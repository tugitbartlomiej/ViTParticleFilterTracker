"""Transport LOCAL -> Eden wyłącznie dla trening.py i ocena.py.

API: dispatch(script, log, kwargs). Odczytuje REMOTE.json obok lokalnego
resume/weights oraz uporządkowane mapping.json datasetu. Wysyła tylko JSON.
ROOT, host, interpreter i skrypt SLURM są stałe; argumenty użytkownika nie
trafiają do zdalnej powłoki. Brak ponowień sbatch, także po utracie odpowiedzi.
Domyślnie odpytuje co 20 s; limit 6 h obejmuje kolejkę i wykonanie 4 h.
Timeout próbuje anulować wyłącznie numer zadania zwrócony przez ten dispatch.
W razie niepewnego wysłania sprawdź request_id zapisany w logu przed ponowieniem.

Przy imporcie nie ma sieci, Torch ani uruchamiania procesów. Transport umożliwia
wstrzyknięcie wykonawcy komend i zegara do testów CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shlex
import subprocess
import tarfile
import tempfile
import time
import uuid

ROOT = '/home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_fastgan_petla'
HOST = 'eden-cluster'
PYTHON = '/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python'
DATA_ARCHIVE = '/home2/faculty/bpiotrowski/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar'
SCRIPTS = {'trening.py', 'ocena.py'}
SCHEMA_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    """Publikacja własnego pliku statusu; katalog rodzica musi już istnieć."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def request_id(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{32}', value):
        raise ValueError('request_id wymaga dokładnie 32 małych znaków hex.')
    return value


def new_request_id():
    return uuid.uuid4().hex


def checked_hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Wymagany SHA-256 zapisany małymi znakami hex.')
    return value


def remote_path(value, *, confined=True):
    if not isinstance(value, str) or '\\' in value or any(ord(c) < 32 for c in value):
        raise ValueError('Niepoprawna ścieżka Edena.')
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or str(path) != value:
        raise ValueError('Ścieżka Edena musi być absolutna, bez przejść do rodzica.')
    if confined and not path.is_relative_to(PurePosixPath(ROOT)):
        raise ValueError('Ścieżka źródłowa wychodzi poza stały ROOT.')
    return value


def validate_request(request):
    if request.get('version') != SCHEMA_VERSION or request.get('script') not in SCRIPTS:
        raise ValueError('Nieobsługiwana wersja lub skrypt żądania.')
    request_id(request['request_id'])
    parent = request['parent']
    for name in ('checkpoint', 'dataset', 'probe', 'config', 'initial_weights', 'pool'):
        remote_path(parent[name])
    if parent.get('image_manifest'):
        remote_path(parent['image_manifest'])
    remote_path(parent['data_root'], confined=False)
    if PurePosixPath(parent['checkpoint']).name != 'FINAL.ckpt':
        raise ValueError('Rodzic musi wskazywać pełny FINAL.ckpt.')
    for name in ('pool_sha256', 'archive_sha256', 'parent_sha256'):
        checked_hash(request[name])
    ids = request['train_ids']
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Wymagane różne uporządkowane ID treningowe.')
    if not isinstance(request.get('added'), bool):
        raise ValueError('added musi być wartością bool.')
    if request['script'] == 'trening.py':
        step = request.get('stop_step')
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError('stop_step musi być dodatnią liczbą całkowitą.')
        if not isinstance(request.get('config'), dict) or request['added'] and len(ids) < 2:
            raise ValueError('Brak konfiguracji lub bazowego podzbioru treningowego.')
    elif request.get('split') not in {'val', 'test'} or request['added']:
        raise ValueError('Ocena wymaga split val/test, bez added.')
    return request


def _provenance(parent, checkpoint_dir):
    digest = parent.get('pool_sha256')
    source = 'REMOTE.json:pool_sha256'
    if digest is None:
        for directory in (checkpoint_dir, *checkpoint_dir.parents):
            candidate = directory / 'INPUTS.json'
            if candidate.is_file() and read_json(candidate).get('pool_sha256'):
                digest, source = read_json(candidate)['pool_sha256'], str(candidate)
                break
    digest = checked_hash(digest)
    archive = parent.get('data_archive_sha256', parent.get('archive_sha256'))
    if parent.get('data_archive_sha256') and parent.get('archive_sha256') and parent['data_archive_sha256'] != parent['archive_sha256']:
        raise ValueError('Sprzeczne skróty archiwum w REMOTE.json.')
    return digest, source, checked_hash(archive)


def make_request(script, kwargs, identifier):
    if script not in SCRIPTS:
        raise ValueError('Zdalnie wolno wykonać tylko trening.py albo ocena.py.')
    request_id(identifier)
    allowed = ({'config', 'dataset', 'probe', 'out', 'initial_weights', 'resume', 'added', 'stop_step', 'device'}
               if script == 'trening.py' else {'weights', 'dataset', 'out', 'split', 'device'})
    if set(kwargs) - allowed:
        raise ValueError(f'Nieznane argumenty: {sorted(set(kwargs) - allowed)}')
    checkpoint = Path(kwargs['resume' if script == 'trening.py' else 'weights']).expanduser().resolve(strict=True)
    required_name = 'FINAL.ckpt' if script == 'trening.py' else 'FINAL.pth'
    if checkpoint.name != required_name:
        raise ValueError(f'Wymagany lokalny {required_name}.')
    parent = read_json(checkpoint.parent / 'REMOTE.json')
    pool_hash, pool_source, archive_hash = _provenance(parent, checkpoint.parent)
    mapping = read_json(Path(kwargs['dataset']) / 'mapping.json')['train']
    if not isinstance(mapping, list) or any(not isinstance(row, dict) for row in mapping):
        raise ValueError('mapping.train musi być listą rekordów.')
    ids = []
    for row in mapping:
        value = row['frame_id']
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError('Niepoprawne frame_id w mapping.json.')
        ids.append(str(value))
    result = dict(version=SCHEMA_VERSION, request_id=identifier, script=script, parent=parent,
                  train_ids=ids, added=kwargs.get('added', False), pool_sha256=pool_hash,
                  pool_sha256_source=pool_source, archive_sha256=archive_hash,
                  archive_hash_provenance=parent.get('archive_hash_provenance', 'REMOTE.json; czas historycznego pomiaru nieznany'),
                  parent_sha256=sha256(checkpoint))
    if script == 'trening.py':
        result.update(stop_step=kwargs['stop_step'], config=read_json(kwargs['config']))
    else:
        result.update(split=kwargs.get('split', 'test'))
    return validate_request(result)


def safe_relative(name):
    """Wspólny bezpieczny podzbiór ścieżek tar dla Linux i Windows."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name or any(ord(c) < 32 for c in name):
        raise ValueError('Niebezpieczna nazwa w archiwum.')
    path = PurePosixPath(name)
    if path.is_absolute() or PureWindowsPath(name).drive or '..' in path.parts:
        raise ValueError('Ścieżka archiwum wychodzi poza katalog docelowy.')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if not path.parts or any(part.endswith((' ', '.')) or part.split('.')[0].upper() in reserved for part in path.parts):
        raise ValueError('Niejednoznaczna nazwa pliku archiwum.')
    return path.as_posix()


def safe_extract(archive_path, destination, *, entries=None, allow_dirs=False):
    """Waliduje cały tar, odrzuca linki i publikuje dopiero sprawdzone pliki.

    entries=None służy wyłącznie lokalnemu archiwum datasetu, wcześniej
    zweryfikowanemu SHA. Wyniki transportu zawsze wymagają pełnego manifestu.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    expected = None
    if entries is not None:
        expected = {}
        for entry in entries:
            name = safe_relative(entry['file'])
            if name in expected or not isinstance(entry['bytes'], int) or isinstance(entry['bytes'], bool) or entry['bytes'] < 0:
                raise ValueError('Niepoprawny manifest archiwum.')
            checked_hash(entry['sha256'])
            expected[name] = entry
    with tarfile.open(archive_path, 'r:*') as archive:
        members, names, files = [], set(), set()
        for member in archive.getmembers():
            if member.isdir() and allow_dirs and member.name in {'.', './'}:
                continue
            name = safe_relative(member.name)
            if name.casefold() in names or (not member.isfile() and not (allow_dirs and member.isdir())):
                raise ValueError('Duplikat, link lub niedozwolony typ wpisu tar.')
            names.add(name.casefold())
            if member.isfile():
                files.add(name)
                if expected is not None and (name not in expected or member.size != expected[name]['bytes']):
                    raise ValueError('Plik tar nie zgadza się z manifestem.')
            members.append((member, name))
        if expected is not None and files != set(expected):
            raise ValueError('Manifest i archiwum nie zawierają dokładnie tych samych plików.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.hybrid-unpack-', dir=destination.parent) as temporary:
            staging = Path(temporary) / 'payload'
            staging.mkdir()
            for member, name in members:
                target = staging / name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.extractfile(member) as source, target.open('xb') as output:
                    for block in iter(lambda: source.read(2**20), b''):
                        digest.update(block)
                        output.write(block)
                if expected is not None and digest.hexdigest() != expected[name]['sha256']:
                    raise ValueError(f'Niezgodny SHA-256 pliku {name}.')
            if destination.exists():
                raise FileExistsError(destination)
            staging.rename(destination)
    return destination


class Transport:
    def __init__(self, *, run=subprocess.run, sleep=time.sleep, monotonic=time.monotonic,
                 poll_seconds=20, timeout_seconds=6 * 3600):
        if poll_seconds <= 0 or timeout_seconds <= 0:
            raise ValueError('Interwał i timeout muszą być dodatnie.')
        self.run, self.sleep, self.monotonic = run, sleep, monotonic
        self.poll_seconds, self.timeout_seconds = poll_seconds, timeout_seconds

    def _execute(self, argv, *, timeout=120):
        result = self.run(argv, check=True, capture_output=True, text=True, timeout=timeout, shell=False)
        return result.stdout.strip()

    def _ssh(self, args):
        return self._execute(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', HOST, shlex.join(map(str, args))])

    def _scp(self, source, destination):
        self._execute(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', str(source), str(destination)], timeout=1800)

    def dispatch(self, script, log, kwargs):
        if os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('dispatch służy wyłącznie kontrolerowi LOCAL.')
        identifier = new_request_id()
        request = make_request(script, kwargs, identifier)
        out = Path(kwargs['out']).expanduser().resolve()
        if out.exists():
            raise FileExistsError(out)
        log = Path(log).expanduser().resolve()
        log.parent.mkdir(parents=True, exist_ok=True)
        remote = f'{ROOT}/hybrid_results/{identifier}'
        remote_request = f'{ROOT}/hybrid_requests/{identifier}.json'
        job_id, terminal = None, False
        with log.open('x', encoding='utf-8') as stream:
            def event(state, **fields):
                stream.write(json.dumps(dict(time=time.time(), request_id=identifier, job_id=job_id,
                                             state=state, **fields), ensure_ascii=False) + '\n')
                stream.flush()
            event('prepared', out=str(out), remote=remote)
            local_request = log.with_name(log.name + '.request.json')
            with local_request.open('x', encoding='utf-8') as handle:
                json.dump(request, handle, ensure_ascii=False, indent=2, allow_nan=False)
            started = self.monotonic()
            try:
                self._ssh(['mkdir', '-p', f'{ROOT}/hybrid_requests', f'{ROOT}/hybrid_results'])
                self._scp(local_request, f'{HOST}:{remote_request}')
                event('submitting')
                response = self._ssh(['sbatch', '--parsable', '--export=NIL',
                    f'--output={ROOT}/hybrid_results/{identifier}.slurm.log',
                    f'{ROOT}/code/run_hybrid_worker.slurm', identifier])
                if not re.fullmatch(r'[1-9][0-9]*(;[A-Za-z0-9_.-]+)?', response):
                    raise RuntimeError(f'Niepewny wynik sbatch: {response!r}. Nie ponawiaj automatycznie żądania {identifier}.')
                job_id = response.split(';')[0]
                event('submitted')
                while True:
                    if self.monotonic() - started >= self.timeout_seconds:
                        raise TimeoutError(f'Przekroczony limit oczekiwania na zadanie {job_id}, request {identifier}.')
                    status = json.loads(self._ssh([PYTHON, '-B', f'{ROOT}/code/hybryda_worker.py',
                                                  'status', '--request-id', identifier, '--job-id', job_id]))
                    if status.get('request_id') != identifier:
                        raise ValueError('Odpowiedź statusu dotyczy innego żądania.')
                    event('poll', status=status)
                    if status.get('state') in {'failed', 'completed'}:
                        terminal = True
                    if status.get('state') == 'failed':
                        raise RuntimeError(f'Zadanie {job_id} nie powiodło się: {status.get("error", status)}')
                    if status.get('state') == 'completed':
                        break
                    self.sleep(self.poll_seconds)
                with tempfile.TemporaryDirectory(prefix=f'hybrid-{identifier}-') as temporary:
                    archive, manifest = Path(temporary) / 'result.tar', Path(temporary) / 'result.manifest.json'
                    self._scp(f'{HOST}:{remote}/result.manifest.json', manifest)
                    self._scp(f'{HOST}:{remote}/result.tar', archive)
                    if sha256(archive) != checked_hash(status['archive_sha256']) or sha256(manifest) != checked_hash(status['manifest_sha256']):
                        raise ValueError('Uszkodzone archiwum lub manifest wyników.')
                    contents = read_json(manifest)
                    if (contents.get('version') != SCHEMA_VERSION or contents.get('request_id') != identifier or
                            contents.get('script') != script):
                        raise ValueError('Manifest wyników dotyczy innego wykonania.')
                    safe_extract(archive, out, entries=contents['files'])
                    # Zachowaj manifest per-file po usunięciu tymczasowego tar.
                    with log.with_name(log.name + '.manifest.json').open('x', encoding='utf-8') as handle:
                        json.dump(contents, handle, ensure_ascii=False, indent=2)
                event('downloaded_and_verified', out=str(out))
                return out
            except BaseException as exc:
                event('failed', error=str(exc), no_automatic_retry=True)
                if job_id and not terminal:
                    try:
                        self._ssh(['scancel', job_id])
                        event('cancellation_requested')
                    except Exception as cancel_error:
                        event('cancellation_unconfirmed', error=str(cancel_error))
                raise


def dispatch(script, log, kwargs):
    """Zamiennik command dla lokalnego kontrolera; wynik Path można zignorować."""
    return Transport().dispatch(script, log, kwargs)
