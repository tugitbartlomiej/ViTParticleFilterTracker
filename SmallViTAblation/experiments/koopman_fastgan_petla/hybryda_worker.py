"""Jednorazowy worker Edena; import i status nie ładują Torch.

run wykonuje tylko skrypt podany w sprawdzonym żądaniu. Odtwarza oryginalny
dataset we własnym scratch. Wyniki, dataset COCO, konfiguracja i metadane
rodzica pozostają w ROOT/hybrid_results/<request_id>. Zmieniają się ścieżki
obrazów, nie przepis treningu ani stop_step. Rebase wymaga dowodu bajtów.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

from hybryda_zdalne import (ROOT, DATA_ARCHIVE, SCHEMA_VERSION, read_json, write_json,
                            sha256, request_id, remote_path, validate_request, safe_extract)


def result_root(identifier):
    return Path(ROOT) / 'hybrid_results' / request_id(identifier)


def source_path(value):
    remote_path(value)
    path = Path(value).resolve(strict=True)
    if not path.is_relative_to(Path(ROOT).resolve(strict=True)):
        raise ValueError('Link źródłowy wychodzi poza ROOT.')
    return path


def rebuild_datasets(request, rows, data_root, output_root, *, parent_mapping):
    from dane import make_dataset

    ids = request['train_ids']
    base_ids = ids[:-1] if request['added'] else ids
    expected = [str(row['frame_id']) for row in parent_mapping]
    if base_ids != expected:
        raise ValueError('Żądanie zmienia kolejność lub tożsamość klatek rodzica.')
    byid = {row['frame_id']: row for row in rows}
    if any(fid not in byid or byid[fid]['split'] != 'train' for fid in ids):
        raise ValueError('Wybór zawiera klatkę spoza puli train.')
    if any(str(row['group_id']) != str(byid[str(row['frame_id'])]['group_id']) for row in parent_mapping):
        raise ValueError('Grupy rodzica różnią się od oryginalnej puli.')
    output_root = Path(output_root)
    base = make_dataset(rows, data_root, base_ids, output_root / 'base_dataset')
    branch = make_dataset(rows, data_root, base_ids, output_root / 'dataset', added=ids[-1] if request['added'] else None)
    return base, branch


def stage_command(request, *, code, dataset, config, resume, out):
    parent = request['parent']
    argv = [sys.executable, '-u', '-B', str(Path(code) / request['script']),
            '--dataset', str(dataset), '--out', str(out), '--device', 'cuda']
    if request['script'] == 'trening.py':
        if resume is None:
            raise ValueError('Brak przeniesionego checkpointu do treningu.')
        argv += ['--config', str(config), '--probe', parent['probe'],
                 '--initial-weights', parent['initial_weights'], '--resume', str(resume),
                 '--stop-step', str(request['stop_step'])]
        if request['added']:
            argv.append('--added')
    elif request['script'] == 'ocena.py':
        argv += ['--weights', str(PurePosixPath(parent['checkpoint']).with_suffix('.pth')),
                 '--split', request['split']]
    else:
        raise ValueError('Niedozwolony skrypt.')
    return argv


def image_manifest(checkpoint, dataset, data_root):
    """Nowy dowód bajtów związany z nowym FINAL, przed usunięciem scratch."""
    data_root = Path(data_root).resolve(strict=True)
    files = {}
    for split in ('train', 'valid'):
        for image in read_json(Path(dataset) / split / '_annotations.coco.json')['images']:
            path = Path(image['file_name']).resolve(strict=True)
            if not path.is_relative_to(data_root) or not path.is_file():
                raise ValueError('Zdjęcie w manifeście wychodzi poza scratch data_root.')
            relative = path.relative_to(data_root).as_posix()
            if relative not in files:
                files[relative] = sha256(path)
    return dict(checkpoint_sha256=sha256(checkpoint), data_root=str(data_root), files=files)


def pack_results(source, prefix, identifier, script):
    """Archiwum zawiera FINAL, historię i wszystkie JSON-y; nie zawiera FULL."""
    source, prefix = Path(source).resolve(), Path(prefix)
    request_id(identifier)
    required = {'REMOTE.json'} | ({'FINAL.ckpt', 'FINAL.pth', 'VALIDATION.json', 'STATUS.json'}
                                   if script == 'trening.py' else {'REPORT.json'})
    candidates = []
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Wyniki nie mogą zawierać linków.')
        if not path.is_file():
            continue
        if (path.suffix in {'.json', '.jsonl', '.log'} or path.name in {'FINAL.ckpt', 'FINAL.pth'} or
                path.suffix == '.npz' and path.name.startswith(('LATENT_', 'CZASTKI_'))):
            candidates.append(path)
    names = {p.relative_to(source).as_posix() for p in candidates}
    if not required <= names or (script == 'trening.py' and not any(p.name.startswith('LATENT_') for p in candidates)):
        raise ValueError('Niekompletne wyniki; brak FINAL, historii lub raportu.')
    archive, manifest = prefix.with_suffix('.tar'), prefix.with_suffix('.manifest.json')
    if archive.exists() or manifest.exists():
        raise FileExistsError('Archiwum wyniku już istnieje.')
    entries = []
    with tarfile.open(archive, 'x') as tar:
        for path in candidates:
            name = path.relative_to(source).as_posix()
            entry = dict(file=name, bytes=path.stat().st_size, sha256=sha256(path))
            tar.add(path, arcname=name, recursive=False)
            entries.append(entry)
    with manifest.open('x', encoding='utf-8') as handle:
        json.dump(dict(version=SCHEMA_VERSION, request_id=identifier, script=script, files=entries),
                  handle, ensure_ascii=False, indent=2, allow_nan=False)
    return archive, manifest


def run_worker(identifier, scratch):
    request_id(identifier)
    job_id = os.environ.get('SLURM_JOB_ID', '')
    scratch = Path(scratch).resolve(strict=True)
    if (not re.fullmatch('[1-9][0-9]*', job_id) or scratch.parent != Path('/tmp') or
            not scratch.name.startswith(f'koopman-hybrid-{job_id}-{identifier}-')):
        raise ValueError('Worker wymaga własnego katalogu scratch zadania SLURM.')
    request = validate_request(read_json(Path(ROOT) / 'hybrid_requests' / f'{identifier}.json'))
    if request['request_id'] != identifier:
        raise ValueError('Niezgodny identyfikator pliku żądania.')
    result = result_root(identifier)
    result.mkdir(parents=True, exist_ok=False)
    status = dict(request_id=identifier, job_id=job_id, state='running', started_unix=time.time())
    def stage(name, **extra):
        status.update(stage=name, updated_unix=time.time(), **extra)
        write_json(result / 'STATUS.json', status)
    stage('verify_inputs')
    try:
        write_json(result / 'REQUEST.json', request)
        parent = request['parent']
        for key in ('dataset', 'probe', 'config', 'initial_weights', 'pool'):
            source_path(parent[key])
        if parent.get('image_manifest'):
            source_path(parent['image_manifest'])
        source_checkpoint = (parent['checkpoint'] if request['script'] == 'trening.py' else
                             str(PurePosixPath(parent['checkpoint']).with_suffix('.pth')))
        if sha256(source_path(source_checkpoint)) != request['parent_sha256']:
            raise ValueError('Lokalny i zdalny checkpoint rodzica mają różne SHA-256.')
        config = read_json(parent['config'])
        if request['script'] == 'trening.py' and config != request['config']:
            raise ValueError('Lokalna receptura różni się od konfiguracji rodzica.')
        if sha256(parent['pool']) != request['pool_sha256']:
            raise ValueError('Zdalny oryginalny manifest puli ma inny SHA-256.')
        archive_hash = sha256(DATA_ARCHIVE)
        if archive_hash != request['archive_sha256']:
            raise ValueError('Archiwum datasetu zmieniło się od przekazania.')
        stage('extract_dataset')
        data_root = safe_extract(DATA_ARCHIVE, scratch / 'data', allow_dirs=True)
        pool = data_root / 'pool_manifest.jsonl'
        if sha256(pool) != request['pool_sha256']:
            raise ValueError('Manifest wypakowany z archiwum nie zgadza się z pulą rodzica.')
        proof = dict(pool_sha256=request['pool_sha256'], pool_sha256_source=request['pool_sha256_source'],
                     archive_sha256=archive_hash, archive_path=DATA_ARCHIVE,
                     archive_hash_provenance=request['archive_hash_provenance'],
                     archive_matches_handoff=True, historical_archive_digest_claimed=False)
        write_json(result / 'DATA_PROOF.json', proof)
        from dane import load_pool
        rows = load_pool(pool)
        for row in rows:
            path = (data_root / row['path']).resolve()
            if not path.is_relative_to(data_root) or row['split'] not in {'train', 'val', 'test'}:
                raise ValueError('Ścieżka lub split puli jest niedozwolony.')
        base, dataset = rebuild_datasets(request, rows, data_root, result,
                                         parent_mapping=read_json(Path(parent['dataset']) / 'mapping.json')['train'])
        local_config = result / 'config.json'
        shutil.copyfile(parent['config'], local_config)
        resume = None
        if request['script'] == 'trening.py':
            stage('rebase_checkpoint')
            from przenosny_checkpoint import rebase_file
            resume = scratch / 'rebased.ckpt'
            audit = rebase_file(checkpoint=parent['checkpoint'], newfile=resume, dataset=base,
                               probe=parent['probe'], old_validation_json=Path(parent['dataset']) / 'valid' / '_annotations.coco.json',
                               old_data_root=parent['data_root'], new_data_root=data_root,
                               image_manifest=parent.get('image_manifest'))
            write_json(result / 'REBASE.json', audit)
        out = result / ('training' if request['script'] == 'trening.py' else 'evaluation')
        code = scratch / 'code'
        command = stage_command(request, code=code, dataset=dataset, config=local_config, resume=resume, out=out)
        write_json(result / 'CODE.json', {p.name: sha256(p) for p in sorted(code.glob('*.py'))})
        write_json(result / 'COMMAND.json', command)
        stage('training' if request['script'] == 'trening.py' else 'evaluation')
        with (result / 'worker.log').open('x', encoding='utf-8') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=code)
        # Dane COCO pozostają na shared, mimo że zawarte w nich stare ścieżki
        # scratch znikną. Następny worker przepisuje je kontrolowanym rebase.
        remote = dict(parent, dataset=str(dataset), data_root=str(data_root), config=str(local_config),
                      pool_sha256=request['pool_sha256'], data_archive_sha256=archive_hash,
                      archive_sha256=archive_hash, archive_hash_provenance=request['archive_hash_provenance'],
                      source_job=job_id, request_id=identifier)
        if request['script'] == 'trening.py':
            remote['checkpoint'] = str(out / 'FINAL.ckpt')
            remote['image_manifest'] = str(out / 'IMAGE_BYTES.json')
            write_json(out / 'IMAGE_BYTES.json', image_manifest(out / 'FINAL.ckpt', dataset, data_root))
            remote.update(checkpoint_sha256=sha256(out / 'FINAL.ckpt'),
                          image_manifest_sha256=sha256(out / 'IMAGE_BYTES.json'),
                          parent_checkpoint=parent['checkpoint'], parent_checkpoint_sha256=request['parent_sha256'],
                          parent_source_job=parent.get('source_job'))
        else:
            # Ocena nie jest nowym rodzicem; zachowuje pełny kontrakt checkpointu.
            remote = dict(parent, evaluation_dataset=str(dataset), evaluation_source_job=job_id)
        write_json(out / 'REMOTE.json', remote)
        for filename in ('REQUEST.json', 'DATA_PROOF.json', 'CODE.json', 'COMMAND.json', 'REBASE.json', 'worker.log'):
            if (result / filename).is_file():
                shutil.copyfile(result / filename, out / filename)
        stage('pack_results')
        archive, manifest = pack_results(out, result / 'result', identifier, request['script'])
        stage('completed', state='completed', archive_sha256=sha256(archive), manifest_sha256=sha256(manifest),
              output=str(out), finished_unix=time.time())
        return out
    except BaseException as exc:
        stage('failed', state='failed', error=str(exc), traceback=traceback.format_exc())
        raise


def status_report(identifier, job_id):
    request_id(identifier)
    if not re.fullmatch('[1-9][0-9]*', job_id):
        raise ValueError('Niepoprawny numer zadania SLURM.')
    path = result_root(identifier) / 'STATUS.json'
    status = read_json(path) if path.is_file() else dict(request_id=identifier, state='pending')
    if status.get('request_id') != identifier or status.get('job_id', job_id) != job_id:
        raise ValueError('Status należy do innego zadania.')
    if status['state'] in {'completed', 'failed'}:
        return status
    process = subprocess.run(['sacct', '-n', '-P', '-j', job_id, '--format=JobIDRaw,State,ExitCode'],
                             check=True, capture_output=True, text=True, timeout=30)
    for line in process.stdout.splitlines():
        fields = line.split('|')
        if len(fields) >= 3 and fields[0] == job_id:
            state = fields[1].split()[0].rstrip('+')
            status['slurm_state'] = state
            if state in {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'COMPLETED'}:
                status.update(state='failed', error=f'SLURM {state}, exit={fields[2]}, bez kompletnego archiwum wyników.')
            break
    return status


def mark_failed(identifier, message):
    root = result_root(identifier)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'STATUS.json'
    previous = read_json(path) if path.exists() else {}
    if previous.get('state') in {'completed', 'failed'}:
        return
    write_json(path, dict(request_id=identifier, job_id=os.environ.get('SLURM_JOB_ID'),
                          state='failed', error=message, updated_unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'status', 'fail'))
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--scratch')
    parser.add_argument('--job-id')
    parser.add_argument('--message', default='Worker SLURM zakończył się błędem.')
    args = parser.parse_args()
    if args.command == 'run':
        run_worker(args.request_id, args.scratch)
    elif args.command == 'status':
        print(json.dumps(status_report(args.request_id, args.job_id), ensure_ascii=False))
    else:
        mark_failed(args.request_id, args.message)


if __name__ == '__main__':
    main()
