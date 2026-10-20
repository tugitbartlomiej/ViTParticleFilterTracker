"""Atomowy zapis wyników, sumy SHA256 i blokady seedów. Zapis nigdy nie nadpisuje zakończonego sharda."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def finite(obj):
    """NaN/inf → None, typy numpy → Python; JSON pozostaje ściśle poprawny (allow_nan=False)."""
    if isinstance(obj, dict):
        return {str(k): finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(finite(value), ensure_ascii=False, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def done_path(path):
    return Path(str(path) + '.done.json')


def verify_npz(path, identity):
    """None: brak sharda; dict: zakończony i zgodny; wyjątek: niezgodność lub niedokończony plik."""
    path = Path(path)
    marker = done_path(path)
    if not marker.exists():
        if path.exists():
            raise ValueError('Unmarked result exists; inspect before resume: ' + str(path))
        return None
    done = read_json(marker)
    if done.get('status') != 'complete' or any(done.get(k) != v for k, v in finite(identity).items()):
        raise ValueError('Shard identity differs: ' + str(path))
    if not path.is_file() or file_sha(path) != done.get('sha256'):
        raise ValueError('Shard hash differs: ' + str(path))
    return done


def save_npz(path, identity, arrays, seconds):
    path = Path(path)
    done = verify_npz(path, identity)
    if done is not None:
        return done
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    done = dict(finite(identity), status='complete', sha256=file_sha(path), seconds=float(seconds),
                bytes=path.stat().st_size, shapes={k: list(np.shape(v)) for k, v in arrays.items()},
                dtypes={k: str(np.asarray(v).dtype) for k, v in arrays.items()})
    save_json(done_path(path), done)
    return done


@contextlib.contextmanager
def seed_lock(directory, name, enabled=True):
    """Blokada fcntl (Linux); zwalniana przez system, gdy zadanie zginie. Lokalnie można wyłączyć."""
    if not enabled:
        yield
        return
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / (name + '.lock')).open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another live worker owns ' + name)
        save_json(directory / (name + '.owner.json'), dict(pid=os.getpid(), job=os.environ.get('SLURM_JOB_ID'),
                  array_task=os.environ.get('SLURM_ARRAY_TASK_ID'), time=time.time()))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def freeze_files(code_dir, opis_dir):
    """Lista plików objętych FREEZE: cały Kod_eksperymentu (bez FREEZE.json i cache) i cały Opis."""
    files = {}
    for base, label in ((Path(code_dir), 'Kod_eksperymentu'), (Path(opis_dir), 'Opis')):
        for p in sorted(base.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name != 'FREEZE.json' and not p.name.endswith('.pyc'):
                files[label + '/' + p.relative_to(base).as_posix()] = file_sha(p)
    return files


def verify_freeze(exp_root):
    exp_root = Path(exp_root)
    freeze = read_json(exp_root / 'Kod_eksperymentu' / 'FREEZE.json')
    current = freeze_files(exp_root / 'Kod_eksperymentu', exp_root / 'Opis')
    if current != freeze['files']:
        changed = sorted(set(current.items()) ^ set(freeze['files'].items()))
        raise ValueError('Frozen experiment files changed: ' + str(changed[:5]))
    return file_sha(exp_root / 'Kod_eksperymentu' / 'FREEZE.json')
