"""Download the already completed Eden measurement; no remote writes or training.

Reads/hashes the remote file, resumes SFTP to one .part, hashes locally, then
renames in place. Reads the NumPy header and small mmap samples, never np.load
without mmap. Guards the remaining disk requirement and preserves every file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
EXPECTED = Path('F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pobranie_latentu_2026-10-01').resolve()
assert HERE == EXPECTED
TARGET = HERE.parent / 'dane' / 'LATENT_GESTY.npy'
PART = TARGET.with_name(TARGET.name + '.part')
REMOTE = '/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/LATENT_GESTY.npy'
RESERVE = 1024**3
OTHER_WRITES = 128 * 1024**2  # LaMa agent reports <50 MB remaining; include margin.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def now():
    return datetime.now(timezone.utc).isoformat()


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def preflight():
    remote_script = '''import ast,datetime,hashlib,json,struct,pathlib
p=pathlib.Path("REMOTE_PLACEHOLDER")
s=p.stat()
with p.open("rb") as f:
 assert f.read(6)==b"\\x93NUMPY"
 version=tuple(f.read(2))
 n=struct.unpack("<H" if version[0]==1 else "<I",f.read(2 if version[0]==1 else 4))[0]
 header=ast.literal_eval(f.read(n).decode("latin1"))
 offset=f.tell()
h=hashlib.sha256()
with p.open("rb") as f:
 for block in iter(lambda:f.read(8*1024**2),b""):h.update(block)
after=p.stat()
assert s.st_size==after.st_size and s.st_mtime_ns==after.st_mtime_ns
print(json.dumps(dict(source_host_alias="eden-cluster",path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest(),npy_version=version,header=header,data_offset=offset,verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_mutations=False)))
'''.replace('REMOTE_PLACEHOLDER', REMOTE)
    print('REMOTE_HASH_STARTED', flush=True)
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                        '-o', 'ServerAliveInterval=30', 'eden-cluster', 'python3 -'],
                       input=remote_script.encode('ascii'), capture_output=True, creationflags=NO_WINDOW)
    (HERE / 'ssh_manifest_stderr.log').write_bytes(r.stderr)
    if r.returncode:
        raise RuntimeError('Read-only SSH manifest failed; see ssh_manifest_stderr.log')
    m = json.loads(r.stdout.decode('utf-8'))
    assert m['path'] == REMOTE and m['bytes'] == 22140518528
    assert m['header'] == {'descr': '<f4', 'fortran_order': False, 'shape': [1001, 12, 600, 768]}
    assert m['data_offset'] == 128
    write('MANIFEST_ZDALNY_2026-10-01.json', m)
    have = PART.stat().st_size if PART.exists() else 0
    free = shutil.disk_usage(TARGET.parent).free
    audit = dict(created_at_utc=now(), free_bytes=free, remote_bytes=m['bytes'],
                 existing_partial_bytes=have, remaining_bytes=m['bytes'] - have,
                 required_reserve_bytes=RESERVE, other_writes_allowance_bytes=OTHER_WRITES,
                 projected_free_after_download_and_other_writes=free - (m['bytes'] - have) - OTHER_WRITES,
                 other_writes_basis='LaMa400 owner reports less than 50 MB remaining; reserve 128 MiB',
                 no_large_new_cache=True, memory_strategy='8 MiB hash blocks, mmap sample verification')
    write('AUDYT_MIEJSCA_2026-10-01.json', audit)
    if not TARGET.exists() and free < m['bytes'] - have + RESERVE + OTHER_WRITES:
        raise RuntimeError('Insufficient space for remaining bytes, 1 GiB reserve and LaMa allowance; no files removed.')
    print(json.dumps(dict(status='preflight_ok', bytes=m['bytes'], sha256=m['sha256'],
                         free_bytes=free, projected_free_bytes=free-m['bytes']+have)), flush=True)


def download():
    m = json.loads((HERE / 'MANIFEST_ZDALNY_2026-10-01.json').read_text(encoding='utf-8'))
    assert m['path'] == REMOTE and len(m['sha256']) == 64
    TARGET.parent.mkdir(exist_ok=True)
    start = now()
    free_before = shutil.disk_usage(TARGET.parent).free
    have_initial = PART.stat().st_size if PART.exists() else 0
    preserved_existing = TARGET.exists()
    if TARGET.exists():
        if TARGET.stat().st_size != m['bytes'] or digest(TARGET) != m['sha256']:
            raise RuntimeError('Existing final file differs; preserved without overwrite.')
    else:
        if have_initial > m['bytes']:
            raise RuntimeError('Oversized partial preserved; manual diagnosis required.')
        if free_before < m['bytes'] - have_initial + RESERVE + OTHER_WRITES:
            raise RuntimeError('Disk guard failed before transfer; partial preserved.')
        batch = HERE / 'sftp_batch.txt'
        batch.write_text(f'reget "{REMOTE}" "{PART.as_posix()}"\n', encoding='utf-8')
        for attempt in range(1, 4):
            if not PART.exists() or PART.stat().st_size < m['bytes']:
                with (HERE / f'sftp_attempt_{attempt}.log').open('wb') as log:
                    p = subprocess.Popen(['sftp', '-b', str(batch), '-o', 'BatchMode=yes',
                                          '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30',
                                          '-o', 'ServerAliveCountMax=3', 'eden-cluster'],
                                         stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
                    print(f'TRANSFER_STARTED pid={p.pid} attempt={attempt} bytes={m["bytes"]}', flush=True)
                    last_print = 0.0
                    while p.poll() is None:
                        count = PART.stat().st_size if PART.exists() else 0
                        free = shutil.disk_usage(TARGET.parent).free
                        projected = free - max(0, m['bytes'] - count)
                        status = dict(status='downloading', updated_at_utc=now(), pid=p.pid,
                                      attempt=attempt, transferred_bytes=count, total_bytes=m['bytes'],
                                      percent=100*count/m['bytes'], free_bytes=free,
                                      projected_free_bytes=projected)
                        write('STATUS.json', status)
                        if projected < RESERVE:
                            p.terminate()
                            p.wait(timeout=30)
                            raise RuntimeError('Disk reserve below 1 GiB; stopped transfer, kept resumable partial.')
                        if time.monotonic() - last_print >= 60:
                            print(json.dumps(status), flush=True)
                            last_print = time.monotonic()
                        time.sleep(10)
                    print(f'SFTP_FINISHED attempt={attempt} exit={p.returncode}', flush=True)
            if PART.exists() and PART.stat().st_size == m['bytes']:
                break
        if not PART.exists() or PART.stat().st_size != m['bytes']:
            raise RuntimeError('Transfer incomplete after retries; resumable partial preserved.')
        write('STATUS.json', dict(status='verifying_sha256', updated_at_utc=now(), bytes=m['bytes']))
        print('LOCAL_HASH_STARTED', flush=True)
        local_hash = digest(PART)
        if local_hash != m['sha256']:
            raise RuntimeError('SHA256 mismatch; partial preserved, no final file published.')
        if TARGET.exists():
            raise RuntimeError('Final file appeared concurrently; refusing overwrite.')
        assert PART.parent.resolve() == TARGET.parent.resolve() == (HERE.parent / 'dane').resolve()
        PART.rename(TARGET)  # Same directory and volume: no duplicate 22 GB copy.
    import numpy as np
    a = np.load(TARGET, mmap_mode='r', allow_pickle=False)
    assert isinstance(a, np.memmap) and a.shape == (1001, 12, 600, 768)
    assert a.dtype == np.dtype('<f4') and a.flags.c_contiguous
    samples = []
    for t in [0, 1, 500, 1000]:
        for image_index in [0, 11]:
            for token_index in [0, 299, 599]:
                v = a[t, image_index, token_index]
                assert np.isfinite(v).all()
                samples.append(dict(step=t, image=image_index, token=token_index,
                                    l2_norm=float(np.linalg.norm(v)), finite=True))
    result = dict(schema='eden-dense-latent-download-v1', status='complete_verified',
                  started_at_utc=start, completed_at_utc=now(), source_host_alias='eden-cluster',
                  remote_path=REMOTE, local_path=str(TARGET), bytes=TARGET.stat().st_size,
                  local_sha256=m['sha256'], remote_sha256=m['sha256'], sha256_match=True,
                  shape=list(a.shape), dtype=str(a.dtype), mmap=True, sampled_vectors=samples,
                  preserved_existing_final=preserved_existing, resumed_bytes=have_initial,
                  free_bytes_before=free_before, free_bytes_after=shutil.disk_usage(TARGET.parent).free,
                  remote_mutations=False, training_or_forward_performed=False,
                  optimizer_state_download=False, second_full_copy=False)
    write('MANIFEST_LOKALNY_2026-10-01.json', result)
    (HERE / 'SHA256SUMS.txt').write_text(m['sha256'] + '  LATENT_GESTY.npy\n')
    write('STATUS.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'bytes', 'shape', 'dtype', 'sha256_match', 'free_bytes_after']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['preflight', 'download'])
    args = p.parse_args()
    try:
        preflight() if args.mode == 'preflight' else download()
    except Exception as exc:
        write('STATUS.json', dict(status='stopped_with_error', updated_at_utc=now(), error=str(exc),
                                 partial_preserved=PART.exists(), final_exists=TARGET.exists()))
        raise
