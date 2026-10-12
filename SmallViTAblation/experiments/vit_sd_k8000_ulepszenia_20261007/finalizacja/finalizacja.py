"""Download completed comparison, verify bytes, and test locally. No training."""
import hashlib, importlib.util, io, json, msvcrt, os, runpy, shlex, subprocess, sys, tarfile, time, traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs/vit_sd_k8000_ulepszenia_20261007'
COPY = OUT / 'eden_kopia'
REMOTE = '/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_vit_sd_k8000_ulepszenia'
BASE = '/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy'
PY = '/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def write(name, obj):
    p = OUT / name; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.replace(tmp, p)

def status(phase, **kw):
    obj = dict(phase=phase, pid=os.getpid(), updated=time.time(), **kw)
    write('FINALIZACJA_LOKALNA.json', obj); print(json.dumps(obj), flush=True)

def ssh(command, script=None):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                        'eden-cluster', command], input=script.encode() if script else None,
                       capture_output=True, timeout=300)
    if r.returncode: raise RuntimeError(r.stderr.decode(errors='replace')[-3000:])
    return r.stdout

def backup_metadata():
    script = '''import pathlib,json,hashlib,subprocess
r=pathlib.Path(ROOT); real=r.resolve()
jobs=subprocess.run(['squeue','-h','-u','bpiotrowski','-o','%j'],capture_output=True,text=True,check=True).stdout.splitlines()
assert 'sdimpk8' not in jobs, 'Comparison is still active'
c=json.loads((r/'wyniki/porownanie/COMPARISON.json').read_text())
assert c['state']=='completed' and not c['smoke']
files={}; models={}
for p in sorted(r.rglob('*')):
 if not p.is_file():continue
 if not p.resolve().is_relative_to(real):raise ValueError('Escaping link: '+str(p))
 rel=p.relative_to(r).as_posix()
 if p.suffix in ('.json','.jsonl','.py','.md','.slurm','.log','.out'):
  files[rel]={'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 elif p.name=='best_sd.pt' and '/porownanie/' in str(p):
  with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  models[rel]={'size':p.stat().st_size,'sha256':h}
for row in c['records']:
 rel=pathlib.Path(row['checkpoint']).relative_to(r).as_posix()
 assert models[rel]['sha256']==row['checkpoint_sha256']
print(json.dumps({'root':str(r),'real_root':str(real),'files':files,'models':models,'comparison':c}))
'''.replace('ROOT', repr(REMOTE))
    inventory = json.loads(ssh(PY + ' -B -', script))
    write('EDEN_AUDIT_KONCOWY.json', inventory)
    paths = list(inventory['files'])
    archive = ssh('tar -czf - -C ' + shlex.quote(REMOTE) + ' -- ' + ' '.join(map(shlex.quote, paths)))
    COPY.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        for member in tar.getmembers():
            if member.name not in inventory['files'] or not member.isfile(): raise ValueError('Unexpected archive member')
            dest = COPY / member.name
            if not dest.resolve().is_relative_to(COPY.resolve()): raise ValueError('Archive path escape')
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != inventory['files'][member.name]['sha256']: raise ValueError('Metadata transfer hash mismatch')
            dest.write_bytes(data)
    manifest = ssh('cat ' + shlex.quote(BASE + '/wejscia/FILMY.json'))
    assert hashlib.sha256(manifest).hexdigest() == inventory['comparison']['sampler']['manifest_sha256']
    (COPY / 'wejscia').mkdir(exist_ok=True)
    (COPY / 'wejscia/FILMY.json').write_bytes(manifest)
    write('TRANSFER_METADANYCH.json', dict(verified=True, files=len(paths), manifest_verified=True, archive_bytes=len(archive)))
    return inventory

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.finalizacja.lock').open('a+b'); lock.seek(0)
    try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError: print('Finalization already running', flush=True); return
    status('metadata_backup')
    inventory = backup_metadata()
    winner = inventory['comparison']['winner']; rel = Path(winner['checkpoint']).relative_to(REMOTE).as_posix()
    dest = COPY / 'wybrany_model/best_sd.pt'; dest.parent.mkdir(parents=True, exist_ok=True)
    old = ROOT / 'experiments/vit_sd_od_zera_k8000_filmy_20261006/pobierz_best.py'
    spec = importlib.util.spec_from_file_location('owned_resumable_download', old)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.OUT = OUT; module.DEST = dest; module.PART = dest.with_suffix('.pt.part')
    module.STATE = OUT / 'TRANSFER_BEST.json'; module.REMOTE = winner['checkpoint']
    module.SHA = winner['checkpoint_sha256']; module.SIZE = inventory['models'][rel]['size']
    status('checkpoint_transfer', winner=winner['arm'], expected_sha256=module.SHA)
    module.main()
    assert sha(dest) == module.SHA
    status('local_test', checkpoint=str(dest), sha256=module.SHA)
    sys.path.insert(0, str(COPY / 'kod'))
    runpy.run_path(str(HERE / 'test_lokalny_ram.py'), run_name='__main__')

if __name__ == '__main__':
    try: main()
    except Exception:
        status('failed', error=traceback.format_exc()); raise
