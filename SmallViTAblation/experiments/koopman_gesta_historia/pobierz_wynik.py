"""Resume-safe download of the completed run, with per-file remote SHA256."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def save(p, value):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(p)

def main():
    run = json.loads((HERE / 'EDEN_RUN.json').read_text(encoding='utf-8-sig'))
    root = run['production_result']
    dest = HERE / 'wyniki' / ('run_' + str(run['production_job']))
    dest.mkdir(parents=True, exist_ok=True)
    statefile = HERE / 'DOWNLOAD_STATE.json'
    state = dict(state='manifest', pid=os.getpid(), destination=str(dest), updated=time.time())
    save(statefile, state)
    code = '''import hashlib,json
from pathlib import Path
p=Path(ROOT)
assert json.loads((p/'REPORT.json').read_text())['state']=='completed'
rows=[]
for f in sorted(p.rglob('*')):
 if f.is_symlink(): raise RuntimeError('Unexpected symlink: '+str(f))
 if not f.is_file(): continue
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(4*1024**2),b''): h.update(b)
 rows.append(dict(path=f.relative_to(p).as_posix(),bytes=f.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(rows))
'''.replace('ROOT', repr(root))
    manifest = dest / 'REMOTE_MANIFEST.json'
    # The run was completed before this immutable per-run manifest was made.
    rows = json.loads(manifest.read_text()) if manifest.is_file() else json.loads(subprocess.run(['ssh', run['ssh_alias'], 'python3 -'], input=code, text=True, capture_output=True, check=True).stdout)
    save(dest / 'REMOTE_MANIFEST.json', rows)
    missing = [r for r in rows if not (dest / r['path']).is_file() or digest(dest / r['path']) != r['sha256']]
    required = sum(r['bytes'] for r in missing)
    if shutil.disk_usage(dest).free < required + 5 * 1024**3:
        raise RuntimeError('Insufficient disk space: need ' + str(required))
    for i, r in enumerate(missing):
        rel = Path(r['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.download')
        state.update(state='downloading', current=r['path'], remaining=len(missing)-i, updated=time.time())
        save(statefile, state)
        for attempt in range(5):
            result = subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3', run['ssh_alias'] + ':' + root + '/' + r['path'], str(partial)])
            if result.returncode == 0:
                break
            state.update(state='retrying', attempt=attempt+1, updated=time.time())
            save(statefile, state)
            if attempt == 4:
                result.check_returncode()
            time.sleep(10 * (attempt+1))
        if digest(partial) != r['sha256']:
            raise RuntimeError('SHA256 mismatch: ' + r['path'])
        partial.replace(target)
    assert all(digest(dest / r['path']) == r['sha256'] for r in rows)
    save(dest / 'VERIFIED.json', dict(files=len(rows), bytes=sum(r['bytes'] for r in rows), sha256_verified=True, completed=time.time()))
    state.update(state='verified', updated=time.time(), files=len(rows))
    save(statefile, state)
    run.update(state='downloaded_verified_analysis_pending', local_result=str(dest))
    save(HERE / 'EDEN_RUN.json', run)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(HERE / 'DOWNLOAD_ERROR.json', dict(error=repr(exc), pid=os.getpid(), updated=time.time()))
        raise
