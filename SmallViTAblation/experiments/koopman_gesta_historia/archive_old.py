"""Back up and verify only old run 1810448 weight files, then remove remote copies."""
import json
import subprocess
from pathlib import Path
from dane import sha256, write_json

HERE=Path(__file__).resolve().parent
REMOTE='/home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_fastgan_petla/results/run_1810448/pretrain'
PREFIX="""
import hashlib,json
from pathlib import Path
p=Path('/home2/faculty/bpiotrowski/smallvit_runs/2026-09-24_koopman_fastgan_petla/results/run_1810448/pretrain')
p=p.resolve()
assert str(p)=='/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/2026-09-24_koopman_fastgan_petla/results/run_1810448/pretrain' and p.is_dir()
assert json.loads((p/'STATUS.json').read_text())['state']=='completed'
def digest(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(2**20),b''):h.update(b)
 return h.hexdigest()
"""
def remote(code):
    return subprocess.run(['ssh','eden-cluster','python3 -'],input=code,text=True,capture_output=True,check=True).stdout

def main():
    dest=HERE/'backup_1810448';dest.mkdir(exist_ok=True)
    rows=json.loads(remote(PREFIX+"print(json.dumps([{'name':f.name,'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(p.iterdir()) if f.is_file() and not f.is_symlink() and f.suffix in ('.pth','.ckpt')]))"))
    assert rows
    write_json(dest/'MANIFEST.json',rows)
    missing=[r for r in rows if not (dest/r['name']).is_file() or sha256(dest/r['name'])!=r['sha256']]
    if missing:
        subprocess.run(['scp','-q',*[f"eden-cluster:{REMOTE}/{r['name']}" for r in missing],str(dest)],check=True)
    assert all(sha256(dest/r['name'])==r['sha256'] for r in rows)
    write_json(dest/'VERIFIED.json',dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),sha256_verified=True))
    # Recheck every remote file BEFORE deleting any; paths are direct children only.
    code=PREFIX+'rows='+repr(rows)+"\n"
    code+='''for r in rows:
 f=p/r['name']
 assert f.parent==p and f.resolve().parent==p and not f.is_symlink()
 assert f.suffix in ('.pth','.ckpt') and digest(f)==r['sha256']
for r in rows:(p/r['name']).unlink()
print(json.dumps({'deleted':[r['name'] for r in rows]}))
'''
    result=json.loads(remote(code));write_json(dest/'REMOTE_DELETED.json',result)
    print(json.dumps(dict(verified_files=len(rows),deleted_files=len(result['deleted']),bytes=sum(r['bytes'] for r in rows))))

if __name__=='__main__':main()
