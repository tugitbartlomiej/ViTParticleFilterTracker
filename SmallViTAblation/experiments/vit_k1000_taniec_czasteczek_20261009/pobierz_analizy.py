"""Collect completed scientific reports and regime labels; never submit jobs."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/vit_k1000_wzorce_pydmd_deeptime_20261009'
REMOTE = '/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009'
JOBS = {'panel64': '1825171', 'population900k': '1825172',
        'population900k_dmdc_supplement': '1825187'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''):h.update(b)
    return h.hexdigest()

def save(value):
    p=OUT/'TRANSFER_WYNIKOW.tmp'
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(p,OUT/'TRANSFER_WYNIKOW.json')

def ssh(command):
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','eden-cluster',command],
                          capture_output=True,text=True,timeout=45)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (OUT/'TRANSFER_WYNIKOW.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError('Collector already running')
        pending=set(JOBS);started=time.time()
        while pending and time.time()-started<48*3600:
            for name in sorted(pending):
                path=f'{REMOTE}/results/{name}'
                code=f'''import json,pathlib,hashlib,subprocess
job={JOBS[name]!r}
account=subprocess.run(['sacct','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode'],capture_output=True,text=True,check=True)
base=[row.split('|') for row in account.stdout.splitlines() if row.split('|')[0]==job]
if not base or base[0][1] in ('PENDING','RUNNING','COMPLETING','CONFIGURING'): raise SystemExit(44)
if base[0][1]!='COMPLETED' or base[0][2]!='0:0': raise RuntimeError('Analysis job did not complete successfully: '+str(base))
p=pathlib.Path({path!r})
if not (p/'PROVENANCE.json').is_file() or not (p/'WYNIKI.md').is_file(): raise SystemExit(44)
v=json.loads((p/'PROVENANCE.json').read_text())
if v.get('status')!='complete' or str(v.get('job_id'))!={JOBS[name]!r}: raise SystemExit(45)
files=[]
for f in sorted(p.rglob('*')):
 if not f.is_file(): continue
 if f.is_symlink(): raise ValueError('Unexpected symlink')
 if f.suffix not in ('.json','.md','.csv','.npz','.txt'): raise ValueError('Unexpected artifact type')
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8388608),b''): h.update(b)
 files.append(dict(path=f.relative_to(p).as_posix(),sha256=h.hexdigest(),bytes=f.stat().st_size))
print(json.dumps(dict(status='complete',files=files,provenance=v)))
'''
                try:r=ssh('python3 -c '+shlex.quote(code))
                except subprocess.TimeoutExpired:continue
                if r.returncode:
                    if r.returncode not in (44,255):
                        raise RuntimeError(f'{name}: remote validation failed: '+r.stderr[-600:])
                    continue
                manifest=json.loads(r.stdout)
                incoming=OUT/'.incoming';incoming.mkdir(exist_ok=True)
                subprocess.run(['scp','-q','-r','-o','BatchMode=yes','-o','ConnectTimeout=20',
                                f'eden-cluster:{path}',str(incoming)],check=True)
                downloaded=incoming/name
                for file in manifest['files']:
                    local=(downloaded/file['path']).resolve();local.relative_to(downloaded.resolve())
                    if local.stat().st_size!=file['bytes'] or sha(local)!=file['sha256']:
                        raise ValueError(f'Transfer mismatch {name}/{file["path"]}')
                # Provenance and all files are verified before final publication.
                destination=OUT/name
                if destination.exists():raise RuntimeError(f'Preserve existing result directory {destination}')
                downloaded.rename(destination)
                (OUT/f'TRANSFER_{name}.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
                pending.remove(name)
            save({'status':'waiting_for_analyses' if pending else 'complete','pid':os.getpid(),
                  'pending':sorted(pending),'jobs':JOBS,'elapsed_seconds':round(time.time()-started),
                  'completed_all_hashes_verified':sorted(set(JOBS)-pending)})
            if pending:time.sleep(45)
        if pending:raise TimeoutError('Analyses did not complete within the waiting period')

if __name__=='__main__':
    try:main()
    except Exception as error:
        if 'already running' not in str(error):save({'status':'failed','error':str(error),'pid':os.getpid()})
        raise
