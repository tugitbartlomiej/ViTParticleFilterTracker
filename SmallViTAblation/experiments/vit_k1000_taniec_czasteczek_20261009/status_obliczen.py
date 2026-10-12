"""Compact read-only status of this experiment; never controls other jobs."""
import json
import shlex
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_k1000_taniec_czasteczek_20261009'

code='''import json,pathlib,subprocess,datetime
p=pathlib.Path('/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009')
records=[json.loads(f.read_text()) for f in (p/'results/production/records').glob('*.json')]
counts={}
for r in records:counts[r['status']]=counts.get(r['status'],0)+1
queue=subprocess.run(['squeue','-u','bpiotrowski','--noheader','--format=%i|%j|%T|%M|%R'],check=True,text=True,capture_output=True).stdout.splitlines()
queue=[q for q in queue if any(k in q for k in ['k1k_r9','k1000_temporal','k1000_dmdc900k'])]
print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'images_complete':len(records),'images_total':1500,'statuses':counts,'queue':queue,'validation_complete':(p/'results/merged/REPLAY_VALIDATION.json').is_file()}))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','eden-cluster','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=45)
if r.returncode:raise RuntimeError(r.stderr[-1000:])
result=json.loads(r.stdout)
(OUT/'STATUS_KLASTRA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
