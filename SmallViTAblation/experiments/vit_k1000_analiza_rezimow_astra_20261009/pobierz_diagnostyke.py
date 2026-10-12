"""Odczytuje Eden bez plików zdalnych; zapis tylko do własnego outputs.

Obala zgodność wersji, jeśli bieżący hash odbiega od historycznego inventory.
"""
import json, subprocess, pathlib, hashlib, datetime
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_k1000_analiza_rezimow_astra_20261009'
FAMILY='vit_k1000_1500_pelne_latenty_90_20261008'
REMOTE='/home2/faculty/bpiotrowski/smallvit_runs/'+FAMILY
PY='/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python'
def ssh_python(script):
    p=subprocess.run(['ssh','-o','BatchMode=yes','eden-cluster',f'env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 {PY} -B -'],input=script,text=True,encoding='utf-8',capture_output=True,timeout=180)
    if p.returncode: raise RuntimeError(p.stderr+'\n'+p.stdout)
    return json.loads(p.stdout)
OUT.mkdir(parents=True,exist_ok=True)
code=(pathlib.Path(__file__).parent/'diagnostyka_eden.py').read_text(encoding='utf-8')
report=ssh_python(code)
(OUT/'DIAGNOSTYKA_EDEN.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('remote diagnostics complete',report['elapsed_seconds'],flush=True)
snapshot=ssh_python("""import pathlib,json,hashlib
r=pathlib.Path(%r)
paths=list((r/'code').glob('*.py'))+[r/'eden/dmdc.slurm',r/'eden/regimes.slurm',r/'eden/submit_production.sh',r/'logs/k1k1500_dmdc_1824880.out',r/'logs/k1k1500_regimes_1824879.out',r/'provenance/CODE_INVENTORY.json']
out=[]
for p in paths:
 if p.exists():
  b=p.read_bytes();out.append({'path':str(p),'relative':str(p.relative_to(r)),'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode('utf-8')})
print(json.dumps(out))
""" % REMOTE)
inventory=json.loads((ROOT/'outputs'/FAMILY/'provenance/CODE_INVENTORY.json').read_text())
expected={x['path']:x['sha256'] for x in inventory['files']}
evidence=[]
for item in snapshot:
    dest=OUT/'snapshot_eden'/item['relative'];dest.parent.mkdir(parents=True,exist_ok=True)
    # Encode without newline translation to preserve exact SHA256.
    dest.write_bytes(item.pop('text').encode('utf-8'))
    local=ROOT/'experiments'/FAMILY/pathlib.Path(item['relative']).relative_to('code') if item['relative'].startswith('code/') else None
    hist='experiment/'+item['relative'][5:] if item['relative'].startswith('code/') else 'experiment/'+item['relative']
    item['historical_inventory_sha256']=expected.get(hist)
    item['matches_historical_inventory']=item['sha256']==expected.get(hist) if hist in expected else None
    item['local_sha256']=hashlib.sha256(local.read_bytes()).hexdigest() if local and local.exists() else None
    evidence.append(item)
sacct=subprocess.run(['ssh','-o','BatchMode=yes','eden-cluster','sacct -j 1824873,1824877,1824878,1824879,1824880,1824979,1824980 -n -P --format=JobID,State,ExitCode,Elapsed,Start,End'],capture_output=True,text=True,encoding='utf-8',timeout=30)
(OUT/'SACCT.txt').write_text(sacct.stdout,encoding='utf-8')
payload={'cutoff_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':evidence,'sacct_exitcode':sacct.returncode}
(OUT/'PROVENANCE_ODCZYTU.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('snapshots',len(evidence),'historical matches',sum(x['matches_historical_inventory'] is True for x in evidence))
