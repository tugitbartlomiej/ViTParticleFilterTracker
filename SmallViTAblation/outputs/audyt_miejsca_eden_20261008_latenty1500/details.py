import json,subprocess
from pathlib import Path
OUT=Path(__file__).resolve().parent
d=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
files=d['files']; paths=[]
interesting=['2026-09-30_dyfuzja_sd_filmy','2026-10-07_maly_vit_sily_czastek','2026-10-06_vit_sd_pelny_latent_filmy','2026-10-06_vit_sd_od_zera_k8000_filmy','2026-10-06_vit_konwerter_dino_filmy','2026-10-02_nndmd_1500_expandedstate_multigpu','2026-10-08_vit_k8000_podmiana_1z256']
for f in files:
 if not any(x in f['path'] for x in interesting):continue
 name=Path(f['path']).name
 if name in ['run.py','trening.py','bieg.py','train.py','dane.py','FINAL.json','STATUS.json','PROVENANCE.json','SMOKE.json','RESUME.json','CONFIG.json','PODSUMOWANIE.json','DANE.json'] or f['path'].endswith('.slurm') or (name.endswith('.out') and any(x in f['path'] for x in ['1823172','1823302','1823430','1819574','1823539'])):
  if f['size']<1_000_000:paths.append(f['path'])
checkpoints=[f['path'] for f in files if (f['size']>10_000_000 and any(x in f['path'] for x in ['ostatni.pt','latest_checkpoint.pt','/smoke/checkpoint_','/smoke_v2/ckpt_','/smoke/ckpt_','/pelny/k1000.pt']))]
code='paths='+repr(paths)+'\ncheckpoints='+repr(checkpoints)+r'''
import os,json,zipfile,pickletools
texts=[];keys=[]
for p in paths:
 try:
  with open(p,encoding='utf-8',errors='replace') as f:t=f.read()
  texts.append(dict(path=p,text=t if not p.endswith('.out') else t[-18000:]))
 except Exception as e:texts.append(dict(path=p,error=str(e)))
for p in checkpoints:
 try:
  with zipfile.ZipFile(p) as z:
   names=[i for i in z.infolist() if i.filename.endswith('data.pkl')]
   if not names or names[0].file_size>8_000_000:continue
   strings=[]
   for op,arg,pos in pickletools.genops(z.read(names[0])):
    if op.name in ['BINUNICODE','SHORT_BINUNICODE','UNICODE'] and isinstance(arg,str):strings.append(arg)
   keys.append(dict(path=p,pickle_strings=list(dict.fromkeys(strings)),zip_members=len(z.infolist())))
 except Exception as e:keys.append(dict(path=p,error=str(e)))
print(json.dumps(dict(texts=texts,checkpoint_keys=keys)))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','eden-cluster','python3 -'],input=code,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
(OUT/'details.stderr.txt').write_text(r.stderr,encoding='utf-8')
if r.returncode:raise RuntimeError(r.stderr)
(OUT/'details.json').write_text(json.dumps(json.loads(r.stdout),ensure_ascii=False,indent=2),encoding='utf-8')
print('texts',len(paths),'checkpoint inspections',len(checkpoints))
# Evidence outside selected runs: only own home top level, no recursion.
cmd='date -Is; ls -la /home2/faculty/bpiotrowski; du -x -B1 --max-depth=1 /home2/faculty/bpiotrowski/smallvit_runs; du -x -B1 --max-depth=1 /home2/faculty/bpiotrowski/.cache; ps -u bpiotrowski -o pid,comm; squeue -u bpiotrowski -o "%.18i %.12T %.50j %.25R"; df -B1 /home2/faculty/bpiotrowski/smallvit_runs; lfs quota -u bpiotrowski /mnt/evafs'
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','eden-cluster',cmd],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
(OUT/'sizes_final.txt').write_text(r.stdout,encoding='utf-8');(OUT/'sizes_final.stderr.txt').write_text(r.stderr,encoding='utf-8')
print(r.stdout)
