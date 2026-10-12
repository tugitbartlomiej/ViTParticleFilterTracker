"""Niezależnie sprawdza zakończone sekwencje, jawny cel i seals prognoz.
Niezgodna trajektoria lub wcześniejsza decyzja unieważnia weryfikację.
"""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import torch
OUT=Path(__file__).resolve().parents[2]/'outputs/vit_langevin_planowanie_paczek_sd'
r=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'))
assert r['status']=='complete' and r['CAV_used'] is False and r['original_SD_checkpoint']
assert r['total_updates']==1500 and r['total_training_images']==48000
with np.load(OUT/'BASELINE.npz') as b:
 target=b['target'].copy();J=b['channels0'].copy();other=b['other640'].copy() if 'other640' in b else np.array([i for i in range(768) if i not in set(J)])
 initial=b['before'].copy()
check={}
for policy in ['controlled','random_choice','balanced_cycle']:
 data=json.loads((OUT/policy/'ACTUAL.json').read_text(encoding='utf-8'))
 lat=np.load(OUT/policy/'TRAJECTORY.npy',mmap_mode='r')
 rows=list(csv.DictReader((OUT/policy/'SCHEDULE.csv').open(encoding='utf-8')))
 seals=[json.loads(x) for x in (OUT/policy/'FORECAST_SEALS.jsonl').read_text().splitlines()]
 assert len(data)==501 and lat.shape==(501,600,768) and len(rows)==len(seals)==500
 assert np.array_equal(lat[0],initial)
 for step in range(501):
  for dim,ix in [('128',J),('640',other),('768',np.arange(768))]:
   mse=float(np.mean((lat[step][:,ix].astype(np.float64)-target[:,ix])**2))
   assert np.isclose(mse,data[step]['mse'+dim],rtol=1e-12,atol=1e-12)
  if step==0:continue
  row=rows[step-1];path=OUT/policy/'decisions'/f'{step:04d}.json'
  d=json.loads(path.read_text(encoding='utf-8'));h=hashlib.sha256(path.read_bytes()).hexdigest()
  assert h==row['seal_sha256']==seals[step-1]['sha256']
  assert d['forecast_origin_step']==step-1 and d['checkpoint_step_before']==1000+step-1
  assert not d['future_information_used'] and d['forecast_horizon']==1
  assert d['selected_arm']==row['arm']==data[step]['arm']
  if policy=='controlled':
   assert len(d['candidate_predictions'])==4
   assert d['selected_arm']==min(d['candidate_predictions'],key=lambda x:x['predicted']['mse128'])['arm']
  if policy=='balanced_cycle':assert d['selected_arm']==str([2026100511,2026100523,2026100537,2026100549][(step-1)%4])
 beststep=min(range(501),key=lambda i:data[i]['mse128'])
 assert beststep==r['policies'][policy]['best']['step']
 cp=torch.load(OUT/f'checkpoints/{policy}_k1500.pt',map_location='cpu',weights_only=True)
 assert cp['krok']==1500 and {float(s['step']) for s in cp['opt']['state'].values()}=={1500.0}
 del cp
 sequence=json.loads((OUT/policy/'BEST_SEQUENCE.json').read_text(encoding='utf-8'))
 assert sequence['sequence']==[x['arm'] for x in rows[:beststep]]
 for forcepath in (OUT/policy/'forces').glob('*.npz'):
  i=int(forcepath.stem)
  with np.load(forcepath) as f:
   assert np.array_equal(f['before'],lat[i-1]) and np.array_equal(f['new']+f['momentum'],f['total'])
   assert np.array_equal(f['before']+f['total'],f['predicted'])
   d=json.loads((OUT/policy/'decisions'/f'{i:04d}.json').read_text(encoding='utf-8'))
   assert hashlib.sha256(f['total'].tobytes()).hexdigest()==d['selected_force_sha256']
 check[policy]=dict(all501_actual_latents_verified=True,all500_decision_seals_and_origins_verified=True,
  checkpoint_optimizer_step=1500,best_step=beststep)
 print(policy,check[policy])
assert r['cost']['candidate_gradient_batches']==3004 and r['cost']['candidate_probe_forwards']==6008
(OUT/'FINAL_AUDIT.json').write_text(json.dumps(dict(status='passed',policies=check,cost_verified=True),indent=2),encoding='utf-8')
print('FINAL AUDIT PASSED')
