"""Count the same beam-pair criterion at every saved historical checkpoint."""
from pathlib import Path
import ast,json,hashlib,csv,time
import numpy as np

EXP=Path(__file__).resolve().parents[2]
HISTORY=Path('/mnt/evafs/groups/transformers_vsc/bpiotrowski/ViTParticleFilterTracker/SmallViTAblation/outputs/beam_survival_angles_training_20261020/run_v1')
PREV=EXP/'Wyniki/Dane/opposite_pair_counts_20261020__18-52-50'
OUT=EXP/'Wyniki/Dane/opposite_pair_counts_by_epoch'
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
if (OUT/'DONE.json').exists():raise SystemExit('Existing completed counts preserved')
start=time.monotonic()
source=EXP/'Kod_eksperymentu/count_opposite_pairs.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='count')
ii,jj=np.triu_indices(768,1);thresholds=[150,160,170,175,179,179.999999]
scope=dict(np=np,II=ii,JJ=jj,THRESHOLDS=thresholds)
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
freeze=json.loads((HISTORY/'EXTRACTION_DONE.json').read_text())['files']
old_done=json.loads((PREV/'DONE.json').read_text())
assert sha(PREV/'COUNTS.json')==old_done['files']['COUNTS.json']
records=json.loads((PREV/'COUNTS.json').read_text())
steps=[0,300,1000,3000,6000,10000,16000,24000,33000,44000,55000,65640]
inputs={str(source):sha(source),str(PREV/'COUNTS.json'):sha(PREV/'COUNTS.json')};errors=[];pairs=[]
for step in steps:
    if step in [0,55000,65640]:continue
    for block in range(1,7):
        path=HISTORY/f'states/b{block}_k{step:05d}.npz'
        h=sha(path);assert h==freeze[str(path.relative_to(HISTORY))];inputs[str(path)]=h
        with np.load(path) as f:
            unpack=lambda k:np.unpackbits(f[k],axis=-1,count=768).astype(bool)
            state=dict(V=f['V'],pos=unpack('positive'),nz=unpack('nonzero'),active=unpack('active_0.05'))
        rr,pp,error=scope['count'](state,block,step)
        records.extend(rr);pairs.extend(pp);errors.append(error)
for r in records:r['epoch_equivalent']=r['step']*32/35000
records.sort(key=lambda r:(r['step'],r['block'],r['threshold_deg']))
assert len(records)==12*6*6
assert len({(r['step'],r['block'],r['threshold_deg']) for r in records})==len(records)
assert all(r['step'] in steps for r in records)
with (OUT/'COUNTS_BY_EPOCH.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
save('COUNTS_BY_EPOCH.json',records)
if pairs:
    with (OUT/'INTERMEDIATE_PAIRS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
save('PROVENANCE.json',dict(status='PASS',steps=steps,epochs=[s*32/35000 for s in steps],thresholds=thresholds,
    seed=3047,pairs_per_block=294528,frames=50,tokens_per_frame=600,dimension=192,
    criterion='Same signed 192D individual contributions for the SAME token; unique neuron pair counted once if any token reaches threshold.',
    historical_sampling='12 saved checkpoints, not one checkpoint per epoch; no interpolated measurements.',
    endpoint_reuse='Verified counts for steps0,55000,65640 reused; remaining nine checkpoints newly counted.',
    direct_dot_max_error=max(errors),inputs=inputs,seconds=time.monotonic()-start))
save('DONE.json',dict(status='PASS',files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
print('COMPLETE',len(records),round(time.monotonic()-start,2),flush=True)
