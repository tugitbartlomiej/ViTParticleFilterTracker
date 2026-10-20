"""Count same-token opposite beam pairs at continuation checkpoints, epochs 60-200."""
from pathlib import Path
import ast,csv,json,sys,time
import numpy as np

EXP=Path(__file__).resolve().parents[1]
CFG=json.loads((EXP/'Kod_eksperymentu/konfiguracja.json').read_text(encoding='utf-8'))
BASE=Path(CFG['base_eden'])
sys.path.insert(0,str(BASE/CFG['measurement_code_relative']))
import common as C
import torch

OUT=EXP/'Wyniki/Dane/pary_epoki_60_200'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'states').mkdir(exist_ok=True)
THRESHOLDS=CFG['thresholds_deg']
II,JJ=np.triu_indices(768,1)
def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def parent_count():
    # Exact count() of the parent experiment, verified by SHA256, not a reimplementation.
    source=BASE/CFG['count_function_source_relative']
    assert C.sha(source)==CFG['count_function_source_sha256'],'Parent count code changed'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='count')
    scope=dict(np=np,II=II,JJ=JJ,THRESHOLDS=THRESHOLDS)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
    return scope['count'],str(source)

def compare_with_parent(records,states):
    # Step 65640 is the parent's final checkpoint: weights, masks and counts must agree.
    prev=BASE/CFG['parent_counts_65640_relative']
    done=json.loads((prev/'DONE.json').read_text())
    assert C.sha(prev/'COUNTS.json')==done['files']['COUNTS.json']
    old={(r['block'],r['threshold_deg']):r for r in json.loads((prev/'COUNTS.json').read_text()) if r['step']==65640}
    new={(r['block'],r['threshold_deg']):r for r in records if r['step']==65640}
    assert old.keys()==new.keys()
    int_keys=[k for k in next(iter(old.values())) if isinstance(next(iter(old.values()))[k],int)]
    count_mismatch=[dict(key=list(k),field=f,parent=old[k][f],now=new[k][f]) for k in old for f in int_keys if old[k][f]!=new[k][f]]
    angle_diff=max(abs(old[k][f]-new[k][f]) for k in old for f in ['maximum_weight_angle','maximum_possible_signed_angle','maximum_observed_signed_angle'])
    masks={}
    for b in range(1,7):
        with np.load(prev/f'b{b}_k65640.npz') as f:
            assert np.array_equal(f['V'],states[b]['V']),f'Block {b} weights differ from parent'
            masks[b]={k:int((np.unpackbits(f[k],axis=-1,count=768).astype(bool).reshape(-1,768)!=states[b][s]).sum()) for k,s in [('positive','pos'),('nonzero','nz'),('active_0_05','active')]}
    return dict(status='PASS' if not count_mismatch else 'COUNT_MISMATCH',weights_equal=True,
        count_mismatches=count_mismatch,max_angle_abs_diff_deg=angle_diff,mask_differences=masks,
        parent_counts_sha256=C.sha(prev/'COUNTS.json'))

def main():
    if (OUT/'DONE.json').exists():raise SystemExit('Existing completed counts preserved')
    start=time.monotonic();net=C.setup()
    assert C.sha(Path(C.__file__))==CFG['measurement_common_sha256']
    assert C.sha(Path(C.M.__file__))==CFG['model_sha256']
    assert C.sha(C.ROOT/'config.json')==CFG['frames_config_sha256']
    rows=C.CFG['rows'];assert len(rows)==CFG['frames']==50
    count,count_source=parent_count()
    images=[C.image(r) for r in rows]
    cache={};handles=[]
    for b,block in enumerate(net.bloki):handles.append(block.mlp.act.register_forward_hook(lambda m,a,o,b=b:cache.update({b:o.detach()})))
    inputs={count_source:CFG['count_function_source_sha256']};records=[];pairs=[];errors=[];qc=None
    for ck in CFG['checkpoints']:
        path=BASE/CFG['checkpoint_dir_relative']/ck['file'];step=ck['step']
        h=C.sha(path);assert h==ck['sha256'],'Checkpoint changed: '+ck['file'];inputs[str(path)]=h
        # Own continuation checkpoint with pinned SHA256; it stores RNG objects, so weights_only=False.
        st=torch.load(path,map_location='cpu',weights_only=False)
        assert st['krok']==step and st['seed']==3047
        net.load_state_dict(st['net'],strict=True);net.eval();del st
        zs={b:[] for b in range(6)}
        with torch.inference_mode():
            for j in range(0,50,CFG['batch']):
                C.forward(net,C.tensor(images[j:j+CFG['batch']]))
                for b in range(6):zs[b].append(cache[b].cpu().numpy().copy())
        states={}
        for b in range(1,7):
            z=np.concatenate(zs.pop(b-1));v=net.bloki[b-1].mlp.fc2.weight.detach().cpu().numpy().T.copy()
            strength=abs(z)*np.linalg.norm(v,axis=1);nz=strength>CFG['nonzero_strength_threshold']
            active=strength>CFG['active_relative_threshold']*strength.max(-1,keepdims=True)
            state=dict(V=v,pos=(z>0)&nz,nz=nz,active=active)
            np.savez_compressed(OUT/f'states/b{b}_k{step:06d}.npz',V=v,positive=np.packbits(state['pos'],axis=-1),
                nonzero=np.packbits(nz,axis=-1),active_0_05=np.packbits(active,axis=-1))
            rr,pp,e=count(state,b,step);records.extend(rr);pairs.extend(pp);errors.append(e)
            if step==65640:states[b]=dict(V=v,pos=state['pos'].reshape(-1,768),nz=nz.reshape(-1,768),active=active.reshape(-1,768))
        if step==65640:qc=compare_with_parent([r for r in records if r['step']==65640],states);save('QC_65640_VS_PARENT.json',qc)
        print('STEP_DONE',step,round(step*32/35000,3),round(time.monotonic()-start,1),flush=True)
    for h in handles:h.remove()
    for r in records:r['epoch_equivalent']=r['step']*32/35000
    records.sort(key=lambda r:(r['step'],r['block'],r['threshold_deg']))
    assert len(records)==len(CFG['checkpoints'])*6*len(THRESHOLDS)
    csvwrite('COUNTS_60_200.csv',records);save('COUNTS_60_200.json',records)
    if pairs:csvwrite('CANDIDATE_PAIRS_60_200.csv',pairs)
    save('PROVENANCE.json',dict(status='PASS',qc_65640=qc['status'],steps=[c['step'] for c in CFG['checkpoints']],
        epochs=[c['step']*32/35000 for c in CFG['checkpoints']],thresholds=THRESHOLDS,seed=3047,frames=50,
        tokens_per_frame=600,beams_per_block=768,pairs_per_block=294528,dimension=192,
        criterion='Unordered neuron ID pair counted once per block and checkpoint if its signed 192D contributions reach the threshold for at least one SAME token among 50 frozen TEST frames (parent count() reused verbatim).',
        inference='eval, FP32, TF32 off, deterministic algorithms; no training',
        inputs=inputs,script_sha256=C.sha(Path(__file__)),config_sha256=C.sha(EXP/'Kod_eksperymentu/konfiguracja.json'),
        frames_config_sha256=CFG['frames_config_sha256'],environment=C.environment(),
        direct_dot_check_max_error=max(errors),seconds=time.monotonic()-start))
    save('DONE.json',dict(status='PASS',files={str(p.relative_to(OUT)):C.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='DONE.json'}))
    print('COMPLETE',len(records),round(time.monotonic()-start,2),flush=True)
if __name__=='__main__':main()
