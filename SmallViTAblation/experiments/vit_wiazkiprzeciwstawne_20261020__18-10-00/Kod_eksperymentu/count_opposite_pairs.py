"""Count individual same-token beam pairs in full 192D, at frozen endpoints."""
from pathlib import Path
import sys, os, json, csv, hashlib, time
import numpy as np

BASE = Path('/mnt/evafs/groups/transformers_vsc/bpiotrowski/ViTParticleFilterTracker/SmallViTAblation')
HISTORY = BASE/'outputs/beam_survival_angles_training_20261020/run_v1'
sys.path.insert(0, str(BASE/'experiments/beam_survival_angles_training_20261020/run_v1'))
import common as C
import torch

EXP = Path(__file__).resolve().parents[1]
OUT = EXP/'Wyniki/Dane/opposite_pair_counts_20261020__18-52-50'
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [150, 160, 170, 175, 179, 179.999999]
II, JJ = np.triu_indices(768, 1)
def unpack(x): return np.unpackbits(x, axis=-1, count=768).astype(bool)
def save(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def readstate(block, step):
    p=HISTORY/f'states/b{block}_k{step:05d}.npz'
    with np.load(p) as f:
        return dict(V=f['V'].copy(), pos=unpack(f['positive']), nz=unpack(f['nonzero']), active=unpack(f['active_0.05'])), str(p), C.sha(p)

def count(state, block, step):
    v=state['V'].astype(np.float64)
    norms=np.linalg.norm(v,axis=1)
    valid=(norms[II]>1e-12)&(norms[JJ]>1e-12)
    u=v/np.maximum(norms[:,None],1e-30)
    angle=np.rad2deg(np.arccos(np.clip((u@u.T)[II,JJ],-1,1)))
    bound=np.maximum(angle,180-angle)
    pos=state['pos'].reshape(-1,768);nz=state['nz'].reshape(-1,768)
    active=state['active'].reshape(-1,768)&nz
    result=[];pairrows=[];cache={}
    def observe(k):
        k=int(k)
        if k not in cache:
            a,b=int(II[k]),int(JJ[k]);co=nz[:,a]&nz[:,b];same=pos[:,a]==pos[:,b]
            act=active[:,a]&active[:,b]
            cache[k]=(int((co&same).sum()),int((co&~same).sum()),int((act&same).sum()),int((act&~same).sum()))
        return cache[k]
    maximum=-1.; maxpair=None
    for k in np.argsort(-bound,kind='stable'):
        if not valid[k]:continue
        if bound[k]<=maximum:break
        ns,no,_,_=observe(k)
        actual=max(float(angle[k]) if ns else -1.,float(180-angle[k]) if no else -1.)
        if actual>maximum:maximum=actual;maxpair=[int(II[k]),int(JJ[k])]
    candidates=np.flatnonzero(valid&(bound>=min(THRESHOLDS)))
    for k in candidates:
        ns,no,acs,aco=observe(k);theta=float(angle[k])
        row=dict(step=step,block=block,neuron_a=int(II[k]),neuron_b=int(JJ[k]),weight_angle=theta,
                 maximum_possible_signed_angle=float(bound[k]),same_sign_nonzero_tokens=ns,opposite_sign_nonzero_tokens=no,
                 same_sign_active05_tokens=acs,opposite_sign_active05_tokens=aco)
        for th in THRESHOLDS:
            row[f'count_ge_{th}']=int(ns*(theta>=th)+no*(180-theta>=th))
            row[f'active05_count_ge_{th}']=int(acs*(theta>=th)+aco*(180-theta>=th))
        pairrows.append(row)
    for th in THRESHOLDS:
        result.append(dict(step=step,block=block,threshold_deg=th,all_unordered_pairs=int(valid.sum()),
            same_token_observed_unique_pairs=sum(r[f'count_ge_{th}']>0 for r in pairrows),
            same_token_observed_pair_token_count=sum(r[f'count_ge_{th}'] for r in pairrows),
            same_token_active05_unique_pairs=sum(r[f'active05_count_ge_{th}']>0 for r in pairrows),
            same_token_active05_pair_token_count=sum(r[f'active05_count_ge_{th}'] for r in pairrows),
            weight_only_unique_pairs=int((valid&(angle>=th)).sum()),
            possible_signed_unique_pairs=int((valid&(bound>=th)).sum()),
            maximum_weight_angle=float(angle[valid].max()),maximum_possible_signed_angle=float(bound[valid].max()),
            maximum_observed_signed_angle=maximum,maximum_observed_pair=maxpair))
    assert all(r['same_token_active05_unique_pairs']<=r['same_token_observed_unique_pairs']<=r['possible_signed_unique_pairs'] for r in result)
    # Independent vector-dot checks for high-angle and ordinary neuron pairs.
    errors=[]
    for k in list(candidates[:5])+[0,101,10001]:
        a,b=II[k],JJ[k];tokens=np.flatnonzero(nz[:,a]&nz[:,b])[:7]
        for t in tokens:
            va=u[a]*(1 if pos[t,a] else -1);vb=u[b]*(1 if pos[t,b] else -1)
            direct=np.rad2deg(np.arccos(np.clip(np.dot(va,vb),-1,1)))
            formula=angle[k] if pos[t,a]==pos[t,b] else 180-angle[k]
            errors.append(abs(float(direct-formula)))
    assert max(errors,default=0)<1e-9
    print('COUNTED',step,block,'unique150',result[0]['same_token_observed_unique_pairs'],'max',round(maximum,6),flush=True)
    return result,pairrows,max(errors,default=0)

def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    start=time.monotonic();net=C.setup();rows=C.CFG['rows'];assert len(rows)==50
    inputs={};records=[];pairs=[];errors=[]
    for step in [0,55000]:
        C.load(net,step);inputs[str(C.weight_path(step))]=C.sha(C.weight_path(step))
        for b in range(1,7):
            state,p,h=readstate(b,step);inputs[p]=h
            assert np.array_equal(state['V'],net.bloki[b-1].mlp.fc2.weight.detach().cpu().numpy().T)
            rr,pp,e=count(state,b,step);records.extend(rr);pairs.extend(pp);errors.append(e)
    step=65640;C.load(net,step);inputs[str(C.weight_path(step))]=C.sha(C.weight_path(step))
    images=[C.image(r) for r in rows];cache={};handles=[];zs={b:[] for b in range(6)}
    for b,block in enumerate(net.bloki):handles.append(block.mlp.act.register_forward_hook(lambda m,a,o,b=b:cache.update({b:o.detach()})))
    with torch.inference_mode():
        for j in range(0,50,10):
            C.forward(net,C.tensor(images[j:j+10]))
            for b in range(6):zs[b].append(cache[b].cpu().numpy().copy())
    for h in handles:h.remove()
    for b in range(1,7):
        z=np.concatenate(zs.pop(b-1));v=net.bloki[b-1].mlp.fc2.weight.detach().cpu().numpy().T.copy()
        strength=abs(z)*np.linalg.norm(v,axis=1);nz=strength>1e-12;active=strength>.05*strength.max(-1,keepdims=True)
        state=dict(V=v,pos=(z>0)&nz,nz=nz,active=active)
        np.savez_compressed(OUT/f'b{b}_k65640.npz',V=v,positive=np.packbits(state['pos'],axis=-1),
            nonzero=np.packbits(nz,axis=-1),active_0_05=np.packbits(active,axis=-1))
        rr,pp,e=count(state,b,step);records.extend(rr);pairs.extend(pp);errors.append(e)
    csvwrite('COUNTS.csv',records);csvwrite('CANDIDATE_PAIRS.csv',pairs)
    save('COUNTS.json',records)
    save('PROVENANCE.json',dict(status='PASS',steps=[0,55000,65640],thresholds=THRESHOLDS,
        seed=3047,frames=50,films=25,tokens_per_frame=600,beams_per_block=768,pairs_per_block=294528,
        dimension=192,best_step=55000,final_step=65640,nonzero_strength_threshold=1e-12,active_relative_threshold=.05,
        pair_definition='Unordered neuron ID pair, counted once per block and checkpoint if its signed 192D contributions reach threshold for at least one SAME token among 50 frozen TEST frames. No mixing tokens or images.',
        possible_definition='Maximum over the two possible relative activation signs; this is a global weight-based bound independent of image.',
        exclusion='A pair with a zero contribution has undefined angle and is excluded for that token.',
        inputs=inputs,frames_manifest_sha256=C.sha(HISTORY/'config.json'),environment=C.environment(),
        direct_dot_check_max_error=max(errors),seconds=time.monotonic()-start))
    save('DONE.json',dict(status='PASS',files={p.name:C.sha(p) for p in OUT.iterdir() if p.is_file()}))
    print('COMPLETE',round(time.monotonic()-start,2),flush=True)
if __name__=='__main__':main()
