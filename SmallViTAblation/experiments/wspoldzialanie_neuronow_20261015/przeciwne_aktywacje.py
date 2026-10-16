"""Signed co-activation of opposed MLP columns; read-only application/checkpoints."""
from pathlib import Path
import json, time, hashlib, sys
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
threadpool_limits(4)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje'
NAMES=['eafb5a973be3d490','c584ef4647149c7c','2cca2743e1c74b14']

def masks(gt):
    p=np.stack([(np.arange(600)%30+.5)*16,(np.arange(600)//30+.5)*16],1)
    inside=((p>=gt[:2])&(p<=gt[2:])).all(1)
    intersects=((p+8>gt[:2])&(p-8<gt[2:])).all(1)
    contained=((p-8>=gt[:2])&(p+8<=gt[2:])).all(1)
    far=np.linalg.norm(np.maximum(np.maximum(gt[:2]-p,p-gt[2:]),0),axis=1)>32
    return dict(all=np.ones(600,bool),bbox=inside,boundary=intersects&~contained,far=far,interior=contained)

def loadv(wd,k):
    return torch.load(wd/f'k{k:05d}.pt',map_location='cpu',weights_only=True)['net']['bloki.5.mlp.fc2.weight'].T.numpy()

def pairs(v):
    vn=v/np.linalg.norm(v,axis=1)[:,None]
    return np.stack(np.where(np.triu(vn@vn.T<-.5,1)),1)

def controls(v, pp,seed):
    rng=np.random.default_rng(seed);norm=np.linalg.norm(v,axis=1);vn=v/norm[:,None];out=[]
    used=set()
    for c,d in pp:
        order=np.argsort(abs(np.log(norm/norm[d])))
        eligible=[int(q) for q in order if q!=c and q!=d and vn[c]@vn[q]>=-.5 and tuple(sorted((int(c),int(q)))) not in used]
        q=int(rng.choice(eligible[:20])); used.add(tuple(sorted((int(c),q))));out.append((c,q))
    return np.array(out)

def measures(z,v,pp,threshold):
    c,d=pp[:,0],pp[:,1];a=z[:,c];b=z[:,d]
    n=v.norm(dim=1);dot=(v[c]*v[d]).sum(-1)
    ac=a.abs()*n[c];bd=b.abs()*n[d];s=ac+bd
    cross=2*a*b*dot
    length=(ac.square()+bd.square()+cross).clamp_min(0).sqrt()
    activea=a.abs()>threshold[c];activeb=b.abs()>threshold[d]
    # Rows token, columns pair. No cancellation when contributions point same way.
    return dict(strength=s,cancel=s-length,balance=2*torch.minimum(ac,bd),
                destructive=s*(cross<0),same_sign=s*(a*b>0),
                both=(activea&activeb).float(),either=(activea|activeb).float(),
                only=(activea^activeb).float(),neither=(~activea&~activeb).float(),
                both_strength=s*(activea&activeb),energy=ac.square()+bd.square(),cross=cross)

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    hh=[np.load(ROOT/'app/cache_uczenia/lasery'/n/'dane.npz') for n in NAMES]
    orig=hh[0]['kroki'];steps=np.arange(1000,65601,100);ix=np.searchsorted(orig,steps)
    assert np.array_equal(orig[ix],steps)
    for h in hh:assert np.array_equal(h['kroki'],orig)
    wd=Path(hh[0]['wagi'].item());vfinal=loadv(wd,65640);vearly=loadv(wd,33000)
    pfinal=pairs(vfinal);pearly=pairs(vearly);assert len(pfinal)==844
    pc=controls(vearly,pearly,15102026)
    sets=dict(final=pfinal,early=pearly,control=pc)
    np.savez(OUT/'pary.npz',**sets,v_early=vearly,v_final=vfinal)
    mm=[masks(h['gt']) for h in hh]
    for i in range(3):assert np.array_equal(mm[i]['bbox'],hh[i]['narz'])
    # A fixed within-group token permutation preserves each marginal distribution.
    rng=np.random.default_rng(262015);perms=[{g:rng.permutation(np.flatnonzero(m)) for g,m in md.items()} for md in mm]
    meta=dict(images=NAMES,steps=steps.tolist(),groups=[{g:int(m.sum()) for g,m in md.items()} for md in mm],
              gt=[h['gt'].tolist() for h in hh],pair_counts={k:len(p) for k,p in sets.items()},seed=15102026,
              rgb_sha256=[hashlib.sha256(h['rgb'].tobytes()).hexdigest() for h in hh])
    old=np.load(ROOT/'outputs/optyka_hipoteza_20261015/pomiar.npz');ids=old['ids']
    cache=[np.load(ROOT/'app/cache_uczenia/lasery'/n/'Z.npy',mmap_mode='r') for n in NAMES]
    vs=np.lib.format.open_memmap(OUT/'V.npy',mode='w+',dtype=np.float32,shape=(len(steps),768,192))
    zs=np.lib.format.open_memmap(OUT/'Z_wybrane.npy',mode='w+',dtype=np.float32,shape=(len(steps),3,24,768))
    np.savez(OUT/'indeksy.npz',steps=steps,original_index=ix,ids=ids,
             bbox=np.stack([m['bbox'][ids[i]] for i,m in enumerate(mm)]),
             far=np.stack([m['far'][ids[i]] for i,m in enumerate(mm)]))
    records=[];late={};select=np.zeros((3,2,768),np.float64);select_n=0;qc=[];started=time.time()
    for ti,(k,oi) in enumerate(zip(steps,ix)):
        vv=loadv(wd,int(k));vs[ti]=vv;v=torch.from_numpy(vv.copy()).to(dev)
        zall=np.stack([np.asarray(cache[b][oi],np.float32) for b in range(3)])
        for b in range(3):zs[ti,b]=zall[b,ids[b]]
        zall=torch.from_numpy(zall).to(dev)
        current=pairs(vv);localsets={**sets,'current':current}
        for b in range(3):
            z=zall[b];threshold=.1*torch.quantile(z.abs(),.95,dim=0)
            if k>=55000:
                for gi,g in enumerate(['bbox','far']):
                    select[b,gi]+=(z[mm[b][g]].abs()*v.norm(dim=1)).mean(0).cpu().numpy()
            for name,pp in localsets.items():
                if len(pp)==0:continue
                pt=torch.as_tensor(pp,device=dev);m=measures(z,v,pt,threshold)
                for g,mask in mm[b].items():
                    if not mask.any():continue
                    ids_g=np.flatnonzero(mask);mg={key:value[ids_g] for key,value in m.items()}
                    strength=mg['strength'].sum();den=strength.clamp_min(1e-20)
                    r=dict(step=int(k),epoch=k/1094,image=b+1,pair_set=name,group=g,n_tokens=len(ids_g),n_pairs=len(pp))
                    vals=[mg[key].sum()/den for key in ['cancel','balance','destructive','same_sign','both_strength']]
                    vals.extend([mg[key].mean() for key in ['both','either','only','neither']])
                    vals.extend([mg['cross'].sum()/mg['energy'].sum().clamp_min(1e-20),strength/len(ids_g)/len(pp)])
                    names=['cancellation','balance','destructive_weight','same_sign_weight','both_weight','both','either','only','neither','cross_energy_ratio','mean_strength']
                    r.update(zip(names,torch.stack(vals).cpu().numpy().tolist()))
                    # Shuffle partner d within this same spatial group.
                    # Need unpermuted c also when a neuron is in both roles, so explicit pair expression.
                    c,d=pt[:,0],pt[:,1];aa=z[ids_g][:,c];bb=z[perms[b][g]][:,d];nn=v.norm(dim=1)
                    ac=aa.abs()*nn[c];bd=bb.abs()*nn[d];ss=ac+bd
                    ll=(ac.square()+bd.square()+2*aa*bb*(v[c]*v[d]).sum(-1)).clamp_min(0).sqrt()
                    r['cancellation_shuffle']=float(((ss-ll).sum()/ss.sum().clamp_min(1e-20)).cpu())
                    records.append(r)
                    if k>=55000 and name in sets and g in ['bbox','far']:
                        key=(name,b,g)
                        agg=torch.stack([mg[q].sum(0) for q in ['strength','cancel','balance','destructive','both','either','only']]).cpu().numpy().astype(float)
                        late[key]=late.get(key,0)+agg
            if k>=55000 and ti%10==0:
                pt=torch.as_tensor(pfinal,device=dev)
                for th in [.05,.2]:
                    met=measures(z,v,pt,threshold*(th/.1))
                    for g in ['bbox','far']:
                        ma=mm[b][g];both=met['both'][ma].mean();either=met['either'][ma].mean()
                        qc.append(dict(kind='threshold',step=int(k),image=b+1,group=g,threshold=th,both=float(both),either=float(either)))
        if k>=55000:select_n+=1
        if ti%100==0 or ti==len(steps)-1:
            print(f'{ti+1}/{len(steps)}; {time.time()-started:.1f}s',flush=True)
            c,d=pfinal[0];zt=zall[0,0];direct=torch.linalg.vector_norm(zt[c]*v[c]+zt[d]*v[d])
            formula=((zt[c]*v[c]).square().sum()+(zt[d]*v[d]).square().sum()+2*zt[c]*zt[d]*(v[c]*v[d]).sum()).sqrt()
            qc.append(dict(kind='norm_identity',step=int(k),absolute_error=float(abs(direct-formula))))
    pd.DataFrame(records).to_csv(OUT/'historia.csv',index=False)
    pairrows=[]
    for (name,b,g),a in late.items():
        for j,(c,d) in enumerate(sets[name]):
            pairrows.append(dict(pair_set=name,image=b+1,group=g,c=int(c),d=int(d),
                cancellation=a[1,j]/max(a[0,j],1e-20),balance=a[2,j]/max(a[0,j],1e-20),
                destructive_weight=a[3,j]/max(a[0,j],1e-20),both_given_either=a[4,j]/max(a[5,j],1e-20),
                strength=a[0,j],only_given_either=a[6,j]/max(a[5,j],1e-20)))
    pd.DataFrame(pairrows).to_csv(OUT/'pary_pozny_trening.csv',index=False)
    neurons=[]
    for b in range(3):
        for c in range(768):neurons.append(dict(image=b+1,neuron=c,bbox_mean_strength=select[b,0,c]/select_n,
            far_mean_strength=select[b,1,c]/select_n,bbox_far_ratio=select[b,0,c]/max(select[b,1,c],1e-20)))
    pd.DataFrame(neurons).to_csv(OUT/'selektywnosc_neuronow.csv',index=False)
    vs.flush();zs.flush()
    # Float16 cache independently checked against float32 activation reconstruction.
    sys.path.insert(0,str(ROOT/'app'));import lasery_dane as LD
    net=LD.modul_modelu().MalyViT().eval();pre=old['x_pre']
    for k in [1000,33000,65600]:
        oi=int(np.where(orig==k)[0][0]);net.load_state_dict(torch.load(wd/f'k{k:05d}.pt',weights_only=True,map_location='cpu')['net'])
        with torch.no_grad():
            x=torch.from_numpy(pre[oi]);mlp=net.bloki[-1];actual=mlp.mlp.act(mlp.mlp.fc1(mlp.norm2(x))).numpy()
        cached=np.asarray(zs[np.where(steps==k)[0][0]])
        qc.append(dict(kind='float16',step=k,relative_rms=float(np.linalg.norm(actual-cached)/np.linalg.norm(actual)),max_abs=float(abs(actual-cached).max())))
    meta['elapsed_s']=time.time()-started;meta['qc']=qc
    meta['control_norm_ratio_median']=float(np.median(np.linalg.norm(vearly[pc[:,1]],axis=1)/np.linalg.norm(vearly[pearly[:,1]],axis=1)))
    meta['control_norm_relative_max']=float(np.max(abs(np.linalg.norm(vearly[pc[:,1]],axis=1)/np.linalg.norm(vearly[pearly[:,1]],axis=1)-1)))
    meta['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'METADANE.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('DONE',flush=True)

if __name__=='__main__':main()
