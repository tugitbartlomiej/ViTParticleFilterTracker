"""Sensitivity to reselecting full-768D neighbors at each measurement origin.
Also compute token-level turning fractions without projection or Gram aggregation.
"""
from pathlib import Path
import numpy as np, torch, json
def measure(x,steps,mode,im,out):
    ids={int(k):i for i,k in enumerate(steps)}
    ages=([300,600,1000,3000,6000,10000,15000,19000] if mode=='dense' else [300,1000,6000,10000,19000,24000,30000,40000,50000,60000])
    results=[]
    with torch.no_grad():
        for age in ages:
            early=mode=='dense' and age<1000
            offsets=[0,10,20,30,40] if early else [0,100,200,300,400]
            lags=([1,2,5,10,20,35,50,80,100,200,300] if early else ([50,100,200,500,1000,2000,3000] if mode=='dense' else [100,200,500,1000,2000,3000,5000]))
            vals=[];overlaps=[];ls=[];spacing=[]
            cache={}
            for offset in offsets:
                k=age+offset
                if k not in ids:continue
                b=torch.from_numpy(np.array(x[ids[k]],dtype=np.float32)).cuda()
                ds=torch.cdist(b,b).square();ds.fill_diagonal_(float('inf'))
                ds,nn=ds.topk(6,largest=False);cache[k]=(b,nn,ds)
                spacing.append(float(ds.mean().cpu()))
            for lag in lags:
                v=[];q=[]
                for k,(b,nn,ds) in cache.items():
                    if k+lag not in ids:continue
                    if mode=='long' and k<22974<k+lag:continue
                    z=torch.from_numpy(np.array(x[ids[k+lag]],dtype=np.float32)).cuda()
                    d=z-b;d-=d[nn].mean(1);rr=d.square().sum(1)
                    v.append(float(rr.mean().cpu()))
                    q.append([float((rr<factor**2*ds.mean(1)).float().mean().cpu()) for factor in [.1,.2,.5]])
                if len(v)==len(offsets):vals.append(np.mean(v));overlaps.append(np.mean(q,axis=0).tolist());ls.append(lag)
            results.append(dict(age=age,lag=ls,mean_cage_msd=vals,mean_shell_distance_sq=float(np.mean(spacing)),
                                shell_relative_overlap=overlaps,threshold_factors=[.1,.2,.5]))
            del cache
        turns=[]
        windows=([(300,1094,d) for d in [1,5,10,50]] if mode=='dense' else [(1000,6000,100),(10000,15000,100),(50000,65000,100)])
        for start,end,dt in windows:
            starts=[k for k in range(start,end-2*dt+1,dt) if k in ids and k+dt in ids and k+2*dt in ids]
            if len(starts)>64: starts=[starts[i] for i in np.linspace(0,len(starts)-1,64,dtype=int)]
            sums=np.zeros(6,np.float64)
            for k0 in range(0,len(starts),8):
                kk=starts[k0:k0+8]
                a=torch.from_numpy(np.array(x[[ids[k] for k in kk]],dtype=np.float32)).cuda()
                b=torch.from_numpy(np.array(x[[ids[k+dt] for k in kk]],dtype=np.float32)).cuda()
                c=torch.from_numpy(np.array(x[[ids[k+2*dt] for k in kk]],dtype=np.float32)).cuda()
                v=b-a;w=c-b;dot=(v*w).sum(-1);e=v.square().sum(-1);f=w.square().sum(-1)
                den=(e*f).sqrt();valid=den>0
                sums+=np.array([float(dot.sum().cpu()),float(((e+f)/2).sum().cpu()),float((dot<0).sum().cpu()),dot.numel(),float((dot[valid]/den[valid]).sum().cpu()),float(valid.sum().cpu())])
            turns.append(dict(start=start,end=end,increment=dt,adjacent_correlation=float(sums[0]/sums[1]),
                              turning_fraction=float(sums[2]/sums[3]),mean_token_cosine=float(sums[4]/sums[5]),pairs=int(sums[3])))
    (out/f'{mode}_image{im}_direct.json').write_text(json.dumps(dict(dynamic_cage=results,turning=turns),indent=2),encoding='utf-8')
if __name__=='__main__':
    import compute_grams as C
    fs=sorted((C.SRC/'wyniki/gesty/gesty').glob('pelne768_*.npz'))
    fullsteps=np.concatenate([np.load(p)['kroki'] for p in fs]);available=set(fullsteps.tolist());wanted=set()
    for age in [300,600,1000,3000,6000,10000,15000,19000]:
        early=age<1000
        offsets=[0,10,20,30,40] if early else [0,100,200,300,400]
        lags=[1,2,5,10,20,35,50,80,100,200,300] if early else [50,100,200,500,1000,2000,3000]
        for off in offsets:
            wanted.add(age+off)
            wanted.update(age+off+lag for lag in lags if age+off+lag in available)
    for dt in [1,5,10,50]:
        st=np.arange(300,1094-2*dt+1,dt)
        if len(st)>64:st=st[np.linspace(0,len(st)-1,64,dtype=int)]
        wanted.update(st.tolist());wanted.update((st+dt).tolist());wanted.update((st+2*dt).tolist())
    steps=np.array(sorted(wanted&available));ind={int(k):i for i,k in enumerate(steps)}
    x=np.empty((len(steps),8,600,768),np.float16)
    C.log('loading direct subset',len(steps),x.nbytes/1e9,'GB')
    for p in fs:
        z=np.load(p);ss=z['kroki'];which=np.flatnonzero(np.isin(ss,steps))
        if len(which):
            a=z['latent']
            for j in which:x[ind[int(ss[j])]]=a[j]
            del a
        z.close();C.log('subset source',p.name)
    for im in range(8):
        if (C.OUT/f'dense_image{im}_direct.json').exists():continue
        measure(x[:,im],steps,'dense',im,C.OUT)
        C.log('dense direct controls',im)
