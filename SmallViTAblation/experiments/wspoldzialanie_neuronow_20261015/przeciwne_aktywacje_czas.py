"""Descriptive lag analysis, frozen pair selection, same-token signed projections."""
from pathlib import Path
import json,time,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits
threadpool_limits(4)
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje'

def corr_columns(a,b):
    aa=a-a.mean(0);bb=b-b.mean(0)
    den=np.sqrt(np.sum(aa*aa,0)*np.sum(bb*bb,0))
    return np.divide(np.sum(aa*bb,0),den,out=np.full_like(den,np.nan),where=den>1e-16)

def lagcorr(a,b,lag):
    if lag>0:return corr_columns(a[:-lag],b[lag:])
    if lag<0:return corr_columns(a[-lag:],b[:lag])
    return corr_columns(a,b)

def detrend(a,B):
    shape=a.shape;flat=a.reshape(len(a),-1)
    return (flat-B@(B.T@flat)).reshape(shape)

def main():
    torch.set_num_threads(4);started=time.time()
    torch.backends.cuda.matmul.allow_tf32=False
    info=np.load(OUT/'indeksy.npz');steps=info['steps'];keep=steps>=33400
    # Reconstruct float32 GELU on saved float32 pre-MLP states: avoid differencing float16 noise.
    old=np.load(ROOT/'outputs/optyka_hipoteza_20261015/pomiar.npz')
    pre=old['x_pre'][info['original_index'][keep]].reshape(sum(keep),72,192)
    h=np.load(ROOT/'app/cache_uczenia/lasery/eafb5a973be3d490/dane.npz');wd=Path(h['wagi'].item())
    sys.path.insert(0,str(ROOT/'app'));import lasery_dane as LD
    block=LD.modul_modelu().MalyViT().bloki[-1];eps=block.norm2.eps;approx=block.mlp.act.approximate
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    z=np.zeros((sum(keep),72,768),np.float32)
    for ti,k in enumerate(steps[keep]):
        state=torch.load(wd/f'k{int(k):05d}.pt',weights_only=True,map_location='cpu')['net']
        w={q:state['bloki.5.'+q].to(dev) for q in ['norm2.weight','norm2.bias','mlp.fc1.weight','mlp.fc1.bias']}
        x=F.layer_norm(torch.tensor(pre[ti],device=dev),(192,),w['norm2.weight'],w['norm2.bias'],eps)
        z[ti]=F.gelu(F.linear(x,w['mlp.fc1.weight'],w['mlp.fc1.bias']),approximate=approx).cpu().numpy()
    np.save(OUT/'Z_wybrane_float32_pozne.npy',z)
    z16=np.load(OUT/'Z_wybrane.npy',mmap_mode='r')[keep].reshape(sum(keep),72,768)
    cache_relative=float(np.linalg.norm(z-z16)/np.linalg.norm(z))
    difference_relative=float(np.linalg.norm(np.diff(z,axis=0)-np.diff(z16,axis=0))/np.linalg.norm(np.diff(z,axis=0)))
    print(f'Float32 late activations: {time.time()-started:.1f}s',flush=True)
    vv=np.load(OUT/'V.npy',mmap_mode='r')[keep].copy()
    pairs=np.load(OUT/'pary.npz');early=pairs['early'];control=pairs['control'];v0=pairs['v_early']
    masks={'all':np.ones(72,bool),'bbox':info['bbox'].ravel(),'far':info['far'].ravel()}
    groups={(b+1,g):ma&np.repeat(np.arange(3)==b,24) for b in range(3) for g,ma in masks.items()}
    B=SplineTransformer(n_knots=6,degree=3,include_bias=True).fit_transform(steps[keep,None])
    B=np.linalg.qr(B)[0].astype(np.float32)
    rows=[];nullrows=[];detail=[]
    # Fixed shift list, minimum 20 intervals. Preserve individual series autocorrelation.
    shifts=np.random.default_rng(271015).choice(np.arange(20,len(z)-20),32,replace=False)
    for name,pp in [('early',early),('control',control)]:
        c,d=pp.T;e=v0[c]-v0[d];e/=np.linalg.norm(e,axis=1)[:,None]
        pc=np.einsum('tpi,pi->tp',vv[:,c],e);pd_=np.einsum('tpi,pi->tp',vv[:,d],e)
        a=z[:,:,c]*pc[:,None,:];b=z[:,:,d]*pd_[:,None,:]
        for kind,aa,bb in [('increments',np.diff(a,axis=0),np.diff(b,axis=0)),('detrended_levels',detrend(a,B),detrend(b,B))]:
            for lag in range(-5,6):
                rr=lagcorr(aa,bb,lag)
                for (im,g),ma in groups.items():
                    v=rr[ma];rows.append(dict(pair_set=name,series=kind,lag=lag,lag_steps=lag*100,image=im,group=g,
                        n_tokens=int(ma.sum()),n_pairs=len(pp),mean=float(np.nanmean(v)),median=float(np.nanmedian(v)),
                        q25=float(np.nanquantile(v,.25)),q75=float(np.nanquantile(v,.75))))
                if kind=='increments' and lag in [0,1,-1]:
                    for j,(c0,d0) in enumerate(pp):
                        detail.append(dict(pair_set=name,c=int(c0),d=int(d0),lag=lag,mean_r=float(np.nanmean(rr[:,j]))))
            for shift in shifts:
                rr=lagcorr(aa,np.roll(bb,int(shift),axis=0),1)
                for (im,g),ma in groups.items():
                    nullrows.append(dict(pair_set=name,series=kind,lag=1,image=im,group=g,shift=int(shift),mean=float(np.nanmean(rr[ma]))))
        print(f'{name}: correlations {time.time()-started:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'opoznienia.csv',index=False)
    pd.DataFrame(nullrows).to_csv(OUT/'opoznienia_kontrola_przesuniec.csv',index=False)
    pd.DataFrame(detail).to_csv(OUT/'opoznienia_par.csv',index=False)
    # Full 192-dimensional counter-motion with all neurons (not only projections).
    xp=old['x_post'][info['original_index'][keep]].reshape(len(z),72,192)
    dx=np.diff(xp,axis=0)
    family=np.zeros(768,bool);family[np.unique(early)]=True
    dev='cuda' if torch.cuda.is_available() else 'cpu';motion=[]
    for t in range(2,len(z)):
        vt=torch.tensor(vv[t],device=dev);vp=torch.tensor(vv[t-1],device=dev)
        zt=torch.tensor(z[t],device=dev);zp=torch.tensor(z[t-1],device=dev)
        nu=(zt.square()*vt.square().sum(1)+zp.square()*vp.square().sum(1)-2*zt*zp*(vt*vp).sum(1)).clamp_min(0).sqrt()
        for lag in [0,1,3,5]:
            if t-1-lag<0:continue
            dd=torch.tensor(dx[t-1-lag],device=dev)
            dot=(dd@vt.T)*zt-(dd@vp.T)*zp
            den=dd.norm(dim=1)[:,None]*nu
            for (im,g),ma in groups.items():
                if g=='all':continue
                for nm,ne in [('opposed_member',family),('other',~family)]:
                    ds=dot[ma][:,ne];ns=den[ma][:,ne]
                    motion.append(dict(step=int(steps[keep][t]),lag=lag,image=im,group=g,neuron_group=nm,
                        weighted_cos=float(ds.sum()/ns.sum().clamp_min(1e-20)),
                        opposing_weight=float((ns*(ds<0)).sum()/ns.sum().clamp_min(1e-20))))
    pd.DataFrame(motion).to_csv(OUT/'przeciwdzialanie_ruchowi.csv',index=False)
    (OUT/'CZAS_QC.json').write_text(json.dumps(dict(first_step=int(steps[keep][0]),last_step=int(steps[keep][-1]),n_checkpoints=int(sum(keep)),
        group_counts={f'{im}_{g}':int(m.sum()) for (im,g),m in groups.items()},
        frozen_pair_count=len(early),opposed_neuron_count=int(family.sum()),elapsed_s=time.time()-started,
        norm2_eps=eps,gelu_approximate=approx,float16_relative_rms=cache_relative,float16_difference_relative_rms=difference_relative,
        interpretation='Lag means optimizer steps, not firing delay inside one MLP. Correlations are descriptive, not independent replications or causal evidence.'),indent=2),encoding='utf-8')
    print('DONE',flush=True)

if __name__=='__main__':main()
