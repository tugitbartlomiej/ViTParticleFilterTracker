"""Joint controlled lifting/state-input memory. No KF or actual future batches."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json,time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import skew,kurtosis
from dane import load,HERE,sha256
from koopman_obserwator import raw_images,means
from krotki_horyzont import metric,majority

OUT=HERE/'wyniki/joint_controlled'
HS=(1,3,5,10,20,50)
WINDOWS=((0,1),(1,5),(5,20))

def preprocess(d,q,cut):
    t=np.arange(1000,cut+1,5);x=d['S'][t]-means(d['S'],t)
    X=q[t].astype(float);mu=X.mean(0);V=np.linalg.svd(X-mu,full_matrices=False)[2][:4].T
    pc=(q-mu)@V;sd=pc[t].std(0)+1e-9
    return dict(mu=float(x.mean()),sd=float(x.std()+1e-9),qmu=mu,V=V,qsd=sd)

def lifting(m,x):
    z=(x-m['mu'])/m['sd'];return np.stack([z,z*z],-1)

def inputs(m,d,q,kind):
    if kind=='simple':return d['q'][:,:3]
    return (q-m['qmu'])@m['V']/m['qsd']

def features(m,d,q,t,kind,controlled=False):
    t=np.asarray(t,int);a=means(d['S'],t);P=d['S'].shape[1]
    z=lifting(m,d['S'][t]-a);blocks=[z]
    if kind!='only':
        for j in range(20):blocks.append(lifting(m,d['S'][t-j]-a)-lifting(m,d['S'][t-j-1]-a))
    if kind in ('images','simple','opt'):
        u=inputs(m,d,q,kind)
        if controlled:blocks.append(np.broadcast_to(u[t+1,None,:],(len(t),P,u.shape[-1])))
        for lo,hi in WINDOWS:
            b=np.stack([u[r-hi+1:r-lo+1].mean(0) for r in t])
            blocks.append(np.broadcast_to(b[:,None,:],(len(t),P,b.shape[1])))
    if kind=='opt':
        for key in ('N','P'):
            for lo,hi in WINDOWS:blocks.append(np.stack([d[key][r-hi+1:r-lo+1].mean(0) for r in t])[...,None])
    X=np.concatenate(blocks,-1);assert np.isfinite(X).all()
    return X,z,a

def fit(d,q,pre,kind,cut,ridge,controlled=False):
    t=np.arange(1000,cut-(1 if controlled else 50)+1,5)
    X,z,a=features(pre,d,q,t,kind,controlled);X=X.reshape(-1,X.shape[-1])
    horizons=(1,) if controlled else HS
    Y=np.concatenate([lifting(pre,d['S'][t+h]-a)-z for h in horizons],-1).reshape(len(X),-1)
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1;A=(X-mu)/sd
    off=Y.mean(0);W=np.linalg.solve(A.T@A+len(A)*ridge*np.eye(A.shape[1]),A.T@(Y-off))
    singular=np.linalg.svd(A,compute_uv=False)
    return dict(pre=pre,kind=kind,mu=mu,sd=sd,W=W,offset=off,ridge=ridge,controlled=controlled,
                rank=int(np.linalg.matrix_rank(A)),condition=float(singular[0]/max(singular[-1],1e-14)),
                smallest_singular=float(singular[-1]),columns=A.shape[1],origins=len(t),rows=len(A))

def forecast(model,d,q,t):
    X,_,_=features(model['pre'],d,q,t,model['kind'])
    out=(X-model['mu'])/model['sd']@model['W']+model['offset']
    return out[...,::2]*model['pre']['sd']

def rollout(model,d,q,t):
    # Future q is modeled as past20 mean; do not read q[t+1:].
    t=np.asarray(t,int);m=model['pre'];a=means(d['S'],t);P=d['S'].shape[1]
    z=lifting(m,d['S'][t]-a);initial=z.copy()
    mem=[lifting(m,d['S'][t-j]-a)-lifting(m,d['S'][t-j-1]-a) for j in range(20)]
    u=inputs(m,d,q,model['kind']);hist=np.stack([u[t-j] for j in range(20)],1)
    future=hist.mean(1);out=[]
    for h in range(1,51):
        blocks=[z]+mem
        blocks.append(np.broadcast_to(future[:,None,:],(len(t),P,future.shape[-1])))
        for lo,hi in WINDOWS:
            b=hist[:,lo:hi].mean(1);blocks.append(np.broadcast_to(b[:,None,:],(len(t),P,b.shape[-1])))
        X=np.concatenate(blocks,-1)
        dz=(X-model['mu'])/model['sd']@model['W']+model['offset'];z=z+dz
        mem=[dz]+mem[:-1];hist=np.concatenate([future[:,None,:],hist[:,:-1]],1)
        if h in HS:out.append((z[...,0]-initial[...,0])*m['sd'])
    return np.stack(out,-1)

def diag(Y,P):
    E=Y-P;center=E-E.mean(0);var=np.mean(center**2)
    return dict(skew=float(skew(E.ravel())),excess_kurtosis=float(kurtosis(E.ravel())),
        residual_acf_origin_stride5=[float(np.mean(center[j:]*center[:-j])/max(var,1e-12)) for j in (1,2,5)],
        error_squared_prediction_abs_corr=float(np.corrcoef((E*E).ravel(),np.abs(P).ravel())[0,1]))

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    d={s:load(s) for s in (47,48)};q={s:raw_images(s) for s in d}
    pre={c:preprocess(d[47],q[47],c) for c in (1800,2500)}
    val=np.arange(1800,2001,5);tests={47:np.arange(2500,2951,5),48:np.arange(2100,2951,5)}
    Yv=np.stack([d[47]['S'][val+h]-d[47]['S'][val] for h in HS],-1)
    models={};vp={};vr={};config={}
    for kind in ('only','memory','images','simple','opt','controlled_images'):
        controlled=kind.startswith('controlled');actual='images' if controlled else kind
        candidates=[]
        for ridge in (.001,.1,10):
            m=fit(d[47],q[47],pre[1800],actual,1800,ridge,controlled)
            p=rollout(m,d[47],q[47],val) if controlled else forecast(m,d[47],q[47],val)
            candidates.append((m,p))
        for i,H in enumerate(HS):
            k=min(range(3),key=lambda j:np.mean((candidates[j][1][...,i]-Yv[...,i])**2))
            m,p=candidates[k];name=kind;key=(H,name)
            models[key]=fit(d[47],q[47],pre[2500],actual,2500,m['ridge'],controlled)
            vp[key]=p[...,i];vr[key]=metric(Yv[...,i],p[...,i])
            config[f'H{H}_{name}']={n:m[n] for n in ('ridge','rank','condition','smallest_singular','columns','origins','rows','controlled')}
    results={};preds={};checks={};saved={}
    for i,H in enumerate(HS):
        names=[k[1] for k in models if k[0]==H];chosen=min(names,key=lambda n:vr[(H,n)]['mse'])
        rr=dict(selected=chosen,validation={n:vr[(H,n)] for n in names},test={})
        for seed,t in tests.items():
            Y=d[seed]['S'][t+H]-d[seed]['S'][t];rr['test'][str(seed)]={}
            for name in names:
                m=models[(H,name)];p=(rollout(m,d[seed],q[seed],t) if m['controlled'] else forecast(m,d[seed],q[seed],t))[...,i]
                a=metric(Y,p);lo,hi=np.quantile(Yv[...,i]-vp[(H,name)],[.05,.95])
                a.update(coverage90_percent=100*float(np.mean((Y>=p+lo)&(Y<=p+hi))),width90=float(hi-lo))
                if name==chosen:a['residual_diagnostics']=diag(Y,p)
                rr['test'][str(seed)][name]=a;preds[f'H{H}_s{seed}_{name}']=p.astype(np.float32)
            rr['test'][str(seed)]['zero']=metric(Y,np.zeros_like(Y));rr['test'][str(seed)]['majority']=metric(Y,majority(d[seed],t,H))
            preds[f'H{H}_s{seed}_truth']=Y.astype(np.float32);preds[f'H{H}_s{seed}_times']=t
        for name in names:
            m=models[(H,name)];r=2200;pref={k:(v[:r+1] if isinstance(v,np.ndarray) else v) for k,v in d[48].items()}
            fn=rollout if m['controlled'] else forecast
            aa=fn(m,d[48],q[48],[r]);bb=fn(m,pref,q[48][:r+1],[r]);assert np.array_equal(aa,bb)
            checks[f'H{H}_{name}']=True
            for k in ('mu','sd','W','offset'):saved[f'H{H}_{name}_{k}']=m[k]
        results[str(H)]=rr;a=rr['test']['48'][chosen]
        print(H,chosen,round(a['skill_percent'],2),round(a['sign_percent'],2),flush=True)
    for cut,m in pre.items():
        for k,v in m.items():saved[f'pre{cut}_{k}']=v
    old=json.loads((HERE/'wyniki/krotki_horyzont/METRYKI.json').read_text());edmd=json.loads((HERE/'wyniki/koopman_obserwator/METRYKI.json').read_text())
    record=dict(created=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat(),seconds=time.time()-start,
        code_sha256=sha256(__file__),exploratory_s48=True,train_cut=1800,refit_cut=2500,hs=HS,
        results=results,configs=config,prefix_invariance=checks,
        old_direct=old['results'],old_edmd_without_kf={h:{s:{n:a for n,a in b.items() if not n.endswith('_kf')} for s,b in r['test'].items()} for h,r in edmd['results'].items()})
    (OUT/'METRYKI.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'PROGNOZY.npz',**preds);np.savez_compressed(OUT/'MODELE.npz',**saved)

if __name__=='__main__':main()
