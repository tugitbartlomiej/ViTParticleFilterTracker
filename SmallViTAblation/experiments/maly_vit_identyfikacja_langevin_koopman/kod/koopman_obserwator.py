"""Czy operator liftingu, obrazowe wejście i przyczynowy KF poprawiają prognozę?
Obala to brak poprawy na eksploracyjnym s48; nie odczytuje przyszłych paczek.
Finite-lag EDMD z zamrożoną referencją, nie globalna linearyzacja ViT.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch
from scipy.stats import skew,kurtosis
from dane import load,HERE,sha256
from krotki_horyzont import metric,majority

torch.set_num_threads(1)
OUT=HERE/'wyniki/koopman_obserwator'


def means(S,t):return np.stack([S[r-499:r+1].mean(0) for r in t])


def raw_images(seed):
    f=OUT/f'OBRAZY_s{seed}.npy'
    if f.exists():return np.load(f)
    p=HERE.parent/'maly_vit_rownanie_latentu'/f'nagranie_s{seed}_sily'
    L=np.load(p/'paczka_lat.npy',mmap_mode='r');mask=np.load(p/'paczka_ma_narz.npy',mmap_mode='r')
    q=np.full((len(L)+1,1536),np.nan,dtype=np.float32)
    for a in range(0,len(L),64):
        b=min(a+64,len(L));v=L[a:b].astype(np.float32);m=mask[a:b].astype(np.float32)
        tool=(v[:,:,0]*m[:,:,None]).sum(1)/np.maximum(m.sum(1)[:,None],1)
        q[a+1:b+1]=np.concatenate([tool,v[:,:,1].mean(1)],1)
    np.save(f,q)
    return q


def pca_fit(q,cut):
    X=q[1000:cut+1:5].astype(float);mu=X.mean(0)
    V=np.linalg.svd(X-mu,full_matrices=False)[2][:8].T
    sd=((X-mu)@V).std(0)+1e-9
    return dict(mu=mu,V=V,sd=sd)


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=torch.nn.Sequential(torch.nn.Linear(1,16),torch.nn.Tanh(),torch.nn.Linear(16,4))
        self.dec=torch.nn.Sequential(torch.nn.Linear(4,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
        self.delta=torch.nn.Linear(4,4)
        torch.nn.init.zeros_(self.delta.weight);torch.nn.init.zeros_(self.delta.bias)


def lifting_fit(S,cut,kind):
    ts=np.arange(1000,cut,5);m=means(S,ts)
    x=(S[ts]-m).ravel();n=(S[ts+1]-m).ravel()
    mu=x.mean();sd=x.std()+1e-9
    obj=dict(kind=kind,mu=mu,sd=sd,seconds=0.,epochs=0)
    if kind=='fixed':return obj
    torch.manual_seed(1101+cut)
    net=AE();opt=torch.optim.Adam(net.parameters(),lr=.002)
    X=torch.tensor((x-mu)/sd,dtype=torch.float32)[:,None]
    N=torch.tensor((n-mu)/sd,dtype=torch.float32)[:,None]
    vscale=max(float((N-X).std()),1e-3);start=time.time()
    for epoch in range(18):
        order=torch.randperm(len(X))
        for idx in order.split(512):
            z=net.enc(X[idx]);rec=net.dec(z)
            loss=((rec-X[idx])**2).mean()
            if epoch>=8:
                zp=z+net.delta(z)
                delta=net.dec(zp)-rec
                loss=loss+((delta-(N[idx]-X[idx]))/vscale).square().mean()
                loss=loss+.05*((net.enc(N[idx])-zp)**2).mean()
            opt.zero_grad();loss.backward();opt.step()
    obj.update(net=net,epochs=18,seconds=time.time()-start)
    with torch.no_grad():
        rec=net.dec(net.enc(X))
        obj['reconstruction_mse']=float(((rec-X)**2).mean())
    torch.save(net.state_dict(),OUT/f'AE_cut{cut}.pt')
    return obj


def psi(lift,x):
    z=(np.asarray(x)-lift['mu'])/lift['sd']
    if lift['kind']=='fixed':return np.stack([np.ones_like(z),z,z*z,z*z*z],-1)
    with torch.no_grad():
        a=lift['net'].enc(torch.tensor(z.reshape(-1,1),dtype=torch.float32)).numpy()
    return np.concatenate([np.ones((len(a),1)),a],1).reshape(*z.shape,5)


def decode(lift,z):
    if lift['kind']=='fixed':return z[...,1]*lift['sd']+lift['mu']
    with torch.no_grad():
        a=lift['net'].dec(torch.tensor(z[...,1:].reshape(-1,4),dtype=torch.float32)).numpy().reshape(z.shape[:-1])
    return a*lift['sd']+lift['mu']


def drift_fit(d,H,lift,cut):
    ts=np.arange(1000,cut-H+1,5);m=means(d['S'],ts)
    X=psi(lift,d['S'][ts]-m).reshape(-1,4 if lift['kind']=='fixed' else 5)
    Y=psi(lift,d['S'][ts+H]-m).reshape(X.shape)
    scale=X.std(0);scale[scale<1e-9]=1
    A=X/scale
    delta=np.linalg.solve(A.T@A+1e-6*len(A)*np.eye(A.shape[1]),A.T@(Y-X))/scale[:,None]
    K=np.eye(len(delta))+delta
    err=np.mean((X@K-Y)**2)
    return dict(lift=lift,K=K,closure_mse=float(err),rank=int(np.linalg.matrix_rank(A)),
                condition=float(np.linalg.cond(A)),spectral_radius=float(max(abs(np.linalg.eigvals(K)))))


def drift_predict(m,d,t):
    x=d['S'][t]-means(d['S'],t);Z=psi(m['lift'],x)
    # Difference of decoded endpoints cancels reconstruction offset of AE.
    return decode(m['lift'],Z@m['K'])-decode(m['lift'],Z)


def history(d,q,t,pca,variant):
    P=d['S'].shape[1]
    f=[d['S'][t-j]-d['S'][t-j-1] for j in range(20)]
    if variant in ('images','opt'):
        for w in (1,5,20):
            a=np.stack([q[r-w+1:r+1].mean(0) for r in t])
            a=(a-pca['mu'])@pca['V']/pca['sd']
            f.extend([np.broadcast_to(a[:,j,None],(len(t),P)) for j in range(8)])
    if variant=='opt':
        for name in ('N','P'):
            for w in (1,5,20):f.append(np.stack([d[name][r-w+1:r+1].mean(0) for r in t]))
    X=np.stack(f,-1);assert np.isfinite(X).all()
    return X


def complete_fit(d,q,H,op,pca,variant,ridge,first,last):
    model=dict(op=op,pca=pca,variant=variant,H=H)
    if variant=='only':return model
    t=np.arange(first,last-H+1,5)
    A=history(d,q,t,pca,variant);X=A.reshape(-1,A.shape[-1])
    Y=(d['S'][t+H]-d['S'][t]-drift_predict(op,d,t)).ravel()
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-9]=1
    X=(X-mu)/sd;W=np.linalg.solve(X.T@X+ridge*len(X)*np.eye(X.shape[1]),X.T@(Y-Y.mean()))
    model.update(mu=mu,sd=sd,W=W,offset=Y.mean(),ridge=ridge,
                 rank=int(np.linalg.matrix_rank(X)),condition=float(np.linalg.cond(X)))
    return model


def forecast(model,d,q,t):
    t=np.asarray(t,int)
    p=drift_predict(model['op'],d,t)
    if model['variant']!='only':
        X=history(d,q,t,model['pca'],model['variant'])
        p=p+(X-model['mu'])/model['sd']@model['W']+model['offset']
    return p


def completed_errors(model,d,q,start,stop):
    H=model['H'];t=np.arange(start,stop-H+1)
    E=d['S'][t+H]-d['S'][t]-forecast(model,d,q,t)
    return t+H,E


def noise_fit(model,d,q,first,last,fraction):
    j,E=completed_errors(model,d,q,first,last)
    var=float(np.var(E));cov=float(np.mean((E[1:]-E.mean())*(E[:-1]-E.mean())))
    R=max(fraction*var,1e-12);hidden=max(var-R,1e-12)
    rho=float(np.clip(cov/hidden,-.98,.98));Q=max((1-rho*rho)*hidden,1e-12)
    return dict(rho=rho,Q=Q,R=R,variance=var,fraction=fraction,
                convention='eta indexed by label completion, not forecast origin')


def filtered_forecast(model,noise,d,q,t,start):
    # y_j is only available at j: its forecast origin is j-H.
    j,E=completed_errors(model,d,q,start,int(max(t)))
    eta=np.zeros(d['S'].shape[1]);P=noise['variance'];est={}
    rho,Q,R=noise['rho'],noise['Q'],noise['R']
    for k,e in zip(j,E):
        eta=rho*eta;P=rho*rho*P+Q;gain=P/(P+R)
        eta=eta+gain*(e-eta);P=(1-gain)*P
        if k in t:est[int(k)]=eta.copy()
    correction=np.stack([rho**model['H']*est[int(k)] for k in t])
    return forecast(model,d,q,t)+correction


def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    d={s:load(s) for s in (47,48)};q={s:raw_images(s) for s in d}
    validation=np.arange(1800,2001,5);tests={47:np.arange(2500,2951,5),48:np.arange(2100,2951,5)}
    lifts={(cut,kind):lifting_fit(d[47]['S'],cut,kind) for cut in (1400,2000) for kind in ('fixed','learned')}
    pc={cut:pca_fit(q[47],cut) for cut in (1400,2000)}
    results={};preds={};checks={};old=json.loads((HERE/'wyniki/krotki_horyzont/METRYKI.json').read_text())
    for H in (1,3,5,10,20,50):
        Yv=d[47]['S'][validation+H]-d[47]['S'][validation]
        vr={};vp={};configs={};fitted={};validation_models={}
        for kind in ('fixed','learned'):
            op=drift_fit(d[47],H,lifts[(1400,kind)],1400)
            op2=drift_fit(d[47],H,lifts[(2000,kind)],2000)
            for variant in ('only','memory','images','opt'):
                name=kind+'_'+variant;best=None
                for ridge in ([.001] if variant=='only' else [.001,.1,10]):
                    m=complete_fit(d[47],q[47],H,op,pc[1400],variant,ridge,1400,1800)
                    p=forecast(m,d[47],q[47],validation);e=np.mean((Yv-p)**2)
                    if best is None or e<best[0]:best=(e,ridge,m,p)
                _,ridge,m,p=best;validation_models[name]=m;vp[name]=p;vr[name]=metric(Yv,p)
                fitted[name]=complete_fit(d[47],q[47],H,op2,pc[2000],variant,ridge,2000,2500)
                configs[name]=dict(kind=kind,variant=variant,ridge=ridge,closure=op['closure_mse'],rank=op['rank'],
                    condition=op['condition'],spectral_radius=op['spectral_radius'])
                # KF is a separate arm; select R only on historical validation.
                best_filter=None
                for fraction in (.1,.5,.9):
                    noise=noise_fit(m,d[47],q[47],1400,1800,fraction)
                    pp=filtered_forecast(m,noise,d[47],q[47],validation,1000)
                    e=np.mean((Yv-pp)**2)
                    if best_filter is None or e<best_filter[0]:best_filter=(e,fraction,pp,noise)
                _,fraction,pp,nv=best_filter;fn=name+'_kf'
                vr[fn]=metric(Yv,pp);vp[fn]=pp
                fitted[fn]=(fitted[name],noise_fit(fitted[name],d[47],q[47],2000,2500,fraction))
                configs[fn]=dict(configs[name],noise=nv)
        selected=min(vr,key=lambda n:vr[n]['mse']);rr=dict(selected=selected,validation=vr,configs=configs,test={})
        for seed,t in tests.items():
            Y=d[seed]['S'][t+H]-d[seed]['S'][t];rr['test'][str(seed)]={}
            for name,model in fitted.items():
                p=filtered_forecast(*model,d[seed],q[seed],t,1000 if seed==47 else 2020) if isinstance(model,tuple) else forecast(model,d[seed],q[seed],t)
                a=metric(Y,p);lo,hi=np.quantile(Yv-vp[name],[.05,.95])
                a.update(coverage90_percent=100*float(np.mean((Y>=p+lo)&(Y<=p+hi))),width90=float(hi-lo))
                if name==selected:
                    E=Y-p;a['residual_skew']=float(skew(E.ravel()));a['residual_kurtosis']=float(kurtosis(E.ravel()))
                rr['test'][str(seed)][name]=a;preds[f'H{H}_s{seed}_{name}']=p.astype(np.float32)
            rr['test'][str(seed)]['zero']=metric(Y,np.zeros_like(Y))
            rr['test'][str(seed)]['majority_past']=metric(Y,majority(d[seed],t,H))
            rr['old_comparison']=old['results'][str(H)]['test']
            preds[f'H{H}_s{seed}_truth']=Y.astype(np.float32);preds[f'H{H}_s{seed}_times']=t
        # Prefix check includes replay of only matured residuals and image prefixes.
        model=fitted[selected];r=2200;pref={k:(v[:r+1] if isinstance(v,np.ndarray) else v) for k,v in d[48].items()}
        if isinstance(model,tuple):
            a=filtered_forecast(*model,d[48],q[48],[r],2020);b=filtered_forecast(*model,pref,q[48][:r+1],[r],2020)
        else:a=forecast(model,d[48],q[48],[r]);b=forecast(model,pref,q[48][:r+1],[r])
        assert np.array_equal(a,b)
        checks[str(H)]=dict(prefix_invariance=True,label_delay=H,actual_future_input=False)
        results[str(H)]=rr;v=rr['test']['48'][selected]
        print(H,selected,'skill',round(v['skill_percent'],2),'sign',round(v['sign_percent'],2),flush=True)
    now=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()
    maximum={field:max([int(h) for h,rr in results.items() if rr['test']['48'][rr['selected']][field]>=80],default=None)
             for field in ('sign_percent','balanced_sign_percent','skill_percent')}
    report=dict(created=now,seconds=time.time()-start,results=results,checks=checks,max_measured_H_at80=maximum,
        source_sha256=sha256(__file__),cpu_threads=1,ae_epochs=18,exploratory=True,
        ae={str(cut):{k:v for k,v in lifts[(cut,'learned')].items() if k!='net'} for cut in (1400,2000)})
    (OUT/'METRYKI.json').write_text(json.dumps(report,indent=2),encoding='utf-8');np.savez_compressed(OUT/'PROGNOZY.npz',**preds)
    lines=['# Pilot EDMD/learned lifting i obserwatora: wynik eksploracyjny','',now,'',
        'S48 wcześniej oglądano. Wybór wyłącznie na walidacji s47. Stan jest skalaremLDA z poolingu, nie trajektorią cząsteczek.', '',
        '| H | Wybór walidacji | RedukcjaMSE s48 [%] | Znak [%] | Balanced [%] | Dawny Langevin MSE [%] | Wcześniejszy direct lifting+GLE MSE [%] |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for h,rr in results.items():
        a=rr['test']['48'][rr['selected']];b=old['results'][h]['test']['48']['old_langevin20'];c=old['results'][h]['test']['48']['koopman_lift_gle20_input']
        lines.append(f"| {h} | {rr['selected']} | {a['skill_percent']:.2f} | {a['sign_percent']:.2f} | {a['balanced_sign_percent']:.2f} | {b['skill_percent']:.2f} | {c['skill_percent']:.2f} |")
    lines += ['', 'Maksymalne zmierzone H z progiem80: '+json.dumps(maximum)+'. Brak monotoniczności lub gwarancji. Pełne ramiona, KF i wyniki s47 wMETRYKI.json.', '',
        'EDMD fituje CAŁY lifting celu przy referencji m_t zamrożonej na czas okna. Learned lifting uczy encoder/decoder i dynamikę; potem osobny K_H. KażdyH to inny operator lokalny, nie iteracjaK1 ani dowód semigrupy. Sekwencyjny podział dryf/pamięć/sterowanie jest umowną projekcją; drugi etap widzi chronologiczne reszty dryfu poza jego treningiem. Nie identyfikuje mechanizmu przyczynowego. Obrazowe sterowanie pochodzi z rzeczywistych paczka_lat, pooled narzędzie/całość i PCA8 z treningu; N/P są oddzielnymi wpływami. Przyszłe wejście jest marginalizowane empirycznie przez historyczną estymację, nie znane.', '',
        'KF aktualizuje ukrytą AR1-resztę indeksowaną KOŃCEM etykiety, wykorzystując e_H(t−H). Prognozuje rho^H eta_t. Przy H>5 okna testu nakładają się, a także błędy pomiarowe mogą być kolorowe; Gaussian KF jest aproksymacją drugich momentów. Parametry wariancji z dopasowanych reszt mogą być optymistyczne. EKF/UKF nie uruchamiano; wymagają konkretnej realizacji i oceny modelu pomiaru. Nie dowiedziono obserwowalności/sterowalności ViT.', '',
        'MałyAE na CPU18epok; nie trenowano ViT, nie użytoGPU/Edenu, nie pobierano22GB. Wszystkie negatywy zachowano. Świeży przebieg po zamrożeniu modelu i miary jest wymagany do potwierdzenia.']
    (OUT/'WYNIKI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
