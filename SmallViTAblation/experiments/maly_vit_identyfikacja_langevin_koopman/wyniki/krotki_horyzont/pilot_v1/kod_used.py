"""Czy bliski zapis osiąga80% znaku albo redukcjiMSE bez przyszłej paczki?
Obala to wynik zamrożonego wyboru poniżej celu na diagnostycznym s48.
Test s48 był wcześniej oglądany; wynik pozostaje eksploracyjny.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import skew,kurtosis
from dane import load,HERE,sha256

OUT=HERE/'wyniki/krotki_horyzont'


def features(d,t,spec):
    S=d['S']; t=np.asarray(t,int)
    x=S[t]-np.stack([S[r-499:r+1].mean(0) for r in t])
    f=[x]+[S[t-j]-S[t-j-1] for j in range(spec['p'])]
    if spec.get('input'):
        for key in ('N','P'):
            f += [d[key][t-j] for j in range(5)]
            f += [np.stack([d[key][r-w+1:r+1].mean(0) for r in t]) for w in (10,20,50)]
    if spec.get('desc'):
        q=np.stack([d['q'][r-9:r+1].mean(0) for r in t])
        f += [np.broadcast_to(q[:,j,None],x.shape) for j in range(q.shape[1])]
    if spec.get('lift'):
        f += [x*x,x*f[1],f[1]*f[1]]
    out=np.stack(f,-1)
    assert np.isfinite(out).all()
    return out


def fit(d,t,H,spec,cut):
    assert max(t)+H<=cut
    A=features(d,t,spec);X=A.reshape(-1,A.shape[-1])
    Y=(d['S'][t+H]-d['S'][t]).ravel()
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1
    Xm=(X-mu)/sd
    W=np.linalg.solve(Xm.T@Xm+spec['ridge']*len(X)*np.eye(X.shape[1]),Xm.T@(Y-Y.mean()))
    return dict(mu=mu,sd=sd,W=W,offset=Y.mean(),spec=spec)


def predict(m,d,t):return (features(d,t,m['spec'])-m['mu'])/m['sd']@m['W']+m['offset']


def metric(Y,P):
    mse=float(np.mean((Y-P)**2));base=float(np.mean(Y*Y))
    pos=Y>0;neg=Y<0
    balanced=.5*(np.mean(P[pos]>0)+np.mean(P[neg]<0))
    big=np.abs(Y)>=np.median(np.abs(Y),axis=1)[:,None]
    return dict(mae=float(np.mean(np.abs(Y-P))),rmse=mse**.5,mse=mse,
        skill_percent=100*(1-mse/base),sign_percent=100*float(np.mean(np.sign(Y)==np.sign(P))),
        balanced_sign_percent=100*float(balanced),sign_large_percent=100*float(np.mean((np.sign(Y)==np.sign(P))[big])),
        positive_percent=100*float(np.mean(pos)),zero_percent=100*float(np.mean(Y==0)))


def majority(d,t,H):
    # Only matured historical H-step changes, with endpoints <= current time.
    return np.stack([np.where((d['S'][r-499:r+1]-d['S'][r-499-H:r+1-H]>0).mean(0)>=.5,1.,-1.) for r in t])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d={s:load(s) for s in (47,48)}
    hs=[1,3,5,10,20,50];val=np.arange(1800,2001,5);test=np.arange(2100,2951,5)
    config=dict(horizons=hs,train_cut=1800,val=val.tolist(),refit_cut=2500,test=test.tolist(),
        fit_stride=5,seed_train=47,seed_test=48,criterion='validation MSE',exploratory=True,
        target80=['sign_percent_all_probes','skill_percent_MSE_reduction'],input_index='source row < t')
    families={'ar':dict(p=1),'gle5':dict(p=5),'gle20':dict(p=20),'gle100':dict(p=100),
        'gle20_input':dict(p=20,input=True),'gle100_input':dict(p=100,input=True),
        'gle20_input_desc':dict(p=20,input=True,desc=True),
        'koopman_lift_gle20':dict(p=20,lift=True),
        'koopman_lift_gle20_input':dict(p=20,lift=True,input=True)}
    results={};artifacts={};coefficients={};checks={}
    for H in hs:
        train=np.arange(1000,1800-H+1,5);refit=np.arange(1000,2500-H+1,5)
        Yv=d[47]['S'][val+H]-d[47]['S'][val]
        chosen={};vp={};scores={}
        for n,b in families.items():
            best=None
            for ridge in (1e-6,.001,.1,10):
                spec=dict(b,ridge=ridge);m=fit(d[47],train,H,spec,1800)
                p=predict(m,d[47],val);e=np.mean((Yv-p)**2)
                if best is None or e<best[0]:best=(e,spec,p)
            chosen[n]=best[1];vp[n]=best[2];scores[n]=metric(Yv,best[2])
        vp['zero']=np.zeros_like(Yv)
        vp['velocity']=(d[47]['S'][val]-d[47]['S'][val-1])*H
        vp['majority_past']=majority(d[47],val,H)*np.sqrt(np.mean(Yv*Yv))
        for n in ('zero','velocity','majority_past'):scores[n]=metric(Yv,vp[n])
        selected=min(families,key=lambda n:scores[n]['mse'])
        overall=min(vp,key=lambda n:scores[n]['mse'])
        rr=dict(specs=chosen,validation=scores,selected_model=selected,selected_overall=overall,test={})
        fitted={n:fit(d[47],refit,H,s,2500) for n,s in chosen.items()}
        for seed,ts in ((47,np.arange(2500,2951,5)),(48,test)):
            Y=d[seed]['S'][ts+H]-d[seed]['S'][ts]
            pp={n:predict(m,d[seed],ts) for n,m in fitted.items()}
            pp['zero']=np.zeros_like(Y);pp['velocity']=(d[seed]['S'][ts]-d[seed]['S'][ts-1])*H
            pp['majority_past']=majority(d[seed],ts,H)*np.sqrt(np.mean(Yv*Yv))
            rr['test'][str(seed)]={}
            for n,p in pp.items():
                a=metric(Y,p);lo,hi=np.quantile(Yv-vp[n],[.05,.95])
                a.update(coverage90_percent=100*float(np.mean((Y>=p+lo)&(Y<=p+hi))),width90=float(hi-lo))
                if n==selected:
                    E=Y-p
                    x=d[seed]['S'][ts]-np.stack([d[seed]['S'][r-499:r+1].mean(0) for r in ts])
                    a['residuals']=dict(skew=float(skew(E.ravel())),excess_kurtosis=float(kurtosis(E.ravel())),
                        acf_lag1=float(np.corrcoef(E[:-1].ravel(),E[1:].ravel())[0,1]),
                        corr_past_state=float(np.corrcoef(E.ravel(),x.ravel())[0,1]),
                        corr_past_input=float(np.corrcoef(E.ravel(),d[seed]['N'][ts].ravel())[0,1]),
                        variance_first=float(np.var(E[:len(E)//2])),variance_second=float(np.var(E[len(E)//2:])),
                        overlaps=H>5)
                rr['test'][str(seed)][n]=a
                artifacts[f'H{H}_s{seed}_{n}']=p.astype(np.float32)
            artifacts[f'H{H}_s{seed}_truth']=Y.astype(np.float32)
            artifacts[f'H{H}_s{seed}_times']=ts
        m=fitted[selected];r=2200
        pref={k:(v[:r+1] if isinstance(v,np.ndarray) else v) for k,v in d[48].items()}
        assert np.array_equal(predict(m,pref,[r]),predict(m,d[48],[r]))
        for key in ('W','mu','sd'):coefficients[f'H{H}_{key}']=m[key]
        coefficients[f'H{H}_offset']=np.array(m['offset'])
        rr['threshold80_s48']={k:rr['test']['48'][selected][k]>=80 for k in ('sign_percent','balanced_sign_percent','skill_percent')}
        results[str(H)]=rr;checks[str(H)]=dict(prefix_invariant=True,mature_labels=True,historical_only=True)
        a=rr['test']['48'][selected]
        print(H,selected,'skill',round(a['skill_percent'],2),'sign',round(a['sign_percent'],2),'balanced',round(a['balanced_sign_percent'],2),flush=True)
    now=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()
    report=dict(created=now,config=config,results=results,checks=checks,source_sha256=sha256(__file__),
                independent_confirmation=False,semantic_identity_proven=False)
    (OUT/'METRYKI.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'CONFIG.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    np.savez_compressed(OUT/'PROGNOZY.npz',**artifacts)
    np.savez_compressed(OUT/'MODELE_WYBRANE.npz',**coefficients)
    lines=['# Pilot bliższego zapisu: wynik eksploracyjny','',now+' (Europe/Warsaw).','',
        'Stały skalar LDA, a nie pełny wektor768. Stan co1 aktualizację; H podano w aktualizacjach. '
        'S48 wcześniej obejrzano; wynik nie jest nowym niezależnym potwierdzeniem. '
        'Wybór wariantu przez MSE walidacji s47, bez przyszłych paczek.','',
        '| H | Wybrany model | RedukcjaMSE s48 [%] | Znak wszystkie [%] | Znak zbilansowany [%] | Większość z przeszłości [%] | Pokrycie90 [%] |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for h,rr in results.items():
        n=rr['selected_model'];a=rr['test']['48'][n];maj=rr['test']['48']['majority_past']
        lines.append(f"| {h} | {n} | {a['skill_percent']:.2f} | {a['sign_percent']:.2f} | {a['balanced_sign_percent']:.2f} | {maj['sign_percent']:.2f} | {a['coverage90_percent']:.1f} |")
    lines += ['', 'Wszystkie ramiona, MAE/RMSE, pomocnicza połowa dużych zmian, reszty i wynik s47: METRYKI.json. '
        'Progi80 są sprawdzone osobno w threshold80_s48. Cel80 nie jest gwarancją. '
        'Rzeczywista paczka powodująca cel jest niedostępna. Pchnięcia przesunięto do stanu po wykonaniu; '
        'u[t] odpowiada źródłowemu wierszowi t-1. Dryf lokalny/lifting x,x²,x*v,v² jest aproksymacją; '
        '20 lub100 opóźnień przyrostów daje pamięć, a historyczne N/P i deskryptory osobne wejścia. '
        'Przyszłe wymuszenie jest uwzględniane przez bezpośrednią regresję warunkowej zmiany, nie przez podgląd przyszłych danych.', '',
        'Cele H>5 nakładają się w diagnostyce co5, więc ich ACF nie dowodzi kolorowego szumu pojedynczego kroku. '
        'Przedziały kalibrowano historycznymi błędami walidacji; brak gwarancji IID. '
        'Dwa biegi i zależne sondy nie stanowią wieloziarnowego potwierdzenia. '
        'Znany kolejny batch z dawnych eksperymentów nie spełnia tego zadania. '
        'Nie badano semantycznej swoistości innych narzędzi/tła ani przyszłych zmian wszystkich składowych. '
        'Te etapy są zaplanowane w PLAN.md; wymagają zdefiniowanej podprzestrzeni i dodatkowych kontroli.', '',
        'Nie uruchomiono nowego treningu, Edenu ani pobierania tokenów. WynikiH500 pozostały w poprzednich plikach.']
    (OUT/'WYNIKI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
