"""Czy historyczne paczki poprawiają H=500 poza dopasowaniem?

Obala to przewaga kontroli GLE/stanu na zamrożonym teście s48. Warianty
wybiera wyłącznie walidacja s47; predykcja nie otrzymuje przyszłych wejść.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import time
import numpy as np
from scipy.stats import skew, kurtosis
from dane import load, HERE, sha256


def moving_mean(A, times, width):
    # Slices end at r inclusive; no future element is read.
    return np.stack([A[max(0,r-width+1):r+1].mean(0) for r in times])


def features(d, times, spec):
    S, times = d['S'], np.asarray(times, int)
    P = S.shape[1]
    window = spec.get('level',500)
    x = S[times]-moving_mean(S,times,window)
    base = []
    kind = spec['kind']
    if kind == 'legacy_gle':
        # exact 20 lags of increments at resolution 10, and mean of 50 records
        base = [S[times-10*j]-S[times-10*(j+1)] for j in range(20)]
        m = np.stack([S[r-490:r+1:10].mean(0) for r in times])
        base.append(S[times]-m)
    elif kind == 'ar':
        base = [x,(S[times]-S[times-10])/10]
    else:
        base = [x]+[(S[times]-S[times-width])/width for width in (10,20,50,100,200)]
    if spec.get('lifting'):
        base += [x*x, x*base[3]]
    if spec.get('desc'):
        for width in (10,50,100):
            q = moving_mean(d['q'],times,width)
            base += [np.broadcast_to(q[:,j,None],(len(times),P)) for j in range(q.shape[1])]
    if spec.get('effect'):
        base += [moving_mean(d['N'],times,width) for width in (10,50,100)]
    if spec.get('ped'):
        base += [moving_mean(d['P'],times,width) for width in (10,50,100)]
    X = np.stack(base,-1)
    assert np.isfinite(X).all(), (spec, times[[0,-1]])
    return X


def fit(d, times, H, spec, cutoff):
    assert np.max(times)+H <= cutoff
    X = features(d,times,spec).reshape(-1,features(d,times,spec).shape[-1])
    y = (d['S'][times+H]-d['S'][times]).reshape(-1)
    # Original GLE has no intercept and scales std without centering.
    legacy = spec['kind']=='legacy_gle'
    mean = np.zeros(X.shape[1]) if legacy else X.mean(0)
    scale = X.std(0)+1e-12
    Xm=(X-mean)/scale
    ym=0. if legacy else y.mean()
    W=np.linalg.solve(Xm.T@Xm+spec['ridge']*len(X)*np.eye(X.shape[1]), Xm.T@(y-ym))
    sv=np.linalg.svd(Xm,compute_uv=False)
    return dict(W=W,mean=mean,scale=scale,ym=ym,spec=spec,
        diagnostics=dict(rank=int(np.linalg.matrix_rank(Xm)),features=int(Xm.shape[1]),
                         condition=float(sv[0]/max(sv[-1],1e-14)),rows=int(len(X))))


def predict(model,d,times):
    X=features(d,times,model['spec'])
    return (X-model['mean'])/model['scale'] @ model['W'] + model['ym']


def metrics(Y,p):
    e=Y-p
    mse=float(np.mean(e*e)); zero=float(np.mean(Y*Y))
    large=np.abs(Y)>=np.median(np.abs(Y),axis=1)[:,None]
    return dict(mse=mse,mae=float(np.mean(np.abs(e))),rmse=mse**.5,
        skill_percent=100*(1-mse/max(zero,1e-30)),sign=float(np.mean(np.sign(Y)==np.sign(p))),
        sign_large=float(np.mean((np.sign(Y)==np.sign(p))[large])),
        baseline_mse=zero,positive_fraction=float(np.mean(Y>0)))


def legacy_fit(d,cutoff,kind):
    # Copied regression family from prognoza.py. Blocks are means of 5 records.
    S=d['S'][1000:cutoff+1:10]
    trim=(len(S)//5)*5
    B=S[len(S)-trim:].reshape(-1,5,S.shape[1]).mean(1)
    D=np.diff(B,axis=0)
    X=[]; y=[]
    for t in range(4,len(D)):
        f=[D[t-1-j,:,None] for j in range(4)]
        if kind=='langevin': f=[B[t,:,None]]+f+[np.eye(B.shape[1])]
        X.append(np.concatenate(f,1));y.append(D[t])
    X=np.concatenate(X);y=np.concatenate(y)
    A=X.T@X
    W=np.linalg.solve(A+1e-2*np.diag(np.diag(A))+1e-9*np.eye(len(A)),X.T@y)
    return W


def legacy_predict(W,d,times,H,kind):
    preds=[]
    # H divisible by 50 in this diagnostic; extrapolate exactly H/50 blocks.
    for r in times:
        # Last block ends at r and is fully historical.
        B=np.stack([d['S'][r-40-50*j:r+1-50*j:10].mean(0) for j in range(5)][::-1])
        B=list(B);D=list(np.diff(B,axis=0))
        for _ in range(H//50):
            f=[D[-1-j][:,None] for j in range(4)]
            if kind=='langevin': f=[B[-1][:,None]]+f+[np.eye(len(B[-1]))]
            dd=np.concatenate(f,1)@W
            B.append(B[-1]+dd);D.append(dd)
        # Original predicts a block mean. Compare absolute prediction to endpoint.
        preds.append(B[-1]-d['S'][r])
    return np.stack(preds)


def diagnostics(Y,p,d,times):
    E=Y-p
    flat=E.ravel()
    q=moving_mean(d['N'],times,100)
    x=d['S'][times]-moving_mean(d['S'],times,500)
    def corr(a,b):
        a=np.asarray(a).ravel();b=np.asarray(b).ravel()
        return float(np.corrcoef(a,b)[0,1]) if a.std()*b.std()>1e-20 else None
    return dict(skew=float(skew(flat)),excess_kurtosis=float(kurtosis(flat)),
        residual_lag1=corr(E[:-1],E[1:]) if len(E)>1 else None,
        residual_lag5=corr(E[:-5],E[5:]) if len(E)>5 else None,
        corr_past_state=corr(E,x),corr_past_batch_effect=corr(E,q),
        variance_first_half=float(np.var(E[:max(1,len(E)//2)])),
        variance_second_half=float(np.var(E[len(E)//2:])),
        warning='H-step overlap induces residual autocorrelation; not a whiteness test of one-step noise')


def causal_tests(d, model, r=2000):
    original=predict(model,d,[r])
    prefix={k:(v[:r+1].copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
    assert np.array_equal(original,predict(model,prefix,[r]))
    modified={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
    for key in ('S','N','P','q'):
        modified[key][r+1:]=1e7
    assert np.array_equal(original,predict(model,modified,[r]))
    # Row after cutoff, the ongoing/future package, must have no effect.
    assert np.array_equal(features(d,[r],model['spec']),features(modified,[r],model['spec']))
    # A completed effect can change an effect feature: validates nonzero usage.
    if model['spec'].get('effect'):
        changed={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
        changed['N'][r]+=1
        assert not np.array_equal(features(d,[r],model['spec']),features(changed,[r],model['spec']))
    return dict(prefix_invariance=True,future_suffix_invariance=True,completed_input_alignment=True)


def main():
    t0=time.time()
    out=HERE/'wyniki'
    d={s:load(s) for s in (47,48)}
    config=dict(train_seed=47,transfer_seed=48,calibration_step=1000,train_cutoff=1800,
        validation_origins=list(range(1800,2001,10)),refit_cutoff=2500,
        test_origins_s48=list(range(2100,2501,10)),test_origins_s47=[2500],
        horizons=[10,50,100,500],ridge_grid=[.001,.1,10],level_grid=[500,1000],
        time_resolution=10,primary_origin=2500)
    (HERE/'CONFIG.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'AUDYT.json').write_text(json.dumps({str(s):d[s]['audit'] for s in d},indent=2),encoding='utf-8')
    results={}; predictions={}; models={}; validation_records=[]
    families={
        'ar':dict(kind='ar'), 'gle_compact':dict(kind='compact'),
        'gle_descriptor':dict(kind='compact',desc=True),
        'gle_effect':dict(kind='compact',effect=True),
        'gle_descriptor_effect':dict(kind='compact',desc=True,effect=True),
        'gle_descriptor_effect_ped':dict(kind='compact',desc=True,effect=True,ped=True),
        'lifting_quadratic':dict(kind='compact',lifting=True)}
    all_checks={}
    for H in config['horizons']:
        train=np.arange(1000,1800-H+1,10)
        val=np.arange(1800,2001,10)
        Yv=d[47]['S'][val+H]-d[47]['S'][val]
        specs={}; fitted={}; val_pred={}
        for name,base in families.items():
            best=None
            for ridge in config['ridge_grid']:
                for level in config['level_grid']:
                    spec=dict(base,ridge=ridge,level=level)
                    model=fit(d[47],train,H,spec,1800)
                    pred=predict(model,d[47],val)
                    mse=float(np.mean((Yv-pred)**2))
                    validation_records.append(dict(H=H,family=name,spec=spec,mse=mse))
                    if best is None or mse<best[0]:best=(mse,spec,model,pred)
            specs[name]=best[1];val_pred[name]=best[3]
        specs['legacy_gle']=dict(kind='legacy_gle',ridge=.001,level=500)
        m=fit(d[47],train,H,specs['legacy_gle'],1800)
        val_pred['legacy_gle']=predict(m,d[47],val)
        val_pred['zero']=np.zeros_like(Yv)
        val_pred['drift']=(d[47]['S'][val]-d[47]['S'][val-100])*H/100
        legacy_models={}
        if H%50==0:
            for kind in ('koopman','langevin'):
                name='legacy_'+kind
                W=legacy_fit(d[47],1800,kind)
                val_pred[name]=legacy_predict(W,d[47],val,H,kind)
                legacy_models[name]=legacy_fit(d[47],2500,kind)
        selected=min(val_pred,key=lambda n:np.mean((Yv-val_pred[n])**2))
        selected_new=min(families,key=lambda n:np.mean((Yv-val_pred[n])**2))
        refit=np.arange(1000,2500-H+1,10)
        for name,spec in specs.items():
            fitted[name]=fit(d[47],refit,H,spec,2500)
            if H==500:all_checks[name]=causal_tests(d[47],fitted[name])
        results[str(H)]=dict(selected_validation=selected,selected_new_validation=selected_new,
            validation={n:metrics(Yv,p) for n,p in val_pred.items()},specs=specs,test={})
        for seed in (47,48):
            ts=np.array(config[f'test_origins_s{seed}'])
            Y=d[seed]['S'][ts+H]-d[seed]['S'][ts]
            pp={n:predict(m,d[seed],ts) for n,m in fitted.items()}
            pp['zero']=np.zeros_like(Y)
            pp['drift']=(d[seed]['S'][ts]-d[seed]['S'][ts-100])*H/100
            for name,W in legacy_models.items():
                pp[name]=legacy_predict(W,d[seed],ts,H,name.split('_')[1])
            results[str(H)]['test'][str(seed)]={}
            for name,p in pp.items():
                # Fixed historical validation calibration for an approximate interval.
                lo,hi=np.quantile(Yv-val_pred[name],[.05,.95])
                met=metrics(Y,p)
                met['primary_t2500']=metrics(Y[-1:],p[-1:])
                met['interval90_coverage']=float(np.mean((Y>=p+lo)&(Y<=p+hi)))
                met['interval90_width']=float(hi-lo)
                if name in (selected_new,'legacy_gle'):
                    met['residual_diagnostics']=diagnostics(Y,p,d[seed],ts)
                results[str(H)]['test'][str(seed)][name]=met
                predictions[f'H{H}_s{seed}_{name}']=p.astype(np.float32)
            predictions[f'H{H}_s{seed}_truth']=Y.astype(np.float32)
            predictions[f'H{H}_s{seed}_times']=ts
        for name,m in fitted.items():
            models[f'H{H}_{name}_W']=m['W'];models[f'H{H}_{name}_mean']=m['mean']
            models[f'H{H}_{name}_scale']=m['scale'];models[f'H{H}_{name}_intercept']=np.array(m['ym'])
            results[str(H)].setdefault('fit_diagnostics',{})[name]=m['diagnostics']
        print('H',H,'selected',selected,'new',selected_new,'s48 skill',
              results[str(H)]['test']['48'][selected_new]['skill_percent'],flush=True)
    stamp=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()
    report=dict(created=stamp,seconds=time.time()-t0,config=config,results=results,
        causal_checks=all_checks,validation_trials=validation_records,
        code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        primary_replications=2,confirmatory=False,
        intervals='validation pooled residual quantiles, empirical only, not guaranteed calibration')
    (out/'METRYKI.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(out/'PROGNOZY.npz',**predictions)
    np.savez_compressed(out/'MODELE.npz',**models)
    print('seconds',report['seconds'],flush=True)


if __name__=='__main__':main()
