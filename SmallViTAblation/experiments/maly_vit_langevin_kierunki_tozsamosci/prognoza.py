"""Validation selects frozen causal GLE; future suffix poisoning must not change forecasts."""
import argparse,time,shutil
import numpy as np
from threadpoolctl import threadpool_limits
from dane import HERE,read,js,sha
from gle import fit,rollout,measure,block_bootstrap

RIDGES=(1e-4,1e-3,1e-2,.1,1)
LAGS=(0,1,2,4,8)

def series():
    a=dict(np.load(HERE/'TRAJEKTORIE_PRZYCZYNOWE_K500.npz'));n=int(a['tool_count'])
    return a['q500'][:,:n],a['tool_weights'],a

def save_models(dt,bank):
    arr={};meta={}
    for name,model in bank.items():
        meta[name]={k:v for k,v in model.items() if not isinstance(v,np.ndarray)}
        for key,value in model.items():
            if isinstance(value,np.ndarray):arr[f'{name}_{key}']=value
    np.savez_compressed(HERE/f'MACIERZE_DT{dt}.npz',**arr);js(HERE/f'MODELE_DT{dt}.json',meta)

def load_models(dt):
    meta=read(HERE/f'MODELE_DT{dt}.json');arr=dict(np.load(HERE/f'MACIERZE_DT{dt}.npz'))
    for name,model in meta.items():
        for key,value in arr.items():
            if key.startswith(name+'_'):model[key[len(name)+1:]]=value
    return meta

def validation(q,w,dt):
    origins=np.arange(700,750,dt);hs=[1,10,50] if dt==1 else [10,50];grid=[]
    for L in LAGS:
        for ridge in RIDGES:
            model=fit(q,w,L,ridge,dt,700);forecast=rollout(q,origins,max(hs),model);metrics={}
            for H in hs:
                m,_=measure(forecast[H],q[origins+H],q[origins],w,model['epsilon'],model['majority_sign']);metrics[str(H)]={k:m[k] for k in ('MSE','persistence_MSE','skill')}
            score=float(np.mean([m['skill'] for m in metrics.values()]));grid.append(dict(L=L,ridge=ridge,mean_skill=score,horizons=metrics))
    selected=max(grid,key=lambda r:r['mean_skill']);best1=max(grid,key=lambda r:r['horizons'][str(dt)]['skill'])
    chosen={f'L{L}':max([r for r in grid if r['L']==L],key=lambda r:r['mean_skill']) for L in LAGS}
    chosen.update(selected=selected,best_H1=best1)
    bank={name:fit(q,w,r['L'],r['ridge'],dt,799 if dt==1 else 790) for name,r in chosen.items()}
    save_models(dt,bank)
    js(HERE/f'WYBOR_WALIDACJA_DT{dt}.json',dict(stage='validation_frozen_before_test',dt=dt,origin_updates=origins,target_max=int(origins[-1]+max(hs)),equal_weights_per_horizon=True,
        selection_horizons=hs,grid=grid,chosen=chosen,final_fit_target_max=799 if dt==1 else 790,trajectories_sha256=sha(HERE/'TRAJEKTORIE_PRZYCZYNOWE_K500.npz'),
        matrices_sha256=sha(HERE/f'MACIERZE_DT{dt}.npz'),basis500_sha256=sha(HERE/'IDENTYFIKACJA_K500.npz')))
    print('Frozen validation',dt,'selected',selected,'best short',best1,flush=True)

def noise(q,w,model,dt):
    from scipy.stats import skew,kurtosis,jarque_bera,chi2
    origins=np.arange(800,1000,dt);forecast=rollout(q,origins,dt,model)[dt];e=q[origins+dt]-forecast
    maximum=min(10,len(e)//5);records=[]
    for name,sequence in [('shared',np.einsum('tnd,n->td',e,w))]+[(f'token{k}',e[:,k]) for k in range(e.shape[1])]:
        for j in range(8):
            value=sequence[:,j];centered=value-value.mean();den=np.sum(centered**2);acf=np.array([np.sum(centered[:-lag]*centered[lag:])/den if den>1e-30 else 0 for lag in range(1,maximum+1)])
            Q=len(value)*(len(value)+2)*np.sum(acf**2/(len(value)-np.arange(1,maximum+1)));jb=jarque_bera(value)
            records.append(dict(series=name,coordinate0=j,N=len(value),acf=acf,LjungBox_Q=float(Q),LjungBox_lags=maximum,LjungBox_raw_p=float(chi2.sf(Q,maximum)),
                skew=float(skew(value,bias=False)),excess_kurtosis=float(kurtosis(value,bias=False)),JarqueBera_p=float(jb.pvalue),variance_first_half=float(np.var(value[:len(value)//2])),variance_second_half=float(np.var(value[len(value)//2:])),mean=float(value.mean())))
    correlations=[]
    for j in range(8):
        C=np.corrcoef(e[:,:,j].T);off=C[np.triu_indices(len(w),1)];off=off[np.isfinite(off)]
        correlations.append(dict(coordinate0=j,mean=float(off.mean()) if len(off) else None,median=float(np.median(off)) if len(off) else None,pairs=len(off)))
    return dict(dt=dt,target_updates=origins+dt,residuals=e,per_series_diagnostics=records,cross_token_same_coordinate_correlations=correlations,
        limitations='Exploratory time diagnostics; rawp uncorrected multiple tests/parameter estimation; Gaussianity does not imply whiteness; no FDT')

def test(q,w,a,dt):
    bank=load_models(dt);selection=read(HERE/f'WYBOR_WALIDACJA_DT{dt}.json');assert selection['matrices_sha256']==sha(HERE/f'MACIERZE_DT{dt}.npz')
    origins=np.arange(800,951,dt);hs=[1,2,5,10,20,50] if dt==1 else [10,20,50];all_metrics={};saved={'origins':origins,'token_ids0':a['token_ids'][:len(w)]};audit=[]
    baseline_model=bank['selected']
    for name in list(bank)+['persistence','velocity']:
        model=bank.get(name,baseline_model);kind=name if name in ('persistence','velocity') else 'gle';forecasts=rollout(q,origins,max(hs),model,kind)
        metric={}
        for H in hs:
            m,details=measure(forecasts[H],q[origins+H],q[origins],w,model['epsilon'],model['majority_sign']);m['bootstrap']=block_bootstrap(details,dt,H);metric[str(H)]=m
            if name in ('selected','best_H1','L0','persistence','velocity'):saved[f'{name}_H{H}']=forecasts[H]
        origins100=np.arange(800,901,dt);pred100=rollout(q,origins100,100,model,kind)[100];m,_=measure(pred100,q[origins100+100],q[origins100],w,model['epsilon'],model['majority_sign']);m.update(bootstrap=dict(reported=False,reason='about one independent block'),origin_updates=origins100);metric['100']=m
        all_metrics[name]=metric
        for origin in (800,950):
            reference=rollout(q,[origin],50,model,kind)[50];prefix=rollout(q[:origin+1],[origin],50,model,kind)[50]
            poison=q.copy();poison[origin+1:]=np.nan;changed=rollout(poison,[origin],50,model,kind)[50]
            assert np.array_equal(reference,prefix) and np.array_equal(reference,changed)
            audit.append(dict(model=name,origin=origin,prefix_exact=True,poisoned_future_exact=True))
    # Common co10 origins: dt1 comparisons do not conflate sample sets.
    common={}
    if dt==1:
        ori=np.arange(800,951,10)
        for name in ('selected','L0','persistence','velocity'):
            model=bank.get(name,baseline_model);kind=name if name in ('persistence','velocity') else 'gle';forecast=rollout(q,ori,50,model,kind)
            common[name]={str(H):measure(forecast[H],q[ori+H],q[ori],w,model['epsilon'],model['majority_sign'])[0] for H in (10,20,50)}
    n=len(w);bg=a['q500'][:,n:];bg_model=dict(baseline_model,anchor=bg[500]);bgpred=rollout(bg,origins,50,bg_model);background={str(H):measure(bgpred[H],bg[origins+H],bg[origins],np.ones(8)/8,bg_model['epsilon'],bg_model['majority_sign'])[0] for H in hs}
    np.savez_compressed(HERE/f'PROGNOZY_DT{dt}.npz',**saved)
    diagnostics=noise(q,w,baseline_model,dt);residual=diagnostics.pop('residuals');np.savez_compressed(HERE/f'RESZTY_DT{dt}.npz',residuals=residual,target_updates=diagnostics['target_updates'])
    js(HERE/f'SZUM_DT{dt}.json',diagnostics);js(HERE/f'AUDYT_PREFIX_DT{dt}.json',audit)
    js(HERE/f'METRYKI_DT{dt}.json',dict(dt=dt,origin_updates=origins,horizons=hs,models=all_metrics,common_co10_origins=common,background_transfer=background,
        MSE_units='physical U500 projected activation squared, averaged8coordinates',selection=selection['chosen'],all_finite=True,prefix_audit_passed=True))
    print('Test complete dt',dt,{H:dict(skill=all_metrics['selected'][str(H)]['skill'],sign=all_metrics['selected'][str(H)]['sign_accuracy']) for H in hs},flush=True)

def main():
    start=time.time();threadpool_limits(4);p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);args=p.parse_args();q,w,a=series()
    for dt in (1,10):
        if args.stage=='validation':validation(q,w,dt)
        else:test(q,w,a,dt)
    js(HERE/f'CZAS_{args.stage}.json',dict(seconds=time.time()-start,stage=args.stage));assert shutil.disk_usage(HERE).free>1e9

if __name__=='__main__':main()
