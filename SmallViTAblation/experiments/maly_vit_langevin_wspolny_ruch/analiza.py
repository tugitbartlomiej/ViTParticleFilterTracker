"""Frozen validation then one exploratory test; no future auxiliary states."""
from common import *
from model import fit,rollout,spectrum
from metryki_bazowe import measure,block_bootstrap,fit as old_fit,rollout as old_rollout
from threadpoolctl import threadpool_limits
import time,argparse,shutil
RIDGES=[.001,.01,.1,1,10]
FAMILIES=[dict(name='localU8',method='PCA',aux=0,coupled=False),dict(name='coupledU8',method='PCA',aux=0,coupled=True)]+[dict(name=f'{kind}{method}{r}',method=method,aux=r,coupled=kind=='coupled') for kind in ('local','coupled') for method in ('PCA','PLS') for r in (4,8)]

def data(a,probe,family):
 D=8+family['aux'];method=family['method'];return a[f's_{probe}_{method}'][:,:,:D],a[f'bg_{probe}_{method}'][:,:D],a[f'w_{probe}']

def save_bank(dt,bank):
 arrays={};meta={}
 for name,m in bank.items():
  meta[name]={}
  for k,v in m.items():
   if k in ('fg','bg'):
    meta[name][k]=None if v is None else {}
    if v is not None:
     for kk,vv in v.items():
      if isinstance(vv,np.ndarray):arrays[f'{name}__{k}__{kk}']=vv
      else:meta[name][k][kk]=vv
   elif isinstance(v,np.ndarray):arrays[f'{name}__{k}']=v
   else:meta[name][k]=v
 np.savez_compressed(HERE/f'MACIERZE_DT{dt}.npz',**arrays);js(HERE/f'MODELE_DT{dt}.json',meta)

def load_bank(dt):
 meta=read(HERE/f'MODELE_DT{dt}.json');a=dict(np.load(HERE/f'MACIERZE_DT{dt}.npz'))
 for key,v in a.items():
  parts=key.split('__');m=meta[parts[0]]
  if len(parts)==3:m[parts[1]][parts[2]]=v
  else:m[parts[1]]=v
 return meta

def old_bank(dt):
 meta=read(OLD/f'MODELE_DT{dt}.json');arr=dict(np.load(OLD/f'MACIERZE_DT{dt}.npz'))
 for name,m in meta.items():
  for k,v in arr.items():
   if k.startswith(name+'_'):m[k[len(name)+1:]]=v
 return meta

def validation(a,dt):
 start=time.time();origins=np.arange(700,750,dt);hs=[1,10,50] if dt==1 else [10,50];grid=[];lags=[0,2,4,8] if dt==1 else [0,1,2,4,8];bank={};choices={};bestshort=None
 for family in FAMILIES:
  s,bg,w=data(a,2,family);entries=[]
  for L in lags:
   for ridge in RIDGES:
    model=fit(s,bg,w,L,ridge,dt,700,family['coupled']);record=dict(family=family,L=L,ridge=ridge,status='finite')
    try:
     pred,maxnorm=rollout(s,bg,w,origins,50,model);vals={str(H):measure(pred[H],s[origins+H,:,:8],s[origins,:,:8],w,model['epsilon_q'],model['majority_q'])[0] for H in hs}
     record.update(mean_skill=float(np.mean([v['skill'] for v in vals.values()])),short_skill=vals[str(dt)]['skill'],metrics={h:{k:v[k] for k in ('MSE','persistence_MSE','skill')} for h,v in vals.items()},max_rollout_absolute_state=maxnorm)
    except (FloatingPointError,OverflowError):record.update(status='nonfinite',mean_skill=None,short_skill=None)
    entries.append(record);grid.append(record)
  valid=[r for r in entries if r['status']=='finite' and np.isfinite(r['mean_skill'])];assert valid
  selected=max(valid,key=lambda r:r['mean_skill']);short=max(valid,key=lambda r:r['short_skill']);choices[family['name']]=selected
  bank[family['name']]=fit(s,bg,w,selected['L'],selected['ridge'],dt,799 if dt==1 else 790,family['coupled']);bank[family['name']]['family']=family;bank[family['name']]['spectrum']=spectrum(bank[family['name']])
  if bestshort is None or short['short_skill']>bestshort['short_skill']:bestshort=short
  print('val',dt,family['name'],selected['mean_skill'],flush=True)
 oldselect=read(OLD/f'WYBOR_WALIDACJA_DT{dt}.json')['chosen'];choices['oldGLE']=dict(family=dict(name='oldGLE'),mean_skill=oldselect['selected']['mean_skill'],short_skill=oldselect['selected']['horizons'][str(dt)]['skill'],source_previous_frozen_validation=True)
 winner=max(choices,key=lambda name:choices[name]['mean_skill']);oldshort=oldselect['best_H1']['horizons'][str(dt)]['skill']
 if bestshort['short_skill']>oldshort:
  f=bestshort['family'];s,bg,w=data(a,2,f);bank['best_short']=fit(s,bg,w,bestshort['L'],bestshort['ridge'],dt,799 if dt==1 else 790,f['coupled']);bank['best_short']['family']=f;bank['best_short']['spectrum']=spectrum(bank['best_short']);shortname='best_short'
 else:shortname='old_best_short'
 save_bank(dt,bank)
 js(HERE/f'WYBOR_DT{dt}.json',dict(status='frozen_before_new_test',dt=dt,grid=grid,family_choices=choices,selected=winner,best_short=shortname,best_new_short=bestshort,old_best_short_validation=oldshort,
  selection_horizons=hs,origin_updates=origins,final_fit_max=799 if dt==1 else 790,V_refit=False,matrices_sha=sha(HERE/f'MACIERZE_DT{dt}.npz'),data_sha=sha(HERE/'DANE_PROJEKCJE.npz'),seconds=time.time()-start))
 print('FREEZE',dt,winner,shortname,'seconds',time.time()-start,flush=True)

def paired_bootstrap(new,old,stride,H):
 N=len(new['mse']);block=max(1,max(50,H)//stride);rng=np.random.default_rng(917+stride+H);rows=[]
 for _ in range(300):
  ids=np.concatenate([(k+np.arange(block))%N for k in rng.integers(0,N,int(np.ceil(N/block)))])[:N]
  nm=new['mse'][ids].mean();om=old['mse'][ids].mean();pm=old['persistence'][ids].mean();rows.append([(om-nm)/pm,1-nm/om,new['sign'][ids].mean()-old['sign'][ids].mean()])
 rows=np.array(rows);return dict(descriptive_only=True,block_updates=block*stride,repeats=300,CI95={k:np.quantile(rows[:,j],[.025,.975]) for j,k in enumerate(('skill_difference_vs_old','MSE_reduction_vs_old','sign_difference_vs_old'))},point=dict(skill_difference_vs_old=float((old['mse'].mean()-new['mse'].mean())/old['persistence'].mean()),MSE_reduction_vs_old=float(1-new['mse'].mean()/old['mse'].mean()),sign_difference_vs_old=float(new['sign'].mean()-old['sign'].mean())))

def test(a,dt):
 start=time.time();choice=read(HERE/f'WYBOR_DT{dt}.json');assert choice['matrices_sha']==sha(HERE/f'MACIERZE_DT{dt}.npz');assert choice['data_sha']==sha(HERE/'DANE_PROJEKCJE.npz')
 bank=load_bank(dt);old=old_bank(dt);origins=np.arange(800,951,dt);hs=[1,2,5,10,20,50] if dt==1 else [10,20,50];allprobes={};audits=[];saved={}
 for probe in (2,8):
  q=a[f's_{probe}_PCA'][:,:,:8];w=a[f'w_{probe}'];baseline=dict(old['selected'],anchor=q[500]);epsilon=baseline['epsilon'];majority=baseline['majority_sign'];details={};metrics={};common={}
  methods=list(bank)+['oldGLE','old_best_short','persistence','velocity']
  for name in methods:
   model=bank.get(name)
   if model is not None:
    s,bg,w=data(a,probe,model['family']);pred,maxnorm=rollout(s,bg,w,origins,50,model)
    for origin in (800,950):
     ref,_=rollout(s,bg,w,[origin],50,model);prefix,_=rollout(s[:origin+1],bg[:origin+1],w,[origin],50,model)
     ps=s.copy();pb=bg.copy();ps[origin+1:]=np.nan;pb[origin+1:]=np.nan;poison,_=rollout(ps,pb,w,[origin],50,model)
     assert all(np.array_equal(ref[h],prefix[h]) and np.array_equal(ref[h],poison[h]) for h in ref)
     # Means built INSIDE rollout after slicing; poisoned q/r and raw projected background are inaccessible.
     audits.append(dict(probe=probe,method=name,origin=origin,q_r_bg_prefix_and_poison_exact=True))
   else:
    mod=dict(old['best_H1'] if name=='old_best_short' else old['selected'],anchor=q[500]);pred=old_rollout(q,origins,50,mod,name if name in ('persistence','velocity') else 'gle');maxnorm=float(np.max(np.abs(pred[50])))
   metrics[name]={};details[name]={}
   for H in hs:
    m,d=measure(pred[H],q[origins+H],q[origins],w,epsilon,majority);m['bootstrap']=block_bootstrap(d,dt,H);metrics[name][str(H)]=m;details[name][str(H)]=d
    if name in (choice['selected'],choice['best_short'],'oldGLE') and H in (1,10,50):saved[f'probe{probe}_{name}_H{H}']=pred[H]
   metrics[name]['max_rollout_absolute_state']=maxnorm
   if dt==1:
    oi=np.arange(0,len(origins),10);common[name]={str(H):measure(pred[H][oi],q[origins[oi]+H],q[origins[oi]],w,epsilon,majority)[0] for H in (10,20,50)}
  contrasts={name:{str(H):paired_bootstrap(details[name][str(H)],details['oldGLE'][str(H)],dt,H) for H in hs} for name in methods if name!='oldGLE'}
  allprobes[str(probe)]=dict(models=metrics,contrasts_vs_oldGLE=contrasts,refit=False,anchors_replaced_only=True,origin_updates=origins,horizons=hs,common_co10=common)
 np.savez_compressed(HERE/f'PROGNOZY_DT{dt}.npz',**saved);js(HERE/f'AUDYT_PREFIX_DT{dt}.json',audits)
 js(HERE/f'METRYKI_DT{dt}.json',dict(dt=dt,selection=choice['selected'],best_short=choice['best_short'],probes=allprobes,all_finite=True,seconds=time.time()-start))
 print('TEST',dt,'selected',choice['selected'],{H:allprobes['2']['models'][choice['selected']][str(H)]['skill'] for H in hs},flush=True)

def main():
 threadpool_limits(4);p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);arg=p.parse_args();a=dict(np.load(HERE/'DANE_PROJEKCJE.npz'))
 for dt in (1,10):
  if arg.stage=='validation':validation(a,dt)
  else:test(a,dt)
 assert shutil.disk_usage(HERE).free>1e9
if __name__=='__main__':main()
