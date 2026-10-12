"""Full K native-coordinate forecasts; no identity rotations, no future teacher forcing."""
from common import *
from model_native import fit,rollout,spectrum
from metryki import measure,block_bootstrap,rollout as floor_rollout
from threadpoolctl import threadpool_limits
import argparse,time,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
RIDGES=[.01,.1,1,10]
LAGS=[0,1,2,4]

def save_bank(dt,bank):
 arr={};meta={}
 for name,m in bank.items():
  meta[name]={}
  for k,v in m.items():
   if k in ('fg','bg'):
    meta[name][k]=None if v is None else {}
    if v is not None:
     for kk,vv in v.items():
      if isinstance(vv,np.ndarray):arr[f'{name}__{k}__{kk}']=vv
      else:meta[name][k][kk]=vv
   elif isinstance(v,np.ndarray):arr[f'{name}__{k}']=v
   else:meta[name][k]=v
 np.savez_compressed(HERE/f'MACIERZE_DT{dt}.npz',**arr);js(HERE/f'MODELE_DT{dt}.json',meta)

def load_bank(dt):
 meta=read(HERE/f'MODELE_DT{dt}.json');arr=dict(np.load(HERE/f'MACIERZE_DT{dt}.npz'))
 for key,v in arr.items():
  pp=key.split('__')
  if len(pp)==3:meta[pp[0]][pp[1]][pp[2]]=v
  else:meta[pp[0]][pp[1]]=v
 return meta

def validation(a,dt):
 start=time.time();q=a['s_2'].astype(float);bg=a['bg_2'];w=a['w_2'];origins=np.arange(700,750,dt);hs=[1,10,50] if dt==1 else [10,50];grid=[]
 for family,coupled in [('local',False),('coupled',True)]:
  for L in LAGS:
   for ridge in RIDGES:
    model=fit(q,bg,w,L,ridge,dt,700,coupled);record=dict(family=family,coupled=coupled,L=L,ridge=ridge,status='finite')
    try:
     pred,maxnorm=rollout(q,bg,w,origins,50,model);vals={str(H):measure(pred[H],q[origins+H],q[origins],w,model['epsilon_q'],model['majority_q'])[0] for H in hs}
     assert all(np.isfinite(v[k]) for v in vals.values() for k in ('MSE','persistence_MSE','skill'))
     record.update(mean_skill=float(np.mean([v['skill'] for v in vals.values()])),short_skill=vals[str(dt)]['skill'],metrics={h:{k:v[k] for k in ('MSE','persistence_MSE','skill')} for h,v in vals.items()},max_rollout_absolute_state=maxnorm)
    except (FloatingPointError,OverflowError,AssertionError):record.update(status='nonfinite_error_preserved',mean_skill=None,short_skill=None)
    grid.append(record);print('val native',dt,family,L,ridge,record['mean_skill'],flush=True)
 valid=[r for r in grid if r['status']=='finite'];assert valid
 choices={name:max([r for r in valid if r['family']==name],key=lambda r:r['mean_skill']) for name in ('local','coupled')}
 choices.update({name+'_L0':max([r for r in valid if r['family']==name and r['L']==0],key=lambda r:r['mean_skill']) for name in ('local','coupled')})
 choices['selected']=max(valid,key=lambda r:r['mean_skill']);choices['best_short']=max(valid,key=lambda r:r['short_skill']);bank={}
 for name,r in choices.items():
  bank[name]=fit(q,bg,w,r['L'],r['ridge'],dt,799 if dt==1 else 790,r['coupled']);bank[name]['spectrum']=spectrum(bank[name])
 save_bank(dt,bank);record=dict(status='native_dynamics_frozen_before_test',dt=dt,K=q.shape[-1],channels0=a['channels0'],channels1=a['channels1'],selection_horizons=hs,choices=choices,grid=grid,origin_updates=origins,
  final_fit_target_max=799 if dt==1 else 790,basis_rotation=False,target='all selected native coordinates physical activation MSE averaged K',data_sha=sha(HERE/'DYNAMIKA_NATYWNE_J.npz'),matrices_sha=sha(HERE/f'MACIERZE_DT{dt}.npz'),channels_model_sha=sha(HERE/'KANALY_MODEL.npz'),seconds=time.time()-start,completed=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat())
 js(HERE/f'WYBOR_DYNAMIKA_DT{dt}.json',record);print('DYNAMICS FREEZE',dt,choices['selected'],choices['best_short'],record['seconds'],flush=True)

def paired(new,old,stride,H):
 N=len(new['mse']);block=max(1,max(50,H)//stride);rng=np.random.default_rng(20261002+stride+H);values=[]
 for _ in range(300):
  ii=np.concatenate([(s+np.arange(block))%N for s in rng.integers(0,N,int(np.ceil(N/block)))])[:N]
  nm=new['mse'][ii].mean();om=old['mse'][ii].mean();pm=old['persistence'][ii].mean();values.append([(om-nm)/pm,1-nm/om,new['sign'][ii].mean()-old['sign'][ii].mean()])
 vals=np.array(values);return dict(descriptive_only=True,block_updates=block*stride,origin_stride=stride,repeats=300,point=dict(skill_difference=float((old['mse'].mean()-new['mse'].mean())/old['persistence'].mean()),MSE_reduction=float(1-new['mse'].mean()/old['mse'].mean()),sign_difference=float(new['sign'].mean()-old['sign'].mean())),CI95={k:np.quantile(vals[:,j],[.025,.975]) for j,k in enumerate(('skill_difference','MSE_reduction','sign_difference'))})

def test(a,dt):
 assert not (HERE/f'METRYKI_DT{dt}.json').exists(),'Frozen native test already exists, no automatic repeat';start=time.time();choice=read(HERE/f'WYBOR_DYNAMIKA_DT{dt}.json');assert choice['matrices_sha']==sha(HERE/f'MACIERZE_DT{dt}.npz');assert choice['data_sha']==sha(HERE/'DYNAMIKA_NATYWNE_J.npz');assert choice['channels_model_sha']==sha(HERE/'KANALY_MODEL.npz')
 bank=load_bank(dt);origins=np.arange(800,951,dt);hs=[1,2,5,10,20,50] if dt==1 else [10,20,50];records={};audits=[];saved={}
 for probe in (2,8):
  q=a[f's_{probe}'].astype(float);bg=a[f'bg_{probe}'];w=a[f'w_{probe}'];D=q.shape[-1];assert D==choice['K'];metrics={};details={};common={};base=bank['local'];epsilon=base['epsilon_q'];majority=base['majority_q']
  for name in list(bank)+['persistence','velocity']:
   if name in bank:
    mod=bank[name];pred,maxnorm=rollout(q,bg,w,origins,50,mod)
    for origin in (800,950):
     reference=rollout(q,bg,w,[origin],50,mod)[0];prefix=rollout(q[:origin+1],bg[:origin+1],w,[origin],50,mod)[0];pq=q.copy();pb=bg.copy();pq[origin+1:]=np.nan;pb[origin+1:]=np.nan;poison=rollout(pq,pb,w,[origin],50,mod)[0]
     assert all(np.array_equal(reference[h],prefix[h]) and np.array_equal(reference[h],poison[h]) for h in reference)
     audits.append(dict(probe=probe,model=name,origin=origin,prefix_poison_exact=True,all_K=D))
   else:
    floor=dict(dt=dt,L=0,anchor=q[500]);pred=floor_rollout(q,origins,50,floor,name);maxnorm=float(np.max(np.abs(pred[50])))
   metrics[name]={};details[name]={}
   for H in hs:
    assert pred[H].shape==q[origins+H].shape and pred[H].shape[-1]==D
    m,d=measure(pred[H],q[origins+H],q[origins],w,epsilon,majority);m['bootstrap']=block_bootstrap(d,dt,H);metrics[name][str(H)]=m;details[name][str(H)]=d
    # Persist only selected forecasts; all model scores and matrices survive.
    if name=='selected' and H in (1,10,50):saved[f'probe{probe}_H{H}']=pred[H]
   metrics[name]['max_rollout_absolute_state']=maxnorm
   if dt==1:
    ii=np.arange(0,len(origins),10);common[name]={str(H):measure(pred[H][ii],q[origins[ii]+H],q[origins[ii]],w,epsilon,majority)[0] for H in (10,20,50)}
  contrasts={name:{str(H):paired(details[name][str(H)],details['local'][str(H)],dt,H) for H in hs} for name in list(bank) if name!='local'}
  records[str(probe)]=dict(models=metrics,contrasts_vs_local=contrasts,common_co10=common,origin_updates=origins,horizons=hs,K=D,transfer_refit=False,threshold_source='main probe local finalfit delta_sd, same for every method and both probes')
 np.savez_compressed(HERE/f'PROGNOZY_DT{dt}.npz',**saved);js(HERE/f'AUDYT_PREFIX_DT{dt}.json',audits);js(HERE/f'METRYKI_DT{dt}.json',dict(status='native_dynamics_single_test_complete',dt=dt,selection=choice['choices']['selected'],best_short=choice['choices']['best_short'],probes=records,all_finite=True,seconds=time.time()-start,completed=datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()))
 print('DYNAMICS TEST',dt,{H:(records['2']['models']['selected'][str(H)]['skill'],records['2']['models']['selected'][str(H)]['sign_accuracy']) for H in hs},flush=True)

def main():
 threadpool_limits(4);p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);arg=p.parse_args();a=dict(np.load(HERE/'DYNAMIKA_NATYWNE_J.npz'))
 for dt in (1,10):validation(a,dt) if arg.stage=='validation' else test(a,dt)
 assert shutil.disk_usage(HERE).free>1e9
if __name__=='__main__':main()
