"""Zamrożona walidacja -> audyt -> osobna ocena H500; bez doboru na teście."""
import dane as data
import modele_liniowe as linear
import modele_obserwatora as observer
from metryki import measure,aggregate
import numpy as np, argparse,time
from datetime import datetime
from zoneinfo import ZoneInfo
from threadpoolctl import threadpool_limits
HERE=data.HERE
FOLDS=[dict(end=250,origin=300,H=[1,10,50,100,200]),dict(end=350,origin=400,H=[1,10,50,100])]
TEST_H=[1,2,5,10,20,50,100,200,300,400,500]
def now():return datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()
def module(c):return linear if c['id'].startswith('linear_') else observer
def code_hashes():return {f:data.sha(HERE/f) for f in ['dane.py','modele_liniowe.py','modele_obserwatora.py','metryki.py','badanie.py','PLAN.md']}
def dump_models(models):
 arrays={}
 def encode(v,path):
  if isinstance(v,np.ndarray):arrays[path]=v;return {'__array__':path}
  if isinstance(v,dict):return {k:encode(x,path+'__'+k) for k,x in v.items()}
  if isinstance(v,(tuple,list)):return [encode(x,path+'__'+str(j)) for j,x in enumerate(v)]
  if isinstance(v,np.generic):return v.item()
  return v
 for m in models.values():
  for key in ('fg','bg'):
   if key in m and m[key] is not None and 'gram' in m[key]:
    G=m[key].pop('gram');m[key]['gram_diagonal']=np.diag(G);m[key]['gram_trace']=float(np.trace(G))
 data.save_json(HERE/'MODELE_PRZED_TESTEM.json',encode(models,'model'))
 np.savez_compressed(HERE/'MACIERZE_PRZED_TESTEM.npz',**arrays)
def load_models():
 metadata=data.read(HERE/'MODELE_PRZED_TESTEM.json')
 with np.load(HERE/'MACIERZE_PRZED_TESTEM.npz') as z:
  def decode(v):
   if isinstance(v,dict):return z[v['__array__']].copy() if '__array__' in v else {k:decode(x) for k,x in v.items()}
   if isinstance(v,list):return [decode(x) for x in v]
   return v
  return decode(metadata)
def candidates():
 out=[]
 for dt in (1,10):
  for c in linear.configs():
   c=dict(c,dt=dt);c['id']+='__dt%d'%dt;c['family']+='__dt%d'%dt;out.append(c)
 return out+observer.configs()
def fit_model(qs,bgs,ws,end,c):
 dt=c.get('dt',1);mod=module(c)
 m=mod.fit([q[:end+1:dt] for q in qs],[bg[:end+1:dt] for bg in bgs],ws,end//dt,c)
 m['driver_dt']=dt;return m
def predict_model(q,bg,w,origin,H,m):
 dt=m.get('driver_dt',1);mod=module(m['config']);assert origin%dt==0
 coarse=mod.predict(q[::dt],bg[::dt],w,origin//dt,(H+dt-1)//dt,m)
 if dt==1:return coarse
 h=np.arange(H+1);j=h//dt;fraction=(h%dt)/dt;k=np.minimum(j+1,len(coarse)-1)
 return coarse[j]*(1-fraction[:,None,None])+coarse[k]*fraction[:,None,None]
def rank(rows):
 useful=[r for r in rows if r['useful']]
 pool=useful if useful else rows
 pool=sorted(pool,key=lambda r:(not r['finite'],-r['minimum_sign'],-r['mean_sign'],-r['mean_skill'],r['config']['id']))
 return pool[0],bool(useful)
def score(row):
 byH={}
 for rec in row['fold_records']:
  for h,m in rec['metrics'].items():byH.setdefault(h,[]).append(m)
 ag={h:aggregate(ms) for h,ms in byH.items()};row['by_horizon']=ag
 row['finite']=all(m['finite'] for m in ag.values())
 if row['finite']:
  row['minimum_sign']=min(m['sign_accuracy'] for m in ag.values());row['mean_sign']=float(np.mean([m['sign_accuracy'] for m in ag.values()]));row['mean_skill']=float(np.mean([m['skill'] for m in ag.values()]));row['long_skill']=float(np.mean([m['skill'] for h,m in ag.items() if int(h)>=100]));row['useful']=row['mean_skill']>0 and row['long_skill']>=0
 else:row.update(minimum_sign=-1.,mean_sign=-1.,mean_skill=-1e300,long_skill=-1e300,useful=False)
 return row

def validation():
 assert not (HERE/'WYBOR_PRZED_TESTEM.json').exists(),'selection already frozen'
 qs,bgs,ws=data.load();configs=candidates();assert len(configs)==99
 rows=[];started=now();hashes=code_hashes()
 for idx,c in enumerate(configs):
  row=dict(config=c,fold_records=[]);mod=module(c);t=time.time()
  for fold in FOLDS:
   m=fit_model(qs[:6],bgs[:6],ws[:6],fold['end'],c)
   for i in (6,7):
    try:
     pred=predict_model(qs[i],bgs[i],ws[i],fold['origin'],max(fold['H']),m)
     metrics={str(h):measure(qs[i][fold['origin']+h],pred[h],qs[i][fold['origin']],ws[i]) for h in fold['H']}
     maximum=float(np.max(np.abs(pred))) if np.isfinite(pred).all() else None
    except (FloatingPointError,np.linalg.LinAlgError) as e:
     metrics={str(h):dict(finite=False,status='failure',reason=repr(e)) for h in fold['H']};maximum=None
    row['fold_records'].append(dict(end=fold['end'],origin=fold['origin'],probe=i,metrics=metrics,max_prediction_abs=maximum))
  row['seconds']=time.time()-t;score(row);rows.append(row)
  print('validation',idx+1,len(configs),c['id'],'mean skill',row['mean_skill'],'minimum sign',row['minimum_sign'],flush=True)
  data.save_json(HERE/'WALIDACJA_POSTEP.json',dict(started=started,completed=len(rows),total=len(configs),rows=rows))
 selected,gate=rank(rows);bestmse=max(rows,key=lambda r:(r['finite'],r['mean_skill'],-configs.index(r['config'])))
 families={}
 for fam in sorted(set(r['config']['family'] for r in rows)):
  winner,passed=rank([r for r in rows if r['config']['family']==fam]);families[fam]=dict(id=winner['config']['id'],gate_passed=passed)
 selectedids=sorted(set([selected['config']['id'],bestmse['config']['id'],'persistence','velocity']+[r['id'] for r in families.values()]))
 models={};audits=[]
 for ident in selectedids:
  c=next(c for c in configs if c['id']==ident);mod=module(c);m=fit_model(qs[:8],bgs[:8],ws[:8],500,c);models[ident]=m
  for i in (2,8):
   origin=500;q=qs[i];bg=bgs[i];w=ws[i]
   full=predict_model(q,bg,w,origin,20,m);prefix=predict_model(q[:origin+1],bg[:origin+1],w,origin,20,m)
   poisonq=q.copy();poisonbg=bg.copy();poisonq[origin+1:]=np.nan;poisonbg[origin+1:]=np.nan
   poisoned=predict_model(poisonq,poisonbg,w,origin,20,m)
   passed=np.array_equal(full,prefix,equal_nan=True) and np.array_equal(full,poisoned,equal_nan=True)
   assert passed,ident
   assert full.shape==(21,len(w),128) and np.array_equal(full[0],q[origin])
   audits.append(dict(id=ident,probe=i,prefix_equal=True,poison_equal=True,channels=128,finite=bool(np.isfinite(full).all())))
 dump_models(models)
 choice=dict(status='frozen_before_H500_test',started=started,completed=now(),global_winner=selected['config']['id'],gate_passed=gate,best_mean_mse_skill=bestmse['config']['id'],families=families,selected_ids=selectedids,rows=rows,code_hashes=hashes,data_sha=data.sha(HERE/'DANE_J128.npz'),models_sha=data.sha(HERE/'MACIERZE_PRZED_TESTEM.npz'),models_metadata_sha=data.sha(HERE/'MODELE_PRZED_TESTEM.json'),audits=audits,test_evaluated=False,scope='exploratory; targets after500 previously viewed in earlier experiments')
 data.save_json(HERE/'WYBOR_PRZED_TESTEM.json',choice);print('FROZEN',choice['global_winner'],choice['gate_passed'],flush=True)

def test():
 assert not (HERE/'OCENA_H500.json').exists(),'test already evaluated'
 choice=data.read(HERE/'WYBOR_PRZED_TESTEM.json');assert choice['code_hashes']==code_hashes();assert choice['data_sha']==data.sha(HERE/'DANE_J128.npz');assert choice['models_sha']==data.sha(HERE/'MACIERZE_PRZED_TESTEM.npz');assert choice['models_metadata_sha']==data.sha(HERE/'MODELE_PRZED_TESTEM.json')
 models=load_models()
 qs,bgs,ws=data.load();records={};started=now()
 for ident,m in models.items():
  mod=module(m['config']);perimage=[];stored={};samples={}
  for i in range(12):
   pred=predict_model(qs[i],bgs[i],ws[i],500,500,m)
   assert pred.shape==(501,len(ws[i]),128)
   metrics={str(h):measure(qs[i][500+h],pred[h],qs[i][500],ws[i]) for h in range(1,501)}
   perimage.append(dict(probe=i,by_horizon=metrics))
   if np.isfinite(pred).all() and np.max(abs(pred))<np.finfo(np.float32).max:stored[f'q_{i}']=pred.astype(np.float32)
   else:stored[f'q_{i}']=pred
   # Consecutive microincrements are distinct from cumulative motion.
   samples[str(i)]=aggregate([measure(qs[i][500+h]-qs[i][499+h],pred[h]-pred[h-1],np.zeros_like(qs[i][0]),ws[i]) for h in range(1,501)])
  groups={}
  for group,ids in [('fit',range(8)),('transfer',range(8,12)),('all',range(12))]:
   curve={str(h):aggregate([perimage[i]['by_horizon'][str(h)] for i in ids]) for h in range(1,501)}
   successes=[h for h in range(1,501) if curve[str(h)]['finite'] and curve[str(h)]['sign_accuracy']>=.85]
   prefix=0
   for h in range(1,501):
    if h in successes:prefix=h
    else:break
   groups[group]=dict(by_horizon=curve,maximum_pointwise_h85=max(successes) if successes else None,continuous_prefix_h85=prefix,sampled_horizons={str(h):curve[str(h)] for h in TEST_H})
  np.savez_compressed(HERE/f'PROGNOZY_{ident}.npz',**stored)
  records[ident]=dict(config=m['config'],perimage=perimage,groups=groups,forecast_storage='float32 if representable; metrics computed float64',forecast_sha=data.sha(HERE/f'PROGNOZY_{ident}.npz'),microincrements=samples,interpolated_nonmultiples10=m.get('driver_dt',1)==10)
  print('test',ident,'fit H500',groups['fit']['by_horizon']['500']['sign_accuracy'],'transfer',groups['transfer']['by_horizon']['500']['sign_accuracy'],flush=True)
 data.save_json(HERE/'OCENA_H500.json',dict(started=started,completed=now(),selection_sha=data.sha(HERE/'WYBOR_PRZED_TESTEM.json'),selected_ids=choice['selected_ids'],global_winner=choice['global_winner'],records=records,scope='exploratory; one origin500, dependent particles/images, no independent time CI'))
if __name__=='__main__':
 threadpool_limits(4);p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);a=p.parse_args();validation() if a.stage=='validation' else test()
