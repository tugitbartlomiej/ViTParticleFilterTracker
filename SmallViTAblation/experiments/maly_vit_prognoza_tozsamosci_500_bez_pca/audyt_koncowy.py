"""Powtórzenie prognoz zamrożonych modeli; bez fitu i zmian wyboru."""
import badanie as b
import dane as data
from metryki import measure
import numpy as np
from threadpoolctl import threadpool_limits

def main():
 threadpool_limits(4);choice=data.read(data.HERE/'WYBOR_PRZED_TESTEM.json');test=data.read(data.HERE/'OCENA_H500.json')
 assert choice['code_hashes']==b.code_hashes()
 assert choice['models_sha']==data.sha(data.HERE/'MACIERZE_PRZED_TESTEM.npz')
 assert choice['models_metadata_sha']==data.sha(data.HERE/'MODELE_PRZED_TESTEM.json')
 assert choice['data_sha']==data.sha(data.HERE/'DANE_J128.npz')
 assert test['selection_sha']==data.sha(data.HERE/'WYBOR_PRZED_TESTEM.json')
 qs,bgs,ws=data.load();models=b.load_models();cases=[];operator_errors=[]
 for ident,m in models.items():
  for key in ('fg','bg'):
   if key in m and m[key] is not None:
    op=m[key];X=op['center_x'][None]+np.random.default_rng(508).normal(size=(4,len(op['center_x'])))*op['scale_x'][None]
    standard=op['B']*op['scale_x'][:,None]/op['scale_y'][None]
    e=float(np.max(abs((X@op['B']+op['b'])-((X-op['center_x'])/op['scale_x']@standard*op['scale_y']+op['center_y']))))
    assert e<1e-8;operator_errors.append(dict(id=ident,operator=key,max_error=e))
  with np.load(data.HERE/f'PROGNOZY_{ident}.npz') as saved:
   for i in (2,8):
    q=qs[i];bg=bgs[i];w=ws[i]
    prefix=b.predict_model(q[:501],bg[:501],w,500,500,m)
    qp=q.copy();bp=bg.copy();qp[501:]=np.nan;bp[501:]=np.nan
    poison=b.predict_model(qp,bp,w,500,500,m)
    assert np.array_equal(prefix,poison,equal_nan=True)
    assert prefix.shape==(501,len(w),128)
    err=float(np.max(abs(prefix-saved[f'q_{i}'].astype(float))));assert np.array_equal(prefix.astype(saved[f'q_{i}'].dtype),saved[f'q_{i}'])
    errors=[]
    for h in (1,10,100,500):
     rec=measure(q[500+h],prefix[h],q[500],w);orig=test['records'][ident]['perimage'][i]['by_horizon'][str(h)]
     for k in ('mse','skill','sign_accuracy','cosine','norm_ratio'):
      if rec[k] is not None:errors.append(abs(rec[k]-orig[k]))
    assert max(errors,default=0)<1e-12
    cases.append(dict(id=ident,probe=i,H=500,prefix_poison_equal=True,forecast_exact_after_storage_cast=True,float32_storage_maxabs=err,metrics_maxabs=max(errors,default=0),finite=bool(np.isfinite(prefix).all())))
 with np.load(data.HERE/'DANE_J128.npz') as d:
  J=d['channels0'];source=np.load(data.SOURCE/'LATENT_GESTY.npy',mmap_mode='r');sourcechecks=[]
  for i in range(12):
   ids=d[f'ids_{i}'];errors=[float(np.max(abs(source[t,i,ids][:,J]-d[f'q_{i}'][t]))) for t in (0,500,1000)]
   assert max(errors)==0;sourcechecks.append(dict(probe=i,states=[0,500,1000],maxabs=0))
 result=dict(status='passed',completed=b.now(),code_hashes=b.code_hashes(),selection_sha=data.sha(data.HERE/'WYBOR_PRZED_TESTEM.json'),test_sha=data.sha(data.HERE/'OCENA_H500.json'),cases=cases,native_standardized_checks=operator_errors,native_source_checks=sourcechecks,refits_performed=False,warning='Tests verify arithmetic and causal use of observations; they do not guarantee predictive accuracy or independence.')
 data.save_json(data.HERE/'AUDYT_KONCOWY.json',result);print('AUDIT PASSED',len(cases),len(operator_errors),flush=True)
if __name__=='__main__':main()
