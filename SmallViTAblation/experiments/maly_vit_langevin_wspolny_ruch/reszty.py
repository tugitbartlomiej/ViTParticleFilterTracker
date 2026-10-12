"""Postfit diagnostics, no tuning: Gaussianity is distinct from whiteness."""
from common import *
from analiza import load_bank,data,old_bank
from model import rollout
from metryki_bazowe import rollout as old_rollout
from scipy.stats import skew,kurtosis,jarque_bera,chi2
from threadpoolctl import threadpool_limits

def diagnostics(e,w):
 T=len(e);lags=min(10,T//5);records=[];shared=np.einsum('tnd,n->td',e,w)
 for name,seq in [('shared',shared)]+[(f'token{j}',e[:,j]) for j in range(len(w))]:
  for j in range(8):
   v=seq[:,j];c=v-v.mean();den=np.sum(c*c);acf=np.array([np.sum(c[:-k]*c[k:])/den if den>1e-30 else 0. for k in range(1,lags+1)]);Q=T*(T+2)*np.sum(acf*acf/(T-np.arange(1,lags+1)));jb=jarque_bera(v)
   records.append(dict(series=name,coordinate0=j,N=T,acf=acf,LjungBox_raw_p=float(chi2.sf(Q,lags)),JarqueBera_p=float(jb.pvalue),skew=float(skew(v,bias=False)),excess_kurtosis=float(kurtosis(v,bias=False)),variance_first_half=float(np.var(v[:T//2])),variance_second_half=float(np.var(v[T//2:]))))
 corr=[]
 for j in range(8):
  C=np.corrcoef(e[:,:,j].T);off=C[np.triu_indices(len(w),1)];off=off[np.isfinite(off)];corr.append(dict(coordinate0=j,median=float(np.median(off)),mean=float(np.mean(off)),count=len(off)))
 energy=float(np.einsum('tnd,n->',e*e,w));common=float(np.sum(shared*shared));return dict(series=records,LB_rejections_raw=sum(r['LjungBox_raw_p']<.05 for r in records),JB_rejections_raw=sum(r['JarqueBera_p']<.05 for r in records),series_count=len(records),cross_token_correlations=corr,shared_energy_fraction=common/energy,
  warning='Time diagnostics per continuous series; rawp uncorrected multiplicity/fitDOF, not iid independent replications; nonreject normality is not Gaussian proof or whiteness; no FDT.')

def main():
 threadpool_limits(4);a=dict(np.load(HERE/'DANE_PROJEKCJE.npz'))
 for dt in (1,10):
  choice=read(HERE/f'WYBOR_DT{dt}.json');selected=choice['selected'];bank=load_bank(dt);old=old_bank(dt);origins=np.arange(800,1000,dt);records={};arrays={}
  for probe in (2,8):
   q=a[f's_{probe}_PCA'][:,:,:8];w=a[f'w_{probe}']
   for name in (selected,'oldGLE'):
    if name=='oldGLE':pred=old_rollout(q,origins,dt,dict(old['selected'],anchor=q[500]))[dt]
    else:
     s,bg,w=data(a,probe,bank[name]['family']);pred=rollout(s,bg,w,origins,dt,bank[name])[0][dt]
    e=q[origins+dt]-pred;assert np.isfinite(e).all();arrays[f'probe{probe}_{name}']=e;records[f'probe{probe}_{name}']=diagnostics(e,w)
  np.savez_compressed(HERE/f'RESZTY_DT{dt}.npz',target_updates=origins+dt,**arrays);js(HERE/f'SZUM_DT{dt}.json',dict(selected=selected,dt=dt,target_updates=origins+dt,postfit_diagnostic_only=True,records=records))
  print('NOISE',dt,{k:(v['LB_rejections_raw'],v['JB_rejections_raw'],v['shared_energy_fraction']) for k,v in records.items()},flush=True)
if __name__=='__main__':main()
