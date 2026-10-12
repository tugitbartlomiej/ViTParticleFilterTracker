"""Selected native K-coordinate model only, postfit time diagnostics."""
from common import *
from dynamika import load_bank
from model_native import rollout
from scipy.stats import jarque_bera,chi2,skew,kurtosis
from threadpoolctl import threadpool_limits

def main():
 threadpool_limits(4);a=dict(np.load(HERE/'DYNAMIKA_NATYWNE_J.npz'))
 for dt in (1,10):
  model=load_bank(dt)['selected'];origins=np.arange(800,1000,dt);records={};saved={}
  for probe in (2,8):
   q=a[f's_{probe}'].astype(float);bg=a[f'bg_{probe}'];w=a[f'w_{probe}'];pred=rollout(q,bg,w,origins,dt,model)[0][dt];e=q[origins+dt]-pred;D=e.shape[-1];assert D==len(a['channels0'])==128 and np.isfinite(e).all();saved[f'probe{probe}']=e;shared=np.einsum('tnd,n->td',e,w);T=len(e);lags=min(10,T//5);series=[];degenerate=0
   for name,vectors in [('shared',shared)]+[(f'token{j}',e[:,j]) for j in range(len(w))]:
    for j in range(D):
     v=vectors[:,j];c=v-v.mean();den=np.sum(c*c)
     if den<=1e-30:series.append(dict(series=name,coordinate=j,native_channel0=int(a['channels0'][j]),N=T,degenerate=True));degenerate+=1;continue
     acf=np.array([np.sum(c[:-k]*c[k:])/den for k in range(1,lags+1)]);Q=T*(T+2)*np.sum(acf*acf/(T-np.arange(1,lags+1)));jb=jarque_bera(v)
     series.append(dict(series=name,coordinate=j,native_channel0=int(a['channels0'][j]),N=T,degenerate=False,acf=acf,LjungBox_raw_p=float(chi2.sf(Q,lags)),JarqueBera_p=float(jb.pvalue),skew=float(skew(v,bias=False)),excess_kurtosis=float(kurtosis(v,bias=False)),variance_first_half=float(np.var(v[:T//2])),variance_second_half=float(np.var(v[T//2:]))))
   good=[r for r in series if not r['degenerate']];corr=[]
   for j in range(D):
    C=np.corrcoef(e[:,:,j].T);off=C[np.triu_indices(len(w),1)];off=off[np.isfinite(off)];corr.append(dict(coordinate=j,native_channel0=int(a['channels0'][j]),median=float(np.median(off)) if len(off) else None,count=len(off)))
   energy=np.einsum('tnd,n->',e*e,w);records[str(probe)]=dict(K=D,series=series,series_count=len(series),degenerate_count=degenerate,LB_rejections_raw=sum(r['LjungBox_raw_p']<.05 for r in good),JB_rejections_raw=sum(r['JarqueBera_p']<.05 for r in good),cross_token_correlation=corr,shared_energy_fraction=float(np.sum(shared*shared)/energy))
  np.savez_compressed(HERE/f'RESZTY_DT{dt}.npz',target_updates=origins+dt,**saved);js(HERE/f'SZUM_DT{dt}.json',dict(dt=dt,K=128,target_updates=origins+dt,probes=records,postfit_only=True,limitations='Time series pertoken/percoordinate not flattened; Gaussianity distinct from whiteness; rawp uncorrected multiplicity/fitDOF; dt10 only20 samples, noFDT'))
  print('NATIVE RESIDUAL',dt,{p:(r['LB_rejections_raw'],r['JB_rejections_raw'],r['series_count']) for p,r in records.items()},flush=True)
if __name__=='__main__':main()
