"""Basis fit <=700, never refitted; full 768D states remain in RAM."""
from common import *
import time,shutil
from threadpoolctl import threadpool_limits

def main():
 threadpool_limits(4);start=time.time();assert (HERE/'PLAN.md').exists()
 latent=np.load(SOURCE/'dane/LATENT_GESTY.npy',mmap_mode='r');m=dict(np.load(SOURCE/'dane/maski.npz'));images=read(SOURCE/'dane/images.json')
 U=np.load(OLD/'IDENTYFIKACJA_K500.npz')['U500'];raw={};background={};meta={}
 for probe in (2,8):
  ids=np.flatnonzero(m['tool'][probe]);bgids=np.flatnonzero(m['background'][probe]);weights=m['coverage'][probe,ids].astype(float);weights/=weights.sum()
  raw[probe]=np.empty((1001,len(ids),768),np.float32);background[probe]=np.empty((1001,768),np.float64)
  meta[probe]=dict(probe=probe,frame_id=images[probe]['frame_id'],source_role=images[probe]['role'],split=images[probe]['split'],tool_ids0=ids,tool_ids1=ids+1,weights=weights,background_ids0=bgids,coverage=m['coverage'][probe,ids])
 for t in range(1001):
  for probe in (2,8):
   ids=meta[probe]['tool_ids0'];bgids=meta[probe]['background_ids0'];raw[probe][t]=latent[t,probe,ids];background[probe][t]=np.mean(latent[t,probe,bgids],axis=0,dtype=np.float64)
  if t%200==0:print('mmap',t,flush=True)
 z=raw[2].astype(float);w=meta[2]['weights'];delta=z[501:701]-z[500:700];delta-=((delta@U)@U.T);mu=np.einsum('tnd,n->d',delta,w)/200;centered=(delta-mu).reshape(-1,768);weights=np.tile(w/200,200)
 covariance=centered.T@(centered*weights[:,None]);eigen,vec=np.linalg.eigh(covariance);Vpca=vec[:,np.argsort(eigen)[::-1][:8]];Vpca-=U@(U.T@Vpca);Vpca=np.linalg.qr(Vpca)[0]
 # Completed fit pairs: residual increment at t=500..699 -> Delta q(t+1), targets 501..700.
 d0=z[500:700]-z[499:699];d0-=((d0@U)@U.T);y=(z[501:701]-z[500:700])@U
 mx=np.einsum('tnd,n->d',d0,w)/200;my=np.einsum('tnd,n->d',y,w)/200
 cross=(d0-mx).reshape(-1,768).T@((y-my).reshape(-1,8)*weights[:,None]);Vpls,singular,_=np.linalg.svd(cross,full_matrices=False);Vpls-=U@(U.T@Vpls);Vpls=np.linalg.qr(Vpls)[0]
 bases={'PCA':Vpca,'PLS':Vpls};out={};info=dict(probes=meta,source_sha_from_receipt=read(OLD/'DANE.json')['latent_source_sha256_from_verified_receipt'],U500_sha=sha(OLD/'IDENTYFIKACJA_K500.npz'),fit_target_max=700,PLS_predictors_t=[500,699],PCA_increment_targets=[501,700],basis_refit=False)
 for method,V in bases.items():
  assert np.max(np.abs(U.T@V))<1e-12 and np.max(np.abs(V.T@V-np.eye(8)))<1e-12
  out['V_'+method]=V
  for probe in (2,8):out[f's_{probe}_{method}']=np.concatenate([raw[probe]@U,raw[probe]@V],axis=-1);out[f'bg_{probe}_{method}']=np.concatenate([background[probe]@U,background[probe]@V],axis=-1)
  info[method]=dict(U_V_max=float(np.max(np.abs(U.T@V))),orthogonality_max=float(np.max(np.abs(V.T@V-np.eye(8)))))
 out['U500']=U;out['PCA_eigenvalues']=eigen[::-1];out['PLS_singularvalues']=singular;out['PCA_center']=mu;out['PLS_x_center']=mx;out['PLS_y_center']=my
 for probe in (2,8):out[f'w_{probe}']=meta[probe]['weights'];out[f'ids_{probe}']=meta[probe]['tool_ids0']
 assert all(np.isfinite(v).all() for v in out.values());np.savez_compressed(HERE/'DANE_PROJEKCJE.npz',**out)
 info.update(seconds=time.time()-start,output_sha=sha(HERE/'DANE_PROJEKCJE.npz'),all_finite=True,raw768_persisted=False,bytes_output=(HERE/'DANE_PROJEKCJE.npz').stat().st_size)
 js(HERE/'DANE.json',info);assert shutil.disk_usage(HERE).free>1e9;print('ready',info['seconds'],info['bytes_output'],flush=True)
if __name__=='__main__':main()
