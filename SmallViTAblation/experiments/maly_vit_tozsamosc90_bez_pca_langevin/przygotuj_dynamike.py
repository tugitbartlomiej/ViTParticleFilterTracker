"""Extract exactly the frozen native coordinates J, without any rotated basis."""
from common import *
from threadpoolctl import threadpool_limits
import time,shutil

def main():
 threadpool_limits(4);start=time.time();choice=read(HERE/'WYBOR_WALIDACJA.json');assert choice['status']=='frozen_before_native_classifier_test';assert choice['model_sha']==sha(HERE/'KANALY_MODEL.npz')
 J=np.load(HERE/'KANALY_MODEL.npz')['channels0'];m=dict(np.load(SOURCE/'dane/maski.npz'));images=read(SOURCE/'dane/images.json');latent=np.load(SOURCE/'dane/LATENT_GESTY.npy',mmap_mode='r');assert latent.shape==(1001,12,600,768);out={'channels0':J,'channels1':J+1};probes={}
 for probe in (2,8):
  ids=np.flatnonzero(m['tool'][probe]);bgids=np.flatnonzero(m['background'][probe]);w=m['coverage'][probe,ids].astype(float);w/=w.sum();out[f's_{probe}']=np.empty((1001,len(ids),len(J)),np.float32);out[f'bg_{probe}']=np.empty((1001,len(J)),float);out[f'w_{probe}']=w;out[f'ids_{probe}']=ids
  probes[str(probe)]=dict(frame_id=images[probe]['frame_id'],source_role=images[probe]['role'],split=images[probe]['split'],tool_count=len(ids),tool_ids0=ids,tool_ids1=ids+1,background_ids0=bgids,coverage=m['coverage'][probe,ids],weights=w)
 for t in range(1001):
  for probe in (2,8):
   out[f's_{probe}'][t]=latent[t,probe,out[f'ids_{probe}']][:,J];out[f'bg_{probe}'][t]=latent[t,probe,probes[str(probe)]['background_ids0']][:,J].mean(axis=0,dtype=float)
  if t%250==0:print('native mmap',t,flush=True)
 assert all(np.isfinite(v).all() for v in out.values());np.savez_compressed(HERE/'DYNAMIKA_NATYWNE_J.npz',**out)
 js(HERE/'DANE_DYNAMIKI.json',dict(probes=probes,K=len(J),channels0=J,channels1=J+1,coordinate_definition='s_i,j(t)=z_i,J_j(t), native coordinates noPCA/noSVD/noU/noV',data_sha=sha(HERE/'DYNAMIKA_NATYWNE_J.npz'),channels_model_sha=choice['model_sha'],source_latent_sha_from_verified_receipt=read(OLD/'DANE.json')['latent_source_sha256_from_verified_receipt'],source_masks_sha=sha(SOURCE/'dane/maski.npz'),seconds=time.time()-start,outputs_bytes=(HERE/'DYNAMIKA_NATYWNE_J.npz').stat().st_size,all_finite=True))
 assert shutil.disk_usage(HERE).free>1e9
if __name__=='__main__':main()
