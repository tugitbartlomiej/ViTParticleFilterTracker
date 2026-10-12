"""Natywne J128 dla 12 stałych obrazów; brak fitu i zmiany bazy."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
from pathlib import Path
import numpy as np, json, hashlib, time
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane'
PREVIOUS=HERE.parent/'maly_vit_tozsamosc90_bez_pca_langevin'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(p,x):
 def conv(v):
  if isinstance(v,np.ndarray): return v.tolist()
  if isinstance(v,np.generic): return v.item()
  if isinstance(v,Path): return str(v)
  raise TypeError(type(v))
 Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=conv,allow_nan=False)+'\n',encoding='utf-8')
def load():
 with np.load(HERE/'DANE_J128.npz') as a:
  return ([a[f'q_{i}'].astype(np.float64) for i in range(12)],
          [a[f'bg_{i}'].astype(np.float64) for i in range(12)],
          [a[f'w_{i}'].copy() for i in range(12)])
def main():
 start=time.time()
 choice=read(PREVIOUS/'WYBOR_WALIDACJA.json')
 assert choice['model_sha']==sha(PREVIOUS/'KANALY_MODEL.npz')
 with np.load(PREVIOUS/'KANALY_MODEL.npz') as a: J=a['channels0'].copy()
 assert len(J)==128 and len(set(J.tolist()))==128
 with np.load(SOURCE/'maski.npz') as a: masks={k:a[k] for k in a.files}
 images=read(SOURCE/'images.json'); meta=read(SOURCE/'metadata.json')
 z=np.load(SOURCE/'LATENT_GESTY.npy',mmap_mode='r')
 assert z.shape==(1001,12,600,768) and z.dtype==np.float32
 out={'channels0':J,'channels1':J+1,'steps':np.arange(1001)};probes=[]
 for i,row in enumerate(images):
  ids=np.flatnonzero(masks['tool'][i]); bg=np.flatnonzero(masks['background'][i])
  cov=masks['coverage'][i,ids].astype(np.float64);w=cov/cov.sum()
  out[f'q_{i}']=np.empty((1001,len(ids),128),np.float32)
  out[f'bg_{i}']=np.empty((1001,128),np.float64)
  out[f'w_{i}']=w;out[f'ids_{i}']=ids;out[f'coverage_{i}']=cov
  probes.append(dict(index=i,frame_id=row['frame_id'],source_role=row['role'],split=row['split'],tool_count=len(ids),background_count=len(bg),tool_ids0=ids,tool_ids1=ids+1,background_ids0=bg,coverage=cov,weights=w))
 for t in range(1001):
  for i,r in enumerate(probes):
   out[f'q_{i}'][t]=z[t,i,r['tool_ids0']][:,J]
   out[f'bg_{i}'][t]=z[t,i,r['background_ids0']][:,J].mean(axis=0,dtype=np.float64)
  if t%250==0: print('mmap preparation',t,flush=True)
 assert sum(r['tool_count'] for r in probes)==218
 assert all(np.isfinite(v).all() for v in out.values())
 np.savez_compressed(HERE/'DANE_J128.npz',**out)
 save_json(HERE/'DANE.json',dict(created_at=time.strftime('%Y-%m-%d %H:%M:%S'),timezone='Europe/Warsaw',source=str(SOURCE/'LATENT_GESTY.npy'),source_latent_sha256_from_metadata=meta['latent_sha256'],source_masks_sha256=sha(SOURCE/'maski.npz'),source_images_sha256=sha(SOURCE/'images.json'),channels_model_sha256=choice['model_sha'],channels0=J,channels1=J+1,probes=probes,states=1001,updates=1000,foreground_tokens=218,coordinate_definition='q_i,j(t)=z_i,J_j(t); native indices, no rotation or dimensionality reduction',background_definition='uniform mean over fixed source background mask; foreground weighted by original coverage per image',finite=True,seconds=time.time()-start,data_sha256=sha(HERE/'DANE_J128.npz'),bytes=(HERE/'DANE_J128.npz').stat().st_size,fit_performed=False))
if __name__=='__main__': main()
