"""Frozen eval checkpoints; primary full spatial J128, no training/PCA."""

import os

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='4'

from pathlib import Path

import json,time,argparse,platform,shutil

import numpy as np,torch

from threadpoolctl import threadpool_limits

from datetime import datetime

from zoneinfo import ZoneInfo

from dobor import HERE,SOURCE,JROOT,read,sha,js

from model_snapshot import MalyViT



def stamp():return datetime.now(ZoneInfo('Europe/Warsaw')).isoformat()

def compare(a,b):

 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);delta=a-b;na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))

 return dict(maxabs=float(np.max(abs(delta))),relative_l2=float(np.linalg.norm(delta)/max(nb,1e-30)),rmse=float(np.sqrt(np.mean(delta**2))),cosine=float(np.sum(a*b)/(na*nb)) if na*nb else None,within_maxabs1e_5=bool(np.max(abs(delta))<=1e-5))

def network():

 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

 assert torch.cuda.is_available()

 net=MalyViT().cuda().eval()

 for p in net.parameters():p.requires_grad_(False)

 net.zapisuj_latent(True);return net



def extract(net,X,batch=4):

 buffers={n:b.detach().clone() for n,b in net.named_buffers()};parts=[]

 with torch.inference_mode():

  for j in range(0,len(X),batch):

   t=torch.from_numpy(np.ascontiguousarray(X[j:j+batch])).permute(0,3,1,2).contiguous().float().cuda()/255.

   _,z=net(t);parts.append(z.detach().float().cpu().numpy().copy())

 assert all(torch.equal(b,buffers[n]) for n,b in net.named_buffers()),'model buffers changed'

 result=np.concatenate(parts);assert result.shape==(len(X),600,768) and result.dtype==np.float32 and np.isfinite(result).all()

 return result



def main(pilot=False):

 threadpool_limits(4);start=time.time();manifest=read(HERE/'MANIFEST.json');assert manifest['count']==34 and manifest['fit_count']==30 and manifest['holdout_indices']==[8,9,10,11]

 X=np.load(HERE/'OBRAZY.npy');assert X.shape==(34,320,480,3) and X.dtype==np.uint8

 oldX=np.load(SOURCE/'dane/obrazy.npy');assert np.array_equal(X[:12],oldX)

 with np.load(HERE/'KANALY_J128.npz') as z:J=z['channels0'];assert len(J)==128

 with np.load(JROOT/'KANALY_MODEL.npz') as z:assert np.array_equal(J,z['channels0'])

 assert sha(HERE/'model_snapshot.py')==sha(SOURCE.parents[1]/'kod/model.py')

 net=network();source=np.load(SOURCE/'dane/LATENT_GESTY.npy',mmap_mode='r');assert source.shape==(1001,12,600,768)

 existing_j=np.load(HERE/'LATENT_J128.npy',mmap_mode='r')

 steps=[0,500,1000] if pilot else list(range(0,1001,10));dest=HERE/'LATENT_768.npy';records=[];checkpoints=[]

 if not pilot:

  assert not dest.exists(),'do not overwrite existing extraction'

  assert shutil.disk_usage(HERE).free>17e9

  output=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(101,34,600,768));output[:]=np.nan;output.flush()

 for slot,step in enumerate(steps):

  path=SOURCE/'wagi'/f'k{step:04d}.pt';ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['krok']==step

  net.load_state_dict(ck['net'],strict=True);net.eval();before=time.time()

  z=extract(net,X[:12] if pilot else X)

  old=np.asarray(source[step]);stats=compare(z[:12,:,J],old[:,:,J])

  row=dict(step=step,primary_J128=stats,gpu_seconds=time.time()-before,buffers_unchanged=True,finite=True)

  if step in (0,500,1000):row['full768']=compare(z[:12],old)

  if not pilot:assert np.array_equal(z[:,:,J], existing_j[slot]), 'J128 differs from previous extraction'

  if not pilot:output[slot]=z;output.flush()

  records.append(row);checkpoints.append(dict(step=step,path=str(path),sha256=sha(path),bytes=path.stat().st_size))

  js(HERE/('PILOT_768_POSTEP.json' if pilot else 'EKSTRAKCJA_768_POSTEP.json'),dict(status='running',started=stamp(),completed_steps=len(records),records=records,checkpoint_steps_completed=[r['step'] for r in records]))

  print('checkpoint',step,'maxabs J',stats['maxabs'],'elapsed',round(time.time()-start,1),flush=True)

 aggregate=dict(primary_maxabs=max(r['primary_J128']['maxabs'] for r in records),primary_relative_l2_max=max(r['primary_J128']['relative_l2'] for r in records),all_within_requested1e_5=all(r['primary_J128']['within_maxabs1e_5'] for r in records),full768_maxabs=max(r['full768']['maxabs'] for r in records if 'full768' in r))

 if pilot:

  js(HERE/'PILOT_768_ZGODNOSC.json',dict(status='complete',completed=stamp(),checkpoints=checkpoints,records=records,aggregate=aggregate,seconds=time.time()-start,device=torch.cuda.get_device_name(0)))

 else:

  assert all(np.isfinite(output[k]).all() for k in range(len(output)));del output

  js(HERE/'DANE_768.json',dict(status='complete',completed=stamp(),shape=[101,34,600,768],dtype='float32',checkpoint_steps=steps,model_updates=1000,dataset_presentations=1000*32/1500,fit_indices=manifest['fit_indices'],holdout_indices=manifest['holdout_indices'],channels0=J,channels1=J+1,latent_path='LATENT_768.npy',latent_sha256=sha(dest),bytes=dest.stat().st_size,manifest_sha256=sha(HERE/'MANIFEST.json'),input_images_sha256=sha(HERE/'OBRAZY.npy'),masks_sha256=sha(HERE/'MASKI.npz'),channels_sha256=sha(HERE/'KANALY_J128.npz'),source_arch_sha256=sha(HERE/'model_snapshot.py'),source_latent_sha256_from_metadata=read(SOURCE/'dane/metadata.json')['latent_sha256'],checkpoints=checkpoints,consistency_records=records,consistency_aggregate=aggregate,measurement=dict(eval=True,no_grad=True,autocast=False,tf32=False,batch_size=4,normalization='RGB uint8/255 then source ImageNet mean/std inside forward',layer='bloki[-1].mlp.act GELU',grid=[20,30],buffers_unchanged=True,movement_includes='weights and BatchNorm running statistics'),versions=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version()),device=torch.cuda.get_device_name(0),seconds=time.time()-start,new_training=False,all_finite=True,existing_J128_unchanged=True,existing_J128_sha256=sha(HERE/'LATENT_J128.npy'),plan_sha256=sha(HERE/'PLAN_EKSTRAKCJA_768.md'),extraction_code_sha256=sha(HERE/'ekstrakcja_768.py')))

 progress=read(HERE/('PILOT_768_POSTEP.json' if pilot else 'EKSTRAKCJA_768_POSTEP.json'));progress.update(status='complete',completed=stamp(),seconds=time.time()-start);js(HERE/('PILOT_768_POSTEP.json' if pilot else 'EKSTRAKCJA_768_POSTEP.json'),progress)
 print('COMPLETE',pilot,aggregate,round(time.time()-start,1),flush=True)

if __name__=='__main__':

 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)

