"""Pełne stany wszystkich1500: błędny provenance/partial slot obala zgodność; brak subsamplingu tokenów."""
from pathlib import Path
import os,time,hashlib,json,random
import numpy as np
HERE=Path(__file__).resolve().parent
from przygotuj_eden import read,sha,js,stamp,SOURCE,IMAGES
def shard_indices(rank,world,count=1500):
 assert 0<=rank<world and count>=0
 return np.arange(rank,count,world,dtype=np.int64)
class FrozenWindowReader:
 def __init__(self,cache_dir=None,batch=4,offload=True):
  self.manifest=read(HERE/'MANIFEST_1500.json');
  for r in self.manifest['checkpoints']:r['path']=str(SOURCE/'wagi'/('k%04d.pt'%r['optimizer_step']))
  self.manifest_hash=sha(HERE/'MANIFEST_1500.json');assert self.manifest['count']==1500 and self.manifest['checkpoint_count']==101
  self.X=np.load(IMAGES,mmap_mode='r');assert self.X.shape==(1500,320,480,3)
  self.shape=(101,1500,600,768);self.dtype=np.dtype('float32');self.batch=int(batch);self.offload=offload
  self.cache=Path(cache_dir or os.environ['EXPANDED_CACHE']);self.cache.mkdir(parents=True,exist_ok=True);assert self.cache.resolve().is_relative_to(Path(os.environ['EXPANDED_SCRATCH']).resolve())
  self.cache_mode=os.environ.get('EXPANDED_CACHE_MODE','rolling13');assert self.cache_mode in ('rolling13','full101');self.verified_slots={};self.net=None;self.phase='unset';self.epoch=None;self.active_origin=None;self.events=HERE/'CACHE_EVENTS.jsonl'
  import torch.distributed as dist
  self.world=dist.get_world_size() if dist.is_initialized() else 1
  self.rank=dist.get_rank() if dist.is_initialized() else 0
 def begin_epoch(self,phase='train',epoch=0):
  self.phase=str(phase);self.epoch=int(epoch);self._event(dict(type='begin_epoch',phase=self.phase,epoch=self.epoch))
 def _event(self,row):
  if self.rank!=0:return
  row.update(timestamp=stamp(),phase=self.phase,epoch=self.epoch)
  with self.events.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
 def _slot(self,cp):return cp if self.cache_mode=='full101' else cp%13
 def _paths(self,cp):return self.cache/f'slot_{self._slot(cp):03d}.npy',self.cache/f'slot_{self._slot(cp):03d}.json'
 def _valid(self,cp):
  path,meta=self._paths(cp)
  if not path.exists() or not meta.exists():return False
  row=read(meta)
  if not (row.get('status')=='complete' and row.get('checkpoint_index')==cp and row.get('manifest_sha256')==self.manifest_hash and row.get('checkpoint_sha256')==self.manifest['checkpoints'][cp]['sha256']):return False
  if row.get('shape')!=[1500,600,768] or row.get('dtype')!='float32' or row.get('bytes')!=path.stat().st_size:return False
  if path.stat().st_size!=1500*600*768*4+128:return False
  array=np.load(path,mmap_mode='r')
  if array.shape!=(1500,600,768) or array.dtype!=np.float32:return False
  key=(cp,row['latent_sha256'],path.stat().st_mtime_ns)
  if self.verified_slots.get(self._slot(cp))!=key:
   if sha(path)!=row['latent_sha256']:return False
   self.verified_slots[self._slot(cp)]=key
  return True
 def _network(self):
  import torch
  if self.net is None:
   cpu=torch.random.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
   from model_snapshot import MalyViT
   torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
   self.net=MalyViT().eval();self.net.zapisuj_latent(True)
   for p in self.net.parameters():p.requires_grad_(False)
   torch.random.set_rng_state(cpu)
   if cuda is not None:torch.cuda.set_rng_state_all(cuda)
  return self.net
 def _extract(self,net,imageindices):
  import torch
  inds=np.asarray(imageindices,dtype=np.int64);X=np.ascontiguousarray(self.X[inds]);t=torch.from_numpy(X).permute(0,3,1,2).contiguous().float().cuda()/255.
  with torch.inference_mode():_,z=net(t)
  result=z.detach().float().cpu().numpy().copy();assert result.shape==(len(inds),600,768) and np.isfinite(result).all()
  return result
 def _build(self,cp):
  import torch,shutil
  assert 0<=cp<=100
  path,meta=self._paths(cp);reserve=2*1500*600*768*4
  assert shutil.disk_usage(self.cache).free>reserve,'less than two checkpoint reserves'
  record=self.manifest['checkpoints'][cp];assert sha(record['path'])==record['sha256'];start=time.time()
  js(meta,dict(status='building',checkpoint_index=cp,checkpoint_step=cp*10,shape=[1500,600,768],manifest_sha256=self.manifest_hash))
  net=self._network();ck=torch.load(record['path'],map_location='cpu',weights_only=True);assert ck['krok']==cp*10;net.load_state_dict(ck['net'],strict=True);net.cuda().eval()
  buffers={n:b.detach().clone() for n,b in net.named_buffers()}
  output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(1500,600,768));n=0
  try:
   for start_i in range(0,1500,self.batch):
    inds=np.arange(start_i,min(start_i+self.batch,1500));output[inds]=self._extract(net,inds);n+=len(inds)
   output.flush();assert n==1500;assert all(torch.equal(b,buffers[nm]) for nm,b in net.named_buffers());del output
   stats=[]
   if cp in (0,50,100):
    old=np.load(SOURCE/'dane/LATENT_GESTY.npy',mmap_mode='r');arr=np.load(path,mmap_mode='r')
    for row in self.manifest['matched_source_probes']:
     a=np.asarray(arr[row['index1500']],np.float64);b=np.asarray(old[cp*10,row['source_probe_index']],np.float64);maxabs=float(np.max(abs(a-b)));assert maxabs<=1e-5
     stats.append(dict(**row,maxabs=maxabs,relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b))))
   complete=dict(status='complete',checkpoint_index=cp,checkpoint_step=cp*10,shape=[1500,600,768],dtype='float32',all_finite=True,completed_image_count=1500,image_indices=[0,1499],image_ids_sha256=hashlib.sha256('\n'.join(r['frame_id'] for r in self.manifest['records']).encode()).hexdigest(),manifest_sha256=self.manifest_hash,checkpoint_sha256=record['sha256'],latent_sha256=sha(path),bytes=path.stat().st_size,buffers_unchanged=True,source_comparison=stats,seconds=time.time()-start)
   js(meta,complete);self.verified_slots[self._slot(cp)]=(cp,complete['latent_sha256'],path.stat().st_mtime_ns);self._event(dict(type='checkpoint_complete',**complete));print('cache complete',cp,'step',cp*10,'seconds',round(complete['seconds'],2),flush=True)
  finally:
   if self.offload:net.cpu();torch.cuda.empty_cache()
 def ensure_window(self,origin,width=10):
  origin=int(origin);assert width==10 and 0<=origin<=90
  for cp in range(max(0,origin-2),origin+width+1):
   if self.world==1:
    if not self._valid(cp):self._build(cp)
   else:self._ensure_distributed(cp)
  self.active_origin=origin
  return [str(self._paths(cp)[0]) for cp in range(origin,origin+11)]
 def get_batch(self,image_indices,origin=None):
  origin=self.active_origin if origin is None else int(origin);assert origin is not None
  ids=np.asarray(image_indices,dtype=np.int64);assert ids.ndim==1 and len(ids)>0 and np.all((0<=ids)&(ids<1500))
  assert all(self._valid(cp) for cp in range(origin,origin+11)),'window is not ready'
  return np.stack([np.load(self._paths(cp)[0],mmap_mode='r')[ids] for cp in range(origin,origin+11)]).astype(np.float32,copy=False)
 def get_history_batch(self,image_indices,origin=None):
  origin=self.active_origin if origin is None else int(origin)
  ids=np.asarray(image_indices,dtype=np.int64)
  # Clamp unavailable negative history to cp0; missing flags remain explicit.
  cps=[max(0,origin-2),max(0,origin-1)]+list(range(origin,origin+11))
  assert all(self._valid(cp) for cp in set(cps))
  return np.stack([np.load(self._paths(cp)[0],mmap_mode='r')[ids] for cp in cps]).astype(np.float32,copy=False)
 def get_state(self,image_indices,cp):
  """A single checkpoint; caller phase determines legal cp range."""
  cp=int(cp);assert 0<=cp<=100
  ids=np.asarray(image_indices,dtype=np.int64);assert ids.ndim==1 and np.all((ids>=0)&(ids<1500))
  if self.world>1:assert self._valid(cp),'all ranks must ensure_window before read-only get_state'
  elif not self._valid(cp):self._build(cp)
  return np.array(np.load(self._paths(cp)[0],mmap_mode='r')[ids],dtype=np.float32,copy=True)
 def _gate(self,error,stage):
  """All ranks reach each gate; process-group timeout handles process death."""
  import torch.distributed as dist
  errors=[None]*self.world;dist.all_gather_object(errors,error)
  if any(e is not None for e in errors):raise RuntimeError(f'collective source {stage}: {errors}')
 def _ensure_distributed(self,cp):
  import torch,torch.distributed as dist,shutil
  path,meta=self._paths(cp);record=self.manifest['checkpoints'][cp]
  payload=[None];error=None
  if self.rank==0:
   try:payload[0]=dict(valid=self._valid(cp),meta=read(meta) if meta.exists() else None)
   except Exception as exc:error=repr(exc)
  self._gate(error,'cache validation');dist.broadcast_object_list(payload,src=0)
  if payload[0]['valid']:
   row=payload[0]['meta'];self.verified_slots[self._slot(cp)]=(cp,row['latent_sha256'],path.stat().st_mtime_ns);return
  start=time.monotonic();error=None
  if self.rank==0:
   try:
    assert shutil.disk_usage(self.cache).free>2*1500*600*768*4,'less than two checkpoint reserves'
    assert sha(record['path'])==record['sha256']
    js(meta,dict(status='building',checkpoint_index=cp,checkpoint_step=cp*10,shape=[1500,600,768],manifest_sha256=self.manifest_hash,world_size=self.world))
    output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(1500,600,768));output.flush();del output
   except Exception as exc:error=repr(exc)
  self._gate(error,'header single writer');dist.barrier()
  error=None;net=None;n=0
  try:
   net=self._network();ck=torch.load(record['path'],map_location='cpu',weights_only=True);assert ck['krok']==cp*10
   net.load_state_dict(ck['net'],strict=True);net.cuda().eval();buffers={name:b.detach().clone() for name,b in net.named_buffers()}
   output=np.lib.format.open_memmap(path,mode='r+');indices=shard_indices(self.rank,self.world)
   torch.cuda.synchronize();extract_start=time.monotonic()
   for offset in range(0,len(indices),self.batch):
    inds=indices[offset:offset+self.batch];output[inds]=self._extract(net,inds);n+=len(inds)
   output.flush();del output;torch.cuda.synchronize();seconds=time.monotonic()-extract_start
   assert n==len(indices) and all(torch.equal(b,buffers[name]) for name,b in net.named_buffers())
  except Exception as exc:error=repr(exc)
  finally:
   if net is not None and self.offload:net.cpu();torch.cuda.empty_cache()
  self._gate(error,'disjoint shard extraction')
  shards=[None]*self.world;dist.all_gather_object(shards,dict(rank=self.rank,count=n,first=self.rank,stride=self.world,seconds=seconds))
  error=None;payload=[None]
  if self.rank==0:
   try:
    assert sum(r['count'] for r in shards)==1500
    arr=np.load(path,mmap_mode='r');stats=[]
    if cp in (0,50,100):
     old=np.load(SOURCE/'dane/LATENT_GESTY.npy',mmap_mode='r')
     for row in self.manifest['matched_source_probes']:
      a=np.asarray(arr[row['index1500']],np.float64);b=np.asarray(old[cp*10,row['source_probe_index']],np.float64)
      maxabs=float(np.max(abs(a-b)));assert maxabs<=1e-5
      stats.append(dict(**row,maxabs=maxabs,relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(b))))
    del arr
    complete=dict(status='complete',checkpoint_index=cp,checkpoint_step=cp*10,shape=[1500,600,768],dtype='float32',all_finite=True,completed_image_count=1500,image_indices=[0,1499],image_ids_sha256=hashlib.sha256('\n'.join(r['frame_id'] for r in self.manifest['records']).encode()).hexdigest(),manifest_sha256=self.manifest_hash,checkpoint_sha256=record['sha256'],latent_sha256=sha(path),bytes=path.stat().st_size,buffers_unchanged=True,source_comparison=stats,seconds=time.monotonic()-start,world_size=self.world,shards=shards,single_metadata_writer=0)
    js(meta,complete);self._event(dict(type='checkpoint_complete',**complete));payload[0]=complete
    print('parallel cache complete',cp,'step',cp*10,'seconds',round(complete['seconds'],2),flush=True)
   except Exception as exc:error=repr(exc)
  self._gate(error,'hash complete publication');dist.broadcast_object_list(payload,src=0)
  row=payload[0];self.verified_slots[self._slot(cp)]=(cp,row['latent_sha256'],path.stat().st_mtime_ns)
