"""Gloo smoke of actual reader shard/gate and shared NPY single-header publication."""
from pathlib import Path
from datetime import timedelta
import os,json,tempfile
import numpy as np
import torch.distributed as dist
from dane import FrozenWindowReader,shard_indices
from przygotuj_eden import js,read

def main():
 dist.init_process_group('gloo',timeout=timedelta(seconds=90))
 rank=dist.get_rank();world=dist.get_world_size()
 reader=FrozenWindowReader.__new__(FrozenWindowReader);reader.rank=rank;reader.world=world
 root=[tempfile.mkdtemp(prefix='reader_owned_') if rank==0 else None]
 dist.broadcast_object_list(root,src=0);root=Path(root[0]);path=root/'tiny.npy';meta=root/'tiny.json'
 try:
  if rank==0:
   js(meta,dict(status='building'))
   arr=np.lib.format.open_memmap(path,mode='w+',shape=(37,3,4),dtype=np.float32);arr.flush();del arr
  dist.barrier();arr=np.lib.format.open_memmap(path,mode='r+')
  inds=shard_indices(rank,world,37);arr[inds]=inds[:,None,None];arr.flush();del arr
  reader._gate(None,'test completed shards')
  if rank==0:
   arr=np.load(path,mmap_mode='r');np.testing.assert_array_equal(arr[:,0,0],np.arange(37))
   groups=[set(shard_indices(r,world,37).tolist()) for r in range(world)]
   assert sum(len(g) for g in groups)==len(set.union(*groups))==37
   js(meta,dict(status='complete',count=37));assert read(meta)['status']=='complete'
  dist.barrier();assert read(meta)['status']=='complete'
  failed=False
  try:reader._gate('intentional shard failure' if rank==world-1 else None,'injected')
  except RuntimeError as exc:failed='intentional shard failure' in str(exc)
  assert failed
  if rank==0:print(json.dumps(dict(status='PASS',world_size=world,disjoint_complete_rows=37,single_header_writer=True,all_rank_error_propagation=True)),flush=True)
 finally:
  dist.barrier()
  if rank==0:
   path.unlink();meta.unlink();root.rmdir()
  dist.destroy_process_group()
if __name__=='__main__':main()
