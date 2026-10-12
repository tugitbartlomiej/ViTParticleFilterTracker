"""Parallel exact-source window preflight, no NNDMD fitting."""
from datetime import timedelta
from pathlib import Path
import os,json,time
import torch
import torch.distributed as dist
from dane import FrozenWindowReader
from przygotuj_eden import js
def main():
 local=0 if torch.cuda.device_count()==1 else int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
 backend=os.environ.get('EXPANDED_DISTRIBUTED_BACKEND','nccl')
 dist.init_process_group(backend,timeout=timedelta(minutes=30),**({'device_id':torch.device('cuda',local)} if backend=='nccl' else {}))
 rank=dist.get_rank();reader=FrozenWindowReader();start=time.monotonic()
 try:
  reader.begin_epoch('parallel_source_preflight',0);reader.ensure_window(0,10)
  error=None
  try:
   ids=list(range(rank,32,dist.get_world_size()));raw=reader.get_history_batch(ids,0)
   assert raw.shape==(13,len(ids),600,768)
  except Exception as exc:error=repr(exc)
  reader._gate(error,'preflight history read')
  if rank==0:
   row=dict(status='PASS',world_size=dist.get_world_size(),source_window=[0,10],
    images_per_checkpoint=1500,all600=True,all768=True,seconds=time.monotonic()-start,
    scratch=os.environ['EXPANDED_SCRATCH'],cache=os.environ['EXPANDED_CACHE'],
    cache_mode=os.environ['EXPANDED_CACHE_MODE'],actual_fit_steps=0)
   js(Path(__file__).resolve().parent/'PARALLEL_SOURCE_PREFLIGHT.json',row);print(json.dumps(row),flush=True)
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
