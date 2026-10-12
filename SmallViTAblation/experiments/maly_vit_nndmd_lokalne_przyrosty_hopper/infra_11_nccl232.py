"""Per-rank assigned UUID binding BEFORE torch import; no NN/model fit."""
from pathlib import Path
from datetime import timedelta,datetime,timezone
import os,json,time,traceback
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
out=Path(os.environ['INFRA_OUTPUT']);mapping=json.loads((out/'UUID_MAP.json').read_text())
device=mapping['devices'][local];os.environ['CUDA_VISIBLE_DEVICES']=device['uuid']
import torch
import torch.distributed as dist
def main():
 import ctypes
 lib=ctypes.CDLL(os.environ['LD_PRELOAD']); actual=ctypes.c_int(); assert lib.ncclGetVersion(ctypes.byref(actual))==0
 loaded_nccl=[x.strip() for x in open('/proc/self/maps') if 'libnccl' in x]
 row=dict(actual_nccl_version=actual.value,loaded_nccl=loaded_nccl,rank=rank,local_rank=local,world_size=world,assigned=device,torch=torch.__version__,cuda_runtime=torch.version.cuda,nccl=list(torch.cuda.nccl.version()),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_device_count=torch.cuda.device_count(),status='enumerating',actual_fit_steps=0,time=datetime.now(timezone.utc).isoformat())
 def save():
  p=out/f'rank_{rank:02d}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2),encoding='utf-8');os.replace(tmp,p)
 save();print(json.dumps(row),flush=True)
 try:
  assert actual.value==23203,'isolated NCCL not loaded; torch.cuda.nccl.version may be compile-time' 
  assert world==11 and row['cuda_device_count']==1,'assigned UUID must expose exactly one device'
  torch.cuda.set_device(0);props=torch.cuda.get_device_properties(0)
  row.update(device_name=props.name,total_memory=props.total_memory,observed_uuid=str(getattr(props,'uuid','unavailable')),compute_capability=[props.major,props.minor],status='device_selected');save()
  dist.init_process_group('nccl',timeout=timedelta(seconds=90),device_id=torch.device('cuda',0))
  value=torch.tensor(float(rank+1),device='cuda:0');torch.cuda.synchronize();start=time.monotonic();dist.all_reduce(value);torch.cuda.synchronize()
  assert float(value.cpu())==66
  row.update(status='PASS',all_reduce_sum=float(value.cpu()),seconds=time.monotonic()-start);save();print(json.dumps(row),flush=True)
 except BaseException:row.update(status='FAIL',error=traceback.format_exc());save();raise
 finally:
  if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
