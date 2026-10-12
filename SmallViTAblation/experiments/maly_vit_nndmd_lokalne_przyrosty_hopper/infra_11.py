"""Eleven CUDA devices communication attempt; four full H200 plus seven MIG, no NN fit."""
from pathlib import Path
from datetime import timedelta,datetime,timezone
import os,json,time,traceback
import torch
import torch.distributed as dist
def main():
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
 out=Path(os.environ['INFRA_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
 row=dict(rank=rank,local_rank=local,world_size=world,torch=torch.__version__,cuda_runtime=torch.version.cuda,
  nccl=list(torch.cuda.nccl.version()),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
  cuda_device_count=torch.cuda.device_count(),status='enumerating',actual_fit_steps=0,
  scope='infrastructure attempt:4fullH200+7MIG, not11full GPUs',time=datetime.now(timezone.utc).isoformat())
 def save():
  p=out/f'rank_{rank:02d}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2),encoding='utf-8');os.replace(tmp,p)
 save();print(json.dumps(row),flush=True)
 try:
  assert world==11
  assert row['cuda_device_count']>=world,f'CUDA sees {row["cuda_device_count"]}, requested11 devices; rank{rank}'
  torch.cuda.set_device(local);props=torch.cuda.get_device_properties(local)
  row.update(device_name=props.name,total_memory=props.total_memory,device_uuid=str(getattr(props,'uuid','unavailable')),
   compute_capability=[props.major,props.minor],status='device_selected');save()
  dist.init_process_group('nccl',timeout=timedelta(seconds=90),device_id=torch.device('cuda',local))
  value=torch.tensor(float(rank+1),device='cuda');torch.cuda.synchronize();start=time.monotonic()
  dist.all_reduce(value);torch.cuda.synchronize()
  assert float(value.cpu())==66
  row.update(status='PASS',all_reduce_sum=float(value.cpu()),seconds=time.monotonic()-start);save();print(json.dumps(row),flush=True)
 except BaseException:
  row.update(status='FAIL',error=traceback.format_exc());save();raise
 finally:
  if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
