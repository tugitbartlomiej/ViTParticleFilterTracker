"""Two GPU infra proof + frozen single-GPU real model optimizer step. NOT distributed model equivalence."""
import os,json,time,hashlib
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local);torch.set_num_threads(4)
dist.init_process_group('nccl',timeout=timedelta(minutes=10))
assert world==2 and torch.cuda.device_count()==2
root=Path(os.environ['INFRA_RUN']);root.mkdir(exist_ok=True)
value=torch.tensor([rank+1.],device='cuda');dist.all_reduce(value);assert value.item()==3.
proof={'kind':'two-GPU infrastructure preflight, not distributed model-gradient equivalence','world_size':world,'backend':dist.get_backend(),'all_reduce_expected':3.,'all_reduce_actual':float(value.item()),'rank':rank,'gpu':torch.cuda.get_device_name(local),'compute_capability':list(torch.cuda.get_device_capability(local))}
error=None
if rank==0:
 try:
  from model import ExpandedNNDMD,optimizer_step
  from state import AuxHistory,Normalizer
  old=Path(os.environ['FROZEN_OLD_RUN']);aux=AuxHistory.load(old/'AUX_HISTORY.npz');norm=Normalizer.fit(1.,aux,80)
  raw=np.random.default_rng(20261002).normal(0,.2,size=(13,32,600,768)).astype(np.float32)
  model=ExpandedNNDMD(aux.q,seed=20261002,hidden=768,lr_network=.001,lr_generator=1e-5,device='cuda')
  start=time.monotonic();stats=optimizer_step(model,raw,2,aux,norm,80,micro_tokens=1024,rec_weight=1.,aux_weight=.1,identity_weight=.001,clip_norm=1.)
  torch.cuda.synchronize();assert model.optimizer_steps==1
  proof.update(real_single_gpu_optimizer_step=True,actual_preflight_optimizer_steps=1,actual_fit_steps=0,logical_batch_images=32,particles_per_image=600,horizon=10,fixture='deterministic synthetic; not exact1500 training',step_seconds=time.monotonic()-start,stats=stats,distributed_model_equivalence=False)
  source=Path(os.environ['FROZEN_EXECUTED_SOURCE'])
  proof['frozen_source_hashes']={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['model.py','state.py']}
  torch.save({'state_dict':model.module.state_dict(),'optimizer':model.optimizer.state_dict(),'optimizer_steps':model.optimizer_steps,'proof':proof},root/'INFRA_CHECKPOINT.pt')
  (root/'INFRA_TELEMETRY.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n',encoding='utf-8')
  with (root/'INFRA_STEPS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(proof,allow_nan=False)+'\n')
  print(json.dumps(proof,allow_nan=False),flush=True)
 except Exception as exc:error=f'{type(exc).__name__}: {exc}'
errors=[error];dist.broadcast_object_list(errors,src=0)
if errors[0]:raise RuntimeError(errors[0])
dist.barrier();dist.destroy_process_group()
