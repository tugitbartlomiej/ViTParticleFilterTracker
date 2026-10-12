"""Real GPU steps, synthetic data equivalence benchmark; not a1500-image fit."""
from pathlib import Path
from datetime import timedelta
import argparse,json,os,time
import numpy as np
import torch
import torch.distributed as dist
from model import ExpandedNNDMD,optimizer_step,distributed_optimizer_step,collective_error_gate,rollout
from state import AuxHistory,Normalizer
from train import training_record,append,atomic_json,save_checkpoint

def run(tokens,output):
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world in (2,3,4,8,11)
    local=0 if torch.cuda.device_count()==1 else local
    torch.cuda.set_device(local)
    backend=os.environ.get('EXPANDED_DISTRIBUTED_BACKEND','nccl')
    assert backend in ('nccl','gloo')
    dist.init_process_group(backend,timeout=timedelta(minutes=10))
    here=Path(__file__).resolve().parent;out=Path(output)
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    cfg=json.loads((here/'CONFIG.json').read_text(encoding='utf-8-sig'))
    aux=AuxHistory.load(here/'AUX_HISTORY.npz');assert aux.q==59
    norm=Normalizer.fit(1.,aux,80)
    def create():return ExpandedNNDMD(59,native_dim=768,learned_dim=768,hidden=768,seed=cfg['seed'],
        device='cuda:'+str(local),lr_network=cfg['lr_network'],lr_generator=cfg['lr_generator'])
    parallel=create();reference=create() if rank==0 else None
    options={k:cfg[k] for k in ['micro_tokens','rec_weight','aux_weight','identity_weight','clip_norm']}
    rows=[]
    try:
        for case,b in enumerate([32,28]):
            raw=np.random.default_rng(800+case).normal(size=(13,b,tokens,768)).astype(np.float32)
            ids=np.arange(b);subset=np.array_split(ids,world)[rank]
            error=None;one=None;seconds_one=None
            if rank==0:
                try:
                    torch.cuda.synchronize();start=time.monotonic()
                    one=optimizer_step(reference,raw,20,aux,norm,80,**options)
                    torch.cuda.synchronize();seconds_one=time.monotonic()-start
                except Exception as exc:error=repr(exc)
            collective_error_gate(error,'reference GPUstep')
            torch.cuda.synchronize()
            start=time.monotonic()
            many=distributed_optimizer_step(parallel,raw[:,subset],20,aux,norm,80,global_images=b,**options)
            torch.cuda.synchronize();seconds=time.monotonic()-start;error=None
            if rank==0:
                try:
                    grad_error=0.;weight_error=0.
                    assert abs(one['loss_normalized']-many['loss_normalized'])<=3e-5+3e-4*abs(one['loss_normalized'])
                    np.testing.assert_allclose(one['prediction_mse_raw'],many['prediction_mse_raw'],rtol=3e-4,atol=3e-5)
                    for (name,a),(other,c) in zip(reference.module.named_parameters(),parallel.module.named_parameters()):
                        assert name==other
                        torch.testing.assert_close(a.grad,c.grad,rtol=3e-4,atol=3e-5)
                        torch.testing.assert_close(a,c,rtol=3e-4,atol=2e-5)
                        grad_error=max(grad_error,float((a.grad-c.grad).abs().max().cpu()))
                        weight_error=max(weight_error,float((a-c).abs().max().cpu()))
                    saved=aux.values[21:].copy();aux.values[21:]=np.nan
                    pred1=rollout(reference,raw[:3],20,aux,norm,10,cfg['micro_tokens'])[0]
                    pred2=rollout(parallel,raw[:3],20,aux,norm,10,cfg['micro_tokens'])[0]
                    aux.values[21:]=saved
                    np.testing.assert_allclose(pred1,pred2,rtol=3e-4,atol=3e-5)
                    row=training_record(many,'synthetic_GPU_equivalence',1,20,case,ids,case+1,seconds,0.)
                    row.update(gradient_max_error=grad_error,weights_max_error=weight_error,
                        decoded_prediction_max_error=float(np.max(np.abs(pred1-pred2))),source='deterministic synthetic fixture',
                        all600=tokens==600,actual_fit_steps=0,actual_preflight_optimizer_steps=case+1,
                        single_GPU_seconds=seconds_one,distributed_GPU_seconds=seconds,
                        timing_note='CUDA synchronized; two cold/early steps, not a warmed epoch ETA')
                    append(out/'TRAINING.jsonl',row);rows.append(row)
                    atomic_json(out/'status.json',dict(phase='GPU_smoke',actual_preflight_optimizer_steps=case+1,actual_fit_steps=0,world_size=world))
                    save_checkpoint(out/'latest.pt',parallel,1,norm,cfg,case+1,'GPU_smoke',20)
                    restored=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
                    assert restored['optimizer_steps']==case+1 and restored['total_optimizer_steps']==case+1
                except Exception as exc:error=repr(exc)
            collective_error_gate(error,'GPU equivalence/telemetry/checkpoint')
        if rank==0:
            atomic_json(out/'RESULT.json',dict(status='PASS',world_size=world,distributed_backend=backend,global_batches=[32,28],tokens_per_image=tokens,
                horizon=10,dimensions={'native':768,'learned':768,'aux':59,'input':2544,'decoder':827},
                source='deterministic synthetic fixture, real GPU optimizer steps; not dataset1500 fitting',
                actual_fit_steps=0,actual_preflight_optimizer_steps=2,comparisons=rows,
                tolerances={'gradient_atol':3e-5,'weights_atol':2e-5,'rtol':3e-4,'prediction_atol':3e-5}))
    finally:dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokens',type=int,default=4);p.add_argument('--output',required=True)
    a=p.parse_args();assert a.tokens in (4,600);run(a.tokens,a.output)
