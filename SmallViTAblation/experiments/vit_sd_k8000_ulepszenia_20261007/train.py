"""Controlled follow-up arms from own K8000 best: no media/latent disk cache."""
import argparse, contextlib, hashlib, json, math, os, random, signal, sys, time
from datetime import timedelta
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from model import new_vae,Denoiser,teacher,features,schedule,VIT_SHA
from dane import VideoFrames,load_manifest
from sampler import MixedVideoFrames
from evaluation import evaluate,fixed_samples,aggregate_records
from training_utils import ARMS,BASE_SHA,EMA,learning_rate,roi_relative_error,roi_ddp_weight,final_from_checkpoint,stop_reason

INTERRUPTED=False
def interrupt(*_):
    global INTERRUPTED
    INTERRUPTED=True
for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,interrupt)
if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1,interrupt)
def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def atomic_json(path,obj):
    path=Path(path); tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8'); os.replace(tmp,path)
def atomic_model(path,obj):
    tmp=path.with_suffix('.pt.tmp'); torch.save(obj,tmp); os.replace(tmp,path)
def finite_or_none(x): return float(x) if math.isfinite(x) else None

def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','vit','checkpoint','output','sampler-selection'): p.add_argument('--'+name,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--steps',type=int,default=4000); p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--frame-stride',type=int,default=600); p.add_argument('--film-buffer',type=int,default=8)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.eval_every,a.batch_size,a.frame_stride,a.film_buffer)<1 or a.workers<0: p.error('Invalid positive parameters')
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(hours=2))
    def barrier():
        if world>1: dist.barrier()
    def broadcast(obj):
        box=[obj]
        if world>1: dist.broadcast_object_list(box,src=0,device=device)
        return box[0]
    def gather(obj):
        boxes=[None]*world
        if world>1: dist.all_gather_object(boxes,obj)
        else: boxes=[obj]
        return boxes
    root=Path(a.output); out=root/a.arm
    out.mkdir(parents=True,exist_ok=True)
    def halt():
        x=torch.tensor(2 if (root/'STOP').exists() else int(INTERRUPTED or (root/'STOP_TIME').exists()),device=device)
        if world>1: dist.all_reduce(x,op=dist.ReduceOp.MAX)
        return int(x.item())
    if (out/'FINAL.json').exists(): return 0
    if (root/'STOP').exists(): return 0
    rows=load_manifest(a.manifest)
    train_films={r['video_id'] for r in rows if r['split']=='train'}
    val_films={r['video_id'] for r in rows if r['split']=='val'}
    if not train_films or not val_films or train_films & val_films: raise ValueError('Require disjoint TRAIN/VAL films')
    selection=json.loads(Path(a.sampler_selection).read_text(encoding='utf8'))
    if not a.smoke and not selection.get('production_selection_valid'):
        raise ValueError('A smoke result cannot select a production sampler')
    ddim_steps=2 if a.smoke else int(selection['steps'])
    spacing=selection['spacing']
    seeds=(20261007,) if a.smoke else (20261007,20261008)
    if sha(a.checkpoint)!=BASE_SHA: raise ValueError('Wrong starting best checkpoint')
    base=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if base['signature']['teacher_sha256']!=VIT_SHA or base['step']!=7500: raise ValueError('Wrong baseline provenance')
    if base['signature']['manifest_sha256']!=sha(a.manifest): raise ValueError('Different split from baseline')
    signature=dict(version=1,arm=a.arm,base_sha256=BASE_SHA,teacher_sha256=VIT_SHA,
        manifest_sha256=sha(a.manifest),steps=a.steps,eval_every=a.eval_every,
        batch_size=a.batch_size,workers=a.workers,world_size=world,frame_stride=a.frame_stride,
        film_buffer=a.film_buffer,smoke=a.smoke,ddim_steps=ddim_steps,spacing=spacing,
        seeds=list(seeds),initialization='own_random_origin_sd_step7500_new_optimizer',
        full_latent_shape=[600,768],roi_coefficient=.05 if a.arm=='D_roi' else 0)
    latest=out/'latest_checkpoint.pt'
    state=torch.load(latest,map_location='cpu',weights_only=False) if latest.exists() else None
    if state and state['signature']!=signature: raise ValueError('Incompatible resume settings')
    if state and state.get('finished'):
        if state.get('reason')=='manual_stop': return 0
        if rank==0: atomic_json(out/'FINAL.json',final_from_checkpoint(state,a.arm,a.smoke))
        barrier()
        return 0
    torch.manual_seed(20261007); random.seed(20261007); np.random.seed(20261007)
    vae=new_vae().to(device); net=Denoiser().to(device); vit=teacher(a.vit,device)
    vae.load_state_dict(base['vae'],strict=True)
    net.load_state_dict(state['net'] if state else base['net'],strict=True)
    scale=float(base['scale']); vae.eval().requires_grad_(False); vit.eval().requires_grad_(False)
    net.train().requires_grad_(True)
    ema=EMA(net) if a.arm in ARMS[2:] else None
    if state and ema: ema.load_state_dict(state['ema'])
    model=DDP(net,device_ids=[local],broadcast_buffers=False) if world>1 else net
    opt=torch.optim.AdamW(model.parameters(),lr=learning_rate(a.arm,0,a.steps),weight_decay=.01)
    if state: opt.load_state_dict(state['optimizer'])
    step=int(state['new_steps']) if state else 0
    seen=set(state.get('seen_films',[])) if state else set()
    best=float(state['best']) if state else float('inf')
    stale=int(state['stale']) if state else 0
    degradation=int(state.get('degradation',0)) if state else 0
    result=state.get('evaluation') if state else None
    baseline_score=state.get('baseline_score') if state else None
    # DataLoader worker stream is deliberately restarted after a restart.
    # Epoch offset is recorded; weights/optimizer/EMA and per-rank RNG resume.
    epoch_offset=int(state.get('data_restart_epoch',0))+1 if state else 0
    data=(VideoFrames(a.manifest,rank=rank,world_size=world,seed=20261007,repeat=True,frame_stride=a.frame_stride)
          if a.arm=='A_control' else MixedVideoFrames(a.manifest,rank=rank,world_size=world,seed=20261007,
              repeat=True,frame_stride=a.frame_stride,film_buffer=a.film_buffer,epoch_offset=epoch_offset))
    loader=torch.utils.data.DataLoader(data,batch_size=a.batch_size,num_workers=a.workers,pin_memory=True,
         **(dict(prefetch_factor=1,multiprocessing_context='spawn') if a.workers else {}))
    # Load fixed natural references in memory only; deterministic film partitions.
    val_rows=[r for r in rows if r['split']=='val']
    train_rows=[r for r in rows if r['split']=='train']
    if a.smoke:
        val_rows=val_rows[:max(2,world)]
        train_rows=train_rows[:max(2,world)]
    else:
        ids=np.linspace(0,len(train_rows)-1,3,dtype=int).tolist()
        train_rows=[train_rows[i] for i in ids]
    local_val_rows=val_rows[rank::world]
    local_train_rows=train_rows[rank::world]
    fractions=(.33,) if a.smoke else (.33,.67)
    val_samples=[]; train_samples=[]; rejected=[]
    err=None
    try:
        if local_val_rows:
            val_samples,bad=fixed_samples(local_val_rows,'val',fractions=fractions); rejected+=bad
        if local_train_rows:
            train_samples,bad=fixed_samples(local_train_rows,'train',fractions=fractions); rejected+=bad
    except Exception as e: err=f'rank{rank}: {type(e).__name__}: {e}'
    errors=gather(err)
    if any(errors): raise RuntimeError(str(errors))
    if not any(gather(bool(val_samples))): raise RuntimeError('No validation samples')
    rejection_all=sum(gather(rejected),[])
    if rank==0:
        atomic_json(out/'PROVENANCE.json',dict(signature=signature,base_step=7500,
             new_optimizer=not bool(state),resume_new_steps=step,frozen_vae=True,frozen_vit=True,
             data_restart_epoch=epoch_offset,bitwise_dataloader_resume=False,
             no_disk_images_or_latents=True,rejected_evaluation=rejection_all,
             code_sha256={x.name:sha(x) for x in Path(__file__).parent.glob('*.py')}))
    barrier()
    if state:
        rng=state['rng'][rank]
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'])
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    else:
        torch.manual_seed(20261007+rank); random.seed(20261007+rank); np.random.seed(20261007+rank)
    del base
    stream=iter(loader)
    scheduler=schedule()
    start=time.time()
    def coverage():
        seen.update(f for part in gather(sorted(seen)) for f in part)
        return len(seen & train_films)/len(train_films)
    def checkpoint(finished=False,reason=None):
        cov=coverage()
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                 python=random.getstate(),numpy=np.random.get_state())
        rngs=gather(rng)
        if rank==0:
            obj=dict(signature=signature,net=net.state_dict(),vae=vae.state_dict(),scale=scale,
                optimizer=opt.state_dict(),ema=ema.state_dict() if ema else None,rng=rngs,
                new_steps=step,step=7500+step,best=best,stale=stale,degradation=degradation,
                evaluation=result,baseline_score=baseline_score,seen_films=sorted(seen),
                data_restart_epoch=epoch_offset,finished=finished,reason=reason)
            atomic_model(latest,obj)
            atomic_json(out/'LATEST.json',dict(arm=a.arm,new_steps=step,step=7500+step,
                best=finite_or_none(best),finished=finished,reason=reason,coverage=cov))
        barrier()
    def run_eval():
        records_val=[]; records_train=[]; error=None
        try:
            with (ema.applied(net) if ema else contextlib.nullcontext()):
                if val_samples:
                    records_val=evaluate(vae,net,vit,val_samples,scale,device,steps=ddim_steps,
                        spacing=spacing,seeds=seeds,stop_check=lambda:INTERRUPTED or
                        (root/'STOP').exists() or (root/'STOP_TIME').exists())['frames']
                if train_samples:
                    records_train=evaluate(vae,net,vit,train_samples,scale,device,steps=ddim_steps,
                        spacing=spacing,seeds=seeds,stop_check=lambda:INTERRUPTED or
                        (root/'STOP').exists() or (root/'STOP_TIME').exists())['frames']
        except Exception as e: error=f'{type(e).__name__}: {e}'
        errors=gather(error)
        if any(errors):
            if all(e is None or e.startswith('InterruptedError') for e in errors): raise InterruptedError(str(errors))
            raise RuntimeError(str(errors))
        all_val=sum(gather(records_val),[]); all_train=sum(gather(records_train),[])
        net.train()
        return dict(validation=aggregate_records(all_val),
                    train_diagnostic=aggregate_records(all_train) if all_train else None,
                    frames=all_val,train_frames=all_train,new_steps=step,step=7500+step,
                    weight_view='ema' if ema else 'raw',seeds=list(seeds),ddim_steps=ddim_steps,spacing=spacing)
    def eval_update():
        nonlocal result,best,stale,baseline_score,degradation
        result=run_eval()
        # aggregate_records contract is {'film_mean':..., ...}
        metrics=result['validation']['film_mean']
        score=float(metrics['full_relative_mse'])
        if not math.isfinite(score): raise FloatingPointError('Nonfinite validation')
        if baseline_score is None: baseline_score=score
        improved=score<best*.995
        if improved: best=score; stale=0
        elif step>0: stale+=1
        degradation=degradation+1 if score>best*1.03 else 0
        cov=coverage()
        reason=stop_reason(step,a.steps,cov,stale,degradation,metrics,a.smoke)
        result.update(arm=a.arm,best=best,stale=stale,degradation=degradation,
             train_film_coverage=cov,train_films_seen=len(seen & train_films),
             baseline_score=baseline_score,reason=reason)
        if rank==0:
            if improved:
                atomic_model(out/'best_sd.pt',dict(signature=signature,vae=vae.state_dict(),
                    net=ema.shadow if ema else net.state_dict(),scale=scale,step=7500+step,
                    new_steps=step,evaluation=result,weight_view='ema' if ema else 'raw'))
                atomic_json(out/'BEST.json',result)
            atomic_json(out/'EVAL.json',result)
            with (out/'EVAL_HISTORY.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(result,allow_nan=False)+'\n')
            print(json.dumps(dict(arm=a.arm,new_steps=step,validation=metrics,best=best,
                  coverage=cov,stale=stale,reason=reason)),flush=True)
        return reason
    reason=None
    try:
        if result is None:
            if rank==0: atomic_json(root/'STATUS.json',dict(arm=a.arm,state='baseline_validation',new_steps=step))
            eval_update(); checkpoint()
        while step<a.steps:
            stop=halt()
            if stop:
                checkpoint(finished=stop==2,reason='manual_stop' if stop==2 else 'time_limit')
                return 0 if stop==2 else 75
            b=None; error=None
            try: b=next(stream)
            except Exception as e: error=f'{type(e).__name__}: {e}'
            errors=gather(error)
            if any(errors): raise RuntimeError(str(errors))
            seen.update(b['film'])
            x=b['image'].to(device,non_blocking=True)*2-1
            for g in opt.param_groups: g['lr']=learning_rate(a.arm,step,a.steps)
            opt.zero_grad(set_to_none=True)
            roi_value=0.
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.no_grad():
                    z=features(vit,b['vit_image'].to(device,non_blocking=True)).detach()
                    h=vae.encode(x).latent_dist.mode()*scale
                t=torch.randint(0,1000,(len(x),),device=device)
                if a.smoke: t.fill_(100)
                noise=torch.randn_like(h); noisy=scheduler.add_noise(h,noise,t)
                pred=model(noisy,t,z); loss=F.mse_loss(pred.float(),noise.float())
                if step%4==0:
                    roi=pred.sum()*0
                    roi_count=torch.zeros((),device=device,dtype=torch.float32)
                    mask=t<250
                    if mask.any():
                        alpha=scheduler.alphas_cumprod.to(device)[t[mask]][:,None,None,None]
                        estimate=(noisy[mask]-torch.sqrt(1-alpha)*pred[mask])/torch.sqrt(alpha)
                        decoded=((vae.decode(estimate/scale).sample+1)/2).clamp(0,1)
                        actual=features(vit,F.interpolate(decoded.float(),(320,480),mode='bilinear',align_corners=False))
                        target=z[mask].float()
                        full=(actual.float()-target).square().mean()/target.square().mean().clamp_min(1e-6)
                        loss=loss+.05*full
                        if a.arm=='D_roi':
                            roi,roi_count=roi_relative_error(actual,target,b['bbox'].to(device)[mask],b['has_tool'].to(device)[mask],return_count=True)
                    if a.arm=='D_roi':
                        global_count=roi_count.detach().float().clone()
                        if world>1: dist.all_reduce(global_count,op=dist.ReduceOp.SUM)
                        loss=loss+.05*roi*roi_ddp_weight(world,roi_count,global_count)
                        roi_value=float(roi.detach())
            good=torch.tensor(int(torch.isfinite(loss)),device=device)
            if world>1: dist.all_reduce(good,op=dist.ReduceOp.MIN)
            if not good.item(): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if ema: ema.update(net)
            step+=1
            if rank==0 and (step%10==0 or a.smoke):
                stat=dict(arm=a.arm,state='training',new_steps=step,step=7500+step,
                    loss=float(loss.detach()),roi_loss=roi_value,lr=opt.param_groups[0]['lr'],
                    seconds=time.time()-start)
                atomic_json(out/'STATUS.json',stat); atomic_json(root/'STATUS.json',stat)
                print(json.dumps(stat),flush=True)
            if step%a.eval_every==0 or step==a.steps:
                reason=eval_update(); checkpoint(finished=bool(reason),reason=reason)
                if reason: break
        if rank==0:
            atomic_json(out/'FINAL.json',dict(arm=a.arm,state='completed',reason=reason or 'budget_complete',
                new_steps=step,best=best,evaluation=result,smoke=a.smoke))
        barrier()
        return 0
    except InterruptedError:
        stop=halt(); checkpoint(finished=stop==2,reason='manual_stop' if stop==2 else 'time_limit')
        return 0 if stop==2 else 75
    finally:
        # No frames persist; worker shutdown releases compressed videos and RAM buffers.
        if hasattr(stream,'_shutdown_workers'): stream._shutdown_workers()

if __name__=='__main__':
    try: sys.exit(main())
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
