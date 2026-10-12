"""Two-stage from-scratch latent diffusion. Falsification: held-out generation fails
or shuffled conditions work equally well. Only code/metrics/model states persist.
"""
import argparse, hashlib, json, os, random, signal, sys, time
from datetime import timedelta
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import DDIMScheduler
from model import new_vae, Denoiser, teacher, features, schedule, VIT_SHA
from dane import VideoFrames, load_manifest, iter_video, FrameCountMismatch

STOP = False
def stop(*args):
    global STOP
    STOP = True
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
if hasattr(signal,'SIGUSR1'): signal.signal(signal.SIGUSR1, stop)

def json_write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp,path)

def save(path, obj):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

def ssim(x,y):
    # Local-window SSIM in image range [0,1].
    mx=F.avg_pool2d(x,11,1,5); my=F.avg_pool2d(y,11,1,5)
    vx=(F.avg_pool2d(x*x,11,1,5)-mx*mx).clamp_min(0)
    vy=(F.avg_pool2d(y*y,11,1,5)-my*my).clamp_min(0)
    cov=F.avg_pool2d(x*y,11,1,5)-mx*my
    return (((2*mx*my+.0001)*(2*cov+.0009))/((mx*mx+my*my+.0001)*(vx+vy+.0009))).mean()

class VAETrain(nn.Module):
    def __init__(self,vae):
        super().__init__(); self.vae=vae
    def forward(self,x):
        p=self.vae.encode(x).latent_dist
        return self.vae.decode(p.sample()).sample, p.kl().mean()

def val_samples(rows, smoke):
    import csv
    csv.field_size_limit(32 * 1024 * 1024)
    samples=[]; rejected=[]
    for row in [r for r in rows if r['split']=='val']:
        n=row.get('frame_count',row.get('annotation_count',1))
        wanted={max(1,int(n*.33)),max(1,int(n*.67))}
        found=[]
        try:
            for sample in iter_video(row,selected_frames=wanted):
                if sample['frame'] in wanted:
                    found.append(sample)
                if smoke and found: break
            samples.extend(found)
        except FrameCountMismatch as e:
            rejected.append({'film':row['video_id'],'reason':str(e)})
        if smoke and samples: break
    if not samples: raise ValueError('No valid held-out frames')
    return samples,rejected

@torch.no_grad()
def evaluate(vae,net,vit,samples,stage,scale,device,smoke,stop_check=lambda:False):
    vae.eval(); net.eval()
    results=[]
    generation_films=set()
    for sample in samples:
        if stop_check(): raise InterruptedError('Validation interrupted for checkpoint')
        x=sample['image'][None].to(device)
        ref=sample['vit_image'][None].to(device)
        recon=((vae.decode(vae.encode(x*2-1).latent_dist.mode()).sample+1)/2).clamp(0,1)
        mse=(recon-x).square().mean().item()
        record={'film':sample['film'],'frame':sample['frame'],'psnr':-10*np.log10(max(mse,1e-12)),
                'ssim':ssim(recon,x).item()}
        if stage=='sd' and sample['film'] not in generation_films:
            generation_films.add(sample['film'])
            z=features(vit,ref).detach()
            # Same noise and different condition: test actual conditioning effect.
            def generate(condition):
                scheduler=DDIMScheduler.from_config(schedule().config)
                scheduler.set_timesteps(2 if smoke else 50,device=device)
                g=torch.Generator(device=device).manual_seed(10000+sample['frame'])
                h=torch.randn((1,4,48,72),generator=g,device=device)
                for t in scheduler.timesteps:
                    if stop_check(): raise InterruptedError('Generation interrupted for checkpoint')
                    with torch.autocast('cuda',dtype=torch.bfloat16): pred=net(h,t,condition)
                    h=scheduler.step(pred.float(),t,h).prev_sample
                    if not torch.isfinite(h).all(): raise FloatingPointError('Nonfinite generation')
                return ((vae.decode(h/scale).sample+1)/2).clamp(0,1)
            image=generate(z)
            other=next((s for s in samples if s['film']!=sample['film']),None)
            wrong_z=features(vit,other['vit_image'][None].to(device)).detach() if other else None
            wrong=generate(wrong_z) if wrong_z is not None else None
            def score(image):
                actual=features(vit,F.interpolate(image,(320,480),mode='bilinear',align_corners=False))
                return ((actual-z).square().mean()/z.square().mean().clamp_min(1e-6)).item(), F.cosine_similarity(actual.flatten(1),z.flatten(1)).item()
            err,cos=score(image)
            record.update(latent_relative_mse=err,cosine=cos,
                image_std=image.flatten(2).std(dim=-1).mean().item())
            if wrong is not None:
                wrong_err,_=score(wrong)
                record.update(wrong_condition_mse=wrong_err,
                    condition_image_mse=(image-wrong).square().mean().item())
        results.append(record)
    # Each film has equal weight even if valid frame counts differ.
    films=sorted(set(r['film'] for r in results))
    keys=[k for k in results[0] if k not in ('film','frame')]
    means={k:float(np.mean([np.mean([r[k] for r in results if r['film']==film and k in r])
                           for film in films if any(r['film']==film and k in r for r in results)])) for k in keys}
    return {'film_mean':means,'frames':results,'films':len(films)}

def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','vit','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--vae-steps',type=int,default=30000); p.add_argument('--sd-steps',type=int,default=100000)
    p.add_argument('--eval-every',type=int,default=500); p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=2); p.add_argument('--resume',default='auto')
    p.add_argument('--frame-stride',type=int,default=600)
    p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    if min(a.vae_steps,a.sd_steps,a.eval_every,a.batch_size,a.frame_stride)<1 or a.workers<0:
        p.error('Steps, eval interval and batch size must be positive; workers nonnegative')
    rank=int(os.getenv('RANK','0')); world=int(os.getenv('WORLD_SIZE','1')); local=int(os.getenv('LOCAL_RANK','0'))
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(hours=2))
    def barrier():
        if world>1: dist.barrier()
    def broadcast(obj):
        box=[obj]
        if world>1: dist.broadcast_object_list(box,src=0,device=device)
        return box[0]
    def collective_stop():
        # Manual STOP wins over the time-limit continuation request.
        value=torch.tensor(2 if (out/'STOP').exists() else
                           int(STOP or (out/'STOP_TIME').exists()),device=device)
        if world>1: dist.all_reduce(value,op=dist.ReduceOp.MAX)
        return int(value.item())
    torch.manual_seed(20261006); random.seed(20261006); np.random.seed(20261006)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    rows=load_manifest(a.manifest)
    train={r['video_id'] for r in rows if r['split']=='train'}; val={r['video_id'] for r in rows if r['split']=='val'}
    if not train or not val or train&val: raise ValueError('Require disjoint films')
    signature={'version':2,'teacher_sha256':VIT_SHA,'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
               'initialization':'random_vae_unet_adapter','architecture':'vae64-128-256-256_unet128-256-384-512',
               'vae_steps':a.vae_steps,'sd_steps':a.sd_steps,'batch_size':a.batch_size,'world_size':world,
               'smoke':a.smoke,'eval_every':a.eval_every,'workers':a.workers}
    vae=new_vae().to(device); net=Denoiser().to(device); vit=teacher(a.vit,device)
    state=None
    latest=out/'latest_checkpoint.pt'
    resume=latest if a.resume=='auto' else (None if a.resume.lower() in ('none','false') else Path(a.resume))
    if resume is not None and (resume.exists() or a.resume!='auto'):
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state['signature']!=signature: raise ValueError('Incompatible resume provenance')
        if state.get('finished') or state.get('needs_review'): return 0
        vae.load_state_dict(state['vae']); net.load_state_dict(state['net'])
    torch.manual_seed(20261006+rank); random.seed(20261006+rank)
    stage=state['stage'] if state else 'vae'; step=state['step'] if state else 0
    scale=state['scale'] if state else 1.0
    seen=set(state.get('seen_films',[])) if state else set()
    samples,rejected=None,None
    validation_error=None
    if rank==0:
        try: samples,rejected=val_samples(rows,a.smoke)
        except Exception as e: validation_error=f'{type(e).__name__}: {e}'
    validation_error=broadcast(validation_error)
    if validation_error: raise RuntimeError(validation_error)
    if rank==0:
        json_write(out/'PROVENANCE.json',{**signature,'train_frame_stride':a.frame_stride,
            'resumed_from_step':state['step'] if state else None,
            'previous_train_frame_stride':state.get('train_frame_stride',1) if state else None,
            'train_films':len(train),'val_films':len(val),
            'rejected_validation':rejected,'image_or_latent_disk_cache':False,'parameters_vae':sum(p.numel() for p in vae.parameters()),
            'parameters_diffusion':sum(p.numel() for p in net.parameters())})
    barrier()
    data=VideoFrames(a.manifest,rank=rank,world_size=world,repeat=True,frame_stride=a.frame_stride)
    loader=torch.utils.data.DataLoader(data,batch_size=a.batch_size,num_workers=a.workers,pin_memory=True,
        **({'prefetch_factor':1,'multiprocessing_context':'spawn'} if a.workers else {}))
    stream=iter(loader)
    def batch():
        value,error=None,None
        try: value=next(stream)
        except Exception as e: error=f'rank {rank}: {type(e).__name__}: {e}'
        failed=torch.tensor(int(error is not None),device=device)
        if world>1: dist.all_reduce(failed,op=dist.ReduceOp.MAX)
        if failed.item(): raise RuntimeError(error or 'Another rank failed to read training data')
        return value
    result=state.get('evaluation') if state else None
    def coverage():
        boxes=[None]*world
        if world>1: dist.all_gather_object(boxes,sorted(seen))
        else: boxes=[sorted(seen)]
        seen.update(film for films in boxes for film in films)
        return len(seen & train)/len(train)
    while True:
        vae.requires_grad_(stage=='vae'); vae.train(stage=='vae'); net.train(stage=='sd')
        raw=VAETrain(vae) if stage=='vae' else net
        model=DDP(raw,device_ids=[local]) if world>1 else raw
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4 if stage=='vae' else 5e-5,weight_decay=.01)
        best=state.get('best',float('inf')) if state else float('inf')
        stale=state.get('stale',0) if state else 0
        if state and state['stage']==stage:
            opt.load_state_dict(state['optimizer'])
            if 'rng' in state:
                torch.set_rng_state(state['rng'][rank]['cpu']); torch.cuda.set_rng_state(state['rng'][rank]['cuda'])
                random.setstate(state['rng'][rank]['python']); np.random.set_state(state['rng'][rank]['numpy'])
        limit=a.vae_steps if stage=='vae' else a.sd_steps
        scheduler=schedule()
        def checkpoint(finished=False,needs_review=False):
            fraction=coverage()
            rng={'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),
                 'python':random.getstate(),'numpy':np.random.get_state()}
            all_rng=[None]*world
            if world>1: dist.all_gather_object(all_rng,rng)
            else: all_rng=[rng]
            if rank==0:
                obj={'signature':signature,'train_frame_stride':a.frame_stride,'stage':stage,'step':step,'scale':scale,'best':best,'stale':stale,
                     'vae':vae.state_dict(),'net':net.state_dict(),'optimizer':opt.state_dict(),'rng':all_rng,
                     'finished':finished,'needs_review':needs_review,'evaluation':result,
                     'seen_films':sorted(seen),'train_film_coverage':fraction,'phase_done':phase_done}
                save(latest,obj)
                json_write(out/'LATEST.json',{'stage':stage,'step':step,'best':best if np.isfinite(best) else None,'finished':finished,
                    'state':'needs_review' if needs_review else ('finished' if finished else 'running'),
                    'train_film_coverage':fraction})
            barrier()
        phase_done=state.get('phase_done',False) if state else False
        if stage=='sd' and step==0: checkpoint()
        halt=collective_stop()
        if halt:
            checkpoint(finished=halt==2); return 0 if halt==2 else 75
        while step<limit and not phase_done:
            b=batch(); seen.update(b['film']); x=b['image'].to(device,non_blocking=True)*2-1
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                if stage=='vae':
                    recon,kl=model(x)
                    loss=F.l1_loss(recon.float(),x)+.25*(1-ssim((recon.float()+1)/2,(x+1)/2))+1e-6*kl.float()/ (48*72*4)
                else:
                    with torch.no_grad():
                        z=features(vit,b['vit_image'].to(device)).detach()
                        h=vae.encode(x).latent_dist.mode()*scale
                    t=torch.randint(0,1000,(len(x),),device=device)
                    noise=torch.randn_like(h); noisy=scheduler.add_noise(h,noise,t)
                    pred=model(noisy,t,z)
                    loss=F.mse_loss(pred.float(),noise.float())
                    if step%4==0:
                        eligible=t<250
                        if eligible.any():
                            alpha=scheduler.alphas_cumprod.to(device)[t[eligible]][:,None,None,None]
                            estimate=(noisy[eligible]-torch.sqrt(1-alpha)*pred[eligible])/torch.sqrt(alpha)
                            image=((vae.decode(estimate/scale).sample+1)/2).clamp(0,1)
                            actual=features(vit,F.interpolate(image.float(),(320,480),mode='bilinear',align_corners=False))
                            target=z[eligible].float()
                            loss=loss+.05*(actual.float()-target).square().mean()/target.square().mean().clamp_min(1e-6)
            finite=torch.tensor(int(torch.isfinite(loss)),device=device)
            if world>1: dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            if not finite.item(): raise FloatingPointError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True); opt.step(); step+=1
            if rank==0 and (step%10==0 or a.smoke):
                json_write(out/'STATUS.json',{'stage':stage,'step':step,'loss':loss.item(),'time':time.time(),'state':'running'})
            halt=collective_stop()
            if halt:
                checkpoint(finished=halt==2); return 0 if halt==2 else 75
            if step%a.eval_every==0 or step==limit:
                fraction=coverage()
                barrier()
                result=None
                evaluation_error=None
                if rank==0:
                    try: result=evaluate(vae,net,vit,samples,stage,scale,device,a.smoke,
                        lambda:STOP or (out/'STOP').exists() or (out/'STOP_TIME').exists())
                    except Exception as e: evaluation_error=f'{type(e).__name__}: {e}'
                evaluation_error=broadcast(evaluation_error)
                if evaluation_error and evaluation_error.startswith('InterruptedError:'):
                    halt=collective_stop()
                    checkpoint(finished=halt==2); return 0 if halt==2 else 75
                if evaluation_error: raise RuntimeError(evaluation_error)
                if rank==0:
                    result.update(train_film_coverage=fraction,train_films_seen=len(seen & train))
                    mean=result['film_mean']; score=1-mean['ssim'] if stage=='vae' else mean['latent_relative_mse']
                    improved=score<best*.995
                    stale=0 if improved else stale+1
                    if improved:
                        best=score; save(out/f'best_{stage}.pt',{'vae':vae.state_dict(),'net':net.state_dict(),'scale':scale,'step':step,'signature':signature,'evaluation':result})
                    quality=(mean['psnr']>=28 and mean['ssim']>=.85) if stage=='vae' else (
                        mean['latent_relative_mse']<=.10 and mean['cosine']>=.90 and mean['image_std']>.06
                        and mean.get('wrong_condition_mse',0)>mean['latent_relative_mse']*1.10)
                    plateau=step>=(5000 if stage=='vae' else 10000) and stale>=(20 if stage=='vae' else 12)
                    result.update(stage=stage,step=step,score=score,best=best,stale=stale,quality=quality,plateau=plateau)
                    json_write(out/'EVAL.json',result)
                    with (out/'EVAL_HISTORY.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(result)+'\n')
                result=broadcast(result); best=result['best']; stale=result['stale']
                representative=a.smoke or fraction>=.90
                phase_done=((result['quality'] or result['plateau']) and representative) or step==limit
                if phase_done and stage=='sd' and rank==0: json_write(out/'FINAL.json',result)
                checkpoint(finished=phase_done and stage=='sd')
                vae.train(stage=='vae'); net.train(stage=='sd')
                if phase_done: break
        if stage=='sd':
            if rank==0: json_write(out/'FINAL.json',result)
            barrier(); return 0
        # Switch using the best reconstruction model, not the final weights.
        best_vae=torch.load(out/'best_vae.pt',map_location='cpu',weights_only=False)
        if best_vae['signature']!=signature: raise ValueError('Incompatible best VAE provenance')
        mean=best_vae['evaluation']['film_mean']
        adequate=a.smoke or (mean['psnr']>=24 and mean['ssim']>=.75 and len(seen & train)/len(train)>=.90)
        if not adequate:
            if rank==0:
                json_write(out/'FINAL.json',{'stage':'vae','step':step,'state':'needs_review',
                    'reason':'VAE below PSNR 24 / SSIM 0.75 or training film coverage below 90%; SD not started',
                    'evaluation':best_vae['evaluation']})
            checkpoint(needs_review=True); return 0
        del model,raw,opt
        vae.load_state_dict(best_vae['vae']); vae.eval().requires_grad_(False)
        # Calibration uses TRAIN only and is fixed thereafter; no latent is saved.
        sums=torch.zeros(3,device=device,dtype=torch.float64)
        with torch.no_grad():
            for _ in range(2 if a.smoke else 16):
                h=vae.encode(batch()['image'].to(device)*2-1).latent_dist.mode().double()
                sums+=torch.stack([h.sum(),h.square().sum(),h.new_tensor(h.numel())])
        if world>1: dist.all_reduce(sums)
        variance=sums[1]/sums[2]-(sums[0]/sums[2]).square()
        if not torch.isfinite(variance) or variance<1e-6:
            raise FloatingPointError('Collapsed/nonfinite VAE latent variance; SD not started')
        scale=float(variance.rsqrt().item())
        vae_coverage=len(seen & train)/len(train)
        stage='sd'; step=0; state=None; seen=set()
        if rank==0: json_write(out/'VAE_COMPLETED.json',{'latent_scale':scale,
            'evaluation':best_vae['evaluation'],'train_film_coverage':vae_coverage,
            'smoke_quality_bypass':a.smoke})

if __name__=='__main__':
    try: sys.exit(main())
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
