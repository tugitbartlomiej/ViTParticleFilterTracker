"""EDEN streaming driver: full image batches, chronological selection/refit/test."""
from pathlib import Path
import argparse,hashlib,json,os,time,traceback,shutil
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from model import ExpandedNNDMD,rollout,optimizer_step,distributed_optimizer_step,collective_error_gate
from state import AuxHistory,Normalizer,expanded_input

HERE=Path(__file__).resolve().parent
def stamp():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def atomic_json(path,row):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(row,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def append(path,row):
    with open(path,'a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def training_record(stats,phase,epoch,origin,batch_index,ids,total_steps,seconds,extract_seconds):
    assert stats['origin']==origin
    # Merge dictionaries explicitly: stats already contains origin.
    return {**stats,'phase':phase,'epoch':epoch,'origin':origin,'batch':batch_index,
            'image_indices':ids.tolist(),'updated':stamp(),'total_optimizer_steps':total_steps,
            'optimizer_seconds':seconds,'extract_seconds':extract_seconds}
def image_batches(count,size,seed=None):
    ids=np.arange(count)
    if seed is not None:np.random.default_rng(seed).shuffle(ids)
    for first in range(0,count,size):yield ids[first:first+size]
def training_origins(last,horizon=10):
    assert last in (80,90) and horizon==10
    return range(last-horizon+1)
def select_epoch(rows):
    return min(rows,key=lambda r:(r['full768']['aggregate_rmse_state'],r['epoch']))['epoch']
def early_stop(rows,patience,min_delta):
    best=float('inf');stale=0
    for row in rows:
        score=row['full768']['aggregate_rmse_state']
        if score<best-min_delta:best=score;stale=0
        else:stale+=1
    return stale>=patience
def save_checkpoint(path,model,epoch,norm,config,total_steps,phase=None,origin=None):
    tmp=Path(str(path)+'.tmp')
    torch.save(dict(state_dict=model.module.state_dict(),optimizer=model.optimizer.state_dict(),epoch=epoch,
                    model_config=model.config,optimizer_steps=model.optimizer_steps,total_optimizer_steps=total_steps,
                    normalization_fit_last=norm.fit_last,config=config,phase=phase,last_completed_origin=origin),tmp);os.replace(tmp,path)

def collective_call(function,phase,rank0_only=False):
    """Matching calls on every rank; singlewriter errors propagate collectively."""
    rank=dist.get_rank() if dist.is_initialized() else 0;error=None;result=None
    if not rank0_only or rank==0:
        try:result=function()
        except Exception as exc:error=repr(exc)
    collective_error_gate(error,phase)
    return result

def broadcast_object(value):
    if not dist.is_initialized():return value
    box=[value];dist.broadcast_object_list(box,src=0);return box[0]

def initialize_runtime():
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    assert world in (1,2,4)
    if world>1:
        torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    return rank,world,local

def dataset_preflight(reader,aux,norm,cfg,create,progress,rank,world):
    """Guard actual all600 dataset32/tail28 steps before any fitting updates."""
    parallel=create();reference=create() if rank==0 else None;reports=[]
    options={k:cfg[k] for k in ['micro_tokens','rec_weight','aux_weight','identity_weight','clip_norm']}
    for case,ids in enumerate([np.arange(32),np.arange(1472,1500)]):
        local_ids=np.array_split(ids,world)[rank]
        raw=collective_call(lambda:reader.get_history_batch(local_ids,0),'preflight local actual dataset')
        assert raw.shape==(13,len(local_ids),600,768)
        one=None;full=None;one_seconds=None
        def single_step():
            nonlocal one,full,one_seconds
            full=reader.get_history_batch(ids,0);torch.cuda.synchronize();start=time.monotonic()
            one=optimizer_step(reference,full,0,aux,norm,80,**options)
            torch.cuda.synchronize();one_seconds=time.monotonic()-start
        collective_call(single_step,'preflight reference actual dataset',True)
        torch.cuda.synchronize();start=time.monotonic()
        many=distributed_optimizer_step(parallel,raw,0,aux,norm,80,global_images=len(ids),**options)
        torch.cuda.synchronize();seconds=time.monotonic()-start
        def compare_and_save():
            np.testing.assert_allclose(one['loss_normalized'],many['loss_normalized'],rtol=3e-4,atol=3e-5)
            np.testing.assert_allclose(one['prediction_mse_raw'],many['prediction_mse_raw'],rtol=3e-4,atol=3e-5)
            grad_error=0.;weight_error=0.
            for (name,a),(other,b) in zip(reference.module.named_parameters(),parallel.module.named_parameters()):
                assert name==other
                torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-5);torch.testing.assert_close(a,b,rtol=3e-4,atol=2e-5)
                grad_error=max(grad_error,float((a.grad-b.grad).abs().max().cpu()));weight_error=max(weight_error,float((a-b).abs().max().cpu()))
            row=training_record(many,'actual_dataset_GPU_preflight',1,0,case,ids,case+1,seconds,0.)
            row.update(single_GPU_seconds=one_seconds,distributed_GPU_seconds=seconds,gradient_max_error=grad_error,
                       weights_max_error=weight_error,actual_fit_steps=0,actual_preflight_optimizer_steps=case+1,
                       timing_note='CUDA synchronized early steps; not a warmed epoch ETA')
            append(HERE/'PREFLIGHT_TRAINING.jsonl',row);reports.append(row)
            save_checkpoint(HERE/'PREFLIGHT_MODEL.pt',parallel,1,norm,cfg,case+1,'preflight',0)
            saved=torch.load(HERE/'PREFLIGHT_MODEL.pt',map_location='cpu',weights_only=False)
            assert saved['optimizer_steps']==case+1
            progress(phase='actual_dataset_GPU_preflight',actual_preflight_optimizer_steps=case+1)
        collective_call(compare_and_save,'actual dataset GPU proof/telemetry/checkpoint',True)
    collective_call(lambda:atomic_json(HERE/'DATASET_GPU_PREFLIGHT.json',dict(status='PASS',source='exact1500 actual latent cache',
        world_size=world,global_batches=[32,28],particles_per_image=600,horizon=10,actual_preflight_optimizer_steps=2,
        actual_fit_steps=0,comparisons=reports,note='Both preflight models discarded; fitting starts fresh from fixed seed')),
        'actual dataset preflight result',True)
    del parallel,reference;torch.cuda.empty_cache()

class Metrics:
    def __init__(self,horizon,channels):
        self.horizon=horizon;self.channels=np.asarray(channels,int);self.count=0
        self.sums={name:{key:np.zeros(horizon,np.float64) for key in
            ('model_sse','persistence_sse','continuation_sse','historical_mean_sse','sign_sum','persistence_sign','continuation_sign','historical_mean_sign','cos_sum','normratio_sum','cos_count','norm_count')}
            for name in ('full768','J128')}
        self.rec_sse={name:0. for name in self.sums};self.aux_sum=None;self.aux_sq=None;self.aux_count=0
    def add(self,truth,pred,origin,previous,reconstruction,proxy,historical_mean):
        b,p,d=origin.shape;n=b*p;self.count+=n
        for name,ids in [('full768',slice(None)),('J128',self.channels)]:
            z0=origin[...,ids].astype(np.float64);dz=z0-previous[...,ids].astype(np.float64)
            self.rec_sse[name]+=float(np.sum((reconstruction[...,ids].astype(np.float64)-z0)**2))
            acc=self.sums[name]
            for k in range(self.horizon):
                actual=truth[k,...,ids] if isinstance(ids,slice) else truth[k][...,ids]
                forecast=pred[k,...,ids] if isinstance(ids,slice) else pred[k][...,ids]
                delta=actual.astype(np.float64)-z0;estimate=forecast.astype(np.float64)-z0
                acc['model_sse'][k]+=np.sum((estimate-delta)**2)
                acc['persistence_sse'][k]+=np.sum(delta**2)
                acc['continuation_sse'][k]+=np.sum(((k+1)*dz-delta)**2)
                acc['historical_mean_sse'][k]+=np.sum((forecast*0+historical_mean[...,ids].astype(np.float64)-actual)**2)
                acc['persistence_sign'][k]+=np.sum(np.sign(delta)==0)
                acc['continuation_sign'][k]+=np.sum(np.sign((k+1)*dz)==np.sign(delta))
                acc['historical_mean_sign'][k]+=np.sum(np.sign(historical_mean[...,ids]-z0)==np.sign(delta))
                acc['sign_sum'][k]+=np.sum(np.sign(estimate)==np.sign(delta))
                tn=np.linalg.norm(delta,axis=-1);pn=np.linalg.norm(estimate,axis=-1)
                mask=(tn>0)&(pn>0);acc['cos_count'][k]+=mask.sum()
                acc['cos_sum'][k]+=np.sum(np.sum(delta*estimate,axis=-1)[mask]/(tn[mask]*pn[mask]))
                valid=tn>0;acc['norm_count'][k]+=valid.sum();acc['normratio_sum'][k]+=np.sum(pn[valid]/tn[valid])
        us=proxy.sum(axis=1,dtype=np.float64);uq=np.square(proxy.astype(np.float64)).sum(axis=1)
        self.aux_sum=us if self.aux_sum is None else self.aux_sum+us
        self.aux_sq=uq if self.aux_sq is None else self.aux_sq+uq;self.aux_count+=n
    def finish(self,origin,aux):
        result={'origin_checkpoint':origin,'origin_source_updates':origin*10,'token_count':self.count,
                'weighting':'equal images; all600 tokens; equal horizons and raw native components',
                'rmse_units':'raw source latent coordinates','sign_definition':'all coordinates, sign(0)=0, no epsilon filtering'}
        for name,d in [('full768',768),('J128',len(self.channels))]:
            acc=self.sums[name];den=self.count*d;rows=[]
            for k in range(self.horizon):
                mse=float(acc['model_sse'][k]/den);base=float(acc['persistence_sse'][k]/den)
                cont=float(acc['continuation_sse'][k]/den)
                rows.append(dict(horizon_checkpoints=k+1,horizon_source_updates=(k+1)*10,mse_state=mse,
                  rmse_state=float(np.sqrt(mse)),mse_motion=mse,rmse_motion=float(np.sqrt(mse)),
                  mse_persistence=base,rmse_persistence=float(np.sqrt(base)),rmse_continuation=float(np.sqrt(cont)),
                  rmse_historical_mean=float(np.sqrt(acc['historical_mean_sse'][k]/den)),
                  rmse_ratio=None if base==0 else float(np.sqrt(mse/base)),mse_skill=None if base==0 else 1-mse/base,
                  sign_all_fraction=float(acc['sign_sum'][k]/den),
                  persistence_sign_all_fraction=float(acc['persistence_sign'][k]/den),
                  continuation_sign_all_fraction=float(acc['continuation_sign'][k]/den),
                  historical_mean_sign_all_fraction=float(acc['historical_mean_sign'][k]/den),
                  cosine_motion=None if acc['cos_count'][k]==0 else float(acc['cos_sum'][k]/acc['cos_count'][k]),
                  motion_norm_ratio=None if acc['norm_count'][k]==0 else float(acc['normratio_sum'][k]/acc['norm_count'][k]),
                  cosine_fraction_included=float(acc['cos_count'][k]/self.count),
                  norm_fraction_included=float(acc['norm_count'][k]/self.count)))
            result[name]={'horizons':rows,'aggregate_rmse_state':float(np.sqrt(acc['model_sse'].mean()/den)),
                          'aggregate_rmse_persistence':float(np.sqrt(acc['persistence_sse'].mean()/den)),
                          'reconstruction_origin_rmse':float(np.sqrt(self.rec_sse[name]/den))}
        mean=self.aux_sum/self.aux_count;std=np.sqrt(np.maximum(self.aux_sq/self.aux_count-mean**2,0))
        labels=[]
        for k in range(self.horizon):
            target,valid=aux.target(origin+k+1,origin+self.horizon)
            labels.append(dict(checkpoint=origin+k+1,target_values=target.tolist(),target_valid=valid.tolist(),
                               prediction_mean=mean[k].tolist(),token_disagreement_std=std[k].tolist(),
                               observed_moment_age_if_no_new_measurement=float(aux.age[origin]+10*(k+1))))
        result['auxiliary_proxy']={'interpretation':'token-local estimates of global summary targets; aggregated diagnostic, no rollout feedback; not full AdamW state',
                                   'horizons':labels}
        return result

def reconstruction(model,past,origin,aux,norm,micro):
    shape=past.shape[1:];flat=past.reshape(3,-1,768);out=np.empty(shape,np.float32).reshape(-1,768)
    with torch.no_grad():
        for first in range(0,len(out),micro):
            raw=torch.tensor(flat[:,first:first+micro],device=model.device)
            out[first:first+micro]=(model.decode(model.encode(expanded_input(raw,origin,aux,norm,origin)))[:,:768]*norm.native_scale).cpu().numpy()
    return out.reshape(shape)

def evaluate(model,reader,origin,aux,norm,cfg,progress,prediction_path=None,window_prepared=False):
    if origin==90:
        freeze=json.loads((HERE/'FREEZE.json').read_text(encoding='utf-8'))
        assert freeze['fit_last']==90 and freeze['test_origin']==90 and freeze['test_last']==100
        assert sha(HERE/'FROZEN_MODEL.pt')==freeze['hashes']['FROZEN_MODEL.pt']
    progress(phase='evaluation_extract',origin=origin)
    if not window_prepared:reader.ensure_window(origin,10)
    meter=Metrics(10,cfg['channels_j128']);saved=None
    probes={}
    examples=cfg['prediction_example_indices']
    if prediction_path:
        saved=np.lib.format.open_memmap(prediction_path,mode='w+',dtype='float32',shape=(10,len(examples),600,128))
    for ids in image_batches(1500,cfg['image_batch']):
        raw=reader.get_history_batch(ids,origin)
        # Only the three origin/past states enter rollout; future states are labels below.
        pred,_,_,proxy=rollout(model,raw[:3],origin,aux,norm,10,cfg['micro_tokens'])
        rec=reconstruction(model,raw[:3],origin,aux,norm,cfg['micro_tokens'])
        meter.add(raw[3:],pred,raw[2],raw[1],rec,proxy,raw[:3].mean(axis=0))
        if saved is not None:
            for j,image in enumerate(ids):
                if int(image) in examples:saved[:,examples.index(int(image))]=pred[:,j][...,cfg['channels_j128']]
        if prediction_path:
            for j,image in enumerate(ids):
                if int(image) in cfg['full768_audit_probe_indices']:
                    probes[int(image)]=(pred[:,j].copy(),rec[j].copy())
            for j,image in enumerate(ids):
                per={'image_index':int(image),'origin':origin,'horizons':[]}
                for h in range(10):
                    row={'horizon_source_updates':10*(h+1)}
                    for name,ch in [('full768',slice(None)),('J128',cfg['channels_j128'])]:
                        dz=raw[3+h,j][...,ch].astype(np.float64)-raw[2,j][...,ch].astype(np.float64)
                        dp=pred[h,j][...,ch].astype(np.float64)-raw[2,j][...,ch].astype(np.float64)
                        row[name]={'mse':float(np.mean((dp-dz)**2)),'rmse':float(np.sqrt(np.mean((dp-dz)**2))),
                                   'persistence_rmse':float(np.sqrt(np.mean(dz**2))),'sign_all_fraction':float(np.mean(np.sign(dp)==np.sign(dz)))}
                    per['horizons'].append(row)
                append(HERE/'TEST_PER_IMAGE.jsonl',per)
        progress(phase='evaluation',origin=origin,evaluated_images=int(ids[-1])+1)
    if saved is not None:saved.flush();del saved
    if prediction_path:
        keys=cfg['full768_audit_probe_indices'];assert all(i in probes for i in keys)
        np.savez(HERE/'FORECAST_FULL768_AUDIT_PROBES.npz',image_indices=keys,
                 predictions=np.stack([probes[i][0] for i in keys],axis=1),reconstruction_h0=np.stack([probes[i][1] for i in keys]))
    return meter.finish(origin,aux)

def main(config_path):
    rank,world,local=initialize_runtime()
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    assert cfg['image_count']==1500 and cfg['image_batch']==32 and cfg['horizon']==10
    assert cfg['inner_last']==80 and cfg['final_last']==90 and cfg['test_last']==100
    assert cfg['native_dim']==cfg['learned_dim']==768 and len(cfg['channels_j128'])==128
    assert cfg['max_epochs']==10 and cfg['early_stopping_patience']==2
    from dane import FrozenWindowReader
    reader=collective_call(lambda:FrozenWindowReader(),'reader initialization');aux=AuxHistory.load(HERE/'AUX_HISTORY.npz');assert aux.q==59
    manifest=json.loads((HERE/'MANIFEST_1500.json').read_text(encoding='utf-8-sig'))
    assert manifest['status']=='verified' and manifest['count']==1500 and len(manifest['records'])==1500
    assert all(r['cache_pixels_match_source'] for r in manifest['records'])
    total_steps=0;status={'submitted':True,'job_id':os.environ.get('SLURM_JOB_ID'),'logpath':os.environ.get('EXPANDED_LOGPATH'),
      'code_ready':True,'validated':True,'exact_images':1500,'image_batch':32,'horizon_checkpoints':10,
      'native_dim':768,'learned_dim':768,'expanded_input_dim':2544,'auxiliary_dim':59,'nonlinear_decoder':True,
      'method':'PyKoopman NNDMD neural components with explicit expanded-state streaming/image-aware fit adapter',
      'AdamW_moments_available':'actual coarse snapshots; causal hold-last summaries and age/mask',
      'actual_training_steps':0,'local_GPU_training_cancelled':True,'world_size':world,
      'parallelization':'global image batch; SUM image/token weighted gradients; rank0 penalty; clip after SUM'}
    def progress(**items):
        if rank==0:
            status.update(items,updated=stamp(),actual_training_steps=total_steps);atomic_json(HERE/'status.json',status)
    def shared_progress(**items):return collective_call(lambda:progress(**items),'status write',True)
    try:
        assert torch.cuda.is_available(),'EDEN GPU required'
        shared_progress(phase='normalization_extract',gpu=torch.cuda.get_device_name(local));reader.begin_epoch('calibration',0)
        collective_call(lambda:reader.ensure_window(0,10),'normalization parallel extraction')
        def native_scale():
            total=0.;sq=0.;n=0
            for ids in image_batches(1500,32):
                z=reader.get_state(ids,0).astype(np.float64);total+=z.sum();sq+=np.sum(z*z);n+=z.size
            return max(float(np.sqrt(max(sq/n-(total/n)**2,0))),1e-8)
        scale=broadcast_object(collective_call(native_scale,'normalization rank0',True))
        collective_call(lambda:atomic_json(HERE/'NORMALIZATION_PROVENANCE.json',dict(native_scalar=scale,native_calibration_checkpoint=0,
            native_calibration_images=1500,native_centering=False,aux_fit_inner_max=80,aux_fit_refit_max=90)),'normalization provenance',True)
        source_hashes={name:sha(HERE/name) for name in ['train.py','model.py','state.py','dane.py','CONFIG.json','PLAN_MODELU.md','MANIFEST_1500.json','AUX_HISTORY.npz']}
        def source_snapshot():
            snapshot=HERE/'EXECUTED_SOURCE';snapshot.mkdir(exist_ok=True)
            for name in source_hashes:
                dest=snapshot/name
                if dest.exists():assert sha(dest)==source_hashes[name],'Refusing to replace executed-source snapshot'
                else:shutil.copyfile(HERE/name,dest)
            atomic_json(HERE/'EXECUTION_PROVENANCE.json',dict(hashes=source_hashes,started=stamp(),job_id=status['job_id'],
                torch_version=torch.__version__,cuda=torch.version.cuda,configuration=cfg,world_size=world))
        collective_call(source_snapshot,'executed source snapshot',True)
        def make_model():return ExpandedNNDMD(aux.q,seed=cfg['seed'],hidden=cfg['hidden'],
            lr_network=cfg['lr_network'],lr_generator=cfg['lr_generator'],device='cuda:'+str(local))
        norm=Normalizer.fit(scale,aux,80);collective_call(lambda:norm.save(HERE/'NORMALIZER_INNER.npz'),'inner normalizer save',True)
        if world>1 and not os.environ.get('EXPANDED_RESUME_CHECKPOINT'):
            dataset_preflight(reader,aux,norm,cfg,make_model,progress,rank,world)
        model=make_model()
        options={key:cfg[key] for key in ['micro_tokens','rec_weight','aux_weight','identity_weight','clip_norm']}
        resume_path=os.environ.get('EXPANDED_RESUME_CHECKPOINT');resume=None
        val_rows=[]
        if resume_path:
            resume=torch.load(resume_path,map_location='cpu',weights_only=False)
            assert resume['config']==cfg and resume['phase'] in ('inner_train','final_refit')
            assert resume['normalization_fit_last']==(80 if resume['phase']=='inner_train' else 90)
            total_steps=resume['total_optimizer_steps']
            val_rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(HERE.glob('VALIDATION_EPOCH_*.json'))]
        def restore(model):
            model.module.load_state_dict(resume['state_dict']);model.optimizer.load_state_dict(resume['optimizer']);model.optimizer_steps=resume['optimizer_steps']
        def train_epoch(model,norm,last,phase,epoch):
            nonlocal total_steps
            reader.begin_epoch(phase,epoch);epoch_start=time.monotonic()
            for origin in training_origins(last):
                if resume and resume['phase']==phase and epoch==resume['epoch'] and origin<=resume['last_completed_origin']:continue
                shared_progress(phase=phase+'_extract',epoch=epoch,origin=origin,fit_last=last);start=time.monotonic()
                collective_call(lambda:reader.ensure_window(origin,10),'parallel source extraction')
                extract_seconds=time.monotonic()-start
                for batch_index,ids in enumerate(image_batches(1500,32,cfg['seed']+epoch*10000+origin)):
                    local_ids=np.array_split(ids,world)[rank]
                    raw=collective_call(lambda:reader.get_history_batch(local_ids,origin),'local raw read');start=time.monotonic()
                    if world>1:stats=distributed_optimizer_step(model,raw,origin,aux,norm,last,global_images=len(ids),**options)
                    else:stats=optimizer_step(model,raw,origin,aux,norm,last,**options)
                    seconds=time.monotonic()-start;total_steps+=1
                    def write_step():
                        if total_steps<=3:
                            atomic_json(HERE/'GPU_TRAINING_SMOKE.json',dict(status='actual optimizer step completed',actual_training_steps=total_steps,
                                logical_batch_images=len(ids),particles_per_image=600,horizon=10,micro_tokens=cfg['micro_tokens'],world_size=world,
                                seconds=seconds,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name(local),
                                model_dimensions=[2544,768,827],estimated_nn_only_seconds_per_inner_epoch=seconds*3337,
                                estimated_nn_only_seconds_per_refit_epoch=seconds*3807,
                                caveat='first steps include warmup; source extraction and IO are additional; full benchmark is real training'))
                        record=training_record(stats,phase,epoch,origin,batch_index,ids,total_steps,seconds,extract_seconds)
                        append(HERE/'TRAINING.jsonl',record)
                        progress(phase=phase,epoch=epoch,origin=origin,batch=batch_index,stage_optimizer_steps=model.optimizer_steps,
                            latest_loss=stats['loss_normalized'],latest_rmse_full768=float(np.sqrt(np.mean(stats['prediction_mse_raw']))),
                            latest_optimizer_seconds=seconds)
                    collective_call(write_step,'telemetry after global step',True)
                collective_call(lambda:save_checkpoint(HERE/(phase+'_latest.pt'),model,epoch,norm,cfg,total_steps,phase,origin),'origin checkpoint',True)
            return time.monotonic()-epoch_start
        if resume and resume['phase']=='inner_train':restore(model)
        start_epoch=resume['epoch'] if resume and resume['phase']=='inner_train' else 1
        inner_finished=bool(resume and resume['phase']=='final_refit')
        for epoch in range(start_epoch,cfg['max_epochs']+1):
            if inner_finished:break
            if early_stop(val_rows,cfg['early_stopping_patience'],cfg['early_stopping_min_delta_raw_rmse']):break
            if any(v['epoch']==epoch for v in val_rows):continue
            duration=train_epoch(model,norm,80,'inner_train',epoch)
            collective_call(lambda:reader.ensure_window(80,10),'parallel validation extraction')
            val=broadcast_object(collective_call(lambda:evaluate(model,reader,80,aux,norm,cfg,progress,window_prepared=True),'rank0 validation',True))
            val.update(epoch=epoch,training_epoch_seconds=duration);val_rows.append(val)
            collective_call(lambda:atomic_json(HERE/('VALIDATION_EPOCH_%02d.json'%epoch),val),'validation save',True)
            collective_call(lambda:save_checkpoint(HERE/('INNER_EPOCH_%02d.pt'%epoch),model,epoch,norm,cfg,total_steps),'epoch checkpoint',True)
            if early_stop(val_rows,cfg['early_stopping_patience'],cfg['early_stopping_min_delta_raw_rmse']):break
        chosen=select_epoch(val_rows);collective_call(lambda:atomic_json(HERE/'SELECTION.json',dict(selected_epoch=chosen,
            objective='minimum full768 aggregate raw validation RMSE across h1..10 at origin80; ties earlier epoch',
            candidates=[{'epoch':v['epoch'],'full768_rmse':v['full768']['aggregate_rmse_state'],
                         'persistence_rmse':v['full768']['aggregate_rmse_persistence'],'J128_rmse':v['J128']['aggregate_rmse_state']} for v in val_rows],
            test_opened=False)),'selection save',True)
        del model;torch.cuda.empty_cache();norm=Normalizer.fit(scale,aux,90)
        collective_call(lambda:norm.save(HERE/'NORMALIZER_FINAL.npz'),'final normalizer save',True);model=make_model()
        refit_start=1
        if resume and resume['phase']=='final_refit':restore(model);refit_start=resume['epoch']
        for epoch in range(refit_start,chosen+1):train_epoch(model,norm,90,'final_refit',epoch)
        collective_call(lambda:save_checkpoint(HERE/'FROZEN_MODEL.pt',model,chosen,norm,cfg,total_steps),'frozen checkpoint',True)
        a=model.operator().detach().cpu().numpy();g=model.module._koopman_propagator.get_K().detach().cpu().numpy()
        collective_call(lambda:(np.save(HERE/'KOOPMAN_A_768.npy',a),np.save(HERE/'GENERATOR_G_768.npy',g)),'operator save',True)
        freeze=dict(frozen=stamp(),hashes={**source_hashes,'FROZEN_MODEL.pt':sha(HERE/'FROZEN_MODEL.pt'),
            'NORMALIZER_FINAL.npz':sha(HERE/'NORMALIZER_FINAL.npz'),'KOOPMAN_A_768.npy':sha(HERE/'KOOPMAN_A_768.npy')},
            fit_last=90,test_origin=90,test_last=100,selected_epoch=chosen,total_optimizer_steps=total_steps,
            final_optimizer_steps=model.optimizer_steps,forecast='origin-only encoder, learned A^h, nonlinear decoder; no future signals')
        assert all(sha(HERE/name)==value for name,value in source_hashes.items()),'Execution sources changed before freeze'
        collective_call(lambda:atomic_json(HERE/'FREEZE.json',freeze),'freeze write',True);shared_progress(phase='frozen',selected_epoch=chosen)
        art=Path(os.environ.get('EXPANDED_ARTIFACT_DIR',str(HERE)));art.mkdir(parents=True,exist_ok=True)
        predpath=art/'PREDICTION_TEST_J128.npy'
        collective_call(lambda:reader.ensure_window(90,10),'parallel frozen-test extraction')
        result=collective_call(lambda:evaluate(model,reader,90,aux,norm,cfg,progress,str(predpath),window_prepared=True),'rank0 final test',True)
        if rank!=0:return
        result['freeze_sha256']=sha(HERE/'FREEZE.json')
        result['prediction_j128']={'path':str(predpath),'sha256':sha(predpath),'shape':[10,len(cfg['prediction_example_indices']),600,128],
                                  'image_indices':cfg['prediction_example_indices'],'scope':'12 predeclared examples; metrics use all1500'}
        atomic_json(HERE/'METRICS_TEST.json',result)
        diagnostic=dict(spectral_radius=float(np.max(np.abs(np.linalg.eigvals(a.astype(np.float64))))),
                        operator_norm=float(np.linalg.norm(a.astype(np.float64),2)),
                        power10_norm=float(np.linalg.norm(np.linalg.matrix_power(a.astype(np.float64),10),2)),
                        note='identity proximity penalty does not enforce spectral stability; learned axes differ from native768')
        atomic_json(HERE/'OPERATOR_DIAGNOSTICS.json',diagnostic)
        lines=['# Wynik rozszerzonego NNDMD1500 na Edenie','',
          'Wybrana epoka: %d. Całkowite kroki Adam: %d. Final refit: %d kroków.'%(chosen,total_steps,model.optimizer_steps),
          'Fit obejmuje 90 przejść między checkpointami, czyli 900 aktualizacji źródłowego ViT. Test obejmuje 10 przejść, czyli 100 aktualizacji.',
          'Test dotyczy odłożonych przyszłych celów tego fitu, tych samych1500 obrazów. Historia źródłowa była już eksplorowana wcześniejszymi wariantami, więc to eksploracyjny test, nie nietknięte niezależne potwierdzenie. J128 odczytano po dekodowaniu pełnych768.','',
          '|Kanały|H aktualizacji|RMSE|Persistence RMSE|RMSE ratio|MSE skill|Sign all|','|---|---:|---:|---:|---:|---:|---:|']
        for name in ['full768','J128']:
            for r in result[name]['horizons']:
                lines.append('|%s|%d|%.8g|%.8g|%.5g|%.5g|%.5g|'%(name,r['horizon_source_updates'],r['rmse_state'],
                    r['rmse_persistence'],r['rmse_ratio'] or 0,r['mse_skill'] or 0,r['sign_all_fraction']))
        lines+=['','Encoder użyto jeden raz. Rozszerzona historia i ostatnie rzeczywiste momenty AdamW służą inicjalizacji; przyszłe auxiliary są wyłącznie estymatami decoderu, bez sprzężenia do rolloutu.',
          'Użyto sieci i propagatora PyKoopman1.2.1 z jawnym własnym adapterem fit/loss, zamiast niezmodyfikowanego NNDMD.fit. Decoder jest nieliniowy. Learnedstate768 kompresuje wejście2544 i nie oznacza tożsamości z oryginalnymi osiami768.',
          'Adam minimalizuje lokalnie ważony wielokrokowy MSE plus rekonstrukcję i maskedauxiliary. RMSE tego samego ważenia ma ten sam minimizer; globalne minimum ani cel85procent nie są gwarantowane.',
          'Równe wagi1500 obrazów i wszystkich600 cząsteczek. Adnotacje źródłowe mają bbox, nie pełne maski segmentacji. Rekonstrukcja nie stanowi ścisłej dolnej granicy błędu przyszłości.']
        (HERE/'RAPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');progress(phase='complete',metrics_path=str(HERE/'METRICS_TEST.json'))
    except BaseException:
        progress(phase='failed',error=traceback.format_exc());raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
