"""Actual PyKoopman NNDMD. Sampler/normalization adapters only; package loss unchanged."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
from pathlib import Path
import gc, json, time, importlib.metadata as metadata
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader,WeightedRandomSampler
from pykoopman_fit import write_json,sha256,now,metrics_one,majority_sign
from report_pykoopman import sign_all
SEED=20261002
HORIZONS=[1,2,5,10,20]


def prepare_trajectories(latent,coverage,tool,indices,last):
    """Prefix only; trajectory=image+token. Coverage/equal-image weights."""
    trajectories=[]; weights=[]; records=[]
    first=np.zeros(latent.shape[-1],dtype=np.float64)
    second=first.copy()
    for image in indices:
        ids=np.flatnonzero(tool[image]&(coverage[image]>0))
        cw=coverage[image,ids].astype(np.float64); cw/=cw.sum()
        for token,weight in zip(ids,cw):
            trajectory=np.array(latent[:last+1,image,token],dtype=np.float32,copy=True)
            assert np.isfinite(trajectory).all()
            wt=float(weight/len(indices))
            trajectories.append(trajectory); weights.append(wt)
            first+=wt*trajectory.astype(np.float64).mean(axis=0)
            second+=wt*(trajectory.astype(np.float64)**2).mean(axis=0)
            records.append({"image_index":int(image),"token_index":int(token),"trajectory_weight":wt,
                            "states":last+1,"max_source_update":last*10})
    std=np.sqrt(np.maximum(second-first*first,0))
    scale=float(max(std.max(),1e-3))
    return trajectories,np.array(weights,dtype=np.float64),scale,{
        "records":records,"images":list(indices),"count_trajectories":len(trajectories),
        "states_per_trajectory":last+1,"fit_max_update":last*10,
        "common_scale":scale,"weighted_channel_mean_for_std":first.tolist(),
        "weighted_channel_std":std.tolist(),"normalization_mean_used":0,
        "weight_sum":float(sum(weights))}


def window_weights(trajectories,weights,look_forward):
    counts=[max(len(t)-look_forward,1) for t in trajectories]
    result=np.concatenate([np.full(n,w/n,dtype=np.float64) for n,w in zip(counts,weights)])
    assert abs(result.sum()-1)<1e-10
    return result


def adapter_class(original,weights,scale,seed):
    """Sampling/scaling adapter around package data module, unchanged loss."""
    from pykoopman.regression._nndmd import TensorNormalize,InverseTensorNormalize
    class WeightedDataModule(original):
        def prepare_data(self):
            super().prepare_data()
            self.mean=torch.zeros_like(self.mean)
            self.std=torch.full_like(self.std,float(scale))
            self.normalization=TensorNormalize(self.mean,self.std)
            self.inverse_transform=InverseTensorNormalize(self.mean,self.std)
        def train_dataloader(self):
            probabilities=window_weights(self.data_tr,weights,self.look_forward)
            assert len(probabilities)==len(self.tr_dataset)
            generator=torch.Generator().manual_seed(seed)
            sampler=WeightedRandomSampler(torch.from_numpy(probabilities),len(probabilities),True,generator=generator)
            return DataLoader(self.tr_dataset,batch_size=self.batch_size,sampler=sampler,
                              collate_fn=self.collate_fn,num_workers=0)
    return WeightedDataModule


def learned_rollout(module,start,scale,n_steps):
    """Encode origin once, propagate learned128, decode without measurements."""
    module.eval()
    with torch.no_grad():
        x=torch.as_tensor(np.asarray(start,dtype=np.float32),device=module.device)
        phi=module._encoder(x/scale)
        a=getattr(module,"_fixed_inference_operator",None)
        if a is None: a=module._koopman_propagator.get_discrete_time_Koopman_Operator()
        preds=[]
        for h in range(n_steps):
            phi=phi@a.T
            preds.append((module._decoder(phi)*scale).detach().cpu().numpy())
    result=np.stack(preds)
    assert np.isfinite(result).all()
    return result


class VectorizedMaskedMSE(torch.nn.Module):
    """Exactly the package masked mean, without per-sample CUDA synchronization."""
    def __init__(self,max_look_forward):
        super().__init__(); self.max_look_forward=int(max_look_forward)
    def forward(self,output,target,target_lens):
        if target.shape[1]<self.max_look_forward: output=output[:,:target.shape[1]]
        lengths=target_lens.to(output.device).clamp(max=self.max_look_forward)
        mask=(torch.arange(output.shape[1],device=output.device)[None,:]<lengths[:,None])[:,:,None].expand_as(output)
        squared=torch.pow(output-target,2)
        return torch.where(mask,squared,torch.zeros_like(squared)).sum()/mask.sum()


def install_loss_and_rollout_adapter(module,channels,beta):
    """Actual package networks/optimizer; explicit full+identity loss extension."""
    from types import MethodType
    module.register_buffer("_identity_channels",torch.as_tensor(channels,dtype=torch.long))
    module.identity_beta=float(beta)
    module.masked_loss_metric=VectorizedMaskedMSE(module.look_forward)
    def propagate(self,encoded,n):
        a=self._koopman_propagator.get_discrete_time_Koopman_Operator()
        seq=[]
        for k in range(int(n)):
            encoded=encoded@a.T; seq.append(encoded)
        return torch.stack(seq,1)
    def training_step(self,batch,batch_idx):
        x,y,ys=batch; n=int(ys.max())
        encoded=self._encoder(x)
        phi=self._propagate_encoded_n_steps(encoded,n)
        predicted=torch.stack([self._decoder(phi[:,k]) for k in range(n)],1)
        reconstructed_x=self._decoder(encoded)
        reconstructed_y=torch.stack([self._decoder(self._encoder(y[:,k])) for k in range(n)],1)
        rnn_all=self.masked_loss_metric(predicted,y,ys)
        rec_all=torch.nn.functional.mse_loss(reconstructed_x,x)+self.masked_loss_metric(reconstructed_y,y,ys)
        ids=self._identity_channels
        rnn_j=self.masked_loss_metric(predicted.index_select(-1,ids),y.index_select(-1,ids),ys)
        rec_j=torch.nn.functional.mse_loss(reconstructed_x.index_select(-1,ids),x.index_select(-1,ids))
        rec_j=rec_j+self.masked_loss_metric(reconstructed_y.index_select(-1,ids),y.index_select(-1,ids),ys)
        rnn=rnn_all+self.identity_beta*rnn_j; rec=rec_all+self.identity_beta*rec_j
        loss=rnn+rec
        for name,value in {"loss":loss,"rnn_loss":rnn,"rec_loss":rec,
                           "rnn_full":rnn_all,"rec_full":rec_all,"rnn_J128":rnn_j,"rec_J128":rec_j}.items():
            self.log(name,value,prog_bar=False,on_step=True,on_epoch=False)
        return loss
    module._propagate_encoded_n_steps=MethodType(propagate,module)
    module.training_step=MethodType(training_step,module)


def validate_module(module,latent,coverage,tool,indices,origin,scale,horizons=HORIZONS,channels=None):
    errors={h:0. for h in horizons}; rec_mse=0.; future_rec={h:0. for h in horizons}
    jerrors={h:0. for h in horizons}; jrec=0.
    d=latent.shape[-1]
    for image in indices:
        ids=np.flatnonzero(tool[image]&(coverage[image]>0))
        w=coverage[image,ids].astype(np.float64); w/=w.sum()
        start=np.asarray(latent[origin,image,ids],dtype=np.float32)
        pred=learned_rollout(module,start,scale,max(horizons))
        with torch.no_grad():
            xx=torch.as_tensor(start,device=module.device)/scale
            reconstructed=(module._decoder(module._encoder(xx))*scale).cpu().numpy()
            rec_mse+=float(np.sum(w[:,None]*(reconstructed.astype(np.float64)-start)**2)/d)/len(indices)
            if channels is not None:
                jrec+=float(np.sum(w[:,None]*(reconstructed[:,channels].astype(np.float64)-start[:,channels])**2)/len(channels))/len(indices)
        for h in horizons:
            truth=np.asarray(latent[origin+h,image,ids],dtype=np.float64)
            errors[h]+=float(np.sum(w[:,None]*(pred[h-1].astype(np.float64)-truth)**2)/d)/len(indices)
            if channels is not None:
                jerrors[h]+=float(np.sum(w[:,None]*(pred[h-1][:,channels].astype(np.float64)-truth[:,channels])**2)/len(channels))/len(indices)
            with torch.no_grad():
                target=torch.as_tensor(truth.astype(np.float32),device=module.device)/scale
                reconstructed=(module._decoder(module._encoder(target))*scale).cpu().numpy()
                future_rec[h]+=float(np.sum(w[:,None]*(reconstructed.astype(np.float64)-truth)**2)/d)/len(indices)
    result={"rmse_by_horizon_updates":{str(h*10):float(np.sqrt(errors[h])) for h in horizons},
            "mse_by_horizon_updates":{str(h*10):errors[h] for h in horizons},
            "mean_multihorizon_rmse":float(np.mean([np.sqrt(errors[h]) for h in horizons])),
            "reconstruction_rmse_origin":float(np.sqrt(rec_mse)),
            "reconstruction_rmse_future_horizons":{str(h*10):float(np.sqrt(future_rec[h])) for h in horizons}}
    if channels is not None:
        result["identity_J128"]={"rmse_by_horizon_updates":{str(h*10):float(np.sqrt(jerrors[h])) for h in horizons},
                                  "mean_multihorizon_rmse":float(np.mean([np.sqrt(jerrors[h]) for h in horizons])),
                                  "reconstruction_rmse_origin":float(np.sqrt(jrec))}
    return result


class PrefixValidation(Callback):
    def __init__(self,root,label,latent,coverage,tool,indices,origin,scale,epochs,channels):
        self.root=root; self.label=label; self.latent=latent; self.coverage=coverage; self.tool=tool
        self.indices=indices; self.origin=origin; self.scale=scale; self.epochs=epochs
        self.channels=channels
        self.history=[]; self.best_score=float("inf"); self.best_state=None; self.best_epoch=None; self.best_metric=None
    def on_train_epoch_end(self,trainer,pl_module):
        epoch=trainer.current_epoch+1
        if epoch%5 and epoch!=self.epochs: return
        was_training=pl_module.training
        values=validate_module(pl_module,self.latent,self.coverage,self.tool,self.indices,self.origin,self.scale,channels=self.channels)
        logs={f"package_{k}_last_batch":float(v.detach().cpu()) for k,v in trainer.callback_metrics.items()
              if k in ["loss","rec_loss","rnn_loss","rnn_full","rec_full","rnn_J128","rec_J128"]}
        record={"epoch":epoch,**logs,**values}
        self.history.append(record)
        score=values["identity_J128"]["mean_multihorizon_rmse"]
        if score<self.best_score:
            self.best_score=score; self.best_epoch=epoch; self.best_metric=record
            self.best_state={k:v.detach().cpu().clone() for k,v in pl_module.state_dict().items()}
            torch.save({"state_dict":self.best_state,"epoch":epoch,"metrics":values},self.root/f"NNDMD_{self.label}_BEST_VALIDATION.pt")
        write_json(self.root/f"NNDMD_{self.label}_HISTORIA.json",self.history)
        write_json(self.root/"NNDMD_POSTEP.json",{"phase":"internal_selection","candidate":self.label,
                   "epoch":epoch,"max_epochs":self.epochs,"metrics":values,"best_epoch":self.best_epoch,
                   "best_rmse":self.best_score,"updated":now()})
        print(f"{self.label} epoch{epoch}: valJ128multiRMSE={score:.7f}, fullRMSE={values['mean_multihorizon_rmse']:.7f}, best={self.best_score:.7f}",flush=True)
        if was_training: pl_module.train()


class RefitLog(Callback):
    def __init__(self,root,epochs): self.root=root; self.epochs=epochs; self.history=[]
    def on_train_epoch_end(self,trainer,pl_module):
        epoch=trainer.current_epoch+1
        logs={k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items() if k in ["loss","rec_loss","rnn_loss","rnn_full","rec_full","rnn_J128","rec_J128"]}
        self.history.append({"epoch":epoch,"package_last_batch":logs})
        write_json(self.root/"NNDMD_REFIT_HISTORIA.json",self.history)
        if epoch%5==0 or epoch==self.epochs:
            write_json(self.root/"NNDMD_POSTEP.json",{"phase":"refit30_prefix500","epoch":epoch,
                       "max_epochs":self.epochs,"losses_last_batch":logs,"updated":now()})
            print(f"Refit epoch{epoch}/{self.epochs}: {logs}",flush=True)


def fit_package(trajectories,weights,scale,look_forward,epochs,root,label,callback=None,
                input_dim=768,latent_dim=768,accelerator="gpu",channels=None,beta=1.):
    import pykoopman as pk
    import pykoopman.regression._nndmd as package_module
    L.seed_everything(SEED,workers=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    config_encoder={"input_size":input_dim,"hidden_sizes":[768] if input_dim==768 else [8],
                    "output_size":latent_dim,"activations":"swish"}
    config_decoder={"input_size":latent_dim,"hidden_sizes":[],"output_size":input_dim,"activations":"linear"}
    logger=CSVLogger(save_dir=str(root/"nndmd_training_logs"),name=label,version=0)
    reg=pk.regression.NNDMD(dt=1,look_forward=look_forward,config_encoder=config_encoder,
                            config_decoder=config_decoder,batch_size=32,lbfgs=False,
                            normalize=True,normalize_mode="max",normalize_std_factor=1.,std_koopman=1e-3,
                            trainer_kwargs={"max_epochs":epochs,"accelerator":accelerator,"devices":1,
                                            "precision":"32-true","deterministic":True,"gradient_clip_val":1.,
                                            "logger":logger,"callbacks":[callback] if callback else [],
                                            "enable_checkpointing":False,"enable_progress_bar":False,
                                            "log_every_n_steps":1,"num_sanity_val_steps":0})
    if channels is not None: install_loss_and_rollout_adapter(reg._regressor,channels,beta)
    original=package_module.SeqDataModule
    package_module.SeqDataModule=adapter_class(original,weights,scale,SEED)
    try: reg.fit(trajectories)
    finally: package_module.SeqDataModule=original
    assert reg.A.shape==(latent_dim,latent_dim)
    assert np.isfinite(reg.A).all()
    return reg,{"input_dim":input_dim,"latent_dim":latent_dim,"output_dim":input_dim,
                "encoder":config_encoder,"decoder":config_decoder,"look_forward":look_forward,
                "epochs":epochs,"seed":SEED,"scale":scale,"accelerator":accelerator,"identity_beta":beta,
                "channels0":channels.tolist() if channels is not None else None}


def smoke(root):
    rng=np.random.default_rng(1001)
    trajectories=[]; a=np.array([[.99,.02,0],[0,.98,.01],[0,0,.97]],dtype=np.float32)
    for j in range(5):
        seq=[rng.normal(size=3).astype(np.float32)]
        for t in range(8): seq.append(seq[-1]@a.T)
        trajectories.append(np.stack(seq))
    reg,config=fit_package(trajectories,np.ones(5)/5,1.,2,2,root,"SMOKE",input_dim=3,latent_dim=3,accelerator="cpu")
    start=np.stack([t[0] for t in trajectories])
    rollout=learned_rollout(reg._regressor,start,1.,5)
    differences=[]
    for h in [1,2,5]: differences.append(float(np.max(np.abs(reg.predict(start,n=h)-rollout[h-1]))))
    assert max(differences)<1e-5
    write_json(root/"NNDMD_SMOKE.json",{"actual_package_fit_completed":True,"device":"cpu","known_system_A":a.tolist(),
                "learned_A_shape":list(reg.A.shape),"finite":bool(np.isfinite(reg.A).all()),
                "package_predict_vs_encode_once_rollout_maxabs":max(differences),
                "learned_A_vs_propagator_maxabs":float(np.max(np.abs(reg.A-reg._regressor._koopman_propagator.get_discrete_time_Koopman_Operator().detach().numpy()))),
                "epochs":2,"note":"API/orientation smoke; convergence to known operator is not asserted"})
    print("Actual NNDMD CPU smoke complete; package predict matches encoded rollout",flush=True)


def save_nn(root,label,reg,config,state=None):
    module=reg._regressor
    if state is not None: module.load_state_dict(state)
    module.cpu().eval()
    with torch.no_grad():
        a=module._koopman_propagator.get_discrete_time_Koopman_Operator().detach().cpu().numpy()
        generator=module._koopman_propagator.get_K().detach().cpu().numpy()
    assert a.shape==(config["latent_dim"],config["latent_dim"])
    checkpoint={"state_dict":{k:v.detach().cpu() for k,v in module.state_dict().items()},"config":config,
                "A_learned":a,"generator":generator,"architecture":"actual pykoopman.regression.NNDMD",
                "loss_adapter":"full768 maskedMSE+beta*decodedJ128 maskedMSE; analogous origin/future reconstruction"}
    torch.save(checkpoint,root/f"NNDMD_{label}.pt")
    np.save(root/f"NNDMD_{label}_A_LEARNED_768.npy",a)
    np.save(root/f"NNDMD_{label}_GENERATOR_768.npy",generator)
    write_json(root/f"NNDMD_{label}_CONFIG.json",config)


def train(root):
    started=time.perf_counter()
    assert torch.cuda.is_available(),"CUDA required for full768 actual NNDMD"
    assert not (root/"WYNIKI_NNDMD.json").exists(),"Refuse retuning after final evaluation"
    dane=json.loads((root/"DANE_768.json").read_text(encoding="utf-8"))
    assert dane["status"]=="complete" and dane["all_finite"]
    latent=np.load(root/"LATENT_768.npy",mmap_mode="r")
    assert latent.shape==(101,34,600,768) and latent.dtype==np.float32
    channels=np.load(root/"KANALY_J128.npz")["channels0"]
    masks=np.load(root/"MASKI.npz"); cov=masks["coverage"]; tool=masks["tool"]
    fitids=list(range(8))+list(range(12,34)); trids=fitids[:24]; valid=fitids[24:]
    protocol_hash=sha256(root/"PLAN_NNDMD.md")
    write_json(root/"NNDMD_RUNTIME.json",{"created":now(),"versions":{k:metadata.version(k) for k in ["pykoopman","pydmd","torch","torchvision","numpy","scipy","lightning"]},
                "cuda_available":True,"cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),
                "gpu_total_bytes":torch.cuda.get_device_properties(0).total_memory,
                "protocol_sha256":protocol_hash,"script_sha256":sha256(__file__),"data_sha256":dane["latent_sha256"],
                "source_nndmd_sha256":sha256(Path(metadata.distribution("pykoopman").locate_file("pykoopman/regression/_nndmd.py")))})
    trajectories,w,scale,meta=prepare_trajectories(latent,cov,tool,trids,30)
    write_json(root/"NNDMD_INTERNAL_TRAJEKTORIE.json",meta)
    print(f"ACTUAL NNDMD GPU TRAIN START: native768/learned768/decoder768; {len(trajectories)} trajectories31states",flush=True)
    candidates=[]
    for lf,beta in [(5,1.),(10,1.),(20,1.),(10,0.)]:
        label=f"LF{lf}_B{int(beta)}"
        callback=PrefixValidation(root,label,latent,cov,tool,valid,30,scale,20,channels)
        reg,config=fit_package(trajectories,w,scale,lf,20,root,label,callback,channels=channels,beta=beta)
        assert callback.best_state is not None
        save_nn(root,label,reg,{**config,"selected_epoch":callback.best_epoch},callback.best_state)
        candidate={"label":label,"look_forward":lf,"beta":beta,"best_epoch":callback.best_epoch,
                   "validation_J128_mean_multihorizon_rmse":callback.best_score,
                   "best_validation_metrics":callback.best_metric,
                   "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated()}
        candidates.append(candidate)
        write_json(root/"NNDMD_KANDYDACI.json",candidates)
        del reg,callback; gc.collect(); torch.cuda.empty_cache()
    chosen=min(candidates,key=lambda c:(c["validation_J128_mean_multihorizon_rmse"],c["look_forward"],c["best_epoch"],c["beta"]))
    selection={"selected_at":now(),"selected":chosen,"candidates":candidates,
               "train_indices":trids,"validation_indices":valid,"train_max_update":300,
               "validation_origin_update":300,"validation_end_update":500,
               "criterion":"mean raw decodedJ128 RMSE at H10,20,50,100,200, coverage/equal-image weights",
               "holdout_images_not_used":[8,9,10,11],"protocol_sha256":protocol_hash}
    write_json(root/"WYBOR_NNDMD.json",selection)
    del trajectories; gc.collect()
    trajectories,w,scale,meta=prepare_trajectories(latent,cov,tool,fitids,50)
    write_json(root/"NNDMD_REFIT_TRAJEKTORIE.json",meta)
    callback=RefitLog(root,chosen["best_epoch"])
    print(f"NNDMD REFIT30 prefix500, selected{chosen['label']}, epochs{chosen['best_epoch']}",flush=True)
    reg,config=fit_package(trajectories,w,scale,chosen["look_forward"],chosen["best_epoch"],root,"REFIT30",callback,
                           channels=channels,beta=chosen["beta"])
    save_nn(root,"PREFIX500",reg,config)
    # Reproducibility/orientation check against the actual fitted package.
    probe=trajectories[0][:2]
    predicted=learned_rollout(reg._regressor,probe,scale,5)
    maxerr=max(float(np.max(np.abs(reg.predict(probe,n=h)-predicted[h-1]))) for h in [1,2,5])
    assert maxerr<1e-4
    write_json(root/"ZAMROZENIE_NNDMD.json",{"frozen_at":now(),"model_file":"NNDMD_PREFIX500.pt",
                "model_sha256":sha256(root/"NNDMD_PREFIX500.pt"),"protocol_sha256":protocol_hash,
                "script_sha256":sha256(__file__),"selected":chosen,"refit_config":config,
                "source_data_sha256":dane["latent_sha256"],"fit_indices":fitids,"fit_max_update":500,
                "holdout_indices":[8,9,10,11],"forecast_origin_update":500,"forecast_end_update":1000,
                "no_holdout_images_used_for_selection":True,"no_future_states_used_for_refit":True,
                "input_dim":768,"learned_dim":768,"output_dim":768,"learned_A_shape":[768,768],
                "package_predict_vs_encode_once_rollout_maxabs":maxerr,
                "seconds_train_selection_refit":time.perf_counter()-started,
                "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated()})
    write_json(root/"NNDMD_POSTEP.json",{"phase":"frozen","completed":now(),"model_file":"NNDMD_PREFIX500.pt"})
    print("NNDMD full768 frozen; ready for separate final evaluation",flush=True)


def aggregate_rows(rows):
    result=[]
    for h in range(1,51):
        part=[r for r in rows if r["horizon_updates"]==h*10]
        means={k:float(np.mean([r[k] for r in part if r[k] is not None]))
               if any(r[k] is not None for r in part) else None
               for k in part[0] if k not in ["image_index","horizon_updates"]}
        means["mse_ratio_vs_persistence"]=means["mse_state"]/means["mse_persistence"]
        means["mse_skill_vs_persistence"]=1-means["mse_ratio_vs_persistence"]
        means["rmse_state"]=float(np.sqrt(means["mse_state"]))
        means["rmse_motion"]=float(np.sqrt(means["mse_motion"]))
        means["rmse_persistence"]=float(np.sqrt(means["mse_persistence"]))
        means["rmse_ratio_vs_persistence"]=float(np.sqrt(means["mse_ratio_vs_persistence"]))
        result.append({"horizon_updates":h*10,"images":len(part),**means})
    return result


def evaluate(root):
    assert not (root/"WYNIKI_NNDMD.json").exists(),"Final exploratory evaluation already exists"
    frozen=json.loads((root/"ZAMROZENIE_NNDMD.json").read_text(encoding="utf-8"))
    assert sha256(root/frozen["model_file"])==frozen["model_sha256"]
    assert sha256(root/"PLAN_NNDMD.md")==frozen["protocol_sha256"]
    assert sha256(__file__)==frozen["script_sha256"],"Script changed after freezing"
    checkpoint=torch.load(root/frozen["model_file"],map_location="cpu")
    config=checkpoint["config"]
    import pykoopman as pk
    reg=pk.regression.NNDMD(dt=1,look_forward=config["look_forward"],config_encoder=config["encoder"],
                            config_decoder=config["decoder"],normalize=True,normalize_mode="max")
    channels=np.asarray(config["channels0"])
    install_loss_and_rollout_adapter(reg._regressor,channels,config["identity_beta"])
    reg._regressor.load_state_dict(checkpoint["state_dict"])
    module=reg._regressor.cuda().eval()
    # Use the exact saved discrete operator; no repeated exponential at inference.
    module._fixed_inference_operator=torch.as_tensor(checkpoint["A_learned"],device=module.device)
    assert module._fixed_inference_operator.shape==(768,768)
    latent=np.load(root/"LATENT_768.npy",mmap_mode="r")
    masks=np.load(root/"MASKI.npz"); coverage=masks["coverage"]; tool=masks["tool"]
    fitids=list(range(8))+list(range(12,34)); holdids=[8,9,10,11]
    majority,_=majority_sign(latent,coverage,tool,fitids,0,50)
    mean=np.zeros(768,dtype=np.float64)
    for image in fitids:
        ids=np.flatnonzero(tool[image]&(coverage[image]>0))
        w=coverage[image,ids].astype(np.float64); w/=w.sum()
        mean+=np.sum(w[:,None]*np.asarray(latent[50,image,ids],dtype=np.float64),axis=0)/len(fitids)
    np.save(root/"NNDMD_REVERSION_STATE_768.npy",mean)
    rows={"full768":[],"identity_J128":[],"reversion_full768":[],"reversion_J128":[]}
    prediction_file=root/"NNDMD_PROGNOZA_HOLDOUT_FULL768.npy"
    predictions=np.lib.format.open_memmap(prediction_file,mode="w+",dtype=np.float32,shape=(50,4,600,768))
    reconstruction=[]
    print("NNDMD FROZEN exploratory final test:4images500..1000; decodedFULL768 thenJ128",flush=True)
    for position,image in enumerate(holdids):
        origin=np.asarray(latent[50,image],dtype=np.float32)
        pred=learned_rollout(module,origin,config["scale"],50)
        predictions[:,position]=pred
        ids=np.flatnonzero(tool[image]&(coverage[image]>0))
        w=coverage[image,ids].astype(np.float64); w/=w.sum()
        with torch.no_grad():
            start=torch.as_tensor(origin[ids],device=module.device)/config["scale"]
            rec=(module._decoder(module._encoder(start))*config["scale"]).cpu().numpy()
        recerr=rec.astype(np.float64)-origin[ids]
        reconstruction.append({"image_index":image,"mse_full768":float(np.sum(w[:,None]*recerr**2)/768),
                               "mse_J128":float(np.sum(w[:,None]*recerr[:,channels]**2)/128)})
        for h in range(1,51):
            truth=np.asarray(latent[50+h,image,ids],dtype=np.float64)
            start=origin[ids].astype(np.float64); predicted=pred[h-1,ids].astype(np.float64)
            reverted=np.broadcast_to(mean,predicted.shape)
            for key,truth_v,start_v,pred_v,maj in [
                ("full768",truth,start,predicted,majority),
                ("identity_J128",truth[:,channels],start[:,channels],predicted[:,channels],majority[channels]),
                ("reversion_full768",truth,start,reverted,majority),
                ("reversion_J128",truth[:,channels],start[:,channels],reverted[:,channels],majority[channels])]:
                values=metrics_one(start_v,truth_v,pred_v,w,maj)
                signs=sign_all(truth_v-start_v,pred_v-start_v,w,maj)
                if "full768" in key:
                    signs={k.replace("all128","all768"):v for k,v in signs.items()}
                rows[key].append({"image_index":image,"horizon_updates":h*10,**values,**signs})
        print(f"NNDMD final image{image} completed",flush=True)
        predictions.flush()
    del predictions
    complete={key:{"per_image_horizon":value,"aggregate":aggregate_rows(value)} for key,value in rows.items()}
    static=json.loads((root/"SIGN_ALL_COMPARABILITY.json").read_text(encoding="utf-8"))["holdout"]
    complete["static_EDMD_J128"]=static
    complete["reconstruction_origin"]={"per_image":reconstruction,
            "rmse_full768":float(np.sqrt(np.mean([r["mse_full768"] for r in reconstruction]))),
            "rmse_J128":float(np.sqrt(np.mean([r["mse_J128"] for r in reconstruction])))}
    write_json(root/"NNDMD_METRYKI_HOLDOUT.json",complete)
    identity=complete["identity_J128"]["aggregate"]; full=complete["full768"]["aggregate"]
    contiguous=0
    for row in identity:
        if row["sign_accuracy_all128"]>=.85: contiguous=row["horizon_updates"]
        else: break
    summary={"status":"complete","completed":now(),"method":"actual PyKoopman.NNDMD with explicit identity loss+sampler adapter",
             "input_dim":768,"learned_dim":768,"decoder_output_dim":768,"A_shape":[768,768],
             "seed":SEED,"fit_images":30,"holdout_images":4,"fit_max_update":500,
             "forecast_origin_update":500,"forecast_end_update":1000,"forecast_step_updates":10,
             "exploratory_not_pristine":True,"selected":frozen["selected"],
             "identity_J128_H10":identity[0],"identity_J128_H100":identity[9],"identity_J128_H500":identity[-1],
             "full768_H10":full[0],"full768_H500":full[-1],
             "static_EDMD_J128_H500":static["aggregate"][-1],
             "reversion_J128_H500":complete["reversion_J128"]["aggregate"][-1],
             "contiguous_sign85_horizon_updates":contiguous,
             "any_horizons_sign85":[r["horizon_updates"] for r in identity if r["sign_accuracy_all128"]>=.85],
             "model_sha256":frozen["model_sha256"],"protocol_sha256":frozen["protocol_sha256"],
             "normalization_scale":config["scale"],"reconstruction_origin":complete["reconstruction_origin"],
             "prediction_shape":[50,4,600,768],"prediction_sha256":sha256(prediction_file),
             "no_future_measurements_for_forecast":True,"learned_axes_not_native_axes":True,
             "limitations":["20epochs/candidate, single seed; global neural optimum not established",
                            "four old holdout images already evaluated in prior methods; exploratory comparison",
                            "J128 selected using500; inner validation300..500 retrospective in channel-selection basis",
                            "all34 images belong to ViT train; dynamics holdout only",
                            "one ViT training trajectory; no independent replicate",
                            "API1.2.1 decoder linear, learned nonlinear encoder and full768 generator",
                            "stochastic importance sampling weights exact in expectation, not each minibatch",
                            "package rec/rnn logs are last minibatch values, callback RMSE is exact weighted validation"]}
    write_json(root/"WYNIKI_NNDMD.json",summary)
    import csv
    with (root/"NNDMD_HORYZONTY_J128.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(identity[0])); writer.writeheader(); writer.writerows(identity)
    write_json(root/"NNDMD_POSTEP.json",{"phase":"complete","completed":now(),
                 "H500_J128_RMSE":identity[-1]["rmse_state"],"H500_J128_sign":identity[-1]["sign_accuracy_all128"]})
    print(json.dumps({"H500_J128_RMSE":identity[-1]["rmse_state"],"persistence_RMSE":identity[-1]["rmse_persistence"],
                      "H500_J128_sign":identity[-1]["sign_accuracy_all128"],"sign85_horizon":contiguous},indent=2),flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["smoke","train","evaluate"],required=True)
    args=parser.parse_args()
    root=Path(__file__).parent
    if args.stage=="smoke": smoke(root)
    elif args.stage=="train": train(root)
    else: evaluate(root)
