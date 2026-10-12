"""PyKoopman NNDMD with explicit nonlinear/image-batch streaming fit extension.

The package neural networks and exp(G) propagator are used. Its native fit cannot
materialize this streaming dataset or postprocess a nonlinear decoder as C.
This adapter deliberately provides no linear C/modes for the nonlinear decoder.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'4')
import numpy as np
import torch
import pykoopman as pk


class ImageBatchNNDMD(pk.regression.NNDMD):
    """Actual package NNDMD neural core; image-aware streaming training adapter."""
    def __init__(self,dim=768,hidden=None,seed=20261002,device='cuda'):
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        hidden=[dim] if hidden is None else hidden
        cfg={'input_size':dim,'hidden_sizes':hidden,'output_size':dim,'activations':'swish'}
        super().__init__(dt=1,look_forward=10,config_encoder=cfg,config_decoder=cfg,batch_size=32,
                         normalize=False,lbfgs=False,std_koopman=1e-3,
                         trainer_kwargs={'enable_checkpointing':False,'enable_progress_bar':False})
        self._regressor=self._regressor.to(device)
        self.input_dim=dim;self.config_network=cfg;self.seed=seed
        self.scale=1.;self.optimizer=self._regressor.configure_optimizers()
        self._optimizer_steps=0
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    def fit(self,*args,**kwargs):
        raise RuntimeError('Use explicit streaming fit protocol, not native materializing fit/C postprocess')
    @property
    def module(self):return self._regressor
    @property
    def device(self):return next(self.module.parameters()).device
    def operator(self):return self.module._koopman_propagator.get_discrete_time_Koopman_Operator()
    def encode(self,x):return self.module._encoder(x)
    def decode(self,z):return self.module._decoder(z)


def direct_loss(model,window,rec_weight=1.):
    """Reference gradient on whole logical batch: [H+1,B,P,D], equal weights."""
    steps,b,p,d=window.shape;h=steps-1
    x=window[0].reshape(-1,d)/model.scale
    targets=window[1:].reshape(h,-1,d)/model.scale
    phi=model.encode(x);a=model.operator();rnn=x.new_zeros(());rec=x.new_zeros(())
    rec=rec+torch.mean((model.decode(phi)-x)**2)
    for k in range(h):
        phi=phi@a.T
        rnn=rnn+torch.mean((model.decode(phi)-targets[k])**2)/h
        rec=rec+torch.mean((model.decode(model.encode(targets[k]))-targets[k])**2)/h
    return rnn+rec_weight*rec


def microbatch_gradients(model,window,micro_tokens=512,rec_weight=1.,stability_weight=0.):
    """Exactly one B-image gradient, no optimizer.step or clipping inside chunks.

    A is a differentiable leaf for chunk accumulation. Its total gradient is
    passed through exp(G) once at the end, mathematically the same chain rule.
    Each token chunk uses its actual size/(B*P), including the last chunk.
    """
    steps,b,p,d=window.shape;h=steps-1;n=b*p
    assert h==10 or d!=768,'Production windows must have10 transitions'
    flat=window.reshape(steps,n,d)
    model.module.train();model.optimizer.zero_grad(set_to_none=True)
    a_graph=model.operator();a_leaf=a_graph.detach().requires_grad_(True)
    raw_prediction_sse=np.zeros(h,dtype=np.float64)
    raw_reconstruction_origin_sse=0.;raw_reconstruction_future_sse=0.;total_loss=0.
    for first in range(0,n,micro_tokens):
        last=min(first+micro_tokens,n)
        if isinstance(flat,np.ndarray):
            batch=torch.tensor(flat[:,first:last],device=model.device,dtype=torch.float32)
        else:batch=flat[:,first:last].to(model.device)
        x=batch[0]/model.scale;targets=batch[1:]/model.scale
        phi=model.encode(x);rec_origin=torch.sum((model.decode(phi)-x)**2)
        rnn=x.new_zeros(());rec_future=x.new_zeros(());sse=[]
        for k in range(h):
            phi=phi@a_leaf.T
            er=torch.sum((model.decode(phi)-targets[k])**2)
            rnn=rnn+er/(h*n*d);sse.append(er.detach())
            rec_future=rec_future+torch.sum((model.decode(model.encode(targets[k]))-targets[k])**2)
        rec=rec_origin/(n*d)+rec_future/(h*n*d)
        loss=rnn+rec_weight*rec
        loss.backward()
        raw_prediction_sse+=torch.stack(sse).detach().cpu().numpy().astype(np.float64)*model.scale**2
        raw_reconstruction_origin_sse+=float(rec_origin.detach().cpu())*model.scale**2
        raw_reconstruction_future_sse+=float(rec_future.detach().cpu())*model.scale**2
        total_loss+=float(loss.detach().cpu())
        del batch,x,targets,phi,loss,rec,rec_origin,rec_future,rnn,sse
    assert a_leaf.grad is not None
    penalty=stability_weight*torch.sum((a_graph-torch.eye(d,device=model.device))**2)/d
    # Dot product is only a gradient carrier, not a loss reported to the user.
    surrogate=torch.sum(a_graph*a_leaf.grad.detach())+penalty
    surrogate.backward()
    total_loss+=float(penalty.detach().cpu())
    return {'normalized_loss':total_loss,'prediction_sse_raw_by_horizon':raw_prediction_sse.tolist(),
            'prediction_mse_raw_by_horizon':(raw_prediction_sse/(n*d)).tolist(),
            'reconstruction_mse_raw_origin':raw_reconstruction_origin_sse/(n*d),
            'reconstruction_mse_raw_future':raw_reconstruction_future_sse/(h*n*d),
            'images':b,'particles_per_image':p,'horizons':h,'micro_tokens':micro_tokens}


def optimizer_step(model,window,micro_tokens=512,rec_weight=1.,stability_weight=0.,clip_norm=1.):
    values=microbatch_gradients(model,window,micro_tokens,rec_weight,stability_weight)
    grads=[p.grad for p in model.module.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads),'Nonfinite gradient'
    norm=torch.nn.utils.clip_grad_norm_(model.module.parameters(),clip_norm)
    model.optimizer.step();model._optimizer_steps+=1
    values['gradient_norm_before_clip']=float(norm.detach().cpu())
    values['optimizer_step']=model._optimizer_steps
    return values


def rollout(model,start,n=10,micro_tokens=512):
    """Full native state encoded once, then A in learnedstate, nonlinear decoder."""
    model.module.eval();shape=start.shape;d=shape[-1];flat=np.asarray(start,dtype=np.float32).reshape(-1,d)
    out=np.empty((n,len(flat),d),dtype=np.float32)
    with torch.no_grad():
        a=model.operator()
        for first in range(0,len(flat),micro_tokens):
            last=min(first+micro_tokens,len(flat));x=torch.tensor(flat[first:last],device=model.device)/model.scale
            phi=model.encode(x)
            for h in range(n):
                phi=phi@a.T;out[h,first:last]=(model.decode(phi)*model.scale).cpu().numpy()
    assert np.isfinite(out).all()
    return out.reshape((n,)+shape)
