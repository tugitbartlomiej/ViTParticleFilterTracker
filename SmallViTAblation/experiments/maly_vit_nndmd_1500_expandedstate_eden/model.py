"""Expanded-state PyKoopman NNDMD neural core, image-aware fit extension.

Nonlinear decoder prevents native package linear-C postprocessing. We retain the
package FFNN/generator/exp(G), and explicitly replace fit/data/loss postprocessing.
No future true auxiliary observation is consumed by learned A^h rollout.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
import numpy as np
import torch
import torch.distributed as dist
import pykoopman as pk
from state import expanded_input,target_auxiliary


class ExpandedNNDMD(pk.regression.NNDMD):
    def __init__(self,q,native_dim=768,learned_dim=768,hidden=768,seed=20261002,device='cuda',
                 lr_network=1e-3,lr_generator=1e-5):
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        self.q=int(q);self.native_dim=int(native_dim);self.learned_dim=int(learned_dim)
        self.hidden=hidden;self.seed=seed;self.lr_network=lr_network;self.lr_generator=lr_generator
        self.input_dim=3*self.native_dim+4*self.q+4;self.output_dim=self.native_dim+self.q
        ce={'input_size':self.input_dim,'hidden_sizes':[hidden],'output_size':self.learned_dim,'activations':'swish'}
        cd={'input_size':self.learned_dim,'hidden_sizes':[hidden],'output_size':self.output_dim,'activations':'swish'}
        super().__init__(dt=1,look_forward=10,config_encoder=ce,config_decoder=cd,batch_size=32,normalize=False,
                         lbfgs=False,std_koopman=1e-3,trainer_kwargs={'logger':False,'enable_checkpointing':False,'enable_progress_bar':False})
        self._regressor=self._regressor.to(device);self.optimizer_steps=0
        self.config={'q':self.q,'native_dim':self.native_dim,'learned_dim':self.learned_dim,'hidden':hidden,'seed':seed,
                     'lr_network':lr_network,'lr_generator':lr_generator,'encoder':ce,'decoder':cd}
        self.optimizer=torch.optim.Adam([
          {'params':list(self._regressor._encoder.parameters())+list(self._regressor._decoder.parameters()),'lr':lr_network},
          {'params':self._regressor._koopman_propagator.parameters(),'lr':lr_generator}],betas=(.9,.999),eps=1e-8)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    @property
    def module(self):return self._regressor
    @property
    def device(self):return next(self.module.parameters()).device
    def operator(self):return self.module._koopman_propagator.get_discrete_time_Koopman_Operator()
    def encode(self,x):return self.module._encoder(x)
    def decode(self,x):return self.module._decoder(x)
    def fit(self,batch_factory,epochs,aux,norm,allowed,last_callback=None,**options):
        """Streaming/image-aware extension, actual package NNDMD neural core."""
        for epoch in range(1,epochs+1):
            for origin,raw,info in batch_factory(epoch):
                stats=optimizer_step(self,raw,origin,aux,norm,allowed=allowed,**options)
                if last_callback:last_callback('batch',epoch,origin,stats,info)
            if last_callback:last_callback('epoch',epoch,None,None,None)
        return self


def _loss_piece(model,raw,origin,aux,norm,allowed,a,rec_weight,aux_weight,n_total):
    """Raw[H+3,M,D], padded causal history; each chunk weighted by M/n_total."""
    steps,m,d=raw.shape;horizon=steps-3
    x=expanded_input(raw[:3],origin,aux,norm,allowed);phi=model.encode(x)
    out=model.decode(phi);z0=raw[2]/norm.native_scale
    native_rec=torch.sum((out[:,:d]-z0)**2)/(n_total*d)
    u0,mask=target_auxiliary(origin,aux,norm,allowed,model.device)
    aux_rec=torch.sum((out[:,d:][:,mask]-u0[mask])**2)/(n_total*max(int(mask.sum()),1))
    prediction=raw.new_zeros(());aux_prediction=raw.new_zeros(())
    native_sse=[];aux_sse=[]
    for k in range(1,horizon+1):
        phi=phi@a.T;out=model.decode(phi);truth=raw[k+2]/norm.native_scale
        err=torch.sum((out[:,:d]-truth)**2);native_sse.append(err.detach())
        prediction=prediction+err/(horizon*n_total*d)
        target,valid=target_auxiliary(origin+k,aux,norm,allowed,model.device)
        ae=torch.sum((out[:,d:][:,valid]-target[valid])**2)
        aux_sse.append(ae.detach());aux_prediction=aux_prediction+ae/(horizon*n_total*max(int(valid.sum()),1))
        true_input=expanded_input(raw[k:k+3],origin+k,aux,norm,allowed)
        reconstruction=model.decode(model.encode(true_input))
        native_rec=native_rec+torch.sum((reconstruction[:,:d]-truth)**2)/(horizon*n_total*d)
        aux_rec=aux_rec+torch.sum((reconstruction[:,d:][:,valid]-target[valid])**2)/(horizon*n_total*max(int(valid.sum()),1))
    loss=prediction+rec_weight*native_rec+aux_weight*(aux_prediction+rec_weight*aux_rec)
    return loss,torch.stack(native_sse),torch.stack(aux_sse),native_rec.detach(),aux_rec.detach()


def direct_loss(model,raw,origin,aux,norm,allowed,rec_weight=1.,aux_weight=.1,identity_weight=1e-3):
    steps,b,p,d=raw.shape;flat=raw.reshape(steps,b*p,d)
    a=model.operator();loss,*_= _loss_piece(model,flat,origin,aux,norm,allowed,a,rec_weight,aux_weight,b*p)
    return loss+identity_weight*torch.sum((a-torch.eye(model.learned_dim,device=model.device))**2)/model.learned_dim


def accumulated_gradients(model,raw,origin,aux,norm,allowed,micro_tokens=1024,rec_weight=1.,aux_weight=.1,identity_weight=1e-3,
                          global_tokens=None,penalty_multiplier=1.):
    steps,b,p,d=raw.shape;local_n=b*p;n=local_n if global_tokens is None else int(global_tokens);horizon=steps-3
    assert n>=local_n and local_n>0
    assert origin+horizon<=allowed,'Window crosses fit boundary'
    flat=raw.reshape(steps,local_n,d);model.module.train();model.optimizer.zero_grad(set_to_none=True)
    a_graph=model.operator();a_leaf=a_graph.detach().requires_grad_(True)
    total=0.;native_sse=np.zeros(horizon);aux_sse=np.zeros(horizon);native_rec=0.;aux_rec=0.
    for first in range(0,local_n,micro_tokens):
        last=min(first+micro_tokens,local_n)
        chunk=torch.as_tensor(flat[:,first:last],device=model.device,dtype=torch.float32)
        loss,ss,us,rr,ur=_loss_piece(model,chunk,origin,aux,norm,allowed,a_leaf,rec_weight,aux_weight,n)
        loss.backward();total+=float(loss.detach().cpu());native_sse+=ss.cpu().numpy().astype(float)*norm.native_scale**2
        aux_sse+=us.cpu().numpy().astype(float);native_rec+=float(rr.cpu())*norm.native_scale**2;aux_rec+=float(ur.cpu())
        del chunk,loss,ss,us,rr,ur
    penalty=penalty_multiplier*identity_weight*torch.sum((a_graph-torch.eye(model.learned_dim,device=model.device))**2)/model.learned_dim
    (torch.sum(a_graph*a_leaf.grad.detach())+penalty).backward();total+=float(penalty.detach().cpu())
    return {'loss_normalized':total,'prediction_mse_raw':(native_sse/(n*d)).tolist(),'aux_prediction_normalized_SSE':aux_sse.tolist(),
            'reconstruction_native_mse_raw':native_rec,'reconstruction_aux_masked_mse_normalized':aux_rec,
            'images':b,'particles':p,'horizon':horizon,'origin':origin,'micro_tokens':micro_tokens}


def collective_error_gate(error,phase):
    """All ranks enter after local work; ordinary errors propagate everywhere."""
    errors=[str(error) if error else None]
    if dist.is_available() and dist.is_initialized():
        errors=[None]*dist.get_world_size();dist.all_gather_object(errors,str(error) if error else None)
    if any(errors):raise RuntimeError(phase+': '+'; '.join('rank%d %s'%(i,e) for i,e in enumerate(errors) if e))


def distributed_optimizer_step(model,raw,origin,aux,norm,allowed,global_images,clip_norm=1.,**options):
    """One global image batch, SUM gradients, one penalty, clip after SUM."""
    world=dist.get_world_size();rank=dist.get_rank();local_images=int(raw.shape[1]);particles=int(raw.shape[2])
    image_count=torch.tensor(local_images,device=model.device,dtype=torch.int64)
    dist.all_reduce(image_count,op=dist.ReduceOp.SUM)
    assert int(image_count.cpu())==global_images,'Actual local images do not sum to global logical batch'
    error=None;stats=None
    try:
        stats=accumulated_gradients(model,raw,origin,aux,norm,allowed,global_tokens=global_images*particles,
                                    penalty_multiplier=float(rank==0),**options)
        if not all(torch.isfinite(p.grad).all() for p in model.module.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite local gradient')
    except Exception as exc:error=repr(exc)
    collective_error_gate(error,'local gradient accumulation')
    # Same exp(G) Jacobian on every rank. SUM of local G-VJPs equals global VJP.
    for parameter in model.module.parameters():
        if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad,op=dist.ReduceOp.SUM)
    error=None if all(torch.isfinite(p.grad).all() for p in model.module.parameters()) else 'Nonfinite aggregated gradient'
    collective_error_gate(error,'aggregated gradient')
    grad=torch.nn.utils.clip_grad_norm_(model.module.parameters(),clip_norm)
    collective_error_gate(None if torch.isfinite(grad) else 'Nonfinite global gradient norm','global clipping')
    model.optimizer.step();model.optimizer_steps+=1
    keys=['loss_normalized','reconstruction_native_mse_raw','reconstruction_aux_masked_mse_normalized']
    scalars=torch.tensor([stats[k] for k in keys],device=model.device,dtype=torch.float64)
    prediction=torch.tensor(stats['prediction_mse_raw'],device=model.device,dtype=torch.float64)
    auxiliary=torch.tensor(stats['aux_prediction_normalized_SSE'],device=model.device,dtype=torch.float64)
    for values in [scalars,prediction,auxiliary]:dist.all_reduce(values,op=dist.ReduceOp.SUM)
    stats.update(dict(zip(keys,scalars.cpu().tolist())),prediction_mse_raw=prediction.cpu().tolist(),
                 aux_prediction_normalized_SSE=auxiliary.cpu().tolist(),images=int(global_images),local_images=local_images,
                 world_size=world,optimizer_step=model.optimizer_steps,gradient_norm_before_clip=float(grad.detach().cpu()))
    return stats


def optimizer_step(model,*args,clip_norm=1.,**kwargs):
    stats=accumulated_gradients(model,*args,**kwargs)
    assert all(torch.isfinite(p.grad).all() for p in model.module.parameters() if p.grad is not None),'Nonfinite gradient'
    grad=torch.nn.utils.clip_grad_norm_(model.module.parameters(),clip_norm)
    model.optimizer.step();model.optimizer_steps+=1
    stats.update(optimizer_step=model.optimizer_steps,gradient_norm_before_clip=float(grad.detach().cpu()))
    return stats


def rollout(model,past_native,origin,aux,norm,n=10,micro_tokens=1024):
    """Encode origin once. Never reads aux/latent after origin."""
    assert past_native.shape[0]==3
    shape=past_native.shape[1:];d=shape[-1];flat=np.asarray(past_native,dtype=np.float32).reshape(3,-1,d)
    token_count=flat.shape[1]
    output=np.empty((n,token_count,d),np.float32);proxy=np.empty((n,token_count,model.q),np.float32)
    model.module.eval()
    with torch.no_grad():
        a=model.operator()
        for first in range(0,token_count,micro_tokens):
            last=min(first+micro_tokens,token_count);raw=torch.tensor(flat[:,first:last],device=model.device)
            phi=model.encode(expanded_input(raw,origin,aux,norm,allowed=origin))
            for h in range(n):
                phi=phi@a.T;decoded=model.decode(phi)
                output[h,first:last]=(decoded[:,:d]*norm.native_scale).cpu().numpy()
                proxy[h,first:last]=decoded[:,d:].cpu().numpy()*norm.target_std+norm.target_mean
    assert np.isfinite(output).all() and np.isfinite(proxy).all()
    # One common estimate and token disagreement; never feedback into A^h.
    global_mean=proxy.mean(axis=1,dtype=np.float64);global_std=proxy.std(axis=1,dtype=np.float64)
    return output.reshape((n,)+shape),global_mean,global_std,proxy
