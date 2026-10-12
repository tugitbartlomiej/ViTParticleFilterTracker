"""Prawdziwe API bibliotek; jawne pary, skale train i fizyczny krok 1."""
from dataclasses import dataclass
import hashlib
import numpy as np

@dataclass
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls,x):
        x=np.asarray(x,dtype=np.float64)
        s=x.std(0); s=np.where(s>1e-30,s,1.)
        return cls(x.mean(0),s)

    def transform(self,x): return (x-self.mean)/self.scale
    def inverse(self,x): return x*self.scale+self.mean

def film_split(films):
    unique=sorted(set(films),key=lambda f:hashlib.sha256(('20261009:'+f).encode()).hexdigest())
    if len(unique)<5: raise ValueError('Need at least five distinct films')
    mapping={f:('test' if i%5==0 else 'validation' if i%5==1 else 'train') for i,f in enumerate(unique)}
    return [mapping[f] for f in films]

def window_starts(phase,horizon):
    phase=np.asarray(phase);n=len(phase)
    return np.asarray([t for t in range(n-horizon+1) if np.all(phase[t:t+horizon]==phase[t])],dtype=int)

def adam_input(z,g,m,v,update_number,lr):
    t=np.asarray(update_number)[...,None];eta=np.asarray(lr)[...,None]
    mh=(.9*m+.1*g)/(1-.9**t)
    vh=(.999*v+.001*g*g)/(1-.999**t)
    return np.maximum(z-eta*mh/(np.sqrt(vh)+1e-8),-.16997121)-z

def validate_arrays(a):
    z=a['control_z'];T=z.shape[0]-1
    if z.ndim!=3 or z.shape[-1]!=768: raise ValueError('Expected full 768 dimensions')
    for k in ['control_gradient_before_step','control_adam_m1_before_step','control_adam_m2_before_step']:
        if k not in a:raise ValueError('Missing '+k)
        if a[k].shape!=(T,z.shape[1],768):raise ValueError('Misaligned '+k)
    for k in ['telemetry_step','telemetry_phase_min_norm','telemetry_learning_rate','telemetry_top_token']:
        if k not in a or a[k].shape!=(T+1,):raise ValueError('Missing/misaligned '+k)
    if not np.array_equal(a['telemetry_step'],np.arange(T+1)):raise ValueError('Non-unit steps or padding')
    if len(a['control_tokens'])!=z.shape[1] or len(set(a['control_tokens']))!=z.shape[1]:raise ValueError('Token identities')
    for k in ['control_z','control_gradient_before_step','control_adam_m1_before_step','control_adam_m2_before_step','telemetry_learning_rate']:
        if not np.isfinite(a[k]).all():raise ValueError('Nonfinite '+k)
    if (a['control_adam_m2_before_step']<0).any():raise ValueError('Negative Adam variance')

def fit_pydmd_pairs(x,y,u,known_B=None):
    from pydmd import DMDc
    x,y,u=map(lambda a:np.asarray(a,dtype=float),(x,y,u))
    if len(x)!=len(y) or len(x)!=len(u):raise ValueError('Unpaired input')
    omega=np.concatenate([x,u],axis=1)
    singular=np.linalg.svd(omega,compute_uv=False)
    effective=int((singular>singular[0]*1e-8).sum())
    # Public lag adapter yields exactly x=X[:,:-N], y=X[:,N:]. No bridge pairs.
    model=DMDc(svd_rank=x.shape[1],svd_rank_omega=effective,lag=len(x))
    model.fit(np.concatenate([x.T,y.T],axis=1),u.T,B=known_B)
    basis=model.basis
    A=np.real_if_close(basis@model.operator.as_numpy_array@basis.T.conj()).real
    B=np.real_if_close(model.B).real
    if not np.isfinite(A).all() or not np.isfinite(B).all():raise ValueError('Nonfinite DMDc operator')
    return {'A':A,'B':B,'backend':'pydmd.DMDc.fit','pairs':len(x),'physical_lag':1,'api_packing_lag':len(x),
      'effective_rank_omega':effective,'full_rank_omega':omega.shape[1],
      'condition_retained':float(singular[0]/singular[effective-1]),'singular_values':singular,
      'eigenvalues':np.linalg.eigvals(A),'known_B':known_B is not None}

def fit_deeptime_pairs(x,y):
    from deeptime.decomposition import DMD
    # Bias is an observable, propagated as 1. Paired fit, never concatenated timeline.
    xa=np.column_stack([x,np.ones(len(x))]);ya=np.column_stack([y,np.ones(len(y))])
    model=DMD(mode='exact',rank=xa.shape[1]).fit((xa,ya)).fetch_model()
    full=np.real_if_close(model.transform(np.eye(xa.shape[1]))).real.T
    return {'A':full[:-1,:-1],'b':full[:-1,-1],'backend':'deeptime.decomposition.DMD.fit'}

def error_metrics(pred,true,start):
    error=np.asarray(pred,dtype=float)-true
    movement=true-start;delta=pred-start
    norm=np.linalg.norm(movement,axis=-1);pn=np.linalg.norm(delta,axis=-1)
    active=(norm>1e-4)&(pn>1e-4)
    cos=np.sum(movement*delta,axis=-1)/np.maximum(norm*pn,1e-30)
    return {'n':len(error),'rmse':float(np.sqrt(np.mean(error**2))),
      'increment_rmse':float(np.sqrt(np.mean((delta-movement)**2))),
      'true_net_norm_mean':float(norm.mean()),'pred_net_norm_mean':float(pn.mean()),
      'direction_cos_mean_active':float(cos[active].mean()) if active.any() else None,
      'direction_active_n':int(active.sum()),'max_abs_error':float(np.abs(error).max())}
