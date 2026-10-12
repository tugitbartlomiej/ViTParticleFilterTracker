"""Causal expanded inputs and separate native/auxiliary normalization."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


class AuxHistory:
    def __init__(self,values,valid,targets,target_valid,age):
        self.values=np.asarray(values,dtype=np.float32);self.valid=np.asarray(valid,dtype=bool)
        self.targets=np.asarray(targets,dtype=np.float32);self.target_valid=np.asarray(target_valid,dtype=bool)
        self.age=np.asarray(age,dtype=np.float32)
        assert self.values.shape==self.valid.shape==self.targets.shape==self.target_valid.shape
        assert self.values.shape[0]==101 and self.age.shape==(101,)
        self.q=self.values.shape[1]
    @classmethod
    def load(cls,path):
        data=np.load(path)
        return cls(data['input_values'],data['input_valid'],data['target_values'],data['target_valid'],data['moment_age_steps'])
    def input(self,t,allowed):
        if t>allowed:raise ValueError('Future true auxiliary input forbidden')
        if t<0:return np.zeros(self.q,np.float32),np.zeros(self.q,bool),0.
        assert np.isfinite(self.values[t,self.valid[t]]).all()
        return self.values[t],self.valid[t],float(self.age[t])
    def target(self,t,allowed):
        if t>allowed:raise ValueError('Future labels outside fitting/evaluation boundary')
        assert np.isfinite(self.targets[t,self.target_valid[t]]).all()
        return self.targets[t],self.target_valid[t]


def _statistics(values,valid):
    counts=valid.sum(axis=0)
    safe=np.where(valid,values,0).astype(np.float64)
    mean=safe.sum(axis=0)/np.maximum(counts,1)
    variance=np.where(valid,(values-mean)**2,0).sum(axis=0)/np.maximum(counts,1)
    floor=np.maximum(np.sqrt((safe*safe).sum(axis=0)/np.maximum(counts,1))*1e-3,1e-10)
    return mean.astype(np.float32),np.maximum(np.sqrt(variance),floor).astype(np.float32)


@dataclass
class Normalizer:
    native_scale:float
    input_mean:np.ndarray
    input_std:np.ndarray
    target_mean:np.ndarray
    target_std:np.ndarray
    fit_last:int
    @classmethod
    def fit(cls,native_scale,aux,last):
        assert 0<last<=90 and native_scale>0
        assert np.isfinite(aux.values[:last+1][aux.valid[:last+1]]).all()
        assert np.isfinite(aux.targets[:last+1][aux.target_valid[:last+1]]).all()
        a,b=_statistics(aux.values[:last+1],aux.valid[:last+1])
        c,d=_statistics(aux.targets[:last+1],aux.target_valid[:last+1])
        return cls(float(native_scale),a,b,c,d,int(last))
    def save(self,path):
        np.savez(path,native_scale=self.native_scale,input_mean=self.input_mean,input_std=self.input_std,
                 target_mean=self.target_mean,target_std=self.target_std,fit_last=self.fit_last)
    @classmethod
    def load(cls,path):
        a=np.load(path)
        return cls(float(a['native_scale']),a['input_mean'],a['input_std'],a['target_mean'],a['target_std'],int(a['fit_last']))


def expanded_input(native_triplet,t,aux,norm,allowed):
    """[z_t,dz_t,dz_(t-1),aux_t,aux_(t-1),masks,history_flags,ages]."""
    assert native_triplet.shape[0]==3
    device=native_triplet.device;n,d=native_triplet.shape[1:]
    if t>allowed:raise ValueError('Future native input forbidden')
    prevprev,prev,current=native_triplet
    delta=current-prev if t>0 else torch.zeros_like(current)
    older=prev-prevprev if t>1 else torch.zeros_like(current)
    blocks=[current/norm.native_scale,delta/norm.native_scale,older/norm.native_scale]
    ages=[];valids=[]
    for cp in [t,t-1]:
        values,valid,age=aux.input(cp,allowed)
        standardized=np.where(valid,(values-norm.input_mean)/norm.input_std,0).astype(np.float32)
        blocks.append(torch.tensor(standardized,device=device).expand(n,-1))
        valids.append(torch.tensor(valid.astype(np.float32),device=device).expand(n,-1));ages.append(age/100.)
    blocks.extend(valids)
    flags=torch.tensor([float(t>0),float(t>1),*ages],device=device).expand(n,-1)
    blocks.append(flags)
    return torch.cat(blocks,dim=-1)


def target_auxiliary(t,aux,norm,allowed,device):
    values,valid=aux.target(t,allowed)
    normalized=np.where(valid,(values-norm.target_mean)/norm.target_std,0).astype(np.float32)
    return torch.tensor(normalized,device=device),torch.tensor(valid,device=device)


def forecast_history(origin_native,prediction):
    """Mechanically shift decoded latent differences; no source future history."""
    assert origin_native.shape[0]==3
    chain=np.concatenate([origin_native[-2:],prediction],axis=0)
    return np.diff(chain,axis=0)[1:]
