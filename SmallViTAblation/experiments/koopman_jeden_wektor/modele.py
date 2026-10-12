"""Full-token linear models; failure to beat temporal controls falsifies utility."""
import numpy as np
from koopman_przyrostowy import IncrementKoopman
from koopman_pamiec_full import FullDelayKoopman

CONFIGS = [(name, ridge) for name in ('state','delay','velocity') for ridge in (.001,.1)] + [('scalar_velocity',0.),('diagonal_velocity',.001)]


class TokenModel:
    def fit(self, x, v, y, name, ridge):
        self.name=name;self.ridge=ridge;self.d=x.shape[1]
        if name=='state': self.m=IncrementKoopman().fit(x,y,ridge)
        elif name=='delay': self.m=FullDelayKoopman().fit(x,v,y,ridge)
        elif name=='velocity': self.m=IncrementKoopman().fit(v,y-x,ridge)
        else:
            denom=np.sum(v*v,axis=0);num=np.sum(v*(y-x),axis=0)
            if name=='scalar_velocity':
                self.alpha=np.full(self.d,float(num.sum()/denom.sum()) if denom.sum()>1e-30 else 1.)
            elif name=='diagonal_velocity':
                penalty=ridge*max(float(denom.mean()),1e-30)
                self.alpha=(num+penalty)/(denom+penalty)
            else: raise ValueError(name)
        return self

    def predict(self,z,v,h=1):
        if self.name=='state':return self.m.predict(z,h)
        if self.name=='delay':return self.m.predict(z,v,h)
        zz=z.copy();vv=v.copy()
        for _ in range(h):
            vv=self.m.predict(vv) if self.name=='velocity' else vv*self.alpha
            zz=zz+vv
        return zz

    def explicit(self):
        """Row-vector convention [z,v] @ A + b. No truncation."""
        d=self.d;eye=np.eye(d);zero=np.zeros((d,d))
        if self.name=='state':
            B=self.m.xc.T@self.m.coeff
            return eye+B,self.m.md-self.m.mx@B
        if self.name=='delay':
            P=self.m.u.T@self.m.coeff/self.m.sx
            Q=self.m.v.T@self.m.coeff/self.m.sv
            b=self.m.md-self.m.mx@P-self.m.mv@Q
            return np.block([[eye+P,P],[eye+Q,eye+Q]]),np.r_[b,b]
        if self.name=='velocity':
            B=self.m.xc.T@self.m.coeff;K=eye+B
            b=self.m.md-self.m.mx@B
        else:K=np.diag(self.alpha);b=np.zeros(d)
        return np.block([[eye,zero],[K,K]]),np.r_[b,b]


def window_indices(steps,origin,window=8):
    steps=np.asarray(steps)
    if not np.all(np.diff(steps)==5):raise ValueError('Expected 5-update spacing')
    ids=np.flatnonzero(steps==origin)
    if len(ids)!=1 or ids[0]<window+1:raise ValueError('Insufficient past')
    j=int(ids[0]);triples=[(k-1,k,k+1) for k in range(j-window,j)]
    assert max(c for a,b,c in triples)==j
    return triples


def training(data,steps,origin,indices,token,window=8):
    triples=window_indices(steps,origin,window)
    x=np.concatenate([data[b,indices,token] for a,b,c in triples]).astype(np.float64)
    # Convert BEFORE subtracting to avoid unnecessary float32 cancellation.
    v=np.concatenate([data[b,indices,token].astype(np.float64)-data[a,indices,token].astype(np.float64) for a,b,c in triples])
    y=np.concatenate([data[c,indices,token] for a,b,c in triples]).astype(np.float64)
    return x,v,y


def metrics(pred,truth,z):
    dp=pred-z;dt=truth-z
    err=np.mean((pred-truth)**2,axis=-1)
    dpnorm=np.linalg.norm(dp,axis=-1);dtnorm=np.linalg.norm(dt,axis=-1)
    cos=np.sum(dp*dt,axis=-1)/np.maximum(dpnorm*dtnorm,1e-30)
    return err,np.mean(dt*dt,axis=-1),cos,dpnorm/np.maximum(dtnorm,1e-30)
