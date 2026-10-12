"""Independent affine residual DMD per trajectory; no mixing across batch units.

The identity prior is explicit. Dual factors represent a full 384x384 map,
not a uniquely identified unrestricted map or a compressed state.
"""
import torch

class Operators:
    def fit(self,x,y,relative_ridge):
        if x.dtype!=torch.float64 or y.dtype!=torch.float64:raise ValueError('float64 required')
        if x.ndim!=3 or x.shape!=y.shape or x.shape[1]<2:raise ValueError('BxNxD matched trajectories required')
        self.mx=x.mean(1);delta=y-x;self.md=delta.mean(1)
        self.xc=x-self.mx[:,None,:];self.dc=delta-self.md[:,None,:]
        gram=self.xc@self.xc.transpose(1,2)
        relative=torch.as_tensor(relative_ridge,dtype=x.dtype,device=x.device)
        if relative.ndim==0:relative=relative.expand(len(x))
        if relative.shape!=(len(x),) or torch.any(relative<=0):raise ValueError('Positive independent ridge')
        self.ridge=relative*torch.clamp(gram.diagonal(dim1=1,dim2=2).mean(1),min=torch.finfo(x.dtype).eps)
        system=gram+self.ridge[:,None,None]*torch.eye(x.shape[1],dtype=x.dtype,device=x.device)
        self.coeff=torch.linalg.solve(system,self.dc)
        self.solver_residual=torch.linalg.vector_norm(system@self.coeff-self.dc,dim=(1,2))/torch.clamp(torch.linalg.vector_norm(self.dc,dim=(1,2)),min=1e-30)
        self.gram=gram
        return self

    def predict(self,z,horizon=1):
        if z.ndim!=3 or z.shape[0]!=len(self.mx):raise ValueError('Independent BxQxD input required')
        if horizon<0 or not isinstance(horizon,int):raise ValueError('Integer nonnegative horizon')
        if horizon==0:return z.clone()
        q=(z-self.mx[:,None,:])@self.xc.transpose(1,2)
        if horizon==1:return z+self.md[:,None,:]+q@self.coeff
        # Exact small recurrence; all output coordinates remain present.
        transition=self.coeff@self.xc.transpose(1,2)
        offset=(self.md[:,None,:]@self.xc.transpose(1,2))
        accum=torch.zeros_like(q)
        for _ in range(horizon):
            accum=accum+q;q=q+offset+q@transition
        return z+horizon*self.md[:,None,:]+accum@self.coeff

    def explicit(self):
        B=self.xc.transpose(1,2)@self.coeff
        K=B+torch.eye(B.shape[-1],device=B.device,dtype=B.dtype)
        b=self.md-(self.mx[:,None,:]@B)[:,0,:]
        return K,b

def scalar_alpha(states,through):
    v=states[:,1:through+1]-states[:,:through]
    a=v[:,:-1];b=v[:,1:]
    denom=(a*a).sum((1,2));num=(a*b).sum((1,2))
    return torch.where(denom>1e-30,num/torch.clamp(denom,min=1e-30),torch.ones_like(num))

def select_ridge(errors):
    # errors has Bx2: exact tie prefers 0.1, no unit shares target data.
    return torch.where(errors[:,0]<errors[:,1],errors.new_tensor(.001),errors.new_tensor(.1))

def metrics(pred,truth,z):
    err=((pred-truth)**2).mean(-1);dt=truth-z;dp=pred-z
    change=(dt*dt).mean(-1);dn=torch.linalg.vector_norm(dt,dim=-1);pn=torch.linalg.vector_norm(dp,dim=-1)
    cos=(dt*dp).sum(-1)/torch.clamp(dn*pn,min=1e-30)
    ratio=pn/torch.clamp(dn,min=1e-30)
    return dict(mse=err,change_mse=change,cosine=cos,increment_ratio=ratio)
