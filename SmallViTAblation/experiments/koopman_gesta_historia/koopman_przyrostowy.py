"""Full-state affine ridge around identity, exact dual computation (no PCA)."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np

class IncrementKoopman:
    def fit(self,x,y,relative_ridge=1e-3):
        x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
        if x.ndim!=2 or x.shape!=y.shape or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Finite paired NxD states required')
        if relative_ridge<=0: raise ValueError('Positive regularization required')
        self.mx=x.mean(0)
        delta=y-x
        self.md=delta.mean(0)
        self.xc=x-self.mx
        self.dc=delta-self.md
        gram=self.xc@self.xc.T
        self.ridge=relative_ridge*max(float(np.trace(gram)/len(x)),np.finfo(float).eps)
        self.coeff=np.linalg.solve(gram+self.ridge*np.eye(len(x)),self.dc)
        # B = Xc.T @ coeff; K = I + B. Never materialize D x D.
        self.transition=self.coeff@self.xc.T
        self.offset=self.md@self.xc.T
        return self

    def predict(self,z,horizon=1):
        z=np.asarray(z,dtype=np.float64)
        if z.shape[-1]!=len(self.mx) or not np.isfinite(z).all(): raise ValueError('Invalid state')
        if not isinstance(horizon,int) or horizon<0: raise ValueError('Integer horizon >=0 required')
        q=(z-self.mx)@self.xc.T
        accum=np.zeros_like(q)
        for _ in range(horizon):
            accum+=q
            q=q+self.offset+q@self.transition
        return z+horizon*self.md+accum@self.coeff

    def save(self,path):
        np.savez_compressed(path,mu_x=self.mx,mean_delta=self.md,X_centered=self.xc,
                            delta_centered=self.dc,coeff=self.coeff,ridge=self.ridge,
                            eigenvalues_learned=1+np.linalg.eigvals(self.transition),
                            identity_prior=True)

def self_test():
    rng=np.random.default_rng(52)
    x=rng.normal(size=(11,6));y=x+.04*rng.normal(size=x.shape)
    model=IncrementKoopman().fit(x,y)
    b=np.linalg.solve(model.xc.T@model.xc+model.ridge*np.eye(6),model.xc.T@model.dc)
    np.testing.assert_allclose(model.xc.T@model.coeff,b,atol=1e-11)
    z=rng.normal(size=(3,6));expected=z.copy()
    for _ in range(20):expected=expected+model.md+(expected-model.mx)@b
    np.testing.assert_allclose(model.predict(z,20),expected,atol=1e-10)
    np.testing.assert_allclose(model.predict(z,0),z,atol=1e-14)
    # Features 3..7 never excited in fit: arbitrary new perturbation survives.
    xx=np.zeros((9,8));xx[:,:3]=rng.normal(size=(9,3))
    yy=xx.copy(); yy[:,:3]+=.03*xx[:,:3]
    mm=IncrementKoopman().fit(xx,yy)
    zz=rng.normal(size=(2,8));v=np.zeros_like(zz);v[:,3:]=rng.normal(size=(2,5))
    np.testing.assert_allclose(mm.predict(zz+v)-mm.predict(zz),v,atol=1e-12)
    # Constant affine drift should be retained, including zero-variance input.
    xx=np.ones((4,5));dd=np.arange(5)*.1
    mm=IncrementKoopman().fit(xx,xx+dd)
    np.testing.assert_allclose(mm.predict(xx,20),xx+20*dd,atol=1e-12)
    mm=IncrementKoopman().fit(x,x)
    np.testing.assert_allclose(mm.predict(z,20),z,atol=1e-12)
    return dict(dual_equals_primal=True,multistep_equals_iteration=True,
                unseen_input_perturbations_preserved=True,zero_horizon=True,
                constant_drift=True,identity_dynamics=True)

if __name__=='__main__':print(self_test())
