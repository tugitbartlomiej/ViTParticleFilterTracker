"""Affine delay DMD on [full state, last increment]; no state reduction.

Tu et al. (2014), Sec. 3.2.2 motivates delay coordinates. Residual learning
follows Lu & Tartakovsky (2021); acceleration prior and scaling are our
adaptation, not a reproduction of either paper's complete algorithm.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np


class FullDelayKoopman:
    def fit(self, x, velocity, y, relative_ridge=1e-3):
        x, velocity, y = [np.asarray(a, dtype=np.float64) for a in (x, velocity, y)]
        if x.ndim != 2 or x.shape != y.shape or x.shape != velocity.shape or len(x) < 2:
            raise ValueError('Matched NxD full states and increments required')
        if not all(np.isfinite(a).all() for a in (x, velocity, y)) or relative_ridge <= 0:
            raise ValueError('Finite inputs and positive ridge required')
        self.mx, self.mv = x.mean(0), velocity.mean(0)
        u, v = x-self.mx, velocity-self.mv
        # Invertible scalar scaling of each FULL block, fitted on training only.
        self.sx = max(float(np.sqrt(np.mean(u*u))), 1e-12)
        self.sv = max(float(np.sqrt(np.mean(v*v))), 1e-12)
        self.u, self.v = u/self.sx, v/self.sv
        residual = y-x-velocity
        self.md = residual.mean(0)
        residual -= self.md
        gram = self.u@self.u.T + self.v@self.v.T
        self.ridge = relative_ridge*max(float(np.trace(gram)/len(x)), np.finfo(float).eps)
        system = gram+self.ridge*np.eye(len(x))
        self.coeff = np.linalg.solve(system, residual)
        self.solver_relative_residual = float(np.linalg.norm(system@self.coeff-residual)/max(np.linalg.norm(residual),1e-30))
        # Small exact recurrence for rollouts, not a truncated latent state.
        self.cx = self.coeff@self.u.T/self.sx
        self.cv = self.coeff@self.v.T/self.sv
        self.dx = self.md@self.u.T/self.sx
        self.dv = self.md@self.v.T/self.sv
        eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0)
        tol = len(x)*np.finfo(float).eps*max(float(eigenvalues[-1]), 1)
        self.diagnostics = dict(samples=len(x),state_dimension=x.shape[1],
            augmented_dimension=2*x.shape[1],numerical_gram_rank=int((eigenvalues>tol).sum()),
            gram_rank_tolerance=tol,effective_ridge_df=float(np.sum(eigenvalues/(eigenvalues+self.ridge))),
            regularized_condition=float((eigenvalues[-1]+self.ridge)/(eigenvalues[0]+self.ridge)),
            solver_relative_residual=self.solver_relative_residual,
            haseli_full_column_rank_assumption=False if len(x)<2*x.shape[1] else None,
            spectral_diagnostics_are_not_semantic_invariants=True)
        self.gram_eigenvalues = eigenvalues
        return self

    def predict(self, z, velocity, horizon=1):
        z, velocity = np.asarray(z,dtype=np.float64),np.asarray(velocity,dtype=np.float64)
        if z.shape != velocity.shape or z.shape[-1] != len(self.mx):
            raise ValueError('State/velocity shape mismatch')
        if not isinstance(horizon,int) or horizon<0 or not np.isfinite(z).all() or not np.isfinite(velocity).all():
            raise ValueError('Finite states and integer nonnegative horizon required')
        q = ((z-self.mx)/self.sx)@self.u.T + ((velocity-self.mv)/self.sv)@self.v.T
        qv = velocity@self.u.T/self.sx
        summed,weighted = np.zeros_like(q),np.zeros_like(q)
        for _ in range(horizon):
            summed += q
            weighted += summed
            ax = self.dx+q@self.cx
            av = self.dv+q@self.cv
            q, qv = q+qv+ax+av, qv+ax
        return z+horizon*velocity+horizon*(horizon+1)/2*self.md+weighted@self.coeff

    def save(self, path):
        np.savez(path,mu_x=self.mx,mu_velocity=self.mv,mean_acceleration=self.md,
                 scale_x=self.sx,scale_velocity=self.sv,coeff=self.coeff,ridge=self.ridge,
                 gram_eigenvalues=self.gram_eigenvalues,
                 note='Recover u and v from audited training pairs; full-state factors, no PCA')


def history_indices(steps, origin, window=4):
    """Only known SAME-image transitions; origin is known, targets later are not."""
    steps=np.asarray(steps)
    if len(steps)<window+2 or not np.all(np.diff(steps)==5):
        raise ValueError('Consecutive checkpoints spaced by five updates required')
    hits=np.flatnonzero(steps==origin)
    if len(hits)!=1 or hits[0]<window+1:
        raise ValueError('Insufficient past for window plus delay')
    j=int(hits[0])
    return [(k-1,k,k+1) for k in range(j-window,j)]


def self_test():
    rng=np.random.default_rng(127)
    x,v=rng.normal(size=(2,13,7)); y=x+.8*v+.03*x
    m=FullDelayKoopman().fit(x,v,y)
    f=np.concatenate([m.u,m.v],axis=1)
    target=y-x-v-m.md
    primal=np.linalg.solve(f.T@f+m.ridge*np.eye(14), f.T@target)
    np.testing.assert_allclose(f.T@m.coeff,primal,rtol=1e-8,atol=1e-10)
    z,vel=rng.normal(size=(2,4,7))
    zz,vv=z.copy(),vel.copy()
    for h in range(1,21):
        accel=m.md+np.concatenate([(zz-m.mx)/m.sx,(vv-m.mv)/m.sv],axis=1)@primal
        vv=vv+accel; zz=zz+vv
        np.testing.assert_allclose(m.predict(z,vel,h),zz,rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(m.predict(z,vel,0),z)
    # Two hidden velocities at identical current states become distinguishable.
    x=np.ones((20,3));v=rng.normal(size=x.shape);y=x+.7*v
    mm=FullDelayKoopman().fit(x,v,y,relative_ridge=1e-6)
    assert np.mean((mm.predict(x,v)-y)**2)<1e-8
    # Full-state constant velocity is exactly preserved, including unexcited axes.
    mm=FullDelayKoopman().fit(x,v,x+v)
    np.testing.assert_allclose(mm.predict(x,v,20),x+20*v,atol=1e-10)
    ids=history_indices(list(range(5,370,5)),210)
    assert max(k for row in ids for k in row)==41
    assert len(ids)==4 and all(b-a==c-b==1 for a,b,c in ids)
    try: history_indices([5,10,20,25,30,35],30)
    except ValueError: pass
    else: raise AssertionError('Irregular spacing accepted')
    return dict(dual_equals_primal=True,rollout_equals_full_iteration=True,
        zero_horizon=True,hidden_velocity_disambiguated=True,constant_velocity=True,
        causal_indices=True,irregular_spacing_rejected=True)

if __name__=='__main__': print(self_test())
