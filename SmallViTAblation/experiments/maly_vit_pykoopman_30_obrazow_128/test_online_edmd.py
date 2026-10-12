import unittest
import numpy as np
import online_edmd as online


@unittest.skip("Online RLS deferred by latest user instruction in favor of actual PyKoopman NNDMD; no implementation/result claimed")
class OnlineSmoke(unittest.TestCase):
    def cls(self):
        self.assertTrue(hasattr(online,"RecursiveEDMD"),"Missing recursive weighted EDMD")
        return online.RecursiveEDMD

    def test_rho1_matches_weighted_batch_with_affine_bias(self):
        cls=self.cls(); rng=np.random.default_rng(918)
        xs=[rng.normal(size=(50,3)) for _ in range(5)]
        ys=[x@np.array([[.9,.1,0],[0,.8,.1],[.1,0,.9]]).T+.3+rng.normal(size=x.shape)*.1 for x in xs]
        weights=[rng.uniform(.01,1,size=50) for _ in xs]
        weights=[w/w.sum() for w in weights]
        state=cls(3,1,1e-2)
        for x,y,w in zip(xs,ys,weights): model=state.update(x,y,w)
        x=np.concatenate(xs); y=np.concatenate(ys); w=np.concatenate(weights)/5
        alpha=1e-2*np.sum(w[:,None]*x*x)/3
        dx=np.vstack([np.c_[x,np.ones(len(x))]*np.sqrt(w[:,None]),np.c_[np.eye(3)*np.sqrt(alpha),np.zeros(3)]])
        dy=np.vstack([y*np.sqrt(w[:,None]),np.eye(3)*np.sqrt(alpha)])
        reference=np.linalg.lstsq(dx,dy,rcond=None)[0].T
        np.testing.assert_allclose(np.c_[model["A"],model["b"]],reference,atol=1e-10)

    def test_forgetting_matches_time_weighted_batch(self):
        cls=self.cls(); rng=np.random.default_rng(919); rho=.95
        xs=[rng.normal(size=(60,3)) for _ in range(4)]
        ys=[rng.normal(size=(60,3)) for _ in xs]; ws=[np.ones(60)/60 for _ in xs]
        state=cls(3,rho,0)
        for x,y,w in zip(xs,ys,ws): model=state.update(x,y,w)
        x=np.concatenate(xs); y=np.concatenate(ys)
        w=np.concatenate([ws[t]*rho**(3-t) for t in range(4)]); w/=w.sum()
        reference=np.linalg.lstsq(np.c_[x,np.ones(len(x))]*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=None)[0].T
        np.testing.assert_allclose(np.c_[model["A"],model["b"]],reference,atol=1e-10)

    def test_future_poison_never_changes_prefix_fit(self):
        self.assertTrue(hasattr(online,"fit_online"),"Missing chronological prefix-only fit")
        rng=np.random.default_rng(920); data=rng.normal(size=(8,2,8,3))
        cov=np.ones((2,8)); tool=cov>0
        safe=online.fit_online(data,cov,tool,[0,1],4,.95,.01)[0]
        poisoned=data.copy(); poisoned[5:]=np.nan
        other=online.fit_online(poisoned,cov,tool,[0,1],4,.95,.01)[0]
        np.testing.assert_array_equal(safe["A"],other["A"])
        np.testing.assert_array_equal(safe["b"],other["b"])


if __name__=="__main__": unittest.main(verbosity=2)
