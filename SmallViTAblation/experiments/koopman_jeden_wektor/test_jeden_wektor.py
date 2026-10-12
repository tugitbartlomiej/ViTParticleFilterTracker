"""Numerical and split tests; mismatches invalidate the token forecast test."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import unittest,numpy as np
from modele import TokenModel,CONFIGS,window_indices,training

class Tests(unittest.TestCase):
    def test_explicit_matrices(self):
        r=np.random.default_rng(10)
        x,v,y=r.normal(size=(3,24,7));z,w=r.normal(size=(2,5,7))
        for name,ridge in CONFIGS:
            m=TokenModel().fit(x,v,y,name,ridge);A,b=m.explicit()
            s=z.copy() if name=='state' else np.concatenate([z,w],axis=1)
            for h in range(1,6):
                s=s@A+b
                # Compare full-vector relative error, not a ratio at near-zero
                # coordinates. Dual recurrence cancels large coefficients in
                # the deliberately overdetermined random-data stress case.
                error=np.linalg.norm(m.predict(z,w,h)-s[:,:7])/max(np.linalg.norm(s[:,:7]),1)
                self.assertLess(error,1e-8,(name,ridge,h,error))
    def test_known_velocity(self):
        r=np.random.default_rng(8);x,v=r.normal(size=(2,120,384));y=x+.8*v
        m=TokenModel().fit(x,v,y,'scalar_velocity',0)
        np.testing.assert_allclose(m.alpha,.8,atol=1e-14)
        np.testing.assert_allclose(m.predict(x,v),y,atol=1e-14)
    def test_causal_pairing(self):
        steps=list(range(5,370,5));d=np.arange(73*5*2*3).reshape(73,5,2,3).astype('f4')
        triples=window_indices(steps,105)
        self.assertEqual(len(triples),8);self.assertEqual(steps[max(c for a,b,c in triples)],105)
        x,v,y=training(d,steps,105,np.array([1,4]),1)
        np.testing.assert_array_equal(y[-2:],d[20,[1,4],1])
        self.assertEqual(x.shape,(16,3))
    def test_spacing(self):
        with self.assertRaises(ValueError):window_indices([5,10,20],20)

if __name__=='__main__':unittest.main(verbosity=2)
