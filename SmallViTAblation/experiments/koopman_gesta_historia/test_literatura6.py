"""Regression tests for mathematical pitfalls identified in the six-paper audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
import unittest
import numpy as np
from koopman_pamiec_full import self_test
from audyt_literatura6 import groups_split,select_ridge


class LiteratureAuditTests(unittest.TestCase):
    def test_full_delay_exact_dual_and_causal_history(self):
        self.assertTrue(all(self_test().values()))

    def test_interpolation_does_not_establish_linearity(self):
        # Tu2014 Theorem2: independent snapshot columns permit exact fitting,
        # including observations actually produced by a nonlinear map.
        rng=np.random.default_rng(2)
        x=rng.normal(size=(4,9));y=x*x
        k=np.linalg.pinv(x)@y
        np.testing.assert_allclose(x@k,y,atol=1e-11)
        z=rng.normal(size=(6,9))
        self.assertGreater(float(np.mean((z@k-z*z)**2)),.1)

    def test_haseli_full_column_rank_cannot_be_ignored(self):
        # Has22 TheoremV.3 does NOT apply when samples < dictionary dimension.
        rng=np.random.default_rng(5)
        x=rng.normal(size=(3,8));y=rng.normal(size=(3,8))
        px=x@np.linalg.pinv(x);py=y@np.linalg.pinv(y)
        np.testing.assert_allclose(px,py,atol=1e-12)
        kf=np.linalg.pinv(x)@y;kb=np.linalg.pinv(y)@x
        index=max(abs(np.linalg.eigvals(np.eye(8)-kf@kb)))
        self.assertAlmostEqual(float(index),1,places=10)

    def test_state_vs_identity_prior_differ_off_training_span(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(3,8));y=x+.1*rng.normal(size=x.shape)
        a=np.linalg.pinv(x)@y
        b=np.eye(8)+np.linalg.pinv(x)@(y-x)
        np.testing.assert_allclose(x@a,x@b,atol=1e-12)
        np.testing.assert_allclose(b-a,np.eye(8)-np.linalg.pinv(x)@x,atol=1e-12)
        self.assertGreater(np.linalg.norm(b-a),1)

    def test_source_groups_stay_separate(self):
        rows=[dict(group_id=f'g{g:02d}',role='fit' if g<8 else 'holdout') for g in range(12) for _ in range(6)]
        inner,valid,fit,hold=groups_split(rows)
        self.assertEqual(len(inner),36)
        self.assertEqual(len(valid),12)
        self.assertTrue(set(inner).isdisjoint(valid))
        self.assertEqual(set(inner)|set(valid),set(fit))
        self.assertTrue(set(fit).isdisjoint(hold))

    def test_ridge_selection_uses_only_supplied_inner_scores(self):
        selected,means=select_ridge({'0.001':[2.,4.],'0.1':[2.,4.]})
        self.assertEqual(selected,.1)
        self.assertEqual(means[.001],3)

if __name__=='__main__':unittest.main()
