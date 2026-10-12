import unittest
import numpy as np
import run


class TemporalProtocol(unittest.TestCase):
    def test_no_future_gradients_or_normalization(self):
        rng=np.random.default_rng(7); x=rng.normal(size=(101,2,3,4)).astype(np.float32)
        c=np.array([[1,.5,0],[.25,.75,0]],dtype=np.float32)
        a=run.prepare_trajectories(x,c,c>0,[0,1],90)
        poisoned=x.copy();poisoned[91:]=np.nan
        b=run.prepare_trajectories(poisoned,c,c>0,[0,1],90)
        self.assertEqual(a[2],b[2]);self.assertEqual(a[3]['fit_max_update'],900)
        self.assertTrue(all(t.shape==(91,4) for t in a[0]))
        for lf in [5,10,20]:
            weights=run.window_weights(a[0],a[1],lf)
            self.assertEqual(len(weights),4*(91-lf));self.assertAlmostEqual(weights.sum(),1)
        for aa,bb in zip(a[0],b[0]):np.testing.assert_array_equal(aa,bb)

    def test_rmse_same_argmin_as_aggregated_mse(self):
        values={'identity_J128':{'rmse_by_horizon_updates':{'10':1.,'20':3.}}}
        self.assertEqual(run.selection_rmse(values),np.sqrt(5.))
        self.assertNotEqual(run.selection_rmse(values),2.)

    def test_aggregate_recomputes_ratios_after_image_mse(self):
        rows=[{'image_index':i,'horizon_updates':h,'mse_state':e,'mse_motion':e,'mse_persistence':b}
              for h in range(10,101,10) for i,e,b in [(8,1.,2.),(9,9.,4.)]]
        agg=run.aggregate(rows)
        self.assertEqual(len(agg),10);self.assertEqual(agg[0]['rmse_state'],np.sqrt(5.))
        self.assertEqual(agg[0]['rmse_ratio_vs_persistence'],np.sqrt(5./3.))


if __name__=='__main__':unittest.main()
