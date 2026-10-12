import unittest
import numpy as np
import nndmd_fit as experiment


class NNDMDDataSmoke(unittest.TestCase):
    def test_trajectories_keep_token_identity_and_ignore_future(self):
        self.assertTrue(hasattr(experiment,"prepare_trajectories"),"Missing weighted prefix trajectory construction")
        rng=np.random.default_rng(1280); latent=rng.normal(size=(8,3,4,3)).astype(np.float32)
        coverage=np.array([[1.,.25,0,0],[.5,.5,.25,0],[0,1,0,0]],dtype=np.float32)
        traj,w,scale,meta=experiment.prepare_trajectories(latent,coverage,coverage>0,[0,1],4)
        self.assertEqual(len(traj),5)
        self.assertEqual(traj[0].shape,(5,3))
        np.testing.assert_array_equal(traj[0],latent[:5,0,0])
        self.assertAlmostEqual(w[:2].sum(),.5)
        self.assertAlmostEqual(w[2:].sum(),.5)
        poisoned=latent.copy(); poisoned[5:]=np.nan
        other=experiment.prepare_trajectories(poisoned,coverage,coverage>0,[0,1],4)
        self.assertEqual(scale,other[2]); np.testing.assert_array_equal(w,other[1])
        self.assertTrue(all(np.isfinite(t).all() for t in other[0]))

    def test_window_sampler_has_equal_image_coverage_distribution(self):
        self.assertTrue(hasattr(experiment,"window_weights"),"Missing coverage weighted window sampler")
        weights=experiment.window_weights([np.ones((8,3)),np.ones((5,3))],np.array([.7,.3]),3)
        self.assertAlmostEqual(weights.sum(),1)
        self.assertAlmostEqual(weights[:5].sum(),.7)
        self.assertAlmostEqual(weights[5:].sum(),.3)

    def test_beta0_cached_rollout_preserves_package_loss_and_gradient(self):
        import torch
        from pykoopman.regression._nndmd import DLKoopmanRegressor
        torch.manual_seed(903)
        ce=dict(input_size=3,hidden_sizes=[],output_size=3,activations="linear")
        cd=dict(input_size=3,hidden_sizes=[],output_size=3,activations="linear")
        original=DLKoopmanRegressor(None,1,2,ce,cd,False,.001)
        adapted=DLKoopmanRegressor(None,1,2,ce,cd,False,.001)
        adapted.load_state_dict(original.state_dict())
        experiment.install_loss_and_rollout_adapter(adapted,np.array([0,2]),0)
        batch=(torch.randn(4,3),torch.randn(4,2,3),torch.full((4,),2,dtype=torch.long))
        loss_a=original.training_step(batch,0); loss_b=adapted.training_step(batch,0)
        torch.testing.assert_close(loss_a,loss_b,atol=1e-6,rtol=1e-6)
        loss_a.backward(); loss_b.backward()
        for pa,pb in zip(original.parameters(),adapted.parameters()):
            torch.testing.assert_close(pa.grad,pb.grad,atol=1e-6,rtol=1e-5)

    def test_beta1_adds_identity_reconstruction_and_prediction(self):
        import torch
        from pykoopman.regression._nndmd import DLKoopmanRegressor
        torch.manual_seed(904)
        ce=dict(input_size=3,hidden_sizes=[],output_size=3,activations="linear")
        cd=dict(input_size=3,hidden_sizes=[],output_size=3,activations="linear")
        module=DLKoopmanRegressor(None,1,2,ce,cd,False,.001)
        experiment.install_loss_and_rollout_adapter(module,np.array([0,2]),1)
        x=torch.randn(4,3); y=torch.randn(4,2,3); ys=torch.full((4,),2,dtype=torch.long)
        encoded=module._encoder(x); future=module._propagate_encoded_n_steps(encoded,2)
        predicted=torch.stack([module._decoder(future[:,i]) for i in range(2)],1)
        rx=module._decoder(encoded); ry=torch.stack([module._decoder(module._encoder(y[:,i])) for i in range(2)],1)
        full=((predicted-y)**2).mean()+((rx-x)**2).mean()+((ry-y)**2).mean()
        j=[0,2]
        identity=((predicted[...,j]-y[...,j])**2).mean()+((rx[...,j]-x[...,j])**2).mean()+((ry[...,j]-y[...,j])**2).mean()
        loss=module.training_step((x,y,ys),0)
        torch.testing.assert_close(loss,full+identity,atol=1e-6,rtol=1e-6)


if __name__=="__main__": unittest.main(verbosity=2)
