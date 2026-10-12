import unittest
import numpy as np
import torch
from model import ImageBatchNNDMD,direct_loss,microbatch_gradients,optimizer_step,rollout


class LogicalImageBatch(unittest.TestCase):
    def test_nonlinear_full_vs_uneven_micro_gradients_all_parameters(self):
        full=ImageBatchNNDMD(dim=4,hidden=[7],seed=11,device='cpu')
        micro=ImageBatchNNDMD(dim=4,hidden=[7],seed=11,device='cpu')
        w=torch.tensor(np.random.default_rng(2).normal(size=(4,3,5,4)).astype(np.float32))
        full.optimizer.zero_grad(set_to_none=True);loss=direct_loss(full,w,rec_weight=.7);loss.backward()
        stats=microbatch_gradients(micro,w,micro_tokens=6,rec_weight=.7)
        self.assertAlmostEqual(float(loss),stats['normalized_loss'],places=6)
        for (na,pa),(nb,pb) in zip(full.module.named_parameters(),micro.module.named_parameters()):
            self.assertEqual(na,nb);torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-5,atol=1e-6)
        self.assertEqual(micro._optimizer_steps,0)
    def test_step_once_after_whole_image_batch(self):
        model=ImageBatchNNDMD(dim=3,hidden=[5],seed=11,device='cpu')
        w=np.random.default_rng(2).normal(size=(4,3,5,3)).astype(np.float32)
        values=optimizer_step(model,w,micro_tokens=6)
        self.assertEqual(values['images'],3);self.assertEqual(values['optimizer_step'],1)
        steps={int(state['step']) for state in model.optimizer.state.values()}
        self.assertEqual(steps,{1})
    def test_encode_once_rollout_not_reencoding(self):
        model=ImageBatchNNDMD(dim=3,hidden=[5],seed=11,device='cpu')
        x=np.ones((2,5,3),dtype=np.float32);pred=rollout(model,x,n=3,micro_tokens=10)
        with torch.no_grad():
            phi=model.encode(torch.tensor(x.reshape(-1,3)));a=model.operator()
            for h in range(3):
                phi=phi@a.T
                np.testing.assert_allclose(pred[h],model.decode(phi).numpy().reshape(x.shape),rtol=1e-6,atol=1e-6)


if __name__=='__main__':unittest.main()
