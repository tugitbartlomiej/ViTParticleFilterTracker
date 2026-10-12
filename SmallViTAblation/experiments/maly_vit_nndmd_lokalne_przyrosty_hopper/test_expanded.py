import unittest
import copy
import numpy as np
import torch
from state import AuxHistory,Normalizer,expanded_input,forecast_history
from model import ExpandedNNDMD,direct_loss,accumulated_gradients,optimizer_step,rollout


def aux_fixture(q=3):
    rng=np.random.default_rng(20);x=rng.normal(size=(101,q)).astype(np.float32)
    valid=np.ones_like(x,dtype=bool);valid[0]=False
    target=x.copy();tv=valid.copy()
    if q>1:tv[:,1]=np.arange(101)%10==0;tv[0,1]=False
    targets=np.where(tv,target,0)
    return AuxHistory(x,valid,targets,tv,np.arange(101)%10*10)


class ExpandedStateTests(unittest.TestCase):
    def test_anchor_H0_exact_and_identity_operator_persistence(self):
        aux=aux_fixture();norm=Normalizer.fit(2.3,aux,80)
        model=ExpandedNNDMD(3,native_dim=3,learned_dim=4,hidden=7,device='cpu')
        raw=np.random.default_rng(41).normal(size=(3,2,5,3)).astype(np.float32)
        with torch.no_grad():model.module._koopman_propagator.get_K().zero_()
        pred,*_=rollout(model,raw,80,aux,norm,10,6)
        np.testing.assert_array_equal(pred,np.broadcast_to(raw[-1],pred.shape))

    def test_paired_absolute_residual_and_baseline_gradient(self):
        aux=aux_fixture();norm=Normalizer.fit(2.3,aux,80)
        model=ExpandedNNDMD(3,native_dim=3,learned_dim=4,hidden=7,device='cpu')
        raw=np.random.default_rng(41).normal(size=(3,2,5,3)).astype(np.float32)
        pred,*_=rollout(model,raw,80,aux,norm,10,10)
        absolute,*_=rollout(model,raw,80,aux,norm,10,10,residual=False)
        with torch.no_grad():
            phi=model.encode(expanded_input(torch.tensor(raw.reshape(3,10,3)),80,aux,norm,80))
            rec=(model.decode(phi)[:,:3]*norm.native_scale).numpy().reshape(2,5,3)
        np.testing.assert_allclose(pred,raw[-1]+(absolute-rec),rtol=1e-6,atol=1e-6)
        # If A=I, loss derivative through D(phi_h)-D(phi0) is exactly zero.
        model.optimizer.zero_grad(set_to_none=True)
        phi=model.encode(expanded_input(torch.tensor(raw.reshape(3,10,3)),80,aux,norm,80))
        base=model.decode(phi)[:,:3]
        identical=model.decode(phi)[:,:3]
        prediction=torch.tensor(raw[-1].reshape(10,3))+(identical-base)*norm.native_scale
        prediction.square().sum().backward()
        for p in list(model.module._encoder.parameters())+list(model.module._decoder.parameters()):
            self.assertTrue(p.grad is None or bool(torch.all(p.grad==0)))

    def test_all_parameter_gradients_whole_image_batch_vs_micro(self):
        aux=aux_fixture();norm=Normalizer.fit(2.3,aux,80)
        a=ExpandedNNDMD(q=3,native_dim=3,learned_dim=4,hidden=7,seed=19,device='cpu')
        b=ExpandedNNDMD(q=3,native_dim=3,learned_dim=4,hidden=7,seed=19,device='cpu')
        raw=torch.tensor(np.random.default_rng(5).normal(size=(13,3,5,3)).astype(np.float32))
        loss=direct_loss(a,raw,20,aux,norm,80,rec_weight=.7,aux_weight=.2,identity_weight=.03)
        loss.backward()
        values=accumulated_gradients(b,raw,20,aux,norm,80,micro_tokens=6,rec_weight=.7,aux_weight=.2,identity_weight=.03)
        self.assertAlmostEqual(float(loss),values['loss_normalized'],places=6)
        for (name,pa),(other,pb) in zip(a.module.named_parameters(),b.module.named_parameters()):
            self.assertEqual(name,other);torch.testing.assert_close(pa.grad,pb.grad,rtol=3e-5,atol=1e-6)
        self.assertEqual(b.optimizer_steps,0)

    def test_normalizer_and_forecast_ignore_future_poison(self):
        aux=aux_fixture();other=copy.deepcopy(aux)
        other.values[81:]=np.nan;other.targets[81:]=np.nan;other.age[81:]=np.nan
        n1=Normalizer.fit(1.2,aux,80);n2=Normalizer.fit(1.2,other,80)
        for name in ['input_mean','input_std','target_mean','target_std']:
            np.testing.assert_array_equal(getattr(n1,name),getattr(n2,name))
        model=ExpandedNNDMD(q=3,native_dim=3,learned_dim=4,hidden=7,seed=19,device='cpu')
        native=np.random.default_rng(5).normal(size=(3,2,5,3)).astype(np.float32)
        first=rollout(model,native,80,aux,n1,n=10,micro_tokens=6)
        second=rollout(model,native,80,other,n2,n=10,micro_tokens=6)
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)

    def test_origin_only_encoding_A_power_and_predicted_history(self):
        aux=aux_fixture();norm=Normalizer.fit(1.2,aux,80)
        model=ExpandedNNDMD(q=3,native_dim=3,learned_dim=4,hidden=7,seed=19,device='cpu')
        native=np.random.default_rng(5).normal(size=(3,2,5,3)).astype(np.float32)
        pred,*_=rollout(model,native,80,aux,norm,n=10,micro_tokens=10)
        with torch.no_grad():
            phi=model.encode(expanded_input(torch.tensor(native.reshape(3,10,3)),80,aux,norm,80));a=model.operator();base=model.decode(phi)[:,:3]
            for h in range(10):
                phi=phi@a.T;decoded=torch.tensor(native[-1].reshape(10,3))+(model.decode(phi)[:,:3]-base)*norm.native_scale
                np.testing.assert_allclose(pred[h],decoded.numpy().reshape(2,5,3),rtol=1e-6,atol=1e-6)
        diff=forecast_history(native,pred)
        np.testing.assert_array_equal(diff[0],pred[0]-native[-1])
        np.testing.assert_array_equal(diff[1:],np.diff(pred,axis=0))

    def test_initial_padding_flags_and_boundary_guard(self):
        aux=aux_fixture();norm=Normalizer.fit(1.2,aux,80)
        native=torch.ones((3,5,3));inp=expanded_input(native,0,aux,norm,80)
        self.assertTrue(torch.equal(inp[:,3:9],torch.zeros((5,6))))
        self.assertTrue(torch.equal(inp[:,-4:-2],torch.zeros((5,2))))
        model=ExpandedNNDMD(q=3,native_dim=3,learned_dim=4,hidden=7,seed=19,device='cpu')
        with self.assertRaises(AssertionError):
            accumulated_gradients(model,np.zeros((13,1,5,3),np.float32),71,aux,norm,80)

    def test_one_adam_step_after_all_image_particles(self):
        aux=aux_fixture();norm=Normalizer.fit(1.2,aux,80)
        model=ExpandedNNDMD(q=3,native_dim=3,learned_dim=4,hidden=7,seed=19,device='cpu')
        raw=np.random.default_rng(3).normal(size=(13,28,5,3)).astype(np.float32)
        out=optimizer_step(model,raw,20,aux,norm,80,micro_tokens=17)
        self.assertEqual(out['images'],28);self.assertEqual(model.optimizer_steps,1)
        self.assertEqual({int(s['step']) for s in model.optimizer.state.values()},{1})


if __name__=='__main__':unittest.main()
