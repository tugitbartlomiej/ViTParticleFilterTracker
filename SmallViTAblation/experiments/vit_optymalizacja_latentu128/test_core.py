"""Interwencja ma zachować 640 kanałów i gradient tylko w 128 wybranych."""
import unittest
import torch
from core import intervene, project, tail
from model_snapshot import MalyViT

class Tests(unittest.TestCase):
    def test_loss_matches_source(self):
        from run import losses
        from model_snapshot import cele,strata
        import numpy as np
        g=[np.array([[100.,80.,200.,180.]],np.float32),np.array([[30.,40.,150.,160.]],np.float32)]
        t=cele(g,20,30);o=torch.randn(2,20,30,5)*.2
        expected=torch.stack([strata(o[i:i+1],*(v[i:i+1] for v in t))[0] for i in range(2)])
        torch.testing.assert_close(losses(o,t),expected,atol=1e-5,rtol=1e-5)
    def test_channels_and_gradient(self):
        z=torch.randn(2,600,768); j=torch.arange(128)*3
        d=torch.randn(2,600,128,requires_grad=True); sd=torch.ones(128)
        y=intervene(z,d,j,sd); mask=torch.ones(768,dtype=torch.bool);mask[j]=False
        self.assertTrue(torch.equal(y[:,:,mask],z[:,:,mask]))
        y.sum().backward(); self.assertTrue(torch.equal(d.grad,torch.ones_like(d)))
        project(d,.25);self.assertLessEqual(d.abs().max().item(),.25)
    def test_tail_and_zero_intervention(self):
        torch.set_num_threads(2); net=MalyViT().eval(); cache={}
        h1=net.bloki[-1].norm2.register_forward_pre_hook(lambda m,a:cache.update(res=a[0].detach()))
        h2=net.bloki[-1].mlp.act.register_forward_hook(lambda m,a,o:cache.update(z=o.detach()))
        with torch.no_grad():o,_=net(torch.rand(1,3,320,480))
        h1.remove();h2.remove()
        y=tail(net,cache['z'],cache['res'])
        torch.testing.assert_close(y,o,atol=1e-6,rtol=1e-5)
        j=torch.arange(128); d=torch.zeros(1,600,128,requires_grad=True)
        y=tail(net,intervene(cache['z'],d,j,torch.ones(128)),cache['res'])
        torch.testing.assert_close(y,o,atol=1e-6,rtol=1e-5)
        y.square().sum().backward();self.assertGreater(d.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()
