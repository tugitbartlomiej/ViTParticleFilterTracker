"""Invariants needed to interpret this probe, without model weights or GPU."""
import unittest
import numpy as np
import torch
from proba_trudnych import composite, transform, fit_affine, box_iou


class ProbeChecks(unittest.TestCase):
    def test_foreground_and_self_paste(self):
        real=torch.rand(1,3,16,16)
        alpha=torch.zeros(1,1,16,16); alpha[:,:,4:12,7:9]=1
        bg=torch.rand_like(real,requires_grad=True)
        out=composite(real,alpha,bg)
        self.assertTrue(torch.equal(out*alpha,real*alpha))
        self.assertTrue(torch.equal(transform(out,alpha)*alpha,real*alpha))
        self.assertTrue(torch.equal(composite(real,alpha,real),real))
        out.sum().backward()
        self.assertTrue(torch.equal(bg.grad*alpha,torch.zeros_like(bg.grad)))
        self.assertGreater(float(bg.grad.abs().sum()),0)

    def test_affine_operator_handles_offset_outside_fit(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(2000,3)); matrix=rng.normal(size=(3,3)); bias=np.array([.2,-.8,.1])
        y=x@matrix+bias
        mu,scale,k=fit_affine(x,y)
        h=rng.normal(size=(100,3))
        pred=np.c_[(h-mu)/scale,np.ones(len(h))]@k
        pred=pred[:,:-1]*scale+mu
        self.assertLess(float(np.max(np.abs(pred-(h@matrix+bias)))),.03)

    def test_detection_geometry(self):
        gt=torch.tensor([.2,.3,.6,.7])
        boxes=torch.stack([gt,torch.tensor([.7,.8,.9,1.])])
        self.assertTrue(torch.allclose(box_iou(boxes,gt),torch.tensor([1.,0.])))


if __name__=='__main__':
    unittest.main()
