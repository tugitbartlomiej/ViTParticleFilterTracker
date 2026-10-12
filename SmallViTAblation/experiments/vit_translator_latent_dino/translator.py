"""Full spatial ViT latent -> normalized DINO embedding. No image decoder."""
from pathlib import Path
import numpy as np

def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

class Translator:
    def __init__(self, path):
        a=np.load(path, allow_pickle=False)
        self.mean=a['mean']; self.scale=float(a['scale'])
        self.anchors=a['anchors']; self.coefficients=a['coefficients']; self.bias=a['bias']
        self.shape=tuple(a['shape'].tolist())
    def predict(self, latent):
        z=np.asarray(latent,dtype=np.float32)
        if z.shape==self.shape:z=z[None]
        if z.shape[1:]!=self.shape:raise ValueError(f'Expected (N,{self.shape}), got {z.shape}')
        x=(z.reshape(len(z),-1)-self.mean)/self.scale
        return unit(x@self.anchors.T@self.coefficients+self.bias).astype(np.float32)

def save(path, x, y, shape, alpha):
    mean=x.mean(0); scale=max(float(np.sqrt(np.mean((x-mean)**2)*x.shape[1])),1e-12)
    anchors=(x-mean)/scale; bias=y.mean(0)
    gram=anchors@anchors.T
    coef=np.linalg.solve(gram.astype(np.float64)+alpha*np.eye(len(x)),(y-bias).astype(np.float64)).astype(np.float32)
    np.savez(path,mean=mean,scale=scale,anchors=anchors,coefficients=coef,bias=bias,shape=shape,alpha=alpha)
