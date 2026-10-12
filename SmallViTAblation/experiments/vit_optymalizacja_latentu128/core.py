"""Ograniczona interwencja po ostatnim GELU, przed fc2; brak uczenia wag."""
import torch

def intervene(z,delta,channels,scale):
    out=z.clone()
    out[:,:,channels]=z[:,:,channels]+delta*scale
    return out

@torch.no_grad()
def project(delta,radius):
    delta.clamp_(-radius,radius)

def tail(net,z,residual):
    b=net.bloki[-1]
    t=residual+b.drop_path2(b.ls2(b.mlp.drop2(b.mlp.fc2(b.mlp.drop1(z)))))
    return net.glowa(net.norm(t)).view(z.shape[0],20,30,5)
