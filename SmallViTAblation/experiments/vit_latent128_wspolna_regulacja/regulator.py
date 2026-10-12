"""Wspólne parametry J128; hipotezę obala brak transferu poprawy do innych filmów."""
import torch
from torch import nn
from torch.nn import functional as F

class Regulator(nn.Module):
    def __init__(self,kind,mu,sd):
        super().__init__();self.kind=kind
        self.register_buffer('mu',mu.clone());self.register_buffer('sd',sd.clone())
        self.bias=nn.Parameter(torch.zeros(128,device=mu.device))
        if kind=='affine':self.scale=nn.Parameter(torch.zeros(128,device=mu.device))
        if kind=='adapter':
            self.down=nn.Linear(128,16,device=mu.device);self.up=nn.Linear(16,128,device=mu.device)
            nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def delta(self,h):
        d=self.bias.expand_as(h)
        if self.kind=='affine':d=d+self.scale*h/self.sd
        if self.kind=='adapter':d=d+self.up(F.gelu(self.down((h-self.mu)/self.sd)))
        return d
    def forward(self,h):return F.gelu(h+self.sd*self.delta(h))

def tail_fast(net,J,H,Q,reg):
    change=reg(H)-F.gelu(H)
    b=net.bloki[-1]
    # In eval dropout/drop-path are identities; ls2 remains explicit.
    update=b.ls2(F.linear(change,b.mlp.fc2.weight[:,J],None))
    return net.glowa(net.norm(Q+update)).view(len(H),20,30,5)

def hook_for(J,reg):
    def hook(module,args,output):
        result=output.clone();result[:,:,J]=reg(args[0][:,:,J]);return result
    return hook
