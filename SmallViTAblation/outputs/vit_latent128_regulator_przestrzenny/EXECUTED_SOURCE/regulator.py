"""Czy przestrzenny regulator J128 przenosi poprawę? Brak AP poza próbą obala hipotezę."""
import torch
from torch import nn
from torch.nn import functional as F

FLOOR=-0.16997120

class Regulator(nn.Module):
    def __init__(self,config,stats):
        super().__init__();self.config=dict(config)
        for k,v in stats.items():self.register_buffer(k,v.detach().clone())
        kind=config['kind'];width=config.get('width',64)
        if kind=='affine':
            self.scale=nn.Parameter(torch.zeros(128,device=self.z_mu.device))
            self.bias=nn.Parameter(torch.zeros(128,device=self.z_mu.device))
        else:
            inp=128 if kind=='local' else 320
            if config.get('coords'):inp+=2
            layers=[nn.Conv2d(inp,width,1)]
            if kind=='spatial':layers.append(nn.GroupNorm(8,width))
            layers.append(nn.GELU())
            if kind=='spatial':
                for dilation in (1,2,4):
                    layers.extend([nn.Conv2d(width,width,3,padding=dilation,dilation=dilation),
                                   nn.GroupNorm(8,width),nn.GELU(),nn.Dropout(.1)])
            else:layers.extend([nn.Conv2d(width,width,1),nn.GELU(),nn.Dropout(.1)])
            layers.append(nn.Conv2d(width,128,1));self.mapping=nn.Sequential(*layers).to(self.z_mu.device)
            nn.init.zeros_(self.mapping[-1].weight);nn.init.zeros_(self.mapping[-1].bias)

    def correction(self,z,q):
        zn=(z-self.z_mu)/self.z_sd
        if self.config['kind']=='affine':return self.bias+self.scale*zn
        features=[zn]
        if self.config['kind']=='spatial':features.append((q-self.q_mu)/self.q_sd)
        x=torch.cat(features,-1).transpose(1,2).reshape(len(z),-1,20,30)
        if self.config.get('coords'):
            yy,xx=torch.meshgrid(torch.linspace(-1,1,20,device=z.device),torch.linspace(-1,1,30,device=z.device),indexing='ij')
            coords=torch.stack([xx,yy])[None].expand(len(z),-1,-1,-1)
            x=torch.cat([x,coords],1)
        return self.mapping(x).flatten(2).transpose(1,2)

    def forward(self,z,q):return (z+self.z_sd*self.correction(z,q)).clamp(min=FLOOR)

def from_changed(net,J,z,q,changed):
    block=net.bloki[-1]
    update=block.ls2(F.linear(changed-z,block.mlp.fc2.weight[:,J],None))
    return net.glowa(net.norm(q+update)).reshape(len(z),20,30,5)

def predict_tail(net,J,z,q,reg):return from_changed(net,J,z,q,reg(z,q))

def full_with_regulator(net,J,x,reg):
    # First capture the original state; the second pass verifies hook replacement.
    cache={};b=net.bloki[-1]
    h1=b.mlp.act.register_forward_hook(lambda m,a,o:cache.update(z=o[:,:,J].detach()))
    h2=b.register_forward_hook(lambda m,a,o:cache.update(q=o.detach()))
    with torch.no_grad():net(x)
    h1.remove();h2.remove()
    changed=reg(cache['z'],cache['q']);flags={}
    def hook(m,a,o):
        out=o.clone();out[:,:,J]=changed
        mask=torch.ones(768,dtype=torch.bool,device=x.device);mask[J]=False
        flags['unchanged640']=torch.equal(o[:,:,mask],out[:,:,mask])
        flags['min_activation']=float(out[:,:,J].min().detach())
        return out
    h=b.mlp.act.register_forward_hook(hook)
    try:output,_=net(x)
    finally:h.remove()
    return output,cache,flags
