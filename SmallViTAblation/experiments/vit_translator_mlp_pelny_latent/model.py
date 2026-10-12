"""Checkpoint-specific MLP: every ordered ViT activation enters a dense layer."""
import torch
from torch import nn
class FullLatentMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.register_buffer('x_mean',torch.zeros(600*768))
        self.register_buffer('x_scale',torch.ones(600*768))
        self.register_buffer('y_mean',torch.zeros(384))
        self.register_buffer('y_scale',torch.ones(384))
        self.layers=nn.Sequential(nn.Linear(600*768,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Dropout(.15),nn.Linear(hidden,256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,384))
    def forward(self,z):
        if z.shape[1:]!=(600,768):raise ValueError('Expected N x 600 x 768 full ordered latent')
        x=(z.flatten(1)-self.x_mean)/self.x_scale
        residual=self.layers(x)
        pred=torch.nn.functional.normalize(self.y_mean+self.y_scale*residual,dim=-1)
        return pred,residual

def load(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=True)
    model=FullLatentMLP(ck['hidden']).to(device);model.load_state_dict(ck['model']);model.eval()
    return model,ck
