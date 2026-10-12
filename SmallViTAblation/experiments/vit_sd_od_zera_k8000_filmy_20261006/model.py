"""Random VAE/UNet; the only loaded weights belong to the frozen K8000 teacher."""
import torch
from functools import partial
from torch.utils.checkpoint import checkpoint
from torch import nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from model_snapshot import MalyViT

VIT_SHA = '0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b'

def teacher(path, device):
    import hashlib
    with open(path, 'rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != VIT_SHA:
        raise ValueError('This experiment requires exactly the verified K8000 checkpoint')
    m = MalyViT()
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    m.to(device).eval().requires_grad_(False)
    m.zapisuj_latent(True)
    return m

def features(m, x):
    _, z = m(x)
    m._latent = None
    if z.shape[1:] != (600, 768):
        raise ValueError(f'Unexpected full latent {z.shape}')
    return z

def new_vae():
    return AutoencoderKL(in_channels=3, out_channels=3,
        down_block_types=('DownEncoderBlock2D',)*4,
        up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(64,128,256,256), layers_per_block=2,
        latent_channels=4, norm_num_groups=32, sample_size=384,
        scaling_factor=1.0)

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(768,768), nn.GELU(), nn.Linear(768,768))
        self.position = nn.Parameter(torch.randn(1,600,768)*.02)
        self.unet = UNet2DConditionModel(sample_size=48, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D','CrossAttnDownBlock2D','CrossAttnDownBlock2D','DownBlock2D'),
            up_block_types=('UpBlock2D','CrossAttnUpBlock2D','CrossAttnUpBlock2D','CrossAttnUpBlock2D'),
            block_out_channels=(128,256,384,512), layers_per_block=2,
            cross_attention_dim=768, attention_head_dim=8, norm_num_groups=32)
        # Non-reentrant recomputation is compatible with DDP and auxiliary input gradients.
        self.unet.enable_gradient_checkpointing(
            gradient_checkpointing_func=partial(checkpoint,use_reentrant=False))
    def forward(self, x, t, z):
        if z.shape[1:] != (600,768):
            raise ValueError(f'Expected full teacher latent, got {z.shape}')
        return self.unet(x,t,encoder_hidden_states=self.adapter(z)+self.position).sample

def schedule():
    return DDPMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2',
                         prediction_type='epsilon',clip_sample=False)
