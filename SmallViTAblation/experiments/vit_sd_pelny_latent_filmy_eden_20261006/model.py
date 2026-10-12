"""Pełny naturalny latent ViT warunkuje SD; tokeny zachowują kolejność i amplitudę."""
from pathlib import Path
import torch
from torch import nn
from model_snapshot import MalyViT, cele, strata


class TokenAdapter(nn.Module):
    def __init__(self, input_scale=1.0):
        super().__init__()
        if input_scale <= 0:
            raise ValueError('input_scale must be positive')
        self.register_buffer('input_scale', torch.tensor(float(input_scale)))
        self.projection = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768))
        self.rows = nn.Parameter(torch.randn(20, 768) * .02)
        self.cols = nn.Parameter(torch.randn(30, 768) * .02)

    def forward(self, latent):
        if latent.shape[1:] != (600, 768):
            raise ValueError(f'Expected Bx600x768, got {tuple(latent.shape)}')
        position = (self.rows[:, None] + self.cols[None, :]).reshape(1, 600, 768)
        return self.projection(latent / self.input_scale) + position


class Denoiser(nn.Module):
    def __init__(self, unet, input_scale=1.0):
        super().__init__()
        self.unet = unet
        self.adapter = TokenAdapter(input_scale)

    def forward(self, noisy, timestep, latent):
        # No text, bbox, image features or spatial mask enters the UNet.
        sample = torch.cat((noisy, noisy.new_zeros(noisy.shape[0], 3, *noisy.shape[2:])), 1)
        return self.unet(sample, timestep, encoder_hidden_states=self.adapter(latent)).sample


def load_vit(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if set(ck) != {'net', 'krok'} or ck['krok'] != 1000:
        raise ValueError('Expected original MODEL_VIT_K1000.pt dictionary: net, krok=1000')
    vit = MalyViT()
    vit.load_state_dict(ck['net'], strict=True)
    vit.to(device).eval().requires_grad_(False)
    vit.zapisuj_latent(True)  # capture the actual natural forward after final MLP GELU
    return vit


def vit_forward(vit, image):
    output, latent = vit(image)
    vit._latent = None  # release hook reference; returned tensor retains its graph if needed
    if latent is None or latent.shape[1:] != (600, 768):
        raise ValueError('ViT must produce the natural 20x30x768 activation')
    return output, latent


def load_sd(sd_root, vae_root, mode, device, checkpointing=True):
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
    # sd_root is the EMA020000 directory itself, not an SD pipeline directory.
    config = UNet2DConditionModel.load_config(str(Path(sd_root)), local_files_only=True)
    if config['in_channels'] != 7 or config['cross_attention_dim'] != 768 or config['out_channels'] != 4:
        raise ValueError('Expected existing UNet: 7 input channels, 4 output, cross_attention_dim=768')
    dtype = torch.bfloat16 if device.type == 'cuda' and mode == 'cross-attention' else torch.float32
    unet = UNet2DConditionModel.from_pretrained(sd_root, local_files_only=True, torch_dtype=dtype)
    if mode == 'cross-attention':
        unet.requires_grad_(False)
        for name, p in unet.named_parameters():
            if '.attn2.' in name:
                p.requires_grad_(True)
                p.data = p.data.float()
    else:
        unet.requires_grad_(True)
    if checkpointing:
        unet.enable_gradient_checkpointing()
    vae = AutoencoderKL.from_pretrained(vae_root, local_files_only=True, torch_dtype=torch.float32)
    vae.eval().requires_grad_(False).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=.00085, beta_end=.012,
                              beta_schedule='scaled_linear', prediction_type='epsilon', clip_sample=False)
    return unet.to(device), vae, scheduler


def auxiliary_loss(estimate, target_latent, vae, vit, bbox, has_tool, scale, detector_weight):
    # VAE/ViT parameters are frozen, but this branch MUST retain image gradients.
    with torch.autocast(estimate.device.type, enabled=False):
        generated = (vae.decode(estimate.float() / scale).sample + 1) / 2
    image = torch.nn.functional.interpolate(generated.clamp(0, 1), (320, 480),
                                            mode='bilinear', align_corners=False)
    output, actual = vit_forward(vit, image)
    energy = target_latent.float().square().sum(-1).clamp_min(1e-6)
    consistency = ((actual.float() - target_latent.float()).square().sum(-1) / energy).mean()
    indices = [i for i, flag in enumerate(has_tool) if bool(flag)]
    detection = output.sum() * 0
    if indices:
        # Snapshot target builder uses Python round; pass native numeric rows,
        # not zero-dimensional torch tensors (which do not implement round).
        boxes = [(torch.as_tensor(bbox[i], dtype=torch.float32).reshape(-1, 4).cpu()
                  * torch.tensor([480., 320., 480., 320.])).tolist() for i in indices]
        heat, ltrb, positive = cele(boxes, 20, 30)
        detection, _ = strata(output[indices], heat.to(output.device), ltrb.to(output.device),
                             positive.to(output.device))
    return consistency + detector_weight * detection, consistency.detach(), detection.detach()


def trainable_state(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError('Checkpoint trainable parameter keys do not match mode/rank/architecture')
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(state[name].to(device=p.device, dtype=p.dtype))
