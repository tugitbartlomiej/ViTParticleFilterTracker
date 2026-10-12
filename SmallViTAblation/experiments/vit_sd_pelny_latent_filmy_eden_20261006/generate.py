"""DDIM z pełnego warunku Bx600x768 do RGB w RAM; żadnych zapisów plików."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import torch
from model import Denoiser, load_sd, restore_trainable


def _weights_digest(root):
    files = sorted(Path(root).glob('*.safetensors')) or sorted(Path(root).glob('*.bin'))
    result = {}
    for path in files:
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8*1024*1024), b''):
                h.update(chunk)
        result[path.name] = h.hexdigest()
    return result


@dataclass
class TrainedSD:
    model: Denoiser
    vae: object
    scheduler_config: dict
    device: torch.device
    step: int

    def generate(self, condition, steps=50, seed=0):
        return generate(self, condition, steps=steps, seed=seed)


def load_trained(sd_root, vae_root, checkpoint, device='cuda'):
    """Load original frozen base + learned adapter/attn2 (or full trained UNet).

    checkpoint is one checkpoint_XXXXXXXX.pt from run.py. No ViT checkpoint,
    tokenizer, image, annotation or cached natural latent is loaded here.
    """
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ck.get('format_version') != 1:
        raise ValueError('Expected training checkpoint format_version=1')
    signature = ck['signature']
    # Attn2-only checkpoints depend on the exact original frozen UNet/VAE.
    provenance = signature['provenance']
    for root, key in ((sd_root, 'unet_weights'), (vae_root, 'vae_weights')):
        if _weights_digest(root) != provenance[key]:
            raise ValueError(f'Base weights differ from training: {key}')
    unet, vae, scheduler = load_sd(sd_root, vae_root, signature['mode'], device, checkpointing=False)
    model = Denoiser(unet, signature['input_scale']).to(device)
    restore_trainable(model, ck['trainable'])
    model.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    result = TrainedSD(model, vae, dict(scheduler.config), device, int(ck['step']))
    del ck  # optimizer was read as part of the container, but is not retained
    return result


@torch.inference_mode()
def generate(trained, condition, steps=50, seed=0):
    """Return float32 RGB [B,3,384,576] in [0,1] on trained.device; no file writes.

    condition can be a natural or externally supplied artificial full latent.
    It must have the trained ordering/amplitude; no automatic normalization.
    No classifier-free guidance is assumed (unconditional branch was not trained).
    A local generator makes initial noise reproducible without changing global RNG.
    For large B, the caller should split conditions into batches to fit GPU RAM.
    """
    from diffusers import DDIMScheduler
    if not isinstance(condition, torch.Tensor) or condition.ndim != 3 or condition.shape[1:] != (600, 768):
        raise ValueError('Expected floating Tensor Bx600x768')
    if not condition.is_floating_point() or len(condition) == 0 or not torch.isfinite(condition).all():
        raise ValueError('Condition must be nonempty and finite floating point')
    if not isinstance(steps, int) or not 1 <= steps <= 1000:
        raise ValueError('steps must be integer in [1,1000]')
    device = trained.device
    scheduler = DDIMScheduler.from_config(trained.scheduler_config, clip_sample=False,
                                          set_alpha_to_one=False, steps_offset=0 if steps == 1000 else 1)
    scheduler.set_timesteps(steps, device=device)
    rng = torch.Generator(device=device).manual_seed(int(seed))
    sample = torch.randn((len(condition), 4, 48, 72), device=device,
                         dtype=torch.float32, generator=rng) * scheduler.init_noise_sigma
    condition = condition.to(device=device, dtype=torch.float32)
    for timestep in scheduler.timesteps:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            prediction = trained.model(scheduler.scale_model_input(sample, timestep), timestep, condition)
        sample = scheduler.step(prediction.float(), timestep, sample, eta=0.).prev_sample
    with torch.autocast(device.type, enabled=False):
        rgb = (trained.vae.decode(sample.float() / float(trained.vae.config.scaling_factor)).sample + 1) / 2
    return rgb.float().clamp(0, 1)
