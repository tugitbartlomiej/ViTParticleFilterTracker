"""Czy SD potrafi zbliżyć obraz do konkretnego zoptymalizowanego latentu J128?

Brak spadku dystansu na końcowym PNG obala osiągnięcie celu tej próby.
To jeden przykład generowania, bez wnioskowania o poprawie mAP na zbiorze.
Prowadzenie DDIM zaadaptowane z maly_vit_petla_sd_rownanie/kod/prowadzenie.py.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dane import ROOT, OUT, SD, CKPT, ANNOTATIONS, load, sha
from generator_sd import GeneratorSD, SKALA_VAE
from model_snapshot import MalyViT, dekoduj
import uklady

sys.path.insert(0, str(ROOT))
from src.narzedzia.deklaracja_wynikow import zapisz_wynik


def tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float().cuda() / 255


def save_image(path, rgb):
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    seed, steps, start_strength, eta, iterations, last_clean = 20261004, 30, .5, .15, 3, 2
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    row, original, target_data, target_path = load(0)
    net = MalyViT().cuda().eval().requires_grad_(False)
    net.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True)['net'])
    net.zapisuj_latent(True)
    channels = torch.as_tensor(target_data['channels0'], device='cuda', dtype=torch.long)
    other = torch.ones(768, device='cuda', dtype=torch.bool)
    other[channels] = False
    original480 = cv2.resize(original, (480, 320), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        base_out, base_z = net(tensor(original480))
        base_z = base_z.detach().clone()
    torch.testing.assert_close(base_z[0, :, channels].cpu(), torch.from_numpy(target_data['base']), rtol=3e-4, atol=5e-5)
    full_target = base_z.clone()
    full_target[:, :, channels] = torch.from_numpy(target_data['target']).cuda()[None]
    assert torch.equal(full_target[:, :, other], base_z[:, :, other])
    denominator = (base_z[:, :, channels] - full_target[:, :, channels]).square().mean().clamp_min(1e-8)
    other_scale = base_z[:, :, other].square().mean().clamp_min(1e-8)
    np.savez_compressed(OUT / 'LATENT_cel_0000.npz', target=full_target.cpu().numpy(), base=base_z.cpu().numpy(), channels0=target_data['channels0'])
    uklady.ADN = ANNOTATIONS
    layout = uklady.uklad(dict(frame_id=row['frame_id'], native_width=original.shape[1], native_height=original.shape[0]))
    assert layout is not None
    masks = uklady.maski(layout)
    np.save(OUT / 'MASKI_0000.npy', masks)
    print('TARGET_READY', json.dumps(dict(frame=row['frame_id'], mse=float(denominator), min=float(full_target.min()), max=float(full_target.max()))), flush=True)
    G = GeneratorSD(unet=SD / 'ema_020000', vae=SD / 'sd-vae-ft-mse', tekst=SD / 'pusty_tekst.pt')
    G.unet.requires_grad_(False)
    G.vae.requires_grad_(False)
    G.vae.enable_gradient_checkpointing()
    m = torch.from_numpy(masks[None]).cuda().half() / 255
    anchor = cv2.resize(original, (576, 384), interpolation=cv2.INTER_AREA)
    save_image(OUT / 'OBRAZ_zrodlowy.png', anchor)
    G.sched.set_timesteps(steps)
    ts = G.sched.timesteps
    ac = G.sched.alphas_cumprod.cuda()
    first = int(round(steps * (1 - start_strength)))
    with torch.no_grad():
        vae_base = G.vae.encode((tensor(anchor) * 2 - 1).half()).latent_dist.mean * SKALA_VAE
    random = torch.Generator(device='cuda').manual_seed(seed)
    noise = torch.randn(vae_base.shape, generator=random, device='cuda', dtype=torch.float16)
    a0 = ac[ts[first]]
    x = (a0.sqrt() * vae_base.float() + (1-a0).sqrt() * noise.float()).half()

    def energy(clean):
        img = (G.vae.decode(clean.half() / SKALA_VAE).sample.float().clamp(-1, 1) + 1) / 2
        # Full-image path, 600 tokens, not the historical 121-token crop.
        resized = F.interpolate(img, size=(320, 480), mode='bilinear', align_corners=False, antialias=True)
        _, z = net(resized)
        e_j = (z[:, :, channels] - full_target[:, :, channels]).square().mean() / denominator
        e_other = (z[:, :, other] - base_z[:, :, other]).square().mean() / other_scale
        return e_j + .1 * e_other, e_j, e_other

    history = []
    print('SD_LOADED', time.time() - start, flush=True)
    for i in range(first, steps):
        t = ts[i]
        with torch.no_grad():
            # Sequential CFG limits peak memory on the 8 GB card.
            cond = G.unet(torch.cat([x, m], 1), t, encoder_hidden_states=G.emb).sample.float()
            uncond = G.unet(torch.cat([x, torch.zeros_like(m)], 1), t, encoder_hidden_states=G.emb).sample.float()
            eps = uncond + 2 * (cond - uncond)
        a = ac[t].float()
        xf = x.float()
        clean = (xf - (1-a).sqrt() * eps) / a.sqrt()
        if i < steps - last_clean:
            for j in range(iterations):
                variable = clean.detach().requires_grad_(True)
                value, ej, eo = energy(variable)
                grad, = torch.autograd.grad(value, variable)
                assert torch.isfinite(grad).all() and float(grad.square().mean()) > 0
                norm = grad.square().mean().sqrt()
                clean = (variable - eta * (1-a).sqrt() * grad / norm.clamp_min(1e-8)).detach()
                h = dict(step=i, iteration=j, energy=float(value.detach()), relative_mse128=float(ej.detach()), other_relative_mse=float(eo.detach()), gradient_rms=float(norm))
                history.append(h)
            eps = (xf - a.sqrt() * clean) / (1-a).sqrt()
        x = G.sched.step(eps, t, xf).prev_sample.half()
        print('DDIM', i + 1, '/', steps, 'E', history[-1]['energy'] if history else None, flush=True)
        (OUT / 'PROGRESS.json').write_text(json.dumps(dict(step=i+1,total=steps,history=history), indent=2))
    with torch.no_grad():
        img = G.vae.decode(x / SKALA_VAE).sample.float().clamp(-1, 1)
        result = ((img + 1) * 127.5).round().byte()[0].permute(1, 2, 0).cpu().numpy()
    image_path = OUT / 'OBRAZ_SD.png'
    save_image(image_path, result)
    # Evaluate the saved uint8 PNG with exactly the source detector's cv2 preprocessing.
    readback = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        output, z = net(tensor(cv2.resize(readback, (480, 320), interpolation=cv2.INTER_AREA)))
        rel = float((z[:, :, channels] - full_target[:, :, channels]).square().mean() / denominator)
        rest = float((z[:, :, other] - base_z[:, :, other]).square().mean() / other_scale)
    def detection(o):
        boxes, scores = dekoduj(o)
        best = int(scores[0].argmax())
        return dict(top_box_xyxy=boxes[0, best].cpu().tolist(), top_confidence=float(scores[0, best]))
    saved = np.load(ROOT / 'outputs/vit_latent128_maksymalizacja/DETECTIONS_FULL400.npz')
    oracle = torch.from_numpy(saved['optimized'][0:1])
    np.savez_compressed(OUT / 'LATENT_obrazu_sd.npz', latent=z.detach().cpu().numpy(), output=output.detach().cpu().numpy())
    details = dict(ziarno=seed, status='complete', source_index=0, selection='first manifest record, no quality selection', frame_id=row['frame_id'],
        image=str(image_path), image_sha256=sha(image_path), vit_checkpoint=str(CKPT), vit_sha256=sha(CKPT), target_file=str(target_path), target_sha256=sha(target_path),
        sd_sources=G.zrodlo, sd_unet_sha256=sha(SD/'ema_020000/diffusion_pytorch_model.safetensors'),
        sd_vae_sha256=sha(SD/'sd-vae-ft-mse/diffusion_pytorch_model.safetensors'), text_sha256=sha(SD/'pusty_tekst.pt'),
        torch_version=torch.__version__, device=torch.cuda.get_device_name(), seconds=time.time()-start,
        ddim_steps=steps, executed_steps=steps-first, cfg=2, img2img_strength=start_strength, guidance_eta=eta, guidance_iterations=iterations, last_unguided_steps=last_clean,
        target_shape=list(full_target.shape), target_channels=128, source_mse128=float(denominator), generated_relative_mse128=rel,
        distance_reduction_vs_source=1-rel, other_relative_mse=rest, source_detection=detection(base_out), oracle_detection=detection(oracle), generated_detection=detection(output),
        history=history, limitations=['one img2img sample, source image and true segmentation used', 'no unguided SD control, no causal claim about guidance', '0.88475 AP50 describes latent interventions on 400 images, not this generated image', 'GELU feasibility does not guarantee image realizability'])
    (OUT / 'RESULT.json').write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding='utf-8')
    zapisz_wynik(__file__, 'sd_0000', details)
    print('COMPLETE', json.dumps({k: details[k] for k in ('image', 'generated_relative_mse128', 'distance_reduction_vs_source', 'source_detection', 'oracle_detection', 'generated_detection', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
