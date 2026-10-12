"""Stable Diffusion 1.5 dostrojone na klatkach Cataract-LMM (`dyfuzja_sd_filmy`) jako generator pętli wersji 4.

Wejście UNet: szum latentu (4 kanały) + maski narzędzia, rogówki i źrenicy (3 kanały, 72×48), stały pusty tekst.
Wagi: migawka EMA w formacie diffusers (MALY_VIT_SD_UNET, np. .../ema_010000), VAE sd-vae-ft-mse (MALY_VIT_SD_VAE),
pusty embedding tekstu policzony raz (MALY_VIT_SD_TEKST, pusty_tekst.pt). Rysowanie: DDIM od czystego szumu, jak
`probkuj` w `dyfuzja_sd_filmy/kod/trenuj_sd.py` (bez warunku = maski zerowe), CFG wzmacnia warunek masek.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

SKALA_VAE = 0.18215
SZUM = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')


class GeneratorSD:
    def __init__(self, dev='cuda', unet=None, vae=None, tekst=None):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        self.dev = dev
        unet = Path(unet or os.environ['MALY_VIT_SD_UNET'])
        if not (unet / 'config.json').exists():                      # katalog z migawkami: najnowsza ema_XXXXXX
            unet = max(p for p in unet.glob('ema_*') if (p / 'config.json').exists())
        vae = Path(vae or os.environ['MALY_VIT_SD_VAE'])
        tekst = Path(tekst or os.environ['MALY_VIT_SD_TEKST'])
        self.unet = UNet2DConditionModel.from_pretrained(unet, torch_dtype=torch.float16).to(dev).eval()
        self.vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16).to(dev).eval()
        self.emb = torch.load(tekst, map_location=dev).to(torch.float16)
        self.sched = DDIMScheduler(**SZUM, clip_sample=False, set_alpha_to_one=False, steps_offset=1)
        self.zrodlo = dict(unet=str(unet), vae=str(vae), tekst=str(tekst))

    @torch.no_grad()
    def rysuj(self, maski_u8: np.ndarray, ziarno: int, kroki: int = 30, cfg: float = 2.0, paczka: int = 8) -> np.ndarray:
        """maski_u8: (N, 3, 48, 72) uint8. Zwraca obrazy (N, 384, 576, 3) uint8 RGB."""
        out = []
        for s in range(0, len(maski_u8), paczka):
            m = torch.from_numpy(maski_u8[s:s + paczka]).to(self.dev, torch.float16) / 255.
            n = len(m)
            self.sched.set_timesteps(kroki)
            g = torch.Generator(device=self.dev).manual_seed(int(ziarno) * 1000 + s)
            x = torch.randn((n, 4, m.shape[2], m.shape[3]), generator=g, device=self.dev, dtype=torch.float16)
            e_ = self.emb.expand(n, -1, -1)
            for t in self.sched.timesteps:
                if cfg != 1.0:
                    xx = torch.cat([torch.cat([x, m], 1), torch.cat([x, torch.zeros_like(m)], 1)])
                    e = self.unet(xx, t, encoder_hidden_states=torch.cat([e_, e_])).sample.float()
                    e = e[n:] + cfg * (e[:n] - e[n:])
                else:
                    e = self.unet(torch.cat([x, m], 1), t, encoder_hidden_states=e_).sample.float()
                x = self.sched.step(e, t, x.float()).prev_sample.to(torch.float16)
            img = self.vae.decode(x / SKALA_VAE).sample.float().clamp(-1, 1)
            out.append(((img + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)
