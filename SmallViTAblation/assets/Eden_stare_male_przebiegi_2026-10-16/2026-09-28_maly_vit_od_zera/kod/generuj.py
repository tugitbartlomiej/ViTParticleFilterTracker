"""FastGAN: obraz narzędzia, którego latent w bieżącym małym ViT pasuje do latentu przewidzianego (PLAN.md).

Generator: FastGAN 256×256 RGBA uczony na wycinkach narzędzia (fastgan_narzedzie, krok 50 000, wagi EMA), zamrożony.
Dla kotwicy (sonda „original”):
  1. inwersja: z*, dla którego G(z*) odtwarza kotwicę (L1 obrazu + L1 maski), 4 losowe starty + doszlifowanie;
  2. cel: z startuje z z*, Adam minimalizuje (latent narzędzia G(z) − przewidziany latent)² + wierność obrazowi kotwicy
     + bliskość z*; latent czytany na tokenach maski kotwicy (jak w poprzednich pętlach);
  3. kontrola: to samo do OBECNEGO latentu kotwicy (obraz bez przesunięcia prognozy).
Zapis: target.png, current.png, maski, RAPORT.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import os
HERE = Path(__file__).resolve().parent
WEJ = Path(os.environ['MALY_VIT_WEJSCIA'])                      # WERSJA EDEN
CKPT = WEJ / 'fastgan/checkpoint_050000.pth'
VENDOR = WEJ / 'fastgan/models.py'


def laduj_generator(dev='cuda'):
    payload = torch.load(CKPT, map_location='cpu', weights_only=True)
    spec = importlib.util.spec_from_file_location('fastgan_models_vendor', VENDOR)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    with torch.random.fork_rng(devices=[]):
        G = mod.Generator(ngf=64, nz=256, nc=4, im_size=256)
    G.load_state_dict(payload['ema'], strict=True)
    return G.to(dev).eval().requires_grad_(False)


def wyjscie(G, z, seed=42):
    """RGBA w [0,1] przy stałym szumie warstw (NoiseInjection losuje szum)."""
    dev = [torch.cuda.current_device()] if z.is_cuda else []
    with torch.random.fork_rng(devices=dev):
        torch.manual_seed(seed)
        v = G(z)
    v = v[0] if isinstance(v, (tuple, list)) else v
    return (v.clamp(-1, 1) + 1) / 2


def _zapisz(p, x):
    from PIL import Image
    a = (x[0].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(a[..., 0] if a.shape[-1] == 1 else a).save(p)


def generuj(u, G, kotwica, cel, obecny, mu, sd, neur, out, seed=0, S=176):
    """kotwica: dict(obraz256 (1,3,256,256), maska256 (1,1,256,256), tn (121,) bool). cel/obecny: (M,) w z-jednostkach."""
    dev = u.dev
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    img, msk = kotwica['obraz256'].to(dev), kotwica['maska256'].to(dev)
    tn = kotwica['tn'].to(dev).float()
    mu_t = torch.tensor(mu[neur], dtype=torch.float32, device=dev); sd_t = torch.tensor(sd[neur], dtype=torch.float32, device=dev)
    neur_t = torch.tensor(neur, device=dev)
    net = u.net
    wymagane = [p.requires_grad for p in net.parameters()]
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval(); net.zapisuj_latent(True)

    def desk(rgba):
        x = F.interpolate(rgba[:, :3], size=(S, S), mode='area')
        _, lat = net(x)
        nar = (lat[0] * tn[:, None]).sum(0) / tn.sum().clamp(min=1)
        return (nar[neur_t] - mu_t) / sd_t

    def inwersja():
        najl = None
        for r in range(4):
            z = torch.randn((1, 256), generator=torch.Generator().manual_seed(1000 * seed + r)).to(dev).requires_grad_(True)
            opt = torch.optim.Adam([z], lr=.05)
            for it in range(160):
                o = wyjscie(G, z, 42 + r)
                L = (o[:, :3] - img).abs().mean() + (o[:, 3:4] - msk).abs().mean()
                opt.zero_grad(); L.backward(); opt.step()
                if it == 99:
                    for g in opt.param_groups:
                        g['lr'] = .01
            if najl is None or float(L) < najl[0]:
                najl = (float(L), z.detach().clone(), 42 + r)
        return najl

    try:
        l_inw, z_star, szum = inwersja()
        cel_t = torch.tensor(cel, dtype=torch.float32, device=dev); ob_t = torch.tensor(obecny, dtype=torch.float32, device=dev)
        with torch.no_grad():
            d_star = desk(wyjscie(G, z_star, szum))
        rap = dict(inwersja_l1=l_inw, reszta_inwersji_do_obecnego=float(((d_star - ob_t) ** 2).mean().sqrt()),
                   odleglosc_cel_obecny=float(((cel_t - ob_t) ** 2).mean().sqrt()))
        obrazy = {}
        for ramie, c in (('target', cel_t), ('current', ob_t)):
            z = z_star.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=.02)
            for _ in range(250):
                o = wyjscie(G, z, szum)
                L = ((desk(o) - c) ** 2).mean() + (o[:, :3] - img).abs().mean() + 1e-3 * ((z - z_star) ** 2).sum()
                opt.zero_grad(); L.backward(); opt.step()
            with torch.no_grad():
                o = wyjscie(G, z, szum); d = desk(o)
            _zapisz(out / f'{ramie}.png', o[:, :3]); _zapisz(out / f'{ramie}_mask.png', o[:, 3:4])
            obrazy[ramie] = o[:, :3]
            rap[ramie] = dict(reszta=float(((d - c) ** 2).mean().sqrt()), l1_do_kotwicy=float((o[:, :3] - img).abs().mean()),
                              odleglosc_do_obecnego=float(((d - ob_t) ** 2).mean().sqrt()),
                              odleglosc_do_celu=float(((d - cel_t) ** 2).mean().sqrt()))
        rap['l1_target_vs_current'] = float((obrazy['target'] - obrazy['current']).abs().mean())
        _zapisz(out / 'kotwica.png', img)
    finally:
        net.zapisuj_latent(False)
        for p, w in zip(net.parameters(), wymagane):
            p.requires_grad_(w)
    (out / 'RAPORT.json').write_text(json.dumps(rap, indent=1), encoding='utf-8')
    return rap
