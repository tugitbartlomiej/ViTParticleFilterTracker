"""Wspólne: FastGAN (z opcjonalnym sterowaniem cech 8×8), mały ViT z kroku 1000, pełny latent narzędzia, cele Koopmana.

Opis ustawienia: PLAN.md, sekcja „Ustawienie”.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
V1 = EXP / 'maly_vit_od_zera'
sys.path.insert(0, str(V1))
import generuj as GN  # noqa: E402  (lokalne ścieżki FastGAN 50k)
import model as M  # noqa: E402
import tozsamosc as TZ  # noqa: E402


def _modul(nazwa, p):
    spec = importlib.util.spec_from_file_location(nazwa, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


PR2 = _modul('prognoza_v2', EXP / 'maly_vit_wektor_tozsamosci' / 'kod' / 'prognoza.py')
PROBE = V1 / 'eden' / 'wejscia' / 'probe'
BIEG = V1 / 'wyniki' / 'petla_s42'
S, SIATKA, NZ = 176, 11, 256
DEV = 'cuda'
WYN = HERE / 'wyniki'


def laduj_vit(pt=BIEG / 'wspolny' / 'k02.pt'):
    st = torch.load(pt, map_location='cpu', weights_only=False)
    net = M.MalyViT()
    net.load_state_dict(st['net'])
    net.to(DEV).eval().requires_grad_(False)
    net.zapisuj_latent(True)
    return net


def laduj_G():
    return GN.laduj_generator(DEV)


def generuj_rgba(G, z, d8=None, seed=42, d16=None):
    """RGBA w [0,1]; d8: przesunięcie cech 8×8 (B,512,8,8), d16: cech 16×16 (B,256,16,16), albo None.
    Stały szum warstw jak w generuj.wyjscie."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.manual_seed(seed)
        f4 = G.init(z)
        f8 = G.feat_8(f4)
        if d8 is not None:
            f8 = f8 + d8
        f16 = G.feat_16(f8)
        if d16 is not None:
            f16 = f16 + d16
        f32 = G.feat_32(f16)
        f64 = G.se_64(f4, G.feat_64(f32))
        f128 = G.se_128(f8, G.feat_128(f64))
        f256 = G.se_256(f16, G.feat_256(f128))
        v = G.to_big(f256)
    return (v.float().clamp(-1, 1) + 1) / 2


def tokeny_narzedzia(rgba):
    """Binarne wagi tokenów narzędzia (B,121) z maski obrazu, jak w sondach."""
    m = F.adaptive_avg_pool2d(F.interpolate(rgba[:, 3:4].detach(), size=(S, S), mode='nearest').round(), (SIATKA, SIATKA)).flatten(1)
    return (m >= .25).float()


def latent_na_tokenach(net, rgba, w, mu, sd):
    """Pełny latent (B,768) na podanych (stałych) tokenach w, w z-jednostkach."""
    x = F.interpolate(rgba[:, :3], size=(S, S), mode='area')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, lat = net(x)
    lat = lat.float()
    nar = (lat * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-3)
    return (nar - mu) / sd


def latent_narzedzia(net, rgba, mu, sd):
    """Pełny latent (B,768) na tokenach narzędzia, w z-jednostkach; plus liczba tokenów narzędzia.
    Tokeny jak w sondach (dane.sondy): maska → 176 najbliższym sąsiadem → udział w tokenie 16×16 ≥ 0,25, wagi binarne
    i odłączone od gradientu (gradient płynie przez obraz). Sprawdzone: 0,16 z-jednostki od zapisu sond w tym samym
    kroku (wagi miękkie dawały 0,24)."""
    x = F.interpolate(rgba[:, :3], size=(S, S), mode='area')
    m = F.adaptive_avg_pool2d(F.interpolate(rgba[:, 3:4].detach(), size=(S, S), mode='nearest').round(), (SIATKA, SIATKA)).flatten(1)
    w = (m >= .25).float()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        _, lat = net(x)
    lat = lat.float()
    nar = (lat * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-3)
    return (nar - mu) / sd, w.sum(1)


def cele_koopmana(H=10):
    """Latenty sond w kroku 1000 (bloki po 5 zapisów), standaryzacja negatywami, prognoza Koopmana na H bloków."""
    man = json.loads((PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    opis = [dict(wariant=m['variant']) for m in man]
    L = np.concatenate([np.load(BIEG / 'wspolny' / f'k0{k}' / 'LATENT.npz')['latent'] for k in (1, 2)])[1:]
    Lb = L[:len(L) // 5 * 5].astype(np.float32).reshape(-1, 5, *L.shape[1:]).mean(1)       # (20,192,2,768)
    st = TZ.statystyki(Lb[-1], obecn, opis)
    Z = (Lb[:, obecn, 0] - st['mu']) / st['sd']
    cel = PR2.prognoza_bloku(Z.astype(np.float64), H)['koopman']
    poz = np.where(obecn)[0]
    kotw = [j for j, i in enumerate(poz) if man[i]['variant'] == 'original']
    obrazy = []
    for j in kotw:
        m = man[poz[j]]
        img = np.asarray(Image.open(PROBE / m['image']).convert('RGB'), np.float32) / 255
        msk = np.asarray(Image.open(PROBE / m['mask']).convert('L'), np.float32)[..., None] / 255
        obrazy.append(np.concatenate([img, msk], -1))
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)  # noqa: E731
    return dict(mu=t(st['mu']), sd=t(st['sd']), teraz=t(Z[-1]), cel=t(cel), kotwice=kotw,
                obrazy=t(np.stack(obrazy)).permute(0, 3, 1, 2), wszystkie_teraz=t(Z[-1]))


def odl(a, b):
    """Odległość latentów (RMSE po 768 neuronach), dla każdej próbki osobno."""
    return ((a - b) ** 2).mean(1).sqrt()


def zapisz_siatke(p, wiersze, podpisy=None):
    """wiersze: lista tensorów (B,4,256,256) RGBA; zapis PNG: kolumny = wiersze listy, wiersze = próbki."""
    B = wiersze[0].shape[0]
    kafle = [[(w[i, :3].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
              for w in wiersze] for i in range(B)]
    siatka = np.concatenate([np.concatenate(r, 1) for r in kafle], 0)
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(siatka).save(p)
