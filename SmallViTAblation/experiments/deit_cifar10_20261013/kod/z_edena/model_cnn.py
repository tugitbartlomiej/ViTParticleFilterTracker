"""Drugi model do porównania „cząstki jak w szkle”: mały konwolucyjny odpowiednik MalyViT (ta sama łodyga, ta sama gęsta głowa,
ta sama definicja cząstki = 768 aktywacji GELU w MLP ostatniego bloku na każdą komórkę siatki o kroku 16 px).

Blok (w stylu ConvNeXt): mieszanie przestrzenne depthwise k×k (lokalne, zamiast globalnej uwagi) → liniowe 1×1 → MLP dim→768→dim
z GELU. Bez pozycji sin-cos (konwolucja jest równoważna względem przesunięć). Interfejs jak MalyViT: forward(x) → (mapa (B,h,w,5),
latent (B,h·w,768) gdy zapisuj_latent(True)); `bloki[i].mlp.act` do skanu po warstwach (obserwator_rp.py).
Parametrów: ~2,2 M (ViT: 2,9 M) — różnica w mieszaniu przestrzennym (depthwise 5×5: 4,8 k na blok zamiast uwagi 148 k).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import model as M


class Mlp(nn.Module):
    def __init__(self, dim, ukryty):
        super().__init__()
        self.fc1 = nn.Linear(dim, ukryty); self.act = nn.GELU(); self.fc2 = nn.Linear(ukryty, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class BlokConv(nn.Module):
    def __init__(self, dim, mlp=4.0, jadro=5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dw = nn.Conv2d(dim, dim, jadro, 1, jadro // 2, groups=dim)
        self.pw = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp))

    def forward(self, t, h, w):
        B, N, C = t.shape
        y = self.norm1(t).transpose(1, 2).reshape(B, C, h, w)
        y = self.dw(y).flatten(2).transpose(1, 2)
        t = t + self.pw(y)
        return t + self.mlp(self.norm2(t))


class MalyCNN(nn.Module):
    def __init__(self, dim=192, glebokosc=6, mlp=4.0, jadro=5):
        super().__init__()
        self.lodyga = nn.Sequential(M._cba(3, 24), M._cba(24, 48), M._cba(48, 96), nn.Conv2d(96, dim, 3, 2, 1))
        self.bloki = nn.ModuleList([BlokConv(dim, mlp, jadro) for _ in range(glebokosc)])
        self.norm = nn.LayerNorm(dim)
        self.glowa = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 5))
        nn.init.constant_(self.glowa[-1].bias, 0.0)
        with torch.no_grad():
            self.glowa[-1].bias[0] = -4.6
            self.glowa[-1].bias[1:] = math.log(3.0)
        self._latent = None
        self._hak = None

    def zapisuj_latent(self, tak: bool):
        if tak and self._hak is None:
            self._hak = self.bloki[-1].mlp.act.register_forward_hook(lambda m, i, o: setattr(self, '_latent', o))
        if not tak and self._hak is not None:
            self._hak.remove(); self._hak = None; self._latent = None

    def tokeny(self, x):
        x = (x - M.MEAN.to(x.device, x.dtype)) / M.STD.to(x.device, x.dtype)
        f = self.lodyga(x)
        B, C, h, w = f.shape
        t = f.flatten(2).transpose(1, 2)
        for b in self.bloki:
            t = b(t, h, w)
        return t, h, w

    def forward(self, x):
        t, h, w = self.tokeny(x)
        B = x.shape[0]
        o = self.glowa(self.norm(t)).view(B, h, w, 5)
        return o, self._latent


def liczba_parametrow(net):
    return int(sum(p.numel() for p in net.parameters()))
