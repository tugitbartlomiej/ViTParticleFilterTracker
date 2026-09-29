"""Model dyfuzyjny wycinków narzędzia (PLAN.md): UNet, harmonogram szumu, próbkowanie DDIM, średnia wykładnicza wag.

Obraz: 4 kanały (RGB + maska narzędzia), 176×176, wartości w [-1, 1]. Sieć przewiduje dodany szum ε (DDPM, Ho i in.
2020); harmonogram kosinusowy (Nichol i Dhariwal 2021), T = 1000; próbkowanie DDIM (Song i in. 2021).
UNet: poziomy 176, 88, 44, 22 px, kanały 64, 128, 192, 256, po 2 bloki resztkowe na poziom, uwaga na 22 px i w środku.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T_KROKI = 1000


def osadz_czas(t, dim):
    pol = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(pol, device=t.device, dtype=torch.float32) / pol)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class BlokRes(nn.Module):
    def __init__(self, i, o, tdim, drop=0.0):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, i), nn.Conv2d(i, o, 3, padding=1)
        self.t = nn.Linear(tdim, 2 * o)
        self.n2, self.drop, self.c2 = nn.GroupNorm(32, o), nn.Dropout(drop), nn.Conv2d(o, o, 3, padding=1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)
        self.skrot = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.t(F.silu(te))[:, :, None, None].chunk(2, 1)
        h = self.c2(self.drop(F.silu(self.n2(h) * (1 + s) + b)))
        return self.skrot(x) + h


class Uwaga(nn.Module):
    def __init__(self, c, glowy=4):
        super().__init__()
        self.n, self.qkv, self.o, self.g = nn.GroupNorm(32, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1), glowy
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.g, C // self.g, H * W).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.o(h.transpose(-1, -2).reshape(B, C, H, W))


class WGore(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.c(F.interpolate(x, scale_factor=2, mode='nearest'))


class UNet(nn.Module):
    def __init__(self, kanaly=4, baza=64, mnozniki=(1, 2, 3, 4), bloki=2, uwaga_na=(22,), rozmiar=176, drop=0.0):
        super().__init__()
        self.cfg = dict(kanaly=kanaly, baza=baza, mnozniki=list(mnozniki), bloki=bloki, uwaga_na=list(uwaga_na),
                        rozmiar=rozmiar, drop=drop)
        self.baza, td = baza, baza * 4
        self.czas = nn.Sequential(nn.Linear(baza, td), nn.SiLU(), nn.Linear(td, td))
        self.wej = nn.Conv2d(kanaly, baza, 3, padding=1)
        self.dol, kan, c, r = nn.ModuleList(), [baza], baza, rozmiar
        for i, m in enumerate(mnozniki):
            for _ in range(bloki):
                w = [BlokRes(c, baza * m, td, drop)]; c = baza * m
                if r in uwaga_na:
                    w.append(Uwaga(c))
                self.dol.append(nn.ModuleList(w)); kan.append(c)
            if i < len(mnozniki) - 1:
                self.dol.append(nn.ModuleList([nn.Conv2d(c, c, 3, 2, 1)])); kan.append(c); r //= 2
        self.srodek = nn.ModuleList([BlokRes(c, c, td, drop), Uwaga(c), BlokRes(c, c, td, drop)])
        self.gora = nn.ModuleList()
        for i, m in reversed(list(enumerate(mnozniki))):
            for j in range(bloki + 1):
                w = [BlokRes(c + kan.pop(), baza * m, td, drop)]; c = baza * m
                if r in uwaga_na:
                    w.append(Uwaga(c))
                if i > 0 and j == bloki:
                    w.append(WGore(c)); r *= 2
                self.gora.append(nn.ModuleList(w))
        self.wyj = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, kanaly, 3, padding=1))
        nn.init.zeros_(self.wyj[-1].weight); nn.init.zeros_(self.wyj[-1].bias)

    @staticmethod
    def _przez(w, h, te):
        for m in w:
            h = m(h, te) if isinstance(m, BlokRes) else m(h)
        return h

    def forward(self, x, t):
        te = self.czas(osadz_czas(t, self.baza))
        h = self.wej(x); hs = [h]
        for w in self.dol:
            h = self._przez(w, h, te); hs.append(h)
        h = self._przez(self.srodek, h, te)
        for w in self.gora:
            h = self._przez(w, torch.cat([h, hs.pop()], 1), te)
        return self.wyj(h)


class Harmonogram:
    """ᾱ_t dla t = 0..T-1 (harmonogram kosinusowy)."""

    def __init__(self, T=T_KROKI, dev='cpu', s=0.008):
        t = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        beta = (1 - (f[1:] / f[0]) / (f[:-1] / f[0])).clamp(max=0.999)
        self.ab = torch.cumprod(1 - beta, 0).float().to(dev)
        self.T = T

    def a(self, t):
        """ᾱ dla indeksu t (tensor); t = -1 oznacza obraz czysty (ᾱ = 1)."""
        t = torch.as_tensor(t, device=self.ab.device)
        return torch.where(t >= 0, self.ab[t.clamp(min=0)], torch.ones_like(t, dtype=self.ab.dtype))

    def zaszum(self, x0, t, eps):
        a = self.a(t).view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def x0(self, xt, t, eps):
        a = self.a(t).view(-1, 1, 1, 1)
        return (xt - (1 - a).sqrt() * eps) / a.sqrt()

    def eps(self, xt, t, x0):
        a = self.a(t).view(-1, 1, 1, 1)
        return (xt - a.sqrt() * x0) / (1 - a).sqrt()

    def krok_ddim(self, xt, t, t_prev, eps, x0, eta=0.0, gen=None):
        """x_{t_prev} z przewidzianego szumu i obrazu czystego (DDIM; eta = 0 bez losowości)."""
        a, ap = self.a(t).view(-1, 1, 1, 1), self.a(t_prev).view(-1, 1, 1, 1)
        sig = eta * ((1 - ap) / (1 - a) * (1 - a / ap)).clamp(min=0).sqrt()
        x = ap.sqrt() * x0 + (1 - ap - sig ** 2).clamp(min=0).sqrt() * eps
        if eta > 0:
            x = x + sig * torch.randn(x.shape, device=x.device, generator=gen)
        return x

    def cofnij(self, x_prev, t_prev, t, gen=None):
        """Podróż w czasie (FreeDoM): ponowne zaszumienie z t_prev z powrotem do t."""
        a, ap = self.a(t).view(-1, 1, 1, 1), self.a(t_prev).view(-1, 1, 1, 1)
        r = a / ap
        return r.sqrt() * x_prev + (1 - r).sqrt() * torch.randn(x_prev.shape, device=x_prev.device, generator=gen)


def siatka_krokow(T, kroki):
    return torch.linspace(T - 1, 0, kroki).round().long().tolist()


@torch.no_grad()
def probkuj(model, harm, n, kroki=50, eta=0.0, seed=0, dev='cuda', rozmiar=176, kanaly=4):
    """Próbki bez prowadzenia (podgląd uczenia); wynik w [-1, 1]."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((n, kanaly, rozmiar, rozmiar), device=dev, generator=g)
    ts = siatka_krokow(harm.T, kroki)
    for i, t in enumerate(ts):
        tp = ts[i + 1] if i + 1 < len(ts) else -1
        tt = torch.full((n,), t, device=dev, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = model(x, tt).float()
        x0 = harm.x0(x, tt, e).clamp(-1, 1)
        e = harm.eps(x, tt, x0)
        x = harm.krok_ddim(x, tt, torch.full_like(tt, tp), e, x0, eta, g)
    return x


class EMA:
    def __init__(self, model, zanik):
        self.m = copy.deepcopy(model).eval().requires_grad_(False)
        self.zanik = zanik

    @torch.no_grad()
    def aktualizuj(self, model, krok):
        d = min(self.zanik, (1 + krok) / (10 + krok))
        for pe, p in zip(self.m.parameters(), model.parameters()):
            pe.lerp_(p.detach(), 1 - d)
        for be, b in zip(self.m.buffers(), model.buffers()):
            be.copy_(b)


def laduj(sciezka, dev='cuda', ema=True):
    """Model z pliku zapisanego przez trenuj.py (klucze: cfg, ema, model)."""
    p = torch.load(sciezka, map_location='cpu', weights_only=False)
    m = UNet(**p['cfg'])
    m.load_state_dict(p['ema' if ema and 'ema' in p else 'model'])
    return m.to(dev).eval().requires_grad_(False), Harmonogram(dev=dev), p
