"""Równanie ruchu latentu jako model skutku klatek (PLAN.md, wersja 4; docs/23_METODA_PRZEWIDYWANIA_LATENTU.md).

Latent i strata poruszają się jak cząstka z bezwładnością popychana przez paczkę danych:
    Δz(t+1) = a·Δz(t) + b·S(t),   S(u) = z(θ + d_u) − z(θ),   b ≈ 1 (zmierzone na 10 ziarnach),
gdzie d_u to część kroku AdamW pochodząca z gradientu paczki u (bez pędu i bez zaniku wag), liczona przy obecnym
stanie optymalizatora. Ten sam wirtualny krok oceniony na stracie zbioru kontrolnego daje przewidziany skutek klatek
dla detekcji (T5 w dokumencie 44): zysk(u) = L(θ) − L(θ + d_u) ≈ −⟨∇L, d_u⟩.

Nic tu nie zmienia sieci ani optymalizatora: gradient kandydatów liczony przez torch.func.functional_call na kopii
buforów BatchNorm, bez .grad i bez opt.step(). Reguła o przyszłych klatkach: używane są tylko obecne wagi, obecny
stan optymalizatora i kandydaci; prawdziwy skutek po uczeniu służy wyłącznie do oceny prognozy.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call

import model as M

KAWALEK = 64


def _parametry(net):
    return {n: p for n, p in net.named_parameters()}


def _bufory_kopia(net):
    return {n: b.detach().clone() for n, b in net.named_buffers()}


def _wejscie(X_u8, dev, odbicie=False):
    x = torch.from_numpy(np.ascontiguousarray(X_u8)).to(dev).permute(0, 3, 1, 2).float() / 255.
    return x.flip(3) if odbicie else x


def _cele(zb, idx, odbicie=False):
    c = zb.cele[1 if odbicie else 0]
    it = torch.as_tensor(idx)
    return c[0][it], c[1][it], c[2][it]


def gradient_paczki(u, zb, idx, odbicie=False):
    """Gradient straty treningowej na klatkach zb[idx] w trybie uczenia (bf16 jak w treningu), obcięty do normy 1 jak
    w `Uczen.kroki`. Zwraca dict nazwa -> tensor float32. Bufory BatchNorm sieci pozostają nietknięte."""
    x = _wejscie(zb.X[idx], u.dev, odbicie)
    heat, ltrb, poz = (t.to(u.dev) for t in _cele(zb, idx, odbicie))
    return gradient_tensorow(u, x, heat, ltrb, poz)


def gradient_tensorow(u, x, heat, ltrb, poz):
    """Jak `gradient_paczki`, ale dla gotowej paczki (np. dokładnie tej z kroku uczenia, po odbiciu i jasności)."""
    net = u.net; net.train()
    par = _parametry(net)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o, _ = functional_call(net, (par, _bufory_kopia(net)), (x,))
    L, _ = M.strata(o, heat, ltrb, poz)
    g = torch.autograd.grad(L, list(par.values()), allow_unused=True)
    g = {n: (gg.float() if gg is not None else torch.zeros_like(p)) for (n, p), gg in zip(par.items(), g)}
    norma = torch.sqrt(sum((v * v).sum() for v in g.values()))
    if norma > 1.0:
        g = {n: v / (norma + 1e-6) for n, v in g.items()}
    return g


def czesci_kroku(u, g=None):
    """Części kroku AdamW jak w `maly_vit_rownanie_latentu/nagranie.py::kierunki`: 'nowa' (z gradientu g paczki) i
    'ped' (z pamięci optymalizatora). Mianownik z obecnego v i g (jak w torch.optim.AdamW)."""
    nazwy = {id(p): n for n, p in u.net.named_parameters()}
    nowa, ped = {}, {}
    lr = u._lr()
    for grp in u.opt.param_groups:
        (b1, b2), eps = grp['betas'], grp['eps']
        for p in grp['params']:
            n = nazwy[id(p)]
            st = u.opt.state.get(p, {})
            krok = float(st['step']) + 1 if 'step' in st else 1.0
            m = st['exp_avg'] if 'exp_avg' in st else torch.zeros_like(p)
            v = st['exp_avg_sq'] if 'exp_avg_sq' in st else torch.zeros_like(p)
            gg = g[n] if g is not None else torch.zeros_like(p)
            mian = (b2 * v + (1 - b2) * gg * gg).sqrt() / np.sqrt(1 - b2 ** krok) + eps
            nowa[n] = -(lr / (1 - b1 ** krok)) * (1 - b1) * gg / mian
            ped[n] = -(lr / (1 - b1 ** krok)) * b1 * m / mian
    return nowa, ped
