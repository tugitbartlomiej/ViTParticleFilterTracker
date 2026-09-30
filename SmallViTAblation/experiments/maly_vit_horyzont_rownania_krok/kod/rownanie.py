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


class Kontrola:
    """Zbiór kontrolny (klatki odłożonych filmów treningowych): strata w float32, jej gradient i strata po
    wirtualnym kroku. Tryb ewaluacji (bufory BatchNorm z uczenia), bez odbicia."""

    def __init__(self, zb):
        self.zb = zb
        self.idx = np.arange(len(zb))

    def _strata(self, u, par):
        net = u.net; net.eval()
        suma, n = 0.0, 0
        buf = dict(net.named_buffers())
        for s in range(0, len(self.idx), KAWALEK):
            i = self.idx[s:s + KAWALEK]
            x = _wejscie(self.zb.X[i], u.dev)
            heat, ltrb, poz = (t.to(u.dev) for t in _cele(self.zb, i))
            o, _ = functional_call(net, (par, buf), (x,))
            L, _ = M.strata(o, heat, ltrb, poz)
            suma = suma + L * len(i); n += len(i)
        return suma / n

    @torch.no_grad()
    def strata(self, u, d=None):
        par = _parametry(u.net)
        if d is not None:
            par = {k: v + d[k] for k, v in par.items()}
        return float(self._strata(u, par))

    def gradient(self, u):
        """Strata i jej gradient po kawałkach (graf jednego kawałka naraz, żeby zmieścić się w pamięci karty)."""
        net = u.net; net.eval()
        par = _parametry(net); buf = dict(net.named_buffers())
        n_all = len(self.idx)
        suma = 0.0
        acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in par.items()}
        for s in range(0, n_all, KAWALEK):
            i = self.idx[s:s + KAWALEK]
            x = _wejscie(self.zb.X[i], u.dev)
            heat, ltrb, poz = (t.to(u.dev) for t in _cele(self.zb, i))
            o, _ = functional_call(net, (par, buf), (x,))
            L, _ = M.strata(o, heat, ltrb, poz)
            w = len(i) / n_all
            g = torch.autograd.grad(L * w, list(par.values()), allow_unused=True)
            for (n, _p), gg in zip(par.items(), g):
                if gg is not None:
                    acc[n] += gg.float()
            suma += float(L) * w
        return suma, acc


def zysk_pierwszy_rzad(gk, d):
    """Przewidziany spadek straty kontrolnej po kroku d (dodatni = poprawa): −⟨∇L, d⟩."""
    return float(-sum((gk[n] * d[n]).sum() for n in d))


def ocen_grupe(u, zb, idx, gk):
    """Wirtualny krok grupy: zysk pierwszego rzędu dla klatek i ich odbić (rozrzut = ocena szumu prognozy)."""
    z = []
    for odb in (False, True):
        g = gradient_paczki(u, zb, idx, odb)
        d, _ = czesci_kroku(u, g)
        z.append(zysk_pierwszy_rzad(gk, d))
    return dict(zysk=float(np.mean(z)), szum=float(abs(z[0] - z[1]) / 2))


def zysk_bezposredni(u, kontrola, zb, idx, L0=None):
    """Zysk z wirtualnego kroku policzony wprost: L(θ) − L(θ + d_u) (sprawdzenie przybliżenia pierwszego rzędu)."""
    d, _ = czesci_kroku(u, gradient_paczki(u, zb, idx))
    L0 = kontrola.strata(u) if L0 is None else L0
    return float(L0 - kontrola.strata(u, d))


@torch.no_grad()
def latent_narzedzia_sond(u, sondy, d=None):
    """Latent narzędzia 192 sond (średnia po tokenach narzędzia), float32, przy wagach θ (+ d). Jak `z_fp32` w
    `maly_vit_rownanie_latentu/nagranie.py`: tryb ewaluacji, te same bufory, bez kopiowania sieci."""
    X, tn = sondy[0], sondy[1]
    net = u.net; net.eval(); net.zapisuj_latent(True)
    par = _parametry(net)
    if d is not None:
        par = {k: v + d[k] for k, v in par.items()}
    buf = dict(net.named_buffers())
    out = []
    for s in range(0, len(X), 96):
        functional_call(net, (par, buf), (X[s:s + 96].to(u.dev),))
        a = tn[s:s + 96].to(u.dev).float()
        lat = net._latent.float()
        out.append((lat * a[..., None]).sum(1) / a.sum(1, keepdim=True).clamp(min=1))
    net.zapisuj_latent(False)
    return torch.cat(out).cpu().numpy()


def sila_tozsamosci_grupy(u, zb, idx, sondy, obecn, wek, z0):
    """Siła paczki S(u) z równania ruchu, rzutowana na wektor tożsamości: średnia zmiana siły tożsamości 180 sond z
    narzędziem po wirtualnym kroku grupy (diagnostyka; nie wchodzi do wyboru)."""
    import sila as SI
    d, _ = czesci_kroku(u, gradient_paczki(u, zb, idx))
    z1 = latent_narzedzia_sond(u, sondy, d)
    return float((SI.sila(z1[obecn], wek) - SI.sila(z0[obecn], wek)).mean())


def dryf_pedu(u, kontrola, gk, L0=None):
    """Co zrobi sam pęd optymalizatora w następnym kroku (prognoza bez przyszłych klatek)."""
    _, ped = czesci_kroku(u, None)
    L0 = kontrola.strata(u) if L0 is None else L0
    return dict(pierwszy_rzad=zysk_pierwszy_rzad(gk, ped), bezposrednio=float(L0 - kontrola.strata(u, ped)))
