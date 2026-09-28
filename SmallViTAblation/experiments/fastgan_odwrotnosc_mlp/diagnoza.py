#!/usr/bin/env python3
"""Krok 1 (PLAN.md): dlaczego FastGAN nie dochodzi do przewidzianego latentu. Warianty A/B/C na 12 celach Koopmana
i na 12 celach na pewno osiągalnych (latent innego obrazu z FastGAN-a).

    <venv>/python diagnoza.py  ->  wyniki/DIAGNOZA.json, wyniki/DIAGNOZA_koopman.png, wyniki/DIAGNOZA_osiagalne.png
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

import wspolne as W


def inwersja(G, obrazy, starty=4, kroki=160):
    """z*, dla którego G(z*) odtwarza kotwice (L1 obrazu + L1 maski); najlepszy z kilku startów, osobno dla próbki."""
    B = obrazy.shape[0]
    najl_z, najl_L = None, torch.full((B,), 1e9, device=W.DEV)
    for r in range(starty):
        z = torch.randn(B, W.NZ, generator=torch.Generator().manual_seed(1000 + r)).to(W.DEV).requires_grad_(True)
        opt = torch.optim.Adam([z], lr=.05)
        for it in range(kroki):
            o = W.generuj_rgba(W_G, z)
            L = (o - obrazy).abs().mean((1, 2, 3))
            opt.zero_grad(); L.sum().backward(); opt.step()
            if it == 99:
                for g in opt.param_groups:
                    g['lr'] = .01
        with torch.no_grad():
            L = (W.generuj_rgba(W_G, z) - obrazy).abs().mean((1, 2, 3))
            lepsze = L < najl_L
            najl_z = z.detach().clone() if najl_z is None else torch.where(lepsze[:, None], z.detach(), najl_z)
            najl_L = torch.minimum(L, najl_L)
    return najl_z, najl_L


def steruj(net, G, z0, cel, mu, sd, kroki, w0, m0, kotwica=None, z_blisko=0.0, glebokie=False, lr=.01, maska=1.0):
    """Sterowanie wejściem FastGAN-a do latentu `cel`, liczonego na STAŁYCH tokenach w0 obrazu startowego; kara za
    zmianę maski narzędzia (m0) trzyma narzędzie w miejscu. Zwraca najlepszy obraz przebiegu (dla każdej próbki)."""
    z = z0.clone().requires_grad_(True)
    par = [z]
    d8 = None
    if glebokie:
        d8 = torch.zeros(z0.shape[0], 512, 8, 8, device=W.DEV, requires_grad=True); par.append(d8)
    opt = torch.optim.Adam(par, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, kroki, eta_min=lr * .1)
    B = z0.shape[0]
    najl = torch.full((B,), 1e9, device=W.DEV); najl_z = z0.clone(); najl_d8 = None if d8 is None else d8.detach().clone()
    for _ in range(kroki):
        o = W.generuj_rgba(G, z, d8)
        lat = W.latent_na_tokenach(net, o, w0, mu, sd)
        Llat = ((lat - cel) ** 2).mean(1)
        L = Llat + maska * (o[:, 3:4] - m0).abs().mean((1, 2, 3))
        if kotwica is not None:
            L = L + (o[:, :3] - kotwica[:, :3]).abs().mean((1, 2, 3))
        if z_blisko:
            L = L + z_blisko * ((z - z0) ** 2).sum(1)
        if d8 is not None:
            L = L + 1e-4 * (d8 ** 2).mean((1, 2, 3))
        with torch.no_grad():
            lepsze = Llat < najl
            najl = torch.where(lepsze, Llat, najl)
            najl_z = torch.where(lepsze[:, None], z.detach(), najl_z)
            if d8 is not None:
                najl_d8 = torch.where(lepsze[:, None, None, None], d8.detach(), najl_d8)
        opt.zero_grad(); L.sum().backward(); opt.step(); sched.step()
    with torch.no_grad():
        o = W.generuj_rgba(G, najl_z, najl_d8)
        lat = W.latent_na_tokenach(net, o, w0, mu, sd)
        _, masa = W.latent_narzedzia(net, o, mu, sd)
    return o, lat, masa, najl_z


def main() -> int:
    global W_G
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    net, W_G = W.laduj_vit(), W.laduj_G()
    c = W.cele_koopmana()
    mu, sd = c['mu'], c['sd']
    kot = c['obrazy']
    with torch.no_grad():
        la, _ = W.latent_narzedzia(net, kot, mu, sd)                               # latent prawdziwych kotwic
    teraz = c['teraz'][c['kotwice']]; cel = c['cel'][c['kotwice']]
    wyn = dict(zgodnosc_latentu_kotwic=W.odl(la, teraz).tolist(), krok_prognozy=W.odl(teraz, cel).tolist())
    print(f'latent kotwic liczony tu vs zapis sond: {W.odl(la, teraz).mean():.3f}; krok prognozy Koopmana '
          f'{W.odl(teraz, cel).mean():.3f} ({time.time() - t0:.0f} s)', flush=True)
    zs, Linw = inwersja(W_G, kot)
    with torch.no_grad():
        o0 = W.generuj_rgba(W_G, zs)
        w0, m0 = W.tokeny_narzedzia(o0), o0[:, 3:4].clone()
        l0 = W.latent_na_tokenach(net, o0, w0, mu, sd)
    wyn['inwersja_l1'] = Linw.tolist()
    wyn['inwersja_latent_do_kotwicy'] = W.odl(l0, la).tolist()
    print(f'inwersja: L1 {Linw.mean():.3f}, latent obrazu z inwersji do latentu kotwicy {W.odl(l0, la).mean():.3f} '
          f'({time.time() - t0:.0f} s)', flush=True)
    # cele Koopmana. Prognoza dotyczy średniej 5 zapisów (bloku), a pojedynczy stan sieci w kroku 1000 różni się od
    # tej średniej o ~0,84 z-jednostki (więcej niż cały krok prognozy ~0,33), więc sensowny cel dla obrazu liczonego
    # siecią z kroku 1000 to: latent obrazu startowego + przewidziany KROK (cel − teraz). Wariant „cel_wprost”
    # (bezwzględny, jak w dawnych pętlach) tylko dla A, żeby odtworzyć dawną porażkę.
    for nazwa_celu, tgt, warianty in (
            ('krok_od_inwersji', l0 + (cel - teraz), (('A_obecne', dict(kroki=250, kotwica=kot, z_blisko=1e-3, lr=.02, maska=0.0)),
                                                      ('B_dlugie_bez_hamulcow', dict(kroki=400)),
                                                      ('C_glebokie', dict(kroki=400, glebokie=True)))),
            ('krok_x3', l0 + 3 * (cel - teraz), (('B_dlugie_bez_hamulcow', dict(kroki=400)),))):
        d0 = W.odl(l0, tgt)
        obr = [kot, o0]
        for wariant, kw in warianty:
            o, lat, masa, _ = steruj(net, W_G, zs, tgt, mu, sd, w0=w0, m0=m0, **kw)
            post = 1 - W.odl(lat, tgt) / d0
            wyn[f'{nazwa_celu}/{wariant}'] = dict(postep=post.tolist(), postep_mediana=float(post.median()),
                                                  l1_do_kotwicy=(o[:, :3] - kot[:, :3]).abs().mean((1, 2, 3)).tolist(),
                                                  masa_tokenow=masa.tolist())
            obr.append(o)
            print(f'{nazwa_celu:18s} {wariant:24s} postęp mediana {post.median():.2f} (min {post.min():.2f}, max '
                  f'{post.max():.2f}), odległość startu {d0.mean():.3f}  ({time.time() - t0:.0f} s)', flush=True)
        W.zapisz_siatke(W.WYN / f'DIAGNOZA_{nazwa_celu}.png', obr)
    # cele na pewno osiągalne: latent obrazu z innego losowego z
    g = torch.Generator().manual_seed(7)
    za, zb = torch.randn(12, W.NZ, generator=g).to(W.DEV), torch.randn(12, W.NZ, generator=g).to(W.DEV)
    with torch.no_grad():
        oa, ob = W.generuj_rgba(W_G, za), W.generuj_rgba(W_G, zb)
        wa, ma = W.tokeny_narzedzia(oa), oa[:, 3:4].clone()
        la_, lb_ = W.latent_na_tokenach(net, oa, wa, mu, sd), W.latent_na_tokenach(net, ob, wa, mu, sd)
    d0 = W.odl(la_, lb_)
    obr = [oa, ob]
    for wariant, kw in (('B_dlugie_bez_hamulcow', dict(kroki=400, maska=0.0)), ('C_glebokie', dict(kroki=400, glebokie=True, maska=0.0))):
        o, lat, _, _ = steruj(net, W_G, za, lb_, mu, sd, w0=wa, m0=ma, **kw)
        post = 1 - W.odl(lat, lb_) / d0
        wyn[f'osiagalne/{wariant}'] = dict(postep=post.tolist(), postep_mediana=float(post.median()))
        obr.append(o)
        print(f'osiągalne          {wariant:24s} postęp mediana {post.median():.2f}, odległość startu {d0.mean():.3f} '
              f'({time.time() - t0:.0f} s)', flush=True)
    W.zapisz_siatke(W.WYN / 'DIAGNOZA_osiagalne.png', obr)
    (W.WYN / 'DIAGNOZA.json').write_text(json.dumps(wyn, indent=1), encoding='utf-8')
    return 0


W_G = None

if __name__ == '__main__':
    raise SystemExit(main())
