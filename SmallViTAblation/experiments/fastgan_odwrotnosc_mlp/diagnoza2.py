#!/usr/bin/env python3
"""Krok 1b (PLAN.md): ile swobody trzeba, żeby przesunąć latent w kierunku Koopmana.

Cel: latent obrazu z inwersji kotwicy + krok Koopmana (jak w diagnoza.py), tokeny narzędzia stałe.
Warianty:
  C1  wejście z + przesunięcie cech 8×8, bez kary za maskę, 800 kroków;
  D   wejście z + cechy 8×8 + cechy 16×16, bez kary za maskę, 800 kroków;
  P   same piksele obrazu (górna granica: czy ten kierunek w ogóle da się uzyskać na obrazie), 400 kroków.
Miary: postęp (1 − odległość do celu / odległość startu), część kroku wykonana WZDŁUŻ kierunku Koopmana
(1 = pełny krok) i błąd w bok (odległość od prostej kroku, w jednostkach długości kroku), zmiana obrazu (L1).

    <venv>/python diagnoza2.py  ->  wyniki/DIAGNOZA2.json, wyniki/DIAGNOZA2.png
"""
from __future__ import annotations

import json
import time

import torch

import diagnoza as DG
import wspolne as W


def miary(lat, l0, tgt):
    s = tgt - l0
    d = lat - l0
    wzdl = (d * s).sum(1) / (s * s).sum(1)
    bok = (d - wzdl[:, None] * s).norm(dim=1) / s.norm(dim=1)
    post = 1 - W.odl(lat, tgt) / W.odl(l0, tgt)
    return dict(postep=post.tolist(), postep_mediana=float(post.median()), wzdluz=wzdl.tolist(),
                wzdluz_mediana=float(wzdl.median()), bok=bok.tolist(), bok_mediana=float(bok.median()))


def steruj_glebiej(net, G, z0, tgt, mu, sd, w0, kroki, d16_tez, lr=.01):
    B = z0.shape[0]
    z = z0.clone().requires_grad_(True)
    d8 = torch.zeros(B, 512, 8, 8, device=W.DEV, requires_grad=True)
    par = [z, d8]
    d16 = None
    if d16_tez:
        d16 = torch.zeros(B, 256, 16, 16, device=W.DEV, requires_grad=True); par.append(d16)
    opt = torch.optim.Adam(par, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, kroki, eta_min=lr * .1)
    najl = torch.full((B,), 1e9, device=W.DEV)
    stan = [z.detach().clone(), d8.detach().clone(), None if d16 is None else d16.detach().clone()]
    for _ in range(kroki):
        o = W.generuj_rgba(G, z, d8, d16=d16)
        Llat = ((W.latent_na_tokenach(net, o, w0, mu, sd) - tgt) ** 2).mean(1)
        L = Llat + 1e-4 * (d8 ** 2).mean((1, 2, 3)) + (0 if d16 is None else 1e-4 * (d16 ** 2).mean((1, 2, 3)))
        with torch.no_grad():
            lep = Llat < najl
            najl = torch.where(lep, Llat, najl)
            stan[0] = torch.where(lep[:, None], z.detach(), stan[0])
            stan[1] = torch.where(lep[:, None, None, None], d8.detach(), stan[1])
            if d16 is not None:
                stan[2] = torch.where(lep[:, None, None, None], d16.detach(), stan[2])
        opt.zero_grad(); L.sum().backward(); opt.step(); sched.step()
    with torch.no_grad():
        o = W.generuj_rgba(G, stan[0], stan[1], d16=stan[2])
        return o, W.latent_na_tokenach(net, o, w0, mu, sd)


def steruj_pikselami(net, o0, tgt, mu, sd, w0, kroki, lr=.005):
    B = o0.shape[0]
    dx = torch.zeros(B, 3, 256, 256, device=W.DEV, requires_grad=True)
    opt = torch.optim.Adam([dx], lr=lr)
    najl = torch.full((B,), 1e9, device=W.DEV); najl_dx = dx.detach().clone()
    for _ in range(kroki):
        o = torch.cat([(o0[:, :3] + dx).clamp(0, 1), o0[:, 3:4]], 1)
        Llat = ((W.latent_na_tokenach(net, o, w0, mu, sd) - tgt) ** 2).mean(1)
        L = Llat + 1e-2 * dx.abs().mean((1, 2, 3))
        with torch.no_grad():
            lep = Llat < najl
            najl = torch.where(lep, Llat, najl)
            najl_dx = torch.where(lep[:, None, None, None], dx.detach(), najl_dx)
        opt.zero_grad(); L.sum().backward(); opt.step()
    with torch.no_grad():
        o = torch.cat([(o0[:, :3] + najl_dx).clamp(0, 1), o0[:, 3:4]], 1)
        return o, W.latent_na_tokenach(net, o, w0, mu, sd)


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    net, G = W.laduj_vit(), W.laduj_G()
    DG.W_G = G
    c = W.cele_koopmana()
    mu, sd, kot = c['mu'], c['sd'], c['obrazy']
    teraz, cel = c['teraz'][c['kotwice']], c['cel'][c['kotwice']]
    zs, _ = DG.inwersja(G, kot)
    with torch.no_grad():
        o0 = W.generuj_rgba(G, zs)
        w0 = W.tokeny_narzedzia(o0)
        l0 = W.latent_na_tokenach(net, o0, w0, mu, sd)
    tgt = l0 + (cel - teraz)
    wyn, obr = {}, [kot, o0]
    for nazwa, fn in (('C1_z_i_cechy8', lambda: steruj_glebiej(net, G, zs, tgt, mu, sd, w0, 800, False)),
                      ('D_z_cechy8_cechy16', lambda: steruj_glebiej(net, G, zs, tgt, mu, sd, w0, 800, True)),
                      ('P_piksele_gorna_granica', lambda: steruj_pikselami(net, o0, tgt, mu, sd, w0, 400))):
        o, lat = fn()
        m = miary(lat, l0, tgt)
        m['zmiana_obrazu_l1'] = (o[:, :3] - o0[:, :3]).abs().mean((1, 2, 3)).tolist()
        wyn[nazwa] = m
        obr.append(o)
        print(f"{nazwa:26s} postęp {m['postep_mediana']:.2f}  wzdłuż kroku {m['wzdluz_mediana']:.2f}  w bok "
              f"{m['bok_mediana']:.2f}  zmiana obrazu {sum(m['zmiana_obrazu_l1']) / 12:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    W.zapisz_siatke(W.WYN / 'DIAGNOZA2.png', obr)
    (W.WYN / 'DIAGNOZA2.json').write_text(json.dumps(wyn, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
