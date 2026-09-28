#!/usr/bin/env python3
"""Krok 4 (PLAN.md): FastGAN sterowany tylko wzdłuż wektora tożsamości narzędzia, o zmianę siły przewidzianą przez
Langevina (decyzja użytkownika 2026-09-28: prognoza dotyczy tożsamości, nie całego latentu).

Ustawienie (v1, ziarno 42, krok 1000 = 21 epok):
  - wektor tożsamości w (768): liniowy rozdzielacz narzędzie/tło na sondach z kroku 1000, jak w
    `maly_vit_wektor_tozsamosci/wektor_tozsamosci_21ep.py`; siła tożsamości s(x) = rzut latentu narzędzia na w;
  - prognoza: historia siły 180 sond z kroków 50–1000 (bloki po 5 zapisów), Langevin (i dla porównania Koopman)
    z `prognoza.py` wersji 2, horyzont 10 bloków (500 kroków) → przewidziana zmiana Δs dla 12 kotwic;
  - cel dla obrazu: s(start) + Δs, gdzie start = obraz FastGAN-a z inwersji kotwicy; reszta latentu (poza kierunkiem w)
    ma zostać jak w starcie (kara za zmianę „w bok”).
Warianty (300 kroków): Z_1x (samo wejście z), D_1x (z + cechy 8×8 i 16×16), D_1x_E (cel liczony na zaburzonych
kopiach obrazu), D_3x (trzy razy większa zmiana, żeby zobaczyć, jak wygląda na obrazie).
Miary: osiągnięta część zmiany siły (1 = dokładnie Δs), to samo na zaburzonych kopiach, zmiana latentu w bok
(względem |Δs|), zmiana obrazu L1.

    <venv>/python wzdluz_tozsamosci.py  ->  wyniki/WZDLUZ.json, wyniki/WZDLUZ.png
"""
from __future__ import annotations

import importlib.util
import json
import time

import numpy as np
import torch

import diagnoza as DG
import wspolne as W
from sterowanie3 import zaburz, K_ZAB


def _mod(nazwa, p):
    spec = importlib.util.spec_from_file_location(nazwa, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


WEK = _mod('wektor21', W.EXP / 'maly_vit_wektor_tozsamosci' / 'wektor_tozsamosci_21ep.py')


def wektor_i_prognoza():
    man = json.loads((W.PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    war = np.array([m['variant'] for m in man]); zrodlo = np.array([m['source_file'] for m in man])
    L = np.concatenate([np.load(W.BIEG / 'wspolny' / f'k0{k}' / 'LATENT.npz')['latent'] for k in (1, 2)])[1:]
    Lb = L[:len(L) // 5 * 5].astype(np.float64).reshape(-1, 5, *L.shape[1:]).mean(1)          # (20,192,2,768)
    m = WEK.naucz(*WEK.zbior(Lb[-1], obecn, war, zrodlo, np.ones(len(war), bool))[:2])
    S = ((Lb[:, obecn, 0] - m['mu']) / m['sd']) @ m['w']                                     # (20,180)
    Y = S[:, :, None]
    prog = {}
    for mod in ('langevin', 'koopman'):
        Wt = W.PR2.dopasuj(Y, mod)
        prog[mod] = W.PR2.rollout(Y, Wt, 10, mod)[-1, :, 0] - Y[-1, :, 0]
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=W.DEV)  # noqa: E731
    return dict(mu=t(m['mu']), sd=t(m['sd']), w=t(m['w']), S=S, dl=t(prog['langevin']), dk=t(prog['koopman']))


def sila_i_reszta(lat_raw, v):
    """Siła tożsamości s (B,) i latent w jednostkach wektora (B,768)."""
    x = (lat_raw - v['mu']) / v['sd']
    return x @ v['w'], x


def steruj(net, G, z0, w0, v, s_cel, x0, kroki, glebokie, odporny, skala_ds, lr=.01, bok=1.0):
    B = z0.shape[0]
    g = torch.Generator(device=W.DEV).manual_seed(0)
    zero, jeden = torch.zeros(768, device=W.DEV), torch.ones(768, device=W.DEV)
    z = z0.clone().requires_grad_(True); par = [z]
    d8 = d16 = None
    if glebokie:
        d8 = torch.zeros(B, 512, 8, 8, device=W.DEV, requires_grad=True)
        d16 = torch.zeros(B, 256, 16, 16, device=W.DEV, requires_grad=True)
        par += [d8, d16]
    opt = torch.optim.Adam(par, lr=lr)
    najl = torch.full((B,), 1e9, device=W.DEV); stan = [p.detach().clone() for p in par]
    for _ in range(kroki):
        o = W.generuj_rgba(G, z, d8, d16=d16)
        if odporny:
            raw = W.latent_na_tokenach(net, zaburz(o, g), w0.repeat(K_ZAB, 1), zero, jeden).view(K_ZAB, B, -1).mean(0)
        else:
            raw = W.latent_na_tokenach(net, o, w0, zero, jeden)
        s, x = sila_i_reszta(raw, v)
        dx = x - x0
        prost = dx - (dx @ v['w'])[:, None] * v['w'][None]                                  # zmiana poza kierunkiem w
        Ls = ((s - s_cel) / skala_ds) ** 2
        L = Ls + bok * (prost ** 2).mean(1) / skala_ds ** 2 / 50
        if glebokie:
            L = L + 1e-4 * ((d8 ** 2).mean((1, 2, 3)) + (d16 ** 2).mean((1, 2, 3)))
        with torch.no_grad():
            lep = Ls < najl
            najl = torch.where(lep, Ls, najl)
            stan = [torch.where(lep.view(-1, *[1] * (p.dim() - 1)), p.detach(), s_) for p, s_ in zip(par, stan)]
        opt.zero_grad(); L.sum().backward(); opt.step()
    with torch.no_grad():
        return W.generuj_rgba(G, stan[0], stan[1] if glebokie else None, d16=stan[2] if glebokie else None)


def miary_s(net, o, o0, w0, v, ds):
    zero, jeden = torch.zeros(768, device=W.DEV), torch.ones(768, device=W.DEV)
    g = torch.Generator(device=W.DEV).manual_seed(123)
    B = o.shape[0]
    with torch.no_grad():
        s0, x0 = sila_i_reszta(W.latent_na_tokenach(net, o0, w0, zero, jeden), v)
        s1, x1 = sila_i_reszta(W.latent_na_tokenach(net, o, w0, zero, jeden), v)
        r0 = W.latent_na_tokenach(net, zaburz(o0, g), w0.repeat(K_ZAB, 1), zero, jeden).view(K_ZAB, B, -1).mean(0)
        g = torch.Generator(device=W.DEV).manual_seed(123)
        r1 = W.latent_na_tokenach(net, zaburz(o, g), w0.repeat(K_ZAB, 1), zero, jeden).view(K_ZAB, B, -1).mean(0)
        sr0, _ = sila_i_reszta(r0, v); sr1, _ = sila_i_reszta(r1, v)
    dx = x1 - x0
    prost = dx - (dx @ v['w'])[:, None] * v['w'][None]
    czesc = (s1 - s0) / ds
    czesc_r = (sr1 - sr0) / ds
    bok = prost.norm(dim=1) / ds.abs()
    return dict(czesc_zmiany=czesc.tolist(), czesc_zmiany_mediana=float(czesc.median()),
                czesc_zmiany_odporna=czesc_r.tolist(), czesc_zmiany_odporna_mediana=float(czesc_r.median()),
                zgodny_znak_odporny=float((torch.sign(sr1 - sr0) == torch.sign(ds)).float().mean()),
                bok_wzgl_mediana=float(bok.median()), zmiana_obrazu_l1=float((o[:, :3] - o0[:, :3]).abs().mean()))


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    net, G = W.laduj_vit(), W.laduj_G()
    DG.W_G = G
    c = W.cele_koopmana()
    v = wektor_i_prognoza()
    kw = c['kotwice']
    ds = v['dl'][kw]                                                                      # przewidziana zmiana siły
    wyn = dict(sila_kotwic_teraz=v['S'][-1, kw].tolist(), ds_langevin=ds.tolist(), ds_koopman=v['dk'][kw].tolist(),
               ds_langevin_wszystkie_sondy_mediana=float(v['dl'].median()),
               ds_langevin_wszystkie_sondy_sredni_modul=float(v['dl'].abs().mean()))
    print(f"siła tożsamości kotwic ~{np.mean(wyn['sila_kotwic_teraz']):.2f}; przewidziana zmiana Langevina (kotwice) "
          f"{[round(x, 3) for x in wyn['ds_langevin']]}; Koopman {[round(x, 3) for x in wyn['ds_koopman']]} ({time.time() - t0:.0f} s)", flush=True)
    kot = c['obrazy']
    zs, _ = DG.inwersja(G, kot)
    zero, jeden = torch.zeros(768, device=W.DEV), torch.ones(768, device=W.DEV)
    with torch.no_grad():
        o0 = W.generuj_rgba(G, zs)
        w0 = W.tokeny_narzedzia(o0)
        s0, x0 = sila_i_reszta(W.latent_na_tokenach(net, o0, w0, zero, jeden), v)
        g = torch.Generator(device=W.DEV).manual_seed(0)
        r0 = W.latent_na_tokenach(net, zaburz(o0, g), w0.repeat(K_ZAB, 1), zero, jeden).view(K_ZAB, 12, -1).mean(0)
        s0r, x0r = sila_i_reszta(r0, v)
    wyn['sila_inwersji'] = s0.tolist()
    skala = ds.abs().clamp(min=.02)
    obr = [kot, o0]
    for nazwa, mnoz, gl, odp in (('Z_1x', 1, False, False), ('D_1x', 1, True, False), ('D_1x_E', 1, True, True), ('D_3x', 3, True, False)):
        cel = (s0r if odp else s0) + mnoz * ds
        o = steruj(net, G, zs, w0, v, cel, x0r if odp else x0, 300, gl, odp, skala * mnoz)
        m = miary_s(net, o, o0, w0, v, mnoz * ds)
        wyn[nazwa] = m
        obr.append(o)
        print(f"{nazwa:8s} część zmiany siły {m['czesc_zmiany_mediana']:.2f} | odporna {m['czesc_zmiany_odporna_mediana']:.2f} "
              f"(znak zgodny {m['zgodny_znak_odporny']:.2f}) | w bok {m['bok_wzgl_mediana']:.2f} | zmiana obrazu {m['zmiana_obrazu_l1']:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    W.zapisz_siatke(W.WYN / 'WZDLUZ.png', obr)
    (W.WYN / 'WZDLUZ.json').write_text(json.dumps(wyn, indent=1), encoding='utf-8')
    print(f'KONIEC ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
