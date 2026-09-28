#!/usr/bin/env python3
"""Krok 3 (PLAN.md, po przeglądzie literatury): sterowanie FastGAN-em do kroku Koopmana z ochroną przed
„niewidocznymi” rozwiązaniami.

Przegląd (NOTATKA_LITERATURA.md): Rousselot i in. 2026 („Show Me What You Don't Know”) pokazują, że dopasowanie
reprezentacji sieci gradientem w pikselach daje szum w stylu ataku, a sensowne próbki wymagają priorytetu obrazów
(sterowany model dyfuzyjny); Väth i in. 2025 — gradient sieci nieodpornej trzeba stabilizować; Nguyen i in. 2016 —
optymalizacja wejścia generatora jako priorytetu obrazów. Tu, bez nowego generatora:
  E  latent liczony jako średnia z K lekko zaburzonych kopii obrazu (jasność/kontrast, szum, rozmycie) — zmiana musi
     przetrwać zaburzenia, więc drobny szum w stylu ataku przestaje wystarczać;
  R  kara dyskryminatora FastGAN-a (hinge: −D(obraz)) — obraz ma zostać w rozkładzie wycinków narzędzia.
Warianty (400 kroków, z + cechy 8×8 + 16×16 jak D w kroku 1): D0 (bez E i R), D_E, D_ER, oraz P_E (piksele z E).
Miary: postęp czysty i odporny (latent na zaburzonych kopiach), część kroku wzdłuż kierunku, zmiana obrazu, wynik D.

    <venv>/python sterowanie3.py  ->  wyniki/STEROWANIE3.json, wyniki/STEROWANIE3.png
"""
from __future__ import annotations

import importlib.util
import json
import time

import torch
import torch.nn.functional as F

import diagnoza as DG
import wspolne as W
from diagnoza2 import miary

K_ZAB = 4


def laduj_D():
    p = torch.load(W.GN.CKPT, map_location='cpu', weights_only=True)
    spec = importlib.util.spec_from_file_location('fastgan_models_vendor_d', W.GN.VENDOR)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    D = mod.Discriminator(ndf=64, nc=4, im_size=256)
    D.load_state_dict(p['d'], strict=True)
    return D.to(W.DEV).eval().requires_grad_(False)


def wynik_D(D, rgba):
    """Średni wynik dyskryminatora na obraz (wyżej = bardziej „prawdziwy”)."""
    x = rgba * 2 - 1
    B = x.shape[0]
    out = D(x, 'fake')
    per = out.numel() // B
    # rf_0 i rf_1 są sklejone po całej paczce: najpierw B·a wartości rf_0 (a pól na obraz), potem rf_1
    a = _pola_rf0(D, x)
    rf0 = out[:B * a].view(B, a).mean(1)
    rf1 = out[B * a:].view(B, per - a).mean(1)
    return (rf0 + rf1) / 2


_A = {}


def _pola_rf0(D, x):
    if 'a' not in _A:
        with torch.no_grad():
            f = D.down_from_big(F.interpolate(x[:1], size=D.im_size))
            f2 = D.down_4(f); f3 = D.down_8(f2); f4 = D.se_2_16(f, D.down_16(f3))
            f5 = D.se_4_32(f2, D.down_32(f4)); f6 = D.se_8_64(f3, D.down_64(f5))
            _A['a'] = D.rf_big(f6).numel()
    return _A['a']


def zaburz(rgba, g):
    """K zaburzonych kopii RGB (jasność/kontrast ±15 %, szum σ 0,02, lekkie rozmycie); maska bez zmian."""
    B = rgba.shape[0]
    x = rgba[:, :3].repeat(K_ZAB, 1, 1, 1)
    a = torch.empty(B * K_ZAB, 2, device=W.DEV).uniform_(.85, 1.15, generator=g)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * a[:, :1, None, None] + m * a[:, 1:, None, None]
    x = x + .02 * torch.randn(x.shape, device=W.DEV, generator=g)
    k = torch.tensor([1., 2., 1.], device=W.DEV); k = (k[:, None] * k[None]) / 16
    x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), k.expand(3, 1, 3, 3), groups=3)
    return torch.cat([x.clamp(0, 1), rgba[:, 3:4].repeat(K_ZAB, 1, 1, 1)], 1)


def latent_odporny(net, rgba, w0, mu, sd, g):
    B = rgba.shape[0]
    lat = W.latent_na_tokenach(net, zaburz(rgba, g), w0.repeat(K_ZAB, 1), mu, sd)
    return lat.view(K_ZAB, B, -1).mean(0)


def steruj(net, G, D, z0, o0, tgt, mu, sd, w0, kroki, piksele=False, odporny=False, realizm=0.0, lr=.01):
    B = z0.shape[0]
    g = torch.Generator(device=W.DEV).manual_seed(0)
    if piksele:
        dx = torch.zeros(B, 3, 256, 256, device=W.DEV, requires_grad=True); par = [dx]; lr = .005
    else:
        z = z0.clone().requires_grad_(True)
        d8 = torch.zeros(B, 512, 8, 8, device=W.DEV, requires_grad=True)
        d16 = torch.zeros(B, 256, 16, 16, device=W.DEV, requires_grad=True)
        par = [z, d8, d16]
    opt = torch.optim.Adam(par, lr=lr)
    najl = torch.full((B,), 1e9, device=W.DEV); stan = [p.detach().clone() for p in par]

    def obraz(pp):
        if piksele:
            return torch.cat([(o0[:, :3] + pp[0]).clamp(0, 1), o0[:, 3:4]], 1)
        return W.generuj_rgba(G, pp[0], pp[1], d16=pp[2])

    for _ in range(kroki):
        o = obraz(par)
        lat = latent_odporny(net, o, w0, mu, sd, g) if odporny else W.latent_na_tokenach(net, o, w0, mu, sd)
        Llat = ((lat - tgt) ** 2).mean(1)
        L = Llat
        if piksele:
            L = L + 1e-2 * par[0].abs().mean((1, 2, 3))
        else:
            L = L + 1e-4 * (par[1] ** 2).mean((1, 2, 3)) + 1e-4 * (par[2] ** 2).mean((1, 2, 3))
        if realizm:
            L = L - realizm * wynik_D(D, o)
        with torch.no_grad():
            lep = Llat < najl
            najl = torch.where(lep, Llat, najl)
            stan = [torch.where(lep.view(-1, *[1] * (p.dim() - 1)), p.detach(), s) for p, s in zip(par, stan)]
        opt.zero_grad(); L.sum().backward(); opt.step()
    with torch.no_grad():
        return obraz(stan)


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    net, G = W.laduj_vit(), W.laduj_G()
    D = laduj_D()
    DG.W_G = G
    c = W.cele_koopmana()
    mu, sd, kot = c['mu'], c['sd'], c['obrazy']
    krok = c['cel'][c['kotwice']] - c['teraz'][c['kotwice']]
    zs, _ = DG.inwersja(G, kot)
    g_ocena = torch.Generator(device=W.DEV).manual_seed(123)
    with torch.no_grad():
        o0 = W.generuj_rgba(G, zs)
        w0 = W.tokeny_narzedzia(o0)
        l0 = W.latent_na_tokenach(net, o0, w0, mu, sd)
        l0r = latent_odporny(net, o0, w0, mu, sd, g_ocena)
        d0 = wynik_D(D, o0)
    tgt, tgt_r = l0 + krok, l0r + krok
    print(f'start: wynik D inwersji {d0.mean():.3f}, wynik D kotwic {wynik_D(D, kot).mean():.3f} ({time.time() - t0:.0f} s)', flush=True)
    wyn, obr = {}, [kot, o0]
    for nazwa, kw in (('D0_bez_ochrony', dict()), ('D_E_odporny', dict(odporny=True)),
                      ('D_ER_odporny_realizm', dict(odporny=True, realizm=.05)), ('P_E_piksele_odporny', dict(piksele=True, odporny=True))):
        o = steruj(net, G, D, zs, o0, tgt if not kw.get('odporny') else tgt_r, mu, sd, w0, 400, **kw)
        with torch.no_grad():
            lat = W.latent_na_tokenach(net, o, w0, mu, sd)
            latr = latent_odporny(net, o, w0, mu, sd, torch.Generator(device=W.DEV).manual_seed(123))
            dd = wynik_D(D, o)
        m = dict(czysty=miary(lat, l0, tgt), odporny=miary(latr, l0r, tgt_r),
                 zmiana_obrazu_l1=float((o[:, :3] - o0[:, :3]).abs().mean()), wynik_D=float(dd.mean()), wynik_D_start=float(d0.mean()))
        wyn[nazwa] = m
        obr.append(o)
        print(f"{nazwa:24s} postęp czysty {m['czysty']['postep_mediana']:.2f} (wzdłuż {m['czysty']['wzdluz_mediana']:.2f}) | "
              f"odporny {m['odporny']['postep_mediana']:.2f} (wzdłuż {m['odporny']['wzdluz_mediana']:.2f}, w bok {m['odporny']['bok_mediana']:.2f}) | "
              f"zmiana obrazu {m['zmiana_obrazu_l1']:.3f} | D {m['wynik_D']:.3f} (start {m['wynik_D_start']:.3f})  ({time.time() - t0:.0f} s)", flush=True)
    W.zapisz_siatke(W.WYN / 'STEROWANIE3.png', obr)
    (W.WYN / 'STEROWANIE3.json').write_text(json.dumps(wyn, indent=1), encoding='utf-8')
    print(f'KONIEC ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
