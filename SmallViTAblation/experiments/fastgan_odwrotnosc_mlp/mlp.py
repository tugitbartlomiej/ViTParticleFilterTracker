#!/usr/bin/env python3
"""Krok 2 (PLAN.md, wersja po kroku 1): sieć sterująca MLP — latent obrazu startowego l0 i zadany krok Δ
→ poprawka wejścia FastGAN-a δz (256) + poprawki cech 8×8 i 16×16 (współczynniki 2 × K wyuczonych map).

Uczenie: starty z losowego z, kroki Δ z banku prawdziwych zmian latentu sond (v1 s42, horyzont 500 kroków) i prognoz
Koopmana w chwilach innych niż krok 1000; strata ‖latent(G(z0+δz; poprawki)) − (l0+Δ)‖²/‖Δ‖² na stałych tokenach
narzędzia obrazu startowego + małe kary. Ocena na 12 celach Koopmana z kroku 1000 (wyłączonych z uczenia).

    <venv>/python mlp.py [--kroki 2000]  ->  wyniki/MLP.pt, MLP.json, MLP_koopman.png, mlp.log
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

import diagnoza as DG
import wspolne as W
from diagnoza2 import miary

K = 64


class SiecSterujaca(nn.Module):
    def __init__(self, d=768, h=1024, k=K):
        super().__init__()
        self.tulow = nn.Sequential(nn.Linear(2 * d + 1, h), nn.LayerNorm(h), nn.GELU(),
                                   nn.Linear(h, h), nn.LayerNorm(h), nn.GELU())
        self.gz, self.g8, self.g16 = nn.Linear(h, W.NZ), nn.Linear(h, k), nn.Linear(h, k)
        for g in (self.gz, self.g8, self.g16):
            nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)
        self.B8 = nn.Parameter(torch.randn(k, 512, 8, 8) * .02)
        self.B16 = nn.Parameter(torch.randn(k, 256, 16, 16) * .02)

    def forward(self, l0, dl):
        r = dl.pow(2).mean(1, keepdim=True).sqrt() + 1e-6                 # wielkość kroku (RMS)
        h = self.tulow(torch.cat([l0 / 3, dl / r, r.log()], 1))
        dz = r * self.gz(h)
        d8 = torch.einsum('bk,kchw->bchw', r * self.g8(h), self.B8)
        d16 = torch.einsum('bk,kchw->bchw', r * self.g16(h), self.B16)
        return dz, d8, d16


def bank_krokow(mu, sd):
    """Kroki Δ (N,768) w z-jednostkach kroku 1000: prawdziwe zmiany latentu sond na 10 bloków (500 kroków) oraz kroki
    prognoz Koopmana z chwil 500–4000 bez 1000 (v1 s42, rozgrzewka + ramię bez)."""
    import json as _j
    man = _j.loads((W.PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    pl = [W.BIEG / 'wspolny' / f'k0{k}' / 'LATENT.npz' for k in (1, 2, 3)] + [W.BIEG / 'bez' / f'k0{k}' / 'LATENT.npz' for k in range(4, 10)]
    L = np.concatenate([np.load(p)['latent'][:, obecn, 0] for p in pl])[1:]
    Lb = L[:len(L) // 5 * 5].astype(np.float32).reshape(-1, 5, *L.shape[1:]).mean(1)       # (90,180,768)
    Z = (Lb - mu.cpu().numpy()) / sd.cpu().numpy()
    prawd = (Z[10:] - Z[:-10]).reshape(-1, 768)
    koop = []
    for t in range(500, 4001, 250):
        if t == 1000:
            continue
        b0 = t // 50 - 1
        h = Z[:b0 + 1].astype(np.float64)
        koop.append(W.PR2.prognoza_bloku(h, 10)['koopman'] - h[-1])
    koop = np.concatenate(koop)
    return torch.tensor(np.concatenate([prawd, koop]), dtype=torch.float32, device=W.DEV), len(prawd), len(koop)


def dokrec(net, G, z0, dz, d8, d16, tgt, mu, sd, w0, kroki, lr=.01):
    """Dokręcenie gradientem (jak wariant D) od podanego punktu; najlepszy stan przebiegu."""
    B = z0.shape[0]
    z = (z0 + dz).detach().clone().requires_grad_(True)
    a8, a16 = d8.detach().clone().requires_grad_(True), d16.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z, a8, a16], lr=lr)
    najl = torch.full((B,), 1e9, device=W.DEV); st = [z.detach().clone(), a8.detach().clone(), a16.detach().clone()]
    for _ in range(kroki):
        o = W.generuj_rgba(G, z, a8, d16=a16)
        Llat = ((W.latent_na_tokenach(net, o, w0, mu, sd) - tgt) ** 2).mean(1)
        with torch.no_grad():
            lep = Llat < najl
            najl = torch.where(lep, Llat, najl)
            st = [torch.where(lep.view(-1, *[1] * (x.dim() - 1)), x.detach(), s) for x, s in zip((z, a8, a16), st)]
        L = Llat + 1e-4 * (a8 ** 2).mean((1, 2, 3)) + 1e-4 * (a16 ** 2).mean((1, 2, 3))
        opt.zero_grad(); L.sum().backward(); opt.step()
    with torch.no_grad():
        o = W.generuj_rgba(G, st[0], st[1], d16=st[2])
        return o, W.latent_na_tokenach(net, o, w0, mu, sd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--kroki', type=int, default=2000)
    ap.add_argument('--paczka', type=int, default=12)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    t0 = time.time()
    net, G = W.laduj_vit(), W.laduj_G()
    DG.W_G = G
    c = W.cele_koopmana()
    mu, sd = c['mu'], c['sd']
    bank, n_pr, n_ko = bank_krokow(mu, sd)
    print(f'bank kroków: {n_pr} prawdziwych zmian + {n_ko} kroków Koopmana; RMS mediana '
          f'{bank.pow(2).mean(1).sqrt().median():.3f} ({time.time() - t0:.0f} s)', flush=True)
    f = SiecSterujaca().to(W.DEV)
    opt = torch.optim.AdamW(f.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=a.kroki, pct_start=.1)
    log = []
    for k in range(1, a.kroki + 1):
        z0 = torch.randn(a.paczka, W.NZ, device=W.DEV)
        with torch.no_grad():
            o0 = W.generuj_rgba(G, z0)
            w0 = W.tokeny_narzedzia(o0)
            l0 = W.latent_na_tokenach(net, o0, w0, mu, sd)
        ok = (w0.sum(1) >= 4).float()                                    # min. 4 tokeny narzędzia
        # 70 % kroków z prognoz Koopmana (skala jak w użyciu), 30 % prawdziwych zmian przeskalowanych do RMS 0,2–1,0
        ik = torch.randint(n_pr, len(bank), (a.paczka,), device=W.DEV)
        ip = torch.randint(0, n_pr, (a.paczka,), device=W.DEV)
        dp = bank[ip]; dp = dp / dp.pow(2).mean(1, keepdim=True).sqrt() * torch.empty(a.paczka, 1, device=W.DEV).uniform_(.2, 1.)
        wyb = (torch.rand(a.paczka, 1, device=W.DEV) < .7).float()
        dl = wyb * bank[ik] * torch.empty(a.paczka, 1, device=W.DEV).uniform_(.5, 2.) + (1 - wyb) * dp
        dz, d8, d16 = f(l0, dl)
        o = W.generuj_rgba(G, z0 + dz, d8, d16=d16)
        lat = W.latent_na_tokenach(net, o, w0, mu, sd)
        Lg = (((lat - l0 - dl) ** 2).sum(1) / (dl ** 2).sum(1)).clamp(max=10)   # 1 = jak brak ruchu, 0 = trafienie
        L = (Lg * ok).sum() / ok.sum().clamp(min=1) + 1e-3 * (dz ** 2).mean() + 1e-4 * ((d8 ** 2).mean() + (d16 ** 2).mean())
        opt.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
        opt.step(); sched.step()
        if k % 50 == 0:
            r = dict(krok=k, blad_wzgl=float((Lg * ok).sum() / ok.sum()), s=round(time.time() - t0))
            log.append(r); print(r, flush=True)
    torch.save(f.state_dict(), W.WYN / 'MLP.pt')
    # --- ocena na 12 celach Koopmana z kroku 1000
    kot = c['obrazy']
    teraz, cel = c['teraz'][c['kotwice']], c['cel'][c['kotwice']]
    zs, _ = DG.inwersja(G, kot)
    with torch.no_grad():
        o0 = W.generuj_rgba(G, zs)
        w0 = W.tokeny_narzedzia(o0)
        l0 = W.latent_na_tokenach(net, o0, w0, mu, sd)
        dl = cel - teraz
        tgt = l0 + dl
        dz, d8, d16 = f(l0, dl)
        o_m = W.generuj_rgba(G, zs + dz, d8, d16=d16)
        l_m = W.latent_na_tokenach(net, o_m, w0, mu, sd)
    o_mr, l_mr = dokrec(net, G, zs, dz, d8, d16, tgt, mu, sd, w0, 200)
    zero8, zero16 = torch.zeros_like(d8), torch.zeros_like(d16)
    o_r, l_r = dokrec(net, G, zs, torch.zeros_like(dz), zero8, zero16, tgt, mu, sd, w0, 200)
    wyn = dict(log=log, bank=dict(prawdziwe=n_pr, koopman=n_ko),
               sam_mlp=miary(l_m, l0, tgt), mlp_plus_dokrecenie_200=miary(l_mr, l0, tgt), samo_dokrecenie_200=miary(l_r, l0, tgt))
    for kl in ('sam_mlp', 'mlp_plus_dokrecenie_200', 'samo_dokrecenie_200'):
        wyn[kl]['zmiana_obrazu_l1'] = float({'sam_mlp': o_m, 'mlp_plus_dokrecenie_200': o_mr, 'samo_dokrecenie_200': o_r}[kl][:, :3].sub(o0[:, :3]).abs().mean())
    (W.WYN / 'MLP.json').write_text(json.dumps(wyn, indent=1), encoding='utf-8')
    W.zapisz_siatke(W.WYN / 'MLP_koopman.png', [kot, o0, o_m, o_mr, o_r])
    for kl in ('sam_mlp', 'mlp_plus_dokrecenie_200', 'samo_dokrecenie_200'):
        x = wyn[kl]
        print(f"{kl:26s} postęp {x['postep_mediana']:.2f}  wzdłuż {x['wzdluz_mediana']:.2f}  w bok {x['bok_mediana']:.2f}  "
              f"zmiana obrazu {x['zmiana_obrazu_l1']:.3f}", flush=True)
    print(f'KONIEC ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
