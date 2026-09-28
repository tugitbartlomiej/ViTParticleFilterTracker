#!/usr/bin/env python3
"""Krok 2b (PLAN.md): wyuczone kierunki edycji FastGAN-a dla głównych kierunków kroku Koopmana.

Kroki Koopmana leżą głównie w kilku kierunkach latentu (PCA banku kroków z chwil innych niż krok 1000). Dla każdego
z k kierunków u_i uczymy JEDNĄ, wspólną dla wszystkich obrazów edycję generatora E_i = (δz_i, poprawka cech 8×8,
poprawka cech 16×16), tak żeby dodanie Σ a_i·E_i przesuwało latent obrazu o ≈ Σ a_i·u_i. To liniowy odpowiednik
sieci sterującej: mało parametrów, uczenie stabilniejsze niż MLP zależna od obrazu.
Ocena na 12 celach Koopmana z kroku 1000: a = Uᵀ·krok, potem dokręcenie gradientem (200 kroków D) i porównanie z samym
dokręceniem.

    <venv>/python kierunki.py [--kroki 1500 --k 16]  ->  wyniki/KIERUNKI.pt, KIERUNKI.json, KIERUNKI_koopman.png
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn as nn

import diagnoza as DG
import wspolne as W
from diagnoza2 import miary
from mlp import bank_krokow, dokrec


class Kierunki(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.Ez = nn.Parameter(torch.zeros(k, W.NZ))                        # start: brak edycji
        self.E8 = nn.Parameter(torch.zeros(k, 512, 8, 8))
        self.E16 = nn.Parameter(torch.zeros(k, 256, 16, 16))

    def forward(self, a):
        return a @ self.Ez, torch.einsum('bk,kchw->bchw', a, self.E8), torch.einsum('bk,kchw->bchw', a, self.E16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--kroki', type=int, default=1500)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--paczka', type=int, default=12)
    a_ = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    t0 = time.time()
    net, G = W.laduj_vit(), W.laduj_G()
    DG.W_G = G
    c = W.cele_koopmana()
    mu, sd = c['mu'], c['sd']
    bank, n_pr, _ = bank_krokow(mu, sd)
    kop = bank[n_pr:]                                                       # kroki Koopmana (bez chwili 1000)
    Uall = torch.linalg.svd(kop - kop.mean(0), full_matrices=False)[2]       # (768,768) wiersze = kierunki
    U = Uall[:a_.k].T.contiguous()                                          # (768,k)
    sw = torch.linalg.svdvals(kop - kop.mean(0)) ** 2
    teraz, cel = c['teraz'][c['kotwice']], c['cel'][c['kotwice']]
    krok_eval = cel - teraz
    pokrycie = float(((krok_eval @ U) ** 2).sum() / (krok_eval ** 2).sum())
    print(f'k={a_.k}: wariancja kroków Koopmana w k kierunkach {float(sw[:a_.k].sum() / sw.sum()):.3f}; '
          f'część kroków oceny (krok 1000) w tych kierunkach {pokrycie:.3f} ({time.time() - t0:.0f} s)', flush=True)
    skala = (kop @ U).std(0)                                                # typowa wielkość współczynników
    E = Kierunki(a_.k).to(W.DEV)
    opt = torch.optim.Adam(E.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a_.kroki, eta_min=3e-5)
    log = []
    for k in range(1, a_.kroki + 1):
        z0 = torch.randn(a_.paczka, W.NZ, device=W.DEV)
        with torch.no_grad():
            o0 = W.generuj_rgba(G, z0)
            w0 = W.tokeny_narzedzia(o0)
            l0 = W.latent_na_tokenach(net, o0, w0, mu, sd)
        ok = (w0.sum(1) >= 4).float()
        a = torch.randn(a_.paczka, a_.k, device=W.DEV) * skala * torch.empty(a_.paczka, 1, device=W.DEV).uniform_(.5, 2.)
        dz, d8, d16 = E(a)
        o = W.generuj_rgba(G, z0 + dz, d8, d16=d16)
        lat = W.latent_na_tokenach(net, o, w0, mu, sd)
        dl = a @ U.T
        # błąd względem wielkości kroku, normowany na całej paczce (bez ucinania: gradient zawsze płynie)
        Lg = ((lat - l0 - dl) ** 2).sum(1) / (dl ** 2).sum(1)
        L = (((lat - l0 - dl) ** 2).sum(1) * ok).sum() / ((dl ** 2).sum(1) * ok).sum().clamp(min=1e-6)             + 1e-4 * (E.E8.pow(2).mean() + E.E16.pow(2).mean())
        opt.zero_grad(set_to_none=True); L.backward(); opt.step(); sched.step()
        if k % 50 == 0:
            r = dict(krok=k, blad_wzgl=float((Lg * ok).sum() / ok.sum()), s=round(time.time() - t0))
            log.append(r); print(r, flush=True)
    torch.save(dict(E=E.state_dict(), U=U), W.WYN / 'KIERUNKI.pt')
    # --- ocena na 12 celach Koopmana z kroku 1000
    kot = c['obrazy']
    zs, _ = DG.inwersja(G, kot)
    with torch.no_grad():
        o0 = W.generuj_rgba(G, zs)
        w0 = W.tokeny_narzedzia(o0)
        l0 = W.latent_na_tokenach(net, o0, w0, mu, sd)
        tgt = l0 + krok_eval
        dz, d8, d16 = E(krok_eval @ U)
        o_e = W.generuj_rgba(G, zs + dz, d8, d16=d16)
        l_e = W.latent_na_tokenach(net, o_e, w0, mu, sd)
    o_er, l_er = dokrec(net, G, zs, dz, d8, d16, tgt, mu, sd, w0, 200)
    o_r, l_r = dokrec(net, G, zs, torch.zeros_like(dz), torch.zeros_like(d8), torch.zeros_like(d16), tgt, mu, sd, w0, 200)
    wyn = dict(k=a_.k, pokrycie_krokow_oceny=pokrycie, log=log, same_kierunki=miary(l_e, l0, tgt),
               kierunki_plus_dokrecenie_200=miary(l_er, l0, tgt), samo_dokrecenie_200=miary(l_r, l0, tgt))
    for kl, o in (('same_kierunki', o_e), ('kierunki_plus_dokrecenie_200', o_er), ('samo_dokrecenie_200', o_r)):
        wyn[kl]['zmiana_obrazu_l1'] = float((o[:, :3] - o0[:, :3]).abs().mean())
        x = wyn[kl]
        print(f"{kl:30s} postęp {x['postep_mediana']:.2f}  wzdłuż {x['wzdluz_mediana']:.2f}  w bok {x['bok_mediana']:.2f}  "
              f"zmiana obrazu {x['zmiana_obrazu_l1']:.3f}", flush=True)
    (W.WYN / 'KIERUNKI.json').write_text(json.dumps(wyn, indent=1), encoding='utf-8')
    W.zapisz_siatke(W.WYN / 'KIERUNKI_koopman.png', [kot, o0, o_e, o_er, o_r])
    print(f'KONIEC ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
