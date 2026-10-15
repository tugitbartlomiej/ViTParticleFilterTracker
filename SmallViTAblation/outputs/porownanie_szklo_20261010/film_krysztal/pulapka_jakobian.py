"""Pułapka koloidalna z Jakobianu bloku MLP 6: czy 768 wiązek trzyma cząsteczkę (siła zwrotna), czy ją obraca lub wypycha?

Blok MLP działa na token x jako pole sił F(x) = W₂ GELU(W₁ LN(x) + b₁) + b₂, a x ← x + F(x). Linearyzacja wokół tokenu:
    J = ∂F/∂x = W₂ diag(GELU'(a)) W₁ · J_LN,   a = W₁ u + b₁,   u = LN(x),
    J_LN = diag(γ)/σ · (I − 11ᵀ/d − n nᵀ),     n = znormalizowany x̂ (LN usuwa średnią i długość: „sfera” LayerNorm).
Rozkład J = S + A (S = (J+Jᵀ)/2: część potencjalna, sztywność pułapki; A = (J−Jᵀ)/2: wir). Pułapka = ujemne wartości
własne S (siła zwrotna); ślad J < 0 = kontrakcja objętości (dywergencja pola). σ nieznane → J liczone z σ = 1 (skala, nie znak).

u_j odtwarzamy z zapisanych aktywacji z_j (768 po GELU, z wartościami ujemnymi): start z GELU⁻¹ i najmniejszych kwadratów, potem
Gauss–Newton na wszystkich 768 kanałach (jasne i ciemne); sprawdzian: R² odtworzenia wszystkich z_c; x̂ ∝ (u − β)/γ.
Uruchom: F:/Python/3.13/python.exe -X utf8 pulapka_jakobian.py   → PULAPKA_JAKOBIAN.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10'
WAGI = ASSETS / '01_bieg_35000_gesty_zapis/eden/gesty/wagi'
LATENT = ASSETS / '03_pomiar_naturalny_60_epok_eden/ruch60_rp128_50tr_50val/pelne768.npy'
KROKI = [0, 1000, 5300, 11000, 22974]
gelu = lambda a: 0.5 * a * (1 + erf(a / np.sqrt(2)))
dgelu = lambda a: 0.5 * (1 + erf(a / np.sqrt(2))) + a * np.exp(-a * a / 2) / np.sqrt(2 * np.pi)
_A = np.linspace(-0.75, 12, 40000); _Z = gelu(_A)                                 # GELU rosnąca dla a > −0,75
ginv = lambda z: np.interp(z, _Z, _A)


def odtworz_u(z, W1, b1, iters=12, lam=1e-3):
    """u z aktywacjami jasnych i ciemnych laserów: start z odwróconej GELU (gałąź a > −0,75), potem Gauss–Newton na ‖GELU(W₁u+b₁) − z‖²."""
    m = z > -0.12
    u, *_ = np.linalg.lstsq(W1[m], ginv(z[m]) - b1[m], rcond=None)
    for _ in range(iters):
        a = W1 @ u + b1
        r = gelu(a) - z
        Jg = dgelu(a)[:, None] * W1
        u -= np.linalg.solve(Jg.T @ Jg + lam * np.eye(len(u)), Jg.T @ r)
    return u


def main():
    import sys
    sys.path.insert(0, str(HERE))
    from lasery_wspolne import dane
    kroki_all = dane(0)['kroki'].astype(int)
    wynik = {}
    for k in KROKI:
        s = torch.load(WAGI / f'k{k:05d}.pt', map_location='cpu', weights_only=False)['net']
        g = lambda n: s[f'bloki.5.{n}'].double().numpy()
        W1, b1, W2, gam, bet = g('mlp.fc1.weight'), g('mlp.fc1.bias'), g('mlp.fc2.weight'), g('norm2.weight'), g('norm2.bias')
        Zt = np.asarray(np.load(LATENT, mmap_mode='r')[int(np.searchsorted(kroki_all, k)), 0]).astype(np.float64)   # (600, 768)
        d = W1.shape[1]
        r2, neg, tr, rot, lmin, lmax, res = [], [], [], [], [], [], []
        for j in range(Zt.shape[0]):
            z = Zt[j]
            u = odtworz_u(z, W1, b1)
            a = W1 @ u + b1
            r2.append(1 - ((gelu(a) - z) ** 2).sum() / ((z - z.mean()) ** 2).sum())
            xh = (u - bet) / gam; xh -= xh.mean(); n = xh / np.linalg.norm(xh)
            Pln = np.eye(d) - np.ones((d, d)) / d - np.outer(n, n)
            J = W2 @ (dgelu(a)[:, None] * W1) @ (gam[:, None] * Pln)
            # część działająca w podprzestrzeni, którą LN „widzi” (bez kierunku 1 i n)
            Q = np.linalg.svd(Pln)[0][:, :d - 2]
            Jq = Q.T @ J @ Q
            S, A = (Jq + Jq.T) / 2, (Jq - Jq.T) / 2
            ev = np.linalg.eigvalsh(S)
            neg.append((ev < 0).mean()); tr.append(np.trace(Jq) / (d - 2)); lmin.append(ev[0]); lmax.append(ev[-1])
            rot.append(np.linalg.norm(A) / max(np.linalg.norm(S), 1e-12))
            res.append(np.linalg.eigvals(Jq).real.max())
        f = lambda x: dict(med=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))
        wynik[str(k)] = dict(R2_odtworzenia_z=f(r2), udzial_ujemnych_wartosci_S=f(neg), sredni_slad_J=f(tr),
                             wir_do_potencjalu_A_S=f(rot), lambda_min_S=f(lmin), lambda_max_S=f(lmax), max_Re_wartosci_J=f(res))
        print(k, {kk: round(v['med'], 3) for kk, v in wynik[str(k)].items()}, flush=True)
    (HERE / 'PULAPKA_JAKOBIAN.json').write_text(json.dumps(dict(opis=(__doc__ or '').split('\n\n')[0], kroki=KROKI, wyniki=wynik),
                                                           ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
