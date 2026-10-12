"""Czy „reflektor” (blok MLP 6) obraca się jako całość, czy każde lusterko (laser) przestawia się osobno?

Dla par checkpointów biegu 35 000 liczy:
- medianę cos kierunku tego samego lasera (v_c z fc2, w_c z fc1) między checkpointami,
- jaki udział zmiany wyjaśnia jeden wspólny obrót R przestrzeni 192D (Prokrustes: R·A ≈ B),
- ten sam udział dla zera: ta sama zmiana przetasowana między laserami,
- wzrost mediany długości wektorów.

Uruchom: F:/Python/3.13/python.exe -X utf8 sprawdz_obrot_reflektora.py   → OBROT_REFLEKTORA.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WAGI = HERE.parents[2] / 'assets/RuchCzasteczek_ViT_szklo_2026-10/01_bieg_35000_gesty_zapis/eden/gesty/wagi'
PARY = [('k00000', 'k01400'), ('k01400', 'k05300'), ('k05300', 'k11000'), ('k11000', 'k22974'), ('k00000', 'k22974')]


def wez(k):
    s = torch.load(WAGI / f'{k}.pt', map_location='cpu', weights_only=False)['net']
    return dict(v=s['bloki.5.mlp.fc2.weight'].float().numpy(),                # (192, 768): kolumny v_c
                w=s['bloki.5.mlp.fc1.weight'].float().numpy().T)              # (192, 768): kolumny w_c


def wspolny_obrot(A, B):
    U, _, Vt = np.linalg.svd(B @ A.T)
    return 1 - np.linalg.norm(B - U @ Vt @ A) ** 2 / np.linalg.norm(B - A) ** 2


def main():
    rng = np.random.default_rng(0)
    wynik = {}
    for a, b in PARY:
        A, B = wez(a), wez(b)
        for nm in ('v', 'w'):
            X, Y = A[nm], B[nm]
            xn, yn = X / np.linalg.norm(X, axis=0), Y / np.linalg.norm(Y, axis=0)
            zero = [wspolny_obrot(X, X + (Y - X)[:, rng.permutation(X.shape[1])]) for _ in range(3)]
            m = dict(cos_kierunku_med=float(np.median((xn * yn).sum(0))),
                     wspolny_obrot_udzial=float(wspolny_obrot(X, Y)),
                     wspolny_obrot_zero=float(np.mean(zero)),
                     wzrost_dlugosci=float(np.median(np.linalg.norm(Y, axis=0)) / np.median(np.linalg.norm(X, axis=0))))
            wynik[f'{a}->{b} {nm}'] = m
            print(f'{a}->{b} {nm}', {k: round(v, 3) for k, v in m.items()})
    (HERE / 'OBROT_REFLEKTORA.json').write_text(json.dumps(dict(opis=(__doc__ or '').split('\n\n')[0], wyniki=wynik),
                                                           ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
