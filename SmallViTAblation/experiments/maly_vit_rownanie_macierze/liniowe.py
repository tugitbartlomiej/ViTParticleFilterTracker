#!/usr/bin/env python3
"""Pełne macierze liniowe zamiast sieci: czy sprzężenie między 768 współrzędnymi latentu pomaga prognozie kroku.

Co rozstrzyga: równanie z uczonymi sieciami (uczenie.py) przegrywa z jedną liczbą bezwładności, bo wkuwa okres uczenia.
Tu te same dane i ten sam podział, ale człony są pełnymi macierzami 768 × 768 dopasowanymi grzbietowo (ridge),
bez przyszłych klatek:
  L1  Δz_{t+1} = A Δz_t
  L2  Δz_{t+1} = A Δz_t + B Δz_{t−1} + C Δz_{t−2}                 (pamięć trzech kroków)
  L3  Δz_{t+1} = A Δz_t + P S_pęd,t                               (siła pędu AdamW liczona siecią, tylko s47)
Odniesienia: skalar a Δz_t (bezwładność) i a Δz_t + c S_pęd,t. Obali: pełne macierze nie wyżej niż skalar.
Siła λ wybierana na krokach 1800–2000, test 2000–2999 na sondach fit i holdout. Wynik: wyniki/LINIOWE.json.
"""
from __future__ import annotations

import json

import numpy as np

import dane as DN

FIT, WYBOR, TEST = (1000, 1800), (1800, 2000), (2000, 2999)
LAMBDY = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def przyrosty(nag, t0, t1, sondy):
    """dz_k dla kroków t0−2..t1: v[i] = z_{t0−3+i+1} − z_{t0−3+i}."""
    z = nag.z[t0 - 3 - nag.od:t1 + 1 - nag.od, sondy].float().numpy()
    return z[1:] - z[:-1]                                   # v[0] = Δz_{t0−2}, …, v[-1] = Δz_{t1}


def cechy(nag, t0, t1, sondy, wariant):
    v = przyrosty(nag, t0, t1, sondy)
    y = v[3:]                                               # Δz_{t+1} dla t = t0..t1−1
    X = [v[2:-1]]                                           # Δz_t
    if wariant == 'L2':
        X += [v[1:-2], v[:-3]]
    if wariant == 'L3':
        X += [nag.sila(t0, t1, sondy, 'ped')]
    return np.concatenate(X, -1).reshape(-1, 768 * len(X)), y.reshape(-1, 768)


def tozs(nag, y):
    return (y / nag.tozs['sd']) @ nag.tozs['w']


def miary(nag, y, p, n_sond):
    ps, ys = p / nag.tozs['sd'], y / nag.tozs['sd']
    wy = ys.reshape(-1, n_sond, 768).mean(1); wp = ps.reshape(-1, n_sond, 768).mean(1)
    return dict(latent=round(DN.umiejetnosc(y, p), 2), tozsamosc=round(DN.umiejetnosc(tozs(nag, y), tozs(nag, p)), 2),
                wektor=round(DN.umiejetnosc(wy, wp), 2))


def main():
    wynik = {}
    for s in (47, 48):
        nag = DN.Nagranie(s)
        fit, hold = nag.sondy('fit'), nag.sondy('holdout')
        warianty = ('L1', 'L2', 'L3') if nag.sily_od <= FIT[0] else ('L1', 'L2')
        for w in warianty:
            X, y = cechy(nag, *FIT, fit, w)
            XtX, Xty = X.T.astype(np.float64) @ X, X.T.astype(np.float64) @ y
            skala = np.trace(XtX) / len(XtX)
            Xw, yw = cechy(nag, *WYBOR, fit, w)
            najl = None
            for lam in LAMBDY:
                M = np.linalg.solve(XtX + lam * skala * np.eye(len(XtX)), Xty)
                u = DN.umiejetnosc(yw, Xw @ M)
                if najl is None or u > najl[0]:
                    najl = (u, lam, M)
            _, lam, M = najl
            wynik[f's{s}/{w}'] = dict(lambda_=lam, wybor=round(najl[0], 2))
            for rola, sd in (('fit', fit), ('holdout', hold)):
                Xt, yt = cechy(nag, *TEST, sd, w)
                wynik[f's{s}/{w}'][rola] = miary(nag, yt, Xt @ M, len(sd))
            print(s, w, wynik[f's{s}/{w}'], flush=True)
        # skalar bezwładności na tych samych krokach (odniesienie wewnątrz ziarna)
        X, y = cechy(nag, *FIT, fit, 'L1')
        a = float((X * y).sum() / (X * X).sum())
        wynik[f's{s}/skalar_a'] = dict(a=round(a, 4))
        for rola, sd in (('fit', fit), ('holdout', hold)):
            Xt, yt = cechy(nag, *TEST, sd, 'L1')
            wynik[f's{s}/skalar_a'][rola] = miary(nag, yt, a * Xt, len(sd))
        print(s, 'skalar', wynik[f's{s}/skalar_a'], flush=True)
        del nag
    (DN.HERE / 'wyniki' / 'LINIOWE.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
