#!/usr/bin/env python3
"""Równanie Langevina dla skalara tożsamości narzędzia: dryf Koopmana (D), jądro pamięci (JP), wymuszenie
harmonogramem uczenia (W), szum kolorowy AR(q) na dawnych resztach (N). Sterowanie (S) w tym biegu stałe, więc
wchodzi w wyraz wolny; etap 2 w PLAN.md. Plan i bramka: PLAN.md.

Dane tylko do odczytu: ../koopman_tozsamosc_kierunki/wyniki/KIERUNKI_K160_S{S}.npz (a: 72 obrazy x 160 kroków),
OPERATORY_K160_S{S}.npz (odniesienie L4), tokeny_pelne/CONFIG.json (podział fit/holdout).
Zapis: wyniki/REPORT_S{S}.json.

    py -3.11 langevin.py --s-fit 48
    py -3.11 langevin.py --s-fit 96
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'koopman_tozsamosc_kierunki' / 'wyniki'
K = 160
T0 = 8                                   # pierwszy indeks przejścia w dopasowaniu (maks. opóźnienie)
LS = [1, 2, 3, 4, 5, 6, 8]
LAMBDY = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
QS = [0, 1, 2]
KAL = 20


def wczytaj(S):
    kier = np.load(SRC / f'KIERUNKI_K{K}_S{S}.npz'); op = np.load(SRC / f'OPERATORY_K{K}_S{S}.npz')
    rows = json.loads((SRC / 'tokeny_pelne' / 'CONFIG.json').read_text(encoding='utf-8'))['rows']
    fit = np.array([r['role'] == 'fit' for r in rows])
    return kier['a'].astype(float), kier['steps'].astype(float), fit, op


def cechy(a, V, t, steps, fit, cfg):
    """Cechy przejścia t -> t+1 z informacji dostępnej w t, pogrupowane w człony D, JP, W (72 x n)."""
    v = V[:, t - 1]
    cz = {'D': [v]}
    if cfg['poz']:
        cz['D'].append(a[:, t] - a[fit, t].mean())
    cz['JP'] = [V[:, t - l] for l in range(2, cfg['L'] + 1)]
    cz['W'] = []
    if cfg['W']:
        s = steps[t]; one = np.ones(a.shape[0])
        cz['W'] = [one * (s % 100 == 0), one * (s % 100 == 5), one * np.sin(2 * np.pi * s / 100),
                   one * np.cos(2 * np.pi * s / 100), one * (5.0 / s), v * (5.0 / s), v * (s % 100 == 0)]
    return {k: (np.stack(x, 1) if x else np.zeros((a.shape[0], 0))) for k, x in cz.items()}


def wektor(c):
    return np.concatenate([c['D'], c['JP'], c['W']], 1)


class Grzbiet:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-12
        self.my = y.mean()
        Z = (X - self.mx) / self.sx
        self.w = np.linalg.solve(Z.T @ Z + self.lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - self.my))
        return self

    def __call__(self, X):
        return self.my + ((X - self.mx) / self.sx) @ self.w


class Rownanie:
    """D + JP + W (grzbiet, wspólny dla obrazów fit) i N = AR(q) na resztach bazowych (drugi etap)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, a, V, ts, steps, fit):
        X, y = [], []
        for t in ts:
            X.append(wektor(cechy(a, V, t, steps, fit, self.cfg))[fit]); y.append(V[fit, t])
        self.baza = Grzbiet(self.cfg['lam']).fit(np.concatenate(X), np.concatenate(y))
        self.phi = None
        q = self.cfg['q']
        if q > 0:
            r = self.reszty(a, V, steps, fit)
            Xr, yr = [], []
            for t in ts:
                if t - q < T0:
                    continue
                Xr.append(np.stack([r[fit, t - l] for l in range(1, q + 1)], 1)); yr.append(r[fit, t])
            Xr, yr = np.concatenate(Xr), np.concatenate(yr)
            self.phi = np.linalg.solve(Xr.T @ Xr + 1e-6 * len(yr) * np.eye(q), Xr.T @ yr)
        return self

    def baza_krok(self, a, V, t, steps, fit):
        return self.baza(wektor(cechy(a, V, t, steps, fit, self.cfg)))

    def reszty(self, a, V, steps, fit):
        r = np.zeros_like(V)
        for t in range(T0, V.shape[1]):
            r[:, t] = V[:, t] - self.baza_krok(a, V, t, steps, fit)
        return r

    def jeden_krok(self, a, V, ts, steps, fit):
        """Prognoza przyrostu v[t] z prawdziwej historii (przewidywanie, potem obserwacja reszty)."""
        r = self.reszty(a, V, steps, fit) if self.phi is not None else None
        P = np.zeros((a.shape[0], len(ts)))
        for j, t in enumerate(ts):
            p = self.baza_krok(a, V, t, steps, fit)
            if self.phi is not None:
                p = p + np.stack([r[:, t - l] for l in range(1, len(self.phi) + 1)], 1) @ self.phi
            P[:, j] = p
        return P

    def rollout(self, a, V, S, h, steps, fit):
        """Od S przez h kroków bez prawdziwych stanów pośrednich; nieznane reszty zastępuje ich prognoza AR."""
        aa, VV = a.copy(), V.copy()
        rb = None
        if self.phi is not None:
            r = self.reszty(a, V, steps, fit)
            rb = [r[:, S - l] for l in range(len(self.phi), 0, -1)]          # najstarsza pierwsza
        for t in range(S, S + h):
            p = self.baza_krok(aa, VV, t, steps, fit)
            if rb is not None:
                ost = np.stack([rb[-l] for l in range(1, len(self.phi) + 1)], 1)
                rh = ost @ self.phi
                p = p + rh; rb.append(rh)
            VV[:, t] = p; aa[:, t + 1] = aa[:, t] + p
        return aa[:, S + h]


class L4:
    """Zapisany operator odniesienia: wspólna macierz na 4 ostatnich przyrostach (sekcje 2b i 5)."""

    def __init__(self, op):
        self.A, self.mx, self.my = (op['skalar_lda__wspolny_vel_L4__' + k] for k in ('A', 'mx', 'my'))

    def krok(self, V, t):
        Xq = np.stack([V[:, t - l] for l in range(1, 5)], 1)                  # najnowszy pierwszy
        return (self.my + (Xq - self.mx) @ self.A)[:, 0]

    def jeden_krok(self, a, V, ts, steps, fit):
        return np.stack([self.krok(V, t) for t in ts], 1)

    def rollout(self, a, V, S, h, steps, fit):
        aa, VV = a.copy(), V.copy()
        for t in range(S, S + h):
            p = self.krok(VV, t); VV[:, t] = p; aa[:, t + 1] = aa[:, t] + p
        return aa[:, S + h]


def oceny(model, a, V, S, h, steps, fit, rows, ts1):
    P = model.jeden_krok(a, V, ts1, steps, fit)
    e1 = P[rows] - V[rows][:, ts1]
    jeden = float(np.sqrt(np.mean(e1 ** 2) / np.mean(V[rows][:, ts1] ** 2)))
    ah = model.rollout(a, V, S, h, steps, fit)
    roll = float(np.sqrt(np.mean((ah[rows] - a[rows, S + h]) ** 2) / np.mean((a[rows, S] - a[rows, S + h]) ** 2)))
    return jeden, roll, e1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--s-fit', type=int, default=48)
    args = ap.parse_args(argv)
    S = args.s_fit
    a, steps, fit, op = wczytaj(S)
    V = np.diff(a, axis=1)
    hold = ~fit
    Sk = S - KAL
    # 1) wybór na kalibracji: dopasowanie do Sk, ocena na obrazach fit (przejścia Sk..S-1 i rollout Sk -> S)
    siatka = []
    for L, lam, poz, W, q in itertools.product(LS, LAMBDY, (False, True), (False, True), QS):
        cfg = dict(L=L, lam=lam, poz=poz, W=W, q=q)
        m = Rownanie(cfg).fit(a, V, range(T0, Sk), steps, fit)
        j, r, _ = oceny(m, a, V, Sk, KAL, steps, fit, fit, list(range(Sk, S)))
        siatka.append((r, j, cfg))
    siatka.sort(key=lambda x: (x[0], x[1]))
    najl = siatka[0][2]
    # 2) test: dopasowanie do S, ocena na obrazach holdout
    ts1 = list(range(S, K - 1))
    modele = {'langevin': Rownanie(najl).fit(a, V, range(T0, S), steps, fit), 'L4_zapisany': L4(op)}
    abl = {'bez_poz': dict(najl, poz=False), 'bez_JP': dict(najl, L=1), 'bez_W': dict(najl, W=False),
           'bez_N': dict(najl, q=0), 'tylko_D_v': dict(najl, poz=False, L=1, W=False, q=0)}
    for k, c in abl.items():
        modele[k] = Rownanie(c).fit(a, V, range(T0, S), steps, fit)
    horyzonty = sorted(set([h for h in (10, 20, 40, 80) if S + h <= K - 1] + [K - 1 - S]))
    wyn, e_lang = {}, None
    for k, m in modele.items():
        d = {}
        for h in horyzonty:
            j, r, e = oceny(m, a, V, S, h, steps, fit, hold, ts1)
            d[f'h{h}'] = r; d['jeden_krok'] = j
            if k == 'langevin':
                e_lang = e
        wyn[k] = d
    # 3) białość reszt holdout (jeden krok): autokorelacja i korelacja z granicą epoki
    ac = [float(np.mean([np.corrcoef(x[:-l], x[l:])[0, 1] for x in e_lang])) for l in range(1, 6)]
    b0 = np.array([steps[t] % 100 == 0 for t in ts1], float)
    kor_W = float(np.corrcoef(e_lang.mean(0), b0)[0, 1])
    L4r = wyn['L4_zapisany']
    bramka = dict(roll_h20_wzgl_L4=wyn['langevin']['h20'] / L4r['h20'],
                  jeden_krok_wzgl_L4=wyn['langevin']['jeden_krok'] / L4r['jeden_krok'])
    bramka['spelniona_w_tym_oknie'] = bool(bramka['roll_h20_wzgl_L4'] <= 0.95 and bramka['jeden_krok_wzgl_L4'] <= 1.02)
    raport = dict(S=S, krok_S=int(steps[S]), wybrana_konfiguracja=najl,
                  top5_kalibracja=[dict(rollout=r, jeden=j, cfg=c) for r, j, c in siatka[:5]],
                  wyniki_holdout_rmse_wzgl_braku_zmian=wyn, autokorelacja_reszt_opozn_1_5=ac,
                  korelacja_reszt_z_granica_epoki=kor_W, bramka=bramka,
                  uwaga='Sterowanie S(u) stałe w biegu 1810662, wchodzi w wyraz wolny; etap 2 w PLAN.md.')
    (HERE / 'wyniki' / f'REPORT_S{S}.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'S={S} (krok {int(steps[S])}), wybrano {najl}')
    for k, d in wyn.items():
        print(f'  {k:12s} ' + '  '.join(f'{kk} {v:.3f}' for kk, v in d.items()))
    print('  autokorelacja reszt', [round(x, 3) for x in ac], 'kor. z granicą epoki', round(kor_W, 3))
    print('  bramka', bramka)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
