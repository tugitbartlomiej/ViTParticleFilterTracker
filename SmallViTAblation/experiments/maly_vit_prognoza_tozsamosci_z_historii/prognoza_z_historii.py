#!/usr/bin/env python3
"""Prognoza siły tożsamości narzędzia na t+1, t+2, t+3, ... ze stanu i historii do chwili t (pytanie użytkownika
2026-09-30: „w chwili t mamy stan i historię; czy równanie Langevina z jądrem pamięci ma przewidywać latent tożsamości;
jaki jest błąd dla t+1, t+2, t+3; sprawdzaj różne równania Koopmana, Langevina, potem pomieszane z pędem i
przyspieszeniem; wykres błędu”).

Dane: nagrania uczenia małego ViT z latentem sond w KAŻDYM kroku (`../maly_vit_rownanie_latentu/nagranie_s47_sily`,
`nagranie_s48_sily`, tylko odczyt): siła tożsamości s_k(t) 180 sond z narzędziem (rzut latentu narzędzia na kierunek
LDA z kroku 1000, jak w `../maly_vit_tozsamosc_jeden_krok`), pęd optymalizatora S_pęd(t) i paczka S_paczka(t)
rzutowane tak samo. Wiersz t = stan po t krokach; S_pęd(t), S_paczka(t) prowadzą ze stanu t do t+1.

Co wolno w chwili t0: s(≤ t0), S_pęd(t0) (pamięć paczek do t0−1). Paczka kroku t0 tylko w wariancie „znane klatki”.
Oznaczenia: v(t) = s(t) − s(t−1) prędkość, a(t) = v(t) − v(t−1) przyspieszenie, x(t) = s(t) − m(t) odchylenie od
poziomu m(t) = średnia s z ostatnich 500 kroków (poziom równowagi z historii; w prognozie stały, m(t0)).

Modele (współczynniki wspólne dla 180 sond, dopasowane na s47, kroki 1000-1999; test: s47 i s48, starty 2000-2499 co 5
kroków, H do 500; s48 nie bierze udziału w dopasowaniu). Rekurencyjnie = równanie jednego kroku składane H razy:
  bezwładność            v(t+1) = a·v(t)
  Langevin OU            v(t+1) = a·v(t) − κ·x(t)                        (tarcie + sprężyna do poziomu)
  Langevin + jądro p     v(t+1) = Σ_j K_j·v(t−j) − κ·x(t), j < p          (jądro pamięci, p = 20 i 100)
  Koopman opóźnienia     x(t+1) = Σ_j A_j·x(t−j), j < 20                  (każda sonda, delay embedding)
  Koopman DMD            c(t+1) = A1·c(t) + A2·c(t−1), c = 20 modów PCA 180 sond razem
  + pęd                  dodatkowo b·S_pęd; w kolejnych krokach znana część pędu gaśnie jak β1^k = 0,9^k
  + przyspieszenie       dodatkowo g·a(t)
  wprost                 osobne dopasowanie Δs(t0 → t0+H) dla każdego H (bez składania kroków)
  znane klatki           krok t0 → t0+1 z równania S_pęd + S_paczka (paczka t0 znana), dalej Langevin + jądro 20 + pęd
Odniesienie (nie prognoza): równanie S_pęd + S_paczka liczone co krok na prawdziwych wagach i zsumowane.

Miara: błąd = RMSE prognozy przyrostu s(t0+H) − s(t0) / RMSE „braku zmiany”; umiejętność = 100·(1 − błąd²).

    python prognoza_z_historii.py   ->  wyniki/PROGNOZA_Z_HISTORII.json, wyniki/PROGNOZA_Z_HISTORII.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parent
NAGR = HERE.parent / 'maly_vit_rownanie_latentu'
OUT = HERE / 'wyniki'
H_LISTA = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500)
HMAX = max(H_LISTA)
OKNO_M = 500
FIT = (1000, 2000)                  # starty dopasowania (s47)
TEST = np.arange(2000, 2500, 5)     # starty testu (s47, s48)
BETA1 = 0.9
RIDGE = 1e-6


# ---------------------------------------------------------------- dane
def wektor(L, obecn):
    """Kierunek tożsamości jak w `../maly_vit_rownanie_macierze/dane.py::wektor_tozsamosci`."""
    X, y = [], []
    for i in range(len(L)):
        X.append(L[i, 0]); y.append(int(obecn[i]))
        X.append(L[i, 1]); y.append(0)
    X, y = np.array(X, np.float64), np.array(y)
    neg = X[y == 0]
    mu, sd = neg.mean(0), neg.std(0) + 1e-6
    w = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit((X - mu) / sd, y).coef_[0]
    return mu, sd, w / np.linalg.norm(w)


def seria(s):
    """s (T+1, 180), ped (T+1, 180), nowa (T+1, 180): NaN tam, gdzie siły nie nagrano."""
    f = OUT / f'seria_s{s}.npz'
    if f.exists():
        z = np.load(f)
        return z['s'], z['ped'], z['nowa']
    k = NAGR / f'nagranie_s{s}_sily'
    meta = json.loads((k / 'META.json').read_text(encoding='utf-8'))
    ob = np.array(meta['obecn']); od = meta['sily_od']
    L = np.load(k / 'sondy_lat.npy', mmap_mode='r')
    mu, sd, w = wektor(L[960:1001:10].astype(np.float64).mean(0), ob)
    wsd = (w / sd).astype(np.float32); mu32 = mu.astype(np.float32)
    S = np.concatenate([(L[a:a + 100][:, ob, 0].astype(np.float32) - mu32) @ wsd for a in range(0, len(L), 100)])
    out = {}
    for n in ('ped', 'nowa'):
        F = np.load(k / f'sila_{n}.npy', mmap_mode='r')
        P = np.concatenate([F[a:a + 100][:, ob].astype(np.float32) @ wsd for a in range(0, len(F), 100)])
        A = np.full_like(S, np.nan); A[od:od + len(P)] = P
        out[n] = A
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, s=S, ped=out['ped'], nowa=out['nowa'])
    return S, out['ped'], out['nowa']


def poziom(S):
    """m(t): średnia s z kroków t−499..t (na każdą sondę)."""
    c = np.cumsum(np.concatenate([np.zeros((1, S.shape[1])), S.astype(np.float64)]), 0)
    t = np.arange(len(S)); a = np.maximum(t - OKNO_M + 1, 0)
    return (c[t + 1] - c[a]) / (t + 1 - a)[:, None]


class Dane:
    def __init__(self, s):
        self.S, self.P, self.N = (x.astype(np.float64) for x in seria(s))
        self.M = poziom(self.S)
        self.V = np.diff(self.S, axis=0, prepend=self.S[:1])        # V[t] = s(t) − s(t−1)


# ---------------------------------------------------------------- cechy i dopasowanie
def cechy(d: Dane, t, spec):
    """Cechy w chwili t (wektor startów t) dla modelu przyrostu: (len(t), 180, F)."""
    t = np.asarray(t)
    f = [d.V[t - j] for j in range(spec.get('p', 0))]
    if spec.get('x'):
        f.append(d.S[t] - d.M[t])
    if spec.get('acc'):
        f.append(d.V[t] - d.V[t - 1])
    if spec.get('ped'):
        f.append(d.P[t])
    return np.stack(f, -1)


def lsq(X, y):
    X = X.reshape(-1, X.shape[-1]); y = y.reshape(-1)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    sc = X.std(0) + 1e-12
    A = (X / sc).T @ (X / sc) + RIDGE * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(A, (X / sc).T @ y) / sc


def dopasuj_przyrost(d, spec):
    t = np.arange(*FIT)
    return lsq(cechy(d, t, spec), d.S[t + 1] - d.S[t])


def prognoza_przyrost(d, t0, spec, coef, znane_klatki=False):
    """Rekurencja H kroków. Zwraca dict H -> przewidziany przyrost (len(t0), 180)."""
    p = spec.get('p', 0)
    vh = [d.V[t0 - j].copy() for j in range(max(p, 2))]
    x = d.S[t0] - d.M[t0]
    sp0 = d.P[t0].copy()
    cum = np.zeros_like(x); out = {}
    for k in range(HMAX):
        if znane_klatki and k == 0:
            vn = d.P[t0] + d.N[t0]
            sp0 = d.P[t0] + d.N[t0]                                # pęd po kroku t0 zawiera paczkę t0
        else:
            f = [vh[j] for j in range(p)]
            if spec.get('x'):
                f.append(x)
            if spec.get('acc'):
                f.append(vh[0] - vh[1])
            if spec.get('ped'):
                f.append(sp0 * BETA1 ** k)                       # znana część pędu po k krokach
            vn = np.stack(f, -1) @ coef
        cum = cum + vn; x = x + vn
        vh = [vn] + vh[:-1]
        if k + 1 in H_LISTA:
            out[k + 1] = cum.copy()
    return out


# Koopman na poziomach każdej sondy (opóźnienia)
def dopasuj_koopman_ar(d, p):
    t = np.arange(*FIT)
    X = np.stack([d.S[t - j] - d.M[t] for j in range(p)], -1)
    return lsq(X, d.S[t + 1] - d.M[t])


def prognoza_koopman_ar(d, t0, p, coef):
    m = d.M[t0]
    xh = [d.S[t0 - j] - m for j in range(p)]
    x0 = xh[0].copy(); out = {}
    for k in range(HMAX):
        xn = np.stack(xh, -1) @ coef
        xh = [xn] + xh[:-1]
        if k + 1 in H_LISTA:
            out[k + 1] = xn - x0
    return out


# Koopman DMD: 180 sond razem, r modów PCA, 2 opóźnienia (+ pęd jako wejście)
def dopasuj_dmd(d, r=20, ped=False):
    t = np.arange(*FIT)
    m = d.M[t]
    U = np.linalg.svd((d.S[t] - m) - (d.S[t] - m).mean(0), full_matrices=False)[2][:r].T   # (180, r)
    F = [(d.S[t] - m) @ U, (d.S[t - 1] - m) @ U] + ([d.P[t] @ U] if ped else [])
    F = np.concatenate(F, 1)
    Y = (d.S[t + 1] - m) @ U
    sc = F.std(0) + 1e-12
    A = (F / sc).T @ (F / sc) + RIDGE * len(F) * np.eye(F.shape[1])
    W = np.linalg.solve(A, (F / sc).T @ Y) / sc[:, None]                # (2r[+r], r)
    return dict(U=U, W=W, r=r, ped=ped)


def prognoza_dmd(d, t0, mod):
    U, W, r = mod['U'], mod['W'], mod['r']
    m = d.M[t0]
    c0 = (d.S[t0] - m) @ U; c1 = (d.S[t0 - 1] - m) @ U
    sp = d.P[t0] @ U
    start = c0.copy(); out = {}
    for k in range(HMAX):
        F = [c0, c1] + ([sp * BETA1 ** k] if mod['ped'] else [])
        cn = np.concatenate(F, -1) @ W
        c1, c0 = c0, cn
        if k + 1 in H_LISTA:
            out[k + 1] = (cn - start) @ U.T
    return out


# wprost: osobne dopasowanie na każde H
def prognoza_wprost(d_fit, d, t0, spec):
    out = {}
    for H in H_LISTA:
        t = np.arange(FIT[0], FIT[1] - H)
        coef = lsq(cechy(d_fit, t, spec), d_fit.S[t + H] - d_fit.S[t])
        out[H] = cechy(d, t0, spec) @ coef
    return out


def odniesienie(d, t0):
    """Równanie S_pęd + S_paczka co krok na prawdziwych wagach, zsumowane (nie prognoza)."""
    out = {}
    cum = np.zeros((len(t0), d.S.shape[1]))
    for k in range(HMAX):
        cum = cum + d.P[t0 + k] + d.N[t0 + k]
        if k + 1 in H_LISTA:
            out[k + 1] = cum.copy()
    return out


# ---------------------------------------------------------------- ocena
MODELE = {
    'bezwładność': dict(rodzina='Langevin', spec=dict(p=1)),
    'Langevin OU (tarcie + sprężyna)': dict(rodzina='Langevin', spec=dict(p=1, x=True)),
    'Langevin + jądro pamięci 20': dict(rodzina='Langevin', spec=dict(p=20, x=True)),
    'Langevin + jądro pamięci 100': dict(rodzina='Langevin', spec=dict(p=100, x=True)),
    'Koopman: opóźnienia 20, każda sonda': dict(rodzina='Koopman', koopman_ar=20),
    'Koopman DMD: 180 sond razem, 20 modów': dict(rodzina='Koopman', dmd=dict(r=20, ped=False)),
    'Koopman DMD + pęd': dict(rodzina='mieszane', dmd=dict(r=20, ped=True)),
    'Langevin OU + pęd': dict(rodzina='mieszane', spec=dict(p=1, x=True, ped=True)),
    'Langevin OU + pęd + przyspieszenie': dict(rodzina='mieszane', spec=dict(p=2, x=True, ped=True, acc=True)),
    'Langevin + jądro 20 + pęd': dict(rodzina='mieszane', spec=dict(p=20, x=True, ped=True)),
    'Langevin + jądro 20 + pęd + przyspieszenie': dict(rodzina='mieszane', spec=dict(p=20, x=True, ped=True, acc=True)),
    'Langevin + jądro 20 + pęd, wprost': dict(rodzina='wprost', wprost=dict(p=20, x=True, ped=True)),
    'Langevin + jądro 20, wprost (bez pędu)': dict(rodzina='wprost', wprost=dict(p=20, x=True)),
    'znane klatki kroku t, dalej jądro 20 + pęd': dict(rodzina='znane klatki', spec=dict(p=20, x=True, ped=True),
                                                         znane_klatki=True),
    'równanie co krok na prawdziwych wagach (nie prognoza)': dict(rodzina='odniesienie', odniesienie=True),
}


def main() -> int:
    t_start = time.time()
    D = {s: Dane(s) for s in (47, 48)}
    print(f'dane ({time.time() - t_start:.0f} s): s47 {D[47].S.shape}, s48 {D[48].S.shape}', flush=True)
    d_fit = D[47]
    wyn = {}
    for nazwa, mdl in MODELE.items():
        wsp = None
        if 'spec' in mdl:
            wsp = dopasuj_przyrost(d_fit, mdl['spec'])
        elif 'koopman_ar' in mdl:
            wsp = dopasuj_koopman_ar(d_fit, mdl['koopman_ar'])
        elif 'dmd' in mdl:
            wsp = dopasuj_dmd(d_fit, **mdl['dmd'])
        r = dict(rodzina=mdl['rodzina'], blad={}, umiejetnosc={}, umiejetnosc_s47={}, umiejetnosc_s48={})
        sse, ss = {H: 0.0 for H in H_LISTA}, {H: 0.0 for H in H_LISTA}
        for s, d in D.items():
            if 'spec' in mdl:
                pr = prognoza_przyrost(d, TEST, mdl['spec'], wsp, mdl.get('znane_klatki', False))
            elif 'koopman_ar' in mdl:
                pr = prognoza_koopman_ar(d, TEST, mdl['koopman_ar'], wsp)
            elif 'dmd' in mdl:
                pr = prognoza_dmd(d, TEST, wsp)
            elif 'wprost' in mdl:
                pr = prognoza_wprost(d_fit, d, TEST, mdl['wprost'])
            else:
                pr = odniesienie(d, TEST)
            for H in H_LISTA:
                y = d.S[TEST + H] - d.S[TEST]
                e2, y2 = float(((y - pr[H]) ** 2).sum()), float((y ** 2).sum())
                sse[H] += e2; ss[H] += y2
                r[f'umiejetnosc_s{s}'][H] = round(100 * (1 - e2 / y2), 2)
        for H in H_LISTA:
            r['umiejetnosc'][H] = round(100 * (1 - sse[H] / ss[H]), 2)
            r['blad'][H] = round(float(np.sqrt(sse[H] / ss[H])), 4)
        if 'spec' in mdl:
            r['wspolczynniki'] = np.round(wsp, 5).tolist()
        wyn[nazwa] = r
        print(f'{nazwa:52s} ' + ' '.join(f'{H}:{r["umiejetnosc"][H]:.0f}' for H in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500)),
              flush=True)
    ruch = {H: float(np.sqrt(np.mean([((D[s].S[TEST + H] - D[s].S[TEST]) ** 2).mean() for s in D]))) for H in H_LISTA}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'PROGNOZA_Z_HISTORII.json').write_text(json.dumps(dict(
        H=list(H_LISTA), dopasowanie='s47 kroki 1000-1999', test='s47 i s48, starty 2000-2495 co 5', modele=wyn,
        rms_przyrostu=ruch, sekundy=time.time() - t_start), ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'gotowe ({time.time() - t_start:.0f} s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
