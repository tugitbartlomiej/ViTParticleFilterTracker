#!/usr/bin/env python3
"""T1: równanie Langevina z TFT przełożone na latent sond, BEZ sterowania danymi (PLAN.md) — podłoga dla T2+.

Co rozstrzyga: ile przyrostu latentu w jednym kroku da się przewidzieć z samej historii latentu. Obali tezę
„bez danych 80 jest nieosiągalne”: umiejętność testowa T1 bliska 80.

Równanie (każda z 768 współrzędnych osobno, współczynniki wspólne dla sond; wariant `glob` — jedne współczynniki
dla wszystkich współrzędnych):
  Δz_{t+1} = D + JP + N
  D   dryf: −Σ_τ γ_τ (z_t − EMA_τ(z)_t) (powrót do średnich ruchomych, τ = 2..128) + a_v v_t + a_a (v_t − v_{t−1})
  JP  pamięć: Σ_m [α_m Re c^m_t + β_m Im c^m_t], c^m_t = ρ_m e^{iω_m} c^m_{t−1} + v_t (tłumione, obracające się jądra)
  N   szum barwny: te same jądra na własnych resztach r_t = v_t − (D + JP)_{t−1} (drugi etap, jak w TFT)
Wymuszenia i zdarzeń nie ma (użytkownik: „raczej nie będzie wymuszenia”).

Dane: Eden co 10 kroków (10 ziaren, ramię `bez`, kroki 510–4500, dopasowanie do ~2750, test dalej) oraz nagranie
co krok (ziarno 47, dopasowanie kroki 1000–2000, test 2000–3000), jeśli istnieje.

    python t1_bez_sterowania.py  ->  wyniki/T1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import dane as DN

TAU = (2, 4, 8, 16, 32, 64, 128)
RHO = (0.5, 0.8, 0.9, 0.97)
OMEGA = (0.0, np.pi / 2, np.pi)
LAM = 1e-3


def cechy(Z):
    """Z (T, P, C) → cechy (T, P, C, nF) z informacji do chwili t włącznie, do przewidywania Z[t+1] − Z[t]."""
    T = len(Z)
    V = np.zeros_like(Z); V[1:] = Z[1:] - Z[:-1]
    kol = []
    for tau in TAU:
        a = 1 / tau; E = np.empty_like(Z); e = Z[0].copy()
        for t in range(T):
            e = (1 - a) * e + a * Z[t]; E[t] = e
        kol.append(Z - E)
    kol.append(V)
    A = np.zeros_like(Z); A[1:] = V[1:] - V[:-1]; kol.append(A)
    kol += filtry(V)
    return np.stack(kol, -1)


def filtry(V):
    """Tłumione, obracające się jądra na sygnale V (T, P, C): lista tablic (T, P, C)."""
    out = []
    for rho in RHO:
        for om in OMEGA:
            c = np.zeros(V.shape[1:], np.complex64); Cs = np.empty(V.shape, np.complex64)
            f = rho * np.exp(1j * om)
            for t in range(len(V)):
                c = f * c + V[t]; Cs[t] = c
            out.append(Cs.real.astype(np.float32))
            if 0 < om < np.pi:
                out.append(Cs.imag.astype(np.float32))
    return out


def dopasuj_diag(F, Y, fit):
    """Osobne współczynniki dla każdej współrzędnej: F (T, P, C, nF), Y (T, P, C) → W (C, nF)."""
    C, nF = F.shape[2], F.shape[3]
    W = np.zeros((C, nF))
    for c in range(C):
        X = F[fit, :, c].reshape(-1, nF).astype(np.float64); y = Y[fit, :, c].reshape(-1).astype(np.float64)
        A = X.T @ X
        W[c] = np.linalg.solve(A + LAM * np.trace(A) / nF * np.eye(nF), X.T @ y)
    return W


def przebieg(Z, fit, test, etap_n=True):
    """Zwraca umiejętności (glob, diag, diag+N) na test dla całego Z (T, P, 768), liczone kawałkami po współrzędnych."""
    T, P, Dm = Z.shape
    Y = np.zeros_like(Z); Y[:-1] = Z[1:] - Z[:-1]                          # cel w chwili t: Z[t+1] − Z[t]
    wyn = {}
    # glob: akumulacja X^T X po kawałkach
    kaw = int(max(1, min(16, 3e8 / (T * P * 45 * 4))))
    XtX = None; Xty = None
    for c0 in range(0, Dm, kaw):
        F = cechy(Z[:, :, c0:c0 + kaw])
        X = F[fit].reshape(-1, F.shape[-1]).astype(np.float64); y = Y[fit, :, c0:c0 + kaw].reshape(-1).astype(np.float64)
        XtX = X.T @ X if XtX is None else XtX + X.T @ X; Xty = X.T @ y if Xty is None else Xty + X.T @ y
    nF = len(Xty)
    Wg = np.linalg.solve(XtX + LAM * np.trace(XtX) / nF * np.eye(nF), Xty)
    sse = {'glob': 0.0, 'diag': 0.0, 'diag_N': 0.0}; ss = 0.0; fit_sse = {'diag': 0.0}; fit_ss = 0.0
    for c0 in range(0, Dm, kaw):
        Zc = Z[:, :, c0:c0 + kaw]; Yc = Y[:, :, c0:c0 + kaw]
        F = cechy(Zc)
        yt = Yc[test]; ss += float((yt.astype(np.float64) ** 2).sum()); fit_ss += float((Yc[fit].astype(np.float64) ** 2).sum())
        sse['glob'] += float(((yt - F[test] @ Wg) ** 2).sum())
        W = dopasuj_diag(F, Yc, fit)
        Pd = np.einsum('tpcf,cf->tpc', F, W)
        sse['diag'] += float(((yt - Pd[test]) ** 2).sum()); fit_sse['diag'] += float(((Yc[fit] - Pd[fit]) ** 2).sum())
        if etap_n:
            R = np.zeros_like(Zc); R[1:] = Yc[:-1] - Pd[:-1]                  # reszta znana w chwili t: z kroku t−1 → t
            FN = np.concatenate([F, np.stack(filtry(R), -1)], -1)
            WN = dopasuj_diag(FN, Yc, fit)
            sse['diag_N'] += float(((yt - np.einsum('tpcf,cf->tpc', FN[test], WN)) ** 2).sum())
    wyn = {k: round(100 * (1 - v / ss), 2) for k, v in sse.items()}
    wyn['diag_na_dopasowaniu'] = round(100 * (1 - fit_sse['diag'] / fit_ss), 2)
    return wyn


def main() -> int:
    (DN.HERE / 'wyniki').mkdir(exist_ok=True)
    out = {}
    # Eden co 10 kroków
    eden = {}
    for s in DN.ZIARNA_EDEN:
        K0 = np.load(DN.EDEN / f'petla_s{s}' / 'wspolny' / 'k02' / 'LATENT.npz')
        K, Z = DN.eden(s)
        Z = np.concatenate([K0['latent'][:, :, 0].astype(np.float32), Z]); K = np.concatenate([K0['kroki'], K])
        ob = np.linalg.norm(Z[0], axis=1) > 0
        Z = Z[:, ob]
        fit = np.arange(50, int(np.searchsorted(K, 2750))); test = np.arange(int(np.searchsorted(K, 2750)), len(K) - 1)
        eden[s] = przebieg(Z, fit, test)
        print('eden', s, eden[s], flush=True)
    out['eden_10_krokow'] = dict(ziarna=eden, mediana={k: float(np.median([v[k] for v in eden.values()])) for k in eden[47]})
    print('eden mediana', out['eden_10_krokow']['mediana'], flush=True)
    # nagranie co krok
    if (DN.HERE / 'nagranie_s47' / 'META.json').exists():
        N = DN.nagranie(47)
        Z = N['sondy_lat'][:, :, 0].astype(np.float32); ob = np.array(N['META']['obecn'])
        Z = Z[:, ob]
        fit = np.arange(1000, 2000); test = np.arange(2000, len(Z) - 1)
        out['nagranie_1_krok'] = przebieg(Z, fit, test)
        print('nagranie 1 krok', out['nagranie_1_krok'], flush=True)
        Z10 = Z[::10]
        out['nagranie_10_krokow'] = przebieg(Z10, np.arange(100, 200), np.arange(200, len(Z10) - 1))
        print('nagranie 10 krokow', out['nagranie_10_krokow'], flush=True)
    (DN.HERE / 'wyniki' / 'T1.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
