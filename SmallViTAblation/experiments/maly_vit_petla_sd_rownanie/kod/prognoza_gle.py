"""Prognoza siły tożsamości na jeden obrót równaniem Langevina z jądrem pamięci (wersja 4.3, docs/26 krok 3).

Postać z `../../maly_vit_prognoza_tozsamosci_z_historii` (najlepsze równanie z historii: Langevin + jądro pamięci,
dopasowane osobno na docelowy horyzont, „wprost”), przeniesiona na rozdzielczość zapisów pętli (co 10 kroków):
    s(r + H) − s(r) = Σ_{j<p} K_j · v(r − j) − κ · x(r),
gdzie v(r) = s(r) − s(r − 1) (przyrost między zapisami), x(r) = s(r) − m(r) odchylenie od poziomu m(r) = średnia z
ostatnich 50 zapisów (500 kroków), K_j - jądro pamięci (p = 20 zapisów = 200 kroków), κ - sprężyna do poziomu.
Współczynniki wspólne dla 180 sond, dopasowane w chwili prognozy tylko na historii tego biegu (pary (r, r + H) z
r + H ≤ teraz), regresja grzbietowa. Pęd optymalizatora pominięty: na 500 krokach nic nie dodaje (wygasa w ~10 krokach,
`maly_vit_prognoza_tozsamosci_z_historii/WYNIKI.md`).
"""
from __future__ import annotations

import numpy as np

P_JADRO, OKNO_M, RIDGE = 20, 50, 1e-3


def _poziom(S, okno=OKNO_M):
    c = np.cumsum(np.concatenate([np.zeros((1, S.shape[1])), S]), 0)
    t = np.arange(len(S)); a = np.maximum(t - okno + 1, 0)
    return (c[t + 1] - c[a]) / (t + 1 - a)[:, None]


def _cechy(S, V, M, r, p):
    return np.stack([V[r - j] for j in range(p)] + [S[r] - M[r]], -1)          # (len(r), P, p + 1)


def prognoza(S_hist, H, p=P_JADRO):
    """S_hist: (n, P) siła tożsamości w zapisach co 10 kroków. H: horyzont w zapisach (50 = 500 kroków).
    Zwraca (Δs (P,), info). Za krótka historia: Δs = 0."""
    S = np.asarray(S_hist, np.float64)
    n = len(S)
    V = np.diff(S, axis=0, prepend=S[:1]); M = _poziom(S)
    r = np.arange(p, n - H)
    if len(r) < 5:
        return np.zeros(S.shape[1]), dict(par=0, uwaga='za krótka historia')
    X = _cechy(S, V, M, r, p).reshape(-1, p + 1)
    y = (S[r + H] - S[r]).reshape(-1)
    sc = X.std(0) + 1e-12
    A = (X / sc).T @ (X / sc) + RIDGE * len(X) * np.eye(p + 1)
    coef = np.linalg.solve(A, (X / sc).T @ y) / sc
    d = _cechy(S, V, M, np.array([n - 1]), p)[0] @ coef
    pred_in = (X @ coef)
    um = float(100 * (1 - ((y - pred_in) ** 2).sum() / max((y ** 2).sum(), 1e-12)))
    return d, dict(par=int(len(r)), umiejetnosc_dopasowania=um, kappa=float(-coef[-1]), jadro=coef[:-1].round(5).tolist())
