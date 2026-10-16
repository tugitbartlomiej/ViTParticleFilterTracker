"""Prognoza latentu tożsamości w przód (Koopman i Langevin), PLAN.md.

Stan: (T, P, M) — aktywacje śledzonych neuronów na sondach z narzędziem. Odszumianie: średnia z bloków po B zapisów
(zapis co 10 kroków → punkt co 50 kroków). Redukcja: PCA do r wymiarów (wspólna dla sond i czasu).
  koopman    Δy_{t+1} = Σ_{j=0..L-1} A_j Δy_{t−j}                     (jak w poprzednich pętlach, bez hamowania)
  langevin   Δy_{t+1} = K y_t + Σ_{j=0..L-1} B_j Δy_{t−j} + a_p         (dryf z hamowaniem do punktu stałego sondy
             p — osobna stała a_p każdej sondy, bo każda sonda dąży do własnego poziomu — plus pamięć)
Wspólne współczynniki dla wszystkich sond, grzbiet (ridge). Rollout H punktów. Test wsteczny: dopasowanie bez
ostatnich H punktów, prognoza, RMSE względem „braku zmiany”.
"""
from __future__ import annotations

import numpy as np

B_BLOK, R_PCA, L_PAM, LAM = 5, 8, 4, 1e-2


def bloki(Z, b=B_BLOK):
    T = (len(Z) // b) * b
    return Z[len(Z) - T:].reshape(T // b, b, *Z.shape[1:]).mean(1)


def pca(Z, r=R_PCA):
    X = Z.reshape(-1, Z.shape[-1]); m = X.mean(0)
    V = np.linalg.svd(X - m, full_matrices=False)[2][:min(r, X.shape[1])].T
    return m, V


def _cechy(Y, D, t, model):
    """Cechy w chwili t (stan Y[t], przyrosty D[t-1..t-L], które do niego doprowadziły)."""
    c = [D[t - 1 - j] for j in range(L_PAM)]
    if model == 'langevin':
        c = [Y[t]] + c + [np.eye(Y.shape[1])]
    return np.concatenate(c, 1)


def dopasuj(Y, model):
    D = np.diff(Y, axis=0)                                           # D[t] = Y[t+1] − Y[t]
    X, y = [], []
    for t in range(L_PAM, len(D)):
        X.append(_cechy(Y, D, t, model)); y.append(D[t])
    X, y = np.concatenate(X), np.concatenate(y)
    A = X.T @ X                                                      # grzbiet względny: kara ∝ skali kolumny,
    Wt = np.linalg.solve(A + LAM * np.diag(np.diag(A)) + 1e-9 * np.eye(len(A)), X.T @ y)  # stałe sond nie gasną
    return Wt


def rollout(Y, Wt, H, model):
    Y = list(Y); D = [Y[i + 1] - Y[i] for i in range(len(Y) - 1)]
    for _ in range(H):
        t = len(Y) - 1
        x = _cechy(np.array(Y), np.array(D), t, model)
        d = x @ Wt
        Y.append(Y[-1] + d); D.append(d)
    return np.array(Y[-H:])


def prognozuj(Z, H):
    """Z: (T, P, M) surowa historia. Zwraca prognozę w przestrzeni neuronów na H punktów (po blokach) naprzód,
    obecny stan (ostatni blok) i diagnostykę testu wstecznego."""
    Zb = bloki(Z)
    if len(Zb) < L_PAM + 4:                                          # za krótka historia: brak prognozy
        return Zb[-1], Zb[-1], np.zeros(Zb.shape[1]), dict(punktow=int(len(Zb)), uwaga='za krótka historia')
    m, V = pca(Zb)
    Y = (Zb - m) @ V
    diag = {}
    for model in ('koopman', 'langevin'):
        if len(Y) > H + L_PAM + 4:
            W_ = dopasuj(Y[:-H], model)
            pr = rollout(Y[:-H], W_, H, model)[-1]
            e_m = float(np.sqrt(np.mean((pr - Y[-1]) ** 2))); e_0 = float(np.sqrt(np.mean((Y[-H - 1] - Y[-1]) ** 2)))
            diag[model] = dict(rmse=e_m, rmse_brak_zmiany=e_0, wzgledem_braku_zmiany=e_m / max(e_0, 1e-12))
    W_ = dopasuj(Y, 'langevin')
    Yh = rollout(Y, W_, H, 'langevin')
    teraz = Zb[-1]
    przyszlosc = teraz + (Yh[-1] - Y[-1]) @ V.T                      # reszta spoza PCA bez zmian
    diag['hamowanie_slad_K'] = float(np.trace(W_[:Y.shape[-1]]))
    diag['punktow'] = int(len(Y)); diag['wymiar'] = int(Y.shape[-1])
    return przyszlosc, teraz, np.linalg.norm(Yh[-1] - Y[-1], axis=1), diag
