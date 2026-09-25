#!/usr/bin/env python3
"""Najlepszy Koopman dla kierunkow tozsamosci narzedzia: warianty uzasadnione literatura (2026-09-25).

Wejscie: tokeny jak w licz.py oraz KIERUNKI_K{K}_S{S}.npz (w_lda, Q top-32, skalar a) zapisane przez licz.py.
Stan obrazu: y_i(t) = Q_k^T (m_i(t) - srednia okna), m_i = srednia tokenow wazona maska (opis narzedzia).

Modele predkosci v_t = y_t - y_{t-1} (wszystkie przewiduja v_{t+1}, potem z_{t+1} = z_t + v):
  alpha            per jednostka: v̂ = α v_t                                    (zanik wykladniczy)
  hiperb           per jednostka: v̂ = v_t (t+τ)/(t+1+τ), τ z siatki            (prawo 1/t, Gur-Ari i in. 2018:
                                                                               gradient ~ 1/t w gornej podprzestrzeni)
  hiperb_wspolne   jedno τ dla wszystkich jednostek (wybrane na obrazach fit)
  wspolny_vel_L    WSPOLNA miedzy obrazami macierz na L opoznieniach predkosci (Hankel, L = 1..4; Redman i in. 2023
                   uzywaja opoznien 8-64 na wagach), v̂ = c + Σ_l A_l v_{t-l+1}, grzbiet
  wspolny_hiperb   wspolna macierz A na predkosci przeskalowanej prawem 1/t: v̂ = g_τ(t) (c + A v_t)
  kontrole         brak_zmian, ostatni_przyrost
Ocena: RMSE 1 kroku i rollout 20 krokow (100 aktualizacji) od stanu S na obrazach holdout, wzgledem braku zmian.
Wybor τ i grzbietu wylacznie na obrazach fit, na koncowce okna (ostatnie 8 par).

    py -3.11 experiments/koopman_tozsamosc_kierunki/najlepszy.py [--zrodlo DIR] [--s-fit 48]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KS = [1, 4, 8, 16, 32]
TAU = [1., 2., 5., 10., 20., 40., 80., 160., 320., 1e9]


def g_tau(t, tau):
    """Czynnik tlumienia predkosci miedzy stanem t a t+1 dla prawa v ~ 1/(t+tau); t = indeks stanu (0 = krok 5)."""
    t = np.asarray(t, float)
    return (t + tau) / (t + 1 + tau)


def dopasuj_tau(V, s_fit, kandydaci=TAU, wspolne=False):
    """Wybor τ minimalizujacy SSE prognozy v_{t+1} = g(t) v_t na parach okna (t = 1..s_fit-1)."""
    t = np.arange(0, s_fit - 1)
    X = V[:, 0:s_fit - 1]; Y = V[:, 1:s_fit]
    sse = np.stack([((g_tau(t, tau)[None, :, None] * X - Y) ** 2).sum((1, 2)) for tau in kandydaci], 1)  # U x T
    if wspolne:
        return np.full(len(V), kandydaci[int(sse.sum(0).argmin())])
    return np.array(kandydaci)[sse.argmin(1)]


def hankel_pary(Vf, s_fit, L):
    """Pary (stos L opoznien predkosci -> nastepna predkosc) z obrazow fit do stanu s_fit."""
    Xs, Ys = [], []
    for t in range(L - 1, s_fit - 1):                                   # v_t dostepne dla t <= s_fit-1
        Xs.append(np.concatenate([Vf[:, t - l] for l in range(L)], 1)); Ys.append(Vf[:, t + 1])
    return np.concatenate(Xs), np.concatenate(Ys)


def ridge_fit(X, Y, rho):
    mx, my = X.mean(0), Y.mean(0); Xc = X - mx
    G = Xc.T @ Xc; d = X.shape[1]
    A = np.linalg.solve(G + rho * np.trace(G) / d * np.eye(d), Xc.T @ (Y - my))
    return (lambda Xq: my + (Xq - mx) @ A), dict(A=A, mx=mx, my=my, rho=np.array(rho))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zrodlo', type=Path, default=HERE.parent / 'koopman_gesta_historia/wyniki/macierze_wiele_obrazow')
    ap.add_argument('--s-fit', type=int, default=48)
    ap.add_argument('--out', type=Path, default=HERE / 'wyniki')
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = json.loads((args.zrodlo / 'CONFIG.json').read_text(encoding='utf-8'))
    steps = [w['step'] for w in cfg['weights']]; K = len(steps); S = args.s_fit
    kier = np.load(args.out / f'KIERUNKI_K{K}_S{S}.npz')
    Q_all, w_lda = kier['Q'], kier['w_lda']
    rows = cfg['rows']; fit = np.array([r['role'] == 'fit' for r in rows]); hold = ~fit
    maska = np.load(args.zrodlo / 'MASK_WEIGHTS.npy'); tlo = maska == 0; narz = maska >= .5
    Z = np.empty((72, 576, K, 384), np.float32)
    for k, s in enumerate(steps):
        Z[:, :, k, :] = np.load(args.zrodlo / f'TOKENS_{s:07d}.npy', mmap_mode='r')
    M = np.stack([(Z[i] * maska[i][:, None, None]).sum(0) / maska[i].sum() for i in range(72)]).astype(np.float64)
    BG = np.stack([Z[i, tlo[i]].mean(0) for i in range(72)]).astype(np.float64)
    # trajektorie tokenow narzedzia wyciete od razu, zeby zwolnic pelne tokeny (10 GB dla K=160)
    idx = [(i, p) for i in range(72) for p in np.flatnonzero(narz[i])]
    Yt = np.stack([Z[i, p] for i, p in idx]).astype(np.float64); hold_u = np.array([hold[i] for i, _ in idx])
    del Z
    a = (M - BG) @ w_lda
    print(f'wczytane K={K}, {time.time() - t0:.0f} s', flush=True)
    poczatki = np.arange(S, K - 1); H = 20; rollout_ok = S + H < K
    raport = dict(K=K, s_fit=S, krok_s_fit=steps[S], poczatki=[int(poczatki[0]), int(poczatki[-1])], horyzont_rollout=H, wyniki={})
    operatory = dict(w_lda=w_lda, Q32=Q_all[:, :32], steps=np.array(steps), s_fit=np.array(S))

    def ocena_wszystko(Y, nazwa_stanu):
        """Y: 72 x K x d. Zwraca slownik metoda -> miary na holdout (1 krok, rollout)."""
        V = np.diff(Y, axis=1); d = Y.shape[2]
        Yf, Vf = Y[fit], V[fit]
        # --- per jednostka
        X = V[:, 0:S - 1]; Yv = V[:, 1:S]
        alpha = np.einsum('und,und->u', X, Yv) / np.maximum(np.einsum('und,und->u', X, X), 1e-30)
        tau_u = dopasuj_tau(V, S); tau_w = dopasuj_tau(V[fit], S, wspolne=True)[0]
        # --- wspolne modele (fit tylko na obrazach fit); wybor grzbietu na koncowce okna obrazow fit
        wal = np.arange(S - 8, S - 1)                                       # poczatki walidacji wewnetrznej
        def wybierz_rho(budowa):
            best = None
            for rho in (1e-3, 1e-2, 1e-1, 1., 10.):
                pred = budowa(rho, S - 8)                                    # dopasowanie do stanu S-8
                e = np.mean((pred(Vf, wal) - (Yf[:, wal + 1] - Yf[:, wal])) ** 2)
                if best is None or e < best[0]: best = (e, rho)
            return best[1]
        def buduj_hankel(L):
            def budowa(rho, s_end):
                Xh, Yh = hankel_pary(Vf, s_end, L); f, par = ridge_fit(Xh, Yh, rho)
                def pred(Vq, tq):
                    Xq = np.concatenate([Vq[:, tq - 1 - l] for l in range(L)], -1)   # U x Q x (L d)
                    return f(Xq.reshape(-1, L * d)).reshape(len(Vq), len(tq), d)
                pred.par = par
                return pred
            return budowa
        def buduj_hiperb(tau):
            def budowa(rho, s_end):
                t = np.arange(0, s_end - 1); g = g_tau(t, tau)
                Xh = Vf[:, 0:s_end - 1].reshape(-1, d); Yh = (Vf[:, 1:s_end] / g[None, :, None]).reshape(-1, d)
                f, par = ridge_fit(Xh, Yh, rho)
                def pred(Vq, tq):
                    return g_tau(tq - 1, tau)[None, :, None] * f(Vq[:, tq - 1].reshape(-1, d)).reshape(len(Vq), len(tq), d)
                pred.par = dict(par, tau=np.array(tau))
                return pred
            return budowa
        modele = {}
        for L in (1, 2, 3, 4):
            b = buduj_hankel(L); rho = wybierz_rho(b); modele[f'wspolny_vel_L{L}'] = (b(rho, S), dict(rho=rho, L=L))
        b = buduj_hiperb(tau_w); rho = wybierz_rho(b); modele['wspolny_hiperb'] = (b(rho, S), dict(rho=rho, tau=float(tau_w)))
        # zapis operatorow dopasowanych do stanu S (wspolne macierze; per jednostka alpha i tau)
        for m, (pred, _) in modele.items():
            for k_, v_ in pred.par.items():
                operatory[f'{nazwa_stanu}__{m}__{k_}'] = np.asarray(v_)
        operatory[f'{nazwa_stanu}__alpha'] = alpha; operatory[f'{nazwa_stanu}__tau'] = tau_u
        # --- ocena 1 kroku na holdout
        t = poczatki; z = Y[:, t]; prawda = Y[:, t + 1] - z; v = V[:, t - 1]
        prog = {'alpha': alpha[:, None, None] * v, 'hiperb': g_tau(t - 1, tau_u[:, None])[:, :, None] * v,
                'hiperb_wspolne': g_tau(t - 1, tau_w)[None, :, None] * v,
                'brak_zmian': np.zeros_like(v), 'ostatni_przyrost': v}
        for m, (pred, _) in modele.items():
            prog[m] = pred(V, t)
        ref = np.mean(prawda[hold] ** 2)
        wyn = {m: dict(rmse_1krok_wzgl=float(np.sqrt(np.mean((p[hold] - prawda[hold]) ** 2) / ref)),
                       rmse_1krok=float(np.sqrt(np.mean((p[hold] - prawda[hold]) ** 2)))) for m, p in prog.items()}
        # --- rollout H krokow od S (bez prawdziwych stanow posrednich); dla modeli z opoznieniami trzymamy historie
        if rollout_ok:
            cel = Y[:, S + H]; z0 = Y[:, S]
            for m in prog:
                hist = [V[:, S - 1 - l] for l in range(4)][::-1]              # v_{S-4} .. v_{S-1}
                zz = z0.copy(); vv = V[:, S - 1].copy()
                for h in range(H):
                    tt = S + h
                    if m == 'alpha': vn = alpha[:, None] * vv
                    elif m == 'hiperb': vn = g_tau(tt - 1, tau_u)[:, None] * vv
                    elif m == 'hiperb_wspolne': vn = g_tau(tt - 1, tau_w) * vv
                    elif m == 'brak_zmian': vn = np.zeros_like(vv)
                    elif m == 'ostatni_przyrost': vn = vv
                    else:
                        L = modele[m][1].get('L', 1)
                        if m == 'wspolny_hiperb':
                            vn = modele[m][0](np.stack(hist, 1), np.array([len(hist)]))[:, 0]
                        else:
                            vn = modele[m][0](np.stack(hist, 1), np.array([len(hist)]))[:, 0]
                    zz = zz + vn; hist.append(vn); vv = vn
                wyn[m]['rmse_rollout20_wzgl'] = float(np.sqrt(np.mean((zz[hold] - cel[hold]) ** 2) / np.mean((z0[hold] - cel[hold]) ** 2)))
                wyn[m]['rmse_rollout20'] = float(np.sqrt(np.mean((zz[hold] - cel[hold]) ** 2)))
        wyn['_parametry'] = dict(tau_wspolne=float(tau_w), tau_per_jednostka_mediana=float(np.median(tau_u)),
                                 alpha_mediana=float(np.median(alpha)), **{m: p for m, (_, p) in modele.items()})
        raport['wyniki'][nazwa_stanu] = wyn
        print(f'{nazwa_stanu:10s} 1krok:', {m: round(r['rmse_1krok_wzgl'], 3) for m, r in wyn.items() if m != '_parametry'}, flush=True)
        if rollout_ok:
            print(f'{"":10s} 20krok:', {m: round(r['rmse_rollout20_wzgl'], 3) for m, r in wyn.items() if m != '_parametry'}, flush=True)

    ocena_wszystko(a[:, :, None], 'skalar_lda')
    Mc = M - M[:, :S].mean(1, keepdims=True)
    for k in KS:
        ocena_wszystko(Mc @ Q_all[:, :k], f'opis_S{k}')
    # dopelnienie 32 -> reszta (kontrola): losowe 8 kierunkow ortogonalnych do S32
    rng = np.random.default_rng(0)
    R = rng.standard_normal((384, 8)); R -= Q_all @ (Q_all.T @ R); R = np.linalg.qr(R)[0]
    ocena_wszystko(Mc @ R, 'losowe8_poza_S32')
    # poziom tokenow narzedzia: alpha vs hiperb (per jednostka i wspolne τ), pelne 384
    # Yt, hold_u policzone zaraz po wczytaniu tokenow, patrz wyzej
    V = np.diff(Yt, axis=1); t = poczatki; v = V[:, t - 1]; prawda = Yt[:, t + 1] - Yt[:, t]
    X = V[:, 0:S - 1]; Yv = V[:, 1:S]
    alpha = np.einsum('und,und->u', X, Yv) / np.maximum(np.einsum('und,und->u', X, X), 1e-30)
    tau_u = dopasuj_tau(V, S); tau_w = dopasuj_tau(V[~hold_u], S, wspolne=True)[0]
    prog = {'alpha': alpha[:, None, None] * v, 'hiperb': g_tau(t - 1, tau_u[:, None])[:, :, None] * v,
            'hiperb_wspolne': g_tau(t - 1, tau_w)[None, :, None] * v, 'ostatni_przyrost': v, 'brak_zmian': np.zeros_like(v)}
    ref = np.mean(prawda[hold_u] ** 2)
    raport['wyniki']['tokeny_narzedzia_384'] = {m: dict(rmse_1krok_wzgl=float(np.sqrt(np.mean((p[hold_u] - prawda[hold_u]) ** 2) / ref)))
                                                for m, p in prog.items()}
    raport['wyniki']['tokeny_narzedzia_384']['_parametry'] = dict(tau_wspolne=float(tau_w), tau_mediana=float(np.median(tau_u)), alpha_mediana=float(np.median(alpha)))
    print('tokeny 384 1krok:', {m: round(r['rmse_1krok_wzgl'], 3) for m, r in raport['wyniki']['tokeny_narzedzia_384'].items() if m != '_parametry'},
          'tau', tau_w, flush=True)
    raport['sekundy'] = time.time() - t0
    (args.out / f'REPORT_najlepszy_K{K}_S{S}.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(args.out / f'OPERATORY_K{K}_S{S}.npz', **operatory)   # macierze wspolnych operatorow + alpha/tau per jednostka
    print('gotowe', args.out / f'REPORT_najlepszy_K{K}_S{S}.json', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
