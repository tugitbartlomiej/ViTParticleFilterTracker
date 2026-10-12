"""Ile jeszcze da się wycisnąć z samej historii latentu na h = 10, 50, 100 kroków i gdzie siedzi reszta.

Rozstrzyga trzy pytania:
1. Czy bogatszy model z historii (osobne współczynniki dla każdej z 768 współrzędnych, cechy wspólnego ruchu)
   przebija równanie OU (0,22-0,27 na h=100)? Obala "lepszy model wystarczy": wynik poniżej 0,5.
2. Czy błądzenie poziomu jest wspólne dla wszystkich tokenów (jedna zmiana wag)? Wyrocznia znająca przyszły
   WSPÓLNY ruch s_t (średnia po wszystkich jednostkach) pokazuje, ile dałoby przewidzenie jednego wektora 768.
3. Czy część własna tokenu r = x - s jest przewidywalna lepiej niż całość.

Dopasowanie: cele do kroku 700, obrazy 0-5 / 6-11 na krzyż. Test: starty 700, 720, ..., 900. Bez treningu.
Uruchomienie: py -3.11 co_jeszcze_z_historii.py --wyniki ../wyniki/co_jeszcze
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dane

H = (10, 50, 100)
K = (5, 21, 51, 101, 201)
L = (10, 50, 100)
T_FIT_MAX = 700
STARTY_FIT = range(200, 700, 2)
STARTY_TEST = tuple(range(700, 901, 20))
N_LOS = 30
KAPPA = np.concatenate([[0.0], np.logspace(-4, 0.5, 300)])


def wczytaj():
    lat = dane.latent()
    tool, _ = dane.maski()
    rng = np.random.default_rng(20261003)
    Y, obraz_jedn = [], []
    for i in range(dane.N_OBR):
        it = np.flatnonzero(tool[i])
        il = np.sort(rng.choice(dane.N_TOK, N_LOS, replace=False))
        idx = np.concatenate([it, il])
        Y.append(np.asarray(lat[:, i, idx, :]))
        obraz_jedn += [i] * len(idx)
    return np.concatenate(Y, 1), np.array(obraz_jedn)


def cechy(y, cs, t):
    f = [(cs[t + 1] - cs[t + 1 - k]) / k - y[t] for k in K] + [y[t] - y[t - l] for l in L]
    return np.stack(f, 1)                                 # (U, F, D)


def cechy_z_wspolnymi(y, cs, ys, css, t, maska_jedn):
    """Cechy własne jednostki plus cechy wspólnego ruchu s_t (te same dla każdej jednostki): (U, 2F, D)."""
    X = cechy(y, cs, t)[maska_jedn]
    W = cechy(ys[:, None, :], css[:, None, :], t)         # (1, F, D)
    return np.concatenate([X, np.broadcast_to(W, (X.shape[0],) + W.shape[1:])], 1)


def ou_dopasuj(y, cs, maska_jedn):
    """OU ze wspólnym (k, kappa): najlepsze na startach fit z celem <= T_FIT_MAX."""
    hs = np.array(H)
    najl = None
    for k in K[1:]:
        A = np.zeros(len(H)); B = np.zeros(len(H)); C = np.zeros(len(H))
        for t in STARTY_FIT:
            d = ((cs[t + 1] - cs[t + 1 - k]) / k - y[t])[maska_jedn]
            B += np.einsum('ud,ud->', d, d)
            for j, h in enumerate(H):
                if t + h > T_FIT_MAX:
                    continue
                dl = (y[t + h] - y[t])[maska_jedn]
                A[j] += np.einsum('ud,ud->', dl, d); C[j] += np.einsum('ud,ud->', dl, dl)
        # prognoza y + al*d, d = mu - y, błąd = dl - al*d, więc SSE = C - 2*al*A + al^2*B
        al = 1 - np.exp(-np.outer(KAPPA, hs))
        loss = (C - 2 * al * A + al ** 2 * B).sum(1)
        i = int(np.argmin(loss))
        if najl is None or loss[i] < najl[0]:
            najl = (loss[i], k, float(KAPPA[i]))
    return dict(k=najl[1], kappa=najl[2])


def ou_prognoza(y, cs, t, h, par):
    mu = (cs[t + 1] - cs[t + 1 - par['k']]) / par['k']
    return mu + (y[t] - mu) * np.exp(-par['kappa'] * h)


def ridge_dopasuj(y, cs, ys, css, maska_jedn, wspolne, per_coord, lam=1e-3):
    D = y.shape[2]
    F = len(K) + len(L) + (len(K) + len(L) if wspolne else 0)
    G = np.zeros((D, F, F)) if per_coord else np.zeros((F, F))
    b = np.zeros((D, F, len(H))) if per_coord else np.zeros((F, len(H)))
    for t in STARTY_FIT:
        X = cechy_z_wspolnymi(y, cs, ys, css, t, maska_jedn) if wspolne else cechy(y, cs, t)[maska_jedn]
        if per_coord:
            G += np.einsum('ufd,ugd->dfg', X, X)
        else:
            G += np.einsum('ufd,ugd->fg', X, X)
        for j, h in enumerate(H):
            if t + h > T_FIT_MAX:
                continue
            dl = (y[t + h] - y[t])[maska_jedn]
            if per_coord:
                b[:, :, j] += np.einsum('ufd,ud->df', X, dl)
            else:
                b[:, j] += np.einsum('ufd,ud->f', X, dl)
    if per_coord:
        reg = lam * np.trace(G, axis1=1, axis2=2).mean() * np.eye(F)
        beta = np.linalg.solve(G + reg, b)                # (D, F, H)
    else:
        beta = np.linalg.solve(G + lam * np.trace(G) * np.eye(F), b)   # (F, H)
    return beta


def ridge_prognoza(y, cs, ys, css, t, h_idx, maska_jedn, beta, wspolne, per_coord):
    X = cechy_z_wspolnymi(y, cs, ys, css, t, maska_jedn) if wspolne else cechy(y, cs, t)[maska_jedn]
    if per_coord:
        return y[t][maska_jedn] + np.einsum('ufd,df->ud', X, beta[:, :, h_idx])
    return y[t][maska_jedn] + np.einsum('ufd,f->ud', X, beta[:, h_idx])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    y, obraz = wczytaj()
    print('dane', y.shape, f'{time.time() - t0:.0f} s', flush=True)
    T, U, D = y.shape
    s = y.mean(1)                                          # wspólny ruch (T, D)
    r = y - s[:, None, :]                                  # część własna
    cs = np.zeros((T + 1, U, D)); np.cumsum(y, 0, out=cs[1:])
    css = np.zeros((T + 1, D)); np.cumsum(s, 0, out=css[1:])
    csr = np.zeros((T + 1, U, D)); np.cumsum(r, 0, out=csr[1:])
    foldy = [(obraz < 6, obraz >= 6), (obraz >= 6, obraz < 6)]
    modele = ['bez_zmian', 'ou', 'ridge_skalarny', 'ridge_per_coord', 'ridge_per_coord_plus_wspolne',
              'ou_wlasne_plus_wspolne_wyrocznia', 'bez_zmian_plus_wspolne_wyrocznia', 'ou_wspolny_sam',
              'ou_wlasne_sam', 'bez_zmian_wlasne']
    sse = {m: np.zeros(len(H)) for m in modele}
    sse_pers = np.zeros(len(H)); sse_pers_r = np.zeros(len(H)); sse_pers_s = np.zeros(len(H))
    sse_ou_s = np.zeros(len(H)); sse_ou_r = np.zeros(len(H))
    par_log = {}
    for nf, (fit, test) in enumerate(foldy):
        par = ou_dopasuj(y, cs, fit)
        par_r = ou_dopasuj(r, csr, fit)
        par_s = ou_dopasuj(s[:, None, :], css[:, None, :], np.array([True]))
        par_log[f'fold{nf}'] = dict(ou=par, ou_wlasne=par_r, ou_wspolny=par_s)
        print(f'fold {nf}: OU {par}, własne {par_r}, wspólny {par_s}, {time.time() - t0:.0f} s', flush=True)
        b_sk = ridge_dopasuj(y, cs, s, css, fit, False, False)
        b_pc = ridge_dopasuj(y, cs, s, css, fit, False, True)
        b_pcw = ridge_dopasuj(y, cs, s, css, fit, True, True)
        print(f'fold {nf}: ridge dopasowany, {time.time() - t0:.0f} s', flush=True)
        for t in STARTY_TEST:
            for j, h in enumerate(H):
                prawda = y[t + h][test]
                xt = y[t][test]
                e = lambda pred: np.einsum('ud,ud->', pred - prawda, pred - prawda)
                sse_pers[j] += e(xt)
                sse['ou'][j] += e(ou_prognoza(y, cs, t, h, par)[test])
                sse['ridge_skalarny'][j] += e(ridge_prognoza(y, cs, s, css, t, j, test, b_sk, False, False))
                sse['ridge_per_coord'][j] += e(ridge_prognoza(y, cs, s, css, t, j, test, b_pc, False, True))
                sse['ridge_per_coord_plus_wspolne'][j] += e(ridge_prognoza(y, cs, s, css, t, j, test, b_pcw, True, True))
                ds = s[t + h] - s[t]                        # prawdziwy przyszły ruch wspólny (wyrocznia)
                sse['bez_zmian_plus_wspolne_wyrocznia'][j] += e(xt + ds)
                r_hat = ou_prognoza(r, csr, t, h, par_r)[test]
                sse['ou_wlasne_plus_wspolne_wyrocznia'][j] += e(r_hat + s[t + h])
                # składowe osobno
                sse_pers_r[j] += np.einsum('ud,ud->', r[t][test] - r[t + h][test], r[t][test] - r[t + h][test])
                sse_ou_r[j] += np.einsum('ud,ud->', r_hat - r[t + h][test], r_hat - r[t + h][test])
                if nf == 0:
                    sse_pers_s[j] += float(((s[t] - s[t + h]) ** 2).sum())
                    s_hat = ou_prognoza(s[:, None, :], css[:, None, :], t, h, par_s)[0]
                    sse_ou_s[j] += float(((s_hat - s[t + h]) ** 2).sum())
    skill = {m: (1 - sse[m] / sse_pers).tolist() for m in modele if sse[m].any()}
    skill['ou_wlasne_sam'] = (1 - sse_ou_r / sse_pers_r).tolist()
    skill['ou_wspolny_sam'] = (1 - sse_ou_s / sse_pers_s).tolist()
    # jaki udział błędu "bez zmian" to ruch wspólny: SSE_pers(x) = SSE_pers(r) + U * SSE_pers(s) (+ wyraz mieszany ~0);
    # sse_pers_s liczone raz (fold 0), sse_pers po obu foldach, czyli po wszystkich U jednostkach
    udzial = (U * sse_pers_s / sse_pers).tolist()
    wynik = dict(meta=dict(horyzonty=H, starty_test=STARTY_TEST, fit_cele_do=T_FIT_MAX, cechy_k=K, cechy_lagi=L,
                           n_jednostek=int(U), n_los_na_obraz=N_LOS, czas_s=time.time() - t0, parametry=par_log,
                           definicje=dict(
                               skill='1 - SSE / SSE prognozy bez zmian, po jednostkach i 768 współrzędnych, oba foldy',
                               wspolny='s_t = średnia po wszystkich jednostkach (12 obrazów); własne r = x - s',
                               wyrocznia_wspolna='prognoza dostaje PRAWDZIWY przyszły ruch wspólny s_{t+h} - s_t')),
                 skill=skill, udzial_ruchu_wspolnego_w_bledzie_bez_zmian=udzial)
    (out / 'CO_JESZCZE.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(skill, indent=1), 'udział wspólnego:', udzial, flush=True)
    print('zapisano', out / 'CO_JESZCZE.json', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
