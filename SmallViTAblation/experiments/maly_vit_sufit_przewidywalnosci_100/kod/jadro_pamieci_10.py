"""Równanie Langevina z jądrem pamięci (postać z projektu grantowego, bez wymuszenia) na 1..10 kroków.

    Δx_{t+1} = -κ (x_t - μ_t) + Σ_{j=0}^{J-1} K_j Δx_{t-j} + szum,   μ_t = średnia z ostatnich 101 stanów

Rozstrzyga: ile daje pamięć ostatnich pchnięć (jądro K_j) na horyzoncie 10 kroków, gdzie przyrosty są jeszcze
skorelowane (autokorelacja 0,69 przy 1 kroku, ujemna przy 3-5). Obala "jądro pamięci wystarczy na 10 kroków":
wynik na h=10 poniżej 0,5. Sufit wyroczni (średnia ruchoma 21) na h=10 to 0,75-0,78.
Warianty: długość jądra J = 5, 10, 20, 40; współczynniki wspólne (skalarne) albo osobne dla każdej z 768
współrzędnych; równanie iterowane krok po kroku (prawdziwa prognoza wielokrokowa) oraz dopasowanie wprost
na każdy horyzont (kontrola, górna granica dla tej klasy). Dopasowanie na celach do kroku 700, obrazy 0-5 / 6-11
na krzyż, test: wszystkie starty 700..890, horyzonty 1..10.
Uruchomienie: py -3.11 jadro_pamieci_10.py --wyniki ../wyniki/jadro_pamieci
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dane

HMAX = 10
K_MU = 101
J_LIST = (5, 10, 20, 40)
T_FIT_MAX = 700
STARTY_FIT = range(200, 700, 1)
STARTY_TEST = range(700, 891, 1)
N_LOS = 30
LAM = 1e-4


def wczytaj():
    lat = dane.latent(); tool, _ = dane.maski()
    rng = np.random.default_rng(20261003)
    Y, obraz = [], []
    for i in range(dane.N_OBR):
        idx = np.concatenate([np.flatnonzero(tool[i]), np.sort(rng.choice(dane.N_TOK, N_LOS, replace=False))])
        Y.append(np.asarray(lat[:, i, idx, :])); obraz += [i] * len(idx)
    return np.concatenate(Y, 1), np.array(obraz)


def cechy(y, cs, t, J):
    """[x_t - μ_t, Δx_t, Δx_{t-1}, ..., Δx_{t-J+1}] -> (U, 1+J, D)."""
    mu = (cs[t + 1] - cs[t + 1 - K_MU]) / K_MU
    return np.stack([y[t] - mu] + [y[t - j] - y[t - j - 1] for j in range(J)], 1)


def dopasuj(y, cs, maska, J, per_coord, cele):
    """cele='krok': cel Δx_{t+1} (równanie iterowane). cele='wprost': cele Δ_h, h=1..HMAX (osobne współczynniki)."""
    D = y.shape[2]; F = 1 + J; nh = HMAX if cele == 'wprost' else 1
    G = np.zeros((D, F, F)) if per_coord else np.zeros((F, F))
    b = np.zeros((D, F, nh)) if per_coord else np.zeros((F, nh))
    for t in STARTY_FIT:
        X = cechy(y, cs, t, J)[maska]
        G += np.einsum('ufd,ugd->dfg', X, X) if per_coord else np.einsum('ufd,ugd->fg', X, X)
        for h in range(1, nh + 1):
            if t + h > T_FIT_MAX:
                continue
            dl = (y[t + h] - y[t])[maska]
            if per_coord:
                b[:, :, h - 1] += np.einsum('ufd,ud->df', X, dl)
            else:
                b[:, h - 1] += np.einsum('ufd,ud->f', X, dl)
    if per_coord:
        reg = LAM * np.trace(G, axis1=1, axis2=2).mean() * np.eye(F)
        return np.linalg.solve(G + reg, b)
    return np.linalg.solve(G + LAM * np.trace(G) * np.eye(F), b)


def prognoza_iterowana(y, cs, t, J, beta, per_coord, maska):
    """Krok po kroku: przewidziane przyrosty wchodzą do jądra jako nowe opóźnienia; μ_t zamrożone."""
    mu = ((cs[t + 1] - cs[t + 1 - K_MU]) / K_MU)[maska]
    x = y[t][maska].copy()
    lagi = [(y[t - j] - y[t - j - 1])[maska] for j in range(J)]       # lagi[0] = Δx_t
    out = []
    for _ in range(HMAX):
        X = np.stack([x - mu] + lagi, 1)                                # (U, F, D)
        d = np.einsum('ufd,df->ud', X, beta[:, :, 0]) if per_coord else np.einsum('ufd,f->ud', X, beta[:, 0])
        x = x + d
        lagi = [d] + lagi[:-1]
        out.append(x.copy())
    return out


def prognoza_wprost(y, cs, t, J, beta, per_coord, maska):
    X = cechy(y, cs, t, J)[maska]
    xt = y[t][maska]
    if per_coord:
        return [xt + np.einsum('ufd,df->ud', X, beta[:, :, h]) for h in range(HMAX)]
    return [xt + np.einsum('ufd,f->ud', X, beta[:, h]) for h in range(HMAX)]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    y, obraz = wczytaj()
    T, U, D = y.shape
    cs = np.zeros((T + 1, U, D)); np.cumsum(y, 0, out=cs[1:])
    print('dane', y.shape, f'{time.time() - t0:.0f} s', flush=True)
    foldy = [(obraz < 6, obraz >= 6), (obraz >= 6, obraz < 6)]
    warianty = [(J, pc, cele) for J in J_LIST for pc in (False, True) for cele in ('krok', 'wprost')]
    sse = {w: np.zeros(HMAX) for w in warianty}
    sse_p = np.zeros(HMAX)
    wsp = {}
    for nf, (fit, test) in enumerate(foldy):
        for t in STARTY_TEST:
            for h in range(1, HMAX + 1):
                e = y[t + h][test] - y[t][test]
                sse_p[h - 1] += np.einsum('ud,ud->', e, e)
        for w in warianty:
            J, pc, cele = w
            beta = dopasuj(y, cs, fit, J, pc, cele)
            if not pc:
                wsp[f'fold{nf}_J{J}_{cele}'] = beta[:, 0].round(4).tolist() if cele == 'krok' else beta.round(4).tolist()
            for t in STARTY_TEST:
                pred = (prognoza_iterowana if cele == 'krok' else prognoza_wprost)(y, cs, t, J, beta, pc, test)
                for h in range(1, HMAX + 1):
                    e = pred[h - 1] - y[t + h][test]
                    sse[w][h - 1] += np.einsum('ud,ud->', e, e)
            print(f'fold {nf} J={J} per_coord={pc} {cele}: h1 {1 - sse[w][0] / sse_p[0]:.3f} '
                  f'h10 {1 - sse[w][-1] / sse_p[-1]:.3f} ({time.time() - t0:.0f} s)', flush=True)
    skill = {f'J{J}_{"per_coord" if pc else "skalarne"}_{cele}': (1 - sse[(J, pc, cele)] / sse_p).tolist()
             for (J, pc, cele) in warianty}
    wyn = dict(meta=dict(rownanie='dx_{t+1} = -kappa (x_t - mu_t) + sum_j K_j dx_{t-j}; mu_t srednia z 101 stanow',
                         horyzonty=list(range(1, HMAX + 1)), J=J_LIST, fit_cele_do=T_FIT_MAX,
                         starty_test=[STARTY_TEST.start, STARTY_TEST.stop - 1], n_jedn=int(U), lam=LAM,
                         czas_s=time.time() - t0,
                         skill='1 - SSE / SSE prognozy bez zmian, po jednostkach i 768 wspolrzednych, oba foldy'),
               skill=skill, wspolczynniki_skalarne=wsp)
    (out / 'JADRO_PAMIECI.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, v in skill.items():
        print(f'{k:28s}', [round(x, 3) for x in v])
    print('zapisano', out / 'JADRO_PAMIECI.json', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
