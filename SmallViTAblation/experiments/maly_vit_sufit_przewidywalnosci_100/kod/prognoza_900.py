"""Prognoza od kroku 900 (checkpoint 90) do kroku 1000 (checkpoint 100), osobno dla każdego kroku.

Rozstrzyga: jak dobrze równanie Langevina ze ściąganiem do średniej (proces Ornsteina-Uhlenbecka) przewiduje
latent na każdym z kroków 901..1000, w porównaniu z prognozą bez zmian, z przedłużeniem ostatniego ruchu,
z sufitem wyroczni i z NNDMD (ten sam bieg uczenia, wagi k0000..k1000, ale 1500 innych obrazów).
Obala równanie: wynik ujemny (gorzej niż bez zmian) na większości kroków.

Równanie:  x̂(t+h) = μ_t + (x_t - μ_t)·ρ^h,  μ_t = średnia z ostatnich k stanów,  ρ = exp(-κ).
To rozwiązanie dx = -κ (x - μ) dt + szum przy stałym μ. Parametry k i ρ dopasowane WYŁĄCZNIE na parach
(start, cel) z celem do kroku startu włącznie, więc kroki 901..1000 są nietknięte.

Wszystko liczę z sum po jednostkach i współrzędnych dla każdego kroku startu t i horyzontu h, więc dopasowanie
dla dowolnego startu jest natychmiastowe. Uruchomienie: py -3.11 prognoza_900.py --wyniki ../wyniki/prognoza_900
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dane
from sufit import kierunek_lda

HMAX = 100
K_SR = (21, 51, 101, 201)
T0_FIT = 200                 # pierwszy start w dopasowaniu (pełna historia 201 stanów)
STARTY = tuple(range(700, 901, 20))
N_LOS = 30                   # losowe tokeny na obraz (porównanie z NNDMD, który liczył wszystkie tokeny)
CH = 32
KAPPA = np.concatenate([[0.0], np.logspace(-4, 0.5, 500)])
NNDMD = (Path(__file__).resolve().parents[2] / 'maly_vit_nndmd_lokalne_przyrosty_hopper' / 'run1819918'
         / 'METRICS_TEST.json')


def sumy_na_krok(Y):
    """Y (T, U, D) -> sumy po (U, D) dla każdego startu t i horyzontu h (NaN, gdy niezdefiniowane)."""
    # Zera tam, gdzie start lub cel wychodzi poza dane; dopasowanie (t >= 200) i ocena (t0 + h <= 1000)
    # nigdy tych miejsc nie czytają.
    T = Y.shape[0]
    c, e = np.zeros((T, HMAX + 1)), np.zeros((T, HMAX + 1))
    a = {k: np.zeros((T, HMAX + 1)) for k in K_SR}
    b = {k: np.zeros(T) for k in K_SR}
    f, j = np.zeros(T), np.zeros(T)
    for u0 in range(0, Y.shape[1], CH):
        y = np.ascontiguousarray(Y[:, u0:u0 + CH], dtype=np.float64)
        cs = np.zeros((T + 1,) + y.shape[1:])
        np.cumsum(y, axis=0, out=cs[1:])
        s = np.arange(T)                                     # wyrocznia: średnia ruchoma 21, skracana przy końcach
        lo, hi = np.maximum(0, s - 10), np.minimum(T - 1, s + 10)
        M = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)[:, None, None]
        j += np.einsum('tud,tud->t', y - M, y - M)
        del M
        v = np.full_like(y, np.nan)
        v[10:] = (y[10:] - y[:-10]) / 10                     # prędkość z ostatnich 10 kroków
        f[10:] += np.einsum('tud,tud->t', v[10:], v[10:])
        d = {}
        for k in K_SR:
            mu = np.full_like(y, np.nan)
            mu[k - 1:] = (cs[k:] - cs[:-k]) / k
            d[k] = y - mu
            b[k][k - 1:] += np.einsum('tud,tud->t', d[k][k - 1:], d[k][k - 1:])
        for h in range(1, HMAX + 1):
            dl = y[h:] - y[:-h]                              # Δ_h dla startów 0..T-1-h
            c[:T - h, h] += np.einsum('tud,tud->t', dl, dl)
            e[10:T - h, h] += np.einsum('tud,tud->t', dl[10:], v[10:T - h])
            for k in K_SR:
                a[k][k - 1:T - h, h] += np.einsum('tud,tud->t', dl[k - 1:], d[k][k - 1:T - h])
    return dict(c=c, e=e, f=f, j=j, a=a, b=b)


def dopasuj(S, t0):
    """Najlepsze (k, κ) na startach T0_FIT..t0-h dla h = 1..HMAX (cele do t0)."""
    hs = np.arange(1, HMAX + 1)
    najl = None
    for k in K_SR:
        A = np.array([np.nansum(S['a'][k][T0_FIT:t0 - h + 1, h]) for h in hs])
        B = np.array([np.nansum(S['b'][k][T0_FIT:t0 - h + 1]) for h in hs])
        C = np.array([np.nansum(S['c'][T0_FIT:t0 - h + 1, h]) for h in hs])
        al = 1 - np.exp(-np.outer(KAPPA, hs))               # (nκ, h)
        loss = (C + 2 * al * A + al ** 2 * B).sum(1)
        i = int(np.argmin(loss))
        kand = (loss[i], k, float(KAPPA[i]), 1 - loss[i] / C.sum())
        if najl is None or kand[0] < najl[0]:
            najl = kand
    return dict(k=najl[1], kappa=najl[2], wynik_w_dopasowaniu=float(najl[3]))


def ocen(S, t0, par):
    """Błędy kwadratowe dla startu t0 na horyzontach 1..HMAX."""
    hs = np.arange(1, HMAX + 1)
    c = S['c'][t0, hs]
    al = 1 - np.exp(-par['kappa'] * hs)
    k = par['k']
    return dict(
        bez_zmian=c,
        rownanie=c + 2 * al * S['a'][k][t0, hs] + al ** 2 * S['b'][k][t0],
        srednia101=c + 2 * S['a'][101][t0, hs] + S['b'][101][t0],
        przedluzenie=c - 2 * hs * S['e'][t0, hs] + hs ** 2 * S['f'][t0],
        sufit=S['j'][t0 + hs],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wyniki', required=True)
    arg = p.parse_args()
    out = Path(arg.wyniki)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    lat = dane.latent()
    tool, bg = dane.maski()
    w_lda, _ = kierunek_lda(lat, tool, bg)
    rng = np.random.default_rng(20261003)
    tok_n, tok_l, sila = [], [], []
    for i in range(dane.N_OBR):
        it = np.flatnonzero(tool[i])
        il = np.sort(rng.choice(dane.N_TOK, N_LOS, replace=False))
        Xn = np.asarray(lat[:, i, it, :])
        tok_n.append(Xn)
        tok_l.append(np.asarray(lat[:, i, il, :]))
        sila.append((Xn @ w_lda).mean(1))
        print(f'obraz {i} wczytany, {time.time() - t_start:.0f} s', flush=True)
    wielk = {'tokeny_losowe': np.concatenate(tok_l, 1), 'tokeny_narzedzia': np.concatenate(tok_n, 1),
             'sila_tozsamosci': np.stack(sila, 1)[..., None]}
    del tok_n, tok_l
    wynik = dict(meta=dict(rownanie='x(t+h) = mu_t + (x_t - mu_t) * exp(-kappa*h), mu_t = srednia z ostatnich k stanow',
                           starty=STARTY, start_glowny=900, n_los_na_obraz=N_LOS, k_kandydaci=K_SR,
                           skill='1 - suma bledow^2 / suma bledow^2 prognozy bez zmian (po jednostkach i wspolrzednych)'),
                 wielkosci={})
    for q, Y in wielk.items():
        S = sumy_na_krok(Y)
        dq = {}
        for t0 in STARTY:
            par = dopasuj(S, t0)
            bl = ocen(S, t0, par)
            dq[str(t0)] = dict(parametry=par, **{m: (1 - bl[m] / bl['bez_zmian']).tolist() for m in bl if m != 'bez_zmian'},
                               bledy_kw={m: bl[m].tolist() for m in bl})
        # Średnia po startach: suma błędów po startach, potem stosunek
        sr = {}
        for m in ('rownanie', 'srednia101', 'przedluzenie', 'sufit'):
            num = sum(np.array(dq[str(t0)]['bledy_kw'][m]) for t0 in STARTY)
            den = sum(np.array(dq[str(t0)]['bledy_kw']['bez_zmian']) for t0 in STARTY)
            sr[m] = (1 - num / den).tolist()
        dq['srednia_po_startach'] = sr
        wynik['wielkosci'][q] = dq
        print(q, 'start 900', dq['900']['parametry'], 'h100:', {m: round(dq['900'][m][-1], 3) for m in sr},
              f'{time.time() - t_start:.0f} s', flush=True)
    np.save(out / 'SILA_TOZSAMOSCI.npy', wielk['sila_tozsamosci'][..., 0])
    if NNDMD.exists():
        mt = json.loads(NNDMD.read_text(encoding='utf-8'))
        wynik['nndmd'] = {'h': [r['horizon_source_updates'] for r in mt['full768']['horizons']],
                          'skill_mse': [r['mse_skill'] for r in mt['full768']['horizons']],
                          'skill_mse_srednia_historyczna': [1 - r['rmse_historical_mean'] ** 2 / r['rmse_persistence'] ** 2
                                                             for r in mt['full768']['horizons']],
                          'zrodlo': str(NNDMD)}
    (out / 'PROGNOZA_900.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
    print('zapisano', out / 'PROGNOZA_900.json', f'{time.time() - t_start:.0f} s', flush=True)


if __name__ == '__main__':
    main()
