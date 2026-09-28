#!/usr/bin/env python3
"""Bramki G1 i G2 z ../sterowanie_langevin_kalman/PLAN.md policzone na lokalnych wagach Nano (odczyt_dekodera_nano.py).

G1 (stan przewiduje zadanie): jakość sond z dekodera (średnia q po ostatniej warstwie) vs mAP@0,5 całego val MIĘDZY
   ramionami w tym samym (ziarno, epoka) — wartości wycentrowane w grupie, Spearman. Zestawy: nano5ep (4 ramiona ×
   3 ziarna × 5 epok) i nanopilot (3 pętle × 8 obrotów). Warunek z planu: ρ ≥ 0,4 i p < 0,05.
G2 (model bije „brak zmiany”): bieg gęsty 1810662 (co 5 aktualizacji, kroki 5..805). Stan: q każdej z 72 sond po
   ostatniej warstwie. Dopasowanie na krokach ≤ 400, test prognoz z punktów startu > 400 na h = 20 zapisów
   (100 aktualizacji = 1 epoka). Modele (współczynniki wspólne dla sond, stałe sond osobne):
     M0 brak zmiany
     M1 Koopman na przyrostach (8 ostatnich, jak w pętli koopman; bez hamowania)
     M2 dryf z hamowaniem (OU): Δq_i = a_i + b q_i
     M3 Langevin: M2 + pamięć 8 przyrostów + wymuszenie (granica epoki)
     M4 M3 z odszumianiem Kalmana (stan q_i ukryty, odczyt = stan + szum; parametry szumu z danych dopasowania)
   Miara: RMSE prognozy / RMSE braku zmiany (< 1 = lepiej niż nic). Warunek z planu: ≤ 0,8.

    py -3.11 bramki_lokalne.py  ->  wyniki/BRAMKI_LOKALNE.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
L = 8          # pamięć (przyrosty)
H = 20         # horyzont (zapisy co 5 aktualizacji)
PODZIAL = 400  # krok końca dopasowania


def g1(plik):
    d = json.loads((HERE / 'wyniki' / plik).read_text(encoding='utf-8'))['wynik']
    grupy = {}
    for w in d:
        if w['mAP50'] is None:
            continue
        grupy.setdefault((w['ziarno'], w['czas']), []).append((float(np.mean(np.array(w['q'])[:, -1])), w['mAP50'], w['ramie']))
    x, y = [], []
    for g in grupy.values():
        if len(g) < 2:
            continue
        mq = np.mean([a for a, _, _ in g]); mm = np.mean([b for _, b, _ in g])
        x += [a - mq for a, _, _ in g]; y += [b - mm for _, b, _ in g]
    r = spearmanr(x, y)
    return dict(n=len(x), grup=len(grupy), rho=float(r.statistic), p=float(r.pvalue))


def ridge(X, y, lam=1e-3):
    return np.linalg.solve(X.T @ X + lam * len(y) * np.eye(X.shape[1]), X.T @ y)


def g2():
    d = json.loads((HERE / 'wyniki/NANO_nanogesty.json').read_text(encoding='utf-8'))['wynik']
    kroki = np.array([w['czas'] for w in d]); o = np.argsort(kroki); kroki = kroki[o]
    Q = np.array([np.array(d[i]['q'])[:, -1] for i in o])            # (T, 72)
    T, P = Q.shape
    tr = kroki <= PODZIAL
    ep = ((kroki % 100) == 0).astype(float)                             # granica epoki (wymuszenie)
    dQ = np.diff(Q, axis=0)
    # --- dopasowania na części treningowej
    def cechy(t, Qs, dQs, z_pamiecia=True, z_polozeniem=True):
        c = []
        if z_polozeniem:
            c.append(Qs[t])                                             # b q_i
        if z_pamiecia:
            c += [dQs[t - j] for j in range(1, L + 1)]                  # pamięć
            c.append(np.full(P, ep[t + 1]))                             # wymuszenie
        return np.stack(c, 1)
    def fit(Qs, dQs, z_pamiecia, z_polozeniem):
        X, y, idx = [], [], []
        for t in range(L, int(tr.sum()) - 1):
            X.append(cechy(t, Qs, dQs, z_pamiecia, z_polozeniem)); y.append(dQs[t]); idx.append(np.arange(P))
        X = np.concatenate(X); y = np.concatenate(y); idx = np.concatenate(idx)
        D = np.zeros((len(y), P)); D[np.arange(len(y)), idx] = 1          # stałe sond a_i
        w = ridge(np.hstack([X, D]), y)
        return w[:X.shape[1]], w[X.shape[1]:]
    def rollout(t0, Qs, dQs, wk, a, z_pamiecia, z_polozeniem):
        q = Qs[t0].copy(); hist = [dQs[t0 - j] for j in range(L, 0, -1)] if z_pamiecia else []
        for s in range(H):
            c = []
            if z_polozeniem:
                c.append(q)
            if z_pamiecia:
                c += [hist[-j] for j in range(1, L + 1)]; c.append(np.full(P, ep[min(t0 + s + 1, T - 1)]))
            dq = np.stack(c, 1) @ wk + a
            q = q + dq
            if z_pamiecia:
                hist.append(dq)
        return q
    # M1: Koopman na przyrostach (bez położenia i bez stałych) — jak w pętli koopman
    X, y = [], []
    for t in range(L, int(tr.sum()) - 1):
        X.append(np.stack([dQ[t - j] for j in range(1, L + 1)], 1)); y.append(dQ[t])
    wK = ridge(np.concatenate(X), np.concatenate(y), lam=0.1)
    # Kalman (M4): per sonda stan AR(1) wokół celu; odczyt = stan + szum. Szum odczytu z ujemnej autokorelacji
    # przyrostów na części treningowej: dla y = x + v, cov(Δy_t, Δy_{t-1}) ≈ −σ_v².
    dtr = dQ[:int(tr.sum()) - 1]
    sv2 = max(-float(np.mean(dtr[1:] * dtr[:-1])), 1e-6)
    sw2 = max(float(np.var(dtr)) - 2 * sv2, 1e-6)
    def kalman(Y, b, a):
        x = Y[0].copy(); Pk = np.full(P, sv2); out = [x.copy()]
        for t in range(1, len(Y)):
            x = x + a + b * x; Pk = (1 + b) ** 2 * Pk + sw2          # prognoza
            K = Pk / (Pk + sv2); x = x + K * (Y[t] - x); Pk = (1 - K) * Pk
            out.append(x.copy())
        return np.array(out)
    w2, a2 = fit(Q, dQ, False, True)
    Qf = kalman(Q, w2[0], a2); dQf = np.diff(Qf, axis=0)
    wy = {}
    w3, a3 = fit(Q, dQ, True, True)
    w4, a4 = fit(Qf, dQf, True, True)
    starty = [t for t in range(int(tr.sum()) + L, T - H)]
    bl = {k: [] for k in ('M0', 'M1', 'M2', 'M3', 'M4')}
    for t0 in starty:
        prawda = Q[t0 + H]
        bl['M0'].append(prawda - Q[t0])
        hist = [dQ[t0 - j] for j in range(L, 0, -1)]; q = Q[t0].copy()
        for _ in range(H):
            dq = np.stack([hist[-j] for j in range(1, L + 1)], 1) @ wK; q = q + dq; hist.append(dq)
        bl['M1'].append(prawda - q)
        bl['M2'].append(prawda - rollout(t0, Q, dQ, w2, a2, False, True))
        bl['M3'].append(prawda - rollout(t0, Q, dQ, w3, a3, True, True))
        bl['M4'].append(prawda - rollout(t0, Qf, dQf, w4, a4, True, True))
    rm = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in bl.items()}
    wy.update(zapisow=int(T), krokow_dopasowania=int(PODZIAL), horyzont_zapisow=H, startow_testu=len(starty),
              rmse=rm, wzgledem_braku_zmiany={k: v / rm['M0'] for k, v in rm.items()},
              hamowanie_b=float(w2[0]), szum_odczytu=sv2 ** .5, szum_procesu=sw2 ** .5,
              jakosc_srednia=dict(start=float(Q[0].mean()), podzial=float(Q[tr][-1].mean()), koniec=float(Q[-1].mean())))
    return wy


WARIANTY = ('original', 'rotation90', 'black', 'yellow', 'length08', 'foreground_bright')


def g1_po_fakcie(plik, var):
    """Kontrola PO WYNIKU (nie zmienia werdyktu G1): inne odczyty stanu dekodera w tym samym teście między ramionami."""
    d = [w for w in json.loads((HERE / 'wyniki' / plik).read_text(encoding='utf-8'))['wynik'] if w['mAP50'] is not None]
    odczyty = {'ostatnia_warstwa': lambda w: np.array(w['q'])[:, -1].mean(),
               'srednia_warstw': lambda w: np.array(w['q']).mean(),
               'propozycje_iou': lambda w: np.mean(w['propozycje_iou']),
               'najslabszy_wariant': lambda w: min(np.array(w['q'])[var == v, -1].mean() for v in WARIANTY)}
    odczyty.update({f'wariant_{v}': (lambda v: lambda w: np.array(w['q'])[var == v, -1].mean())(v) for v in WARIANTY})
    out = {}
    for nazwa, f in odczyty.items():
        grupy = {}
        for w in d:
            grupy.setdefault((w['ziarno'], w['czas']), []).append((float(f(w)), w['mAP50']))
        x, y = [], []
        for g in grupy.values():
            if len(g) > 1:
                a = np.array(g); x += list(a[:, 0] - a[:, 0].mean()); y += list(a[:, 1] - a[:, 1].mean())
        r = spearmanr(x, y)
        out[nazwa] = dict(rho=float(r.statistic), p=float(r.pvalue))
    # skala różnic: rozrzut mAP między ramionami w grupie vs zmiana mAP między kolejnymi epokami
    grupy = {}
    for w in d:
        grupy.setdefault((w['ziarno'], w['czas']), []).append(w['mAP50'])
    out['rozrzut_mAP_miedzy_ramionami'] = float(np.mean([np.std(g) for g in grupy.values() if len(g) > 1]))
    return out


def g2_po_fakcie():
    """Kontrola PO WYNIKU: G2 dopasowane i testowane na samym plateau (bez fazy startu nowej głowicy), skalar i sondy,
    horyzonty 4 i 20 zapisów. Modele bez stałych sond (czysta dynamika przyrostów) i z hamowaniem do średniej plateau."""
    d = json.loads((HERE / 'wyniki/NANO_nanogesty.json').read_text(encoding='utf-8'))['wynik']
    d = sorted(d, key=lambda w: w['czas'])
    kroki = np.array([w['czas'] for w in d]); Q = np.array([np.array(w['q'])[:, -1] for w in d])
    out = {}
    for stan, X in (('sondy', Q), ('srednia', Q.mean(1, keepdims=True))):
        dX = np.diff(X, axis=0)
        fit_idx = [t for t in range(len(kroki) - 1) if 300 < kroki[t] <= 600 and t >= L]
        mu = X[[t for t in range(len(kroki)) if 300 < kroki[t] <= 600]].mean(0)       # cel hamowania (plateau)
        # AR na przyrostach z hamowaniem: dX_t = b (X_t - mu) + sum_j c_j dX_{t-j}
        A = np.concatenate([np.stack([X[t] - mu] + [dX[t - j] for j in range(1, L + 1)], 1) for t in fit_idx])
        y = np.concatenate([dX[t] for t in fit_idx])
        w = ridge(A, y, lam=1e-2)
        # AR bez hamowania (Koopman na przyrostach, jak M1, ale na plateau)
        wk = ridge(A[:, 1:], y, lam=1e-2)
        for H_ in (4, 20):
            st = [t for t in range(len(kroki)) if kroki[t] > 600 and t + H_ < len(kroki)]
            e = {k: [] for k in ('M0_brak_zmiany', 'M1_koopman_plateau', 'M3_hamowanie_pamiec', 'Mmu_srednia_plateau')}
            for t0 in st:
                pr = X[t0 + H_]
                e['M0_brak_zmiany'].append(pr - X[t0]); e['Mmu_srednia_plateau'].append(pr - mu)
                for nazwa, ww, ham in (('M1_koopman_plateau', wk, False), ('M3_hamowanie_pamiec', w, True)):
                    x = X[t0].copy(); h = [dX[t0 - j] for j in range(L, 0, -1)]
                    for _ in range(H_):
                        c = ([x - mu] if ham else []) + [h[-j] for j in range(1, L + 1)]
                        dx = np.stack(c, 1) @ ww; x = x + dx; h.append(dx)
                    e[nazwa].append(pr - x)
            rm = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in e.items()}
            out[f'{stan}_h{H_}'] = dict(startow=len(st), wzgledem_braku_zmiany={k: v / rm['M0_brak_zmiany'] for k, v in rm.items()},
                                        rmse_brak_zmiany=rm['M0_brak_zmiany'])
        out[f'{stan}_hamowanie_b'] = float(w[0]); out[f'{stan}_pamiec_suma'] = float(w[1:].sum())
    return out


def main() -> int:
    import sys
    sys.path.insert(0, str(HERE.parent / 'petla_od_poczatku_large'))
    import obrot_large as O
    rows, _, _ = O.sondy(); var = np.array([r['variant'] for r in rows])
    wyn = dict(opis=__doc__)
    for z in ('nano5ep', 'nanopilot'):
        if (HERE / f'wyniki/NANO_{z}.json').exists():
            wyn[f'G1_{z}'] = g1(f'NANO_{z}.json')
            wyn[f'G1_{z}_po_fakcie'] = g1_po_fakcie(f'NANO_{z}.json', var)
    if (HERE / 'wyniki/NANO_nanogesty.json').exists():
        wyn['G2_nanogesty'] = g2()
        wyn['G2_nanogesty_plateau_po_fakcie'] = g2_po_fakcie()
    (HERE / 'wyniki/BRAMKI_LOKALNE.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in wyn.items() if k != 'opis'}, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
