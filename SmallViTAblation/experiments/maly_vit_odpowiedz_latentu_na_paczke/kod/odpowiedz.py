"""Funkcja odpowiedzi latentu sond na pchnięcie jednej paczki, z istniejących nagrań (PLAN.md).

Dane: `../../maly_vit_rownanie_latentu/nagranie_s{47,48}_sily` (docs/23): latent 180 sond z narzędziem po każdym
kroku i siły S_nowa, S_ped, S_zanik w stanie t. Bez treningu, bez GPU.

Dla horyzontu h i startu t (współczynniki wspólne dla 768 współrzędnych i 180 sond):
    z(t+h) - z(t) = sum_{k<h} c(h,k) S_nowa(t+k) + a S_ped(t) + w S_zanik(t) + g (z(t) - mu(t)) + reszta
Modele: A = wszystkie pchnięcia w oknie (granica), B = bez pchnięć paczek (uczciwa prognoza),
C = B + pchnięcie paczki bieżącej. Funkcja odpowiedzi R(n) = c(h, h-n).
Wszystko liczone z macierzy Grama zbieranych w jednym przejściu (dopasowanie, test, przeniesienie), osobno dla
całego ruchu, ruchu wspólnego (średnia po sondach) i własnego (odchylenie od niej).

    py -3.11 odpowiedz.py --wyniki ../wyniki            (około 10 minut, około 2 GB RAM)
    py -3.11 odpowiedz.py --wyniki ../wyniki/proba --proba
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

NAG = Path(__file__).resolve().parents[2] / 'maly_vit_rownanie_latentu'
HMAX, K_MU, LAM, BLOK = 50, 101, 1e-6, 50
HS = (1, 2, 3, 5, 10, 20, 50)
I_PED, I_ZANIK, I_ODCH = HMAX, HMAX + 1, HMAX + 2
F = HMAX + 3
FIT = range(1000, 1950)
TEST = range(2000, 2941)


class Nagranie:
    def __init__(self, s, t_od, t_do):
        d = NAG / f'nagranie_s{s}_sily'
        meta = json.loads((d / 'META.json').read_text(encoding='utf-8'))
        self.ob = np.flatnonzero(np.array(meta['obecn']))
        self.od = int(meta['sily_od'])
        lat = np.load(d / 'sondy_lat.npy', mmap_mode='r')
        self.t0 = t_od - K_MU
        self.Z = np.asarray(lat[self.t0:t_do + HMAX + 1, :, 0, :])[:, self.ob].astype(np.float32)   # float16 -> float32 po wyborze sond
        nowa = np.load(d / 'sila_nowa.npy', mmap_mode='r')
        self.ts = t_od
        self.nowa = np.ascontiguousarray(nowa[t_od - self.od:t_do + HMAX - self.od][:, self.ob])     # float16
        self.ped = np.load(d / 'sila_ped.npy', mmap_mode='r')
        self.zanik = np.load(d / 'sila_zanik.npy', mmap_mode='r')

    def x(self, t): return self.Z[t - self.t0]
    def mu(self, t): i = t - self.t0; return self.Z[i + 1 - K_MU:i + 1].mean(0, dtype=np.float64).astype(np.float32)
    def s_nowa(self, t): return self.nowa[t - self.ts].astype(np.float32)
    def s_ped(self, t): return np.asarray(self.ped[t - self.od], dtype=np.float32)[self.ob]
    def s_zanik(self, t): return np.asarray(self.zanik[t - self.od], dtype=np.float32)[self.ob]


def zbierz(N, starty):
    """Macierze Grama cech i celów, w blokach po BLOK startów. Zwraca słownik tablic (bloki, ...)."""
    U = len(N.ob)
    bl = {k: [] for k in ('G', 'B', 'yy', 'Gc', 'Bc', 'yyc', 'n')}
    acc = None
    X = np.empty((F, U, 768), dtype=np.float32)
    for j, t in enumerate(starty):
        if j % BLOK == 0:
            if acc is not None:
                for k in bl: bl[k].append(acc[k])
            acc = dict(G=np.zeros((F, F)), B=np.zeros((F, len(HS))), yy=np.zeros(len(HS)),
                       Gc=np.zeros((F, F)), Bc=np.zeros((F, len(HS))), yyc=np.zeros(len(HS)), n=0)
        for k in range(HMAX):
            X[k] = N.s_nowa(t + k)
        X[I_PED], X[I_ZANIK], X[I_ODCH] = N.s_ped(t), N.s_zanik(t), N.x(t) - N.mu(t)
        Y = np.stack([N.x(t + h) - N.x(t) for h in HS])                                   # (H, U, 768)
        X2, Y2 = X.reshape(F, -1), Y.reshape(len(HS), -1)
        acc['G'] += (X2 @ X2.T).astype(np.float64)
        acc['B'] += (X2 @ Y2.T).astype(np.float64)
        acc['yy'] += np.einsum('hn,hn->h', Y2, Y2, dtype=np.float64)
        Xc, Yc = X.mean(1), Y.mean(1)                                                     # ruch wspólny (F, 768)
        acc['Gc'] += U * (Xc @ Xc.T).astype(np.float64)
        acc['Bc'] += U * (Xc @ Yc.T).astype(np.float64)
        acc['yyc'] += U * np.einsum('hn,hn->h', Yc, Yc, dtype=np.float64)
        acc['n'] += 1
    for k in bl: bl[k].append(acc[k])
    return {k: np.stack(v) for k, v in bl.items()}


def skladowe(S):
    """Sumy po blokach dla: całość, wspólny, własny (własny = całość - wspólny, bo składowe są ortogonalne)."""
    G, B, yy = S['G'].sum(0), S['B'].sum(0), S['yy'].sum(0)
    Gc, Bc, yyc = S['Gc'].sum(0), S['Bc'].sum(0), S['yyc'].sum(0)
    return dict(calosc=(G, B, yy), wspolny=(Gc, Bc, yyc), wlasny=(G - Gc, B - Bc, yy - yyc))


def indeksy(model, h):
    kontr = [I_PED, I_ZANIK, I_ODCH]
    if model == 'A': return list(range(h)) + kontr
    if model == 'C': return [0] + kontr
    return kontr


def dopasuj(G, B, idx, hi):
    g = G[np.ix_(idx, idx)]
    return np.linalg.solve(g + LAM * np.trace(g) / len(idx) * np.eye(len(idx)), B[idx, hi])


def umiejetnosc(beta, idx, hi, G, B, yy):
    sse = yy[hi] - 2 * beta @ B[idx, hi] + beta @ G[np.ix_(idx, idx)] @ beta
    return float(1 - sse / yy[hi])


def odpowiedz(G, B, h):
    """R(n) dla n = 1..h z modelu A o horyzoncie h: R(n) = c(h, h-n)."""
    hi = HS.index(h); idx = indeksy('A', h)
    b = dopasuj(G, B, idx, hi)
    return b[:h][::-1], b[h:]                                                             # R(1..h), [ped, zanik, odch]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wyniki', required=True); p.add_argument('--proba', action='store_true')
    arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    fit, test = (range(1000, 1100), range(2000, 2100)) if arg.proba else (FIT, TEST)
    t0 = time.time()
    N47 = Nagranie(47, fit.start, test.stop - 1)
    print('s47 wczytane', N47.Z.shape, N47.nowa.shape, f'{time.time() - t0:.0f} s', flush=True)
    # skala sił i ruchu (opis)
    probka = list(fit)[::25]
    skala = dict(
        rms_S_nowa=float(np.sqrt(np.mean([np.mean(N47.s_nowa(t) ** 2) for t in probka]))),
        rms_S_ped=float(np.sqrt(np.mean([np.mean(N47.s_ped(t) ** 2) for t in probka]))),
        rms_S_zanik=float(np.sqrt(np.mean([np.mean(N47.s_zanik(t) ** 2) for t in probka]))),
        rms_kroku=float(np.sqrt(np.mean([np.mean((N47.x(t + 1) - N47.x(t)) ** 2) for t in probka]))),
        rms_10_krokow=float(np.sqrt(np.mean([np.mean((N47.x(t + 10) - N47.x(t)) ** 2) for t in probka]))),
        rms_50_krokow=float(np.sqrt(np.mean([np.mean((N47.x(t + 50) - N47.x(t)) ** 2) for t in probka]))),
    )
    Sf = zbierz(N47, fit); print('dopasowanie zebrane', f'{time.time() - t0:.0f} s', flush=True)
    St = zbierz(N47, test); print('test s47 zebrany', f'{time.time() - t0:.0f} s', flush=True)
    del N47
    N48 = Nagranie(48, test.start, test.stop - 1)
    S8 = zbierz(N48, test); print('s48 zebrane', f'{time.time() - t0:.0f} s', flush=True)
    del N48

    # macierze Grama w blokach: wszystkie dalsze warianty regresji da się policzyć z nich bez czytania nagrań
    np.savez_compressed(out / 'GRAMY.npz', **{f'{n}_{k}': v for n, S in (('fit', Sf), ('test47', St), ('s48', S8)) for k, v in S.items()},
                        horyzonty=np.array(HS), hmax=HMAX, sond=180)
    Ff, Ft, F8 = skladowe(Sf), skladowe(St), skladowe(S8)
    wyn = dict(meta=dict(zrodlo=str(NAG), fit=[fit.start, fit.stop - 1], test=[test.start, test.stop - 1], horyzonty=list(HS),
                         hmax=HMAX, k_mu=K_MU, lam=LAM, blok=BLOK, proba=bool(arg.proba),
                         miara='1 - SSE / SSE prognozy bez zmian; 180 sond x 768 wspolrzednych, srednia po tokenach narzedzia',
                         modele=dict(A='wszystkie pchniecia w oknie + ped + zanik + odchylenie (granica)',
                                     B='ped + zanik + odchylenie (uczciwa prognoza)',
                                     C='B + pchniecie paczki biezacej')), skala=skala)
    for skl in ('calosc', 'wspolny', 'wlasny'):
        G, B, yy = Ff[skl]
        r = {}
        um = {m: {} for m in 'ABC'}
        for hi, h in enumerate(HS):
            for m in 'ABC':
                idx = indeksy(m, h); beta = dopasuj(G, B, idx, hi)
                um[m][str(h)] = dict(dopasowanie=umiejetnosc(beta, idx, hi, G, B, yy),
                                     test_s47=umiejetnosc(beta, idx, hi, *Ft[skl]),
                                     s48=umiejetnosc(beta, idx, hi, *F8[skl]))
        for h in HS:
            R, kontr = odpowiedz(G, B, h)
            r[str(h)] = dict(R=R.tolist(), ped=float(kontr[0]), zanik=float(kontr[1]), odchylenie=float(kontr[2]))
        wyn[skl] = dict(umiejetnosc=um, odpowiedz=r, udzial_w_ruchu={str(h): float(yy[i] / Ff['calosc'][2][i]) for i, h in enumerate(HS)})
    # niepewność R(n) z h=50: losowanie bloków startów dopasowania
    rng = np.random.default_rng(20261005); nb = Sf['G'].shape[0]; Rb = []
    for _ in range(500):
        ii = rng.integers(0, nb, nb)
        Rb.append(odpowiedz(Sf['G'][ii].sum(0), Sf['B'][ii].sum(0), HMAX)[0])
    Rb = np.array(Rb)
    wyn['calosc']['odpowiedz_przedzial95_h50'] = dict(dol=np.percentile(Rb, 2.5, 0).tolist(), gora=np.percentile(Rb, 97.5, 0).tolist())
    # funkcja odpowiedzi dopasowana na teście i na s48 (stałość w czasie i między ziarnami)
    wyn['calosc']['odpowiedz_h50_test_s47'] = odpowiedz(*Ft['calosc'][:2], HMAX)[0].tolist()
    wyn['calosc']['odpowiedz_h50_s48'] = odpowiedz(*F8['calosc'][:2], HMAX)[0].tolist()
    wyn['meta']['czas_s'] = time.time() - t0
    (out / 'ODPOWIEDZ.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    R = np.array(wyn['calosc']['odpowiedz']['50']['R'])
    print('skala', {k: round(v, 5) for k, v in skala.items()})
    print('R(n) calosc, n = 1,2,3,5,10,20,50:', [round(float(R[n - 1]), 3) for n in (1, 2, 3, 5, 10, 20, 50)])
    for m in 'ABC':
        print(m, {h: [round(v['test_s47'], 3), round(v['s48'], 3)] for h, v in wyn['calosc']['umiejetnosc'][m].items()})
    print('zapisano', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
