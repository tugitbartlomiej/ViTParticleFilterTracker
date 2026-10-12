"""Odpowiedź latentu na pchnięcie bieżącej paczki w długim horyzoncie (do 500 kroków), projekcją lokalną.

Analiza dodatkowa po wyniku (AMENDMENT_DLUGI_HORYZONT.md). Pierwszy bieg (`odpowiedz.py`) pokazał, że współczynniki
modelu A przy późnych pchnięciach niosą sprzężenie zwrotne, więc funkcję odpowiedzi czytam z modelu C:

    z(t+h) - z(t) = R(h) S_nowa(t) + a(h) S_ped(t) + w(h) S_zanik(t) + g(h) (z(t) - mu(t)) + p(h) S_nowa(t-100) + reszta

Ostatni człon to placebo: pchnięcie paczki sprzed 100 kroków, policzone w tamtym stanie. Jego współczynnik powinien być
bliski zera. Wszystkie regresory są znane w chwili t (paczka bieżąca jest tą, którą właśnie podajemy).

Trzy zbiory startów, każdy dopasowany osobno (stałość w czasie i między ziarnami):
    s47_a: ziarno 47, starty 1000-1749     s47_b: ziarno 47, starty 1750-2499     s48: ziarno 48, starty 2000-2499

    py -3.11 odpowiedz_dluga.py --wyniki ../wyniki/dluga         (kilkanaście minut na dysku talerzowym, około 1,5 GB RAM)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

NAG = Path(__file__).resolve().parents[2] / 'maly_vit_rownanie_latentu'
HS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500)
K_MU, LAM, BLOK, PLACEBO = 101, 1e-6, 50, 100
CECHY = ('pchniecie', 'ped', 'zanik', 'odchylenie', 'placebo')
ZBIORY = {'s47_a': (47, 1000, 1750), 's47_b': (47, 1750, 2500), 's48': (48, 2000, 2500)}


class Nagranie:
    def __init__(self, s, t_od, t_do):
        d = NAG / f'nagranie_s{s}_sily'
        meta = json.loads((d / 'META.json').read_text(encoding='utf-8'))
        self.ob = np.flatnonzero(np.array(meta['obecn']))
        self.od = int(meta['sily_od'])
        lat = np.load(d / 'sondy_lat.npy', mmap_mode='r')
        self.t0 = t_od - K_MU
        t1 = min(t_do + HS[-1], lat.shape[0])
        self.Z = np.concatenate([np.asarray(lat[a:min(a + 64, t1), :, 0, :])[:, self.ob] for a in range(self.t0, t1, 64)])   # float16
        self.sily = {n: np.load(d / f'sila_{n}.npy', mmap_mode='r') for n in ('nowa', 'ped', 'zanik')}
        self.placebo = t_od - PLACEBO >= self.od

    def x(self, t): return self.Z[t - self.t0].astype(np.float32)
    def mu(self, t): i = t - self.t0; return self.Z[i + 1 - K_MU:i + 1].mean(0, dtype=np.float32)
    def blok(self, n, a, b): return np.asarray(self.sily[n][a - self.od:b - self.od])[:, self.ob].astype(np.float32)


def zbierz(N, t_od, t_do):
    U, F, H = len(N.ob), len(CECHY), len(HS)
    bl = {k: [] for k in ('G', 'B', 'yy', 'Gc', 'Bc', 'yyc')}
    X = np.zeros((F, U, 768), dtype=np.float32)
    for a in range(t_od, t_do, BLOK):
        b = min(a + BLOK, t_do)
        nowa, ped, zanik = N.blok('nowa', a, b), N.blok('ped', a, b), N.blok('zanik', a, b)
        plac = N.blok('nowa', a - PLACEBO, b - PLACEBO) if N.placebo else None
        acc = dict(G=np.zeros((F, F)), B=np.zeros((F, H)), yy=np.zeros(H), Gc=np.zeros((F, F)), Bc=np.zeros((F, H)), yyc=np.zeros(H))
        for j, t in enumerate(range(a, b)):
            x = N.x(t)
            X[0], X[1], X[2], X[3] = nowa[j], ped[j], zanik[j], x - N.mu(t)
            if plac is not None: X[4] = plac[j]
            Y = np.stack([N.x(t + h) - x for h in HS])
            X2, Y2 = X.reshape(F, -1), Y.reshape(H, -1)
            acc['G'] += (X2 @ X2.T).astype(np.float64); acc['B'] += (X2 @ Y2.T).astype(np.float64)
            acc['yy'] += np.einsum('hn,hn->h', Y2, Y2, dtype=np.float64)
            Xc, Yc = X.mean(1), Y.mean(1)
            acc['Gc'] += U * (Xc @ Xc.T).astype(np.float64); acc['Bc'] += U * (Xc @ Yc.T).astype(np.float64)
            acc['yyc'] += U * np.einsum('hn,hn->h', Yc, Yc, dtype=np.float64)
        for k in bl: bl[k].append(acc[k])
    return {k: np.stack(v) for k, v in bl.items()}


def skladowe(S, ii=None):
    s = (lambda v: v.sum(0)) if ii is None else (lambda v: v[ii].sum(0))
    G, B, yy, Gc, Bc, yyc = (s(S[k]) for k in ('G', 'B', 'yy', 'Gc', 'Bc', 'yyc'))
    return dict(calosc=(G, B, yy), wspolny=(Gc, Bc, yyc), wlasny=(G - Gc, B - Bc, yy - yyc))


def dopasuj(G, B, idx):
    g = G[np.ix_(idx, idx)]
    return np.linalg.solve(g + LAM * np.trace(g) / len(idx) * np.eye(len(idx)), B[idx])          # (len(idx), H)


def umiej(beta, idx, G, B, yy):
    sse = yy - 2 * np.einsum('fh,fh->h', beta, B[idx]) + np.einsum('fh,fg,gh->h', beta, G[np.ix_(idx, idx)], beta)
    return 1 - sse / yy


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); S = {}
    N = Nagranie(47, 1000, 2500); print('s47 wczytane', N.Z.shape, f'{time.time() - t0:.0f} s', flush=True)
    S['s47_a'] = zbierz(N, 1000, 1750); print('s47_a zebrane', f'{time.time() - t0:.0f} s', flush=True)
    S['s47_b'] = zbierz(N, 1750, 2500); print('s47_b zebrane', f'{time.time() - t0:.0f} s', flush=True)
    del N
    N = Nagranie(48, 2000, 2500); print('s48 wczytane', N.Z.shape, 'placebo', N.placebo, f'{time.time() - t0:.0f} s', flush=True)
    S['s48'] = zbierz(N, 2000, 2500); print('s48 zebrane', f'{time.time() - t0:.0f} s', flush=True)
    del N
    np.savez_compressed(out / 'GRAMY_DLUGA.npz', **{f'{n}_{k}': v for n, d in S.items() for k, v in d.items()}, horyzonty=np.array(HS))

    wyn = dict(meta=dict(zrodlo=str(NAG), horyzonty=list(HS), cechy=list(CECHY), zbiory={k: list(v) for k, v in ZBIORY.items()},
                         k_mu=K_MU, blok=BLOK, placebo_przesuniecie=PLACEBO,
                         opis='projekcja lokalna: wspolczynniki dopasowane osobno na kazdym zbiorze; umiejetnosc = 1 - SSE/SSE bez zmian'))
    rng = np.random.default_rng(20261005)
    C, Bm = [0, 1, 2, 3], [1, 2, 3]
    for skl in ('calosc', 'wspolny', 'wlasny'):
        r = {}
        for n, d in S.items():
            G, B, yy = skladowe(d)[skl]
            idx = C + ([4] if n != 's48' else [])
            bC, bB = dopasuj(G, B, idx), dopasuj(G, B, Bm)
            r[n] = dict(pchniecie=bC[0].tolist(), ped=bC[1].tolist(), odchylenie=bC[3].tolist(),
                        placebo=(bC[4].tolist() if len(idx) == 5 else None),
                        umiejetnosc_C=umiej(bC, idx, G, B, yy).tolist(), umiejetnosc_B=umiej(bB, Bm, G, B, yy).tolist(),
                        rms_przesuniecia=np.sqrt(yy / (d['G'].shape[0] * BLOK * 180 * 768)).tolist())
        # przeniesienie: współczynniki z s47_a oceniane na s47_b i s48 (bez placebo, żeby zestaw cech był wspólny)
        G, B, yy = skladowe(S['s47_a'])[skl]; bC, bB = dopasuj(G, B, C), dopasuj(G, B, Bm)
        r['przeniesienie_z_s47_a'] = {n: dict(umiejetnosc_C=umiej(bC, C, *skladowe(S[n])[skl]).tolist(),
                                              umiejetnosc_B=umiej(bB, Bm, *skladowe(S[n])[skl]).tolist()) for n in ('s47_b', 's48')}
        # przedział z losowania bloków po 50 startów (dla h > 50 zaniżony: okna celów zachodzą na siebie)
        boot = []
        d = {k: np.concatenate([S['s47_a'][k], S['s47_b'][k]]) for k in S['s47_a']}
        nb = d['G'].shape[0]
        for _ in range(500):
            G, B, _ = skladowe(d, rng.integers(0, nb, nb))[skl]
            boot.append(dopasuj(G, B, C)[:2])
        boot = np.array(boot)
        G, B, yy = skladowe(d)[skl]; bC = dopasuj(G, B, C)
        r['s47_razem'] = dict(pchniecie=bC[0].tolist(), ped=bC[1].tolist(), odchylenie=bC[3].tolist(),
                              pchniecie_przedzial95=[np.percentile(boot[:, 0], 2.5, 0).tolist(), np.percentile(boot[:, 0], 97.5, 0).tolist()],
                              ped_przedzial95=[np.percentile(boot[:, 1], 2.5, 0).tolist(), np.percentile(boot[:, 1], 97.5, 0).tolist()])
        wyn[skl] = r
    wyn['meta']['czas_s'] = time.time() - t0
    (out / 'ODPOWIEDZ_DLUGA.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print('h        ', list(HS))
    for n in ('s47_a', 's47_b', 's48'):
        print(f'{n} R  ', np.round(wyn['calosc'][n]['pchniecie'], 2).tolist())
    print('s47 dol  ', np.round(wyn['calosc']['s47_razem']['pchniecie_przedzial95'][0], 2).tolist())
    print('s47 gora ', np.round(wyn['calosc']['s47_razem']['pchniecie_przedzial95'][1], 2).tolist())
    print('placebo a', np.round(wyn['calosc']['s47_a']['placebo'], 2).tolist())
    print('ped s47_a', np.round(wyn['calosc']['s47_a']['ped'], 2).tolist())
    print('C-B s47_a', np.round(np.array(wyn['calosc']['s47_a']['umiejetnosc_C']) - np.array(wyn['calosc']['s47_a']['umiejetnosc_B']), 4).tolist())
    print('zapisano', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
