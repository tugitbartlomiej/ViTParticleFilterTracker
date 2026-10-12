"""Analiza dodatkowa po wyniku, z zapisanych macierzy Grama (`../wyniki/GRAMY.npz`), bez czytania nagrań.

Opisowa, bez kryterium ustalonego z góry (PLAN.md jej nie przewidywał). Rozstrzyga, skąd bierze się ujemny
współczynnik przy ostatnim pchnięciu w modelu A i ile wynosi odpowiedź na pchnięcie liczona projekcją lokalną.

1. Projekcja lokalna: współczynnik przy S_nowa(t) w modelu C (pchnięcie bieżące + pęd + zanik + odchylenie) dla
   każdego horyzontu. W odróżnieniu od modelu A nie zawiera późniejszych pchnięć, które same zależą od stanu.
2. Sprzężenie zwrotne: korelacja pchnięcia w kroku t+k z przesunięciem latentu, które zaszło PRZED tym krokiem.
3. Autokorelacja pchnięć i ich związek z pędem oraz odchyleniem od średniej z chwili t.
4. Ile pchnięcia da się przewidzieć z tego, co znane przed losowaniem paczki.

    py -3.11 analiza_gramow.py --wyniki ../wyniki
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from odpowiedz import HMAX, HS, I_ODCH, I_PED, I_ZANIK, LAM, skladowe


def rozw(G, b, idx):
    g = G[np.ix_(idx, idx)]
    return np.linalg.solve(g + LAM * np.trace(g) / len(idx) * np.eye(len(idx)), b[idx])


def projekcja_lokalna(G, B):
    """Współczynniki modelu C dla każdego horyzontu: [pchnięcie bieżące, pęd, zanik, odchylenie]."""
    idx = [0, I_PED, I_ZANIK, I_ODCH]
    return np.array([rozw(G, B[:, hi], idx) for hi in range(len(HS))])


def kor(G, i, j):
    return float(G[i, j] / np.sqrt(G[i, i] * G[j, j]))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki)
    z = np.load(out / 'GRAMY.npz')
    zb = {n: {k: z[f'{n}_{k}'] for k in ('G', 'B', 'yy', 'Gc', 'Bc', 'yyc', 'n')} for n in ('fit', 'test47', 's48')}
    wyn = dict(meta=dict(zrodlo=str(out / 'GRAMY.npz'), horyzonty=list(HS),
                         opis='analiza dodatkowa po wyniku, opisowa; projekcja lokalna = model C'))

    # 1. projekcja lokalna na trzech zbiorach i dla trzech składowych ruchu, z przedziałem z losowania bloków
    rng = np.random.default_rng(20261005)
    pl = {}
    for skl in ('calosc', 'wspolny', 'wlasny'):
        r = {}
        for n, S in zb.items():
            G, B, _ = skladowe(S)[skl]
            c = projekcja_lokalna(G, B)
            r[n] = dict(pchniecie=c[:, 0].tolist(), ped=c[:, 1].tolist(), odchylenie=c[:, 3].tolist())
        S = zb['fit']; nb = S['G'].shape[0]; boot = []
        for _ in range(500):
            ii = rng.integers(0, nb, nb)
            Sb = {k: v[ii] for k, v in S.items()}
            G, B, _ = skladowe(Sb)[skl]
            boot.append(projekcja_lokalna(G, B))
        boot = np.array(boot)
        r['fit_przedzial95'] = dict(pchniecie=[np.percentile(boot[:, :, 0], 2.5, 0).tolist(), np.percentile(boot[:, :, 0], 97.5, 0).tolist()],
                                    ped=[np.percentile(boot[:, :, 1], 2.5, 0).tolist(), np.percentile(boot[:, :, 1], 97.5, 0).tolist()])
        pl[skl] = r
    wyn['projekcja_lokalna'] = pl

    # 2-4. korelacje na zbiorze dopasowania (całość) i na s48
    for n in ('fit', 's48'):
        G, B, yy = skladowe(zb[n])['calosc']
        d = {}
        # autokorelacja pchnięć: średnia po parach (k, k+op) w oknie
        d['autokorelacja_pchniec'] = {str(op): float(np.mean([kor(G, k, k + op) for k in range(HMAX - op)])) for op in (1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40)}
        d['ped_t_a_pchniecie_t_plus_k'] = {str(k): kor(G, I_PED, k) for k in (0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 49)}
        d['odchylenie_t_a_pchniecie_t_plus_k'] = {str(k): kor(G, I_ODCH, k) for k in (0, 1, 2, 3, 5, 10, 20, 30, 49)}
        d['odchylenie_t_a_ped_t'] = kor(G, I_ODCH, I_PED)
        # sprzężenie: pchnięcie w kroku t+k a przesunięcie z(t+h) - z(t); k >= h znaczy "pchnięcie po oknie"
        sp = {}
        for hi, h in enumerate(HS):
            sp[str(h)] = {str(k): float(B[k, hi] / np.sqrt(G[k, k] * yy[hi])) for k in sorted({0, h - 1, h, h + 1, h + 2, h + 4, h + 9, h + 19} & set(range(HMAX)))}
        d['pchniecie_t_plus_k_a_przesuniecie_h'] = sp
        d['ped_t_a_przesuniecie_h'] = {str(h): float(B[I_PED, hi] / np.sqrt(G[I_PED, I_PED] * yy[hi])) for hi, h in enumerate(HS)}
        d['odchylenie_t_a_przesuniecie_h'] = {str(h): float(B[I_ODCH, hi] / np.sqrt(G[I_ODCH, I_ODCH] * yy[hi])) for hi, h in enumerate(HS)}
        # ile pchnięcia jest wyznaczone przez to, co znane wcześniej
        def r2(cel, idx):
            b = rozw(G, G[:, cel], idx)
            return float((2 * b @ G[idx, cel] - b @ G[np.ix_(idx, idx)] @ b) / G[cel, cel])
        d['r2_pchniecia_t_ze_stanu_t'] = r2(0, [I_PED, I_ZANIK, I_ODCH])
        d['r2_pchniecia_t49_z_wczesniejszych_pchniec_i_stanu_t'] = r2(HMAX - 1, list(range(HMAX - 1)) + [I_PED, I_ZANIK, I_ODCH])
        d['r2_pchniecia_t9_z_wczesniejszych_pchniec_i_stanu_t'] = r2(9, list(range(9)) + [I_PED, I_ZANIK, I_ODCH])
        d['r2_pchniecia_t1_z_pchniecia_t_i_stanu_t'] = r2(1, [0, I_PED, I_ZANIK, I_ODCH])
        d['wariancje'] = dict(pchniecie=float(G[0, 0]), ped=float(G[I_PED, I_PED]), odchylenie=float(G[I_ODCH, I_ODCH]),
                              przesuniecie={str(h): float(yy[hi]) for hi, h in enumerate(HS)})
        wyn[f'korelacje_{n}'] = d
    (out / 'ANALIZA_GRAMOW.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')

    f = pl['calosc']
    print('projekcja lokalna, calosc (horyzonty', list(HS), ')')
    for n in ('fit', 'test47', 's48'):
        print(f'  {n:7s} pchniecie', np.round(f[n]['pchniecie'], 3).tolist(), ' ped', np.round(f[n]['ped'], 3).tolist())
    print('  przedzial pchniecie', np.round(f['fit_przedzial95']['pchniecie'], 3).tolist())
    for skl in ('wspolny', 'wlasny'):
        print(f'  {skl} fit pchniecie', np.round(pl[skl]['fit']['pchniecie'], 3).tolist(), ' ped', np.round(pl[skl]['fit']['ped'], 3).tolist())
    k = wyn['korelacje_fit']
    print('autokorelacja pchniec', {a: round(b, 3) for a, b in k['autokorelacja_pchniec'].items()})
    print('ped(t) a pchniecie(t+k)', {a: round(b, 3) for a, b in k['ped_t_a_pchniecie_t_plus_k'].items()})
    print('odchylenie(t) a pchniecie(t+k)', {a: round(b, 3) for a, b in k['odchylenie_t_a_pchniecie_t_plus_k'].items()})
    print('pchniecie(t+k) a przesuniecie(h)', {h: {a: round(b, 3) for a, b in v.items()} for h, v in k['pchniecie_t_plus_k_a_przesuniecie_h'].items()})
    print('R2 pchniecia ze stanu', round(k['r2_pchniecia_t_ze_stanu_t'], 3), ' t+1 z t i stanu', round(k['r2_pchniecia_t1_z_pchniecia_t_i_stanu_t'], 3),
          ' t+9', round(k['r2_pchniecia_t9_z_wczesniejszych_pchniec_i_stanu_t'], 3), ' t+49', round(k['r2_pchniecia_t49_z_wczesniejszych_pchniec_i_stanu_t'], 3))


if __name__ == '__main__':
    main()
