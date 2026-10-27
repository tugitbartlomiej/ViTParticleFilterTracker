"""Ocena predykcji W1–W6 i F1–F5 według progów z Opis/PREDYKCJE_PRZED_POMIAREM.md na 5 biegach (detekcja s3047/s4047/s5047,
cifar_maly_s47, cifar_deit_s47). Czyta Wyniki/Dane/pomiar/<bieg>/wyniki.npz, pisze Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}.
Napisany przed odczytem wyników."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[2]
DANE = E / 'Wyniki' / 'Dane'
BIEGI = ['detekcja_s3047', 'detekcja_s4047', 'detekcja_s5047', 'cifar_maly_s47', 'cifar_deit_s47']


class Bieg:
    def __init__(self, kat):
        z = np.load(Path(kat) / 'wyniki.npz', allow_pickle=False)
        self.W, self.P, self.F = z['W'], z['P'], z['F']
        if (Path(kat) / 'wyniki_F.npz').exists():                      # część F przeliczona po poprawce (stan 2b+1)
            self.F = np.load(Path(kat) / 'wyniki_F.npz', allow_pickle=False)['F']
        self.kW = list(z['kol_W']); self.kP = list(z['kol_P']); self.kF = list(z['kol_F'])
        self.kroki = [int(k) for k in z['kroki']]; self.k0, self.k1, self.kk = self.kroki[0], self.kroki[1], self.kroki[-1]
        self.NB = int(self.W[:, 2].max()) + 1; self.bst = self.NB // 2                      # blok środkowy (indeks 0-based)

    def w(self, k, blok, miara):
        v = self.W[(self.W[:, 0] == k) & (self.W[:, 2] == blok), self.kW.index(miara)]
        return float(np.median(v)) if len(v) else float('nan')

    def w_bloki(self, k, miara):
        return [self.w(k, b, miara) for b in range(self.NB)]

    def gmax(self, k, blok):
        return max(self.w(k, blok, f'G{i}') for i in range(1, 7))

    def p(self, k, miara):
        v = self.P[self.P[:, 0] == k, self.kP.index(miara)]; return float(np.median(v)) if len(v) else float('nan')

    def f(self, k, blok, delta, rodzaj, miara, abs_=False):
        m = (self.F[:, 0] == k) & (self.F[:, 2] == blok) & (np.isclose(self.F[:, 3], delta)) & (self.F[:, 4] == rodzaj)
        v = self.F[m, self.kF.index(miara)]; v = np.abs(v) if abs_ else v
        return float(np.nanmedian(v)) if len(v) else float('nan')


def test(wyn, nr, prog, wart):
    n = int(sum(bool(v[0]) for v in wart.values())); wyn[nr] = dict(prog=prog, n=len(wart), zaszlo=n, PASS=n >= prog, per_bieg=wart)


def ocena(B):
    wyn = {}
    g = lambda x, k: x.w(k, x.bst, 'G1')
    test(wyn, 'W1', 5, {b: (g(x, x.kk) > g(x, x.k0) and g(x, x.kk) > 1.5, g(x, x.k0), g(x, x.kk)) for b, x in B.items()})
    test(wyn, 'W2', 5, {b: (x.gmax(x.kk, x.bst) >= 2.0, x.gmax(x.kk, x.bst), [round(x.w(x.kk, x.bst, f'G{i}'), 3) for i in range(1, 7)]) for b, x in B.items()})
    test(wyn, 'W3', 5, {b: (x.w(x.kk, x.bst, 'kreiss') > x.w(x.k0, x.bst, 'kreiss'), x.w(x.k0, x.bst, 'kreiss'), x.w(x.kk, x.bst, 'kreiss')) for b, x in B.items()})
    luka = lambda x, k: x.w(k, x.bst, 'omega') - x.w(k, x.bst, 'maxRe')
    test(wyn, 'W4', 5, {b: (luka(x, x.kk) > luka(x, x.k0), luka(x, x.k0), luka(x, x.kk)) for b, x in B.items()})
    test(wyn, 'W5', 5, {b: (x.p(x.kk, 'G_P') > float(np.median(x.w_bloki(x.kk, 'G1'))), x.p(x.kk, 'G_P'), float(np.median(x.w_bloki(x.kk, 'G1')))) for b, x in B.items()})
    test(wyn, 'W6', 5, {b: (1.1 <= g(x, x.k0) <= 1.6, g(x, x.k0)) for b, x in B.items()})
    def f1(x):
        a = x.f(x.kk, x.bst, 0.01, 0, 'amp_med'); pn = x.f(x.kk, x.bst, 0.01, 0, 'pred_nn_med'); pr = x.f(x.kk, x.bst, 0.01, 0, 'pred_norm_med')
        return (abs(a / pn - 1) < 0.10 and pn / pr > 1.15, a, pn, pr)
    test(wyn, 'F1', 5, {b: f1(x) for b, x in B.items()})
    def f2(x):
        a = x.f(x.kk, x.bst, 0.01, 1, 'amp_med'); s = x.f(x.kk, x.bst, 0.01, 1, 'pred_nn_med'); return (abs(a / s - 1) < 0.20, a, s)
    test(wyn, 'F2', 5, {b: f2(x) for b, x in B.items()})
    def f3(x):
        n = x.f(x.kk, x.bst, 0.03, 1, 'wyj_delta', True); l = x.f(x.kk, x.bst, 0.03, 0, 'wyj_delta', True); return (n > 2 * l, l, n)
    test(wyn, 'F3', 5, {b: f3(x) for b, x in B.items()})
    def f4(x):
        v = [x.f(x.kk, b, 0.1, 0, 'wyj_delta', True) for b in range(x.NB)]; return (int(np.nanargmax(v)) == 0, [round(t, 4) for t in v])
    test(wyn, 'F4', 5, {b: f4(x) for b, x in B.items()})
    def f5(x):
        return (x.f(x.kk, x.bst, 0.03, 1, 'wyj_delta', True) > x.f(x.k1, x.bst, 0.03, 1, 'wyj_delta', True),
                x.f(x.k1, x.bst, 0.03, 1, 'wyj_delta', True), x.f(x.kk, x.bst, 0.03, 1, 'wyj_delta', True))
    test(wyn, 'F5', 5, {b: f5(x) for b, x in B.items()})
    return wyn


def main():
    B = {}
    for b in BIEGI:
        k = DANE / 'pomiar' / b
        if (k / 'wyniki.npz').exists() and (k / 'COMPLETE.json').exists():
            B[b] = Bieg(k)
    wyn = dict(biegi=sorted(B))
    if B:
        wyn['ocena'] = ocena(B)
    def konw(o):
        if isinstance(o, dict):
            return {str(k): konw(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [konw(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    (DANE / 'OCENA_PREDYKCJI.json').write_text(json.dumps(konw(wyn), indent=1, ensure_ascii=False), encoding='utf-8')
    linie = [f'biegi: {len(B)}'] + [f"{nr}: {'PASS' if r['PASS'] else 'FAIL'} ({r['zaszlo']}/{r['n']}, próg {r['prog']})" for nr, r in wyn.get('ocena', {}).items()]
    (DANE / 'OCENA_PREDYKCJI.txt').write_text('\n'.join(linie) + '\n', encoding='utf-8')
    print('\n'.join(linie))


if __name__ == '__main__':
    sys.exit(main())
