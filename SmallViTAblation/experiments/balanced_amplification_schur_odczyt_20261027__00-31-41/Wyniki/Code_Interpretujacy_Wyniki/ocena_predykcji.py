"""Ocena predykcji BA1–BA6 według progów z Opis/PREDYKCJE_PRZED_POMIAREM.md na 5 biegach. Czyta Wyniki/Dane/pomiar/<bieg>/wyniki.npz,
pisze Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}. Napisany przed odczytem wyników."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[2]
DANE = E / 'Wyniki' / 'Dane'
BIEGI = ['detekcja_s3047', 'detekcja_s4047', 'detekcja_s5047', 'cifar_maly_s47', 'cifar_deit_s47']
DET = BIEGI[:3]
LOS = 1.0 / 192


class Bieg:
    def __init__(self, kat):
        z = np.load(Path(kat) / 'wyniki.npz', allow_pickle=False)
        self.S, self.O, self.R, self.BLK = z['S'], z['O'], z['R'], z['BLK']
        self.O2 = z['O2'] if 'O2' in z else None; self.kO2 = list(z['kol_O2']) if 'kol_O2' in z else []
        self.kS = list(z['kol_S']); self.kO = list(z['kol_O']); self.kR = list(z['kol_R']); self.kB = list(z['kol_BLK'])
        self.kroki = [int(k) for k in z['kroki']]; self.k0, self.kk = self.kroki[0], self.kroki[-1]
        self.NB = int(self.S[:, 2].max()) + 1; self.bst = self.NB // 2

    def s(self, k, blok, m, narz=None):
        w = (self.S[:, 0] == k) & (self.S[:, 2] == blok)
        if narz is not None:
            w &= self.S[:, 4] == (1.0 if narz else 0.0)
        v = self.S[w, self.kS.index(m)]; return float(np.median(v)) if len(v) else float('nan')

    def o(self, k, blok, m):
        w = (self.O[:, 0] == k) & (self.O[:, 2] == blok); v = self.O[w, self.kO.index(m)]; return float(np.median(v)) if len(v) else float('nan')

    def r(self, k, blok, m, narz=None):
        w = (self.R[:, 0] == k) & (self.R[:, 2] == blok)
        if narz is not None:
            w &= self.R[:, 4] == (1.0 if narz else 0.0)
        v = self.R[w, self.kR.index(m)]; return float(np.median(v)) if len(v) else float('nan')

    def o2(self, k, m):
        if self.O2 is None or not len(self.O2):
            return float('nan')
        w = self.O2[:, 0] == k; v = self.O2[w, self.kO2.index(m)]; return float(np.median(v)) if len(v) else float('nan')

    def blk(self, k, blok, m):
        w = (self.BLK[:, 0] == k) & (self.BLK[:, 1] == blok); v = self.BLK[w, self.kB.index(m)]; return float(v[0]) if len(v) else float('nan')

    def kor_zywe_G1(self, k, blok):
        ws = (self.S[:, 0] == k) & (self.S[:, 2] == blok); wr = (self.R[:, 0] == k) & (self.R[:, 2] == blok)
        a = self.R[wr, self.kR.index('zywe')]; b = self.S[ws, self.kS.index('G1')]
        n = min(len(a), len(b)); return float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 3 else float('nan')


def test(wyn, nr, prog, wart):
    n = int(sum(bool(v[0]) for v in wart.values())); wyn[nr] = dict(prog=prog, n=len(wart), zaszlo=n, PASS=n >= prog, per_bieg=wart)


def ocena(B):
    wyn = {}; b = lambda x: x.bst
    test(wyn, 'BA1', 5, {n: (x.s(x.kk, b(x), 'c5') > x.s(x.k0, b(x), 'c5') and x.s(x.kk, b(x), 'PR_N') < x.s(x.k0, b(x), 'PR_N'),
                             x.s(x.k0, b(x), 'c5'), x.s(x.kk, b(x), 'c5'), x.s(x.k0, b(x), 'PR_N'), x.s(x.kk, b(x), 'PR_N')) for n, x in B.items()})
    test(wyn, 'BA2a', 5, {n: (x.s(x.kk, b(x), 'cos_vu') < 0.3, x.s(x.k0, b(x), 'cos_vu'), x.s(x.kk, b(x), 'cos_vu')) for n, x in B.items()})
    test(wyn, 'BA2b', 5, {n: (x.s(x.kk, b(x), 'zawor') < 0.6, x.s(x.k0, b(x), 'zawor'), x.s(x.kk, b(x), 'zawor')) for n, x in B.items()})
    test(wyn, 'BA3a', 5, {n: (x.o(x.kk, b(x), 'cos2_u_r') > x.o(x.kk, b(x), 'cos2_v_r'), x.o(x.kk, b(x), 'cos2_u_r'), x.o(x.kk, b(x), 'cos2_v_r'), x.o(x.kk, b(x), 'cos2_los_r')) for n, x in B.items()})
    test(wyn, 'BA3b', 5, {n: (x.o(x.kk, b(x), 'cos2_u_r') > 3 * LOS and x.o(x.kk, b(x), 'cos2_u_r') > x.o(x.k0, b(x), 'cos2_u_r'),
                              x.o(x.k0, b(x), 'cos2_u_r'), x.o(x.kk, b(x), 'cos2_u_r'), 3 * LOS) for n, x in B.items()})
    test(wyn, 'BA4a', 5, {n: (x.blk(x.kk, b(x), 'nN_nJ_W2W1') > 0.85, x.blk(x.k0, b(x), 'nN_nJ_W2W1'), x.blk(x.kk, b(x), 'nN_nJ_W2W1')) for n, x in B.items()})
    test(wyn, 'BA4b', 5, {n: (abs(x.blk(x.kk, b(x), 'nN_nJ_W2gW1') - x.s(x.kk, b(x), 'nN_nJ')) < 0.05, x.blk(x.kk, b(x), 'nN_nJ_W2gW1'), x.s(x.kk, b(x), 'nN_nJ')) for n, x in B.items()})
    test(wyn, 'BA5a', 5, {n: (x.kor_zywe_G1(x.kk, b(x)) > 0.3, x.kor_zywe_G1(x.k0, b(x)), x.kor_zywe_G1(x.kk, b(x))) for n, x in B.items()})
    test(wyn, 'BA5b', 3, {n: (x.r(x.kk, 3, 'zywe', True) < x.r(x.kk, 3, 'zywe', False), x.r(x.kk, 3, 'zywe', True), x.r(x.kk, 3, 'zywe', False)) for n, x in B.items() if n in DET})
    if 'E_u_ramka' in next(iter(B.values())).kO:
        test(wyn, 'BA3c', 3, {n: (x.o(x.kk, b(x), 'E_u_ramka') > 3 * 4 / 192 and x.o(x.kk, b(x), 'E_u_ramka') > x.o(x.k0, b(x), 'E_u_ramka'),
                                  x.o(x.k0, b(x), 'E_u_ramka'), x.o(x.kk, b(x), 'E_u_ramka'), x.o(x.kk, b(x), 'E_v_ramka'), x.o(x.kk, b(x), 'E_u_glowa'), x.o(x.kk, b(x), 'E_v_glowa'), x.o(x.kk, b(x), 'E_los_glowa')) for n, x in B.items() if n in DET})
        test(wyn, 'BA3c_cifar', 2, {n: (x.o(x.kk, b(x), 'E_u_glowa') > 3 * 10 / 192, x.o(x.k0, b(x), 'E_u_glowa'), x.o(x.kk, b(x), 'E_u_glowa'), x.o(x.kk, b(x), 'E_v_glowa'), x.o(x.kk, b(x), 'E_los_glowa')) for n, x in B.items() if n not in DET})
        test(wyn, 'BA3d', 5, {n: (x.o2(x.kk, 'cos2_u_nextv') > 3 * LOS and x.o2(x.kk, 'cos2_u_nextv') > x.o2(x.k0, 'cos2_u_nextv'),
                                  x.o2(x.k0, 'cos2_u_nextv'), x.o2(x.kk, 'cos2_u_nextv'), x.o2(x.kk, 'cos2_v_nextv'), x.o2(x.kk, 'cos2_u_nextu')) for n, x in B.items()})
    test(wyn, 'BA6', 5, {n: (x.blk(x.kk, b(x), 'cos_w1w2_waz') < 0.2 and x.blk(x.kk, b(x), 'cos_w1w2_waz') <= 2 * x.blk(x.k0, b(x), 'cos_w1w2_waz'),
                             x.blk(x.k0, b(x), 'cos_w1w2_waz'), x.blk(x.kk, b(x), 'cos_w1w2_waz'), x.blk(x.kk, b(x), 'cos_w1w2_sr')) for n, x in B.items()})
    return wyn


def main():
    B = {}
    for n in BIEGI:
        k = DANE / 'pomiar' / n
        if (k / 'wyniki.npz').exists() and (k / 'COMPLETE.json').exists():
            B[n] = Bieg(k)
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
