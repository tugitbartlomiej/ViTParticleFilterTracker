"""Ocena predykcji R1–R2, Z1–Z2, S1–S2, T1, K1–K3 według progów z Opis/PREDYKCJE_PRZED_POMIAREM.md. Czyta
Wyniki/Dane/pomiar/<bieg>/wyniki.npz (bieg liczony, gdy jest też COMPLETE.json), pisze Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}.
Napisany przed odczytem wyników (2026-10-27, 02:55), w czasie próby dymnej.

Statystyka M[·]: mediana po blokach b = 0…NB−2 z median po tokenach i klatkach w bloku b, na danym checkpoincie.
Nieliniowo: F_X(blok) = mediana_tokenów(dy_pelny − dy_bez_X) / mediana_tokenów|dy_pelny| (δ = 0,03, chyba że podano), M = mediana po blokach;
M|F_X| = mediana po blokach z |F_X(blok)|.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[2]
DANE = E / 'Wyniki' / 'Dane'
RODZIC_OCENA = E.parent / 'odbiorca_zapisu_mlp_uwaga_20261027__01-55-25' / 'Wyniki' / 'Dane' / 'OCENA_PREDYKCJI.json'
BIEGI = ['detekcja_s3047', 'detekcja_s4047', 'detekcja_s5047', 'cifar_maly_s47', 'cifar_deit_s47']
DET = BIEGI[:3]
PROG = 4
KIER = ['v1', 'r', 'los']


def med(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float('nan')


class Bieg:
    def __init__(self, kat):
        z = np.load(Path(kat) / 'wyniki.npz', allow_pickle=False)
        self.L, self.S, self.B = z['L'], z['S'], z['B']
        self.kL, self.kS, self.kB = list(z['kol_L']), list(z['kol_S']), list(z['kol_B'])
        self.kroki = [int(k) for k in z['kroki']]; self.k0, self.kk = self.kroki[0], self.kroki[-1]
        self.bloki = sorted({int(b) for b in self.L[:, 2]})

    def _maska(self, M, kol, k, b, narz=None, kier=None, delta=None):
        w = (M[:, 0] == k) & (M[:, 2] == b)
        if narz is not None:
            w &= M[:, kol.index('narz')] == (1.0 if narz else 0.0)
        if kier is not None:
            w &= M[:, kol.index('kier')] == KIER.index(kier)
        if delta is not None:
            w &= np.isclose(M[:, kol.index('delta')], delta)
        return w

    def mL(self, k, c, narz=None):
        return med([med(self.L[self._maska(self.L, self.kL, k, b, narz), self.kL.index(c)]) for b in self.bloki])

    def mS(self, k, c, kier, narz=None, f=None):
        out = []
        for b in self.bloki:
            v = self.S[self._maska(self.S, self.kS, k, b, narz, kier), self.kS.index(c)]
            out.append(med(f(v) if f else v))
        return med(out)

    def frac_znak(self, k, c, kier, znak):
        v = self.S[(self.S[:, 0] == k) & (self.S[:, self.kS.index('kier')] == KIER.index(kier)), self.kS.index(c)]
        return float(np.mean(np.sign(v) == znak)) if len(v) else float('nan')

    def F_blok(self, k, b, war, kier, delta=0.03, narz=None):
        w = self._maska(self.B, self.kB, k, b, narz, kier, delta)
        p = self.B[w, self.kB.index('dy_pelny')]; q = self.B[w, self.kB.index(f'dy_{war}')]
        mian = med(np.abs(p))
        return med(p - q) / mian if mian and np.isfinite(mian) else float('nan')

    def mF(self, k, war, kier, delta=0.03, narz=None, absol=False):
        v = [self.F_blok(k, b, war, kier, delta, narz) for b in self.bloki]
        return med(np.abs(v) if absol else v)

    def max_F_mean(self):
        p = self.B[:, self.kB.index('dy_pelny')]; q = self.B[:, self.kB.index('dy_bez_mean')]
        return float(np.max(np.abs(p - q)) / max(np.median(np.abs(p)), 1e-300))


def test(wyn, nr, prog, wart):
    n = int(sum(bool(v[0]) for v in wart.values()))
    wyn[nr] = dict(prog=prog, n=len(wart), zaszlo=n, PASS=n >= prog, sila=('silnie' if n == len(wart) else 'słabo') if n >= prog else '',
                   per_bieg=wart)


def ocena(B, rodzic):
    wyn = {}
    test(wyn, 'R1', PROG, {n: (x.mL(x.kk, 'E_rad_u1') > x.mL(x.kk, 'E_mean_u1'), x.mL(x.kk, 'E_rad_u1'), x.mL(x.kk, 'E_mean_u1'),
                               x.mL(x.kk, 'E_rad_los'), x.mL(x.kk, 'E_mean_los')) for n, x in B.items()})
    test(wyn, 'R2', PROG, {n: (x.mL(x.kk, 'E_rad_u1') > 3 * x.mL(x.kk, 'E_rad_los') and x.mL(x.kk, 'E_rad_u1') > x.mL(x.k0, 'E_rad_u1'),
                               x.mL(x.k0, 'E_rad_u1'), x.mL(x.kk, 'E_rad_u1'), x.mL(x.kk, 'E_rad_los')) for n, x in B.items()})
    znaki = {n: np.sign(x.mS(x.kk, 'c_rad', 'v1')) for n, x in B.items()}
    wiek = np.sign(sum(znaki.values())) or 1.0
    test(wyn, 'Z1', PROG, {n: (znaki[n] == wiek and x.frac_znak(x.kk, 'c_rad', 'v1', wiek) > 0.6, x.mS(x.kk, 'c_rad', 'v1'),
                               x.frac_znak(x.kk, 'c_rad', 'v1', wiek), float(wiek), x.mS(x.kk, 'c_rad', 'los')) for n, x in B.items()})
    test(wyn, 'Z2', PROG, {n: (x.mS(x.kk, 'c_rad', 'r') > 0, x.mS(x.k0, 'c_rad', 'r'), x.mS(x.kk, 'c_rad', 'r'), x.mS(x.kk, 'c_rad', 'los'))
                           for n, x in B.items()})
    test(wyn, 'S1', PROG, {n: (x.mF(x.kk, 'bez_rad', 'r') < 0 and x.mF(x.kk, 'bez_rad', 'v1') < 0, x.mF(x.kk, 'bez_rad', 'r'),
                               x.mF(x.kk, 'bez_rad', 'v1'), x.mF(x.kk, 'bez_rad', 'los'), x.mF(x.kk, 'bez_rad', 'r', 0.1)) for n, x in B.items()})

    def udz(x):
        a, p = x.mF(x.kk, 'bez_rad', 'r', absol=True), x.mF(x.kk, 'bez_perp', 'r', absol=True)
        return a / (a + p) if (a + p) > 0 else float('nan')
    test(wyn, 'S2', PROG, {n: (udz(x) > 0.1, udz(x), x.mF(x.kk, 'bez_rad', 'r', absol=True), x.mF(x.kk, 'bez_perp', 'r', absol=True),
                               x.mF(x.kk, 'bez_mlp', 'r')) for n, x in B.items()})
    test(wyn, 'T1', 3, {n: (x.mS(x.kk, 'eta', 'r', True, np.abs) < x.mS(x.kk, 'eta', 'r', False, np.abs), x.mS(x.kk, 'eta', 'r', True, np.abs),
                            x.mS(x.kk, 'eta', 'r', False, np.abs)) for n, x in B.items() if n in DET})
    K1 = {}
    for n, x in B.items():
        s = med([med(x.L[x._maska(x.L, x.kL, x.kk, b), x.kL.index('E_rad_u1')] + x.L[x._maska(x.L, x.kL, x.kk, b), x.kL.index('E_mean_u1')])
                 for b in x.bloki])
        r = rodzic.get(n, float('nan'))
        K1[n] = (bool(abs(s - r) < 0.05), s, r)
    test(wyn, 'K1', len(B), K1)
    test(wyn, 'K2', len(B), {n: (float(np.max(x.S[:, x.kS.index('rc_mean')])) < 1e-6 and x.max_F_mean() < 1e-6,
                                 float(np.max(x.S[:, x.kS.index('rc_mean')])), x.max_F_mean()) for n, x in B.items()})
    test(wyn, 'K3', PROG, {n: (np.sign(x.mS(x.kk, 'F_rad', 'r')) == np.sign(x.mF(x.kk, 'bez_rad', 'r')), x.mS(x.kk, 'F_rad', 'r'),
                               x.mF(x.kk, 'bez_rad', 'r')) for n, x in B.items()})
    return wyn


def opis(B):
    o = {}
    for n, x in B.items():
        r = {}
        for k in x.kroki:
            s = {}
            for g, nz in (('wszystkie', None), ('narz', True), ('tlo', False)):
                if nz is not None and n not in DET:
                    continue
                d = {c: x.mL(k, c, nz) for c in x.kL[5:]}
                for kn in KIER:
                    for c in ('c_rad', 'c_mean', 'eta', 'dy_lin', 'F_id', 'F_rad', 'F_perp'):
                        d[f'{c}_{kn}'] = x.mS(k, c, kn, nz)
                    d[f'frac_crad_dodatni_{kn}'] = (lambda v: float(np.mean(v > 0)) if len(v) else float('nan'))(
                        x.S[(x.S[:, 0] == k) & (x.S[:, x.kS.index('kier')] == KIER.index(kn)) &
                            ((x.S[:, x.kS.index('narz')] == (1.0 if nz else 0.0)) if nz is not None else True), x.kS.index('c_rad')])
                    for dl in (0.03, 0.1):
                        for war in ('bez_rad', 'bez_perp', 'bez_mlp'):
                            d[f'F_{war}_{kn}_d{dl}'] = x.mF(k, war, kn, dl, nz)
                d['per_blok'] = {c: [med(x.L[x._maska(x.L, x.kL, k, b, nz), x.kL.index(c)]) for b in x.bloki]
                                 for c in ('E_rad_u1', 'E_mean_u1', 'wzrost_mlp', 'wzrost_att', 'cos_m0_ea', 'rho_wzgl')}
                d['per_blok'].update({f'F_bez_rad_r_d0.03': [x.F_blok(k, b, 'bez_rad', 'r', 0.03, nz) for b in x.bloki],
                                      'c_rad_r': [med(x.S[x._maska(x.S, x.kS, k, b, nz, 'r'), x.kS.index('c_rad')]) for b in x.bloki]})
                s[g] = d
            r[str(k)] = s
        o[n] = r
    return o


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


def main():
    B = {}
    for n in BIEGI:
        k = DANE / 'pomiar' / n
        if (k / 'wyniki.npz').exists() and (k / 'COMPLETE.json').exists():
            B[n] = Bieg(k)
    rodzic = {}
    if RODZIC_OCENA.exists():
        o8 = json.loads(RODZIC_OCENA.read_text(encoding='utf-8'))['ocena']['O8']['per_bieg']
        rodzic = {n: float(v[2]) for n, v in o8.items()}
    wyn = dict(biegi=sorted(B), enull_rodzica=rodzic)
    if B:
        wyn['ocena'] = ocena(B, rodzic); wyn['opis'] = opis(B)
    (DANE / 'OCENA_PREDYKCJI.json').write_text(json.dumps(konw(wyn), indent=1, ensure_ascii=False), encoding='utf-8')
    linie = [f'biegi: {len(B)} ({", ".join(sorted(B))})']
    for nr, r in wyn.get('ocena', {}).items():
        linie.append(f"{nr}: {'PASS ' + r['sila'] if r['PASS'] else 'FAIL'}  ({r['zaszlo']}/{r['n']}, próg {r['prog']})")
        for b, v in r['per_bieg'].items():
            linie.append(f"    {b}: {'tak' if v[0] else 'nie'}  " + '  '.join(f'{u:.4g}' for u in v[1:]))
    (DANE / 'OCENA_PREDYKCJI.txt').write_text('\n'.join(linie) + '\n', encoding='utf-8')
    print('\n'.join(linie))


if __name__ == '__main__':
    sys.exit(main())
