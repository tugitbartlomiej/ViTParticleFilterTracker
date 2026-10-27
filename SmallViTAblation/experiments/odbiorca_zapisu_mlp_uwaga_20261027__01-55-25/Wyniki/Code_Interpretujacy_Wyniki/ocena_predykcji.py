"""Ocena predykcji O1–O8 i kontroli K1 według progów z Opis/PREDYKCJE_PRZED_POMIAREM.md na 5 biegach. Czyta
Wyniki/Dane/pomiar/<bieg>/wyniki.npz (tylko biegi z COMPLETE.json), pisze Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}. Napisany przed
odczytem wyników (2026-10-27, po teście lokalnym na losowych wagach). Opcjonalny argument: inny katalog Dane (do testu).

Statystyka M[·]: mediana po blokach b = 0…NB−2 z median po tokenach i klatkach w bloku b; c = b+1 (off = 1), ostatni checkpoint.
Stosunki nieliniowe (B, C): w każdym bloku stosunek median po tokenach, potem mediana po blokach."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[2]
DANE = Path(sys.argv[1]) if len(sys.argv) > 1 else E / 'Wyniki' / 'Dane'
BIEGI = ['detekcja_s3047', 'detekcja_s4047', 'detekcja_s5047', 'cifar_maly_s47', 'cifar_deit_s47']
PROG = 4                     # ≥ 4/5 biegów (5/5 = „silnie”, p = 0,031; 4/5 = „słabo”, p = 0,19)
LOS16 = 16 / 192
X_ALL = ['q', 'k', 'v', 'att', 'm', 'mg']
D_ALL = ['u1', 'v1', 'los', 'v1n']


def med(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float('nan')


class Bieg:
    def __init__(self, kat):
        z = np.load(Path(kat) / 'wyniki.npz', allow_pickle=False)
        self.T = {n: z[n] for n in 'AOBC'}
        self.k = {n: list(z['kol_' + n]) for n in 'AOBC'}
        self.kier_B = list(z['kier_B']); self.kier_C = list(z['kier_C'])
        self.kroki = [int(k) for k in z['kroki']]; self.k0, self.kk = self.kroki[0], self.kroki[-1]
        self.NB = int(self.T['A'][:, 2].max()) + 2

    def bloki(self, t, k, maska=None):
        """Lista (b, wiersze) dla checkpointu k (opcjonalna maska wierszy)."""
        T = self.T[t]; w = T[:, 0] == k
        if maska is not None:
            w &= maska
        return [(b, T[w & (T[:, 2] == b)]) for b in range(self.NB - 1)]

    def mA(self, k, kol, off=1):
        T = self.T['A']; j = self.k['A'].index(kol)
        return med([med(r[:, j]) for b, r in self.bloki('A', k, T[:, 3] == off) if len(r)])

    def bloki_A(self, k, kol, off=1):
        T = self.T['A']; j = self.k['A'].index(kol)
        return [med(r[:, j]) for b, r in self.bloki('A', k, T[:, 3] == off)]

    def g_qk(self, k, d, off=1):
        return max(self.mA(k, f'g_q_{d}', off), self.mA(k, f'g_k_{d}', off))

    def mO(self, k, kol):
        j = self.k['O'].index(kol)
        return med([med(r[:, j]) for b, r in self.bloki('O', k) if len(r)])

    def _B(self, r, kol, kier, delta):
        j = self.k['B'].index(kol); ki = self.kier_B.index(kier)
        return med(r[(r[:, 6] == ki) & np.isclose(r[:, 5], delta), j])

    def stosB(self, k, kol, kier, delta, mian='los'):
        return med([self._B(r, kol, kier, delta) / self._B(r, kol, mian, delta) for b, r in self.bloki('B', k) if len(r)])

    def mB(self, k, kol, kier, delta):
        return med([self._B(r, kol, kier, delta) for b, r in self.bloki('B', k) if len(r)])

    def routing_B(self, k, kier, delta):
        """mediana_b [max(med Δattn_q, med Δattn_k) / med Δattn_v]."""
        return med([max(self._B(r, 'dattn_q', kier, delta), self._B(r, 'dattn_k', kier, delta)) / self._B(r, 'dattn_v', kier, delta)
                    for b, r in self.bloki('B', k) if len(r)])

    def udzial_uwagi(self, k, kier, delta):
        """mediana_b [med Δattn (wł.+inne) / (med Δattn + med ΔMLP)] — wskaźnik toru w bloku b+1."""
        out = []
        for b, r in self.bloki('B', k):
            if not len(r):
                continue
            a = np.hypot(self._B(r, 'dattn_wl', kier, delta), self._B(r, 'dattn_inne', kier, delta))
            m = np.hypot(self._B(r, 'dmlp_wl', kier, delta), self._B(r, 'dmlp_inne', kier, delta))
            out.append(a / (a + m))
        return med(out)

    def _C(self, r, kol, kiery):
        j = self.k['C'].index(kol); ki = [self.kier_C.index(x) for x in kiery]
        return med(r[np.isin(r[:, 5], ki), j])

    def stosC(self, k, kier, kol='jvp'):
        los = [x for x in self.kier_C if x.startswith('los')]
        return med([self._C(r, kol, [kier]) / self._C(r, kol, los) for b, r in self.bloki('C', k) if len(r)])


def test(wyn, nr, prog, wart):
    n = int(sum(bool(v[0]) for v in wart.values()))
    wyn[nr] = dict(prog=prog, n=len(wart), zaszlo=n, PASS=n >= prog, sila=('silnie' if n == len(wart) else 'słabo') if n >= prog else '',
                   per_bieg=wart)


def ocena(B):
    wyn = {}
    test(wyn, 'O1', PROG, {n: (x.g_qk(x.kk, 'u1') > 3 * x.g_qk(x.kk, 'los') and x.g_qk(x.kk, 'u1') > x.g_qk(x.k0, 'u1'),
                               x.g_qk(x.k0, 'u1'), x.g_qk(x.kk, 'u1'), x.g_qk(x.kk, 'los'), x.g_qk(x.kk, 'v1n')) for n, x in B.items()})
    test(wyn, 'O2', PROG, {n: (x.g_qk(x.kk, 'u1') > x.mA(x.kk, 'g_v_u1'), x.g_qk(x.kk, 'u1'), x.mA(x.kk, 'g_v_u1'), x.mA(x.kk, 'g_v_los'))
                           for n, x in B.items()})
    test(wyn, 'O3', PROG, {n: (x.mA(x.kk, 'E_att_u1') > x.mA(x.kk, 'E_m_u1'), x.mA(x.kk, 'E_att_u1'), x.mA(x.kk, 'E_m_u1'),
                               x.mA(x.kk, 'E_att_los'), x.mA(x.kk, 'E_m_los'), x.mA(x.kk, 'E_att_v1n'), x.mA(x.kk, 'E_m_v1n')) for n, x in B.items()})
    test(wyn, 'O4', PROG, {n: (x.stosB(x.kk, 'dattn_inne', 'u1', 0.03) >= 2, x.stosB(x.k0, 'dattn_inne', 'u1', 0.03),
                               x.stosB(x.kk, 'dattn_inne', 'u1', 0.03), x.stosB(x.kk, 'dattn_inne', 'v1n', 0.03), x.stosB(x.kk, 'dattn_inne', 'u1', 0.1))
                           for n, x in B.items()})
    test(wyn, 'O5', PROG, {n: (x.stosC(x.kk, 'u1') >= 1, x.stosC(x.k0, 'u1'), x.stosC(x.kk, 'u1'), x.stosC(x.kk, 'v1n'), x.stosC(x.kk, 'v1'))
                           for n, x in B.items()})
    test(wyn, 'O6', PROG, {n: (x.g_qk(x.kk, 'u1', 2) > 3 * x.g_qk(x.kk, 'los', 2), x.g_qk(x.k0, 'u1', 2), x.g_qk(x.kk, 'u1', 2),
                               x.g_qk(x.kk, 'los', 2)) for n, x in B.items()})
    test(wyn, 'O7', PROG, {n: (x.routing_B(x.kk, 'u1', 0.03) > x.routing_B(x.kk, 'los', 0.03), x.routing_B(x.kk, 'u1', 0.03),
                               x.routing_B(x.kk, 'los', 0.03), x.routing_B(x.kk, 'v1n', 0.03)) for n, x in B.items()})
    test(wyn, 'O8', PROG, {n: (x.mA(x.kk, 'Enull_u1') > 3 * 2 / 192, x.mA(x.k0, 'Enull_u1'), x.mA(x.kk, 'Enull_u1'), x.mA(x.kk, 'Enull_los'),
                               x.mA(x.kk, 'Enull_v1n')) for n, x in B.items()})
    test(wyn, 'K1', len(B), {n: (x.mA(x.kk, 'E_m_v1n') > 2 * LOS16, x.mA(x.kk, 'E_m_v1n'), x.mA(x.kk, 'E_m_los')) for n, x in B.items()})
    return wyn


def opis(B):
    """Wartości opisowe (wszystkie checkpointy): M[g_X(d)], M[E_X(d)] dla c = b+1 i b+2, głowa, stosunki B i C, wartości per blok."""
    o = {}
    for n, x in B.items():
        r = {}
        for k in x.kroki:
            s = {}
            for off in (1, 2):
                s[f'off{off}'] = {f'{m}_{X}_{d}': x.mA(k, f'{m}_{X}_{d}', off) for m in ('g', 'E') for X in X_ALL for d in D_ALL}
                s[f'off{off}'].update({f'gmax_{X}': x.mA(k, f'gmax_{X}', off) for X in X_ALL})
                s[f'off{off}'].update({f'Enull_{d}': x.mA(k, f'Enull_{d}', off) for d in D_ALL})
            s['kontrola_rodzica'] = {m: x.mA(k, m) for m in ('cos2_u1_v1n', 'cos2_v1_v1n', 'cos2_u1_u1n', 'sigma1')}
            s['glowa'] = {m: x.mO(k, m) for m in x.k['O'][5:]}
            s['B'] = {}
            for d in (0.03, 0.1):
                for kier in ('u1', 'v1n', 'v1'):
                    s['B'][f'd{d}_{kier}_do_los'] = {kol: x.stosB(k, kol, kier, d) for kol in x.k['B'][8:]}
                for kier in ('u1', 'los', 'v1n', 'v1'):
                    s['B'][f'd{d}_{kier}_bezwzgl'] = {kol: x.mB(k, kol, kier, d) for kol in x.k['B'][8:]}
                    s['B'][f'd{d}_{kier}_routing_qk_do_v'] = x.routing_B(k, kier, d)
                    s['B'][f'd{d}_{kier}_udzial_uwagi'] = x.udzial_uwagi(k, kier, d)
                    s['B'][f'd{d}_{kier}_inne_do_wlasny'] = x.mB(k, 'dattn_inne', kier, d) / x.mB(k, 'dattn_wl', kier, d)
            s['C'] = {f'{kier}_do_los_{kol}': x.stosC(k, kier, kol) for kier in ('u1', 'v1', 'v1n') for kol in ('jvp', 'jvp_cieplo', 'jvp_wl')}
            s['per_blok'] = {m: x.bloki_A(k, m) for m in ('g_q_u1', 'g_k_u1', 'g_v_u1', 'g_m_u1', 'E_att_u1', 'E_m_u1', 'g_q_los', 'g_k_los',
                                                         'g_v_los', 'g_q_v1n', 'g_k_v1n', 'g_v_v1n', 'g_m_v1n')}
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
    wyn = dict(biegi=sorted(B))
    if B:
        wyn['ocena'] = ocena(B); wyn['opis'] = opis(B)
    (DANE / 'OCENA_PREDYKCJI.json').write_text(json.dumps(konw(wyn), indent=1, ensure_ascii=False), encoding='utf-8')
    linie = [f'biegi: {len(B)} ({", ".join(sorted(B))})']
    for nr, r in wyn.get('ocena', {}).items():
        linie.append(f"{nr}: {'PASS' if r['PASS'] else 'FAIL'} {r['sila']} ({r['zaszlo']}/{r['n']}, próg {r['prog']})")
        for b, v in r['per_bieg'].items():
            linie.append(f"    {b}: {'tak' if v[0] else 'nie'}  " + '  '.join(f'{float(a):.4g}' for a in v[1:]))
    (DANE / 'OCENA_PREDYKCJI.txt').write_text('\n'.join(linie) + '\n', encoding='utf-8')
    print('\n'.join(linie))


if __name__ == '__main__':
    sys.exit(main())
