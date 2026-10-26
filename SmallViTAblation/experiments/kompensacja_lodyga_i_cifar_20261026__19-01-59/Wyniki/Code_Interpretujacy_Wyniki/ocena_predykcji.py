"""Ocena predykcji K1–K5 (frozen_both, 3 seedy; odniesienie: base/frozen_mlp/frozen_attn rodzica) i C1–C9 (CIFAR-10, 6 biegów)
według progów z Opis/PREDYKCJE_PRZED_POMIAREM.md. Pisze Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}. Napisany przed odczytem wyników."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[2]
DANE = E / 'Wyniki' / 'Dane'
RODZIC = E.parent / 'mosty_fizyki_24seedy_ablacje_20261026__13-04-31' / 'Wyniki' / 'Dane' / 'ablacje'
sys.path.insert(0, str(E.parent / 'mosty_fizyki_24seedy_ablacje_20261026__13-04-31' / 'Wyniki' / 'Code_Interpretujacy_Wyniki'))
from ocena_predykcji import Seed, KK as KK_DET, K0  # noqa: E402  (klasa Seed rodzica czyta wyniki.npz części K)

SEEDY_K = [3047, 4047, 5047]
BIEGI_C = {'s47_obciecie': 'deit', 's47_bez_obciecia': 'deit', 'maly_s47_obciecie': 'malyvit', 'maly_s48_obciecie': 'malyvit',
           'maly_s47_wd0': 'malyvit', 'maly_s47_losowe_etykiety': 'malyvit'}
UCZONE = ['s47_obciecie', 's47_bez_obciecia', 'maly_s47_obciecie', 'maly_s48_obciecie', 'maly_s47_wd0']
M0, M1 = 'maly_s47_losowe_etykiety', 'maly_s47_obciecie'
KKC = 93780


class Cifar:
    def __init__(self, kat):
        z = np.load(Path(kat) / 'wyniki.npz', allow_pickle=False)
        self.kroki = [int(k) for k in z['kroki']]; self.B = z['B_stan']; self.Bp = z['B_pol']; self.C = z['C']; self.sep = z['sep']; self.acc = z['acc']
        self.A = z['A']; self.mA = list(z['miary_A'])[5:]; self.mB = list(z['miary_B_stan']); self.mBp = list(z['miary_B_pol']); self.mC = list(z['miary_C'])
        self.NB = int(z['n_blokow']); self.cls = bool(z['cls']); self.bst = self.NB // 2                 # blok środkowy b* (indeks 0-based)

    def ik(self, k):
        return self.kroki.index(k)

    def a_med(self, k, blok, miara):
        w = self.A[(self.A[:, 0] == k) & (self.A[:, 2] == blok) & (self.A[:, 4] == 0.0)]         # tokeny obrazu, bez CLS
        v = w[:, 5 + self.mA.index(miara)]
        return float(np.median(v)) if len(v) else float('nan')

    def stosunki_Du(self, k):
        d = np.nanmedian(self.B[self.ik(k), :, :, self.mB.index('D_u')], 0); return d[1:] / d[:-1]

    def iloczyny(self, k):
        s = self.stosunki_Du(k); return float(np.prod(s[0::2])), float(np.prod(s[1::2]))

    def dsep(self, k):
        s = self.sep[self.ik(k), :, 0]; d = s[1:] - s[:-1]; return float(np.nansum(d[0::2])), float(np.nansum(d[1::2]))

    def styczny(self, k):
        return np.nanmedian(self.Bp[self.ik(k), :, :, self.mBp.index('styczny')], 0)

    def argmax_SP(self, k):
        return int(np.argmax(np.nanmedian(self.C[self.ik(k), :, :, self.mC.index('SP')], 0))) + 1

    def acc_P(self, k):
        return np.nanmean(self.C[self.ik(k), :, :, self.mC.index('ok_P')], 0)                       # (NB,)


def test(wyn, nr, prog, wart):
    wart = {s: v for s, v in wart.items() if v is not None}
    n = int(sum(bool(v[0]) for v in wart.values())); wyn[nr] = dict(prog=prog, n=len(wart), zaszlo=n, PASS=n >= prog, per_bieg=wart)


def ocena_K(fb, rodz):
    wyn = {}
    KK = KK_DET
    def ref(w, s):
        d = RODZIC / f'{w}_s{s}' / 'pomiar'
        return Seed(d) if (d / 'wyniki.npz').exists() else None
    k1, k2, k3, k4, k5 = {}, {}, {}, {}, {}
    for s, sd in fb.items():
        fm, fa, ba = ref('frozen_mlp', s), ref('frozen_attn', s), ref('base', s)
        ap = sd.ap(KK)
        k1[s] = (ap < 0.80 and fm is not None and fa is not None and ap <= fm.ap(KK) - 0.03 and ap <= fa.ap(KK) - 0.03,
                 ap, fm.ap(KK) if fm else None, fa.ap(KK) if fa else None, ba.ap(KK) if ba else None)
        du, dm = sd.dsep(KK); k2[s] = (du + dm < 0.5, du, dm)
        s0 = float(sd.mediany_stanow(KK, 'sep_u')[0]); k3[s] = (s0 > 1.10, s0, float(ba.mediany_stanow(KK, 'sep_u')[0]) if ba else None)
        iu = sd.iloczyny(KK)[0]; k4[s] = (iu > 0.85, sd.iloczyny(K0)[0], iu)
        nn = sd.a_med(KK, 3, 'nN_nJ'); k5[s] = (nn < 0.80, nn)
    for nr, w in (('K1', k1), ('K2', k2), ('K3', k3), ('K4', k4), ('K5', k5)):
        test(wyn, nr, 3, w)
    return wyn


def ocena_C(c):
    wyn = {}
    ucz = {b: c[b] for b in UCZONE if b in c}
    test(wyn, 'C1', 5, {b: (x.a_med(KKC, x.bst, 'nN_nJ') > 0.85 and x.a_med(0, x.bst, 'nN_nJ') < 0.80, x.a_med(0, x.bst, 'nN_nJ'), x.a_med(KKC, x.bst, 'nN_nJ')) for b, x in ucz.items()})
    test(wyn, 'C1z', 1, {b: (x.a_med(KKC, x.bst, 'nN_nJ') > 0.85, x.a_med(KKC, x.bst, 'nN_nJ')) for b, x in c.items() if b == M0})
    test(wyn, 'C2', 5, {b: (x.a_med(KKC, x.bst, 'nJ') > x.a_med(0, x.bst, 'nJ'), x.a_med(0, x.bst, 'nJ'), x.a_med(KKC, x.bst, 'nJ')) for b, x in ucz.items()})
    test(wyn, 'C2z', 1, {b: (x.a_med(KKC, x.bst, 'nJ') < 0.5 * x.a_med(0, x.bst, 'nJ'), x.a_med(0, x.bst, 'nJ'), x.a_med(KKC, x.bst, 'nJ')) for b, x in c.items() if b == M0})
    def c3(x):
        v = [x.a_med(KKC, b, 'nN_nJ') for b in range(x.NB)]; return (v[-1] < max(v[1:-1]), [round(t, 3) for t in v])
    test(wyn, 'C3', 5, {b: c3(x) for b, x in ucz.items()})
    test(wyn, 'C4a', 6, {b: (x.iloczyny(0)[0] < 0.9, x.iloczyny(0)[0]) for b, x in c.items()})
    test(wyn, 'C4b', 5, {b: (x.iloczyny(KKC)[0] > x.iloczyny(0)[0] and x.iloczyny(KKC)[1] < x.iloczyny(0)[1], x.iloczyny(0), x.iloczyny(KKC)) for b, x in ucz.items()})
    test(wyn, 'C5', 5, {b: (bool((x.styczny(KKC)[0::2][:-1] > 0.9).all()), x.styczny(KKC)[0::2].round(3).tolist()) for b, x in ucz.items()})
    c6 = {b: (x.dsep(KKC)[1] > x.dsep(KKC)[0], x.dsep(KKC)[0], x.dsep(KKC)[1]) for b, x in ucz.items()}
    test(wyn, 'C6_malyvit', 3, {b: v for b, v in c6.items() if BIEGI_C[b] == 'malyvit'})
    test(wyn, 'C6_deit', 2, {b: v for b, v in c6.items() if BIEGI_C[b] == 'deit'})
    test(wyn, 'C7', 6, {b: (sum(x.dsep(0)) < 0.2 and x.acc[x.ik(0)] < 0.15, sum(x.dsep(0)), float(x.acc[x.ik(0)])) for b, x in c.items()})
    if M0 in c and M1 in c:
        x0, x1 = c[M0], c[M1]
        test(wyn, 'C8', 1, {M0: (x0.acc[x0.ik(KKC)] < 0.15 and x0.dsep(KKC)[1] < 0.5 * x1.dsep(KKC)[1], float(x0.acc[x0.ik(KKC)]), x0.dsep(KKC)[1], x1.dsep(KKC)[1])})
    test(wyn, 'C9a', 5, {b: (x.argmax_SP(KKC) not in (1, x.NB), x.argmax_SP(KKC)) for b, x in ucz.items()})
    def c9b(x):
        ap = x.acc_P(KKC); a = float(x.acc[x.ik(KKC)]); return (bool((ap[1:] >= a - 0.05).all()), a, ap.round(3).tolist())
    test(wyn, 'C9b', 5, {b: c9b(x) for b, x in ucz.items()})
    return wyn


def main():
    fb = {}
    for s in SEEDY_K:
        k = DANE / 'ablacje' / f'frozen_both_s{s}' / 'pomiar'
        if (k / 'wyniki.npz').exists() and (k.parent / 'COMPLETE.json').exists():
            fb[s] = Seed(k)
    c = {}
    for b in BIEGI_C:
        k = DANE / 'cifar' / b
        if (k / 'wyniki.npz').exists() and (k / 'COMPLETE.json').exists():
            c[b] = Cifar(k)
    wyn = dict(biegi_K=sorted(fb), biegi_C=sorted(c))
    if fb:
        wyn['czesc_K'] = ocena_K(fb, None)
    if c:
        wyn['czesc_C'] = ocena_C(c)
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
    linie = [f'biegi części K: {len(fb)}, biegi części C: {len(c)}']
    for cz in ('czesc_K', 'czesc_C'):
        for nr, r in wyn.get(cz, {}).items():
            linie.append(f"{nr}: {'PASS' if r['PASS'] else 'FAIL'} ({r['zaszlo']}/{r['n']}, próg {r['prog']})")
    (DANE / 'OCENA_PREDYKCJI.txt').write_text('\n'.join(linie) + '\n', encoding='utf-8')
    print('\n'.join(linie))


if __name__ == '__main__':
    sys.exit(main())
