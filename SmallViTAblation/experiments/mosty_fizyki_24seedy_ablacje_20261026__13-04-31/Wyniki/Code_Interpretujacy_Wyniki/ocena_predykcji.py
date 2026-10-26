"""Ocena predykcji P1–P10 (część 1, 24 seedy) i A1–A6 (część 2, ablacje) według progów z Opis/PREDYKCJE_PRZED_POMIAREM.md.
Czyta Wyniki/Dane/pomiar/s<seed>/{wyniki.npz, podsumowanie.json} i Wyniki/Dane/ablacje/<wariant>_s<seed>/pomiar/…; pisze
Wyniki/Dane/OCENA_PREDYKCJI.json i .txt (PASS / FAIL z liczbami per seed). Nic nie rysuje. Napisany przed odczytem wyników."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[2]
DANE = E / 'Wyniki' / 'Dane'
K0, KK = 0, 65640
SEEDY = [3047 + 1000 * i for i in range(24)]
WARIANTY = ['base', 'parallel', 'frozen_mlp', 'frozen_attn', 'no_wd', 'shuffled_labels']
SEEDY_ABL = [3047, 4047, 5047]


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


class Seed:
    """Miary jednego biegu z wyniki.npz (per klatka / token) i podsumowanie.json (AP)."""

    def __init__(self, kat):
        self.kat = Path(kat)
        z = np.load(self.kat / 'wyniki.npz', allow_pickle=False)
        self.kroki = [int(k) for k in z['kroki']]; self.B = z['B_stan']; self.Bp = z['B_pol']; self.C = z['C']; self.iou = z['iou']
        self.A = z['A']; self.mA = list(z['miary_A'])[5:]; self.mB = list(z['miary_B_stan']); self.mBp = list(z['miary_B_pol']); self.mC = list(z['miary_C'])
        self.ps = json.loads((self.kat / 'podsumowanie.json').read_text(encoding='utf-8'))
        # miary_A w npz zaczyna się od 5 etykiet wierszy (krok, klatka, blok, token, narz); kolumny miar liczone od 5

    def ik(self, k):
        return self.kroki.index(k)

    def ap(self, k):
        return float(self.ps['ap50'][str(k)])

    # --- A
    def a_med(self, k, blok, miara, klasa=None):
        w = self.A[(self.A[:, 0] == k) & (self.A[:, 2] == blok)]
        if klasa is not None:
            w = w[w[:, 4] == (1.0 if klasa == 'narz' else 0.0)]
        v = w[:, 5 + self.mA.index(miara)]
        return float(np.median(v)) if len(v) else float('nan')

    # --- B
    def mediany_stanow(self, k, miara):
        return np.nanmedian(self.B[self.ik(k), :, :, self.mB.index(miara)], 0)              # (13,)

    def stosunki_Du(self, k):
        d = self.mediany_stanow(k, 'D_u'); return d[1:] / d[:-1]                               # (12,) półkroki 0..11 (parzyste = uwaga)

    def iloczyny(self, k):
        s = self.stosunki_Du(k); return float(np.prod(s[0::2])), float(np.prod(s[1::2]))

    def dsep(self, k):
        s = self.mediany_stanow(k, 'sep_u'); d = s[1:] - s[:-1]; return float(np.nansum(d[0::2])), float(np.nansum(d[1::2]))

    def styczny(self, k):
        return np.nanmedian(self.Bp[self.ik(k), :, :, self.mBp.index('styczny')], 0)          # (12,)

    def rho_sep_iou(self, tylko_k=None):
        ks = [self.ik(tylko_k)] if tylko_k is not None else range(len(self.kroki))
        sep = np.concatenate([self.B[i, :, 12, self.mB.index('sep_u')] for i in ks]); io = np.concatenate([self.iou[i] for i in ks])
        return spearman(sep, io)

    def sep12(self, k):
        return float(np.nanmedian(self.B[self.ik(k), :, 12, self.mB.index('sep_u')]))

    # --- C
    def argmax_SP(self, k):
        return int(np.argmax(np.nanmedian(self.C[self.ik(k), :, :, self.mC.index('SP')], 0))) + 1  # blok 1..6


def ocena_czesci_1(seedy):
    wyn = {}
    def test(nr, prog, wart):
        n = int(sum(bool(v[0]) for v in wart.values())); wyn[nr] = dict(prog=prog, n=len(wart), zaszlo=n, PASS=n >= prog, per_seed=wart)
    test('P1', 20, {s: (sd.a_med(KK, 3, 'nN_nJ') > 0.85 and sd.a_med(K0, 3, 'nN_nJ') < 0.80, sd.a_med(KK, 3, 'nN_nJ'), sd.a_med(K0, 3, 'nN_nJ')) for s, sd in seedy.items()})
    test('P2', 18, {s: (sd.a_med(KK, 3, 'nN_nJ') >= sd.a_med(KK, 5, 'nN_nJ'), sd.a_med(KK, 3, 'nN_nJ'), sd.a_med(KK, 5, 'nN_nJ')) for s, sd in seedy.items()})
    test('P3', 18, {s: (sd.a_med(KK, 3, 'trJ', 'narz') < sd.a_med(KK, 3, 'trJ', 'tlo'), sd.a_med(KK, 3, 'trJ', 'narz'), sd.a_med(KK, 3, 'trJ', 'tlo')) for s, sd in seedy.items()})
    d4 = {}
    for s, sd in seedy.items():
        st = sd.stosunki_Du(KK); d4[s] = (int((st[0::2] < 1).sum()) >= 4 and st[1] > 1, st.round(4).tolist())
    test('P4', 20, d4)
    test('P5', 20, {s: (bool((sd.styczny(KK)[0::2] > 0.9).all()), sd.styczny(KK)[0::2].round(3).tolist()) for s, sd in seedy.items()})
    test('P6', 20, {s: (sd.dsep(KK)[1] > sd.dsep(KK)[0], sd.dsep(KK)[0], sd.dsep(KK)[1]) for s, sd in seedy.items()})
    test('P7a', 20, {s: (sd.rho_sep_iou() > 0.5, sd.rho_sep_iou()) for s, sd in seedy.items()})
    rho = spearman([sd.sep12(KK) for sd in seedy.values()], [sd.ap(KK) for sd in seedy.values()])
    wyn['P7b'] = dict(prog=1, n=1, zaszlo=int(rho > 0.4), PASS=rho > 0.4, rho=rho, sep12={s: sd.sep12(KK) for s, sd in seedy.items()}, ap={s: sd.ap(KK) for s, sd in seedy.items()})
    test('P8', 20, {s: (sd.iloczyny(KK)[0] > sd.iloczyny(K0)[0] and sd.iloczyny(KK)[1] < sd.iloczyny(K0)[1], sd.iloczyny(K0), sd.iloczyny(KK)) for s, sd in seedy.items()})
    test('P9', 18, {s: (sd.argmax_SP(KK) in (4, 5), sd.argmax_SP(KK)) for s, sd in seedy.items()})
    test('P10', 20, {s: (sd.a_med(K0, 3, 'nN_nJ') < 0.80 and sd.dsep(K0)[1] < 0.1 and abs(sd.rho_sep_iou(K0)) < 0.3,
                         sd.a_med(K0, 3, 'nN_nJ'), sd.dsep(K0)[1], sd.rho_sep_iou(K0)) for s, sd in seedy.items()})
    return wyn


def ocena_czesci_2(abl, czesc1):
    wyn = {}
    base = {s: abl.get(('base', s)) for s in SEEDY_ABL}
    def test(nr, wart):
        wart = {s: v for s, v in wart.items() if v is not None}
        n = int(sum(bool(v[0]) for v in wart.values())); wyn[nr] = dict(prog=3, n=len(wart), zaszlo=n, PASS=(n >= 3), per_seed=wart)
    a1 = {}
    for s, sd in base.items():
        if sd is None:
            continue
        p1 = sd.a_med(KK, 3, 'nN_nJ') > 0.85 and sd.a_med(K0, 3, 'nN_nJ') < 0.80
        st = sd.stosunki_Du(KK); p4 = int((st[0::2] < 1).sum()) >= 4 and st[1] > 1
        p6 = sd.dsep(KK)[1] > sd.dsep(KK)[0]; p9 = sd.argmax_SP(KK) in (4, 5)
        a1[s] = (p1 and p4 and p6 and p9, dict(P1=p1, P4=p4, P6=p6, P9=p9))
    test('A1', a1)
    def para(w, f):
        out = {}
        for s in SEEDY_ABL:
            sd, b = abl.get((w, s)), base.get(s)
            out[s] = f(sd, b) if sd is not None else None
        return out
    test('A2', para('parallel', lambda sd, b: (b is not None and sd.ap(KK) <= b.ap(KK) - 0.02, sd.ap(KK), b.ap(KK) if b else None)))
    test('A3', para('frozen_mlp', lambda sd, b: (sd.dsep(KK)[1] < sd.dsep(KK)[0] and sd.ap(KK) < 0.40, sd.dsep(KK), sd.ap(KK))))
    test('A4', para('frozen_attn', lambda sd, b: (abs(sd.iloczyny(KK)[0] - sd.iloczyny(K0)[0]) < 0.05 and (b is not None and abs(b.iloczyny(KK)[0] - b.iloczyny(K0)[0]) > 0.05),
                                                 sd.iloczyny(K0)[0], sd.iloczyny(KK)[0], (b.iloczyny(K0)[0], b.iloczyny(KK)[0]) if b else None)))
    test('A5', para('no_wd', lambda sd, b: (sd.a_med(KK, 3, 'nN_nJ') > 0.85, sd.a_med(KK, 3, 'nN_nJ'))))
    test('A6', para('shuffled_labels', lambda sd, b: (sd.a_med(KK, 3, 'nN_nJ') < 0.85 and sd.dsep(KK)[1] < 0.1 and sd.ap(KK) < 0.10,
                                                      sd.a_med(KK, 3, 'nN_nJ'), sd.dsep(KK)[1], sd.ap(KK))))
    return wyn


def main():
    seedy = {}
    for s in SEEDY:
        k = DANE / 'pomiar' / f's{s}'
        if (k / 'wyniki.npz').exists() and (k / 'COMPLETE.json').exists():
            seedy[s] = Seed(k)
    abl = {}
    for w in WARIANTY:
        for s in SEEDY_ABL:
            k = DANE / 'ablacje' / f'{w}_s{s}' / 'pomiar'
            if (k / 'wyniki.npz').exists() and (k.parent / 'COMPLETE.json').exists():
                abl[(w, s)] = Seed(k)
    wyn = dict(seedy_czesc_1=sorted(seedy), biegi_czesc_2=[f'{w}_s{s}' for (w, s) in sorted(abl)])
    if seedy:
        wyn['czesc_1'] = ocena_czesci_1(seedy)
    if abl:
        wyn['czesc_2'] = ocena_czesci_2(abl, seedy)
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
    linie = [f'seedy części 1: {len(seedy)}, biegi części 2: {len(abl)}']
    for cz in ('czesc_1', 'czesc_2'):
        for nr, r in wyn.get(cz, {}).items():
            linie.append(f"{nr}: {'PASS' if r['PASS'] else 'FAIL'} ({r['zaszlo']}/{r['n']}, próg {r['prog']})")
    (DANE / 'OCENA_PREDYKCJI.txt').write_text('\n'.join(linie) + '\n', encoding='utf-8')
    print('\n'.join(linie))


if __name__ == '__main__':
    sys.exit(main())
