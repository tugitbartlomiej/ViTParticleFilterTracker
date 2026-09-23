#!/usr/bin/env python3
"""Wektor tozsamosci narzedzia w tokenach backbone BEZ etykiet (2026-09-23).

Uczenie: kawalki obrazu wewnatrz ramki PRZEWIDZIANEJ przez detektor (pewnosc > 0,7) wobec kawalkow
daleko poza nia, na polowie filmow. Ocena z etykietami (tylko tutaj): kawalki na trzonku narzedzia
(odcinek od czubka do przeciwleglego rogu ramki) wobec tla, na drugiej polowie filmow.
Porownanie z kierunkiem wyznaczonym z etykiet (rdzen_w_tokenach.py). Wynik 2026-09-23:
AUC 0,975 bez etykiet wobec 0,978 z etykietami, cosinus miedzy kierunkami 0,88.

    py -3.11 experiments/krystalizacja_latentu/rdzen_bez_etykiet.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KORZEN = Path(__file__).resolve().parents[2]
PULA = KORZEN / 'results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500/czesc000.npz'
MANIFEST = KORZEN / 'feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl'
DOWODY = Path(__file__).resolve().parent / 'wyniki/dowody'


def auc(p, n):
    x = np.r_[p, n]; r = np.argsort(np.argsort(x)) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def main() -> int:
    d = np.load(PULA)
    rng = np.random.default_rng(0)
    wyb = np.sort(rng.choice(len(d['frame_ids']), 4000, replace=False))
    P = d['profil_red'][wyb].astype(float); M = d['maska_pred'][wyb]; S = d['score'][wyb]
    fid = [str(f) for f in d['frame_ids'][wyb]]
    man = {}
    for line in MANIFEST.open(encoding='utf-8'):
        r = json.loads(line); man[r['frame_id']] = r
    H = W = 24
    ys, xs = (np.arange(H) + .5) / H * 480, (np.arange(W) + .5) / W * 720
    gy, gx = np.meshgrid(ys, xs, indexing='ij'); gy, gx = gy.ravel(), gx.ravel()
    film = np.array([f.split('/')[-2] for f in fid]); fl = np.unique(film); rng.shuffle(fl)
    A = np.isin(film, fl[:len(fl) // 2]); B = ~A

    ins, out = [], []
    for i in np.flatnonzero(A & (S > .7)):
        m = M[i]
        if m.sum() < 2:
            continue
        ins.append(P[i, m]); out.append(P[i, rng.choice(np.flatnonzero(~m), 20, replace=False)])
    ins = np.concatenate(ins); out = np.concatenate(out)
    Sw = np.cov(np.r_[ins - ins.mean(0), out - out.mean(0)].T) + 1e-6 * np.eye(32)
    w_lda = np.linalg.solve(Sw, ins.mean(0) - out.mean(0)); w_lda /= np.linalg.norm(w_lda)
    w_pat = ins.mean(0) - out.mean(0); w_pat /= np.linalg.norm(w_pat)

    tp, tn = [], []
    for i in np.flatnonzero(B):
        r = man[fid[i]]; tip = r.get('tip_point'); x0, y0, bw, bh = r['annotations'][0]['bbox']
        if tip is None or bw < 8 or bh < 8:
            continue
        rogi = np.array([[x0, y0], [x0 + bw, y0], [x0, y0 + bh], [x0 + bw, y0 + bh]])
        k = rogi[np.argmax(((rogi - tip) ** 2).sum(1))]
        a = np.array(tip, float); v = k - a
        t = np.clip(((np.c_[gx, gy] - a) @ v) / max(v @ v, 1e-9), 0, 1)
        odl = np.linalg.norm(np.c_[gx, gy] - (a + t[:, None] * v), axis=1)
        wew = (gx >= x0) & (gx <= x0 + bw) & (gy >= y0) & (gy <= y0 + bh)
        n_ = np.flatnonzero(wew & (odl < 12))
        poza = np.flatnonzero(~((gx >= x0 - .5 * bw) & (gx <= x0 + 1.5 * bw) & (gy >= y0 - .5 * bh) & (gy <= y0 + 1.5 * bh)))
        if len(n_):
            tp.append(P[i, n_]); tn.append(P[i, rng.choice(poza, 20, replace=False)])
    tp = np.concatenate(tp); tn = np.concatenate(tn)
    w_lab = np.load(DOWODY / 'RDZEN_kierunek_32.npy')
    raport = {}
    for nm, w in (('bez_etykiet_lda', w_lda), ('bez_etykiet_wzorzec', w_pat), ('z_etykietami', w_lab)):
        raport[nm] = dict(auc_obce_filmy=auc(tp @ w, tn @ w), cos_z_etykietami=float(abs(w @ w_lab)))
        print(f'{nm:22s} AUC {raport[nm]["auc_obce_filmy"]:.3f}  cos {raport[nm]["cos_z_etykietami"]:.2f}')
    np.save(DOWODY / 'RDZEN_bez_etykiet_32.npy', w_lda)
    (DOWODY / 'REPORT_rdzen_bez_etykiet.json').write_text(json.dumps(raport, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
