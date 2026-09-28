#!/usr/bin/env python3
"""Analiza odczytu dekodera (odczyt_dekodera.py): czy stan dekodera nadaje się na latent równania Langevina.

  1. związek z mAP w czasie: jakość sond po ostatniej warstwie (średnia po 72 sondach) vs mAP@0,5 pętli koopman,
     k00..k06, wycentrowane w ziarnie; to samo dla skalara tożsamości kręgosłupa (DIAGNOZA.json)
  2. jakość wg wariantu sondy (k00, k06): gdzie detektor nie rozpoznaje narzędzia
  3. niezmienniczość stanu ukrytego wybranego zapytania: R (rozrzut wewnątrz obrazu / między obrazami) i rozpoznanie
  4. ruch stanu dekodera między obrotami (względem rozrzutu sond)

    py -3.11 analiza_dekodera.py      ->  wyniki/ANALIZA_DEKODERA.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'petla_od_poczatku_large'))
import obrot_large as O  # noqa: E402

WARIANTY = ('original', 'rotation90', 'black', 'yellow', 'length08', 'foreground_bright')


def main() -> int:
    rows, _, _ = O.sondy()
    D = json.loads((HERE / 'wyniki/DEKODER.json').read_text(encoding='utf-8'))['wynik']
    H = np.load(HERE / 'wyniki/DEKODER.npz')
    A = json.loads((O.W / 'ANALIZA.json').read_text(encoding='utf-8'))['ziarna']
    diag = json.loads((O.EXP / 'domykanie_tozsamosci_large/wyniki/DIAGNOZA.json').read_text(encoding='utf-8'))['wynik']
    src = np.array([r['source_file'] for r in rows]); var = np.array([r['variant'] for r in rows])
    G = sorted(set(src)); gi = np.array([G.index(x) for x in src])
    wyn = dict(czas={}, warianty={}, niezmienniczosc={}, ruch={})
    xq, xs, y = [], [], []
    for s in ('42', '43', '44'):
        m = [A[s]['k00']] + A[s]['koopman']['mAP50']
        q = [float(np.mean(np.array(D[f's{s}_k{k:02d}']['q'])[:, -1])) for k in range(7)]
        sk = [e['skalar_sredni'] for e in diag[s]['koopman']][1:8]
        wyn['czas'][s] = dict(mAP50=m, jakosc_dekodera=q, skalar_kregoslupa=sk)
        xq += list(np.array(q) - np.mean(q)); xs += list(np.array(sk) - np.mean(sk)); y += list(np.array(m) - np.mean(m))
    wyn['czas']['spearman_dekoder_mAP'] = float(spearmanr(xq, y)[0])
    wyn['czas']['spearman_kregoslup_mAP'] = float(spearmanr(xs, y)[0])
    for k in ('k00', 'k06'):
        Q = np.mean([np.array(D[f's{s}_{k}']['q'])[:, -1] for s in (42, 43, 44)], 0)
        wyn['warianty'][k] = {v: float(Q[var == v].mean()) for v in WARIANTY}

    def R(Z):
        c = np.stack([Z[gi == g].mean(0) for g in range(len(G))])
        return float(np.mean([np.linalg.norm(Z[gi == g] - c[g], axis=1).mean() for g in range(len(G))])
                     / np.linalg.norm(c - Z.mean(0), axis=1).mean())

    def rozp(Z):
        ok = []
        for i in range(len(Z)):
            c = np.stack([Z[(gi == g) & (np.arange(len(Z)) != i)].mean(0) for g in range(len(G))])
            ok.append(np.argmin(((c - Z[i]) ** 2).sum(1)) == gi[i])
        return float(np.mean(ok))
    for k in ('k00', 'k03', 'k06'):
        wyn['niezmienniczosc'][k] = dict(R=float(np.mean([R(H[f's{s}_{k}']) for s in (42, 43, 44)])),
                                         rozpoznanie=float(np.mean([rozp(H[f's{s}_{k}']) for s in (42, 43, 44)])))
    for a, b in (('k00', 'k01'), ('k01', 'k02'), ('k02', 'k03'), ('k03', 'k06')):
        wyn['ruch'][f'{a}_{b}'] = float(np.mean([
            np.linalg.norm(H[f's{s}_{b}'] - H[f's{s}_{a}'], axis=1).mean()
            / np.linalg.norm(H[f's{s}_{a}'] - H[f's{s}_{a}'].mean(0), axis=1).mean() for s in (42, 43, 44)]))
    (HERE / 'wyniki/ANALIZA_DEKODERA.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: wyn['czas'][k] for k in ('spearman_dekoder_mAP', 'spearman_kregoslup_mAP')}),
          json.dumps(wyn['warianty']), json.dumps(wyn['niezmienniczosc']), json.dumps(wyn['ruch']), sep='\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
