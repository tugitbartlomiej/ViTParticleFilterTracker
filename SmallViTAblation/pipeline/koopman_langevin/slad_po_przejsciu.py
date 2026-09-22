#!/usr/bin/env python3
"""Czy kwadrat siatki PAMIETA narzedzie, ktore przez niego przeszlo (opis Eulera).

Pytanie autora, z eksperymentow z niedzwiedziem: cząstka stoi w miejscu, ale czy po przejsciu
obiektu zostaje w niej slad? Test bez zadnego uczenia:

  wzorzec        sredni kierunek tokenow lezacych w prawdziwej ramce narzedzia (400 klatek)
  miara          podobienstwo kosinusowe tokenu do wzorca (jedna liczba na kwadrat i klatke)
  zbior A        kwadraty, w ktorych narzedzia TERAZ nie ma, ale bylo k klatek temu
  zbior B        kwadraty, w ktorych narzedzia nie ma i nie bylo przez ostatnie osiem pomiarow
  wynik          AUC miedzy A i B: 0,5 znaczy brak sladu, wiecej znaczy slad tym silniejszy

Kontrola: ten sam rachunek dla kwadratow, w ktorych narzedzie DOPIERO BEDZIE za k klatek.
Jesli "slad" wychodzi tak samo w przyszlosc, to nie jest pamiec, tylko sasiedztwo narzedzia.
Etykiety (prawdziwe ramki) sluza wylacznie do oceny.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.slad_po_przejsciu \
        --recording .../CZASTKI_b0500 --package .../detector_probe_clips --out REPORT_slad.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .patch_analysis import grid_masks, load_recording


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(len(order)) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / max(len(pos) * len(neg), 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--lags', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    args = ap.parse_args(argv)
    rec = load_recording(args.recording)
    h, w = rec['meta']['grid']
    fids = rec['frame_ids']
    clips = np.array([f.split('/')[-2] for f in fids])
    stride = int(rec['meta'].get('stride_frames') or 1)
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    truth = grid_masks(tracks, fids, (h, w))
    prof = rec['profil_red'].astype(np.float64)
    unit = prof / np.maximum(np.linalg.norm(prof, axis=2, keepdims=True), 1e-9)

    rng = np.random.default_rng(0)
    rows = np.flatnonzero(truth.min(1) > -2)
    wzorzec = np.zeros(prof.shape[2])
    for t in rows[rng.choice(len(rows), 400, replace=False)]:
        m = truth[t] == 1
        if m.sum():
            wzorzec += unit[t, m].mean(0)
    wzorzec /= np.linalg.norm(wzorzec)
    sim = np.einsum('tpk,k->tp', unit, wzorzec)

    raport = {'stride_frames': stride, 'lags': {}}
    print(f'{"odstep":>10s} {"slad po przejsciu":>18s} {"zapowiedz przed":>16s} {"kwadratow":>10s}')
    for lag in args.lags:
        po, przed, tlo = [], [], []
        for c in np.unique(clips):
            idx = np.flatnonzero((clips == c) & (truth.min(1) > -2))
            if len(idx) < 12:
                continue
            m = truth[idx] == 1                                     # [T, P] narzedzie obecne
            s = sim[idx]
            hist = np.zeros_like(m)
            for j in range(1, 9):
                hist[j:] |= m[:-j]
            fut = np.zeros_like(m)
            for j in range(1, 9):
                fut[:-j] |= m[j:]
            for t in range(8, len(idx) - 8):
                brak = ~m[t]
                a = brak & m[t - lag] & ~fut[t]                     # bylo tu lag pomiarow temu i NIE wroci
                b = brak & m[t + lag] & ~hist[t]                    # bedzie tu za lag pomiarow, a NIE bylo
                n = brak & ~hist[t] & ~fut[t]                       # nie bylo i nie bedzie przez osiem pomiarow
                if a.sum() and n.sum():
                    po.append((s[t][a], s[t][n]))
                if b.sum() and n.sum():
                    przed.append((s[t][b], s[t][n]))
                tlo.append(n.sum())
        a_po = auc(np.concatenate([x for x, _ in po]), np.concatenate([y for _, y in po]))
        a_przed = auc(np.concatenate([x for x, _ in przed]), np.concatenate([y for _, y in przed]))
        raport['lags'][lag] = dict(frames=lag * stride, slad_po=a_po, zapowiedz_przed=a_przed,
                                   probek=int(sum(len(x) for x, _ in po)))
        print(f'{lag * stride:7d} kl {a_po:18.3f} {a_przed:16.3f} {sum(len(x) for x, _ in po):10d}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
