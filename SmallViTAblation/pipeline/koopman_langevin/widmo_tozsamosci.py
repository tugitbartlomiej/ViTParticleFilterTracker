#!/usr/bin/env python3
"""Widmo operatora translacyjnego z podzialem na czesc rzeczywista (dodatnia i ujemna) i urojona.

Dotychczasowy kod tozsamosci (`identity_code_pool.py`) liczyl operator PO SYMETRYZACJI, wiec jego
wartosci wlasne sa z definicji rzeczywiste i cala czesc obrotowa przepadala. Tutaj ten sam operator
(token narzedzia w jednym polozeniu -> ten sam token po przesunieciu narzedzia) liczony jest BEZ
symetryzacji, a jego mody dziela sie na trzy rodziny:

  trwale (Re > 0, Im = 0)   wielkosc, ktora po przesunieciu wraca taka sama, tylko slabsza
  przemienne (Re < 0)       wielkosc, ktora po przesunieciu zmienia znak na przeciwny
  obrotowe (Im != 0)        para wielkosci, ktora po przesunieciu obraca sie o staly kat;
                            kat daje "okres": po ilu przesunieciach para wraca do siebie

Dla kazdej rodziny mierzymy to samo i porownujemy:
  odroznienie   czy z tych liczb widac, ktory kwadrat lezy na narzedziu (prawdziwa ramka, tylko do oceny)
  przezycie     czy wartosc przezywa przesuniecie narzedzia (zgodnosc par)
  wydluzenie    ile procent proporcji prawdziwej ramki da sie z nich przewidziec (podzial po filmach)
  zwrot         to samo dla polozenia koncowki wzgledem srodka ramki (tego trwale mody NIE niosly)

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.widmo_tozsamosci \
        --recording .../CZASTKI_b0500 --package .../detector_probe_clips --out REPORT_widmo.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .patch_analysis import grid_masks, load_recording, translational_pairs


def operator(x0: np.ndarray, x1: np.ndarray, reg: float = 1e-3):
    """Niesymetryzowany operator: x1 ~ x0 K, plus jego wartosci i wektory wlasne (zespolone)."""
    mu = x0.mean(0)
    a, b = x0 - mu, x1 - mu
    c00 = a.T @ a / len(a) + reg * np.eye(a.shape[1])
    k = np.linalg.solve(c00, a.T @ b / len(a))
    lam, u = np.linalg.eig(k)                      # f(x) = (x - mu) @ u spelnia f(x1) = lam f(x0)
    order = np.argsort(-np.abs(lam))
    return lam[order], u[:, order], mu


def families(lam: np.ndarray, tol: float = 1e-8) -> dict[str, np.ndarray]:
    real = np.abs(lam.imag) < tol * max(1.0, float(np.abs(lam).max()))
    return {'trwale (Re>0, bez obrotu)': np.flatnonzero(real & (lam.real > 0)),
            'przemienne (Re<0, bez obrotu)': np.flatnonzero(real & (lam.real < 0)),
            'obrotowe (Im!=0)': np.flatnonzero(~real)}


def coordinates(prof: np.ndarray, mask: np.ndarray, score: np.ndarray, mu: np.ndarray,
                u: np.ndarray, cols: np.ndarray, min_score: float = .5) -> np.ndarray:
    """Srednia po tokenach narzedzia; mod zespolony daje dwie liczby (czesc rzeczywista i urojona)."""
    out = np.full((len(prof), 2 * len(cols)), np.nan)
    v = u[:, cols]
    for t in range(len(prof)):
        if score[t] > min_score and mask[t].sum() >= 4:
            c = (prof[t, mask[t]] - mu) @ v
            out[t] = np.concatenate([c.real.mean(0), c.imag.mean(0)])
    return out[:, np.concatenate([np.ones(len(cols), bool), np.abs(u[:, cols].imag).sum(0) > 0])]


def skill(x: np.ndarray, y: np.ndarray, group: np.ndarray, lam: float = 1.0, folds: int = 4, seed: int = 0) -> float:
    names = np.array(sorted(set(group.tolist())))
    fold = {v: i % folds for i, v in enumerate(np.random.default_rng(seed).permutation(names))}
    part = np.array([fold[v] for v in group])
    x = np.hstack([(x - x.mean(0)) / np.maximum(x.std(0), 1e-9), np.ones((len(x), 1))])
    pred = np.zeros_like(y)
    for f in range(folds):
        tr, te = part != f, part == f
        if not tr.any() or not te.any():
            continue
        a = x[tr].T @ x[tr] + lam * np.eye(x.shape[1])
        pred[te] = x[te] @ np.linalg.solve(a, x[tr].T @ y[tr])
    return float(100 * (1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def patch_auc(prof: np.ndarray, truth: np.ndarray, mu: np.ndarray, u: np.ndarray, cols: np.ndarray,
              clips: np.ndarray, rows: np.ndarray, seed: int = 0) -> float:
    """Czy z tych liczb widac, ktory kwadrat lezy na narzedziu; uczenie i ocena na roznych filmach."""
    rng = np.random.default_rng(seed)
    take = rows[rng.choice(len(rows), min(600, len(rows)), replace=False)]
    v = u[:, cols]
    feats, y, g = [], [], []
    for t in take:
        c = (prof[t] - mu) @ v
        f = np.hstack([c.real, c.imag]) if np.abs(v.imag).sum() else c.real
        keep = truth[t] != -1
        feats.append(f[keep]); y.append((truth[t][keep] == 1).astype(float)); g.append(np.full(keep.sum(), clips[t]))
    x, y, g = np.vstack(feats), np.concatenate(y), np.concatenate(g)
    names = np.array(sorted(set(g.tolist())))
    fold = {v_: i % 2 for i, v_ in enumerate(names)}
    part = np.array([fold[v_] for v_ in g])
    x = np.hstack([(x - x.mean(0)) / np.maximum(x.std(0), 1e-9), np.ones((len(x), 1))])
    pred = np.zeros(len(y))
    for f in (0, 1):
        tr, te = part != f, part == f
        a = x[tr].T @ x[tr] + np.eye(x.shape[1])
        pred[te] = x[te] @ np.linalg.solve(a, x[tr].T @ y[tr])
    pos, neg = pred[y == 1], pred[y == 0]
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(len(order)) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--modes', type=int, default=6, help='ile modow z kazdej rodziny')
    args = ap.parse_args(argv)
    rec = load_recording(args.recording)
    h, w = rec['meta']['grid']
    fids = rec['frame_ids']
    clips = np.array([f.split('/')[-2] for f in fids])
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    truth = grid_masks(tracks, fids, (h, w))
    prof = rec['profil_red'].astype(np.float64)
    score, mask = rec['score'].astype(float), rec['maska_pred']
    x0, x1 = translational_pairs(rec, (h, w), per_clip=400)
    lam, u, mu = operator(x0, x1)
    rodziny = families(lam)
    print(f'par {len(x0)}; modow: ' + ', '.join(f'{k} {len(v)}' for k, v in rodziny.items()), flush=True)

    # prawdziwe cele (etykiety tylko do oceny)
    box, tip, have = [], [], []
    for t, fid in enumerate(fids):
        c, n = fid.split('/')[-2], int(fid.split('/')[-1])
        i = tracks[c]['frame'].index(n)
        b, p = tracks[c]['box'][i], tracks[c]['tip'][i]
        have.append(b is not None and p is not None and score[t] > .5 and mask[t].sum() >= 4)
        box.append(b if b else [0, 0, 1, 1]); tip.append(p if p else [0, 0])
    box, tip, have = np.array(box, float), np.array(tip, float), np.array(have)
    rows_all = np.flatnonzero(truth.min(1) > -2)
    cele = {'wydluzenie ramki': np.log(box[:, 2] / np.maximum(box[:, 3], 1)),
            'wielkosc ramki': np.log(box[:, 2] * box[:, 3]),
            'zwrot koncowki (poziomo)': (tip[:, 0] - (box[:, 0] + box[:, 2] / 2)) / np.maximum(box[:, 2], 1),
            'zwrot koncowki (pionowo)': (tip[:, 1] - (box[:, 1] + box[:, 3] / 2)) / np.maximum(box[:, 3], 1)}

    raport = {'pary': int(len(x0)), 'wartosci_wlasne': [[float(v.real), float(v.imag)] for v in lam[:16]], 'rodziny': {}}
    print(f'\n{"rodzina":34s} {"|lam|":>6s} {"okres":>7s} {"odroznienie":>11s} {"przezycie":>10s} '
          + ' '.join(f'{k:>24s}' for k in cele))
    for nazwa, cols in rodziny.items():
        if not len(cols):
            continue
        cols = cols[:args.modes]
        kod = coordinates(prof, mask, score, mu, u, cols)
        ok = have & np.isfinite(kod).all(1)
        a0, a1 = (x0 - mu) @ u[:, cols], (x1 - mu) @ u[:, cols]
        przezycie = float(np.mean([np.corrcoef(a0[:, j].real, a1[:, j].real)[0, 1] for j in range(len(cols))]))
        okres = 2 * np.pi / np.abs(np.angle(lam[cols])) if np.abs(lam[cols].imag).max() > 0 else None
        auc = patch_auc(prof, truth, mu, u, cols, clips, rows_all)
        wyn = {k: skill(kod[ok], y[ok], clips[ok]) for k, y in cele.items()}
        raport['rodziny'][nazwa] = dict(mody=len(cols), lam_abs=[float(abs(v)) for v in lam[cols]],
                                        okres=[float(o) for o in okres] if okres is not None else None,
                                        odroznienie_auc=auc, przezycie=przezycie, klatek=int(ok.sum()), cele=wyn)
        print(f'{nazwa:34s} {abs(lam[cols[0]]):6.2f} {(okres[0] if okres is not None else np.nan):7.1f} '
              f'{auc:11.3f} {przezycie:10.2f} ' + ' '.join(f'{wyn[k]:23.1f}%' for k in cele), flush=True)
    # czy rodziny poza trwalymi DOKLADAJA cos, czego trwale nie maja
    trwale = rodziny['trwale (Re>0, bez obrotu)'][:args.modes]
    laczenia = {'trwale + obrotowe': np.concatenate([trwale, rodziny['obrotowe (Im!=0)'][:args.modes]]),
                'trwale + przemienne': np.concatenate([trwale, rodziny['przemienne (Re<0, bez obrotu)'][:args.modes]]),
                'wszystkie trzy rodziny': np.concatenate([trwale, rodziny['obrotowe (Im!=0)'][:args.modes],
                                                          rodziny['przemienne (Re<0, bez obrotu)'][:args.modes]])}
    raport['laczenia'] = {}
    for nazwa, cols in laczenia.items():
        kod = coordinates(prof, mask, score, mu, u, cols)
        ok = have & np.isfinite(kod).all(1)
        auc = patch_auc(prof, truth, mu, u, cols, clips, rows_all)
        wyn = {k: skill(kod[ok], y[ok], clips[ok]) for k, y in cele.items()}
        raport['laczenia'][nazwa] = dict(mody=len(cols), odroznienie_auc=auc, cele=wyn)
        print(f'{nazwa:34s} {"":6s} {"":7s} {auc:11.3f} {"":10s} ' + ' '.join(f'{wyn[k]:23.1f}%' for k in cele), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
