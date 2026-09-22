#!/usr/bin/env python3
"""Czy z samych niezmiennikow da sie odtworzyc ksztalt narzedzia w ramce (pytanie autora, 2026-09-22).

Bierzemy wspolrzedne klatki w kilku najtrwalszych modach operatora czasowego (te, ktore po klatce
wracaja prawie takie same), zerujemy WSZYSTKO inne i probujemy z nich odtworzyc:

  wycinek obrazu    ramka narzedzia wyciete z klatki, rozciagniete do stalego prostokata
  opis wycinka      te same tokeny, ktore operator widzial (kontrola: ile w ogole da sie odtworzyc)

Odtwarzanie liniowe (grzbietowe), uczenie i ocena na ROZNYCH filmach. Porownania:
  wszystkie wspolrzedne   sufit dla tej reprezentacji
  same niezmienniki       to, o co pyta autor
  tyle samo skladowych PCA   kontrola: czy liczy sie trwalosc, czy sama liczba wymiarow
  srednia po zbiorze uczacym   podloga (odtwarzanie bez zadnej informacji z klatki)

Na koniec rysunek: prawdziwy wycinek, odtworzenie z niezmiennikow, odtworzenie ze wszystkiego.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.ksztalt_z_niezmiennikow \\
        --recording .../CZASTKI_b0500 --package .../detector_probe_clips --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .niezmiennik_czasowy import crop_descriptor, edmd
from .patch_analysis import load_recording


def ridge_cv(x: np.ndarray, y: np.ndarray, group: np.ndarray, lam: float = 1.0, folds: int = 4, seed: int = 0):
    names = np.array(sorted(set(group.tolist())))
    fold = {v: i % folds for i, v in enumerate(np.random.default_rng(seed).permutation(names))}
    part = np.array([fold[v] for v in group])
    xx = np.hstack([(x - x.mean(0)) / np.maximum(x.std(0), 1e-9), np.ones((len(x), 1))])
    pred = np.zeros_like(y)
    for f in range(folds):
        tr, te = part != f, part == f
        if not tr.any() or not te.any():
            continue
        a = xx[tr].T @ xx[tr] + lam * np.eye(xx.shape[1])
        beta = np.linalg.solve(a, xx[tr].T @ y[tr])
        pred[te] = xx[te] @ beta
        pred[te] += y[tr].mean(0) - (xx[tr] @ beta).mean(0)
    r2 = float(1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())
    return r2, pred


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--max-gap', type=int, default=4)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--crop', type=int, nargs=2, default=[32, 24], help='szerokosc i wysokosc wycinka obrazu')
    args = ap.parse_args(argv)
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rec = load_recording(args.recording)
    grid = tuple(rec['meta']['grid'])
    fids = rec['frame_ids']
    at = {f: i for i, f in enumerate(fids)}
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    box = np.full((len(fids), 4), np.nan)
    for clip, t in tracks.items():
        for i, n in enumerate(t['frame']):
            j = at.get(f'cataract_lmm/tracking/{clip}/{n:08d}')
            if j is not None and t['box'][i]:
                box[j] = t['box'][i]
    prof = rec['profil_red'].astype(np.float64)
    x = crop_descriptor(prof, box, grid, args.cells)
    ok = np.isfinite(x).all(1) & np.isfinite(box).all(1)
    rows = np.flatnonzero(ok)
    video = np.array([f.split('/')[-2] for f in fids])
    frame = np.array([int(f.split('/')[-1]) for f in fids], float)
    xm = x[rows].astype(np.float64)
    xm = xm - xm.mean(0)
    _, s, vt = np.linalg.svd(xm, full_matrices=False)
    base = vt[:args.dim].T / np.maximum(s[:args.dim] / np.sqrt(len(xm)), 1e-9)
    z = xm @ base

    pairs = []
    for v in np.unique(video[rows]):
        idx = rows[video[rows] == v]
        idx = idx[np.argsort(frame[idx])]
        d = np.diff(frame[idx])
        pos = {r: k for k, r in enumerate(idx)}
        for i in np.flatnonzero(d <= args.max_gap):
            pairs.append((pos[idx[i]], pos[idx[i + 1]]))
    pairs = np.array(pairs)
    lam, modes, mu = edmd(z[pairs[:, 0]], z[pairs[:, 1]])
    print(f'klatek {len(rows)}, par {len(pairs)}, |lambda| {np.round(np.abs(lam[:args.modes]), 3).tolist()}', flush=True)
    c = (z - mu) @ modes[:, :args.modes]
    niezm = np.hstack([c.real, c.imag]) if np.abs(modes[:, :args.modes].imag).sum() else c.real

    # prawdziwe wycinki obrazu
    w_c, h_c = args.crop
    y = np.zeros((len(rows), h_c * w_c), dtype=np.float32)
    for k, t in enumerate(rows):
        clip, n = fids[t].split('/')[-2], fids[t].split('/')[-1]
        im = Image.open(args.package / 'frames' / clip / f'{n}.jpg').convert('L')
        b = box[t]
        im = im.crop((b[0], b[1], b[0] + b[2], b[1] + b[3])).resize((w_c, h_c))
        y[k] = np.asarray(im, dtype=np.float32).ravel() / 255.0
    print(f'wycinki obrazu: {w_c} na {h_c} pikseli', flush=True)

    warianty = {'wszystkie wspolrzedne opisu': z,
                f'same niezmienniki ({niezm.shape[1]} liczb)': niezm,
                f'kontrola PCA ({niezm.shape[1]} skladowych)': z[:, :niezm.shape[1]]}
    wynik, pred_of = {}, {}
    for nazwa, feat in warianty.items():
        r2_img, pred = ridge_cv(feat, y, video[rows])
        r2_opis, _ = ridge_cv(feat, xm, video[rows])
        wynik[nazwa] = dict(r2_obraz=r2_img, r2_opis=r2_opis, wymiarow=int(feat.shape[1]))
        pred_of[nazwa] = pred
        print(f'{nazwa:38s} wymiarow {feat.shape[1]:3d}  R2 obrazu {r2_img:6.3f}  R2 opisu tokenow {r2_opis:6.3f}', flush=True)

    out_png = args.out.with_suffix('.png')
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), 6, replace=False)
    fig, ax = plt.subplots(3, 6, figsize=(12, 5))
    klucz_n = [k for k in warianty if k.startswith('same')][0]
    klucz_w = [k for k in warianty if k.startswith('wszystkie')][0]
    for j, k in enumerate(pick):
        for r, (img, tytul) in enumerate(((y[k], 'prawdziwy wycinek'), (pred_of[klucz_n][k], 'z samych niezmiennikow'),
                                          (pred_of[klucz_w][k], 'ze wszystkich wspolrzednych'))):
            ax[r, j].imshow(img.reshape(h_c, w_c), cmap='gray', vmin=0, vmax=1)
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if j == 0:
                ax[r, j].set_ylabel(tytul, fontsize=8)
    fig.suptitle('Odtworzenie wycinka narzedzia; uczenie i ocena na roznych filmach', y=.98)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(frames=len(rows), pairs=int(len(pairs)), modes=args.modes,
                                        lam_abs=[float(abs(v)) for v in lam[:args.modes]], warianty=wynik,
                                        rysunek=str(out_png)), indent=2, ensure_ascii=False), encoding='utf-8')
    print('zapisane', args.out, 'oraz', out_png, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
