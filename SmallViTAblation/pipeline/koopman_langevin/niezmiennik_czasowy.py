#!/usr/bin/env python3
"""Niezmienniki Koopmana WYCINKA narzedzia, klatka po klatce (pomysl autora, 2026-09-22).

Zamiast parowac pojedyncze tokeny przy przesunieciu narzedzia, opisujemy CALY wycinek w ramce
narzedzia i pytamy, co w nim nie zmienia sie z klatki na klatke.

  wycinek     tokeny lezace w PRAWDZIWEJ ramce narzedzia, przepisane na stala siatke R na C
              we wspolrzednych wzglednych ramki (lewy gorny rog ramki to zawsze ta sama komorka),
              wiec opis nie zalezy od wielkosci ani polozenia ramki w kadrze
  krok        nastepna klatka tego samego filmu (odstep `--max-gap` klatek albo mniej)
  operator    x_(t+1) ~ x_t K, bez symetryzacji, liczony po redukcji PCA do `--dim` wymiarow
  niezmiennik mod o module wartosci wlasnej najblizszym jedynki: wielkosc, ktora po klatce
              wraca prawie taka sama

Podlogi, bez ktorych liczby nic nie znacza:
  czas potasowany   te same klatki, pary losowane bez kolejnosci (ile "trwalosci" daje sam bezruch)
  pary miedzy filmami   krok laczy rozne filmy (ile daje sama tozsamosc operacji)

Nastepnie sprawdzamy, co niosa niezmienniki (etykiety WYLACZNIE do oceny, podzial po filmach)
i jak maja sie do modow operatora TRANSLACYJNEGO z `widmo_tozsamosci.py` (kosinusy glowne).

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.niezmiennik_czasowy \\
        --recording .../CZASTKI_b0500 --package .../detector_probe_clips --out REPORT.json
    python -P niezmiennik_czasowy.py --recording .../CZASTKI_pool_b0500 \\
        --pool .../pool_manifest.jsonl --max-gap 6 --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .patch_analysis import generalised_koopman, load_recording, principal_cosines, translational_pairs


def crop_descriptor(prof: np.ndarray, box: np.ndarray, grid: tuple[int, int], cells: int) -> np.ndarray:
    """Tokeny z wnetrza prawdziwej ramki przepisane na siatke `cells` na `cells` we wspolrzednych ramki."""
    h, w = grid
    ys = (np.arange(h) + .5) / h * 480
    xs = (np.arange(w) + .5) / w * 720
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    gy, gx = gy.ravel(), gx.ravel()
    k = prof.shape[-1]
    out = np.full((len(prof), cells * cells * k), np.nan, dtype=np.float32)
    for t in range(len(prof)):
        x0, y0, bw, bh = box[t]
        if not np.isfinite(box[t]).all() or bw < 8 or bh < 8:
            continue
        u = (gx - x0) / bw
        v = (gy - y0) / bh
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        if inside.sum() < 4:
            continue
        cell = (np.clip((v[inside] * cells).astype(int), 0, cells - 1) * cells
                + np.clip((u[inside] * cells).astype(int), 0, cells - 1))
        acc = np.zeros((cells * cells, k))
        cnt = np.zeros(cells * cells)
        np.add.at(acc, cell, prof[t][inside])
        np.add.at(cnt, cell, 1)
        empty = cnt == 0
        acc[~empty] /= cnt[~empty, None]
        if empty.any():                                   # puste komorki dostaja srednia z wypelnionych
            acc[empty] = acc[~empty].mean(0)
        out[t] = acc.ravel()
    return out


def edmd(x0: np.ndarray, x1: np.ndarray, reg: float = 1e-3):
    mu = x0.mean(0)
    a, b = x0 - mu, x1 - mu
    c00 = a.T @ a / len(a) + reg * np.eye(a.shape[1])
    k = np.linalg.solve(c00, a.T @ b / len(a))
    lam, u = np.linalg.eig(k)
    order = np.argsort(-np.abs(lam))
    return lam[order], u[:, order], mu


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recording', type=Path, required=True)
    ap.add_argument('--package', type=Path, help='katalog z tracks.json (zestaw gesty)')
    ap.add_argument('--pool', type=Path, help='pool_manifest.jsonl (pula ucząca)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--max-gap', type=int, default=4, help='najwiekszy odstep klatek uznany za krok')
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    rec = load_recording(args.recording)
    grid = tuple(rec['meta']['grid'])
    fids = rec['frame_ids']
    prof = rec['profil_red'].astype(np.float64)
    at = {f: i for i, f in enumerate(fids)}

    box = np.full((len(fids), 4), np.nan)
    video = np.array([f.split('/')[-2] for f in fids])
    frame = np.array([int(f.split('/')[-1]) for f in fids], float)
    if args.package:
        tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
        for clip, t in tracks.items():
            for i, n in enumerate(t['frame']):
                j = at.get(f'cataract_lmm/tracking/{clip}/{n:08d}')
                if j is not None and t['box'][i]:
                    box[j] = t['box'][i]
    else:
        for line in args.pool.open(encoding='utf-8'):
            r = json.loads(line)
            j = at.get(r['frame_id'])
            ann = r.get('annotations') or []
            if j is not None and ann and r.get('split', 'train') == 'train':
                box[j] = ann[0]['bbox']
    have = np.isfinite(box).all(1)
    print(f'klatek {len(fids)}, z zaznaczonym narzedziem {int(have.sum())}, filmow {len(set(video[have].tolist()))}', flush=True)

    x = crop_descriptor(prof, box, grid, args.cells)
    ok = np.isfinite(x).all(1) & have
    xm = x[ok].astype(np.float64)
    xm = xm - xm.mean(0)
    u_pca, s, vt = np.linalg.svd(xm, full_matrices=False)
    base = vt[:args.dim].T / np.maximum(s[:args.dim] / np.sqrt(len(xm)), 1e-9)    # bielenie
    z = np.full((len(fids), args.dim), np.nan)
    z[ok] = xm @ base
    print(f'opis wycinka: {args.cells}x{args.cells} komorek po {prof.shape[-1]} liczb; udzial wariancji w {args.dim} wymiarach '
          f'{100 * (s[:args.dim] ** 2).sum() / (s ** 2).sum():.1f}%', flush=True)

    rows = np.flatnonzero(ok)
    pairs = []
    for v in np.unique(video[rows]):
        idx = rows[video[rows] == v]
        idx = idx[np.argsort(frame[idx])]
        d = np.diff(frame[idx])
        for i in np.flatnonzero(d <= args.max_gap):
            pairs.append((idx[i], idx[i + 1]))
    pairs = np.array(pairs)
    print(f'par klatka-do-nastepnej: {len(pairs)} (odstep do {args.max_gap} klatek)', flush=True)
    rng = np.random.default_rng(args.seed)
    lam, modes, mu = edmd(z[pairs[:, 0]], z[pairs[:, 1]])

    # podlogi
    shuffled = pairs.copy()
    for v in np.unique(video[pairs[:, 0]]):                                     # czas potasowany wewnatrz filmu
        m = np.flatnonzero(video[pairs[:, 0]] == v)
        shuffled[m, 1] = pairs[rng.permutation(m), 1]
    lam_shuf, _, _ = edmd(z[shuffled[:, 0]], z[shuffled[:, 1]])
    across = np.column_stack([rows[rng.permutation(len(rows))], rows[rng.permutation(len(rows))]])
    lam_across, _, _ = edmd(z[across[:, 0]], z[across[:, 1]])

    top = np.abs(lam[:args.modes])
    print(f'\n|lambda| osmiu najtrwalszych modow: {np.round(top, 3).tolist()}')
    print(f'podloga, czas potasowany:            {np.round(np.abs(lam_shuf[:args.modes]), 3).tolist()}')
    print(f'podloga, pary miedzy filmami:        {np.round(np.abs(lam_across[:args.modes]), 3).tolist()}')

    # co niosa niezmienniki
    cols = np.arange(args.modes)
    c = (z[rows] - mu) @ modes[:, cols]
    kod = np.hstack([c.real, c.imag]) if np.abs(modes[:, cols].imag).sum() else c.real
    b = box[rows]
    cele = {'wydluzenie ramki': np.log(b[:, 2] / np.maximum(b[:, 3], 1)),
            'wielkosc ramki': np.log(b[:, 2] * b[:, 3]),
            'polozenie w kadrze (poziomo)': b[:, 0] + b[:, 2] / 2}
    niesie = {k: skill(kod, y, video[rows]) for k, y in cele.items()}
    print('\nco niosa niezmienniki (uczenie i ocena na roznych filmach): '
          + '  '.join(f'{k} {v:.1f}%' for k, v in niesie.items()))
    mid = {}
    for v in np.unique(video[rows]):
        mid[v] = kod[video[rows] == v].mean(0)
    within = ((kod - np.array([mid[v] for v in video[rows]])) ** 2).sum()
    miedzy = float(100 * (1 - within / (kod ** 2).sum()))
    print(f'udzial zmiennosci MIEDZY filmami: {miedzy:.0f}%')

    # jak maja sie do operatora TRANSLACYJNEGO (na tokenach)
    kos = None
    if rec['maska_pred'].sum():
        t0, t1 = translational_pairs(rec, grid, per_clip=200)
        if len(t0) > 1000:
            _, tmod, _ = generalised_koopman(t0, t1)
            # niezmiennik czasowy sprowadzony do przestrzeni tokenu: srednia komorek wycinka
            k = prof.shape[-1]
            w_tok = (base @ modes[:, cols].real).reshape(args.cells * args.cells, k, len(cols)).mean(0)
            kos = principal_cosines(w_tok, tmod[:, :args.modes].real).tolist()
            print(f'kosinusy glowne z modami translacyjnymi: {np.round(kos, 2).tolist()}')

    raport = dict(frames=int(len(fids)), with_box=int(have.sum()), pairs=int(len(pairs)), max_gap=args.max_gap,
                  cells=args.cells, dim=args.dim, variance_share=float((s[:args.dim] ** 2).sum() / (s ** 2).sum()),
                  lam_abs=[float(v) for v in top], lam_shuffled=[float(abs(v)) for v in lam_shuf[:args.modes]],
                  lam_across_videos=[float(abs(v)) for v in lam_across[:args.modes]],
                  carries=niesie, between_video_share=miedzy, principal_cosines_vs_translational=kos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
