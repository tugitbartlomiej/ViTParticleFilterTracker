#!/usr/bin/env python3
"""Koopman spectrum of the detector latent along training: invariants, static/dynamic split, dimension sweep.

For each checkpoint (same frames = same particles) the latent is lifted with one delay,
[z_t, z_{t-1}], and a linear one-step operator K is fitted by ridge (EDMD) on the training
clips only; the test clips give out-of-sample numbers. Eigenvalues lambda of K are sorted into
  invariant   |lambda| > .97 and |angle| < 5 deg   (a quantity that stays the same frame to frame)
  slow real   .8 < |lambda| <= .97, angle ~ 0      (decays over several frames)
  rotating    |angle| >= 5 deg                     (comes back after 360/angle frames, rotated)
  fast        |lambda| <= .8 or negative real part (gone within a frame or two)
For every eigenfunction: share of its variance BETWEEN videos (identity of the operation) and,
labels for scoring only, how much of it the tool state explains.
Dimension sweep: out-of-sample one-step skill of the linear operator for 8..256 latent
directions, next to the nonlinear (random-feature) Langevin model at the same dimension.
The gap between them is the part a Deep Koopman autoencoder would have to make linear.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.koopman_spectrum_axis --package ... \
        --latents coco=... b0500=... b2500=... --out REPORT_spectrum_axis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import epoch_dynamics as ed
from .latent_langevin import LatentLangevin


def lift(z, state):
    prev = state['previous']
    ok = prev >= 0
    ok2 = ok.copy()
    ok2[ok] = prev[prev[ok]] >= 0
    cur = np.flatnonzero(ok2)
    x0 = np.hstack([z[prev[cur]], z[prev[prev[cur]]]])
    x1 = np.hstack([z[cur], z[prev[cur]]])
    return cur, x0, x1


def fit_operator(x0, x1, alpha=1e-2):
    gram = x0.T @ x0 / len(x0)
    gram.flat[::len(gram) + 1] += alpha
    return np.linalg.solve(gram, x0.T @ x1 / len(x0))


def classify(values):
    mod, ang = np.abs(values), np.degrees(np.abs(np.angle(values)))
    kind = np.where((mod > .97) & (ang < 5), 'invariant',
           np.where((mod > .8) & (ang < 5) & (values.real > 0), 'slow_real',
           np.where(ang >= 5, 'rotating', 'fast')))
    return kind, mod, ang


def between_share(c, videos):
    means = {v: c[videos == v].mean() for v in set(videos.tolist())}
    return float(np.var([means[v] for v in videos]) / max(c.var(), 1e-12))


def nanmean(values):
    return float(np.mean(values)) if len(values) else float('nan')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--latents', nargs='+', required=True)
    ap.add_argument('--which', default='box_pool')
    ap.add_argument('--dims', type=int, nargs='+', default=[8, 16, 24, 32, 64, 128, 256])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    stride = float(next(iter(tracks.values()))['stride'])
    report = dict(which=args.which, checkpoints={})
    state = None
    for item in args.latents:
        tag, path = item.split('=', 1)
        with np.load(path, allow_pickle=False) as data:
            raw, ids = data[args.which].astype(np.float64), [str(f) for f in data['frame_ids'].tolist()]
        if state is None:
            state = ed.physical_state(tracks, ids)
            train = state['split'] == 'train'
            videos = state['clip']
            targets = np.hstack([state['targets']['position'], state['targets']['size']])
        z = ed.pca(raw, 24)[0]
        cur, x0, x1 = lift(z, state)
        fit = train[cur]
        k = fit_operator(x0[fit], x1[fit])
        values, right = np.linalg.eig(k)
        order = np.argsort(-np.abs(values))
        values, right = values[order], right[:, order]
        kind, mod, ang = classify(values)
        coords = x1 @ right
        test = ~fit
        pred = (x0 @ k) @ right
        res = np.abs(coords[test] - pred[test]) ** 2
        defect = res.mean(0) / np.maximum((np.abs(coords[test]) ** 2).mean(0), 1e-12)
        modes = []
        for j in range(len(values)):
            c = coords[:, j]
            feat = np.column_stack([c.real, c.imag]) if ang[j] >= 5 else c.real[:, None]
            ok = np.isfinite(targets[cur]).all(1)
            r2 = ed.cross_fitted_r2(targets[cur][ok], feat[ok], state['group'][cur][ok]) if ok.sum() > 100 else float('nan')
            modes.append(dict(modulus=float(mod[j]), angle_deg=float(np.degrees(np.angle(values[j]))), kind=str(kind[j]),
                              period_frames=float(360 / ang[j] * stride) if ang[j] >= 5 else None,
                              between_videos=between_share(c.real, videos[cur]), r2_from_tool_state=float(r2),
                              out_of_sample_defect=float(defect[j])))
        counts = {c: int((kind == c).sum()) for c in ('invariant', 'slow_real', 'rotating', 'fast')}
        inv = [m for m in modes if m['kind'] == 'invariant']
        rot = [m for m in modes if m['kind'] == 'rotating']
        print('%-8s mody: %s | niezmienniki: miedzy filmami %.2f, ze stanu narzedzia %.2f, defekt %.2f | obracajace: okres %.0f klatek, ze stanu %.2f' % (
            tag, counts, nanmean([m['between_videos'] for m in inv]), nanmean([m['r2_from_tool_state'] for m in inv]),
            nanmean([m['out_of_sample_defect'] for m in inv]),
            float(np.median([m['period_frames'] for m in rot])) if rot else float('nan'),
            nanmean([m['r2_from_tool_state'] for m in rot])), flush=True)
        sweep = {}
        rows = ed.rows_of(state)
        for d in args.dims:
            if d > raw.shape[1]:
                continue
            zd = ed.pca(raw, d)[0]
            cur_d, a0, a1 = lift(zd, state)
            fit_d = train[cur_d]
            kd = fit_operator(a0[fit_d], a1[fit_d], alpha=1e-2)
            step_true = a1[~fit_d][:, :d] - a0[~fit_d][:, :d]
            step_pred = (a0[~fit_d] @ kd)[:, :d] - a0[~fit_d][:, :d]
            lin = float(100 * (1 - ((step_pred - step_true) ** 2).sum() / (step_true ** 2).sum()))
            nl = LatentLangevin(rows, zd, seed=42, dim=d, lags=8, folds=4, max_gap_frames=1.5 * stride).summary()['skill_vs_no_movement']
            sweep[d] = dict(linear_one_step_skill=lin, nonlinear_history_skill=float(nl))
            print('   wymiar %3d: liniowy %5.1f  nieliniowy z historia %5.1f  luka %+5.1f' % (d, lin, nl, nl - lin), flush=True)
        report['checkpoints'][tag] = dict(counts=counts, modes=modes, dimension_sweep=sweep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
