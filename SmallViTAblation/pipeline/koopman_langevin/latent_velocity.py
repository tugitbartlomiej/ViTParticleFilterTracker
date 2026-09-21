#!/usr/bin/env python3
"""Is the instrument velocity in the TRAJECTORY of the detector latent, and over what time base?

One latent vector of a one-frame detector cannot hold velocity. The difference of two
neighbouring latents does not recover it either (document 38): the error of reading the
position out of the latent is larger than the motion between frames. This tool asks the next
question: if the position read-out is followed over a window of +-k steps and its local
linear slope is taken, does the slope follow the slope of the ground-truth track?

Three sources of position are compared on the same frames:
  latent box_pool, latent query   cross-fitted linear read-out of position (other operations)
  detector box                    centre of the predicted box, no latent involved (control)
and the weights before training serve as the floor.

Correlation is the measure of information. R^2 of the raw slope is reported as well, but a raw
slope is unbiased and noisy; an estimator with a gain below one (Kalman filter, that is a
Langevin state model with observation noise) is what turns this information into a usable state.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.latent_velocity \
        --package SmallViTAblation/feature_cache/detector_probe_clips \
        --latents coco=.../LATENT_coco.npz b2500=.../LATENT_ut_b2500_best.npz --out .../REPORT_velocity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..latent_dynamics.gle import _folds, _ridge_fit_predict
from . import epoch_dynamics as ed


def out_of_fold_position(z: np.ndarray, state: dict, alpha: float = 1e-1, seed: int = 0) -> np.ndarray:
    y, groups = state['targets']['position'], state['group']
    ok = np.isfinite(y).all(1)
    fold = _folds(groups, 4, seed)
    out = np.full_like(y, np.nan)
    for f in sorted(set(fold.tolist())):
        fit, new = np.flatnonzero((fold != f) & ok), np.flatnonzero(fold == f)
        out[new] = _ridge_fit_predict(z[fit], y[fit], z[new], alpha)
    return out


def local_slope(p: np.ndarray, clip: np.ndarray, number: np.ndarray, k: int) -> np.ndarray:
    """Least-squares slope over +-k steps inside one clip; NaN at the edges and across gaps."""
    v = np.full_like(p, np.nan)
    t = np.arange(-k, k + 1, dtype=float)
    for name in np.unique(clip):
        w = np.flatnonzero(clip == name)
        w = w[np.argsort(number[w])]
        step = np.diff(number[w])
        if len(step) == 0 or not np.all(step == step[0]):
            continue
        for i in range(k, len(w) - k):
            seg = p[w[i - k:i + k + 1]]
            if np.isfinite(seg).all():
                v[w[i]] = (t[:, None] * seg).sum(0) / (t ** 2).sum()
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--latents', nargs='+', required=True, help='tag=path')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--windows', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--dim', type=int, default=24)
    args = ap.parse_args(argv)
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    state, rows = None, []
    for item in args.latents:
        tag, path = item.split('=', 1)
        with np.load(path, allow_pickle=False) as data:
            d = {k: data[k] for k in ('frame_ids', 'query', 'box_pool', 'pred_box')}
        if state is None:
            state = ed.physical_state(tracks, [str(f) for f in d['frame_ids'].tolist()])
        truth = state['targets']['position']
        sources = {'latent box_pool': out_of_fold_position(ed.pca(d['box_pool'].astype(np.float64), args.dim)[0], state),
                   'latent query': out_of_fold_position(ed.pca(d['query'].astype(np.float64), args.dim)[0], state),
                   'detector box': d['pred_box'][:, :2] + d['pred_box'][:, 2:] / 2}
        for k in args.windows:
            v_true = local_slope(truth, state['clip'], state['number'], k)
            speed = np.linalg.norm(v_true, axis=1)
            keep = np.isfinite(speed) & (speed <= np.nanpercentile(speed, 99))     # annotation jumps out
            for name, p in sources.items():
                v = local_slope(p, state['clip'], state['number'], k)
                m = keep & np.isfinite(v).all(1)
                r2 = 1 - ((v[m] - v_true[m]) ** 2).sum() / ((v_true[m] - v_true[m].mean(0)) ** 2).sum()
                corr = float(np.mean([np.corrcoef(v[m][:, j], v_true[m][:, j])[0, 1] for j in range(2)]))
                rows.append(dict(weights=tag, source=name, window_steps=k, correlation=corr, r2_raw_slope=float(r2),
                                 frames=int(m.sum())))
                print(f'{tag:10s} {name:16s} okno +-{k}: korelacja {corr:+.2f}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(rows=rows, labels_used_for='scoring only'), indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
