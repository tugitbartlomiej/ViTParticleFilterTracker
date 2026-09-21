#!/usr/bin/env python3
"""Controlled test: which frame selector helps a downstream model at a fixed label budget?

Synthetic videos with three regimes whose ground truth is known:
  common   most frames; slow motion; label is an easy function of the frame
  rare     few frames, only in some videos; its own regular motion; its own label function,
           so the downstream model improves only if such frames get labelled
  burst    few frames; the latent jumps at random (blur, occlusion, cut); the label is noise,
           so labelling these frames teaches nothing

The downstream model stands in for the detector: ridge regression from frame features to
the label, trained on the selected frames only, evaluated on held-out videos on the frames
whose label is learnable (common and rare). All selectors use the production
`select_batch` and see no labels and no regime names.

ASSUMPTION THAT THIS BENCHMARK DOES NOT TEST: frames whose motion nobody can predict are
also frames whose label is not worth buying. The benchmark shows what follows IF that
holds. Whether it holds for surgical video is decided only by detector training.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.synthetic_selection_benchmark --seeds 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..active_learning.acquisition import select_batch
from ..latent_dynamics.gle import diagnose_gle
from .latent_langevin import LatentLangevin

STRATEGIES = ('random', 'uniform_time', 'coverage', 'coverage_residual', 'coverage_reducible',
              'uniform_time_veto', 'coverage_reducible_veto')
VETO_QUANTILE = .90      # frames whose pool-model error is in the top 10 percent are not offered for labelling


def make_pool(seed: int, *, videos: int = 24, frames: int = 240, dim: int = 8,
              rare_share: float = .10, burst_share: float = .10, split: str = 'train'):
    rng = np.random.default_rng(seed)
    w_common = rng.normal(size=dim)
    w_rare = rng.normal(size=dim)
    rot = np.eye(dim)
    c, s = np.cos(.35), np.sin(.35)
    rot[:2, :2] = [[c, -s], [s, c]]
    offset_rare = np.zeros(dim)
    offset_rare[2] = 3.0
    rows, z_all, y_all, regime_all = [], [], [], []
    for v in range(videos):
        z = np.zeros((frames, dim))
        regime = np.zeros(frames, dtype=int)
        has_rare = v % 3 == 0
        t = 1
        z[0] = rng.normal(scale=.5, size=dim)
        while t < frames:
            u = rng.random()
            if has_rare and u < rare_share / 12:
                length, kind = 12, 1
            elif u < rare_share / 12 + burst_share / 6:
                length, kind = 6, 2
            else:
                length, kind = 1, 0
            for _ in range(length):
                if t >= frames:
                    break
                if kind == 0:
                    z[t] = .9 * z[t - 1] + rng.normal(scale=.12, size=dim)
                elif kind == 1:
                    base = z[t - 1] - offset_rare if regime[t - 1] == 1 else rng.normal(scale=.4, size=dim)
                    z[t] = offset_rare + rot @ base * .97 + rng.normal(scale=.05, size=dim)
                else:
                    z[t] = rng.normal(scale=2.0, size=dim)
                regime[t] = kind
                t += 1
        y = np.where(regime == 1, z @ w_rare, z @ w_common) + rng.normal(scale=.05, size=frames)
        y[regime == 2] = rng.normal(scale=3.0, size=(regime == 2).sum())
        for k in range(frames):
            rows.append(dict(frame_id=f'{split}{v}/f{k}', split='train', group_id=f'{split}{v}',
                             video_id=f'{split}{v}', frame_index=k))
        z_all.append(z)
        y_all.append(y)
        regime_all.append(regime)
    return rows, np.vstack(z_all), np.concatenate(y_all), np.concatenate(regime_all), (w_common, w_rare, offset_rare)


def downstream_error(z_fit, y_fit, z_test, y_test, regime_test) -> dict:
    def design(z):
        return np.column_stack([z, z[:, 2:3] > 1.5, z * (z[:, 2:3] > 1.5), np.ones(len(z))])
    a = design(z_fit)
    g = a.T @ a / len(a)
    g.flat[::len(g) + 1] += 1e-3
    w = np.linalg.solve(g, a.T @ y_fit / len(a))
    p = design(z_test) @ w
    err = (p - y_test) ** 2
    return dict(learnable=float(err[regime_test != 2].mean()), common=float(err[regime_test == 0].mean()),
                rare=float(err[regime_test == 1].mean()))


def run(seed: int, budgets=(60, 120, 240), initial: int = 40) -> dict:
    rows, z, y, regime, truth = make_pool(seed)
    # held-out videos from the same generator parameters
    _, z_test, _, regime_test, _ = make_pool(seed + 77, videos=12, split='test')
    w_common, w_rare, _ = truth
    y_test = np.where(regime_test == 1, z_test @ w_rare, z_test @ w_common)
    residual_rows, _, _ = diagnose_gle(rows, z, seed=seed, dim=z.shape[1], lags=3, folds=3, max_gap_frames=1,
                                       max_fit_windows=5000, residual_modes=4)
    residual_scores = {r['frame_id']: r for r in residual_rows}
    langevin = LatentLangevin(rows, z, seed=seed, dim=z.shape[1], lags=4, folds=3, max_gap_frames=1,
                              neighbourhood_frames=3)
    index = {r['frame_id']: i for i, r in enumerate(rows)}
    surprise = np.full(len(rows), np.nan)
    surprise[langevin.target] = langevin.err_pool
    keep = ~(surprise > np.nanquantile(surprise, VETO_QUANTILE))          # frames without a window stay eligible
    kept_rows, kept_z = [r for r, k in zip(rows, keep) if k], z[keep]
    start, _ = select_batch(rows, z, batch_size=initial, strategy='uniform_time', seed=seed, min_gap_frames=2)
    out = {}
    for strategy in STRATEGIES:
        selected = list(start)
        curve = {}
        for budget in budgets:
            need = budget - len(selected)
            if strategy == 'coverage_reducible':
                scores = langevin.scores(selected)
            elif strategy == 'coverage_residual':
                scores = residual_scores
            else:
                scores = None
            base = strategy.replace('_veto', '')
            if base == 'coverage_reducible':
                scores = langevin.scores(selected)
            pool_rows, pool_z = (kept_rows, kept_z) if strategy.endswith('_veto') else (rows, z)
            kept_ids = {r['frame_id'] for r in pool_rows}
            batch, _ = select_batch(pool_rows, pool_z, batch_size=need, strategy=base, seed=seed,
                                    selected_ids=[f for f in selected if f in kept_ids],
                                    scores=scores, min_gap_frames=2)
            selected.extend(batch)
            idx = np.array([index[f] for f in selected])
            curve[budget] = dict(downstream_error(z[idx], y[idx], z_test, y_test, regime_test),
                                 share_rare=float((regime[idx] == 1).mean()),
                                 share_burst=float((regime[idx] == 2).mean()))
        out[strategy] = curve
    out['pool'] = dict(share_rare=float((regime == 1).mean()), share_burst=float((regime == 2).mean()),
                       langevin=langevin.summary())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args(argv)
    runs = [run(s) for s in range(args.seeds)]
    budgets = sorted(runs[0]['random'])
    table = {}
    for strategy in STRATEGIES:
        table[strategy] = {b: {k: float(np.mean([r[strategy][b][k] for r in runs]))
                               for k in ('learnable', 'common', 'rare', 'share_rare', 'share_burst')} for b in budgets}
    last = budgets[-1]
    wins = {s: int(sum(r[s][last]['learnable'] < r['random'][last]['learnable'] for r in runs))
            for s in STRATEGIES if s != 'random'}
    report = dict(seeds=args.seeds, budgets=budgets, mean=table, seeds_better_than_random_at_last_budget=wins,
                  pool=runs[0]['pool'], labels_seen_by_selectors=False,
                  untested_assumption='frames with unpredictable motion have labels that are not worth buying')
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding='utf-8')
    for strategy in STRATEGIES:
        row = table[strategy][last]
        print(f'{strategy:20s} blad na klatkach uczalnych {row["learnable"]:.3f}  rzadki rezim {row["rare"]:.3f}  '
              f'udzial rzadkich {row["share_rare"]:.2f}  udzial zaklocen {row["share_burst"]:.2f}')
    print('lepsze od losowania (liczba ziaren z', args.seeds, '):', wins, '| pula:', runs[0]['pool'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
