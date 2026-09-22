#!/usr/bin/env python3
"""D-optimal (leverage) frame selection from control theory, tested without any detector training.

System identification says which data identify an operator best: for the ridge regression
psi_{t+1} = K psi_t the covariance of the estimate is proportional to (Psi_S Psi_S^T + lambda I)^-1,
so the frame that shrinks it most is the one with the largest leverage on the current set S

    s_t = psi_t^T (Psi_S Psi_S^T + lambda I)^-1 psi_t          (greedy log det, Ott 2024; Abraham 2019)

Here psi_t = [z_t, z_{t-1}] is the delay-lifted DINOv2 latent of a pool frame. This picks frames
from RARELY VISITED regions of the latent trajectory space, not frames with a large residual.

Arms compared at equal budget, all starting from the same L0:
  uniform_time    evenly in time (the winner of stage 1 among simple arms)
  random
  residual_top    largest Langevin residual, spaced, group-percentile ranked (what K1 buys)
  doptimal        greedy leverage with the same spacing rule
Measured on the unlabelled pool (no labels anywhere):
  logdet          log det(Psi_S Psi_S^T + lambda I): information about the operator
  trace_cov       trace of the inverse: operator uncertainty (lower is better)
  one_step        out-of-sample one-step error of the operator fitted on S, relative to
                  "no movement", on transitions of groups NOT touched by S
The pre-registered expectation: doptimal lowers trace_cov monotonically, residual does not.
Whether the DETECTOR benefits is a separate question that needs training (a stage-2 arm).

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.leverage_selection \
        --pool .../pool_manifest.jsonl --cache .../cache/dinov2/<id> \
        --initial-selection .../variants/tracking_uniform_time_b500_s42.json --out REPORT_leverage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..active_learning.acquisition import select_batch
from .latent_langevin import LatentLangevin
from .select import load_cache, observation_rows, read_jsonl


def leverage_greedy(psi: np.ndarray, start: np.ndarray, candidates: np.ndarray, need: int, lam: float,
                    video: np.ndarray, frame: np.ndarray, min_gap: float) -> list[int]:
    """Greedy argmax of psi^T (Psi_S Psi_S^T + lam I)^-1 psi with a Sherman-Morrison update."""
    d = psi.shape[1]
    inv = np.linalg.inv(psi[start].T @ psi[start] + lam * np.eye(d))
    chosen_times: dict[str, list[float]] = {}
    for i in start:
        chosen_times.setdefault(str(video[i]), []).append(float(frame[i]))
    taken = set(start.tolist())
    picked = []
    for _ in range(need):
        scores = np.einsum('ij,jk,ik->i', psi[candidates], inv, psi[candidates])
        order = np.argsort(-scores)
        for j in order:
            i = int(candidates[j])
            if i in taken:
                continue
            times = chosen_times.get(str(video[i]), [])
            if times and min(abs(float(frame[i]) - t) for t in times) < min_gap:
                continue
            break
        else:
            break
        v = inv @ psi[i]
        inv -= np.outer(v, v) / (1.0 + psi[i] @ v)
        taken.add(i)
        picked.append(i)
        chosen_times.setdefault(str(video[i]), []).append(float(frame[i]))
    return picked


def operator_report(psi: np.ndarray, step: np.ndarray, selected: np.ndarray, holdout: np.ndarray, lam: float) -> dict:
    d = psi.shape[1]
    gram = psi[selected].T @ psi[selected] + lam * np.eye(d)
    sign, logdet = np.linalg.slogdet(gram)
    inv = np.linalg.inv(gram)
    k = inv @ (psi[selected].T @ step[selected])                       # ridge: step ~ psi K
    pred = psi[holdout] @ k
    err = ((pred - step[holdout]) ** 2).sum()
    return dict(logdet=float(logdet), trace_cov=float(np.trace(inv)),
                one_step_skill=float(100 * (1 - err / (step[holdout] ** 2).sum())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--initial-selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[100, 250, 500, 1000])
    ap.add_argument('--dim', type=int, default=24)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--min-gap-frames', type=float, default=5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)
    pool = [r for r in read_jsonl(args.pool) if r.get('split', 'train') == 'train']
    cache_ids, cache_x = load_cache(args.cache)
    position = {f: i for i, f in enumerate(cache_ids)}
    observed = observation_rows(cache_ids)
    group_of = {r['video_id']: r['group_id'] for r in pool}
    for r in observed:
        r['group_id'] = group_of.get(r['video_id'], r['video_id'])
    model = LatentLangevin(observed, cache_x, seed=args.seed, dim=args.dim, lags=3, folds=3, max_gap_frames=7.5)
    w = model.windows                                                   # [n, 4]: t-3 .. t (target)
    n_obs = len(observed)
    psi_full = np.full((n_obs, 2 * args.dim), np.nan)
    step_full = np.full((n_obs, args.dim), np.nan)
    psi_full[w[:, -2]] = np.hstack([model.z[w[:, -2]], model.z[w[:, -3]]])    # lifted state at t-1 ...
    step_full[w[:, -2]] = model.z[w[:, -1]] - model.z[w[:, -2]]               # ... and its step to t
    residual = np.full(n_obs, np.nan)
    residual[w[:, -1]] = model.err_pool
    video = np.array([r['video_id'] for r in observed])
    frame = np.array([float(r['frame_index']) for r in observed])
    group = np.array([r['group_id'] for r in observed])

    pool_rows = np.array([position[r['frame_id']] for r in pool])
    usable = pool_rows[np.isfinite(psi_full[pool_rows]).all(1)]
    raw = json.loads(args.initial_selection.read_text(encoding='utf-8'))
    l0 = np.array([position[f] for f in raw.get('selected_frame_ids', raw.get('frame_ids'))])
    l0_usable = l0[np.isfinite(psi_full[l0]).all(1)]
    print(f'pula {len(pool)} klatek, z podniesionym stanem {len(usable)}; L0 {len(l0)}, z podniesionym stanem {len(l0_usable)}', flush=True)

    pool_by_row = {position[r['frame_id']]: r for r in pool}
    rows_usable = [pool_by_row[i] for i in usable]
    x_usable = cache_x[usable]
    idx_of = {position[r['frame_id']]: k for k, r in enumerate(rows_usable)}
    need = max(args.budgets)
    rng = np.random.default_rng(args.seed)
    pct = {}
    for g in set(group[usable].tolist()):
        members = usable[group[usable] == g]
        order = np.argsort(residual[members])
        for rank, i in enumerate(members[order]):
            pct[i] = (rank + .5) / len(members)
    arms = {}
    l0_ids = [pool_by_row[i]['frame_id'] for i in l0]
    for name, strategy in (('uniform_time', 'uniform_time'), ('random', 'random'), ('residual_top', 'coverage_residual')):
        scores = {r['frame_id']: dict(residual_mahalanobis=pct[position[r['frame_id']]]) for r in rows_usable} if strategy == 'coverage_residual' else None
        chosen, _ = select_batch(rows_usable, x_usable, batch_size=need, strategy=strategy, seed=args.seed,
                                 selected_ids=[f for f in l0_ids if position[f] in idx_of], scores=scores,
                                 min_gap_frames=args.min_gap_frames, coverage_weight=0.0, auxiliary_weight=1.0)
        arms[name] = [position[f] for f in chosen]
    arms['doptimal'] = leverage_greedy(psi_full, l0_usable, usable, need, args.lam, video, frame, args.min_gap_frames)
    report = dict(budgets=args.budgets, lam=args.lam, arms={})
    for name, order in arms.items():
        report['arms'][name] = {}
        for b in args.budgets:
            sel = np.concatenate([l0_usable, np.array(order[:b], dtype=int)])
            touched = set(group[sel].tolist())
            holdout = np.array([i for i in usable if group[i] not in touched and np.isfinite(step_full[i]).all()])
            r = operator_report(psi_full, step_full, sel, holdout, args.lam)
            r['holdout_transitions'] = int(len(holdout))
            report['arms'][name][b] = r
        row = report['arms'][name][max(args.budgets)]
        print(f'{name:14s} logdet {row["logdet"]:8.1f}  trace_cov {row["trace_cov"]:.4f}  one-step skill poza grupami {row["one_step_skill"]:5.1f} '
              f'({row["holdout_transitions"]} przejsc)  |  ' + '  '.join(f'b{b}: {report["arms"][name][b]["trace_cov"]:.4f}' for b in args.budgets), flush=True)
    a, b = set(arms['doptimal'][:1000]), set(arms['residual_top'][:1000])
    report['overlap_doptimal_residual'] = len(a & b) / max(len(a), 1)
    report['overlap_doptimal_uniform'] = len(a & set(arms['uniform_time'][:1000])) / max(len(a), 1)
    print(f'wspolne klatki: D-optymalny z resztowym {report["overlap_doptimal_residual"]:.2f}, z rownomiernym {report["overlap_doptimal_uniform"]:.2f}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
