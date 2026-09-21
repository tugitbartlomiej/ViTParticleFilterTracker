#!/usr/bin/env python3
"""Does the detector latent crystallise the state space of the instrument while it trains?

Input: one `LATENT_<tag>.npz` per checkpoint (see `detector_latent.py`), all on the same
frames, ordered along the training clock, plus the ground-truth tracks of those frames.
Each frame is one particle. Two clocks are analysed:

VIDEO CLOCK (within one checkpoint, frame to frame)
  probe         cross-fitted linear read-out of the physical state from the latent:
                position, size, axis direction, velocity. `single` reads one frame,
                `pair` reads [z_t, z_t - z_{t-1}]. A one-frame detector cannot hold velocity
                in one latent vector; if velocity is readable it must come from the pair.
                `box_pair` is the control: the same read-out from predicted boxes alone.
  langevin      z_{t+1} = z_t + D(z, v, a) + memory + e, cross-fitted over operation groups:
                share of the next step it predicts, gain from the memory kernel,
                lag-1 correlation of the residual (white residual = nothing left to model)
  koopman       linear one-step operator on [z_t, z_{t-1}]; how much of the physical state the
                few slowest eigenfunctions carry, against the same number of leading
                principal directions (control)
  geometry      participation ratio = how many directions the latent really uses

TRAINING CLOCK (same frames, consecutive checkpoints)
  displacement  how far each particle moved between checkpoints after the best rotation
                (orthogonal Procrustes), and linear CKA between the two latents
  detection     mean IoU of the top box with the ground truth, the learning curve itself

Labels are used here for scoring only. Nothing in this file ranks frames.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.epoch_dynamics \
        --package SmallViTAblation/feature_cache/detector_probe_clips \
        --latents coco=.../LATENT_coco.npz b500=.../LATENT_b500.npz b2500=.../LATENT_b2500.npz \
        --out SmallViTAblation/results/koopman_langevin_epochs/report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..latent_dynamics.gle import _folds, _ridge_fit_predict
from .latent_langevin import LatentLangevin

ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
STATE_NAMES = ('position', 'size', 'axis', 'velocity', 'velocity_trimmed')


# --------------------------------------------------------------------------- ground truth

def physical_state(tracks: dict, frame_ids: list[str]) -> dict:
    """Per-frame ground truth aligned with `frame_ids`; NaN where the annotation is missing."""
    n = len(frame_ids)
    clip = np.array([f.split('/')[-2] for f in frame_ids])
    number = np.array([int(f.split('/')[-1]) for f in frame_ids])
    box = np.full((n, 4), np.nan)
    tip = np.full((n, 2), np.nan)
    group = np.empty(n, dtype=object)
    split = np.empty(n, dtype=object)
    for name, t in tracks.items():
        where = np.flatnonzero(clip == name)
        at = {f: i for i, f in enumerate(t['frame'])}
        for i in where:
            k = at[int(number[i])]
            if t['box'][k] is not None:
                box[i] = t['box'][k]
            if t['tip'][k] is not None:
                tip[i] = t['tip'][k]
        group[where], split[where] = t['group_id'], t['split']
    centre = box[:, :2] + box[:, 2:] / 2
    size = np.log(np.maximum(box[:, 2:], 1.0))
    direction = tip - centre
    axis = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
    velocity = np.full((n, 2), np.nan)
    previous = np.full(n, -1)
    for name in np.unique(clip):
        where = np.flatnonzero(clip == name)
        where = where[np.argsort(number[where])]
        step = np.diff(number[where])
        regular = step == np.median(step)
        previous[where[1:][regular]] = where[:-1][regular]
        inner = where[1:-1][regular[:-1] & regular[1:]]
        velocity[inner] = (centre[where[2:]] - centre[where[:-2]])[regular[:-1] & regular[1:]] / 2
    # Added after the first read-out (post hoc, declared in document 38): 1 percent of frames held
    # 58 percent of the squared velocity (annotation jumps, instrument re-entry), so the plain
    # target measures the jumps. The trimmed target drops frames above the 99th percentile of speed.
    speed = np.linalg.norm(velocity, axis=1)
    trimmed = velocity.copy()
    trimmed[speed > np.nanpercentile(speed, 99)] = np.nan
    return dict(clip=clip, number=number, group=group.astype(str), split=split.astype(str), box=box,
                previous=previous, targets=dict(position=centre, size=size, axis=axis, velocity=velocity,
                                                velocity_trimmed=trimmed))


# --------------------------------------------------------------------------- probes

def cross_fitted_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray, *, folds: int = 4, seed: int = 0) -> float:
    """Out-of-group R^2 of a ridge read-out; the ridge strength is chosen inside the fitting groups."""
    ok = np.isfinite(x).all(1) & np.isfinite(y).all(1)
    x, y, groups = x[ok], y[ok], groups[ok]
    if len(set(groups.tolist())) < 3 or len(x) < 50:
        return float('nan')
    fold = _folds(groups, folds, seed)
    sse = sst = 0.0
    for f in sorted(set(fold.tolist())):
        fit, new = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        names = sorted(set(groups[fit].tolist()))
        half = np.isin(groups[fit], names[::2])
        a, b = fit[half], fit[~half]
        errors = [((_ridge_fit_predict(x[a], y[a], x[b], alpha) - y[b]) ** 2).sum() for alpha in ALPHAS]
        p = _ridge_fit_predict(x[fit], y[fit], x[new], ALPHAS[int(np.argmin(errors))])
        sse += ((p - y[new]) ** 2).sum()
        sst += ((y[new] - y[fit].mean(0)) ** 2).sum()
    return float(1 - sse / max(sst, 1e-12))


def probe(z: np.ndarray, state: dict, seed: int = 0) -> dict:
    prev = state['previous']
    has_prev = prev >= 0
    pair = np.full((len(z), 2 * z.shape[1]), np.nan)
    pair[has_prev] = np.hstack([z[has_prev], z[has_prev] - z[prev[has_prev]]])
    out = {}
    for name in STATE_NAMES:
        y = state['targets'][name]
        out[name] = dict(single=cross_fitted_r2(z, y, state['group'], seed=seed),
                         pair=cross_fitted_r2(pair, y, state['group'], seed=seed))
    return out


# --------------------------------------------------------------------------- video clock

def pca(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean(0)
    x = x / max(float(x.std()), 1e-12)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:dim].T, s ** 2


def participation_ratio(spectrum: np.ndarray) -> float:
    return float(spectrum.sum() ** 2 / max((spectrum ** 2).sum(), 1e-24))


def rows_of(state: dict) -> list[dict]:
    # No `split` key on purpose: LatentLangevin guards frame SELECTION against non-training
    # rows. Here it only measures, on validation clips as well, and nothing is ranked.
    return [dict(frame_id=f'{c}/{n:08d}', video_id=str(c), group_id=str(g), frame_index=int(n))
            for c, n, g in zip(state['clip'], state['number'], state['group'])]


def langevin_summary(z: np.ndarray, state: dict, *, seed: int, max_gap: float) -> dict:
    rows = rows_of(state)
    short = LatentLangevin(rows, z, seed=seed, dim=z.shape[1], lags=3, folds=4, max_gap_frames=max_gap)
    long_ = LatentLangevin(rows, z, seed=seed, dim=z.shape[1], lags=8, folds=4, max_gap_frames=max_gap)
    residual = long_.y - long_.pred_pool
    target = long_.target
    follows = (target[1:] == target[:-1] + 1) & (long_.video[1:] == long_.video[:-1])
    a, b = residual[:-1][follows], residual[1:][follows]
    lag1 = float(np.mean([(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in range(a.shape[1])])) if len(a) > 10 else float('nan')
    drift, full = short.summary()['skill_vs_no_movement'], long_.summary()['skill_vs_no_movement']
    # Floor: the same model with the increments permuted among the windows of each clip. The
    # link from history to next step is gone, the distribution of steps is not, so the number
    # is comparable with the skill above. What survives is an artefact of the fit, not dynamics.
    rng = np.random.default_rng(seed)
    y_perm = long_.y.copy()
    for name in np.unique(long_.video):
        where = np.flatnonzero(long_.video == name)
        y_perm[where] = long_.y[rng.permutation(where)]
    error = 0.0
    for f in sorted(set(long_.fold.tolist())):
        fit, new = np.flatnonzero(long_.fold != f), np.flatnonzero(long_.fold == f)
        error += ((_ridge_fit_predict(long_.x[fit], y_perm[fit], long_.x[new], long_.alpha) - y_perm[new]) ** 2).mean(1).sum()
    floor = float(100 * (1 - error / (y_perm ** 2).mean(1).sum()))
    return dict(skill_drift_only=drift, skill_with_memory=full, memory_gain=full - drift, residual_lag1=lag1,
                skill_permuted_floor=floor, windows=int(len(target)))


def koopman_alignment(z: np.ndarray, state: dict, *, modes: int = 6, alpha: float = 1e-2, seed: int = 0) -> dict:
    prev = state['previous']
    index = np.flatnonzero(prev >= 0)
    current = index[prev[index] >= 0]
    current = current[prev[prev[current]] >= 0]              # t, t-1 and t-2 all exist
    lift = lambda i: np.hstack([z[i], z[prev[i]]])
    x0, x1 = lift(prev[current]), lift(current)
    gram = x0.T @ x0 / len(x0)
    gram.flat[::len(gram) + 1] += alpha
    k = np.linalg.solve(gram, x0.T @ x1 / len(x0))           # x1 ~ x0 @ k
    values, right = np.linalg.eig(k)                          # x @ right are eigenfunction coordinates
    order = np.argsort(-np.abs(values))
    values, right = values[order], right[:, order]
    full = np.full((len(z), 2 * z.shape[1]), np.nan)
    full[current] = x1
    coords = full @ right[:, :modes]
    slow = np.hstack([coords.real, coords.imag])
    control = np.full_like(slow, np.nan)
    control[current] = np.hstack([x1[:, :modes], x1[:, z.shape[1]:z.shape[1] + modes]])   # leading PCA directions
    carried = {name: dict(slow_modes=cross_fitted_r2(slow, state['targets'][name], state['group'], seed=seed),
                          leading_pca=cross_fitted_r2(control, state['targets'][name], state['group'], seed=seed))
               for name in STATE_NAMES}
    modulus = np.abs(values)
    return dict(slow_mode_count=int((modulus > .9).sum()), top_moduli=[float(m) for m in modulus[:modes]],
                top_angles_deg=[float(np.degrees(np.angle(v))) for v in values[:modes]], carried=carried)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0, y0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y1 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-9)


# --------------------------------------------------------------------------- training clock

def procrustes_displacement(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-particle squared move from latent a to latent b after the best rotation, and linear CKA."""
    a, b = a - a.mean(0), b - b.mean(0)
    a, b = a / max(float(np.linalg.norm(a)), 1e-12) * len(a) ** .5, b / max(float(np.linalg.norm(b)), 1e-12) * len(b) ** .5
    u, _, vt = np.linalg.svd(a.T @ b, full_matrices=False)
    moved = ((a @ (u @ vt) - b) ** 2).sum(1) / max(float((b ** 2).sum(1).mean()), 1e-12)
    cross = np.linalg.norm(a.T @ b) ** 2
    cka = cross / max(np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b), 1e-24)
    return moved, float(cka)


# --------------------------------------------------------------------------- driver

def analyse(latent: dict, state: dict, *, dim: int, seed: int, max_gap: float, which: str = 'query') -> dict:
    z, spectrum = pca(np.asarray(latent[which], dtype=np.float64), dim)
    labelled = np.isfinite(state['box']).all(1)
    pred_state = np.hstack([latent['pred_box'][:, :2] + latent['pred_box'][:, 2:] / 2,
                            np.log(np.maximum(latent['pred_box'][:, 2:], 1.0))])
    return dict(latent=which, dim=int(z.shape[1]), participation_ratio=participation_ratio(spectrum),
                detection_iou=float(np.nanmean(iou(latent['pred_box'][labelled], state['box'][labelled]))),
                probe=probe(z, state, seed), probe_box_control=probe(pred_state, state, seed),
                probe_gt_state_ceiling=probe(np.hstack([state['targets']['position'], state['targets']['size']]), state, seed),
                langevin=langevin_summary(z, state, seed=seed, max_gap=max_gap),
                koopman=koopman_alignment(z, state, seed=seed))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--latents', nargs='+', required=True, help='tag=path, ordered along the training clock')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dim', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--which', nargs='+', default=['query', 'box_pool', 'global_pool'])
    args = ap.parse_args(argv)
    tracks = json.loads((args.package / 'tracks.json').read_text(encoding='utf-8'))
    stride = float(next(iter(tracks.values()))['stride'])
    loaded, state, reference = [], None, []
    report: dict = dict(checkpoints={}, training_clock=[])
    for item in args.latents:
        tag, path = item.split('=', 1)
        with np.load(path, allow_pickle=False) as data:
            latent = {k: data[k] for k in data.files}
        ids = [str(f) for f in latent['frame_ids'].tolist()]
        if state is None:
            state, reference = physical_state(tracks, ids), ids
        elif ids != reference:
            raise ValueError(f'{tag}: checkpoints must be read on identical frames in identical order')
        report['checkpoints'][tag] = {w: analyse(latent, state, dim=args.dim, seed=args.seed,
                                                 max_gap=1.5 * stride, which=w) for w in args.which}
        loaded.append((tag, latent))
        r = report['checkpoints'][tag][args.which[0]]
        print(f'{tag:14s} IoU {r["detection_iou"]:.3f}  polozenie {r["probe"]["position"]["single"]:.3f}  '
              f'predkosc z jednej klatki {r["probe"]["velocity"]["single"]:.3f}  z pary {r["probe"]["velocity"]["pair"]:.3f}  '
              f'z samych ramek {r["probe_box_control"]["velocity"]["pair"]:.3f}  Langevin {r["langevin"]["skill_with_memory"]:.1f}  '
              f'pamiec {r["langevin"]["memory_gain"]:+.1f}', flush=True)
    moves = {}
    for (tag_a, a), (tag_b, b) in zip(loaded[:-1], loaded[1:]):
        for w in args.which:
            moved, cka = procrustes_displacement(np.asarray(a[w], np.float64), np.asarray(b[w], np.float64))
            report['training_clock'].append(dict(start=tag_a, end=tag_b, latent=w, mean_displacement=float(moved.mean()),
                                                 linear_cka=cka))
            moves[f'{tag_a}__{tag_b}__{w}'] = moved.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if moves:
        np.savez_compressed(args.out.with_suffix('.displacement.npz'), frame_ids=np.array(reference), **moves)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
