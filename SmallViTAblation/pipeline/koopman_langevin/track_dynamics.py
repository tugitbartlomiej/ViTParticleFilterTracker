#!/usr/bin/env python3
"""Koopman invariants and an extended Langevin equation on instrument tracks.

A track is the instrument state per frame, s_t = [cx, cy, log w, log h] (box centre and
size). It can come from ground truth (to measure how much physics there is in the motion)
or from a detector run on unlabelled video (to score frames for annotation).

Koopman part
  EDMD on the lifted state gives eigenvalues. Eigenfunctions with eigenvalue near 1 are
  conserved along the motion. One conserved quantity is known from surgery: a rigid
  instrument passes through the incision, so its axis keeps pointing at one pivot point.
  `pivot_invariant` measures how well that holds.

Extended Langevin part (same structure as the latent equation of the grant project)
  s_{t+1} = s_t + D(s_t, v_t, a_t) + JP(history of s, v) + N(past residuals) + e_{t+1}
  D   drift from the present state, velocity and acceleration
  JP  memory kernel over earlier steps
  N   predictable part of earlier residuals (coloured noise)
  e   what is left: surprise plus whatever the model class cannot express
All predictors are ridge regressions cross-fitted over operation groups, so every residual
belongs to a clip the model has not seen.

Frame scores
  residual_energy   size of e (large for both surprises and model gaps)
  reducible_gain    error of a weak model minus error of the full model on the same frame;
                    by the error decomposition this difference does not contain the surprise
  pivot_violation   distance of the instrument axis from the clip's pivot

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.track_dynamics \
        --dataset-root Dataset/Cataract-LMM --out SmallViTAblation/results/koopman_langevin_smoke/tracks
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- data

def read_tracks(root: Path, stride: int = 1, limit: int | None = None) -> list[dict]:
    """Ground-truth tracks from the tracking CSV files. Images are not opened."""
    audit = json.loads((root / '_audyt_pacjentow.json').read_text(encoding='utf-8'))
    group_of = {v: g for g, videos in audit['grupy_operacji'].items() for v in videos}
    tracks = []
    for archive in sorted((root / '3_Object_Tracking/annotations').glob('*.zip'))[:limit]:
        with zipfile.ZipFile(archive) as z:
            name = [n for n in z.namelist() if PurePosixPath(n).name == 'annotation.csv'][0]
            rows = list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
        rows = [r for r in rows if (int(r['frame_number']) - 1) % stride == 0]
        box = np.full((len(rows), 4), np.nan)
        tip = np.full((len(rows), 2), np.nan)
        for i, r in enumerate(rows):
            if (r.get('cystotome_bbox') or '').strip():
                box[i] = json.loads(r['cystotome_bbox'])
            if (r.get('cystotome_tip') or '').strip():
                tip[i] = json.loads(r['cystotome_tip'])
        tracks.append(dict(video_id=archive.stem, group_id=str(group_of.get(archive.stem, archive.stem)),
                           frame=np.array([int(r['frame_number']) for r in rows]), box=box, tip=tip))
    return tracks


def state_of(box: np.ndarray) -> np.ndarray:
    """[cx, cy, log w, log h]; NaN where the box is missing."""
    s = np.full((len(box), 4), np.nan)
    ok = np.isfinite(box).all(1) & (box[:, 2] > 0) & (box[:, 3] > 0)
    s[ok, 0] = box[ok, 0] + box[ok, 2] / 2
    s[ok, 1] = box[ok, 1] + box[ok, 3] / 2
    s[ok, 2:] = np.log(box[ok, 2:])
    return s


# --------------------------------------------------------------------------- regression

def ridge(x_fit, y_fit, x_new, alpha):
    mx, sx = x_fit.mean(0), x_fit.std(0)
    sx[sx < 1e-8] = 1.0
    my = y_fit.mean(0)
    a = (x_fit - mx) / sx
    g = a.T @ a / len(a)
    g.flat[::len(g) + 1] += alpha
    w = np.linalg.solve(g, a.T @ (y_fit - my) / len(a))
    return (x_new - mx) / sx @ w + my


def windows(tracks: list[dict], lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices (clip, t) of targets t+1 whose whole history t-lags..t+1 has a box."""
    clip, time, group = [], [], []
    for c, tr in enumerate(tracks):
        ok = np.isfinite(tr['state']).all(1)
        run = np.zeros(len(ok), dtype=int)
        for t in range(len(ok)):
            run[t] = (run[t - 1] + 1 if t and ok[t] else int(ok[t]))
        for t in range(lags + 1, len(ok) - 1):
            if run[t + 1] >= lags + 3:                       # t+1 and lags+2 frames before it
                clip.append(c)
                time.append(t)
                group.append(tr['group_id'])
    return np.array(clip), np.array(time), np.array(group)


def features(tracks, clip, time, lags, scale):
    """Nested designs. All use information up to frame t only."""
    s = np.stack([tracks[c]['state'][t] for c, t in zip(clip, time)])
    hist = np.stack([np.stack([tracks[c]['state'][t - l] - tracks[c]['state'][t - l - 1] for l in range(lags + 1)])
                     for c, t in zip(clip, time)]) / scale                     # (n, lags+1, 4) steps v_t, v_{t-1}, ...
    v, a = hist[:, 0], hist[:, 0] - hist[:, 1]
    centred = (s - s.mean(0)) / s.std(0)
    nonlinear = np.concatenate([v * np.abs(v), a * np.abs(a), centred * np.abs(v).sum(1, keepdims=True)], 1)
    drift = np.concatenate([centred, v, a, nonlinear], 1)
    kernel = np.concatenate([drift, hist[:, 2:].reshape(len(s), -1)], 1)
    target = np.stack([tracks[c]['state'][t + 1] - tracks[c]['state'][t] for c, t in zip(clip, time)]) / scale
    return dict(persistence=None, velocity=v, drift=drift, kernel=kernel), target


def langevin(tracks: list[dict], *, lags: int = 8, folds: int = 5, alpha: float = 1e-2, seed: int = 42,
             residual_taps: tuple[int, ...] = (1, 2, 3)) -> tuple[dict, dict]:
    """Cross-fitted nested predictors of the next step. Returns per-frame errors and a summary."""
    clip, time, group = windows(tracks, lags)
    if len(clip) < 100:
        raise ValueError('too few valid windows for the Langevin fit')
    steps = np.concatenate([np.diff(tr['state'], axis=0)[np.isfinite(np.diff(tr['state'], axis=0)).all(1)] for tr in tracks])
    scale = steps.std(0)
    designs, y = features(tracks, clip, time, lags, scale)
    names = sorted(set(group))
    rng = np.random.default_rng(seed)
    fold_of = {g: i % folds for i, g in enumerate(rng.permutation(names))}
    fold = np.array([fold_of[g] for g in group])
    pred = {k: np.zeros_like(y) for k in ('velocity', 'drift', 'kernel')}
    pred['persistence'] = np.zeros_like(y)                                    # "no movement"
    for f in range(folds):
        fit, new = fold != f, fold == f
        if not new.any():
            continue
        pred['velocity'][new] = designs['velocity'][new]                      # constant velocity, no fitting
        for k in ('drift', 'kernel'):
            pred[k][new] = ridge(designs[k][fit], y[fit], designs[k][new], alpha)
    # N: predictable part of earlier residuals of the kernel model, again cross-fitted
    res = y - pred['kernel']
    index = {(c, t): i for i, (c, t) in enumerate(zip(clip, time))}
    taps = np.zeros((len(y), len(residual_taps) * y.shape[1]))
    has_taps = np.ones(len(y), bool)
    for i, (c, t) in enumerate(zip(clip, time)):
        for j, lag in enumerate(residual_taps):
            k = index.get((c, t - lag))
            if k is None:
                has_taps[i] = False
            else:
                taps[i, j * y.shape[1]:(j + 1) * y.shape[1]] = res[k]
    pred['full'] = pred['kernel'].copy()
    for f in range(folds):
        fit, new = (fold != f) & has_taps, (fold == f) & has_taps
        if fit.sum() > 50 and new.any():
            pred['full'][new] = pred['kernel'][new] + ridge(taps[fit], res[fit], taps[new], alpha)
    err = {k: ((y - p) ** 2).sum(1) for k, p in pred.items()}
    ss = (y ** 2).sum()
    skill = {k: float(100 * (1 - e.sum() / ss)) for k, e in err.items()}
    e_full = y - pred['full']
    lag1 = np.array([index.get((c, t - 1), -1) for c, t in zip(clip, time)])
    ok = lag1 >= 0
    autocorr = float((e_full[ok] * e_full[lag1[ok]]).sum() / (e_full[ok] ** 2).sum())
    summary = dict(windows=int(len(y)), clips=len(tracks), groups=len(names), lags=lags, folds=folds,
                   step_scale=scale.tolist(), skill_vs_no_movement=skill,
                   gain_drift_over_velocity=skill['drift'] - skill['velocity'],
                   gain_memory_over_drift=skill['kernel'] - skill['drift'],
                   gain_noise_term_over_memory=skill['full'] - skill['kernel'],
                   residual_lag1_autocorrelation=autocorr,
                   note='cross-fitted over operation groups; skill = 100 (1 - SSE / SS of the step)')
    frames = dict(clip=clip, time=time, residual_energy=err['full'],
                  reducible_gain=err['velocity'] - err['full'], movement=(y ** 2).sum(1))
    return frames, summary


# --------------------------------------------------------------------------- Koopman

def koopman_spectrum(tracks: list[dict], *, lags: int = 8, alpha: float = 1e-3) -> dict:
    """EDMD on the lifted track state: eigenvalues tell which quantities persist and which decay."""
    clip, time, _ = windows(tracks, lags)
    s0 = np.stack([tracks[c]['state'][t] for c, t in zip(clip, time)])
    s1 = np.stack([tracks[c]['state'][t + 1] for c, t in zip(clip, time)])
    v0 = s0 - np.stack([tracks[c]['state'][t - 1] for c, t in zip(clip, time)])
    v1 = s1 - s0
    mu, sd = s0.mean(0), s0.std(0)
    vs = v0.std(0)

    def lift(s, v):
        a, b = (s - mu) / sd, v / vs
        return np.concatenate([a, b, a ** 2, b ** 2, a * b, np.ones((len(a), 1))], 1)

    g0, g1 = lift(s0, v0), lift(s1, v1)
    gram = g0.T @ g0 / len(g0)
    gram.flat[::len(gram) + 1] += alpha
    k = np.linalg.solve(gram, g0.T @ g1 / len(g0))                            # g1 ~ g0 @ k
    lam, vec = np.linalg.eig(k)
    psi0, psi1 = g0 @ vec, g1 @ vec
    quality = 1 - np.abs(psi1 - lam[None] * psi0).var(0) / np.maximum(np.abs(psi1).var(0), 1e-12)
    names = ([f's{i}' for i in range(4)] + [f'v{i}' for i in range(4)] + [f's{i}^2' for i in range(4)]
             + [f'v{i}^2' for i in range(4)] + [f's{i}v{i}' for i in range(4)] + ['1'])
    order = np.argsort(-np.abs(lam))
    modes = []
    for j in order:
        load = np.abs(vec[:, j]) ** 2
        load = load / load.sum()
        position = float(load[[0, 1, 8, 9]].sum())
        size = float(load[[2, 3, 10, 11]].sum())
        speed = float(load[4:8].sum() + load[12:20].sum())
        modes.append(dict(modulus=float(np.abs(lam[j])), angle_rad=float(np.angle(lam[j])),
                          half_life_frames=float(np.log(.5) / np.log(min(np.abs(lam[j]), .999999))),
                          eigenfunction_quality=float(quality[j]), load_position=position, load_size=size,
                          load_velocity=speed, dominant=names[int(np.argmax(load))]))
    slow = [m for m in modes if m['modulus'] > .98 and abs(m['angle_rad']) < .05 and m['dominant'] != '1']
    fast = [m for m in modes if m['modulus'] <= .98]
    return dict(modes=modes, slow_modes=len(slow), fast_modes=len(fast),
                slow_load=dict(position=float(np.mean([m['load_position'] for m in slow])) if slow else None,
                               size=float(np.mean([m['load_size'] for m in slow])) if slow else None,
                               velocity=float(np.mean([m['load_velocity'] for m in slow])) if slow else None),
                fast_load=dict(position=float(np.mean([m['load_position'] for m in fast])) if fast else None,
                               size=float(np.mean([m['load_size'] for m in fast])) if fast else None,
                               velocity=float(np.mean([m['load_velocity'] for m in fast])) if fast else None))


def pivot_invariant(tracks: list[dict]) -> dict:
    """Does the instrument axis keep pointing at one point (the incision) within a clip?

    The axis is the line from the tip to the box corner farthest from the tip. The pivot is the
    least-squares intersection of those lines. Reported: distance of the axes from the pivot
    compared with how far the tip travels. A rigid rod through a fixed incision gives a small ratio.
    """
    per_clip, violation = [], {}
    for tr in tracks:
        ok = np.isfinite(tr['tip']).all(1) & np.isfinite(tr['box']).all(1)
        if ok.sum() < 30:
            continue
        tip, box = tr['tip'][ok], tr['box'][ok]
        corners = np.stack([box[:, :2], box[:, :2] + box[:, 2:] * [1, 0], box[:, :2] + box[:, 2:] * [0, 1],
                            box[:, :2] + box[:, 2:]], 1)                       # (n, 4, 2)
        far = corners[np.arange(len(tip)), np.argmax(((corners - tip[:, None]) ** 2).sum(2), 1)]
        d = far - tip
        length = np.linalg.norm(d, axis=1)
        keep = length > 20
        if keep.sum() < 30:
            continue
        tip, d = tip[keep], d[keep] / length[keep, None]
        normal = np.stack([-d[:, 1], d[:, 0]], 1)
        a = normal
        b = (normal * tip).sum(1)
        pivot = np.linalg.lstsq(a, b, rcond=None)[0]
        dist = np.abs(a @ pivot - b)
        travel = np.linalg.norm(tip - tip.mean(0), axis=1)
        per_clip.append(dict(video_id=tr['video_id'], frames=int(keep.sum()), axis_distance_px_median=float(np.median(dist)),
                             tip_travel_px_median=float(np.median(travel)),
                             ratio=float(np.median(dist) / max(np.median(travel), 1e-9)),
                             pivot=pivot.tolist()))
        full = np.full(len(tr['frame']), np.nan)
        full[np.flatnonzero(ok)[keep]] = dist
        violation[tr['video_id']] = full
    ratios = np.array([c['ratio'] for c in per_clip])
    return dict(clips=len(per_clip), ratio_median=float(np.median(ratios)) if len(ratios) else None,
                ratio_p90=float(np.quantile(ratios, .9)) if len(ratios) else None,
                axis_distance_px_median=float(np.median([c['axis_distance_px_median'] for c in per_clip])) if per_clip else None,
                tip_travel_px_median=float(np.median([c['tip_travel_px_median'] for c in per_clip])) if per_clip else None,
                per_clip=per_clip), violation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', type=Path, default=REPOSITORY / 'Dataset/Cataract-LMM')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--lags', type=int, default=8)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args(argv)
    tracks = read_tracks(args.dataset_root, args.stride, args.limit)
    for tr in tracks:
        tr['state'] = state_of(tr['box'])
    frames, summary = langevin(tracks, lags=args.lags)
    spectrum = koopman_spectrum(tracks, lags=args.lags)
    pivot, _ = pivot_invariant(tracks)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(source='ground_truth_tracks', stride=args.stride, langevin=summary,
                  koopman={k: v for k, v in spectrum.items() if k != 'modes'}, koopman_modes=spectrum['modes'][:12],
                  pivot={k: v for k, v in pivot.items() if k != 'per_clip'})
    (args.out / f'track_dynamics_stride{args.stride}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('langevin', 'koopman', 'pivot')}, indent=1)[:3500])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
