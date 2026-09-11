"""Cross-fitted prediction diagnostics, inspired by regression-based MZ analysis.

https://doi.org/10.1137/22M1506146
These are finite-model loss contrasts, NOT an exact memory kernel, conditional
mutual information, or evidence that more bounding-box labels repair a latent.
All windows stay within one clip; held-out folds contain whole operation groups.
"""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np


def project(features, dim: int, seed: int):
    if dim <= 0:
        raise ValueError('Projection dimension must be positive')
    if features.shape[1] <= dim:
        return features.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(features.shape[1], dim)).astype(np.float32)/np.sqrt(dim)
    return features @ projection


def ridge_predict(x, y, query, alpha):
    """Mean-square objective plus ridge penalty; unpenalized intercept."""
    mx, my = x.mean(0), y.mean(0)
    scale = x.std(0)
    scale[scale < 1e-6] = 1
    design = (x-mx)/scale
    gram = design.T @ design/len(x)
    gram.flat[::len(gram)+1] += alpha
    rhs = design.T @ (y-my)/len(x)
    weights = np.linalg.solve(gram, rhs)
    return ((query-mx)/scale) @ weights + my


def diagnose(rows, features, *, seed=42, dim=24, lags=3, folds=3,
             max_gap_frames=8, alpha=.01, max_fit_windows=20000):
    if lags < 1 or folds < 2 or alpha <= 0 or max_fit_windows < 2 or max_gap_frames <= 0:
        raise ValueError('Invalid dynamics configuration')
    started = time.perf_counter()
    z = project(features, dim, seed).astype(np.float64)
    by_video = defaultdict(list)
    for i, row in enumerate(rows):
        by_video[row['video_id']].append(i)
    windows = []
    for indices in by_video.values():
        indices.sort(key=lambda i: rows[i]['frame_index'])
        for position in range(lags, len(indices)-1):
            seq = indices[position-lags:position+2]
            times = np.array([rows[i]['frame_index'] for i in seq])
            gaps = np.diff(times)
            # Near-uniform h only. Sparse static frames must not become a fake movie.
            if np.any(gaps <= 0) or gaps.max() > max_gap_frames or gaps.max() > 1.1*gaps.min():
                continue
            windows.append([indices[position-j] for j in range(lags+1)]+[indices[position+1]])
    if not windows:
        raise ValueError('No valid temporal windows; use balanced for sparse frames, '
                         'or supply a correctly sampled dense tracking pool')
    windows = np.asarray(windows, dtype=np.int64)
    anchor = windows[:, 0]
    groups = np.array([rows[i]['group_id'] for i in anchor])
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError('Dynamics requires at least two training operation groups')
    folds = min(folds, len(unique))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    fold_by_group = {g: i % folds for i, g in enumerate(shuffled)}
    fold_ids = np.array([fold_by_group[g] for g in groups])
    d = z.shape[1]
    # Fixed random nonlinear maps: no fitting to held-out targets or labels.
    w = rng.normal(size=(d, (lags+1)*d))/np.sqrt(d)
    b = rng.uniform(0, 2*np.pi, w.shape[1])
    names = ('persistence', 'linear', 'nonlinear', 'wide_present', 'history')
    errors = {name: np.full(len(rows), np.nan) for name in names}
    summaries = []
    for fold in range(folds):
        held = np.flatnonzero(fold_ids == fold)
        training = np.flatnonzero(fold_ids != fold)
        # Balance the regression fit over operation groups before capping memory.
        per_group = max(1, max_fit_windows // len(set(groups[training])))
        samples = []
        for group in sorted(set(groups[training])):
            members = training[groups[training] == group]
            samples.extend(rng.choice(members, min(len(members), per_group), replace=False))
        fit = np.sort(np.asarray(samples, dtype=np.int64))
        if len(fit) > max_fit_windows:
            fit = np.sort(rng.choice(fit, max_fit_windows, replace=False))
        if len(fit) < 2:
            raise ValueError('Insufficient independent training windows')
        all_indices = np.concatenate((fit, held))
        current = z[windows[all_indices, 0]]
        mu, sd = current[:len(fit)].mean(0), current[:len(fit)].std(0)
        sd[sd < 1e-6] = 1
        current = (current-mu)/sd
        history = ((z[windows[all_indices, 1:-1]]-mu)/sd).reshape(len(all_indices), -1)
        nonlinear = np.cos(current @ w + b)
        maps = dict(linear=current,
            nonlinear=np.column_stack((current, nonlinear[:, :d])),
            wide_present=np.column_stack((current, nonlinear)),
            history=np.column_stack((current, nonlinear[:, :d], history)))
        target = z[windows[all_indices, -1]]
        # Shared target coordinates and train-only scale for every competing probe.
        target_scale = target[:len(fit)].std(0)
        target_scale[target_scale < 1e-6] = 1
        y = target/target_scale
        index = anchor[held]
        errors['persistence'][index] = np.mean((z[index]/target_scale-y[len(fit):])**2, 1)
        for name, design in maps.items():
            prediction = ridge_predict(design[:len(fit)], y[:len(fit)], design[len(fit):], alpha)
            errors[name][index] = np.mean((prediction-y[len(fit):])**2, 1)
        summaries.append(dict(fold=fold, fit_windows=len(fit), held_out_windows=len(held),
            held_out_groups=sorted(set(groups[held])),
            mse={name: float(errors[name][index].mean()) for name in names}))
        print(f'[dynamics] fold {fold+1}/{folds}: fit={len(fit)}, held={len(held)}', flush=True)
    errors['history_gain'] = errors['wide_present']-errors['history']
    errors['nonlinearity_gain'] = errors['linear']-errors['nonlinear']
    errors['residual'] = errors['nonlinear'].copy()
    valid = np.isfinite(errors['history_gain'])
    return errors, dict(experimental=True, interpretation='out_of_group_prediction_loss_contrasts',
        exact_mori_zwanzig_kernel=False, mutual_information=False, annotation_labels_used=False,
        valid_windows=int(valid.sum()), total_candidates=len(rows), folds=summaries,
        projection_dimension=d, lags=lags, max_gap_frames=max_gap_frames, ridge_alpha=alpha,
        horizon='next_candidate; frame gaps bounded and nearly uniform within each window',
        model_input_dimensions={name: int(x.shape[1]) for name, x in maps.items()},
        wall_seconds=time.perf_counter()-started,
        mean_history_gain=float(errors['history_gain'][valid].mean()))
