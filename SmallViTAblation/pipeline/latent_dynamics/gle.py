"""Finite-dimensional GLE-inspired diagnostics for video representations.

The implementation compares predictors of the same next-frame latent target.
It estimates neither an exact Mori--Zwanzig memory kernel nor physical Langevin
noise.  Every reported residual comes from a model fitted on other operation
groups and is assigned to the frame that was predicted.
"""
from __future__ import annotations

from collections import defaultdict
import time
from typing import Sequence

import numpy as np


def _project(features: np.ndarray, dim: int, seed: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1:
        raise ValueError("features must have shape [n>=2, d>=1]")
    if not np.isfinite(x).all():
        raise ValueError("features contain NaN or infinity")
    if dim < 1:
        raise ValueError("projection dimension must be positive")
    if x.shape[1] <= dim:
        return x.copy()
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(x.shape[1], dim)) / np.sqrt(dim)
    return x @ matrix


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, alpha: float) -> np.ndarray:
    mx = x_train.mean(axis=0)
    sx = x_train.std(axis=0)
    sx[sx < 1e-8] = 1.0
    my = y_train.mean(axis=0)
    xs = (x_train - mx) / sx
    gram = xs.T @ xs / len(xs)
    gram.flat[:: len(gram) + 1] += alpha
    weights = np.linalg.solve(gram, xs.T @ (y_train - my) / len(xs))
    return (x_test - mx) / sx @ weights + my


def _windows(rows: Sequence[dict], lags: int,
             max_gap_frames: float) -> np.ndarray:
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_video[str(row["video_id"])].append(i)
    result = []
    for indices in by_video.values():
        indices.sort(key=lambda i: float(rows[i]["frame_index"]))
        for target_pos in range(lags, len(indices)):
            seq = indices[target_pos - lags:target_pos + 1]
            times = np.asarray([float(rows[i]["frame_index"]) for i in seq])
            gaps = np.diff(times)
            if len(gaps) == 0 or np.any(gaps <= 0) or gaps.max() > max_gap_frames:
                continue
            # Avoid treating irregular sparse annotation snapshots as a movie.
            if gaps.max() > 1.1 * gaps.min():
                continue
            result.append(seq)
    if not result:
        raise ValueError("no valid temporal windows; use dense, near-uniform video candidates")
    return np.asarray(result, dtype=np.int64)


def _folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    unique = np.asarray(sorted(set(map(str, groups))))
    if len(unique) < 2:
        raise ValueError("diagnostics require at least two operation groups")
    n_folds = min(max(2, int(n_folds)), len(unique))
    order = np.random.default_rng(seed).permutation(unique)
    assignment = {group: i % n_folds for i, group in enumerate(order)}
    return np.asarray([assignment[str(group)] for group in groups], dtype=np.int64)


def _balanced_fit_indices(groups: np.ndarray, candidates: np.ndarray,
                          maximum: int, rng: np.random.Generator) -> np.ndarray:
    distinct = sorted(set(map(str, groups[candidates])))
    per_group = max(1, maximum // len(distinct))
    chosen = []
    for group in distinct:
        members = candidates[np.asarray([str(groups[i]) == group for i in candidates])]
        chosen.extend(rng.choice(members, min(len(members), per_group), replace=False))
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    if len(chosen) > maximum:
        chosen = np.sort(rng.choice(chosen, maximum, replace=False))
    return chosen


def _address_proxy(rows: Sequence[dict], z: np.ndarray) -> tuple[np.ndarray, dict]:
    """Causal running video mean: an identity/context proxy, never proof."""
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_video[str(row["video_id"])].append(i)
    address = np.empty_like(z)
    centres = []
    within = []
    for indices in by_video.values():
        indices.sort(key=lambda i: float(rows[i]["frame_index"]))
        trajectory = z[indices]
        causal = np.cumsum(trajectory, axis=0) / np.arange(1, len(indices) + 1)[:, None]
        centre = causal[-1]
        centres.append(centre)
        within.extend(np.mean((trajectory - causal) ** 2, axis=1))
        address[indices] = causal
    centres = np.asarray(centres)
    between = float(np.mean(np.var(centres, axis=0))) if len(centres) > 1 else 0.0
    within_mean = float(np.mean(within)) if within else 0.0
    return address, {
        "definition": "causal_running_video_mean_identity_or_context_proxy",
        "future_frames_used": False,
        "identity_proven": False,
        "within_video_mse": within_mean,
        "between_video_variance": between,
        "between_to_within_ratio": between / max(within_mean, 1e-12),
    }


def diagnose_gle(
    rows: Sequence[dict],
    features: np.ndarray,
    *,
    seed: int = 42,
    dim: int = 24,
    lags: int = 3,
    folds: int = 3,
    max_gap_frames: float = 8,
    alpha: float = .01,
    max_fit_windows: int = 20000,
    covariance_shrinkage: float = .10,
    residual_modes: int = 8,
) -> tuple[list[dict], dict, dict[str, np.ndarray]]:
    """Return per-frame scores, an audit summary, and residual-mode arrays."""
    started = time.perf_counter()
    if len(rows) != len(features):
        raise ValueError("rows and features have different lengths")
    if lags < 1 or alpha <= 0 or max_fit_windows < 2:
        raise ValueError("invalid temporal model configuration")
    if not 0 <= covariance_shrinkage <= 1:
        raise ValueError("covariance_shrinkage must lie in [0, 1]")
    if any(row.get("split", "train") != "train" for row in rows):
        raise ValueError("latent diagnostics may only consume the training pool")

    z = _project(features, dim, seed)
    address, address_summary = _address_proxy(rows, z)
    windows = _windows(rows, lags, max_gap_frames)
    targets = windows[:, -1]
    groups = np.asarray([str(rows[i]["group_id"]) for i in targets])
    fold_id = _folds(groups, folds, seed)
    rng = np.random.default_rng(seed)
    d = z.shape[1]
    nonlinear_w = rng.normal(size=(d, 2 * d)) / np.sqrt(d)
    nonlinear_b = rng.uniform(0, 2 * np.pi, size=2 * d)

    fields = ("persistence_error", "markov_error", "nonlinear_error", "history_error")
    values = {name: np.full(len(rows), np.nan, dtype=np.float64) for name in fields}
    residual = np.full((len(rows), d), np.nan, dtype=np.float64)
    fold_summaries = []
    for fold in sorted(set(fold_id.tolist())):
        held = np.flatnonzero(fold_id == fold)
        candidates = np.flatnonzero(fold_id != fold)
        fit = _balanced_fit_indices(groups, candidates, max_fit_windows, rng)
        if len(fit) < 2 or not len(held):
            raise ValueError("insufficient cross-fitting windows")
        all_idx = np.concatenate([fit, held])
        seq = windows[all_idx]
        # The current causal address is the reference for the entire window and
        # its target. Subtracting a target-frame mean would leak the answer.
        addr = address[seq[:, -2]]
        current = z[seq[:, -2]] - addr
        past = (z[seq[:, :-2]] - addr[:, None, :]).reshape(len(seq), -1)
        nonlinear = np.cos(current @ nonlinear_w + nonlinear_b)
        designs = {
            "markov_error": current,
            "nonlinear_error": np.column_stack([current, nonlinear]),
            "history_error": np.column_stack([current, nonlinear, past]),
        }
        y_raw = z[seq[:, -1]] - addr
        y_mean = y_raw[:len(fit)].mean(axis=0)
        y_scale = y_raw[:len(fit)].std(axis=0)
        y_scale[y_scale < 1e-8] = 1.0
        y = (y_raw - y_mean) / y_scale
        n_fit = len(fit)
        target_rows = targets[held]
        persistence = (current[n_fit:] - y_mean) / y_scale
        values["persistence_error"][target_rows] = np.mean(
            (persistence - y[n_fit:]) ** 2, axis=1)
        predictions = {}
        for name, design in designs.items():
            prediction = _ridge_fit_predict(
                design[:n_fit], y[:n_fit], design[n_fit:], alpha)
            predictions[name] = prediction
            values[name][target_rows] = np.mean((prediction - y[n_fit:]) ** 2, axis=1)
        residual[target_rows] = y[n_fit:] - predictions["history_error"]
        fold_summaries.append({
            "fold": int(fold),
            "fit_windows": int(len(fit)),
            "held_out_windows": int(len(held)),
            "held_out_groups": sorted(set(groups[held].tolist())),
            "mean_error": {name: float(np.mean(values[name][target_rows]))
                           for name in fields},
        })

    valid = np.isfinite(residual).all(axis=1)
    residual_valid = residual[valid]
    if len(residual_valid) < 2:
        raise ValueError("too few cross-fitted residuals")
    centred = residual_valid - residual_valid.mean(axis=0)
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    diagonal = np.diag(np.diag(covariance))
    shrunk = (1 - covariance_shrinkage) * covariance + covariance_shrinkage * diagonal
    scale = max(float(np.trace(shrunk)) / len(shrunk), 1e-8)
    shrunk.flat[:: len(shrunk) + 1] += scale * 1e-6
    inverse = np.linalg.pinv(shrunk, hermitian=True)
    mahalanobis = np.full(len(rows), np.nan)
    mahalanobis[valid] = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    energy = np.full(len(rows), np.nan)
    energy[valid] = np.mean(residual_valid ** 2, axis=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    keep = order[:min(max(1, residual_modes), d)]
    modes = eigenvectors[:, keep]
    mode_scores = np.full((len(rows), len(keep)), np.nan)
    mode_scores[valid] = np.abs(centred @ modes)

    score_rows = []
    for i, row in enumerate(rows):
        record = {
            "frame_id": str(row["frame_id"]),
            "has_valid_window": bool(valid[i]),
            "score_assigned_to": "predicted_target_frame",
        }
        for name in fields:
            record[name] = float(values[name][i]) if np.isfinite(values[name][i]) else None
        record["history_gain"] = (
            float(values["nonlinear_error"][i] - values["history_error"][i])
            if np.isfinite(values["history_error"][i]) else None)
        record["nonlinearity_gain"] = (
            float(values["markov_error"][i] - values["nonlinear_error"][i])
            if np.isfinite(values["nonlinear_error"][i]) else None)
        record["residual_energy"] = float(energy[i]) if np.isfinite(energy[i]) else None
        record["residual_mahalanobis"] = (
            float(mahalanobis[i]) if np.isfinite(mahalanobis[i]) else None)
        record["dominant_residual_mode"] = (
            int(np.argmax(mode_scores[i])) if valid[i] else None)
        record["dominant_residual_mode_score"] = (
            float(np.max(mode_scores[i])) if valid[i] else None)
        score_rows.append(record)

    valid_gain = np.asarray([r["history_gain"] for r in score_rows if r["history_gain"] is not None])
    summary = {
        "schema_version": 1,
        "experimental": True,
        "interpretation": "cross_fitted_finite_model_prediction_contrasts",
        "exact_mori_zwanzig_kernel": False,
        "physical_langevin_noise": False,
        "koopman_identity_proven": False,
        "annotation_labels_used": False,
        "residual_predictors_cross_fitted_by_group": True,
        "residual_mode_basis_cross_fitted": False,
        "residual_mode_basis_note": (
            "descriptive basis fitted to all label-free cross-fitted training-pool residuals; "
            "confirm mode stability in an outer grouped split before mechanistic claims"),
        "valid_windows": int(valid.sum()),
        "total_frames": len(rows),
        "projection_dimension": d,
        "lags": lags,
        "max_gap_frames": max_gap_frames,
        "ridge_alpha": alpha,
        "covariance_shrinkage": covariance_shrinkage,
        "mean_history_gain": float(valid_gain.mean()),
        "positive_history_gain_fraction": float(np.mean(valid_gain > 0)),
        "address_proxy": address_summary,
        "folds": fold_summaries,
        "wall_seconds": time.perf_counter() - started,
    }
    arrays = {
        "residual_mode_vectors": modes.astype(np.float32),
        "residual_mode_eigenvalues": eigenvalues[keep].astype(np.float32),
        "projected_features": z.astype(np.float32),
    }
    return score_rows, summary, arrays
