"""FIT-only feature probes and descriptive, held-out D failure queries.

Callers enforce operation-disjoint FIT/D partitions before calling these array
functions. No D observations are accepted by ``fit_probes``. ``discover_needs``
uses FIT only to set feature thresholds and D only to describe task difficulty.
Queries describe associations; neither the bins nor probe errors establish a
causal explanation of detector failure.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return result


def fit_probes(descriptors: np.ndarray, feature_values: np.ndarray) -> dict:
    """Fit standardized ridge probes from FIT rows, one target per column.

    Missing descriptors exclude that row; missing targets exclude it only from
    that target. Fewer than two valid targets means the probe is unavailable.
    All means/scales and regression coefficients are fitted here, never during
    prediction. State uses NumPy arrays and is suitable for NPZ/checkpoint storage.
    The tiny fixed ridge is numerical regularization, not tuned on D.
    """
    x = _matrix(descriptors, "descriptors")
    y = _matrix(feature_values, "feature_values")
    if len(x) != len(y) or x.shape[1] == 0:
        raise ValueError("FIT descriptors and features must have aligned rows and nonempty descriptors")
    valid_x = np.isfinite(x).all(axis=1)
    if not valid_x.any():
        raise ValueError("At least one finite FIT descriptor is required")
    x_mean = x[valid_x].mean(axis=0)
    x_scale = x[valid_x].std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    z = (x - x_mean) / x_scale
    coefficients = np.full((x.shape[1] + 1, y.shape[1]), np.nan)
    feature_mean = np.full(y.shape[1], np.nan)
    feature_scale = np.full(y.shape[1], np.nan)
    available = np.zeros(y.shape[1], dtype=bool)
    counts = np.zeros(y.shape[1], dtype=np.int64)
    ridge = 1e-5
    for column in range(y.shape[1]):
        valid = valid_x & np.isfinite(y[:, column])
        counts[column] = int(valid.sum())
        if counts[column] < 2:
            continue
        target = y[valid, column]
        feature_mean[column] = float(target.mean())
        target_std = float(target.std())
        feature_scale[column] = target_std if target_std > 1e-12 else 1.0
        normalized_target = (target - feature_mean[column]) / feature_scale[column]
        # This additional centering gives an unpenalized, per-target intercept
        # when different targets have different missing-value masks.
        center = z[valid].mean(axis=0)
        design = z[valid] - center
        if design.shape[1] <= design.shape[0]:
            gram = design.T @ design
            gram.flat[:: gram.shape[0] + 1] += ridge
            slope = np.linalg.solve(gram, design.T @ normalized_target)
        else:
            gram = design @ design.T
            gram.flat[:: gram.shape[0] + 1] += ridge
            slope = design.T @ np.linalg.solve(gram, normalized_target)
        coefficients[0, column] = -float(center @ slope)
        coefficients[1:, column] = slope
        available[column] = True
    return {
        "descriptor_mean": x_mean,
        "descriptor_scale": x_scale,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coefficients": coefficients,
        "available": available,
        "n_fit_samples": int(len(x)),
        "n_feature_samples": counts,
        "ridge": ridge,
    }


def predict_probes(state: dict, descriptors: np.ndarray) -> np.ndarray:
    """Predict ``[N,F]`` without refitting; unavailable values stay NaN."""
    x = _matrix(descriptors, "descriptors")
    mean = np.asarray(state["descriptor_mean"], dtype=np.float64)
    scale = np.asarray(state["descriptor_scale"], dtype=np.float64)
    available = np.asarray(state["available"], dtype=bool)
    coefficients = np.asarray(state["coefficients"], dtype=np.float64)
    if mean.shape != (x.shape[1],) or scale.shape != mean.shape:
        raise ValueError("Probe descriptor dimension differs from the fitted FIT state")
    if coefficients.shape != (x.shape[1] + 1, len(available)):
        raise ValueError("Invalid probe coefficient dimensions")
    result = np.full((len(x), len(available)), np.nan)
    valid_rows = np.flatnonzero(np.isfinite(x).all(axis=1))
    valid_columns = np.flatnonzero(available)
    if len(valid_rows) and len(valid_columns):
        z = (x[valid_rows] - mean) / scale
        design = np.column_stack([np.ones(len(z)), z])
        predictions = design @ coefficients[:, valid_columns]
        predictions = predictions * np.asarray(state["feature_scale"])[valid_columns] + np.asarray(state["feature_mean"])[valid_columns]
        result[np.ix_(valid_rows, valid_columns)] = predictions
    return result


def _group_mean(values: np.ndarray, groups: np.ndarray, mask: np.ndarray) -> float:
    """Each operation contributes equally, regardless of its frame count."""
    finite = mask & np.isfinite(values)
    if not finite.any():
        return 0.0
    group_values = [float(values[finite & (groups == group)].mean()) for group in dict.fromkeys(groups[finite].tolist())]
    return float(np.mean(group_values))


def _bounded_signal(values: np.ndarray, groups: np.ndarray, mask: np.ndarray) -> float:
    """Nonnegative, bounded descriptive contribution in [0, 1]."""
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    positive = np.where(finite, np.maximum(values, 0.0), np.nan)
    positive_support = positive[np.isfinite(positive) & (positive > 0)]
    if not len(positive_support):
        return 0.0
    scale = float(np.median(positive_support))
    signal = _group_mean(positive, groups, mask)
    # Division in this order avoids overflow for large but finite signals.
    return float(1.0 / (1.0 + scale / signal)) if signal > 0 else 0.0


def discover_needs(
    fit_features: np.ndarray,
    diagnostic_features: np.ndarray,
    diagnostic_losses: np.ndarray,
    diagnostic_groups: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    *,
    latent_errors: np.ndarray | None = None,
    dynamics_gain: np.ndarray | None = None,
    min_support: int = 8,
    min_groups: int = 3,
    max_needs: int = 8,
) -> list[dict]:
    """Return supported D difficulty bins with FIT-derived query thresholds.

    Inputs have shapes FIT ``[M,F]``, D ``[N,F]``, losses/groups ``[N]``.
    ``latent_errors`` are nonnegative, FIT-standardized probe squared residuals
    ``[N,F]`` (or pre-aggregated ``[N]``); missing entries are NaN.
    ``dynamics_gain`` is the per-frame ``[N]`` readout proxy
    ``max(observed-token readout BCE - predicted-token readout BCE, 0)``.
    It is neither improvement over a causal/persistence baseline nor measured
    detector learning improvement. Missing history is NaN; only positive
    available values can boost priority.

    A bin must have strictly greater group-balanced task loss than global D,
    at least min_support frames and min_groups operations. Priority equals
    relative excess loss times ``1 + .25*latent_signal + .25*positive_gain``.
    Both signals are scaled by their positive D median and saturated to [0,1].
    Auxiliary evidence can boost task priority by at most 50%, never create it.

    Three bins per nonconstant FIT feature use FIT 1/3 and 2/3 quantiles. For
    interactions, at most 12 supported single bins are paired across different
    features (at most 66 pairs). Identical D memberships are deduplicated with
    simpler queries preferred. This is descriptive selection, not inference
    with a multiple-testing-adjusted confidence level.
    """
    fit = _matrix(fit_features, "fit_features")
    diagnostic = _matrix(diagnostic_features, "diagnostic_features")
    losses = np.asarray(diagnostic_losses, dtype=np.float64)
    groups = np.asarray(diagnostic_groups, dtype=object)
    names = list(feature_names)
    if fit.shape[1] != diagnostic.shape[1] or len(names) != fit.shape[1] or len(set(names)) != len(names):
        raise ValueError("Feature columns must agree with unique feature_names")
    if losses.shape != (len(diagnostic),) or groups.shape != losses.shape:
        raise ValueError("D losses and groups must have one value per diagnostic row")
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("D task losses must be finite and nonnegative")
    if any(group is None or (isinstance(group, (float, np.floating)) and not np.isfinite(group)) for group in groups):
        raise ValueError("Every diagnostic frame requires an operation group")
    for value, name, minimum in ((min_support, "min_support", 1), (min_groups, "min_groups", 1), (max_needs, "max_needs", 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    latent = None if latent_errors is None else np.asarray(latent_errors, dtype=np.float64)
    if latent is not None and latent.shape not in ((len(diagnostic),), diagnostic.shape):
        raise ValueError("latent_errors must have shape [N] or [N,F]")
    if latent is not None and (latent[np.isfinite(latent)] < 0).any():
        raise ValueError("latent_errors must be nonnegative errors or missing")
    gains = None if dynamics_gain is None else np.asarray(dynamics_gain, dtype=np.float64)
    if gains is not None and gains.shape != losses.shape:
        raise ValueError("dynamics_gain must have shape [N]")
    if not len(diagnostic) or max_needs == 0:
        return []
    all_rows = np.ones(len(diagnostic), dtype=bool)
    baseline = _group_mean(losses, groups, all_rows)
    singles: list[dict] = []

    def summarize(conditions: list[dict], mask: np.ndarray) -> dict | None:
        support = int(mask.sum())
        group_count = len(set(groups[mask].tolist()))
        if support < min_support or group_count < min_groups:
            return None
        return {"conditions": conditions, "mask": mask, "support": support, "group_count": group_count, "mean_task_loss": _group_mean(losses, groups, mask)}

    for column, name in enumerate(names):
        observed = fit[np.isfinite(fit[:, column]), column]
        if len(observed) < 2 or float(np.ptp(observed)) <= 1e-12:
            continue
        thresholds = np.unique(np.quantile(observed, [1 / 3, 2 / 3]))
        boundaries = [None, *[float(value) for value in thresholds], None]
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            condition = {"feature": name, "feature_index": column, "lower": lower, "upper": upper, "lower_inclusive": False, "upper_inclusive": True}
            mask = np.isfinite(diagnostic[:, column])
            if lower is not None:
                mask &= diagnostic[:, column] > lower
            if upper is not None:
                mask &= diagnostic[:, column] <= upper
            candidate = summarize([condition], mask)
            if candidate is not None:
                singles.append(candidate)
    # Include pair-only difficulty even if one-dimensional marginal bins happen
    # to lie below the overall loss, but bound the search and require support.
    pool = sorted(singles, key=lambda item: -item["mean_task_loss"])[:12]
    candidates = list(singles)
    for first, second in combinations(pool, 2):
        if first["conditions"][0]["feature_index"] == second["conditions"][0]["feature_index"]:
            continue
        candidate = summarize(first["conditions"] + second["conditions"], first["mask"] & second["mask"])
        if candidate is not None:
            candidates.append(candidate)
    seen_memberships: set[bytes] = set()
    needs = []
    for candidate in candidates:
        mask = candidate.pop("mask")
        signature = np.packbits(mask).tobytes()
        if signature in seen_memberships:
            continue
        seen_memberships.add(signature)
        excess = candidate["mean_task_loss"] - baseline
        if excess <= max(1e-12, abs(baseline) * 1e-12):
            continue
        latent_signal = 0.0
        if latent is not None:
            if latent.ndim == 1:
                frame_errors = latent
            else:
                columns = [condition["feature_index"] for condition in candidate["conditions"]]
                errors = latent[:, columns]
                valid = np.isfinite(errors)
                count = valid.sum(axis=1)
                frame_errors = np.divide(np.where(valid, errors, 0.0).sum(axis=1), count, out=np.full(len(errors), np.nan), where=count > 0)
            latent_signal = _bounded_signal(frame_errors, groups, mask)
        gain_signal = 0.0 if gains is None else _bounded_signal(gains, groups, mask)
        candidate.update(
            baseline_task_loss=baseline,
            task_loss_excess=float(excess),
            latent_error_factor=float(0.25 * latent_signal),
            dynamics_gain_factor=float(0.25 * gain_signal),
            priority=float(excess / max(baseline, 1e-12) * (1 + 0.25 * latent_signal + 0.25 * gain_signal)),
            threshold_source="FIT feature quantiles",
            attribution="Descriptive D task-loss excess; optional FIT-probe error and positive causal-history KG gain are bounded secondary signals, not causal attribution.",
        )
        needs.append(candidate)
    needs.sort(key=lambda item: (-item["priority"], len(item["conditions"]), -item["support"]))
    needs = needs[:max_needs]
    for index, need in enumerate(needs, start=1):
        need["need_id"] = f"need_{index:03d}"
    return needs
