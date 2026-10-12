"""Retrieve FIT images matching diagnosed traits, with explicit random slots."""

from __future__ import annotations

import numpy as np


def _matches(features: np.ndarray, names: list[str], need: dict) -> np.ndarray:
    conditions = need.get("conditions")
    if not conditions:
        raise ValueError("Every need must contain at least one feature condition")
    mask = np.ones(len(features), dtype=bool)
    for condition in conditions:
        feature = condition.get("feature")
        if feature not in names:
            raise ValueError(f"Unknown query feature: {feature}")
        column = names.index(feature)
        if "feature_index" in condition and condition["feature_index"] != column:
            raise ValueError("Query feature_index disagrees with feature_names")
        values = features[:, column]
        mask &= np.isfinite(values)
        lower, upper = condition.get("lower"), condition.get("upper")
        if lower is not None and not np.isfinite(lower):
            raise ValueError("Query lower threshold must be finite or None")
        if upper is not None and not np.isfinite(upper):
            raise ValueError("Query upper threshold must be finite or None")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("Query lower threshold exceeds its upper threshold")
        if lower is not None:
            mask &= values >= lower if condition.get("lower_inclusive", False) else values > lower
        if upper is not None:
            mask &= values <= upper if condition.get("upper_inclusive", True) else values < upper
    return mask


def retrieve_candidates(
    features: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    needs: list[dict],
    frame_ids: list,
    *,
    limit: int,
    exploration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Return exactly ``limit`` unique FIT indices and aligned selection reasons.

    Caller supplies FIT rows only and enforces FIT/D/val operation disjointness.
    Reserve ``floor(limit * exploration_fraction)`` random exploration slots.
    Rank remaining trait slots by the sum of matching positive need priorities;
    seeded random values break ties. Matching requires every query feature to be
    finite and inside its explicit bounds. Random exploration can include any
    remaining FIT row. Insufficient trait matches are filled randomly and labeled
    ``random_fallback``. Neither kind of random slot claims a trait match.

    Each reason is ``{frame_id, selection_reason, need_ids}``, in returned index
    order. ``need_ids`` lists actual matching queries, including for random slots.
    Duplicate frame IDs are rejected instead of silently shrinking the budget.
    """
    features = np.asarray(features, dtype=np.float64)
    names = list(feature_names)
    if features.ndim != 2 or features.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("features must be [N,F] with unique feature_names")
    if len(frame_ids) != len(features):
        raise ValueError("frame_ids must align with feature rows")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame_ids must be unique")
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or not 0 <= limit <= len(features):
        raise ValueError("limit must be an integer between zero and FIT frame count")
    if not np.isfinite(exploration_fraction) or not 0 <= exploration_fraction <= 1:
        raise ValueError("exploration_fraction must be in [0, 1]")
    rng = np.random.default_rng(seed)
    scores = np.zeros(len(features), dtype=np.float64)
    memberships: list[list[str]] = [[] for _ in range(len(features))]
    seen_ids = set()
    for need in needs:
        need_id = need.get("need_id")
        if not isinstance(need_id, str) or not need_id or need_id in seen_ids:
            raise ValueError("Every need requires a unique nonempty need_id")
        seen_ids.add(need_id)
        priority = float(need["priority"])
        if not np.isfinite(priority) or priority <= 0:
            raise ValueError("Need priority must be finite and positive")
        matches = _matches(features, names, need)
        scores[matches] += priority
        for index in np.flatnonzero(matches):
            memberships[int(index)].append(need_id)
    if not np.isfinite(scores).all():
        raise ValueError("Combined need priorities overflowed")
    exploration_count = int(np.floor(limit * exploration_fraction))
    trait_count = limit - exploration_count
    candidates = np.flatnonzero(scores > 0)
    tie_break = rng.random(len(features))
    ranked = candidates[np.lexsort((tie_break[candidates], -scores[candidates]))]
    matched = ranked[:trait_count]
    chosen = [int(index) for index in matched]
    labels = ["feature_match"] * len(chosen)
    available = np.ones(len(features), dtype=bool)
    available[matched] = False
    if exploration_count:
        exploration = rng.choice(np.flatnonzero(available), size=exploration_count, replace=False)
        chosen.extend(int(index) for index in exploration)
        labels.extend(["exploration"] * len(exploration))
        available[exploration] = False
    shortfall = limit - len(chosen)
    if shortfall:
        fallback = rng.choice(np.flatnonzero(available), size=shortfall, replace=False)
        chosen.extend(int(index) for index in fallback)
        labels.extend(["random_fallback"] * len(fallback))
    reasons = [
        {
            "frame_id": frame_ids[index].item() if isinstance(frame_ids[index], np.generic) else frame_ids[index],
            "selection_reason": label,
            "need_ids": memberships[index],
        }
        for index, label in zip(chosen, labels)
    ]
    return np.asarray(chosen, dtype=np.int64), reasons
