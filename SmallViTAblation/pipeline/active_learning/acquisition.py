"""Model-agnostic acquisition with exact budgets and temporal safeguards.

The selector consumes metadata, frozen embeddings and optional scalar scores.
It deliberately never inspects annotations or detector ground truth.  Scores
must have been produced from the training pool only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Mapping, Sequence

import numpy as np


STRATEGIES = (
    "random",
    "uniform_time",
    "coverage",
    "coverage_uncertainty",
    "coverage_residual",
    "coverage_history",
)

SCORE_FIELD = {
    "coverage_uncertainty": "uncertainty",
    "coverage_residual": "residual_mahalanobis",
    "coverage_history": "history_gain",
}


def _stable_noise(seed: int, frame_id: str) -> float:
    raw = hashlib.sha256(f"{seed}:{frame_id}".encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "big") / float(2**64)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("embeddings must be a non-empty [n_frames, n_features] matrix")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain NaN or infinity")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return values / norm


def _rank01(values: np.ndarray) -> np.ndarray:
    """Stable percentile ranks; robust to score scale and constant vectors."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(len(values), dtype=np.float64)
    finite = np.isfinite(values)
    idx = np.flatnonzero(finite)
    if len(idx) < 2:
        return out
    order = idx[np.argsort(values[idx], kind="stable")]
    out[order] = np.arange(len(order), dtype=np.float64) / (len(order) - 1)
    return out


def _validate(rows: Sequence[dict], embeddings: np.ndarray | None,
              selected_ids: set[str], strategy: str) -> dict[str, int]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if embeddings is None and strategy not in {"random", "uniform_time"}:
        raise ValueError(f"strategy {strategy!r} requires real embeddings")
    if embeddings is not None and len(rows) != len(embeddings):
        raise ValueError("rows and embeddings have different lengths")
    ids = [str(row["frame_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate frame_id in acquisition pool")
    non_train = [row["frame_id"] for row in rows if row.get("split", "train") != "train"]
    if non_train:
        raise ValueError(f"acquisition pool contains non-training frames: {non_train[:3]}")
    index = {frame_id: i for i, frame_id in enumerate(ids)}
    missing = sorted(selected_ids - index.keys())
    if missing:
        raise ValueError(f"selected frames are absent from pool: {missing[:3]}")
    for field in ("group_id", "video_id", "frame_index"):
        if any(field not in row for row in rows):
            raise ValueError(f"pool rows require {field}")
    return index


def _temporal_ok(row: dict, chosen_by_video: Mapping[str, list[float]],
                 min_gap_frames: float) -> bool:
    if min_gap_frames <= 0:
        return True
    times = chosen_by_video.get(str(row["video_id"]), [])
    t = float(row["frame_index"])
    return all(abs(t - other) >= min_gap_frames for other in times)


def _coverage_scores(z: np.ndarray, selected: list[int]) -> np.ndarray:
    if selected:
        # Cosine k-center: novelty is distance to the nearest selected frame.
        return np.min(1.0 - z @ z[np.asarray(selected)].T, axis=1)
    # Deterministic medoid-like first point.  Starting with an outlier makes
    # small-budget k-center needlessly unstable.
    centre = z.mean(axis=0)
    centre /= max(np.linalg.norm(centre), 1e-12)
    return z @ centre


def _uniform_scores(rows: Sequence[dict], selected: set[int]) -> np.ndarray:
    chosen = defaultdict(list)
    ranges: dict[str, tuple[float, float]] = {}
    for i, row in enumerate(rows):
        video = str(row["video_id"])
        t = float(row["frame_index"])
        lo, hi = ranges.get(video, (t, t))
        ranges[video] = min(lo, t), max(hi, t)
        if i in selected:
            chosen[video].append(t)
    out = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        video = str(row["video_id"])
        t = float(row["frame_index"])
        lo, hi = ranges[video]
        scale = max(hi - lo, 1.0)
        if chosen[video]:
            out[i] = min(abs(t - old) for old in chosen[video]) / scale
        else:
            # Pick the temporal midpoint first in each unseen video.
            out[i] = 1.0 - abs(t - (lo + hi) / 2.0) / scale
    return out


def _score_vector(rows: Sequence[dict], field: str,
                  scores: Mapping[str, Mapping[str, float]] | None) -> np.ndarray:
    if scores is None:
        raise ValueError(f"strategy requires score field {field!r}")
    values = np.full(len(rows), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        value = scores.get(str(row["frame_id"]), {}).get(field)
        if value is not None:
            values[i] = float(value)
    if not np.isfinite(values).any():
        raise ValueError(f"no finite values found for required score {field!r}")
    # Boundary frames without a valid history remain eligible through coverage,
    # but receive zero auxiliary priority.
    ranked = _rank01(values)
    ranked[~np.isfinite(values)] = 0.0
    return ranked


def select_batch(
    rows: Sequence[dict],
    embeddings: np.ndarray | None,
    *,
    batch_size: int,
    strategy: str,
    seed: int,
    selected_ids: Sequence[str] = (),
    scores: Mapping[str, Mapping[str, float]] | None = None,
    min_gap_frames: float = 0,
    coverage_weight: float = 0.60,
    auxiliary_weight: float = 0.40,
    group_balance_weight: float = 0.15,
) -> tuple[list[str], dict]:
    """Select one incremental batch and return frame ids plus an audit record."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if min_gap_frames < 0:
        raise ValueError("min_gap_frames cannot be negative")
    selected_set = set(map(str, selected_ids))
    index = _validate(rows, embeddings, selected_set, strategy)
    available = len(rows) - len(selected_set)
    if batch_size > available:
        raise ValueError(f"requested {batch_size} frames, only {available} remain")
    if min(coverage_weight, auxiliary_weight, group_balance_weight) < 0:
        raise ValueError("score weights cannot be negative")

    z = _unit_rows(embeddings) if embeddings is not None else None
    selected = [index[x] for x in selected_set]
    chosen_by_video: dict[str, list[float]] = defaultdict(list)
    group_counts = Counter()
    group_sizes = Counter(str(row["group_id"]) for row in rows)
    for i in selected:
        row = rows[i]
        chosen_by_video[str(row["video_id"])].append(float(row["frame_index"]))
        group_counts[str(row["group_id"])] += 1
    group_weight_sum = sum(np.sqrt(v) for v in group_sizes.values())

    random_priority = np.array([_stable_noise(seed, str(r["frame_id"])) for r in rows])
    auxiliary = None
    if strategy in SCORE_FIELD:
        auxiliary = _score_vector(rows, SCORE_FIELD[strategy], scores)

    picked: list[int] = []
    trace: list[dict] = []
    spacing_relaxations = 0
    for _ in range(batch_size):
        blocked = set(selected) | set(picked)
        if strategy == "random":
            base = random_priority.copy()
        elif strategy == "uniform_time":
            base = _uniform_scores(rows, blocked)
        else:
            if z is None:  # guarded by _validate
                raise RuntimeError("coverage selection has no embeddings")
            coverage = _rank01(_coverage_scores(z, list(blocked)))
            base = coverage
            if auxiliary is not None:
                total = coverage_weight + auxiliary_weight
                if total <= 0:
                    raise ValueError("coverage and auxiliary weights sum to zero")
                base = (coverage_weight * coverage + auxiliary_weight * auxiliary) / total

        total_after = len(selected) + len(picked) + 1
        balance = np.array([
            total_after * np.sqrt(group_sizes[str(row["group_id"])]) / group_weight_sum
            - group_counts[str(row["group_id"])]
            for row in rows
        ])
        combined = base + group_balance_weight * _rank01(balance)
        combined += random_priority * 1e-12  # deterministic tie-break only
        if blocked:
            combined[list(blocked)] = -np.inf

        eligible = np.array([
            i not in blocked and _temporal_ok(row, chosen_by_video, min_gap_frames)
            for i, row in enumerate(rows)
        ])
        if eligible.any():
            combined[~eligible] = -np.inf
            relaxed = False
        else:
            spacing_relaxations += 1
            relaxed = True
        winner = int(np.argmax(combined))
        if not np.isfinite(combined[winner]):
            raise RuntimeError("selector could not satisfy the requested exact budget")
        picked.append(winner)
        row = rows[winner]
        video = str(row["video_id"])
        group = str(row["group_id"])
        chosen_by_video[video].append(float(row["frame_index"]))
        group_counts[group] += 1
        trace.append({
            "frame_id": str(row["frame_id"]),
            "rank_in_batch": len(picked),
            "combined_score": float(combined[winner]),
            "spacing_relaxed": relaxed,
        })

    ids = [str(rows[i]["frame_id"]) for i in picked]
    if len(ids) != batch_size or len(ids) != len(set(ids)):
        raise RuntimeError("internal exact-budget invariant failed")
    return ids, {
        "strategy": strategy,
        "seed": int(seed),
        "requested": int(batch_size),
        "selected_before": len(selected_set),
        "selected_after": len(selected_set) + len(ids),
        "min_gap_frames": float(min_gap_frames),
        "spacing_relaxations": spacing_relaxations,
        "annotation_fields_read": False,
        "features_used_for_ranking": strategy not in {"random", "uniform_time"},
        "score_field": SCORE_FIELD.get(strategy),
        "trace": trace,
    }
