"""Task-error acquisition and an operation-local temporal placebo.

Scores use candidate labels and therefore describe a labelled diagnostic
experiment, not label-free active learning. This module fits no model.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from collections import defaultdict
from numbers import Integral

import numpy as np


def _vector(values, name, length=None):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f"{name} must have shape ({length if length is not None else 'n'},)")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} must contain finite, nonnegative values")
    return result


def _validate_rows(rows):
    required = {"frame_id", "group_id", "video_id", "frame_index", "split"}
    frame_ids, physical_frames = set(), set()
    for row in rows:
        if required - row.keys():
            raise ValueError(f"row is missing required fields: {sorted(required - row.keys())}")
        for key in ("frame_id", "group_id", "video_id", "split"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"{key} must be a nonempty string")
        frame = row["frame_index"]
        if isinstance(frame, bool) or not isinstance(frame, Integral) or frame < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        physical = (row["video_id"], int(frame))
        if row["frame_id"] in frame_ids or physical in physical_frames:
            raise ValueError("duplicate frame_id or duplicate video/frame_index")
        frame_ids.add(row["frame_id"])
        physical_frames.add(physical)


def task_errors(pred_box, target_box):
    """Mean SmoothL1(beta=0.05) over normalized cx, cy, width, height.

    Predictions are not clipped: out-of-image predictions must remain errors.
    Conversion to normalized cxcywh is the caller's responsibility.
    """
    prediction = np.asarray(pred_box, dtype=np.float64)
    target = np.asarray(target_box, dtype=np.float64)
    if prediction.ndim != 2 or prediction.shape[1] != 4 or target.shape != prediction.shape:
        raise ValueError("prediction and target must have the same shape (n, 4)")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("boxes must contain finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(prediction - target)
    if not np.isfinite(difference).all():
        raise ValueError("box differences must be finite")
    beta = .05
    loss = difference - beta / 2
    quadratic = difference < beta
    loss[quadratic] = .5 * difference[quadratic] ** 2 / beta
    return loss.mean(axis=1)


def recoverable_scores(e_obs, e_dyn):
    """Positive task-error reduction from the temporal latent prediction."""
    observed = _vector(e_obs, "e_obs")
    dynamic = _vector(e_dyn, "e_dyn", len(observed))
    return np.maximum(0., observed - dynamic)


def placebo_scores(rows, e_obs, e_dyn, seed, bins=4, eligible=None):
    """Permute relative gain inside operation and observed-loss quantile bins.

    The observed loss stays attached to its original frame. Non-training rows
    receive score zero and never participate in a permutation. Audit metadata
    gives each row's bin and the source row of its permuted relative gain.
    Equal observed losses are never split across quantile bins.
    """
    rows = list(rows)
    _validate_rows(rows)
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("bins must be a positive integer")
    observed = _vector(e_obs, "e_obs", len(rows))
    dynamic = _vector(e_dyn, "e_dyn", len(rows))
    eligible = np.ones(len(rows), dtype=bool) if eligible is None else np.asarray(eligible)
    if eligible.shape != (len(rows),) or eligible.dtype != np.bool_:
        raise ValueError('eligible must be a boolean vector aligned with rows')
    gain = np.zeros(len(rows), dtype=np.float64)
    improved = (observed > 0) & (dynamic < observed)
    gain[improved] = 1. - dynamic[improved] / observed[improved]
    rng = np.random.default_rng(seed)
    source = np.arange(len(rows))
    bin_ids = np.full(len(rows), -1, dtype=int)
    group_indices = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] == "train" and eligible[i]:
            group_indices[row["group_id"]].append(i)
    strata, quantile_edges = [], {}
    for group in sorted(group_indices):
        indices = np.asarray(group_indices[group], dtype=int)
        edges = np.quantile(observed[indices], np.arange(1, bins) / bins)
        assignments = np.searchsorted(edges, observed[indices], side="right")
        bin_ids[indices] = assignments
        quantile_edges[group] = edges.tolist()
        for bin_id in sorted(set(assignments.tolist())):
            members = indices[assignments == bin_id]
            shuffled = rng.permutation(members)
            source[members] = shuffled
            strata.append({
                "group_id": group, "bin_id": int(bin_id), "size": len(members),
                "permuted_row_count": int(np.count_nonzero(members != shuffled)),
            })
    scores = np.zeros(len(rows), dtype=np.float64)
    train = bin_ids >= 0
    scores[train] = observed[train] * gain[source[train]]
    return scores, {
        "seed": int(seed), "bins": int(bins),
        "permutation_scope": "eligible_history_within_group_id_and_observed_loss_quantile",
        "source_indices": source.tolist(), "bin_ids": bin_ids.tolist(),
        "quantile_edges": quantile_edges, "strata": strata,
        "excluded_non_train_count": sum(r['split'] != 'train' for r in rows),
        "excluded_missing_history_count": int(np.count_nonzero(~eligible)),
    }


def _compatible(frame, anchors, gap):
    position = bisect_left(anchors, frame)
    return ((position == 0 or frame - anchors[position - 1] >= gap)
            and (position == len(anchors) or anchors[position] - frame >= gap))


def _capacity(rows, ordered_by_video, available, anchors, gap, stop_at=None):
    """Maximum remaining cardinality under uniform per-video separation.

    Earliest-compatible-frame greedy is cardinality-optimal on each timeline.
    Fixed L0/selected anchors are checked independently of this greedy pass.
    """
    count = 0
    for video, indices in ordered_by_video.items():
        last = None
        fixed = anchors[video]
        for index in indices:
            if index not in available:
                continue
            frame = int(rows[index]["frame_index"])
            if last is not None and frame - last < gap:
                continue
            if not _compatible(frame, fixed, gap):
                continue
            last = frame
            count += 1
            if stop_at is not None and count >= stop_at:
                return count
    return count


def rank_select(rows, scores, initial_ids, budget, seed, min_gap_frames=5):
    """Select additional training IDs, never L0 IDs, with no coverage quotas.

    Positive scores rank first; ties and all remaining fill are seed-random.
    scores=None uses exactly the same random rule. Spacing includes L0 and is
    measured within video_id, not operation. If a full budget is impossible,
    the gap is reduced only as far as mathematically necessary and reported.
    The returned IDs contain only newly selected frames, in selection order.
    """
    rows = list(rows)
    _validate_rows(rows)
    for name, value in (("budget", budget), ("min_gap_frames", min_gap_frames)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    values = None if scores is None else _vector(scores, "scores", len(rows))
    initial = set(initial_ids)
    by_id = {row["frame_id"]: i for i, row in enumerate(rows)}
    if initial - by_id.keys():
        raise ValueError("initial_ids contains IDs absent from rows; spacing cannot be verified")
    if any(rows[by_id[item]]["split"] != "train" for item in initial):
        raise ValueError("initial_ids must refer only to training rows")
    candidates = [i for i, row in enumerate(rows)
                  if row["split"] == "train" and row["frame_id"] not in initial]
    if budget > len(candidates):
        raise ValueError(f"budget {budget} exceeds {len(candidates)} available training frames")
    anchors, ordered_by_video = defaultdict(list), defaultdict(list)
    for item in initial:
        row = rows[by_id[item]]
        anchors[row["video_id"]].append(int(row["frame_index"]))
    for fixed in anchors.values():
        fixed.sort()
    for index in candidates:
        ordered_by_video[rows[index]["video_id"]].append(index)
    for indices in ordered_by_video.values():
        indices.sort(key=lambda i: rows[i]["frame_index"])
    available = set(candidates)
    gap = int(min_gap_frames)
    capacity = _capacity(rows, ordered_by_video, available, anchors, gap)
    requested_capacity = capacity
    relaxations = []
    while capacity < budget:
        previous_gap, previous_capacity = gap, capacity
        gap -= 1
        capacity = _capacity(rows, ordered_by_video, available, anchors, gap)
        relaxations.append({
            "from_gap": previous_gap, "to_gap": gap,
            "capacity_before": previous_capacity, "capacity_after": capacity,
            "reason": "full_budget_impossible_at_previous_gap",
        })
    rng = np.random.default_rng(seed)
    random_order = [candidates[i] for i in rng.permutation(len(candidates))]
    if values is None:
        order = random_order
    else:
        # Stable sorting preserves the seeded random order within equal scores.
        positive = sorted((i for i in random_order if values[i] > 0),
                          key=lambda i: -values[i])
        order = positive + [i for i in random_order if values[i] == 0]
    selected = []
    if capacity == len(candidates):
        # Every candidate already satisfies the gap, as in a stride-5 pool.
        selected = order[:budget]
    else:
        for index in order:
            if len(selected) == budget:
                break
            available.remove(index)
            row = rows[index]
            frame, fixed = int(row["frame_index"]), anchors[row["video_id"]]
            if not _compatible(frame, fixed, gap):
                continue
            insort(fixed, frame)
            still_needed = budget - len(selected) - 1
            if not still_needed or _capacity(
                rows, ordered_by_video, available, anchors, gap,
                stop_at=still_needed,
            ) >= still_needed:
                selected.append(index)
            else:
                fixed.remove(frame)
    if len(selected) != budget:
        raise RuntimeError("selection failed despite a feasible budget")
    by_score = 0 if values is None else sum(values[i] > 0 for i in selected)
    by_score = int(by_score)
    random_count = len(selected) - by_score
    metadata = {
        "seed": int(seed), "mode": "random" if values is None else "ranked",
        "budget": int(budget), "selected_count": len(selected),
        "candidate_count": len(candidates), "excluded_initial_count": len(initial),
        "excluded_non_train_count": sum(row["split"] != "train" for row in rows),
        "selected_by_score": by_score, "selected_by_random": random_count,
        "random_fill_count": random_count if values is not None else 0,
        "random_fill_reason": (
            "insufficient_positive_scores_under_spacing"
            if values is not None and random_count else None
        ),
        "positive_candidate_count": (
            int(sum(values[i] > 0 for i in candidates)) if values is not None else None
        ),
        "requested_min_gap_frames": int(min_gap_frames),
        "requested_gap_capacity": requested_capacity,
        "final_min_gap_frames": gap, "spacing_relaxed": bool(relaxations),
        "relaxations": relaxations,
        "spacing_scope": "video_id_including_initial_ids",
    }
    return [rows[i]["frame_id"] for i in selected], metadata
