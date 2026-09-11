"""Temporal anchors, feature coverage and an optional experimental dynamics score.

Coverage and temporal redundancy controls are motivated by COWAL
(https://doi.org/10.1109/WACV57701.2024.00201) and LRTD
(https://doi.org/10.1007/s11548-020-02198-9). This is an adaptation, not their
reproduction. No claim of publication novelty or measured detector improvement.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import time

import numpy as np

from .dynamics import project


def allocate(capacities: dict, budget: int, mode='sqrt') -> dict:
    if budget < 0 or budget > sum(capacities.values()):
        raise ValueError('Budget outside available unique candidate count')
    if mode not in {'equal', 'sqrt', 'proportional'}:
        raise ValueError('Unknown group allocation mode')
    result = dict.fromkeys(capacities, 0)
    remaining = budget
    while remaining:
        active = sorted(k for k, cap in capacities.items() if result[k] < cap)
        weight = {k: (1 if mode == 'equal' else math.sqrt(capacities[k])
                      if mode == 'sqrt' else capacities[k]) for k in active}
        total = sum(weight.values())
        desired = {k: remaining*weight[k]/total for k in active}
        increments = {k: min(capacities[k]-result[k], int(desired[k])) for k in active}
        if not sum(increments.values()):
            for k in sorted(active, key=lambda k: (-desired[k], k))[:remaining]:
                increments[k] = 1
        for k, count in increments.items():
            result[k] += count
            remaining -= count
    return result


def percentile_rank(values):
    """Tie-aware ranks; invalid values have no priority and constant vectors stay zero."""
    result = np.zeros(len(values), dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < 2:
        return result
    unique, inverse, counts = np.unique(values[valid], return_inverse=True, return_counts=True)
    if len(unique) == 1:
        return result
    rank = (np.cumsum(counts)-.5*(counts+1))/(len(valid)-1)
    result[valid] = rank[inverse]
    return result


def choose(rows, features, quality, hashes, *, profile, budget, seed=42,
           allocation='sqrt', anchor_fraction=.25, importance_weight=.5,
           min_gap_frames=5, spacing_policy='relax', dynamics=None,
           signal='history_gain', projection_dim=64):
    if profile not in {'simple', 'balanced', 'dynamics'}:
        raise ValueError('Unknown curation profile')
    if budget <= 0 or not 0 <= anchor_fraction <= 1 or importance_weight < 0 or min_gap_frames < 0:
        raise ValueError('Invalid selection parameters')
    if spacing_policy not in {'strict', 'relax'}:
        raise ValueError('Invalid spacing policy')
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Only training candidates may enter selection')
    started = time.perf_counter()
    embedding = project(features, projection_dim, seed)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
    eligible = np.ones(len(rows), dtype=bool)
    seen, duplicate_of = {}, {}
    for i, value in enumerate(hashes):
        key = (rows[i]['native_width'], rows[i]['native_height'], value)
        if key in seen:
            eligible[i] = False
            duplicate_of[rows[i]['frame_id']] = rows[seen[key]]['frame_id']
        else:
            seen[key] = i
    by_group, by_video = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        if eligible[i]:
            by_group[row['group_id']].append(i)
            by_video[row['video_id']].append(i)
    quotas = allocate({g: len(v) for g, v in by_group.items()}, budget, allocation)
    importance = np.zeros(len(rows))
    if profile == 'dynamics':
        if dynamics is None or signal not in dynamics:
            raise ValueError('Requested dynamics score has not been computed')
        values = dynamics[signal].copy()
        if signal == 'history_gain':
            values[values <= 0] = np.nan
        importance = percentile_rank(values)
    sharpness = percentile_rank(np.log1p(quality[:, 0]))
    selected, reasons = [], {}
    used = np.zeros(len(rows), dtype=bool)
    blocked = np.zeros(len(rows), dtype=bool)
    relaxations = []
    video_indices, positions = {}, {}
    for video, indices in by_video.items():
        indices = np.array(sorted(indices, key=lambda i: rows[i]['frame_index']))
        video_indices[video] = indices
        positions[video] = np.array([rows[i]['frame_index'] for i in indices])

    def take(candidates, utility, reason):
        candidates = np.asarray(candidates, dtype=np.int64)
        available = ~used[candidates]
        spaced = available & ~blocked[candidates]
        relaxed = False
        if not spaced.any():
            if not available.any():
                return None
            if spacing_policy == 'strict':
                raise ValueError('Budget incompatible with strict temporal spacing; reduce K or gap')
            spaced = available
            relaxed = True
        local = np.flatnonzero(spaced)
        # Canonical row ordering resolves ties independently of hash/set ordering.
        winner = int(candidates[local[np.argmax(utility[local])]])
        used[winner] = True
        selected.append(winner)
        reasons[winner] = reason
        video = rows[winner]['video_id']
        t = rows[winner]['frame_index']
        lo = np.searchsorted(positions[video], t-min_gap_frames, side='right')
        hi = np.searchsorted(positions[video], t+min_gap_frames, side='left')
        blocked[video_indices[video][lo:hi]] = True
        if relaxed:
            relaxations.append(rows[winner]['frame_id'])
        return winner

    for group, members in sorted(by_group.items()):
        quota = quotas[group]
        if not quota:
            continue
        reserve = quota if profile == 'simple' else min(quota, max(1, math.ceil(quota*anchor_fraction)))
        vids = defaultdict(list)
        for i in members:
            vids[rows[i]['video_id']].append(i)
        per_video = allocate({v: len(ids) for v, ids in vids.items()}, reserve, 'proportional')
        start = len(selected)
        for video, indices in sorted(vids.items()):
            count = per_video[video]
            if not count:
                continue
            ids = np.array(sorted(indices, key=lambda i: rows[i]['frame_index']))
            times = np.array([rows[i]['frame_index'] for i in ids], dtype=float)
            boundaries = np.linspace(times[0]-.5, times[-1]+.5, count+1)
            for j in range(count):
                bin_ids = ids[(times >= boundaries[j]) & (times < boundaries[j+1])]
                # Sparse segmentation samples can leave a time interval empty.
                if not len(bin_ids) or used[bin_ids].all():
                    bin_ids = ids[~used[ids]]
                t = np.array([rows[i]['frame_index'] for i in bin_ids], dtype=float)
                midpoint = (boundaries[j]+boundaries[j+1])/2
                temporal = -np.abs(t-midpoint)/max(boundaries[j+1]-boundaries[j], 1)
                utility = temporal.copy()
                if profile == 'simple' and j % 4:
                    utility += .5*sharpness[bin_ids]
                take(bin_ids, utility, 'temporal_anchor' if j % 4 == 0 or profile != 'simple'
                     else 'quality_within_temporal_bin')
        local = np.array(members)
        distance = np.full(len(local), np.inf)
        for i in selected[start:]:
            distance = np.minimum(distance, np.sum((embedding[local]-embedding[i])**2, axis=1))
        while len(selected)-start < quota:
            # Distances remain the primary criterion; diagnostics only modulate coverage.
            utility = distance*(1+importance_weight*importance[local])
            winner = take(local, utility, 'coverage' if profile != 'dynamics' else f'coverage+{signal}')
            if winner is None:
                raise RuntimeError('No eligible candidate left within group quota')
            distance = np.minimum(distance, np.sum((embedding[local]-embedding[winner])**2, axis=1))
    if len(selected) != budget or len(set(selected)) != budget:
        raise RuntimeError('Selection failed exact unique frame budget')
    selected.sort()
    meta = dict(profile=profile, unique_budget=budget, input_frames=len(rows),
        unique_pixel_frames=int(eligible.sum()), exact_duplicates_removed=len(duplicate_of),
        group_quotas=quotas, group_allocation=allocation, anchor_fraction=anchor_fraction,
        min_gap_frames=min_gap_frames, spacing_policy=spacing_policy,
        spacing_relaxations=relaxations, projection_dimension=embedding.shape[1],
        importance_weight=importance_weight if profile == 'dynamics' else 0,
        score_source=signal if profile == 'dynamics' else None,
        labels_used_for_ranking=False, offline=True, wall_seconds=time.perf_counter()-started)
    return selected, reasons, duplicate_of, embedding, meta


def measurements(rows, selected, features, quality, seed=42):
    """Descriptive curation metrics; no substitute for downstream detector AP."""
    rng = np.random.default_rng(seed)
    probes = rng.choice(len(rows), min(4096, len(rows)), replace=False)
    distance = np.full(len(probes), np.inf)
    for start in range(0, len(selected), 128):
        centres = features[selected[start:start+128]]
        dist = (features[probes]**2).sum(1)[:, None] + (centres**2).sum(1)[None, :]
        dist -= 2*features[probes] @ centres.T
        distance = np.minimum(distance, np.maximum(dist, 0).min(1))
    by_video = defaultdict(list)
    for i in selected:
        by_video[rows[i]['video_id']].append(rows[i]['frame_index'])
    gaps = [b-a for times in by_video.values() for a, b in zip(sorted(times), sorted(times)[1:])]
    def quantiles(values):
        return dict(zip(('p05', 'p50', 'p95'), map(float, np.quantile(values, [.05, .5, .95])))) if len(values) else None
    selected_rows = [rows[i] for i in selected]
    def box_areas(records):
        return [a['bbox'][2]*a['bbox'][3]/(r['native_width']*r['native_height'])
                for r in records for a in r['annotations']]
    return dict(descriptive_only=True, detector_performance_measured=False,
        feature_coverage=dict(probe_count=len(probes), seed=seed, distance='projected_l2',
            mean=float(np.sqrt(distance).mean()), p95=float(np.quantile(np.sqrt(distance), .95)),
            max=float(np.sqrt(distance).max())),
        temporal_gap_frames=quantiles(gaps), videos_selected=len(by_video),
        group_counts=dict(Counter(r['group_id'] for r in selected_rows)),
        sharpness_pool=quantiles(quality[:, 0]), sharpness_selected=quantiles(quality[selected, 0]),
        annotation_audit=dict(post_selection_only=True,
            target_instances=sum(len(r['annotations']) for r in selected_rows),
            unknown_label_frames=sum(r['label_status'] == 'unknown' for r in selected_rows),
            provided_empty_frames=sum(r['label_status'] == 'provided' and not r['annotations']
                                      for r in selected_rows),
            bbox_area_fraction_pool=quantiles(box_areas(rows)),
            bbox_area_fraction_selected=quantiles(box_areas(selected_rows)),
            source_classes=dict(Counter(a.get('source_category', 'instrument')
                for r in selected_rows for a in r['annotations']))))
