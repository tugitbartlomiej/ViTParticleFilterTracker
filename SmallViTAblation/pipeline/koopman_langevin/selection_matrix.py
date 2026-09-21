#!/usr/bin/env python3
"""Build every arm of the controlled frame-selection experiment from ONE shared start.

All arms begin with the same L0 frames and add the same number of frames under the same
spacing rule, so a difference between two arms has exactly one cause. Arms:

  B0_consecutive   contiguous runs of neighbouring frames (the way video is labelled by hand)
  B1_random        random draw
  B2_uniform_time  evenly spread in time
  B3_coverage      spread in DINOv2 appearance space (core-set style)
  B4_uncertainty   B3 mixed with 1 - confidence of the detector trained on L0 (needs --uncertainty)
  K1_langevin      B3 mixed with the residual of the extended Langevin equation WITH history
  K2_no_history    same mixing, residual of the model that sees the current latent only
  K3_placebo       same mixing, K1 scores shuffled among the frames of each video
  K4_koopman_cover coverage in the coordinates of the slowest Koopman eigenfunctions of the latent

What each contrast isolates:
  K1 - B3   does the dynamics signal add anything to appearance coverage
  K1 - B4   is it better than the standard active-learning signal
  K1 - K2   is the HISTORY (memory, velocity) responsible, or any next-frame residual
  K1 - K3   is the signal itself responsible, or merely mixing a second criterion in
  K4 - B3   does covering the Koopman state space beat covering appearance
  * - B0    gain over labelling consecutive frames

B4, K1, K2 and K3 all go through `coverage_residual`/`coverage_uncertainty` of the production
`select_batch` with identical weights; only the auxiliary score differs. Scores are turned into
percentile ranks inside each operation group, so no arm wins by the scale of its score.
Labels are never read. The Langevin models are fitted on the unlabelled training pool
(transductive, stated as such); validation and test frames are not touched.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.selection_matrix \
        --pool .../pool_manifest.jsonl --cache .../cache/dinov2/<id> \
        --initial-selection .../variants/tracking_uniform_time_b500_s42.json \
        --budget 1500 --out SmallViTAblation/curated/selection_matrix_b1500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..active_learning.acquisition import select_batch
from .latent_langevin import LatentLangevin
from .select import load_cache, observation_rows, read_jsonl


def group_percentile(values: dict[str, float], rows: list[dict]) -> dict[str, float]:
    by_group: dict[str, list[str]] = {}
    for r in rows:
        if r['frame_id'] in values:
            by_group.setdefault(str(r['group_id']), []).append(r['frame_id'])
    out = {}
    for ids in by_group.values():
        order = np.argsort([values[f] for f in ids])
        for rank, k in enumerate(order):
            out[ids[k]] = (rank + .5) / len(ids)
    return out


def koopman_coordinates(model: LatentLangevin, modes: int, alpha: float = 1e-2) -> np.ndarray:
    """Coordinates of every frame in the slowest eigenfunctions of the one-step operator on [z_t, z_{t-1}]."""
    z, w = model.z, model.windows
    x0 = np.hstack([z[w[:, -2]], z[w[:, -3]]])
    x1 = np.hstack([z[w[:, -1]], z[w[:, -2]]])
    gram = x0.T @ x0 / len(x0)
    gram.flat[::len(gram) + 1] += alpha
    values, right = np.linalg.eig(np.linalg.solve(gram, x0.T @ x1 / len(x0)))
    order = np.argsort(-np.abs(values))[:modes]
    lifted = np.hstack([z, z])                                   # frames without a predecessor: static lift
    lifted[w[:, -1]] = x1
    coords = lifted @ right[:, order]
    out = np.hstack([coords.real, coords.imag])
    return out[:, out.std(0) > 1e-9]


def consecutive(rows: list[dict], selected: list[str], need: int, seed: int) -> list[str]:
    """Contiguous runs: a random start in a random video, then the following eligible frames."""
    rng = np.random.default_rng(seed)
    taken = set(selected)
    by_video: dict[str, list[dict]] = {}
    for r in rows:
        by_video.setdefault(str(r['video_id']), []).append(r)
    for v in by_video.values():
        v.sort(key=lambda r: r['frame_index'])
    videos = sorted(by_video)
    run = max(10, need // max(len(videos) // 2, 1))
    out: list[str] = []
    while len(out) < need:
        frames = by_video[videos[rng.integers(len(videos))]]
        start = int(rng.integers(len(frames)))
        for r in frames[start:start + run]:
            if r['frame_id'] not in taken and len(out) < need:
                taken.add(r['frame_id'])
                out.append(r['frame_id'])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--initial-selection', type=Path, required=True)
    ap.add_argument('--uncertainty', type=Path, help='JSON {frame_id: 1 - max confidence} from the L0 detector; enables B4')
    ap.add_argument('--budget', type=int, default=1500)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-gap-frames', type=float, default=5)
    ap.add_argument('--max-gap-frames', type=float, default=7.5)
    ap.add_argument('--dim', type=int, default=24)
    ap.add_argument('--lags', type=int, default=8)
    ap.add_argument('--koopman-modes', type=int, default=8)
    ap.add_argument('--coverage-weight', type=float, default=.50)
    ap.add_argument('--auxiliary-weight', type=float, default=.50)
    args = ap.parse_args(argv)

    started = time.perf_counter()
    uncertainty = None
    if args.uncertainty:                                     # read first: fail before the long fits, not after
        uncertainty = {str(k): float(v) for k, v in json.loads(args.uncertainty.read_text(encoding='utf-8')).items()}
    pool = [r for r in read_jsonl(args.pool) if r.get('split', 'train') == 'train']
    cache_ids, cache_x = load_cache(args.cache)
    position = {f: i for i, f in enumerate(cache_ids)}
    pool_x = cache_x[[position[r['frame_id']] for r in pool]]
    raw = json.loads(args.initial_selection.read_text(encoding='utf-8'))
    l0 = [str(f) for f in raw.get('selected_frame_ids', raw.get('frame_ids'))]
    need = args.budget - len(l0)
    if need <= 0 or len(set(l0)) != len(l0) or not set(l0) <= {r['frame_id'] for r in pool}:
        raise ValueError('L0 must be a duplicate-free subset of the pool, smaller than the budget')

    observed = observation_rows(cache_ids)
    group_of = {r['video_id']: r['group_id'] for r in pool}
    for r in observed:
        r['group_id'] = group_of.get(r['video_id'], r['video_id'])
    common = dict(seed=args.seed, dim=args.dim, lags=args.lags, folds=3, max_gap_frames=args.max_gap_frames)
    with_history = LatentLangevin(observed, cache_x, history=True, **common)
    no_history = LatentLangevin(observed, cache_x, history=False, **common)

    def residual(model):
        values = {observed[i]['frame_id']: float(e) for i, e in zip(model.target, model.err_pool) if np.isfinite(e)}
        return group_percentile(values, pool)

    k1, k2 = residual(with_history), residual(no_history)
    print(f'[matrix] Langevin fits done at {time.perf_counter() - started:.0f}s', flush=True)
    rng = np.random.default_rng(args.seed)
    k3: dict[str, float] = {}
    by_video: dict[str, list[str]] = {}
    for r in pool:
        if r['frame_id'] in k1:
            by_video.setdefault(str(r['video_id']), []).append(r['frame_id'])
    for ids in by_video.values():
        for f, g in zip(ids, rng.permutation(ids)):
            k3[f] = k1[g]
    coords = koopman_coordinates(with_history, args.koopman_modes)
    pool_koopman = coords[[position[r['frame_id']] for r in pool]]
    pool_koopman = (pool_koopman - pool_koopman.mean(0)) / pool_koopman.std(0)

    def batch(strategy, features, scores=None, field='residual_mahalanobis'):
        shaped = {f: {field: v} for f, v in scores.items()} if scores else None
        chosen, _ = select_batch(pool, features, batch_size=need, strategy=strategy, seed=args.seed, selected_ids=l0,
                                 scores=shaped, min_gap_frames=args.min_gap_frames,
                                 coverage_weight=args.coverage_weight, auxiliary_weight=args.auxiliary_weight)
        print(f'[matrix] {strategy} done at {time.perf_counter() - started:.0f}s', flush=True)
        return chosen

    arms = {'B0_consecutive': consecutive(pool, l0, need, args.seed),
            'B1_random': batch('random', pool_x),
            'B2_uniform_time': batch('uniform_time', pool_x),
            'B3_coverage': batch('coverage', pool_x),
            'K1_langevin': batch('coverage_residual', pool_x, k1),
            'K2_no_history': batch('coverage_residual', pool_x, k2),
            'K3_placebo': batch('coverage_residual', pool_x, k3),
            'K4_koopman_cover': batch('coverage', pool_koopman)}
    if uncertainty is not None:
        missing = [r['frame_id'] for r in pool if r['frame_id'] not in uncertainty]
        if missing:
            raise ValueError(f'uncertainty file misses {len(missing)} pool frames, e.g. {missing[:3]}')
        arms['B4_uncertainty'] = batch('coverage_uncertainty', pool_x, group_percentile(uncertainty, pool), field='uncertainty')

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(budget=args.budget, l0=len(l0), added=need, seed=args.seed, arms={},
                    pool_sha256=hashlib.sha256(args.pool.read_bytes()).hexdigest(),
                    l0_sha256=hashlib.sha256('\n'.join(l0).encode()).hexdigest(), labels_used_for_ranking=False,
                    langevin=dict(with_history=with_history.summary(), no_history=no_history.summary()),
                    config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})
    reference = set(arms['K1_langevin'])
    for name, added in arms.items():
        frames = l0 + list(added)
        if len(frames) != args.budget or len(set(frames)) != args.budget:
            raise RuntimeError(f'{name}: expected {args.budget} unique frames, got {len(set(frames))}')
        variant_id = f'matrix_{name}_b{args.budget}_sel{args.seed}'
        (args.out / f'{variant_id}.json').write_text(json.dumps(dict(
            schema_version=1, variant_id=variant_id, run_id=variant_id, strategy=name, seed=args.seed,
            budget=args.budget, frame_ids=frames, selected_frame_ids=frames, labels_used_for_ranking=False)), encoding='utf-8')
        manifest['arms'][name] = dict(file=f'{variant_id}.json', overlap_with_K1=len(reference & set(added)) / need,
                                      sha256=hashlib.sha256('\n'.join(frames).encode()).hexdigest())
        print(f'{name:18s} wspolne z K1: {manifest["arms"][name]["overlap_with_K1"]:.2f}', flush=True)
    (args.out / 'MATRIX_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
