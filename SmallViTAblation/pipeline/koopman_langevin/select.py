#!/usr/bin/env python3
"""Round-aware frame selection driven by the extended Langevin equation on the frame latent.

Writes variant files with the same shape as `pipeline.active_learning.simulate`, so the
existing trainer and queue consume them unchanged. Labels are never read for ranking: the
label-eligible pool only limits WHICH frames may be bought, while the dynamics are fitted on
every observable training frame (roles U and A of the curation protocol).

Strategies
  langevin_reducible      coverage + |h_known - h_pool|^2 / noise, recomputed every round
  langevin_surprise       coverage + residual energy of the pool model (buys surprises)
  langevin_veto_uniform   uniform_time after removing the most surprising frames
  langevin_veto_coverage  coverage after removing the most surprising frames
The two veto variants and the surprise variant point in opposite directions on purpose:
on synthetic data the veto helps, on real clips the surprise marks fast instrument motion
and instrument entry, which a detector may need. Only detector training decides.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.select \
        --pool   SmallViTAblation/feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl \
        --cache  SmallViTAblation/feature_cache/eden_static36/cache/dinov2/<cache_id> \
        --strategy langevin_reducible --budgets 500 1500 --out SmallViTAblation/curated/koopman_langevin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np

from ..active_learning.acquisition import select_batch
from .latent_langevin import LatentLangevin

STRATEGIES = ('langevin_reducible', 'langevin_surprise', 'langevin_veto_uniform', 'langevin_veto_coverage')


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_cache(directory: Path) -> tuple[list[str], np.ndarray]:
    """Feature cache written by pipeline.curation.features.build_features (chunked NPZ)."""
    ids, feats = [], []
    for chunk in sorted(directory.glob('*.npz')):
        with np.load(chunk, allow_pickle=False) as data:
            ids.extend(map(str, data['frame_ids'].tolist()))
            feats.append(np.asarray(data['features'], dtype=np.float32))
    if not ids:
        raise ValueError(f'no feature chunks in {directory}')
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate frame ids in the feature cache')
    return ids, np.vstack(feats)


def observation_rows(frame_ids: list[str]) -> list[dict]:
    """Rows for frames that are observable but not label-eligible: id -> video and frame index."""
    rows = []
    for fid in frame_ids:
        parts = fid.split('/')
        rows.append(dict(frame_id=fid, split='train', video_id=parts[-2], group_id=parts[-2],
                         frame_index=int(parts[-1])))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', type=Path, required=True, help='label-eligible training pool (JSONL)')
    ap.add_argument('--cache', type=Path, required=True, help='chunked DINOv2 cache of all observable train frames')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--strategy', choices=STRATEGIES, required=True)
    ap.add_argument('--budgets', type=int, nargs='+', default=[500, 1500])
    ap.add_argument('--initial-selection', type=Path, help='variant JSON whose frame ids form the shared L0')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-gap-frames', type=float, default=5)
    ap.add_argument('--dim', type=int, default=24)
    ap.add_argument('--lags', type=int, default=4)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--max-gap-frames', type=float, default=7.5)
    ap.add_argument('--neighbourhood-frames', type=float, default=25)
    ap.add_argument('--veto-quantile', type=float, default=.90)
    ap.add_argument('--coverage-weight', type=float, default=.60)
    ap.add_argument('--auxiliary-weight', type=float, default=.40)
    args = ap.parse_args(argv)
    started = time.perf_counter()
    budgets = sorted(set(args.budgets))
    pool = [r for r in read_jsonl(args.pool) if r.get('split', 'train') == 'train']
    cache_ids, cache_x = load_cache(args.cache)
    position = {fid: i for i, fid in enumerate(cache_ids)}
    missing = [r['frame_id'] for r in pool if r['frame_id'] not in position]
    if missing:
        raise ValueError(f'feature cache misses {len(missing)} pool frames: {missing[:3]}')
    group_of = {r['video_id']: r['group_id'] for r in pool}
    observed = observation_rows(cache_ids)
    for r in observed:
        r['group_id'] = group_of.get(r['video_id'], r['video_id'])
    model = LatentLangevin(observed, cache_x, seed=args.seed, dim=args.dim, lags=args.lags, folds=args.folds,
                           max_gap_frames=args.max_gap_frames, neighbourhood_frames=args.neighbourhood_frames)
    pool_x = cache_x[[position[r['frame_id']] for r in pool]]
    surprise = {observed[i]['frame_id']: float(e) for i, e in zip(model.target, model.err_pool) if np.isfinite(e)}
    eligible_rows, eligible_x = pool, pool_x
    vetoed = 0
    if args.strategy.startswith('langevin_veto'):
        limit = float(np.quantile(list(surprise.values()), args.veto_quantile))
        keep = np.array([surprise.get(r['frame_id'], -np.inf) <= limit for r in pool])
        vetoed = int((~keep).sum())
        eligible_rows, eligible_x = [r for r, k in zip(pool, keep) if k], pool_x[keep]
    base = {'langevin_reducible': 'coverage_reducible', 'langevin_surprise': 'coverage_residual',
            'langevin_veto_uniform': 'uniform_time', 'langevin_veto_coverage': 'coverage'}[args.strategy]
    eligible_ids = {r['frame_id'] for r in eligible_rows}
    if args.initial_selection:
        raw = json.loads(args.initial_selection.read_text(encoding='utf-8'))
        selected = [f for f in map(str, raw.get('selected_frame_ids', raw.get('frame_ids', [])))]
        if len(selected) != budgets[0]:
            raise ValueError('initial selection must equal the first budget')
    else:
        selected, _ = select_batch(eligible_rows, eligible_x, batch_size=budgets[0], strategy='uniform_time',
                                   seed=args.seed, min_gap_frames=args.min_gap_frames)
    args.out.mkdir(parents=True, exist_ok=True)
    rounds = []
    for number, budget in enumerate(budgets, 1):
        need = budget - len(selected)
        audit = None
        if need > 0:
            scores = None
            if base == 'coverage_reducible':
                scores = model.scores(selected)
            elif base == 'coverage_residual':
                scores = {fid: dict(residual_mahalanobis=value) for fid, value in surprise.items()}
            batch, audit = select_batch(eligible_rows, eligible_x, batch_size=need, strategy=base,
                                        seed=args.seed + number - 1,
                                        selected_ids=[f for f in selected if f in eligible_ids], scores=scores,
                                        min_gap_frames=args.min_gap_frames, coverage_weight=args.coverage_weight,
                                        auxiliary_weight=args.auxiliary_weight)
            if audit is not None:
                audit.pop('trace', None)
            selected.extend(batch)
        variant_id = f'tracking_{args.strategy}_b{budget}_sel{args.seed}'
        variant = dict(schema_version=1, variant_id=variant_id, run_id=variant_id, strategy=args.strategy,
                       selection_mode='active_latent_only', seed=args.seed, budget=budget, round=number,
                       selected_frame_ids=list(selected), frame_ids=list(selected),
                       pool_sha256=hashlib.sha256(args.pool.read_bytes()).hexdigest(),
                       feature_cache=str(args.cache.name), labels_used_for_ranking=False,
                       detector_in_the_loop=False, langevin=model.summary(), vetoed_pool_frames=vetoed,
                       selection_audit=audit, config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})
        target = args.out / f'{variant_id}.json'
        tmp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        tmp.write_text(json.dumps(variant, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(target)
        rounds.append(dict(budget=budget, path=str(target), n_selected=len(selected)))
    print(json.dumps(dict(strategy=args.strategy, rounds=rounds, langevin=model.summary(), vetoed=vetoed,
                          seconds=round(time.perf_counter() - started, 1)), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
