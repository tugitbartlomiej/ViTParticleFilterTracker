#!/usr/bin/env python3
"""What does coverage in Koopman coordinates (arm K4 of document 39) actually add?

No training. Three questions are answered on the unlabelled pool and the finished arm lists:

1. WHAT THE COORDINATES ARE. For each Koopman coordinate, and for the same number of leading
   principal directions of the same lifted latent: how much of its variance lies BETWEEN videos,
   how well the tool state (position, size) and the time inside the video explain it
   (cross-fitted by video), and how slowly it changes from frame to frame.
2. WHAT THE SELECTION LOOKS LIKE. For every arm: evenness over videos, coverage of the tool
   position grid, of tool size and of time inside the video, and the covering radius in DINOv2
   appearance space and in Koopman space (mean distance of a pool frame to its nearest chosen frame).
3. IS IT KOOPMAN OR JUST LOW DIMENSION. Offline control selections with the same production
   `select_batch`: coverage in the 24-dim projected latent, in 16 principal directions and in 16
   random directions of the lifted latent. Reported: overlap with K4 and the same descriptors.
   Whether these controls TRAIN as well as K4 needs detector runs; this file only says how
   different they are.

Ground-truth boxes are used for describing selections only, never for selecting.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.koopman_cover_analysis \
        --pool .../pool_manifest.jsonl --cache .../cache/dinov2/<id> \
        --arms SmallViTAblation/curated/selection_matrix_stage1 --out .../REPORT_koopman_cover.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..active_learning.acquisition import select_batch
from ..latent_dynamics.gle import _ridge_fit_predict
from .latent_langevin import LatentLangevin
from .select import load_cache, observation_rows, read_jsonl
from .selection_matrix import koopman_coordinates


def r2_by_video(x: np.ndarray, y: np.ndarray, videos: np.ndarray, folds: int = 4) -> float:
    names = np.array(sorted(set(videos.tolist())))
    fold_of = {v: i % folds for i, v in enumerate(np.random.default_rng(0).permutation(names))}
    fold = np.array([fold_of[v] for v in videos])
    sse = sst = 0.0
    for f in range(folds):
        fit, new = fold != f, fold == f
        p = _ridge_fit_predict(x[fit], y[fit], x[new], 1e-2)
        sse += ((p - y[new]) ** 2).sum()
        sst += ((y[new] - y[fit].mean(0)) ** 2).sum()
    return float(1 - sse / max(sst, 1e-12))


def describe_coordinates(coords: np.ndarray, state: np.ndarray, rel_time: np.ndarray, videos: np.ndarray,
                         follows: np.ndarray) -> dict:
    between, by_state, by_time, persistence = [], [], [], []
    ok = np.isfinite(state).all(1)
    for k in range(coords.shape[1]):
        c = coords[:, k]
        means = {v: c[videos == v].mean() for v in set(videos.tolist())}
        between.append(float(np.var([means[v] for v in videos]) / max(c.var(), 1e-12)))
        by_state.append(r2_by_video(state[ok], c[ok, None], videos[ok]))
        by_time.append(r2_by_video(np.column_stack([rel_time, rel_time ** 2])[ok], c[ok, None], videos[ok]))
        persistence.append(float(np.corrcoef(c[follows[:, 0]], c[follows[:, 1]])[0, 1]))
    return dict(variance_between_videos=float(np.mean(between)), r2_from_tool_state=float(np.mean(by_state)),
                r2_from_time_in_video=float(np.mean(by_time)), frame_to_frame_correlation=float(np.mean(persistence)),
                per_coordinate=dict(between=between, tool_state=by_state, time=by_time, persistence=persistence))


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def radius(features: np.ndarray, chosen: np.ndarray, block: int = 2000) -> float:
    ref = features[chosen]
    ref_sq = (ref ** 2).sum(1)
    out = []
    for s in range(0, len(features), block):
        x = features[s:s + block]
        d = (x ** 2).sum(1)[:, None] - 2 * x @ ref.T + ref_sq[None]
        out.append(np.sqrt(np.maximum(d.min(1), 0)))
    return float(np.concatenate(out).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--arms', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-controls', action='store_true')
    args = ap.parse_args(argv)
    started = time.perf_counter()
    pool = [r for r in read_jsonl(args.pool) if r.get('split', 'train') == 'train']
    cache_ids, cache_x = load_cache(args.cache)
    position = {f: i for i, f in enumerate(cache_ids)}
    rows_of_pool = np.array([position[r['frame_id']] for r in pool])
    observed = observation_rows(cache_ids)
    group_of = {r['video_id']: r['group_id'] for r in pool}
    for r in observed:
        r['group_id'] = group_of.get(r['video_id'], r['video_id'])
    model = LatentLangevin(observed, cache_x, seed=args.seed, dim=24, lags=8, folds=3, max_gap_frames=7.5)
    z, w = model.z, model.windows
    lifted = np.hstack([z, z])
    lifted[w[:, -1]] = np.hstack([z[w[:, -1]], z[w[:, -2]]])
    koopman = koopman_coordinates(model, 8)
    centred = lifted - lifted.mean(0)
    _, _, vt = np.linalg.svd(centred[::5], full_matrices=False)
    pca16 = centred @ vt[:koopman.shape[1]].T
    random16 = centred @ np.random.default_rng(args.seed).normal(size=(lifted.shape[1], koopman.shape[1]))

    def standard(a):
        a = a[rows_of_pool]
        return (a - a.mean(0)) / np.maximum(a.std(0), 1e-9)

    spaces = dict(koopman=standard(koopman), pca16=standard(pca16), random16=standard(random16), z24=standard(z))
    dino = cache_x[rows_of_pool].astype(np.float64)
    dino = dino / np.maximum(np.linalg.norm(dino, axis=1, keepdims=True), 1e-9)
    # `select_batch` puts every embedding row on the unit sphere, so coverage there is coverage of DIRECTIONS.
    # The radius is measured in the same geometry.
    koopman_unit = spaces['koopman'] / np.maximum(np.linalg.norm(spaces['koopman'], axis=1, keepdims=True), 1e-9)

    videos = np.array([r['video_id'] for r in pool])
    frame = np.array([float(r['frame_index']) for r in pool])
    rel_time = np.zeros(len(pool))
    for v in set(videos.tolist()):
        m = videos == v
        rel_time[m] = (frame[m] - frame[m].min()) / max(frame[m].max() - frame[m].min(), 1.0)
    state = np.full((len(pool), 4), np.nan)
    for i, r in enumerate(pool):
        if r.get('annotations'):
            x, y, bw, bh = r['annotations'][0]['bbox']
            state[i] = [(x + bw / 2) / r['native_width'], (y + bh / 2) / r['native_height'],
                        np.log(max(bw, 1) / r['native_width']), np.log(max(bh, 1) / r['native_height'])]
    order = np.lexsort((frame, videos))
    pairs = np.array([(a, b) for a, b in zip(order[:-1], order[1:]) if videos[a] == videos[b] and frame[b] - frame[a] <= 7.5])
    report = dict(pool_frames=len(pool), koopman_coordinates=int(koopman.shape[1]), coordinates={}, arms={}, controls={})
    for name in ('koopman', 'pca16', 'random16'):
        report['coordinates'][name] = describe_coordinates(spaces[name], state, rel_time, videos, pairs)
        c = report['coordinates'][name]
        print(f'[coords] {name:9s} miedzy filmami {c["variance_between_videos"]:.2f}  ze stanu narzedzia {c["r2_from_tool_state"]:.2f}  '
              f'z czasu w filmie {c["r2_from_time_in_video"]:.2f}  korelacja klatka-klatka {c["frame_to_frame_correlation"]:.2f}', flush=True)

    index = {r['frame_id']: i for i, r in enumerate(pool)}

    def describe_selection(ids):
        chosen = np.array([index[f] for f in ids])
        per_video = np.array([np.sum(videos[chosen] == v) for v in sorted(set(videos.tolist()))], dtype=float)
        s = state[chosen]
        s = s[np.isfinite(s).all(1)]
        grid, _, _ = np.histogram2d(s[:, 0], s[:, 1], bins=[8, 6], range=[[0, 1], [0, 1]])
        size_hist, _ = np.histogram(s[:, 2], bins=10, range=(np.nanpercentile(state[:, 2], 1), np.nanpercentile(state[:, 2], 99)))
        time_hist, _ = np.histogram(rel_time[chosen], bins=10, range=(0, 1))
        return dict(video_entropy=entropy(per_video), videos=int((per_video > 0).sum()), position_entropy=entropy(grid.ravel()),
                    size_entropy=entropy(size_hist.astype(float)), time_entropy=entropy(time_hist.astype(float)),
                    radius_dino=radius(dino, chosen), radius_koopman=radius(koopman_unit, chosen))

    l0 = None
    for f in sorted(args.arms.glob('matrix_*.json')):
        body = json.loads(f.read_text(encoding='utf-8'))
        if 'strategy' not in body:                      # MATRIX_MANIFEST.json matches the glob on Windows
            continue
        name = body['strategy']
        report['arms'][name] = describe_selection(body['frame_ids'])
        if name == 'K4_koopman_cover':
            k4 = body['frame_ids']
        if name == 'B2_uniform_time':
            l0 = body['frame_ids'][:500]
        a = report['arms'][name]
        print(f'[arm] {name:18s} filmy {a["video_entropy"]:.3f}  polozenie {a["position_entropy"]:.3f}  rozmiar {a["size_entropy"]:.3f}  '
              f'czas {a["time_entropy"]:.3f}  promien DINO {a["radius_dino"]:.3f}  promien Koopman {a["radius_koopman"]:.3f}', flush=True)

    if not args.skip_controls:
        need = len(k4) - len(l0)
        for name in ('koopman', 'z24', 'pca16', 'random16'):
            added, _ = select_batch(pool, spaces[name], batch_size=need, strategy='coverage', seed=args.seed, selected_ids=l0,
                                    min_gap_frames=5)
            ids = l0 + list(added)
            d = describe_selection(ids)
            d['overlap_with_K4'] = len(set(added) & set(k4[500:])) / need
            report['controls'][name] = d
            print(f'[control] {name:9s} wspolne z K4 {d["overlap_with_K4"]:.2f}  filmy {d["video_entropy"]:.3f}  polozenie {d["position_entropy"]:.3f}  '
                  f'promien DINO {d["radius_dino"]:.3f}  promien Koopman {d["radius_koopman"]:.3f}  ({time.perf_counter() - started:.0f}s)', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
