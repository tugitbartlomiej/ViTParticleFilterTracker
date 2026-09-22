#!/usr/bin/env python3
"""Arms K5: frame selection in the IDENTITY-CODE space of the instrument (document 38, section 4j; document 39, 5f).

The identity code of a frame is the mean position of its instrument patches in the slow eigenfunctions of the
TRANSLATIONAL Koopman operator of the current detector (`identity_code_pool.py`): label-free, from the detector's
own tokens, unlike K4 (whole-frame DINOv2 latent = operation fingerprint). Frames where the L0 detector has no
confident box carry no code.

  K5_identity_cover              Euclidean k-centre (farthest point first) in the standardised code, starting from
                                 the L0 frames: rare identity states are labelled first
  K5b_identity_cover_plus_unseen as K5, but a fixed share of the added budget goes uniformly in time to frames
                                 WITHOUT a code ("the network sees no instrument here" is a state worth covering)
  K5c_closure_near               the author's literal idea: frames whose code lies CLOSEST to the prototype, i.e.
                                 where the identity is encoded "correctly"; quota per video proportional to its
                                 share of the pool, spaced in time (otherwise the best-closed frames are one clip)
  K5d_closure_far                mirror of K5c: frames whose code lies FARTHEST from the prototype

Metric note. `select_batch(strategy='coverage')` is a COSINE k-centre and assumes unit vectors; stage 1 fed it
standardised (not unit) coordinates for K4, so K4 was an anti-alignment rule rather than a k-centre. The k-centre
here is Euclidean on the standardised code, and the manifest records for EVERY arm (also the stage-1 ones passed
with --compare) how far a random pool frame lies from its nearest labelled frame in the code space (median, p90,
max), so the coverage claim is measured, not assumed. Labels are never read. Same L0, budget and spacing rule as
the stage-1 matrix.

    py -3.11 -m SmallViTAblation.pipeline.koopman_langevin.identity_cover_arm \
        --pool .../pool_manifest.jsonl --code KOD_TOZSAMOSCI_pool_b0500.npz \
        --initial-selection .../tracking_uniform_time_b500_s42.json --budget 1500 --out curated/selection_matrix_stage1 \
        --compare curated/selection_matrix_stage1/matrix_K4_koopman_cover_b1500_sel42.json ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..active_learning.acquisition import select_batch
from .select import read_jsonl


def read_variant(path: Path) -> list[str]:
    body = json.loads(path.read_text(encoding='utf-8'))
    return [str(f) for f in body.get('selected_frame_ids', body.get('frame_ids'))]


def _spaced(times: dict[str, list[float]], video: str, t: float, gap: float) -> bool:
    return not times.get(video) or min(abs(t - u) for u in times[video]) >= gap


def kcentre(x: np.ndarray, rows: list[dict], start: list[int], need: int, gap: float) -> list[int]:
    """Greedy Euclidean k-centre with the temporal spacing rule; `start` are indices already labelled."""
    video = [str(r['video_id']) for r in rows]
    frame = [float(r['frame_index']) for r in rows]
    times: dict[str, list[float]] = defaultdict(list)
    dmin = np.full(len(rows), np.inf)
    for i in start:
        times[video[i]].append(frame[i])
        dmin = np.minimum(dmin, np.linalg.norm(x - x[i], axis=1))
    if not start:                                              # deterministic first point: nearest to the mean
        dmin = -np.linalg.norm(x - x.mean(0), axis=1)
    taken = set(start)
    picked: list[int] = []
    for _ in range(need):
        for j in np.argsort(-dmin):
            j = int(j)
            if j not in taken and _spaced(times, video[j], frame[j], gap):
                break
        else:
            break
        picked.append(j)
        taken.add(j)
        times[video[j]].append(frame[j])
        dmin = np.minimum(dmin, np.linalg.norm(x - x[j], axis=1))
    return picked


def by_closure(closure: np.ndarray, rows: list[dict], start: list[int], need: int, gap: float, farthest: bool) -> list[int]:
    """Per-video quota proportional to the video's share of the pool; inside a video the smallest (or largest) closure first."""
    video = [str(r['video_id']) for r in rows]
    frame = [float(r['frame_index']) for r in rows]
    members: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(video):
        members[v].append(i)
    raw = {v: need * len(m) / len(rows) for v, m in members.items()}
    quota = {v: int(np.floor(q)) for v, q in raw.items()}
    for v in sorted(raw, key=lambda v: raw[v] - quota[v], reverse=True)[:need - sum(quota.values())]:
        quota[v] += 1                                            # largest remainder
    times: dict[str, list[float]] = defaultdict(list)
    for i in start:
        times[video[i]].append(frame[i])
    taken = set(start)
    picked: list[int] = []
    for v, m in members.items():
        got = 0
        for i in sorted(m, key=lambda i: closure[i], reverse=farthest):
            if got >= quota[v]:
                break
            if i not in taken and _spaced(times, v, frame[i], gap):
                picked.append(i); taken.add(i); times[v].append(frame[i]); got += 1
    for i in sorted(range(len(rows)), key=lambda i: closure[i], reverse=farthest):     # top-up if spacing blocked a video
        if len(picked) >= need:
            break
        if i not in taken and _spaced(times, video[i], frame[i], gap):
            picked.append(i); taken.add(i); times[video[i]].append(frame[i])
    return picked


def cover_stats(x_all: np.ndarray, sample: np.ndarray, labelled: np.ndarray) -> dict:
    """How far a random pool frame (rows `sample`) lies from its nearest labelled frame, in the standardised code."""
    if len(labelled) == 0:
        return dict(median=None, p90=None, max=None)
    d = np.sqrt(((x_all[sample][:, None, :] - x_all[labelled][None, :, :]) ** 2).sum(-1)).min(1)
    return dict(median=float(np.median(d)), p90=float(np.percentile(d, 90)), max=float(d.max()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--code', type=Path, required=True)
    ap.add_argument('--initial-selection', type=Path, required=True)
    ap.add_argument('--budget', type=int, default=1500)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-gap-frames', type=float, default=5)
    ap.add_argument('--no-code-share', type=float, default=.2, help='K5b: share of the added budget spent on frames without a code')
    ap.add_argument('--compare', type=Path, nargs='*', default=[], help='other variant files: overlap and the same coverage statistics')
    args = ap.parse_args(argv)
    pool = [r for r in read_jsonl(args.pool) if r.get('split', 'train') == 'train']
    with np.load(args.code, allow_pickle=False) as data:
        code_of = {str(f): (c, float(cl)) for f, c, cl in zip(data['frame_ids'].tolist(), data['code'], data['closure'])}
        prototype, lam = data['prototype'], data['lambda_slow']
    l0 = read_variant(args.initial_selection)
    need = args.budget - len(l0)
    has = [r for r in pool if r['frame_id'] in code_of and np.isfinite(code_of[r['frame_id']][0]).all()]
    none = [r for r in pool if r['frame_id'] not in code_of or not np.isfinite(code_of[r['frame_id']][0]).all()]
    x = np.array([code_of[r['frame_id']][0] for r in has], dtype=np.float64)
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
    closure = np.array([code_of[r['frame_id']][1] for r in has])
    row_of = {r['frame_id']: i for i, r in enumerate(has)}
    l0_rows = [row_of[f] for f in l0 if f in row_of]
    print(f'pula {len(pool)}: z kodem tozsamosci {len(has)}, bez kodu (L0 nie widzi narzedzia) {len(none)}; L0 z kodem {len(l0_rows)} z {len(l0)}', flush=True)

    arms: dict[str, list[str]] = {}
    arms['K5_identity_cover'] = [has[i]['frame_id'] for i in kcentre(x, has, l0_rows, need, args.min_gap_frames)]
    n_none = min(int(round(args.no_code_share * need)), len(none))
    if n_none:
        none_ids = {r['frame_id'] for r in none}
        part, _ = select_batch(none, None, batch_size=n_none, strategy='uniform_time', seed=args.seed,
                               selected_ids=[f for f in l0 if f in none_ids], min_gap_frames=args.min_gap_frames)
        rest = [has[i]['frame_id'] for i in kcentre(x, has, l0_rows, need - len(part), args.min_gap_frames)]
        arms['K5b_identity_cover_plus_unseen'] = list(part) + rest
    arms['K5c_closure_near'] = [has[i]['frame_id'] for i in by_closure(closure, has, l0_rows, need, args.min_gap_frames, farthest=False)]
    arms['K5d_closure_far'] = [has[i]['frame_id'] for i in by_closure(closure, has, l0_rows, need, args.min_gap_frames, farthest=True)]

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(len(has), min(2000, len(has)), replace=False)
    others = {}
    for path in args.compare:
        body = json.loads(path.read_text(encoding='utf-8'))
        others[body.get('strategy', path.stem)] = read_variant(path)[len(l0):]

    def describe(added: list[str]) -> dict:
        rows = [row_of[f] for f in added if f in row_of]
        cl = closure[rows]
        return dict(videos=len({f.split('/')[-2] for f in added}), without_code=len(added) - len(rows),
                    closure_median=float(np.median(cl)) if len(cl) else None, closure_p90=float(np.percentile(cl, 90)) if len(cl) else None,
                    cover=cover_stats(x, sample, np.array(l0_rows + rows, dtype=int)))

    manifest = dict(budget=args.budget, l0=len(l0), added=need, seed=args.seed, code=str(args.code), frames_with_code=len(has),
                    frames_without_code=len(none), lambda_slow=[float(v) for v in lam], prototype=[float(v) for v in prototype],
                    pool_closure_median=float(np.median(closure)), cover_l0_only=cover_stats(x, sample, np.array(l0_rows, dtype=int)),
                    arms={}, compare={})
    args.out.mkdir(parents=True, exist_ok=True)

    def line(name: str, d: dict, extra: str = '') -> None:
        c = d['cover']
        print(f'{name:32s} filmy {d["videos"]:3d}  bez kodu {d["without_code"]:4d}  domkniecie mediana {d["closure_median"]:.3f}  '
              f'pokrycie kodu: mediana {c["median"]:.3f} p90 {c["p90"]:.3f} max {c["max"]:.3f}  {extra}', flush=True)

    for name, added in arms.items():
        frames = l0 + added
        assert len(frames) == args.budget == len(set(frames)), (name, len(frames), len(set(frames)))
        variant_id = f'matrix_{name}_b{args.budget}_sel{args.seed}'
        (args.out / f'{variant_id}.json').write_text(json.dumps(dict(
            schema_version=1, variant_id=variant_id, run_id=variant_id, strategy=name, seed=args.seed, budget=args.budget,
            frame_ids=frames, selected_frame_ids=frames, labels_used_for_ranking=False)), encoding='utf-8')
        d = describe(added)
        d.update(file=f'{variant_id}.json', sha256=hashlib.sha256('\n'.join(frames).encode()).hexdigest(),
                 overlap={k: len(set(v) & set(added)) / need for k, v in others.items()})
        manifest['arms'][name] = d
        line(name, d, '  '.join(f'{k}:{v:.2f}' for k, v in d['overlap'].items()))
    for name, added in others.items():
        manifest['compare'][name] = describe(added)
        line(name, manifest['compare'][name])
    c = manifest['cover_l0_only']
    print(f'{"L0 alone":32s} pokrycie kodu: mediana {c["median"]:.3f} p90 {c["p90"]:.3f} max {c["max"]:.3f}')
    (args.out / 'K5_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
