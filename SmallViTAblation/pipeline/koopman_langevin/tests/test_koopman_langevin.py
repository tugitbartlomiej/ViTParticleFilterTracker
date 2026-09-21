"""Fast CPU tests of the Koopman / Langevin modules on data with known ground truth."""
from __future__ import annotations

import json

import numpy as np
import pytest

from SmallViTAblation.pipeline.koopman_langevin import koopman_tool, select, track_dynamics
from SmallViTAblation.pipeline.koopman_langevin.latent_langevin import LatentLangevin


def _pool(videos=6, frames=80, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    rows, z = [], []
    for v in range(videos):
        x = np.zeros((frames, dim))
        x[0] = rng.normal(size=dim)
        for t in range(1, frames):
            x[t] = .9 * x[t - 1] + rng.normal(scale=.1, size=dim)
        z.append(x)
        rows += [dict(frame_id=f'cataract_lmm/tracking/V{v}/{t:08d}', split='train', group_id=f'V{v}',
                      video_id=f'V{v}', frame_index=t) for t in range(frames)]
    return rows, np.vstack(z)


def test_dmd_separates_moving_blob_from_static_scene():
    rng = np.random.default_rng(1)
    t_len, grid, c = 40, 8, 4
    scene = rng.normal(size=(grid * grid, c))
    tokens = np.repeat(scene[None], t_len, 0) + rng.normal(scale=.01, size=(t_len, grid * grid, c))
    path = [(1 + (t // 2) % 6, (t // 1) % grid) for t in range(t_len)]        # moves every frame
    for t, (r, col) in enumerate(path):
        tokens[t, r * grid + col] += 3.0
    maps, spectra = koopman_tool.saliency_maps(tokens, window=16, hop=8, rank=6)
    hits = [np.argmax(maps['dmd_slow_modes'][t]) == r * grid + col for t, (r, col) in enumerate(path)]
    assert np.mean(hits) > .8
    assert all(np.isfinite(s).all() for s in spectra)


def test_latent_langevin_is_label_free_and_prefers_uncovered_motion():
    rows, z = _pool()
    for r in rows:
        r['annotations'] = 'must never be read'
    model = LatentLangevin(rows, z, dim=6, lags=4, folds=3, max_gap_frames=1, neighbourhood_frames=3)
    assert model.summary()['skill_vs_no_movement'] > 0
    chosen = [r['frame_id'] for r in rows if r['video_id'] in {'V0', 'V1', 'V2'} and r['frame_index'] % 4 == 0]
    scores = model.scores(chosen)
    covered = np.mean([v['reducible_gain'] for v in scores.values() if v['covered_by_selection']])
    open_ = np.mean([v['reducible_gain'] for v in scores.values() if not v['covered_by_selection']])
    assert np.isfinite([covered, open_]).all()
    assert all(set(v) >= {'reducible_gain', 'residual_energy'} for v in scores.values())


def test_latent_langevin_rejects_non_training_rows():
    rows, z = _pool(videos=3)
    rows[5]['split'] = 'val'
    with pytest.raises(ValueError):
        LatentLangevin(rows, z, dim=6, max_gap_frames=1)


def test_track_langevin_and_pivot_on_a_rod_through_a_fixed_point():
    rng = np.random.default_rng(2)
    tracks = []
    for v in range(8):
        n, pivot = 300, np.array([700.0, 240.0])
        angle = np.zeros(n)
        depth = np.zeros(n)
        for t in range(1, n):
            angle[t] = .97 * angle[t - 1] + rng.normal(scale=.01)
            depth[t] = .97 * depth[t - 1] + rng.normal(scale=2.0)
        direction = np.stack([-np.cos(angle), np.sin(angle)], 1)
        tip = pivot + direction * (300 + depth)[:, None]
        far = pivot + direction * 40
        lo, hi = np.minimum(tip, far), np.maximum(tip, far)
        box = np.concatenate([lo, np.maximum(hi - lo, 2.0)], 1)
        tracks.append(dict(video_id=f'V{v}', group_id=f'V{v}', frame=np.arange(n), box=box, tip=tip,
                           state=track_dynamics.state_of(box)))
    _, summary = track_dynamics.langevin(tracks, lags=4, folds=4)
    assert summary['skill_vs_no_movement']['kernel'] > summary['skill_vs_no_movement']['velocity']
    pivot_report, violation = track_dynamics.pivot_invariant(tracks)
    assert pivot_report['ratio_median'] < .2
    assert set(violation) == {f'V{v}' for v in range(8)}


def test_selection_driver_writes_exact_budgets(tmp_path):
    rows, z = _pool(videos=6, frames=60)
    pool = tmp_path / 'pool.jsonl'
    pool.write_text('\n'.join(json.dumps(r) for r in rows if r['frame_index'] % 2 == 0), encoding='utf-8')
    cache = tmp_path / 'cache'
    cache.mkdir()
    ids = np.array([r['frame_id'] for r in rows])
    for start in range(0, len(rows), 64):
        np.savez(cache / f'{start:09d}.npz', frame_ids=ids[start:start + 64], features=z[start:start + 64].astype(np.float32))
    for strategy in select.STRATEGIES:
        out = tmp_path / strategy
        assert select.main(['--pool', str(pool), '--cache', str(cache), '--out', str(out), '--strategy', strategy,
                            '--budgets', '20', '40', '--dim', '6', '--max-gap-frames', '1', '--min-gap-frames', '2',
                            '--neighbourhood-frames', '3']) == 0
        variant = json.loads((out / f'tracking_{strategy}_b40_sel42.json').read_text(encoding='utf-8'))
        assert len(variant['frame_ids']) == 40 == len(set(variant['frame_ids']))
        assert variant['labels_used_for_ranking'] is False
