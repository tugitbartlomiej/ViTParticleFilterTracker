"""Epoch diagnostics on synthetic particles whose ground truth is known."""
from __future__ import annotations

import numpy as np

from SmallViTAblation.pipeline.koopman_langevin import epoch_dynamics as ed


def _world(noise: float, seed: int = 0, clips: int = 8, frames: int = 160, dim: int = 12):
    """A one-frame encoder: the latent is a linear image of position and size plus noise."""
    rng = np.random.default_rng(seed)
    mix = np.random.default_rng(99).normal(size=(4, dim))
    tracks, ids, z = {}, [], []
    for c in range(clips):
        pos = np.zeros((frames, 2))
        vel = rng.normal(scale=2.0, size=2)
        pos[0] = rng.uniform(100, 500, size=2)
        for t in range(1, frames):
            vel = .95 * vel + rng.normal(scale=.6, size=2)
            pos[t] = pos[t - 1] + vel
        wh = np.tile([200.0, 50.0], (frames, 1)) * np.exp(rng.normal(scale=.02, size=(frames, 2)).cumsum(0))
        box = np.hstack([pos - wh / 2, wh])
        name = f'C{c}'
        tracks[name] = dict(split='val', group_id=f'G{c}', stride=1, frame=list(range(1, frames + 1)),
                            box=box.tolist(), tip=(pos + [wh[0, 0] / 2, 0]).tolist())
        ids += [f'cataract_lmm/tracking/{name}/{t:08d}' for t in range(1, frames + 1)]
        truth = np.hstack([(pos - 300) / 100, np.log(wh) - 4])
        z.append(truth @ mix + rng.normal(scale=noise, size=(frames, dim)))
    return tracks, ids, np.vstack(z)


def test_velocity_lives_in_the_pair_not_in_one_frame():
    tracks, ids, z = _world(noise=.02)
    state = ed.physical_state(tracks, ids)
    out = ed.probe(z, state)
    assert out['position']['single'] > .9
    assert out['velocity']['single'] < .3
    assert out['velocity']['pair'] > out['velocity']['single'] + .3


def test_probe_rises_as_the_latent_sharpens():
    tracks, ids, blurry = _world(noise=10.0)
    _, _, sharp = _world(noise=.05)
    state = ed.physical_state(tracks, ids)
    assert ed.probe(sharp, state)['position']['single'] > ed.probe(blurry, state)['position']['single'] + .2


def test_langevin_beats_its_time_shuffled_floor():
    tracks, ids, z = _world(noise=.001)
    state = ed.physical_state(tracks, ids)
    summary = ed.langevin_summary(ed.pca(z, 4)[0], state, seed=0, max_gap=1.5)
    assert summary['skill_with_memory'] > summary['skill_permuted_floor'] + 10
    assert summary['skill_permuted_floor'] < 5


def test_rotation_of_the_latent_is_not_motion():
    _, _, z = _world(noise=.02)
    q = np.linalg.qr(np.random.default_rng(3).normal(size=(z.shape[1], z.shape[1])))[0]
    moved, cka = ed.procrustes_displacement(z, z @ q)
    assert moved.mean() < 1e-10 and cka > .999
    moved, _ = ed.procrustes_displacement(z, np.random.default_rng(4).normal(size=z.shape))
    assert moved.mean() > .5


def test_koopman_alignment_reports_both_arms():
    tracks, ids, z = _world(noise=.02)
    state = ed.physical_state(tracks, ids)
    out = ed.koopman_alignment(ed.pca(z, 8)[0], state)
    assert set(out['carried']['position']) == {'slow_modes', 'leading_pca'}
    assert np.isfinite(out['top_moduli']).all()
