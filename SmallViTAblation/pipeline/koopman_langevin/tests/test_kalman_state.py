"""The Langevin state filter on a synthetic track with known velocity and bursty observations."""
from __future__ import annotations

import numpy as np

from SmallViTAblation.pipeline.koopman_langevin import kalman_state as ks


def _tracks(seed=0, n=12, t_len=200, noise=4.0):
    rng = np.random.default_rng(seed)
    v = np.zeros((n, t_len)); a = np.zeros((n, t_len)); p = np.zeros((n, t_len))
    for t in range(1, t_len):
        a[:, t] = .8 * a[:, t - 1] + rng.normal(scale=.3, size=n)
        v[:, t] = .9 * v[:, t - 1] + a[:, t]
        p[:, t] = p[:, t - 1] + v[:, t - 1]
    y = p + rng.normal(scale=noise, size=p.shape)
    burst = rng.random(p.shape) < .02
    y[burst] += rng.normal(scale=80.0, size=burst.sum())                 # detector jumps
    return y, v


def test_filter_recovers_velocity_better_than_a_difference():
    y, v = _tracks()
    theta = ks.fit(y[:8])
    filtered, innovation, variance, _, smoothed = ks.kalman(y[8:], theta, smooth=True)
    difference = np.diff(y[8:], axis=1, prepend=y[8:, :1])
    score = lambda est: np.corrcoef(est[:, 10:].ravel(), v[8:, 10:].ravel())[0, 1]
    assert score(filtered[:, :, 1]) > score(difference) + .2
    assert score(smoothed[:, :, 1]) > score(filtered[:, :, 1])
    assert np.isfinite(innovation).all() and (variance > 0).all()


def test_auc_orders_scores():
    score = np.array([.1, .2, .3, .9, .8])
    assert ks.auc(score, np.array([False, False, False, True, True])) == 1.0
    assert ks.auc(-score, np.array([False, False, False, True, True])) == 0.0
