"""Extended Langevin equation on the frame latent and the reducible-error frame score.

Equation (same structure as in the grant project, without external forcing, because a
surgical video has no input known in advance):

    z_{t+1} = z_t + D(z_t, v_t, a_t) + JP(z_{t-1}, z_{t-2}, ...) + e_{t+1}

Why a new score. The residual e of any predictor h built from the history splits into
    |z - h|^2 = |z - m|^2 (surprise, not removable) + |m - h|^2 (removable)
where m is the best prediction the history allows (proved in Lean in the grant repository,
`SpmProofs/DiagnostykaInnowacji.lean`, theorems `rozklad_bledu` and
`roznica_strat_bez_zaskoczenia`). The size of e cannot tell the two parts apart, so ranking
frames by residual size buys blurred, occluded and cut frames.

Two predictors see the same history:
  h_pool   full model cross-fitted on the whole unlabelled pool (other operation groups)
  h_known  same model fitted only on transitions next to frames that are already selected

The theorem says the DIFFERENCE OF THEIR EXPECTED errors contains no surprise. Per frame this
is not enough: |z-h_known|^2 - |z-h_pool|^2 = |m-h_known|^2 - |m-h_pool|^2 + 2<e, h_pool-h_known>,
and the last term is driven by the surprise e of that one frame. The first version of this score
used the error difference and the synthetic benchmark showed it still bought 44 percent noise
frames. The score therefore uses the distance between the two PREDICTIONS, which does not
involve the realised next frame at all, divided by the local noise level:

    reducible_gain(i) = |h_known(i) - h_pool(i)|^2 / (err_pool(i) + median err_pool)

A frame scores high when its motion is predictable in principle, but not from what the
selected set already covers. It scores near zero when nobody can predict it (surprise) or
when the selected set already explains it (redundant). The score depends on the selected
set, so it changes every round: this is an active criterion, not a one-off ranking.

Labels are never read. Only `split == 'train'` rows are accepted.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..latent_dynamics.gle import _folds, _project, _ridge_fit_predict, _windows


def _design(z: np.ndarray, windows: np.ndarray, rff_w: np.ndarray, rff_b: np.ndarray,
            history: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Features known at frame t for the step to t+1. windows[:, -1] is the target frame.

    history=False is the ablation arm: the current latent only, with the SAME number of random
    nonlinear features, so that a gain of the full model cannot be credited to flexibility.
    Velocity and acceleration already are history and are left out of that arm.
    """
    cur = z[windows[:, -2]]
    if not history:
        return np.column_stack([cur, np.cos(cur @ rff_w[:cur.shape[1]] + rff_b)]), z[windows[:, -1]] - cur
    prev = z[windows[:, -3]]
    v = cur - prev
    a = v - (prev - z[windows[:, -4]]) if windows.shape[1] >= 4 else np.zeros_like(v)
    drift = np.column_stack([cur, v, a, np.cos(np.column_stack([cur, v]) @ rff_w + rff_b)])
    past = (z[windows[:, :-3]] - cur[:, None, :]).reshape(len(windows), -1) if windows.shape[1] > 3 else np.zeros((len(windows), 0))
    return np.column_stack([drift, past]), z[windows[:, -1]] - cur


class LatentLangevin:
    """Cross-fitted pool model plus a cheap refit on the transitions the selected set covers."""

    ALPHAS = (.01, .1, 1.0, 10.0, 100.0)

    def _choose_alpha(self, fit: np.ndarray, groups: np.ndarray) -> float:
        """Ridge strength picked inside the fitting groups only (half of them fit, half score)."""
        names = sorted(set(groups[fit].tolist()))
        if len(names) < 2:
            return max(self.ALPHAS)
        first = set(names[::2])
        a = fit[np.isin(groups[fit], list(first))]
        b = fit[~np.isin(groups[fit], list(first))]
        errors = [float(((_ridge_fit_predict(self.x[a], self.y[a], self.x[b], alpha) - self.y[b]) ** 2).mean())
                  for alpha in self.ALPHAS]
        return float(self.ALPHAS[int(np.argmin(errors))])

    def __init__(self, rows: Sequence[dict], features: np.ndarray, *, seed: int = 42, dim: int = 24,
                 lags: int = 4, folds: int = 3, alpha: float | None = None, max_gap_frames: float = 8,
                 max_fit_windows: int = 40000, neighbourhood_frames: float = 10, history: bool = True):
        if any(r.get('split', 'train') != 'train' for r in rows):
            raise ValueError('latent dynamics may only consume the training pool')
        if lags < 3:
            raise ValueError('need at least three lags for velocity and acceleration')
        self.rows, self.neighbourhood = rows, float(neighbourhood_frames)
        z = _project(features, dim, seed)
        self.z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-8)
        self.windows = _windows(rows, lags, max_gap_frames)
        rng = np.random.default_rng(seed)
        d = self.z.shape[1]
        self.x, y = _design(self.z, self.windows, rng.normal(size=(2 * d, 2 * d)) / np.sqrt(2 * d),
                            rng.uniform(0, 2 * np.pi, size=2 * d), history=history)
        self.scale = np.maximum(y.std(0), 1e-8)
        self.y = y / self.scale
        self.target = self.windows[:, -1]
        groups = np.asarray([str(rows[i]['group_id']) for i in self.target])
        self.fold = _folds(groups, folds, seed)
        self.rng, self.max_fit = rng, max_fit_windows
        self.err_pool = np.full(len(self.target), np.nan)
        self.pred_pool = np.full_like(self.y, np.nan)
        self.err_stay = (self.y ** 2).mean(1)
        self.alphas: list[float] = []
        for f in sorted(set(self.fold.tolist())):
            new = np.flatnonzero(self.fold == f)
            fit = np.flatnonzero(self.fold != f)
            if len(fit) > max_fit_windows:
                fit = np.sort(rng.choice(fit, max_fit_windows, replace=False))
            chosen = alpha if alpha is not None else self._choose_alpha(fit, groups)
            self.alphas.append(chosen)
            p = _ridge_fit_predict(self.x[fit], self.y[fit], self.x[new], chosen)
            self.pred_pool[new] = p
            self.err_pool[new] = ((p - self.y[new]) ** 2).mean(1)
        self.alpha = float(np.median(self.alphas))
        self.video = np.asarray([str(rows[i]['video_id']) for i in self.target])
        self.time = np.asarray([float(rows[i]['frame_index']) for i in self.target])

    def summary(self) -> dict:
        skill = float(100 * (1 - np.nansum(self.err_pool) / self.err_stay.sum()))
        return dict(windows=int(len(self.target)), frames=len(self.rows), dim=int(self.z.shape[1]),
                    skill_vs_no_movement=skill, ridge_alpha_per_fold=self.alphas,
                    cross_fitted_by_group=True, labels_used=False)

    def known_mask(self, selected_ids: Sequence[str]) -> np.ndarray:
        """Transitions whose target lies within the neighbourhood of a selected frame of the same video."""
        by_video: dict[str, list[float]] = {}
        index = {str(r['frame_id']): r for r in self.rows}
        for fid in selected_ids:
            r = index[str(fid)]
            by_video.setdefault(str(r['video_id']), []).append(float(r['frame_index']))
        mask = np.zeros(len(self.target), bool)
        for video, times in by_video.items():
            where = np.flatnonzero(self.video == video)
            if len(where):
                t = np.sort(np.asarray(times))
                pos = np.searchsorted(t, self.time[where])
                left = np.abs(self.time[where] - t[np.clip(pos - 1, 0, len(t) - 1)])
                right = np.abs(self.time[where] - t[np.clip(pos, 0, len(t) - 1)])
                mask[where] = np.minimum(left, right) <= self.neighbourhood
        return mask

    def scores(self, selected_ids: Sequence[str]) -> dict[str, dict[str, float]]:
        """Per-frame score rows for `select_batch` (field names as in SCORE_FIELD)."""
        known = self.known_mask(selected_ids)
        if known.sum() < 5 * self.x.shape[1] ** .5:
            p = np.zeros_like(self.y)                         # nothing is known yet: "no movement" predictor
        else:
            fit = np.flatnonzero(known)
            if len(fit) > self.max_fit:
                fit = np.sort(self.rng.choice(fit, self.max_fit, replace=False))
            p = _ridge_fit_predict(self.x[fit], self.y[fit], self.x, max(self.alpha, 1.0 / len(fit) ** .5))
        err_known = ((p - self.y) ** 2).mean(1)
        disagreement = ((p - self.pred_pool) ** 2).mean(1)
        gain = disagreement / (self.err_pool + np.nanmedian(self.err_pool))
        out: dict[str, dict[str, float]] = {}
        for i, row_index in enumerate(self.target):
            if np.isfinite(gain[i]):
                out[str(self.rows[row_index]['frame_id'])] = dict(
                    reducible_gain=float(gain[i]), prediction_disagreement=float(disagreement[i]),
                    residual_energy=float(self.err_pool[i]),
                    known_error=float(err_known[i]), covered_by_selection=bool(known[i]))
        return out
