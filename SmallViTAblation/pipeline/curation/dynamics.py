"""Compatibility adapter for the shared latent-dynamics implementation.

The diagnostics are finite-model, group-cross-fitted prediction contrasts.
They are not an exact Mori--Zwanzig kernel, conditional mutual information or
physical Langevin noise. Scores are assigned to the predicted target frame.
"""
from __future__ import annotations

import numpy as np

from ..latent_dynamics.gle import diagnose_gle


def project(features, dim: int, seed: int):
    """Deterministic random projection retained for curation coverage."""
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("features must be a non-empty matrix")
    if dim <= 0:
        raise ValueError("projection dimension must be positive")
    if values.shape[1] <= dim:
        return values.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(values.shape[1], dim)).astype(np.float32) / np.sqrt(dim)
    return values @ matrix


def diagnose(rows, features, *, seed=42, dim=24, lags=3, folds=3,
             max_gap_frames=8, alpha=.01, max_fit_windows=20000):
    """Return legacy array names backed by the audited shared GLE diagnostic."""
    score_rows, summary, _ = diagnose_gle(
        rows, features, seed=seed, dim=dim, lags=lags, folds=folds,
        max_gap_frames=max_gap_frames, alpha=alpha,
        max_fit_windows=max_fit_windows)

    def column(name: str) -> np.ndarray:
        return np.asarray([
            np.nan if row.get(name) is None else float(row[name])
            for row in score_rows
        ], dtype=np.float64)

    errors = {
        "persistence": column("persistence_error"),
        "linear": column("markov_error"),
        "nonlinear": column("nonlinear_error"),
        "history": column("history_error"),
        "history_gain": column("history_gain"),
        "nonlinearity_gain": column("nonlinearity_gain"),
        "residual": column("residual_mahalanobis"),
        "residual_energy": column("residual_energy"),
    }
    summary = {**summary, "compatibility_adapter": "pipeline.curation.dynamics"}
    return errors, summary
