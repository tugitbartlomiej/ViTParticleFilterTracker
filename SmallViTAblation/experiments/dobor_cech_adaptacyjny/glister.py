"""Chunked greedy first-order GLISTER-style selection on selected head gradients.

This is an explicit approximation, not the official full GLISTER algorithm.
The callback owns detector parameter restoration and evaluates the D gradient at
theta - learning_rate * mean(candidate_gradients[selected]). It must restore the
real model, optimizer state and buffers after every virtual evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def select_gradients(
    gradients: np.ndarray,
    reference_gradient: Callable[[Sequence[int]], np.ndarray],
    budget: int,
    *,
    rounds: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Select unique local candidate indices with at most ``rounds`` refreshes.

    Positive ``g_candidate dot g_D`` predicts a reduction in D loss after a
    gradient-descent update. The reference callback receives all indices chosen
    so far, including an empty tuple on its first call. Each refresh selects a
    chunk using that reference. Raw gradients are not normalized. Ties use the
    provided seed, reproducibly for the same candidate order and seed.
    """
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[1] == 0 or not np.isfinite(gradients).all():
        raise ValueError("gradients must be a finite [candidates, parameters] matrix")
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or not 0 <= budget <= len(gradients):
        raise ValueError("budget must be an integer between zero and candidate count")
    if isinstance(rounds, (bool, np.bool_)) or not isinstance(rounds, (int, np.integer)) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    if not callable(reference_gradient):
        raise ValueError("reference_gradient must be callable")
    tie_break = np.random.default_rng(seed).random(len(gradients))
    remaining = np.ones(len(gradients), dtype=bool)
    chosen: list[int] = []
    chunk_size = (budget + min(rounds, budget) - 1) // min(rounds, budget)
    while len(chosen) < budget:
        reference = np.asarray(reference_gradient(tuple(chosen)), dtype=np.float64)
        if reference.shape != (gradients.shape[1],) or not np.isfinite(reference).all():
            raise ValueError("reference gradient must be finite and match the parameter dimension")
        candidates = np.flatnonzero(remaining)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = gradients[candidates] @ reference
        if not np.isfinite(scores).all():
            raise ValueError("Gradient alignment overflowed; rescale gradients before selection")
        order = np.lexsort((tie_break[candidates], -scores))
        selected = candidates[order[: min(chunk_size, budget - len(chosen))]]
        chosen.extend(int(index) for index in selected)
        remaining[selected] = False
    return np.asarray(chosen, dtype=np.int64)
