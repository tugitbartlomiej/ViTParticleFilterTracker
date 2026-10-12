"""Hand-derived Taylor directions; tests are written but not run locally."""

import numpy as np
import pytest

from SmallViTAblation.experiments.dobor_cech_adaptacyjny.glister import select_gradients


def test_reference_refresh_changes_second_choice_after_virtual_step():
    gradients = np.array([[3.0, 0.0], [-2.0, 0.0], [1.0, 0.0]])

    def reference(selected):
        # A quadratic D objective has its optimum at virtual update 1.0.
        virtual_update = gradients[np.asarray(selected, dtype=int)].mean(axis=0) if len(selected) else np.zeros(2)
        return np.array([1.0, 0.0]) - virtual_update

    selected = select_gradients(gradients, reference, 2, rounds=2, seed=42)
    assert selected.tolist() == [0, 1]
    stale = select_gradients(gradients, reference, 2, rounds=1, seed=42)
    assert stale.tolist() == [0, 2]


def test_positive_gradient_alignment_predicts_loss_reduction():
    gradients = np.array([[-5.0], [2.0], [1.0]])
    selected = select_gradients(gradients, lambda _: np.array([1.0]), 1)
    assert selected.tolist() == [1]


def test_chunked_selection_has_exact_deterministic_unique_budget():
    gradients = np.ones((11, 3))
    first = select_gradients(gradients, lambda _: np.ones(3), 7, rounds=3, seed=4)
    second = select_gradients(gradients, lambda _: np.ones(3), 7, rounds=3, seed=4)
    assert len(first) == len(set(first.tolist())) == 7
    np.testing.assert_array_equal(first, second)


def test_zero_budget_never_evaluates_reference():
    def forbidden(_):
        raise AssertionError("No reference needed for empty budget")

    assert select_gradients(np.ones((2, 2)), forbidden, 0).size == 0


@pytest.mark.parametrize("gradients,budget", [(np.array([[np.nan]]), 1), (np.ones((2, 1)), 3), (np.ones((2, 1)), -1), (np.ones((2, 1)), 1.5)])
def test_invalid_gradients_or_budget_are_rejected(gradients, budget):
    with pytest.raises(ValueError):
        select_gradients(gradients, lambda _: np.ones(1), budget)


def test_nonfinite_or_wrong_size_reference_cannot_silently_rank_candidates():
    with pytest.raises(ValueError, match="reference"):
        select_gradients(np.ones((3, 2)), lambda _: np.array([np.inf, 0]), 1)
    with pytest.raises(ValueError, match="reference"):
        select_gradients(np.ones((3, 2)), lambda _: np.ones(3), 1)
