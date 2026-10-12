"""Selection contracts: task relevance, local placebo and leakage-free budgets."""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def _module():
    return importlib.import_module(
        "SmallViTAblation.experiments.dobor_klatek_stan_sieci.selekcja"
    )


def _rows(frames, *, video="V0", group="G0", split="train"):
    return [
        dict(frame_id=f"{video}/{frame:08d}", group_id=group,
             video_id=video, frame_index=frame, split=split)
        for frame in frames
    ]


def test_task_error_uses_mean_smooth_l1_with_beta_point_zero_five():
    result = _module().task_errors(
        np.array([[0., .025, .05, .1], [.02, .02, .02, .02]]),
        np.zeros((2, 4)),
    )
    np.testing.assert_allclose(result, [.0265625, .004])


def test_recoverable_score_requires_task_improvement_not_surprise():
    score = _module().recoverable_scores(
        np.array([1., 2., 0., 5.]), np.array([.25, 2.5, 1., 5.])
    )
    np.testing.assert_allclose(score, [.75, 0., 0., 0.])


@pytest.mark.parametrize("obs,dyn", [([np.nan], [0.]), ([1.], [np.inf]),
                                      ([-1.], [0.]), ([1.], [-.1]),
                                      ([1., 2.], [0.]), ([[1.]], [[0.]])])
def test_recoverable_score_rejects_invalid_errors(obs, dyn):
    with pytest.raises(ValueError):
        _module().recoverable_scores(obs, dyn)


@pytest.mark.parametrize("pred,target", [
    (np.zeros((2, 3)), np.zeros((2, 3))),
    (np.zeros((2, 4)), np.zeros((1, 4))),
    (np.full((1, 4), np.nan), np.zeros((1, 4))),
])
def test_task_error_rejects_invalid_boxes(pred, target):
    with pytest.raises(ValueError):
        _module().task_errors(pred, target)


def test_rank_selection_excludes_initial_validation_and_test():
    rows = _rows(range(0, 40, 5))
    rows += _rows([0], video="Val", split="val")
    rows += _rows([0], video="Test", split="test")
    scores = np.array([100., 8., 7., 6., 5., 4., 3., 2., 1000., 1000.])
    chosen, metadata = _module().rank_select(
        rows, scores, [rows[0]["frame_id"]], 3, seed=42
    )
    assert chosen == [row["frame_id"] for row in rows[1:4]]
    assert metadata["selected_count"] == 3
    assert metadata["excluded_non_train_count"] == 2
    assert not metadata["spacing_relaxed"]


def test_spacing_applies_to_initial_labels_and_selected_frames():
    rows = _rows([0, 1, 4, 5, 9, 10])
    chosen, metadata = _module().rank_select(
        rows, [0., 99., 98., 9., 97., 8.], [rows[0]["frame_id"]], 2, seed=7
    )
    assert chosen == [rows[3]["frame_id"], rows[5]["frame_id"]]
    assert metadata["final_min_gap_frames"] == 5


def test_spacing_is_per_video_not_per_operation():
    rows = _rows([0], video="A") + _rows([0], video="B")
    chosen, metadata = _module().rank_select(rows, [2., 1.], [], 2, seed=3)
    assert len(chosen) == 2
    assert not metadata["spacing_relaxed"]


def test_rank_does_not_relax_spacing_when_a_full_spaced_budget_exists():
    rows = _rows([0, 4, 8])
    chosen, metadata = _module().rank_select(rows, [1., 10., 2.], [], 2, seed=1)
    assert set(chosen) == {rows[0]["frame_id"], rows[2]["frame_id"]}
    assert not metadata["spacing_relaxed"]


def test_unavoidable_spacing_relaxation_is_minimal_and_reported():
    rows = _rows([0, 1, 2])
    chosen, metadata = _module().rank_select(rows, [3., 2., 1.], [], 3, seed=1)
    assert len(chosen) == len(set(chosen)) == 3
    assert metadata["spacing_relaxed"]
    assert metadata["requested_min_gap_frames"] == 5
    assert metadata["final_min_gap_frames"] == 1
    assert metadata["relaxations"]


def test_zero_score_fallback_is_the_same_seeded_random_rule_as_random_arm():
    rows = _rows(range(0, 100, 5))
    actual, meta = _module().rank_select(rows, np.zeros(20), [], 7, seed=23)
    random_ids, _ = _module().rank_select(rows, None, [], 7, seed=23)
    assert actual == random_ids
    assert meta["random_fill_count"] == 7
    assert meta["selected_by_score"] == 0


def test_positive_scores_are_selected_before_random_fill():
    rows = _rows(range(0, 100, 5))
    scores = np.zeros(20)
    scores[8] = 1.
    chosen, metadata = _module().rank_select(rows, scores, [], 5, seed=14)
    assert chosen[0] == rows[8]["frame_id"]
    assert metadata["selected_by_score"] == 1
    assert metadata["random_fill_count"] == 4


def test_ties_are_seeded_random_and_repeatable_without_duplicates():
    rows = _rows(range(0, 150, 5))
    first, _ = _module().rank_select(rows, np.ones(30), [], 10, seed=9)
    again, _ = _module().rank_select(rows, np.ones(30), [], 10, seed=9)
    other, _ = _module().rank_select(rows, np.ones(30), [], 10, seed=10)
    assert first == again
    assert first != other
    assert len(first) == len(set(first)) == 10


@pytest.mark.parametrize("scores", [[1.], [1., np.nan], [1., -1.], [1., np.inf]])
def test_rank_rejects_missing_or_invalid_scores(scores):
    with pytest.raises(ValueError):
        _module().rank_select(_rows([0, 5]), scores, [], 1, seed=0)


def test_rank_rejects_duplicate_ids_and_unknown_initial_ids():
    rows = _rows([0, 5])
    with pytest.raises(ValueError):
        _module().rank_select(rows + [rows[0]], None, [], 1, seed=0)
    with pytest.raises(ValueError):
        _module().rank_select(rows, None, ["not-in-pool"], 1, seed=0)


def test_rank_rejects_duplicate_physical_frames_under_different_ids():
    rows = _rows([0])
    duplicate = dict(rows[0], frame_id="alias")
    with pytest.raises(ValueError):
        _module().rank_select(rows + [duplicate], None, [], 1, seed=0)


def test_rank_rejects_missing_split_and_infeasible_count_budget():
    rows = _rows([0, 5])
    del rows[0]["split"]
    with pytest.raises(ValueError):
        _module().rank_select(rows, None, [], 1, seed=0)
    with pytest.raises(ValueError):
        _module().rank_select(_rows([0, 5]), None, [], 3, seed=0)


def test_placebo_permutes_relative_gain_only_within_group_and_loss_bin():
    rows = _rows(range(0, 80, 5), video="A", group="GA")
    rows += _rows(range(0, 80, 5), video="B", group="GB")
    observed = np.tile(np.arange(1., 17.), 2)
    gains = np.concatenate([np.linspace(.05, .45, 16), np.linspace(.55, .95, 16)])
    dynamic = observed * (1. - gains)
    scores, metadata = _module().placebo_scores(rows, observed, dynamic, seed=21)
    source = metadata["source_indices"]
    bins = metadata["bin_ids"]
    assert sorted(source) == list(range(32))
    assert any(i != j for i, j in enumerate(source))
    for i, j in enumerate(source):
        assert rows[i]["group_id"] == rows[j]["group_id"]
        assert bins[i] == bins[j] == (i % 16) // 4
        assert scores[i] == pytest.approx(observed[i] * gains[j])
    repeated, repeated_metadata = _module().placebo_scores(
        rows, observed, dynamic, seed=21
    )
    np.testing.assert_array_equal(scores, repeated)
    assert metadata == repeated_metadata


def test_placebo_preserves_loss_scale_and_handles_zero_error():
    rows = _rows([0, 5, 10, 15])
    scores, metadata = _module().placebo_scores(
        rows, [0., 1., 2., 3.], [0., .5, 1., 1.5], seed=2, bins=1
    )
    assert scores[0] == 0.
    assert np.isfinite(scores).all()
    for i, j in enumerate(metadata["source_indices"]):
        gain = 0. if j == 0 else .5
        assert scores[i] == pytest.approx([0., 1., 2., 3.][i] * gain)


def test_placebo_never_uses_validation_gain():
    rows = _rows([0, 5]) + _rows([0], video="Val", split="val")
    scores, metadata = _module().placebo_scores(
        rows, [1., 1., 1000.], [.5, .5, 0.], seed=1, bins=1
    )
    np.testing.assert_allclose(scores, [.5, .5, 0.])
    assert metadata["bin_ids"][2] == -1
    assert set(metadata["source_indices"][:2]) == {0, 1}
