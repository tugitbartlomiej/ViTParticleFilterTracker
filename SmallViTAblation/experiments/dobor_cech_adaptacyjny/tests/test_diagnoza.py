"""Tests prepared before implementation; execution is deferred to Eden."""

import numpy as np
import pytest

from SmallViTAblation.experiments.dobor_cech_adaptacyjny.diagnoza import (
    discover_needs,
    fit_probes,
    predict_probes,
)


def test_probes_learn_fit_relation_and_preserve_unobserved_features_as_nan():
    descriptors = np.arange(8.0)[:, None]
    features = np.column_stack([2 * descriptors[:, 0] + 3, np.full(8, np.nan)])
    state = fit_probes(descriptors, features)
    mean_before = state["descriptor_mean"].copy()
    predictions = predict_probes(state, np.array([[10.0], [1000.0], [np.nan]]))
    np.testing.assert_allclose(predictions[:2, 0], [23.0, 2003.0], rtol=1e-4)
    assert np.isnan(predictions[:, 1]).all()
    assert np.isnan(predictions[2]).all()
    np.testing.assert_array_equal(state["descriptor_mean"], mean_before)
    assert state["n_feature_samples"].tolist() == [8, 0]


def test_missing_targets_are_excluded_not_imputed_as_zero():
    descriptors = np.arange(5.0)[:, None]
    targets = np.array([[7.0], [np.nan], [7.0], [np.nan], [7.0]])
    predicted = predict_probes(fit_probes(descriptors, targets), np.array([[1.0], [3.0]]))
    np.testing.assert_allclose(predicted, [[7.0], [7.0]], atol=1e-8)


def diagnostic_fixture():
    fit = np.arange(30.0)[:, None]
    diagnostic = np.concatenate([np.ones(9), np.full(9, 20.0)])[:, None]
    losses = np.concatenate([np.full(9, 4.0), np.ones(9)])
    groups = np.tile(["op_a", "op_b", "op_c"], 6)
    return fit, diagnostic, losses, groups


def test_needs_use_fit_thresholds_and_held_out_task_difficulty():
    fit, diagnostic, losses, groups = diagnostic_fixture()
    needs = discover_needs(fit, diagnostic, losses, groups, ["brightness"])
    assert len(needs) == 1
    need = needs[0]
    assert need["conditions"][0]["upper"] == pytest.approx(29.0 / 3.0)
    assert need["support"] == 9
    assert need["group_count"] == 3
    assert need["mean_task_loss"] == 4.0
    assert need["baseline_task_loss"] == 2.5
    assert need["priority"] > 0
    changed = diagnostic.copy()
    changed[:9, 0] = -1000.0
    needs_changed = discover_needs(fit, changed, losses, groups, ["brightness"])
    assert needs_changed[0]["conditions"] == need["conditions"]


def test_missing_measurements_and_single_operation_cannot_create_supported_need():
    fit, diagnostic, losses, groups = diagnostic_fixture()
    diagnostic[:8, 0] = np.nan
    assert discover_needs(fit, diagnostic, losses, groups, ["brightness"]) == []
    fit, diagnostic, losses, groups = diagnostic_fixture()
    groups[:9] = "op_a"
    assert discover_needs(fit, diagnostic, losses, groups, ["brightness"]) == []


def test_latent_and_dynamics_evidence_cannot_create_need_without_task_difficulty():
    fit, diagnostic, losses, groups = diagnostic_fixture()
    assert discover_needs(fit, diagnostic, np.ones(18), groups, ["brightness"], latent_errors=np.full((18, 1), 1e9), dynamics_gain=np.full(18, 1e9)) == []
    ordinary = discover_needs(fit, diagnostic, losses, groups, ["brightness"])[0]
    augmented = discover_needs(fit, diagnostic, losses, groups, ["brightness"], latent_errors=np.full((18, 1), 1e9), dynamics_gain=np.full(18, 1e9))[0]
    assert ordinary["priority"] <= augmented["priority"] <= 1.5 * ordinary["priority"]
    negative_gain = discover_needs(fit, diagnostic, losses, groups, ["brightness"], dynamics_gain=np.full(18, -100.0))[0]
    assert negative_gain["priority"] == ordinary["priority"]


def test_absent_fit_feature_cannot_get_threshold_from_diagnostic_set():
    fit, diagnostic, losses, groups = diagnostic_fixture()
    fit[:] = np.nan
    assert discover_needs(fit, diagnostic, losses, groups, ["brightness"]) == []


def test_mismatched_diagnostic_lengths_and_nonfinite_task_losses_fail_closed():
    fit, diagnostic, losses, groups = diagnostic_fixture()
    with pytest.raises(ValueError):
        discover_needs(fit, diagnostic, losses[:-1], groups, ["brightness"])
    losses[0] = np.nan
    with pytest.raises(ValueError, match="loss"):
        discover_needs(fit, diagnostic, losses, groups, ["brightness"])
