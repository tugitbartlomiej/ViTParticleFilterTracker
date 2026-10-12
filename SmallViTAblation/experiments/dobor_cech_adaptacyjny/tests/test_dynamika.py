"""Eden-only runtime tests, written before the adaptive dynamics implementation.

These catch pooling, D-conditioned fitting, future-token leakage, and treating
missing history as a positive dynamics gain. They are not run during preparation.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from SmallViTAblation.experiments.dobor_cech_adaptacyjny.dynamika import (
    LivePatchRecorder,
    compute_dynamics,
    positive_gain,
)


def fixture():
    rng = np.random.default_rng(817)
    rows, tokens, inputs, support = [], [], [], []
    for group in range(5):
        for frame in range(7):
            patch_id = (group + frame // 2) % 4
            cx, cy = .25 + .5 * (patch_id % 2), .25 + .5 * (patch_id // 2)
            row = dict(frame_id=f"g{group}_f{frame}", group_id=f"g{group}",
                       video_id=f"v{group}", frame_index=frame * 5, split="train",
                       role="D" if group == 4 else "FIT", native_width=100,
                       native_height=100,
                       annotations=[dict(bbox=[100 * cx - 5, 100 * cy - 5, 10., 10.])])
            rows.append(row)
            value = rng.normal(0, .05, (4, 3)).astype(np.float32)
            value[:, 0] += np.where(np.arange(4) == patch_id, 1., -1.)
            value[:, 1] += np.sin(frame / 5. + np.arange(4))
            tokens.append(value)
            inputs.append([.8, cx, cy, .1, .1, .5])
            if group < 4 and frame % 2 == 0:
                support.append(row["frame_id"])
    return rows, np.asarray(tokens), np.asarray(inputs, dtype=np.float32), support


def fit(rows, tokens, inputs, support, out, **options):
    config = dict(grid_shape=(2, 2), context=3, step=5, folds=2, epochs=1,
                  max_windows=12, seed=42, device="cpu")
    config.update(options)
    return compute_dynamics(rows, tokens, inputs, rows, support, {"g4"}, out, **config)


def test_d_tokens_inputs_and_labels_cannot_condition_any_fitted_artifact(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    rows, tokens, inputs, support = fixture()
    before = fit(rows, tokens, inputs, support, tmp_path / "before")
    altered_rows, altered_tokens, altered_inputs = copy.deepcopy(rows), tokens.copy(), inputs.copy()
    excluded = np.array([row["group_id"] == "g4" for row in rows])
    altered_tokens[excluded] *= -7.
    altered_inputs[excluded] += 8.
    for row in altered_rows:
        if row["group_id"] == "g4":
            row["annotations"][0]["bbox"] = [70., 70., 10., 10.]
    after = fit(altered_rows, altered_tokens, altered_inputs, support, tmp_path / "after")
    assert before["audit"]["excluded_fit_groups"] == ["g4"]
    assert not before["audit"]["spatial_pooling"]
    assert len(before["diagnostics"]) == len(rows)
    np.testing.assert_array_equal(before["e_obs"][~excluded], after["e_obs"][~excluded])
    for fold in before["audit"]["folds"]:
        assert "g4" not in fold["fit_groups"]
        assert set(fold["fit_groups"]).isdisjoint(fold["held_groups"])
        assert set(fold["readout_support_ids"]) <= set(support)
        for key in ("token_scaler_fit_groups", "input_scaler_fit_groups",
                    "readout_fit_groups", "gle_fit_groups", "koopman_fit_groups"):
            assert set(fold[key]) <= set(fold["fit_groups"])
        relative = f"GLE_fold{fold['fold']}.pt"
        first = torch.load(tmp_path / "before" / relative, weights_only=False, map_location="cpu")
        second = torch.load(tmp_path / "after" / relative, weights_only=False, map_location="cpu")
        for key in ("token_scaling", "input_scaling"):
            for statistic in ("mean", "scale"):
                np.testing.assert_array_equal(first[key][statistic], second[key][statistic])
        np.testing.assert_array_equal(first["readout"]["coefficients"], second["readout"]["coefficients"])
        np.testing.assert_array_equal(first["koopman"]["operator"], second["koopman"]["operator"])
        for name in first["state_dict"]:
            torch.testing.assert_close(first["state_dict"][name], second["state_dict"][name], rtol=0, atol=0)


def test_missing_history_is_ineligible_for_gain(tmp_path):
    pytest.importorskip("torch")
    rows, tokens, inputs, support = fixture()
    result = fit(rows, tokens, inputs, support, tmp_path / "scores", step=7)
    assert np.isfinite(result["e_obs"]).all()
    assert not result["valid_history"].any()
    assert np.isnan(result["e_dyn"]).all()
    np.testing.assert_array_equal(result["positive_gain"], np.zeros(len(rows)))
    assert all(not item["valid_history"] and item["positive_gain"] == 0.
               for item in result["diagnostics"])


def test_nonfinite_and_nonpositive_gain_never_select_a_frame():
    gain = positive_gain(np.array([3., 1., np.nan, 2., 4., 5.]),
                         np.array([1., 3., 1., np.inf, 1., 1.]),
                         np.array([True, True, True, True, False, True]))
    np.testing.assert_array_equal(gain, [2., 0., 0., 0., 0., 4.])


@pytest.mark.parametrize("invalid", ["support_d", "eval_cache", "missing_d_exclusion", "role_mismatch"])
def test_partition_errors_are_rejected_before_fitting(tmp_path, invalid):
    rows, tokens, inputs, support = fixture()
    pool = copy.deepcopy(rows)
    excluded = {"g4"}
    if invalid == "support_d":
        support.append("g4_f0")
    elif invalid == "eval_cache":
        rows[0]["split"] = "val"
    elif invalid == "missing_d_exclusion":
        excluded = set()
    else:
        pool[0]["group_id"] = "g4"
    with pytest.raises(ValueError):
        compute_dynamics(rows, tokens, inputs, pool, support, excluded, tmp_path / invalid,
                         grid_shape=(2, 2), context=3, folds=2, epochs=1,
                         max_windows=12, device="cpu")


def test_fixed_patch_trajectory_ignores_every_other_patch():
    pytest.importorskip("torch")
    from SmallViTAblation.experiments.dobor_cech_adaptacyjny.vendor.token_dopasowanie import particle_trajectories
    tokens = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    windows = np.array([[0, 1, 2], [2, 3, 4]])
    patches = np.array([1, 3])
    got = particle_trajectories(tokens, windows, patches, np.zeros(3), np.ones(3))
    np.testing.assert_array_equal(got[0], [[3, 4, 5], [15, 16, 17], [27, 28, 29]])
    np.testing.assert_array_equal(got[1], [[33, 34, 35], [45, 46, 47], [57, 58, 59]])
    tokens[:, 2] += 1000
    np.testing.assert_array_equal(got, particle_trajectories(tokens, windows, patches, np.zeros(3), np.ones(3)))


def test_live_recorder_preserves_patch_positions_and_uses_current_weights():
    torch = pytest.importorskip("torch")

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, value):
            return [value * self.scale]

    backbone = torch.nn.Module()
    backbone.encoder = Encoder()
    net = torch.nn.Module()
    net.backbone = torch.nn.ModuleList([backbone])
    image = torch.zeros(1, 3, 2, 2)
    image[:, 0, 0, 0] = 1.
    image[:, 0, 1, 1] = 3.
    with LivePatchRecorder(net, dim=5, seed=42) as recorder:
        output = net.backbone[0].encoder(image)
        assert output[0].requires_grad
        first = recorder.pop()
        assert first.shape == (1, 4, 5)
        assert recorder.grid_shape == (2, 2)
        np.testing.assert_array_equal(first[0, 1:3], np.zeros((2, 5)))
        np.testing.assert_allclose(first[0, 3], 3. * first[0, 0])
        assert np.linalg.norm(first[0, 0]) > 0
        with torch.no_grad():
            net.backbone[0].encoder.scale.fill_(2.)
        net.backbone[0].encoder(image)
        np.testing.assert_allclose(recorder.pop(), 2. * first)
        with pytest.raises(RuntimeError):
            recorder.pop()
    assert not net.backbone[0].encoder._forward_hooks


def test_scaled_gle_with_nonzero_residual_path_is_strictly_causal():
    torch = pytest.importorskip("torch")
    from SmallViTAblation.experiments.dobor_cech_adaptacyjny.vendor.model import LangevinConfig
    from SmallViTAblation.experiments.dobor_cech_adaptacyjny.vendor.token_model import ScaledTokenLangevin
    torch.manual_seed(19)
    model = ScaledTokenLangevin(LangevinConfig(latent_dim=4, context=8, hidden_dim=16),
                                np.ones(4), 2. * np.ones(4))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "weight" in name or "bias" in name:
                parameter.normal_(0., .03)
        model.N.a1.fill_(.15)
        model.N.a7.fill_(.1)
    z = torch.randn(2, 10, 4)
    inputs = torch.randn(2, 10, 6)
    observed = torch.rand(2, 10) > .5
    before = model(z, inputs, observed)["prediction"][:, :6].detach()
    z[:, 6:] += 100.
    inputs[:, 6:] -= 100.
    observed[:, 6:] = ~observed[:, 6:]
    after = model(z, inputs, observed)["prediction"][:, :6].detach()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
