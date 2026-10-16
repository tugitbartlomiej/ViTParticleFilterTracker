"""Testuje kontrakt replay; obala go brak prawdziwej osi czasu, targetu lub kontroli Adama."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from fit_analysis_pca32 import spatial_tokens
from replay_core import (
    DENSE_STEPS,
    INTERVAL_END_STEPS,
    PCA_BASIS_SHA256,
    basis_sha256,
    select_panel_tokens,
    validate_replay_arrays,
    write_npz_atomic,
)


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = Path(
    os.environ.get(
        "PROVENANCE", str(ROOT / "outputs/vit_k1000_replay900k_co5_20261009/provenance")
    )
)


def test_fixed_visual_basis_has_expected_hash():
    with np.load(PROVENANCE / "PCA3_FIXED.npz", allow_pickle=False) as data:
        assert basis_sha256(data["pca_mean"], data["pca_components"]) == PCA_BASIS_SHA256


def test_dense_steps_are_exact_real_prefix_without_interpolation():
    assert np.array_equal(DENSE_STEPS, np.arange(0, 501, 5, dtype=np.int32))
    source = (Path(__file__).parent / "replay_core.py").read_text(encoding="utf-8")
    assert "np.interp" not in source


def test_panel_selection_always_contains_target_top1_and_competitors():
    output = torch.zeros(1, 20, 30, 5)
    output.reshape(-1, 5)[:, 0] = torch.linspace(-3, 2, 600)
    output.reshape(-1, 5)[42, 0] = 8.0
    selected = select_panel_tokens(output, target_index=17)
    tokens = selected["tokens"]
    roles = selected["role_mask"]
    assert tokens.shape == (16,)
    assert len(np.unique(tokens)) == 16
    assert 17 in tokens and 42 in tokens
    assert np.any(roles & 1)
    assert np.any(roles & 2)
    assert np.count_nonzero(roles & 4) == 14


def test_panel_split_is_video_disjoint_train_validation_test():
    payload = json.loads((PROVENANCE / "PANEL_64.json").read_text(encoding="utf-8"))
    entries = payload["entries"]
    assert len(entries) == 64
    assert {split: sum(e["analysis_split"] == split for e in entries) for split in ("train", "validation", "test")} == {
        "train": 40,
        "validation": 12,
        "test": 12,
    }
    videos = {
        split: {e["video_id"] for e in entries if e["analysis_split"] == split}
        for split in ("train", "validation", "test")
    }
    assert videos["train"].isdisjoint(videos["validation"])
    assert videos["train"].isdisjoint(videos["test"])
    assert videos["validation"].isdisjoint(videos["test"])


def test_per_image_contract_roundtrip(tmp_path):
    arrays = {
        "projected_positions": np.zeros((101, 600, 3), np.float32),
        "analysis_positions_pca32": np.zeros((101, 600, 32), np.float32),
        "analysis_state_residual_norm": np.zeros((101, 600), np.float32),
        "analysis_increment_residual_norm": np.zeros((100, 600), np.float32),
        "movement_norm": np.zeros((101, 600), np.float32),
        "steps": DENSE_STEPS.copy(),
        "descriptor_interval_end_steps": INTERVAL_END_STEPS.copy(),
        "interval_increment_norm": np.zeros((100, 600), np.float32),
        "interval_previous_increment_dot": np.zeros((100, 600), np.float32),
        "interval_previous_increment_cosine": np.zeros((100, 600), np.float32),
        "interval_previous_increment_valid": np.r_[np.uint8(0), np.ones(99, np.uint8)],
        "interval_gradient_l2_mean": np.zeros((100, 600), np.float32),
        "interval_gradient_l2_last": np.zeros((100, 600), np.float32),
        "interval_gradient_l2_start": np.zeros((100, 600), np.float32),
        "interval_adam_m1_l2_start": np.zeros((100, 600), np.float32),
        "interval_adam_m2_l2_start": np.zeros((100, 600), np.float32),
        "interval_phase_min_norm_fraction": np.zeros(100, np.float32),
        "interval_phase_min_norm_start": np.zeros(100, np.uint8),
        "token_role_mask": np.zeros(600, np.uint8),
        "image_index": np.asarray(7, np.int32),
        "telemetry_step": np.arange(501, dtype=np.int32),
        "telemetry_phase_min_norm": np.zeros(501, np.int32),
        "telemetry_learning_rate": np.zeros(501, np.float32),
        "telemetry_top_token": np.zeros(501, np.int32),
    }
    validate_replay_arrays(arrays)
    path = tmp_path / "000007.npz"
    write_npz_atomic(path, arrays)
    with np.load(path, allow_pickle=False) as restored:
        validate_replay_arrays({name: restored[name] for name in restored.files})
        assert int(restored["image_index"]) == 7


def test_storage_budget_is_gigabytes_not_terabytes():
    viewer = 1500 * 101 * 600 * (3 + 1) * 4
    panel = 64 * (501 * 16 * 768 + 3 * 500 * 16 * 768) * 4
    assert viewer == 1_454_400_000
    descriptors = 1500 * 100 * 600 * 8 * 4
    analytical_pca32 = 1500 * 101 * 600 * 32 * 4
    analytical_residuals = 1500 * (101 + 100) * 600 * 4
    assert panel == 6_294_601_728
    assert descriptors == 2_880_000_000
    assert analytical_pca32 == 11_635_200_000
    assert analytical_residuals == 723_600_000
    assert viewer + panel + descriptors + analytical_pca32 + analytical_residuals < 23_000_000_000


def test_zeno_required_panel_names_are_emitted():
    source = "\n".join(
        (Path(__file__).parent / name).read_text(encoding="utf-8")
        for name in ("replay_core.py", "uruchom_replay.py", "scal_replay.py")
    )
    for name in (
        "control_z",
        "control_gradient_before_step",
        "control_adam_m1_before_step",
        "control_adam_m2_before_step",
        "control_tokens",
        "telemetry_phase_min_norm",
    ):
        assert name in source


def test_pca32_token_sample_covers_every_spatial_stratum():
    tokens = spatial_tokens(137)
    assert tokens.shape == (16,)
    assert len(np.unique(tokens)) == 16
    def column_group(column: int) -> int:
        return 0 if column < 8 else 1 if column < 16 else 2 if column < 23 else 3

    cells = {(int(token) // 30 // 5, column_group(int(token) % 30)) for token in tokens}
    assert cells == {(row, column) for row in range(4) for column in range(4)}


def test_main_dense_exporter_copy_is_unmodified():
    import hashlib

    experiment = Path(__file__).parent
    digest = hashlib.sha256((experiment / "export_dense_timeline.py").read_bytes()).hexdigest()
    helper = hashlib.sha256((experiment / "dense_timeline.py").read_bytes()).hexdigest()
    assert digest == "3168b3e50420306afbb114b66e6978f5fe009a23b2aa672af4cce380ae72acfa"
    assert helper == "9e2cc12673c76f30a6c4751ab9eabca41136a2b420fef0ae81a9a3b3e0c471d3"


def test_population_pca32_and_causal_start_controls_are_in_contract():
    source = "\n".join(
        (Path(__file__).parent / name).read_text(encoding="utf-8")
        for name in ("replay_core.py", "uruchom_replay.py", "scal_replay.py")
    )
    for name in (
        "analysis_positions_pca32",
        "analysis_state_residual_norm",
        "analysis_increment_residual_norm",
        "interval_gradient_l2_start",
        "interval_adam_m1_l2_start",
        "interval_adam_m2_l2_start",
    ):
        assert name in source
