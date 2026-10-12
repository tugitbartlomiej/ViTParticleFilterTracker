"""Testy kontraktu: fast tail, dokładny manifest oraz bezstratny zapis latentów."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dane import build_manifest
from model_snapshot import MalyViT
from pelne_latenty import (
    GELU_FLOOR,
    capture_baseline,
    choose_target_token,
    fast_tail,
    learning_rate,
    load_artifact,
    optimize_latent,
    state_fingerprint,
    top1_metrics,
    write_artifact_atomic,
)
from korekta_model import MeanCorrection, RegimeCorrection
from analiza_regimow import fit_regime_model
from artifact_runtime import CorrectedK1000, load_executable_artifact
from metryki_source import detection_map


def test_manifest_preserves_coco_order_scales_boxes_and_hashes_bytes(tmp_path: Path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"first-image")
    image_b.write_bytes(b"second-image")
    coco = {
        "images": [
            {"id": 8, "file_name": str(image_b), "width": 720, "height": 480},
            {"id": 3, "file_name": str(image_a), "width": 720, "height": 480},
        ],
        "annotations": [
            {"id": 30, "image_id": 3, "bbox": [3, 6, 30, 60]},
            {"id": 80, "image_id": 8, "bbox": [12, 15, 90, 120]},
        ],
    }
    coco_path = tmp_path / "annotations.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    rows = build_manifest(coco_path, lambda name: Path(name).read_bytes())

    assert [r["image_id"] for r in rows] == [8, 3]
    assert rows[0]["index"] == 0
    assert rows[0]["bbox_xyxy_480"] == [8.0, 10.0, 68.0, 90.0]
    assert rows[1]["bbox_xyxy_480"] == [2.0, 4.0, 22.0, 44.0]
    assert rows[0]["image_sha256"] == hashlib.sha256(b"second-image").hexdigest()


def test_full_fast_tail_matches_real_hook_forward_and_keeps_model_frozen():
    torch.manual_seed(7)
    net = MalyViT().eval().requires_grad_(False)
    image = torch.rand(1, 3, 32, 32)
    fingerprint_before = state_fingerprint(net)
    baseline, z, q = capture_baseline(net, image)
    changed = (z + 0.01 * torch.randn_like(z)).clamp_min(GELU_FLOOR).requires_grad_()

    expected = fast_tail(net, z, q, changed)
    hook = net.bloki[-1].mlp.act.register_forward_hook(lambda _m, _a, _o: changed)
    actual, _ = net(image)
    hook.remove()

    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(fast_tail(net, z, q, z), baseline, rtol=1e-5, atol=2e-5)
    expected.square().mean().backward()
    assert changed.grad is not None and torch.isfinite(changed.grad).all()
    assert all(p.grad is None for p in net.parameters())
    assert state_fingerprint(net) == fingerprint_before


def test_artifact_roundtrip_is_bitwise_and_resume_metadata_is_preserved(tmp_path: Path):
    rng = np.random.default_rng(11)
    before = rng.standard_normal((4, 7), dtype=np.float32)
    after = np.maximum(before + 0.125, GELU_FLOOR).astype(np.float32)
    delta = (after - before).astype(np.float32)
    trajectory = np.stack([before, after]).astype(np.float16)
    metadata = {
        "index": 17,
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "status": "success",
    }
    path = tmp_path / "000017.npz"

    write_artifact_atomic(path, before, after, delta, trajectory, [0, 5], metadata)
    loaded = load_artifact(path)

    assert np.array_equal(loaded["before"], before)
    assert np.array_equal(loaded["after"], after)
    assert np.array_equal(loaded["delta"], delta)
    assert np.array_equal(loaded["trajectory"], trajectory)
    assert loaded["trajectory_steps"].tolist() == [0, 5]
    assert loaded["metadata"] == metadata


def test_target_token_reports_geometric_iou_bound_for_tiny_boxes():
    ordinary = choose_target_token(np.array([16.0, 16.0, 80.0, 80.0]), 20, 30)
    tiny = choose_target_token(np.array([10.0, 10.0, 11.0, 11.0]), 20, 30)

    assert ordinary["geometry_iou_upper_bound"] == 1.0
    assert ordinary["target_ltrb"].min() > 0
    assert tiny["geometry_iou_upper_bound"] < 0.90
    assert tiny["target_ltrb"].min() > 0


def test_common_schedule_is_defined_for_every_real_step():
    assert learning_rate(1, 500) == 0.05
    assert learning_rate(100, 500) == 0.05
    assert learning_rate(101, 500) == 0.15
    assert learning_rate(250, 500) == 0.15
    assert learning_rate(251, 500) == 0.05
    assert learning_rate(401, 500) == 0.01
    assert learning_rate(500, 500) == 0.01


def test_top1_metrics_use_actual_top_detection_and_ground_truth():
    output = torch.full((1, 2, 2, 5), -8.0)
    output[..., 1:] = 0.0
    output[0, 1, 1, 0] = 4.0
    # Anchor (24,24), exp(0)*16 gives [8,8,40,40].
    metrics = top1_metrics(output, torch.tensor([8.0, 8.0, 40.0, 40.0]))

    assert metrics["token_index"] == 3
    assert metrics["iou"] == 1.0
    assert metrics["confidence"] > 0.98


def test_mean_and_regime_corrections_transform_each_natural_latent_without_targets():
    z0 = torch.tensor(
        [[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    )
    mean = MeanCorrection(torch.ones(2, 3) * 0.25, floor=-10.0)
    torch.testing.assert_close(mean(z0), z0 + 0.25)

    # Assignment features are z0 projection + token row/col + image projection mean.
    regime = RegimeCorrection(
        z_mean=torch.zeros(3),
        z_components=torch.eye(3)[:2],
        feature_mean=torch.zeros(6),
        feature_scale=torch.ones(6),
        assignment_centroids=torch.tensor([[0.0] * 6, [5.0] * 6]),
        correction_by_regime_token=torch.stack(
            [torch.zeros(2, 3), torch.ones(2, 3)], dim=0
        ),
        grid_shape=(1, 2),
        floor=-10.0,
    )
    first = regime(z0)
    # There is deliberately no oracle-delta argument. Repeated assignment from z0 is stable.
    second = regime(z0.clone())
    torch.testing.assert_close(first, second)
    assert regime.last_regime_ids.shape == (2, 2)


def test_optimizer_records_only_real_steps_on_a_common_prefix():
    torch.manual_seed(19)
    net = MalyViT().eval().requires_grad_(False)
    image = torch.rand(1, 3, 32, 32)
    _baseline, z0, q0 = capture_baseline(net, image)
    target = choose_target_token(np.array([8.0, 8.0, 24.0, 24.0]), 2, 2)
    frozen = state_fingerprint(net)

    result = optimize_latent(
        net,
        z0,
        q0,
        torch.tensor([8.0, 8.0, 24.0, 24.0]),
        target,
        horizon=2,
        milestones=(1, 2),
        control_tokens=(),
    )

    assert result["telemetry"]["step"].tolist() == [0, 1, 2]
    assert result["trajectory_steps"].tolist() == [1, 2]
    assert result["trajectory"].shape == (2, 4, 768)
    assert result["final_state"].shape == z0.shape
    assert result["after_step"] in (0, 1, 2)
    assert state_fingerprint(net) == frozen
    assert all(parameter.grad is None for parameter in net.parameters())


def test_regime_training_discovers_with_delta_but_exports_z0_only_assignment():
    rng = np.random.default_rng(31)
    z0 = rng.normal(size=(6, 4, 6)).astype(np.float32)
    delta = np.empty_like(z0)
    delta[:3] = 0.2
    delta[3:] = -0.2
    model, diagnostics = fit_regime_model(
        z0,
        delta,
        np.arange(6),
        components=2,
        regimes=2,
        pca_sample=20,
        cluster_sample=20,
        seed=4,
        grid_shape=(2, 2),
    )

    assert model["mean_delta"].shape == (4, 6)
    assert model["correction_by_regime_token"].shape == (2, 4, 6)
    assert model["assignment_centroids"].shape[0] == 2
    assert diagnostics["discovery_uses_oracle_delta"] is True
    assert diagnostics["assignment_inputs"] == ["natural_z0", "token_row_col", "image_z0_context"]


def test_executable_wrapper_corrects_natural_hook_latent_in_full_forward():
    torch.manual_seed(37)
    base = MalyViT().eval().requires_grad_(False)
    image = torch.rand(1, 3, 32, 32)
    correction = MeanCorrection(torch.full((4, 768), 0.01), floor=GELU_FLOOR)
    executable = CorrectedK1000(base, correction).eval()

    output, natural, corrected = executable(image)

    assert natural.shape == corrected.shape == (1, 4, 768)
    torch.testing.assert_close(corrected, (natural + 0.01).clamp_min(GELU_FLOOR))
    hook = base.bloki[-1].mlp.act.register_forward_hook(lambda _m, _a, _o: corrected)
    expected, _ = base(image)
    hook.remove()
    torch.testing.assert_close(output, expected)


def test_detection_map_uses_ranked_one_to_one_matching_and_coco_thresholds():
    ground_truth = [
        np.asarray([[0.0, 0.0, 10.0, 10.0]], np.float32),
        np.asarray([[20.0, 20.0, 30.0, 30.0]], np.float32),
    ]
    perfect = [
        (np.asarray([[0.0, 0.0, 10.0, 10.0]], np.float32), np.asarray([0.9], np.float32)),
        (np.asarray([[20.0, 20.0, 30.0, 30.0]], np.float32), np.asarray([0.8], np.float32)),
    ]
    duplicate_before_second_true_positive = [
        (
            np.asarray([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], np.float32),
            np.asarray([0.9, 0.85], np.float32),
        ),
        (np.asarray([[20.0, 20.0, 30.0, 30.0]], np.float32), np.asarray([0.8], np.float32)),
    ]

    exact = detection_map(perfect, ground_truth)
    duplicated = detection_map(duplicate_before_second_true_positive, ground_truth)

    assert exact["AP50"] == 1.0
    assert exact["AP75"] == 1.0
    assert exact["mAP50_95"] == 1.0
    assert duplicated["AP50"] < 1.0
    assert duplicated["AP50"] == duplicated["AP75"]


def test_executable_artifact_fresh_load_verifies_hashes_and_applies_rule(tmp_path: Path):
    package = tmp_path / "model_mean"
    package.mkdir()
    net = MalyViT().eval()
    checkpoint = package / "MODEL_VIT_K1000.pt"
    correction = package / "CORRECTION.pt"
    torch.save({"net": net.state_dict(), "krok": 1000}, checkpoint)
    torch.save({"mean_delta": torch.full((4, 768), 0.02)}, correction)

    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    config = {
        "method": "mean600x768",
        "checkpoint_sha256": file_hash(checkpoint),
        "correction_sha256": file_hash(correction),
        "gelu_floor": GELU_FLOOR,
        "grid_shape": [2, 2],
    }
    (package / "config.json").write_text(json.dumps(config), encoding="utf-8")
    files = {
        path.name: file_hash(path)
        for path in (checkpoint, correction, package / "config.json")
    }
    (package / "FILES_SHA256.json").write_text(json.dumps(files), encoding="utf-8")

    loaded, loaded_config = load_executable_artifact(package)
    output, natural, corrected = loaded(torch.rand(1, 3, 32, 32))

    assert loaded_config == config
    assert output.shape == (1, 2, 2, 5)
    torch.testing.assert_close(corrected, (natural + 0.02).clamp_min(GELU_FLOOR))
