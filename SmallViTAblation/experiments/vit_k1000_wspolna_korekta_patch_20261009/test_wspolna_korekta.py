"""Testy kontraktu jednej wspolnej korekty patch przed wyslaniem na Eden."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wspolna_korekta import (
    GELU_FLOOR,
    SharedPatchCorrection,
    atomic_torch_save,
    exact_mean_backward,
    joint_loss_per_image,
    load_training_checkpoint,
    should_stop_for_plateau,
)


def test_shared_patch_delta_is_broadcast_per_image_and_clamped() -> None:
    delta = torch.zeros(3, 4)
    delta[1, 2] = -10.0
    module = SharedPatchCorrection(delta, GELU_FLOOR)
    z0 = torch.randn(2, 3, 4)
    corrected = module(z0)
    assert corrected.shape == z0.shape
    assert torch.all(corrected >= GELU_FLOOR)
    assert torch.equal(corrected[:, 0], z0[:, 0].clamp_min(GELU_FLOOR))
    assert torch.all(corrected[:, 1, 2] == GELU_FLOOR)


def test_exact_microbatch_accumulation_matches_full_mean() -> None:
    initial = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float32)
    samples = torch.tensor(
        [[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0], [0.1, -0.2, 0.3], [4.0, 1.0, -2.0], [2.0, -1.0, 0.5]],
        dtype=torch.float32,
    )
    direct = initial.clone().requires_grad_(True)
    direct_losses = ((direct[None] - samples) ** 2).sum(dim=1)
    (direct_losses.mean() + 0.03 * direct.square().mean()).backward()

    accumulated = initial.clone().requires_grad_(True)
    batches = [samples[:2], samples[2:]]

    def objective(parameter: torch.Tensor, batch: torch.Tensor):
        losses = ((parameter[None] - batch) ** 2).sum(dim=1)
        return losses, {"quadratic": losses}

    audit = exact_mean_backward(
        accumulated,
        batches,
        total_count=len(samples),
        objective_fn=objective,
        regularization_weight=0.03,
    )
    assert torch.allclose(accumulated.grad, direct.grad, rtol=2e-6, atol=2e-6)
    assert audit["examples"] == 5
    assert audit["weight_sum"] == pytest.approx(1.0)


def test_joint_loss_combines_confidence_rank_box_and_iou() -> None:
    output = torch.zeros(2, 2, 2, 5, requires_grad=True)
    target_index = torch.tensor([0, 3])
    target_ltrb = torch.tensor([[1.0, 1.5, 2.0, 2.5], [0.5, 0.75, 1.25, 1.5]])
    losses, components = joint_loss_per_image(output, target_index, target_ltrb)
    assert losses.shape == (2,)
    assert set(components) == {"positive", "negative", "rank", "box", "giou"}
    assert torch.isfinite(losses).all()
    losses.mean().backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()


def test_checkpoint_round_trip_preserves_delta_optimizer_and_rng(tmp_path: Path) -> None:
    delta = torch.nn.Parameter(torch.randn(3, 4))
    optimizer = torch.optim.Adam([delta], lr=0.01)
    (delta.square().mean()).backward()
    optimizer.step()
    state = {
        "format_version": 1,
        "completed_pass": 7,
        "delta": delta.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": ["MT19937", [1, 2, 3], 4, 0, 0.0],
        "config_sha256": "a" * 64,
    }
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(path, state)
    loaded = load_training_checkpoint(path, expected_config_sha256="a" * 64)
    assert loaded["completed_pass"] == 7
    assert torch.equal(loaded["delta"], state["delta"])
    assert loaded["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.01)


def test_plateau_contract_has_minimum_pass_and_patience() -> None:
    assert not should_stop_for_plateau(25, evaluations_without_improvement=20, min_passes=30, patience_evaluations=6)
    assert not should_stop_for_plateau(30, evaluations_without_improvement=5, min_passes=30, patience_evaluations=6)
    assert should_stop_for_plateau(30, evaluations_without_improvement=6, min_passes=30, patience_evaluations=6)


def test_plan_declares_in_sample_selection() -> None:
    plan = (Path(__file__).with_name("PLAN.md")).read_text(encoding="utf-8")
    assert "in-sample" in plan
    assert "mAP50:95" in plan
    assert "1500" in plan


def test_results_declaration_has_separate_family() -> None:
    value = json.loads(Path(__file__).with_name("WYNIKI.json").read_text(encoding="utf-8"))
    assert value["rodzina"] == "vit_k1000_wspolna_korekta_patch_20261009"
    assert value["klasa"] == "badawcza"
