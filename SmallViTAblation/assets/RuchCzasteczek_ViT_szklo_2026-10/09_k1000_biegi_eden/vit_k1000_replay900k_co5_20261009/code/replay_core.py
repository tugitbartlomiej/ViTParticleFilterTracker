"""Odtwarza prawdziwe stany oracle; obala go rozjazd z krokami 100/500 starego biegu."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from pelne_latenty import (
    GELU_FLOOR,
    _fit_objective,
    _minimum_norm_objective,
    fast_tail,
    learning_rate,
    top1_metrics,
)

HORIZON = 500
DENSE_STEPS = np.arange(0, HORIZON + 1, 5, dtype=np.int32)
INTERVAL_END_STEPS = DENSE_STEPS[1:].copy()
TOKENS = 600
DIMENSIONS = 768
PANEL_TOKENS = 16
PCA_BASIS_SHA256 = "8d3cb28ad2b1581efeca89fab659b9d6fa518bffe4d452b64c68ea7dcd1b9cca"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def basis_sha256(mean: np.ndarray, components: np.ndarray) -> str:
    mean = np.asarray(mean, dtype="<f8")
    components = np.asarray(components, dtype="<f8")
    return hashlib.sha256(mean.tobytes() + components.tobytes()).hexdigest()


def encode_json(value: object) -> np.ndarray:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return np.frombuffer(payload, dtype=np.uint8).copy()


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publikuje nieskompresowany NPZ atomowo; zapis panelu nie zużywa czasu na ZIP deflate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".npz.tmp", prefix=path.name + ".", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_panel_tokens(output: torch.Tensor, target_index: int) -> dict[str, np.ndarray | int]:
    """Wybiera target, naturalny top-1 i najsilniejszych naturalnych konkurentów."""
    if output.shape != (1, 20, 30, 5):
        raise ValueError(output.shape)
    probability = output[0, ..., 0].float().sigmoid()
    maxima = torch.nn.functional.max_pool2d(probability[None, None], 3, 1, 1)[0, 0]
    natural_top1 = int((probability * (probability == maxima)).reshape(-1).argmax().item())
    scores = probability.reshape(-1).detach().cpu().numpy()
    order = np.argsort(-scores, kind="stable")
    chosen: list[int] = []
    for token in (int(target_index), natural_top1, *[int(v) for v in order]):
        if token not in chosen:
            chosen.append(token)
        if len(chosen) == PANEL_TOKENS:
            break
    if len(chosen) != PANEL_TOKENS:
        raise RuntimeError(chosen)
    roles = np.zeros(PANEL_TOKENS, dtype=np.uint8)
    for slot, token in enumerate(chosen):
        if token == target_index:
            roles[slot] |= 1
        if token == natural_top1:
            roles[slot] |= 2
        if token != target_index and token != natural_top1:
            roles[slot] |= 4
    ranks = np.empty(PANEL_TOKENS, dtype=np.int32)
    inverse = np.empty(TOKENS, dtype=np.int32)
    inverse[order] = np.arange(1, TOKENS + 1, dtype=np.int32)
    for slot, token in enumerate(chosen):
        ranks[slot] = inverse[token]
    result = {
        "tokens": np.asarray(chosen, dtype=np.int32),
        "role_mask": roles,
        "baseline_score": scores[np.asarray(chosen)].astype(np.float32),
        "baseline_rank": ranks,
        "natural_top1": natural_top1,
    }
    if not np.any(result["tokens"] == target_index):
        raise RuntimeError("Panel nie zawiera target tokenu")
    if not np.any(result["tokens"] == natural_top1):
        raise RuntimeError("Panel nie zawiera naturalnego top-1")
    return result


def _new_telemetry() -> dict[str, np.ndarray]:
    float_names = (
        "learning_rate",
        "fit_objective",
        "confidence",
        "iou",
        "delta_rms",
        "delta_l2",
        "gradient_rms",
        "target_confidence",
        "natural_top1_confidence",
        "max_other_confidence",
    )
    int_names = (
        "step",
        "phase_min_norm",
        "top_token",
        "target_is_top",
        "success",
        "target_rank",
    )
    result = {name: np.empty(HORIZON + 1, dtype=np.float32) for name in float_names}
    result.update({name: np.empty(HORIZON + 1, dtype=np.int32) for name in int_names})
    return result


def replay_latent(
    net: torch.nn.Module,
    baseline_latent: torch.Tensor,
    baseline_block_output: torch.Tensor,
    ground_truth: torch.Tensor,
    target: dict,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    analysis_pca_mean: torch.Tensor,
    analysis_pca_components: torch.Tensor,
    *,
    panel: bool,
    reference: dict | None = None,
) -> dict:
    """Wykonuje ten sam Adam co oracle i zapisuje rzeczywiste stany bez interpolacji."""
    if baseline_latent.shape != (1, TOKENS, DIMENSIONS):
        raise ValueError(baseline_latent.shape)
    if basis_sha256(pca_mean, pca_components) != PCA_BASIS_SHA256:
        raise RuntimeError("Niewłaściwa baza PCA3")
    if analysis_pca_mean.shape != (DIMENSIONS,) or analysis_pca_components.shape != (DIMENSIONS, 32):
        raise ValueError((analysis_pca_mean.shape, analysis_pca_components.shape))
    if analysis_pca_mean.device != baseline_latent.device or analysis_pca_components.device != baseline_latent.device:
        raise ValueError("PCA32 nie jest na urządzeniu latentu")
    target_index = int(target["token_index"])
    target_ltrb = torch.as_tensor(
        target["target_ltrb"], device=baseline_latent.device, dtype=torch.float32
    )
    ground_truth = ground_truth.to(device=baseline_latent.device, dtype=torch.float32)
    changed = baseline_latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([changed], lr=learning_rate(1, HORIZON))

    baseline_np = baseline_latent[0].detach().cpu().numpy().astype(np.float32, copy=True)
    mean64 = np.asarray(pca_mean, dtype=np.float64)
    components64 = np.asarray(pca_components, dtype=np.float64)
    projected = np.empty((len(DENSE_STEPS), TOKENS, 3), dtype=np.float32)
    analysis_positions_pca32 = np.empty((len(DENSE_STEPS), TOKENS, 32), dtype=np.float32)
    analysis_state_residual_norm = np.empty((len(DENSE_STEPS), TOKENS), dtype=np.float32)
    analysis_increment_residual_norm = np.empty((len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32)
    movement_norm = np.empty((len(DENSE_STEPS), TOKENS), dtype=np.float32)
    interval_increment_norm = np.empty((len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32)
    interval_previous_increment_dot = np.zeros(
        (len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32
    )
    interval_previous_increment_cosine = np.zeros(
        (len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32
    )
    interval_previous_increment_valid = np.ones(len(DENSE_STEPS) - 1, dtype=np.uint8)
    interval_previous_increment_valid[0] = 0
    interval_gradient_l2_mean = np.empty(
        (len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32
    )
    interval_gradient_l2_last = np.empty(
        (len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32
    )
    interval_gradient_l2_start = np.empty(
        (len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32
    )
    interval_adam_m1_l2_start = np.empty(
        (len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32
    )
    interval_adam_m2_l2_start = np.empty(
        (len(DENSE_STEPS) - 1, TOKENS), dtype=np.float32
    )
    interval_phase_min_norm_fraction = np.empty(len(DENSE_STEPS) - 1, dtype=np.float32)
    interval_phase_min_norm_start = np.empty(len(DENSE_STEPS) - 1, dtype=np.uint8)
    telemetry = _new_telemetry()
    phase_before_update = np.empty(HORIZON, dtype=np.uint8)

    output = fast_tail(net, baseline_latent, baseline_block_output, changed)
    selection = select_panel_tokens(output, target_index)
    natural_top1 = int(selection["natural_top1"])
    token_role_mask = np.zeros(TOKENS, dtype=np.uint8)
    for token, role in zip(selection["tokens"], selection["role_mask"], strict=True):
        token_role_mask[int(token)] |= int(role)
    panel_arrays: dict[str, np.ndarray] = {}
    selected = np.asarray(selection["tokens"], dtype=np.int32)
    if panel:
        panel_arrays = {
            "control_z": np.empty((HORIZON + 1, PANEL_TOKENS, DIMENSIONS), np.float32),
            "control_gradient_before_step": np.empty((HORIZON, PANEL_TOKENS, DIMENSIONS), np.float32),
            "control_adam_m1_before_step": np.empty((HORIZON, PANEL_TOKENS, DIMENSIONS), np.float32),
            "control_adam_m2_before_step": np.empty((HORIZON, PANEL_TOKENS, DIMENSIONS), np.float32),
        }
        panel_arrays["control_z"][0] = baseline_np[selected]

    comparison = {
        "before_exact": None,
        "before_max_abs": None,
        "step_100_exact": None,
        "step_100_max_abs": None,
        "step_500_exact": None,
        "step_500_max_abs": None,
    }
    reference_states: dict[int, np.ndarray] = {}
    if reference is not None:
        comparison["before_exact"] = bool(np.array_equal(reference["before"], baseline_np))
        comparison["before_max_abs"] = float(
            np.max(np.abs(reference["before"].astype(np.float64) - baseline_np.astype(np.float64)))
        )
        for slot, step in enumerate(reference["trajectory_steps"].tolist()):
            reference_states[int(step)] = reference["trajectory"][slot]

    baseline64 = baseline_np.astype(np.float64)
    previous_dense_state64: np.ndarray | None = None
    previous_interval_increment64: np.ndarray | None = None

    def record_dense(step: int) -> np.ndarray:
        nonlocal previous_dense_state64, previous_interval_increment64
        state = changed[0].detach().cpu().numpy().astype(np.float32, copy=True)
        state64 = state.astype(np.float64)
        slot = step // 5
        projected[slot] = ((state64 - mean64) @ components64).astype(np.float32)
        with torch.no_grad():
            analysis_centered = changed[0].detach().float() - analysis_pca_mean
            analysis_projected = analysis_centered @ analysis_pca_components
            total_square = analysis_centered.square().sum(dim=-1)
            captured_square = analysis_projected.square().sum(dim=-1)
            residual = (total_square - captured_square).clamp_min(0.0).sqrt()
        analysis_positions_pca32[slot] = analysis_projected.cpu().numpy().astype(np.float32)
        analysis_state_residual_norm[slot] = residual.cpu().numpy().astype(np.float32)
        if step == 0:
            movement_norm[slot].fill(0.0)
        else:
            movement_norm[slot] = np.linalg.norm(state64 - baseline64, axis=-1).astype(np.float32)
            interval_slot = slot - 1
            increment64 = state64 - previous_dense_state64
            increment_norm64 = np.linalg.norm(increment64, axis=-1)
            interval_increment_norm[interval_slot] = increment_norm64.astype(np.float32)
            projected_increment64 = (
                analysis_positions_pca32[slot].astype(np.float64)
                - analysis_positions_pca32[slot - 1].astype(np.float64)
            )
            projected_increment_square = np.square(projected_increment64).sum(axis=-1)
            analysis_increment_residual_norm[interval_slot] = np.sqrt(
                np.maximum(np.square(increment_norm64) - projected_increment_square, 0.0)
            ).astype(np.float32)
            if previous_interval_increment64 is not None:
                dot64 = np.einsum(
                    "ij,ij->i", increment64, previous_interval_increment64, optimize=True
                )
                previous_norm64 = np.linalg.norm(previous_interval_increment64, axis=-1)
                denominator = increment_norm64 * previous_norm64
                cosine64 = np.divide(
                    dot64,
                    denominator,
                    out=np.zeros_like(dot64),
                    where=denominator > 0.0,
                )
                interval_previous_increment_dot[interval_slot] = dot64.astype(np.float32)
                interval_previous_increment_cosine[interval_slot] = np.clip(
                    cosine64, -1.0, 1.0
                ).astype(np.float32)
            previous_interval_increment64 = increment64
        previous_dense_state64 = state64
        if step in reference_states:
            difference = state.astype(np.float64) - reference_states[step].astype(np.float64)
            comparison[f"step_{step}_exact"] = bool(np.array_equal(state, reference_states[step]))
            comparison[f"step_{step}_max_abs"] = float(np.max(np.abs(difference)))
        return state

    first_success_step: int | None = None

    def inspect(step: int, own_output: torch.Tensor, fit_loss: torch.Tensor, grad_rms: float) -> None:
        nonlocal first_success_step
        metrics = top1_metrics(own_output, ground_truth)
        difference = changed.detach() - baseline_latent
        success = metrics["confidence"] >= 0.90 and metrics["iou"] >= 0.90
        if success and first_success_step is None:
            first_success_step = step
        logits = own_output.reshape(-1, 5)[:, 0]
        probability = logits.float().sigmoid()
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[target_index] = False
        target_logit = logits[target_index]
        telemetry["step"][step] = step
        telemetry["learning_rate"][step] = 0.0 if step == 0 else learning_rate(step, HORIZON)
        telemetry["phase_min_norm"][step] = int(first_success_step is not None)
        telemetry["fit_objective"][step] = float(fit_loss.detach().item())
        telemetry["confidence"][step] = metrics["confidence"]
        telemetry["iou"][step] = metrics["iou"]
        telemetry["delta_rms"][step] = float(difference.square().mean().sqrt().item())
        telemetry["delta_l2"][step] = float(difference.float().norm().item())
        telemetry["gradient_rms"][step] = grad_rms
        telemetry["top_token"][step] = int(metrics["token_index"])
        telemetry["target_is_top"][step] = int(metrics["token_index"] == target_index)
        telemetry["success"][step] = int(success)
        telemetry["target_confidence"][step] = float(probability[target_index].item())
        telemetry["natural_top1_confidence"][step] = float(probability[natural_top1].item())
        telemetry["max_other_confidence"][step] = float(probability[mask].max().item())
        telemetry["target_rank"][step] = int((logits > target_logit).sum().item()) + 1

    fit_loss = _fit_objective(output, changed, baseline_latent, target_index, target_ltrb)
    inspect(0, output, fit_loss, 0.0)
    record_dense(0)
    interval_gradient_l2_sum = np.zeros(TOKENS, dtype=np.float64)

    for step in range(1, HORIZON + 1):
        optimizer.param_groups[0]["lr"] = learning_rate(step, HORIZON)
        optimizer.zero_grad(set_to_none=True)
        output = fast_tail(net, baseline_latent, baseline_block_output, changed)
        phase_before_update[step - 1] = int(first_success_step is not None)
        if first_success_step is None:
            loss = _fit_objective(output, changed, baseline_latent, target_index, target_ltrb)
        else:
            loss = _minimum_norm_objective(
                output, changed, baseline_latent, ground_truth, target_index
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Niefinitywna strata w kroku {step}")
        loss.backward()
        if changed.grad is None or not torch.isfinite(changed.grad).all():
            raise FloatingPointError(f"Niefinitywny gradient w kroku {step}")
        gradient_rms = float(changed.grad.detach().square().mean().sqrt().item())
        gradient_l2 = (
            changed.grad[0].detach().float().norm(dim=-1).cpu().numpy().astype(np.float32)
        )
        interval_gradient_l2_sum += gradient_l2.astype(np.float64)
        state = optimizer.state.get(changed, {})
        if (step - 1) % 5 == 0:
            interval_slot = (step - 1) // 5
            interval_gradient_l2_start[interval_slot] = gradient_l2
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None:
                interval_adam_m1_l2_start[interval_slot].fill(0.0)
                interval_adam_m2_l2_start[interval_slot].fill(0.0)
            else:
                interval_adam_m1_l2_start[interval_slot] = (
                    exp_avg[0].detach().float().norm(dim=-1).cpu().numpy().astype(np.float32)
                )
                interval_adam_m2_l2_start[interval_slot] = (
                    exp_avg_sq[0].detach().float().norm(dim=-1).cpu().numpy().astype(np.float32)
                )
            interval_phase_min_norm_start[interval_slot] = phase_before_update[step - 1]
        if panel:
            zeros = torch.zeros_like(changed)
            panel_arrays["control_gradient_before_step"][step - 1] = (
                changed.grad[0, selected].detach().cpu().numpy()
            )
            panel_arrays["control_adam_m1_before_step"][step - 1] = (
                state.get("exp_avg", zeros)[0, selected].detach().cpu().numpy()
            )
            panel_arrays["control_adam_m2_before_step"][step - 1] = (
                state.get("exp_avg_sq", zeros)[0, selected].detach().cpu().numpy()
            )
        optimizer.step()
        with torch.no_grad():
            changed.clamp_(min=GELU_FLOOR)
        if panel:
            panel_arrays["control_z"][step] = changed[0, selected].detach().cpu().numpy()
        output = fast_tail(net, baseline_latent, baseline_block_output, changed)
        fit_loss = _fit_objective(output, changed, baseline_latent, target_index, target_ltrb)
        inspect(step, output, fit_loss, gradient_rms)
        if step % 5 == 0:
            record_dense(step)
            interval_slot = step // 5 - 1
            interval_gradient_l2_mean[interval_slot] = (
                interval_gradient_l2_sum / 5.0
            ).astype(np.float32)
            interval_gradient_l2_last[interval_slot] = gradient_l2
            interval_phase_min_norm_fraction[interval_slot] = float(
                phase_before_update[step - 5 : step].mean(dtype=np.float64)
            )
            interval_gradient_l2_sum.fill(0.0)

    telemetry_comparison: dict[str, float | bool] = {}
    if reference is not None:
        for name in (
            "step",
            "learning_rate",
            "phase_min_norm",
            "fit_objective",
            "confidence",
            "iou",
            "delta_rms",
            "delta_l2",
            "gradient_rms",
            "top_token",
            "target_is_top",
            "success",
        ):
            expected = reference[f"telemetry_{name}"]
            actual = telemetry[name]
            telemetry_comparison[f"{name}_exact"] = bool(np.array_equal(expected, actual))
            telemetry_comparison[f"{name}_max_abs"] = float(
                np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))
            )
    return {
        "projected_positions": projected,
        "analysis_positions_pca32": analysis_positions_pca32,
        "analysis_state_residual_norm": analysis_state_residual_norm,
        "analysis_increment_residual_norm": analysis_increment_residual_norm,
        "movement_norm": movement_norm,
        "steps": DENSE_STEPS.copy(),
        "descriptor_interval_end_steps": INTERVAL_END_STEPS.copy(),
        "interval_increment_norm": interval_increment_norm,
        "interval_previous_increment_dot": interval_previous_increment_dot,
        "interval_previous_increment_cosine": interval_previous_increment_cosine,
        "interval_previous_increment_valid": interval_previous_increment_valid,
        "interval_gradient_l2_mean": interval_gradient_l2_mean,
        "interval_gradient_l2_last": interval_gradient_l2_last,
        "interval_gradient_l2_start": interval_gradient_l2_start,
        "interval_adam_m1_l2_start": interval_adam_m1_l2_start,
        "interval_adam_m2_l2_start": interval_adam_m2_l2_start,
        "interval_phase_min_norm_fraction": interval_phase_min_norm_fraction,
        "interval_phase_min_norm_start": interval_phase_min_norm_start,
        "token_role_mask": token_role_mask,
        "telemetry": telemetry,
        "phase_before_update": phase_before_update,
        "first_success_step": first_success_step,
        "status": "success" if first_success_step is not None else "budget_exhausted",
        "target_index": target_index,
        "natural_top1": natural_top1,
        "panel_selection": selection,
        "panel_arrays": panel_arrays,
        "state_comparison": comparison,
        "telemetry_comparison": telemetry_comparison,
    }


def validate_replay_arrays(arrays: dict[str, np.ndarray]) -> None:
    required = {
        "projected_positions": ((101, 600, 3), np.dtype("float32")),
        "analysis_positions_pca32": ((101, 600, 32), np.dtype("float32")),
        "analysis_state_residual_norm": ((101, 600), np.dtype("float32")),
        "analysis_increment_residual_norm": ((100, 600), np.dtype("float32")),
        "movement_norm": ((101, 600), np.dtype("float32")),
        "steps": ((101,), np.dtype("int32")),
        "descriptor_interval_end_steps": ((100,), np.dtype("int32")),
        "interval_increment_norm": ((100, 600), np.dtype("float32")),
        "interval_previous_increment_dot": ((100, 600), np.dtype("float32")),
        "interval_previous_increment_cosine": ((100, 600), np.dtype("float32")),
        "interval_previous_increment_valid": ((100,), np.dtype("uint8")),
        "interval_gradient_l2_mean": ((100, 600), np.dtype("float32")),
        "interval_gradient_l2_last": ((100, 600), np.dtype("float32")),
        "interval_gradient_l2_start": ((100, 600), np.dtype("float32")),
        "interval_adam_m1_l2_start": ((100, 600), np.dtype("float32")),
        "interval_adam_m2_l2_start": ((100, 600), np.dtype("float32")),
        "interval_phase_min_norm_fraction": ((100,), np.dtype("float32")),
        "interval_phase_min_norm_start": ((100,), np.dtype("uint8")),
        "token_role_mask": ((600,), np.dtype("uint8")),
        "image_index": ((), np.dtype("int32")),
        "telemetry_step": ((501,), np.dtype("int32")),
        "telemetry_phase_min_norm": ((501,), np.dtype("int32")),
        "telemetry_learning_rate": ((501,), np.dtype("float32")),
        "telemetry_top_token": ((501,), np.dtype("int32")),
    }
    for name, (shape, dtype) in required.items():
        if name not in arrays or arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError((name, arrays.get(name, None), shape, dtype))
    if not np.array_equal(arrays["steps"], DENSE_STEPS):
        raise ValueError("Nieprawdziwa oś czasu co 5")
    if not np.array_equal(arrays["descriptor_interval_end_steps"], INTERVAL_END_STEPS):
        raise ValueError("Nieprawdziwa oś interwałów deskryptorów")
    if not np.array_equal(arrays["telemetry_step"], np.arange(501, dtype=np.int32)):
        raise ValueError("Niepełna telemetria")
    if not np.isfinite(arrays["projected_positions"]).all():
        raise ValueError("Niefinitywne pozycje PCA3")
    if not np.isfinite(arrays["analysis_positions_pca32"]).all():
        raise ValueError("Niefinitywne pozycje analitycznej PCA32")
    for name in ("analysis_state_residual_norm", "analysis_increment_residual_norm"):
        if not np.isfinite(arrays[name]).all() or np.any(arrays[name] < 0):
            raise ValueError(f"Nieprawidłowe residuum PCA32: {name}")
    if not np.isfinite(arrays["movement_norm"]).all() or np.any(arrays["movement_norm"] < 0):
        raise ValueError("Nieprawidłowe normy 768D")
    if not np.array_equal(arrays["movement_norm"][0], np.zeros(600, np.float32)):
        raise ValueError("Norma w kroku 0 nie jest dokładnie zerem")
    for name in (
        "interval_increment_norm",
        "interval_previous_increment_dot",
        "interval_previous_increment_cosine",
        "interval_gradient_l2_mean",
        "interval_gradient_l2_last",
        "interval_gradient_l2_start",
        "interval_adam_m1_l2_start",
        "interval_adam_m2_l2_start",
        "interval_phase_min_norm_fraction",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Niefinitywny deskryptor {name}")
    for name in (
        "interval_increment_norm",
        "interval_gradient_l2_mean",
        "interval_gradient_l2_last",
        "interval_gradient_l2_start",
        "interval_adam_m1_l2_start",
        "interval_adam_m2_l2_start",
    ):
        if np.any(arrays[name] < 0):
            raise ValueError(f"Ujemna norma {name}")
    if np.any(np.abs(arrays["interval_previous_increment_cosine"]) > 1.000001):
        raise ValueError("Cosinus kolejnych przyrostów poza [-1,1]")
    if not np.array_equal(
        arrays["interval_previous_increment_valid"],
        np.r_[np.uint8(0), np.ones(99, dtype=np.uint8)],
    ):
        raise ValueError("Nieprawidłowa maska pierwszego kąta")
    if np.any((arrays["interval_phase_min_norm_fraction"] < 0) | (arrays["interval_phase_min_norm_fraction"] > 1)):
        raise ValueError("Nieprawidłowy udział fazy")
    if not np.all(np.isin(arrays["interval_phase_min_norm_start"], (0, 1))):
        raise ValueError("Nieprawidłowa faza na początku interwału")
    if not np.all(np.isin(arrays["token_role_mask"], np.arange(8, dtype=np.uint8))):
        raise ValueError("Nieprawidłowa maska roli tokenu")


def validate_panel_arrays(arrays: dict[str, np.ndarray], target_index: int, natural_top1: int) -> None:
    shapes = {
        "control_z": (501, 16, 768),
        "control_gradient_before_step": (500, 16, 768),
        "control_adam_m1_before_step": (500, 16, 768),
        "control_adam_m2_before_step": (500, 16, 768),
        "control_tokens": (16,),
        "telemetry_step": (501,),
        "telemetry_phase_min_norm": (501,),
        "telemetry_learning_rate": (501,),
        "telemetry_top_token": (501,),
    }
    for name, shape in shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError((name, arrays.get(name, None), shape))
    for name in (
        "control_z",
        "control_gradient_before_step",
        "control_adam_m1_before_step",
        "control_adam_m2_before_step",
    ):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(name)
    if np.any(arrays["control_adam_m2_before_step"] < 0):
        raise ValueError("Druga chwila Adama ma wartości ujemne")
    tokens = arrays["control_tokens"]
    if len(np.unique(tokens)) != 16 or target_index not in tokens or natural_top1 not in tokens:
        raise ValueError("Panel nie zawiera wymaganych tokenów")
