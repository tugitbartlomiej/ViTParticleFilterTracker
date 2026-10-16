"""Rdzeń interwencji: dokładny fast tail i atomowe artefakty per obraz."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

GELU_FLOOR = -0.16997121
STRIDE = 16.0


def state_fingerprint(net: torch.nn.Module) -> str:
    """Hash nazw, typów, kształtów i surowych bajtów wszystkich wag i buforów."""
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def capture_baseline(net: torch.nn.Module, image: torch.Tensor):
    """Zwraca pełny output, post-GELU `z` i wynik ostatniego bloku `q`."""
    cache: dict[str, torch.Tensor] = {}
    block = net.bloki[-1]
    h_z = block.mlp.act.register_forward_hook(
        lambda _module, _args, output: cache.update(z=output.detach().clone())
    )
    h_q = block.register_forward_hook(
        lambda _module, _args, output: cache.update(q=output.detach().clone())
    )
    try:
        with torch.no_grad():
            baseline, _ = net(image)
    finally:
        h_z.remove()
        h_q.remove()
    if set(cache) != {"z", "q"}:
        raise RuntimeError(f"Hooki nie zebrały obu tensorów: {sorted(cache)}")
    return baseline.detach(), cache["z"], cache["q"]


def _grid_shape(tokens: int) -> tuple[int, int]:
    pairs = [(h, tokens // h) for h in range(1, int(tokens**0.5) + 1) if tokens % h == 0]
    if not pairs:
        raise ValueError(f"Nie można rozłożyć {tokens} tokenów na siatkę")
    return min(pairs, key=lambda hw: abs(hw[1] / hw[0] - 1.5))


def fast_tail(
    net: torch.nn.Module,
    baseline_latent: torch.Tensor,
    baseline_block_output: torch.Tensor,
    changed_latent: torch.Tensor,
) -> torch.Tensor:
    """Przelicza część sieci za post-GELU po podmianie wszystkich 768 osi."""
    if changed_latent.shape != baseline_latent.shape:
        raise ValueError((changed_latent.shape, baseline_latent.shape))
    block = net.bloki[-1]
    delta = block.ls2(
        F.linear(changed_latent - baseline_latent, block.mlp.fc2.weight, None)
    )
    flat = net.glowa(net.norm(baseline_block_output + delta))
    height, width = _grid_shape(flat.shape[1])
    return flat.reshape(flat.shape[0], height, width, 5)


def learning_rate(step: int, horizon: int = 500) -> float:
    """Wspólny schedule; `step` jest realną liczbą wykonanych aktualizacji."""
    if horizon != 500:
        # Smoke może użyć krótszego prefiksu, ale nie przeskalowuje osi czasu.
        if not 1 <= step <= horizon or horizon > 500:
            raise ValueError((step, horizon))
    elif not 1 <= step <= 500:
        raise ValueError(step)
    if step <= 100:
        return 0.05
    if step <= 250:
        return 0.15
    if step <= 400:
        return 0.05
    return 0.01


def choose_target_token(box_xyxy: np.ndarray, height: int = 20, width: int = 30) -> dict:
    """Wybiera kotwicę o największym geometrycznie możliwym IoU z ramką GT."""
    box = np.asarray(box_xyxy, dtype=np.float64)
    x0, y0, x1, y1 = box.tolist()
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"Zdegenerowana ramka {box.tolist()}")
    xs = (np.arange(width, dtype=np.float64) + 0.5) * STRIDE
    ys = (np.arange(height, dtype=np.float64) + 0.5) * STRIDE
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    union_x0 = np.minimum(cx, x0)
    union_y0 = np.minimum(cy, y0)
    union_x1 = np.maximum(cx, x1)
    union_y1 = np.maximum(cy, y1)
    upper = ((x1 - x0) * (y1 - y0)) / (
        (union_x1 - union_x0) * (union_y1 - union_y0)
    )
    distance = (cx - (x0 + x1) / 2.0) ** 2 + (cy - (y0 + y1) / 2.0) ** 2
    best_upper = float(upper.max())
    candidates = np.flatnonzero(np.isclose(upper.reshape(-1), best_upper, rtol=0, atol=1e-12))
    flat_distance = distance.reshape(-1)
    index = int(candidates[np.argmin(flat_distance[candidates])])
    row, column = divmod(index, width)
    anchor_x, anchor_y = xs[column], ys[row]
    ltrb = np.array(
        [anchor_x - x0, anchor_y - y0, x1 - anchor_x, y1 - anchor_y],
        dtype=np.float32,
    ) / STRIDE
    ltrb = np.maximum(ltrb, np.float32(1e-4))
    return {
        "token_index": index,
        "token_row": row,
        "token_col": column,
        "anchor_xy": [float(anchor_x), float(anchor_y)],
        "target_ltrb": ltrb,
        "target_log_ltrb": np.log(ltrb).astype(np.float32),
        "geometry_iou_upper_bound": best_upper,
    }


def _iou_one(box: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    x0 = torch.maximum(box[0], ground_truth[0])
    y0 = torch.maximum(box[1], ground_truth[1])
    x1 = torch.minimum(box[2], ground_truth[2])
    y1 = torch.minimum(box[3], ground_truth[3])
    intersection = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    area_box = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area_gt = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
    return intersection / (area_box + area_gt - intersection + 1e-9)


@torch.no_grad()
def top1_metrics(output: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    """Metryki faktycznej top detekcji, z lokalnymi maksimami jak `dekoduj`."""
    if output.shape[0] != 1:
        raise ValueError("top1_metrics wymaga jednego obrazu")
    _, height, width, _ = output.shape
    probability = output[..., 0].float().sigmoid()
    maxima = F.max_pool2d(probability[:, None], 3, 1, 1)[:, 0]
    peaks = probability * (probability == maxima)
    token = int(peaks.reshape(-1).argmax().item())
    row, column = divmod(token, width)
    distances = output.reshape(-1, 5)[token, 1:].float().clamp(-20, 20).exp() * STRIDE
    center_x = (column + 0.5) * STRIDE
    center_y = (row + 0.5) * STRIDE
    box = torch.stack(
        [
            distances.new_tensor(center_x) - distances[0],
            distances.new_tensor(center_y) - distances[1],
            distances.new_tensor(center_x) + distances[2],
            distances.new_tensor(center_y) + distances[3],
        ]
    )
    gt = ground_truth.to(device=box.device, dtype=box.dtype)
    return {
        "confidence": float(probability.reshape(-1)[token].item()),
        "iou": float(_iou_one(box, gt).item()),
        "bbox_xyxy": [float(v) for v in box.cpu().tolist()],
        "token_index": token,
        "token_row": row,
        "token_col": column,
    }


def _giou_ltrb(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (
        torch.minimum(predicted[0], target[0]) + torch.minimum(predicted[2], target[2])
    ) * (
        torch.minimum(predicted[1], target[1]) + torch.minimum(predicted[3], target[3])
    )
    area_predicted = (predicted[0] + predicted[2]) * (predicted[1] + predicted[3])
    area_target = (target[0] + target[2]) * (target[1] + target[3])
    union = area_predicted + area_target - intersection
    cover = (
        torch.maximum(predicted[0], target[0]) + torch.maximum(predicted[2], target[2])
    ) * (
        torch.maximum(predicted[1], target[1]) + torch.maximum(predicted[3], target[3])
    )
    return intersection / union.clamp_min(1e-9) - (cover - union) / cover.clamp_min(1e-9)


def _fit_objective(
    output: torch.Tensor,
    changed: torch.Tensor,
    baseline: torch.Tensor,
    target_index: int,
    target_ltrb: torch.Tensor,
) -> torch.Tensor:
    flat = output.reshape(-1, 5)
    logits = flat[:, 0]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[target_index] = False
    positive = F.softplus(5.5 - logits[target_index])
    negative = F.softplus(logits[mask] + 2.0).mean()
    rank = F.cross_entropy(logits[None], logits.new_tensor([target_index], dtype=torch.long))
    predicted = flat[target_index, 1:].clamp(-20, 20).exp()
    box = F.smooth_l1_loss(flat[target_index, 1:], target_ltrb.log())
    giou = _giou_ltrb(predicted, target_ltrb)
    regularization = 1e-6 * (changed - baseline).square().mean()
    return positive + negative + 0.5 * rank + 20.0 * box + 10.0 * (1.0 - giou) + regularization


def _target_box(
    output: torch.Tensor, target_index: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = output.reshape(-1, 5)
    row, column = divmod(target_index, width)
    distances = flat[target_index, 1:].clamp(-20, 20).exp() * STRIDE
    center_x = distances.new_tensor((column + 0.5) * STRIDE)
    center_y = distances.new_tensor((row + 0.5) * STRIDE)
    box = torch.stack(
        [
            center_x - distances[0],
            center_y - distances[1],
            center_x + distances[2],
            center_y + distances[3],
        ]
    )
    return box, flat[target_index, 0], flat[:, 0]


def _minimum_norm_objective(
    output: torch.Tensor,
    changed: torch.Tensor,
    baseline: torch.Tensor,
    ground_truth: torch.Tensor,
    target_index: int,
) -> torch.Tensor:
    width = output.shape[2]
    box, target_logit, logits = _target_box(output, target_index, width)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[target_index] = False
    other_max = logits[mask].max()
    iou = _iou_one(box, ground_truth.to(box))
    confidence_margin = F.relu(target_logit.new_tensor(2.442347) - target_logit).square()
    rank_margin = F.relu(other_max + 0.25 - target_logit).square()
    iou_margin = F.relu(iou.new_tensor(0.905) - iou).square()
    norm = (changed - baseline).square().mean()
    return norm + 200.0 * confidence_margin + 100.0 * rank_margin + 500.0 * iou_margin


def optimize_latent(
    net: torch.nn.Module,
    baseline_latent: torch.Tensor,
    baseline_block_output: torch.Tensor,
    ground_truth: torch.Tensor,
    target: dict,
    *,
    horizon: int = 500,
    milestones: tuple[int, ...] = (100, 500),
    control_tokens: tuple[int, ...] = (),
    confidence_threshold: float = 0.90,
    iou_threshold: float = 0.90,
) -> dict:
    """Wykonuje wspólny rzeczywisty prefix Adama; nigdy nie dopełnia trajektorii."""
    if not (1 <= horizon <= 500):
        raise ValueError(horizon)
    milestones = tuple(sorted(set(int(step) for step in milestones if 1 <= step <= horizon)))
    target_index = int(target["token_index"])
    target_ltrb = torch.as_tensor(
        target["target_ltrb"], device=baseline_latent.device, dtype=torch.float32
    )
    ground_truth = ground_truth.to(device=baseline_latent.device, dtype=torch.float32)
    changed = baseline_latent.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([changed], lr=learning_rate(1, horizon))

    telemetry: dict[str, list] = {
        "step": [],
        "learning_rate": [],
        "phase_min_norm": [],
        "fit_objective": [],
        "confidence": [],
        "iou": [],
        "delta_rms": [],
        "delta_l2": [],
        "gradient_rms": [],
        "top_token": [],
        "target_is_top": [],
        "success": [],
    }
    trajectory: dict[int, np.ndarray] = {}
    selected = tuple(int(token) for token in control_tokens)
    if any(token < 0 or token >= baseline_latent.shape[1] for token in selected):
        raise ValueError(selected)
    control_z = np.empty((horizon + 1, len(selected), baseline_latent.shape[2]), np.float32)
    control_gradient = np.empty((horizon, len(selected), baseline_latent.shape[2]), np.float32)
    control_m1 = np.empty_like(control_gradient)
    control_m2 = np.empty_like(control_gradient)
    if selected:
        control_z[0] = baseline_latent[0, selected].detach().cpu().numpy()

    first_success_step: int | None = None
    best_success: torch.Tensor | None = None
    best_success_step: int | None = None
    best_success_norm = float("inf")
    best_failure = baseline_latent.detach().clone()
    best_failure_step = 0
    best_failure_quality = (-float("inf"), -float("inf"), -float("inf"))

    def inspect(step: int, output: torch.Tensor, loss: torch.Tensor, gradient_rms: float) -> dict:
        nonlocal first_success_step, best_success, best_success_step, best_success_norm
        nonlocal best_failure, best_failure_step, best_failure_quality
        metrics = top1_metrics(output, ground_truth)
        difference = changed.detach() - baseline_latent
        rms = float(difference.square().mean().sqrt().item())
        l2 = float(difference.float().norm().item())
        success = metrics["confidence"] >= confidence_threshold and metrics["iou"] >= iou_threshold
        if success and first_success_step is None:
            first_success_step = step
        if success and rms < best_success_norm:
            best_success_norm = rms
            best_success = changed.detach().clone()
            best_success_step = step
        quality = (
            min(metrics["confidence"] / confidence_threshold, metrics["iou"] / iou_threshold),
            metrics["confidence"] + metrics["iou"],
            -float(loss.detach().item()),
        )
        if quality > best_failure_quality:
            best_failure_quality = quality
            best_failure = changed.detach().clone()
            best_failure_step = step
        telemetry["step"].append(step)
        telemetry["learning_rate"].append(0.0 if step == 0 else learning_rate(step, horizon))
        telemetry["phase_min_norm"].append(int(first_success_step is not None))
        telemetry["fit_objective"].append(float(loss.detach().item()))
        telemetry["confidence"].append(metrics["confidence"])
        telemetry["iou"].append(metrics["iou"])
        telemetry["delta_rms"].append(rms)
        telemetry["delta_l2"].append(l2)
        telemetry["gradient_rms"].append(gradient_rms)
        telemetry["top_token"].append(metrics["token_index"])
        telemetry["target_is_top"].append(int(metrics["token_index"] == target_index))
        telemetry["success"].append(int(success))
        return metrics

    output = fast_tail(net, baseline_latent, baseline_block_output, changed)
    fit_loss = _fit_objective(output, changed, baseline_latent, target_index, target_ltrb)
    baseline_metrics = inspect(0, output, fit_loss, 0.0)

    for step in range(1, horizon + 1):
        lr = learning_rate(step, horizon)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        output = fast_tail(net, baseline_latent, baseline_block_output, changed)
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
        if selected:
            state = optimizer.state.get(changed, {})
            zeros = torch.zeros_like(changed)
            control_gradient[step - 1] = changed.grad[0, selected].detach().cpu().numpy()
            control_m1[step - 1] = state.get("exp_avg", zeros)[0, selected].detach().cpu().numpy()
            control_m2[step - 1] = state.get("exp_avg_sq", zeros)[0, selected].detach().cpu().numpy()
        optimizer.step()
        with torch.no_grad():
            changed.clamp_(min=GELU_FLOOR)
        if selected:
            control_z[step] = changed[0, selected].detach().cpu().numpy()
        output = fast_tail(net, baseline_latent, baseline_block_output, changed)
        fit_loss = _fit_objective(output, changed, baseline_latent, target_index, target_ltrb)
        inspect(step, output, fit_loss, gradient_rms)
        if step in milestones:
            trajectory[step] = changed[0].detach().cpu().numpy().astype(np.float32, copy=True)

    after_tensor = best_success if best_success is not None else best_failure
    after_step = best_success_step if best_success is not None else best_failure_step
    status = "success" if best_success is not None else "budget_exhausted"
    telemetry_arrays = {
        key: np.asarray(
            values,
            dtype=(np.int32 if key in {"step", "phase_min_norm", "top_token", "target_is_top", "success"} else np.float32),
        )
        for key, values in telemetry.items()
    }
    return {
        "status": status,
        "baseline_metrics": baseline_metrics,
        "first_success_step": first_success_step,
        "after": after_tensor,
        "after_step": int(after_step),
        "final_state": changed.detach(),
        "trajectory_steps": np.asarray(sorted(trajectory), dtype=np.int32),
        "trajectory": np.stack([trajectory[step] for step in sorted(trajectory)], axis=0),
        "telemetry": telemetry_arrays,
        "control_tokens": np.asarray(selected, dtype=np.int32),
        "control_z": control_z,
        "control_gradient": control_gradient,
        "control_adam_m1_before": control_m1,
        "control_adam_m2_before": control_m2,
    }


def write_artifact_atomic(
    path: Path,
    before: np.ndarray,
    after: np.ndarray,
    delta: np.ndarray,
    trajectory: np.ndarray,
    trajectory_steps: list[int],
    metadata: dict,
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    """Zapisuje kompletny NPZ i publikuje go pojedynczym `os.replace`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.frombuffer(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"),
        dtype=np.uint8,
    )
    handle = tempfile.NamedTemporaryFile(
        mode="w+b", suffix=".npz.tmp", prefix=path.name + ".", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            arrays = {
                "before": np.asarray(before, dtype=np.float32),
                "after": np.asarray(after, dtype=np.float32),
                "delta": np.asarray(delta, dtype=np.float32),
                "trajectory": np.asarray(trajectory, dtype=np.float32),
                "trajectory_steps": np.asarray(trajectory_steps, dtype=np.int32),
                "metadata_json_utf8": encoded,
            }
            if extras:
                overlap = set(arrays).intersection(extras)
                if overlap:
                    raise ValueError(f"Powtórzone klucze artefaktu: {sorted(overlap)}")
                arrays.update({name: np.asarray(value) for name, value in extras.items()})
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_artifact(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name].copy() for name in data.files if name != "metadata_json_utf8"}
        result["metadata"] = json.loads(data["metadata_json_utf8"].tobytes().decode("utf-8"))
    return result
