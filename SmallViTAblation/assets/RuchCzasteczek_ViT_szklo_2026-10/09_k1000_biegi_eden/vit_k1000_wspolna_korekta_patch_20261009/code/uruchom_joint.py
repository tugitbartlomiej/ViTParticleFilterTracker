"""Rozstrzyga, czy jedna joint delta 600x768 przebija delta-zero na tych samych 1500 obrazach."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dane import TarImageSource, decode_resize, load_manifest
from metryki_source import detection_map, iou
from model_snapshot import MalyViT, dekoduj
from wspolna_korekta import (
    DIMENSIONS,
    EXPECTED_BASELINE,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    GELU_FLOOR,
    TOKENS,
    atomic_torch_save,
    capture_prefix,
    exact_mean_backward,
    fast_tail,
    joint_loss_per_image,
    json_atomic,
    learning_rate,
    load_training_checkpoint,
    sha256,
    should_stop_for_plateau,
    state_fingerprint,
    tensor_sha256,
)

SEED = 20261009
REGULARIZATION_WEIGHT = 1e-6
MAX_PASSES = 100
EVAL_EVERY = 5
MIN_MAP_IMPROVEMENT = 1e-5
MIN_PASSES = 30
PATIENCE_EVALUATIONS = 6
BASELINE_TOLERANCE = 1e-12
FALSE_POSITIVE_SCORE_THRESHOLDS = (0.5, 0.9)


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def load_model(checkpoint: Path, device: torch.device) -> tuple[MalyViT, str]:
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Niewłaściwy K1000: {checkpoint_hash}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"net", "krok"} or int(payload["krok"]) != 1000:
        raise RuntimeError("Nieprawidłowy kontener K1000")
    model = MalyViT().float().eval().requires_grad_(False)
    model.load_state_dict(payload["net"], strict=True)
    model = model.to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Model nie został zamrożony")
    return model, state_fingerprint(model)


def _image_tensor(source: TarImageSource, row: dict, device: torch.device) -> torch.Tensor:
    payload = source.read_bytes(row["image_name"])
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != row["image_sha256"]:
        raise RuntimeError(f"Hash obrazu {row['index']} nie zgadza się z manifestem")
    rgb = decode_resize(payload)
    return (
        torch.from_numpy(rgb.copy())
        .permute(2, 0, 1)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def _target_from_box(box: list[float]) -> dict:
    from wspolna_korekta import choose_target_token

    return choose_target_token(np.asarray(box, dtype=np.float64), 20, 30)


def _record_batch(
    output: torch.Tensor,
    ground_truth: torch.Tensor,
    target_index: torch.Tensor,
    rows: list[dict],
) -> list[dict]:
    boxes, scores = dekoduj(output.detach(), k=10)
    output_np = output.detach().float().cpu().contiguous().numpy()
    probability = output[..., 0].float().sigmoid()
    maxima = F.max_pool2d(probability[:, None], 3, 1, 1)[:, 0]
    peaks = probability * (probability == maxima)
    top_tokens = peaks.reshape(output.shape[0], -1).argmax(dim=1)
    flat_probability = probability.reshape(output.shape[0], -1)
    batch_ids = torch.arange(output.shape[0], device=output.device)
    target_confidence = flat_probability[batch_ids, target_index]

    boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
    scores_np = scores.detach().cpu().numpy().astype(np.float32)
    gt_np = ground_truth.detach().cpu().numpy().astype(np.float32)
    target_conf_np = target_confidence.detach().cpu().numpy().astype(np.float64)
    top_tokens_np = top_tokens.detach().cpu().numpy()
    records: list[dict] = []
    for local, row in enumerate(rows):
        top_iou = float(iou(boxes_np[local, 0], gt_np[local]))
        false_positives: dict[str, int] = {}
        true_positives: dict[str, int] = {}
        for threshold in FALSE_POSITIVE_SCORE_THRESHOLDS:
            selected = np.flatnonzero(scores_np[local] >= threshold)
            matched = False
            fp = 0
            for detection_index in selected:
                overlap = float(iou(boxes_np[local, detection_index], gt_np[local]))
                if not matched and overlap >= 0.5:
                    matched = True
                else:
                    fp += 1
            key = f"score_{threshold:.1f}_iou_0.5"
            false_positives[key] = int(fp)
            true_positives[key] = int(matched)
        records.append(
            {
                "index": int(row["index"]),
                "image_id": int(row["image_id"]),
                "frame_id": row["frame_id"],
                "image_sha256": row["image_sha256"],
                "confidence": float(scores_np[local, 0]),
                "iou": top_iou,
                "both90": bool(scores_np[local, 0] >= 0.90 and top_iou >= 0.90),
                "top1_token_index": int(top_tokens_np[local]),
                "target_token_index": int(target_index[local].item()),
                "top1_is_target_token": bool(top_tokens_np[local] == int(target_index[local].item())),
                "target_token_confidence": float(target_conf_np[local]),
                "false_positives": false_positives,
                "true_positives": true_positives,
                "all_boxes_xyxy": boxes_np[local].astype(float).tolist(),
                "all_scores": scores_np[local].astype(float).tolist(),
                "output_tensor_sha256": _output_array_sha256(output_np[local : local + 1]),
            }
        )
    return records


def _output_array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"{q:.2f}": float(np.quantile(values, q))
        for q in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    }


def _comparison(records: list[dict], baseline: list[dict], tolerance: float = 1e-6) -> dict:
    if [record["index"] for record in records] != [record["index"] for record in baseline]:
        raise RuntimeError("Porównanie per-image ma różne indeksy")
    confidence_delta = np.asarray(
        [record["confidence"] - base["confidence"] for record, base in zip(records, baseline)],
        dtype=np.float64,
    )
    iou_delta = np.asarray(
        [record["iou"] - base["iou"] for record, base in zip(records, baseline)],
        dtype=np.float64,
    )

    def counts(values: np.ndarray) -> dict[str, int]:
        return {
            "improved": int((values > tolerance).sum()),
            "unchanged": int((np.abs(values) <= tolerance).sum()),
            "worse": int((values < -tolerance).sum()),
        }

    unchanged = (np.abs(confidence_delta) <= tolerance) & (np.abs(iou_delta) <= tolerance)
    pareto_improved = (
        (confidence_delta >= -tolerance) & (iou_delta >= -tolerance) & ~unchanged
    )
    pareto_worse = (
        (confidence_delta <= tolerance) & (iou_delta <= tolerance) & ~unchanged
    )
    mixed = ~(unchanged | pareto_improved | pareto_worse)
    return {
        "confidence": counts(confidence_delta),
        "iou": counts(iou_delta),
        "joint": {
            "pareto_improved": int(pareto_improved.sum()),
            "unchanged": int(unchanged.sum()),
            "pareto_worse": int(pareto_worse.sum()),
            "mixed": int(mixed.sum()),
        },
        "delta_confidence_quantiles": _quantiles(confidence_delta),
        "delta_iou_quantiles": _quantiles(iou_delta),
    }


def summarize_records(
    records: list[dict],
    ground_truth: torch.Tensor,
    objective: dict,
    delta: torch.Tensor,
    baseline_records: list[dict] | None = None,
) -> dict:
    detections = [
        (
            np.asarray(record["all_boxes_xyxy"], dtype=np.float32),
            np.asarray(record["all_scores"], dtype=np.float32),
        )
        for record in records
    ]
    gt_np = ground_truth.detach().cpu().numpy().astype(np.float32)
    gt_list = [np.asarray([box], dtype=np.float32) for box in gt_np]
    confidence = np.asarray([record["confidence"] for record in records], dtype=np.float64)
    overlaps = np.asarray([record["iou"] for record in records], dtype=np.float64)
    summary = {
        "count": len(records),
        "detection": detection_map(detections, gt_list),
        "confidence_quantiles": _quantiles(confidence),
        "iou_quantiles": _quantiles(overlaps),
        "both90_count": int(sum(record["both90"] for record in records)),
        "top1_is_target_token_count": int(sum(record["top1_is_target_token"] for record in records)),
        "target_token_confidence_quantiles": _quantiles(
            np.asarray([record["target_token_confidence"] for record in records], np.float64)
        ),
        "false_positives": {
            key: {
                "count": int(sum(record["false_positives"][key] for record in records)),
                "images_with_any": int(sum(record["false_positives"][key] > 0 for record in records)),
                "true_positive_count": int(sum(record["true_positives"][key] for record in records)),
            }
            for key in records[0]["false_positives"]
        },
        "objective": objective,
        "delta": {
            "sha256": tensor_sha256(delta),
            "l2": float(delta.detach().float().norm().item()),
            "rms": float(delta.detach().float().square().mean().sqrt().item()),
            "max_abs": float(delta.detach().float().abs().max().item()),
            "nonzero": int(torch.count_nonzero(delta.detach()).item()),
        },
    }
    if baseline_records is not None:
        summary["comparison_vs_baseline"] = _comparison(records, baseline_records)
    return summary


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _cache_slice(cache: dict, start: int, end: int, device: torch.device) -> dict:
    return {
        "z0": cache["z0"][start:end].to(device=device, non_blocking=True),
        "q0": cache["q0"][start:end].to(device=device, non_blocking=True),
        "gt": cache["gt"][start:end].to(device=device, non_blocking=True),
        "target_index": cache["target_index"][start:end].to(device=device, non_blocking=True),
        "target_ltrb": cache["target_ltrb"][start:end].to(device=device, non_blocking=True),
        "rows": cache["rows"][start:end],
    }


def build_cache(
    model: MalyViT,
    rows: list[dict],
    archive: Path,
    device: torch.device,
    capture_batch_size: int,
    prefer_cuda_cache: bool,
) -> dict:
    started = time.perf_counter()
    count = len(rows)
    bytes_estimated = count * TOKENS * (DIMENSIONS + 192) * 4
    free_gpu, total_gpu = torch.cuda.mem_get_info(device)
    cache_on_cuda = bool(
        prefer_cuda_cache and bytes_estimated + (4 << 30) < free_gpu * 0.8
    )
    cache_device = device if cache_on_cuda else torch.device("cpu")
    z0 = torch.empty((count, TOKENS, DIMENSIONS), dtype=torch.float32, device=cache_device)
    q0 = torch.empty((count, TOKENS, 192), dtype=torch.float32, device=cache_device)
    gt = torch.empty((count, 4), dtype=torch.float32, device=cache_device)
    target_index = torch.empty((count,), dtype=torch.long, device=cache_device)
    target_ltrb = torch.empty((count, 4), dtype=torch.float32, device=cache_device)
    baseline_records: list[dict] = []
    objective_sums = {name: 0.0 for name in ("positive", "negative", "rank", "box", "giou")}
    objective_total = 0.0
    zero_tail_max_abs_error = 0.0
    torch.cuda.reset_peak_memory_stats(device)
    source = TarImageSource(archive)
    try:
        for start in range(0, count, capture_batch_size):
            end = min(start + capture_batch_size, count)
            batch_rows = rows[start:end]
            images = torch.stack([_image_tensor(source, row, device) for row in batch_rows])
            output, batch_z0, batch_q0 = capture_prefix(model, images)
            targets = [_target_from_box(row["bbox_xyxy_480"]) for row in batch_rows]
            batch_gt = torch.tensor(
                [row["bbox_xyxy_480"] for row in batch_rows], device=device, dtype=torch.float32
            )
            batch_target_index = torch.tensor(
                [target["token_index"] for target in targets], device=device, dtype=torch.long
            )
            batch_target_ltrb = torch.tensor(
                np.stack([target["target_ltrb"] for target in targets]),
                device=device,
                dtype=torch.float32,
            )
            zero = batch_z0.new_zeros((TOKENS, DIMENSIONS))
            zero_output = fast_tail(model, batch_z0, batch_q0, zero)
            zero_tail_max_abs_error = max(
                zero_tail_max_abs_error, float((zero_output - output).abs().max().item())
            )
            losses, components = joint_loss_per_image(output, batch_target_index, batch_target_ltrb)
            objective_total += float(losses.sum().item())
            for name, values in components.items():
                objective_sums[name] += float(values.sum().item())
            z0[start:end].copy_(batch_z0.to(cache_device))
            q0[start:end].copy_(batch_q0.to(cache_device))
            gt[start:end].copy_(batch_gt.to(cache_device))
            target_index[start:end].copy_(batch_target_index.to(cache_device))
            target_ltrb[start:end].copy_(batch_target_ltrb.to(cache_device))
            baseline_records.extend(
                _record_batch(output, batch_gt, batch_target_index, batch_rows)
            )
            print(
                json.dumps({"cache": [end, count], "elapsed_seconds": time.perf_counter() - started}),
                flush=True,
            )
    finally:
        source.close()
    torch.cuda.synchronize(device)
    cache = {
        "z0": z0,
        "q0": q0,
        "gt": gt,
        "target_index": target_index,
        "target_ltrb": target_ltrb,
        "rows": rows,
        "baseline_records": baseline_records,
        "baseline_objective": {
            "data_loss": objective_total / count,
            "regularization": 0.0,
            "total_loss": objective_total / count,
            "components": {name: value / count for name, value in objective_sums.items()},
        },
        "cache_audit": {
            "device": str(cache_device),
            "estimated_bytes": bytes_estimated,
            "actual_tensor_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in (z0, q0, gt, target_index, target_ltrb)
            ),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_bytes": int(total_gpu),
            "gpu_free_before_bytes": int(free_gpu),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "build_seconds": time.perf_counter() - started,
            "disk_latent_cache": False,
            "zero_delta_fast_tail_max_abs_error": zero_tail_max_abs_error,
        },
    }
    return cache


@torch.no_grad()
def evaluate_cache(
    model: MalyViT,
    cache: dict,
    delta: torch.Tensor,
    device: torch.device,
    batch_size: int,
    baseline_records: list[dict] | None,
) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    count = len(cache["rows"])
    records: list[dict] = []
    loss_sum = 0.0
    component_sums = {name: 0.0 for name in ("positive", "negative", "rank", "box", "giou")}
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        batch = _cache_slice(cache, start, end, device)
        output = fast_tail(model, batch["z0"], batch["q0"], delta)
        losses, components = joint_loss_per_image(
            output, batch["target_index"], batch["target_ltrb"]
        )
        loss_sum += float(losses.sum().item())
        for name, values in components.items():
            component_sums[name] += float(values.sum().item())
        records.extend(
            _record_batch(output, batch["gt"], batch["target_index"], batch["rows"])
        )
    regularization = REGULARIZATION_WEIGHT * float(delta.float().square().mean().item())
    objective = {
        "data_loss": loss_sum / count,
        "regularization": regularization,
        "total_loss": loss_sum / count + regularization,
        "components": {name: value / count for name, value in component_sums.items()},
    }
    summary = summarize_records(
        records,
        cache["gt"],
        objective,
        delta,
        baseline_records=baseline_records,
    )
    summary["elapsed_seconds"] = time.perf_counter() - started
    return summary, records


def verify_full_baseline(summary: dict) -> None:
    if summary["count"] != 1500:
        raise RuntimeError("Pełna podłoga nie obejmuje 1500 obrazów")
    detection = summary["detection"]
    differences = {
        key: float(detection[key] - EXPECTED_BASELINE[key]) for key in EXPECTED_BASELINE
    }
    if any(abs(value) > BASELINE_TOLERANCE for value in differences.values()):
        raise RuntimeError(f"Delta-zero nie odtwarza historycznej podłogi: {differences}")


def _iter_training_batches(cache: dict, batch_size: int, device: torch.device):
    for start in range(0, len(cache["rows"]), batch_size):
        yield _cache_slice(cache, start, min(start + batch_size, len(cache["rows"])), device)


def _objective_for_batch(model: MalyViT):
    def objective(delta: torch.Tensor, batch: dict):
        output = fast_tail(model, batch["z0"], batch["q0"], delta)
        return joint_loss_per_image(output, batch["target_index"], batch["target_ltrb"])

    return objective


def _objective_no_grad(model: MalyViT, cache: dict, delta: torch.Tensor, device: torch.device) -> float:
    total = 0.0
    with torch.no_grad():
        for batch in _iter_training_batches(cache, len(cache["rows"]), device):
            output = fast_tail(model, batch["z0"], batch["q0"], delta)
            losses, _ = joint_loss_per_image(output, batch["target_index"], batch["target_ltrb"])
            total += float(losses.sum().item())
    return total / len(cache["rows"]) + REGULARIZATION_WEIGHT * float(
        delta.square().mean().item()
    )


def _hook_output(model: MalyViT, image: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    def replace(_module, _args, natural):
        return (natural + delta).clamp_min(GELU_FLOOR)

    hook = model.bloki[-1].mlp.act.register_forward_hook(replace)
    try:
        with torch.no_grad():
            output, _ = model(image)
    finally:
        hook.remove()
    return output


def run_smoke(args, model: MalyViT, model_hash: str, rows: list[dict], device: torch.device) -> dict:
    selected = json.loads(args.smoke_indices.read_text(encoding="utf-8"))
    smoke_rows = [rows[int(index)] for index in selected]
    cache = build_cache(
        model,
        smoke_rows,
        args.archive,
        device,
        capture_batch_size=len(smoke_rows),
        prefer_cuda_cache=True,
    )
    before_hash = state_fingerprint(model)
    zero = torch.zeros((TOKENS, DIMENSIONS), device=device, dtype=torch.float32)
    baseline_summary, _ = evaluate_cache(
        model, cache, zero, device, len(smoke_rows), cache["baseline_records"]
    )

    direct = nn.Parameter(zero.clone())
    direct_losses, _ = _objective_for_batch(model)(
        direct, _cache_slice(cache, 0, len(smoke_rows), device)
    )
    (direct_losses.mean() + REGULARIZATION_WEIGHT * direct.square().mean()).backward()
    direct_gradient = direct.grad.detach().clone()

    accumulated = nn.Parameter(zero.clone())
    uneven_batches = [
        _cache_slice(cache, 0, 3, device),
        _cache_slice(cache, 3, len(smoke_rows), device),
    ]
    accumulated_audit = exact_mean_backward(
        accumulated,
        uneven_batches,
        total_count=len(smoke_rows),
        objective_fn=_objective_for_batch(model),
        regularization_weight=REGULARIZATION_WEIGHT,
    )
    accumulated_gradient = accumulated.grad.detach().clone()
    gradient_max_abs_error = float((direct_gradient - accumulated_gradient).abs().max().item())
    gradient_relative_error = gradient_max_abs_error / max(
        float(direct_gradient.abs().max().item()), 1e-12
    )

    optimizer = torch.optim.Adam([accumulated], lr=0.01)
    before_objective = accumulated_audit["total_loss"]
    optimizer.step()
    after_objective = _objective_no_grad(model, cache, accumulated, device)
    first_step_improvement = before_objective - after_objective

    first = _cache_slice(cache, 0, 1, device)
    fast_output = fast_tail(model, first["z0"], first["q0"], accumulated)
    source = TarImageSource(args.archive)
    try:
        image = _image_tensor(source, smoke_rows[0], device)[None]
    finally:
        source.close()
    hook_output = _hook_output(model, image, accumulated.detach())
    hook_max_abs_error = float((fast_output - hook_output).abs().max().item())

    smoke_delta_path = args.output / "smoke" / "SMOKE_DELTA.pt"
    atomic_torch_save(
        smoke_delta_path,
        {
            "delta": accumulated.detach().cpu(),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
    )
    restored = torch.load(smoke_delta_path, map_location=device, weights_only=True)["delta"]
    restored_output = fast_tail(model, first["z0"], first["q0"], restored)
    fresh_load_max_abs_error = float((restored_output - fast_output).abs().max().item())
    after_hash = state_fingerprint(model)
    model_gradients_none = all(parameter.grad is None for parameter in model.parameters())
    checks = {
        "delta_zero_matches_natural_forward": cache["cache_audit"][
            "zero_delta_fast_tail_max_abs_error"
        ]
        <= 2e-6,
        "fast_tail_matches_hook_forward": hook_max_abs_error <= 2e-6,
        "uneven_microbatches_match_full_mean_gradient": gradient_relative_error <= 2e-5,
        "gradient_finite_nonzero": bool(
            torch.isfinite(accumulated_gradient).all() and torch.count_nonzero(accumulated_gradient) > 0
        ),
        "gradient_reaches_only_delta": model_gradients_none,
        "weights_buffers_unchanged": before_hash == after_hash == model_hash,
        "fresh_load_bitwise_equal": fresh_load_max_abs_error == 0.0,
        "fixed_objective_improves_after_first_adam_step": first_step_improvement > 0.0,
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "indices": selected,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "weights_buffers_hash": model_hash,
        "precision": "FP32; TF32 disabled",
        "baseline_subset": baseline_summary,
        "objective_before": before_objective,
        "objective_after_one_adam_step_lr_0.01": after_objective,
        "objective_improvement": first_step_improvement,
        "gradient": {
            "l2": float(accumulated_gradient.norm().item()),
            "max_abs": float(accumulated_gradient.abs().max().item()),
            "microbatch_vs_full_max_abs_error": gradient_max_abs_error,
            "microbatch_vs_full_relative_error": gradient_relative_error,
        },
        "fast_tail_hook_max_abs_error": hook_max_abs_error,
        "fresh_load_max_abs_error": fresh_load_max_abs_error,
        "cache": cache["cache_audit"],
    }
    json_atomic(args.output / "smoke" / "SMOKE_RESULT.json", result)
    if result["status"] != "pass":
        raise RuntimeError(f"Smoke nie przeszedł: {checks}")
    return result


def _checkpoint_state(
    completed_pass: int,
    delta: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
    history: list[dict],
    best_delta: torch.Tensor,
    best_summary: dict,
    best_pass: int,
    no_improvement: int,
    model_hash: str,
) -> dict:
    return {
        "format_version": 1,
        "completed_pass": completed_pass,
        "delta": delta.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "config_sha256": config_sha256,
        "history": history,
        "best_delta": best_delta.detach().cpu(),
        "best_summary": best_summary,
        "best_pass": best_pass,
        "evaluations_without_improvement": no_improvement,
        "weights_buffers_hash": model_hash,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }


def run_training(
    args,
    model: MalyViT,
    model_hash: str,
    rows: list[dict],
    device: torch.device,
    config: dict,
    config_sha256: str,
) -> dict:
    output = args.output / "training"
    output.mkdir(parents=True, exist_ok=True)
    cache = build_cache(
        model,
        rows,
        args.archive,
        device,
        capture_batch_size=args.capture_batch_size,
        prefer_cuda_cache=True,
    )
    zero = torch.zeros((TOKENS, DIMENSIONS), device=device, dtype=torch.float32)
    baseline_summary = summarize_records(
        cache["baseline_records"],
        cache["gt"],
        cache["baseline_objective"],
        zero,
    )
    verify_full_baseline(baseline_summary)
    baseline_summary["cache"] = cache["cache_audit"]
    json_atomic(output / "BASELINE.json", baseline_summary)
    write_jsonl(output / "BASELINE_RECORDS.jsonl", cache["baseline_records"])

    old_summary = json.loads(args.old_evaluation_summary.read_text(encoding="utf-8"))
    old_mean_reference = {
        "source": str(args.old_evaluation_summary),
        "source_sha256": sha256(args.old_evaluation_summary),
        "method": "mean_in_sample",
        "metrics": old_summary["methods"]["mean_in_sample"],
        "comparison": old_summary["comparisons"]["mean_in_sample"],
        "used_for_training": False,
    }
    json_atomic(output / "OLD_MEAN_A_REFERENCE.json", old_mean_reference)

    delta = nn.Parameter(zero.clone())
    optimizer = torch.optim.Adam([delta], lr=learning_rate(1))
    history: list[dict] = []
    best_delta = zero.clone()
    best_summary = baseline_summary
    best_pass = 0
    no_improvement = 0
    start_pass = 1
    latest_path = output / "CHECKPOINT_LATEST.pt"
    if latest_path.exists():
        loaded = load_training_checkpoint(latest_path, config_sha256)
        delta.data.copy_(loaded["delta"].to(device))
        optimizer.load_state_dict(loaded["optimizer"])
        torch.set_rng_state(loaded["torch_rng_state"])
        torch.cuda.set_rng_state_all(loaded["cuda_rng_state_all"])
        np.random.set_state(loaded["numpy_rng_state"])
        history = loaded["history"]
        best_delta = loaded["best_delta"].to(device)
        best_summary = loaded["best_summary"]
        best_pass = int(loaded["best_pass"])
        no_improvement = int(loaded["evaluations_without_improvement"])
        start_pass = int(loaded["completed_pass"]) + 1
        print(json.dumps({"resume_from_pass": start_pass - 1}), flush=True)
    else:
        atomic_torch_save(
            output / "BEST_DELTA.pt",
            {
                "delta": best_delta.cpu(),
                "selected_pass": 0,
                "selection_metric": "training_mAP50_95",
                "summary": best_summary,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "config_sha256": config_sha256,
            },
        )
        write_jsonl(output / "BEST_RECORDS.jsonl", cache["baseline_records"])

    stop_requested = {"value": False}

    def request_stop(_signal, _frame):
        stop_requested["value"] = True

    signal.signal(signal.SIGTERM, request_stop)
    training_started = time.perf_counter()
    stop_reason = "max_passes"
    last_summary = baseline_summary if start_pass == 1 else None
    last_records = cache["baseline_records"] if start_pass == 1 else None
    for full_pass in range(start_pass, MAX_PASSES + 1):
        pass_started = time.perf_counter()
        lr = learning_rate(full_pass)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        audit = exact_mean_backward(
            delta,
            _iter_training_batches(cache, args.microbatch_size, device),
            total_count=1500,
            objective_fn=_objective_for_batch(model),
            regularization_weight=REGULARIZATION_WEIGHT,
        )
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError("Niefinitywny gradient delty")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Gradient dotarł do zamrożonego K1000")
        gradient_l2 = float(delta.grad.norm().item())
        gradient_max_abs = float(delta.grad.abs().max().item())
        optimizer.step()
        if not torch.isfinite(delta).all():
            raise RuntimeError("Niefinitywna delta po Adam.step")
        pass_record = {
            "pass": full_pass,
            "lr": lr,
            "loss_before_update": audit,
            "gradient_l2": gradient_l2,
            "gradient_max_abs": gradient_max_abs,
            "delta_l2_after": float(delta.detach().norm().item()),
            "delta_rms_after": float(delta.detach().square().mean().sqrt().item()),
            "seconds": time.perf_counter() - pass_started,
            "full_dataset_examples": audit["examples"],
            "optimizer_updates_total": full_pass,
        }
        history.append(pass_record)

        if full_pass % EVAL_EVERY == 0 or full_pass == MAX_PASSES:
            summary, records = evaluate_cache(
                model,
                cache,
                delta.detach(),
                device,
                args.eval_batch_size,
                cache["baseline_records"],
            )
            summary.update(
                {
                    "pass": full_pass,
                    "selection_on_training_data": True,
                    "selection_metric": "mAP50_95",
                    "weights_buffers_hash": state_fingerprint(model),
                }
            )
            json_atomic(output / f"EVAL_PASS_{full_pass:03d}.json", summary)
            last_summary, last_records = summary, records
            current_map = float(summary["detection"]["mAP50_95"])
            best_map = float(best_summary["detection"]["mAP50_95"])
            if current_map > best_map + MIN_MAP_IMPROVEMENT:
                best_delta = delta.detach().clone()
                best_summary = summary
                best_pass = full_pass
                no_improvement = 0
                atomic_torch_save(
                    output / "BEST_DELTA.pt",
                    {
                        "delta": best_delta.cpu(),
                        "selected_pass": best_pass,
                        "selection_metric": "training_mAP50_95",
                        "summary": best_summary,
                        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                        "config_sha256": config_sha256,
                    },
                )
                write_jsonl(output / "BEST_RECORDS.jsonl", records)
            else:
                no_improvement += 1
            print(
                json.dumps(
                    {
                        "evaluation_pass": full_pass,
                        "mAP50_95": current_map,
                        "AP50": summary["detection"]["AP50"],
                        "AP75": summary["detection"]["AP75"],
                        "both90": summary["both90_count"],
                        "false_positives": summary["false_positives"],
                        "best_pass": best_pass,
                        "best_mAP50_95": best_summary["detection"]["mAP50_95"],
                        "no_improvement": no_improvement,
                    }
                ),
                flush=True,
            )
        atomic_torch_save(
            latest_path,
            _checkpoint_state(
                full_pass,
                delta,
                optimizer,
                config_sha256,
                history,
                best_delta,
                best_summary,
                best_pass,
                no_improvement,
                model_hash,
            ),
        )
        json_atomic(output / "PROGRESS.json", {"completed_pass": full_pass, "history": history})
        print(json.dumps({"pass_complete": full_pass, **pass_record}), flush=True)
        if stop_requested["value"]:
            stop_reason = "sigterm_after_complete_pass"
            break
        if should_stop_for_plateau(full_pass, no_improvement, MIN_PASSES, PATIENCE_EVALUATIONS):
            stop_reason = "plateau"
            break

    completed_pass = history[-1]["pass"] if history else start_pass - 1
    if last_summary is None or int(last_summary.get("pass", -1)) != completed_pass:
        last_summary, last_records = evaluate_cache(
            model,
            cache,
            delta.detach(),
            device,
            args.eval_batch_size,
            cache["baseline_records"],
        )
        last_summary["pass"] = completed_pass
    atomic_torch_save(
        output / "FINAL_DELTA.pt",
        {
            "delta": delta.detach().cpu(),
            "completed_pass": completed_pass,
            "summary": last_summary,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "config_sha256": config_sha256,
        },
    )
    if last_records is not None:
        write_jsonl(output / "FINAL_RECORDS.jsonl", last_records)
    result = {
        "status": "complete" if stop_reason != "sigterm_after_complete_pass" else "interrupted_resumable",
        "stop_reason": stop_reason,
        "completed_pass": completed_pass,
        "optimizer_updates": completed_pass,
        "best_pass": best_pass,
        "baseline": baseline_summary,
        "best": best_summary,
        "final": last_summary,
        "old_mean_A_reference": old_mean_reference,
        "history": history,
        "cache": cache["cache_audit"],
        "weights_buffers_hash_before": model_hash,
        "weights_buffers_hash_after": state_fingerprint(model),
        "weights_buffers_unchanged": state_fingerprint(model) == model_hash,
        "elapsed_seconds": time.perf_counter() - training_started,
        "selection_on_training_data": True,
        "generalization_claim": False,
    }
    json_atomic(output / "TRAINING_SUMMARY.json", result)
    if result["status"] != "complete":
        return result
    if not result["weights_buffers_unchanged"]:
        raise RuntimeError("K1000 zmienił się podczas treningu delty")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-indices", type=Path)
    parser.add_argument("--old-evaluation-summary", type=Path)
    parser.add_argument("--capture-batch-size", type=int, default=16)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.mode == "smoke" and args.smoke_indices is None:
        parser.error("--smoke-indices jest wymagane dla smoke")
    if args.mode == "train" and args.old_evaluation_summary is None:
        parser.error("--old-evaluation-summary jest wymagane dla train")
    if not torch.cuda.is_available():
        raise RuntimeError("Bieg wymaga CUDA")
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    manifest_hash = sha256(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Niewłaściwy manifest: {manifest_hash}")
    rows = load_manifest(args.manifest)
    if len(rows) != 1500:
        raise RuntimeError(f"Manifest ma {len(rows)} zamiast 1500")
    if [row["index"] for row in rows] != list(range(1500)):
        raise RuntimeError("Manifest ma niespójną kolejność")
    model, model_hash = load_model(args.checkpoint, device)
    config = {
        "family": "vit_k1000_wspolna_korekta_patch_20261009",
        "mode": args.mode,
        "seed": SEED,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "precision": "FP32; TF32 disabled",
        "delta_shape": [TOKENS, DIMENSIONS],
        "delta_initialization": "exact zeros",
        "gelu_floor": GELU_FLOOR,
        "loss": {
            "constant_for_all_images_and_passes": True,
            "positive": "softplus(5.5-target_logit)",
            "negative": "mean softplus(other_logits+2.0)",
            "top1_rank": "0.5*cross_entropy over 600 tokens",
            "box": "20*smooth_l1(log_ltrb)",
            "giou": "10*(1-GIoU)",
            "delta_regularization": REGULARIZATION_WEIGHT,
        },
        "optimizer": "one Adam over shared delta only",
        "full_dataset_gradient": True,
        "microbatch_size": args.microbatch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_passes": MAX_PASSES,
        "lr_schedule": [[1, 40, 0.01], [41, 70, 0.003], [71, 100, 0.001]],
        "evaluation_every_passes": EVAL_EVERY,
        "selection_metric": "training mAP50_95",
        "minimum_map_improvement": MIN_MAP_IMPROVEMENT,
        "plateau": {"min_passes": MIN_PASSES, "patience_evaluations": PATIENCE_EVALUATIONS},
        "selection_on_same_1500": True,
        "generalization_claim": False,
        "disk_image_or_latent_cache": False,
    }
    config_sha256 = canonical_sha(config)
    config["config_sha256"] = config_sha256
    config_path = args.output / f"RUN_CONFIG_{args.mode}.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(f"Istniejąca konfiguracja różni się: {config_path}")
    else:
        json_atomic(config_path, config)
    if args.mode == "smoke":
        result = run_smoke(args, model, model_hash, rows, device)
    else:
        result = run_training(args, model, model_hash, rows, device, config, config_sha256)
    print(json.dumps({"FINAL": result}, ensure_ascii=False), flush=True)
    return 0 if result.get("status") in {"pass", "complete"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
