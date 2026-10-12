"""Rozstrzyga oracle per-image; obala go brak obu progów w kroku 0-500."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from dane import LocalImageSource, TarImageSource, decode_resize, load_manifest
from model_snapshot import MalyViT, dekoduj
from pelne_latenty import (
    GELU_FLOOR,
    capture_baseline,
    choose_target_token,
    fast_tail,
    load_artifact,
    optimize_latent,
    state_fingerprint,
    top1_metrics,
    write_artifact_atomic,
)

EXPECTED_CHECKPOINT_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
EXPECTED_MANIFEST_SHA256 = "dd70797c916fe9223755db97b949b11f13709f45f1118e65c4fb3ab646d39ca0"
MILESTONES = (100, 500)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def detection_record(output: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    top = top1_metrics(output, ground_truth)
    boxes, scores = dekoduj(output.detach(), k=10)
    return {
        **top,
        "all_boxes_xyxy": boxes[0].cpu().numpy().astype(float).tolist(),
        "all_scores": scores[0].cpu().numpy().astype(float).tolist(),
    }


def load_model(checkpoint: Path, device: torch.device) -> tuple[MalyViT, str]:
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Niewłaściwy checkpoint {checkpoint_hash}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("krok") != 1000 or set(payload) != {"net", "krok"}:
        raise RuntimeError(f"Niewłaściwy kontener checkpointa: {payload.keys()}")
    model = MalyViT().to(device).float().eval().requires_grad_(False)
    model.load_state_dict(payload["net"], strict=True)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Parametr modelu pozostał trenowalny")
    return model, state_fingerprint(model)


def validate_resume(record_path: Path, artifact_path: Path) -> bool:
    if not record_path.exists() and not artifact_path.exists():
        return False
    if not record_path.exists() or not artifact_path.exists():
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Obcy checkpoint w {record_path}")
    if record.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Obcy manifest w {record_path}")
    if record.get("artifact_sha256") != sha256(artifact_path):
        raise RuntimeError(f"Uszkodzony artefakt {artifact_path}")
    artifact = load_artifact(artifact_path)
    if artifact["before"].shape != (600, 768) or artifact["after"].shape != (600, 768):
        raise RuntimeError(f"Zły kształt {artifact_path}")
    if any(artifact[name].dtype != np.float32 for name in ("before", "after", "delta", "trajectory")):
        raise RuntimeError(f"Zła precyzja FP32 w {artifact_path}")
    if not np.array_equal(artifact["delta"], artifact["after"] - artifact["before"]):
        raise RuntimeError(f"Delta nie jest bitowo równa after-before w {artifact_path}")
    if artifact["trajectory_steps"].tolist() != [100, 500]:
        raise RuntimeError(f"Złe milestone'y w {artifact_path}")
    if not np.array_equal(artifact["telemetry_step"], np.arange(501, dtype=np.int32)):
        raise RuntimeError(f"Niepełna rzeczywista oś czasu w {artifact_path}")
    if artifact["metadata"].get("horizon") != 500:
        raise RuntimeError(f"Zły horyzont w {artifact_path}")
    if artifact["metadata"].get("index") != record.get("index"):
        raise RuntimeError(f"Niespójna identyfikacja {artifact_path}")
    return True


def parse_indices(args, count: int) -> list[int]:
    if args.indices_json:
        indices = [int(value) for value in json.loads(args.indices_json.read_text(encoding="utf-8"))]
    else:
        if args.shard_id is None or args.num_shards is None:
            raise ValueError("Wymagane --indices-json albo --shard-id i --num-shards")
        indices = list(range(args.shard_id, count, args.num_shards))
    if len(indices) != len(set(indices)) or any(index < 0 or index >= count for index in indices):
        raise ValueError(indices)
    return indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--local-image-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--indices-json", type=Path)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--control-panel", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Ten bieg wymaga GPU CUDA")
    torch.set_num_threads(8)
    torch.manual_seed(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    manifest_hash = sha256(args.manifest)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Niewłaściwy manifest {manifest_hash}")
    rows = load_manifest(args.manifest)
    if len(rows) != 1500:
        raise RuntimeError(f"Manifest ma {len(rows)} wierszy")
    indices = parse_indices(args, len(rows))
    control_panel = {
        int(index): tuple(int(token) for token in tokens)
        for index, tokens in json.loads(args.control_panel.read_text(encoding="utf-8")).items()
    }
    image_source = LocalImageSource(args.local_image_root) if args.local_image_root else TarImageSource(args.archive)
    model, model_hash = load_model(args.checkpoint, device)

    shard_name = "selected" if args.indices_json else f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": args.mode,
        "indices": indices,
        "shard": shard_name,
        "horizon": args.horizon,
        "milestones": [step for step in MILESTONES if step <= args.horizon],
        "schedule": [[1, 100, 0.05], [101, 250, 0.15], [251, 400, 0.05], [401, 500, 0.01]],
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "precision": "FP32; TF32 disabled",
        "gelu_floor": GELU_FLOOR,
        "confidence_threshold": 0.90,
        "iou_threshold": 0.90,
        "trajectory_policy": "real states only; k0=before; no repeated terminal padding",
    }
    config_path = output / f"RUN_CONFIG_{shard_name}.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError(f"Niezgodna istniejąca konfiguracja {config_path}")
    if not config_path.exists():
        json_atomic(config_path, config)

    started = time.perf_counter()
    completed = 0
    skipped = 0
    summaries = []
    try:
        for ordinal, index in enumerate(indices, start=1):
            row = rows[index]
            artifact_path = output / "latents" / f"{index:06d}.npz"
            record_path = output / "records" / f"{index:06d}.json"
            if validate_resume(record_path, artifact_path):
                skipped += 1
                print(json.dumps({"resume_skip": index, "ordinal": ordinal, "total": len(indices)}), flush=True)
                continue
            image_started = time.perf_counter()
            try:
                payload = image_source.read_bytes(row["image_name"])
                if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
                    raise RuntimeError(f"Hash obrazu {index} nie zgadza się z manifestem")
                rgb = decode_resize(payload)
                image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32) / 255.0
                ground_truth = torch.tensor(row["bbox_xyxy_480"], device=device, dtype=torch.float32)
                torch.cuda.reset_peak_memory_stats(device)
                baseline_output, z0, q0 = capture_baseline(model, image)
                target = choose_target_token(np.asarray(row["bbox_xyxy_480"]), 20, 30)
                torch.cuda.synchronize(device)
                optimize_started = time.perf_counter()
                optimized = optimize_latent(
                    model,
                    z0,
                    q0,
                    ground_truth,
                    target,
                    horizon=args.horizon,
                    milestones=tuple(step for step in MILESTONES if step <= args.horizon),
                    control_tokens=control_panel.get(index, ()),
                )
                torch.cuda.synchronize(device)
                optimize_seconds = time.perf_counter() - optimize_started
                before = z0[0].detach().cpu().numpy().astype(np.float32, copy=True)
                after = optimized["after"][0].detach().cpu().numpy().astype(np.float32, copy=True)
                delta = (after - before).astype(np.float32, copy=False)
                metadata = {
                    "index": index,
                    "image_id": row["image_id"],
                    "annotation_id": row["annotation_id"],
                    "image_name": row["image_name"],
                    "image_sha256": row["image_sha256"],
                    "frame_id": row["frame_id"],
                    "bbox_xyxy_480": row["bbox_xyxy_480"],
                    "target": {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in target.items()
                    },
                    "status": optimized["status"],
                    "first_success_step": optimized["first_success_step"],
                    "after_step": optimized["after_step"],
                    "horizon": args.horizon,
                    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "weights_buffers_hash": model_hash,
                }
                extras = {f"telemetry_{key}": value for key, value in optimized["telemetry"].items()}
                if len(optimized["control_tokens"]):
                    extras.update(
                        control_tokens=optimized["control_tokens"],
                        control_z=optimized["control_z"],
                        control_gradient_before_step=optimized["control_gradient"],
                        control_adam_m1_before_step=optimized["control_adam_m1_before"],
                        control_adam_m2_before_step=optimized["control_adam_m2_before"],
                    )
                serialize_started = time.perf_counter()
                write_artifact_atomic(
                    artifact_path,
                    before,
                    after,
                    delta,
                    optimized["trajectory"],
                    optimized["trajectory_steps"].tolist(),
                    metadata,
                    extras=extras,
                )
                serialize_seconds = time.perf_counter() - serialize_started
                verify_started = time.perf_counter()
                restored = load_artifact(artifact_path)
                if not np.array_equal(restored["after"], after):
                    raise RuntimeError("Ponowne wczytanie after nie jest bitowo zgodne")
                restored_tensor = torch.from_numpy(restored["after"])[None].to(device)
                hook = model.bloki[-1].mlp.act.register_forward_hook(
                    lambda _module, _args, _output: restored_tensor
                )
                try:
                    with torch.no_grad():
                        actual_output, _ = model(image)
                finally:
                    hook.remove()
                expected_output = fast_tail(model, z0, q0, restored_tensor)
                max_tail_error = float((actual_output - expected_output).abs().max().item())
                torch.testing.assert_close(actual_output, expected_output, rtol=2e-5, atol=2e-5)
                baseline = detection_record(baseline_output, ground_truth)
                final = detection_record(actual_output, ground_truth)
                actual_success = final["confidence"] >= 0.90 and final["iou"] >= 0.90
                if actual_success != (optimized["status"] == "success"):
                    raise RuntimeError(
                        f"Status fast tail i hook-forward różnią się dla {index}: {optimized['status']} {final}"
                    )
                if any(parameter.grad is not None for parameter in model.parameters()):
                    raise RuntimeError("Gradient wyciekł do wag modelu")
                verify_seconds = time.perf_counter() - verify_started
                artifact_hash = sha256(artifact_path)
                record = {
                    **metadata,
                    "baseline": baseline,
                    "after": final,
                    "both90": actual_success,
                    "artifact": str(artifact_path.relative_to(output)).replace("\\", "/"),
                    "artifact_sha256": artifact_hash,
                    "artifact_bytes": artifact_path.stat().st_size,
                    "trajectory_steps": optimized["trajectory_steps"].tolist(),
                    "latent_reload_bitwise_equal": True,
                    "full_forward_tail_max_abs_error": max_tail_error,
                    "model_parameters_require_grad_false": True,
                    "optimizer_updates_model": False,
                    "timing_seconds": {
                        "optimization": optimize_seconds,
                        "serialization": serialize_seconds,
                        "reload_and_hook_verification": verify_seconds,
                        "total": time.perf_counter() - image_started,
                        "per_real_step": optimize_seconds / args.horizon,
                    },
                    "gpu_peak_bytes": {
                        "allocated": int(torch.cuda.max_memory_allocated(device)),
                        "reserved": int(torch.cuda.max_memory_reserved(device)),
                    },
                }
                json_atomic(record_path, record)
                completed += 1
                summaries.append(record)
                progress = {
                    "status": "running",
                    "shard": shard_name,
                    "completed_this_invocation": completed,
                    "skipped_valid": skipped,
                    "last_index": index,
                    "total_assigned": len(indices),
                    "elapsed_seconds": time.perf_counter() - started,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                }
                json_atomic(output / f"PROGRESS_{shard_name}.json", progress)
                print(
                    json.dumps(
                        {
                            "index": index,
                            "ordinal": ordinal,
                            "total": len(indices),
                            "status": optimized["status"],
                            "first_success_step": optimized["first_success_step"],
                            "after_step": optimized["after_step"],
                            "baseline": [baseline["confidence"], baseline["iou"]],
                            "after": [final["confidence"], final["iou"]],
                            "seconds": record["timing_seconds"],
                            "artifact_bytes": record["artifact_bytes"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception:
                failure = {
                    "index": index,
                    "row": row,
                    "traceback": traceback.format_exc(),
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                }
                json_atomic(output / "failures" / f"{index:06d}.json", failure)
                raise
    finally:
        image_source.close()

    ending_hash = state_fingerprint(model)
    if ending_hash != model_hash:
        raise RuntimeError(f"Wagi lub bufory zmieniły się: {model_hash} -> {ending_hash}")
    all_records = [
        json.loads((output / "records" / f"{index:06d}.json").read_text(encoding="utf-8"))
        for index in indices
    ]
    summary = {
        "status": "complete",
        "mode": args.mode,
        "shard": shard_name,
        "assigned": len(indices),
        "completed_this_invocation": completed,
        "skipped_valid": skipped,
        "successes": sum(bool(record["both90"]) for record in all_records),
        "budget_exhausted": sum(not bool(record["both90"]) for record in all_records),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "weights_buffers_hash_before": model_hash,
        "weights_buffers_hash_after": ending_hash,
        "weights_buffers_unchanged": True,
        "device": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "elapsed_seconds": time.perf_counter() - started,
        "artifact_bytes": sum(int(record["artifact_bytes"]) for record in all_records),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    json_atomic(output / f"SUMMARY_{shard_name}.json", summary)
    json_atomic(output / f"PROGRESS_{shard_name}.json", summary)
    print("COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
