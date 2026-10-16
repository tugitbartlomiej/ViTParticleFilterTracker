"""Liczy brakujące stany 900k cząsteczek; obala go rozjazd ze starym oracle."""
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

from dane import TarImageSource, decode_resize, load_manifest
from model_snapshot import MalyViT
from pelne_latenty import capture_baseline, choose_target_token, load_artifact, state_fingerprint
from replay_core import (
    PCA_BASIS_SHA256,
    basis_sha256,
    replay_latent,
    sha256,
    validate_panel_arrays,
    validate_replay_arrays,
    write_npz_atomic,
    encode_json,
)

EXPECTED_CHECKPOINT_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
EXPECTED_MANIFEST_SHA256 = "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7"


def json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_model(checkpoint: Path, device: torch.device) -> tuple[MalyViT, str]:
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Niewłaściwy checkpoint K1000")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"net", "krok"} or payload["krok"] != 1000:
        raise RuntimeError("Niewłaściwy kontener K1000")
    model = MalyViT().to(device).float().eval().requires_grad_(False)
    model.load_state_dict(payload["net"], strict=True)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Parametr modelu pozostał trenowalny")
    return model, state_fingerprint(model)


def load_old_index(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    result = {int(row["index"]): row for row in rows}
    if len(rows) != 1500 or len(result) != 1500 or set(result) != set(range(1500)):
        raise RuntimeError("Stary indeks oracle nie zawiera dokładnie 1500 unikalnych obrazów")
    return result


def load_panel(path: Path) -> dict[int, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {int(row["image_index"]): row for row in payload["entries"]}
    if len(result) != 64:
        raise RuntimeError("Panel nie ma 64 obrazów")
    splits = [row["analysis_split"] for row in result.values()]
    if splits.count("train") != 40 or splits.count("validation") != 12 or splits.count("test") != 12:
        raise RuntimeError("Niewłaściwy split panelu")
    split_videos = {
        split: {row["video_id"] for row in result.values() if row["analysis_split"] == split}
        for split in ("train", "validation", "test")
    }
    if any(split_videos[a] & split_videos[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Filmy panelu przeciekają między splitami")
    return result


def parse_indices(args: argparse.Namespace) -> list[int]:
    if args.indices_json:
        indices = [int(v) for v in json.loads(args.indices_json.read_text(encoding="utf-8"))]
    else:
        if args.shard_id is None or args.num_shards is None:
            raise ValueError("Brak definicji sharda")
        indices = list(range(args.shard_id, 1500, args.num_shards))
    if len(indices) != len(set(indices)) or any(i < 0 or i >= 1500 for i in indices):
        raise ValueError(indices)
    return indices


def validate_resume(
    record_path: Path, replay_path: Path, panel_path: Path | None, analysis_pca32_basis_sha256: str
) -> bool:
    expected = [record_path, replay_path] + ([] if panel_path is None else [panel_path])
    if not any(path.exists() for path in expected):
        return False
    if not all(path.exists() for path in expected):
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Obcy checkpoint w {record_path}")
    if record["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Obcy manifest w {record_path}")
    if record["pca_basis_sha256"] != PCA_BASIS_SHA256:
        raise RuntimeError(f"Obca baza PCA w {record_path}")
    if record.get("analysis_pca32_basis_sha256") != analysis_pca32_basis_sha256:
        raise RuntimeError(f"Obca analityczna PCA32 w {record_path}")
    if sha256(replay_path) != record["replay_sha256"]:
        raise RuntimeError(f"Uszkodzony replay {replay_path}")
    with np.load(replay_path, allow_pickle=False) as data:
        validate_replay_arrays({name: data[name] for name in data.files})
    if panel_path is not None:
        if sha256(panel_path) != record["panel_sha256"]:
            raise RuntimeError(f"Uszkodzony panel {panel_path}")
        with np.load(panel_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
            validate_panel_arrays(arrays, record["target_index"], record["natural_top1_token"])
    return True


def assert_replay_matches_old(result: dict, reference: dict, state_atol: float, metric_atol: float) -> dict:
    comparison = result["state_comparison"]
    for label in ("before", "step_100", "step_500"):
        if comparison[f"{label}_max_abs"] is None or comparison[f"{label}_max_abs"] > state_atol:
            raise RuntimeError(f"Replay {label} przekracza tolerancję: {comparison}")
    telemetry = result["telemetry_comparison"]
    for name in ("step", "phase_min_norm", "top_token", "target_is_top", "success"):
        if not telemetry[f"{name}_exact"]:
            raise RuntimeError(f"Dyskretna telemetria różni się: {name}")
    for name in (
        "learning_rate",
        "fit_objective",
        "confidence",
        "iou",
        "delta_rms",
        "delta_l2",
        "gradient_rms",
    ):
        if telemetry[f"{name}_max_abs"] > metric_atol:
            raise RuntimeError(f"Telemetria {name} przekracza tolerancję: {telemetry[name + '_max_abs']}")
    old_first = reference["metadata"].get("first_success_step")
    if result["first_success_step"] != old_first or result["status"] != reference["metadata"]["status"]:
        raise RuntimeError("Status replay różni się od starego oracle")
    return {"states": comparison, "telemetry": telemetry, "old_first_success_step": old_first}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--analysis-pca32", type=Path, required=True)
    parser.add_argument("--analysis-pca32-manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument("--old-production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--indices-json", type=Path)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--state-atol", type=float, default=2e-6)
    parser.add_argument("--metric-atol", type=float, default=2e-6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Replay wymaga CUDA")
    if sha256(args.manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Niewłaściwy manifest")
    rows = load_manifest(args.manifest)
    if len(rows) != 1500:
        raise RuntimeError(len(rows))
    with np.load(args.pca, allow_pickle=False) as data:
        pca_mean = data["pca_mean"].copy()
        pca_components = data["pca_components"].copy()
    if basis_sha256(pca_mean, pca_components) != PCA_BASIS_SHA256:
        raise RuntimeError("Niewłaściwa baza PCA")
    analysis_pca32_manifest = json.loads(args.analysis_pca32_manifest.read_text(encoding="utf-8"))
    if analysis_pca32_manifest.get("status") != "complete":
        raise RuntimeError("Analityczna PCA32 nie jest kompletna")
    with np.load(args.analysis_pca32, allow_pickle=False) as data:
        analysis_pca32_mean_np = data["pca_mean"].copy()
        analysis_pca32_components_np = data["pca_components"].copy()
    analysis_pca32_basis_sha256 = basis_sha256(
        analysis_pca32_mean_np, analysis_pca32_components_np
    )
    if analysis_pca32_basis_sha256 != analysis_pca32_manifest.get("basis_sha256"):
        raise RuntimeError("Hash analitycznej PCA32 nie zgadza się z manifestem")
    if analysis_pca32_mean_np.shape != (768,) or analysis_pca32_components_np.shape != (768, 32):
        raise RuntimeError("Niewłaściwy kształt analitycznej PCA32")
    panel = load_panel(args.panel)
    old_index = load_old_index(args.old_index)
    indices = parse_indices(args)

    torch.set_num_threads(8)
    torch.manual_seed(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    model, model_hash = load_model(args.checkpoint, device)
    analysis_pca32_mean = torch.as_tensor(
        analysis_pca32_mean_np, device=device, dtype=torch.float32
    )
    analysis_pca32_components = torch.as_tensor(
        analysis_pca32_components_np, device=device, dtype=torch.float32
    )
    torch.cuda.reset_peak_memory_stats(device)
    source = TarImageSource(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    shard = "selected" if args.indices_json else f"shard_{args.shard_id:02d}_of_{args.num_shards:02d}"
    config = {
        "mode": args.mode,
        "shard": shard,
        "indices": indices,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "pca_basis_sha256": PCA_BASIS_SHA256,
        "analysis_pca32_basis_sha256": analysis_pca32_basis_sha256,
        "steps": list(range(0, 501, 5)),
        "panel_steps": list(range(501)),
        "panel_tokens": 16,
        "state_atol": args.state_atol,
        "metric_atol": args.metric_atol,
        "precision": "FP32 optimization; TF32 disabled; PCA formula in float64 then cast FP32",
    }
    config_path = args.output / f"RUN_CONFIG_{shard}.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise RuntimeError(f"Konflikt konfiguracji {config_path}")
    if not config_path.exists():
        json_atomic(config_path, config)

    started = time.perf_counter()
    completed = skipped = 0
    max_state_error = max_metric_error = 0.0
    panel_completed = 0
    try:
        for ordinal, index in enumerate(indices, 1):
            row = rows[index]
            replay_path = args.output / "per_image" / f"{index:06d}.npz"
            record_path = args.output / "records" / f"{index:06d}.json"
            panel_path = args.output / "panel" / f"{index:06d}.npz" if index in panel else None
            if validate_resume(
                record_path, replay_path, panel_path, analysis_pca32_basis_sha256
            ):
                skipped += 1
                print(json.dumps({"resume_skip": index, "ordinal": ordinal, "total": len(indices)}), flush=True)
                continue
            image_started = time.perf_counter()
            try:
                payload = source.read_bytes(row["image_name"])
                if hashlib.sha256(payload).hexdigest() != row["image_sha256"]:
                    raise RuntimeError(f"Hash obrazu {index} nie zgadza się z manifestem")
                rgb = decode_resize(payload)
                image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(
                    device=device, dtype=torch.float32
                ) / 255.0
                ground_truth = torch.tensor(row["bbox_xyxy_480"], device=device, dtype=torch.float32)
                _, z0, q0 = capture_baseline(model, image)
                target = choose_target_token(np.asarray(row["bbox_xyxy_480"]), 20, 30)
                old_record = old_index[index]
                old_artifact = args.old_production / "latents" / f"{index:06d}.npz"
                if sha256(old_artifact) != old_record["artifact_sha256"]:
                    raise RuntimeError(f"Hash starego artefaktu {index} nie zgadza się")
                reference = load_artifact(old_artifact)
                torch.cuda.synchronize(device)
                optimize_started = time.perf_counter()
                result = replay_latent(
                    model,
                    z0,
                    q0,
                    ground_truth,
                    target,
                    pca_mean,
                    pca_components,
                    analysis_pca32_mean,
                    analysis_pca32_components,
                    panel=index in panel,
                    reference=reference,
                )
                torch.cuda.synchronize(device)
                optimize_seconds = time.perf_counter() - optimize_started
                validation = assert_replay_matches_old(
                    result, reference, args.state_atol, args.metric_atol
                )
                state_error = max(
                    float(validation["states"]["before_max_abs"]),
                    float(validation["states"]["step_100_max_abs"]),
                    float(validation["states"]["step_500_max_abs"]),
                )
                metric_error = max(
                    float(value)
                    for key, value in validation["telemetry"].items()
                    if key.endswith("_max_abs")
                )
                max_state_error = max(max_state_error, state_error)
                max_metric_error = max(max_metric_error, metric_error)
                metadata = {
                    "schema_version": 1,
                    "family": "vit_k1000_replay900k_co5_20261009",
                    "index": index,
                    "image_id": row["image_id"],
                    "annotation_id": row["annotation_id"],
                    "image_name": row["image_name"],
                    "image_sha256": row["image_sha256"],
                    "frame_id": row["frame_id"],
                    "film_id": row["frame_id"].rsplit("/", 1)[0],
                    "status": result["status"],
                    "first_success_step": result["first_success_step"],
                    "target": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in target.items()},
                    "natural_top1_token": result["natural_top1"],
                    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "pca_basis_sha256": PCA_BASIS_SHA256,
                    "analysis_pca32_basis_sha256": analysis_pca32_basis_sha256,
                    "analysis_projection_role": "train_only_analytical_PCA32_for_population_dynamics",
                    "analysis_projection_fit_uses_validation_or_test": False,
                    "weights_buffers_hash": model_hash,
                    "steps_are_real": True,
                    "interpolation": False,
                    "projection_role": "visualization_only_not_dynamics_model_basis",
                    "movement_norm": "L2(z_step-z0) in full 768D",
                    "global_motion_descriptors": {
                        "source_space": "full_768D_before_visual_projection",
                        "interval": "five actual Adam updates; endpoint in descriptor_interval_end_steps",
                        "increment_norm": "L2(z_k-z_k-5)",
                        "previous_increment_dot": "dot(z_k-z_k-5, z_k-5-z_k-10); row 0 invalid",
                        "previous_increment_cosine": "cosine of consecutive five-update increments; row 0 invalid",
                        "gradient_l2_mean": "mean per-token L2 gradient over the five updates",
                        "gradient_l2_last": "per-token L2 gradient before the interval's final update",
                        "gradient_l2_start": "per-token L2 gradient at z_start before the interval's first update; causal start control",
                        "adam_m1_l2_start": "per-token L2 Adam m1 before incorporating the interval's first gradient",
                        "adam_m2_l2_start": "per-token L2 Adam m2 before incorporating the interval's first gradient",
                        "role_mask_bits": {"1": "target", "2": "natural_top1", "4": "strongest_natural_competitor"},
                        "phase_fraction": "fraction of transitions using minimum-norm objective",
                    },
                    "reference_validation": validation,
                }
                replay_arrays = {
                    "projected_positions": result["projected_positions"],
                    "analysis_positions_pca32": result["analysis_positions_pca32"],
                    "analysis_state_residual_norm": result["analysis_state_residual_norm"],
                    "analysis_increment_residual_norm": result["analysis_increment_residual_norm"],
                    "movement_norm": result["movement_norm"],
                    "steps": result["steps"],
                    "descriptor_interval_end_steps": result["descriptor_interval_end_steps"],
                    "interval_increment_norm": result["interval_increment_norm"],
                    "interval_previous_increment_dot": result["interval_previous_increment_dot"],
                    "interval_previous_increment_cosine": result["interval_previous_increment_cosine"],
                    "interval_previous_increment_valid": result["interval_previous_increment_valid"],
                    "interval_gradient_l2_mean": result["interval_gradient_l2_mean"],
                    "interval_gradient_l2_last": result["interval_gradient_l2_last"],
                    "interval_gradient_l2_start": result["interval_gradient_l2_start"],
                    "interval_adam_m1_l2_start": result["interval_adam_m1_l2_start"],
                    "interval_adam_m2_l2_start": result["interval_adam_m2_l2_start"],
                    "interval_phase_min_norm_fraction": result["interval_phase_min_norm_fraction"],
                    "interval_phase_min_norm_start": result["interval_phase_min_norm_start"],
                    "token_role_mask": result["token_role_mask"],
                    "image_index": np.asarray(index, dtype=np.int32),
                    "token_indices": np.arange(600, dtype=np.int32),
                    "target_token": np.asarray(result["target_index"], dtype=np.int32),
                    "natural_top1_token": np.asarray(result["natural_top1"], dtype=np.int32),
                    "phase_before_update": result["phase_before_update"],
                    "metadata_json_utf8": encode_json(metadata),
                }
                replay_arrays.update({f"telemetry_{k}": v for k, v in result["telemetry"].items()})
                validate_replay_arrays(replay_arrays)
                write_npz_atomic(replay_path, replay_arrays)
                with np.load(replay_path, allow_pickle=False) as restored:
                    validate_replay_arrays({name: restored[name] for name in restored.files})

                panel_hash = None
                panel_bytes = 0
                if panel_path is not None:
                    panel_entry = panel[index]
                    selection = result["panel_selection"]
                    panel_metadata = {
                        **metadata,
                        "analysis_split": panel_entry["analysis_split"],
                        "video_disjoint_split": True,
                        "control_token_roles": {
                            "bit_1": "target",
                            "bit_2": "natural_top1",
                            "bit_4": "strongest_natural_competitor",
                        },
                        "control_inputs_alignment": "g,m1,m2 before update z_t to z_t+1",
                        "adam_betas": [0.9, 0.999],
                        "adam_eps": 1e-8,
                        "clamp_floor": -0.16997121,
                    }
                    panel_arrays = {
                        **result["panel_arrays"],
                        "control_tokens": np.asarray(selection["tokens"], dtype=np.int32),
                        "control_token_role_mask": np.asarray(selection["role_mask"], dtype=np.uint8),
                        "control_token_baseline_score": np.asarray(selection["baseline_score"], dtype=np.float32),
                        "control_token_baseline_rank": np.asarray(selection["baseline_rank"], dtype=np.int32),
                        "telemetry_step": result["telemetry"]["step"],
                        "telemetry_phase_min_norm": result["telemetry"]["phase_min_norm"],
                        "telemetry_learning_rate": result["telemetry"]["learning_rate"],
                        "telemetry_top_token": result["telemetry"]["top_token"],
                        "phase_before_update": result["phase_before_update"],
                        "image_index": np.asarray(index, dtype=np.int32),
                        "metadata_json_utf8": encode_json(panel_metadata),
                    }
                    validate_panel_arrays(
                        panel_arrays, result["target_index"], result["natural_top1"]
                    )
                    write_npz_atomic(panel_path, panel_arrays)
                    with np.load(panel_path, allow_pickle=False) as restored:
                        validate_panel_arrays(
                            {name: restored[name] for name in restored.files},
                            result["target_index"],
                            result["natural_top1"],
                        )
                    panel_hash = sha256(panel_path)
                    panel_bytes = panel_path.stat().st_size
                    panel_completed += 1

                record = {
                    "index": index,
                    "frame_id": row["frame_id"],
                    "status": result["status"],
                    "first_success_step": result["first_success_step"],
                    "target_index": result["target_index"],
                    "natural_top1_token": result["natural_top1"],
                    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "pca_basis_sha256": PCA_BASIS_SHA256,
                    "analysis_pca32_basis_sha256": analysis_pca32_basis_sha256,
                    "weights_buffers_hash": model_hash,
                    "replay_path": str(replay_path.relative_to(args.output)).replace("\\", "/"),
                    "replay_sha256": sha256(replay_path),
                    "replay_bytes": replay_path.stat().st_size,
                    "panel_path": None if panel_path is None else str(panel_path.relative_to(args.output)).replace("\\", "/"),
                    "panel_sha256": panel_hash,
                    "panel_bytes": panel_bytes,
                    "reference_validation": validation,
                    "timing_seconds": {
                        "optimization_projection_serialization": time.perf_counter() - image_started,
                        "optimization_projection": optimize_seconds,
                        "per_real_update": optimize_seconds / 500,
                    },
                }
                json_atomic(record_path, record)
                completed += 1
                json_atomic(
                    args.output / f"PROGRESS_{shard}.json",
                    {
                        "status": "running",
                        "shard": shard,
                        "completed_this_invocation": completed,
                        "skipped_valid": skipped,
                        "last_index": index,
                        "total_assigned": len(indices),
                        "panel_completed_this_invocation": panel_completed,
                        "max_state_abs_error": max_state_error,
                        "max_telemetry_abs_error": max_metric_error,
                        "elapsed_seconds": time.perf_counter() - started,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(
                    json.dumps(
                        {
                            "index": index,
                            "ordinal": ordinal,
                            "total": len(indices),
                            "panel": panel_path is not None,
                            "status": result["status"],
                            "state_error": state_error,
                            "metric_error": metric_error,
                            "seconds": record["timing_seconds"],
                            "bytes": record["replay_bytes"] + record["panel_bytes"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception:
                json_atomic(
                    args.output / "failures" / f"{index:06d}.json",
                    {"index": index, "traceback": traceback.format_exc(), "time_utc": datetime.now(timezone.utc).isoformat()},
                )
                raise
    finally:
        source.close()

    ending_hash = state_fingerprint(model)
    if ending_hash != model_hash or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Model nie pozostał zamrożony")
    records = [
        json.loads((args.output / "records" / f"{index:06d}.json").read_text(encoding="utf-8"))
        for index in indices
    ]
    summary = {
        "status": "complete",
        "mode": args.mode,
        "shard": shard,
        "assigned": len(indices),
        "completed_this_invocation": completed,
        "skipped_valid": skipped,
        "panel_count": sum(record["panel_path"] is not None for record in records),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "pca_basis_sha256": PCA_BASIS_SHA256,
        "analysis_pca32_basis_sha256": analysis_pca32_basis_sha256,
        "weights_buffers_hash_before": model_hash,
        "weights_buffers_hash_after": ending_hash,
        "weights_buffers_unchanged": True,
        "max_state_abs_error_vs_old_0_100_500": max(
            float(record["reference_validation"]["states"][key])
            for record in records
            for key in ("before_max_abs", "step_100_max_abs", "step_500_max_abs")
        ),
        "max_telemetry_abs_error_vs_old": max(
            float(value)
            for record in records
            for key, value in record["reference_validation"]["telemetry"].items()
            if key.endswith("_max_abs")
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "elapsed_seconds": time.perf_counter() - started,
        "total_bytes": sum(record["replay_bytes"] + record["panel_bytes"] for record in records),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    json_atomic(args.output / f"SUMMARY_{shard}.json", summary)
    json_atomic(args.output / f"PROGRESS_{shard}.json", summary)
    print("COMPLETE", json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
