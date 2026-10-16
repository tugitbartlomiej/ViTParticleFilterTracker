"""Scala replay; obala kompletność dowolny brak obrazu, kroku, panelu lub zgodności 100/500."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from replay_core import (
    DENSE_STEPS,
    PCA_BASIS_SHA256,
    sha256,
    validate_panel_arrays,
    validate_replay_arrays,
)

CHECKPOINT_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
MANIFEST_SHA256 = "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7"


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--panel-contract", type=Path, required=True)
    parser.add_argument("--analysis-pca32-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel_contract = json.loads(args.panel_contract.read_text(encoding="utf-8"))
    analysis_pca32_manifest = json.loads(
        args.analysis_pca32_manifest.read_text(encoding="utf-8")
    )
    analysis_pca32_basis_sha256 = analysis_pca32_manifest.get("basis_sha256")
    analysis_pca32_path = args.analysis_pca32_manifest.with_name("PCA32_ANALYTICAL.npz")
    if analysis_pca32_manifest.get("status") != "complete" or not analysis_pca32_path.is_file():
        raise RuntimeError("Brak kompletnej analitycznej PCA32")
    if sha256(analysis_pca32_path) != analysis_pca32_manifest.get("container_sha256"):
        raise RuntimeError("Kontener analitycznej PCA32 ma niewłaściwy hash")
    panel_expected = {int(row["image_index"]): row for row in panel_contract["entries"]}
    if len(panel_expected) != 64:
        raise RuntimeError("Kontrakt panelu nie ma 64 obrazów")
    replay_files = sorted((args.production / "per_image").glob("*.npz"))
    record_files = sorted((args.production / "records").glob("*.json"))
    panel_files = sorted((args.production / "panel").glob("*.npz"))
    if len(replay_files) != 1500 or len(record_files) != 1500 or len(panel_files) != 64:
        raise RuntimeError(
            f"Niekompletne wyniki: replay={len(replay_files)} records={len(record_files)} panel={len(panel_files)}"
        )

    index_rows = []
    panel_rows = []
    weights_hashes = set()
    array_job_ids = set()
    status_counts = {"success": 0, "budget_exhausted": 0}
    replay_bytes = panel_bytes = 0
    max_state_error = max_telemetry_error = 0.0
    exact_state_counts = {"before": 0, "step_100": 0, "step_500": 0}
    for index in range(1500):
        replay_path = args.production / "per_image" / f"{index:06d}.npz"
        record_path = args.production / "records" / f"{index:06d}.json"
        if not replay_path.exists() or not record_path.exists():
            raise RuntimeError(f"Brak obrazu {index}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["index"] != index:
            raise RuntimeError(f"Niespójny indeks {index}")
        if record["checkpoint_sha256"] != CHECKPOINT_SHA256 or record["manifest_sha256"] != MANIFEST_SHA256:
            raise RuntimeError(f"Niespójne wejścia {index}")
        if record["pca_basis_sha256"] != PCA_BASIS_SHA256:
            raise RuntimeError(f"Niespójna PCA {index}")
        if record.get("analysis_pca32_basis_sha256") != analysis_pca32_basis_sha256:
            raise RuntimeError(f"Niespójna analityczna PCA32 {index}")
        if sha256(replay_path) != record["replay_sha256"]:
            raise RuntimeError(f"Hash replay {index}")
        with np.load(replay_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
            validate_replay_arrays(arrays)
            if int(arrays["image_index"]) != index:
                raise RuntimeError(f"Wewnętrzny indeks {index}")
        validation = record["reference_validation"]
        for label in exact_state_counts:
            exact_state_counts[label] += int(bool(validation["states"][f"{label}_exact"]))
            max_state_error = max(max_state_error, float(validation["states"][f"{label}_max_abs"]))
        max_telemetry_error = max(
            max_telemetry_error,
            max(float(v) for k, v in validation["telemetry"].items() if k.endswith("_max_abs")),
        )
        status_counts[record["status"]] += 1
        weights_hashes.add(record["weights_buffers_hash"])
        replay_bytes += int(record["replay_bytes"])
        row = {
            "image_index": index,
            "frame_id": record["frame_id"],
            "status": record["status"],
            "first_success_step": record["first_success_step"],
            "target_index": record["target_index"],
            "natural_top1_token": record["natural_top1_token"],
            "path": os.path.relpath(replay_path, args.output).replace("\\", "/"),
            "sha256": record["replay_sha256"],
            "bytes": record["replay_bytes"],
        }
        index_rows.append(row)
        if index in panel_expected:
            panel_path = args.production / "panel" / f"{index:06d}.npz"
            if record["panel_path"] is None or sha256(panel_path) != record["panel_sha256"]:
                raise RuntimeError(f"Hash panelu {index}")
            with np.load(panel_path, allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
                validate_panel_arrays(arrays, record["target_index"], record["natural_top1_token"])
                roles = arrays["control_token_role_mask"]
                if not np.any(roles & 1) or not np.any(roles & 2):
                    raise RuntimeError(f"Brak ról target/top1 w panelu {index}")
            panel_bytes += int(record["panel_bytes"])
            panel_rows.append(
                {
                    "path": os.path.relpath(panel_path, args.output).replace("\\", "/"),
                    "sha256": record["panel_sha256"],
                    "bytes": record["panel_bytes"],
                    "image_index": index,
                    "frame_id": record["frame_id"],
                    "analysis_split": panel_expected[index]["analysis_split"],
                    "target_index": record["target_index"],
                    "natural_top1_token": record["natural_top1_token"],
                }
            )

    if len(weights_hashes) != 1:
        raise RuntimeError(f"Różne fingerprinty modelu: {weights_hashes}")
    summaries = sorted(args.production.glob("SUMMARY_shard_*.json"))
    if len(summaries) != 8:
        raise RuntimeError(f"Brak ośmiu summary shardów: {len(summaries)}")
    for path in summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["status"] != "complete" or not summary["weights_buffers_unchanged"]:
            raise RuntimeError(f"Nieprawidłowy shard {path}")
        if summary.get("slurm_job_id"):
            array_job_ids.add(str(summary["slurm_job_id"]).split("_")[0])

    split_counts = {
        split: sum(row["analysis_split"] == split for row in panel_rows)
        for split in ("train", "validation", "test")
    }
    if split_counts != {"train": 40, "validation": 12, "test": 12}:
        raise RuntimeError(split_counts)

    args.output.mkdir(parents=True, exist_ok=True)
    index_payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in index_rows
    ).encode("utf-8")
    atomic_bytes(args.output / "REPLAY_INDEX.jsonl", index_payload)
    validation = {
        "schema_version": 1,
        "status": "pass",
        "count_images": 1500,
        "count_particles": 900000,
        "steps": DENSE_STEPS.tolist(),
        "real_states_per_particle": 101,
        "interpolation": False,
        "exact_match_counts_vs_old": exact_state_counts,
        "max_state_abs_error_vs_old_at_0_100_500": max_state_error,
        "max_telemetry_abs_error_vs_old_0_500": max_telemetry_error,
        "determinism_tolerance_state": 2e-6,
        "determinism_tolerance_telemetry": 2e-6,
        "weights_buffers_hash": next(iter(weights_hashes)),
        "weights_buffers_unchanged": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "pca_basis_sha256": PCA_BASIS_SHA256,
        "pca_role": "visualization_only_not_dynamics_model_basis",
        "analysis_pca32_basis_sha256": analysis_pca32_basis_sha256,
        "analysis_pca32_role": "train_only_population_dynamics_basis",
        "status_counts": status_counts,
        "split_counts": split_counts,
        "replay_bytes": replay_bytes,
        "panel_bytes": panel_bytes,
        "total_tensor_bytes": replay_bytes + panel_bytes,
        "producer_array_job_ids": sorted(array_job_ids),
        "merge_job_id": os.environ.get("SLURM_JOB_ID"),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output / "REPLAY_VALIDATION.json", validation)
    panel_manifest = {
        "schema_version": 1,
        "status": "complete",
        "step_stride": 1,
        "dimensions": 768,
        "tokens_per_image": 16,
        "count_images": 64,
        "required_arrays": [
            "control_z",
            "control_gradient_before_step",
            "control_adam_m1_before_step",
            "control_adam_m2_before_step",
            "control_tokens",
            "telemetry_step",
            "telemetry_phase_min_norm",
            "telemetry_learning_rate",
            "telemetry_top_token",
            "metadata_json_utf8",
        ],
        "files": sorted(panel_rows, key=lambda row: row["image_index"]),
        "producer_job_id": sorted(array_job_ids),
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "source_manifest_sha256": MANIFEST_SHA256,
        "validation_path": "REPLAY_VALIDATION.json",
        "analysis_split": {"unit": "video", **split_counts, "video_disjoint": True},
        "control_alignment": "g,m1,m2 before update z_t to z_t+1; no future delta",
        "visual_pca_is_analysis_basis": False,
        "analysis_pca32_basis_sha256": analysis_pca32_basis_sha256,
    }
    atomic_json(args.output / "TEMPORAL_PANEL_MANIFEST.json", panel_manifest)
    visualization_manifest = {
        "schema_version": 1,
        "status": "complete",
        "point_count": 900000,
        "image_count": 1500,
        "tokens_per_image": 600,
        "steps": DENSE_STEPS.tolist(),
        "pca_basis_sha256": PCA_BASIS_SHA256,
        "interpolation": False,
        "dtype": "float32",
        "axis_order": ["image", "step", "token", "pca_axis"],
        "keys": {
            "position": "projected_positions",
            "norm": "movement_norm",
            "steps": "steps",
            "telemetry_step": "telemetry_step",
            "confidence": "telemetry_confidence",
            "iou": "telemetry_iou",
        },
        "full_768d_scalar_descriptors": {
            "interval_end_steps": "descriptor_interval_end_steps",
            "net_displacement_l2": "movement_norm[1:]",
            "increment_l2": "interval_increment_norm",
            "previous_increment_dot": "interval_previous_increment_dot",
            "previous_increment_cosine": "interval_previous_increment_cosine",
            "previous_increment_valid": "interval_previous_increment_valid",
            "gradient_l2_mean": "interval_gradient_l2_mean",
            "gradient_l2_last": "interval_gradient_l2_last",
            "gradient_l2_start": "interval_gradient_l2_start",
            "adam_m1_l2_start": "interval_adam_m1_l2_start",
            "adam_m2_l2_start": "interval_adam_m2_l2_start",
            "phase_min_norm_fraction": "interval_phase_min_norm_fraction",
            "phase_min_norm_start": "interval_phase_min_norm_start",
            "token_role_mask": "token_role_mask",
            "source_space": "full_768D_before_visual_projection",
        },
        "files": [
            {"path": row["path"], "sha256": row["sha256"], "image_index": row["image_index"]}
            for row in index_rows
        ],
        "validation_path": "REPLAY_VALIDATION.json",
        "producer_array_job_ids": sorted(array_job_ids),
        "producer_merge_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(args.output / "VISUALIZATION_MANIFEST.json", visualization_manifest)
    descriptor_manifest = {
        "schema_version": 1,
        "status": "complete",
        "image_count": 1500,
        "particle_count": 900000,
        "token_transition_count": 90000000,
        "step_stride": 5,
        "interval_end_steps": DENSE_STEPS[1:].tolist(),
        "source_dimensions": 768,
        "stored_direction_dimensions": 32,
        "stored_directional_control_dimensions_full_population": 0,
        "source_space": "full_768D_before_visual_projection",
        "analysis_pca32": {
            "positions": "analysis_positions_pca32",
            "shape_per_image": [101, 600, 32],
            "state_residual_l2_full768": "analysis_state_residual_norm",
            "increment_residual_l2_full768": "analysis_increment_residual_norm",
            "basis_sha256": analysis_pca32_basis_sha256,
            "basis_fit": "train films only; validation/test excluded",
        },
        "visual_pca_is_analysis_basis": False,
        "causal_start_controls": {
            "gradient_l2": "interval_gradient_l2_start",
            "adam_m1_l2": "interval_adam_m1_l2_start",
            "adam_m2_l2": "interval_adam_m2_l2_start",
            "phase_min_norm": "interval_phase_min_norm_start",
            "alignment": "row r is known at z_start before the first update of z_start to z_start+5",
            "oracle_dependency": "gradient and phase use the GT-conditioned oracle objective",
        },
        "teacher_forced_diagnostics_not_valid_as_same_transition_forecast_inputs": {
            "gradient_l2_mean": "interval_gradient_l2_mean",
            "gradient_l2_last": "interval_gradient_l2_last",
            "phase_min_norm_fraction": "interval_phase_min_norm_fraction",
        },
        "outcome_descriptors_not_controls": {
            "net_displacement_l2": "movement_norm[1:]",
            "increment_l2": "interval_increment_norm",
            "previous_increment_dot": "interval_previous_increment_dot",
            "previous_increment_cosine": "interval_previous_increment_cosine",
            "previous_increment_valid": "interval_previous_increment_valid",
        },
        "static_token_context": {
            "role_mask": "token_role_mask",
            "role_mask_bits": {"1": "target", "2": "natural_top1", "4": "strongest_natural_competitor"},
        },
        "limitation": "PCA32 preserves shared reduced directions plus full768 residual norms, but full-population directional g/m1/m2 controls are not stored; controlled DMDc with vector controls remains limited to the full-768D panel, while whole-population PCA32 supports autonomous/statistical regime analysis",
        "files": [
            {"path": row["path"], "sha256": row["sha256"], "image_index": row["image_index"]}
            for row in index_rows
        ],
        "panel_manifest": "TEMPORAL_PANEL_MANIFEST.json",
        "validation_path": "REPLAY_VALIDATION.json",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
    }
    atomic_json(args.output / "GLOBAL_MOTION_DESCRIPTOR_MANIFEST.json", descriptor_manifest)
    analytical_pca32_output = {
        **analysis_pca32_manifest,
        "status": "complete",
        "basis_path": os.path.relpath(analysis_pca32_path, args.output).replace("\\", "/"),
        "per_image_key": "analysis_positions_pca32",
        "per_image_shape": [101, 600, 32],
        "state_residual_norm_key": "analysis_state_residual_norm",
        "increment_residual_norm_key": "analysis_increment_residual_norm",
        "steps_key": "steps",
        "files": [
            {"path": row["path"], "sha256": row["sha256"], "image_index": row["image_index"]}
            for row in index_rows
        ],
        "validation_path": "REPLAY_VALIDATION.json",
    }
    atomic_json(args.output / "ANALYTICAL_PCA32_MANIFEST.json", analytical_pca32_output)
    contract = {
        "schema_version": 1,
        "status": "complete",
        "per_image_root": str((args.production / "per_image").resolve()),
        "per_image_key": "projected_positions",
        "per_image_shape": [101, 600, 3],
        "movement_norm_key": "movement_norm",
        "movement_norm_shape": [101, 600],
        "global_motion_descriptors": {
            "source_space": "full_768D_before_visual_projection",
            "interval_end_steps": "descriptor_interval_end_steps",
            "interval_shape_per_token": [100, 600],
            "increment_norm": "interval_increment_norm",
            "previous_increment_dot": "interval_previous_increment_dot",
            "previous_increment_cosine": "interval_previous_increment_cosine",
            "previous_increment_valid": "interval_previous_increment_valid",
            "gradient_l2_mean": "interval_gradient_l2_mean",
            "gradient_l2_last": "interval_gradient_l2_last",
            "gradient_l2_start": "interval_gradient_l2_start",
            "adam_m1_l2_start": "interval_adam_m1_l2_start",
            "adam_m2_l2_start": "interval_adam_m2_l2_start",
            "phase_min_norm_fraction": "interval_phase_min_norm_fraction",
            "phase_min_norm_start": "interval_phase_min_norm_start",
            "token_role_mask": "token_role_mask",
            "role_mask_bits": {"1": "target", "2": "natural_top1", "4": "strongest_natural_competitor"},
        },
        "steps": DENSE_STEPS.tolist(),
        "dtype": "float32",
        "projection": "(z_float64-pca_mean)@pca_components then cast float32",
        "pca_basis_sha256": PCA_BASIS_SHA256,
        "point_order": "image-major then token-major",
        "dense_timeline_export_owner": "viewer/exporter task; replay does not create step_NNN.bin",
        "panel_manifest": "TEMPORAL_PANEL_MANIFEST.json",
        "replay_index": "REPLAY_INDEX.jsonl",
        "validation": "REPLAY_VALIDATION.json",
        "visualization_manifest": "VISUALIZATION_MANIFEST.json",
        "global_motion_descriptor_manifest": "GLOBAL_MOTION_DESCRIPTOR_MANIFEST.json",
        "analytical_pca32_manifest": "ANALYTICAL_PCA32_MANIFEST.json",
    }
    atomic_json(args.output / "DATASET_CONTRACT.json", contract)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
