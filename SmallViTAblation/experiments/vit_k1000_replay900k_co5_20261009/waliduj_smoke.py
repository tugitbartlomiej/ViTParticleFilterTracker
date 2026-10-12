"""Publikuje smoke PASS dopiero po fresh-load, zgodności starego oracle i kontroli panelu."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from replay_core import sha256, validate_panel_arrays, validate_replay_arrays
from uruchom_replay import json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--analysis-pca32-manifest", type=Path, required=True)
    args = parser.parse_args()

    indices = [int(value) for value in json.loads(args.indices.read_text(encoding="utf-8"))]
    panel_indices = {
        int(row["image_index"])
        for row in json.loads(args.panel.read_text(encoding="utf-8"))["entries"]
    }
    if indices != [91, 684, 0]:
        raise RuntimeError(f"Niezatwierdzony smoke: {indices}")
    summary_path = args.output / "SUMMARY_selected.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pca32 = json.loads(args.analysis_pca32_manifest.read_text(encoding="utf-8"))
    if pca32.get("status") != "complete" or pca32.get("validation_and_test_used_for_fit") is not False:
        raise RuntimeError("PCA32 nie jest poprawnie zamrożona przed smoke")
    if summary["status"] != "complete" or summary["assigned"] != len(indices):
        raise RuntimeError(summary)
    if not summary["weights_buffers_unchanged"]:
        raise RuntimeError("Wagi lub bufory zmieniły się w smoke")

    records = []
    exact_counts = {"before": 0, "step_100": 0, "step_500": 0}
    max_state = max_telemetry = 0.0
    total_bytes = 0
    for index in indices:
        record_path = args.output / "records" / f"{index:06d}.json"
        replay_path = args.output / "per_image" / f"{index:06d}.npz"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if sha256(replay_path) != record["replay_sha256"]:
            raise RuntimeError(f"Hash replay {index}")
        with np.load(replay_path, allow_pickle=False) as data:
            validate_replay_arrays({name: data[name] for name in data.files})
        validation = record["reference_validation"]
        for label in exact_counts:
            exact_counts[label] += int(bool(validation["states"][f"{label}_exact"]))
            max_state = max(max_state, float(validation["states"][f"{label}_max_abs"]))
        max_telemetry = max(
            max_telemetry,
            max(
                float(value)
                for key, value in validation["telemetry"].items()
                if key.endswith("_max_abs")
            ),
        )
        panel_path = args.output / "panel" / f"{index:06d}.npz"
        if index in panel_indices:
            if not panel_path.is_file() or sha256(panel_path) != record["panel_sha256"]:
                raise RuntimeError(f"Panel {index}")
            with np.load(panel_path, allow_pickle=False) as data:
                validate_panel_arrays(
                    {name: data[name] for name in data.files},
                    record["target_index"],
                    record["natural_top1_token"],
                )
        elif panel_path.exists() or record["panel_path"] is not None:
            raise RuntimeError(f"Nadmiarowy panel {index}")
        total_bytes += int(record["replay_bytes"]) + int(record["panel_bytes"])
        records.append(record)

    if max_state > 2e-6 or max_telemetry > 2e-6:
        raise RuntimeError((max_state, max_telemetry))
    if {record["status"] for record in records} != {"success", "budget_exhausted"}:
        raise RuntimeError("Smoke nie obejmuje obu statusów")
    if sum(record["panel_path"] is not None for record in records) != 2:
        raise RuntimeError("Smoke nie obejmuje dwóch paneli")

    optimization = [record["timing_seconds"]["optimization_projection"] for record in records]
    total = [record["timing_seconds"]["optimization_projection_serialization"] for record in records]
    median_optimization = statistics.median(optimization)
    median_total = statistics.median(total)
    estimate = {
        "basis": "three-image smoke; includes success and budget-exhausted paths; two panel images",
        "median_optimization_projection_seconds_per_image": median_optimization,
        "median_total_seconds_per_image": median_total,
        "seconds_per_real_update": median_optimization / 500.0,
        "eight_gpu_wall_seconds_simple": median_total * 188,
        "eight_gpu_wall_hours_simple": median_total * 188 / 3600.0,
        "upper_budget_updates": 1500 * 500,
        "production_panel_images": 64,
        "raw_tensor_budget_bytes": 22_987_801_728,
        "dense_export_bytes": 1_454_400_000,
        "compression_credit_assumed": 0,
    }
    report = {
        "schema_version": 1,
        "status": "pass",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "indices": indices,
        "count_images": len(indices),
        "count_panel_images": 2,
        "statuses": {status: sum(record["status"] == status for record in records) for status in ("success", "budget_exhausted")},
        "first_success_steps": {str(record["index"]): record["first_success_step"] for record in records},
        "exact_match_counts_vs_old": exact_counts,
        "max_state_abs_error_vs_old_at_0_100_500": max_state,
        "max_telemetry_abs_error_vs_old": max_telemetry,
        "weights_buffers_unchanged": True,
        "analysis_pca32_basis_sha256": pca32["basis_sha256"],
        "analysis_pca32_energy_audit": pca32["energy_audit_fixed_basis"],
        "cuda_peak_memory_allocated_bytes": summary["cuda_peak_memory_allocated_bytes"],
        "cuda_peak_memory_reserved_bytes": summary["cuda_peak_memory_reserved_bytes"],
        "device": summary["device"],
        "smoke_output_bytes": total_bytes,
        "benchmark": estimate,
        "records": [
            {
                "index": record["index"],
                "status": record["status"],
                "first_success_step": record["first_success_step"],
                "timing_seconds": record["timing_seconds"],
                "bytes": record["replay_bytes"] + record["panel_bytes"],
            }
            for record in records
        ],
    }
    json_atomic(args.output / "SMOKE_VALIDATION.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
