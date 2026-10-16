"""Twarda bramka techniczna smoke przed kosztownym biegiem 1500 obrazów."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pelne_latenty import load_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = [int(value) for value in json.loads(args.indices.read_text(encoding="utf-8"))]
    details = []
    early_improved = 0
    any_improved = 0
    loss_decreased = 0
    descent_dot = None
    for index in indices:
        record_path = args.smoke / "records" / f"{index:06d}.json"
        artifact_path = args.smoke / "latents" / f"{index:06d}.npz"
        if not record_path.is_file() or not artifact_path.is_file():
            raise RuntimeError(f"Brak smoke dla {index}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        artifact = load_artifact(artifact_path)
        steps = artifact["telemetry_step"]
        if not np.array_equal(steps, np.arange(501, dtype=np.int32)):
            raise RuntimeError(f"Niepełna rzeczywista oś czasu {index}")
        if artifact["trajectory_steps"].tolist() != [100, 500]:
            raise RuntimeError(f"Złe milestone'y {index}")
        confidence = artifact["telemetry_confidence"].astype(np.float64)
        iou = artifact["telemetry_iou"].astype(np.float64)
        objective = artifact["telemetry_fit_objective"].astype(np.float64)
        if not np.isfinite(confidence).all() or not np.isfinite(iou).all() or not np.isfinite(objective).all():
            raise RuntimeError(f"Niefinitywna telemetria {index}")
        quality = np.minimum(confidence / 0.90, iou / 0.90)
        own_early = float(quality[1:33].max() - quality[0])
        own_any = float(quality[1:].max() - quality[0])
        own_loss = float(objective[1:].min() - objective[0])
        early_improved += own_early > 1e-6
        any_improved += own_any > 1e-6
        loss_decreased += own_loss < -1e-6
        if index == 0 and "control_z" in artifact:
            displacement = artifact["control_z"][1] - artifact["control_z"][0]
            gradient = artifact["control_gradient_before_step"][0]
            descent_dot = float(np.sum(displacement.astype(np.float64) * gradient))
        if record["full_forward_tail_max_abs_error"] > 2e-5:
            raise RuntimeError(f"Fast tail niezgodny dla {index}")
        if not record["weights_buffers_hash"] or not record["model_parameters_require_grad_false"]:
            raise RuntimeError(f"Model niezamrożony {index}")
        details.append(
            {
                "index": index,
                "status": record["status"],
                "early_best_joint_quality_delta": own_early,
                "all_steps_best_joint_quality_delta": own_any,
                "best_fit_objective_delta": own_loss,
                "first_success_step": record["first_success_step"],
                "seconds_per_step": record["timing_seconds"]["per_real_step"],
                "gpu_peak_reserved_bytes": record["gpu_peak_bytes"]["reserved"],
                "artifact_bytes": record["artifact_bytes"],
            }
        )

    summaries = list(args.smoke.glob("SUMMARY_*.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Oczekiwano jednego summary smoke, znaleziono {len(summaries)}")
    run_summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    checks = {
        "all_images_complete": run_summary.get("status") == "complete" and run_summary.get("assigned") == len(indices),
        "weights_and_buffers_unchanged": bool(run_summary.get("weights_buffers_unchanged")),
        "at_least_one_early_metric_improvement": early_improved >= 1,
        "at_least_one_any_metric_improvement": any_improved >= 1,
        "fit_objective_decreases_for_majority": loss_decreased >= (len(indices) // 2 + 1),
        "first_adam_update_is_descent_on_control_panel": descent_dot is not None and descent_dot < 0,
    }
    passed = all(checks.values())
    seconds_per_step = np.asarray([item["seconds_per_step"] for item in details])
    seconds_per_image = seconds_per_step * 500
    report = {
        "status": "pass" if passed else "blocked",
        "production_allowed_by_smoke": passed,
        "checks": checks,
        "alarm_from_independent_local_pilot_addressed": {
            "all_smoke_images_move_away_during_first_32_steps": early_improved == 0,
            "early_improved_count": early_improved,
            "any_improved_count": any_improved,
            "fit_loss_decreased_count": loss_decreased,
            "first_update_gradient_dot_displacement": descent_dot,
        },
        "benchmark": {
            "gpu": run_summary.get("device"),
            "seconds_per_step_median": float(np.median(seconds_per_step)),
            "seconds_per_step_range": [float(seconds_per_step.min()), float(seconds_per_step.max())],
            "seconds_per_image_500_median": float(np.median(seconds_per_image)),
            "estimated_1500_gpu_seconds_serial_equivalent": float(seconds_per_image.sum() / len(seconds_per_image) * 1500),
            "estimated_8_shard_seconds_compute_only": float(seconds_per_image.sum() / len(seconds_per_image) * 1500 / 8),
            "artifact_bytes_mean": float(np.mean([item["artifact_bytes"] for item in details])),
            "gpu_peak_reserved_bytes_max": int(max(item["gpu_peak_reserved_bytes"] for item in details)),
            "excludes_queue_and_shared_filesystem_variability": True,
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
