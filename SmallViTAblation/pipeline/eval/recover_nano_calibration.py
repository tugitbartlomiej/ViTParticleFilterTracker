#!/usr/bin/env python3
"""Finish last/best validation reports for already trained Nano calibration runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def checkpoint_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validated_metrics(path: Path, expected_images: int) -> dict | None:
    if not path.exists():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    block = metrics.get("results", {}).get("val")
    if block is None or int(block.get("n_images", -1)) != expected_images:
        return None
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run = args.run_root / args.run_id
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model") != "rf-detr-nano" or int(manifest.get("epochs_completed", -1)) != 30:
        raise RuntimeError("Recovery requires a complete 30-epoch RF-DETR Nano run")
    expected_images = int(manifest["dataset"]["n_val_images"])
    started = time.monotonic()
    checkpoints = {}

    for filename, rule in (("last.pth", "last_epoch"), ("best.pth", "best_val")):
        weights = run / filename
        output = run / ("eval_" + rule)
        metrics_path = output / "metrics.json"
        block = validated_metrics(metrics_path, expected_images)
        if block is None:
            predictions = output / "predictions_val.json"
            command = [
                sys.executable, str(Path(__file__).with_name("evaluate.py")),
                "--protocol", str(args.protocol), "--run-id", args.run_id,
                "--checkpoint-rule", rule, "--set", "val",
                "--annotations", str(args.data_root / "annotations_val.json"),
                "--images-dir", str(args.data_root), "--device", "cuda",
                "--model-name", "rf-detr-nano",
                "--resolution", str(manifest["resolution"]),
                "--output-dir", str(output),
            ]
            if predictions.exists():
                command += ["--predictions-file", str(predictions)]
            else:
                command += ["--checkpoint", str(weights)]
            subprocess.run(command, check=True)
            block = validated_metrics(metrics_path, expected_images)
            if block is None:
                raise RuntimeError(f"Incomplete validation aggregate: {metrics_path}")
        checkpoints[rule] = {
            "checkpoint_sha256": checkpoint_sha256(weights),
            "metrics": block,
        }

    report = {
        "status": "passed",
        "scope": "training_recipe_calibration",
        "recovered_evaluation_only": True,
        "training_seed": manifest["seed"],
        "training_source_commit": manifest.get("git_commit"),
        "evaluation_source_commit": os.environ.get("SMALLVIT_SOURCE_COMMIT"),
        "recovery_slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "recovery_seconds": time.monotonic() - started,
        "evaluation_set": "val",
        "test_evaluated": False,
        "checkpoints": checkpoints,
    }
    (run / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
