"""Scala dokładnie 1500 oracle; dowolny brak, duplikat lub zły hash obala kompletność."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from pelne_latenty import load_artifact

EXPECTED_CHECKPOINT_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
EXPECTED_MANIFEST_SHA256 = "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def json_atomic(path: Path, value: object) -> None:
    atomic(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def quantiles(values: np.ndarray) -> dict:
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    confidence = np.empty((1500, 501), np.float32)
    iou = np.empty_like(confidence)
    delta_rms = np.empty_like(confidence)
    success_timeline = np.empty((1500, 501), np.uint8)
    index_lines = []
    total_bytes = 0
    for index in range(1500):
        record_path = args.production / "records" / f"{index:06d}.json"
        artifact_path = args.production / "latents" / f"{index:06d}.npz"
        if not record_path.exists() or not artifact_path.exists():
            raise RuntimeError(f"Brak pary wyników dla indeksu {index}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["index"] != index:
            raise RuntimeError(f"Niespójny indeks w {record_path}")
        if record["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(f"Zły checkpoint w {record_path}")
        if record["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError(f"Zły manifest w {record_path}")
        actual_hash = sha256(artifact_path)
        if record["artifact_sha256"] != actual_hash:
            raise RuntimeError(f"Zły hash {artifact_path}")
        artifact = load_artifact(artifact_path)
        if artifact["before"].shape != (600, 768) or artifact["after"].shape != (600, 768):
            raise RuntimeError(f"Zły latent {artifact_path}")
        steps = artifact["telemetry_step"]
        if not np.array_equal(steps, np.arange(501, dtype=np.int32)):
            raise RuntimeError(f"Oś czasu nie jest rzeczywistym wspólnym prefixem dla {index}")
        if artifact["trajectory_steps"].tolist() != [100, 500]:
            raise RuntimeError(f"Złe milestone'y dla {index}")
        confidence[index] = artifact["telemetry_confidence"]
        iou[index] = artifact["telemetry_iou"]
        delta_rms[index] = artifact["telemetry_delta_rms"]
        success_timeline[index] = artifact["telemetry_success"]
        total_bytes += artifact_path.stat().st_size
        records.append(record)
        index_lines.append(
            json.dumps(
                {
                    "index": index,
                    "frame_id": record["frame_id"],
                    "image_name": record["image_name"],
                    "status": record["status"],
                    "first_success_step": record["first_success_step"],
                    "after_step": record["after_step"],
                    "artifact": str(artifact_path),
                    "artifact_sha256": actual_hash,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic(args.output / "ORACLE_INDEX.jsonl", "".join(index_lines).encode("utf-8"))
    np.savez_compressed(
        args.output / "TIMELINE_AGGREGATE.npz",
        steps=np.arange(501, dtype=np.int32),
        confidence_mean=confidence.mean(axis=0),
        confidence_median=np.median(confidence, axis=0),
        iou_mean=iou.mean(axis=0),
        iou_median=np.median(iou, axis=0),
        delta_rms_mean=delta_rms.mean(axis=0),
        success_count=success_timeline.sum(axis=0),
    )
    successful = np.asarray([bool(record["both90"]) for record in records])
    legacy_mismatch = np.asarray(
        [bool(record["campaign_legacy_bbox_differs"]) for record in records], dtype=bool
    )
    first = np.asarray(
        [record["first_success_step"] if record["first_success_step"] is not None else 501 for record in records],
        dtype=np.int32,
    )
    summary = {
        "status": "complete",
        "count": len(records),
        "success_count": int(successful.sum()),
        "budget_exhausted_count": int((~successful).sum()),
        "success_fraction": float(successful.mean()),
        "first_success_step_quantiles_success_only": quantiles(first[successful]) if successful.any() else None,
        "baseline_confidence_quantiles": quantiles(np.asarray([r["baseline"]["confidence"] for r in records])),
        "baseline_iou_quantiles": quantiles(np.asarray([r["baseline"]["iou"] for r in records])),
        "after_confidence_quantiles": quantiles(np.asarray([r["after"]["confidence"] for r in records])),
        "after_iou_quantiles": quantiles(np.asarray([r["after"]["iou"] for r in records])),
        "actual_artifact_bytes": total_bytes,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "timeline": {
            "real_steps": [0, 500],
            "full_latent_milestones": [0, 100, 500],
            "step0_stored_as": "before",
            "terminal_padding": False,
        },
        "ground_truth_resize_audit": {
            "per_axis_correct_gt_count": 1500,
            "campaign_legacy_bbox_same_count": int((~legacy_mismatch).sum()),
            "campaign_legacy_bbox_different_count": int(legacy_mismatch.sum()),
            "success_count_legacy_same": int((successful & ~legacy_mismatch).sum()),
            "success_count_legacy_different": int((successful & legacy_mismatch).sum()),
            "legacy_box_used_as_gt": False,
        },
    }
    json_atomic(args.output / "ORACLE_SUMMARY.json", summary)
    failures = [
        {
            "index": r["index"],
            "frame_id": r["frame_id"],
            "confidence": r["after"]["confidence"],
            "iou": r["after"]["iou"],
            "geometry_iou_upper_bound": r["target"]["geometry_iou_upper_bound"],
            "campaign_legacy_bbox_differs": r["campaign_legacy_bbox_differs"],
        }
        for r in records
        if not r["both90"]
    ]
    json_atomic(args.output / "ORACLE_UNMET.json", failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
