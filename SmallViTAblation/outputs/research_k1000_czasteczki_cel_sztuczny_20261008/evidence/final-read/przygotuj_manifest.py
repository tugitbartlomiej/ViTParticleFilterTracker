"""Rozstrzyga tożsamość dokładnej bazy 1500; błąd dowolnego hasha blokuje bieg."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from dane import LocalImageSource, TarImageSource, build_manifest, canonical_jsonl
from pelne_latenty import choose_target_token

EXPECTED_COCO_SHA256 = "3667dd12c8fddcb695c9ce6144ca4bf3f5300f70a47322a8da12b92de6d98f75"
EXPECTED_CHECKPOINT_SHA256 = "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e"
COUNT = 1500
TOKENS = 600
DIMENSIONS = 768
HORIZON = 500
MILESTONES = [100, 500]
CONTROL_IMAGES = 16
CONTROL_TOKENS = 8
FIXED_SMOKE_INDICES = [1361, 406, 1298, 739, 21, 665, 0, 997]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
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
    write_atomic(
        path,
        (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"),
    )


def smoke_indices(rows: list[dict]) -> list[int]:
    """Zamrożony panel: skrajne pola/geometria plus indeks 0 z kontrolą gradientu."""
    if len(rows) != COUNT or any(index < 0 or index >= len(rows) for index in FIXED_SMOKE_INDICES):
        raise ValueError("Nie można zastosować zamrożonego panelu smoke")
    return FIXED_SMOKE_INDICES.copy()


def _derive_smoke_indices_for_audit(rows: list[dict]) -> list[int]:
    """Rekonstrukcja kryterium użytego przy pierwotnym doborze; nie zmienia panelu."""
    area = np.asarray(
        [
            (row["bbox_xyxy_480"][2] - row["bbox_xyxy_480"][0])
            * (row["bbox_xyxy_480"][3] - row["bbox_xyxy_480"][1])
            for row in rows
        ]
    )
    geometry = np.asarray(
        [choose_target_token(np.asarray(row["bbox_xyxy_480"]))["geometry_iou_upper_bound"] for row in rows]
    )
    chosen: list[int] = []
    for values, quantiles in (
        (area, [0.0, 0.1, 0.5, 0.9, 1.0]),
        (geometry, [0.0, 0.5, 1.0]),
    ):
        for quantile in quantiles:
            target = float(np.quantile(values, quantile))
            for index in np.argsort(np.abs(values - target)):
                if int(index) not in chosen:
                    chosen.append(int(index))
                    break
    return chosen


def control_panel() -> dict[str, list[int]]:
    images = np.rint(np.linspace(0, COUNT - 1, CONTROL_IMAGES)).astype(int).tolist()
    tokens = [0, 29, 145, 284, 315, 454, 570, 599]
    return {str(index): tokens for index in images}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-image-root", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()

    coco_hash = sha256(args.coco)
    checkpoint_hash = sha256(args.checkpoint)
    if coco_hash != EXPECTED_COCO_SHA256:
        raise RuntimeError(f"Niewłaściwy COCO: {coco_hash}")
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Niewłaściwy K1000: {checkpoint_hash}")

    image_source = (
        LocalImageSource(args.local_image_root)
        if args.local_image_root
        else TarImageSource(args.archive)
    )
    try:
        rows = build_manifest(args.coco, image_source.read_bytes)
    finally:
        image_source.close()
    if len(rows) != COUNT:
        raise RuntimeError(f"Manifest ma {len(rows)} zamiast {COUNT} obrazów")
    payload = canonical_jsonl(rows)
    manifest_hash = hashlib.sha256(payload).hexdigest()
    if args.verify_existing is not None:
        existing = args.verify_existing.read_bytes()
        if existing != payload:
            raise RuntimeError(
                f"Manifest różni się: istniejący {hashlib.sha256(existing).hexdigest()}, "
                f"odtworzony {manifest_hash}"
            )

    provenance = args.output
    provenance.mkdir(parents=True, exist_ok=True)
    write_atomic(provenance / "MANIFEST_1500.jsonl", payload)
    json_atomic(provenance / "SMOKE_INDICES.json", smoke_indices(rows))
    json_atomic(provenance / "CONTROL_PANEL.json", control_panel())

    endpoint_bytes = COUNT * TOKENS * DIMENSIONS * 4 * 3
    trajectory_bytes = COUNT * TOKENS * DIMENSIONS * 4 * len(MILESTONES)
    control_particles = CONTROL_IMAGES * CONTROL_TOKENS
    control_bytes = control_particles * DIMENSIONS * 4 * ((HORIZON + 1) + 3 * HORIZON)
    metadata = {
        "status": "verified",
        "count": COUNT,
        "annotations": COUNT,
        "one_annotation_per_image": True,
        "coco_sha256": coco_hash,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "source_mode": "local" if args.local_image_root else "tar_stream_without_extraction",
        "shape_per_image": [TOKENS, DIMENSIONS],
        "horizon": HORIZON,
        "milestones_fp32": MILESTONES,
        "raw_bytes": {
            "one_fp32_snapshot_all_images": COUNT * TOKENS * DIMENSIONS * 4,
            "one_hundred_fp32_snapshots": COUNT * TOKENS * DIMENSIONS * 4 * 100,
            "one_thousand_fp32_snapshots": COUNT * TOKENS * DIMENSIONS * 4 * 1000,
            "before_after_delta": endpoint_bytes,
            "trajectory_milestones": trajectory_bytes,
            "dense_control_panel": control_bytes,
            "main_tensors_plus_control_panel": endpoint_bytes + trajectory_bytes + control_bytes,
        },
    }
    json_atomic(provenance / "MANIFEST_METADATA.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
