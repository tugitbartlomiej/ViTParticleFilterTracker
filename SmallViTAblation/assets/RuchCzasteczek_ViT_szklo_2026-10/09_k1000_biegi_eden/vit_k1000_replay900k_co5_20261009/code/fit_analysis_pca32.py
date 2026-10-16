"""Dopasowuje jedną zamrożoną PCA32 wyłącznie ze starych stanów filmów train."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pelne_latenty import load_artifact
from replay_core import basis_sha256, sha256, write_npz_atomic
from uruchom_replay import json_atomic

EXPECTED_MANIFEST_SHA256 = "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7"
EXPECTED_OLD_INDEX_SHA256 = "e990f9032efc8b8e13727752d099488688d0f643e1b49e6d0ac601eb3caee3fb"
EXPECTED_PANEL_SHA256 = "4cd91f3147e8872b128246293349c34c6307d3107621dc6ccf54399aefc57e75"
FIT_STEPS = (0, 100, 500)
GRID_ROWS = 4
GRID_COLS = 4
SELECTION_SEED = "pca32-train-only-spatial-v1-20261009"


def spatial_tokens(image_index: int) -> np.ndarray:
    """Jeden deterministyczny token z każdej z 16 komórek siatki 4x4."""
    row_groups = np.array_split(np.arange(20, dtype=np.int32), GRID_ROWS)
    col_groups = np.array_split(np.arange(30, dtype=np.int32), GRID_COLS)
    result = []
    for row_group_index, rows in enumerate(row_groups):
        for col_group_index, columns in enumerate(col_groups):
            digest = hashlib.sha256(
                f"{SELECTION_SEED}|{image_index}|{row_group_index}|{col_group_index}".encode()
            ).digest()
            row = int(rows[int.from_bytes(digest[:4], "little") % len(rows)])
            col = int(columns[int.from_bytes(digest[4:8], "little") % len(columns)])
            result.append(row * 30 + col)
    tokens = np.asarray(result, dtype=np.int32)
    if tokens.shape != (16,) or len(np.unique(tokens)) != 16:
        raise RuntimeError(tokens)
    return tokens


def states_from_artifact(path: Path) -> np.ndarray:
    artifact = load_artifact(path)
    mapping = {int(step): artifact["trajectory"][slot] for slot, step in enumerate(artifact["trajectory_steps"])}
    if 100 not in mapping or 500 not in mapping:
        raise RuntimeError(f"Brak historycznych kroków 100/500: {path}")
    states = np.stack([artifact["before"], mapping[100], mapping[500]], axis=0)
    if states.shape != (3, 600, 768) or states.dtype != np.float32 or not np.isfinite(states).all():
        raise RuntimeError((path, states.shape, states.dtype))
    return states


def split_indices(old_rows: list[dict], panel: dict) -> dict[str, list[int]]:
    films = {
        split: {row["video_id"] for row in panel["entries"] if row["analysis_split"] == split}
        for split in ("train", "validation", "test")
    }
    if {key: len(value) for key, value in films.items()} != {"train": 40, "validation": 12, "test": 12}:
        raise RuntimeError("Niewłaściwy split filmów panelu")
    if films["train"] & films["validation"] or films["train"] & films["test"] or films["validation"] & films["test"]:
        raise RuntimeError("Przeciek filmów")
    return {
        split: [
            int(row["index"])
            for row in old_rows
            if row["frame_id"].rsplit("/", 1)[0] in films[split]
        ]
        for split in films
    }


def energy_audit(
    indices: list[int], old_rows: list[dict], old_production: Path, mean: np.ndarray, components: np.ndarray
) -> dict:
    state_total = state_captured = movement_total = movement_captured = 0.0
    samples = movement_samples = 0
    for index in indices:
        row = old_rows[index]
        path = old_production / "latents" / f"{index:06d}.npz"
        states = states_from_artifact(path).astype(np.float64)
        tokens = spatial_tokens(index)
        selected = states[:, tokens]
        centered = selected - mean
        projected = centered @ components
        state_total += float(np.square(centered).sum(dtype=np.float64))
        state_captured += float(np.square(projected).sum(dtype=np.float64))
        samples += int(centered.shape[0] * centered.shape[1])
        increments = np.diff(selected, axis=0)
        projected_increments = increments @ components
        movement_total += float(np.square(increments).sum(dtype=np.float64))
        movement_captured += float(np.square(projected_increments).sum(dtype=np.float64))
        movement_samples += int(increments.shape[0] * increments.shape[1])
    return {
        "images": len(indices),
        "state_samples": samples,
        "movement_samples": movement_samples,
        "state_energy_total": state_total,
        "state_energy_captured": state_captured,
        "state_energy_fraction": state_captured / state_total,
        "movement_energy_total": movement_total,
        "movement_energy_captured": movement_captured,
        "movement_energy_fraction": movement_captured / movement_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument("--old-production", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    input_hashes = {
        "exact1500_manifest": sha256(args.manifest),
        "old_oracle_index": sha256(args.old_index),
        "panel_split": sha256(args.panel),
    }
    expected = {
        "exact1500_manifest": EXPECTED_MANIFEST_SHA256,
        "old_oracle_index": EXPECTED_OLD_INDEX_SHA256,
        "panel_split": EXPECTED_PANEL_SHA256,
    }
    if input_hashes != expected:
        raise RuntimeError((input_hashes, expected))
    if args.output.exists() or args.manifest_output.exists():
        if not args.output.is_file() or not args.manifest_output.is_file():
            raise RuntimeError("Częściowa publikacja PCA32")
        existing = json.loads(args.manifest_output.read_text(encoding="utf-8"))
        with np.load(args.output, allow_pickle=False) as data:
            digest = basis_sha256(data["pca_mean"], data["pca_components"])
        if existing.get("status") != "complete" or existing.get("basis_sha256") != digest:
            raise RuntimeError("Istniejąca PCA32 nie przechodzi fresh-load")
        print(json.dumps({"status": "reused", "basis_sha256": digest}), flush=True)
        return 0

    exact_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    old_rows = [json.loads(line) for line in args.old_index.read_text(encoding="utf-8").splitlines() if line]
    if len(exact_rows) != 1500 or len(old_rows) != 1500:
        raise RuntimeError("Niewłaściwa liczba obrazów")
    for index, (exact, old) in enumerate(zip(exact_rows, old_rows, strict=True)):
        if exact["index"] != index or old["index"] != index or exact["frame_id"] != old["frame_id"]:
            raise RuntimeError(f"Rozjazd manifestów {index}")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    split = split_indices(old_rows, panel)
    if not split["train"]:
        raise RuntimeError("Brak obrazów train")

    sum_vector = np.zeros(768, dtype=np.float64)
    gram = np.zeros((768, 768), dtype=np.float64)
    sample_count = 0
    for ordinal, index in enumerate(split["train"], 1):
        row = old_rows[index]
        path = args.old_production / "latents" / f"{index:06d}.npz"
        if sha256(path) != row["artifact_sha256"]:
            raise RuntimeError(f"Hash train oracle {index}")
        states = states_from_artifact(path).astype(np.float64)
        samples = states[:, spatial_tokens(index)].reshape(-1, 768)
        sum_vector += samples.sum(axis=0, dtype=np.float64)
        gram += samples.T @ samples
        sample_count += samples.shape[0]
        if ordinal % 50 == 0:
            print(json.dumps({"pca_fit_images": ordinal, "samples": sample_count}), flush=True)

    mean = sum_vector / sample_count
    covariance = (gram - np.outer(sum_vector, sum_vector) / sample_count) / (sample_count - 1)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order[:32]].copy()
    for axis in range(components.shape[1]):
        pivot = int(np.abs(components[:, axis]).argmax())
        if components[pivot, axis] < 0:
            components[:, axis] *= -1
    orthogonality_error = float(np.max(np.abs(components.T @ components - np.eye(32))))
    if orthogonality_error > 1e-10:
        raise RuntimeError(f"PCA32 nie jest ortonormalna: {orthogonality_error}")
    basis_digest = basis_sha256(mean, components)
    audit = {
        name: energy_audit(indices, old_rows, args.old_production, mean, components)
        for name, indices in split.items()
    }
    arrays = {
        "pca_mean": mean.astype(np.float64),
        "pca_components": components.astype(np.float64),
        "eigenvalues": eigenvalues.astype(np.float64),
        "fit_steps": np.asarray(FIT_STEPS, dtype=np.int32),
        "fit_image_indices": np.asarray(split["train"], dtype=np.int32),
    }
    write_npz_atomic(args.output, arrays)
    if sha256(args.output) == "":
        raise RuntimeError("Nieosiągalna kontrola zapisu")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "role": "train_only_analytical_PCA32_not_visual_PCA3",
        "fit_source": "historical exact oracle states at actual steps 0,100,500",
        "fit_split": "40 panel-contract train films only",
        "validation_and_test_used_for_fit": False,
        "fit_steps": list(FIT_STEPS),
        "spatial_sampling": {
            "tokens_per_image": 16,
            "method": "one deterministic token from each 4x4 spatial stratum of the 20x30 grid",
            "seed": SELECTION_SEED,
        },
        "fit_image_count": len(split["train"]),
        "fit_sample_count": sample_count,
        "split_image_counts": {name: len(indices) for name, indices in split.items()},
        "fit_image_indices": split["train"],
        "dimensions_input": 768,
        "dimensions_output": 32,
        "centering": "single train-only float64 mean shared by all images, tokens, states and steps",
        "solver": "exact float64 covariance eigen-decomposition; component signs canonicalized",
        "basis_sha256": basis_digest,
        "container_sha256": sha256(args.output),
        "orthogonality_max_abs": orthogonality_error,
        "explained_variance_fraction_train_covariance": float(eigenvalues[:32].sum() / np.maximum(eigenvalues, 0).sum()),
        "energy_audit_fixed_basis": audit,
        "input_hashes": input_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    json_atomic(args.manifest_output, manifest)
    with np.load(args.output, allow_pickle=False) as restored:
        if basis_sha256(restored["pca_mean"], restored["pca_components"]) != basis_digest:
            raise RuntimeError("Fresh-load PCA32 nie zgadza się")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
