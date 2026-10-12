"""Weryfikuje chronione wejścia, stare oracle, miejsce i brak duplikatu przed SLURM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from replay_core import PCA_BASIS_SHA256, basis_sha256
from uruchom_replay import load_model


EXPECTED = {
    "checkpoint": "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e",
    "adamw": "e7a7727ccf19dd3801bd8ec4de65887480252ccdf7e21d385346c13a139259b6",
    "coco": "3667dd12c8fddcb695c9ce6144ca4bf3f5300f70a47322a8da12b92de6d98f75",
    "archive": "5850b5c8dd4411159061e254e1f9b32e2b4b633036d3990049b553bea767d8b1",
    "manifest": "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7",
    "old_index": "e990f9032efc8b8e13727752d099488688d0f643e1b49e6d0ac601eb3caee3fb",
    "pca": "3e031f0555f0a08b2de0d6e6cbbc32dc15113705e950574152bd1713fdb536f7",
    "panel": "4cd91f3147e8872b128246293349c34c6307d3107621dc6ccf54399aefc57e75",
    "smoke_indices": "3378d1697dabe5bd014e5dfc9e4a7c91046cd3b302775899d5de26cb05639fc2",
}
MINIMUM_FREE_BYTES = 35_000_000_000
RAW_TENSOR_BUDGET_BYTES = 22_987_801_728
DENSE_EXPORT_BYTES = 1_454_400_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adamw", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-index", type=Path, required=True)
    parser.add_argument("--old-production", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--smoke-indices", type=Path, required=True)
    parser.add_argument("--analysis-pca32", type=Path)
    parser.add_argument("--analysis-pca32-manifest", type=Path)
    parser.add_argument("--verify-old-artifact-hashes", action="store_true")
    args = parser.parse_args()

    paths = {
        "checkpoint": args.checkpoint,
        "adamw": args.adamw,
        "coco": args.coco,
        "archive": args.archive,
        "manifest": args.manifest,
        "old_index": args.old_index,
        "pca": args.pca,
        "panel": args.panel,
        "smoke_indices": args.smoke_indices,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Brak chronionych wejść: {missing}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise RuntimeError({name: (hashes[name], EXPECTED[name]) for name in hashes if hashes[name] != EXPECTED[name]})

    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(manifest) != 1500 or [row["index"] for row in manifest] != list(range(1500)):
        raise RuntimeError("Manifest nie jest dokładnym uporządkowanym exact1500")
    if len({row["image_sha256"] for row in manifest}) != 1500:
        raise RuntimeError("Manifest exact1500 zawiera duplikaty obrazów")
    old_rows = [json.loads(line) for line in args.old_index.read_text(encoding="utf-8").splitlines() if line]
    if len(old_rows) != 1500 or [row["index"] for row in old_rows] != list(range(1500)):
        raise RuntimeError("Stary indeks oracle nie odpowiada exact1500")
    old_files = sorted((args.old_production / "latents").glob("*.npz"))
    if len(old_files) != 1500:
        raise RuntimeError(f"Stare oracle ma {len(old_files)} plików zamiast 1500")
    verified_old_hashes = 0
    if args.verify_old_artifact_hashes:
        for position, row in enumerate(old_rows, 1):
            path = args.old_production / "latents" / f"{row['index']:06d}.npz"
            if sha256(path) != row["artifact_sha256"]:
                raise RuntimeError(f"Niezgodny historyczny artefakt: {path}")
            verified_old_hashes += 1
            if position % 100 == 0:
                print(f"verified_old_artifacts={position}/1500", flush=True)
    elif args.mode == "production":
        prior_path = args.root / "preflight" / "PREFLIGHT_SMOKE.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("status") != "pass" or prior.get("verified_old_artifact_hashes") != 1500:
            raise RuntimeError("Produkcja wymaga pełnej kontroli hashy starego oracle ze smoke preflight")
        verified_old_hashes = 1500

    analysis_pca32 = None
    if args.mode == "production":
        if args.analysis_pca32 is None or args.analysis_pca32_manifest is None:
            raise RuntimeError("Produkcja wymaga analitycznej PCA32")
        pca32_meta = json.loads(args.analysis_pca32_manifest.read_text(encoding="utf-8"))
        if pca32_meta.get("status") != "complete" or sha256(args.analysis_pca32) != pca32_meta.get("container_sha256"):
            raise RuntimeError("Analityczna PCA32 nie przechodzi preflight")
        with np.load(args.analysis_pca32, allow_pickle=False) as data:
            pca32_basis = basis_sha256(data["pca_mean"], data["pca_components"])
            if data["pca_components"].shape != (768, 32):
                raise RuntimeError("Niewłaściwy wymiar PCA32")
        if pca32_basis != pca32_meta.get("basis_sha256") or pca32_meta.get("validation_and_test_used_for_fit") is not False:
            raise RuntimeError("Niespójna lub przeciekająca PCA32")
        analysis_pca32 = {
            "basis_sha256": pca32_basis,
            "container_sha256": pca32_meta["container_sha256"],
            "fit_image_count": pca32_meta["fit_image_count"],
            "energy_audit_fixed_basis": pca32_meta["energy_audit_fixed_basis"],
        }

    with np.load(args.pca, allow_pickle=False) as pca:
        if basis_sha256(pca["pca_mean"], pca["pca_components"]) != PCA_BASIS_SHA256:
            raise RuntimeError("Niewłaściwa wizualizacyjna baza PCA")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if len(panel["entries"]) != 64:
        raise RuntimeError("Panel nie ma 64 obrazów")
    if json.loads(args.smoke_indices.read_text(encoding="utf-8")) != [91, 684, 0]:
        raise RuntimeError("Niewłaściwy smoke")

    model, model_fingerprint = load_model(args.checkpoint, torch.device("cpu"))
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Model nie jest zamrożony")
    disk = shutil.disk_usage(args.root.parent)
    queue = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", "bpiotrowski"), "-o", "%i|%j|%T"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    relevant_jobs = [line for line in queue if any(token in line for token in ("k1k_r9", "k1k1500", "k1k_patch_joint"))]
    duplicates = [line for line in queue if "k1k_r9" in line]
    processes = subprocess.run(
        ["ps", "-u", os.environ.get("USER", "bpiotrowski"), "-o", "pid=,args="],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    destructive = [line for line in processes if any(token in line for token in ("PLAN_USUNIECIA", "rm -rf", " -delete"))]
    marker = args.root / "submission" / f"{args.mode.upper()}_SUBMITTED.json"
    checks = {
        "exact_input_hashes": True,
        "exact1500_order_and_unique_images": True,
        "old_oracle_count": len(old_files) == 1500,
        "old_oracle_hashes_verified": verified_old_hashes == 1500,
        "visual_pca_basis_hash": True,
        "checkpoint_strict_load": True,
        "model_parameters_frozen": True,
        "full_adamw_protected": args.adamw.stat().st_size == 35_212_799,
        "runtime_environment_exists": Path(sys.executable).is_file(),
        "analysis_pca32_ready_for_production": args.mode != "production" or analysis_pca32 is not None,
        "free_space_with_reserve": disk.free >= MINIMUM_FREE_BYTES,
        "no_active_replay_duplicate": not duplicates,
        "no_active_destructive_process": not destructive,
        "no_submission_marker": not marker.exists(),
    }
    report = {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "blocked",
        "mode": args.mode,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "hashes": hashes,
        "manifest_count": len(manifest),
        "old_oracle_count": len(old_files),
        "verified_old_artifact_hashes": verified_old_hashes,
        "model_weights_buffers_fingerprint": model_fingerprint,
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total, "used_bytes": disk.used},
        "storage_budget": {
            "raw_tensor_budget_bytes": RAW_TENSOR_BUDGET_BYTES,
            "dense_export_bytes": DENSE_EXPORT_BYTES,
            "total_large_output_bytes": RAW_TENSOR_BUDGET_BYTES + DENSE_EXPORT_BYTES,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "compression_credit_assumed": 0,
        },
        "relevant_active_jobs": relevant_jobs,
        "active_replay_duplicates": duplicates,
        "active_destructive_processes": destructive,
        "python": sys.version,
        "analysis_pca32": analysis_pca32,
        "protected_dependencies": [str(path) for path in paths.values()],
    }
    atomic_json(args.root / "preflight" / f"PREFLIGHT_{args.mode.upper()}.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
