"""Tworzy powtarzalny snapshot kodu, konfiguracji, środowiska i wejść biegu."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--eden", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-adamw", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code_files = []
    for root in (args.code, args.eden):
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in {".py", ".sh", ".slurm", ".md", ".json"}:
                code_files.append(path)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available_at_snapshot": torch.cuda.is_available(),
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
        },
        "code_sha256": {
            f"{path.parent.name}/{path.name}": sha256(path) for path in code_files
        },
        "inputs": {
            "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
            "checkpoint_adamw_protected": {
                "path": str(args.checkpoint_adamw),
                "bytes": args.checkpoint_adamw.stat().st_size,
                "sha256": sha256(args.checkpoint_adamw),
                "used_for_inference": False,
                "must_preserve_for_recovery": True,
            },
            "coco": {"path": str(args.coco), "sha256": sha256(args.coco)},
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "archive": {
                "path": str(args.archive),
                "bytes": args.archive.stat().st_size,
                "sha256": sha256(args.archive),
                "access": "stream members; never extract or save new images",
            },
        },
        "method": {
            "precision": "FP32; TF32 disabled",
            "latent_shape": [600, 768],
            "horizon": 500,
            "full_fp32_states": [0, 100, 500],
            "optimizer": "Adam over per-image latent only",
            "schedule": [[1, 100, 0.05], [101, 250, 0.15], [251, 400, 0.05], [401, 500, 0.01]],
            "gelu_floor": -0.16997121,
            "model_weights_and_buffers": "eval and frozen",
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
