"""Blokuje wysłanie, jeśli exact1500, K1000, miejsce albo brak duplikatu nie są potwierdzone."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    "checkpoint": "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e",
    "adamw": "e7a7727ccf19dd3801bd8ec4de65887480252ccdf7e21d385346c13a139259b6",
    "coco": "3667dd12c8fddcb695c9ce6144ca4bf3f5300f70a47322a8da12b92de6d98f75",
    "archive": "5850b5c8dd4411159061e254e1f9b32e2b4b633036d3990049b553bea767d8b1",
    "manifest": "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7",
}
MIN_FREE_BYTES = 20_000_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deploy", "production"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adamw", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--old-evaluation", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "checkpoint": args.checkpoint,
        "adamw": args.adamw,
        "coco": args.coco,
        "archive": args.archive,
        "manifest": args.manifest,
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise RuntimeError({name: (hashes[name], EXPECTED[name]) for name in hashes if hashes[name] != EXPECTED[name]})
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1500 or [row.get("index") for row in rows] != list(range(1500)):
        raise RuntimeError("Manifest nie jest dokładnym uporządkowanym zbiorem 1500")
    if len({row["image_sha256"] for row in rows}) != 1500:
        raise RuntimeError("Manifest zawiera powtórzone obrazy")
    old = json.loads(args.old_evaluation.read_text(encoding="utf-8"))
    if old.get("count") != 1500 or abs(old["methods"]["baseline"]["detection"]["mAP50_95"] - 0.24549623827449926) > 1e-15:
        raise RuntimeError("Historyczna ewaluacja nie odpowiada exact1500")
    free = shutil.disk_usage(args.root).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"Za mało miejsca: {free}")
    queue = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%T|%j"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    duplicates = [line for line in queue if "k1k_patch_joint" in line]
    if duplicates:
        raise RuntimeError(f"Aktywny duplikat: {duplicates}")
    if args.mode == "production":
        smoke_path = args.root / "results" / "smoke" / "SMOKE_RESULT.json"
        if not smoke_path.exists() or json.loads(smoke_path.read_text(encoding="utf-8")).get("status") != "pass":
            raise RuntimeError("Brak poprawnego smoke")
        if (args.root / "submission" / "PRODUCTION_SUBMITTED.json").exists():
            raise RuntimeError("Produkcja została już zgłoszona")
    result = {
        "status": "pass",
        "mode": args.mode,
        "hashes": hashes,
        "manifest_count": len(rows),
        "free_bytes": free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "active_duplicate_jobs": duplicates,
        "python": sys.version,
        "old_evaluation_sha256": sha256(args.old_evaluation),
    }
    target = args.root / "preflight" / f"PREFLIGHT_{args.mode.upper()}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
