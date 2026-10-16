"""Sprawdza SHA-256 wszystkich plików wymienionych w DEPLOY_MANIFEST.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(
        (args.root / "provenance/DEPLOY_MANIFEST.json").read_text(encoding="utf-8")
    )
    failures = []
    for row in manifest["files"]:
        path = args.root / row["destination"]
        actual = sha256(path) if path.is_file() else None
        if actual != row["sha256"]:
            failures.append(
                {"path": row["destination"], "actual": actual, "expected": row["sha256"]}
            )
    print(json.dumps({"files": len(manifest["files"]), "failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
