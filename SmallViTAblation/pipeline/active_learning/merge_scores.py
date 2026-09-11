#!/usr/bin/env python3
"""Join label-free acquisition scores by frame_id with conflict checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(paths: list[Path]) -> list[dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        seen = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                frame_id = str(row.get("frame_id", ""))
                if not frame_id:
                    raise ValueError(f"{path}:{line_number}: missing frame_id")
                if frame_id in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate frame_id={frame_id}")
                seen.add(frame_id)
                target = merged.setdefault(frame_id, {"frame_id": frame_id})
                for key, value in row.items():
                    if key == "frame_id":
                        continue
                    if key in target and target[key] != value:
                        raise ValueError(
                            f"conflicting {key!r} for frame_id={frame_id} while reading {path}")
                    target[key] = value
    return [merged[frame_id] for frame_id in sorted(merged)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = merge(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_name(args.out.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(args.out)
    manifest = {
        "schema_version": 1,
        "rows": len(rows),
        "inputs": [{"path": str(path.resolve()), "sha256": _hash(path)}
                   for path in args.inputs],
        "output_sha256": _hash(args.out),
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
