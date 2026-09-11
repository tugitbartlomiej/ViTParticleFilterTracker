#!/usr/bin/env python3
"""One-way zero-look gate for an external evaluation.

The ledger records access before evaluation starts. A failed external run still
counts as an access and requires an explicit protocol amendment, rather than a
silent retry with another checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket

from . import evaluate


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint-rule", choices=["last_epoch", "best_val"], required=True)
    ap.add_argument("--set", dest="set_name",
                    choices=["external_full", "external_clean"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--operating-threshold", type=float, required=True,
                    help="threshold frozen on internal validation")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--resolution", type=int, default=None)
    args = ap.parse_args(argv)

    lock = args.ledger.with_suffix(args.ledger.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        existing = []
        if args.ledger.exists():
            existing = [json.loads(line) for line in args.ledger.read_text(
                encoding="utf-8").splitlines() if line.strip()]
        checkpoint_hash = _sha256(args.checkpoint)
        prior_run = [row for row in existing if row.get("run_id") == args.run_id
                     and row.get("event") == "started"]
        prior_set = [row for row in prior_run if row.get("set") == args.set_name]
        if prior_set:
            raise RuntimeError(
                f"external set {args.set_name} was already opened for run {args.run_id}")
        incompatible = [row for row in prior_run
                        if row.get("checkpoint_sha256") != checkpoint_hash
                        or row.get("checkpoint_rule") != args.checkpoint_rule]
        if incompatible:
            raise RuntimeError(
                f"run {args.run_id} already opened an external view with another "
                "checkpoint or checkpoint rule")
        access_id = hashlib.sha256(
            f"{args.run_id}:{args.set_name}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:20]
        common = {
            "access_id": access_id, "run_id": args.run_id, "set": args.set_name,
            "checkpoint_rule": args.checkpoint_rule,
            "checkpoint_sha256": checkpoint_hash,
            "annotations_sha256": _sha256(args.annotations),
            "hostname": socket.gethostname(),
        }
        _append(args.ledger, {**common, "event": "started",
                             "at": datetime.now(timezone.utc).isoformat()})
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass

    forwarded = [
        "--protocol", str(args.protocol), "--run-id", args.run_id,
        "--checkpoint-rule", args.checkpoint_rule, "--set", args.set_name,
        "--checkpoint", str(args.checkpoint), "--annotations", str(args.annotations),
        "--images-dir", str(args.images_dir), "--output-dir", str(args.output_dir),
        "--operating-threshold", str(args.operating_threshold),
    ]
    if args.device:
        forwarded += ["--device", args.device]
    if args.model_name:
        forwarded += ["--model-name", args.model_name]
    if args.resolution:
        forwarded += ["--resolution", str(args.resolution)]
    try:
        return_code = evaluate.main(forwarded)
    except Exception as exc:
        _append(args.ledger, {**common, "event": "failed",
                             "at": datetime.now(timezone.utc).isoformat(),
                             "error": str(exc)[:2000]})
        raise
    _append(args.ledger, {**common, "event": "completed",
                         "at": datetime.now(timezone.utc).isoformat(),
                         "return_code": return_code})
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
