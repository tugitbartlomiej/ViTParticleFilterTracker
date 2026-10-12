"""Atomowy dziennik zgłaszania jobów; także częściowy stan blokuje duplikat."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--action", choices=("begin", "add", "finish"), required=True)
    parser.add_argument("--kind")
    parser.add_argument("--key")
    parser.add_argument("--job-id")
    parser.add_argument("--final-status", default="submitted")
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    if args.action == "begin":
        if args.path.exists():
            raise RuntimeError(f"Marker już istnieje: {args.path}")
        if not args.kind:
            raise ValueError("--kind jest wymagane dla begin")
        value = {
            "status": "submitting",
            "kind": args.kind,
            "jobs": {},
            "started_utc": now,
            "updated_utc": now,
        }
    else:
        if not args.path.is_file():
            raise RuntimeError(f"Brak markera: {args.path}")
        value = json.loads(args.path.read_text(encoding="utf-8"))
        if args.action == "add":
            if not args.key or not args.job_id:
                raise ValueError("--key i --job-id są wymagane dla add")
            if args.key in value.get("jobs", {}):
                raise RuntimeError(f"Job {args.key} jest już zapisany")
            value.setdefault("jobs", {})[args.key] = args.job_id
        else:
            value["status"] = args.final_status
            value["finished_submission_utc"] = now
        value["updated_utc"] = now
    _atomic(args.path, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
