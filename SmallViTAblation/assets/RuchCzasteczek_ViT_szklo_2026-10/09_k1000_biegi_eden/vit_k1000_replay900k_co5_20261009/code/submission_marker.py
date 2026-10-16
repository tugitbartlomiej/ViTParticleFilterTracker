"""Atomowy dziennik zgłoszeń; obecność częściowego markera blokuje duplikat."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
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
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--action", choices=("begin", "add", "finish"), required=True)
    parser.add_argument("--kind")
    parser.add_argument("--key")
    parser.add_argument("--job-id")
    args = parser.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    if args.action == "begin":
        if args.path.exists():
            raise RuntimeError(f"Marker już istnieje: {args.path}")
        value = {"kind": args.kind, "status": "submitting", "jobs": {}, "started_utc": now}
    else:
        value = json.loads(args.path.read_text(encoding="utf-8"))
        if args.action == "add":
            if args.key in value["jobs"]:
                raise RuntimeError(f"Powtórzony job {args.key}")
            value["jobs"][args.key] = args.job_id
        else:
            value["status"] = "submitted"
            value["finished_submission_utc"] = now
    value["updated_utc"] = now
    atomic(args.path, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
