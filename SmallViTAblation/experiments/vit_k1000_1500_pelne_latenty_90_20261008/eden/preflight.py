"""Read-only checks of protected inputs plus a small, atomic preflight report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

EXPECTED = {
    "checkpoint": "3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e",
    "checkpoint_adamw": "e7a7727ccf19dd3801bd8ec4de65887480252ccdf7e21d385346c13a139259b6",
    "coco": "3667dd12c8fddcb695c9ce6144ca4bf3f5300f70a47322a8da12b92de6d98f75",
    "archive": "5850b5c8dd4411159061e254e1f9b32e2b4b633036d3990049b553bea767d8b1",
    "manifest": "33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7",
}
RAW_LARGE_TENSORS = 14_610_825_216
SMALL_AND_SMOKE_ALLOWANCE = 800_000_000
OPERATING_RESERVE = 5_000_000_000
MINIMUM_FREE_FOR_PRODUCTION = RAW_LARGE_TENSORS + SMALL_AND_SMOKE_ALLOWANCE + OPERATING_RESERVE
EXPECTED_ARCHIVE_BYTES = 2_108_682_240
EXPECTED_ADAMW_BYTES = 35_212_799
RUNTIME_PY = Path("/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python")
TEST_PY = Path("/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python")


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


def python_probe(executable: Path, source: str, *arguments: str) -> dict:
    if not executable.is_file():
        return {
            "ok": False,
            "executable": str(executable),
            "returncode": None,
            "stdout": "",
            "stderr": "missing executable",
        }
    completed = subprocess.run(
        [str(executable), "-c", source, *arguments],
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        timeout=180,
        check=False,
    )
    return {
        "ok": completed.returncode == 0,
        "executable": str(executable),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "production"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-adamw", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-existing-submission", action="store_true")
    args = parser.parse_args()

    paths = {
        "checkpoint": args.checkpoint,
        "checkpoint_adamw": args.checkpoint_adamw,
        "coco": args.coco,
        "archive": args.archive,
        "manifest": args.manifest,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Chronione wejścia nie istnieją: {missing}")
    hashes = {
        "checkpoint": sha256(args.checkpoint),
        "checkpoint_adamw": sha256(args.checkpoint_adamw),
        "coco": sha256(args.coco),
        "archive": sha256(args.archive),
        "manifest": sha256(args.manifest),
    }
    if hashes != EXPECTED:
        raise RuntimeError(f"Niezgodne hashe wejść: {hashes}")
    manifest_count = sum(1 for line in args.manifest.open("rb") if line.strip())
    if manifest_count != 1500:
        raise RuntimeError(f"Manifest ma {manifest_count} wierszy")

    disk = shutil.disk_usage(args.root.parent)
    required = MINIMUM_FREE_FOR_PRODUCTION if args.phase == "production" else 1_000_000_000
    space_ok = disk.free >= required
    try:
        queue = subprocess.check_output(
            ["squeue", "-h", "-u", os.environ.get("USER", "bpiotrowski"), "-o", "%i|%j|%T"],
            text=True,
        ).splitlines()
        queue_ok = True
    except Exception as error:
        queue = [f"SQUEUE_ERROR:{error}"]
        queue_ok = False
    own_active = [line for line in queue if "k1k1500" in line]
    marker = args.root / "submission" / f"{args.phase.upper()}_SUBMITTED.json"
    runtime_probe = python_probe(
        RUNTIME_PY,
        "import json,sys; from pathlib import Path; import cv2,numpy,torch; "
        "from uruchom_oracle import load_model; "
        "model,digest=load_model(Path(sys.argv[1]),torch.device('cpu')); "
        "print(json.dumps({'weights_buffers_fingerprint':digest,'parameters':sum(p.numel() for p in model.parameters()),"
        "'torch':torch.__version__,'numpy':numpy.__version__,'cv2':cv2.__version__},sort_keys=True))",
        str(args.checkpoint),
    )
    test_probe = python_probe(
        TEST_PY,
        "import json,cv2,numpy,pytest,torch; import model_snapshot; "
        "print(json.dumps({'pytest':pytest.__version__,'torch':torch.__version__,"
        "'numpy':numpy.__version__,'cv2':cv2.__version__},sort_keys=True))",
    )
    checks = {
        "exact_input_hashes": True,
        "exact_manifest_count": True,
        "protected_full_adamw_state_exists": args.checkpoint_adamw.is_file(),
        "protected_full_adamw_state_size": args.checkpoint_adamw.stat().st_size == EXPECTED_ADAMW_BYTES,
        "source_archive_exists_without_extraction": args.archive.is_file(),
        "source_archive_exact_size": args.archive.stat().st_size == EXPECTED_ARCHIVE_BYTES,
        "space_threshold_met": space_ok,
        "no_existing_submission_marker": (not marker.exists()) or args.allow_existing_submission,
        "scheduler_query_succeeded": queue_ok,
        "no_active_pipeline_job": not own_active,
        "runtime_imports_and_strict_checkpoint_load": runtime_probe["ok"],
        "test_environment_imports": test_probe["ok"],
    }
    report = {
        "status": "pass" if all(checks.values()) else "blocked",
        "phase": args.phase,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": hashes,
        "archive_bytes": args.archive.stat().st_size,
        "checkpoint_adamw_bytes": args.checkpoint_adamw.stat().st_size,
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "budget": {
            "raw_large_tensors": RAW_LARGE_TENSORS,
            "small_artifacts_and_smoke_allowance": SMALL_AND_SMOKE_ALLOWANCE,
            "operating_reserve": OPERATING_RESERVE,
            "minimum_free_for_production": MINIMUM_FREE_FOR_PRODUCTION,
            "compression_credit_assumed": 0,
        },
        "active_pipeline_jobs": own_active,
        "runtime_probe": runtime_probe,
        "test_environment_probe": test_probe,
        "allow_existing_submission_for_resume": args.allow_existing_submission,
        "protected_dependencies": [
            str(args.checkpoint),
            str(args.checkpoint_adamw),
            str(args.coco),
            str(args.archive),
        ],
    }
    atomic_json(args.root / "preflight" / f"PREFLIGHT_{args.phase.upper()}.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
