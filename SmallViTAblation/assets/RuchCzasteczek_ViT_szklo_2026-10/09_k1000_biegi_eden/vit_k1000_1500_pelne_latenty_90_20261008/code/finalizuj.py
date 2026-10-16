"""Scala status zależnych etapów bez przeliczania wyników naukowych."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    required = {
        "smoke": root / "results/smoke/SMOKE_VALIDATION.json",
        "oracle": root / "results/merged/ORACLE_SUMMARY.json",
        "movement": root / "results/movement/MOVEMENT_STATS.json",
        "regimes": root / "results/regimes/REGIME_DIAGNOSTICS.json",
        "dmdc": root / "results/dmdc/DMDC_REPORT.json",
        "evaluation": root / "results/evaluation/EVALUATION_SUMMARY.json",
        "fresh_load": root / "results/evaluation/FRESH_LOAD_VERIFICATION.json",
        "freeze": root / "results/evaluation/MODEL_FREEZE_VERIFICATION.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Brak wyników: {missing}")
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in required.items()}
    if payloads["smoke"]["status"] != "pass":
        raise RuntimeError("Smoke nie przeszedł")
    if payloads["oracle"]["status"] != "complete":
        raise RuntimeError("Oracle niekompletny")
    if payloads["evaluation"]["status"] != "complete":
        raise RuntimeError("Ewaluacja niekompletna")
    final = {
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in required.items()
        },
        "oracle": payloads["oracle"],
        "evaluation": payloads["evaluation"],
        "methodological_limits": [
            "per-image oracle and all-data A/B fit use the same exact 1500 images",
            "in-sample fit is not evidence of generalization",
            "video-group holdout is internal and does not replace an external test set",
            "DMD/DMDc is descriptive and is not used to assign the transfer correction",
        ],
    }
    output = root / "results/FINAL_STATUS.json"
    output.write_text(
        json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
