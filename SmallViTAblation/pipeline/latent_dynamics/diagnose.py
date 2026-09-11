#!/usr/bin/env python3
"""Run cross-fitted GLE-inspired diagnostics on a frozen training latent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .gle import diagnose_gle


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _features(path: Path, rows: list[dict]) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    ids = [str(x) for x in data["frame_ids"].tolist()]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate frame_ids in feature cache")
    index = {frame_id: i for i, frame_id in enumerate(ids)}
    missing = [row["frame_id"] for row in rows if row["frame_id"] not in index]
    if missing:
        raise ValueError(f"feature cache misses {len(missing)} frames: {missing[:3]}")
    return np.asarray(data["features"])[[index[row["frame_id"]] for row in rows]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=24)
    ap.add_argument("--lags", type=int, default=3)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--max-gap-frames", type=float, default=8)
    ap.add_argument("--alpha", type=float, default=.01)
    ap.add_argument("--max-fit-windows", type=int, default=20000)
    ap.add_argument("--covariance-shrinkage", type=float, default=.10)
    ap.add_argument("--residual-modes", type=int, default=8)
    args = ap.parse_args(argv)
    rows = [row for row in _rows(args.pool) if row.get("split", "train") == "train"]
    scores, summary, arrays = diagnose_gle(
        rows, _features(args.features, rows), seed=args.seed, dim=args.dim,
        lags=args.lags, folds=args.folds, max_gap_frames=args.max_gap_frames,
        alpha=args.alpha, max_fit_windows=args.max_fit_windows,
        covariance_shrinkage=args.covariance_shrinkage,
        residual_modes=args.residual_modes)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "frame_scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in scores:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(args.out / "residual_modes.npz", **arrays)
    print(json.dumps({"valid_windows": summary["valid_windows"],
                      "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
