#!/usr/bin/env python3
"""CPU benchmark checking whether the diagnostic distinguishes known causes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .gle import diagnose_gle


def make_case(kind: str, *, seed: int, groups: int = 6,
              frames: int = 180, dim: int = 6) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, chunks = [], []
    for group in range(groups):
        z = np.zeros((frames, dim), dtype=np.float64)
        z[0] = rng.normal(scale=.3, size=dim)
        z[1] = rng.normal(scale=.3, size=dim)
        context = rng.normal(scale=.4, size=dim)
        forcing = rng.normal(scale=.8, size=dim)
        for t in range(2, frames):
            eps = rng.normal(scale=.06, size=dim)
            if kind == "markov":
                z[t] = .78 * z[t - 1] + .08 * np.tanh(z[t - 1]) + eps
            elif kind == "hidden_memory":
                z[t] = .78 * z[t - 1] - .48 * z[t - 2] + eps
            elif kind == "nonlinear":
                z[t] = .45 * z[t - 1] + .38 * np.sin(2.5 * z[t - 1]) + eps
            elif kind == "exogenous":
                pulse = forcing if frames // 3 <= t < frames // 3 + 8 else 0.0
                z[t] = .72 * z[t - 1] + pulse + eps
            elif kind == "noise":
                z[t] = rng.normal(scale=.35, size=dim)
            else:
                raise ValueError(f"unknown synthetic case {kind}")
        z += context
        chunks.append(z)
        for t in range(frames):
            rows.append({"frame_id": f"g{group}/f{t}", "split": "train",
                         "group_id": f"g{group}", "video_id": f"g{group}",
                         "frame_index": t})
    return rows, np.vstack(chunks)


def run_benchmark(seed: int = 42) -> dict:
    report = {}
    for kind in ("markov", "hidden_memory", "nonlinear", "exogenous", "noise"):
        rows, z = make_case(kind, seed=seed)
        scores, summary, _ = diagnose_gle(
            rows, z, seed=seed, dim=z.shape[1], lags=3, folds=3,
            max_gap_frames=1, max_fit_windows=5000, residual_modes=4)
        valid = [row for row in scores if row["has_valid_window"]]
        report[kind] = {
            "mean_history_gain": summary["mean_history_gain"],
            "positive_history_gain_fraction": summary["positive_history_gain_fraction"],
            "mean_residual_mahalanobis": float(np.mean(
                [row["residual_mahalanobis"] for row in valid])),
            "valid_windows": len(valid),
        }
    report["checks"] = {
        "hidden_memory_gain_exceeds_markov": (
            report["hidden_memory"]["mean_history_gain"]
            > report["markov"]["mean_history_gain"]),
        "benchmark_is_diagnostic_not_physical_proof": True,
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    report = run_benchmark(args.seed)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["checks"]["hidden_memory_gain_exceeds_markov"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
