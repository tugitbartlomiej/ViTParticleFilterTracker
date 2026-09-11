#!/usr/bin/env python3
"""Aggregate paired-seed detection results without frame-level pseudoreplication."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def _metric(block: dict, name: str) -> float:
    aliases = {
        "map_50_95": ("mAP_0.5:0.95", "map_50_95"),
        "map_50": ("mAP_0.5", "map_50"),
        "map_75": ("mAP_0.75", "map_75"),
        "ar_100": ("AR@100", "ar_100"),
        "f1": ("f1",),
    }
    for key in aliases.get(name, (name,)):
        if key in block:
            return float(block[key])
    raise KeyError(f"metric {name!r} absent from result block")


def load_run(run_dir: Path, split: str, metric: str) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    experiment = manifest.get("experiment", {})
    block = metrics.get("results", {}).get(split)
    if block is None:
        raise ValueError(f"{run_dir}: no {split!r} metrics block")
    return {
        "run_id": manifest["run_id"],
        "arch": manifest.get("arch", manifest.get("model", "unknown")),
        "strategy": experiment.get("strategy", "unknown"),
        "budget": experiment.get("budget"),
        "selection_seed": experiment.get("selection_seed"),
        "train_seed": int(experiment.get("train_seed", manifest["seed"])),
        "value": _metric(block, metric),
    }


def aggregate(records: list[dict], baseline: str) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (record["arch"], record["strategy"], record["budget"])
        groups[key].append(record)
    rows = []
    by_pair = {(r["arch"], r["budget"], r["train_seed"], r["selection_seed"], r["strategy"]): r
               for r in records}
    for (arch, strategy, budget), values in sorted(groups.items(), key=lambda x: str(x[0])):
        raw = [r["value"] for r in values]
        deltas = []
        for record in values:
            base = by_pair.get((arch, budget, record["train_seed"],
                                record["selection_seed"], baseline))
            if base is not None:
                deltas.append(record["value"] - base["value"])
        rows.append({
            "arch": arch, "strategy": strategy, "budget": budget,
            "n": len(raw), "mean": statistics.fmean(raw),
            "std": statistics.stdev(raw) if len(raw) > 1 else None,
            "raw": raw,
            "paired_baseline": baseline,
            "paired_n": len(deltas),
            "paired_delta_mean": statistics.fmean(deltas) if deltas else None,
            "paired_delta_std": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "paired_deltas": deltas,
        })

    auc = []
    curves: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if isinstance(row["budget"], (int, float)):
            curves[(row["arch"], row["strategy"])].append(row)
    for (arch, strategy), points in curves.items():
        points.sort(key=lambda row: row["budget"])
        area = sum((b["budget"] - a["budget"]) * (a["mean"] + b["mean"]) / 2
                   for a, b in zip(points, points[1:]))
        width = points[-1]["budget"] - points[0]["budget"]
        auc.append({"arch": arch, "strategy": strategy,
                    "normalized_budget_auc": area / width if width > 0 else None,
                    "budgets": [point["budget"] for point in points]})
    return {
        "groups": rows,
        "budget_curves": auc,
        "inference_note": (
            "Seed summaries are descriptive. Confidence intervals for detector quality "
            "must be obtained by resampling independent operation groups from predictions, "
            "never individual video frames."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--metric", default="map_50_95")
    ap.add_argument("--baseline", default="uniform_time")
    args = ap.parse_args(argv)
    run_dirs = sorted({path.parent for path in args.results.rglob("metrics.json")})
    records, errors = [], []
    for run_dir in run_dirs:
        try:
            records.append(load_run(run_dir, args.split, args.metric))
        except Exception as exc:
            errors.append({"run_dir": str(run_dir), "error": str(exc)})
    if not records:
        raise ValueError(f"no complete runs under {args.results}; errors={errors[:3]}")
    report = aggregate(records, args.baseline)
    report.update({"split": args.split, "metric": args.metric,
                   "runs_loaded": len(records), "skipped": errors})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aggregate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["arch", "strategy", "budget", "n", "mean", "std", "paired_n",
                  "paired_delta_mean", "paired_delta_std"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["groups"]:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({"runs": len(records), "groups": len(report["groups"]),
                      "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
