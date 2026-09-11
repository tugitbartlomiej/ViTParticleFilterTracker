#!/usr/bin/env python3
"""Derive a transparent detection uncertainty proxy from COCO predictions.

This is least-confidence plus optional between-checkpoint score disagreement.
It is not BALD and it must not be described as a calibrated epistemic measure.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _predictions(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["annotations"] if isinstance(raw, dict) and "annotations" in raw else raw


def score_predictions(coco: dict, prediction_sets: list[list[dict]]) -> list[dict]:
    if not prediction_sets:
        raise ValueError("at least one prediction file is required")
    by_set = []
    for predictions in prediction_sets:
        grouped = defaultdict(list)
        for pred in predictions:
            score = float(pred["score"])
            if not 0 <= score <= 1 or not np.isfinite(score):
                raise ValueError("prediction scores must be finite probabilities")
            grouped[int(pred["image_id"])].append(score)
        by_set.append(grouped)
    out = []
    for image in coco["images"]:
        image_id = int(image["id"])
        top = np.asarray([max(group.get(image_id, [0.0])) for group in by_set])
        least_confidence = 1.0 - float(top.mean())
        disagreement = float(top.std(ddof=0)) if len(top) > 1 else 0.0
        out.append({
            "frame_id": str(image.get("frame_id", image_id)),
            "image_id": image_id,
            "uncertainty": least_confidence + disagreement,
            "least_confidence": least_confidence,
            "checkpoint_score_std": disagreement,
            "n_prediction_sets": len(by_set),
            "interpretation": "least_confidence_plus_score_disagreement_not_BALD",
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    rows = score_predictions(coco, [_predictions(path) for path in args.predictions])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"frames": len(rows), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
