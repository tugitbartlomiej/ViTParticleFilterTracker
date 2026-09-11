#!/usr/bin/env python3
"""Paired cluster bootstrap of COCO metrics over independent operations.

Every bootstrap draw is shared by all compared methods. Repeated operations
receive fresh image and annotation identifiers so COCOeval preserves their
multiplicity. This estimates test-cohort uncertainty for fixed checkpoints;
variation across training seeds must be reported separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .common import compute_map_metrics, load_gt_dataset, load_predictions


METRICS = ("mAP_0.5:0.95", "mAP_0.5", "mAP_0.75", "AR@100")


def _method_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected METHOD=predictions.json")
    return name, Path(raw_path)


def _group_lookup(gt: dict, mapping: dict | None) -> dict[int, str]:
    result = {}
    for image in gt["images"]:
        image_id = int(image["id"])
        group = image.get("group_id") or image.get("operation_id")
        if group is None and mapping is not None:
            for key in (str(image_id), str(image.get("frame_id", "")),
                        str(image.get("file_name", ""))):
                if key and key in mapping:
                    group = mapping[key]
                    break
        if group is None:
            raise ValueError(
                f"image_id={image_id} has no group_id/operation_id; supply --group-map")
        result[image_id] = str(group)
    if len(set(result.values())) < 2:
        raise ValueError("cluster bootstrap requires at least two operation groups")
    return result


def _resample(gt: dict, predictions: dict[str, list[dict]], groups: dict[int, str],
              draw: list[str]) -> tuple[dict, dict[str, list[dict]]]:
    images_by_group: dict[str, list[dict]] = defaultdict(list)
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    predictions_by_method_image: dict[str, dict[int, list[dict]]] = {}
    for image in gt["images"]:
        images_by_group[groups[int(image["id"])]].append(image)
    for annotation in gt["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    for method, rows in predictions.items():
        by_image: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_image[int(row["image_id"])].append(row)
        predictions_by_method_image[method] = by_image

    sampled_images, sampled_annotations = [], []
    sampled_predictions = {method: [] for method in predictions}
    next_image_id = next_annotation_id = 1
    for draw_index, group in enumerate(draw):
        for source_image in images_by_group[group]:
            source_image_id = int(source_image["id"])
            new_image_id = next_image_id
            next_image_id += 1
            sampled_images.append({**source_image, "id": new_image_id,
                                   "bootstrap_draw": draw_index,
                                   "bootstrap_source_image_id": source_image_id})
            for source_annotation in annotations_by_image[source_image_id]:
                sampled_annotations.append({**source_annotation,
                                            "id": next_annotation_id,
                                            "image_id": new_image_id})
                next_annotation_id += 1
            for method, by_image in predictions_by_method_image.items():
                sampled_predictions[method].extend(
                    {**source_prediction, "image_id": new_image_id}
                    for source_prediction in by_image.get(source_image_id, []))
    sampled_gt = {**gt, "images": sampled_images, "annotations": sampled_annotations}
    return sampled_gt, sampled_predictions


def _summary(values: list[float], ci_level: float) -> dict:
    alpha = (1.0 - ci_level) / 2.0
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else None,
        "ci_low": float(np.quantile(array, alpha)),
        "ci_high": float(np.quantile(array, 1.0 - alpha)),
        "raw": [float(value) for value in values],
    }


def paired_group_bootstrap(gt: dict, predictions: dict[str, list[dict]], *,
                           baseline: str, iterations: int, seed: int,
                           ci_level: float = .95,
                           group_map: dict | None = None) -> dict:
    if baseline not in predictions:
        raise ValueError(f"baseline {baseline!r} is absent from predictions")
    if iterations < 1 or not 0 < ci_level < 1:
        raise ValueError("iterations must be positive and ci_level must lie in (0, 1)")
    groups = _group_lookup(gt, group_map)
    known_images = set(groups)
    for method, rows in predictions.items():
        unknown = {int(row["image_id"]) for row in rows} - known_images
        if unknown:
            raise ValueError(f"{method}: predictions contain unknown image ids {sorted(unknown)[:3]}")
    names = sorted(set(groups.values()))
    rng = np.random.default_rng(seed)
    values = {method: {metric: [] for metric in METRICS} for method in predictions}
    differences = {method: {metric: [] for metric in METRICS}
                   for method in predictions if method != baseline}
    for _ in range(iterations):
        draw = rng.choice(names, size=len(names), replace=True).tolist()
        sampled_gt, sampled_predictions = _resample(gt, predictions, groups, draw)
        measured = {method: compute_map_metrics(sampled_gt, rows)
                    for method, rows in sampled_predictions.items()}
        for method in predictions:
            for metric in METRICS:
                values[method][metric].append(float(measured[method][metric]))
        for method in differences:
            for metric in METRICS:
                differences[method][metric].append(
                    float(measured[method][metric] - measured[baseline][metric]))
    point = {method: compute_map_metrics(gt, rows) for method, rows in predictions.items()}
    return {
        "schema_version": 1,
        "unit": "operation_group",
        "paired_draws": True,
        "fixed_checkpoint_test_cohort_uncertainty": True,
        "training_seed_variability_included": False,
        "groups": len(names),
        "iterations": iterations,
        "seed": seed,
        "ci_level": ci_level,
        "baseline": baseline,
        "point_estimates": point,
        "bootstrap": {method: {metric: _summary(raw, ci_level)
                                for metric, raw in metric_values.items()}
                      for method, metric_values in values.items()},
        "paired_differences": {method: {metric: _summary(raw, ci_level)
                                         for metric, raw in metric_values.items()}
                               for method, metric_values in differences.items()},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--predictions", type=_method_path, nargs="+", required=True,
                    metavar="METHOD=FILE")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--group-map", type=Path,
                    help="optional JSON map keyed by image_id, frame_id or file_name")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--ci-level", type=float, default=.95)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    prediction_paths = dict(args.predictions)
    if len(prediction_paths) != len(args.predictions):
        raise ValueError("prediction method names must be unique")
    report = paired_group_bootstrap(
        load_gt_dataset(args.annotations),
        {name: load_predictions(path) for name, path in prediction_paths.items()},
        baseline=args.baseline, iterations=args.iterations, seed=args.seed,
        ci_level=args.ci_level,
        group_map=(json.loads(args.group_map.read_text(encoding="utf-8"))
                   if args.group_map else None),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({"groups": report["groups"], "iterations": report["iterations"],
                      "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
