"""Paired film-cluster bootstrap of saved detection measurements; no inference.

Run from any directory. Results are conditional on the measured training seed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 20261021
REPLICATES = 2000
FINAL_STEP = 218750
BASELINE_STEPS = (55000, 44000)
RESULTS = Path(__file__).resolve().parents[1]
SOURCE = RESULTS / "Dane/detekcja_0_200/DETEKCJA_OBRAZY_0_200.csv"
OUT = RESULTS / "Dane/audyt_wnioskow_20261021"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> dict[tuple[str, int], dict[str, dict]]:
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in ("VAL", "TEST"):
                continue
            row["step"] = int(row["step"])
            row["iou"] = float(row["iou"]) if row["iou"].strip() else float("nan")
            row["score"] = float(row["score"])
            assert np.isfinite(row["score"]), row
            assert 0 <= row["score"] <= 1, row
            if np.isfinite(row["iou"]):
                assert 0 <= row["iou"] <= 1, row
            group = groups.setdefault((row["split"], row["step"]), {})
            assert row["id"] not in group, (row["split"], row["step"], row["id"])
            group[row["id"]] = row
    return groups


def summarize_difference(baseline, final, mask, film_index, draws):
    """Resample whole films; the same image contributes both paired endpoints."""
    count = np.bincount(film_index[mask], minlength=draws.shape[1])
    deltas = final - baseline
    sums = np.bincount(film_index[mask], weights=deltas[mask], minlength=draws.shape[1])
    denominator = count[draws].sum(axis=1)
    numerator = sums[draws].sum(axis=1)
    valid = denominator > 0
    bootstrap = numerator[valid] / denominator[valid]
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "images": int(mask.sum()),
        "films_with_eligible_images": int((count > 0).sum()),
        "baseline_mean": float(baseline[mask].mean()),
        "final_mean": float(final[mask].mean()),
        "mean_paired_difference_final_minus_baseline": float(deltas[mask].mean()),
        "percentile_95_ci": [float(low), float(high)],
        "bootstrap_valid_replicates": int(valid.sum()),
        "bootstrap_replicates_without_eligible_images": int((~valid).sum()),
        "ci_contains_zero": bool(low <= 0 <= high),
    }


def main():
    groups = load_rows()
    rng = np.random.default_rng(SEED)
    result = {
        "input": str(SOURCE),
        "input_sha256": sha256(SOURCE),
        "script_sha256": sha256(Path(__file__).resolve()),
        "method": {
            "seed": SEED,
            "replicates": REPLICATES,
            "interval": "two-sided 95% percentile, pointwise; no multiplicity correction",
            "unit": "whole film, sampled with replacement separately within VAL and TEST",
            "estimand": "mean image-level paired difference, final minus baseline; eligible frames weighted equally",
            "pairing": "exact split/image ID match and unchanged film and positive/empty status",
            "joint_draws": "same film draws used for both baseline comparisons and all metrics within a split",
            "positive_definition": "finite IoU in source CSV (including IoU=0); empty images have absent IoU",
            "empty_score": "maximum-token sigmoid score on annotated empty frames; not a calibrated false-positive rate",
            "scope": "one training seed3047; uncertainty over sampled films only; descriptive post-hoc contrasts",
        },
        "final_step": FINAL_STEP,
        "final_epoch": FINAL_STEP * 32 / 35000,
        "pairing_checks": {},
        "comparisons": [],
    }
    for split in ("VAL", "TEST"):
        endpoint = groups[(split, FINAL_STEP)]
        ids = sorted(endpoint)
        films = sorted({endpoint[key]["film"] for key in ids})
        film_lookup = {name: i for i, name in enumerate(films)}
        film_index = np.array([film_lookup[endpoint[key]["film"]] for key in ids])
        final_iou = np.array([endpoint[key]["iou"] for key in ids])
        final_score = np.array([endpoint[key]["score"] for key in ids])
        positive = np.isfinite(final_iou)
        steps = sorted(step for panel, step in groups if panel == split)
        assert len(steps) == 26, (split, len(steps))
        for step in steps:
            current = groups[(split, step)]
            assert set(current) == set(endpoint), (split, step, "image IDs differ")
            for key in ids:
                assert current[key]["film"] == endpoint[key]["film"], (split, step, key, "film differs")
                assert np.isfinite(current[key]["iou"]) == np.isfinite(endpoint[key]["iou"]), (
                    split, step, key, "positive/empty status differs")
        assert len(ids) == 500 and len(films) == 25, (split, len(ids), len(films))
        result["pairing_checks"][split] = {
            "all_26_checkpoints_same_image_ids": True,
            "film_and_positive_empty_status_match": True,
            "images": len(ids),
            "films": len(films),
            "positive_images": int(positive.sum()),
            "empty_images": int((~positive).sum()),
        }
        draws = rng.integers(0, len(films), size=(REPLICATES, len(films)))
        for step in BASELINE_STEPS:
            baseline = groups[(split, step)]
            baseline_iou = np.array([baseline[key]["iou"] for key in ids])
            baseline_score = np.array([baseline[key]["score"] for key in ids])
            result["comparisons"].append({
                "split": split,
                "baseline_step": step,
                "baseline_epoch": step * 32 / 35000,
                "positive_mean_top1_iou": summarize_difference(
                    baseline_iou, final_iou, positive, film_index, draws),
                "positive_mean_top1_score": summarize_difference(
                    baseline_score, final_score, positive, film_index, draws),
                "empty_mean_top1_score": summarize_difference(
                    baseline_score, final_score, ~positive, film_index, draws),
            })

    validation = []
    for (split, step), records in groups.items():
        if split == "VAL":
            vals = np.array([row["iou"] for row in records.values()])
            validation.append((float(vals[np.isfinite(vals)].mean()), step))
    val_mean, val_step = max(validation)
    result["highest_measured_VAL_mean_iou_among_26"] = {
        "step": val_step,
        "epoch": val_step * 32 / 35000,
        "mean_top1_iou_positive": val_mean,
        "note": "descriptive maximum on this VAL panel; no TEST checkpoint selection and no claim of statistical superiority",
        "all_VAL_means": [
            {"step": step, "epoch": step * 32 / 35000, "mean_top1_iou_positive": mean}
            for mean, step in sorted(validation, key=lambda item: item[1])
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "PAIRED_FILM_BOOTSTRAP.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
