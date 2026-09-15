#!/usr/bin/env python3
"""Freeze the CATARACT-LMM static-selection pilot used by the Eden campaign.

The selector sees training frames only. Validation is copied unchanged from the
audited split, and test data are never materialized. Selection randomness is
frozen independently of the three RF-DETR training seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from SmallViTAblation.pipeline.curation.data import ImageSource, write_json
from SmallViTAblation.pipeline.curation.features import build_features
from SmallViTAblation.pipeline.curation.selection import choose
from SmallViTAblation.tools.prepare_tracking_calibration import materialize, temporal_sample


STRATEGIES = {
    "random": "random_global_without_replacement",
    "uniform_time": "sqrt_group_quota_midpoint_time_bins",
    "dino_coverage": "dinov2_group_kcenter_one_temporal_initialiser_per_group",
    "dino_temporal_hybrid": "dinov2_group_kcenter_with_25pct_temporal_anchors",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--dino-hub-repo", required=True,
                    help="Pinned torch.hub repository, e.g. facebookresearch/dinov2:<commit>")
    ap.add_argument("--selection-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if args.out.exists() or args.out.with_suffix(".tar").exists():
        raise FileExistsError("Use a new campaign path; prepared data are immutable")
    if ":main" in args.dino_hub_repo or args.dino_hub_repo.endswith(":master"):
        raise ValueError("DINO source must be pinned to an immutable commit")

    started = time.monotonic()
    manifest_path = args.index_dir / "pool_manifest.jsonl"
    missing_path = args.index_dir / "missing_images_audit.json"
    rows = read_jsonl(manifest_path)
    missing_doc = json.loads(missing_path.read_text(encoding="utf-8"))
    missing_ids = {row["frame_id"] for row in missing_doc["missing"]}
    eligible = [row for row in rows
                if row["label_status"] == "provided" and row["frame_id"] not in missing_ids]
    train = sorted((row for row in eligible if row["split"] == "train"),
                   key=lambda row: row["frame_id"])
    val = sorted((row for row in eligible if row["split"] == "val"),
                 key=lambda row: row["frame_id"])
    if len(train) < 2500 or not val:
        raise ValueError("Audited train/validation pools are too small")
    if {r["group_id"] for r in train} & {r["group_id"] for r in val}:
        raise ValueError("Train/validation group leakage")

    source = ImageSource()
    try:
        features, quality, pixel_hashes, feature_meta = build_features(
            train, source, args.cache_root, semantic=True,
            hub_repo=args.dino_hub_repo, batch_size=args.batch_size,
            model="dinov2_vits14", image_size=224, device="cuda", amp=True)

        # All methods operate on the same exact-pixel-deduplicated training pool.
        seen = set()
        keep = []
        for index, (row, pixel_hash) in enumerate(zip(train, pixel_hashes)):
            key = (row["native_width"], row["native_height"], pixel_hash)
            if key not in seen:
                seen.add(key)
                keep.append(index)
        train = [train[i] for i in keep]
        features = features[keep]
        quality = quality[keep]
        pixel_hashes = [pixel_hashes[i] for i in keep]

        budgets = [500, 1500, 2500]
        rng = np.random.default_rng(args.selection_seed)
        random_order = rng.permutation(len(train)).tolist()
        selections: dict[tuple[str, int], list[dict]] = {}
        selector_meta = {}
        for budget in budgets:
            selections[("random", budget)] = [train[i] for i in random_order[:budget]]
            selections[("uniform_time", budget)] = temporal_sample(train, budget)
            for strategy, anchor_fraction in [("dino_coverage", 0.0),
                                              ("dino_temporal_hybrid", 0.25)]:
                selected, _, _, _, meta = choose(
                    train, features, quality, pixel_hashes, profile="balanced",
                    budget=budget, seed=args.selection_seed, allocation="sqrt",
                    anchor_fraction=anchor_fraction, min_gap_frames=5,
                    spacing_policy="relax", projection_dim=64)
                selections[(strategy, budget)] = [train[i] for i in selected]
                selector_meta[f"{strategy}_b{budget}"] = meta

        args.out.mkdir(parents=True)
        (args.out / "variants").mkdir()
        selected_union = {row["frame_id"]: row for pool in selections.values() for row in pool}
        export_train = [selected_union[key] for key in sorted(selected_union)]
        exported = {}
        pixel_splits = {}
        for split, pool in [("train", export_train), ("val", val)]:
            final_rows, images, annotations = [], [], []
            for row, images, annotations in materialize(pool, args.out, source, pixel_splits):
                final_rows.append(row)
            exported[split] = final_rows
            if split == "val":
                write_json(args.out / "annotations_val.json", {
                    "images": images, "annotations": annotations,
                    "categories": [{"id": 1, "name": "instrument", "supercategory": "tool"}],
                })
    finally:
        source.close()

    exported_by_id = {row["frame_id"]: row for row in exported["train"]}
    with (args.out / "pool_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in exported["train"] + exported["val"]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_seeds = [42, 1337, 2024]
    for (strategy, budget), pool in sorted(selections.items()):
        ids = [row["frame_id"] for row in pool]
        if len(ids) != budget or len(set(ids)) != budget or not set(ids) <= exported_by_id.keys():
            raise RuntimeError(f"Invalid frozen selection: {strategy}, {budget}")
        for train_seed in train_seeds:
            run_id = f"tracking_{strategy}_b{budget}_s{train_seed}"
            write_json(args.out / "variants" / f"{run_id}.json", {
                "run_id": run_id, "strategy": strategy, "strategy_definition": STRATEGIES[strategy],
                "budget": budget, "seed": args.selection_seed,
                "selection_seed": args.selection_seed, "train_seed": train_seed,
                "seed_roles_separated": True, "frame_ids": ids,
                "scope": "static_curation_pilot", "candidate_grid_stride": 5,
                "test_used_for_selection": False,
            })

    write_json(args.out / "split_manifest.json",
               json.loads((args.index_dir / "split_manifest.json").read_text(encoding="utf-8")))
    write_json(args.out / "missing_images_audit.json", {
        "source_sha256": sha256(missing_path), "excluded_before_selection": True,
        "count": len(missing_ids), "split_counts": missing_doc.get("split_counts"),
    })
    write_json(args.out / "selector_manifest.json", {
        "strategies": STRATEGIES, "selection_seed": args.selection_seed,
        "training_seeds": train_seeds, "budgets": budgets,
        "dino_hub_repo": args.dino_hub_repo, "dino_feature_cache": feature_meta,
        "selector_details": selector_meta,
        "static_candidate_only": True,
        "koopman_langevin_or_mori_zwanzig_used": False,
        "labels_used_for_ranking": False,
    })
    summary = {
        "scope": "cataract_lmm_tracking_static_curation_pilot",
        "source_target": "cystotome_instrument_box",
        "split_unit": "conservative_operation_groups",
        "patient_disjointness_verified": False,
        "annotation_policy": "provided_bbox_only",
        "empty_labels_not_interpreted_as_negative": True,
        "test_images_exported": 0,
        "candidate_counts": dict(Counter(row["split"] for row in rows)),
        "eligible_counts": dict(Counter(row["split"] for row in eligible)),
        "unique_train_candidates_after_exact_pixel_dedup": len(train),
        "materialized_train_union": len(exported["train"]),
        "validation_images": len(exported["val"]),
        "variants": len(STRATEGIES) * len(budgets) * len(train_seeds),
        "candidate_index_sha256": sha256(manifest_path),
        "preparation_seconds": time.monotonic() - started,
    }
    write_json(args.out / "data_manifest.json", summary)
    archive = args.out.with_suffix(".tar")
    with tarfile.open(archive, "w") as output:
        for path in sorted(args.out.rglob("*")):
            if path.is_file():
                output.add(path, arcname=path.relative_to(args.out), recursive=False)
    write_json(archive.with_suffix(".tar.sha256.json"), {
        "sha256": sha256(archive), "bytes": archive.stat().st_size,
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
