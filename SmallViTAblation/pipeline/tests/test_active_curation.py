from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from SmallViTAblation.pipeline.active_learning.acquisition import select_batch
from SmallViTAblation.pipeline.active_learning.simulate import main as simulate_main
from SmallViTAblation.pipeline.active_learning.merge_scores import merge as merge_scores
from SmallViTAblation.pipeline.latent_dynamics.synthetic_benchmark import make_case
from SmallViTAblation.pipeline.latent_dynamics.gle import diagnose_gle
from SmallViTAblation.pipeline.eval.group_bootstrap import paired_group_bootstrap
from SmallViTAblation.pipeline.transfer.transfer_common import build_coco
from SmallViTAblation.pipeline.transfer.make_fixtures import build as build_transfer_fixture
from SmallViTAblation.pipeline.transfer.train_transfer import (
    _rtdetr_config, apply_profile, main as transfer_main)
from SmallViTAblation.pipeline.train.train_rfdetr import iter_predictions
from SmallViTAblation.pipeline.features.el2n_from_proxy import (
    detection_error, prediction_arrays)
from SmallViTAblation.pipeline.features.train_el2n_proxy import build_proxy_coco_dir
from SmallViTAblation.pipeline.selection.materialize import build_coco as materialize_coco
from SmallViTAblation.tools.build_active_manifest import main as build_manifest_main


def toy_pool(n=24):
    rows = []
    for i in range(n):
        rows.append({"frame_id": f"f{i}", "split": "train",
                     "group_id": f"g{i % 3}", "video_id": f"v{i % 3}",
                     "frame_index": i // 3, "path": f"image{i}.jpg",
                     "native_width": 100, "native_height": 80,
                     "annotations": []})
    features = np.column_stack([np.arange(n), np.arange(n) % 5, np.ones(n)])
    return rows, features


def test_exact_incremental_budget_is_deterministic():
    rows, features = toy_pool()
    first, audit = select_batch(rows, features, batch_size=8,
                                strategy="uniform_time", seed=42,
                                min_gap_frames=2)
    second, _ = select_batch(rows, features, batch_size=7,
                             strategy="coverage", seed=42,
                             selected_ids=first, min_gap_frames=2)
    repeated, _ = select_batch(rows, features, batch_size=7,
                               strategy="coverage", seed=42,
                               selected_ids=first, min_gap_frames=2)
    assert len(first) == 8 and len(second) == 7
    assert len(set(first + second)) == 15
    assert second == repeated
    assert audit["annotation_fields_read"] is False


def test_non_train_frame_is_rejected():
    rows, features = toy_pool()
    rows[-1]["split"] = "val"
    try:
        select_batch(rows, features, batch_size=2, strategy="random", seed=1)
    except ValueError as exc:
        assert "non-training" in str(exc)
    else:
        raise AssertionError("validation frame reached acquisition")


def test_simulation_writes_cumulative_compatible_variants(tmp_path: Path):
    rows, features = toy_pool()
    pool = tmp_path / "pool.jsonl"
    with pool.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    cache = tmp_path / "features.npz"
    np.savez(cache, frame_ids=np.asarray([r["frame_id"] for r in rows]),
             features=features)
    out = tmp_path / "variants"
    assert simulate_main(["--pool", str(pool), "--embeddings", str(cache),
                          "--out", str(out), "--strategy", "coverage",
                          "--budgets", "6", "12", "--min-gap-frames", "1"]) == 0
    variants = sorted(out.glob("active-coverage-*.json"))
    assert len(variants) == 2
    small, large = [json.loads(path.read_text(encoding="utf-8")) for path in variants]
    if small["budget"] > large["budget"]:
        small, large = large, small
    assert len(small["frame_ids"]) == 6
    assert len(large["frame_ids"]) == 12
    assert set(small["frame_ids"]).issubset(large["frame_ids"])


def test_hidden_memory_has_larger_history_gain_than_markov():
    summaries = {}
    for kind in ("markov", "hidden_memory"):
        rows, features = make_case(kind, seed=7, groups=5, frames=100, dim=4)
        scores, summary, arrays = diagnose_gle(
            rows, features, seed=7, dim=4, lags=3, folds=3,
            max_gap_frames=1, max_fit_windows=2000, residual_modes=3)
        summaries[kind] = summary
        valid = [row for row in scores if row["has_valid_window"]]
        assert valid and all(row["score_assigned_to"] == "predicted_target_frame"
                             for row in valid)
        assert arrays["residual_mode_vectors"].shape == (4, 3)
    assert summaries["hidden_memory"]["mean_history_gain"] > summaries["markov"]["mean_history_gain"]


def test_multi_object_coco_contract():
    rows, _ = toy_pool(1)
    rows[0]["annotations"] = [
        {"bbox": [1, 2, 10, 12], "iscrowd": 0},
        {"bbox": [20, 22, 8, 9], "iscrowd": 0},
    ]
    coco = build_coco(rows, Path("."))
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 2
    assert {ann["image_id"] for ann in coco["annotations"]} == {1}
    assert coco["categories"][0]["name"] == "instrument"


def test_legacy_materializer_preserves_multiple_objects(tmp_path: Path):
    rows, _ = toy_pool(1)
    rows[0]["annotations"] = [
        {"bbox": [1, 2, 10, 12], "iscrowd": 0},
        {"bbox": [20, 22, 8, 9], "iscrowd": 0},
    ]
    manifest = tmp_path / "pool_manifest.jsonl"
    manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    variant = {"run_id": "multi", "seed": 1, "budget": 1,
               "frame_ids": [rows[0]["frame_id"]]}
    coco = materialize_coco(variant, manifest, absolute=False)
    assert len(coco["annotations"]) == 2


def test_active_manifest_pairs_seeds_and_writes_lock(tmp_path: Path):
    variants = tmp_path / "variants"
    variants.mkdir()
    for seed in (1, 2):
        value = {"strategy": "uniform_time", "budget": 2, "seed": seed,
                 "frame_ids": [f"f{seed}a", f"f{seed}b"]}
        (variants / f"active-uniform_time-s{seed}-b2.json").write_text(
            json.dumps(value), encoding="utf-8")
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        "version: 1\nprofiles:\n  p:\n    models: [yolov8n]\n"
        "    budgets: [2]\n    seeds: [1, 2]\n    seed_pairing: matched\n"
        "    strategies: [uniform_time]\n    max_runs: 2\n"
        "active_curation:\n  strategies: [uniform_time]\n",
        encoding="utf-8")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("split:\n  seed: 99\n", encoding="utf-8")
    out = tmp_path / "runs.jsonl"
    assert build_manifest_main([
        "--variants-dir", str(variants), "--profile", "p",
        "--profiles-config", str(profiles), "--protocol", str(protocol),
        "--out", str(out),
    ]) == 0
    runs = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(runs) == 2
    assert {(run["selection_seed"], run["train_seed"]) for run in runs} == {(1, 1), (2, 2)}
    assert all(run["split_seed"] == 99 for run in runs)
    lock = json.loads((tmp_path / "runs.jsonl.lock.json").read_text(encoding="utf-8"))
    assert lock["runs"] == 2 and lock["manifest_sha256"]


def test_score_files_are_joined_by_frame_id(tmp_path: Path):
    first = tmp_path / "uncertainty.jsonl"
    second = tmp_path / "dynamics.jsonl"
    first.write_text(
        '{"frame_id":"b","uncertainty":0.7}\n'
        '{"frame_id":"a","uncertainty":0.2}\n', encoding="utf-8")
    second.write_text(
        '{"frame_id":"a","history_gain":0.4}\n'
        '{"frame_id":"b","history_gain":0.1}\n', encoding="utf-8")
    rows = merge_scores([first, second])
    assert rows == [
        {"frame_id": "a", "uncertainty": 0.2, "history_gain": 0.4},
        {"frame_id": "b", "uncertainty": 0.7, "history_gain": 0.1},
    ]


def test_operation_bootstrap_uses_paired_draws():
    gt = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 20, "height": 20, "group_id": "g1"},
            {"id": 2, "file_name": "b.jpg", "width": 20, "height": 20, "group_id": "g2"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [2, 2, 5, 5], "area": 25, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "instrument"}],
    }
    perfect = [
        {"image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5], "score": .9},
        {"image_id": 2, "category_id": 1, "bbox": [2, 2, 5, 5], "score": .9},
    ]
    empty = []
    report = paired_group_bootstrap(
        gt, {"uniform_time": empty, "curated": perfect}, baseline="uniform_time",
        iterations=8, seed=7)
    assert report["groups"] == 2 and report["paired_draws"] is True
    assert report["paired_differences"]["curated"]["mAP_0.5:0.95"]["ci_low"] > 0


def test_fast_and_transformer_adapters_dry_run_without_model_packages(tmp_path: Path):
    data = tmp_path / "data"
    fixture = build_transfer_fixture(data, "FIXTURE", n_train=4, n_val=2, n_test=1)
    protocol = Path(__file__).resolve().parents[2] / "configs" / "protocol.yaml"
    profiles = Path(__file__).resolve().parents[2] / "configs" / "profiles.yaml"
    cases = [("yolov8n", "smoke"), ("rtdetrv2-s", "vit_smoke")]
    for arch, profile in cases:
        run_id = f"FIXTURE-{arch}"
        out = tmp_path / "results"
        assert transfer_main([
            "--run-id", run_id,
            "--variant-file", fixture["variant_file"],
            "--protocol", str(protocol),
            "--profiles-config", str(profiles),
            "--profile", profile,
            "--data-root", str(data),
            "--out", str(out),
            "--arch", arch,
            "--train-seed", "42",
            "--dry-run",
        ]) == 0
        manifest = json.loads((out / run_id / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["arch"] == arch
        assert manifest["experiment_profile"] == profile
        assert manifest["epochs_completed"] == 0
        expected_resolution = 384 if arch == "yolov8n" else 640
        assert manifest["resolution"] == expected_resolution
        assert manifest["iso_resource"]["precision"] == "fp16"
        assert manifest["iso_resource"]["train_time_augmentation"] == "none"


def test_profile_locks_architecture_specific_resolution(tmp_path: Path):
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text(
        "profiles:\n  p:\n    models: [rtdetrv2-s]\n    epochs: 3\n"
        "    resolution_by_model: {rtdetrv2-s: 640}\n", encoding="utf-8")
    resolved = apply_profile(
        {"model": {"resolution": 512}, "training": {"epochs": 60}},
        profiles, "p", "rtdetrv2-s")
    assert resolved["model"]["resolution"] == 640
    assert resolved["training"]["epochs"] == 3


def test_rtdetr_config_replaces_inherited_augmentations_and_resolution():
    iso = {"resolution": 640, "epochs": 3, "batch_size": 16,
           "precision": "bf16"}
    protocol = {"training": {"lr": 1e-4, "lr_backbone": 1e-5,
                              "weight_decay": 1e-4, "warmup_epochs": 2}}
    text = _rtdetr_config(
        Path("base.yml"), (Path("train"), Path("train.json")),
        (Path("val"), Path("val.json")), Path("out"), iso, protocol, 33)
    cfg = yaml.safe_load(text)
    train = cfg["train_dataloader"]
    types = [op["type"] for op in train["dataset"]["transforms"]["ops"]]
    assert types == ["Resize", "SanitizeBoundingBoxes", "ConvertPILImage", "ConvertBoxes"]
    assert train["dataset"]["transforms"]["ops"][0]["size"] == [640, 640]
    assert train["dataset"]["transforms"]["policy"] is None
    assert train["collate_fn"]["scales"] is None
    assert cfg["val_dataloader"]["dataset"]["transforms"]["ops"][0]["size"] == [640, 640]
    assert cfg["lr_warmup_scheduler"]["warmup_duration"] == 6


def test_rfdetr_detection_objects_and_el2n_use_one_to_one_matching():
    raw = SimpleNamespace(
        xyxy=np.asarray([[0, 0, 10, 10], [30, 30, 40, 40]], dtype=float),
        confidence=np.asarray([0.9, 0.8]), class_id=np.asarray([0, 0]))
    assert list(iter_predictions(raw))[0] == (0.0, 0.0, 10.0, 10.0, 0.9, 0)
    boxes, conf = prediction_arrays(raw)
    assert boxes.shape == (2, 4) and conf.tolist() == [0.9, 0.8]
    # One prediction overlaps both identical GT boxes and may satisfy only one.
    score = detection_error([[0, 0, 10, 10], [0, 0, 10, 10]], boxes[:1], conf[:1], 0.5)
    assert score == pytest.approx(0.55)
    assert detection_error([], boxes, conf, 0.5) == pytest.approx(0.9)


def test_el2n_proxy_validation_is_train_only_and_checkpoint_neutral(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fixture")
    records = [{"frame_id": "f1", "split": "train", "path": image.name,
                "native_width": 20, "native_height": 10,
                "label_status": "labeled",
                "annotations": [{"bbox": [1, 1, 5, 4]}]}]
    dataset = build_proxy_coco_dir(records, tmp_path, tmp_path / "proxy")
    train = json.loads((dataset / "train" / "_annotations.coco.json").read_text())
    valid = json.loads((dataset / "valid" / "_annotations.coco.json").read_text())
    assert train == valid
    assert Path(train["images"][0]["file_name"]).resolve() == image.resolve()


def test_contract_validator_accepts_legacy_multi_stage_fixture_without_writing(tmp_path: Path):
    build_transfer_fixture(tmp_path, "FIXTURE__s42__b20")
    validator = Path(__file__).resolve().parents[2] / "agents" / "ops" / "validate_contracts.py"
    result = subprocess.run(
        [sys.executable, str(validator), "--root", str(tmp_path),
         "--only", "pool", "variants"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / ".pool_train_ids.json").exists()
