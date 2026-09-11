#!/usr/bin/env python3
"""CPU-only reproduction of five findings from the read-only evaluation audit.

Run: py -3.11 -B reproduce_eval_bugs.py
Repeat without writing: py -3.11 -B reproduce_eval_bugs.py --stdout
Requires numpy, PyYAML and pycocotools. Does not import detector libraries,
load weights, run training, create fixtures on disk or modify repository files.
Only the adjacent eval_repro_results.json is created, with exclusive creation.
Exit 0: all five audited bugs reproduced; exit 1: a finding was not reproduced
or a test failed; exit 2: result file already exists (use --stdout).
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DEFAULT_ROOT = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], node


def make_gt(boxes_by_image):
    annotations = []
    for image_id, boxes in boxes_by_image.items():
        for box in boxes:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": 1, "bbox": box,
                                "area": box[2] * box[3], "iscrowd": 0})
    return {"info": {}, "images": [{"id": i} for i in boxes_by_image],
            "categories": [{"id": 1, "name": "instrument_tip"}],
            "annotations": annotations}


def prediction(image_id=1, box=None, score=0.9, category=1):
    return {"image_id": image_id, "category_id": category,
            "bbox": [0, 0, 10, 10] if box is None else box, "score": score}


def counts(values):
    return dict(zip(("TP", "FP", "FN"), map(int, values)))


def short_counts(block):
    return {key: block[key] for key in ("TP", "FP", "FN")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT,
                        help="Root of SmallViTAblation, not its parent repository")
    parser.add_argument("--stdout", action="store_true", help="Repeat without creating any file")
    args = parser.parse_args()
    result_path = Path(__file__).resolve().with_name("eval_repro_results.json")
    if not args.stdout and result_path.exists():
        print(f"Refusing to overwrite {result_path}; use --stdout to repeat.", file=sys.stderr)
        return 2
    root = args.repo_root.resolve()
    paths = {"common": root / "pipeline/eval/common.py",
             "evaluate": root / "pipeline/eval/evaluate.py",
             "transfer_metrics": root / "pipeline/transfer/metrics.py",
             "train": root / "pipeline/train/train_rfdetr.py"}
    before = {str(p.relative_to(root)): digest(p) for p in paths.values()}
    common = load_module("audit_eval_common", paths["common"])
    transfer = load_module("audit_transfer_metrics", paths["transfer_metrics"])
    cases = []

    def run_case(name, function):
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = function()
            result.update(id=name, execution="ok")
        except Exception as exc:
            result = {"id": name, "execution": "error", "bug_reproduced": False,
                      "error": {"type": type(exc).__name__, "message": str(exc)}}
        result["captured_stdout"] = stdout.getvalue()
        result["captured_stderr"] = stderr.getvalue()
        cases.append(result)

    def missing_images():
        gt = make_gt({1: [[0, 0, 10, 10]], 2: [[0, 0, 10, 10]]})
        dt = [prediction()]
        observed = common.compute_map_metrics(gt, dt)
        cg = COCO()
        cg.dataset = json.loads(json.dumps(gt))
        cg.createIndex()
        cd = cg.loadRes(dt)
        ev = COCOeval(cg, cd, "bbox")
        ev.params.imgIds = [1, 2]  # All evaluated images, including an image with no detections.
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        expected = {"mAP_0.5": float(ev.stats[1]), "mAP_0.5:0.95": float(ev.stats[0]),
                    "mAP_0.75": float(ev.stats[2]), "AR@100": float(ev.stats[8])}
        assert np.isclose(expected["mAP_0.5:0.95"], 51 / 101)
        assert np.isclose(expected["AR@100"], 0.5)
        return {"source": "pipeline/eval/common.py:245", "fixture": {"gt": gt, "predictions": dt},
                "observed": observed, "expected": expected,
                "expected_basis": "Independent COCOeval with explicit imgIds=[1,2]; 101 recall samples.",
                "bug_reproduced": bool(observed["mAP_0.5:0.95"] > expected["mAP_0.5:0.95"] + 0.1),
                "severity": "P1"}

    def matcher():
        gt = make_gt({1: [[0, 0, 10, 10], [3, 0, 10, 10]]})
        dt = [prediction(score=0.9), prediction(score=0.8)]
        observed = counts(transfer.match_at_threshold(transfer.parse_preds(dt), transfer.parse_gt(gt)[0], 0, 0.5))
        expected = {"TP": 2, "FP": 0, "FN": 0}
        reference = short_counts(common.counting_metrics(common.build_gt_index(gt), dt, 0, 0.5))
        assert reference == expected
        return {"source": "pipeline/transfer/metrics.py:76", "fixture": {"gt": gt, "predictions": dt, "iou_threshold": 0.5},
                "observed": observed, "expected": expected, "reference_eval_matcher": reference,
                "expected_basis": "Second detection matches the remaining GT: IoU=70/130 > 0.5.",
                "bug_reproduced": observed != expected, "severity": "P1"}

    def wrong_category():
        gt = make_gt({1: [[0, 0, 10, 10]]})
        dt = [prediction(category=99)]
        observed_transfer = counts(transfer.match_at_threshold(transfer.parse_preds(dt), transfer.parse_gt(gt)[0], 0, 0.5))
        remapped, applied = common.remap_dt_categories_single_class(gt, dt)
        observed_remap = short_counts(common.counting_metrics(common.build_gt_index(gt), remapped, 0, 0.5))
        reference = short_counts(common.counting_metrics(common.build_gt_index(gt), dt, 0, 0.5))
        expected = {"TP": 0, "FP": 1, "FN": 1}
        assert reference == expected
        return {"source": ["pipeline/transfer/metrics.py:144", "pipeline/transfer/metrics.py:159", "pipeline/eval/common.py:211"],
                "fixture": {"gt": gt, "predictions": dt},
                "observed": {"transfer": observed_transfer, "eval_with_single_class_remap": observed_remap, "remap_applied": applied},
                "expected": {"transfer": expected, "eval_with_single_class_remap": expected},
                "expected_basis": "Category 99 has no authorized mapping to GT category 1; localization alone is insufficient for TP.",
                "reference_without_remap": reference,
                "bug_reproduced": observed_transfer != expected and observed_remap != expected,
                "severity": "P1", "limitation": "Synthetic invalid class ID; not a claim about actual detector outputs on Eden."}

    def test_threshold():
        function, node = load_function(paths["evaluate"], "resolve_operating_threshold",
                                      {"Path": Path, "common": common, "np": np, "sys": sys})
        class AbsentOperatingPoint:
            def __truediv__(self, name):
                assert name == "operating_point.json"
                return self
            def exists(self):
                return False
        gt = make_gt({1: [[0, 0, 10, 10]]})
        dt = [prediction(score=0.9), prediction(box=[30, 30, 10, 10], score=0.8)]
        threshold, source = function(AbsentOperatingPoint(), {}, gt, dt, 0.5, "test")
        at_threshold = common.counting_metrics(common.build_gt_index(gt), dt, threshold, 0.5)
        return {"source": f"pipeline/eval/evaluate.py:{node.lineno}", "fixture": {"gt": gt, "predictions": dt, "set_name": "test", "operating_point_exists": False},
                "observed": {"action": "selected_threshold_on_test", "threshold": threshold, "threshold_selected_on": source, "f1": at_threshold["f1"]},
                "expected": {"action": "reject_missing_validation_threshold", "threshold": None},
                "expected_basis": "protocol.yaml evaluation.operating_threshold=select_on_val; absence of a validation threshold must not trigger test optimization.",
                "bug_reproduced": source == "max_f1_on_set:test", "severity": "P1",
                "limitation": "Exercises actual function with an in-memory absent-path stub; no real test set was accessed."}

    def checkpoint_sort():
        tree = ast.parse(paths["train"].read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_training")
        assignments = {}
        for node in function.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("epoch_ckpts", "last_ckpt"):
                        assignments[target.id] = node
        if set(assignments) != {"epoch_ckpts", "last_ckpt"}:
            raise RuntimeError("Checkpoint selection code changed; review the reproduction harness.")
        fixture = {"checkpoint.pth": 59, "checkpoint0009.pth": 9, "checkpoint0059.pth": 59,
                   "checkpoint_best_ema.pth": 20, "checkpoint_best_regular.pth": 17,
                   "checkpoint_best_total.pth": 20}
        class CheckpointDirectory:
            def rglob(self, pattern):
                assert pattern == "checkpoint*.pth"
                return [Path(name) for name in fixture]
        namespace = {"ckpt_dir": CheckpointDirectory()}
        for name in ("epoch_ckpts", "last_ckpt"):
            exec(compile(ast.Module(body=[assignments[name]], type_ignores=[]), str(paths["train"]), "exec"), namespace)
        selected = namespace["last_ckpt"].name
        latest_epoch = max(fixture.values())
        return {"source": [f"pipeline/train/train_rfdetr.py:{assignments[name].lineno}" for name in ("epoch_ckpts", "last_ckpt")],
                "fixture": {"checkpoint_epoch_metadata": fixture, "epoch_indexing": "zero_based"},
                "observed": {"sorted_names": [p.name for p in namespace["epoch_ckpts"]], "selected_filename": selected, "selected_epoch": fixture[selected]},
                "expected": {"selected_epoch": latest_epoch, "valid_filenames": [name for name, epoch in fixture.items() if epoch == latest_epoch]},
                "expected_basis": "last_epoch must refer to the final training epoch, independent of lexical filename order.",
                "bug_reproduced": fixture[selected] != latest_epoch, "severity": "P1",
                "limitation": "Actual source assignments executed against synthetic filenames and declared epoch metadata; no checkpoint weights or Eden artifacts inspected."}

    for name, function in [("map_missing_images", missing_images), ("transfer_matcher", matcher),
                           ("wrong_category", wrong_category), ("fallback_test_threshold", test_threshold),
                           ("lexicographic_checkpoint_sort", checkpoint_sort)]:
        run_case(name, function)
    after = {str(p.relative_to(root)): digest(p) for p in paths.values()}
    passed = all(c["execution"] == "ok" and c["bug_reproduced"] for c in cases)
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "repo_root": str(root), "python": platform.python_version(),
              "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "PyYAML", "pycocotools")},
              "source_sha256_before": before, "source_sha256_after": after,
              "audited_sources_unchanged_during_run": before == after,
              "script_sha256": digest(Path(__file__)),
              "scope": "Five CPU-only reproductions of local evaluation/training source logic.",
              "limitations": ["No GPU, detector inference, training or real dataset evaluation.",
                              "RF-DETR API claims in the companion report were checked against upstream 1.2.1 only; this script does not test those API claims.",
                              "Eden environment, library versions, historical results and real checkpoint metadata remain unverified."],
              "summary": {"cases": len(cases), "bugs_reproduced": sum(c["bug_reproduced"] for c in cases),
                          "all_five_bugs_reproduced": passed}, "cases": cases}
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.stdout:
        print(payload, end="")
    else:
        with result_path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(f"Created {result_path}")
        for case in cases:
            print(f"{case['id']}: execution={case['execution']}; bug_reproduced={case['bug_reproduced']}")
        print(f"Audited source hashes unchanged: {before == after}")
    return 0 if passed and before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
