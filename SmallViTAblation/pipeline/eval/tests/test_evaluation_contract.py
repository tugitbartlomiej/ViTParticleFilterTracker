"""CPU regressions for complete-image COCO evaluation and validation-only tuning."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from SmallViTAblation.pipeline.eval import common, evaluate, operating_point
from SmallViTAblation.pipeline.transfer import metrics as transfer


@pytest.fixture
def gt():
    return {
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100}
                   for i in (1, 2)],
        "categories": [{"id": 1, "name": "tip"}],
        "annotations": [{"id": i, "image_id": i, "category_id": 1,
                         "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
                        for i in (1, 2)],
    }


@pytest.fixture
def predictions():
    return [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.8}]


@pytest.fixture
def protocol():
    return {"evaluation": {"map_conf_threshold": 0.001, "iou_threshold": 0.5},
            "eval_agent3": {"evaluation": {"single_class_remap": False},
                            "operating_point": {"f1_curve_max_points": 200,
                                                "low_confidence_probe_threshold": 0.3}}}


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_coco_keeps_images_without_predictions(gt, predictions):
    result = common.compute_map_metrics(gt, predictions)
    # 51 of the 101 recall thresholds remain reachable with recall=0.5.
    assert result["mAP_0.5"] == pytest.approx(51 / 101)
    assert result["AR@100"] == pytest.approx(0.5)


def test_empty_predictions_with_positive_gt_are_zero(gt):
    assert common.compute_map_metrics(gt, [])["mAP_0.5"] == 0


def test_unknown_prediction_image_is_rejected(gt, predictions):
    predictions[0]["image_id"] = 999
    with pytest.raises(ValueError, match="image_id"):
        common.compute_map_metrics(gt, predictions)


def test_single_class_remap_rejects_unexpected_class(gt, predictions):
    predictions[0]["category_id"] = 99
    with pytest.raises(ValueError, match="category_id"):
        common.remap_dt_categories_single_class(gt, predictions)


def test_zero_based_class_mapping_preserves_input(gt, predictions):
    predictions[0]["category_id"] = 0
    mapped, changed = common.remap_dt_categories_single_class(gt, predictions)
    assert mapped[0]["category_id"] == 1
    assert changed
    assert predictions[0]["category_id"] == 0


@pytest.mark.parametrize("empty", [False, True])
def test_import_without_sidecar_evaluates_full_annotation_set(tmp_path, protocol, gt, predictions, empty):
    source = write_json(tmp_path / "input.json", [] if empty else predictions)
    annotations = write_json(tmp_path / "gt.json", gt)
    sidecar = evaluate.prepare_predictions(tmp_path / "out", protocol, "val", annotations,
                                           predictions_file=source)
    assert sidecar["image_ids"] == [1, 2]
    result = evaluate.compute_block(tmp_path / "out", protocol, "val")
    assert result["block"]["FN"] == (2 if empty else 1)


def test_import_rejects_unknown_evaluated_ids(tmp_path, protocol, gt, predictions):
    source = write_json(tmp_path / "input.json", predictions)
    annotations = write_json(tmp_path / "gt.json", gt)
    ids = write_json(tmp_path / "ids.json", {"image_ids": [1, 999]})
    with pytest.raises(ValueError, match="image_id"):
        evaluate.prepare_predictions(tmp_path / "out", protocol, "val", annotations,
                                     predictions_file=source, evaluated_images_file=ids)


@pytest.mark.parametrize("set_name", ["test", "external_full", "external_clean"])
def test_test_threshold_cannot_be_fitted_without_validation(tmp_path, protocol, gt, predictions, set_name):
    with pytest.raises(ValueError, match="walidac|validation"):
        evaluate.resolve_operating_threshold(tmp_path, protocol, gt, predictions, 0.5, set_name)


@pytest.mark.parametrize("source", ["test", None])
def test_operating_point_requires_validation_provenance(tmp_path, protocol, gt, predictions, source):
    op = {"operating_threshold": 0.4}
    if source is not None:
        op["threshold_selected_on"] = source
    write_json(tmp_path / "operating_point.json", op)
    with pytest.raises(ValueError, match="walidac|validation"):
        evaluate.resolve_operating_threshold(tmp_path, protocol, gt, predictions, 0.5, "test")


def test_validation_threshold_transfers_unchanged(tmp_path, protocol, gt, predictions):
    write_json(tmp_path / "operating_point.json",
               {"operating_threshold": 0.4, "threshold_selected_on": "val"})
    assert evaluate.resolve_operating_threshold(tmp_path, protocol, gt, predictions, 0.5, "test") == (0.4, "val")


@pytest.mark.parametrize("metrics_name", ["metrics__last_epoch.json", "metrics_summary.json"])
def test_assemble_is_repeatable_with_named_aggregate(tmp_path, protocol, gt, predictions, metrics_name):
    source = write_json(tmp_path / "input.json", predictions)
    annotations = write_json(tmp_path / "gt.json", gt)
    evaluate.prepare_predictions(tmp_path, protocol, "val", annotations, predictions_file=source)
    evaluate.compute_block(tmp_path, protocol, "val", operating_threshold=0.4, threshold_source="cli")
    for _ in range(2):
        path = evaluate.assemble(tmp_path, protocol, "demo", "last_epoch",
                                 metrics_name, operating_threshold=0.4,
                                 threshold_selected_on="cli")
    assert set(json.loads(path.read_text())["results"]) == {"val"}


def test_operating_point_respects_disabled_remap(tmp_path, protocol, gt, predictions):
    predictions[0]["category_id"] = 0
    annotations = write_json(tmp_path / "gt.json", gt)
    pred = write_json(tmp_path / "predictions_val.json", predictions)
    write_json(tmp_path / "evaluated_images_val.json",
               {"image_ids": [1, 2], "annotations_file": str(annotations), "predictions_file": str(pred)})
    sets = operating_point.load_sets(tmp_path, protocol)
    assert sets["val"]["dt"][0]["category_id"] == 0


def test_transfer_uses_best_unmatched_ground_truth():
    boxes = np.array([[0, 0, 10, 10], [2, 0, 10, 10]], dtype=float)
    pred = {1: (np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float), np.array([0.9, 0.8]))}
    assert transfer.match_at_threshold(pred, {1: boxes}, 0.1, 0.5) == (2, 0, 0)


def test_transfer_counts_are_class_aware(gt, predictions):
    gt = copy.deepcopy(gt)
    gt["categories"].append({"id": 2, "name": "other"})
    predictions[0]["category_id"] = 2
    result = transfer.compute_block(gt, predictions, 0.1, 0.5)
    assert (result["TP"], result["FP"], result["FN"]) == (0, 1, 2)


def test_transfer_threshold_search_includes_low_scores():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    threshold, f1 = transfer.select_operating_threshold({1: (boxes, np.array([0.005]))}, {1: boxes}, 0.5)
    assert threshold <= 0.005
    assert f1 == 1.0


def test_transfer_threshold_tuning_does_not_match_another_class(gt, predictions):
    predictions[0]["category_id"] = 2
    gt["categories"].append({"id": 2, "name": "other"})
    _, f1 = transfer.select_operating_threshold(transfer.parse_preds(predictions),
                                               transfer.parse_gt(gt)[0], 0.5)
    assert f1 == 0.0


def test_invalid_import_does_not_replace_previous_predictions(tmp_path, protocol, gt, predictions):
    out = tmp_path / "out"
    out.mkdir()
    saved = write_json(out / "predictions_val.json", [])
    previous = saved.read_bytes()
    annotations = write_json(tmp_path / "gt.json", gt)
    ids = write_json(tmp_path / "ids.json", {"image_ids": [999]})
    source = write_json(tmp_path / "input.json", predictions)
    with pytest.raises(ValueError):
        evaluate.prepare_predictions(out, protocol, "val", annotations,
                                     predictions_file=source, evaluated_images_file=ids)
    assert saved.read_bytes() == previous


def test_cli_validation_to_test_workflow(tmp_path, protocol, gt, predictions):
    import yaml
    cfg = tmp_path / "protocol.yaml"
    cfg.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    annotations = write_json(tmp_path / "gt.json", gt)
    source = write_json(tmp_path / "input.json", predictions)
    out = tmp_path / "out"
    scripts = Path(evaluate.__file__).parent
    base = ["--protocol", str(cfg), "--output-dir", str(out)]
    def run(script, *args):
        result = subprocess.run([sys.executable, str(scripts / script), *base, *args],
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert result.returncode == 0, result.stdout + result.stderr
    run("evaluate.py", "--set", "val", "--annotations", str(annotations), "--predictions-file", str(source))
    run("operating_point.py")
    run("evaluate.py", "--set", "test", "--annotations", str(annotations), "--predictions-file", str(source))
    run("evaluate.py", "--assemble-only", "--run-id", "demo", "--checkpoint-rule", "last_epoch")
    result = json.loads((out / "metrics.json").read_text())
    assert result["threshold_selected_on"] == "val"
    assert result["results"]["test"]["FN"] == 1
    assert result["results"]["test"]["mAP_0.5"] == pytest.approx(51 / 101)


def test_image_subset_rejects_ambiguous_basenames(tmp_path):
    for folder in ("operation_a", "operation_b"):
        target = tmp_path / folder / "frame.jpg"
        target.parent.mkdir()
        target.write_bytes(b"jpeg")
    dataset = {
        "images": [{"id": 1, "file_name": "frame.jpg"}],
        "annotations": [], "categories": [{"id": 1, "name": "instrument"}],
    }
    with pytest.raises(ValueError, match="Niejednoznaczna"):
        common.evaluated_image_ids_from_dir(dataset, tmp_path)
