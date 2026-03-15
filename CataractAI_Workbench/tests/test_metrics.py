"""Tests for CataractAI_Workbench.backend.metrics module.

Covers compute_iou, compute_metrics, load_coco_annotations, and batch_evaluate.
"""

import json
from pathlib import Path

import pytest

from CataractAI_Workbench.backend.metrics import (
    batch_evaluate,
    compute_iou,
    compute_metrics,
    load_coco_annotations,
)


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------

class TestComputeIoU:
    """Tests for compute_iou([x1,y1,x2,y2], [x1,y1,x2,y2])."""

    def test_perfect_overlap(self):
        box = [0, 0, 10, 10]
        iou = compute_iou(box, box)
        assert iou == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        iou = compute_iou(box1, box2)
        assert iou == pytest.approx(0.0, abs=1e-5)

    def test_partial_overlap(self):
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        # Intersection: [5,5,10,10] -> 5*5 = 25
        # Union: 100 + 100 - 25 = 175
        expected = 25.0 / (175.0 + 1e-6)
        iou = compute_iou(box1, box2)
        assert iou == pytest.approx(expected, abs=1e-4)

    def test_identical_boxes(self):
        box = [50, 60, 200, 300]
        iou = compute_iou(box, box)
        assert iou == pytest.approx(1.0, abs=1e-5)

    def test_one_box_inside_other(self):
        outer = [0, 0, 100, 100]
        inner = [25, 25, 75, 75]
        # Intersection = 50*50 = 2500; Union = 10000 + 2500 - 2500 = 10000
        expected = 2500.0 / (10000.0 + 1e-6)
        iou = compute_iou(outer, inner)
        assert iou == pytest.approx(expected, abs=1e-4)

    def test_zero_area_box(self):
        """A zero-area box (line or point) should produce IoU close to 0."""
        box_zero = [5, 5, 5, 5]  # point
        box_normal = [0, 0, 10, 10]
        iou = compute_iou(box_zero, box_normal)
        assert iou == pytest.approx(0.0, abs=1e-5)

    def test_touching_edges_no_overlap(self):
        box1 = [0, 0, 10, 10]
        box2 = [10, 0, 20, 10]
        iou = compute_iou(box1, box2)
        assert iou == pytest.approx(0.0, abs=1e-5)

    def test_symmetry(self):
        box1 = [10, 20, 50, 60]
        box2 = [30, 40, 70, 80]
        assert compute_iou(box1, box2) == pytest.approx(
            compute_iou(box2, box1), abs=1e-6
        )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for compute_metrics(predictions, ground_truth, iou_threshold)."""

    def test_all_correct(self):
        gt = [{"bbox": [0, 0, 10, 10]}, {"bbox": [20, 20, 30, 30]}]
        preds = [
            {"bbox": [0, 0, 10, 10], "score": 0.9},
            {"bbox": [20, 20, 30, 30], "score": 0.8},
        ]
        m = compute_metrics(preds, gt)
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["precision"] == pytest.approx(1.0, abs=1e-4)
        assert m["recall"] == pytest.approx(1.0, abs=1e-4)
        assert m["f1"] == pytest.approx(1.0, abs=1e-4)

    def test_all_wrong(self):
        gt = [{"bbox": [0, 0, 10, 10]}]
        preds = [{"bbox": [100, 100, 110, 110], "score": 0.9}]
        m = compute_metrics(preds, gt)
        assert m["tp"] == 0
        assert m["fp"] == 1
        assert m["fn"] == 1
        assert m["precision"] == pytest.approx(0.0, abs=1e-4)
        assert m["recall"] == pytest.approx(0.0, abs=1e-4)
        assert m["f1"] == pytest.approx(0.0, abs=1e-4)

    def test_mixed(self):
        gt = [{"bbox": [0, 0, 10, 10]}, {"bbox": [50, 50, 60, 60]}]
        preds = [
            {"bbox": [0, 0, 10, 10], "score": 0.95},  # TP
            {"bbox": [200, 200, 210, 210], "score": 0.5},  # FP
        ]
        m = compute_metrics(preds, gt)
        assert m["tp"] == 1
        assert m["fp"] == 1
        assert m["fn"] == 1  # second GT not matched
        assert m["precision"] == pytest.approx(0.5, abs=1e-4)
        assert m["recall"] == pytest.approx(0.5, abs=1e-4)

    def test_empty_predictions(self):
        gt = [{"bbox": [0, 0, 10, 10]}]
        m = compute_metrics([], gt)
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 1
        assert m["precision"] == pytest.approx(0.0, abs=1e-4)
        assert m["recall"] == pytest.approx(0.0, abs=1e-4)
        assert m["f1"] == pytest.approx(0.0, abs=1e-4)

    def test_empty_ground_truth(self):
        preds = [{"bbox": [0, 0, 10, 10], "score": 0.9}]
        m = compute_metrics(preds, [])
        assert m["tp"] == 0
        assert m["fp"] == 1
        assert m["fn"] == 0
        assert m["precision"] == pytest.approx(0.0, abs=1e-4)
        assert m["recall"] == pytest.approx(0.0, abs=1e-4)

    def test_both_empty(self):
        m = compute_metrics([], [])
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["precision"] == pytest.approx(0.0, abs=1e-4)
        assert m["recall"] == pytest.approx(0.0, abs=1e-4)
        assert m["f1"] == pytest.approx(0.0, abs=1e-4)

    def test_strict_iou_threshold(self):
        """With a very high threshold, a slightly offset prediction becomes FP."""
        gt = [{"bbox": [0, 0, 10, 10]}]
        preds = [{"bbox": [2, 2, 12, 12], "score": 0.9}]
        m_loose = compute_metrics(preds, gt, iou_threshold=0.3)
        m_strict = compute_metrics(preds, gt, iou_threshold=0.9)
        assert m_loose["tp"] == 1
        assert m_strict["tp"] == 0
        assert m_strict["fp"] == 1

    def test_score_ordering_matters(self):
        """Higher-scored predictions should be matched first."""
        gt = [{"bbox": [0, 0, 10, 10]}]
        preds = [
            {"bbox": [0, 0, 10, 10], "score": 0.5},  # exact match, lower score
            {"bbox": [1, 1, 11, 11], "score": 0.9},  # near match, higher score
        ]
        m = compute_metrics(preds, gt, iou_threshold=0.5)
        # The higher-scored pred is matched first; both have high IoU with gt,
        # so the 0.9-scored one grabs the GT, leaving the 0.5-scored as FP.
        assert m["tp"] == 1
        assert m["fp"] == 1

    def test_predictions_without_score_key(self):
        """Predictions missing 'score' should default to 0 and still work."""
        gt = [{"bbox": [0, 0, 10, 10]}]
        preds = [{"bbox": [0, 0, 10, 10]}]  # no "score" key
        m = compute_metrics(preds, gt)
        assert m["tp"] == 1
        assert m["fp"] == 0


# ---------------------------------------------------------------------------
# load_coco_annotations
# ---------------------------------------------------------------------------

class TestLoadCocoAnnotations:
    """Tests for load_coco_annotations(json_path)."""

    def test_valid_file(self, sample_coco_annotations):
        result = load_coco_annotations(sample_coco_annotations)

        # Check images dict keyed by filename
        assert "img_001.jpg" in result["images"]
        assert "img_002.jpg" in result["images"]
        assert result["images"]["img_001.jpg"]["width"] == 640

        # Check categories
        assert result["categories"][1] == "tool"
        assert result["categories"][2] == "background"

        # Check annotations converted from [x,y,w,h] to [x1,y1,x2,y2]
        anns_img1 = result["annotations"]["img_001.jpg"]
        assert len(anns_img1) == 2
        # First annotation: [100,100,50,50] -> [100,100,150,150]
        assert anns_img1[0]["bbox"] == [100, 100, 150, 150]
        # Second annotation: [200,200,30,40] -> [200,200,230,240]
        assert anns_img1[1]["bbox"] == [200, 200, 230, 240]

        anns_img2 = result["annotations"]["img_002.jpg"]
        assert len(anns_img2) == 1
        # [50,50,100,80] -> [50,50,150,130]
        assert anns_img2[0]["bbox"] == [50, 50, 150, 130]

    def test_missing_file(self, tmp_path):
        missing = str(tmp_path / "nonexistent.json")
        with pytest.raises((FileNotFoundError, OSError)):
            load_coco_annotations(missing)

    def test_malformed_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json content", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_coco_annotations(str(bad_file))

    def test_empty_annotations(self, tmp_path):
        """A valid COCO file with no annotations returns empty dicts."""
        data = {"images": [], "annotations": [], "categories": []}
        path = tmp_path / "empty_coco.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = load_coco_annotations(str(path))
        assert result["images"] == {}
        assert result["annotations"] == {}
        assert result["categories"] == {}

    def test_annotation_with_unknown_image_id_skipped(self, tmp_path):
        """Annotations referencing a non-existent image_id are skipped."""
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                {"id": 2, "image_id": 999, "category_id": 1, "bbox": [0, 0, 5, 5]},  # unknown image
            ],
            "categories": [{"id": 1, "name": "tool"}],
        }
        path = tmp_path / "partial.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = load_coco_annotations(str(path))
        assert "a.jpg" in result["annotations"]
        assert len(result["annotations"]["a.jpg"]) == 1

    def test_preserves_area_and_iscrowd(self, sample_coco_annotations):
        result = load_coco_annotations(sample_coco_annotations)
        first_ann = result["annotations"]["img_001.jpg"][0]
        assert first_ann["area"] == 2500
        assert first_ann["iscrowd"] == 0


# ---------------------------------------------------------------------------
# batch_evaluate
# ---------------------------------------------------------------------------

class TestBatchEvaluate:
    """Tests for batch_evaluate(inference_fn, image_paths, annotations, ...)."""

    def _make_inference_fn(self, mapping: dict):
        """Create a mock inference function from a filename->predictions mapping."""
        def fn(path):
            fname = Path(path).name
            return mapping.get(fname, [])
        return fn

    def test_basic_evaluation(self, tmp_path):
        annotations = {
            "img1.jpg": [{"bbox": [0, 0, 10, 10]}],
            "img2.jpg": [{"bbox": [20, 20, 30, 30]}],
        }
        pred_map = {
            "img1.jpg": [{"bbox": [0, 0, 10, 10], "score": 0.9}],
            "img2.jpg": [{"bbox": [20, 20, 30, 30], "score": 0.8}],
        }
        paths = [str(tmp_path / "img1.jpg"), str(tmp_path / "img2.jpg")]
        result = batch_evaluate(
            self._make_inference_fn(pred_map), paths, annotations
        )
        assert result["total_tp"] == 2
        assert result["total_fp"] == 0
        assert result["total_fn"] == 0
        assert result["precision"] == pytest.approx(1.0, abs=1e-4)
        assert result["recall"] == pytest.approx(1.0, abs=1e-4)
        assert result["f1"] == pytest.approx(1.0, abs=1e-4)
        assert result["images_tested"] == 2
        assert len(result["per_image"]) == 2
        assert result["avg_confidence"] == pytest.approx(0.85, abs=1e-4)

    def test_empty_image_list(self):
        result = batch_evaluate(
            lambda p: [], [], {}
        )
        assert result["images_tested"] == 0
        assert result["total_tp"] == 0
        assert result["per_image"] == []
        assert result["avg_confidence"] == pytest.approx(0.0, abs=1e-4)

    def test_no_annotations_for_images(self, tmp_path):
        """Predictions exist but no ground truth -- all FP."""
        pred_map = {
            "img.jpg": [{"bbox": [0, 0, 10, 10], "score": 0.7}],
        }
        paths = [str(tmp_path / "img.jpg")]
        result = batch_evaluate(
            self._make_inference_fn(pred_map), paths, {}
        )
        assert result["total_fp"] == 1
        assert result["total_tp"] == 0
        assert result["total_fn"] == 0

    def test_progress_callback(self, tmp_path):
        annotations = {"a.jpg": [{"bbox": [0, 0, 5, 5]}]}
        pred_map = {"a.jpg": [{"bbox": [0, 0, 5, 5], "score": 0.6}]}
        paths = [str(tmp_path / "a.jpg")]

        callback_log = []

        def on_progress(current, total, fname):
            callback_log.append((current, total, fname))

        batch_evaluate(
            self._make_inference_fn(pred_map),
            paths,
            annotations,
            progress_callback=on_progress,
        )
        assert len(callback_log) == 1
        assert callback_log[0] == (1, 1, "a.jpg")

    def test_per_image_details(self, tmp_path):
        annotations = {
            "x.jpg": [{"bbox": [0, 0, 10, 10]}, {"bbox": [50, 50, 60, 60]}],
        }
        pred_map = {
            "x.jpg": [{"bbox": [0, 0, 10, 10], "score": 0.95}],  # 1 TP, 1 FN
        }
        paths = [str(tmp_path / "x.jpg")]
        result = batch_evaluate(
            self._make_inference_fn(pred_map), paths, annotations
        )
        pi = result["per_image"][0]
        assert pi["filename"] == "x.jpg"
        assert pi["tp"] == 1
        assert pi["fp"] == 0
        assert pi["fn"] == 1
        assert pi["gt_count"] == 2
        assert pi["pred_count"] == 1

    def test_custom_iou_threshold(self, tmp_path):
        """A strict IoU threshold turns a near-miss into FP."""
        annotations = {"img.jpg": [{"bbox": [0, 0, 10, 10]}]}
        pred_map = {"img.jpg": [{"bbox": [3, 3, 13, 13], "score": 0.9}]}
        paths = [str(tmp_path / "img.jpg")]

        result_loose = batch_evaluate(
            self._make_inference_fn(pred_map), paths, annotations, iou_threshold=0.2
        )
        result_strict = batch_evaluate(
            self._make_inference_fn(pred_map), paths, annotations, iou_threshold=0.9
        )
        assert result_loose["total_tp"] == 1
        assert result_strict["total_tp"] == 0
        assert result_strict["total_fp"] == 1
