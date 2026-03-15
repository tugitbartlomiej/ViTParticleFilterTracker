"""Tests for CataractAI Workbench analysis services.

Covers CheckpointAnalysisService, ParameterRecommendationEngine,
and DatasetAnalysisService.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CataractAI_Workbench.app.core.services.checkpoint_service import (
    CheckpointAnalysisService,
)
from CataractAI_Workbench.app.core.services.parameter_service import (
    ParameterRecommendationEngine,
)
from CataractAI_Workbench.app.core.services.dataset_service import (
    DatasetAnalysisService,
)


# ---------------------------------------------------------------------------
# Helpers: fake tensors that mimic torch.Tensor API
# ---------------------------------------------------------------------------

class FakeTensor:
    """Lightweight mock that behaves like a torch.Tensor for stat methods."""

    def __init__(self, values):
        self._arr = np.array(values, dtype=np.float32)

    # shape / numel
    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return "torch.float32"

    def numel(self):
        return int(self._arr.size)

    # math
    def float(self):
        return self

    def mean(self):
        return FakeScalar(float(self._arr.mean()))

    def std(self):
        return FakeScalar(float(self._arr.std()))

    def min(self):
        return FakeScalar(float(self._arr.min()))

    def max(self):
        return FakeScalar(float(self._arr.max()))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def flatten(self):
        return self._arr.flatten()


class FakeScalar:
    """Wraps a float so .item() works like a 0-dim tensor."""

    def __init__(self, val: float):
        self._val = val

    def item(self):
        return self._val


# ===================================================================
# CheckpointAnalysisService
# ===================================================================


class TestCheckpointAnalysisServiceLoad:
    """Tests for analyze(action='load')."""

    def setup_method(self):
        self.svc = CheckpointAnalysisService()

    @patch("CataractAI_Workbench.app.core.services.checkpoint_service.torch",
           create=True)
    def test_load_checkpoint_with_model_state_dict_key(self, mock_torch_module):
        """torch.load returns dict with 'model_state_dict' key."""
        fake_sd = {"layer.weight": FakeTensor([1, 2, 3])}
        raw = {"model_state_dict": fake_sd, "epoch": 10, "loss": 0.05}

        # We need to patch at the import site inside the static method.
        # The method does `import torch` locally, so we patch builtins.
        import builtins
        real_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name == "torch":
                mod = MagicMock()
                mod.load.return_value = raw
                return mod
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_patched_import):
            result = self.svc.analyze({"action": "load", "path": "/fake/model.pth"})

        assert result["path"] == "/fake/model.pth"
        assert result["state_dict"] is fake_sd
        assert result["metadata"]["epoch"] == 10
        assert result["metadata"]["loss"] == 0.05
        assert "model_state_dict" not in result["metadata"]

    @patch("CataractAI_Workbench.app.core.services.checkpoint_service.torch",
           create=True)
    def test_load_checkpoint_raw_state_dict(self, _):
        """torch.load returns a plain dict (raw state dict, no wrapping key)."""
        fake_sd = {"conv.weight": FakeTensor([0.1, 0.2])}

        import builtins
        real_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name == "torch":
                mod = MagicMock()
                mod.load.return_value = fake_sd
                return mod
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_patched_import):
            result = self.svc.analyze({"action": "load", "path": "/fake/raw.pth"})

        assert result["state_dict"] is fake_sd
        assert result["metadata"] == {}

    @patch("CataractAI_Workbench.app.core.services.checkpoint_service.torch",
           create=True)
    def test_load_checkpoint_with_state_dict_key(self, _):
        """torch.load returns dict with 'state_dict' key."""
        fake_sd = {"bn.weight": FakeTensor([1])}
        raw = {"state_dict": fake_sd, "optimizer": {"lr": 1e-4}}

        import builtins
        real_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name == "torch":
                mod = MagicMock()
                mod.load.return_value = raw
                return mod
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_patched_import):
            result = self.svc.load_checkpoint("/fake/sd.pth")

        assert result["state_dict"] is fake_sd
        assert "optimizer" in result["metadata"]

    def test_analyze_dispatches_to_load(self):
        """analyze with action='load' delegates to load_checkpoint."""
        with patch.object(
            CheckpointAnalysisService, "load_checkpoint", return_value={"ok": True}
        ) as mock_load:
            result = self.svc.analyze({"action": "load", "path": "/x.pth"})
            mock_load.assert_called_once_with("/x.pth")
            assert result == {"ok": True}


class TestCheckpointAnalysisServiceStats:
    """Tests for analyze(action='stats') / compute_layer_stats."""

    def setup_method(self):
        self.svc = CheckpointAnalysisService()

    def test_compute_layer_stats_returns_correct_keys(self):
        sd = {"layer1.weight": FakeTensor([1.0, 2.0, 3.0, 4.0])}
        result = self.svc.compute_layer_stats(sd, "layer1.weight")

        assert result["layer_name"] == "layer1.weight"
        assert result["shape"] == [4]
        assert result["numel"] == 4
        assert result["dtype"] == "torch.float32"
        assert abs(result["mean"] - 2.5) < 1e-4
        assert result["min"] == pytest.approx(1.0)
        assert result["max"] == pytest.approx(4.0)
        assert len(result["values_flat"]) == 4

    def test_compute_layer_stats_missing_layer(self):
        sd = {"a": FakeTensor([1])}
        result = self.svc.compute_layer_stats(sd, "nonexistent")
        assert "error" in result

    def test_compute_layer_stats_non_tensor(self):
        sd = {"meta": "some_string"}
        result = self.svc.compute_layer_stats(sd, "meta")
        assert "error" in result

    def test_analyze_stats_action(self):
        sd = {"w": FakeTensor([5.0, 6.0])}
        result = self.svc.analyze({
            "action": "stats",
            "state_dict": sd,
            "layer_name": "w",
        })
        assert result["layer_name"] == "w"
        assert result["numel"] == 2


class TestCheckpointAnalysisServiceCompare:
    """Tests for analyze(action='compare') / compare_state_dicts."""

    def setup_method(self):
        self.svc = CheckpointAnalysisService()

    def test_compare_state_dicts_basic(self):
        sd_a = {"layer.weight": FakeTensor([1, 2, 3])}
        sd_b = {"layer.weight": FakeTensor([4, 5, 6])}

        result = self.svc.compare_state_dicts(sd_a, sd_b, "model_v1", "model_v2")
        assert result["name_a"] == "model_v1"
        assert result["name_b"] == "model_v2"
        assert len(result["rows"]) == 1
        layer, std_a, std_b, delta = result["rows"][0]
        assert layer == "layer.weight"
        assert delta >= 0

    def test_compare_only_common_keys(self):
        sd_a = {"shared": FakeTensor([1, 2]), "only_a": FakeTensor([3])}
        sd_b = {"shared": FakeTensor([3, 4]), "only_b": FakeTensor([5])}

        result = self.svc.compare_state_dicts(sd_a, sd_b)
        layers = [r[0] for r in result["rows"]]
        assert "shared" in layers
        assert "only_a" not in layers
        assert "only_b" not in layers

    def test_compare_skips_non_tensor_keys(self):
        sd_a = {"w": FakeTensor([1, 2]), "meta": "text"}
        sd_b = {"w": FakeTensor([3, 4]), "meta": "other"}

        result = self.svc.compare_state_dicts(sd_a, sd_b)
        layers = [r[0] for r in result["rows"]]
        assert "w" in layers
        assert "meta" not in layers

    def test_compare_sorted_by_delta_descending(self):
        sd_a = {
            "small_change": FakeTensor([1, 1, 1]),
            "big_change": FakeTensor([0, 0, 0]),
        }
        sd_b = {
            "small_change": FakeTensor([1.01, 1.01, 1.01]),
            "big_change": FakeTensor([100, 200, 300]),
        }
        result = self.svc.compare_state_dicts(sd_a, sd_b)
        assert result["rows"][0][0] == "big_change"

    def test_analyze_compare_action(self):
        sd_a = {"x": FakeTensor([1])}
        sd_b = {"x": FakeTensor([2])}
        result = self.svc.analyze({
            "action": "compare",
            "state_dict_a": sd_a,
            "state_dict_b": sd_b,
            "name_a": "A",
            "name_b": "B",
        })
        assert result["name_a"] == "A"

    def test_analyze_unknown_action_raises(self):
        with pytest.raises(ValueError, match="Unknown action"):
            self.svc.analyze({"action": "foobar"})


class TestCheckpointScanDirectory:
    """Tests for scan_directory."""

    def test_scan_directory_with_pth_files(self, tmp_path):
        svc = CheckpointAnalysisService()
        (tmp_path / "model_a.pth").write_bytes(b"\x00" * 1024)
        (tmp_path / "model_b.pth").write_bytes(b"\x00" * 2048)
        (tmp_path / "readme.txt").write_text("ignore me")

        results = svc.scan_directory(str(tmp_path))
        assert len(results) == 2
        names = {r["name"] for r in results}
        assert names == {"model_a.pth", "model_b.pth"}
        for r in results:
            assert "size_mb" in r
            assert r["size_mb"] > 0

    def test_scan_directory_empty(self, tmp_path):
        svc = CheckpointAnalysisService()
        results = svc.scan_directory(str(tmp_path))
        assert results == []


class TestCheckpointCountParams:
    """Tests for count_params."""

    def test_count_params_basic(self):
        svc = CheckpointAnalysisService()
        sd = {
            "layer1.weight": FakeTensor(np.zeros((3, 3))),
            "layer1.bias": FakeTensor(np.zeros(3)),
            "layer2.weight": FakeTensor(np.zeros((5, 3))),
        }
        total, names = svc.count_params(sd)
        assert total == 9 + 3 + 15  # 27
        assert len(names) == 3

    def test_count_params_skips_non_tensor(self):
        svc = CheckpointAnalysisService()
        sd = {
            "w": FakeTensor([1, 2, 3]),
            "config": {"lr": 1e-3},
        }
        total, names = svc.count_params(sd)
        assert total == 3
        assert names == ["w"]


# ===================================================================
# ParameterRecommendationEngine
# ===================================================================

def _make_config(lr=2e-6, bs=2, epochs=5, warmup=1, patience=6,
                 scheduler="StepLR", use_amp=True, base_model="detr"):
    """Helper to build a nested config dict."""
    return {
        "training_parameters": {
            "learning_rate": lr,
            "batch_size": bs,
            "epochs": epochs,
            "warmup_epochs": warmup,
            "weight_decay": 1e-4,
        },
        "optimization": {
            "scheduler": {"type": scheduler},
            "use_amp": use_amp,
        },
        "early_stopping": {"enabled": True, "patience": patience},
        "model_settings": {"base_model": base_model},
        "dataset_paths": {},
    }


class TestParameterRecommendationEngineAnalyze:
    """Tests for analyze() entry point."""

    def setup_method(self):
        self.engine = ParameterRecommendationEngine()

    def test_analyze_config_only(self):
        config = _make_config()
        result = self.engine.analyze({"config": config})
        assert "summary" in result
        assert "recommendations" in result
        assert "history_analysis" not in result

    def test_analyze_config_with_dataset_dir(self, tmp_path):
        # Create image dir + annotation file
        img_dir = tmp_path / "all_images"
        img_dir.mkdir()
        (img_dir / "img1.jpg").write_bytes(b"\xff")
        (img_dir / "img2.png").write_bytes(b"\xff")

        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        coco = {
            "images": [{"id": 1}, {"id": 2}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
            "categories": [{"id": 1, "name": "tool"}],
        }
        (ann_dir / "mixed_annotations.json").write_text(json.dumps(coco))

        config = _make_config()
        result = self.engine.analyze({
            "config": config,
            "dataset_dir": str(tmp_path),
        })
        assert "summary" in result
        assert "Images:" in result["summary"]

    def test_analyze_config_with_history(self):
        config = _make_config()
        history = [
            {"loss": 0.5, "val_loss": 0.6, "lr": 1e-5},
            {"loss": 0.3, "val_loss": 0.4, "lr": 1e-5},
            {"loss": 0.1, "val_loss": 0.2, "lr": 1e-6},
        ]
        result = self.engine.analyze({
            "config": config,
            "history": history,
        })
        assert "history_analysis" in result
        assert "text" in result["history_analysis"]
        assert len(result["history_analysis"]["train_losses"]) == 3


class TestBuildSummary:
    """Tests for _build_summary."""

    def setup_method(self):
        self.engine = ParameterRecommendationEngine()

    def test_summary_contains_config_values(self):
        config = _make_config(lr=5e-5, bs=4, epochs=10, warmup=2)
        summary = self.engine._build_summary(config, dataset_dir=None)
        assert "10" in summary          # epochs
        assert "4" in summary           # batch_size
        assert "5" in summary or "5.0e-05" in summary or "5e-05" in summary  # LR

    def test_summary_without_dataset(self):
        config = _make_config()
        summary = self.engine._build_summary(config, dataset_dir=None)
        assert "Set dataset dir" in summary

    def test_summary_with_dataset(self, tmp_path):
        img_dir = tmp_path / "all_images"
        img_dir.mkdir()
        for i in range(5):
            (img_dir / f"img_{i}.jpg").write_bytes(b"\xff")

        config = _make_config()
        summary = self.engine._build_summary(config, str(tmp_path))
        assert "5" in summary  # 5 images


class TestLrRecs:
    """Tests for _lr_recs."""

    def setup_method(self):
        self.engine = ParameterRecommendationEngine()

    def test_lr_too_low(self):
        recs = self.engine._lr_recs(1e-7, 100)
        assert len(recs) == 1
        assert "[!]" in recs[0]
        assert "conservative" in recs[0]

    def test_lr_too_high(self):
        recs = self.engine._lr_recs(5e-4, 100)
        assert len(recs) == 1
        assert "[!]" in recs[0]
        assert "high" in recs[0]

    def test_lr_good_range(self):
        recs = self.engine._lr_recs(1e-5, 100)
        assert len(recs) == 1
        assert "[OK]" in recs[0]

    def test_lr_too_low_includes_img_count(self):
        recs = self.engine._lr_recs(5e-7, 500)
        assert "500 images" in recs[0]

    def test_lr_too_low_no_images(self):
        recs = self.engine._lr_recs(5e-7, 0)
        assert "images" not in recs[0]


class TestVramRecs:
    """Tests for _vram_recs."""

    def setup_method(self):
        self.engine = ParameterRecommendationEngine()

    def test_vram_ok_detr(self):
        # batch_size=2, detr uses 2.5 GB/batch -> 5 GB, under 10 -> OK
        recs = self.engine._vram_recs(2, "detr")
        assert "[OK]" in recs[0]
        assert "DETR" in recs[0]

    def test_vram_warning_detr(self):
        # batch_size=8, detr uses 2.5 GB/batch -> 20 GB, over 10 -> warning
        recs = self.engine._vram_recs(8, "detr")
        assert "[!]" in recs[0]

    def test_vram_ok_yolo(self):
        # batch_size=8, yolo uses 1.0 GB/batch -> 8 GB, under 10 -> OK
        recs = self.engine._vram_recs(8, "yolo")
        assert "[OK]" in recs[0]

    def test_vram_warning_yolo(self):
        # batch_size=16, yolo uses 1.0 GB/batch -> 16 GB, over 10 -> warning
        recs = self.engine._vram_recs(16, "yolo")
        assert "[!]" in recs[0]

    def test_vram_unknown_model_defaults_to_2_5(self):
        # Unknown model type defaults to 2.5 GB/batch
        recs = self.engine._vram_recs(5, "unknown_model")
        # 5 * 2.5 = 12.5 -> over 10 -> warning
        assert "[!]" in recs[0]


# ===================================================================
# DatasetAnalysisService
# ===================================================================


class TestDatasetAnalysisServiceAnalyze:
    """Tests for DatasetAnalysisService.analyze."""

    def setup_method(self):
        self.svc = DatasetAnalysisService()

    def test_analyze_valid_coco(self):
        coco = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "b.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "c.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [10, 20, 50, 60]},
                {"id": 2, "image_id": 2, "category_id": 1,
                 "bbox": [30, 40, 70, 80]},
            ],
            "categories": [{"id": 1, "name": "tool"}],
        }
        result = self.svc.analyze({"coco_data": coco})

        assert "composition" in result
        assert "bboxes" in result
        assert "images" in result
        assert "recommendations" in result
        assert "category_names" in result
        assert result["category_names"] == ["tool"]
        assert result["bboxes"].shape == (2, 4)
        assert "3" in result["composition"]  # 3 images

    def test_analyze_with_filtered_category(self):
        coco = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [10, 20, 50, 60]},
                {"id": 2, "image_id": 1, "category_id": 2,
                 "bbox": [100, 200, 30, 40]},
            ],
            "categories": [
                {"id": 1, "name": "tool"},
                {"id": 2, "name": "iris"},
            ],
        }
        result = self.svc.analyze({
            "coco_data": coco,
            "selected_category": "tool",
        })
        # Only 1 annotation for "tool"
        assert result["bboxes"].shape == (1, 4)

    def test_analyze_empty_annotations(self):
        coco = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480}],
            "annotations": [],
            "categories": [],
        }
        result = self.svc.analyze({"coco_data": coco})
        assert result["bboxes"].shape == (0, 4)

    def test_analyze_counts_images_from_disk(self, tmp_path):
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        (img_dir / "img1.jpg").write_bytes(b"\xff")
        (img_dir / "img2.png").write_bytes(b"\xff")
        (img_dir / "readme.txt").write_text("not an image")

        coco = {"images": [], "annotations": [], "categories": []}
        result = self.svc.analyze({
            "coco_data": coco,
            "img_dir": str(img_dir),
        })
        # images list is empty, so n_images falls back to dir_img_count = 2
        assert "2" in result["composition"]


class TestDatasetGenerateRecommendations:
    """Tests for _generate_recommendations."""

    def test_background_heavy_dataset(self):
        """Over 80% background triggers warning."""
        recs = DatasetAnalysisService._generate_recommendations(
            n_images=100,
            n_annotations=10,
            zero_box_count=85,
            bboxes=np.array([[10, 20, 50, 60]] * 10),
            categories=[{"id": 1, "name": "tool"}],
            coco_data={"annotations": [{"category_id": 1}] * 10},
        )
        combined = "\n".join(recs)
        assert "85%" in combined
        assert "[!]" in combined

    def test_moderate_background(self):
        """50-80% background triggers moderate warning."""
        recs = DatasetAnalysisService._generate_recommendations(
            n_images=100,
            n_annotations=30,
            zero_box_count=65,
            bboxes=np.array([[10, 20, 50, 60]] * 30),
            categories=[{"id": 1, "name": "tool"}],
            coco_data={"annotations": [{"category_id": 1}] * 30},
        )
        combined = "\n".join(recs)
        assert "[~]" in combined
        assert "65%" in combined

    def test_very_small_dataset_warning(self):
        recs = DatasetAnalysisService._generate_recommendations(
            n_images=50,
            n_annotations=50,
            zero_box_count=0,
            bboxes=np.array([[10, 20, 50, 60]] * 50),
            categories=[{"id": 1, "name": "tool"}],
            coco_data={"annotations": [{"category_id": 1}] * 50},
        )
        combined = "\n".join(recs)
        assert "Very small dataset" in combined

    def test_small_dataset_warning(self):
        recs = DatasetAnalysisService._generate_recommendations(
            n_images=200,
            n_annotations=200,
            zero_box_count=0,
            bboxes=np.array([[10, 20, 50, 60]] * 200),
            categories=[{"id": 1, "name": "tool"}],
            coco_data={"annotations": [{"category_id": 1}] * 200},
        )
        combined = "\n".join(recs)
        assert "Small dataset" in combined

    def test_reasonable_dataset(self):
        bboxes = np.array([[10, 20, 50, 60]] * 600)
        recs = DatasetAnalysisService._generate_recommendations(
            n_images=600,
            n_annotations=600,
            zero_box_count=0,
            bboxes=bboxes,
            categories=[{"id": 1, "name": "tool"}],
            coco_data={"annotations": [{"category_id": 1}] * 600},
        )
        combined = "\n".join(recs)
        assert "[OK]" in combined

    def test_high_box_size_variation(self):
        """CV > 1 should trigger multi-scale warning."""
        bboxes = np.array([
            [0, 0, 5, 5],       # area = 25
            [0, 0, 500, 500],   # area = 250000
        ])
        recs = DatasetAnalysisService._generate_recommendations(
            n_images=600,
            n_annotations=2,
            zero_box_count=0,
            bboxes=bboxes,
            categories=[{"id": 1, "name": "tool"}],
            coco_data={"annotations": [{"category_id": 1}] * 2},
        )
        combined = "\n".join(recs)
        assert "variation" in combined.lower() or "multi-scale" in combined.lower()

    def test_category_imbalance_severe(self):
        """Max/min ratio > 10 triggers severe imbalance warning."""
        recs = DatasetAnalysisService._generate_recommendations(
            n_images=600,
            n_annotations=110,
            zero_box_count=0,
            bboxes=np.array([[10, 20, 50, 60]] * 110),
            categories=[
                {"id": 1, "name": "tool"},
                {"id": 2, "name": "iris"},
            ],
            coco_data={
                "annotations": (
                    [{"category_id": 1}] * 100
                    + [{"category_id": 2}] * 5
                ),
            },
        )
        combined = "\n".join(recs)
        assert "imbalance" in combined.lower()

    def test_no_issues(self):
        """No warnings when dataset is well-formed."""
        recs = DatasetAnalysisService._generate_recommendations(
            n_images=600,
            n_annotations=600,
            zero_box_count=10,
            bboxes=np.array([[10, 20, 50, 60]] * 600),
            categories=[{"id": 1, "name": "tool"}],
            coco_data={"annotations": [{"category_id": 1}] * 600},
        )
        # Should have at least one [OK]
        combined = "\n".join(recs)
        assert "[OK]" in combined
