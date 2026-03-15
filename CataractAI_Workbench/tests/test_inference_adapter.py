"""Tests for the polymorphic inference adapter layer.

Covers ModelInference ABC, InferenceFactory, DETRInference, and YOLOInference.
Heavy use of mocks since torch/transformers/ultralytics may not be installed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CataractAI_Workbench.backend.inference_adapter import (
    ModelInference,
    DETRInference,
    YOLOInference,
    InferenceFactory,
    _require,
)


# ===================================================================
# ModelInference ABC
# ===================================================================


class TestModelInferenceABC:
    """Verify that ModelInference cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ModelInference()

    def test_subclass_must_implement_load(self):
        """Subclass missing 'load' cannot be instantiated."""
        class Incomplete(ModelInference):
            def predict(self, image_path, confidence=0.5):
                return []

            @property
            def model_type(self):
                return "test"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_predict(self):
        """Subclass missing 'predict' cannot be instantiated."""
        class Incomplete(ModelInference):
            def load(self, checkpoint_path, device="cuda", **kwargs):
                pass

            @property
            def model_type(self):
                return "test"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_model_type(self):
        """Subclass missing 'model_type' cannot be instantiated."""
        class Incomplete(ModelInference):
            def load(self, checkpoint_path, device="cuda", **kwargs):
                pass

            def predict(self, image_path, confidence=0.5):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_can_be_instantiated(self):
        """A fully-implemented subclass works fine."""
        class Complete(ModelInference):
            def load(self, checkpoint_path, device="cuda", **kwargs):
                pass

            def predict(self, image_path, confidence=0.5):
                return []

            @property
            def model_type(self):
                return "test"

        obj = Complete()
        assert obj.model_type == "test"
        assert obj.is_loaded is False  # default from base class


# ===================================================================
# InferenceFactory
# ===================================================================


class TestInferenceFactory:
    """Tests for the InferenceFactory."""

    def test_create_detr(self):
        obj = InferenceFactory.create("detr")
        assert isinstance(obj, DETRInference)

    def test_create_yolo(self):
        obj = InferenceFactory.create("yolo")
        assert isinstance(obj, YOLOInference)

    def test_create_case_insensitive(self):
        obj = InferenceFactory.create("DETR")
        assert isinstance(obj, DETRInference)

    def test_create_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            InferenceFactory.create("unknown")

    def test_available_types_contains_detr_and_yolo(self):
        types = InferenceFactory.available_types()
        assert "detr" in types
        assert "yolo" in types

    def test_available_types_returns_list(self):
        types = InferenceFactory.available_types()
        assert isinstance(types, list)


# ===================================================================
# DETRInference
# ===================================================================


class TestDETRInference:
    """Tests for DETRInference adapter."""

    def test_model_type(self):
        inf = DETRInference()
        assert inf.model_type == "detr"

    def test_is_loaded_initially_false(self):
        inf = DETRInference()
        assert inf.is_loaded is False

    def test_model_initially_none(self):
        inf = DETRInference()
        assert inf.model is None

    def test_processor_initially_none(self):
        inf = DETRInference()
        assert inf.processor is None

    def test_device_default_cpu(self):
        inf = DETRInference()
        assert inf.device == "cpu"

    def test_predict_before_load_raises(self):
        """Calling predict without loading a model should fail."""
        inf = DETRInference()
        # predict tries to use self._processor which is None.
        # The exact error depends on what happens when calling None(...),
        # but it should raise some exception.
        with pytest.raises(Exception):
            inf.predict("/fake/image.jpg", confidence=0.5)

    @patch("CataractAI_Workbench.backend.inference_adapter._HAS_TORCH", True)
    @patch("CataractAI_Workbench.backend.inference_adapter._HAS_TRANSFORMERS", True)
    @patch("CataractAI_Workbench.backend.inference_adapter.DetrForObjectDetection")
    @patch("CataractAI_Workbench.backend.inference_adapter.DetrImageProcessor")
    @patch("CataractAI_Workbench.backend.inference_adapter.torch")
    def test_load_sets_model(self, mock_torch, mock_processor_cls,
                             mock_model_cls):
        """After successful load, is_loaded should be True."""
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_torch.load.return_value = {}
        mock_processor_cls.from_pretrained.return_value = MagicMock()

        inf = DETRInference()
        inf.load("/fake/checkpoint.pth", device="cpu")

        assert inf.is_loaded is True
        assert inf.model is mock_model
        assert inf.device == "cpu"

    @patch("CataractAI_Workbench.backend.inference_adapter._HAS_TORCH", False)
    def test_load_without_torch_raises_import_error(self):
        inf = DETRInference()
        with pytest.raises(ImportError, match="PyTorch"):
            inf.load("/fake/checkpoint.pth")


# ===================================================================
# YOLOInference
# ===================================================================


class TestYOLOInference:
    """Tests for YOLOInference adapter."""

    def test_model_type(self):
        inf = YOLOInference()
        assert inf.model_type == "yolo"

    def test_is_loaded_initially_false(self):
        inf = YOLOInference()
        assert inf.is_loaded is False

    def test_model_initially_none(self):
        inf = YOLOInference()
        assert inf.model is None

    def test_processor_always_none(self):
        inf = YOLOInference()
        assert inf.processor is None

    def test_device_always_cpu(self):
        inf = YOLOInference()
        assert inf.device == "cpu"

    def test_predict_before_load_raises(self):
        """Calling predict without loading a model should fail."""
        inf = YOLOInference()
        # self._model is None -> calling None(...) raises TypeError
        with pytest.raises(Exception):
            inf.predict("/fake/image.jpg", confidence=0.5)

    @patch("CataractAI_Workbench.backend.inference_adapter._HAS_ULTRALYTICS", True)
    @patch("CataractAI_Workbench.backend.inference_adapter._YOLO")
    def test_load_sets_model(self, mock_yolo_cls):
        """After successful load, is_loaded should be True."""
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        inf = YOLOInference()
        inf.load("/fake/yolo.pt", device="cpu")

        assert inf.is_loaded is True
        assert inf.model is mock_model
        mock_yolo_cls.assert_called_once_with("/fake/yolo.pt")

    @patch("CataractAI_Workbench.backend.inference_adapter._HAS_ULTRALYTICS", False)
    def test_load_without_ultralytics_raises_import_error(self):
        inf = YOLOInference()
        with pytest.raises(ImportError, match="ultralytics"):
            inf.load("/fake/yolo.pt")


# ===================================================================
# _require helper
# ===================================================================


class TestRequireHelper:
    """Tests for the _require guard function."""

    def test_require_passes_when_available(self):
        _require(True, "SomeLib")  # should not raise

    def test_require_raises_when_not_available(self):
        with pytest.raises(ImportError, match="MissingLib"):
            _require(False, "MissingLib")

    def test_require_includes_install_hint(self):
        with pytest.raises(ImportError, match="pip install some-package"):
            _require(False, "MissingLib", install_hint="some-package")
