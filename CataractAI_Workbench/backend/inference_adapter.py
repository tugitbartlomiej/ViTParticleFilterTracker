"""Polymorphic model inference adapters for DETR and YOLO.

Uses ABC-based design so each model type encapsulates its own loading,
prediction, and dependency-checking logic. The ``InferenceFactory``
provides a single entry point for creating the correct adapter.

Metrics and COCO parsing are delegated to ``metrics.py``.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Guard optional imports
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import DetrForObjectDetection, DetrImageProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    from ultralytics import YOLO as _YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _require(available: bool, name: str, install_hint: str = "") -> None:
    """Raise ImportError if a dependency is missing."""
    if not available:
        msg = f"{name} is required but not installed."
        if install_hint:
            msg += f" Install with: pip install {install_hint}"
        raise ImportError(msg)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ModelInference(ABC):
    """Abstract interface for model loading and single-image prediction."""

    @abstractmethod
    def load(self, checkpoint_path: str, device: str = "cuda", **kwargs) -> None:
        """Load model weights from *checkpoint_path* onto *device*."""

    @abstractmethod
    def predict(self, image_path: str, confidence: float = 0.5) -> List[dict]:
        """Run inference and return detections.

        Each detection is ``{bbox: [x1,y1,x2,y2], score: float, label: int}``.
        """

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return a short identifier such as ``"detr"`` or ``"yolo"``."""

    @property
    def is_loaded(self) -> bool:
        """Whether a model checkpoint has been successfully loaded."""
        return False


# ---------------------------------------------------------------------------
# DETR
# ---------------------------------------------------------------------------

class DETRInference(ModelInference):
    """DETR (facebook/detr-resnet-50) inference adapter."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device: str = "cpu"

    @property
    def model_type(self) -> str:
        return "detr"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def device(self) -> str:
        return self._device

    def load(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        *,
        num_labels: int = 1,
    ) -> None:
        _require(_HAS_TORCH, "PyTorch")
        _require(_HAS_TRANSFORMERS, "transformers", "transformers")

        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        self._model = model
        self._processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50",
        )
        self._device = device

    def predict(self, image_path: str, confidence: float = 0.5) -> List[dict]:
        _require(_HAS_TORCH, "PyTorch")
        _require(_HAS_PIL, "Pillow")

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor(
            [image.size[::-1]], device=self._device,
        )
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence,
        )

        detections: List[dict] = []
        if results:
            r = results[0]
            boxes = r["boxes"].cpu().tolist()
            scores = r["scores"].cpu().tolist()
            labels = r["labels"].cpu().tolist()
            for box, score, label in zip(boxes, scores, labels):
                detections.append({
                    "bbox": [round(v, 1) for v in box],
                    "score": round(score, 4),
                    "label": int(label),
                })
        return detections


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

class YOLOInference(ModelInference):
    """Ultralytics YOLO inference adapter."""

    def __init__(self) -> None:
        self._model = None

    @property
    def model_type(self) -> str:
        return "yolo"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return None

    @property
    def device(self) -> str:
        return "cpu"

    def load(self, checkpoint_path: str, device: str = "cuda", **kwargs) -> None:
        _require(_HAS_ULTRALYTICS, "ultralytics", "ultralytics")
        self._model = _YOLO(checkpoint_path)

    def predict(self, image_path: str, confidence: float = 0.5) -> List[dict]:
        results = self._model(image_path, conf=confidence, verbose=False)
        detections: List[dict] = []
        if results:
            r = results[0]
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().tolist()
                detections.append({
                    "bbox": [round(v, 1) for v in xyxy],
                    "score": round(float(box.conf[0].cpu()), 4),
                    "label": int(box.cls[0].cpu()),
                })
        return detections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[ModelInference]] = {
    "detr": DETRInference,
    "yolo": YOLOInference,
}


class InferenceFactory:
    """Create ``ModelInference`` instances by model-type string."""

    @staticmethod
    def create(model_type: str) -> ModelInference:
        cls = _REGISTRY.get(model_type.lower())
        if cls is None:
            raise ValueError(
                f"Unknown model type: {model_type!r}. "
                f"Available: {', '.join(_REGISTRY)}"
            )
        return cls()

    @staticmethod
    def available_types() -> List[str]:
        return list(_REGISTRY)
