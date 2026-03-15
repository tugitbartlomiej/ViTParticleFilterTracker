"""COCO prediction parsing and rendering for model comparison.

Handles loading COCO-format prediction files (both results-only and
full annotation formats), organizing predictions by image, and rendering
overlays with model badges using the shared BBoxPainter.
"""

import json
from pathlib import Path
from typing import Dict, List

from PyQt6.QtGui import QColor, QPixmap

from ...widgets.bbox_painter import BBoxPainter


class PredictionRenderer:
    """Loads COCO-format predictions and renders them onto images.

    This class separates pure data + rendering logic from the Qt widget
    layer, making it reusable and testable.
    """

    def __init__(self):
        self._predictions: Dict[int, List[dict]] = {}  # image_id -> [predictions]
        self._categories: Dict[int, str] = {}
        self._image_files: Dict[int, str] = {}  # image_id -> filename
        self._name: str = ""

    @property
    def name(self) -> str:
        return self._name

    @property
    def predictions(self) -> Dict[int, List[dict]]:
        return self._predictions

    @property
    def categories(self) -> Dict[int, str]:
        return self._categories

    @categories.setter
    def categories(self, value: Dict[int, str]):
        self._categories = value

    @property
    def image_files(self) -> Dict[int, str]:
        return self._image_files

    @image_files.setter
    def image_files(self, value: Dict[int, str]):
        self._image_files = value

    def load(self, predictions_path: str) -> bool:
        """Load predictions from a COCO-format JSON file.

        Accepted formats:

        - COCO results: ``[{image_id, category_id, bbox, score}, ...]``
        - Full COCO with images/annotations:
          ``{images: [...], annotations: [...], categories: [...]}``

        Returns
        -------
        bool
            True if the file was loaded successfully.
        """
        p = Path(predictions_path)
        if not p.is_file():
            return False

        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._name = p.stem
        self._predictions.clear()

        if isinstance(raw, list):
            self._parse_results_list(raw)
        elif isinstance(raw, dict):
            self._parse_coco_dict(raw)

        return True

    def render(self, pixmap: QPixmap, image_id: int, color: QColor) -> QPixmap:
        """Render predictions for *image_id* onto *pixmap*.

        Draws bounding boxes and a model-name badge with detection count.

        Parameters
        ----------
        pixmap:
            Base image (not modified).
        image_id:
            COCO image id to look up predictions for.
        color:
            Box and badge color.

        Returns
        -------
        QPixmap
            New pixmap with overlays drawn.
        """
        preds = self._predictions.get(image_id, [])

        result = BBoxPainter.draw_boxes(
            pixmap, preds, color, categories=self._categories,
        )

        # Model name badge
        next_y = BBoxPainter.draw_badge(result, self._name, color)

        # Detection count badge
        BBoxPainter.draw_badge(
            result, f"{len(preds)} detections",
            QColor(0, 0, 0), y_offset=next_y,
        )

        return result

    def summarize(self) -> dict:
        """Compute summary statistics from loaded predictions.

        Returns
        -------
        dict
            Metrics dict with keys like ``Total Detections``,
            ``Mean Score``, etc. Empty dict if no predictions loaded.
        """
        if not self._predictions:
            return {}

        total_dets = sum(len(v) for v in self._predictions.values())
        n_images = len(self._predictions)
        scores = [
            d["score"]
            for dets in self._predictions.values()
            for d in dets
            if d.get("score") is not None
        ]

        metrics: dict = {
            "Total Detections": total_dets,
            "Images with Dets": n_images,
            "Avg Dets/Image": round(total_dets / max(n_images, 1), 2),
        }
        if scores:
            metrics["Mean Score"] = round(sum(scores) / len(scores), 4)
            metrics["Min Score"] = round(min(scores), 4)
            metrics["Max Score"] = round(max(scores), 4)

        return metrics

    def get_all_image_ids(self) -> set:
        """Return the set of all image ids that have predictions."""
        return set(self._predictions.keys())

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_results_list(self, raw: list):
        """Parse COCO results format: ``[{image_id, category_id, bbox, score}]``."""
        for entry in raw:
            img_id = entry.get("image_id")
            if img_id is not None:
                self._predictions.setdefault(img_id, []).append(entry)

    def _parse_coco_dict(self, raw: dict):
        """Parse full COCO format with images, annotations, categories."""
        for ann in raw.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id is not None:
                self._predictions.setdefault(img_id, []).append(ann)

        for cat in raw.get("categories", []):
            self._categories[cat["id"]] = cat.get("name", f"cat_{cat['id']}")

        for img in raw.get("images", []):
            self._image_files[img["id"]] = img.get("file_name", "")
