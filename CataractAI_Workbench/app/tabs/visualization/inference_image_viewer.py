"""Image viewer widget with bounding-box overlay for inference results.

Displays a single image with optional ground-truth (green dashed) and
prediction (red solid) bounding boxes, labels, confidence scores, and
a legend badge.
"""

from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, QSizePolicy,
)
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QRect

# Drawing colours
_GT_COLOR = QColor(107, 203, 119)       # green - ground truth
_PRED_COLOR = QColor(255, 107, 107)     # red - predictions

_FONT_FAMILY = "Consolas"
_FONT_SIZE = 9


class InferenceImageViewer(QWidget):
    """Scrollable image viewer that draws bounding-box overlays."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._categories: Dict[int, str] = {}
        self._init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setStyleSheet(
            "QScrollArea { background-color: #1E1E1E; border: 1px solid #3C3C3C; }"
        )

        self._image_label = QLabel(
            "Load a model and dataset, then click Run Inference"
        )
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self._image_label.setStyleSheet("color: #7F7F7F; font-size: 14px;")
        self._scroll_area.setWidget(self._image_label)

        layout.addWidget(self._scroll_area)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_categories(self, categories: Dict[int, str]) -> None:
        self._categories = categories

    def show_placeholder(self, text: str) -> None:
        self._image_label.setText(text)

    def display(
        self,
        image_path: str,
        predictions: List[dict],
        gt_boxes: List[dict],
    ) -> None:
        """Load *image_path*, draw overlays, and scale to fit."""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            from pathlib import Path
            self._image_label.setText(f"Cannot load: {Path(image_path).name}")
            return

        drawn = self._draw_boxes(pixmap, predictions, gt_boxes)

        available = self._scroll_area.size()
        scaled = drawn.scaled(
            available.width() - 20,
            available.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_boxes(
        self,
        pixmap: QPixmap,
        predictions: List[dict],
        gt_boxes: List[dict],
    ) -> QPixmap:
        result = QPixmap(pixmap)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont(_FONT_FAMILY, _FONT_SIZE)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()

        self._draw_gt_boxes(painter, fm, gt_boxes)
        self._draw_pred_boxes(painter, fm, predictions, has_gt=bool(gt_boxes))
        self._draw_legend(painter, fm, predictions, gt_boxes)

        painter.end()
        return result

    def _draw_gt_boxes(self, painter: QPainter, fm, gt_boxes: List[dict]) -> None:
        for gt in gt_boxes:
            bbox = gt.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
            rect = QRect(x1, y1, x2 - x1, y2 - y1)

            pen = QPen(_GT_COLOR, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(rect)

            cat_id = gt.get("category_id", 0)
            cat_name = self._categories.get(cat_id, f"id:{cat_id}")
            label_text = f"GT: {cat_name}"

            text_w = fm.horizontalAdvance(label_text) + 6
            text_h = fm.height() + 4
            bg = QColor(_GT_COLOR)
            bg.setAlpha(160)
            label_y = max(0, y1 - text_h)
            painter.fillRect(x1, label_y, text_w, text_h, bg)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(x1 + 3, label_y + fm.ascent() + 2, label_text)

    def _draw_pred_boxes(
        self,
        painter: QPainter,
        fm,
        predictions: List[dict],
        has_gt: bool,
    ) -> None:
        for pred in predictions:
            bbox = pred.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
            rect = QRect(x1, y1, x2 - x1, y2 - y1)

            pen = QPen(_PRED_COLOR, 2)
            painter.setPen(pen)
            painter.drawRect(rect)

            score = pred.get("score", 0)
            label_id = pred.get("label", 0)
            cat_name = self._categories.get(label_id, f"id:{label_id}")
            label_text = f"{cat_name} {score:.2f}"

            text_w = fm.horizontalAdvance(label_text) + 6
            text_h = fm.height() + 4
            bg = QColor(_PRED_COLOR)
            bg.setAlpha(180)
            if has_gt:
                label_y = y2 + 2  # below box to avoid overlap with GT
            else:
                label_y = max(0, y1 - text_h)
            painter.fillRect(x1, label_y, text_w, text_h, bg)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(x1 + 3, label_y + fm.ascent() + 2, label_text)

    def _draw_legend(
        self,
        painter: QPainter,
        fm,
        predictions: List[dict],
        gt_boxes: List[dict],
    ) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        legend_y = 4
        badge_h = fm.height() + 8

        if gt_boxes:
            gt_text = f"GT: {len(gt_boxes)} annotations"
            gt_w = fm.horizontalAdvance(gt_text) + 12
            bg_gt = QColor(_GT_COLOR)
            bg_gt.setAlpha(200)
            painter.fillRect(4, legend_y, gt_w, badge_h, bg_gt)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(10, legend_y + fm.ascent() + 4, gt_text)
            legend_y += badge_h + 2
            painter.setPen(Qt.PenStyle.NoPen)

        if predictions:
            pred_text = f"Predictions: {len(predictions)} detections"
            pred_w = fm.horizontalAdvance(pred_text) + 12
            bg_pred = QColor(_PRED_COLOR)
            bg_pred.setAlpha(200)
            painter.fillRect(4, legend_y, pred_w, badge_h, bg_pred)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(10, legend_y + fm.ascent() + 4, pred_text)
