"""Shared bounding-box drawing utilities for QPixmap overlays.

Used by dataset_explorer, model_comparison, and inference_tester to
eliminate duplicate QPainter drawing code.
"""

from typing import Dict, List, Optional

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap


# Default category color palette (cycled for >10 categories)
CATEGORY_COLORS = [
    QColor(78, 154, 255),   # blue
    QColor(255, 107, 107),  # red
    QColor(107, 203, 119),  # green
    QColor(255, 217, 61),   # yellow
    QColor(180, 130, 255),  # purple
    QColor(255, 165, 80),   # orange
    QColor(100, 220, 220),  # cyan
    QColor(220, 150, 180),  # pink
    QColor(160, 200, 90),   # lime
    QColor(200, 120, 100),  # brown
]


class BBoxPainter:
    """Draws bounding boxes with labels on a QPixmap.

    This is a stateless utility class. All methods are static and
    produce a new QPixmap without mutating the input.
    """

    @staticmethod
    def draw_boxes(
        pixmap: QPixmap,
        boxes: List[dict],
        color: QColor,
        categories: Optional[Dict[int, str]] = None,
        label_key: str = "category_id",
        score_key: str = "score",
        line_width: int = 2,
        style: Qt.PenStyle = Qt.PenStyle.SolidLine,
        font_size: int = 9,
        color_by_category: Optional[Dict[int, QColor]] = None,
        label_position: str = "above",
    ) -> QPixmap:
        """Draw bounding boxes on a copy of *pixmap*.

        Parameters
        ----------
        pixmap:
            Source image. Not modified.
        boxes:
            List of dicts each containing ``bbox`` as ``[x, y, w, h]``
            (COCO format).
        color:
            Default box/label color when *color_by_category* is not used.
        categories:
            Mapping from category id to human-readable name.
        label_key:
            Dict key for the category identifier in each box dict.
        score_key:
            Dict key for the confidence score (may be absent).
        line_width:
            Pen width for box outlines.
        style:
            Qt pen style (solid, dashed, etc.).
        font_size:
            Label font size in points.
        color_by_category:
            If provided, each box is colored according to its category id.
        label_position:
            ``"above"`` (default) or ``"below"`` the box.

        Returns
        -------
        QPixmap
            A new pixmap with boxes drawn.
        """
        if categories is None:
            categories = {}

        result = QPixmap(pixmap)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Consolas", font_size)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()

        for box in boxes:
            bbox = box.get("bbox", [])
            if len(bbox) < 4:
                continue

            x, y, w, h = [int(round(v)) for v in bbox[:4]]
            rect = QRect(x, y, w, h)

            cat_id = box.get(label_key, -1)
            box_color = (
                color_by_category.get(cat_id, color)
                if color_by_category
                else color
            )

            pen = QPen(box_color, line_width, style)
            painter.setPen(pen)
            painter.drawRect(rect)

            cat_name = categories.get(cat_id, f"id:{cat_id}")
            score = box.get(score_key)
            label_text = (
                f"{cat_name} {score:.2f}" if score is not None else cat_name
            )

            text_w = fm.horizontalAdvance(label_text) + 6
            text_h = fm.height() + 4

            bg = QColor(box_color)
            bg.setAlpha(180)

            if label_position == "below":
                label_y = y + h + 2
            else:
                label_y = max(0, y - text_h)

            painter.fillRect(x, label_y, text_w, text_h, bg)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(x + 3, label_y + fm.ascent() + 2, label_text)

        painter.end()
        return result

    @staticmethod
    def draw_badge(
        pixmap: QPixmap,
        text: str,
        color: QColor,
        y_offset: int = 4,
        x_offset: int = 4,
        font_size: int = 9,
    ) -> int:
        """Draw a text badge (e.g. model name or count) on *pixmap*.

        Draws directly onto *pixmap* (mutating).

        Parameters
        ----------
        pixmap:
            Target pixmap to draw onto.
        text:
            Badge text content.
        color:
            Background color for the badge.
        y_offset:
            Vertical position of the badge top edge.
        x_offset:
            Horizontal position of the badge left edge.
        font_size:
            Font size in points.

        Returns
        -------
        int
            The y position just below this badge (for stacking badges).
        """
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Consolas", font_size)
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()

        badge_w = fm.horizontalAdvance(text) + 12
        badge_h = fm.height() + 8

        bg = QColor(color)
        bg.setAlpha(200)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.fillRect(x_offset, y_offset, badge_w, badge_h, bg)

        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(x_offset + 6, y_offset + fm.ascent() + 4, text)

        painter.end()
        return y_offset + badge_h + 2
