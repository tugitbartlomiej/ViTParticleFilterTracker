"""Statistics panel and detection details table for inference results.

Two widgets:
- ``StatsPanel``: running totals (images tested, TP/FP/FN, precision/recall/F1).
- ``DetectionDetailsTable``: per-image GT and prediction rows.
- ``ConfusionMatrixPanel``: batch-level confusion matrix display.
"""

from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal

_GT_COLOR = QColor(107, 203, 119)
_PRED_COLOR = QColor(255, 107, 107)


# =====================================================================
# Running statistics panel
# =====================================================================

class StatsPanel(QGroupBox):
    """Displays running totals accumulated during single-image inference."""

    reset_requested = pyqtSignal()

    _STAT_NAMES = [
        "Images tested", "Avg confidence",
        "True Positives", "False Positives", "False Negatives",
        "Precision", "Recall", "F1 Score",
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Statistics", parent)
        self._labels: Dict[str, QLabel] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(2)

        for name in self._STAT_NAMES:
            row = QHBoxLayout()
            lbl_name = QLabel(f"{name}:")
            lbl_name.setFixedWidth(110)
            lbl_name.setStyleSheet("color: #CCCCCC;")
            lbl_val = QLabel("--")
            lbl_val.setStyleSheet("color: #D4D4D4; font-weight: bold;")
            row.addWidget(lbl_name)
            row.addWidget(lbl_val)
            row.addStretch()
            lay.addLayout(row)
            self._labels[name] = lbl_val

        btn = QPushButton("Reset Statistics")
        btn.clicked.connect(self.reset_requested.emit)
        lay.addWidget(btn)

    # -- public --

    def update_stats(
        self,
        tested_count: int,
        total_tp: int,
        total_fp: int,
        total_fn: int,
        all_scores: List[float],
    ) -> None:
        """Recompute and display running statistics."""
        self._labels["Images tested"].setText(str(tested_count))

        avg_conf = sum(all_scores) / len(all_scores) if all_scores else 0.0
        self._labels["Avg confidence"].setText(f"{avg_conf:.4f}")
        self._labels["True Positives"].setText(str(total_tp))
        self._labels["False Positives"].setText(str(total_fp))
        self._labels["False Negatives"].setText(str(total_fn))

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        self._labels["Precision"].setText(f"{prec:.3f}")
        self._labels["Recall"].setText(f"{rec:.3f}")
        self._labels["F1 Score"].setText(f"{f1:.3f}")

    def reset(self) -> None:
        for lbl in self._labels.values():
            lbl.setText("--")


# =====================================================================
# Detection details table
# =====================================================================

class DetectionDetailsTable(QGroupBox):
    """Table showing per-detection GT and prediction entries."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Detection Details", parent)
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "#", "Type", "Class/Label", "Confidence", "BBox [x1,y1,x2,y2]",
        ])
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background-color: #1E1E1E; color: #D4D4D4; "
            "gridline-color: #3C3C3C; alternate-background-color: #252526; }"
            "QHeaderView::section { background-color: #2D2D30; color: #D4D4D4; "
            "padding: 4px; border: 1px solid #3C3C3C; }"
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        lay.addWidget(self._table)

    def fill(
        self,
        predictions: List[dict],
        gt_boxes: List[dict],
        categories: Dict[int, str],
    ) -> None:
        """Populate table with GT and prediction rows."""
        total = len(gt_boxes) + len(predictions)
        self._table.setRowCount(total)
        row = 0

        for i, gt in enumerate(gt_boxes):
            bbox = gt.get("bbox", [])
            cat_id = gt.get("category_id", 0)
            cat_name = categories.get(cat_id, f"id:{cat_id}")
            bbox_str = ", ".join(f"{v:.0f}" for v in bbox) if bbox else ""
            cells = [str(i + 1), "GT", cat_name, "--", f"[{bbox_str}]"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 1:
                    item.setForeground(_GT_COLOR)
                self._table.setItem(row, col, item)
            row += 1

        for i, pred in enumerate(predictions):
            bbox = pred.get("bbox", [])
            score = pred.get("score", 0)
            label_id = pred.get("label", 0)
            cat_name = categories.get(label_id, f"id:{label_id}")
            bbox_str = ", ".join(f"{v:.0f}" for v in bbox) if bbox else ""
            cells = [str(i + 1), "Pred", cat_name, f"{score:.4f}", f"[{bbox_str}]"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 1:
                    item.setForeground(_PRED_COLOR)
                self._table.setItem(row, col, item)
            row += 1


# =====================================================================
# Confusion matrix panel
# =====================================================================

class ConfusionMatrixPanel(QGroupBox):
    """Batch-level confusion matrix summary."""

    _ITEMS = [
        ("TP", "True Positives"),
        ("FP", "False Positives"),
        ("FN", "False Negatives"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("F1", "F1 Score"),
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Confusion Matrix", parent)
        self._labels: Dict[str, QLabel] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        for key, display_name in self._ITEMS:
            row = QHBoxLayout()
            name_lbl = QLabel(f"{display_name}:")
            name_lbl.setFixedWidth(110)
            name_lbl.setStyleSheet("color: #CCCCCC;")
            val_lbl = QLabel("--")
            val_lbl.setStyleSheet("color: #D4D4D4; font-weight: bold;")
            row.addWidget(name_lbl)
            row.addWidget(val_lbl)
            row.addStretch()
            lay.addLayout(row)
            self._labels[key] = val_lbl

        lay.addStretch()

    def update_from_results(self, results: dict) -> None:
        self._labels["TP"].setText(str(results["total_tp"]))
        self._labels["FP"].setText(str(results["total_fp"]))
        self._labels["FN"].setText(str(results["total_fn"]))
        self._labels["Precision"].setText(f"{results['precision']:.3%}")
        self._labels["Recall"].setText(f"{results['recall']:.3%}")
        self._labels["F1"].setText(f"{results['f1']:.3%}")
