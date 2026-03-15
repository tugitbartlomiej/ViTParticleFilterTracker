"""Side-by-side model comparison viewer for pre-computed predictions."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QSizePolicy, QScrollArea, QMessageBox,
)
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtCore import Qt

from ...widgets.file_picker import FilePicker
from .prediction_renderer import PredictionRenderer

# Distinct colors for Model 1 and Model 2
_MODEL1_COLOR = QColor(78, 154, 255)   # blue
_MODEL2_COLOR = QColor(255, 107, 107)  # red


class ModelComparison(QWidget):
    """Side-by-side viewer for comparing two models' predictions."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._renderer1 = PredictionRenderer()
        self._renderer2 = PredictionRenderer()
        self._renderer1._name = "Model 1"
        self._renderer2._name = "Model 2"
        self._images_dir: str = ""
        self._image_ids: List[int] = []
        self._current_idx: int = -1
        self._model1_metrics: dict = {}
        self._model2_metrics: dict = {}
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        root.addLayout(self._build_selectors())
        root.addLayout(self._build_load_row())

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(self._build_image_panels())

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self._build_navigation())
        bottom_layout.addWidget(self._build_metrics_table())

        v_splitter.addWidget(bottom)
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)

        root.addWidget(v_splitter)

    def _build_pred_group(self, title: str) -> tuple:
        """Create a model prediction picker group box. Returns (group, picker)."""
        group = QGroupBox(title)
        lay = QHBoxLayout(group)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(QLabel("Predictions:"))
        picker = FilePicker(
            label="Browse...", mode="file",
            filter_str="JSON Files (*.json);;All Files (*)",
        )
        lay.addWidget(picker)
        return group, picker

    def _build_selectors(self) -> QHBoxLayout:
        selectors = QHBoxLayout()
        g1, self._pred1_picker = self._build_pred_group("Model 1")
        selectors.addWidget(g1)
        g2, self._pred2_picker = self._build_pred_group("Model 2")
        selectors.addWidget(g2)
        return selectors

    def _build_load_row(self) -> QHBoxLayout:
        load_row = QHBoxLayout()
        load_row.addWidget(QLabel("Images dir:"))
        self._images_picker = FilePicker(label="Browse...", mode="dir")
        load_row.addWidget(self._images_picker)

        load_row.addWidget(QLabel("Annotations (optional):"))
        self._ann_picker = FilePicker(
            label="Browse...", mode="file",
            filter_str="JSON Files (*.json);;All Files (*)",
        )
        load_row.addWidget(self._ann_picker)

        self._btn_load = QPushButton("Load && Compare")
        self._btn_load.setFixedWidth(130)
        self._btn_load.clicked.connect(self._on_load)
        load_row.addWidget(self._btn_load)

        return load_row

    def _build_scroll_panel(self, title: str, placeholder: str) -> tuple:
        """Create a group box with a scroll area and image label."""
        group = QGroupBox(title)
        lay = QVBoxLayout(group)
        lay.setContentsMargins(2, 2, 2, 2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet(
            "QScrollArea { background-color: #1E1E1E; border: 1px solid #3C3C3C; }"
        )

        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        label.setStyleSheet("color: #7F7F7F;")
        scroll.setWidget(label)
        lay.addWidget(scroll)

        return group, scroll, label

    def _build_image_panels(self) -> QWidget:
        panels = QWidget()
        panels_layout = QHBoxLayout(panels)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(4)

        p1, self._scroll1, self._img_label1 = self._build_scroll_panel(
            "Model 1", "No predictions loaded",
        )
        panels_layout.addWidget(p1)

        p2, self._scroll2, self._img_label2 = self._build_scroll_panel(
            "Model 2", "No predictions loaded",
        )
        panels_layout.addWidget(p2)

        return panels

    def _build_navigation(self) -> QWidget:
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 4, 0, 4)

        self._btn_prev = QPushButton("< Previous")
        self._btn_prev.clicked.connect(self._on_prev)
        nav_layout.addWidget(self._btn_prev)

        self._image_combo = QComboBox()
        self._image_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self._image_combo.currentIndexChanged.connect(self._on_image_combo_changed)
        nav_layout.addWidget(self._image_combo)

        self._pos_label = QLabel("0 / 0")
        self._pos_label.setFixedWidth(80)
        self._pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._pos_label)

        self._btn_next = QPushButton("Next >")
        self._btn_next.clicked.connect(self._on_next)
        nav_layout.addWidget(self._btn_next)

        return nav_widget

    def _build_metrics_table(self) -> QGroupBox:
        metrics_group = QGroupBox("Metrics Comparison")
        metrics_lay = QVBoxLayout(metrics_group)
        metrics_lay.setContentsMargins(2, 2, 2, 2)

        self._metrics_table = QTableWidget()
        self._metrics_table.setAlternatingRowColors(True)
        self._metrics_table.setStyleSheet(
            "QTableWidget { background-color: #1E1E1E; color: #D4D4D4; "
            "gridline-color: #3C3C3C; alternate-background-color: #252526; }"
            "QHeaderView::section { background-color: #2D2D30; color: #D4D4D4; "
            "padding: 4px; border: 1px solid #3C3C3C; }"
        )
        self._metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        metrics_lay.addWidget(self._metrics_table)

        return metrics_group

    def load_predictions(self, model_name: str, predictions_path: str):
        """Load COCO-format predictions for ``"model1"`` or ``"model2"``."""
        renderer = self._renderer1 if model_name == "model1" else self._renderer2
        renderer.load(predictions_path)

    def show_comparison(self, image_id: int):
        """Show the same image with both models' predictions overlaid."""
        all_files = self._merged_image_files()
        fname = all_files.get(image_id, "")
        if not fname:
            return

        img_path = Path(self._images_dir) / fname
        if not img_path.is_file():
            self._img_label1.setText(f"Not found: {fname}")
            self._img_label2.setText(f"Not found: {fname}")
            return

        base_pixmap = QPixmap(str(img_path))
        if base_pixmap.isNull():
            self._img_label1.setText(f"Cannot load: {fname}")
            self._img_label2.setText(f"Cannot load: {fname}")
            return

        self._render_panel(self._renderer1, base_pixmap, image_id,
                           _MODEL1_COLOR, self._scroll1, self._img_label1)
        self._render_panel(self._renderer2, base_pixmap, image_id,
                           _MODEL2_COLOR, self._scroll2, self._img_label2)

    def update_metrics(self, model1_metrics: dict, model2_metrics: dict):
        """Update the metrics comparison table."""
        self._model1_metrics = model1_metrics
        self._model2_metrics = model2_metrics
        self._rebuild_metrics_table()

    def _merged_image_files(self) -> Dict[int, str]:
        return {**self._renderer1.image_files, **self._renderer2.image_files}

    @staticmethod
    def _render_panel(renderer, base_pixmap, image_id, color, scroll, label):
        pm = renderer.render(QPixmap(base_pixmap), image_id, color)
        scaled = pm.scaled(
            scroll.width() - 20, scroll.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def _rebuild_metrics_table(self):
        """Populate the metrics comparison table."""
        all_keys = sorted(
            set(list(self._model1_metrics.keys()) + list(self._model2_metrics.keys()))
        )

        if not all_keys:
            self._metrics_table.setRowCount(0)
            self._metrics_table.setColumnCount(0)
            return

        self._metrics_table.setColumnCount(3)
        self._metrics_table.setHorizontalHeaderLabels([
            "Metric", self._renderer1.name, self._renderer2.name,
        ])
        self._metrics_table.setRowCount(len(all_keys))

        for row, key in enumerate(all_keys):
            v1 = self._model1_metrics.get(key)
            v2 = self._model2_metrics.get(key)
            self._set_metric_row(row, key, v1, v2)

    def _set_metric_row(self, row: int, key: str, v1, v2):
        """Fill a single row of the metrics table with value highlighting."""
        items = [
            key,
            f"{v1:.4f}" if isinstance(v1, (int, float)) else str(v1 or ""),
            f"{v2:.4f}" if isinstance(v2, (int, float)) else str(v2 or ""),
        ]

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            if col > 0 and v1 is not None and v2 is not None:
                if self._is_better_value(key, col, v1, v2):
                    item.setForeground(QColor(107, 203, 119))

            self._metrics_table.setItem(row, col, item)

    @staticmethod
    def _is_better_value(key: str, col: int, v1, v2) -> bool:
        """Check whether the value in *col* is the better of v1/v2."""
        try:
            fv1, fv2 = float(v1), float(v2)
        except (ValueError, TypeError):
            return False

        is_loss = "loss" in key.lower()
        if is_loss:
            return (col == 1 and fv1 < fv2) or (col == 2 and fv2 < fv1)
        return (col == 1 and fv1 > fv2) or (col == 2 and fv2 > fv1)

    def _on_load(self):
        """Load predictions and build image list."""
        images_dir = self._images_picker.path()
        if not images_dir:
            QMessageBox.warning(self, "Error", "Please select an images directory.")
            return
        self._images_dir = images_dir

        self._load_coco_annotations(self._ann_picker.path())
        self._load_pred_files()
        self._discover_images_if_needed()
        self._build_image_list()

        m1, m2 = self._renderer1.summarize(), self._renderer2.summarize()
        if m1 or m2:
            self.update_metrics(m1, m2)
        if self._image_ids:
            self._current_idx = 0
            self._show_current()

    def _load_coco_annotations(self, ann_path: str):
        """Load shared annotations/categories from a COCO file."""
        if not ann_path or not Path(ann_path).is_file():
            return
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        shared_files = {img["id"]: img.get("file_name", "") for img in coco.get("images", [])}
        shared_cats = {cat["id"]: cat.get("name", f"cat_{cat['id']}") for cat in coco.get("categories", [])}
        for renderer in (self._renderer1, self._renderer2):
            renderer.image_files.update(shared_files)
            renderer.categories.update(shared_cats)

    def _load_pred_files(self):
        """Load prediction files for both models."""
        for name, picker in [("model1", self._pred1_picker), ("model2", self._pred2_picker)]:
            path = picker.path()
            if path and Path(path).is_file():
                self.load_predictions(name, path)

    def _discover_images_if_needed(self):
        """Scan images directory when no filenames are known."""
        if self._merged_image_files():
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        for i, fp in enumerate(sorted(Path(self._images_dir).iterdir())):
            if fp.suffix.lower() in exts:
                self._renderer1.image_files[i] = fp.name
                self._renderer2.image_files[i] = fp.name

    def _build_image_list(self):
        """Build sorted image id list and populate the combo box."""
        all_ids = set(self._renderer1.image_files.keys())
        all_ids.update(self._renderer2.image_files.keys())
        all_ids.update(self._renderer1.get_all_image_ids())
        all_ids.update(self._renderer2.get_all_image_ids())
        self._image_ids = sorted(all_ids)

        all_files = self._merged_image_files()
        self._image_combo.blockSignals(True)
        self._image_combo.clear()
        for img_id in self._image_ids:
            self._image_combo.addItem(all_files.get(img_id, f"image_{img_id}"))
        self._image_combo.blockSignals(False)

    def _on_image_combo_changed(self, index: int):
        if 0 <= index < len(self._image_ids):
            self._current_idx = index
            self._show_current()

    def _on_prev(self):
        self._navigate_to(self._current_idx - 1)

    def _on_next(self):
        self._navigate_to(self._current_idx + 1)

    def _navigate_to(self, index: int):
        """Navigate to the image at *index* if within bounds."""
        if index < 0 or index >= len(self._image_ids):
            return
        self._current_idx = index
        self._image_combo.blockSignals(True)
        self._image_combo.setCurrentIndex(index)
        self._image_combo.blockSignals(False)
        self._show_current()

    def _show_current(self):
        if self._current_idx < 0 or self._current_idx >= len(self._image_ids):
            return
        self._pos_label.setText(f"{self._current_idx + 1} / {len(self._image_ids)}")
        self.show_comparison(self._image_ids[self._current_idx])
