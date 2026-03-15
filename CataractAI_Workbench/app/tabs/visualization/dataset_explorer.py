"""Dataset explorer with COCO annotation overlay and filtering."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QGroupBox, QScrollArea, QSizePolicy, QMessageBox,
)
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtCore import Qt

from ...widgets.bbox_painter import BBoxPainter, CATEGORY_COLORS
from ...widgets.file_picker import FilePicker


class DatasetExplorer(QWidget):
    """Browse datasets with COCO-format annotations and bounding-box overlays."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._images: List[dict] = []
        self._annotations: List[dict] = []
        self._categories: Dict[int, str] = {}
        self._cat_colors: Dict[int, QColor] = {}
        self._ann_by_image: Dict[int, List[dict]] = {}
        self._image_by_id: Dict[int, dict] = {}
        self._images_dir: str = ""
        self._current_image_id: Optional[int] = None
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        root.addLayout(self._build_toolbar())

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(self._build_main_area())
        v_splitter.addWidget(self._build_annotation_table())
        v_splitter.setStretchFactor(0, 4)
        v_splitter.setStretchFactor(1, 1)

        root.addWidget(v_splitter)

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Images:"))
        self._dir_picker = FilePicker(label="Browse...", mode="dir")
        toolbar.addWidget(self._dir_picker)
        toolbar.addWidget(QLabel("Annotations:"))
        self._ann_picker = FilePicker(
            label="Browse...", mode="file",
            filter_str="JSON Files (*.json);;All Files (*)",
        )
        toolbar.addWidget(self._ann_picker)
        self._btn_load = QPushButton("Load")
        self._btn_load.setFixedWidth(70)
        self._btn_load.clicked.connect(self._on_load)
        toolbar.addWidget(self._btn_load)
        toolbar.addWidget(QLabel("Category:"))
        self._cat_combo = QComboBox()
        self._cat_combo.setFixedWidth(160)
        self._cat_combo.addItem("All")
        self._cat_combo.currentTextChanged.connect(self._on_category_filter)
        toolbar.addWidget(self._cat_combo)
        toolbar.addStretch()
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #7F7F7F;")
        toolbar.addWidget(self._info_label)
        return toolbar

    def _build_main_area(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Filter images...")
        self._search_edit.textChanged.connect(self._on_search_changed)
        left_layout.addWidget(self._search_edit)
        self._image_list = QListWidget()
        self._image_list.setStyleSheet(
            "QListWidget { background-color: #1E1E1E; color: #D4D4D4; "
            "border: 1px solid #3C3C3C; }"
            "QListWidget::item:selected { background-color: #264F78; }"
        )
        self._image_list.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self._image_list)
        splitter.addWidget(left_widget)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll_area.setStyleSheet(
            "QScrollArea { background-color: #1E1E1E; border: 1px solid #3C3C3C; }"
        )

        self._image_label = QLabel("Load a dataset to begin")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self._image_label.setStyleSheet("color: #7F7F7F; font-size: 14px;")
        self._scroll_area.setWidget(self._image_label)
        center_layout.addWidget(self._scroll_area)

        center_layout.addLayout(self._build_navigation())
        splitter.addWidget(center_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        return splitter

    def _build_navigation(self) -> QHBoxLayout:
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("< Previous")
        self._btn_prev.clicked.connect(self._on_prev)
        nav.addWidget(self._btn_prev)
        self._pos_label = QLabel("0 / 0")
        self._pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self._pos_label)
        self._btn_next = QPushButton("Next >")
        self._btn_next.clicked.connect(self._on_next)
        nav.addWidget(self._btn_next)
        return nav

    def _build_annotation_table(self) -> QGroupBox:
        table_group = QGroupBox("Annotations")
        table_lay = QVBoxLayout(table_group)
        table_lay.setContentsMargins(2, 2, 2, 2)

        self._ann_table = QTableWidget()
        self._ann_table.setColumnCount(6)
        self._ann_table.setHorizontalHeaderLabels([
            "ID", "Category", "BBox (x,y,w,h)", "Area", "Score", "IsCrowd",
        ])
        self._ann_table.setAlternatingRowColors(True)
        self._ann_table.setStyleSheet(
            "QTableWidget { background-color: #1E1E1E; color: #D4D4D4; "
            "gridline-color: #3C3C3C; alternate-background-color: #252526; }"
            "QHeaderView::section { background-color: #2D2D30; color: #D4D4D4; "
            "padding: 4px; border: 1px solid #3C3C3C; }"
        )
        self._ann_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        table_lay.addWidget(self._ann_table)

        return table_group

    def load_dataset(self, images_dir: str, annotations_path: str):
        """Load a COCO-format JSON and index images."""
        ann_path = Path(annotations_path)
        img_dir = Path(images_dir)

        if not ann_path.is_file():
            QMessageBox.warning(self, "Error", f"Annotations file not found:\n{ann_path}")
            return
        if not img_dir.is_dir():
            QMessageBox.warning(self, "Error", f"Images directory not found:\n{img_dir}")
            return

        self._images_dir = str(img_dir)

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self._images = coco.get("images", [])
        self._annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        self._categories.clear()
        self._cat_colors.clear()
        for i, cat in enumerate(categories):
            cid = cat["id"]
            self._categories[cid] = cat.get("name", f"cat_{cid}")
            self._cat_colors[cid] = CATEGORY_COLORS[i % len(CATEGORY_COLORS)]

        self._ann_by_image.clear()
        for ann in self._annotations:
            img_id = ann["image_id"]
            self._ann_by_image.setdefault(img_id, []).append(ann)

        self._image_by_id.clear()
        for img in self._images:
            self._image_by_id[img["id"]] = img

        self._cat_combo.blockSignals(True)
        self._cat_combo.clear()
        self._cat_combo.addItem("All")
        for cid in sorted(self._categories):
            self._cat_combo.addItem(self._categories[cid])
        self._cat_combo.blockSignals(False)

        self._populate_image_list()

        self._info_label.setText(
            f"{len(self._images)} images, {len(self._annotations)} annotations, "
            f"{len(self._categories)} categories"
        )

    def show_image(self, image_path: str):
        """Display an image with overlaid bounding boxes."""
        p = Path(image_path)
        if not p.is_file():
            self._image_label.setText(f"File not found:\n{p.name}")
            return

        pixmap = QPixmap(str(p))
        if pixmap.isNull():
            self._image_label.setText(f"Cannot load image:\n{p.name}")
            return

        # Find image entry by filename
        fname = p.name
        img_entry = None
        for img in self._images:
            if img.get("file_name", "") == fname:
                img_entry = img
                break

        if img_entry:
            self._current_image_id = img_entry["id"]
            anns = self._ann_by_image.get(img_entry["id"], [])
            pixmap = BBoxPainter.draw_boxes(
                pixmap, anns, QColor(200, 200, 200),
                categories=self._categories,
                color_by_category=self._cat_colors,
                font_size=10,
            )
            self._fill_annotation_table(anns)
        else:
            self._current_image_id = None
            self._ann_table.setRowCount(0)

        available = self._scroll_area.size()
        scaled = pixmap.scaled(
            available.width() - 20, available.height() - 20,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def filter_by_category(self, category: str):
        """Filter the image list to only show images containing the given category."""
        self._cat_combo.setCurrentText(category)

    def _fill_annotation_table(self, annotations: List[dict]):
        """Populate the annotation details table."""
        self._ann_table.setRowCount(len(annotations))
        for row, ann in enumerate(annotations):
            cat_id = ann.get("category_id", -1)
            cat_name = self._categories.get(cat_id, str(cat_id))

            bbox = ann.get("bbox", [])
            bbox_str = ", ".join(f"{v:.1f}" for v in bbox) if bbox else ""

            area = ann.get("area", "")
            score = ann.get("score")
            is_crowd = ann.get("iscrowd", 0)

            items = [
                str(ann.get("id", "")),
                cat_name,
                bbox_str,
                f"{area:.1f}" if isinstance(area, (int, float)) else str(area),
                f"{score:.4f}" if score is not None else "",
                str(is_crowd),
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._ann_table.setItem(row, col, item)

    def _populate_image_list(self, category_filter: Optional[str] = None,
                             text_filter: str = ""):
        """Rebuild the image list with optional filters."""
        self._image_list.blockSignals(True)
        self._image_list.clear()

        matching_ids = self._get_matching_image_ids(category_filter)

        for img in self._images:
            fname = img.get("file_name", "")

            if matching_ids is not None and img["id"] not in matching_ids:
                continue
            if text_filter and text_filter.lower() not in fname.lower():
                continue

            item = QListWidgetItem(fname)
            item.setData(Qt.ItemDataRole.UserRole, img["id"])
            self._image_list.addItem(item)

        self._image_list.blockSignals(False)

        total = self._image_list.count()
        self._pos_label.setText(f"0 / {total}")

        if total > 0:
            self._image_list.setCurrentRow(0)

    def _get_matching_image_ids(self, category_filter: Optional[str]) -> Optional[set]:
        """Return image ids matching the category filter, or None for no filter."""
        if not category_filter or category_filter == "All":
            return None

        cat_id = None
        for cid, cname in self._categories.items():
            if cname == category_filter:
                cat_id = cid
                break

        if cat_id is None:
            return set()

        return {
            ann["image_id"]
            for ann in self._annotations
            if ann.get("category_id") == cat_id
        }

    def _on_load(self):
        images_dir = self._dir_picker.path()
        ann_path = self._ann_picker.path()
        if images_dir and ann_path:
            self.load_dataset(images_dir, ann_path)

    def _on_category_filter(self, category: str):
        text_filter = self._search_edit.text()
        self._populate_image_list(category_filter=category, text_filter=text_filter)

    def _on_search_changed(self, text: str):
        category = self._cat_combo.currentText()
        self._populate_image_list(category_filter=category, text_filter=text)

    def _on_image_selected(self, row: int):
        if row < 0:
            return
        item = self._image_list.item(row)
        if item is None:
            return

        fname = item.text()
        img_path = Path(self._images_dir) / fname

        total = self._image_list.count()
        self._pos_label.setText(f"{row + 1} / {total}")

        self.show_image(str(img_path))

    def _on_prev(self):
        row = self._image_list.currentRow()
        if row > 0:
            self._image_list.setCurrentRow(row - 1)

    def _on_next(self):
        row = self._image_list.currentRow()
        if row < self._image_list.count() - 1:
            self._image_list.setCurrentRow(row + 1)
