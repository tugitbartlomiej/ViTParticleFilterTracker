"""Dataset Analyzer widget -- UI for COCO annotation analysis and charts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...widgets.file_picker import FilePicker
from ...core.analysis_services import DatasetAnalysisService
from .theme_constants import (
    ACCENT_BLUE,
    ACCENT_GREEN,
    ACCENT_YELLOW,
    BG,
    BORDER,
    FG,
    GREEN_BUTTON_STYLE,
    GROUP_STYLE,
    TEXT_STYLE,
    make_canvas,
    style_axis,
)


class DatasetAnalyzerWidget(QWidget):
    """Analyze COCO dataset: composition, box distribution, spatial heatmap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._service = DatasetAnalysisService()
        self._coco_data: dict | None = None
        self._build_ui()

    # ---------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel(), stretch=1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(320)
        panel.setMinimumWidth(240)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        grp_path = QGroupBox("Dataset Path")
        grp_path.setStyleSheet(GROUP_STYLE)
        path_layout = QVBoxLayout(grp_path)

        path_layout.addWidget(QLabel("Images dir:"))
        self._img_picker = FilePicker(label="Browse...", mode="dir")
        path_layout.addWidget(self._img_picker)

        path_layout.addWidget(QLabel("Annotations (COCO JSON):"))
        self._ann_picker = FilePicker(
            label="Browse...", mode="file",
            filter_str="JSON (*.json);;All (*)",
        )
        path_layout.addWidget(self._ann_picker)

        self._btn_load = QPushButton("Load & Analyze")
        self._btn_load.setStyleSheet(GREEN_BUTTON_STYLE)
        self._btn_load.clicked.connect(self._load_and_analyze)
        path_layout.addWidget(self._btn_load)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItem("All")
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._filter_combo, stretch=1)
        path_layout.addLayout(filter_row)

        layout.addWidget(grp_path)

        # Recommendations
        grp_recs = QGroupBox("Recommendations")
        grp_recs.setStyleSheet(GROUP_STYLE)
        recs_layout = QVBoxLayout(grp_recs)
        self._recs_text = QTextEdit()
        self._recs_text.setReadOnly(True)
        self._recs_text.setFont(QFont("Consolas", 9))
        self._recs_text.setStyleSheet(TEXT_STYLE)
        recs_layout.addWidget(self._recs_text)
        layout.addWidget(grp_recs)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Composition analysis text
        grp_comp = QGroupBox("Composition Analysis")
        grp_comp.setStyleSheet(GROUP_STYLE)
        comp_layout = QVBoxLayout(grp_comp)

        self._comp_text = QTextEdit()
        self._comp_text.setReadOnly(True)
        self._comp_text.setFont(QFont("Consolas", 9))
        self._comp_text.setStyleSheet(TEXT_STYLE)
        self._comp_text.setMaximumHeight(160)
        comp_layout.addWidget(self._comp_text)

        layout.addWidget(grp_comp)

        # Charts: box size + aspect ratio side by side
        charts_row = QHBoxLayout()

        self._box_fig, self._box_canvas = make_canvas()
        self._box_canvas.setMinimumHeight(200)
        charts_row.addWidget(self._box_canvas)

        self._aspect_fig, self._aspect_canvas = make_canvas()
        self._aspect_canvas.setMinimumHeight(200)
        charts_row.addWidget(self._aspect_canvas)

        layout.addLayout(charts_row)

        # Spatial heatmap
        self._heatmap_fig, self._heatmap_canvas = make_canvas()
        self._heatmap_canvas.setMinimumHeight(220)
        layout.addWidget(self._heatmap_canvas)

        return panel

    # ---------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------

    def _load_and_analyze(self) -> None:
        ann_path = self._ann_picker.path()
        if not ann_path or not Path(ann_path).is_file():
            QMessageBox.warning(
                self, "No Annotations",
                "Please select a valid COCO annotations JSON file.",
            )
            return

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                self._coco_data = json.load(f)
        except Exception as exc:
            QMessageBox.warning(self, "Parse Error", f"Cannot parse annotations: {exc}")
            return

        # Populate category filter
        self._filter_combo.blockSignals(True)
        self._filter_combo.clear()
        self._filter_combo.addItem("All")
        for cat in self._coco_data.get("categories", []):
            self._filter_combo.addItem(cat.get("name", f"id={cat.get('id')}"))
        self._filter_combo.blockSignals(False)

        self._run_analysis()

    def _on_filter_changed(self, _text: str) -> None:
        if self._coco_data is not None:
            self._run_analysis()

    # ---------------------------------------------------------------
    # Analysis orchestration
    # ---------------------------------------------------------------

    def _run_analysis(self) -> None:
        if self._coco_data is None:
            return

        result = self._service.analyze({
            "coco_data": self._coco_data,
            "img_dir": self._img_picker.path(),
            "selected_category": self._filter_combo.currentText(),
        })

        self._comp_text.setPlainText(result["composition"])
        self._recs_text.setPlainText(result["recommendations"])

        bboxes = result["bboxes"]
        images = result["images"]
        self._draw_box_size_distribution(bboxes)
        self._draw_aspect_ratio_distribution(bboxes)
        self._draw_spatial_heatmap(bboxes, images)

    # ---------------------------------------------------------------
    # Chart drawing
    # ---------------------------------------------------------------

    def _draw_box_size_distribution(self, bboxes: np.ndarray) -> None:
        self._box_fig.clear()
        ax = self._box_fig.add_subplot(111)
        style_axis(ax, title="Box Size Distribution", xlabel="Area (px^2)", ylabel="Count")

        if len(bboxes) == 0:
            ax.text(0.5, 0.5, "No annotations", transform=ax.transAxes,
                    ha="center", va="center", color=FG, fontsize=12)
            self._box_fig.tight_layout()
            self._box_canvas.draw()
            return

        areas = bboxes[:, 2] * bboxes[:, 3]
        ax.hist(areas, bins=50, color=ACCENT_BLUE, alpha=0.8,
                edgecolor=BG, linewidth=0.5)

        mean_area = float(np.mean(areas))
        ax.axvline(mean_area, color=ACCENT_YELLOW, linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_area:.0f}")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=FG)

        self._box_fig.tight_layout()
        self._box_canvas.draw()

    def _draw_aspect_ratio_distribution(self, bboxes: np.ndarray) -> None:
        self._aspect_fig.clear()
        ax = self._aspect_fig.add_subplot(111)
        style_axis(ax, title="Aspect Ratio Distribution",
                   xlabel="Width / Height", ylabel="Count")

        if len(bboxes) == 0:
            ax.text(0.5, 0.5, "No annotations", transform=ax.transAxes,
                    ha="center", va="center", color=FG, fontsize=12)
            self._aspect_fig.tight_layout()
            self._aspect_canvas.draw()
            return

        heights = bboxes[:, 3].copy()
        heights[heights == 0] = 1
        ratios = bboxes[:, 2] / heights
        ratios_clipped = ratios[(ratios > 0.1) & (ratios < 10)]

        ax.hist(ratios_clipped, bins=50, color="#B39DDB", alpha=0.8,
                edgecolor=BG, linewidth=0.5)

        mean_ratio = float(np.mean(ratios_clipped)) if len(ratios_clipped) > 0 else 0
        ax.axvline(mean_ratio, color=ACCENT_YELLOW, linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_ratio:.2f}")
        ax.axvline(1.0, color=ACCENT_GREEN, linestyle=":", linewidth=1,
                   label="Square (1.0)")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=FG)

        self._aspect_fig.tight_layout()
        self._aspect_canvas.draw()

    def _draw_spatial_heatmap(self, bboxes: np.ndarray, images: list) -> None:
        self._heatmap_fig.clear()
        ax = self._heatmap_fig.add_subplot(111)
        style_axis(ax, title="Spatial Heatmap of Box Centers",
                   xlabel="X (normalized)", ylabel="Y (normalized)")

        if len(bboxes) == 0:
            ax.text(0.5, 0.5, "No annotations", transform=ax.transAxes,
                    ha="center", va="center", color=FG, fontsize=12)
            self._heatmap_fig.tight_layout()
            self._heatmap_canvas.draw()
            return

        if images:
            max_w = max((img.get("width", 1920) for img in images), default=1920)
            max_h = max((img.get("height", 1080) for img in images), default=1080)
        else:
            max_w, max_h = 1920, 1080

        cx = np.clip((bboxes[:, 0] + bboxes[:, 2] / 2) / max_w, 0, 1)
        cy = np.clip((bboxes[:, 1] + bboxes[:, 3] / 2) / max_h, 0, 1)

        heatmap, _xedges, _yedges = np.histogram2d(
            cx, cy, bins=30, range=[[0, 1], [0, 1]],
        )

        im = ax.imshow(
            heatmap.T, origin="lower", extent=[0, 1, 0, 1],
            cmap="hot", aspect="auto", interpolation="gaussian",
        )
        self._heatmap_fig.colorbar(im, ax=ax, label="Count", shrink=0.8)

        self._heatmap_fig.tight_layout()
        self._heatmap_canvas.draw()
