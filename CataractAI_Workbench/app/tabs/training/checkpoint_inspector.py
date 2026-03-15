"""Checkpoint Inspector widget -- UI for inspecting .pth checkpoint weights."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...widgets.file_picker import FilePicker
from ...core.analysis_services import CheckpointAnalysisService
from .theme_constants import (
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_YELLOW,
    BLUE_BUTTON_STYLE,
    FG,
    GROUP_STYLE,
    PURPLE_BUTTON_STYLE,
    TEXT_STYLE,
    TREE_STYLE,
    make_canvas,
    style_axis,
    BG,
    ACCENT_BLUE,
    BORDER,
)


# -------------------------------------------------------------------
# Background loader thread
# -------------------------------------------------------------------

class _CheckpointLoader(QThread):
    """Load a ``.pth`` file in a background thread."""

    loaded = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, path: str, service: CheckpointAnalysisService, parent=None):
        super().__init__(parent)
        self._path = path
        self._service = service

    def run(self) -> None:
        try:
            result = self._service.load_checkpoint(self._path)
            self.loaded.emit(result)
        except Exception as exc:
            self.error.emit(f"Failed to load {self._path}: {exc}")


# -------------------------------------------------------------------
# CheckpointInspector widget
# -------------------------------------------------------------------

class CheckpointInspector(QWidget):
    """Inspect checkpoint weights: layer stats, histograms, epoch comparison."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._service = CheckpointAnalysisService()
        self._checkpoints: Dict[str, dict] = {}
        self._current_path: str = ""
        self._loader: Optional[_CheckpointLoader] = None
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

        grp = QGroupBox("Checkpoint Selector")
        grp.setStyleSheet(GROUP_STYLE)
        grp_layout = QVBoxLayout(grp)

        self._dir_picker = FilePicker(label="Browse...", mode="dir")
        grp_layout.addWidget(QLabel("Directory:"))
        grp_layout.addWidget(self._dir_picker)

        self._btn_scan = QPushButton("Scan Checkpoints")
        self._btn_scan.setStyleSheet(BLUE_BUTTON_STYLE)
        self._btn_scan.clicked.connect(self._scan_directory)
        grp_layout.addWidget(self._btn_scan)

        self._ckpt_tree = QTreeWidget()
        self._ckpt_tree.setHeaderLabels(["Checkpoint", "Size"])
        self._ckpt_tree.setRootIsDecorated(False)
        self._ckpt_tree.setAlternatingRowColors(True)
        self._ckpt_tree.setStyleSheet(TREE_STYLE)
        self._ckpt_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        header = self._ckpt_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._ckpt_tree.itemClicked.connect(self._on_ckpt_selected)
        grp_layout.addWidget(self._ckpt_tree)

        self._lbl_total_params = QLabel("Total params: --")
        self._lbl_total_params.setStyleSheet(f"color: {ACCENT_GREEN};")
        self._lbl_trainable = QLabel("Trainable: --")
        self._lbl_trainable.setStyleSheet(f"color: {ACCENT_GREEN};")
        self._lbl_size = QLabel("Size: --")
        self._lbl_size.setStyleSheet(f"color: {FG};")
        grp_layout.addWidget(self._lbl_total_params)
        grp_layout.addWidget(self._lbl_trainable)
        grp_layout.addWidget(self._lbl_size)

        self._btn_compare = QPushButton("Compare Selected")
        self._btn_compare.setStyleSheet(PURPLE_BUTTON_STYLE)
        self._btn_compare.clicked.connect(self._compare_checkpoints)
        grp_layout.addWidget(self._btn_compare)

        layout.addWidget(grp)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Layer analysis group
        grp_layer = QGroupBox("Layer Analysis")
        grp_layer.setStyleSheet(GROUP_STYLE)
        layer_layout = QVBoxLayout(grp_layer)

        layer_sel_row = QHBoxLayout()
        layer_sel_row.addWidget(QLabel("Layer:"))
        self._layer_combo = QComboBox()
        self._layer_combo.setMinimumWidth(300)
        self._layer_combo.currentTextChanged.connect(self._on_layer_changed)
        layer_sel_row.addWidget(self._layer_combo, stretch=1)
        layer_layout.addLayout(layer_sel_row)

        self._layer_stats_text = QTextEdit()
        self._layer_stats_text.setReadOnly(True)
        self._layer_stats_text.setMaximumHeight(120)
        self._layer_stats_text.setFont(QFont("Consolas", 9))
        self._layer_stats_text.setStyleSheet(TEXT_STYLE)
        layer_layout.addWidget(self._layer_stats_text)

        self._hist_fig, self._hist_canvas = make_canvas()
        self._hist_canvas.setMinimumHeight(200)
        layer_layout.addWidget(self._hist_canvas)

        layout.addWidget(grp_layer, stretch=3)

        # Epoch comparison group
        grp_compare = QGroupBox("Epoch Comparison")
        grp_compare.setStyleSheet(GROUP_STYLE)
        compare_layout = QVBoxLayout(grp_compare)

        self._compare_tree = QTreeWidget()
        self._compare_tree.setHeaderLabels(["Layer", "Ckpt A (Std)", "Ckpt B (Std)", "Delta"])
        self._compare_tree.setRootIsDecorated(False)
        self._compare_tree.setAlternatingRowColors(True)
        self._compare_tree.setStyleSheet(TREE_STYLE)
        cmp_header = self._compare_tree.header()
        cmp_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            cmp_header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        compare_layout.addWidget(self._compare_tree)

        layout.addWidget(grp_compare, stretch=2)
        return panel

    # ---------------------------------------------------------------
    # Directory scanning
    # ---------------------------------------------------------------

    def _scan_directory(self) -> None:
        dir_path = self._dir_picker.path()
        if not dir_path or not Path(dir_path).is_dir():
            QMessageBox.warning(self, "Invalid Path", "Please select a valid directory.")
            return

        self._ckpt_tree.clear()
        self._checkpoints.clear()

        entries = self._service.scan_directory(dir_path)

        if not entries:
            self._ckpt_tree.addTopLevelItem(QTreeWidgetItem(["No .pth files found", ""]))
            return

        for entry in entries:
            item = QTreeWidgetItem([entry["name"], f"{entry['size_mb']:.1f} MB"])
            item.setData(0, Qt.ItemDataRole.UserRole, entry["path"])
            self._ckpt_tree.addTopLevelItem(item)

    # ---------------------------------------------------------------
    # Checkpoint loading
    # ---------------------------------------------------------------

    def _on_ckpt_selected(self, item: QTreeWidgetItem, _col: int) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return

        if path in self._checkpoints:
            self._show_checkpoint(path)
            return

        self._current_path = path
        self._lbl_total_params.setText("Loading...")
        self._loader = _CheckpointLoader(path, self._service, self)
        self._loader.loaded.connect(self._on_checkpoint_loaded)
        self._loader.error.connect(self._on_load_error)
        self._loader.start()

    def _on_checkpoint_loaded(self, data: dict) -> None:
        path = data["path"]
        self._checkpoints[path] = data
        if path == self._current_path:
            self._show_checkpoint(path)

    def _on_load_error(self, msg: str) -> None:
        self._lbl_total_params.setText("Load failed")
        QMessageBox.warning(self, "Load Error", msg)

    def _show_checkpoint(self, path: str) -> None:
        sd = self._checkpoints[path]["state_dict"]
        total_params, layer_names = self._service.count_params(sd)

        size_mb = Path(path).stat().st_size / (1024 * 1024)
        self._lbl_total_params.setText(f"Total params: {total_params:,}")
        self._lbl_trainable.setText(f"Trainable: {total_params:,}")
        self._lbl_size.setText(f"Size: {size_mb:.1f} MB")

        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        self._layer_combo.addItems(layer_names)
        self._layer_combo.blockSignals(False)

        if layer_names:
            self._layer_combo.setCurrentIndex(0)
            self._on_layer_changed(layer_names[0])

    # ---------------------------------------------------------------
    # Layer analysis
    # ---------------------------------------------------------------

    def _on_layer_changed(self, layer_name: str) -> None:
        if not layer_name or not self._current_path:
            return

        data = self._checkpoints.get(self._current_path)
        if data is None:
            return

        stats = self._service.compute_layer_stats(data["state_dict"], layer_name)

        if "error" in stats:
            self._layer_stats_text.setPlainText(stats["error"])
            return

        text = (
            f"Layer:  {stats['layer_name']}\n"
            f"Shape:  {stats['shape']}   Params: {stats['numel']:,}\n"
            f"Mean:   {stats['mean']:.6f}   Std: {stats['std']:.6f}\n"
            f"Min:    {stats['min']:.6f}   Max: {stats['max']:.6f}\n"
            f"Dtype:  {stats['dtype']}"
        )
        self._layer_stats_text.setPlainText(text)
        self._draw_histogram(stats["values_flat"])

    def _draw_histogram(self, values: np.ndarray) -> None:
        self._hist_fig.clear()
        ax = self._hist_fig.add_subplot(111)
        style_axis(ax, title="Weight Distribution", xlabel="Value", ylabel="Count")

        q1, q99 = np.percentile(values, [1, 99])
        clipped = values[(values >= q1) & (values <= q99)]

        ax.hist(clipped, bins=80, color=ACCENT_BLUE, alpha=0.8,
                edgecolor=BG, linewidth=0.5)

        mean_val = float(np.mean(values))
        ax.axvline(mean_val, color=ACCENT_YELLOW, linestyle="--", linewidth=1.5,
                   label=f"Mean: {mean_val:.4f}")
        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=FG)

        self._hist_fig.tight_layout()
        self._hist_canvas.draw()

    # ---------------------------------------------------------------
    # Checkpoint comparison
    # ---------------------------------------------------------------

    def _compare_checkpoints(self) -> None:
        selected = self._ckpt_tree.selectedItems()
        if len(selected) < 2:
            QMessageBox.information(
                self, "Select Checkpoints",
                "Please select exactly 2 checkpoints (Ctrl+Click) to compare.",
            )
            return

        paths = [item.data(0, Qt.ItemDataRole.UserRole) for item in selected[:2]]

        for p in paths:
            if p not in self._checkpoints:
                QMessageBox.information(
                    self, "Load First",
                    f"Please click on '{Path(p).name}' first to load it, then compare.",
                )
                return

        result = self._service.compare_state_dicts(
            self._checkpoints[paths[0]]["state_dict"],
            self._checkpoints[paths[1]]["state_dict"],
            name_a=Path(paths[0]).stem,
            name_b=Path(paths[1]).stem,
        )

        self._compare_tree.clear()
        self._compare_tree.setHeaderLabels([
            "Layer", result["name_a"], result["name_b"], "Delta (Std)",
        ])

        for name, std_a, std_b, delta in result["rows"]:
            item = QTreeWidgetItem([
                name, f"{std_a:.6f}", f"{std_b:.6f}", f"{delta:.6f}",
            ])
            if delta > 0.01:
                item.setForeground(3, QColor(ACCENT_RED))
            elif delta > 0.001:
                item.setForeground(3, QColor(ACCENT_YELLOW))
            else:
                item.setForeground(3, QColor(ACCENT_GREEN))
            self._compare_tree.addTopLevelItem(item)
