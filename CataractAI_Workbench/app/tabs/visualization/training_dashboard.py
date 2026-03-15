"""Training metrics dashboard with live pyqtgraph charts and epoch table."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QComboBox,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSlot

from ...core.signal_bus import get_signal_bus
from ...widgets.file_picker import FilePicker


# Palette for overlaying multiple runs
_RUN_COLORS = [
    (78, 154, 255),    # blue
    (255, 107, 107),   # red
    (107, 203, 119),   # green
    (255, 217, 61),    # yellow
    (180, 130, 255),   # purple
    (255, 165, 80),    # orange
    (100, 220, 220),   # cyan
    (220, 150, 180),   # pink
]


class TrainingDashboard(QWidget):
    """Unified training metrics viewer.

    Shows loss and accuracy/mAP curves from local training, Eden jobs,
    or loaded JSON files.  Multiple runs can be overlaid for comparison.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._runs: Dict[str, List[dict]] = {}  # name -> [{epoch, loss, ...}, ...]
        self._color_idx = 0
        self._run_colors: Dict[str, tuple] = {}
        self._live_run_name: Optional[str] = None
        self._init_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # --- Top toolbar --------------------------------------------------
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Data source:"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(["Local Training", "Eden Job", "Load from file"])
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._source_combo.setFixedWidth(180)
        toolbar.addWidget(self._source_combo)

        self._file_picker = FilePicker(
            label="Browse...",
            mode="file",
            filter_str="JSON Files (*.json);;All Files (*)",
        )
        self._file_picker.setVisible(False)
        self._file_picker.path_changed.connect(self._on_file_selected)
        toolbar.addWidget(self._file_picker)

        self._btn_load = QPushButton("Load")
        self._btn_load.setFixedWidth(60)
        self._btn_load.setVisible(False)
        self._btn_load.clicked.connect(self._on_load_clicked)
        toolbar.addWidget(self._btn_load)

        toolbar.addStretch()

        self._btn_clear = QPushButton("Clear All")
        self._btn_clear.setFixedWidth(80)
        self._btn_clear.clicked.connect(self.clear_runs)
        toolbar.addWidget(self._btn_clear)

        self._btn_export = QPushButton("Export")
        self._btn_export.setToolTip("Export current data to JSON")
        self._btn_export.setFixedWidth(70)
        self._btn_export.clicked.connect(self._on_export)
        toolbar.addWidget(self._btn_export)

        root.addLayout(toolbar)

        # --- Charts -------------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Vertical)

        charts_widget = QWidget()
        charts_layout = QHBoxLayout(charts_widget)
        charts_layout.setContentsMargins(0, 0, 0, 0)

        pg.setConfigOptions(background="#1E1E1E", foreground="#D4D4D4")

        # Left: Loss chart
        loss_group = QGroupBox("Loss")
        loss_lay = QVBoxLayout(loss_group)
        loss_lay.setContentsMargins(2, 2, 2, 2)
        self._loss_plot = pg.PlotWidget()
        self._loss_plot.setLabel("left", "Loss")
        self._loss_plot.setLabel("bottom", "Epoch")
        self._loss_plot.addLegend(offset=(10, 10))
        self._loss_plot.showGrid(x=True, y=True, alpha=0.3)
        loss_lay.addWidget(self._loss_plot)
        charts_layout.addWidget(loss_group)

        # Right: mAP / accuracy chart
        acc_group = QGroupBox("mAP / Accuracy")
        acc_lay = QVBoxLayout(acc_group)
        acc_lay.setContentsMargins(2, 2, 2, 2)
        self._acc_plot = pg.PlotWidget()
        self._acc_plot.setLabel("left", "mAP / Accuracy")
        self._acc_plot.setLabel("bottom", "Epoch")
        self._acc_plot.addLegend(offset=(10, 10))
        self._acc_plot.showGrid(x=True, y=True, alpha=0.3)
        acc_lay.addWidget(self._acc_plot)
        charts_layout.addWidget(acc_group)

        splitter.addWidget(charts_widget)

        # --- Epoch table --------------------------------------------------
        table_group = QGroupBox("Epoch Metrics")
        table_lay = QVBoxLayout(table_group)
        table_lay.setContentsMargins(2, 2, 2, 2)

        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget { background-color: #1E1E1E; color: #D4D4D4; "
            "gridline-color: #3C3C3C; alternate-background-color: #252526; }"
            "QHeaderView::section { background-color: #2D2D30; color: #D4D4D4; "
            "padding: 4px; border: 1px solid #3C3C3C; }"
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table_lay.addWidget(self._table)

        splitter.addWidget(table_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

    def _connect_signals(self):
        """Connect to global signal bus for live updates."""
        bus = get_signal_bus()
        bus.epoch_completed.connect(self.on_epoch_completed)
        bus.training_started.connect(self._on_training_started)
        bus.training_finished.connect(self._on_training_finished)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_training_run(self, name: str, data: List[dict]):
        """Add (or replace) a run's epoch data.

        Parameters
        ----------
        name : str
            Display name for the run (used in legend).
        data : list[dict]
            Each dict should have at least ``epoch`` and ``loss``.
            Optional keys: ``lr``, ``val_loss``, ``mAP``, ``mAP_50``,
            ``accuracy``, ``precision``, ``recall``.
        """
        self._runs[name] = list(data)
        if name not in self._run_colors:
            self._run_colors[name] = _RUN_COLORS[self._color_idx % len(_RUN_COLORS)]
            self._color_idx += 1
        self._redraw()

    def clear_runs(self):
        """Remove all runs and reset charts."""
        self._runs.clear()
        self._run_colors.clear()
        self._color_idx = 0
        self._live_run_name = None
        self._loss_plot.clear()
        self._acc_plot.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)

    def load_from_json(self, path: str):
        """Load training history from a JSON file.

        Accepted formats:
        - List of dicts: ``[{epoch, loss, ...}, ...]``
        - Dict with ``"epochs"`` key: ``{"epochs": [...]}``
        - Dict with named runs: ``{"run_name": [...], ...}``
        """
        p = Path(path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            self.add_training_run(p.stem, raw)
        elif isinstance(raw, dict):
            if "epochs" in raw and isinstance(raw["epochs"], list):
                name = raw.get("name", p.stem)
                self.add_training_run(name, raw["epochs"])
            else:
                # Treat each key with a list value as a separate run
                loaded_any = False
                for key, val in raw.items():
                    if isinstance(val, list) and val and isinstance(val[0], dict):
                        self.add_training_run(key, val)
                        loaded_any = True
                if not loaded_any:
                    # Single flat dict with metrics -- wrap in list
                    self.add_training_run(p.stem, [raw])

    @pyqtSlot(dict)
    def on_epoch_completed(self, data: dict):
        """Receive a live epoch update from signal_bus.

        Parameters
        ----------
        data : dict
            Must contain ``epoch`` and ``loss`` at minimum.
        """
        name = self._live_run_name or data.get("model", "Live Training")
        if name not in self._runs:
            self._runs[name] = []
            self._run_colors[name] = _RUN_COLORS[self._color_idx % len(_RUN_COLORS)]
            self._color_idx += 1
        self._runs[name].append(dict(data))
        self._redraw()

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    @pyqtSlot(str)
    def _on_training_started(self, model_name: str):
        self._live_run_name = model_name

    @pyqtSlot(str, dict)
    def _on_training_finished(self, model_name: str, metrics: dict):
        self._live_run_name = None

    def _on_source_changed(self, index: int):
        is_file = index == 2  # "Load from file"
        self._file_picker.setVisible(is_file)
        self._btn_load.setVisible(is_file)

    def _on_file_selected(self, path: str):
        pass  # wait for Load button click

    def _on_load_clicked(self):
        path = self._file_picker.path()
        if path and Path(path).is_file():
            self.load_from_json(path)

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Training Data", "", "JSON Files (*.json)"
        )
        if not path:
            return
        export = {}
        for name, data in self._runs.items():
            export[name] = data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _redraw(self):
        """Replot all runs on both charts and rebuild the table."""
        self._loss_plot.clear()
        self._acc_plot.clear()

        for name, data in self._runs.items():
            if not data:
                continue
            color = self._run_colors.get(name, (200, 200, 200))
            pen = pg.mkPen(color=color, width=2)
            symbol_brush = pg.mkBrush(color)

            epochs = [d.get("epoch", i) for i, d in enumerate(data)]

            # --- Loss ---
            losses = [d.get("loss") for d in data]
            if any(v is not None for v in losses):
                clean_epochs = [e for e, v in zip(epochs, losses) if v is not None]
                clean_losses = [v for v in losses if v is not None]
                self._loss_plot.plot(
                    clean_epochs, clean_losses,
                    pen=pen, symbol="o", symbolSize=5,
                    symbolBrush=symbol_brush, name=name,
                )

            # Val loss (dashed)
            val_losses = [d.get("val_loss") for d in data]
            if any(v is not None for v in val_losses):
                dash_pen = pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine)
                ce = [e for e, v in zip(epochs, val_losses) if v is not None]
                cv = [v for v in val_losses if v is not None]
                self._loss_plot.plot(
                    ce, cv, pen=dash_pen, symbol="s", symbolSize=4,
                    symbolBrush=symbol_brush, name=f"{name} (val)",
                )

            # --- mAP / accuracy ---
            for metric_key, label_suffix in [
                ("mAP", "mAP"),
                ("mAP_50", "mAP@50"),
                ("accuracy", "Acc"),
            ]:
                values = [d.get(metric_key) for d in data]
                if any(v is not None for v in values):
                    ce = [e for e, v in zip(epochs, values) if v is not None]
                    cv = [v for v in values if v is not None]
                    self._acc_plot.plot(
                        ce, cv, pen=pen, symbol="o", symbolSize=5,
                        symbolBrush=symbol_brush,
                        name=f"{name} ({label_suffix})",
                    )

        self._rebuild_table()

    def _rebuild_table(self):
        """Populate the epoch table with all runs' data."""
        if not self._runs:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            return

        # Collect all column keys across all runs
        all_keys: list[str] = []
        for data in self._runs.values():
            for d in data:
                for k in d:
                    if k not in all_keys:
                        all_keys.append(k)

        # Ensure "run" and "epoch" come first
        ordered = ["run"]
        if "epoch" in all_keys:
            ordered.append("epoch")
            all_keys.remove("epoch")
        for k in all_keys:
            if k not in ordered:
                ordered.append(k)

        self._table.setColumnCount(len(ordered))
        self._table.setHorizontalHeaderLabels(ordered)

        total_rows = sum(len(d) for d in self._runs.values())
        self._table.setRowCount(total_rows)

        row = 0
        for name, data in self._runs.items():
            for entry in data:
                for col, key in enumerate(ordered):
                    if key == "run":
                        value = name
                    else:
                        value = entry.get(key, "")
                    if isinstance(value, float):
                        text = f"{value:.6g}"
                    else:
                        text = str(value)
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._table.setItem(row, col, item)
                row += 1
