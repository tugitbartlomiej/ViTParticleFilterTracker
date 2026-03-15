"""Widget for viewing and charting benchmark results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...widgets.chart_widget import ChartWidget
from ...widgets.file_picker import FilePicker
from .results_charts import draw_bar_chart, draw_line_chart


# Column definitions for the results table
_COLUMNS = [
    ("Model", "model"),
    ("Epoch", "epoch"),
    ("mAP@50", "mAP@0.5"),
    ("mAP@50:95", "mAP@0.5:0.95"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1_score"),
]

# Alternate key mappings for normalization
_KEY_ALIASES = {
    "mAP_50": "mAP@0.5",
    "mAP_50_95": "mAP@0.5:0.95",
    "f1": "f1_score",
}


class ResultsViewer(QWidget):
    """Table + chart for benchmark results viewing and comparison."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._rows: List[dict] = []
        self._init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -- Top bar: load / previous runs / export -------------------------
        top = QHBoxLayout()

        self._picker = FilePicker(
            label="Load JSON",
            mode="file",
            filter_str="JSON Files (*.json);;All Files (*)",
        )
        self._picker.path_changed.connect(self._on_file_selected)
        top.addWidget(self._picker, stretch=1)

        self._combo_runs = QComboBox()
        self._combo_runs.setMinimumWidth(220)
        self._combo_runs.setPlaceholderText("Previous Runs...")
        self._combo_runs.currentIndexChanged.connect(self._on_run_selected)
        top.addWidget(self._combo_runs)

        self._btn_export = QPushButton("Export CSV")
        self._btn_export.setFixedWidth(90)
        self._btn_export.clicked.connect(self._on_export)
        top.addWidget(self._btn_export)

        layout.addLayout(top)

        # -- Results table ---------------------------------------------------
        self._table = QTableWidget()
        self._table.setColumnCount(len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._table, stretch=2)

        # -- Chart type selector + chart ------------------------------------
        chart_bar = QHBoxLayout()
        self._combo_chart = QComboBox()
        self._combo_chart.addItems([
            "Bar Chart (Model Comparison)",
            "Line Chart (Epoch Progression)",
        ])
        self._combo_chart.currentIndexChanged.connect(self._refresh_chart)
        chart_bar.addWidget(self._combo_chart)
        chart_bar.addStretch()
        layout.addLayout(chart_bar)

        self._chart = ChartWidget.matplotlib(parent=self)
        self._chart.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._chart, stretch=3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_results(self, data: Union[dict, List[dict]]):
        """Populate the table and chart from benchmark data.

        *data* can be either:
        - A list of flat result dicts: ``[{model, epoch, mAP@0.5, ...}]``
        - A dict keyed by model name (the format produced by the benchmark
          scripts, e.g. ``{"YOLO_epoch70": {...}, ...}``).
        """
        self.clear()

        if isinstance(data, list):
            for entry in data:
                self.add_result_row(entry)
        elif isinstance(data, dict):
            self._load_nested_results(data)
        self._refresh_chart()

    def add_result_row(self, result: dict):
        """Add a single result row to the table."""
        row = self._normalize_result(result)
        self._rows.append(row)

        row_idx = self._table.rowCount()
        self._table.insertRow(row_idx)

        for col_idx, (_, key) in enumerate(_COLUMNS):
            value = row.get(key, "")
            if isinstance(value, float):
                text = f"{value:.2f}"
            else:
                text = str(value) if value is not None else ""
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row_idx, col_idx, item)

    def clear(self):
        """Remove all rows and reset the chart."""
        self._rows.clear()
        self._table.setRowCount(0)
        self._chart.clear()

    def export_table(self, path: str):
        """Export the current table contents to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([c[0] for c in _COLUMNS])
            for row in self._rows:
                writer.writerow(
                    [row.get(key, "") for _, key in _COLUMNS]
                )

    def refresh_previous_runs(self):
        """Reload the list of previous benchmark runs."""
        from CataractAI_Workbench.backend.benchmark_adapter import BenchmarkAdapter

        adapter = BenchmarkAdapter()
        runs = adapter.list_previous_runs()

        self._combo_runs.blockSignals(True)
        self._combo_runs.clear()
        self._combo_runs.addItem("-- Previous Runs --", None)
        for run in runs:
            label = run["name"]
            if run.get("date"):
                label += f"  ({run['date']})"
            self._combo_runs.addItem(label, run)
        self._combo_runs.blockSignals(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_nested_results(self, data: dict):
        """Load results from the nested dict format produced by scripts."""
        for model_key, model_data in data.items():
            model_type = model_data.get("type", model_key)
            epoch = model_data.get("epoch", "")

            metrics = self._extract_metrics(model_data)

            row = {
                "model": model_type,
                "epoch": epoch,
                "mAP@0.5": metrics.get("mAP@0.5", ""),
                "mAP@0.5:0.95": metrics.get("mAP@0.5:0.95", ""),
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1_score": metrics.get("f1_score", ""),
            }
            self.add_result_row(row)

    @staticmethod
    def _extract_metrics(model_data: dict) -> dict:
        """Extract the best available metrics from nested model data."""
        metrics = model_data.get("aggregated", {})
        if not metrics:
            splits = model_data.get("splits", {})
            for split_data in splits.values():
                metrics = split_data.get("metrics", {})
                if metrics:
                    break

        if "mAP@0.5" not in metrics:
            splits = model_data.get("splits", {})
            for split_name in ("original_test", "test", "valid", "train"):
                split_data = splits.get(split_name, {})
                split_metrics = split_data.get("metrics", {})
                if "mAP@0.5" in split_metrics:
                    for k in ("mAP@0.5", "mAP@0.5:0.95", "mAP@0.75", "AR@100"):
                        if k in split_metrics and k not in metrics:
                            metrics[k] = split_metrics[k]
                    break

        return metrics

    @staticmethod
    def _normalize_result(result: dict) -> dict:
        """Normalize a result dict to the flat format used by the table."""
        metrics = result.get("metrics", {})
        if metrics:
            flat = {"model": result.get("model", ""), "epoch": result.get("epoch", "")}
            flat.update(metrics)
        else:
            flat = dict(result)

        # Map alternate key names
        for alias, canonical in _KEY_ALIASES.items():
            if alias in flat and canonical not in flat:
                flat[canonical] = flat[alias]

        return flat

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_file_selected(self, path: str):
        """Load results from a user-selected JSON file."""
        if not path or not Path(path).is_file():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_results(data)
        except Exception as exc:
            self.add_result_row({"model": f"Error: {exc}", "epoch": ""})

    def _on_run_selected(self, index: int):
        """Load results from a previously-run benchmark."""
        run = self._combo_runs.currentData()
        if run is None or not run.get("result_files"):
            return
        result_path = run["result_files"][0]
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.load_results(data)
        except Exception:
            pass

    def _on_export(self):
        """Prompt user for save path and export CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.export_table(path)

    def _refresh_chart(self, _index: int = 0):
        """Redraw the chart based on the current chart type selection."""
        if not self._rows:
            self._chart.clear()
            return

        chart_type = self._combo_chart.currentIndex()
        if chart_type == 0:
            fig = draw_bar_chart(self._rows)
        else:
            fig = draw_line_chart(self._rows)
        self._chart.update_figure(fig)
