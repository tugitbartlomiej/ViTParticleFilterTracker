"""Main Benchmarks tab assembling runner, viewer, log, and report widgets."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from .benchmark_runner import BenchmarkRunner
from .results_viewer import ResultsViewer
from .report_generator import ReportGenerator
from ...widgets.log_console import LogConsole


class BenchmarksTab(QWidget):
    """Top-level benchmarks tab with four quadrants.

    Layout::

        +--[Benchmark Runner]--+--[Results Viewer]------------------+
        | Script: [combo]      | [Table: Model|Epoch|mAP|Prec|...]  |
        | --arg1: [edit]       |                                     |
        | --arg2: [edit]       | [Chart - comparison visualization]  |
        | [Run] [Stop]         |                                     |
        | Progress: [========] |                                     |
        +--[Log Console]-------+--[Report Generator]----------------+
        | Running benchmark... | Format: [JSON]  Output: [path]     |
        | Evaluating epoch 100 | [Generate Report]                   |
        +----------------------+-------------------------------------+
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()
        self._initial_load()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Main horizontal splitter: left column | right column
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ----- Left column: Runner (top) + Log (bottom) -------------------
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        self._runner = BenchmarkRunner()
        left_splitter.addWidget(self._runner)

        self._log = LogConsole()
        left_splitter.addWidget(self._log)

        left_splitter.setStretchFactor(0, 3)  # runner gets more space
        left_splitter.setStretchFactor(1, 2)

        # ----- Right column: Results (top) + Report (bottom) --------------
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self._results = ResultsViewer()
        right_splitter.addWidget(self._results)

        self._report = ReportGenerator()
        right_splitter.addWidget(self._report)

        right_splitter.setStretchFactor(0, 4)  # results get more space
        right_splitter.setStretchFactor(1, 1)

        # Assemble main splitter
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)

        layout.addWidget(main_splitter)

    def _connect_signals(self):
        """Wire up communication between sub-widgets."""
        # Runner stdout -> Log console
        self._runner.output_line.connect(self._log.append)

        # Runner finished -> Results viewer + Report generator
        self._runner.benchmark_finished.connect(self._on_benchmark_finished)

    def _initial_load(self):
        """Populate the script list and previous runs on first display."""
        self._runner.refresh_scripts()
        self._results.refresh_previous_runs()

    def _on_benchmark_finished(self, result: dict):
        """Handle benchmark completion: load results and update report."""
        data = result.get("data")
        if data:
            self._results.load_results(data)

        # Push the current table rows to the report generator
        self._report.set_results(self._results._rows)

        # Also refresh the previous-runs dropdown
        self._results.refresh_previous_runs()
