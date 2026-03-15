"""Main Dataset Selection tab assembling config, runner, viz, and results."""

import logging
from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QPushButton, QMessageBox,
)
from PyQt6.QtCore import Qt

from .config_editor import ConfigEditor
from .pipeline_runner import DatasetSelectionWorker
from .feature_viz import FeatureViz
from .results_viewer import ResultsViewer
from ...widgets.log_console import LogConsole
from ...widgets.progress_panel import ProgressPanel
from ...core.signal_bus import get_signal_bus

logger = logging.getLogger(__name__)

_TASK_ID = "dataset_selection"


class DatasetSelectionTab(QWidget):
    """Top-level widget for the Dataset Selection tab.

    Layout
    ------
    ::

        +--[Config Panel]--+--[Results Panel]----------------------------+
        | Method, Strategy | [Run]  [Stop]  ProgressPanel                |
        | Target, Weights  |                                              |
        | Paths, Cache     | +--[Feature Viz]--------+--[Gallery]------+ |
        | [Load] [Save]    | | PCA Fourier SAM EL2N  | Stats + Thumbs  | |
        |                  | +--[Log Console]--------+----------------+ |
        +------------------+-+------------------------------------------++
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[DatasetSelectionWorker] = None
        self._signal_bus = get_signal_bus()
        self._init_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left: Config editor ----------------------------------------
        self._config_editor = ConfigEditor()
        self._config_editor.setMinimumWidth(260)
        self._config_editor.setMaximumWidth(380)
        main_splitter.addWidget(self._config_editor)

        # ---- Right: toolbar + results + log -----------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._btn_run = QPushButton("Run Pipeline")
        self._btn_run.setStyleSheet(
            "QPushButton { background: #2A6E2A; font-weight: bold; }"
            "QPushButton:hover { background: #348F34; }"
        )
        self._btn_run.setFixedHeight(32)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setFixedHeight(32)

        self._progress = ProgressPanel()

        toolbar.addWidget(self._btn_run)
        toolbar.addWidget(self._btn_stop)
        toolbar.addWidget(self._progress, stretch=1)
        right_layout.addLayout(toolbar)

        # Middle: feature viz + results in tabs
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self._result_tabs = QTabWidget()
        self._feature_viz = FeatureViz()
        self._results_viewer = ResultsViewer()
        self._result_tabs.addTab(self._feature_viz, "Feature Viz")
        self._result_tabs.addTab(self._results_viewer, "Gallery / Stats")

        right_splitter.addWidget(self._result_tabs)

        # Log console
        self._log = LogConsole()
        right_splitter.addWidget(self._log)

        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(right_splitter)

        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        root.addWidget(main_splitter)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._btn_run.clicked.connect(self._on_run)
        self._btn_stop.clicked.connect(self._on_stop)

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Pipeline Running",
                                "A pipeline is already running. Stop it first.")
            return

        config_path = self._config_editor.get_config_path()
        target_size = self._config_editor.get_target_size()
        config_overrides = self._config_editor.get_config()

        self._log.clear()
        self._log.append(f"[{datetime.now():%H:%M:%S}] Starting dataset selection pipeline...")
        self._log.append(f"  Config: {config_path}")
        self._log.append(f"  Target size: {target_size}")
        self._log.append(f"  Method: {config_overrides.get('selection', {}).get('method', '?')}")

        self._progress.add_task(_TASK_ID, "Dataset Selection")
        self._progress.update_task(_TASK_ID, 0, "Starting...")

        self._worker = DatasetSelectionWorker(
            config_path=config_path,
            target_size=target_size,
            config_overrides=config_overrides,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.stage_changed.connect(self._on_stage)
        self._worker.log_message.connect(self._on_log)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._signal_bus.pipeline_started.emit("dataset_selection")
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._log.append("WARNING - Cancellation requested...")
            self._worker.cancel()
            self._btn_stop.setEnabled(False)

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def _on_progress(self, percent: int, message: str):
        self._progress.update_task(_TASK_ID, percent, message)

    def _on_stage(self, stage: str):
        self._log.append(f"INFO - Stage: {stage}")
        self._progress.update_task(_TASK_ID, -1, stage)

    def _on_log(self, line: str):
        self._log.append(line)

    def _on_result(self, results: dict):
        self._log.append(f"INFO - Pipeline completed successfully.")

        stats = results.get("statistics", {})
        orig = stats.get("original_count", "?")
        final = stats.get("final_count", "?")
        self._log.append(f"INFO - Selected {final} / {orig} images.")

        # Update visualizations
        self._feature_viz.update_plots(results)

        # Update results viewer
        self._results_viewer.update_results(results)

        # Switch to Gallery/Stats tab
        self._result_tabs.setCurrentIndex(1)

        self._signal_bus.pipeline_finished.emit("dataset_selection", results)

    def _on_error(self, err_type: str, message: str):
        self._log.append(f"ERROR - {err_type}: {message}")
        self._progress.update_task(_TASK_ID, 0, f"Error: {err_type}")
        QMessageBox.critical(self, f"Pipeline Error ({err_type})", message)
        self._signal_bus.pipeline_error.emit("dataset_selection", message)

    def _on_finished(self):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.remove_task(_TASK_ID)
        self._worker = None
