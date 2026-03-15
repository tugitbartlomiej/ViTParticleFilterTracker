"""Main Training tab widget for CataractAI Workbench."""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.signal_bus import get_signal_bus
from ...widgets.log_console import LogConsole
from .checkpoint_manager import CheckpointManager
from .config_editor import ConfigEditor
from .live_chart import LiveChart
from .training_analyzer import TrainingAnalyzer
from .training_runner import TrainingRunner


class TrainingTab(QWidget):
    """Top-level Training tab with two sub-tabs: Training and Training Analyzer.

    Layout
    ------
    +--[Training | Training Analyzer]-----------------------------------+
    |                                                                    |
    | Training sub-tab:                                                  |
    | +--[Config Panel]--+--[Training Monitor]-------------------+       |
    | | (scroll area)    | [Start] [Stop]   status label         |       |
    | |                  | +--[LiveChart]----------------------+ |       |
    | |                  | | Loss curve       LR curve         | |       |
    | |                  | +-----------------------------------+ |       |
    | |                  | +--[Checkpoints]--+--[LogConsole]---+ |       |
    | |                  | | ep5_best.pth    | Epoch 3 ...     | |       |
    | +------------------+ +-----------------+-----------------+ |       |
    |                                                                    |
    | Training Analyzer sub-tab:                                         |
    | +--[Checkpoint Inspector | Parameter Advisor | Dataset Analyzer]--+|
    | | (analysis tools for checkpoints, parameters, datasets)          ||
    | +-----------------------------------------------------------------+|
    +--------------------------------------------------------------------+
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner: TrainingRunner | None = None
        self._bus = get_signal_bus()
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Top-level sub-tab widget
        self._sub_tabs = QTabWidget()
        self._sub_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._sub_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #3C3C3C; }"
            "QTabBar::tab { background: #2D2D2D; color: #D4D4D4; padding: 6px 20px; "
            "border: 1px solid #3C3C3C; border-bottom: none; margin-right: 2px; "
            "font-weight: bold; }"
            "QTabBar::tab:selected { background: #1E1E1E; color: #6BCB77; "
            "border-bottom: 2px solid #6BCB77; }"
            "QTabBar::tab:hover { background: #383838; }"
        )

        # --- Sub-tab 1: Training (original UI) ---
        training_page = QWidget()
        root_layout = QHBoxLayout(training_page)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # ── Left: config editor in a scroll area ──
        self._config_editor = ConfigEditor()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._config_editor)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(380)
        scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #3C3C3C; background: #1E1E1E; }"
        )
        root_layout.addWidget(scroll)

        # ── Right: training monitor ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Toolbar row
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._btn_start = QPushButton("Start Training")
        self._btn_start.setFixedHeight(32)
        self._btn_start.setStyleSheet(
            "QPushButton { background-color: #2EA043; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 0 16px; }"
            "QPushButton:hover { background-color: #3FB950; }"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            "QPushButton { background-color: #DA3633; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 0 16px; }"
            "QPushButton:hover { background-color: #F85149; }"
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )

        self._status_label = QLabel("Idle")
        self._status_label.setFont(QFont("Segoe UI", 9))
        self._status_label.setStyleSheet("color: #8B949E;")

        toolbar.addWidget(self._btn_start)
        toolbar.addWidget(self._btn_stop)
        toolbar.addSpacing(12)
        toolbar.addWidget(self._status_label)
        toolbar.addStretch()

        right_layout.addLayout(toolbar)

        # Live chart area
        self._live_chart = LiveChart()
        self._live_chart.setMinimumHeight(200)
        right_layout.addWidget(self._live_chart, stretch=3)

        # Bottom splitter: checkpoints | log console
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._ckpt_manager = CheckpointManager()
        self._ckpt_manager.setMinimumWidth(200)
        bottom_splitter.addWidget(self._ckpt_manager)

        self._log_console = LogConsole()
        bottom_splitter.addWidget(self._log_console)

        bottom_splitter.setSizes([300, 500])
        right_layout.addWidget(bottom_splitter, stretch=2)

        root_layout.addWidget(right, stretch=1)

        self._sub_tabs.addTab(training_page, "Training")

        # --- Sub-tab 2: Training Analyzer ---
        self._training_analyzer = TrainingAnalyzer()
        self._sub_tabs.addTab(self._training_analyzer, "Training Analyzer")

        outer_layout.addWidget(self._sub_tabs)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop.clicked.connect(self._on_stop)

        # Config editor buttons
        self._config_editor.btn_load.clicked.connect(self._on_load_config)
        self._config_editor.btn_save.clicked.connect(self._on_save_config)
        self._config_editor.btn_defaults.clicked.connect(self._on_defaults)

        # Checkpoint manager
        self._ckpt_manager.load_checkpoint.connect(self._on_load_checkpoint)

        # When checkpoint_dir changes, auto-scan
        self._config_editor.checkpoint_dir.path_changed.connect(
            self._ckpt_manager.scan_directory
        )

        # Global bus epoch signal -> live chart
        self._bus.epoch_completed.connect(self._live_chart.on_epoch_complete)

    # ------------------------------------------------------------------
    # Training start / stop
    # ------------------------------------------------------------------

    def _on_start(self):
        if self._runner is not None and self._runner.isRunning():
            return

        config = self._config_editor.get_config()

        # Basic validation
        ds_dir = config.get("dataset_paths", {}).get("mixed_dataset_dir", "")
        if not ds_dir or not Path(ds_dir).is_dir():
            QMessageBox.warning(
                self,
                "Invalid config",
                "Please set a valid dataset directory.",
            )
            return

        self._live_chart.clear()
        self._log_console.clear()

        self._runner = TrainingRunner(config)
        self._runner.log_message.connect(self._log_console.append)
        self._runner.progress.connect(self._on_progress)
        self._runner.result_ready.connect(self._on_finished)
        self._runner.error_occurred.connect(self._on_error)

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._status_label.setText("Training...")
        self._status_label.setStyleSheet("color: #FFD93D;")

        self._runner.start()

    def _on_stop(self):
        if self._runner is not None:
            self._runner.cancel()
            self._status_label.setText("Stopping...")
            self._status_label.setStyleSheet("color: #F0883E;")
            self._log_console.append("Cancellation requested -- finishing current epoch...")

    # ------------------------------------------------------------------
    # Runner callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, percent: int, message: str):
        self._status_label.setText(message)

    def _on_finished(self, result: dict):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

        best = result.get("best_loss", "?")
        ran = result.get("epochs_ran", "?")
        final_model = result.get("final_model", "")

        if isinstance(best, float):
            best_str = f"{best:.4f}"
        else:
            best_str = str(best)

        self._status_label.setText(
            f"Done -- best loss: {best_str}, epochs: {ran}"
        )
        self._status_label.setStyleSheet("color: #6BCB77;")

        # Refresh checkpoint list
        ckpt_dir = self._config_editor.checkpoint_dir.path()
        if ckpt_dir:
            self._ckpt_manager.scan_directory(ckpt_dir)

        self._log_console.append(
            f"Training complete. Best loss={best_str}  Epochs ran={ran}"
        )
        if final_model:
            self._log_console.append(f"Final model: {final_model}")

        self._runner = None

    def _on_error(self, err_type: str, message: str):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._status_label.setText(f"ERROR: {err_type}")
        self._status_label.setStyleSheet("color: #FF6B6B;")
        self._log_console.append(f"ERROR [{err_type}] {message}")
        self._runner = None

    # ------------------------------------------------------------------
    # Config load / save / defaults
    # ------------------------------------------------------------------

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Training Config",
            "",
            "JSON (*.json);;All (*)",
        )
        if not path:
            return
        try:
            from CataractAI_Workbench.backend.training_adapter import TrainingAdapter

            adapter = TrainingAdapter(config_path=path)
            cfg = adapter.load_config()
            self._config_editor.set_config(cfg)
            self._log_console.append(f"Config loaded from {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Load error", str(exc))

    def _on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Config",
            "",
            "JSON (*.json);;All (*)",
        )
        if not path:
            return
        try:
            from CataractAI_Workbench.backend.training_adapter import TrainingAdapter

            cfg = self._config_editor.get_config()
            adapter = TrainingAdapter()
            adapter.save_config(cfg, path)
            self._log_console.append(f"Config saved to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save error", str(exc))

    def _on_defaults(self):
        from CataractAI_Workbench.backend.training_adapter import TrainingAdapter

        adapter = TrainingAdapter()
        self._config_editor.set_config(adapter.get_default_config())
        self._log_console.append("Reset to default config")

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _on_load_checkpoint(self, path: str):
        """Set the loaded checkpoint path in the config editor."""
        self._config_editor.checkpoint_path.set_path(path)
        self._log_console.append(f"Checkpoint selected: {Path(path).name}")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_initial_config(self):
        """Try to load the project's default TrainingSettings.json on tab init."""
        try:
            from CataractAI_Workbench.backend.training_adapter import TrainingAdapter

            adapter = TrainingAdapter()
            cfg = adapter.load_config()
            self._config_editor.set_config(cfg)

            # Auto-scan checkpoint dir
            ckpt_dir = cfg.get("checkpointing", {}).get("checkpoint_dir", "")
            if ckpt_dir and Path(ckpt_dir).is_dir():
                self._ckpt_manager.scan_directory(ckpt_dir)
        except FileNotFoundError:
            pass
