"""Main application window with tabbed interface."""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QLabel, QMenuBar, QWidget,
    QVBoxLayout,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction

from .core.signal_bus import get_signal_bus
from .widgets.log_console import LogConsole


class PlaceholderTab(QWidget):
    """Temporary placeholder for tabs not yet implemented."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(f"{name}\n\nModule ready for implementation.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #7F7F7F; font-size: 14px;")
        layout.addWidget(label)


class MainWindow(QMainWindow):
    """Main application window with 6 tabs and status bar."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CataractAI Workbench")
        self.setMinimumSize(1200, 800)
        self._signal_bus = get_signal_bus()

        self._init_menu()
        self._init_tabs()
        self._init_status_bar()
        self._init_timers()
        self._connect_signals()

    def _init_menu(self):
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")
        open_action = QAction("&Open Config...", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menu.addMenu("&Tools")
        clear_cache = QAction("Clear &Cache", self)
        tools_menu.addAction(clear_cache)

        # Help menu
        help_menu = menu.addMenu("&Help")
        about_action = QAction("&About", self)
        help_menu.addAction(about_action)

    def _init_tabs(self):
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)

        # Import real tab implementations (with fallback to placeholders)
        try:
            from .tabs.dataset_selection.tab_widget import DatasetSelectionTab
            self._tab_dataset = DatasetSelectionTab()
        except Exception:
            self._tab_dataset = PlaceholderTab("Dataset Selection")

        try:
            from .tabs.training.tab_widget import TrainingTab
            self._tab_training = TrainingTab()
        except Exception:
            self._tab_training = PlaceholderTab("Training")

        try:
            from .tabs.benchmarks.tab_widget import BenchmarksTab
            self._tab_benchmarks = BenchmarksTab()
        except Exception:
            self._tab_benchmarks = PlaceholderTab("Benchmarks")

        try:
            from .tabs.eden.tab_widget import EdenTab
            self._tab_eden = EdenTab()
        except Exception:
            self._tab_eden = PlaceholderTab("Eden HPC")

        try:
            from .tabs.visualization.tab_widget import VisualizationTab
            self._tab_viz = VisualizationTab()
        except Exception:
            self._tab_viz = PlaceholderTab("Visualization")

        try:
            from .tabs.experiments.tab_widget import ExperimentsTab
            self._tab_experiments = ExperimentsTab()
        except Exception:
            self._tab_experiments = PlaceholderTab("Experiments")

        self._tabs.addTab(self._tab_dataset, "Dataset Selection")
        self._tabs.addTab(self._tab_training, "Training")
        self._tabs.addTab(self._tab_benchmarks, "Benchmarks")
        self._tabs.addTab(self._tab_eden, "Eden HPC")
        self._tabs.addTab(self._tab_viz, "Visualization")
        self._tabs.addTab(self._tab_experiments, "Experiments")

        self.setCentralWidget(self._tabs)

    def _init_status_bar(self):
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._gpu_label = QLabel("GPU: detecting...")
        self._eden_label = QLabel("Eden: Disconnected")
        self._job_label = QLabel("")

        self._status_bar.addPermanentWidget(self._gpu_label)
        self._status_bar.addPermanentWidget(QLabel(" | "))
        self._status_bar.addPermanentWidget(self._eden_label)
        self._status_bar.addPermanentWidget(QLabel(" | "))
        self._status_bar.addPermanentWidget(self._job_label)

    def _init_timers(self):
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu_status)
        self._gpu_timer.start(5000)
        # Initial update
        self._update_gpu_status()

    def _connect_signals(self):
        bus = self._signal_bus
        bus.status_message.connect(self._status_bar.showMessage)
        bus.ssh_connected.connect(lambda h: self._eden_label.setText(f"Eden: Connected ({h})"))
        bus.ssh_disconnected.connect(lambda _: self._eden_label.setText("Eden: Disconnected"))
        bus.job_status_changed.connect(
            lambda jid, st: self._job_label.setText(f"Job #{jid}: {st}")
        )

    def _update_gpu_status(self):
        try:
            import psutil
            # Try pynvml for NVIDIA GPU info
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_gb = mem.used / (1024 ** 3)
                total_gb = mem.total / (1024 ** 3)
                self._gpu_label.setText(f"GPU: {name} | VRAM: {used_gb:.1f}/{total_gb:.0f}GB")
                pynvml.nvmlShutdown()
            except Exception:
                self._gpu_label.setText(f"GPU: N/A | CPU: {psutil.cpu_percent():.0f}%")
        except ImportError:
            self._gpu_label.setText("GPU: psutil not installed")
