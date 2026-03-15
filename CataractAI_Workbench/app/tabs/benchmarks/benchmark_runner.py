"""Widget for configuring and running benchmark scripts via QProcess."""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import pyqtSignal, QProcess
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)


class BenchmarkRunner(QWidget):
    """Configure and launch a benchmark script in a subprocess.

    Signals
    -------
    benchmark_finished(dict)
        Emitted when the benchmark completes.  The dict contains at least
        ``{"exit_code": int, "script": str}``.
    output_line(str)
        Emitted for every stdout/stderr line produced by the child process.
    """

    benchmark_finished = pyqtSignal(dict)
    output_line = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._adapter = None  # lazy
        self._scripts: List[dict] = []
        self._current_args: List[dict] = []
        self._arg_widgets: Dict[str, QWidget] = {}
        self._process: Optional[QProcess] = None

        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -- Script selector ------------------------------------------------
        header = QHBoxLayout()
        header.addWidget(QLabel("Script:"))
        self._combo_scripts = QComboBox()
        self._combo_scripts.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._combo_scripts.currentIndexChanged.connect(self._on_script_changed)
        header.addWidget(self._combo_scripts)

        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setFixedWidth(70)
        self._btn_refresh.clicked.connect(self.refresh_scripts)
        header.addWidget(self._btn_refresh)

        layout.addLayout(header)

        # -- Description label ----------------------------------------------
        self._lbl_description = QLabel("")
        self._lbl_description.setWordWrap(True)
        self._lbl_description.setStyleSheet("color: #A8A8A8; font-style: italic;")
        layout.addWidget(self._lbl_description)

        # -- Dynamic argument form (scrollable) -----------------------------
        self._args_group = QGroupBox("Arguments")
        self._args_form = QFormLayout()
        self._args_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self._args_group.setLayout(self._args_form)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._args_group)
        layout.addWidget(scroll, stretch=1)

        # -- Controls -------------------------------------------------------
        controls = QHBoxLayout()
        self._btn_run = QPushButton("Run")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #2D7D46; color: white; font-weight: bold; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #3A9D5A; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._btn_run.clicked.connect(self.run_benchmark)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet(
            "QPushButton { background-color: #8B2020; color: white; font-weight: bold; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #B03030; }"
            "QPushButton:disabled { background-color: #555; color: #999; }"
        )
        self._btn_stop.clicked.connect(self.stop_benchmark)

        controls.addWidget(self._btn_run)
        controls.addWidget(self._btn_stop)
        controls.addStretch()
        layout.addLayout(controls)

        # -- Progress bar ---------------------------------------------------
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Running...")
        layout.addWidget(self._progress)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_scripts(self):
        """Re-scan the scripts directory and populate the combo box."""
        adapter = self._get_adapter()
        self._scripts = adapter.discover_scripts()

        self._combo_scripts.blockSignals(True)
        self._combo_scripts.clear()
        for s in self._scripts:
            self._combo_scripts.addItem(s["name"], s)
        self._combo_scripts.blockSignals(False)

        if self._scripts:
            self._combo_scripts.setCurrentIndex(0)
            self._on_script_changed(0)

    def set_script(self, script_info: dict):
        """Populate the argument form from *script_info*."""
        self._lbl_description.setText(script_info.get("description", ""))

        # Clear previous form
        self._clear_form()

        # Fetch arguments
        adapter = self._get_adapter()
        self._current_args = adapter.get_script_args(script_info["path"])

        for arg in self._current_args:
            widget = self._create_arg_widget(arg)
            label_text = f"--{arg['name']}"
            if arg.get("required"):
                label_text += " *"
            self._args_form.addRow(label_text, widget)
            self._arg_widgets[arg["name"]] = widget

    def run_benchmark(self):
        """Start the currently selected benchmark script as a QProcess."""
        idx = self._combo_scripts.currentIndex()
        if idx < 0 or idx >= len(self._scripts):
            self.output_line.emit("ERROR: No benchmark script selected.")
            return

        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self.output_line.emit("WARNING: A benchmark is already running.")
            return

        script_info = self._scripts[idx]
        args = self._collect_args()

        adapter = self._get_adapter()
        self._process = adapter.run_benchmark(
            script_path=script_info["path"],
            args=args,
            on_output=self._on_process_output,
            parent=self,
        )
        self._process.finished.connect(self._on_process_finished)

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress.setVisible(True)

        self.output_line.emit(f"INFO: Started benchmark: {script_info['name']}")

    def stop_benchmark(self):
        """Kill the running QProcess."""
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self.output_line.emit("WARNING: Stopping benchmark...")
            self._process.kill()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_adapter(self):
        if self._adapter is None:
            from CataractAI_Workbench.backend.benchmark_adapter import BenchmarkAdapter
            self._adapter = BenchmarkAdapter()
        return self._adapter

    def _on_script_changed(self, index: int):
        if 0 <= index < len(self._scripts):
            self.set_script(self._scripts[index])

    def _clear_form(self):
        """Remove all widgets from the argument form."""
        while self._args_form.rowCount() > 0:
            self._args_form.removeRow(0)
        self._arg_widgets.clear()

    def _create_arg_widget(self, arg: dict) -> QWidget:
        """Create the appropriate input widget for an argument."""
        choices = arg.get("choices")
        if choices:
            combo = QComboBox()
            combo.addItems(choices)
            default = arg.get("default")
            if default and default in choices:
                combo.setCurrentText(default)
            return combo

        arg_type = arg.get("type", "str")

        if arg_type == "int":
            spin = QSpinBox()
            spin.setRange(-999999, 999999)
            spin.setSpecialValueText("")
            default = arg.get("default")
            if default is not None:
                try:
                    spin.setValue(int(default))
                except (ValueError, TypeError):
                    pass
            return spin

        if arg_type == "float":
            dspin = QDoubleSpinBox()
            dspin.setRange(-1e12, 1e12)
            dspin.setDecimals(6)
            default = arg.get("default")
            if default is not None:
                try:
                    dspin.setValue(float(default))
                except (ValueError, TypeError):
                    pass
            return dspin

        # Default: line edit
        edit = QLineEdit()
        default = arg.get("default")
        if default is not None and default.lower() != "none":
            edit.setText(str(default))
        placeholder = arg.get("help", "")
        if placeholder:
            edit.setPlaceholderText(placeholder[:80])
        return edit

    def _collect_args(self) -> Dict[str, str]:
        """Collect current argument values from the form widgets."""
        args: Dict[str, str] = {}
        for arg in self._current_args:
            name = arg["name"]
            widget = self._arg_widgets.get(name)
            if widget is None:
                continue

            if isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QSpinBox):
                value = str(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                value = str(widget.value())
            elif isinstance(widget, QLineEdit):
                value = widget.text().strip()
            else:
                continue

            if value:
                args[name] = value

        return args

    def _on_process_output(self, line: str):
        """Forward process output to the signal."""
        self.output_line.emit(line)

    def _on_process_finished(self, exit_code: int, exit_status):
        """Handle process completion."""
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)

        idx = self._combo_scripts.currentIndex()
        script_name = self._scripts[idx]["name"] if 0 <= idx < len(self._scripts) else "unknown"

        status = "OK" if exit_code == 0 else f"FAILED (code {exit_code})"
        self.output_line.emit(f"INFO: Benchmark finished: {script_name} -- {status}")

        result = {
            "exit_code": exit_code,
            "script": script_name,
        }

        # Try to auto-detect result files from the most recent benchmark directory
        if exit_code == 0:
            adapter = self._get_adapter()
            runs = adapter.list_previous_runs()
            if runs:
                latest = runs[0]
                if latest.get("result_files"):
                    try:
                        data = adapter.load_results(latest["result_files"][0])
                        result["data"] = data
                        result["result_path"] = latest["result_files"][0]
                        result["run_dir"] = latest["path"]
                    except Exception:
                        pass

        self.benchmark_finished.emit(result)
