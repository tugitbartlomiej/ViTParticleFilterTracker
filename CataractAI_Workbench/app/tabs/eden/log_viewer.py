"""Live log viewer with streaming from Eden via SSH."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
)
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot

from ...core.log_parser import LogParser
from ...core.ssh_manager import SSHManager
from ...widgets.log_console import LogConsole


class _StreamWorker(QThread):
    """Background thread that streams ``tail -f`` output line-by-line."""

    line_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_streaming = pyqtSignal()

    def __init__(self, ssh_manager: SSHManager, log_path: str, parent=None):
        super().__init__(parent)
        self._ssh = ssh_manager
        self._log_path = log_path
        self._running = True

    def run(self):
        try:
            for line in self._ssh.stream(f"tail -f {self._log_path}"):
                if not self._running:
                    break
                self.line_received.emit(line)
        except Exception as exc:
            if self._running:
                self.error_occurred.emit(str(exc))
        finally:
            self.finished_streaming.emit()

    def stop(self):
        self._running = False


class LogViewer(QWidget):
    """Widget for live log streaming from Eden with parsed metric colours."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ssh_manager: SSHManager | None = None
        self._worker: _StreamWorker | None = None
        self._parser = LogParser()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- path / controls --
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Log path:"))

        self._edit_path = QLineEdit()
        self._edit_path.setPlaceholderText(
            "/mnt/evafs/faculty/home/bpiotrowski/DETR/logs/slurm-XXXXXX.out"
        )
        controls.addWidget(self._edit_path)

        self._btn_start = QPushButton("Start")
        self._btn_start.setFixedWidth(80)
        self._btn_start.clicked.connect(self._on_start)
        controls.addWidget(self._btn_start)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setFixedWidth(80)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        controls.addWidget(self._btn_stop)

        layout.addLayout(controls)

        # -- status --
        self._lbl_status = QLabel("Idle")
        self._lbl_status.setStyleSheet("color: #7F7F7F;")
        layout.addWidget(self._lbl_status)

        # -- log console --
        self._console = LogConsole()
        layout.addWidget(self._console)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_ssh_manager(self, ssh_manager: SSHManager):
        """Assign the SSH connection used for streaming."""
        self._ssh_manager = ssh_manager

    def start_streaming(self, log_path: str):
        """Begin streaming *log_path* from Eden."""
        self._edit_path.setText(log_path)
        self._on_start()

    def stop_streaming(self):
        """Stop the current streaming session."""
        self._on_stop()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_start(self):
        if self._ssh_manager is None or not self._ssh_manager.is_connected():
            self._lbl_status.setText("Not connected to Eden")
            self._lbl_status.setStyleSheet("color: #FF6B6B;")
            return

        log_path = self._edit_path.text().strip()
        if not log_path:
            return

        self.stop_streaming()

        self._worker = _StreamWorker(self._ssh_manager, log_path)
        self._worker.line_received.connect(self._on_line)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished_streaming.connect(self._on_finished)
        self._worker.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_status.setText(f"Streaming: {log_path}")
        self._lbl_status.setStyleSheet("color: #6BCB77;")

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.quit()
            self._worker.wait(3000)
            self._worker = None

        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_status.setText("Stopped")
        self._lbl_status.setStyleSheet("color: #7F7F7F;")

    @pyqtSlot(str)
    def _on_line(self, line: str):
        # Parse for metrics and annotate
        info = self._parser.parse_line(line)
        if info is not None:
            fmt_type = info.get("type", "").upper()
            parts = [f"[{fmt_type}]"]
            for key in ("epoch", "avg_loss", "box_loss", "cls_loss", "mAP50_95", "lr"):
                if key in info:
                    parts.append(f"{key}={info[key]}")
            self._console.append(" ".join(parts))
        else:
            self._console.append(line)

    @pyqtSlot(str)
    def _on_error(self, msg: str):
        self._console.append(f"ERROR streaming: {msg}")
        self._lbl_status.setText(f"Error: {msg}")
        self._lbl_status.setStyleSheet("color: #FF6B6B;")

    @pyqtSlot()
    def _on_finished(self):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._lbl_status.setText("Stream ended")
        self._lbl_status.setStyleSheet("color: #7F7F7F;")
