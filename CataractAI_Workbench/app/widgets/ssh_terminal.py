"""Simple SSH terminal widget with command input and history."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
)
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QTextCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from typing import Optional

from ..core.ssh_manager import SSHManager


class _CommandWorker(QThread):
    """Runs a single SSH command in the background."""

    output_ready = pyqtSignal(str, str, int)  # stdout, stderr, exit_code

    def __init__(self, ssh_manager: SSHManager, cmd: str, parent=None):
        super().__init__(parent)
        self._ssh = ssh_manager
        self._cmd = cmd

    def run(self):
        try:
            out, err, rc = self._ssh.execute(self._cmd, timeout=60)
            self.output_ready.emit(out, err, rc)
        except Exception as exc:
            self.output_ready.emit("", str(exc), -1)


class SSHTerminal(QWidget):
    """Interactive SSH terminal: monospace output pane + command line.

    Up/Down arrow keys cycle through command history.
    """

    MAX_HISTORY = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ssh_manager: Optional[SSHManager] = None
        self._history: list[str] = []
        self._history_idx = -1
        self._worker: Optional[_CommandWorker] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- output pane --
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 9))
        self._output.setStyleSheet(
            "QTextEdit {"
            "  background-color: #0C0C0C;"
            "  color: #CCCCCC;"
            "  border: 1px solid #3C3C3C;"
            "}"
        )
        layout.addWidget(self._output)

        # -- command input --
        self._input = QLineEdit()
        self._input.setFont(QFont("Consolas", 9))
        self._input.setPlaceholderText(
            "Type a command and press Enter (e.g. squeue, nvidia-smi)"
        )
        self._input.setStyleSheet(
            "QLineEdit {"
            "  background-color: #1A1A1A;"
            "  color: #CCCCCC;"
            "  border: 1px solid #3C3C3C;"
            "  padding: 4px;"
            "}"
        )
        self._input.returnPressed.connect(self._on_enter)
        self._input.installEventFilter(self)
        layout.addWidget(self._input)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_ssh_manager(self, ssh_manager: SSHManager):
        self._ssh_manager = ssh_manager
        self._append_text("Connected to Eden. Type commands below.\n", "#6BCB77")

    # ------------------------------------------------------------------
    # Event filter for Up/Down history
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self._input and event.type() == event.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Up:
                self._navigate_history(-1)
                return True
            if key == Qt.Key.Key_Down:
                self._navigate_history(1)
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_enter(self):
        cmd = self._input.text().strip()
        if not cmd:
            return

        if self._ssh_manager is None or not self._ssh_manager.is_connected():
            self._append_text("Not connected to Eden.\n", "#FF6B6B")
            return

        if self._worker is not None and self._worker.isRunning():
            self._append_text("Previous command still running...\n", "#FFD93D")
            return

        # Record history
        if not self._history or self._history[-1] != cmd:
            self._history.append(cmd)
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]
        self._history_idx = -1

        # Show prompt
        self._append_text(f"$ {cmd}\n", "#45AAF2")
        self._input.clear()

        # Execute
        self._worker = _CommandWorker(self._ssh_manager, cmd)
        self._worker.output_ready.connect(self._on_output)
        self._worker.start()

    @pyqtSlot(str, str, int)
    def _on_output(self, stdout: str, stderr: str, exit_code: int):
        if stdout:
            self._append_text(stdout)
            if not stdout.endswith("\n"):
                self._append_text("\n")
        if stderr:
            self._append_text(stderr, "#FF6B6B")
            if not stderr.endswith("\n"):
                self._append_text("\n")
        if exit_code != 0:
            self._append_text(
                f"[exit code {exit_code}]\n", "#FF6B6B"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_text(self, text: str, color: str = "#CCCCCC"):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text, fmt)
        self._output.setTextCursor(cursor)
        scrollbar = self._output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _navigate_history(self, direction: int):
        """Navigate command history. direction: -1 = older, +1 = newer."""
        if not self._history:
            return

        if self._history_idx == -1:
            if direction == -1:
                self._history_idx = len(self._history) - 1
            else:
                return
        else:
            self._history_idx += direction

        if self._history_idx < 0:
            self._history_idx = 0
        elif self._history_idx >= len(self._history):
            self._history_idx = -1
            self._input.clear()
            return

        self._input.setText(self._history[self._history_idx])
