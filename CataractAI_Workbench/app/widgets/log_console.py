"""Scrolling log console widget with syntax coloring."""

from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QWidget, QHBoxLayout, QPushButton
from PyQt6.QtGui import QTextCharFormat, QColor, QFont
from PyQt6.QtCore import Qt, pyqtSlot
import re


class LogConsole(QWidget):
    """Log console with colored output, auto-scroll, and line limit."""

    MAX_LINES = 5000

    # Color mapping for log levels
    COLORS = {
        "ERROR": QColor("#FF6B6B"),
        "WARNING": QColor("#FFD93D"),
        "INFO": QColor("#6BCB77"),
        "DEBUG": QColor("#A8A8A8"),
        "CRITICAL": QColor("#FF4444"),
    }

    _LEVEL_RE = re.compile(r"\b(ERROR|WARNING|INFO|DEBUG|CRITICAL)\b")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._auto_scroll = True
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar
        controls = QHBoxLayout()
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setFixedWidth(60)
        self._btn_clear.clicked.connect(self.clear)

        self._btn_scroll = QPushButton("Auto-scroll: ON")
        self._btn_scroll.setFixedWidth(120)
        self._btn_scroll.clicked.connect(self._toggle_auto_scroll)

        controls.addStretch()
        controls.addWidget(self._btn_scroll)
        controls.addWidget(self._btn_clear)
        layout.addLayout(controls)

        # Text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Consolas", 9))
        self._text.setStyleSheet(
            "QTextEdit { background-color: #1E1E1E; color: #D4D4D4; border: 1px solid #3C3C3C; }"
        )
        layout.addWidget(self._text)

    @pyqtSlot(str)
    def append(self, text: str):
        """Append a line to the log with color detection."""
        fmt = QTextCharFormat()

        match = self._LEVEL_RE.search(text)
        if match:
            color = self.COLORS.get(match.group(1), QColor("#D4D4D4"))
            fmt.setForeground(color)
        else:
            fmt.setForeground(QColor("#D4D4D4"))

        cursor = self._text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text + "\n", fmt)

        # Enforce line limit
        doc = self._text.document()
        if doc.blockCount() > self.MAX_LINES:
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(
                cursor.MoveOperation.Down,
                cursor.MoveMode.KeepAnchor,
                doc.blockCount() - self.MAX_LINES,
            )
            cursor.removeSelectedText()

        if self._auto_scroll:
            scrollbar = self._text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(str, str)
    def append_with_source(self, source: str, message: str):
        """Append log with source prefix."""
        self.append(f"[{source}] {message}")

    def clear(self):
        """Clear all log content."""
        self._text.clear()

    def _toggle_auto_scroll(self):
        self._auto_scroll = not self._auto_scroll
        state = "ON" if self._auto_scroll else "OFF"
        self._btn_scroll.setText(f"Auto-scroll: {state}")
