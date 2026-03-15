"""File/directory picker widget with validation."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog,
)
from PyQt6.QtCore import pyqtSignal


class FilePicker(QWidget):
    """Line edit + browse button for selecting files or directories."""

    path_changed = pyqtSignal(str)

    def __init__(
        self,
        label: str = "Browse...",
        mode: str = "file",  # "file", "dir", "save"
        filter_str: str = "All Files (*)",
        parent=None,
    ):
        super().__init__(parent)
        self._mode = mode
        self._filter = filter_str

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText(f"Select {'directory' if mode == 'dir' else 'file'}...")
        self._edit.textChanged.connect(self._on_text_changed)

        self._btn = QPushButton(label)
        self._btn.setFixedWidth(80)
        self._btn.clicked.connect(self._browse)

        layout.addWidget(self._edit)
        layout.addWidget(self._btn)

    def _browse(self):
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select Directory", self._edit.text())
        elif self._mode == "save":
            path, _ = QFileDialog.getSaveFileName(self, "Save As", self._edit.text(), self._filter)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open File", self._edit.text(), self._filter)

        if path:
            self._edit.setText(path)

    def _on_text_changed(self, text: str):
        # Highlight invalid paths
        exists = Path(text).exists() if text else False
        color = "" if exists or not text else "border: 1px solid #FF6B6B;"
        self._edit.setStyleSheet(color)
        self.path_changed.emit(text)

    def path(self) -> str:
        return self._edit.text()

    def set_path(self, path: str):
        self._edit.setText(path)
