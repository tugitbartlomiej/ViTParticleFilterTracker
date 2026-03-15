"""Widget for browsing and managing model checkpoints."""

from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt, QFileSystemWatcher
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class CheckpointManager(QWidget):
    """Tree view of ``.pth`` checkpoint files with context menu actions."""

    load_checkpoint = pyqtSignal(str)  # full path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_dir: str = ""
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_dir_changed)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setFixedWidth(70)
        self._btn_refresh.clicked.connect(self._refresh)
        toolbar.addStretch()
        toolbar.addWidget(self._btn_refresh)
        layout.addLayout(toolbar)

        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Filename", "Size (MB)", "Date", "Epoch"])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        self._tree.setStyleSheet(
            "QTreeWidget { background-color: #1E1E1E; color: #D4D4D4; "
            "border: 1px solid #3C3C3C; alternate-background-color: #252526; }"
            "QHeaderView::section { background-color: #2D2D2D; color: #D4D4D4; "
            "border: 1px solid #3C3C3C; padding: 3px; }"
        )

        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._tree)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_directory(self, path: str) -> None:
        """Scan *path* for .pth files and populate the tree."""
        # Update watcher
        if self._current_dir and self._current_dir in self._watcher.directories():
            self._watcher.removePath(self._current_dir)
        self._current_dir = path
        if Path(path).is_dir():
            self._watcher.addPath(path)
        self._refresh()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self):
        self._tree.clear()
        d = Path(self._current_dir)
        if not d.is_dir():
            return

        for p in sorted(d.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = p.stat()
            size_mb = f"{stat.st_size / (1024 * 1024):.1f}"

            from datetime import datetime
            date_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            # Try to parse epoch from filename
            epoch_str = ""
            for part in p.stem.split("_"):
                if part.startswith("epoch"):
                    epoch_str = part.replace("epoch", "")
                    break

            item = QTreeWidgetItem([p.name, size_mb, date_str, epoch_str])
            item.setData(0, Qt.ItemDataRole.UserRole, str(p))
            self._tree.addTopLevelItem(item)

    def _on_dir_changed(self, _path: str):
        self._refresh()

    def _on_double_click(self, item: QTreeWidgetItem, _col: int):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            self.load_checkpoint.emit(path)

    def _show_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if not item:
            return

        path = item.data(0, Qt.ItemDataRole.UserRole)
        menu = QMenu(self)

        act_load = QAction("Load checkpoint", self)
        act_load.triggered.connect(lambda: self.load_checkpoint.emit(path))
        menu.addAction(act_load)

        act_copy = QAction("Copy path", self)
        act_copy.triggered.connect(
            lambda: QApplication.clipboard().setText(path)
        )
        menu.addAction(act_copy)

        menu.addSeparator()

        act_delete = QAction("Delete", self)
        act_delete.triggered.connect(lambda: self._delete_checkpoint(path))
        menu.addAction(act_delete)

        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _delete_checkpoint(self, path: str):
        reply = QMessageBox.question(
            self,
            "Delete checkpoint",
            f"Delete {Path(path).name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                Path(path).unlink()
                self._refresh()
            except Exception as exc:
                QMessageBox.warning(self, "Error", str(exc))
