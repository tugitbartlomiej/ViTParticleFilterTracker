"""Browser widget for remote Claude SSH sessions on Eden."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal
from typing import Optional

from ...core.ssh_manager import SSHManager


_SESSION_DIRS = [
    "/mnt/evafs/faculty/home/bpiotrowski/ClaudeSshSession",
    "/mnt/evafs/faculty/home/bpiotrowski/.claude/sessions",
    "/mnt/evafs/faculty/home/bpiotrowski/sessions",
]


class SessionBrowser(QWidget):
    """Tree view that lists remote session directories on Eden.

    Emits :pyqtSignal:`session_selected(str)` with the log file path
    when the user double-clicks a session entry.
    """

    session_selected = pyqtSignal(str)   # remote log path
    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ssh_manager: Optional[SSHManager] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Sessions")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Name", "Size"])
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemDoubleClicked.connect(self._on_double_click)

        tree_header = self._tree.header()
        tree_header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        tree_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self._tree)

        btn = QPushButton("Refresh Sessions")
        btn.clicked.connect(self._load_sessions)
        layout.addWidget(btn)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def set_ssh_manager(self, ssh_manager: SSHManager):
        self._ssh_manager = ssh_manager
        self._load_sessions()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_sessions(self):
        """Scan known remote directories for session folders."""
        self._tree.clear()

        if self._ssh_manager is None or not self._ssh_manager.is_connected():
            return

        for session_dir in _SESSION_DIRS:
            out, err, rc = self._ssh_manager.execute(
                f"test -d {session_dir} && echo EXISTS", timeout=10
            )
            if "EXISTS" not in out:
                continue

            try:
                entries = self._ssh_manager.list_dir(session_dir)
            except Exception:
                continue

            root_item = QTreeWidgetItem([session_dir.split("/")[-1], ""])
            self._tree.addTopLevelItem(root_item)

            for entry in entries:
                if entry["is_dir"]:
                    child = QTreeWidgetItem(
                        [entry["name"], ""]
                    )
                    child.setData(
                        0, 0x0100, f"{session_dir}/{entry['name']}"
                    )
                    root_item.addChild(child)

                    # Try to list log files inside
                    try:
                        subdir = f"{session_dir}/{entry['name']}"
                        sub_entries = self._ssh_manager.list_dir(subdir)
                        for se in sub_entries:
                            if not se["is_dir"]:
                                size_str = self._format_size(se["size"])
                                log_item = QTreeWidgetItem(
                                    [se["name"], size_str]
                                )
                                log_item.setData(
                                    0,
                                    0x0100,
                                    f"{subdir}/{se['name']}",
                                )
                                child.addChild(log_item)
                    except Exception:
                        pass

            root_item.setExpanded(True)

    def _on_double_click(self, item: QTreeWidgetItem, column: int):
        """Emit the remote path when a leaf (file) is double-clicked."""
        remote_path = item.data(0, 0x0100)
        if remote_path and item.childCount() == 0:
            self.session_selected.emit(remote_path)

    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        if size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        return f"{size / (1024 * 1024 * 1024):.1f} GB"
