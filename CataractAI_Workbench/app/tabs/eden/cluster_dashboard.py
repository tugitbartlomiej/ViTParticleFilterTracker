"""Cluster dashboard showing Eden HPC node status in a tree view."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QLabel,
    QHeaderView,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal


# State -> (foreground colour, tooltip)
_STATE_COLORS: dict[str, QColor] = {
    "idle": QColor("#6BCB77"),
    "mixed": QColor("#FFD93D"),
    "allocated": QColor("#FF6B6B"),
    "down": QColor("#7F7F7F"),
    "drained": QColor("#7F7F7F"),
    "draining": QColor("#FF9F43"),
    "completing": QColor("#45AAF2"),
}


class ClusterDashboard(QWidget):
    """QTreeWidget-based view of cluster nodes with colour-coded state."""

    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- toolbar --
        toolbar = QHBoxLayout()
        self._lbl_summary = QLabel("Nodes: --")
        toolbar.addWidget(self._lbl_summary)
        toolbar.addStretch()

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(90)
        btn_refresh.clicked.connect(self.refresh_requested.emit)
        toolbar.addWidget(btn_refresh)
        layout.addLayout(toolbar)

        # -- tree --
        self._tree = QTreeWidget()
        self._tree.setColumnCount(6)
        self._tree.setHeaderLabels(
            ["Node", "Partition", "GPUs", "CPUs", "Memory (MB)", "State"]
        )
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(False)

        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._tree)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_info(self, cluster_info: dict):
        """Populate the tree from *cluster_info* returned by
        :meth:`EdenClient.get_cluster_info`.
        """
        self._tree.clear()
        nodes = cluster_info.get("nodes", [])

        for node in nodes:
            item = QTreeWidgetItem(
                [
                    str(node.get("name", "")),
                    str(node.get("partition", "")),
                    str(node.get("gpus", "")),
                    str(node.get("cpus", "")),
                    str(node.get("memory", "")),
                    str(node.get("state", "")),
                ]
            )

            # Colour by state
            state = str(node.get("state", "")).lower().split("+")[0]
            color = _STATE_COLORS.get(state, QColor("#D4D4D4"))
            for col in range(6):
                item.setForeground(col, color)

            self._tree.addTopLevelItem(item)

        self._lbl_summary.setText(f"Nodes: {len(nodes)}")
