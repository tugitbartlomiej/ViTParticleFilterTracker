"""SLURM job manager widget with submit, cancel, and refresh controls."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QMenu,
    QMessageBox,
    QLabel,
    QAbstractItemView,
)
from PyQt6.QtGui import QColor, QAction
from PyQt6.QtCore import Qt, pyqtSignal
from typing import List, Optional


_STATUS_COLORS: dict[str, QColor] = {
    "RUNNING": QColor("#6BCB77"),
    "PENDING": QColor("#FFD93D"),
    "COMPLETED": QColor("#7F7F7F"),
    "COMPLETING": QColor("#45AAF2"),
    "FAILED": QColor("#FF6B6B"),
    "CANCELLED": QColor("#FF9F43"),
    "TIMEOUT": QColor("#FF6B6B"),
}


class JobManager(QWidget):
    """Table-based view of SLURM jobs with submit / cancel / refresh."""

    refresh_requested = pyqtSignal()
    submit_requested = pyqtSignal(str)       # script path
    cancel_requested = pyqtSignal(str)       # job id
    view_log_requested = pyqtSignal(str)     # job id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: List[dict] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- toolbar --
        toolbar = QHBoxLayout()
        self._lbl_count = QLabel("Jobs: --")
        toolbar.addWidget(self._lbl_count)
        toolbar.addStretch()

        btn_submit = QPushButton("Submit Job")
        btn_submit.setFixedWidth(100)
        btn_submit.clicked.connect(self._on_submit)
        toolbar.addWidget(btn_submit)

        btn_cancel = QPushButton("Cancel Selected")
        btn_cancel.setFixedWidth(120)
        btn_cancel.clicked.connect(self._on_cancel_selected)
        toolbar.addWidget(btn_cancel)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(90)
        btn_refresh.clicked.connect(self.refresh_requested.emit)
        toolbar.addWidget(btn_refresh)

        layout.addLayout(toolbar)

        # -- table --
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["Job ID", "Name", "Partition", "Status", "Time", "Node", "GPUs"]
        )
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._table.customContextMenuRequested.connect(self._show_context_menu)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._table)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update_jobs(self, jobs: List[dict]):
        """Replace the table contents with *jobs*."""
        self._jobs = jobs
        self._table.setRowCount(len(jobs))

        for row, job in enumerate(jobs):
            values = [
                job.get("job_id", ""),
                job.get("name", ""),
                job.get("partition", ""),
                job.get("state", ""),
                job.get("time", ""),
                job.get("node", ""),
                job.get("gpus", ""),
            ]
            status = str(values[3]).upper()
            color = _STATUS_COLORS.get(status, QColor("#D4D4D4"))

            for col, val in enumerate(values):
                item = QTableWidgetItem(str(val))
                item.setForeground(color)
                self._table.setItem(row, col, item)

        self._lbl_count.setText(f"Jobs: {len(jobs)}")

    def selected_job_id(self) -> Optional[str]:
        """Return the job ID of the currently selected row, or None."""
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        item = self._table.item(row, 0)
        return item.text() if item else None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_submit(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SLURM batch script",
            "",
            "Shell Scripts (*.sh *.slurm);;All Files (*)",
        )
        if path:
            self.submit_requested.emit(path)

    def _on_cancel_selected(self):
        job_id = self.selected_job_id()
        if job_id is None:
            return
        reply = QMessageBox.question(
            self,
            "Cancel Job",
            f"Cancel job {job_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.cancel_requested.emit(job_id)

    def _show_context_menu(self, pos):
        job_id = self.selected_job_id()
        if job_id is None:
            return

        menu = QMenu(self)

        act_cancel = QAction("Cancel Job", self)
        act_cancel.triggered.connect(lambda: self.cancel_requested.emit(job_id))
        menu.addAction(act_cancel)

        act_log = QAction("View Log", self)
        act_log.triggered.connect(lambda: self.view_log_requested.emit(job_id))
        menu.addAction(act_log)

        act_copy = QAction("Copy Job ID", self)
        act_copy.triggered.connect(
            lambda: self._copy_to_clipboard(job_id)
        )
        menu.addAction(act_copy)

        menu.exec(self._table.viewport().mapToGlobal(pos))

    @staticmethod
    def _copy_to_clipboard(text: str):
        from PyQt6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)
