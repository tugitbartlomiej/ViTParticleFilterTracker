"""Multi-task progress panel widget."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame,
)
from PyQt6.QtCore import pyqtSlot


class ProgressPanel(QWidget):
    """Panel showing multiple named progress bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bars: dict[str, tuple[QLabel, QProgressBar, QLabel]] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

    def add_task(self, task_id: str, label: str = ""):
        """Add a named progress bar."""
        if task_id in self._bars:
            return

        row = QHBoxLayout()
        name_label = QLabel(label or task_id)
        name_label.setFixedWidth(150)
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(True)
        status_label = QLabel("")
        status_label.setFixedWidth(200)

        row.addWidget(name_label)
        row.addWidget(bar)
        row.addWidget(status_label)

        container = QFrame()
        container.setLayout(row)
        self._layout.addWidget(container)
        self._bars[task_id] = (name_label, bar, status_label)

    @pyqtSlot(str, int, str)
    def update_task(self, task_id: str, percent: int, message: str = ""):
        """Update a task's progress."""
        if task_id not in self._bars:
            self.add_task(task_id)
        _, bar, status = self._bars[task_id]
        bar.setValue(min(percent, 100))
        if message:
            status.setText(message)

    def remove_task(self, task_id: str):
        """Remove a progress bar."""
        if task_id in self._bars:
            label, bar, status = self._bars.pop(task_id)
            # Find and remove the container frame
            for i in range(self._layout.count()):
                item = self._layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    layout = widget.layout()
                    if layout and layout.indexOf(bar) >= 0:
                        self._layout.removeWidget(widget)
                        widget.deleteLater()
                        break

    def clear(self):
        """Remove all progress bars."""
        for task_id in list(self._bars.keys()):
            self.remove_task(task_id)
