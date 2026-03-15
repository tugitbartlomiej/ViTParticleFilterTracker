"""Dialog for creating a new experiment session."""

from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QFont

from .session_manager import SessionManager


class NewSessionDialog(QDialog):
    """Modal dialog for creating a new experiment session."""

    def __init__(self, manager: SessionManager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self.setWindowTitle("New Experiment Session")
        self.setMinimumWidth(500)
        self.setMinimumHeight(450)
        self._topic_checkboxes: list[QCheckBox] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Create New Session")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #E6EDF3;")
        layout.addWidget(title)

        self._build_form(layout)
        self._build_topics_section(layout)
        self._build_objective_section(layout)
        self._build_dialog_buttons(layout)

    def _build_form(self, layout: QVBoxLayout) -> None:
        form = QFormLayout()
        form.setSpacing(8)

        self._desc_input = QLineEdit()
        self._desc_input.setPlaceholderText("Short description (becomes folder name)")
        self._desc_input.setFixedHeight(30)
        form.addRow("Description:", self._desc_input)

        self._type_combo = QComboBox()
        for session_type in self._manager.SESSION_TYPES:
            self._type_combo.addItem(session_type)
        self._type_combo.setFixedHeight(30)
        form.addRow("Type:", self._type_combo)

        layout.addLayout(form)

    def _build_topics_section(self, layout: QVBoxLayout) -> None:
        label = QLabel("Topics:")
        label.setStyleSheet("color: #D4D4D4; font-weight: bold;")
        layout.addWidget(label)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(8, 0, 0, 0)
        container_layout.setSpacing(2)

        columns = 3
        row_layout: Optional[QHBoxLayout] = None
        topics = self._manager.KNOWN_TOPICS

        for i, topic in enumerate(topics):
            if i % columns == 0:
                row_layout = QHBoxLayout()
                row_layout.setSpacing(12)
                container_layout.addLayout(row_layout)
            cb = QCheckBox(topic)
            cb.setStyleSheet("QCheckBox { color: #D4D4D4; }")
            self._topic_checkboxes.append(cb)
            if row_layout is not None:
                row_layout.addWidget(cb)

        # Fill remaining spots in last row with stretches
        if row_layout is not None:
            remaining = columns - (len(topics) % columns)
            if remaining < columns:
                for _ in range(remaining):
                    row_layout.addStretch()

        layout.addWidget(container)

    def _build_objective_section(self, layout: QVBoxLayout) -> None:
        label = QLabel("Objective:")
        label.setStyleSheet("color: #D4D4D4; font-weight: bold;")
        layout.addWidget(label)

        self._obj_edit = QTextEdit()
        self._obj_edit.setPlaceholderText(
            "What is the goal of this experiment session?"
        )
        self._obj_edit.setMaximumHeight(100)
        self._obj_edit.setStyleSheet(
            "QTextEdit { background-color: #0D1117; color: #E6EDF3; "
            "border: 1px solid #30363D; border-radius: 4px; padding: 6px; }"
        )
        layout.addWidget(self._obj_edit)

    def _build_dialog_buttons(self, layout: QVBoxLayout) -> None:
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Create Session")
        ok_btn.setStyleSheet(
            "QPushButton { background-color: #238636; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #2EA043; }"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> dict:
        """Return the form data as a dict."""
        topics = [cb.text() for cb in self._topic_checkboxes if cb.isChecked()]
        return {
            "description": self._desc_input.text().strip(),
            "type": self._type_combo.currentText(),
            "topics": topics,
            "objective": self._obj_edit.toPlainText().strip(),
        }
