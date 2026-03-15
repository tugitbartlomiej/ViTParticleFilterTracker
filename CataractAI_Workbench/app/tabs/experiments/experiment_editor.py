"""Rich experiment documentation editor with read/edit modes and markdown rendering."""

import shutil
from typing import Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from .markdown_renderer import markdown_to_html
from .session_manager import SessionManager
from .topic_tags_widget import TopicTagsBar


# Style constants
_STYLE_BTN_EDIT = (
    "QPushButton { background-color: #238636; color: white; font-weight: bold; "
    "border-radius: 4px; }"
    "QPushButton:hover { background-color: #2EA043; }"
    "QPushButton:disabled { background-color: #21262D; color: #484F58; }"
)
_STYLE_BTN_CANCEL = (
    "QPushButton { background-color: #DA3633; color: white; "
    "font-weight: bold; border-radius: 4px; }"
    "QPushButton:hover { background-color: #F85149; }"
)
_STYLE_BTN_SAVE = (
    "QPushButton { background-color: #1F6FEB; color: white; font-weight: bold; "
    "border-radius: 4px; }"
    "QPushButton:hover { background-color: #388BFD; }"
)
_STYLE_BTN_EXPORT = (
    "QPushButton { background-color: #2D2D2D; color: #D4D4D4; "
    "border-radius: 4px; border: 1px solid #3C3C3C; }"
    "QPushButton:hover { background-color: #3C3C3C; }"
    "QPushButton:disabled { background-color: #21262D; color: #484F58; }"
)
_STYLE_BTN_DELETE = (
    "QPushButton { background-color: #DA3633; color: white; font-weight: bold; "
    "border-radius: 4px; }"
    "QPushButton:hover { background-color: #F85149; }"
    "QPushButton:disabled { background-color: #21262D; color: #484F58; }"
)

_SESSION_TYPE_COLORS: dict[str, str] = {
    "training": "#2EA043",
    "benchmark": "#388BFD",
    "analysis": "#8957E5",
    "ssh": "#F0883E",
    "writing": "#D29922",
    "fix": "#DA3633",
    "debug": "#DA3633",
    "development": "#58A6FF",
    "mixed": "#6E7681",
}


def _type_color(session_type: str) -> str:
    """Return a badge color for the given session type."""
    lower = session_type.lower()
    for keyword, color in _SESSION_TYPE_COLORS.items():
        if keyword in lower:
            return color
    return "#8B949E"


class ExperimentEditor(QWidget):
    """Widget for viewing and editing a single experiment session.

    Features:
    - Header with session name, type badge, and status
    - Topics displayed as colored tag buttons
    - Read mode: renders markdown as HTML in a QTextBrowser
    - Edit mode: plain text editing in a QTextEdit
    - Save / Export / Delete actions
    """

    session_saved = pyqtSignal(str)   # dir_name
    session_deleted = pyqtSignal(str)  # dir_name

    def __init__(self, session_manager: SessionManager, parent=None):
        super().__init__(parent)
        self._manager = session_manager
        self._current_dir: Optional[str] = None
        self._editing = False
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._build_header(layout)
        self._build_metadata_row(layout)
        self._topics_bar = TopicTagsBar()
        layout.addWidget(self._topics_bar)
        self._build_content_area(layout)
        self._build_action_buttons(layout)

    def _build_header(self, layout: QVBoxLayout) -> None:
        header = QHBoxLayout()
        header.setSpacing(8)
        self._title_label = QLabel("No session selected")
        self._title_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self._title_label.setStyleSheet("color: #E6EDF3;")
        self._title_label.setWordWrap(True)
        header.addWidget(self._title_label, stretch=1)
        layout.addLayout(header)

    def _build_metadata_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._type_label = QLabel("")
        self._type_label.setFixedHeight(26)
        self._type_label.setStyleSheet(
            "QLabel { background-color: #2EA043; color: white; border-radius: 4px; "
            "padding: 2px 10px; font-weight: bold; font-size: 11px; }"
        )
        row.addWidget(self._type_label)

        self._status_combo = QComboBox()
        self._status_combo.addItems(["In Progress", "Completed", "Failed", "Paused"])
        self._status_combo.setFixedWidth(140)
        self._status_combo.setEnabled(False)
        row.addWidget(QLabel("Status:"))
        row.addWidget(self._status_combo)

        self._date_label = QLabel("")
        self._date_label.setStyleSheet("color: #8B949E; font-size: 11px;")
        row.addWidget(self._date_label)

        row.addStretch()
        layout.addLayout(row)

    def _build_content_area(self, layout: QVBoxLayout) -> None:
        self._browser = QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._browser.setFont(QFont("Segoe UI", 10))
        self._browser.setStyleSheet(
            "QTextBrowser { background-color: #161B22; color: #E6EDF3; "
            "border: 1px solid #30363D; border-radius: 6px; padding: 12px; }"
        )
        layout.addWidget(self._browser, stretch=1)

        self._editor = QTextEdit()
        self._editor.setFont(QFont("Consolas", 10))
        self._editor.setStyleSheet(
            "QTextEdit { background-color: #0D1117; color: #E6EDF3; "
            "border: 1px solid #30363D; border-radius: 6px; padding: 8px; }"
        )
        self._editor.setVisible(False)
        layout.addWidget(self._editor, stretch=1)

    def _build_action_buttons(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._btn_edit = self._make_button("Edit", _STYLE_BTN_EDIT, enabled=False)
        self._btn_edit.clicked.connect(self._toggle_edit)
        row.addWidget(self._btn_edit)

        self._btn_save = self._make_button("Save", _STYLE_BTN_SAVE)
        self._btn_save.setVisible(False)
        self._btn_save.clicked.connect(self._save)
        row.addWidget(self._btn_save)

        self._btn_export = self._make_button("Export MD", _STYLE_BTN_EXPORT, enabled=False)
        self._btn_export.clicked.connect(self._export)
        row.addWidget(self._btn_export)

        self._btn_delete = self._make_button("Delete", _STYLE_BTN_DELETE, enabled=False)
        self._btn_delete.clicked.connect(self._delete)
        row.addWidget(self._btn_delete)

        row.addStretch()
        layout.addLayout(row)

    @staticmethod
    def _make_button(text: str, style: str, enabled: bool = True) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(32)
        btn.setFixedWidth(100)
        btn.setEnabled(enabled)
        btn.setStyleSheet(style)
        return btn

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_session(self, session_dir: str) -> None:
        """Load and display a session."""
        self._current_dir = session_dir
        data = self._manager.get_session(session_dir)

        self._title_label.setText(
            data.get("description") or data.get("title") or session_dir
        )

        stype = data.get("type", "Unknown")
        color = _type_color(stype)
        self._type_label.setText(stype)
        self._type_label.setStyleSheet(
            f"QLabel {{ background-color: {color}; color: white; "
            f"border-radius: 4px; padding: 2px 10px; font-weight: bold; "
            f"font-size: 11px; }}"
        )

        self._set_status(data.get("status", "In Progress"))
        self._date_label.setText(f"{data.get('date', '')} {data.get('time', '')}")
        self._topics_bar.set_topics(data.get("topics", []))

        raw = data.get("raw_content", "")
        self._browser.setHtml(markdown_to_html(raw))
        self._editor.setPlainText(raw)

        self._btn_edit.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._btn_delete.setEnabled(True)

        if self._editing:
            self._toggle_edit()

    def clear(self) -> None:
        """Reset to empty state."""
        self._current_dir = None
        self._title_label.setText("No session selected")
        self._type_label.setText("")
        self._date_label.setText("")
        self._status_combo.setEnabled(False)
        self._topics_bar.clear_topics()
        self._browser.clear()
        self._editor.clear()
        self._btn_edit.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_delete.setEnabled(False)
        if self._editing:
            self._toggle_edit()

    # ------------------------------------------------------------------ #
    # Edit / Save / Export / Delete
    # ------------------------------------------------------------------ #

    def _toggle_edit(self) -> None:
        self._editing = not self._editing
        if self._editing:
            self._browser.setVisible(False)
            self._editor.setVisible(True)
            self._btn_edit.setText("Cancel")
            self._btn_edit.setStyleSheet(_STYLE_BTN_CANCEL)
            self._btn_save.setVisible(True)
        else:
            self._editor.setVisible(False)
            self._browser.setVisible(True)
            self._btn_edit.setText("Edit")
            self._btn_edit.setStyleSheet(_STYLE_BTN_EDIT)
            self._btn_save.setVisible(False)
            if self._current_dir:
                data = self._manager.get_session(self._current_dir)
                raw = data.get("raw_content", "")
                self._browser.setHtml(markdown_to_html(raw))

    def _save(self) -> None:
        if not self._current_dir:
            return

        content = self._editor.toPlainText()
        try:
            self._manager.save_session(self._current_dir, content)
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return

        self._browser.setHtml(markdown_to_html(content))

        parsed = self._manager.parse_session_summary(content)
        self._title_label.setText(
            parsed.get("description") or parsed.get("title") or self._current_dir
        )
        self._topics_bar.set_topics(parsed.get("topics", []))
        self._toggle_edit()
        self.session_saved.emit(self._current_dir)

    def _export(self) -> None:
        if not self._current_dir:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Session as Markdown",
            f"{self._current_dir}.md",
            "Markdown (*.md);;All (*)",
        )
        if not path:
            return
        data = self._manager.get_session(self._current_dir)
        raw = data.get("raw_content", "")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(raw)
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _delete(self) -> None:
        if not self._current_dir:
            return
        reply = QMessageBox.question(
            self,
            "Delete Session",
            f"Are you sure you want to delete session:\n{self._current_dir}?\n\n"
            "This will remove the entire session directory.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        dir_path = self._manager.SESSIONS_DIR / self._current_dir
        try:
            shutil.rmtree(dir_path)
        except OSError as exc:
            QMessageBox.critical(self, "Delete Error", str(exc))
            return

        dir_name = self._current_dir
        self.clear()
        self.session_deleted.emit(dir_name)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _set_status(self, status: str) -> None:
        """Set the status combo, adding the value if not already present."""
        idx = self._status_combo.findText(status)
        if idx >= 0:
            self._status_combo.setCurrentIndex(idx)
        else:
            self._status_combo.addItem(status)
            self._status_combo.setCurrentText(status)
        self._status_combo.setEnabled(True)
