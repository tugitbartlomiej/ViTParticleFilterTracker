"""Session list panel with filtering, search, and visualization buttons."""

import webbrowser
from typing import Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QFont

from .session_manager import SessionManager


# Status -> foreground color mapping
_STATUS_COLORS: dict[str, str] = {
    "completed": "#3FB950",
    "in progress": "#D29922",
    "failed": "#F85149",
}


class SessionListWidget(QWidget):
    """Left-side panel listing experiment sessions with filter and search.

    Signals:
        session_selected(str): emitted with the dir_name when a session is clicked.
    """

    session_selected = pyqtSignal(str)

    def __init__(self, manager: SessionManager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        title = QLabel("Experiment Journal")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #E6EDF3;")
        layout.addWidget(title)

        self._build_top_buttons(layout)
        self._build_filter_row(layout)
        self._build_session_list(layout)
        self._build_viz_buttons(layout)

    def _build_top_buttons(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)

        self._btn_new = QPushButton("+ New Session")
        self._btn_new.setFixedHeight(30)
        self._btn_new.setStyleSheet(
            "QPushButton { background-color: #238636; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 0 12px; }"
            "QPushButton:hover { background-color: #2EA043; }"
        )
        row.addWidget(self._btn_new)

        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setFixedHeight(30)
        self._btn_refresh.setStyleSheet(
            "QPushButton { background-color: #2D2D2D; color: #D4D4D4; "
            "border-radius: 4px; padding: 0 12px; border: 1px solid #3C3C3C; }"
            "QPushButton:hover { background-color: #3C3C3C; }"
        )
        row.addWidget(self._btn_refresh)
        layout.addLayout(row)

    def _build_filter_row(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(4)
        label = QLabel("Filter:")
        label.setStyleSheet("color: #8B949E; font-size: 11px;")
        row.addWidget(label)

        self._filter_combo = QComboBox()
        self._filter_combo.addItem("All Types")
        for session_type in self._manager.SESSION_TYPES:
            self._filter_combo.addItem(session_type)
        row.addWidget(self._filter_combo, stretch=1)
        layout.addLayout(row)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search sessions...")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.setFixedHeight(28)
        layout.addWidget(self._search_input)

    def _build_session_list(self, layout: QVBoxLayout) -> None:
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setStyleSheet(
            "QListWidget { background-color: #0D1117; border: 1px solid #30363D; "
            "border-radius: 6px; }"
            "QListWidget::item { padding: 6px 8px; border-bottom: 1px solid #21262D; }"
            "QListWidget::item:selected { background-color: #1F6FEB; color: white; }"
            "QListWidget::item:hover { background-color: #161B22; }"
        )
        layout.addWidget(self._list, stretch=1)

    def _build_viz_buttons(self, layout: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        style = (
            "QPushButton { background-color: #2D2D2D; color: #D4D4D4; "
            "border-radius: 4px; border: 1px solid #3C3C3C; font-size: 11px; }"
            "QPushButton:hover { background-color: #3C3C3C; }"
        )

        self._btn_timeline = QPushButton("Timeline View")
        self._btn_timeline.setFixedHeight(28)
        self._btn_timeline.setStyleSheet(style)
        row.addWidget(self._btn_timeline)

        self._btn_graph = QPushButton("Graph View")
        self._btn_graph.setFixedHeight(28)
        self._btn_graph.setStyleSheet(style)
        row.addWidget(self._btn_graph)

        layout.addLayout(row)

    # ------------------------------------------------------------------ #
    # Signal wiring
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self._btn_refresh.clicked.connect(self.refresh)
        self._filter_combo.currentTextChanged.connect(lambda _: self.refresh())
        self._search_input.textChanged.connect(lambda _: self.refresh())
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._btn_timeline.clicked.connect(self._open_timeline)
        self._btn_graph.clicked.connect(self._open_graph)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def new_session_button(self) -> QPushButton:
        """Expose the '+ New Session' button so the parent can connect it."""
        return self._btn_new

    def refresh(self) -> None:
        """Reload the session list from disk, applying current filters."""
        type_filter = self._filter_combo.currentText()
        if type_filter == "All Types":
            type_filter = None

        search = self._search_input.text().strip() or None
        sessions = self._manager.list_sessions(
            type_filter=type_filter, search_query=search,
        )

        self._list.blockSignals(True)
        self._list.clear()

        for session in sessions:
            self._list.addItem(self._make_item(session))

        self._list.blockSignals(False)

    def select_session(self, dir_name: str) -> None:
        """Programmatically select a session by directory name."""
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == dir_name:
                self._list.setCurrentItem(item)
                return

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _on_item_changed(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            self.session_selected.emit("")
            return
        dir_name = current.data(Qt.ItemDataRole.UserRole)
        if dir_name:
            self.session_selected.emit(dir_name)

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def _open_timeline(self) -> None:
        analysis = self._manager.SESSIONS_DIR / "analysis"
        self._open_html_or_warn(
            primary=analysis / "hybrid_timeline.html",
            fallback=analysis / "session_timeline_2d.html",
            generate_hint="py -3.11 .sessions/tools/visualize_sessions_tree.py",
            label="Timeline",
        )

    def _open_graph(self) -> None:
        analysis = self._manager.SESSIONS_DIR / "analysis"
        self._open_html_or_warn(
            primary=analysis / "graph_view.html",
            fallback=analysis / "session_graph_3d.html",
            generate_hint="py -3.11 .sessions/tools/generate_graph_view.py",
            label="Graph",
        )

    def _open_html_or_warn(
        self, primary, fallback, generate_hint: str, label: str,
    ) -> None:
        for path in (primary, fallback):
            if path.is_file():
                webbrowser.open(str(path))
                return
        QMessageBox.information(
            self,
            "Not Found",
            f"{label} visualization not found.\n\n"
            f"Generate it by running:\n{generate_hint}",
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_item(session: dict) -> QListWidgetItem:
        """Build a QListWidgetItem from a session dict."""
        item = QListWidgetItem()

        date = session.get("date", "")
        desc = session.get("description", "Untitled")
        stype = session.get("type", "")
        status = session.get("status", "")

        if len(desc) > 40:
            desc = desc[:37] + "..."

        display = f"{date}  {desc}"
        if stype:
            type_short = stype.split("/")[0].split("|")[0].strip()[:12]
            display += f"  [{type_short}]"
        item.setText(display)
        item.setData(Qt.ItemDataRole.UserRole, session.get("dir_name", ""))

        tooltip_parts = [f"Session: {session.get('dir_name', '')}"]
        if stype:
            tooltip_parts.append(f"Type: {stype}")
        if status:
            tooltip_parts.append(f"Status: {status}")
        topics = session.get("topics", [])
        if topics:
            tooltip_parts.append(f"Topics: {', '.join(topics)}")
        obj = session.get("objective", "")
        if obj:
            tooltip_parts.append(f"Objective: {obj[:100]}")
        item.setToolTip("\n".join(tooltip_parts))

        status_lower = status.lower()
        color = _STATUS_COLORS.get(status_lower, "")
        if not color:
            for key, val in _STATUS_COLORS.items():
                if key in status_lower:
                    color = val
                    break
        item.setForeground(QColor(color or "#8B949E"))

        return item
