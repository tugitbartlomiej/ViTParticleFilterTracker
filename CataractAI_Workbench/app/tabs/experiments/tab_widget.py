"""Main Experiments tab -- orchestrates session list, editor, and dialogs."""

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from ...core.signal_bus import get_signal_bus
from .experiment_editor import ExperimentEditor
from .new_session_dialog import NewSessionDialog
from .session_list_widget import SessionListWidget
from .session_manager import SessionManager


class ExperimentsTab(QWidget):
    """Top-level Experiments tab for browsing and editing experiment sessions.

    Layout
    ------
    +--[Session List]--------+--[Session Detail / Editor]---------------------+
    | [New Session] [Refresh]| Session title, type badge, status              |
    | Filter: [All Types v]  | Topics: [DETR] [GPU] [CUDA]                   |
    | Search: [________]     |                                                |
    |                        | ## Objective                                    |
    | * 2026-01-26 RAG M3    | Description text...                            |
    | * 2026-01-18 GPU Fix   |                                                |
    | * 2026-01-18 EL2N Opt  | ## Actions Taken                               |
    |   ...                  | 1. Step one                                    |
    |                        |                                                |
    | [Timeline] [Graph]     | [Edit] [Save] [Export MD] [Delete]             |
    +------------------------+------------------------------------------------+
    | Stats: Sessions: 42 | Types: Training(10) Analysis(10) Mixed(15)       |
    +------------------------+------------------------------------------------+
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._manager = SessionManager()
        self._build_ui()
        self._connect_signals()
        self._session_list.refresh()
        self._update_stats()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: session list with filters
        self._session_list = SessionListWidget(self._manager)
        self._session_list.setMinimumWidth(260)
        self._session_list.setMaximumWidth(400)
        splitter.addWidget(self._session_list)

        # Right panel: editor
        self._editor = ExperimentEditor(self._manager)
        splitter.addWidget(self._editor)

        splitter.setSizes([300, 700])
        root.addWidget(splitter, stretch=1)

        # Status bar
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(
            "QLabel { color: #8B949E; font-size: 11px; padding: 4px 8px; "
            "background-color: #161B22; border: 1px solid #21262D; "
            "border-radius: 4px; }"
        )
        root.addWidget(self._stats_label)

    # ------------------------------------------------------------------ #
    # Signal wiring
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self._session_list.new_session_button.clicked.connect(self._on_new_session)
        self._session_list.session_selected.connect(self._on_session_selected)
        self._editor.session_saved.connect(self._on_session_saved)
        self._editor.session_deleted.connect(self._on_session_deleted)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _on_session_selected(self, dir_name: str) -> None:
        if dir_name:
            self._editor.load_session(dir_name)
        else:
            self._editor.clear()

    def _on_session_saved(self, dir_name: str) -> None:
        self._session_list.refresh()
        self._session_list.select_session(dir_name)
        self._update_stats()
        get_signal_bus().status_message.emit(f"Session saved: {dir_name}")

    def _on_session_deleted(self, dir_name: str) -> None:
        self._session_list.refresh()
        self._editor.clear()
        self._update_stats()
        get_signal_bus().status_message.emit(f"Session deleted: {dir_name}")

    def _on_new_session(self) -> None:
        dialog = NewSessionDialog(self._manager, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        data = dialog.get_data()
        try:
            new_dir = self._manager.create_session(
                description=data["description"],
                session_type=data["type"],
                topics=data["topics"],
                objective=data["objective"],
            )
        except OSError as exc:
            QMessageBox.critical(
                self, "Error", f"Failed to create session:\n{exc}"
            )
            return

        self._session_list.refresh()
        self._session_list.select_session(new_dir)
        self._update_stats()
        get_signal_bus().status_message.emit(f"New session created: {new_dir}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _update_stats(self) -> None:
        """Update the stats label at the bottom."""
        stats = self._manager.get_statistics()
        type_parts = [
            f"{name}({count})"
            for name, count in list(stats["by_type"].items())[:6]
        ]
        self._stats_label.setText(
            f"Sessions: {stats['total']}  |  Types: {'  '.join(type_parts)}"
        )
