"""Training Analyzer -- thin orchestrator combining analysis sub-tabs."""

from __future__ import annotations

from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .checkpoint_inspector import CheckpointInspector
from .dataset_analyzer_widget import DatasetAnalyzerWidget
from .parameter_advisor import ParameterAdvisor
from .theme_constants import TAB_STYLE


class TrainingAnalyzer(QWidget):
    """Training analysis tool with checkpoint inspection, parameter advice,
    and dataset analysis sub-tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tab_widget = QTabWidget()
        self._tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self._tab_widget.setStyleSheet(TAB_STYLE)

        self._checkpoint_inspector = CheckpointInspector()
        self._parameter_advisor = ParameterAdvisor()
        self._dataset_analyzer = DatasetAnalyzerWidget()

        self._tab_widget.addTab(self._checkpoint_inspector, "Checkpoint Inspector")
        self._tab_widget.addTab(self._parameter_advisor, "Parameter Advisor")
        self._tab_widget.addTab(self._dataset_analyzer, "Dataset Analyzer")

        layout.addWidget(self._tab_widget)
