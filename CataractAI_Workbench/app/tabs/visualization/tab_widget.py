"""Main Visualization tab containing Training Dashboard, Dataset Explorer,
and Model Comparison sub-tabs."""

from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from ...core.signal_bus import get_signal_bus
from .training_dashboard import TrainingDashboard
from .dataset_explorer import DatasetExplorer
from .model_comparison import ModelComparison
from .inference_tester import InferenceTester


class VisualizationTab(QWidget):
    """Top-level widget for the Visualization tab.

    Contains a QTabWidget with four sub-tabs:
    - Training Dashboard: live and historical training metrics
    - Dataset Explorer: COCO dataset browsing with annotation overlays
    - Model Comparison: side-by-side prediction comparison
    - Inference Tester: interactive visual testing of DETR/YOLO models
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Sub-tabs
        self._training_dashboard = TrainingDashboard()
        self._dataset_explorer = DatasetExplorer()
        self._model_comparison = ModelComparison()
        self._inference_tester = InferenceTester()

        self._tabs.addTab(self._training_dashboard, "Training Dashboard")
        self._tabs.addTab(self._dataset_explorer, "Dataset Explorer")
        self._tabs.addTab(self._model_comparison, "Model Comparison")
        self._tabs.addTab(self._inference_tester, "Inference Tester")

        layout.addWidget(self._tabs)

    def _connect_signals(self):
        """Wire signal_bus.epoch_completed to the training dashboard."""
        bus = get_signal_bus()
        bus.epoch_completed.connect(self._training_dashboard.on_epoch_completed)

    # ------------------------------------------------------------------
    # Public accessors for sub-tabs
    # ------------------------------------------------------------------

    @property
    def training_dashboard(self) -> TrainingDashboard:
        return self._training_dashboard

    @property
    def dataset_explorer(self) -> DatasetExplorer:
        return self._dataset_explorer

    @property
    def model_comparison(self) -> ModelComparison:
        return self._model_comparison

    @property
    def inference_tester(self) -> InferenceTester:
        return self._inference_tester
