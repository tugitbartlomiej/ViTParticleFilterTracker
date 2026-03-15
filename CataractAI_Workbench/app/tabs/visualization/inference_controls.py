"""Control panels for model configuration, dataset loading, and batch testing.

Emits signals so the parent orchestrator can wire actions without the
controls needing to know about the model or image viewer.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QDoubleSpinBox, QSpinBox,
    QGroupBox, QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ...widgets.file_picker import FilePicker


# =====================================================================
# Styled button helpers
# =====================================================================

_BTN_BLUE = (
    "QPushButton { background-color: #0E639C; color: white; padding: 6px; "
    "border-radius: 3px; font-weight: bold; }"
    "QPushButton:hover { background-color: #1177BB; }"
    "QPushButton:disabled { background-color: #3C3C3C; color: #7F7F7F; }"
)

_BTN_GREEN = (
    "QPushButton { background-color: #388E3C; color: white; padding: 6px; "
    "border-radius: 3px; font-weight: bold; }"
    "QPushButton:hover { background-color: #43A047; }"
)

_BTN_ORANGE = (
    "QPushButton { background-color: #E65100; color: white; padding: 4px; "
    "border-radius: 3px; font-weight: bold; }"
    "QPushButton:hover { background-color: #F57C00; }"
    "QPushButton:disabled { background-color: #3C3C3C; color: #7F7F7F; }"
)


# =====================================================================
# Model configuration panel
# =====================================================================

class ModelConfigPanel(QGroupBox):
    """Model type, checkpoint, confidence, device, and load button."""

    load_requested = pyqtSignal()
    model_type_changed = pyqtSignal(str)  # "detr" / "yolo"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Model Configuration", parent)
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        # Model type
        row = QHBoxLayout()
        row.addWidget(QLabel("Model Type:"))
        self._model_type_combo = QComboBox()
        self._model_type_combo.addItems(["DETR", "YOLO"])
        self._model_type_combo.currentTextChanged.connect(
            self._on_type_changed,
        )
        row.addWidget(self._model_type_combo)
        lay.addLayout(row)

        # Checkpoint
        row = QHBoxLayout()
        row.addWidget(QLabel("Checkpoint:"))
        self._ckpt_picker = FilePicker(
            label="Browse...", mode="file",
            filter_str="Model Files (*.pth *.pt);;All Files (*)",
        )
        row.addWidget(self._ckpt_picker)
        lay.addLayout(row)

        # Confidence threshold
        row = QHBoxLayout()
        row.addWidget(QLabel("Confidence:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 0.99)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.50)
        self._conf_spin.setFixedWidth(70)
        row.addWidget(self._conf_spin)
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(1, 99)
        self._conf_slider.setValue(50)
        self._conf_slider.valueChanged.connect(
            lambda v: self._conf_spin.setValue(v / 100.0),
        )
        self._conf_spin.valueChanged.connect(
            lambda v: self._conf_slider.setValue(int(v * 100)),
        )
        row.addWidget(self._conf_slider)
        lay.addLayout(row)

        # Num labels (DETR only)
        row = QHBoxLayout()
        row.addWidget(QLabel("Num Labels:"))
        self._num_labels_spin = QSpinBox()
        self._num_labels_spin.setRange(1, 100)
        self._num_labels_spin.setValue(1)
        self._num_labels_spin.setFixedWidth(70)
        row.addWidget(self._num_labels_spin)
        row.addStretch()
        self._nlabels_widget = QWidget()
        self._nlabels_widget.setLayout(row)
        lay.addWidget(self._nlabels_widget)

        # Device
        row = QHBoxLayout()
        row.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItems(["cuda", "cpu"])
        self._device_combo.setFixedWidth(80)
        row.addWidget(self._device_combo)
        row.addStretch()
        lay.addLayout(row)

        # Load button
        self._btn_load = QPushButton("Load Model")
        self._btn_load.setStyleSheet(_BTN_BLUE)
        self._btn_load.clicked.connect(self.load_requested.emit)
        lay.addWidget(self._btn_load)

        # Status
        self._status = QLabel("No model loaded")
        self._status.setStyleSheet("color: #7F7F7F; font-style: italic;")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

    # -- public accessors --

    @property
    def checkpoint_path(self) -> str:
        return self._ckpt_picker.path()

    @property
    def model_type_str(self) -> str:
        return self._model_type_combo.currentText().lower()

    @property
    def confidence(self) -> float:
        return self._conf_spin.value()

    @property
    def num_labels(self) -> int:
        return self._num_labels_spin.value()

    @property
    def device(self) -> str:
        return self._device_combo.currentText()

    def set_load_enabled(self, enabled: bool) -> None:
        self._btn_load.setEnabled(enabled)

    def set_status(self, text: str, color: str = "#7F7F7F") -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-style: italic;")

    # -- internal --

    def _on_type_changed(self, text: str) -> None:
        self._nlabels_widget.setVisible(text.upper() == "DETR")
        self.model_type_changed.emit(text.lower())


# =====================================================================
# Dataset panel
# =====================================================================

class DatasetPanel(QGroupBox):
    """Image directory, annotation file, and load button."""

    load_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Dataset", parent)
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Images:"))
        self._img_picker = FilePicker(label="Browse...", mode="dir")
        row.addWidget(self._img_picker)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Annotations:"))
        self._ann_picker = FilePicker(
            label="Browse...", mode="file",
            filter_str="JSON Files (*.json);;All Files (*)",
        )
        row.addWidget(self._ann_picker)
        lay.addLayout(row)

        self._btn_load = QPushButton("Load Dataset")
        self._btn_load.setStyleSheet(_BTN_GREEN)
        self._btn_load.clicked.connect(self.load_requested.emit)
        lay.addWidget(self._btn_load)

        self._status = QLabel("No dataset loaded")
        self._status.setStyleSheet("color: #7F7F7F; font-style: italic;")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

    @property
    def image_dir(self) -> str:
        return self._img_picker.path()

    @property
    def annotations_path(self) -> str:
        return self._ann_picker.path()

    def set_status(self, text: str, color: str = "#7F7F7F") -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color}; font-style: italic;")


# =====================================================================
# Batch test panel (IoU, run/cancel, progress, export)
# =====================================================================

class BatchTestPanel(QGroupBox):
    """Batch inference controls with IoU threshold, progress, and export."""

    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    export_csv_requested = pyqtSignal()
    export_json_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Batch Test", parent)
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)

        # IoU threshold
        row = QHBoxLayout()
        row.addWidget(QLabel("IoU Threshold:"))
        self._iou_spin = QDoubleSpinBox()
        self._iou_spin.setRange(0.30, 0.90)
        self._iou_spin.setSingleStep(0.05)
        self._iou_spin.setValue(0.50)
        self._iou_spin.setFixedWidth(70)
        row.addWidget(self._iou_spin)
        self._iou_slider = QSlider(Qt.Orientation.Horizontal)
        self._iou_slider.setRange(30, 90)
        self._iou_slider.setValue(50)
        self._iou_slider.valueChanged.connect(
            lambda v: self._iou_spin.setValue(v / 100.0),
        )
        self._iou_spin.valueChanged.connect(
            lambda v: self._iou_slider.setValue(int(v * 100)),
        )
        row.addWidget(self._iou_slider)
        lay.addLayout(row)

        # Run / Cancel
        row = QHBoxLayout()
        self._btn_run = QPushButton("Run All")
        self._btn_run.setStyleSheet(_BTN_BLUE)
        self._btn_run.clicked.connect(self.run_requested.emit)
        row.addWidget(self._btn_run)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self.cancel_requested.emit)
        row.addWidget(self._btn_cancel)
        lay.addLayout(row)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #3C3C3C; border-radius: 3px; "
            "text-align: center; background-color: #1E1E1E; color: #D4D4D4; }"
            "QProgressBar::chunk { background-color: #0E639C; }"
        )
        lay.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #7F7F7F; font-size: 11px;")
        lay.addWidget(self._progress_label)

        # Export
        row = QHBoxLayout()
        btn_csv = QPushButton("Export CSV")
        btn_csv.clicked.connect(self.export_csv_requested.emit)
        row.addWidget(btn_csv)
        btn_json = QPushButton("Export JSON")
        btn_json.clicked.connect(self.export_json_requested.emit)
        row.addWidget(btn_json)
        lay.addLayout(row)

    # -- public accessors --

    @property
    def iou_threshold(self) -> float:
        return self._iou_spin.value()

    def set_running(self, running: bool) -> None:
        self._btn_run.setEnabled(not running)
        self._btn_cancel.setEnabled(running)

    def set_progress(self, current: int, total: int, filename: str) -> None:
        pct = int(current / total * 100) if total else 0
        self._progress_bar.setValue(pct)
        self._progress_label.setText(f"{current}/{total}  |  {filename}")

    def set_finished(self, images_tested: int) -> None:
        self._progress_bar.setValue(100)
        self._progress_label.setText(
            f"Done. {images_tested} images tested."
        )

    def set_error(self, message: str) -> None:
        self._progress_label.setText(f"Error: {message}")

    def set_cancelling(self) -> None:
        self._progress_label.setText("Cancelling...")


# =====================================================================
# Navigation bar
# =====================================================================

class NavigationBar(QWidget):
    """Prev / Next / Random / Run Inference buttons with position label."""

    prev_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    random_clicked = pyqtSignal()
    run_inference_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._btn_prev = QPushButton("< Prev")
        self._btn_prev.setFixedWidth(70)
        self._btn_prev.clicked.connect(self.prev_clicked.emit)
        lay.addWidget(self._btn_prev)

        self._pos_label = QLabel("No images loaded")
        self._pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pos_label.setStyleSheet(
            "color: #CCCCCC; font-weight: bold;",
        )
        lay.addWidget(self._pos_label)

        self._btn_next = QPushButton("Next >")
        self._btn_next.setFixedWidth(70)
        self._btn_next.clicked.connect(self.next_clicked.emit)
        lay.addWidget(self._btn_next)

        self._btn_random = QPushButton("Random")
        self._btn_random.setFixedWidth(70)
        self._btn_random.clicked.connect(self.random_clicked.emit)
        lay.addWidget(self._btn_random)

        self._btn_run = QPushButton("Run Inference")
        self._btn_run.setFixedWidth(110)
        self._btn_run.setStyleSheet(_BTN_ORANGE)
        self._btn_run.clicked.connect(self.run_inference_clicked.emit)
        lay.addWidget(self._btn_run)

    def set_position(self, text: str) -> None:
        self._pos_label.setText(text)

    def set_run_enabled(self, enabled: bool) -> None:
        self._btn_run.setEnabled(enabled)
