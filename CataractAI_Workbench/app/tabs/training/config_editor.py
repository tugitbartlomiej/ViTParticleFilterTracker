"""Form-based editor for TrainingSettings.json configuration."""

from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
)

from ...widgets.file_picker import FilePicker


class ConfigEditor(QWidget):
    """Panel with form fields for every TrainingSettings.json section."""

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()
        self._connect_change_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # --- Model settings ------------------------------------------------
        grp_model = QGroupBox("Model")
        form_model = QFormLayout(grp_model)

        self.base_model = QLineEdit("facebook/detr-resnet-50")
        form_model.addRow("Base model:", self.base_model)

        self.checkpoint_path = FilePicker(
            label="Browse...", mode="file", filter_str="PyTorch (*.pth *.pt);;All (*)"
        )
        form_model.addRow("Checkpoint:", self.checkpoint_path)

        self.num_labels = QSpinBox()
        self.num_labels.setRange(1, 1000)
        self.num_labels.setValue(1)
        form_model.addRow("Num labels:", self.num_labels)

        root.addWidget(grp_model)

        # --- Dataset paths -------------------------------------------------
        grp_data = QGroupBox("Dataset")
        form_data = QFormLayout(grp_data)

        self.mixed_dataset_dir = FilePicker(label="Browse...", mode="dir")
        form_data.addRow("Dataset dir:", self.mixed_dataset_dir)

        self.images_dir = QLineEdit("all_images")
        form_data.addRow("Images sub-dir:", self.images_dir)

        self.annotations_file = QLineEdit("annotations/mixed_annotations.json")
        form_data.addRow("Annotations:", self.annotations_file)

        root.addWidget(grp_data)

        # --- Training parameters -------------------------------------------
        grp_train = QGroupBox("Training Parameters")
        form_train = QFormLayout(grp_train)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(5)
        form_train.addRow("Epochs:", self.epochs)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(2)
        form_train.addRow("Batch size:", self.batch_size)

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(9)
        self.learning_rate.setRange(1e-9, 1e-1)
        self.learning_rate.setSingleStep(1e-7)
        self.learning_rate.setValue(2e-6)
        form_train.addRow("Learning rate:", self.learning_rate)

        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setDecimals(8)
        self.weight_decay.setRange(0, 1)
        self.weight_decay.setSingleStep(1e-6)
        self.weight_decay.setValue(1e-5)
        form_train.addRow("Weight decay:", self.weight_decay)

        self.warmup_epochs = QSpinBox()
        self.warmup_epochs.setRange(0, 100)
        self.warmup_epochs.setValue(1)
        form_train.addRow("Warmup epochs:", self.warmup_epochs)

        self.gradient_clip = QDoubleSpinBox()
        self.gradient_clip.setDecimals(2)
        self.gradient_clip.setRange(0.0, 100.0)
        self.gradient_clip.setSingleStep(0.1)
        self.gradient_clip.setValue(1.0)
        form_train.addRow("Gradient clip:", self.gradient_clip)

        root.addWidget(grp_train)

        # --- Optimization --------------------------------------------------
        grp_opt = QGroupBox("Optimization")
        form_opt = QFormLayout(grp_opt)

        self.optimizer = QComboBox()
        self.optimizer.addItems(["AdamW", "Adam", "SGD"])
        form_opt.addRow("Optimizer:", self.optimizer)

        self.use_amp = QCheckBox("Mixed precision (AMP)")
        self.use_amp.setChecked(True)
        form_opt.addRow(self.use_amp)

        self.scheduler_type = QComboBox()
        self.scheduler_type.addItems(["StepLR", "CosineAnnealingLR"])
        form_opt.addRow("Scheduler:", self.scheduler_type)

        root.addWidget(grp_opt)

        # --- Early stopping ------------------------------------------------
        grp_es = QGroupBox("Early Stopping")
        form_es = QFormLayout(grp_es)

        self.es_enabled = QCheckBox("Enabled")
        self.es_enabled.setChecked(True)
        form_es.addRow(self.es_enabled)

        self.es_patience = QSpinBox()
        self.es_patience.setRange(1, 100)
        self.es_patience.setValue(6)
        form_es.addRow("Patience:", self.es_patience)

        self.es_min_delta = QDoubleSpinBox()
        self.es_min_delta.setDecimals(6)
        self.es_min_delta.setRange(0, 1)
        self.es_min_delta.setSingleStep(0.0001)
        self.es_min_delta.setValue(0.0005)
        form_es.addRow("Min delta:", self.es_min_delta)

        root.addWidget(grp_es)

        # --- Checkpointing -------------------------------------------------
        grp_ckpt = QGroupBox("Checkpointing")
        form_ckpt = QFormLayout(grp_ckpt)

        self.save_best = QCheckBox("Save best")
        self.save_best.setChecked(True)
        form_ckpt.addRow(self.save_best)

        self.save_every_n = QSpinBox()
        self.save_every_n.setRange(0, 100)
        self.save_every_n.setValue(3)
        form_ckpt.addRow("Save every N:", self.save_every_n)

        self.checkpoint_dir = FilePicker(label="Browse...", mode="dir")
        form_ckpt.addRow("Checkpoint dir:", self.checkpoint_dir)

        root.addWidget(grp_ckpt)

        # --- Device --------------------------------------------------------
        grp_dev = QGroupBox("Device")
        form_dev = QFormLayout(grp_dev)

        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        form_dev.addRow("Device:", self.device)

        root.addWidget(grp_dev)

        # --- Load / Save buttons ------------------------------------------
        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Config")
        self.btn_save = QPushButton("Save Config")
        self.btn_defaults = QPushButton("Defaults")
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_defaults)
        root.addLayout(btn_row)

        root.addStretch()

    # ------------------------------------------------------------------
    # Change tracking
    # ------------------------------------------------------------------

    def _connect_change_signals(self):
        """Emit config_changed whenever any field value changes."""
        # Line edits
        for w in (self.base_model, self.images_dir, self.annotations_file):
            w.textChanged.connect(self.config_changed)
        # File pickers
        for w in (self.checkpoint_path, self.mixed_dataset_dir, self.checkpoint_dir):
            w.path_changed.connect(self.config_changed)
        # Spin boxes
        for w in (
            self.num_labels,
            self.epochs,
            self.batch_size,
            self.warmup_epochs,
            self.es_patience,
            self.save_every_n,
        ):
            w.valueChanged.connect(self.config_changed)
        # Double spin boxes
        for w in (
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip,
            self.es_min_delta,
        ):
            w.valueChanged.connect(self.config_changed)
        # Combo boxes
        for w in (self.optimizer, self.scheduler_type, self.device):
            w.currentTextChanged.connect(self.config_changed)
        # Check boxes
        for w in (self.use_amp, self.es_enabled, self.save_best):
            w.stateChanged.connect(self.config_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Build a full config dict from form field values."""
        return {
            "training_configuration": {
                "description": "DETR training configuration",
                "version": "2.0",
            },
            "model_settings": {
                "checkpoint_path": self.checkpoint_path.path(),
                "base_model": self.base_model.text(),
                "num_labels": self.num_labels.value(),
                "ignore_mismatched_sizes": True,
            },
            "dataset_paths": {
                "mixed_dataset_dir": self.mixed_dataset_dir.path(),
                "images_dir": self.images_dir.text(),
                "annotations_file": self.annotations_file.text(),
                "auto_use_dino_subsets": True,
                "dino_subset_suffix": ".dino",
            },
            "training_parameters": {
                "epochs": self.epochs.value(),
                "batch_size": self.batch_size.value(),
                "learning_rate": self.learning_rate.value(),
                "weight_decay": self.weight_decay.value(),
                "warmup_epochs": self.warmup_epochs.value(),
                "gradient_clip": self.gradient_clip.value(),
                "images_per_epoch": None,
            },
            "optimization": {
                "optimizer": self.optimizer.currentText(),
                "scheduler": {
                    "type": self.scheduler_type.currentText(),
                    "step_size": 2,
                    "gamma": 0.7,
                },
                "use_amp": self.use_amp.isChecked(),
            },
            "early_stopping": {
                "enabled": self.es_enabled.isChecked(),
                "patience": self.es_patience.value(),
                "min_delta": self.es_min_delta.value(),
                "monitor": "loss",
            },
            "checkpointing": {
                "save_best": self.save_best.isChecked(),
                "save_every_n_epochs": self.save_every_n.value(),
                "save_final": True,
                "checkpoint_dir": self.checkpoint_dir.path(),
            },
            "device_settings": {
                "device": self.device.currentText(),
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": True,
            },
            "logging": {
                "log_level": "INFO",
                "save_training_plots": True,
                "save_training_history": True,
                "progress_bar": False,
            },
            "augmentation": {
                "enabled": True,
                "random_seed": 42,
                "reproducible": True,
            },
            "validation": {
                "validation_split": 0.0,
                "validation_frequency": 1,
            },
        }

    def set_config(self, cfg: dict) -> None:
        """Populate form fields from a config dict."""
        ms = cfg.get("model_settings", {})
        self.base_model.setText(ms.get("base_model", "facebook/detr-resnet-50"))
        self.checkpoint_path.set_path(ms.get("checkpoint_path", ""))
        self.num_labels.setValue(ms.get("num_labels", 1))

        dp = cfg.get("dataset_paths", {})
        self.mixed_dataset_dir.set_path(dp.get("mixed_dataset_dir", ""))
        self.images_dir.setText(dp.get("images_dir", "all_images"))
        self.annotations_file.setText(
            dp.get("annotations_file", "annotations/mixed_annotations.json")
        )

        tp = cfg.get("training_parameters", {})
        self.epochs.setValue(tp.get("epochs", 5))
        self.batch_size.setValue(tp.get("batch_size", 2))
        self.learning_rate.setValue(tp.get("learning_rate", 2e-6))
        self.weight_decay.setValue(tp.get("weight_decay", 1e-5))
        self.warmup_epochs.setValue(tp.get("warmup_epochs", 1))
        self.gradient_clip.setValue(tp.get("gradient_clip", 1.0))

        opt = cfg.get("optimization", {})
        idx = self.optimizer.findText(opt.get("optimizer", "AdamW"))
        if idx >= 0:
            self.optimizer.setCurrentIndex(idx)
        self.use_amp.setChecked(opt.get("use_amp", True))
        sched = opt.get("scheduler", {})
        sidx = self.scheduler_type.findText(sched.get("type", "StepLR"))
        if sidx >= 0:
            self.scheduler_type.setCurrentIndex(sidx)

        es = cfg.get("early_stopping", {})
        self.es_enabled.setChecked(es.get("enabled", True))
        self.es_patience.setValue(es.get("patience", 6))
        self.es_min_delta.setValue(es.get("min_delta", 0.0005))

        ckpt = cfg.get("checkpointing", {})
        self.save_best.setChecked(ckpt.get("save_best", True))
        self.save_every_n.setValue(ckpt.get("save_every_n_epochs", 3))
        self.checkpoint_dir.set_path(ckpt.get("checkpoint_dir", ""))

        dev = cfg.get("device_settings", {})
        didx = self.device.findText(dev.get("device", "auto"))
        if didx >= 0:
            self.device.setCurrentIndex(didx)
