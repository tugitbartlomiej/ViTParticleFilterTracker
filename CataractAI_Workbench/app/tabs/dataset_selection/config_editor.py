"""Config editor panel for dataset selection pipeline."""

from pathlib import Path
from typing import Dict, Optional

import yaml
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QLabel, QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, Qt

from ...widgets.file_picker import FilePicker
from ...core.project_paths import DATASET_SELECTION_CONFIG


class ConfigEditor(QWidget):
    """Panel exposing the most important config.yaml fields for
    the dataset selection pipeline.

    Signals
    -------
    config_changed()
        Emitted whenever a field value is modified by the user.
    """

    config_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config_path: str = str(DATASET_SELECTION_CONFIG)
        self._init_ui()
        self._load_from_file()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        title = QLabel("Pipeline Configuration")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        root.addWidget(title)

        # --- Selection method -------------------------------------------
        method_group = QGroupBox("Selection Method")
        method_form = QFormLayout(method_group)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["cluster", "kcenter"])
        self._method_combo.currentTextChanged.connect(self._on_changed)
        method_form.addRow("Method:", self._method_combo)

        self._strategy_combo = QComboBox()
        self._strategy_combo.addItems(["centroid", "max_el2n", "medoid"])
        self._strategy_combo.currentTextChanged.connect(self._on_changed)
        method_form.addRow("Strategy:", self._strategy_combo)

        self._target_spin = QSpinBox()
        self._target_spin.setRange(100, 999_999)
        self._target_spin.setSingleStep(1000)
        self._target_spin.setValue(20_000)
        self._target_spin.valueChanged.connect(self._on_changed)
        method_form.addRow("Target Size:", self._target_spin)

        self._pca_spin = QSpinBox()
        self._pca_spin.setRange(2, 1024)
        self._pca_spin.setValue(32)
        self._pca_spin.valueChanged.connect(self._on_changed)
        method_form.addRow("DINO PCA Dim:", self._pca_spin)

        self._apply_weights_cb = QCheckBox("Apply feature weights")
        self._apply_weights_cb.setChecked(True)
        self._apply_weights_cb.toggled.connect(self._on_changed)
        method_form.addRow(self._apply_weights_cb)

        root.addWidget(method_group)

        # --- Weights ----------------------------------------------------
        weights_group = QGroupBox("Feature Weights")
        weights_form = QFormLayout(weights_group)

        self._w_dino = self._make_weight_spin(0.35)
        weights_form.addRow("DINO:", self._w_dino)

        self._w_fourier = self._make_weight_spin(0.15)
        weights_form.addRow("Fourier:", self._w_fourier)

        self._w_sam = self._make_weight_spin(0.20)
        weights_form.addRow("SAM:", self._w_sam)

        self._w_el2n = self._make_weight_spin(0.30)
        weights_form.addRow("EL2N:", self._w_el2n)

        root.addWidget(weights_group)

        # --- Dataset paths ----------------------------------------------
        paths_group = QGroupBox("Datasets")
        paths_form = QFormLayout(paths_group)

        self._existing_picker = FilePicker(mode="dir")
        self._existing_picker.path_changed.connect(self._on_changed)
        paths_form.addRow("Existing:", self._existing_picker)

        self._new_picker = FilePicker(mode="dir")
        self._new_picker.path_changed.connect(self._on_changed)
        paths_form.addRow("New Source:", self._new_picker)

        self._output_picker = FilePicker(mode="dir")
        self._output_picker.path_changed.connect(self._on_changed)
        paths_form.addRow("Output:", self._output_picker)

        root.addWidget(paths_group)

        # --- Cache ------------------------------------------------------
        cache_group = QGroupBox("Processing")
        cache_form = QFormLayout(cache_group)

        self._use_cache_cb = QCheckBox("Use feature cache")
        self._use_cache_cb.setChecked(True)
        self._use_cache_cb.toggled.connect(self._on_changed)
        cache_form.addRow(self._use_cache_cb)

        self._cache_picker = FilePicker(mode="dir")
        self._cache_picker.path_changed.connect(self._on_changed)
        cache_form.addRow("Cache Dir:", self._cache_picker)

        self._gen_viz_cb = QCheckBox("Generate visualizations")
        self._gen_viz_cb.setChecked(True)
        self._gen_viz_cb.toggled.connect(self._on_changed)
        cache_form.addRow(self._gen_viz_cb)

        root.addWidget(cache_group)

        # --- Buttons ----------------------------------------------------
        btn_row = QHBoxLayout()
        self._btn_load = QPushButton("Load")
        self._btn_load.setToolTip("Load config from YAML file")
        self._btn_load.clicked.connect(self._on_load)

        self._btn_save = QPushButton("Save")
        self._btn_save.setToolTip("Save config to YAML file")
        self._btn_save.clicked.connect(self._on_save)

        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_save)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> Dict:
        """Return current form values as a config dict."""
        return {
            "selection": {
                "method": self._method_combo.currentText(),
                "strategy": self._strategy_combo.currentText(),
                "dino_pca_dim": self._pca_spin.value(),
                "apply_weights": self._apply_weights_cb.isChecked(),
            },
            "weights": {
                "dino": self._w_dino.value(),
                "fourier": self._w_fourier.value(),
                "sam": self._w_sam.value(),
                "el2n": self._w_el2n.value(),
            },
            "datasets": {
                "existing": self._existing_picker.path(),
                "new_source": self._new_picker.path(),
                "output": self._output_picker.path(),
            },
            "output": {
                "target_size": self._target_spin.value(),
                "generate_visualizations": self._gen_viz_cb.isChecked(),
            },
            "processing": {
                "use_cache": self._use_cache_cb.isChecked(),
                "cache_dir": self._cache_picker.path(),
            },
        }

    def set_config(self, cfg: Dict):
        """Populate form from a config dict (missing keys are ignored)."""
        sel = cfg.get("selection", {})
        self._method_combo.setCurrentText(sel.get("method", "cluster"))
        self._strategy_combo.setCurrentText(sel.get("strategy", "centroid"))
        self._pca_spin.setValue(sel.get("dino_pca_dim", 32))
        self._apply_weights_cb.setChecked(sel.get("apply_weights", True))

        w = cfg.get("weights", {})
        self._w_dino.setValue(w.get("dino", 0.35))
        self._w_fourier.setValue(w.get("fourier", 0.15))
        self._w_sam.setValue(w.get("sam", 0.20))
        self._w_el2n.setValue(w.get("el2n", 0.30))

        ds = cfg.get("datasets", {})
        self._existing_picker.set_path(ds.get("existing", ""))
        self._new_picker.set_path(ds.get("new_source", ""))
        self._output_picker.set_path(ds.get("output", ""))

        out = cfg.get("output", {})
        self._target_spin.setValue(out.get("target_size", 20_000))
        self._gen_viz_cb.setChecked(out.get("generate_visualizations", True))

        proc = cfg.get("processing", {})
        self._use_cache_cb.setChecked(proc.get("use_cache", True))
        self._cache_picker.set_path(proc.get("cache_dir", ""))

    def get_target_size(self) -> int:
        return self._target_spin.value()

    def get_config_path(self) -> str:
        return self._config_path

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_changed(self):
        self.config_changed.emit()

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config", str(Path(self._config_path).parent),
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self._config_path = path
            self._load_from_file()

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", self._config_path,
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self._config_path = path
            self._save_to_file()

    def _load_from_file(self):
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            self.set_config(cfg)
        except FileNotFoundError:
            pass  # use defaults
        except Exception as exc:
            QMessageBox.warning(self, "Config Error", f"Failed to load config:\n{exc}")

    def _save_to_file(self):
        try:
            # Load existing config to preserve keys we don't expose
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    full = yaml.safe_load(f) or {}
            except FileNotFoundError:
                full = {}

            # Merge our fields
            updates = self.get_config()
            self._deep_merge(full, updates)

            with open(self._config_path, "w", encoding="utf-8") as f:
                yaml.dump(full, f, default_flow_style=False, allow_unicode=True)

            QMessageBox.information(self, "Saved", f"Config saved to:\n{self._config_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", f"Failed to save config:\n{exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_weight_spin(self, default: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(0.0, 1.0)
        sb.setDecimals(2)
        sb.setSingleStep(0.05)
        sb.setValue(default)
        sb.valueChanged.connect(self._on_changed)
        return sb

    @staticmethod
    def _deep_merge(base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                ConfigEditor._deep_merge(base[k], v)
            else:
                base[k] = v
