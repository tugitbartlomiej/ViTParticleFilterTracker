"""Dynamic form generator from dict/schema with nested QGroupBox support."""

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QScrollArea,
    QLabel,
)
from PyQt6.QtCore import pyqtSignal, Qt


class ConfigForm(QWidget):
    """Dynamically generates a form from a nested dict.

    Supported value types:
    - str  -> QLineEdit
    - int  -> QSpinBox
    - float -> QDoubleSpinBox
    - bool -> QCheckBox
    - list of str (enum) -> QComboBox (pass via *enums* parameter)
    - dict -> QGroupBox (recursive)

    Parameters
    ----------
    schema : dict
        Nested dictionary whose leaf values set the initial widget values
        and determine widget types.
    enums : dict | None
        Mapping of dotted-key -> list[str] for fields that should use QComboBox.
        Example: ``{"selection.method": ["cluster", "kcenter"]}``
    parent : QWidget | None
    """

    value_changed = pyqtSignal()  # emitted on any field change

    def __init__(
        self,
        schema: Optional[Dict] = None,
        enums: Optional[Dict[str, List[str]]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._widgets: Dict[str, QWidget] = {}  # dotted-key -> widget
        self._enums = enums or {}
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(4, 4, 4, 4)
        self._inner_layout.setSpacing(6)
        self._scroll.setWidget(self._inner)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self._scroll)

        if schema:
            self.build(schema)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, schema: Dict):
        """Build (or rebuild) the form from *schema*."""
        self._clear()
        self._build_level(schema, self._inner_layout, prefix="")
        self._inner_layout.addStretch()

    def get_values(self) -> Dict:
        """Return current form values as a nested dict."""
        flat: Dict[str, Any] = {}
        for key, widget in self._widgets.items():
            flat[key] = self._read_widget(widget)
        return self._unflatten(flat)

    def set_values(self, data: Dict):
        """Set widget values from a nested dict (keys that don't exist are
        silently ignored)."""
        flat = self._flatten(data)
        for key, value in flat.items():
            widget = self._widgets.get(key)
            if widget is not None:
                self._write_widget(widget, value)

    # ------------------------------------------------------------------
    # Internal build helpers
    # ------------------------------------------------------------------

    def _clear(self):
        """Remove all widgets from the inner layout."""
        self._widgets.clear()
        while self._inner_layout.count():
            item = self._inner_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _build_level(self, data: Dict, parent_layout: QVBoxLayout, prefix: str):
        """Recursively build widgets for one nesting level."""
        for key, value in data.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                group = QGroupBox(self._label(key))
                group_layout = QVBoxLayout(group)
                group_layout.setContentsMargins(8, 12, 8, 8)
                group_layout.setSpacing(4)
                self._build_level(value, group_layout, dotted)
                parent_layout.addWidget(group)
            else:
                form = self._get_or_create_form(parent_layout)
                widget = self._create_widget(dotted, value)
                form.addRow(self._label(key) + ":", widget)
                self._widgets[dotted] = widget

    def _get_or_create_form(self, parent_layout: QVBoxLayout) -> QFormLayout:
        """Return the last QFormLayout child of *parent_layout*, or create one."""
        if parent_layout.count() > 0:
            last = parent_layout.itemAt(parent_layout.count() - 1)
            if last and last.layout() and isinstance(last.layout(), QFormLayout):
                return last.layout()
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        parent_layout.addLayout(form)
        return form

    def _create_widget(self, dotted_key: str, value: Any) -> QWidget:
        """Create the appropriate QWidget for *value*."""
        # Enum override?
        if dotted_key in self._enums:
            combo = QComboBox()
            combo.addItems(self._enums[dotted_key])
            if isinstance(value, str) and value in self._enums[dotted_key]:
                combo.setCurrentText(value)
            combo.currentTextChanged.connect(lambda: self.value_changed.emit())
            return combo

        if isinstance(value, bool):
            cb = QCheckBox()
            cb.setChecked(value)
            cb.toggled.connect(lambda: self.value_changed.emit())
            return cb

        if isinstance(value, int):
            sb = QSpinBox()
            sb.setRange(0, 999_999)
            sb.setValue(value)
            sb.valueChanged.connect(lambda: self.value_changed.emit())
            return sb

        if isinstance(value, float):
            dsb = QDoubleSpinBox()
            dsb.setRange(0.0, 999_999.0)
            dsb.setDecimals(4)
            dsb.setSingleStep(0.01)
            dsb.setValue(value)
            dsb.valueChanged.connect(lambda: self.value_changed.emit())
            return dsb

        # Default: string
        le = QLineEdit()
        le.setText(str(value) if value is not None else "")
        le.textChanged.connect(lambda: self.value_changed.emit())
        return le

    # ------------------------------------------------------------------
    # Read / write helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_widget(widget: QWidget) -> Any:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    @staticmethod
    def _write_widget(widget: QWidget, value: Any):
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value) if value is not None else 0)
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value) if value is not None else 0.0)
        elif isinstance(widget, QComboBox):
            idx = widget.findText(str(value))
            if idx >= 0:
                widget.setCurrentIndex(idx)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value) if value is not None else "")

    # ------------------------------------------------------------------
    # Dict flatten / unflatten
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(d: Dict, prefix: str = "") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(ConfigForm._flatten(v, new_key))
            else:
                items[new_key] = v
        return items

    @staticmethod
    def _unflatten(flat: Dict[str, Any]) -> Dict:
        result: Dict = {}
        for dotted, value in flat.items():
            parts = dotted.split(".")
            d = result
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
        return result

    @staticmethod
    def _label(key: str) -> str:
        """Convert snake_case key to Title Case label."""
        return key.replace("_", " ").title()
