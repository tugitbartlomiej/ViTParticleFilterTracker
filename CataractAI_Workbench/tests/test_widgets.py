"""Tests for reusable Qt widgets."""
import sys
from pathlib import Path

import pytest

# Skip entire module if no PyQt6
PyQt6 = pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtCore import Qt

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CataractAI_Workbench.app.widgets.log_console import LogConsole
from CataractAI_Workbench.app.widgets.progress_panel import ProgressPanel
from CataractAI_Workbench.app.widgets.config_form import ConfigForm
from CataractAI_Workbench.app.widgets.file_picker import FilePicker
from CataractAI_Workbench.app.widgets.image_gallery import ImageGallery
from CataractAI_Workbench.app.widgets.bbox_painter import BBoxPainter


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def qapp_fixture(qapp):
    """Per-test fixture to ensure QApplication exists."""
    return qapp


# -----------------------------------------------------------------------
# LogConsole tests
# -----------------------------------------------------------------------


class TestLogConsole:
    """Tests for LogConsole widget."""

    def test_append_adds_text(self, qapp_fixture):
        """Verify that append() inserts text into the log."""
        console = LogConsole()
        console.append("Hello World")
        plain = console._text.toPlainText()
        assert "Hello World" in plain

    def test_append_error_colored(self, qapp_fixture):
        """ERROR lines should get red-ish foreground color."""
        console = LogConsole()
        console.append("ERROR something went wrong")

        # The cursor was used to insert with a QTextCharFormat that has
        # the ERROR color (#FF6B6B). Verify the first block has that color.
        doc = console._text.document()
        block = doc.begin()
        it = block.begin()
        if not it.atEnd():
            fragment = it.fragment()
            fmt = fragment.charFormat()
            fg = fmt.foreground().color()
            assert fg.red() == 0xFF and fg.green() == 0x6B and fg.blue() == 0x6B

    def test_append_with_source(self, qapp_fixture):
        """append_with_source should prefix message with [source]."""
        console = LogConsole()
        console.append_with_source("DETR", "training started")
        plain = console._text.toPlainText()
        assert "[DETR]" in plain
        assert "training started" in plain

    def test_clear(self, qapp_fixture):
        """clear() should remove all content."""
        console = LogConsole()
        console.append("line 1")
        console.append("line 2")
        console.clear()
        plain = console._text.toPlainText()
        assert plain.strip() == ""

    def test_max_lines(self, qapp_fixture):
        """Block count should stay within MAX_LINES + small margin."""
        console = LogConsole()
        num_lines = 5010
        for i in range(num_lines):
            console.append(f"Line {i}")
        doc = console._text.document()
        # blockCount includes trailing empty block, allow a small margin
        assert doc.blockCount() <= LogConsole.MAX_LINES + 10


# -----------------------------------------------------------------------
# ProgressPanel tests
# -----------------------------------------------------------------------


class TestProgressPanel:
    """Tests for ProgressPanel widget."""

    def test_add_task(self, qapp_fixture):
        """add_task creates a new progress bar entry."""
        panel = ProgressPanel()
        panel.add_task("train", label="Training")
        assert "train" in panel._bars
        name_label, bar, status_label = panel._bars["train"]
        assert name_label.text() == "Training"
        assert bar.value() == 0

    def test_update_task(self, qapp_fixture):
        """update_task sets the progress value and message."""
        panel = ProgressPanel()
        panel.add_task("train")
        panel.update_task("train", 42, "Epoch 2/5")
        _, bar, status = panel._bars["train"]
        assert bar.value() == 42
        assert status.text() == "Epoch 2/5"

    def test_update_creates_if_missing(self, qapp_fixture):
        """update_task auto-creates the task if it does not exist."""
        panel = ProgressPanel()
        panel.update_task("new_task", 50, "halfway")
        assert "new_task" in panel._bars
        _, bar, status = panel._bars["new_task"]
        assert bar.value() == 50
        assert status.text() == "halfway"

    def test_remove_task(self, qapp_fixture):
        """remove_task removes the task from the panel."""
        panel = ProgressPanel()
        panel.add_task("task_a")
        panel.add_task("task_b")
        panel.remove_task("task_a")
        assert "task_a" not in panel._bars
        assert "task_b" in panel._bars

    def test_clear_all(self, qapp_fixture):
        """clear() removes all tasks."""
        panel = ProgressPanel()
        panel.add_task("task_1")
        panel.add_task("task_2")
        panel.add_task("task_3")
        panel.clear()
        assert len(panel._bars) == 0


# -----------------------------------------------------------------------
# ConfigForm tests
# -----------------------------------------------------------------------


class TestConfigForm:
    """Tests for ConfigForm widget."""

    def test_build_flat_schema(self, qapp_fixture):
        """Build form with str, int, float, bool fields."""
        schema = {
            "name": "experiment_1",
            "epochs": 10,
            "learning_rate": 0.001,
            "use_amp": True,
        }
        form = ConfigForm(schema=schema)
        assert "name" in form._widgets
        assert "epochs" in form._widgets
        assert "learning_rate" in form._widgets
        assert "use_amp" in form._widgets

        assert isinstance(form._widgets["name"], QLineEdit)
        assert isinstance(form._widgets["epochs"], QSpinBox)
        assert isinstance(form._widgets["learning_rate"], QDoubleSpinBox)
        assert isinstance(form._widgets["use_amp"], QCheckBox)

    def test_get_values(self, qapp_fixture):
        """get_values() returns correct types and values."""
        schema = {
            "name": "test",
            "epochs": 5,
            "lr": 0.01,
            "amp": False,
        }
        form = ConfigForm(schema=schema)
        values = form.get_values()
        assert values["name"] == "test"
        assert values["epochs"] == 5
        assert abs(values["lr"] - 0.01) < 1e-6
        assert values["amp"] is False

    def test_set_values(self, qapp_fixture):
        """set_values() updates widget values and get_values() reflects them."""
        schema = {
            "name": "original",
            "epochs": 1,
            "lr": 0.001,
            "amp": False,
        }
        form = ConfigForm(schema=schema)
        form.set_values({"name": "updated", "epochs": 20, "lr": 0.1, "amp": True})
        values = form.get_values()
        assert values["name"] == "updated"
        assert values["epochs"] == 20
        assert abs(values["lr"] - 0.1) < 1e-6
        assert values["amp"] is True

    def test_nested_schema(self, qapp_fixture):
        """Nested dict creates QGroupBox and dotted keys."""
        schema = {
            "model": "detr",
            "training": {
                "epochs": 10,
                "batch_size": 4,
            },
        }
        form = ConfigForm(schema=schema)
        assert "model" in form._widgets
        assert "training.epochs" in form._widgets
        assert "training.batch_size" in form._widgets

        # Verify nested values round-trip
        values = form.get_values()
        assert values["model"] == "detr"
        assert values["training"]["epochs"] == 10
        assert values["training"]["batch_size"] == 4

        # Verify a QGroupBox was created inside the inner widget
        group_boxes = form._inner.findChildren(QGroupBox)
        assert len(group_boxes) >= 1
        group_titles = [gb.title() for gb in group_boxes]
        assert "Training" in group_titles

    def test_enum_creates_combobox(self, qapp_fixture):
        """Enum fields create QComboBox with correct items."""
        schema = {"method": "cluster"}
        enums = {"method": ["cluster", "kcenter", "random"]}
        form = ConfigForm(schema=schema, enums=enums)
        widget = form._widgets["method"]
        assert isinstance(widget, QComboBox)
        items = [widget.itemText(i) for i in range(widget.count())]
        assert items == ["cluster", "kcenter", "random"]
        assert widget.currentText() == "cluster"


# -----------------------------------------------------------------------
# FilePicker tests
# -----------------------------------------------------------------------


class TestFilePicker:
    """Tests for FilePicker widget."""

    def test_initial_path_empty(self, qapp_fixture):
        """path() returns empty string initially."""
        picker = FilePicker()
        assert picker.path() == ""

    def test_set_path(self, qapp_fixture):
        """set_path updates the internal line edit text."""
        picker = FilePicker()
        picker.set_path("/some/path/to/file.txt")
        assert picker.path() == "/some/path/to/file.txt"

    def test_path_changed_signal(self, qapp_fixture):
        """path_changed signal is emitted when text changes."""
        picker = FilePicker()
        received = []
        picker.path_changed.connect(lambda p: received.append(p))
        picker.set_path("/new/path")
        assert len(received) == 1
        assert received[0] == "/new/path"


# -----------------------------------------------------------------------
# ImageGallery tests
# -----------------------------------------------------------------------


class TestImageGallery:
    """Tests for ImageGallery widget."""

    def test_set_images(self, qapp_fixture, tmp_path):
        """set_images populates the gallery grid with thumbnail labels."""
        # Create small dummy image files
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            # Create a 10x10 red pixmap and save it
            pix = QPixmap(10, 10)
            pix.fill(QColor(255, 0, 0))
            pix.save(str(p))
            paths.append(str(p))

        gallery = ImageGallery()
        gallery.set_images(paths)

        assert len(gallery._labels) == 3
        assert gallery._count_label.text() == "3 images"
        # Each label stores its image path
        for lbl, expected_path in zip(gallery._labels, paths):
            assert lbl.image_path == expected_path

    def test_set_thumbnail_size(self, qapp_fixture):
        """set_thumbnail_size updates stored dimensions."""
        gallery = ImageGallery()
        assert gallery._thumb_w == 150
        assert gallery._thumb_h == 120
        gallery.set_thumbnail_size(200, 160)
        assert gallery._thumb_w == 200
        assert gallery._thumb_h == 160


# -----------------------------------------------------------------------
# BBoxPainter tests
# -----------------------------------------------------------------------


class TestBBoxPainter:
    """Tests for BBoxPainter utility class."""

    def _make_pixmap(self, w=200, h=200):
        """Create a blank white pixmap for testing."""
        pix = QPixmap(w, h)
        pix.fill(QColor(255, 255, 255))
        return pix

    def test_draw_boxes_basic(self, qapp_fixture):
        """draw_boxes returns a QPixmap with drawings on it."""
        pix = self._make_pixmap()
        boxes = [
            {"bbox": [10, 10, 50, 50], "category_id": 1, "score": 0.95},
            {"bbox": [80, 80, 40, 30], "category_id": 2},
        ]
        result = BBoxPainter.draw_boxes(
            pix,
            boxes,
            color=QColor(255, 0, 0),
            categories={1: "tool", 2: "bg"},
        )
        assert isinstance(result, QPixmap)
        assert not result.isNull()
        assert result.width() == 200
        assert result.height() == 200

        # The result should differ from a plain white pixmap
        # (boxes were drawn on it). Compare via QImage.
        orig_img = self._make_pixmap().toImage()
        result_img = result.toImage()
        differ = False
        for x_pos in range(result_img.width()):
            for y_pos in range(result_img.height()):
                if orig_img.pixel(x_pos, y_pos) != result_img.pixel(x_pos, y_pos):
                    differ = True
                    break
            if differ:
                break
        assert differ, "draw_boxes should have modified the pixmap"

    def test_draw_boxes_empty(self, qapp_fixture):
        """No boxes means the result pixmap equals the original."""
        pix = self._make_pixmap()
        result = BBoxPainter.draw_boxes(pix, [], color=QColor(0, 0, 255))
        assert isinstance(result, QPixmap)
        assert not result.isNull()
        # With no boxes drawn, pixels should be identical
        orig_img = pix.toImage()
        result_img = result.toImage()
        all_same = True
        for x_pos in range(result_img.width()):
            for y_pos in range(result_img.height()):
                if orig_img.pixel(x_pos, y_pos) != result_img.pixel(x_pos, y_pos):
                    all_same = False
                    break
            if not all_same:
                break
        assert all_same, "Empty boxes list should not modify the pixmap"

    def test_draw_badge(self, qapp_fixture):
        """draw_badge returns a y position below the badge for stacking."""
        pix = self._make_pixmap()
        y_after = BBoxPainter.draw_badge(
            pix,
            text="DETR",
            color=QColor(78, 154, 255),
            y_offset=4,
            x_offset=4,
        )
        assert isinstance(y_after, int)
        # The returned y must be greater than the input y_offset
        assert y_after > 4
