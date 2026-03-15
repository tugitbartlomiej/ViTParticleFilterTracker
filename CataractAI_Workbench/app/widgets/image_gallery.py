"""Scrollable image gallery with lazy-loaded thumbnails."""

from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QLabel, QVBoxLayout, QSizePolicy,
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSignal, Qt, QTimer, QSize


class _ThumbnailLabel(QLabel):
    """Clickable thumbnail with filename caption."""

    clicked = pyqtSignal(str)  # image path

    def __init__(self, path: str, thumb_w: int, thumb_h: int, parent=None):
        super().__init__(parent)
        self.image_path = path
        self._thumb_size = QSize(thumb_w, thumb_h)
        self._loaded = False

        self.setFixedSize(thumb_w, thumb_h + 20)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QLabel { border: 1px solid #3C3C3C; background: #252526; }"
            "QLabel:hover { border: 1px solid #2A82DA; }"
        )
        self.setToolTip(Path(path).name)

    def load(self):
        """Load the thumbnail from disk (called lazily)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            pixmap = QPixmap(self.image_path)
            if pixmap.isNull():
                self.setText(Path(self.image_path).name[:12])
                return
            scaled = pixmap.scaled(
                self._thumb_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        except Exception:
            self.setText("err")

    def mousePressEvent(self, ev):
        self.clicked.emit(self.image_path)
        super().mousePressEvent(ev)


class ImageGallery(QWidget):
    """Grid of thumbnails inside a QScrollArea with lazy loading.

    Signals
    -------
    image_clicked(str)
        Emitted with the full path when a thumbnail is clicked.
    """

    image_clicked = pyqtSignal(str)

    COLUMNS = 4

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._thumb_w = 150
        self._thumb_h = 120
        self._labels: List[_ThumbnailLabel] = []
        self._load_index = 0  # next label to load

        # Layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._count_label = QLabel("0 images")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._count_label.setStyleSheet("color: #7F7F7F; padding: 2px 6px;")
        outer.addWidget(self._count_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        outer.addWidget(self._scroll)

        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setSpacing(6)
        self._scroll.setWidget(self._grid_widget)

        # Lazy-load timer
        self._timer = QTimer(self)
        self._timer.setInterval(10)  # ms per batch
        self._timer.timeout.connect(self._load_batch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_images(self, paths: List[str]):
        """Replace gallery contents with *paths*."""
        self._timer.stop()
        self._clear()

        for idx, path in enumerate(paths):
            lbl = _ThumbnailLabel(path, self._thumb_w, self._thumb_h)
            lbl.clicked.connect(self.image_clicked.emit)
            row, col = divmod(idx, self.COLUMNS)
            self._grid.addWidget(lbl, row, col)
            self._labels.append(lbl)

        self._count_label.setText(f"{len(paths)} images")
        self._load_index = 0
        self._timer.start()

    def set_thumbnail_size(self, w: int, h: int):
        """Change thumbnail dimensions (takes effect on next set_images)."""
        self._thumb_w = w
        self._thumb_h = h

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clear(self):
        self._labels.clear()
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _load_batch(self):
        """Load a small batch of thumbnails per timer tick."""
        batch = 8
        end = min(self._load_index + batch, len(self._labels))
        for i in range(self._load_index, end):
            self._labels[i].load()
        self._load_index = end
        if self._load_index >= len(self._labels):
            self._timer.stop()
