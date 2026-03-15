"""Real-time training charts using pyqtgraph."""

from typing import List

from PyQt6.QtWidgets import QHBoxLayout, QWidget

import pyqtgraph as pg


class LiveChart(QWidget):
    """Side-by-side Loss and Learning-Rate curves updated in real time.

    Left plot  -- Training Loss (yellow, epoch vs loss)
    Right plot -- Learning Rate schedule (cyan, epoch vs lr)
    """

    # Dark palette matching the workbench theme
    _BG = "#1E1E1E"
    _GRID_PEN = pg.mkPen(color="#3C3C3C", style=pg.QtCore.Qt.PenStyle.DotLine)
    _LOSS_PEN = pg.mkPen(color="#FFD93D", width=2)
    _LR_PEN = pg.mkPen(color="#6BCBF0", width=2)
    _LOSS_SYMBOL_BRUSH = pg.mkBrush("#FFD93D")
    _LR_SYMBOL_BRUSH = pg.mkBrush("#6BCBF0")

    def __init__(self, parent=None):
        super().__init__(parent)

        self._epochs: List[int] = []
        self._losses: List[float] = []
        self._lrs: List[float] = []

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # -- Loss plot --
        self._loss_pw = pg.PlotWidget(title="Training Loss")
        self._setup_plot(self._loss_pw, y_label="Loss")
        self._loss_curve = self._loss_pw.plot(
            pen=self._LOSS_PEN,
            symbol="o",
            symbolSize=6,
            symbolBrush=self._LOSS_SYMBOL_BRUSH,
        )
        layout.addWidget(self._loss_pw)

        # -- LR plot --
        self._lr_pw = pg.PlotWidget(title="Learning Rate")
        self._setup_plot(self._lr_pw, y_label="LR")
        self._lr_curve = self._lr_pw.plot(
            pen=self._LR_PEN,
            symbol="o",
            symbolSize=6,
            symbolBrush=self._LR_SYMBOL_BRUSH,
        )
        layout.addWidget(self._lr_pw)

    def _setup_plot(self, pw: pg.PlotWidget, y_label: str):
        pw.setBackground(self._BG)
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel("bottom", "Epoch")
        pw.setLabel("left", y_label)
        pw.getAxis("bottom").setPen(pg.mkPen("#888888"))
        pw.getAxis("left").setPen(pg.mkPen("#888888"))
        pw.getAxis("bottom").setTextPen(pg.mkPen("#AAAAAA"))
        pw.getAxis("left").setTextPen(pg.mkPen("#AAAAAA"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_epoch_complete(self, data: dict) -> None:
        """Append a single data point and redraw curves.

        *data* must contain at least ``epoch``, ``loss``, ``lr``.
        """
        ep = data.get("epoch", len(self._epochs) + 1)
        self._epochs.append(ep)
        self._losses.append(data.get("loss", 0.0))
        self._lrs.append(data.get("lr", 0.0))
        self._redraw()

    def clear(self) -> None:
        """Reset both charts to empty."""
        self._epochs.clear()
        self._losses.clear()
        self._lrs.clear()
        self._redraw()

    def set_data(self, losses: List[float], lrs: List[float]) -> None:
        """Bulk-set full history (e.g. when loading a saved run)."""
        n = max(len(losses), len(lrs))
        self._epochs = list(range(1, n + 1))
        self._losses = list(losses)
        self._lrs = list(lrs)
        # Pad shorter list if needed
        while len(self._losses) < n:
            self._losses.append(0.0)
        while len(self._lrs) < n:
            self._lrs.append(0.0)
        self._redraw()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _redraw(self):
        self._loss_curve.setData(self._epochs, self._losses)
        self._lr_curve.setData(self._epochs, self._lrs)
