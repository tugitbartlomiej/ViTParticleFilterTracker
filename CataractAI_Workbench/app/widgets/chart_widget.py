"""Chart container widget supporting matplotlib and pyqtgraph backends."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


class ChartWidget(QWidget):
    """Embed either a matplotlib FigureCanvas or a pyqtgraph PlotWidget.

    Use the class methods to construct in the desired mode:

        chart = ChartWidget.matplotlib()
        chart = ChartWidget.pyqtgraph()

    Matplotlib mode
    ---------------
    - ``update_figure(fig)`` replaces the current figure.
    - ``get_canvas()`` returns the FigureCanvasQTAgg instance (or None).

    pyqtgraph mode
    ---------------
    - ``get_plot_widget()`` returns the PlotWidget (or None).
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mode: str = ""  # "matplotlib" | "pyqtgraph"
        self._canvas = None  # FigureCanvasQTAgg
        self._plot_widget = None  # pyqtgraph.PlotWidget
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def matplotlib(cls, fig=None, parent: Optional[QWidget] = None) -> "ChartWidget":
        """Create a ChartWidget in matplotlib mode.

        Parameters
        ----------
        fig : matplotlib.figure.Figure | None
            Initial figure to display. If *None* an empty canvas is created.
        """
        widget = cls(parent)
        widget._mode = "matplotlib"

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        if fig is None:
            fig = Figure(figsize=(5, 4), dpi=100)
            fig.patch.set_facecolor("#1E1E1E")

        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        widget._canvas = canvas
        widget._layout.addWidget(canvas)
        return widget

    @classmethod
    def pyqtgraph(cls, parent: Optional[QWidget] = None) -> "ChartWidget":
        """Create a ChartWidget in pyqtgraph mode."""
        widget = cls(parent)
        widget._mode = "pyqtgraph"

        import pyqtgraph as pg

        pg.setConfigOptions(background="#1E1E1E", foreground="#D4D4D4")
        plot_widget = pg.PlotWidget()
        widget._plot_widget = plot_widget
        widget._layout.addWidget(plot_widget)
        return widget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_figure(self, fig):
        """Replace the matplotlib figure (matplotlib mode only).

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        """
        if self._mode != "matplotlib":
            return

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        # Remove old canvas
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()

        fig.patch.set_facecolor("#1E1E1E")
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas = canvas
        self._layout.addWidget(canvas)
        canvas.draw()

    def get_canvas(self):
        """Return the FigureCanvasQTAgg (matplotlib mode) or *None*."""
        return self._canvas

    def get_plot_widget(self):
        """Return the pyqtgraph PlotWidget or *None*."""
        return self._plot_widget

    def clear(self):
        """Clear the current plot content."""
        if self._mode == "matplotlib" and self._canvas is not None:
            fig = self._canvas.figure
            fig.clear()
            fig.patch.set_facecolor("#1E1E1E")
            self._canvas.draw()
        elif self._mode == "pyqtgraph" and self._plot_widget is not None:
            self._plot_widget.clear()
