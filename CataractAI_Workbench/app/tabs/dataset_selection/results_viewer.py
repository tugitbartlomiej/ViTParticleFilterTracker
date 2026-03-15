"""Results viewer showing selection statistics and selected-image gallery."""

from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QGroupBox,
)
from PyQt6.QtCore import Qt

from ...widgets.image_gallery import ImageGallery


class ResultsViewer(QWidget):
    """Two-part panel: statistics table (top) and image gallery (bottom).

    Call :meth:`update_results` after the pipeline finishes.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ---- Stats table ------------------------------------------------
        stats_group = QGroupBox("Selection Statistics")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(4, 8, 4, 4)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents,
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        stats_layout.addWidget(self._table)
        splitter.addWidget(stats_group)

        # ---- Image gallery ----------------------------------------------
        gallery_group = QGroupBox("Selected Images")
        gallery_layout = QVBoxLayout(gallery_group)
        gallery_layout.setContentsMargins(4, 8, 4, 4)

        self._gallery = ImageGallery()
        gallery_layout.addWidget(self._gallery)
        splitter.addWidget(gallery_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_results(self, results: Dict, selected_paths: Optional[List[str]] = None):
        """Populate the table and gallery from pipeline *results*.

        Parameters
        ----------
        results : dict
            Pipeline result dict (must contain ``statistics`` key).
        selected_paths : list[str] | None
            If *None* will be read from ``results["selected_paths"]``.
        """
        stats = results.get("statistics", {})
        if selected_paths is None:
            selected_paths = results.get("selected_paths", [])

        self._populate_table(stats, results)
        if selected_paths:
            self._gallery.set_images(selected_paths)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _populate_table(self, stats: Dict, results: Dict):
        rows = []

        # Core counts
        if "original_count" in stats:
            rows.append(("Total Images (input)", str(stats["original_count"])))
        if "valid_count" in stats:
            rows.append(("Valid Images", str(stats["valid_count"])))
        if "final_count" in stats:
            rows.append(("Selected (output)", str(stats["final_count"])))
        if "reduction_ratio" in stats:
            ratio = stats["reduction_ratio"]
            rows.append(("Reduction Ratio", f"{ratio * 100:.1f}%"))
            if ratio > 0:
                rows.append(("Compression Factor", f"{1.0 / ratio:.1f}x"))

        # Cluster-specific
        if "n_clusters" in stats:
            rows.append(("Number of Clusters", str(stats["n_clusters"])))
        if "strategy" in stats:
            rows.append(("Strategy", str(stats["strategy"])))

        # Fourier / k-center stages
        if "after_fourier" in stats:
            rows.append(("After Fourier Filter", str(stats["after_fourier"])))
        if "after_k_center" in stats:
            rows.append(("After k-Center", str(stats["after_k_center"])))

        # Time info
        if "start_time" in results and "end_time" in results:
            rows.append(("Start Time", results["start_time"]))
            rows.append(("End Time", results["end_time"]))

        # Selection method from config
        cfg = results.get("config", {})
        method = cfg.get("selection", {}).get("method", "")
        if method:
            rows.append(("Selection Method", method))
        strategy = cfg.get("selection", {}).get("strategy", "")
        if strategy:
            rows.append(("Cluster Strategy", strategy))

        # Weights
        weights = cfg.get("weights", {})
        if weights:
            w_str = ", ".join(f"{k}={v}" for k, v in weights.items())
            rows.append(("Weights", w_str))

        # Fill table
        self._table.setRowCount(len(rows))
        for i, (metric, value) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(metric))
            self._table.setItem(i, 1, QTableWidgetItem(value))

    def clear(self):
        """Reset to empty state."""
        self._table.setRowCount(0)
        self._gallery.set_images([])
