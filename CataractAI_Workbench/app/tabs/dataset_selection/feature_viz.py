"""Feature visualization panel with sub-tabs for PCA, Fourier, SAM, EL2N, and Clusters."""

from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

from ...widgets.chart_widget import ChartWidget


class FeatureViz(QTabWidget):
    """Tabbed container showing pipeline feature visualizations.

    Each sub-tab holds a :class:`ChartWidget` in matplotlib mode.
    Call :meth:`update_plots` after the pipeline finishes to populate
    the charts from the result dict.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._charts: Dict[str, ChartWidget] = {}
        self._tab_names = ["PCA", "Fourier", "SAM", "EL2N", "Cluster"]

        for name in self._tab_names:
            chart = ChartWidget.matplotlib()
            self._charts[name] = chart
            self.addTab(chart, name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_plots(self, results: Dict):
        """Create matplotlib figures from pipeline *results* and display them.

        Expected keys in *results*:

        - ``statistics`` (dict)  with original_count, final_count, etc.
        - ``selected_indices`` (list[int])
        - ``selected_paths`` (list[str])

        Additionally, the pipeline stores feature arrays on the selector
        object.  Since we may not have direct access to those from the
        serialized results dict, this method also accepts optional numpy
        arrays under the following keys (added by the adapter or worker):

        - ``dino_features``  (np.ndarray, shape N x D)
        - ``el2n_scores``    (np.ndarray, shape N)
        - ``sam_scores``     (np.ndarray or list[float])
        - ``fourier_features`` (np.ndarray, shape N x F)
        - ``cluster_labels`` (np.ndarray, shape N)
        - ``combined_features`` (np.ndarray, shape N x M)
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        selected_indices = results.get("selected_indices", [])
        stats = results.get("statistics", {})

        self._plot_pca(results, selected_indices, plt)
        self._plot_fourier(results, plt)
        self._plot_sam(results, selected_indices, plt)
        self._plot_el2n(results, selected_indices, plt)
        self._plot_cluster(results, selected_indices, stats, plt)

    def update_from_images(self, image_paths: List[str]):
        """Show placeholder info about loaded images (before pipeline run)."""
        chart = self._charts.get("PCA")
        if chart is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1E1E1E")
        ax.set_facecolor("#252526")
        ax.text(
            0.5, 0.5,
            f"{len(image_paths)} images loaded\nRun pipeline to see features",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=14, color="#7F7F7F",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        chart.update_figure(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Private plot methods
    # ------------------------------------------------------------------

    def _plot_pca(self, results: Dict, selected_indices, plt):
        chart = self._charts["PCA"]
        features = results.get("dino_features")
        if features is None:
            self._placeholder(chart, "PCA", "No DINO features available", plt)
            return

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            self._placeholder(chart, "PCA", "scikit-learn not installed", plt)
            return

        pca = PCA(n_components=2)
        pts = pca.fit_transform(features)

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#1E1E1E")
        ax.set_facecolor("#252526")

        ax.scatter(pts[:, 0], pts[:, 1], c="#555555", alpha=0.3, s=8, label="All")
        if selected_indices:
            mask = np.zeros(len(pts), dtype=bool)
            valid = [i for i in selected_indices if i < len(pts)]
            mask[valid] = True
            ax.scatter(
                pts[mask, 0], pts[mask, 1],
                c="#FF4444", alpha=0.8, s=20, label="Selected",
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color="#D4D4D4")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color="#D4D4D4")
        ax.set_title("DINO Feature Space (PCA)", color="#D4D4D4")
        ax.legend(facecolor="#2D2D2D", edgecolor="#3C3C3C", labelcolor="#D4D4D4")
        ax.tick_params(colors="#7F7F7F")
        fig.tight_layout()
        chart.update_figure(fig)
        plt.close(fig)

    def _plot_fourier(self, results: Dict, plt):
        chart = self._charts["Fourier"]
        features = results.get("fourier_features")
        if features is None:
            self._placeholder(chart, "Fourier", "No Fourier features available", plt)
            return

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#1E1E1E")
        for ax in axes:
            ax.set_facecolor("#252526")

        # Band energy box plot
        if features.shape[1] >= 3:
            bp = axes[0].boxplot(
                [features[:, 0], features[:, 1], features[:, 2]],
                labels=["Low", "Mid", "High"],
                patch_artist=True,
            )
            for patch, color in zip(bp["boxes"], ["#2A82DA", "#6BCB77", "#FF6B6B"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            axes[0].set_title("Band Energy", color="#D4D4D4")
            axes[0].set_ylabel("Energy", color="#D4D4D4")
            axes[0].tick_params(colors="#7F7F7F")

        # Entropy histogram
        if features.shape[1] >= 4:
            axes[1].hist(features[:, 3], bins=40, color="#9B59B6", alpha=0.7, edgecolor="#1E1E1E")
            axes[1].set_title("Spectral Entropy", color="#D4D4D4")
            axes[1].set_xlabel("Entropy", color="#D4D4D4")
            axes[1].set_ylabel("Count", color="#D4D4D4")
            axes[1].tick_params(colors="#7F7F7F")

        fig.tight_layout()
        chart.update_figure(fig)
        plt.close(fig)

    def _plot_sam(self, results: Dict, selected_indices, plt):
        chart = self._charts["SAM"]
        scores = results.get("sam_scores")
        if scores is None:
            self._placeholder(chart, "SAM", "No SAM complexity scores available", plt)
            return

        scores = np.asarray(scores)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#1E1E1E")
        for ax in axes:
            ax.set_facecolor("#252526")

        axes[0].hist(scores, bins=40, color="#1ABC9C", alpha=0.7, edgecolor="#1E1E1E")
        axes[0].axvline(np.mean(scores), color="#FF4444", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
        axes[0].set_title("All Samples", color="#D4D4D4")
        axes[0].set_xlabel("Complexity", color="#D4D4D4")
        axes[0].set_ylabel("Count", color="#D4D4D4")
        axes[0].legend(facecolor="#2D2D2D", edgecolor="#3C3C3C", labelcolor="#D4D4D4")
        axes[0].tick_params(colors="#7F7F7F")

        if selected_indices:
            sel = [scores[i] for i in selected_indices if i < len(scores)]
            if sel:
                axes[1].hist(sel, bins=30, color="#E67E22", alpha=0.7, edgecolor="#1E1E1E")
                axes[1].axvline(np.mean(sel), color="#FF4444", linestyle="--", label=f"Mean: {np.mean(sel):.3f}")
                axes[1].legend(facecolor="#2D2D2D", edgecolor="#3C3C3C", labelcolor="#D4D4D4")
        axes[1].set_title("Selected", color="#D4D4D4")
        axes[1].set_xlabel("Complexity", color="#D4D4D4")
        axes[1].tick_params(colors="#7F7F7F")

        fig.tight_layout()
        chart.update_figure(fig)
        plt.close(fig)

    def _plot_el2n(self, results: Dict, selected_indices, plt):
        chart = self._charts["EL2N"]
        scores = results.get("el2n_scores")
        if scores is None:
            self._placeholder(chart, "EL2N", "No EL2N scores available", plt)
            return

        scores = np.asarray(scores)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#1E1E1E")
        for ax in axes:
            ax.set_facecolor("#252526")

        axes[0].hist(scores, bins=50, color="#3498DB", alpha=0.7, edgecolor="#1E1E1E")
        axes[0].axvline(np.mean(scores), color="#FF4444", linestyle="--", label=f"Mean: {np.mean(scores):.3f}")
        axes[0].set_title("All EL2N Scores", color="#D4D4D4")
        axes[0].set_xlabel("Score", color="#D4D4D4")
        axes[0].set_ylabel("Count", color="#D4D4D4")
        axes[0].legend(facecolor="#2D2D2D", edgecolor="#3C3C3C", labelcolor="#D4D4D4")
        axes[0].tick_params(colors="#7F7F7F")

        if selected_indices:
            sel = scores[np.array([i for i in selected_indices if i < len(scores)])]
            if len(sel):
                axes[1].hist(sel, bins=30, color="#E74C3C", alpha=0.7, edgecolor="#1E1E1E")
                axes[1].axvline(np.mean(sel), color="#FFD93D", linestyle="--", label=f"Mean: {np.mean(sel):.3f}")
                axes[1].legend(facecolor="#2D2D2D", edgecolor="#3C3C3C", labelcolor="#D4D4D4")
        axes[1].set_title("Selected EL2N Scores", color="#D4D4D4")
        axes[1].set_xlabel("Score", color="#D4D4D4")
        axes[1].tick_params(colors="#7F7F7F")

        fig.tight_layout()
        chart.update_figure(fig)
        plt.close(fig)

    def _plot_cluster(self, results: Dict, selected_indices, stats: Dict, plt):
        chart = self._charts["Cluster"]
        labels = results.get("cluster_labels")

        # If no cluster data, show the selection summary funnel instead
        if labels is None:
            self._plot_summary_fallback(chart, stats, plt)
            return

        labels = np.asarray(labels)
        combined = results.get("combined_features")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor("#1E1E1E")
        for ax in axes:
            ax.set_facecolor("#252526")

        # Cluster size bar chart
        unique, counts = np.unique(labels, return_counts=True)
        n_show = min(len(unique), 30)
        axes[0].bar(range(n_show), counts[:n_show], color="#2A82DA", alpha=0.7, edgecolor="#1E1E1E")
        if len(unique) > 30:
            axes[0].set_xlabel(f"Cluster ID (first 30 of {len(unique)})", color="#D4D4D4")
        else:
            axes[0].set_xlabel("Cluster ID", color="#D4D4D4")
        axes[0].set_ylabel("Size", color="#D4D4D4")
        axes[0].set_title("Cluster Sizes", color="#D4D4D4")
        axes[0].tick_params(colors="#7F7F7F")

        # PCA of combined features colored by cluster
        if combined is not None:
            try:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2)
                pts = pca.fit_transform(combined)
                scatter = axes[1].scatter(
                    pts[:, 0], pts[:, 1],
                    c=labels, cmap="tab20", alpha=0.4, s=8,
                )
                if selected_indices:
                    mask = np.zeros(len(pts), dtype=bool)
                    valid = [i for i in selected_indices if i < len(pts)]
                    mask[valid] = True
                    axes[1].scatter(
                        pts[mask, 0], pts[mask, 1],
                        c="#FF4444", marker="*", s=60, edgecolors="white", linewidths=0.5,
                        label=f"Selected ({len(selected_indices)})",
                    )
                    axes[1].legend(facecolor="#2D2D2D", edgecolor="#3C3C3C", labelcolor="#D4D4D4")
                axes[1].set_title("Clusters (PCA)", color="#D4D4D4")
                axes[1].tick_params(colors="#7F7F7F")
            except ImportError:
                axes[1].text(0.5, 0.5, "sklearn not installed", transform=axes[1].transAxes,
                             ha="center", va="center", color="#7F7F7F", fontsize=12)

        fig.tight_layout()
        chart.update_figure(fig)
        plt.close(fig)

    def _plot_summary_fallback(self, chart: ChartWidget, stats: Dict, plt):
        """Show a selection-stages funnel when cluster data is unavailable."""
        if not stats:
            self._placeholder(chart, "Cluster", "No cluster / summary data available", plt)
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#1E1E1E")
        ax.set_facecolor("#252526")

        stages = ["Original", "Valid", "Final"]
        counts = [
            stats.get("original_count", 0),
            stats.get("valid_count", 0),
            stats.get("final_count", 0),
        ]
        colors = ["#3498DB", "#2ECC71", "#E74C3C"]
        bars = ax.barh(stages, counts, color=colors, alpha=0.8, edgecolor="#1E1E1E")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", color="#D4D4D4", fontsize=10)
        ax.set_title("Selection Pipeline Stages", color="#D4D4D4")
        ax.set_xlabel("Count", color="#D4D4D4")
        ax.tick_params(colors="#7F7F7F")
        ax.invert_yaxis()
        fig.tight_layout()
        chart.update_figure(fig)
        plt.close(fig)

    @staticmethod
    def _placeholder(chart: ChartWidget, name: str, msg: str, plt):
        import matplotlib
        matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#1E1E1E")
        ax.set_facecolor("#252526")
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#7F7F7F")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, color="#D4D4D4")
        chart.update_figure(fig)
        plt.close(fig)
