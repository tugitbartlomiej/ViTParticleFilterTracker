"""Chart rendering logic for benchmark results.

Extracts the matplotlib Figure creation from ResultsViewer so that
results_viewer.py stays focused on table and data management.
"""

from typing import Dict, List

from matplotlib.figure import Figure
import numpy as np


# Shared dark-theme styling constants
_BG_COLOR = "#1E1E1E"
_AXES_BG = "#2D2D2D"
_TEXT_COLOR = "#D4D4D4"
_SPINE_COLOR = "#555"
_GRID_COLOR = "#666"


def _style_axes(ax):
    """Apply consistent dark-theme styling to axes."""
    ax.set_facecolor(_AXES_BG)
    ax.tick_params(colors=_TEXT_COLOR)
    ax.spines["bottom"].set_color(_SPINE_COLOR)
    ax.spines["left"].set_color(_SPINE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_bar_chart(rows: List[dict]) -> Figure:
    """Create a grouped bar chart comparing models on key metrics.

    Parameters
    ----------
    rows:
        Flat result dicts with keys ``model``, ``epoch``, ``mAP@0.5``,
        ``precision``, ``recall``, ``f1_score``.

    Returns
    -------
    Figure
        Matplotlib figure ready to be rendered.
    """
    fig = Figure(figsize=(8, 5), dpi=100)
    fig.patch.set_facecolor(_BG_COLOR)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    metrics_to_plot = ["mAP@0.5", "precision", "recall", "f1_score"]
    metric_labels = ["mAP@50", "Precision", "Recall", "F1"]

    labels = []
    values_per_metric: Dict[str, list] = {m: [] for m in metrics_to_plot}

    for row in rows:
        model = str(row.get("model", "?"))
        epoch = row.get("epoch", "")
        label = f"{model} ep{epoch}" if epoch else model
        labels.append(label)
        for m in metrics_to_plot:
            v = row.get(m, 0)
            try:
                values_per_metric[m].append(float(v))
            except (ValueError, TypeError):
                values_per_metric[m].append(0.0)

    x = np.arange(len(labels))
    n_metrics = len(metrics_to_plot)
    width = 0.8 / max(n_metrics, 1)
    colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373"]

    for i, (m, ml) in enumerate(zip(metrics_to_plot, metric_labels)):
        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(
            x + offset, values_per_metric[m], width,
            label=ml, color=colors[i], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8, color=_TEXT_COLOR)
    ax.set_ylabel("Score (%)", color=_TEXT_COLOR)
    ax.set_title("Model Comparison", color=_TEXT_COLOR, fontsize=12)
    ax.legend(
        fontsize=8, facecolor=_AXES_BG,
        edgecolor=_SPINE_COLOR, labelcolor=_TEXT_COLOR,
    )
    fig.tight_layout()

    return fig


def draw_line_chart(rows: List[dict]) -> Figure:
    """Create a line chart showing metric progression across epochs.

    Parameters
    ----------
    rows:
        Flat result dicts with keys ``model``, ``epoch``, ``mAP@0.5``,
        ``f1_score``.

    Returns
    -------
    Figure
        Matplotlib figure ready to be rendered.
    """
    fig = Figure(figsize=(8, 5), dpi=100)
    fig.patch.set_facecolor(_BG_COLOR)
    ax = fig.add_subplot(111)
    _style_axes(ax)

    # Group data by model type
    model_groups: Dict[str, Dict[str, list]] = {}
    for row in rows:
        model = str(row.get("model", "Unknown"))
        epoch = row.get("epoch", 0)
        try:
            epoch = int(epoch)
        except (ValueError, TypeError):
            continue

        if model not in model_groups:
            model_groups[model] = {"epochs": [], "mAP@0.5": [], "f1_score": []}

        model_groups[model]["epochs"].append(epoch)
        for m in ("mAP@0.5", "f1_score"):
            v = row.get(m, 0)
            try:
                model_groups[model][m].append(float(v))
            except (ValueError, TypeError):
                model_groups[model][m].append(0.0)

    colors_map = {
        0: ("#4FC3F7", "#81C784"),
        1: ("#FFB74D", "#E57373"),
        2: ("#CE93D8", "#F48FB1"),
        3: ("#80CBC4", "#A5D6A7"),
    }

    for idx, (model, data) in enumerate(model_groups.items()):
        if not data["epochs"]:
            continue
        order = np.argsort(data["epochs"])
        epochs = np.array(data["epochs"])[order]
        col_pair = colors_map.get(idx, ("#FFFFFF", "#CCCCCC"))

        mAP = np.array(data["mAP@0.5"])[order]
        ax.plot(
            epochs, mAP, "o-", color=col_pair[0],
            label=f"{model} mAP@50", markersize=4,
        )

        f1 = np.array(data["f1_score"])[order]
        ax.plot(
            epochs, f1, "s--", color=col_pair[1],
            label=f"{model} F1", markersize=4,
        )

    ax.set_xlabel("Epoch", color=_TEXT_COLOR)
    ax.set_ylabel("Score (%)", color=_TEXT_COLOR)
    ax.set_title("Epoch Progression", color=_TEXT_COLOR, fontsize=12)
    ax.legend(
        fontsize=8, facecolor=_AXES_BG,
        edgecolor=_SPINE_COLOR, labelcolor=_TEXT_COLOR,
    )
    ax.grid(True, alpha=0.2, color=_GRID_COLOR)
    fig.tight_layout()

    return fig
