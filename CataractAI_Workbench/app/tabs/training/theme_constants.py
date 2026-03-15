"""Shared dark-theme constants and matplotlib helpers for training widgets."""

from __future__ import annotations

from PyQt6.QtWidgets import QSizePolicy

# -------------------------------------------------------------------
# Color palette
# -------------------------------------------------------------------

BG = "#1E1E1E"
FG = "#D4D4D4"
BORDER = "#3C3C3C"
ALT_ROW = "#252526"
ACCENT_GREEN = "#6BCB77"
ACCENT_YELLOW = "#FFD93D"
ACCENT_RED = "#FF6B6B"
ACCENT_BLUE = "#6BCBF0"

# -------------------------------------------------------------------
# Reusable style sheets
# -------------------------------------------------------------------

TREE_STYLE = (
    f"QTreeWidget {{ background-color: {BG}; color: {FG}; "
    f"border: 1px solid {BORDER}; alternate-background-color: {ALT_ROW}; }}"
    f"QHeaderView::section {{ background-color: #2D2D2D; color: {FG}; "
    f"border: 1px solid {BORDER}; padding: 3px; }}"
)

GROUP_STYLE = (
    f"QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; "
    f"margin-top: 8px; padding-top: 14px; color: {FG}; }}"
    f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; "
    f"padding: 0 4px; color: {ACCENT_BLUE}; }}"
)

TEXT_STYLE = (
    f"QTextEdit {{ background-color: {BG}; color: {FG}; "
    f"border: 1px solid {BORDER}; }}"
)

BLUE_BUTTON_STYLE = (
    "QPushButton { background-color: #2A6DB0; color: white; "
    "border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
    "QPushButton:hover { background-color: #3A8DD0; }"
)

PURPLE_BUTTON_STYLE = (
    "QPushButton { background-color: #6A4C93; color: white; "
    "border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
    "QPushButton:hover { background-color: #8A6CB3; }"
)

GREEN_BUTTON_STYLE = (
    "QPushButton { background-color: #2EA043; color: white; "
    "border-radius: 4px; padding: 8px 16px; font-weight: bold; font-size: 12px; }"
    "QPushButton:hover { background-color: #3FB950; }"
)

TAB_STYLE = (
    f"QTabWidget::pane {{ border: 1px solid {BORDER}; }}"
    f"QTabBar::tab {{ background: #2D2D2D; color: {FG}; padding: 6px 16px; "
    f"border: 1px solid {BORDER}; border-bottom: none; margin-right: 2px; }}"
    f"QTabBar::tab:selected {{ background: {BG}; color: {ACCENT_BLUE}; "
    f"border-bottom: 2px solid {ACCENT_BLUE}; }}"
    f"QTabBar::tab:hover {{ background: #383838; }}"
)


# -------------------------------------------------------------------
# Matplotlib helpers
# -------------------------------------------------------------------

def make_canvas(fig=None):
    """Return ``(figure, canvas)`` using the matplotlib Qt backend."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    if fig is None:
        fig = Figure(figsize=(5, 3), dpi=100)
    fig.patch.set_facecolor(BG)
    canvas = FigureCanvasQTAgg(fig)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return fig, canvas


def style_axis(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply the dark theme to a matplotlib *Axes*."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
