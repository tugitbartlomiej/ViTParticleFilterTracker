"""Reusable topic-tag bar: colored pill-shaped buttons for session topics."""

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget
from PyQt6.QtCore import Qt


# Color palette shared with the rest of the experiments tab
TOPIC_COLORS: dict[str, str] = {
    "DETR": "#2EA043",
    "YOLO": "#DA3633",
    "Dataset Selection": "#8957E5",
    "SSH Eden": "#F0883E",
    "GPU": "#58A6FF",
    "Pipeline": "#3FB950",
    "IEEE": "#D29922",
    "Fourier": "#A371F7",
    "EL2N": "#79C0FF",
    "DINO": "#56D364",
    "Query 81": "#DB6D28",
    "SAM": "#F778BA",
    "K-Center": "#BC8CFF",
    "K-Means": "#B392F0",
    "Benchmark": "#388BFD",
    "Visualization": "#39D353",
    "RAG": "#E3B341",
    "CataractAI Workbench": "#58A6FF",
}

DEFAULT_TOPIC_COLOR = "#8B949E"


class TopicTag(QPushButton):
    """Small colored pill button representing a single topic."""

    def __init__(self, topic: str, parent=None):
        super().__init__(topic, parent)
        color = TOPIC_COLORS.get(topic, DEFAULT_TOPIC_COLOR)
        self.setFixedHeight(24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: white; "
            f"border: none; border-radius: 10px; padding: 2px 10px; "
            f"font-size: 11px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {color}; opacity: 0.8; }}"
        )


class TopicTagsBar(QWidget):
    """Horizontal bar that displays a set of topic tags.

    Usage::

        bar = TopicTagsBar()
        bar.set_topics(["DETR", "GPU", "Benchmark"])
        bar.clear_topics()
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self._label = QLabel("Topics:")
        self._label.setStyleSheet("color: #8B949E; font-size: 11px;")
        self._layout.addWidget(self._label)
        self._layout.addStretch()

    def set_topics(self, topics: list[str]) -> None:
        """Replace the current tags with new ones."""
        self.clear_topics()
        for topic in topics:
            tag = TopicTag(topic)
            # Insert before the trailing stretch
            self._layout.insertWidget(self._layout.count() - 1, tag)

    def clear_topics(self) -> None:
        """Remove all tag widgets, keeping the label and stretch."""
        while self._layout.count() > 2:
            item = self._layout.takeAt(1)
            if item and item.widget():
                item.widget().deleteLater()
