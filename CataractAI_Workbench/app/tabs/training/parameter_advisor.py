"""Parameter Advisor widget -- UI for analyzing training config and recommendations."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...widgets.file_picker import FilePicker
from ...core.analysis_services import ParameterRecommendationEngine
from .theme_constants import (
    ACCENT_YELLOW,
    ACCENT_RED,
    BG,
    BORDER,
    FG,
    GREEN_BUTTON_STYLE,
    GROUP_STYLE,
    TEXT_STYLE,
    make_canvas,
    style_axis,
)


class ParameterAdvisor(QWidget):
    """Analyze training config and provide rule-based recommendations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._service = ParameterRecommendationEngine()
        self._build_ui()

    # ---------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel(), stretch=1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMaximumWidth(320)
        panel.setMinimumWidth(240)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        grp = QGroupBox("Current Config")
        grp.setStyleSheet(GROUP_STYLE)
        grp_layout = QVBoxLayout(grp)

        grp_layout.addWidget(QLabel("Config JSON:"))
        self._config_picker = FilePicker(
            label="Browse...", mode="file",
            filter_str="JSON (*.json);;All (*)",
        )
        grp_layout.addWidget(self._config_picker)

        grp_layout.addWidget(QLabel("Dataset dir (optional):"))
        self._dataset_picker = FilePicker(label="Browse...", mode="dir")
        grp_layout.addWidget(self._dataset_picker)

        grp_layout.addWidget(QLabel("Training history (optional):"))
        self._history_picker = FilePicker(
            label="Browse...", mode="file",
            filter_str="JSON (*.json);;All (*)",
        )
        grp_layout.addWidget(self._history_picker)

        self._btn_analyze = QPushButton("Analyze")
        self._btn_analyze.setStyleSheet(GREEN_BUTTON_STYLE)
        self._btn_analyze.clicked.connect(self._run_analysis)
        grp_layout.addWidget(self._btn_analyze)

        self._config_summary = QTextEdit()
        self._config_summary.setReadOnly(True)
        self._config_summary.setFont(QFont("Consolas", 9))
        self._config_summary.setStyleSheet(TEXT_STYLE)
        grp_layout.addWidget(self._config_summary)

        layout.addWidget(grp)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Recommendations
        grp_rec = QGroupBox("Recommendations")
        grp_rec.setStyleSheet(GROUP_STYLE)
        rec_layout = QVBoxLayout(grp_rec)

        self._rec_text = QTextEdit()
        self._rec_text.setReadOnly(True)
        self._rec_text.setFont(QFont("Consolas", 9))
        self._rec_text.setStyleSheet(TEXT_STYLE)
        rec_layout.addWidget(self._rec_text)

        layout.addWidget(grp_rec, stretch=1)

        # Training history analysis
        grp_hist = QGroupBox("Training History Analysis")
        grp_hist.setStyleSheet(GROUP_STYLE)
        hist_layout = QVBoxLayout(grp_hist)

        self._history_text = QTextEdit()
        self._history_text.setReadOnly(True)
        self._history_text.setFont(QFont("Consolas", 9))
        self._history_text.setStyleSheet(TEXT_STYLE)
        self._history_text.setMaximumHeight(180)
        hist_layout.addWidget(self._history_text)

        self._loss_fig, self._loss_canvas = make_canvas()
        self._loss_canvas.setMinimumHeight(180)
        hist_layout.addWidget(self._loss_canvas)

        layout.addWidget(grp_hist, stretch=1)
        return panel

    # ---------------------------------------------------------------
    # Analysis orchestration
    # ---------------------------------------------------------------

    def _run_analysis(self) -> None:
        config_path = self._config_picker.path()
        if not config_path or not Path(config_path).is_file():
            QMessageBox.warning(
                self, "No Config",
                "Please select a valid TrainingSettings.json file.",
            )
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as exc:
            QMessageBox.warning(self, "Parse Error", f"Cannot parse config: {exc}")
            return

        # Load optional training history
        history = None
        history_path = self._history_picker.path()
        if history_path and Path(history_path).is_file():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception as exc:
                self._history_text.setPlainText(f"Cannot parse history: {exc}")

        result = self._service.analyze({
            "config": config,
            "dataset_dir": self._dataset_picker.path(),
            "history": history,
        })

        self._config_summary.setPlainText(result["summary"])
        self._rec_text.setPlainText(result["recommendations"])

        if "history_analysis" in result:
            ha = result["history_analysis"]
            self._history_text.setPlainText(ha["text"])
            self._plot_loss_curve(ha["train_losses"], ha["val_losses"])
        else:
            self._history_text.setPlainText(
                "No training history file selected.\n"
                "Load a training_history.json to analyze loss curves."
            )
            self._loss_fig.clear()
            self._loss_canvas.draw()

    # ---------------------------------------------------------------
    # Loss curve chart
    # ---------------------------------------------------------------

    def _plot_loss_curve(self, train_losses: list, val_losses: list) -> None:
        self._loss_fig.clear()
        ax = self._loss_fig.add_subplot(111)
        style_axis(ax, title="Loss Curve", xlabel="Epoch", ylabel="Loss")

        epochs = list(range(1, len(train_losses) + 1))
        ax.plot(epochs, train_losses, color=ACCENT_YELLOW, marker="o",
                markersize=4, linewidth=1.5, label="Train Loss")

        if val_losses:
            val_epochs = list(range(1, len(val_losses) + 1))
            ax.plot(val_epochs, val_losses, color=ACCENT_RED, marker="s",
                    markersize=4, linewidth=1.5, label="Val Loss")

        ax.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=FG)
        self._loss_fig.tight_layout()
        self._loss_canvas.draw()
