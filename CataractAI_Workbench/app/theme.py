"""Dark/light theme for the application."""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


def apply_dark_theme(app: QApplication):
    """Apply dark Fusion theme to the application."""
    app.setStyle("Fusion")

    palette = QPalette()

    # Base colors
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Text, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(212, 212, 212))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    # Disabled colors
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(127, 127, 127))

    app.setPalette(palette)

    # Additional QSS for fine-tuning
    app.setStyleSheet("""
        QTabWidget::pane {
            border: 1px solid #3C3C3C;
            background: #1E1E1E;
        }
        QTabBar::tab {
            background: #2D2D2D;
            color: #D4D4D4;
            padding: 8px 16px;
            border: 1px solid #3C3C3C;
            border-bottom: none;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #1E1E1E;
            border-bottom: 2px solid #2A82DA;
        }
        QTabBar::tab:hover {
            background: #3C3C3C;
        }
        QGroupBox {
            border: 1px solid #3C3C3C;
            margin-top: 8px;
            padding-top: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QSplitter::handle {
            background: #3C3C3C;
            width: 2px;
            height: 2px;
        }
        QStatusBar {
            background: #252526;
            color: #D4D4D4;
            border-top: 1px solid #3C3C3C;
        }
        QPushButton {
            padding: 5px 15px;
            border: 1px solid #3C3C3C;
            border-radius: 3px;
            background: #2D2D2D;
        }
        QPushButton:hover {
            background: #3C3C3C;
        }
        QPushButton:pressed {
            background: #1E1E1E;
        }
        QPushButton:disabled {
            color: #7F7F7F;
        }
        QProgressBar {
            border: 1px solid #3C3C3C;
            border-radius: 3px;
            text-align: center;
            background: #1E1E1E;
        }
        QProgressBar::chunk {
            background: #2A82DA;
            border-radius: 2px;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            padding: 4px;
            border: 1px solid #3C3C3C;
            border-radius: 3px;
            background: #1E1E1E;
            color: #D4D4D4;
        }
        QTreeWidget, QTableWidget, QListWidget {
            border: 1px solid #3C3C3C;
            background: #1E1E1E;
            alternate-background-color: #252526;
        }
        QHeaderView::section {
            background: #2D2D2D;
            color: #D4D4D4;
            padding: 4px;
            border: 1px solid #3C3C3C;
        }
        QScrollBar:vertical {
            background: #1E1E1E;
            width: 12px;
        }
        QScrollBar::handle:vertical {
            background: #3C3C3C;
            border-radius: 4px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #505050;
        }
    """)


def apply_light_theme(app: QApplication):
    """Reset to default light theme."""
    app.setStyle("Fusion")
    app.setPalette(QApplication.style().standardPalette())
    app.setStyleSheet("")
