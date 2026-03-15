"""CataractAI Workbench - Entry point."""

import sys
from pathlib import Path

# Add project root to path so we can import existing modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt

from .theme import apply_dark_theme
from .main_window import MainWindow


def create_splash(app: QApplication) -> QSplashScreen:
    """Create a splash screen."""
    pixmap = QPixmap(500, 300)
    pixmap.fill(Qt.GlobalColor.black)

    splash = QSplashScreen(pixmap)
    splash.setFont(QFont("Segoe UI", 12))
    splash.showMessage(
        "CataractAI Workbench\nLoading modules...",
        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom,
        Qt.GlobalColor.white,
    )
    return splash


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CataractAI Workbench")
    app.setOrganizationName("PhD Project")

    apply_dark_theme(app)

    splash = create_splash(app)
    splash.show()
    app.processEvents()

    window = MainWindow()
    window.show()
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
