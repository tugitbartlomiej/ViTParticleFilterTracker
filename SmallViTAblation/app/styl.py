"""Wspólny styl przycisków: hierarchia ważności zamiast jednakowych szarych przycisków.

- rola „glowny”: główna akcja zakładki (Generuj, Wykryj, Auto-szukanie, Konwertuj, Odtwarzaj), niebieski, pogrubiony;
- rola „narzedzie”: małe przyciski widoku (kamera, PNG, osobne okno), niższe i węższe;
- bez roli: zwykły przycisk w stylu systemu.
Styl dotyczy tylko przycisków z ustawioną rolą, reszta zostaje w motywie systemu (ciemny Windows 11).
"""
from PyQt6.QtWidgets import QPushButton

STYL = """
QPushButton[rola="glowny"] {
    background: #2f6fd6; color: white; font-weight: bold; border: 1px solid #4c86e6; border-radius: 4px;
    padding: 5px 16px;
}
QPushButton[rola="glowny"]:hover { background: #3b7ce6; }
QPushButton[rola="glowny"]:pressed, QPushButton[rola="glowny"]:checked { background: #2459ad; }
QPushButton[rola="glowny"]:disabled { background: #3a3d44; color: #8a8d93; border-color: #4a4d54; }
QPushButton[rola="narzedzie"] {
    background: #2d2f34; color: #dcdcdc; border: 1px solid #45484f; border-radius: 3px; padding: 2px 8px;
}
QPushButton[rola="narzedzie"]:hover { background: #383b41; }
QPushButton[rola="narzedzie"]:checked { background: #2459ad; color: white; }
QGroupBox { font-weight: bold; }
QGroupBox QLabel, QGroupBox QCheckBox, QGroupBox QPushButton, QGroupBox QComboBox { font-weight: normal; }
QTabWidget#opcje_widoku::pane { border: 1px solid #3c3f45; border-top: none; }
"""


def rola(przycisk: QPushButton, nazwa: str) -> QPushButton:
    """Ustawia rolę przycisku (patrz STYL) i odświeża jego wygląd; zwraca ten sam przycisk."""
    przycisk.setProperty('rola', nazwa)
    przycisk.style().unpolish(przycisk); przycisk.style().polish(przycisk)
    return przycisk
