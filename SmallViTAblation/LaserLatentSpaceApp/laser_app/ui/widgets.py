"""Proste widżety wspólne dla zakładki i rzutów: zwijana sekcja panelu (CollapsibleSection), pole liczbowe (_spin),
lista liczb z tekstu (lista_liczb), napis o stałym układzie (_staly_uklad)."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QDoubleSpinBox, QGroupBox, QSizePolicy, QSpinBox, QToolButton, QWidget


class CollapsibleSection(QGroupBox):
    """Sekcja panelu zwijana strzałką w nagłówku: ▾ rozwinięta, ▸ zwinięta (klik w cały nagłówek). Zwinięta pokazuje tylko nagłówek.
    Nagłówek leży nad treścią (górny margines treści = jego wysokość), więc sekcje budują się jak zwykły QGroupBox: QFormLayout(g)."""

    def __init__(self, tytul):
        super().__init__()
        self.naglowek = QToolButton(self)
        self.naglowek.setText(tytul); self.naglowek.setCheckable(True); self.naglowek.setChecked(True)
        self.naglowek.setArrowType(Qt.ArrowType.DownArrow)
        self.naglowek.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.naglowek.setAutoRaise(True)
        self.naglowek.setStyleSheet('QToolButton { border: none; font-weight: bold; padding: 3px 2px; }'
                                    'QToolButton:hover { background: rgba(255, 255, 255, 0.07); }')
        self.naglowek.setToolTip('Kliknij, żeby zwinąć (▸) albo rozwinąć (▾) sekcję.')
        self.naglowek.toggled.connect(self._zwin)
        self._wys = 0
        self._przelicz()

    def _przelicz(self):
        """Wysokość nagłówka z jego faktycznego rozmiaru (pogrubiona czcionka, arkusz stylów, skalowanie DPI) i margines treści równy
        tej wysokości. Liczona po nałożeniu stylu, nie w konstruktorze: tam sizeHint jest za mały i treść wchodziła pod nagłówek."""
        self.naglowek.ensurePolished()
        wys = self.naglowek.sizeHint().height() + 6
        if wys != self._wys:
            self._wys = wys
            self.naglowek.setGeometry(3, 1, max(self.width() - 6, 10), wys - 2)
            if self.naglowek.isChecked():
                self.setContentsMargins(0, wys, 0, 0)
            else:
                self.setMaximumHeight(wys + 3)
        elif self.contentsMargins().top() != wys and self.naglowek.isChecked():
            self.setContentsMargins(0, wys, 0, 0)       # QGroupBox sam przelicza ramkę przy zmianie stylu i zeruje nasz margines

    def event(self, ev):
        wynik = super().event(ev)
        if ev.type() in (QEvent.Type.Polish, QEvent.Type.StyleChange, QEvent.Type.FontChange, QEvent.Type.Show):
            self._przelicz()
        return wynik

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._przelicz()
        self.naglowek.setGeometry(3, 1, max(self.width() - 6, 10), self._wys - 2)
        self.naglowek.raise_()

    def zwinieta(self):
        return not self.naglowek.isChecked()

    def _zwin(self, on):
        self.naglowek.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        for w in self.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            if w is not self.naglowek:
                w.setVisible(on)
        self.setMaximumHeight(16777215 if on else self._wys + 3)       # zwinięta: sam nagłówek (bez marginesów pustej treści)


def _staly_uklad(lb):
    """Napis zmieniany w każdej klatce nie może zmieniać układu okna: inaczej Qt przelicza doki i zmienia rozmiar widoków OpenGL
    (ponowne bufory, przerwy ~100 ms). Szerokość z układu, nie z tekstu; wysokość stała (jedna linia)."""
    from PyQt6.QtWidgets import QSizePolicy
    lb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    lb.setFixedHeight(lb.fontMetrics().height() + 8)
    lb.setWordWrap(False)


def lista_liczb(tekst, n):
    """'12, 40 100-105' → [12, 40, 100, …, 105] (bez powtórzeń, kolejność wpisania, tylko 0..n−1)."""
    out, byly = [], set()
    for kaw in tekst.replace(';', ',').replace(',', ' ').split():
        a, _, b = kaw.partition('-')
        try:
            lo = int(a); hi = int(b) if b else lo
        except ValueError:
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        for x in range(max(lo, 0), min(hi, n - 1) + 1):                 # zakres przycięty do 0..n−1 (np. „0-100000000” bez zawieszenia)
            if x not in byly:
                byly.add(x); out.append(x)
    return out


def _spin(lo, hi, v, krok=1, dec=None, sufiks=''):
    s = QDoubleSpinBox() if dec is not None else QSpinBox()
    if dec is not None:
        s.setDecimals(dec)
    s.setRange(lo, hi); s.setSingleStep(krok); s.setValue(v)
    if sufiks:
        s.setSuffix(sufiks)
    return s
