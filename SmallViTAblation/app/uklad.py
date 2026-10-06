"""Układ „przepływowy”: kontrolki w wierszu, a gdy brakuje miejsca, zawijają się do następnej linii
(jak tekst). Dzięki temu panele z wieloma opcjami nie rozpychają okna na szerokość."""
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLabel, QLayout, QWidgetItem


class Przeplyw(QLayout):
    def __init__(self, rodzic=None, odstep=6):
        super().__init__(rodzic)
        self._el = []
        self._odstep = odstep
        if rodzic is not None:
            self.setContentsMargins(0, 0, 0, 0)

    # zgodność z QHBoxLayout: addWidget(w, rozciąganie) i addStretch() bez skutku
    def addWidget(self, w, *_):
        super().addWidget(w)

    def addStretch(self, *_):
        pass

    def addItem(self, el):
        self._el.append(el)

    def count(self):
        return len(self._el)

    def itemAt(self, i):
        return self._el[i] if 0 <= i < len(self._el) else None

    def takeAt(self, i):
        return self._el.pop(i) if 0 <= i < len(self._el) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, szer):
        return self._uloz(QRect(0, 0, szer, 0), tylko_licz=True)

    def setGeometry(self, r):
        super().setGeometry(r)
        self._uloz(r, tylko_licz=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for el in self._el:
            s = s.expandedTo(el.minimumSize())
        m = self.contentsMargins()
        return s + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _uloz(self, r, tylko_licz):
        m = self.contentsMargins()
        x0, y = r.x() + m.left(), r.y() + m.top()
        prawa = r.right() - m.right()
        x, wys_linii = x0, 0
        for el in self._el:
            if isinstance(el, QWidgetItem) and el.widget() is not None and el.widget().isHidden():
                continue
            h = el.sizeHint()
            if x + h.width() > prawa and wys_linii > 0:
                x, y = x0, y + wys_linii + self._odstep // 2
                wys_linii = 0
            if not tylko_licz:
                el.setGeometry(QRect(QPoint(x, y), h))
            x += h.width() + self._odstep
            wys_linii = max(wys_linii, h.height())
        return y + wys_linii - r.y() + m.bottom()


def dodaj(uklad, nazwa, widget, opis):
    """Etykieta i kontrolka obok siebie w układzie, ten sam opis jako podpowiedź obu; zwraca kontrolkę."""
    e = QLabel(nazwa); e.setToolTip(opis); widget.setToolTip(opis)
    uklad.addWidget(e); uklad.addWidget(widget)
    return widget
