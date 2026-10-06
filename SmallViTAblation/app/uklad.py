"""Układ „przepływowy”: kontrolki w wierszu, a gdy brakuje miejsca, zawijają się do następnej linii
(jak tekst). Dzięki temu panele z wieloma opcjami nie rozpychają okna na szerokość."""
from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLayout, QPushButton, QTabWidget, QVBoxLayout, QWidget, QWidgetItem


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
    if isinstance(uklad, Przeplyw):                 # etykieta i pole w jednym kawałku: zawijają się razem
        para = QWidget(); l = QHBoxLayout(para); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(4)
        l.addWidget(e); l.addWidget(widget)
        uklad.addWidget(para)
    else:
        uklad.addWidget(e); uklad.addWidget(widget)
    return widget


class PanelOpcji:
    """Opcje widoku: pasek przycisków zawsze widoczny (z przyciskiem „Opcje”, który chowa resztę) i zakładki
    z grupami opcji. Dzięki temu widok 3D dostaje prawie całą wysokość, a opcje są pogrupowane tematycznie."""

    def __init__(self, uklad, strony):
        from styl import rola
        self.pasek = Przeplyw(); uklad.addLayout(self.pasek)
        self.btn_opcje = rola(QPushButton('Opcje ▾'), 'narzedzie'); self.btn_opcje.setCheckable(True)
        self.btn_opcje.setChecked(True)
        self.btn_opcje.setToolTip('Pokaż albo schowaj opcje widoku (więcej miejsca na widok).')
        self.pasek.addWidget(self.btn_opcje)
        self.zakladki = QTabWidget(); self.zakladki.setObjectName('opcje_widoku'); self.zakladki.setDocumentMode(True)
        self._strony = {}
        for n in strony:
            s = QWidget(); lay = QVBoxLayout(s); lay.setContentsMargins(6, 4, 6, 4); lay.setSpacing(2)
            self.zakladki.addTab(s, n); self._strony[n] = lay
        uklad.addWidget(self.zakladki)
        self.btn_opcje.toggled.connect(lambda v: (self.zakladki.setVisible(v),
                                                  self.btn_opcje.setText('Opcje ▾' if v else 'Opcje ▸')))

    def wiersz(self, strona):
        w = Przeplyw(); self._strony[strona].addLayout(w)
        return w

    def strona(self, strona):
        return self._strony[strona]

    def zakoncz(self):
        for lay in self._strony.values():
            lay.addStretch(1)
