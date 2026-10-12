"""Pasek głównych zakładek okna: każda zakładka w przewijanym polu i leniwe budowanie zakładek (zastępnik
podmieniany na prawdziwą zakładkę przy pierwszym wejściu na kartę, patrz `app.Okno.zbuduj_zakladke`)."""
from PyQt6.QtWidgets import QScrollArea, QTabWidget, QWidget


class _StronaBezHfw(QWidget):
    """Pośrednik między przewijanym polem a zakładką, bez własnego układu. QScrollArea pyta układ swojego widżetu
    o wysokość zależną od szerokości (heightForWidth); zakładki z układami przepływowymi i zawijanymi napisami
    zwracały wtedy coraz większą wysokość i strona rosła ponad okno (2700 -> 3800 px). Tu zakładka dostaje
    dokładnie rozmiar pośrednika, a pośrednik zgłasza tylko jej minimalny rozmiar."""

    def __init__(self, strona):
        super().__init__()
        self.strona = strona
        strona.setParent(self); strona.show()
        strona.installEventFilter(self)

    def podmien(self, nowa):
        """Wstawia w miejsce zakładki-zastępnika prawdziwą zakładkę (leniwe budowanie zakładek)."""
        stara = self.strona
        stara.removeEventFilter(self)
        stara.setParent(None); stara.deleteLater()
        self.strona = nowa
        nowa.setParent(self)
        nowa.setGeometry(0, 0, self.width(), self.height())
        nowa.show()
        nowa.installEventFilter(self)
        self.updateGeometry()

    def minimumSizeHint(self):
        return self.strona.minimumSizeHint()

    def sizeHint(self):
        return self.strona.minimumSizeHint()

    def resizeEvent(self, e):
        self.strona.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(e)

    def eventFilter(self, obj, ev):
        from PyQt6.QtCore import QEvent
        if obj is self.strona and ev.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()                  # minimum zakładki się zmieniło: pole przewijane przelicza paski
        return False


class ZakladkiGlowne(QTabWidget):
    """Główne zakładki, każda w przewijanym polu. QTabWidget ma minimalny rozmiar równy największemu minimum ze
    wszystkich zakładek, więc jedna wysoka zakładka (np. detekcja, około 1240 px) podnosiła minimalną wysokość całego
    okna ponad ekran; przy każdej zmianie układu (pokaż / schowaj opcje) Qt powiększał wtedy okno poza ekran i psuł
    maksymalizację. W przewijanym polu za wysoka zakładka dostaje pasek przewijania, a okno zostaje w ekranie.
    widget / currentWidget / indexOf / setCurrentWidget działają na samych zakładkach (bez pola)."""

    def addTab(self, w, tekst):
        from PyQt6.QtWidgets import QFrame
        sc = QScrollArea(); sc.setWidget(_StronaBezHfw(w)); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.Shape.NoFrame)
        return super().addTab(sc, tekst)

    @staticmethod
    def _wew(sc):
        return sc.widget().strona if isinstance(sc, QScrollArea) else sc

    def widget(self, i):
        return self._wew(super().widget(i))

    def currentWidget(self):
        return self._wew(super().currentWidget())

    def indexOf(self, w):
        for i in range(self.count()):
            if self.widget(i) is w or super().widget(i) is w:
                return i
        return -1

    def setCurrentWidget(self, w):
        i = self.indexOf(w)
        if i >= 0:
            self.setCurrentIndex(i)

    def podmien(self, zastepnik, nowa):
        """Podmienia zastępnik zakładki na prawdziwą zakładkę; tytuł i miejsce na pasku kart zostają."""
        i = self.indexOf(zastepnik)
        if i >= 0:
            super().widget(i).widget().podmien(nowa)


class _Odnosnik:
    """Leniwy odnośnik do zakładki: pierwsze użycie dowolnego atrybutu buduje zakładkę (np. korekta i pchnięcie
    czytają dane zakładki uczenia, ale jej budowanie nie ma sensu, dopóki nikt ich nie potrzebuje)."""

    def __init__(self, okno, nazwa):
        object.__setattr__(self, '_okno', okno)
        object.__setattr__(self, '_nazwa', nazwa)

    def __getattr__(self, n):
        return getattr(getattr(self._okno, self._nazwa), n)
