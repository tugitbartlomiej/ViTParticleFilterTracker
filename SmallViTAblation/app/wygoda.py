"""Wygodne wprowadzanie parametrów w całej aplikacji.

- Kółko myszy nad polem liczbowym, listą albo suwakiem, w które nie kliknięto, nie zmienia wartości, tylko
  przewija panel (najbliższy obszar przewijany). Wcześniej przewijanie listy 64 kanałów albo opcji zmieniało
  po drodze wartości pól pod kursorem. Kliknięte pole (z fokusem) dalej reaguje na kółko.
- Pola liczbowe zatwierdzają wpisaną wartość Enterem albo wyjściem z pola (bez keyboardTracking), więc widoki
  nie przeliczają się po każdej cyfrze; strzałki i kółko zmieniają wartość od razu.
- Przytrzymana strzałka pola przyspiesza (setAccelerated).
"""
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QAbstractScrollArea, QAbstractSlider, QAbstractSpinBox, QComboBox, QScrollBar


class _KolkoTylkoZFokusem(QObject):
    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.Wheel and not obj.hasFocus():
            w = obj.parentWidget()
            while w is not None and not isinstance(w, QAbstractScrollArea):
                w = w.parentWidget()
            if w is not None:                                   # przewiń panel zamiast zmieniać wartość
                pasek = w.verticalScrollBar()
                pasek.setValue(pasek.value() - int(e.angleDelta().y() / 120 * pasek.singleStep() * 3))
            return True
        return False


_filtr = _KolkoTylkoZFokusem()


def ulatw(korzen):
    """Stosuje zasady z opisu modułu do wszystkich pól, list i suwaków pod widżetem `korzen`. Można wołać
    wielokrotnie (np. dla osobnych okien); każdy widżet dostaje filtr tylko raz."""
    for w in korzen.findChildren(QAbstractSpinBox):
        w.setKeyboardTracking(False)
        w.setAccelerated(True)
        _podepnij(w)
    for typ in (QComboBox, QAbstractSlider):
        for w in korzen.findChildren(typ):
            if isinstance(w, QScrollBar):
                continue
            _podepnij(w)


def _podepnij(w):
    if w.property('wygoda'):
        return
    w.setProperty('wygoda', True)
    w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    w.installEventFilter(_filtr)
