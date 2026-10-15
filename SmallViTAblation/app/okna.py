"""Maksymalizacja każdego okna na ekranie, na którym właśnie jest (także na drugim monitorze).

- `przelacz(w)`: przełącza maksymalizację okna albo okienka-doku, w którym leży widżet `w`. Okienko-dok (QDockWidget,
  np. okienka 1–11 zakładki „Lasery i koloidy”) wyskakuje, nawet przypięte, i wypełnia ekran, na którym leży jego
  środek; drugie przełączenie przypina je z powrotem albo oddaje poprzednie położenie.
- `PasekDoku`: pasek tytułu okienka-doku z przyciskami ⛶ (maksymalizacja na tym ekranie), ❐ (odepnij / przypnij) i ✕.
  Podwójne kliknięcie w pasek też maksymalizuje (zamiast domyślnego przypinania); przeciąganie paska działa jak dawniej.
- `skrot_f11(okno)`: F11 w całej aplikacji (okno główne, okna odpięte, okienka-doki) przełącza maksymalizację tego okna,
  w którym jest kursor klawiatury.
"""
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QCursor, QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QDockWidget, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget


def ekran(w):
    """Ekran, na którym leży środek widżetu (ekran kursora, gdy środek wypadł poza wszystkie ekrany)."""
    srodek = w.mapToGlobal(w.rect().center())
    return QGuiApplication.screenAt(srodek) or QGuiApplication.screenAt(QCursor.pos()) or w.screen()


def _blisko(a: QRect, b: QRect, tol=24):
    return all(abs(x - y) <= tol for x, y in ((a.x(), b.x()), (a.y(), b.y()), (a.width(), b.width()), (a.height(), b.height())))


def maksymalizuj_na(okno, e):
    """Maksymalizuje okno najwyższego poziomu na ekranie `e` (najpierw przenosi je na ten ekran)."""
    if okno.isMaximized():
        okno.showNormal()
    cel = e.availableGeometry()
    if QGuiApplication.screenAt(okno.frameGeometry().center()) is not e:
        okno.move(cel.x() + 40, cel.y() + 40)
    okno.showMaximized()
    okno.raise_(); okno.activateWindow()


def przelacz_okno(okno):
    okno = okno.window()
    if okno.isMaximized() or okno.isFullScreen():
        okno.showNormal()
    else:
        maksymalizuj_na(okno, ekran(okno))


def przelacz_dok(d):
    """Okienko-dok: wyskakuje i wypełnia swój ekran; drugi raz wraca (przypięte albo na poprzednie miejsce)."""
    m = getattr(d, '_okna_maks', None)
    if m is not None and d.isFloating() and _blisko(d.geometry(), m[2]):
        d._okna_maks = None
        if m[0]:
            d.setGeometry(m[1])
        else:
            d.setFloating(False)
        return
    cel = ekran(d).availableGeometry()
    d._okna_maks = (d.isFloating(), d.geometry(), cel)
    if not d.isFloating():
        d.setFloating(True)
    d.show()
    d.setGeometry(cel)
    d.raise_(); d.activateWindow()


def przelacz(w):
    """Maksymalizacja okienka-doku, w którym leży `w`, a gdy go nie ma, okna najwyższego poziomu."""
    x = w
    while x is not None:
        if isinstance(x, QDockWidget):
            return przelacz_dok(x)
        x = x.parentWidget()
    przelacz_okno(w)


class PasekDoku(QWidget):
    """Pasek tytułu okienka-doku: tytuł (nadąża za setWindowTitle) i przyciski ⛶, ❐, ✕."""

    def __init__(self, dok):
        super().__init__(dok)
        self.dok = dok
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 1, 2, 1); lay.setSpacing(1)
        self.tytul = QLabel(dok.windowTitle())
        self.tytul.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # długi tytuł nie poszerza okienka
        lay.addWidget(self.tytul, 1)
        for znak, opis, fn in (
                ('⛶', 'Maksymalizuj na tym ekranie, na którym jest okienko (wyskakuje z układu i wypełnia ekran).\n'
                      'Drugie kliknięcie przypina je z powrotem. To samo: podwójne kliknięcie w pasek albo F11.',
                 lambda: przelacz_dok(dok)),
                ('❐', 'Odepnij do osobnego okienka albo przypnij z powrotem.', lambda: dok.setFloating(not dok.isFloating())),
                ('✕', 'Zamknij okienko (wraca z menu okienek zakładki).', dok.close)):
            b = QToolButton(); b.setText(znak); b.setAutoRaise(True); b.setToolTip(opis); b.setFixedSize(22, 18)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(fn); lay.addWidget(b)
        dok.windowTitleChanged.connect(self.tytul.setText)

    def mouseDoubleClickEvent(self, e):
        przelacz_dok(self.dok)
        e.accept()


def pasek_doku(d):
    """Ustawia okienku-dokowi pasek tytułu z maksymalizacją; zwraca okienko."""
    d.setTitleBarWidget(PasekDoku(d))
    return d


def skrot_f11(okno):
    """F11 w całej aplikacji: maksymalizacja okna albo okienka-doku z kursorem klawiatury na jego ekranie."""
    def f11():
        w = QApplication.focusWidget() or QApplication.activeWindow() or okno
        przelacz(w)
    s = QShortcut(QKeySequence(Qt.Key.Key_F11), okno)
    s.setContext(Qt.ShortcutContext.ApplicationShortcut)
    s.activated.connect(f11)
    return s
