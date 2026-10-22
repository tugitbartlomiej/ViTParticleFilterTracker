"""Maksymalizacja każdego okna na ekranie, na którym właśnie jest (także na drugim monitorze).

- `przelacz(w)`: przełącza maksymalizację okna albo okienka-doku, w którym leży widżet `w`. Okienko-dok (QDockWidget,
  np. okienka 1–11 zakładki „Lasery i koloidy”) wyskakuje, nawet przypięte, i wypełnia ekran, na którym leży jego
  środek; drugie przełączenie przypina je z powrotem albo oddaje poprzednie położenie.
- `PasekDoku`: pasek tytułu okienka-doku z przyciskami ⛶ (maksymalizacja na tym ekranie), ❐ (odepnij / przypnij) i ✕.
  Podwójne kliknięcie w pasek też maksymalizuje (zamiast domyślnego przypinania); przeciąganie paska działa jak dawniej.
- `skrot_f11(okno)`: F11 w całej aplikacji (okno główne, okna odpięte, okienka-doki) przełącza maksymalizację tego okna,
  w którym jest kursor klawiatury.
"""
from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QCursor, QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QDockWidget, QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton, QWidget


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


def _pelny_dok(d):
    """Okienko-dok w prawdziwym trybie pełnoekranowym (bez ramki i paska zadań) na ekranie, na którym leży; drugi raz wraca.
    Wymagane m.in. do stereo SBS na monitorze 3D. Esc też wychodzi z pełnego ekranu."""
    if d.isFullScreen():
        m = getattr(d, '_okna_pelny', None)
        d.showNormal()
        pasek = d.titleBarWidget()
        if pasek is not None:
            pasek.show()
        if m is not None:
            if m[0]:
                d.setGeometry(m[1])
            else:
                d.setFloating(False)
        d._okna_pelny = None
        return
    d._okna_pelny = (d.isFloating(), d.geometry())
    e = ekran(d)
    if not d.isFloating():
        d.setFloating(True)
    d.show()
    d.setGeometry(e.geometry())
    pasek = d.titleBarWidget()
    if pasek is not None:
        pasek.hide()
    d.showFullScreen()
    d.raise_(); d.activateWindow()
    if not hasattr(d, '_okna_esc'):
        d._okna_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), d)
        d._okna_esc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        d._okna_esc.activated.connect(lambda d=d: _pelny_dok(d) if d.isFullScreen() else None)


def przelacz(w):
    """Okienko-dok, w którym leży `w`: prawdziwy pełny ekran (F11); gdy go nie ma, maksymalizacja okna najwyższego poziomu."""
    x = w
    while x is not None:
        if isinstance(x, QDockWidget):
            return _pelny_dok(x)
        x = x.parentWidget()
    przelacz_okno(w)


class _Oslona(QWidget):
    """Pojemnik bez układu z jednym napisem w środku. QLabel.setText woła updateGeometry, a to przelicza układ rodzica aż do
    QMainWindow; przy skalowaniu ekranu 150 % każde takie przeliczenie gubiło piksel na zaokrągleniach doków (doki pełzły o 1–4 px
    na klatkę), a każdy widok OpenGL dostawał resize i nowy bufor ramki w każdej klatce odtwarzania. Pojemnik bez układu
    przyjmuje to zgłoszenie i nic dalej nie przekazuje; jego rozmiar ustala układ, napis wypełnia go w resizeEvent."""

    def __init__(self, lb, wys=None):
        super().__init__()
        self.lb = lb
        lb.setParent(self)
        self.setFixedHeight(int(wys if wys is not None else max(lb.height(), lb.minimumHeight(), lb.sizeHint().height())))
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(40, self.height())

    def minimumSizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(10, self.height())

    def resizeEvent(self, ev):
        self.lb.setGeometry(self.rect())
        super().resizeEvent(ev)


def bez_ukladu(lb, wys=None):
    """Napis zmieniany często (co klatkę) w osłonie, która nie przelicza układu okna (_Oslona); do layoutu dodaje się wynik."""
    return _Oslona(lb, wys)


class PasekDoku(QWidget):
    """Pasek tytułu okienka-doku: tytuł (nadąża za setWindowTitle) i przyciski ⛶, ❐, ✕."""

    def __init__(self, dok):
        super().__init__(dok)
        self.dok = dok
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 1, 2, 1); lay.setSpacing(1)
        self.tytul = QLabel(dok.windowTitle())
        self.tytul.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # długi tytuł nie poszerza okienka
        lay.addWidget(bez_ukladu(self.tytul, self.tytul.sizeHint().height() + 2), 1)   # tytuł zmieniany w czasie (11): bez przeliczeń
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
    """Ustawia okienku-dokowi pasek tytułu z maksymalizacją; zwraca okienko. Odpięte okienko jest zwykłym oknem (_niezalezne)."""
    d.setTitleBarWidget(PasekDoku(d))
    d.topLevelChanged.connect(lambda plywa, d=d: _niezalezne(d, plywa))
    return d


def _niezalezne(d, plywa):
    """Odpięte okienko-dok: Qt robi z niego okno narzędziowe (Qt.Tool), które zawsze leży NAD oknem głównym i nad innymi odpiętymi
    (panel parametrów i podkroki zasłaniały okno i siebie nawzajem). Zwykłe okno bez ramki systemu (pasek tytułu jest nasz): można
    je schować pod inne, przełączać Alt+Tab, ma własną pozycję na pasku zadań. Ponowne przypięcie: ❐ albo przeciągnięcie paska."""
    if not plywa or not d.isFloating() or d.windowType() != Qt.WindowType.Tool:
        return
    if QApplication.mouseButtons() != Qt.MouseButton.NoButton:          # w trakcie przeciągania: zmiana okna przerwałaby ruch
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(150, lambda d=d: _niezalezne(d, d.isFloating()))
        return
    g = d.geometry()
    d.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
    d.setGeometry(g)
    d.show(); d.raise_(); d.activateWindow()


class Przewijane(QScrollArea):
    """Zawartość okienka sterującego (podkroki, panel) w przewijanym polu: jej minimalny rozmiar (np. 1146 px szerokości paska podkroków
    z drugim wierszem opcji) nie rozpycha już okna głównego, więc przypięte okienka nie wchodzą na siebie ani na panel parametrów.
    Gdy brakuje miejsca, pojawia się pasek przewijania; zwykły rozmiar (sizeHint) to rozmiar zawartości."""
    def __init__(self, w, min_szer=200, min_wys=40):
        super().__init__()
        self.setWidget(w); self.setWidgetResizable(True); self.setFrameShape(QFrame.Shape.NoFrame)
        self._min = QSize(min_szer, min_wys)

    def sizeHint(self):
        w = self.widget()
        s = w.sizeHint().expandedTo(w.minimumSizeHint()) if w is not None else QSize()
        return s + QSize(2 * self.frameWidth(), 2 * self.frameWidth())

    def minimumSizeHint(self):
        w = self.widget()
        wys = min(self._min.height() if w is None else w.minimumSizeHint().height(), 120)   # wysokość: jak treść, najwyżej 120 px
        return QSize(self._min.width(), max(self._min.height(), wys))


def skrot_f11(okno):
    """F11 w całej aplikacji: okienko-dok pod kursorem myszy przechodzi do prawdziwego pełnego ekranu (drugi raz albo Esc wraca), a inne okno jest maksymalizowane."""
    def f11():
        w = QApplication.widgetAt(QCursor.pos()) or QApplication.focusWidget() or QApplication.activeWindow() or okno   # okienko pod kursorem
        przelacz(w)
    s = QShortcut(QKeySequence(Qt.Key.Key_F11), okno)
    s.setContext(Qt.ShortcutContext.ApplicationShortcut)
    s.activated.connect(f11)
    return s
