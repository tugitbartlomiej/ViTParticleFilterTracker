"""Maksymalizacja każdego okna na ekranie, na którym właśnie jest (także na drugim monitorze).

- `przelacz(w)`: przełącza maksymalizację okna albo okienka-doku, w którym leży widżet `w`. Okienko-dok (QDockWidget,
  np. okienka 1–11 zakładki „Lasery i koloidy”) wyskakuje, nawet przypięte, i wypełnia ekran, na którym leży jego
  środek; drugie przełączenie przypina je z powrotem albo oddaje poprzednie położenie.
- `DockTitleBar`: pasek tytułu okienka-doku z przyciskami ⛶ (maksymalizacja na tym ekranie), ❐ (odepnij / przypnij) i ✕.
  Podwójne kliknięcie w pasek też maksymalizuje (zamiast domyślnego przypinania); przeciąganie paska działa jak dawniej.
- `skrot_f11(okno)`: F11 w całej aplikacji (okno główne, okna odpięte, okienka-doki) przełącza maksymalizację tego okna,
  w którym jest kursor klawiatury.
"""
from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QCursor, QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QDockWidget, QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton,
                             QWidget)


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


# Pełny ekran a nakładki (konwersja SBS → 3D w Odyssey 3D Hub). Hub włącza się, gdy jakiś program przejdzie w pełny ekran
# (okno równe monitorowi), a potem jego okno musi leżeć nad tym programem. Okno OpenGL, którego obszar rysowania pokrywa się
# z monitorem co do piksela, sterownik bierze za pełny ekran na wyłączność (jak gra) i nic nie położy się nad nim (YouTube
# w przeglądarce działa, bo przeglądarka składa obraz przez DWM). Okno o 2 px większe od ekranu nie jest na wyłączność, ale
# Hub nie uznaje go za pełny ekran i się nie włącza. Dlatego okno jest dokładnie na cały monitor, a ma styl WS_BORDER: ramka
# 1 px zmniejsza obszar rysowania OpenGL o 1 px z każdej strony (3838 × 2158 na monitorze 3840 × 2160), więc sterownik
# zostawia je zwykłym oknem. To samo robi Qt-owe QWindowsWindowFunctions::setHasBorderInFullScreen (PyQt6 go nie udostępnia);
# w oknie z FramelessWindowHint Qt nie oddaje ramki z obszaru klienta, więc na czas pełnego ekranu okno traci tę flagę.
# PELNY_DLA_NAKLADEK = False: klasyczny pełny ekran bez ramki.
PELNY_DLA_NAKLADEK = True
_GWL_STYLE, _WS_BORDER = -16, 0x00800000


def _styl_okna(okno, dodaj=0):
    """Windows: styl okna najwyższego poziomu (GWL_STYLE), po dodaniu bitów `dodaj` (z przeliczeniem ramki); inny system: None."""
    import sys
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    pobierz, ustaw = u.GetWindowLongPtrW, u.SetWindowLongPtrW
    pobierz.restype = ustaw.restype = ctypes.c_ssize_t
    pobierz.argtypes = [wintypes.HWND, ctypes.c_int]; ustaw.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    hwnd = wintypes.HWND(int(okno.winId()))
    styl = pobierz(hwnd, _GWL_STYLE)
    if dodaj and (styl & dodaj) != dodaj:
        ustaw(hwnd, _GWL_STYLE, styl | dodaj)
        # SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED: przelicza ramkę, okno zostaje na miejscu
        u.SetWindowPos(hwnd, None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)
        styl = pobierz(hwnd, _GWL_STYLE)
    return styl


def _pilnuj_ramki(d):
    """Okienko w pełnym ekranie dla nakładek: ramka WS_BORDER dalej jest (Qt przy zmianie stanu okna potrafi ustawić styl od nowa)."""
    if getattr(d, '_okna_pelny', None) is not None and getattr(d, '_okna_ramka', False):
        _styl_okna(d, _WS_BORDER)


def w_pelnym_ekranie(d):
    """Czy okienko-dok jest w pełnym ekranie (w którymkolwiek trybie)."""
    return d.isFullScreen() or getattr(d, '_okna_pelny', None) is not None


def ustaw_pelny_dla_nakladek(wlacz):
    """Przełącza tryb pełnego ekranu (dla nakładek albo klasyczny); okienka, które już są w pełnym ekranie, wchodzą w nowy tryb."""
    global PELNY_DLA_NAKLADEK
    if PELNY_DLA_NAKLADEK == bool(wlacz):
        return
    PELNY_DLA_NAKLADEK = bool(wlacz)
    for w in QApplication.topLevelWidgets():
        if isinstance(w, QDockWidget) and w_pelnym_ekranie(w):
            _pelny_dok(w); _pelny_dok(w)


def _pelny_dok(d):
    """Okienko-dok na cały ekran, na którym leży (bez paska tytułu i paska zadań); drugi raz wraca. Wymagane m.in. do stereo
    SBS na monitorze 3D. Esc też wychodzi z pełnego ekranu. Tryb: PELNY_DLA_NAKLADEK (opis wyżej)."""
    if w_pelnym_ekranie(d):
        m = getattr(d, '_okna_pelny', None)
        d._okna_pelny = None
        d.showNormal()
        ramka, d._okna_ramka = getattr(d, '_okna_ramka', False), False
        pasek = d.titleBarWidget()
        if pasek is not None:
            pasek.show()
        if m is not None and not m[0]:
            d.setFloating(False)                                     # przypięcie samo ustawia flagi okna doku
        else:
            if ramka:                                                # z powrotem zwykłe okno bez ramki systemu (_niezalezne)
                d.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
            if m is not None:
                d.setGeometry(m[1])
        d.show(); d.raise_()                                         # przypięte w grupie kart: jego karta na wierzch
        return
    d._okna_pelny = (d.isFloating(), d.geometry())
    e = ekran(d)
    if not d.isFloating():
        d.setFloating(True)
    pasek = d.titleBarWidget()
    if pasek is not None:
        pasek.hide()
    d._okna_ramka = PELNY_DLA_NAKLADEK
    if PELNY_DLA_NAKLADEK:
        d.setWindowFlags(Qt.WindowType.Window)                       # bez FramelessWindowHint: ramka 1 px zabiera obszar klienta
    d.show()
    d.setGeometry(e.geometry())
    d.showFullScreen()
    if PELNY_DLA_NAKLADEK:
        _styl_okna(d, _WS_BORDER)
        from PyQt6.QtCore import QTimer
        for ms in (100, 500):
            QTimer.singleShot(ms, lambda d=d: _pilnuj_ramki(d))
    d.raise_(); d.activateWindow()
    if not hasattr(d, '_okna_esc'):
        d._okna_esc = QShortcut(QKeySequence(Qt.Key.Key_Escape), d)
        d._okna_esc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        d._okna_esc.activated.connect(lambda d=d: _pelny_dok(d) if w_pelnym_ekranie(d) else None)


def przelacz(w):
    """Okienko-dok, w którym leży `w`: prawdziwy pełny ekran (F11); gdy go nie ma, maksymalizacja okna najwyższego poziomu."""
    x = w
    while x is not None:
        if isinstance(x, QDockWidget):
            return _pelny_dok(x)
        x = x.parentWidget()
    przelacz_okno(w)


class _LayoutShield(QWidget):
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
    """Napis zmieniany często (co klatkę) w osłonie, która nie przelicza układu okna (_LayoutShield); do layoutu dodaje się wynik."""
    return _LayoutShield(lb, wys)


class DockTitleBar(QWidget):
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
    d.setTitleBarWidget(DockTitleBar(d))
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


class ScrollWrapper(QScrollArea):
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
