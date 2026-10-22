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


# Pełny ekran a nakładki (np. konwersja SBS → 3D w Odyssey 3D Hub): okno OpenGL bez ramki dokładnie na cały monitor sterownik
# Windows bierze za pełny ekran na wyłączność (jak gra) i nic nie da się położyć nad nim, także okna zawsze-na-wierzchu innych
# programów (YouTube w przeglądarce działa, bo przeglądarka składa obraz przez DWM). Dlatego domyślnie okienko nie przechodzi
# w stan pełnego ekranu, tylko staje się oknem bez ramki o 2 px większym od ekranu (wystaje krawędzią, przy której nie ma
# innego monitora; 1 px przy skalowaniu 150 % Qt zaokrągla do rozmiaru ekranu i sam przełącza okno w pełny ekran). Windows
# dalej chowa pasek zadań, ale okno nie pokrywa się z monitorem co do piksela, więc zostaje zwykłym oknem składanym przez DWM
# (SHQueryUserNotificationState: 2 = zwykły pełny ekran zamiast 3 = Direct3D na wyłączność). Ramka WS_BORDER, czyli Qt-owe
# setHasBorderInFullScreen, w oknie bez ramki nie zmienia obszaru klienta, więc tu nie pomaga.
# PELNY_DLA_NAKLADEK = False: klasyczny pełny ekran (showFullScreen).
PELNY_DLA_NAKLADEK = True

WYSTAJE = 2                                                       # o tyle pikseli logicznych okno wystaje poza ekran
_KRAWEDZIE = ((0, 0, 0, WYSTAJE), (0, -WYSTAJE, 0, 0), (0, 0, WYSTAJE, 0), (-WYSTAJE, 0, 0, 0))   # dół, góra, prawa, lewa


def krawedz_dla_nakladek(e):
    """Krawędź ekranu e, przy której okno „pełny ekran dla nakładek” wystaje o WYSTAJE px: pierwsza (dół, góra, prawa, lewa),
    za którą nie ma innego monitora; gdy wszystkie z kimś sąsiadują, dół. Najpierw dół i góra: obraz SBS nie przesuwa się
    w poziomie, więc połówki oczu zostają na swoich miejscach. Zwraca przesunięcia rogów (x0, y0, x1, y1)."""
    g = e.geometry(); n = WYSTAJE
    inne = [s.geometry() for s in QGuiApplication.screens() if s is not e]
    pasy = (QRect(g.left(), g.bottom() + 1, g.width(), n), QRect(g.left(), g.top() - n, g.width(), n),
            QRect(g.right() + 1, g.top(), n, g.height()), QRect(g.left() - n, g.top(), n, g.height()))
    for pas, k in zip(pasy, _KRAWEDZIE):
        if not any(pas.intersects(x) for x in inne):
            return k
    return _KRAWEDZIE[0]


def _pilnuj_nakladek(d):
    """Okno „pełny ekran dla nakładek”, które Qt przełączył w stan pełnego ekranu (rozmiar równy ekranowi): z powrotem większe."""
    if getattr(d, '_okna_pelny', None) is not None and PELNY_DLA_NAKLADEK and d.isFullScreen():
        d.setWindowState(Qt.WindowState.WindowNoState)
        d.setGeometry(d._okna_cel)


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
    """Okienko-dok na cały ekran, na którym leży (bez ramki i paska zadań); drugi raz wraca. Wymagane m.in. do stereo SBS
    na monitorze 3D. Esc też wychodzi z pełnego ekranu. Tryb: PELNY_DLA_NAKLADEK (opis wyżej)."""
    if w_pelnym_ekranie(d):
        m = getattr(d, '_okna_pelny', None)
        if d.isFullScreen():
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
    pasek = d.titleBarWidget()
    if pasek is not None:
        pasek.hide()
    if PELNY_DLA_NAKLADEK:
        d._okna_cel = e.geometry().adjusted(*krawedz_dla_nakladek(e))
        d.setGeometry(d._okna_cel)
        from PyQt6.QtCore import QTimer
        for ms in (100, 500):                                        # Qt potrafi sam przełączyć okno na cały ekran: wtedy wraca
            QTimer.singleShot(ms, lambda d=d: _pilnuj_nakladek(d))
    else:
        d.setGeometry(e.geometry())
        d.showFullScreen()
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
