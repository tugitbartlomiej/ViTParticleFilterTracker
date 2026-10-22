"""Panel parametrów i podręczne ustawienia okienek aplikacji „Lasery i koloidy”.

- `PanelParametrow`: zwarta lista sekcji (po lewej) i jedna strona naraz (po prawej, przewijana) zamiast długiej kolumny
  zwijanych sekcji. Leży w okienku-doku, które można odpiąć jako osobne okno (np. na drugi monitor) albo schować (F2).
- `PopupUstawien`: prawy klik w okienko (widok 3D, wykres, pasek tytułu) otwiera przy kursorze ustawienia właśnie tego
  okienka. Popup nie ma własnych kontrolek: wypożycza sekcje z panelu (te same pola, ta sama logika i zapis ustawień);
  w panelu stoi w tym czasie zastępnik, a po zamknięciu sekcja wraca na swoje miejsce. 📌 przypina popup jako zwykłe
  okienko narzędziowe, które zostaje otwarte i można je przesunąć gdziekolwiek.
- `podlacz_menu(korzen, fn)`: prawy klik bez przeciągania w każdym widoku OpenGL (WidokGL.prawy_px), w wykresie pyqtgraph
  i w pozostałych miejscach okienka woła fn(pozycja_globalna). Prawe przeciąganie w widokach 3D dalej zbliża i oddala.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QGuiApplication
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QScrollArea, QSizePolicy, QStackedWidget,
                             QToolButton, QVBoxLayout, QWidget)

TLO_POPUPU = '#1b1e23'


def _przewijana(tresc):
    sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.Shape.NoFrame)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sc.setWidget(tresc)
    return sc


class PanelParametrow(QWidget):
    """Lista sekcji i strona wybranej sekcji. Strona to pionowy stos widżetów (zwykle sekcji `Zwijana`)."""

    zmiana_strony = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        self.lista = QListWidget()
        self.lista.setFixedWidth(118)
        self.lista.setStyleSheet('QListWidget { background: #17191d; border: none; font-size: 9pt; }'
                                 'QListWidget::item { padding: 5px 4px; border-bottom: 1px solid #23262b; }'
                                 'QListWidget::item:selected { background: #2459ad; color: white; }')
        self.lista.setWordWrap(True)
        self.strony = QStackedWidget()
        lay.addWidget(self.lista); lay.addWidget(self.strony, 1)
        self._klucze = []
        self.lista.currentRowChanged.connect(self._wybrano)

    def dodaj_strone(self, klucz, nazwa, widgety, dymek=''):
        """Strona `klucz` z widżetami jeden pod drugim (na dole odstęp); zwraca jej układ (do dopisywania)."""
        tresc = QWidget(); v = QVBoxLayout(tresc); v.setContentsMargins(2, 2, 4, 2); v.setSpacing(4)
        for w in widgety:
            v.addWidget(w)
        v.addStretch(1)
        self.strony.addWidget(_przewijana(tresc))
        it = QListWidgetItem(nazwa); it.setToolTip(dymek or nazwa); it.setSizeHint(QSize(110, 30))
        self.lista.addItem(it)
        self._klucze.append(klucz)
        if self.lista.currentRow() < 0:
            self.lista.setCurrentRow(0)
        return v

    def pokaz(self, klucz):
        if klucz in self._klucze:
            self.lista.setCurrentRow(self._klucze.index(klucz))

    def biezaca(self):
        i = self.lista.currentRow()
        return self._klucze[i] if 0 <= i < len(self._klucze) else None

    def _wybrano(self, i):
        if 0 <= i < len(self._klucze):
            self.strony.setCurrentIndex(i)
            self.zmiana_strony.emit(self._klucze[i])


class _Wypozyczenie:
    """Widżet przeniesiony z układu do popupu; w układzie stoi zastępnik o tej samej szerokości."""

    def __init__(self, w):
        from zakladka_laserow import Zwijana
        self.w = w
        self.zastepnik = QLabel('↗ Ustawienia otwarte w okienku podręcznym (prawy klik w okienko).')
        self.zastepnik.setWordWrap(True); self.zastepnik.setStyleSheet('color:#8a8d93; padding:6px; font-style:italic;')
        self.uklad = w.parentWidget().layout() if w.parentWidget() is not None else None
        self.zwinieta = isinstance(w, Zwijana) and w.zwinieta()
        if self.uklad is None or self.uklad.replaceWidget(w, self.zastepnik) is None:
            self.uklad = None                                # nie w układzie: tylko przeniesienie
        if self.zwinieta:
            w.naglowek.setChecked(True)

    def oddaj(self):
        if self.uklad is not None:
            self.uklad.replaceWidget(self.zastepnik, self.w)
            self.w.show()
        if self.zwinieta:
            self.w.naglowek.setChecked(False)
        self.zastepnik.deleteLater()


class PopupUstawien(QFrame):
    """Ustawienia jednego okienka przy kursorze. Zamyka się klikiem obok (jak menu), Esc albo ✕; 📌 przypina."""

    otwarte = []                                         # przypięte popupy (żeby nie zniknęły po wyjściu z funkcji)
    _u_kogo = {}                                         # id(sekcji) → popup, który ją teraz wypożycza

    def __init__(self, tytul, sekcje, akcje=(), na_panel=None):
        super().__init__(None, Qt.WindowType.Popup)
        self.setObjectName('popup_ustawien')
        self.setStyleSheet(f'QFrame#popup_ustawien {{ background: {TLO_POPUPU}; border: 1px solid #4c86e6; }}')
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._wyp = []
        self._przypiety = False
        v = QVBoxLayout(self); v.setContentsMargins(4, 3, 4, 4); v.setSpacing(3)
        gora = QHBoxLayout(); gora.setSpacing(2)
        lb = QLabel(f'<b>{tytul}</b>'); lb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setWindowTitle(f'Ustawienia: {tytul}')
        gora.addWidget(lb, 1)

        def przycisk(znak, opis, fn):
            b = QToolButton(); b.setText(znak); b.setToolTip(opis); b.setAutoRaise(True); b.setFixedSize(26, 22)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus); b.clicked.connect(fn); gora.addWidget(b)
            return b
        for znak, opis, fn in akcje:
            przycisk(znak, opis, lambda _=False, fn=fn: (fn(), self.close()))
        if na_panel is not None:
            przycisk('⇤', 'Pokaż te ustawienia w panelu parametrów (zamyka okienko podręczne).', lambda: (self.close(), na_panel()))
        self.bt_pin = przycisk('📌', 'Przypnij: okienko zostaje otwarte jako osobne okno (można je przesunąć), aż je zamkniesz.',
                               self.przypnij)
        przycisk('✕', 'Zamknij (Esc).', self.close)
        v.addLayout(gora)
        tresc = QWidget(); tv = QVBoxLayout(tresc); tv.setContentsMargins(0, 0, 0, 0); tv.setSpacing(4)
        self._tv = tv
        for s in sekcje:                                 # sekcja już w innym (przypiętym) popupie: tamten zamykamy, sekcja wraca
            inny = PopupUstawien._u_kogo.get(id(s))
            if inny is not None and inny is not self:
                inny.close()
        for s in sekcje:
            self._wyp.append(_Wypozyczenie(s))
            PopupUstawien._u_kogo[id(s)] = self
            tv.addWidget(s); s.show()
        if not sekcje:
            info = QLabel('To okienko nie ma własnych ustawień rysunku; oś czasu i wybór cząsteczki są w panelu „Dane” '
                          'i na pasku u góry.')
            info.setWordWrap(True); tv.addWidget(info)
        tv.addStretch(1)
        self.sc = _przewijana(tresc)
        v.addWidget(self.sc, 1)

    def przypnij(self):
        """Popup → zwykłe okienko narzędziowe z ramką (zostaje otwarte, można je przesunąć i zmienić mu rozmiar)."""
        if self._przypiety:
            return
        g = self.geometry()
        self._przypiety = True
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.bt_pin.setEnabled(False)
        PopupUstawien.otwarte.append(self)
        self.setGeometry(g)
        self.show()

    def pokaz_przy(self, pos: QPoint):
        """Pokazuje popup przy kursorze, w całości na ekranie, na którym jest kursor."""
        tresc = self.sc.widget()
        tresc.adjustSize()
        e = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
        r = e.availableGeometry()
        szer = min(max(tresc.sizeHint().width() + 28, 380), 560)
        wys = min(tresc.sizeHint().height() + 46, int(r.height() * 0.85))
        x = min(max(pos.x(), r.left()), r.right() - szer); y = min(max(pos.y(), r.top()), r.bottom() - wys)
        self.setGeometry(x, y, szer, wys)
        self.show()
        self.activateWindow()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        for w in reversed(self._wyp):
            self._tv.removeWidget(w.w)
            w.oddaj()
            if PopupUstawien._u_kogo.get(id(w.w)) is self:
                del PopupUstawien._u_kogo[id(w.w)]
        self._wyp = []
        if self in PopupUstawien.otwarte:
            PopupUstawien.otwarte.remove(self)
        super().closeEvent(e)


class _FiltrMenu(QObject):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def eventFilter(self, obj, e):
        if e.type() == QEvent.Type.ContextMenu:
            self.fn(e.globalPos())
            return True
        return False


def podlacz_menu(korzen, fn):
    """Prawy klik w okienku `korzen` (dok albo jego zawartość) → fn(QPoint globalny). Widoki OpenGL: tylko klik bez przeciągania
    (prawe przeciąganie zbliża), wykresy pyqtgraph: zamiast ich własnego menu."""
    import pyqtgraph as pg
    from widok_gl import WidokGL
    for v in [korzen] + korzen.findChildren(QWidget):
        if isinstance(v, WidokGL):
            v.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
            v.prawy_px.connect(lambda x, y, v=v: fn(v.mapToGlobal(QPoint(int(x), int(y)))))
        elif isinstance(v, pg.PlotWidget):
            v.setMenuEnabled(False)
            v.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
            v.scene().sigMouseClicked.connect(
                lambda ev: fn(QCursor.pos()) if ev.button() == Qt.MouseButton.RightButton and not ev.isAccepted() else None)
    filtr = _FiltrMenu(fn)
    korzen._filtr_menu = filtr                            # trzymamy referencję (filtr żyje razem z okienkiem)
    korzen.installEventFilter(filtr)
    tb = getattr(korzen, 'titleBarWidget', lambda: None)()
    if tb is not None:
        tb.installEventFilter(filtr)
