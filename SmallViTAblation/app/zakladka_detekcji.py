"""Zakładka „Detekcja ViT i optymalizacja latentu”.

1. Wczytaj zdjęcie (albo weź obraz wygenerowany przez SD), 2. wybierz checkpoint ViT, Wykryj:
ViT zaznacza narzędzie, pokazuje pewność, IoU do etykiety, AP@0,5 i mAP@0,5:0,95.
3. Auto-szukanie: optymalizacja wartości latentu (J128 albo wszystkie 768 kanałów, w limitach
makrowektorów z edytora), żeby IoU i pewność dążyły do 100 %. Porównanie przed/po na zdjęciu i na latencie.
"""
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtCore import QEvent, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
                             QLabel, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QTabWidget,
                             QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.colors import to_rgba_array
from matplotlib.figure import Figure

import silnik
from styl import rola
from uklad import Przeplyw, dodaj
from wygoda import LeniwyModul

D = LeniwyModul('detekcja')            # timm i torchvision dopiero przy pierwszej detekcji

TESTOWE = Path(__file__).resolve().parent / 'zdjeciatestowe'

# ciemny motyw wykresów (te same barwy co w zakładce filmu)
TLO_FIG, TLO_OSI, TEKST, RAMKA, JASNY = '#1e1f22', '#26282c', '#d4d4d4', '#55585e', '#c8c8c8'


def _czarny(c):
    """Czy kolor (albo jednolita tablica kolorów) jest czarny: domyślny kolor tekstu i linii w matplotlib."""
    try:
        a = to_rgba_array(c)
    except (ValueError, TypeError):
        return False
    return len(a) == 1 and a[0, 3] > 0 and not a[0, :3].any()


def _motyw_osi(ax):
    """Ciemny motyw jednej osi: tło, ramki, podziałki, napisy, czarne linie / słupki / punkty na jasne, legenda."""
    ax.set_facecolor(TLO_OSI)
    for s in ax.spines.values():
        s.set_edgecolor(RAMKA)
    ax.tick_params(which='both', colors=TEKST)
    for t in [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + ax.texts:
        if _czarny(t.get_color()):
            t.set_color(TEKST)
    for l in ax.lines:
        if _czarny(l.get_color()):
            l.set_color(JASNY)
    for p in ax.patches:
        if _czarny(p.get_facecolor()):
            p.set_facecolor(JASNY)
    for c in ax.collections:
        if _czarny(c.get_facecolor()):
            c.set_facecolor(JASNY)
        if _czarny(c.get_edgecolor()):
            c.set_edgecolor(JASNY)
    leg = ax.get_legend()
    if leg is not None:
        leg.get_frame().set_facecolor(TLO_OSI); leg.get_frame().set_edgecolor(RAMKA)
        for t in leg.get_texts():
            if _czarny(t.get_color()):
                t.set_color(TEKST)


def ciemny_motyw(fig):
    """Przemalowuje figurę na ciemno tuż przed rysowaniem: tło, ramki osi, podziałki, napisy, legendy,
    a czarne linie, słupki i punkty na jasne. Działa po każdym fig.clear()/ax.clear(), bo wołane przy draw."""
    fig.set_facecolor(TLO_FIG)
    for t in fig.texts:
        if _czarny(t.get_color()):
            t.set_color(TEKST)
    for ax in fig.axes:
        _motyw_osi(ax)


class PlotnoCiemne(FigureCanvasQTAgg):
    """Płótno matplotlib w ciemnym motywie aplikacji (motyw nakładany przy każdym rysowaniu).

    Ruchome elementy (pionowy kursor kroku, ramki, tytuł, a nawet cała oś) można zarejestrować przez
    `ustaw_animowane`: pełne rysowanie robi się wtedy tylko przy zmianie reszty wykresu, a przy każdym kroku
    `odswiez_animowane` odtwarza zapamiętane tło i dorysowuje same ruchome elementy (blitting, kilka ms zamiast
    setek). Niewidoczne płótno nic nie rysuje; po pokazaniu odświeża się raz."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._animowane, self._tlo, self._zalegle = [], None, False
        self.mpl_connect('draw_event', self._po_rysowaniu)

    def draw(self):
        ciemny_motyw(self.figure)
        super().draw()

    def ustaw_animowane(self, artysci):
        """Rejestruje ruchome elementy (artyści albo osie); wymusza pełne rysowanie z nowym tłem."""
        nowe = [x for x in artysci if x is not None]
        for x in self._animowane:
            if not any(x is y for y in nowe):                       # wyrejestrowany: znów rysowany razem z resztą
                x.set_animated(False)
        self._animowane = nowe
        for x in nowe:
            x.set_animated(True)
        self._tlo = None

    def _rysuj_animowane(self):
        for x in self._animowane:
            if getattr(x, 'figure', None) is None:                 # usunięty przez fig.clear()
                continue
            if hasattr(x, 'spines'):                                # cała oś: motyw jak przy pełnym rysowaniu
                _motyw_osi(x)
            self.figure.draw_artist(x)

    def _po_rysowaniu(self, _zdarzenie):
        if self._animowane:
            self._tlo = self.copy_from_bbox(self.figure.bbox)       # tło bez ruchomych elementów
            self._rysuj_animowane()
        self._zalegle = False

    def odswiez_animowane(self):
        """Szybkie odświeżenie ruchomych elementów po zmianie ich danych (bez rysowania całego wykresu)."""
        if not self.isVisible():
            self._zalegle = True
            return
        if self._tlo is None:
            self.draw_idle()
            return
        self.restore_region(self._tlo)
        self._rysuj_animowane()
        self.blit(self.figure.bbox)

    def odswiez_pelne(self):
        """Pełne rysowanie, gdy płótno jest widoczne; niewidoczne odświeży się raz po pokazaniu."""
        if self.isVisible():
            self.draw_idle()
        else:
            self._zalegle = True

    def showEvent(self, e):
        super().showEvent(e)
        if self._zalegle:
            self.draw_idle()


class PanelDetekcji(QWidget):
    """Obraz 480x320 z ramką etykiety (cyjan) i ramką ViT (zielona); opcjonalnie rysowanie etykiety myszą."""

    etykietaNarysowana = pyqtSignal(list)

    def __init__(self, tytul):
        super().__init__()
        self.setMinimumSize(480, 320)
        self.tytul, self.rgb, self.gt, self.ramka, self.opis = tytul, None, None, None, ''
        self.rysowanie, self._start, self._koniec = False, None, None

    def ustaw(self, rgb480=None, gt=None, ramka=None, opis=''):
        if rgb480 is not None:
            self.rgb = np.ascontiguousarray(rgb480)
        self.gt, self.ramka, self.opis = gt, ramka, opis
        self.update()

    def _obszar(self):
        w, h = self.width(), self.height()
        if w * 2 > h * 3:
            s = h * 3 / 2
            return QRectF((w - s) / 2, 0, s, h)
        return QRectF(0, (h - w * 2 / 3) / 2, w, w * 2 / 3)

    def _na_ekran(self, b):
        o = self._obszar(); k = o.width() / 480
        return QRectF(o.left() + b[0] * k, o.top() + b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#111'))
        o = self._obszar()
        if self.rgb is not None:
            h, w = self.rgb.shape[:2]
            img = QImage(self.rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
            p.drawImage(o, img)
        else:
            p.setPen(QColor('#888'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '(wczytaj zdjęcie)')
        if self.gt is not None:
            p.setPen(QPen(QColor(0, 230, 255), 2)); p.drawRect(self._na_ekran(self.gt))
        if self.ramka is not None:
            p.setPen(QPen(QColor(60, 255, 60), 2)); p.drawRect(self._na_ekran(self.ramka))
        if self._start is not None and self._koniec is not None:
            p.setPen(QPen(QColor(0, 230, 255), 1, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(self._start, self._koniec).normalized())
        p.setPen(QColor(255, 255, 255))
        p.drawText(o.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                   f'{self.tytul}\n{self.opis}\ncyjan = etykieta, zielony = ViT')
        p.end()

    def mousePressEvent(self, e):
        if self.rysowanie and e.button() == Qt.MouseButton.LeftButton:
            self._start = self._koniec = e.position()

    def mouseMoveEvent(self, e):
        if self._start is not None:
            self._koniec = e.position(); self.update()

    def mouseReleaseEvent(self, e):
        if self._start is None:
            return
        o = self._obszar(); k = o.width() / 480
        r = QRectF(self._start, e.position()).normalized()
        self._start = self._koniec = None
        b = [(r.left() - o.left()) / k, (r.top() - o.top()) / k, (r.right() - o.left()) / k, (r.bottom() - o.top()) / k]
        b = [min(max(b[0], 0), 480), min(max(b[1], 0), 320), min(max(b[2], 0), 480), min(max(b[3], 0), 320)]
        if b[2] - b[0] > 3 and b[3] - b[1] > 3:
            self.etykietaNarysowana.emit(b)
        self.update()


class Watek(QThread):
    """Uruchamia funkcję w tle; postep(krok, slownik) dla optymalizacji."""
    gotowy = pyqtSignal(object)
    blad = pyqtSignal(str)
    postep = pyqtSignal(int, object)

    def __init__(self, funkcja, parent=None):
        super().__init__(parent)
        self.funkcja = funkcja

    def run(self):
        try:
            self.gotowy.emit(self.funkcja(lambda k, d: self.postep.emit(k, d)))
        except Exception:
            self.blad.emit(traceback.format_exc())


class PanelWykresu(QWidget):
    """Wykres matplotlib z paskiem narzędzi (przesuwanie, lupa, zapis), powiększeniem szerokości i wysokości
    w procentach z przewijaniem lewo/prawo/góra/dół, Ctrl + kółko myszy oraz przyciskiem „Osobne okno”."""

    def __init__(self, fig, wysokosc_bazowa=900, osobne_okno=None):
        super().__init__()
        self.fig, self.wys_baz = fig, wysokosc_bazowa
        self.canvas = PlotnoCiemne(fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        w = QHBoxLayout()
        self.pasek = NavigationToolbar2QT(self.canvas, self)
        self.pasek.setToolTip('Strzałki na krzyż = przesuwanie myszą, lupa = powiększenie zaznaczonego prostokąta,\n'
                              'domek = widok domyślny, dyskietka = zapis wykresu do PNG.')
        w.addWidget(self.pasek)
        w.addWidget(QLabel('Szerokość'))
        self.spin_x = QSpinBox(); self.spin_x.setRange(100, 2000); self.spin_x.setSingleStep(25); self.spin_x.setSuffix(' %')
        self.spin_x.setToolTip('Rozciągnięcie wykresu w poziomie; przewijaj suwakiem lewo-prawo pod wykresem.')
        w.addWidget(self.spin_x)
        w.addWidget(QLabel('Wysokość'))
        self.spin_y = QSpinBox(); self.spin_y.setRange(50, 1000); self.spin_y.setSingleStep(25); self.spin_y.setSuffix(' %')
        self.spin_y.setValue(100)
        self.spin_y.setToolTip(f'100% = {wysokosc_bazowa} px; przewijaj suwakiem góra-dół z prawej.\n'
                               'Ctrl + kółko myszy na wykresie powiększa oba kierunki.')
        w.addWidget(self.spin_y)
        b = rola(QPushButton('Dopasuj'), 'narzedzie'); b.setToolTip('Szerokość 100%, wysokość 100%.')
        b.clicked.connect(lambda: (self.spin_x.setValue(100), self.spin_y.setValue(100)))
        w.addWidget(b)
        if osobne_okno is not None:
            bo = rola(QPushButton('Osobne okno'), 'narzedzie')
            bo.setToolTip('Ten sam wykres w osobnym, zmaksymalizowanym oknie.')
            bo.clicked.connect(osobne_okno)
            w.addWidget(bo)
        w.addStretch(1)
        lay.addLayout(w)
        self.przewijanie = QScrollArea()
        self.przewijanie.setWidgetResizable(False)
        self.przewijanie.setWidget(self.canvas)
        lay.addWidget(self.przewijanie, 1)
        self.spin_x.valueChanged.connect(self._rozmiar)
        self.spin_y.valueChanged.connect(self._rozmiar)
        self.canvas.installEventFilter(self)
        self.przewijanie.viewport().installEventFilter(self)

    def eventFilter(self, obiekt, zdarzenie):
        if obiekt is self.przewijanie.viewport() and zdarzenie.type() == QEvent.Type.Resize:
            self._rozmiar()
            return False
        if zdarzenie.type() == QEvent.Type.Wheel and zdarzenie.modifiers() & Qt.KeyboardModifier.ControlModifier:
            krok = 25 if zdarzenie.angleDelta().y() > 0 else -25
            self.spin_x.setValue(self.spin_x.value() + krok)
            self.spin_y.setValue(self.spin_y.value() + krok)
            return True
        return False

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rozmiar()

    def showEvent(self, e):
        super().showEvent(e)
        self._rozmiar()

    def _rozmiar(self):
        vp = self.przewijanie.viewport().size()
        szer_widoku = self.przewijanie.width() - 2 * self.przewijanie.frameWidth()
        szer = max(200, int(szer_widoku * self.spin_x.value() / 100))
        wys = max(150, int(self.wys_baz * self.spin_y.value() / 100))
        if self.spin_x.value() == 100 and wys > vp.height():
            szer = max(200, szer - self.przewijanie.verticalScrollBar().sizeHint().width())
        self.canvas.setFixedSize(szer, wys)
        self.canvas.draw_idle()


def osobne_okno(rodzic, tytul, fabryka):
    """Okno z nową instancją widoku (fabryka() -> QWidget z narysowanymi danymi), zmaksymalizowane."""
    d = QDialog(rodzic)
    d.setWindowTitle(tytul)
    QVBoxLayout(d).addWidget(fabryka())
    d.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
    d.showMaximized()
    rodzic._okna = getattr(rodzic, '_okna', []) + [d]


class WidokLatentu(QWidget):
    """Mapy cieplne tokeny x kanały: przed, po, różnica, oraz średnie kanałów przed/po."""

    def __init__(self, tytul):
        super().__init__()
        self.tytul = tytul
        self.dane = None
        self.fig = Figure(figsize=(9, 5.2), tight_layout=True)
        self.panel = PanelWykresu(self.fig, 760, self._osobne)
        self.canvas = self.panel.canvas
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.panel)

    def _osobne(self):
        if self.dane is None:
            return
        def nowy():
            w = WidokLatentu(self.tytul); w.pokaz(*self.dane[:2], kanaly=self.dane[2])
            w.panel.wys_baz = 1100
            return w
        osobne_okno(self, f'Latent: {self.tytul}', nowy)

    def pokaz(self, przed, po=None, kanaly=None):
        self.dane = (przed, po, kanaly)
        self.fig.clear()
        etyk = np.asarray(kanaly) if kanaly is not None else np.arange(przed.shape[1])
        P = przed[:, etyk]
        Q = po[:, etyk] if po is not None else None
        n = 3 if Q is not None else 1
        gs = self.fig.add_gridspec(2, n, height_ratios=[3, 1.3])
        vmax = float(max(P.max(), Q.max() if Q is not None else P.max()))
        panele = [('przed', P, 'magma', -0.17, vmax)]
        if Q is not None:
            d = Q - P; m = float(np.abs(d).max()) or 1.0
            panele += [('po', Q, 'magma', -0.17, vmax), ('różnica po − przed', d, 'coolwarm', -m, m)]
        for k, (nazwa, A, cmap, lo, hi) in enumerate(panele):
            ax = self.fig.add_subplot(gs[0, k])
            im = ax.imshow(A, aspect='auto', cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
            ax.set_title(f'{self.tytul}: {nazwa}', fontsize=9)
            ax.set_xlabel('kanał (makrowektor)' + (' — kolejność J128' if kanaly is not None else ''), fontsize=8)
            ax.set_ylabel('token (0–599)', fontsize=8)
            ax.tick_params(labelsize=7)
            self.fig.colorbar(im, ax=ax, fraction=.046)
        ax = self.fig.add_subplot(gs[1, :])
        x = np.arange(len(etyk))
        ax.plot(x, P.mean(0), lw=.8, label='średnia przed')
        if Q is not None:
            ax.plot(x, Q.mean(0), lw=.8, label='średnia po')
        ax.set_xlim(0, len(etyk) - 1); ax.legend(fontsize=7); ax.tick_params(labelsize=7)
        ax.set_ylabel('średnia po tokenach', fontsize=8)
        if kanaly is not None and len(etyk) <= 128:
            ax.set_xticks(x[::8], [str(c) for c in etyk[::8]], fontsize=6, rotation=90)
        self.canvas.draw_idle()


class WidokMakrowektorow(QWidget):
    """Latent globalnie: jedna wartość na każdy z 768 kanałów (makrowektorów), zebrana po 600 tokenach.

    Słupki w górę / w dół od zera, oś symetryczna −M..+M. Panele: przed i po, różnica po − przed
    oraz same kanały J128 w kolejności siły sztucznej korekty.

    To statystyka do oglądania, nie wielkość obecna w modelu: ViT i adapter SD przetwarzają każdy token
    osobno. Przetasowanie tokenów daje identyczne makrowektory, a inną detekcję (po auto-szukaniu J128
    pewność 0,84 -> 0,35, IoU 1,00 -> 0,73; test 2026-10-06).
    """

    AGREGACJE = ['średnia po tokenach (ze znakiem)', 'suma po tokenach (ze znakiem)',
                 'RMS po tokenach (siła, bez znaku)', 'maksimum po tokenach']

    def __init__(self, kanaly_j128):
        super().__init__()
        self.j128 = np.asarray(kanaly_j128)
        self.przed = self.po = None
        lay = QVBoxLayout(self)
        w = QHBoxLayout()
        w.addWidget(QLabel('Wartość makrowektora:'))
        self.combo = QComboBox(); self.combo.addItems(self.AGREGACJE)
        self.combo.setToolTip('Jak z 600 wartości kanału (po jednej na token) zrobić jedną wartość globalną.\n'
                              'Średnia i suma zachowują znak (−0,17 to minimum GELU); RMS mierzy samą siłę.')
        self.combo.currentIndexChanged.connect(lambda _: self._rysuj())
        w.addWidget(self.combo)
        w.addWidget(QLabel('Skala osi:'))
        self.combo_skala = QComboBox()
        self.combo_skala.addItems(['symetryczna logarytmiczna (widać małe ujemne)', 'liniowa'])
        self.combo_skala.setToolTip('Po GELU wartość nie spada poniżej −0,17, więc ujemne makrowektory są małe\n'
                                    '(rzędu −0,05) przy dodatnich do kilku jednostek. Skala symetryczna logarytmiczna\n'
                                    'rozciąga okolice zera w obie strony, żeby piki w dół były widoczne.')
        self.combo_skala.currentIndexChanged.connect(lambda _: self._rysuj())
        w.addWidget(self.combo_skala)
        w.addStretch(1)
        lay.addLayout(w)
        self.fig = Figure(figsize=(9, 6), tight_layout=True)
        self.panel = PanelWykresu(self.fig, 1000, self._osobne)
        self.canvas = self.panel.canvas
        lay.addWidget(self.panel, 1)

    def _osobne(self):
        if self.przed is None:
            return
        def nowy():
            w = WidokMakrowektorow(self.j128)
            w.combo.setCurrentIndex(self.combo.currentIndex()); w.combo_skala.setCurrentIndex(self.combo_skala.currentIndex())
            w.panel.wys_baz = 1300
            w.pokaz(self.przed, self.po)
            return w
        osobne_okno(self, 'Makrowektory: 768 wartości globalnie', nowy)

    def _agreguj(self, L):
        k = self.combo.currentIndex()
        if k == 0:
            return L.mean(0)
        if k == 1:
            return L.sum(0)
        if k == 2:
            return np.sqrt((L ** 2).mean(0))
        return L.max(0)

    def pokaz(self, przed, po=None):
        self.przed, self.po = przed, po
        self._rysuj()

    def _os(self, ax, wartosci):
        m = float(np.abs(wartosci).max()) or 1.0
        ax.axhline(0, color='k', lw=.6)
        if self.combo_skala.currentIndex() == 0:
            prog = max(float(np.percentile(np.abs(wartosci[wartosci != 0]), 20)) if np.any(wartosci != 0) else 1e-3, 1e-4)
            ax.set_yscale('symlog', linthresh=prog, linscale=.6)
            podz = [t for t in (0.01, 0.1, 1, 10, 100) if prog * 0.99 <= t <= m * 1.15]
            ticks = sorted([-t for t in podz] + [0] + podz)
            ax.set_yticks(ticks, [f'{t:g}'.replace('.', ',') for t in ticks])
            ax.yaxis.set_minor_locator(__import__('matplotlib').ticker.NullLocator())
        ax.set_ylim(-m * 1.15, m * 1.15)
        ax.tick_params(labelsize=7)

    def _slupki(self, ax, x, v, szer=1.0, przesuniecie=0.0, alpha=.9):
        ax.bar(x + przesuniecie, np.where(v >= 0, v, 0), width=szer, color='tab:red', alpha=alpha)
        ax.bar(x + przesuniecie, np.where(v < 0, v, 0), width=szer, color='tab:blue', alpha=alpha)

    def _rysuj(self):
        if self.przed is None:
            return
        self.fig.clear()
        a = self._agreguj(self.przed)
        b = self._agreguj(self.po) if self.po is not None else None
        x = np.arange(len(a))
        nazwa = self.AGREGACJE[self.combo.currentIndex()]
        panele = [('PRZED', a)] + ([('PO', b), ('różnica po − przed', b - a)] if b is not None else [])
        gs = self.fig.add_gridspec(len(panele) + 1, 1, height_ratios=[1] * len(panele) + [1.1])
        for k, (tytul, v) in enumerate(panele):
            ax = self.fig.add_subplot(gs[k])
            self._slupki(ax, x, v)
            self._os(ax, v)
            ax.scatter(self.j128, np.zeros(len(self.j128)), marker='|', s=10, color='k', alpha=.35, zorder=3)
            ax.set_xlim(-1, len(a))
            n_dod, n_uj = int((v > 0).sum()), int((v < 0).sum())
            ax.set_title(f'{tytul}: 768 makrowektorów, {nazwa} — czerwony w górę: {n_dod}, niebieski w dół: {n_uj}'
                         + (' (podsumowanie; model działa na tokenach osobno)' if k == 0 else ''), fontsize=8)
        ax.set_xlabel('kanał (makrowektor) 0–767; szare kreski przy zerze = kanały J128', fontsize=8)

        ax = self.fig.add_subplot(gs[-1])
        xj = np.arange(len(self.j128))
        vj = [a[self.j128]] + ([b[self.j128]] if b is not None else [])
        szer = .8 / len(vj)
        for i, v in enumerate(vj):
            self._slupki(ax, xj, v, szer, (i - (len(vj) - 1) / 2) * szer, alpha=.95 if i == 0 else .5)
        self._os(ax, np.concatenate(vj))
        ax.set_xlim(-1, len(xj))
        ax.set_xticks(xj[::4], [str(c) for c in self.j128[::4]], fontsize=6, rotation=90)
        ax.set_title('128 wektorów tożsamości J128 (kolejność siły korekty)'
                     + ('; lewy słupek pary = przed, prawy jaśniejszy = po' if b is not None else ''), fontsize=8)
        self.canvas.draw_idle()


class ZakladkaDetekcji(QWidget):
    def __init__(self, okno):
        super().__init__()
        self.okno = okno
        self.det = None
        self.ckpt = silnik.VIT_CHECKPOINT
        self.rgb480 = None
        self.gt = None                  # xyxy w pikselach 480x320
        self.nazwa = None
        self.opis_gt = 'brak etykiety'
        self.analiza = None
        self.z_po = None
        self.watek = None
        self._przerwij = False
        self._buduj()

    # ---------------------------------------------------------------- UI
    def _buduj(self):
        lay = QVBoxLayout(self)
        gora = QHBoxLayout(); lay.addLayout(gora)

        # ---- grupa 1: zdjęcie, checkpoint, detekcja
        g = QGroupBox('Zdjęcie i detekcja ViT')
        v = QVBoxLayout(g); v.setSpacing(6)
        w1 = Przeplyw(); v.addLayout(w1)
        b1 = QPushButton('Wczytaj zdjęcie…')
        b1.setToolTip('Dowolne zdjęcie. Etykieta narzędzia jest wczytywana automatycznie z annotation.csv,\n'
                      'jeśli nazwa pliku wskazuje film i klatkę (np. TR_0001_S1_P03_klatka_000870.jpg).')
        b1.clicked.connect(self._wczytaj_zdjecie)
        bsd = QPushButton('Obraz z Edytora SD')
        bsd.setToolTip('Bierze ostatni obraz wygenerowany przez SD i sprawdza na nim detekcję ViT.\n'
                       'Etykieta: ta sama co dla zdjęcia, z którego policzono latent bazowy (jeśli była).')
        bsd.clicked.connect(self._obraz_z_sd)
        b2 = QPushButton('Checkpoint ViT…')
        b2.setToolTip('Słownik z kluczem net (dowolny krok uczenia małego ViT).')
        b2.clicked.connect(self._wybierz_checkpoint)
        for b in (b1, bsd, b2):
            w1.addWidget(b)
        import historie_checkpointow as HC
        self.combo_hist = QComboBox(); self.combo_hist.addItem('— historia uczenia —', '')
        for h in HC.HISTORIE:
            self.combo_hist.addItem(h.nazwa, h.klucz)
        self.combo_hist.setToolTip('Checkpoint z zapisanej historii uczenia małego ViT (K1000 co 10 kroków albo bieg '
                                   '35 000 klatek co epokę). Krok wybierasz w polu obok.')
        self.combo_hkrok = QComboBox(); self.combo_hkrok.setMinimumWidth(170)
        self.combo_hkrok.setToolTip('Krok albo epoka wybranej historii: detekcja i auto-szukanie użyją tego modelu.')
        self.combo_hist.currentIndexChanged.connect(lambda _: self._kroki_historii())
        self.combo_hkrok.activated.connect(lambda _: self._checkpoint_z_historii())
        w1.addWidget(self.combo_hist); w1.addWidget(self.combo_hkrok)
        w1b = Przeplyw(); v.addLayout(w1b)
        self.btn_wykryj = rola(QPushButton('Wykryj'), 'glowny')
        self.btn_wykryj.setToolTip('Uruchamia ViT na bieżącym zdjęciu: zaznacza wykryte narzędzie, podaje pewność, IoU do etykiety\n'
                                   'oraz AP@0,5 i mAP@0,5:0,95 (gdy zdjęcie ma etykietę).')
        self.btn_wykryj.clicked.connect(self._wykryj)
        bmap = QPushButton('mAP na zdjęciach testowych')
        bmap.setToolTip(f'AP@0,5 i mAP@0,5:0,95 ViT na wszystkich zdjęciach z etykietą w {TESTOWE.name}/.')
        bmap.clicked.connect(self._map_testowe)
        self.chk_rysuj = QCheckBox('Rysuj etykietę myszą')
        self.chk_rysuj.setToolTip('Przeciągnij prostokąt na obrazie PRZED, aby ustawić etykietę narzędzia ręcznie.')
        self.chk_rysuj.toggled.connect(lambda on: setattr(self.panel_przed, 'rysowanie', on))
        for b in (self.btn_wykryj, bmap, self.chk_rysuj):
            w1b.addWidget(b)
        self.etykieta_info = QLabel(f'Checkpoint: {Path(self.ckpt).name} (ładowany przy pierwszej detekcji). Etykieta: —')
        self.etykieta_info.setWordWrap(True)
        rola(self.etykieta_info, 'opis')
        v.addWidget(self.etykieta_info)
        v.addStretch(1)
        gora.addWidget(g, 1)

        # ---- grupa 2: auto-szukanie latentu (ustawienia, cele, akcje)
        g2 = QGroupBox('Auto-szukanie latentu')
        v2 = QVBoxLayout(g2); v2.setSpacing(6)

        def naglowek(tekst, opis=''):
            e = QLabel(tekst); e.setStyleSheet('font-weight:bold;'); e.setToolTip(opis)
            return e

        w2 = Przeplyw(); v2.addLayout(w2)          # wiersz 1: zakres i optymalizator
        w2.addWidget(naglowek('Optymalizacja:', 'Co wolno zmieniać i jak szybko (optymalizator Adam).'))
        self.combo_kanaly = QComboBox()
        self.combo_kanaly.addItems(['128 kanałów narzędzia (J128)', 'Wszystkie 768 kanałów'])
        self.combo_kanaly.setToolTip('Które kanały latentu wolno zmieniać (w każdym z 600 tokenów).\n'
                                     'J128: tylko 128 kanałów tożsamości narzędzia, pozostałe 640 bez zmian.\n'
                                     'Wszystkie 768: pełny latent; zwykle łatwiej osiągnąć cel, ale zmiana jest szersza.')
        w2.addWidget(self.combo_kanaly)
        self.combo_patche_opt = QComboBox()
        self.combo_patche_opt.addItems(['we wszystkich 600 patchach', 'tylko patche w ramce etykiety',
                                        'ramka etykiety + 1 patch obwódki', 'tylko patch środka etykiety',
                                        'tylko wybrany patch (klik w widoku cząsteczek)'])
        self.combo_patche_opt.setToolTip('W których patchach (tokenach) wolno zmieniać latent; w pozostałych zostaje naturalny:\n'
                                         '- wszystkie 600: jak dotąd;\n'
                                         '- ramka etykiety: patche, których środek leży w ramce narzędzia;\n'
                                         '- ramka + obwódka: to samo powiększone o 1 patch z każdej strony;\n'
                                         '- patch środka etykiety: jeden patch, w którym ViT ma wskazać środek narzędzia;\n'
                                         '- wybrany patch: patch kliknięty w widoku cząsteczek 2D albo 3D poniżej.\n'
                                         'Mniej patchy = zmiana bardziej lokalna; pokazuje, ile wystarczy zmienić, żeby ViT trafił.')
        w2.addWidget(self.combo_patche_opt)

        def pole(nazwa, opis, widget, szer=None):
            if szer:
                widget.setFixedWidth(szer)
            return dodaj(w2, nazwa, widget, opis)

        self.spin_kroki = pole('Kroki', 'Największa liczba kroków optymalizatora Adam. Szukanie kończy się wcześniej,\n'
                               'gdy wszystkie cele są spełnione. Pomiar celów co 25 kroków.', QSpinBox(), 80)
        self.spin_kroki.setRange(25, 50000); self.spin_kroki.setValue(1500)
        self.spin_lr = pole('lr', 'Współczynnik uczenia Adama: o ile mniej więcej zmienia się wartość kanału w jednym kroku.\n'
                            'Zalecane 0,05 (jak w eksperymencie J128; restarty 0,15 i 0,3). Powyżej ~0,5\n'
                            'wartości skaczą między limitami i wynik bywa niestabilny.', QDoubleSpinBox(), 85)
        self.spin_lr.setRange(1e-4, 5); self.spin_lr.setDecimals(4); self.spin_lr.setValue(0.05)
        self.combo_harm = pole('Harmonogram lr', 'Jak lr zmienia się w trakcie szukania:\n'
                               '- stały: przez cały czas lr,\n'
                               '- cosinusowy: łagodnie od lr do lr min (szybko na początku, dokładnie na końcu),\n'
                               '- liniowy: równomiernie od lr do lr min,\n'
                               '- krokowy: lr spada o połowę 4 razy w trakcie (nie niżej niż lr min),\n'
                               '- plateau: lr spada o połowę, gdy strata przez 100 kroków nie spadła o 1 %,\n'
                               '- restarty: 3 etapy po 1/3 kroków z lr x1, x3, x6, każdy od najlepszego\n'
                               '  dotąd wyniku i z nowym Adamem (jak w eksperymencie J128: 0,05; 0,15; 0,3).',
                               QComboBox())
        self.combo_harm.addItems(['stały', 'cosinusowy', 'liniowy', 'krokowy', 'plateau', 'restarty x1/x3/x6'])
        self.spin_lr_min = pole('lr min', 'Najmniejszy lr dla harmonogramów cosinusowego, liniowego, krokowego i plateau.',
                                QDoubleSpinBox(), 90)
        self.spin_lr_min.setRange(0, 5); self.spin_lr_min.setDecimals(5); self.spin_lr_min.setValue(0.005)
        self.spin_rozgrzewka = pole('rozgrzewka', 'Liczba pierwszych kroków, w których lr rośnie liniowo od 0 do lr\n'
                                    '(łagodny start; 0 = bez rozgrzewki). W restartach dotyczy każdego etapu.', QSpinBox(), 70)
        self.spin_rozgrzewka.setRange(0, 10000); self.spin_rozgrzewka.setValue(0)
        self.chk_limity = QCheckBox('w limitach makrowektorów')
        self.chk_limity.setChecked(True)
        self.chk_limity.setToolTip('Włączone: każdy kanał w [min_c, max_c] z grupy Limity w Edytorze.\n'
                                   'Wyłączone: tylko twarde minimum GELU −0,17, bez górnej granicy.')
        w2.addWidget(self.chk_limity)

        w2 = Przeplyw(); v2.addLayout(w2)          # wiersz 2: cele i wagi (pole() dokłada do bieżącego w2)
        w2.addWidget(naglowek('Cele, do których dąży ViT:',
                              'Szukanie kończy się, gdy jednocześnie spełnione są cele pewności, IoU i mAP\n'
                              '(albo po wyczerpaniu kroków / przerwaniu). Wartości w procentach.'))

        def procent(nazwa, opis, wartosc, minimum=0.0):
            w = QDoubleSpinBox(); w.setRange(minimum, 100.0); w.setDecimals(1); w.setSingleStep(0.5)
            w.setSuffix(' %'); w.setValue(wartosc)
            return pole(nazwa, opis, w, 85)

        self.spin_cel_p = procent('pewność', 'Pewność 0–100 %, z jaką ViT ma wskazać narzędzie w miejscu etykiety\n'
                                  '(ta sama pewność, którą ViT pokazuje przy ramce).\n'
                                  'Szukanie podnosi pewność do tego poziomu (z małym zapasem), nie wyżej.\n'
                                  'Wyższy cel = mocniejsze zmiany latentu.', 99.0)
        self.spin_cel_i = procent('IoU', 'IoU 0–100 %: jak bardzo ramka ViT pokrywa się z etykietą narzędzia\n'
                                  '(pole części wspólnej podzielone przez pole sumy obu ramek).\n'
                                  '100 % = ramka dokładnie na etykiecie. Szukanie zawsze dopasowuje ramkę\n'
                                  'jak najlepiej; ten cel mówi, kiedy wynik jest już wystarczający.', 99.0)
        self.spin_cel_map = procent('mAP', 'mAP 0–100 % (mAP@0,5:0,95) na tym jednym obrazie: ile z dziesięciu progów\n'
                                    'IoU (50 %, 55 %, ..., 95 %) przekracza ramka ViT. 100 % wymaga IoU >= 95 %,\n'
                                    '50 % to IoU od 70 do 75 %. 0 % = nie sprawdzaj.', 100.0)
        w2.addWidget(naglowek('  Wagi straty:', 'Jak bardzo szukanie dba o pewność względem dopasowania ramki.'))
        self.spin_waga_p = pole('pewności', 'Jak bardzo szukanie dba o pewność względem dopasowania ramki.\n'
                                '1 = tak jak w eksperymencie J128; 3 = trzy razy ważniejsza pewność; 0 = pomijana.',
                                QDoubleSpinBox(), 75)
        self.spin_waga_p.setRange(0.0, 100.0); self.spin_waga_p.setDecimals(2); self.spin_waga_p.setValue(1.0)
        self.spin_waga_r = pole('ramki', 'Jak bardzo szukanie dba o dopasowanie ramki (IoU) względem pewności.\n'
                                '1 = tak jak w eksperymencie J128; większa = ważniejsze IoU; 0 = pomijane.',
                                QDoubleSpinBox(), 75)
        self.spin_waga_r.setRange(0.0, 100.0); self.spin_waga_r.setDecimals(2); self.spin_waga_r.setValue(1.0)

        w4 = QHBoxLayout(); v2.addLayout(w4)        # wiersz 3: akcje i postęp razem
        self.btn_auto = rola(QPushButton('Auto-szukanie latentu'), 'glowny')
        self.btn_auto.setToolTip('Optymalizacja (Adam) wartości latentu tego zdjęcia, aby IoU i pewność ViT dążyły do 100%.\n'
                                 'Wagi ViT i piksele bez zmian; zmieniają się tylko wybrane kanały latentu,\n'
                                 'po każdym kroku przycinane do limitów makrowektorów (grupa Limity w Edytorze).')
        self.btn_auto.clicked.connect(self._auto)
        w4.addWidget(self.btn_auto)
        self.btn_stop = QPushButton('Przerwij'); self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip('Zatrzymuje szukanie po najbliższym pomiarze (co 25 kroków) i pokazuje bieżący wynik.')
        self.btn_stop.clicked.connect(lambda: setattr(self, '_przerwij', True))
        w4.addWidget(self.btn_stop)
        self.pasek = QProgressBar(); self.pasek.setMinimumWidth(120)
        w4.addWidget(self.pasek, 1)
        bprz = QPushButton('Latent PO → baza Edytora SD')
        bprz.setToolTip('Ustawia zoptymalizowany latent jako latent bazowy w Edytorze, aby wygenerować z niego obraz SD.')
        bprz.clicked.connect(self._do_edytora)
        w4.addWidget(bprz)
        gora.addWidget(g2, 2)

        w3 = QHBoxLayout()
        self.panel_przed = PanelDetekcji('PRZED (naturalny latent)')
        self.panel_przed.etykietaNarysowana.connect(self._etykieta_reczna)
        self.panel_po = PanelDetekcji('PO auto-szukaniu (ten sam obraz, zmieniony latent)')
        w3.addWidget(self.panel_przed, 1); w3.addWidget(self.panel_po, 1)
        lay.addLayout(w3, 3)
        self.etykieta_metryk = QLabel('')
        self.etykieta_metryk.setStyleSheet('font-family: Consolas, monospace;')
        lay.addWidget(self.etykieta_metryk)

        self.zakladki_latentu = QTabWidget()
        self.widok_768 = WidokLatentu('768 kanałów')
        self.widok_j128 = WidokLatentu('J128')
        self.zakladki_latentu.addTab(self.widok_768, 'Latent: wszystkie składowe 600 x 768')
        self.zakladki_latentu.addTab(self.widok_j128, 'Latent: wektory tożsamości J128 (600 x 128)')
        self.widok_makro = WidokMakrowektorow([w['kanal'] for w in self.okno.narzedzie])
        self.zakladki_latentu.addTab(self.widok_makro, 'Makrowektory: 768 wartości globalnie (−/+), podsumowanie')
        from czasteczki import WidokCzasteczek
        self.widok_cz = WidokCzasteczek([w['kanal'] for w in self.okno.narzedzie], PanelWykresu, osobne_okno)
        self.zakladki_latentu.addTab(self.widok_cz, 'Cząsteczki: okręgi 768 wektorów na patchach')
        try:
            from czasteczki3d import WidokCzasteczek3D
            self.widok_cz3d = WidokCzasteczek3D([w['kanal'] for w in self.okno.narzedzie])
            self.zakladki_latentu.addTab(self.widok_cz3d, 'Cząsteczki 3D: globalny układ (GPU, OpenGL)')
        except Exception as e:                       # brak OpenGL albo pyqtgraph: reszta aplikacji działa
            self.widok_cz3d = None
            self.zakladki_latentu.addTab(QLabel(f'Widok 3D niedostępny: {e}'), 'Cząsteczki 3D')
        lay.addWidget(self.zakladki_latentu, 4)

    # ------------------------------------------------------------ dane
    def _status(self, t, ms=8000):
        self.okno.statusBar().showMessage(t, ms)

    def _opis_info(self, etykieta):
        krok = f'krok {self.det.krok}' if self.det is not None else 'nieładowany'
        self.etykieta_info.setText(f'Checkpoint: {Path(self.ckpt).name} ({krok}). Zdjęcie: {self.nazwa or "—"}. {etykieta}')

    def _ustaw_obraz(self, rgb, nazwa, gt_natywna=None, opis_gt='brak etykiety'):
        H, W = np.asarray(rgb).shape[:2] if np.asarray(rgb).shape[-1] == 3 else np.asarray(rgb).shape[1:]
        self.rgb480 = D.do_480(rgb)
        self.nazwa = nazwa
        self.gt = None if gt_natywna is None else [gt_natywna[0] * 480 / W, gt_natywna[1] * 320 / H,
                                                   gt_natywna[2] * 480 / W, gt_natywna[3] * 320 / H]
        self.analiza = self.z_po = None
        self.opis_gt = opis_gt
        self.panel_przed.ustaw(self.rgb480, self.gt, None, '')
        self.panel_po.ustaw(self.rgb480, self.gt, None, '')
        self.etykieta_metryk.setText('')
        self._opis_info(opis_gt)

    def _wczytaj_zdjecie(self):
        s, _ = QFileDialog.getOpenFileName(self, 'Zdjęcie do detekcji', str(TESTOWE), 'Obrazy (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not s:
            return
        rgb = np.asarray(Image.open(s).convert('RGB'))
        gt, opis = D.etykieta_dla_zdjecia(s)
        self._ustaw_obraz(rgb, Path(s).name, gt, opis)
        self._wykryj()

    def _obraz_z_sd(self):
        if self.okno.obraz is None:
            QMessageBox.information(self, 'Brak obrazu', 'Najpierw wygeneruj obraz w zakładce Edytor SD.')
            return
        gt, opis = (None, 'brak etykiety (baza nie pochodzi ze zdjęcia z etykietą)')
        sb = self.okno.sciezka_bazy
        if sb:
            gt_n, opis_n = D.etykieta_dla_zdjecia(sb)
            if gt_n is not None:
                Hn, Wn = np.asarray(Image.open(sb)).shape[:2]
                gt = [gt_n[0] / Wn * 576, gt_n[1] / Hn * 384, gt_n[2] / Wn * 576, gt_n[3] / Hn * 384]
                opis = f'{opis_n} — etykieta zdjęcia bazy przeniesiona na obraz SD (przybliżenie)'
        self._ustaw_obraz(self.okno.obraz, 'obraz wygenerowany przez SD', gt, opis)
        self._wykryj()

    def _wybierz_checkpoint(self):
        s, _ = QFileDialog.getOpenFileName(self, 'Checkpoint ViT', str(Path(self.ckpt).parent), 'PyTorch (*.pt *.pth)')
        if not s:
            return
        self.ckpt, self.det = Path(s), None
        self._opis_info(self.opis_gt)
        if self.rgb480 is not None:
            self._wykryj()

    def _kroki_historii(self):
        import historie_checkpointow as HC
        self.combo_hkrok.clear()
        klucz = self.combo_hist.currentData()
        if not klucz:
            return
        h = HC.historia(klucz)
        kroki = h.kroki()
        if not kroki:
            self.combo_hkrok.addItem(f'brak wag w {h.wagi.name}', None); return
        for k in (kroki if not h.na_epoke else [k for k in kroki if k % h.na_epoke == 0]):
            self.combo_hkrok.addItem(h.opis_kroku(k), k)
        self.combo_hkrok.setCurrentIndex(self.combo_hkrok.count() - 1)
        self._checkpoint_z_historii()

    def _checkpoint_z_historii(self):
        import historie_checkpointow as HC
        krok, klucz = self.combo_hkrok.currentData(), self.combo_hist.currentData()
        if krok is None or not klucz:
            return
        self.ckpt, self.det = HC.historia(klucz).sciezka(krok), None
        self._opis_info(self.opis_gt)
        if self.rgb480 is not None:
            self._wykryj()

    def _etykieta_reczna(self, b):
        self.gt = b
        self.panel_przed.ustaw(None, self.gt, self.panel_przed.ramka, self.panel_przed.opis)
        self.opis_gt = 'etykieta narysowana ręcznie'
        self._opis_info(self.opis_gt)
        if self.analiza is not None:
            self._pokaz_przed()

    def _detektor(self):
        if self.det is None:
            self.det = D.Detektor(self.ckpt)
        return self.det

    # ------------------------------------------------------------ detekcja
    def _uruchom(self, funkcja, gotowy, postep=None):
        if self.watek is not None:
            return
        self.watek = Watek(funkcja, self)
        self.watek.gotowy.connect(lambda w: (setattr(self, 'watek', None), gotowy(w)))
        self.watek.blad.connect(self._blad)
        if postep:
            self.watek.postep.connect(postep)
        self.watek.start()

    def _blad(self, t):
        self.watek = None
        self.btn_auto.setEnabled(True); self.btn_stop.setEnabled(False)
        QMessageBox.warning(self, 'Błąd', t)

    def _wykryj(self):
        if self.rgb480 is None:
            QMessageBox.information(self, 'Brak zdjęcia', 'Wczytaj zdjęcie (przycisk 1).')
            return
        self._status('ViT liczy detekcję...')
        self._uruchom(lambda _p: self._detektor().analizuj(self.rgb480), self._po_detekcji)

    def _po_detekcji(self, a):
        self.analiza, self.z_po = a, None
        self._opis_info(self.opis_gt)
        self._pokaz_przed()
        self.panel_po.ustaw(self.rgb480, self.gt, None, '(uruchom auto-szukanie)')
        z = a['z'][0].cpu().numpy()
        self.widok_768.pokaz(z)
        self.widok_j128.pokaz(z, kanaly=[w['kanal'] for w in self.okno.narzedzie])
        self.widok_makro.pokaz(z)
        det, rgb = self.det, self.rgb480
        self.widok_cz.pokaz(z, None, rgb, lambda b: det.atencja(rgb, b))
        if self.widok_cz3d is not None:
            self.widok_cz3d.pokaz(z, None, rgb, lambda b: det.atencja(rgb, b))
        self._status('Detekcja gotowa.')

    @staticmethod
    def _tekst(m):
        f = lambda v: '—' if v is None else f'{100 * v:.2f} %'
        return (f'pewność {f(m["pewnosc"])} | IoU {f(m["iou"])} | AP@0,5 {f(m["ap50"])} | '
                f'mAP@0,5:0,95 {f(m["map5095"])}')

    def _pokaz_przed(self):
        a = self.analiza
        self.m_przed = D.metryki(a['boxes'], a['scores'], self.gt)
        self.panel_przed.ustaw(self.rgb480, self.gt, self.m_przed['ramka'], self._tekst(self.m_przed))
        self.etykieta_metryk.setText('PRZED: ' + self._tekst(self.m_przed))

    def _map_testowe(self):
        pliki = sorted(TESTOWE.glob('*.jpg'))

        def licz(_p):
            det, gts, nazwy = [], [], []
            for p in pliki:
                gt, _ = D.etykieta_dla_zdjecia(p)
                rgb = np.asarray(Image.open(p).convert('RGB')); H, W = rgb.shape[:2]
                a = self._detektor().analizuj(D.do_480(rgb))
                det.append((a['boxes'], a['scores']))
                gts.append(np.zeros((0, 4), np.float32) if gt is None else
                           np.array([[gt[0] * 480 / W, gt[1] * 320 / H, gt[2] * 480 / W, gt[3] * 320 / H]], np.float32))
                nazwy.append(p.name)
            return (D.ap(det, gts, .5), float(np.mean([D.ap(det, gts, t) for t in np.arange(.5, .96, .05)])),
                    sum(len(g) for g in gts), len(pliki))
        self._status('mAP na zdjęciach testowych...')
        self._uruchom(licz, lambda w: QMessageBox.information(
            self, 'mAP ViT', f'Checkpoint {Path(self.ckpt).name}\n{w[3]} zdjęć, {w[2]} z etykietą narzędzia\n'
                             f'AP@0,5 = {100 * w[0]:.2f} %\nmAP@0,5:0,95 = {100 * w[1]:.2f} %'))

    # ------------------------------------------------------------ auto-szukanie
    def _auto(self):
        if self.analiza is None:
            QMessageBox.information(self, 'Najpierw detekcja', 'Wczytaj zdjęcie i kliknij Wykryj.')
            return
        if self.gt is None:
            QMessageBox.information(self, 'Brak etykiety', 'Auto-szukanie potrzebuje etykiety narzędzia: '
                                    'zdjęcie z annotation.csv albo zaznacz „Rysuj etykietę myszą”.')
            return
        kanaly = [w['kanal'] for w in self.okno.narzedzie] if self.combo_kanaly.currentIndex() == 0 else list(range(768))
        if self.chk_limity.isChecked():
            lo, hi = self.okno.lim_lo, self.okno.lim_hi
        else:
            lo, hi = np.full(768, silnik.GELU_MIN, np.float32), np.full(768, 1e6, np.float32)
        kroki = self.spin_kroki.value()
        self.pasek.setMaximum(kroki); self.pasek.setValue(0)
        self._przerwij = False
        self.btn_auto.setEnabled(False); self.btn_stop.setEnabled(True)
        a, gt = self.analiza, list(self.gt)
        cele = dict(lr=self.spin_lr.value(), cel_pewnosc=self.spin_cel_p.value() / 100, cel_iou=self.spin_cel_i.value() / 100,
                    cel_map=self.spin_cel_map.value() / 100, waga_pewnosci=self.spin_waga_p.value(),
                    waga_ramki=self.spin_waga_r.value(),
                    harmonogram=('staly', 'cos', 'liniowy', 'krokowy', 'plateau', 'restarty')[self.combo_harm.currentIndex()],
                    lr_min=self.spin_lr_min.value(), rozgrzewka=self.spin_rozgrzewka.value())
        tokeny = self._tokeny_opt(gt)
        if tokeny is not None and len(tokeny) == 0:
            QMessageBox.information(self, 'Brak patchy', 'Wybierz patch w widoku cząsteczek (klik) albo inną opcję patchy.')
            self.btn_auto.setEnabled(True); self.btn_stop.setEnabled(False)
            return
        cele['tokeny'] = tokeny
        self._kanaly_opt, self._cele = kanaly, cele
        self._status('Auto-szukanie latentu...', 0)
        self._uruchom(lambda post: self._detektor().optymalizuj(a, gt, kanaly, lo, hi, kroki, postep=post,
                                                                przerwij=lambda: self._przerwij, **cele),
                      self._po_auto, self._postep_auto)

    def _tokeny_opt(self, gt):
        """Indeksy patchy do zmiany wg combo_patche_opt (None = wszystkie 600)."""
        i = self.combo_patche_opt.currentIndex()
        if i == 0:
            return None
        g = np.asarray(gt, np.float32)
        t = np.arange(600); sx, sy = (t % 30 + .5) * 16, (t // 30 + .5) * 16
        if i in (1, 2):
            m = 16 * (i == 2)
            return t[(sx >= g[0] - m) & (sx <= g[2] + m) & (sy >= g[1] - m) & (sy <= g[3] + m)]
        if i == 3:
            cx = int(np.clip(round((g[0] + g[2]) / 32 - .5), 0, 29)); cy = int(np.clip(round((g[1] + g[3]) / 32 - .5), 0, 19))
            return np.array([cy * 30 + cx])
        for w in (self.widok_cz3d, self.widok_cz):
            if w is not None and w.token is not None:
                return np.array([int(w.token)])
        return np.array([], int)

    def _postep_auto(self, krok, d):
        self.pasek.setValue(krok)
        self.etykieta_metryk.setText('PRZED: ' + self._tekst(self.m_przed) +
                                     f'\nszukanie: krok {krok}, lr {d["lr"]:.4g}, pewność {100 * d["pewnosc"]:.2f} % (cel {100 * self._cele["cel_pewnosc"]:.1f}), '
                                     f'IoU {100 * d["iou"]:.2f} % (cel {100 * self._cele["cel_iou"]:.1f}), '
                                     f'mAP@0,5:0,95 {100 * d["map5095"]:.0f} % (cel {100 * self._cele["cel_map"]:.0f})')

    def _po_auto(self, wynik):
        self.btn_auto.setEnabled(True); self.btn_stop.setEnabled(False)
        self.z_po, historia = wynik
        bx, sc = self.det.detekcja_z_latentu(self.analiza, self.z_po)
        m = D.metryki(bx, sc, self.gt)
        self.panel_po.ustaw(self.rgb480, self.gt, m['ramka'], self._tekst(m))
        przed = self.analiza['z'][0].cpu().numpy(); po = self.z_po[0].cpu().numpy()
        zm = int((np.abs(po - przed) > 1e-6).sum())
        self.etykieta_metryk.setText('PRZED: ' + self._tekst(self.m_przed) + '\nPO:    ' + self._tekst(m) +
                                     f'\n{historia[-1]["krok"] if historia else 0} kroków, zmienionych wartości latentu {zm} '
                                     f'z {po.size}; maks. po {po.max():.3f} (przed {przed.max():.3f})')
        self.widok_768.pokaz(przed, po)
        self.widok_j128.pokaz(przed, po, kanaly=[w['kanal'] for w in self.okno.narzedzie])
        self.widok_makro.pokaz(przed, po)
        det, rgb = self.det, self.rgb480
        self.widok_cz.pokaz(przed, po, rgb, lambda b: det.atencja(rgb, b))
        if self.widok_cz3d is not None:
            self.widok_cz3d.pokaz(przed, po, rgb, lambda b: det.atencja(rgb, b))
        self._zapisz(przed, po, m, historia)
        self.pasek.setValue(self.pasek.maximum())

    def _zapisz(self, przed, po, m, historia):
        import json
        from datetime import datetime
        silnik.WYJSCIE.mkdir(parents=True, exist_ok=True)
        zn = datetime.now().strftime('%Y%m%d_%H%M%S')
        sciezka = silnik.WYJSCIE / f'auto_szukanie_{zn}.npz'
        np.savez(sciezka, przed=przed, po=po, kanaly=np.asarray(self._kanaly_opt), h_min=self.okno.lim_lo,
                 h_max=self.okno.lim_hi, gt480=np.asarray(self.gt), zdjecie=str(self.nazwa), checkpoint=str(self.ckpt))
        sciezka.with_suffix('.json').write_text(json.dumps(dict(
            zdjecie=self.nazwa, checkpoint=str(self.ckpt), kanaly=len(self._kanaly_opt), cele=self._cele,
            limity=self.okno.zrodlo_limitow if self.chk_limity.isChecked() else 'tylko min GELU',
            przed={k: v for k, v in self.m_przed.items()}, po={k: v for k, v in m.items()}, historia=historia),
            indent=2, ensure_ascii=False, default=float), encoding='utf-8')
        self._status(f'Zapisano: {sciezka} (+ JSON)')

    def _do_edytora(self):
        if self.z_po is None:
            QMessageBox.information(self, 'Brak wyniku', 'Najpierw uruchom auto-szukanie.')
            return
        self.okno._ustaw_baze(self.z_po[0].cpu().numpy(), f'latent PO auto-szukaniu ({self.nazwa})')
        self._status('Latent PO ustawiony jako baza Edytora SD.')
