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
from matplotlib.figure import Figure

import detekcja as D
import silnik

TESTOWE = Path(__file__).resolve().parent / 'zdjeciatestowe'


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
        self.canvas = FigureCanvasQTAgg(fig)
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
        b = QPushButton('Dopasuj'); b.setToolTip('Szerokość 100%, wysokość 100%.')
        b.clicked.connect(lambda: (self.spin_x.setValue(100), self.spin_y.setValue(100)))
        w.addWidget(b)
        if osobne_okno is not None:
            bo = QPushButton('Osobne okno'); bo.setToolTip('Ten sam wykres w osobnym, zmaksymalizowanym oknie.')
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
        w.addWidget(self.combo, 1)
        w.addWidget(QLabel('Skala osi:'))
        self.combo_skala = QComboBox()
        self.combo_skala.addItems(['symetryczna logarytmiczna (widać małe ujemne)', 'liniowa'])
        self.combo_skala.setToolTip('Po GELU wartość nie spada poniżej −0,17, więc ujemne makrowektory są małe\n'
                                    '(rzędu −0,05) przy dodatnich do kilku jednostek. Skala symetryczna logarytmiczna\n'
                                    'rozciąga okolice zera w obie strony, żeby piki w dół były widoczne.')
        self.combo_skala.currentIndexChanged.connect(lambda _: self._rysuj())
        w.addWidget(self.combo_skala, 1)
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
        g = QGroupBox('Zdjęcie, checkpoint ViT i auto-szukanie latentu')
        v = QVBoxLayout(g)
        w1 = QHBoxLayout()
        b1 = QPushButton('1. Wczytaj zdjęcie...')
        b1.setToolTip('Dowolne zdjęcie. Etykieta narzędzia jest wczytywana automatycznie z annotation.csv,\n'
                      'jeśli nazwa pliku wskazuje film i klatkę (np. TR_0001_S1_P03_klatka_000870.jpg).')
        b1.clicked.connect(self._wczytaj_zdjecie)
        bsd = QPushButton('Obraz z SD (zakładka Edytor)')
        bsd.setToolTip('Bierze ostatni obraz wygenerowany przez SD i sprawdza na nim detekcję ViT.\n'
                       'Etykieta: ta sama co dla zdjęcia, z którego policzono latent bazowy (jeśli była).')
        bsd.clicked.connect(self._obraz_z_sd)
        b2 = QPushButton('2. Checkpoint ViT...')
        b2.setToolTip('Słownik z kluczem net (dowolny krok uczenia małego ViT).')
        b2.clicked.connect(self._wybierz_checkpoint)
        self.btn_wykryj = QPushButton('Wykryj')
        self.btn_wykryj.clicked.connect(self._wykryj)
        bmap = QPushButton('mAP na zdjęciach testowych')
        bmap.setToolTip(f'AP@0,5 i mAP@0,5:0,95 ViT na wszystkich zdjęciach z etykietą w {TESTOWE.name}/.')
        bmap.clicked.connect(self._map_testowe)
        for b in (b1, bsd, b2, self.btn_wykryj, bmap):
            w1.addWidget(b)
        self.chk_rysuj = QCheckBox('Rysuj etykietę myszą')
        self.chk_rysuj.setToolTip('Przeciągnij prostokąt na obrazie PRZED, aby ustawić etykietę narzędzia ręcznie.')
        self.chk_rysuj.toggled.connect(lambda on: setattr(self.panel_przed, 'rysowanie', on))
        w1.addWidget(self.chk_rysuj)
        v.addLayout(w1)

        w2 = QHBoxLayout()
        self.btn_auto = QPushButton('3. Auto-szukanie latentu')
        self.btn_auto.setStyleSheet('font-weight:bold; padding:4px;')
        self.btn_auto.setToolTip('Optymalizacja (Adam) wartości latentu tego zdjęcia, aby IoU i pewność ViT dążyły do 100%.\n'
                                 'Wagi ViT i piksele bez zmian; zmieniają się tylko wybrane kanały latentu,\n'
                                 'po każdym kroku przycinane do limitów makrowektorów (grupa Limity w Edytorze).')
        self.btn_auto.clicked.connect(self._auto)
        w2.addWidget(self.btn_auto)
        self.combo_kanaly = QComboBox()
        self.combo_kanaly.addItems(['128 kanałów narzędzia (J128)', 'Wszystkie 768 kanałów'])
        self.combo_kanaly.setToolTip('Które kanały latentu wolno zmieniać (w każdym z 600 tokenów).\n'
                                     'J128: tylko 128 kanałów tożsamości narzędzia, pozostałe 640 bez zmian.\n'
                                     'Wszystkie 768: pełny latent; zwykle łatwiej osiągnąć cel, ale zmiana jest szersza.')
        w2.addWidget(self.combo_kanaly)

        def pole(nazwa, opis, widget):
            e = QLabel(nazwa); e.setToolTip(opis); widget.setToolTip(opis)
            w2.addWidget(e); w2.addWidget(widget)
            return widget

        self.spin_kroki = pole('Kroki', 'Największa liczba kroków optymalizatora Adam. Szukanie kończy się wcześniej,\n'
                               'gdy wszystkie cele są spełnione. Pomiar celów co 25 kroków.', QSpinBox())
        self.spin_kroki.setRange(25, 50000); self.spin_kroki.setValue(1500)
        self.spin_lr = pole('lr', 'Współczynnik uczenia Adama: o ile mniej więcej zmienia się wartość kanału w jednym kroku.\n'
                            'Zalecane 0,05 (jak w eksperymencie J128; restarty 0,15 i 0,3). Powyżej ~0,5\n'
                            'wartości skaczą między limitami i wynik bywa niestabilny.', QDoubleSpinBox())
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
                                QDoubleSpinBox())
        self.spin_lr_min.setRange(0, 5); self.spin_lr_min.setDecimals(5); self.spin_lr_min.setValue(0.005)
        self.spin_rozgrzewka = pole('rozgrzewka', 'Liczba pierwszych kroków, w których lr rośnie liniowo od 0 do lr\n'
                                    '(łagodny start; 0 = bez rozgrzewki). W restartach dotyczy każdego etapu.', QSpinBox())
        self.spin_rozgrzewka.setRange(0, 10000); self.spin_rozgrzewka.setValue(0)
        w2.addStretch(1)
        v.addLayout(w2)
        w2 = QHBoxLayout()          # drugi wiersz: cele i wagi (pole() dokłada do bieżącego w2)
        e_cele = QLabel('Cele, do których dąży ViT:')
        e_cele.setStyleSheet('font-weight:bold;')
        e_cele.setToolTip('Szukanie kończy się, gdy jednocześnie spełnione są cele pewności, IoU i mAP\n'
                          '(albo po wyczerpaniu kroków / przerwaniu). Wartości w procentach.')
        w2.addWidget(e_cele)

        def procent(nazwa, opis, wartosc, minimum=0.0):
            w = QDoubleSpinBox(); w.setRange(minimum, 100.0); w.setDecimals(1); w.setSingleStep(0.5)
            w.setSuffix(' %'); w.setValue(wartosc)
            return pole(nazwa, opis, w)

        self.spin_cel_p = procent('Cel pewności', 'Pewność 0–100 %, z jaką ViT ma wskazać narzędzie w miejscu etykiety\n'
                                  '(ta sama pewność, którą ViT pokazuje przy ramce).\n'
                                  'Szukanie podnosi pewność do tego poziomu (z małym zapasem), nie wyżej.\n'
                                  'Wyższy cel = mocniejsze zmiany latentu.', 99.0)
        self.spin_cel_i = procent('Cel IoU', 'IoU 0–100 %: jak bardzo ramka ViT pokrywa się z etykietą narzędzia\n'
                                  '(pole części wspólnej podzielone przez pole sumy obu ramek).\n'
                                  '100 % = ramka dokładnie na etykiecie. Szukanie zawsze dopasowuje ramkę\n'
                                  'jak najlepiej; ten cel mówi, kiedy wynik jest już wystarczający.', 99.0)
        self.spin_cel_map = procent('Cel mAP', 'mAP 0–100 % (mAP@0,5:0,95) na tym jednym obrazie: ile z dziesięciu progów\n'
                                    'IoU (50 %, 55 %, ..., 95 %) przekracza ramka ViT. 100 % wymaga IoU >= 95 %,\n'
                                    '50 % to IoU od 70 do 75 %. 0 % = nie sprawdzaj.', 100.0)
        self.spin_waga_p = pole('Waga pewności', 'Jak bardzo szukanie dba o pewność względem dopasowania ramki.\n'
                                '1 = tak jak w eksperymencie J128; 3 = trzy razy ważniejsza pewność; 0 = pomijana.', QDoubleSpinBox())
        self.spin_waga_p.setRange(0.0, 100.0); self.spin_waga_p.setDecimals(2); self.spin_waga_p.setValue(1.0)
        self.spin_waga_r = pole('Waga ramki', 'Jak bardzo szukanie dba o dopasowanie ramki (IoU) względem pewności.\n'
                                '1 = tak jak w eksperymencie J128; większa = ważniejsze IoU; 0 = pomijane.', QDoubleSpinBox())
        self.spin_waga_r.setRange(0.0, 100.0); self.spin_waga_r.setDecimals(2); self.spin_waga_r.setValue(1.0)
        self.chk_limity = QCheckBox('w limitach makrowektorów')
        self.chk_limity.setChecked(True)
        self.chk_limity.setToolTip('Włączone: każdy kanał w [min_c, max_c] z grupy Limity w Edytorze.\n'
                                   'Wyłączone: tylko twarde minimum GELU −0,17, bez górnej granicy.')
        w2.addWidget(self.chk_limity)
        self.btn_stop = QPushButton('Przerwij'); self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip('Zatrzymuje szukanie po najbliższym pomiarze (co 25 kroków) i pokazuje bieżący wynik.')
        self.btn_stop.clicked.connect(lambda: setattr(self, '_przerwij', True))
        w2.addWidget(self.btn_stop)
        bprz = QPushButton('Latent PO → baza Edytora SD')
        bprz.setToolTip('Ustawia zoptymalizowany latent jako latent bazowy w Edytorze, aby wygenerować z niego obraz SD.')
        bprz.clicked.connect(self._do_edytora)
        w2.addWidget(bprz)
        v.addLayout(w2)
        self.pasek = QProgressBar(); v.addWidget(self.pasek)
        self.etykieta_info = QLabel(f'Checkpoint: {Path(self.ckpt).name} (ładowany przy pierwszej detekcji). Etykieta: —')
        self.etykieta_info.setWordWrap(True)
        v.addWidget(self.etykieta_info)
        lay.addWidget(g)

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
        self._kanaly_opt, self._cele = kanaly, cele
        self._status('Auto-szukanie latentu...', 0)
        self._uruchom(lambda post: self._detektor().optymalizuj(a, gt, kanaly, lo, hi, kroki, postep=post,
                                                                przerwij=lambda: self._przerwij, **cele),
                      self._po_auto, self._postep_auto)

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
