"""Okienka R1–R8 zakładki „Lasery i koloidy”: rzuty z 192 wymiarów do 3D z miernikiem wierności (obliczenia: projections.py,
uzasadnienie: reports/Rzutowanie wielowymiarowego ruchu do 3D.md).

Po lewej zwijana sekcja „Rzuty z 192 wymiarów”: liczenie stanów 192D (raz na historię i zdjęcie, potem z dysku), przyciski
okienek i opcje. Jedno okienko = jeden rzut:
  R1 A — mapa klas: oś 1 = Δ narzędzie − tło (100 % separacji), osie 2–3 = PCA reszty chmury (wspólna albo w każdym checkpoincie);
  R2 S — siły: 3 kierunki największej energii sił netto bloku 6;
  R3 W — wiązki ważone aktywacją (następca P3) i wiązki wybranej cząsteczki (sieć silna);
  R4 B — rama siły wybranej cząsteczki: F̂ i płaszczyzna, w której jej wiązki się znoszą;
  R5 C — rama Schura: podprzestrzeń niezmiennicza pełnego jakobianu, pole lokalnej deformacji, płaszczyzna wiru i kierunek własny;
  R6 D — wędrówka w czasie uczenia: wszystkie, narzędzie, wybrane albo jedna cząsteczka, z torami zerowymi (błądzenie losowe);
  R7 F — 7 widoków płaszczyzny Fano nad 7 kierunkami (każda para kierunków razem w dokładnie jednym widoku);
  R8 — bez rzutu: wierność widoków w czasie, separacja klas wobec permutacji etykiet, kosinusy w 192D, pary wiązek, widmo J;
  R9 L — separacja narzędzia od tła: oś 1 = LDA ze skurczem (Δ wybielone kowariancją wewnątrz klas, dopasowane na puli ostatnich
       checkpointów), osie 2–3 do wyboru; w podpisie sylwetka, trafienia kNN, margines i stosunek Fishera z zerem permutacyjnym.
W okienkach globalnych (R1–R3, R6, R7) ślady to wędrówka cząsteczek przez checkpointy uczenia, a strzałki to siła netto bloku 6
(dokładny rzut; krycie = widoczna część długości ρ = |Pu|/|u|). Wszystkie pokazują chwilę osi czasu zakładki.
Rysowanie na karcie (projections_gl.ProjectionGPU, GLSL 4.30): cała historia w bazie okienka idzie do bufora raz (przy zmianie bazy, środka,
skali, wygładzania), w klatce tylko czas; ramy lokalne (wiązki, pole Schura) idą raz na checkpoint i cząsteczkę. Rzutowanie
192D → 3D całej historii liczy karta (projections.ProjectionData.wsp, torch).
"""
from __future__ import annotations

import traceback

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy,
                             QVBoxLayout, QWidget)

from laser_app.compute import history as hist
from laser_app.compute import projections
from laser_app.ui.layout_widgets import NarrowCombo
from laser_app.ui.gl_view import GLView

TLO = '#03060a'
S_EKR = 10.0                                         # promień RMS chmury na ekranie (jednostki sceny)
KOL_NARZ, KOL_TLO = (1.0, 0.36, 0.30), (0.56, 0.64, 0.72)
KOL_WYBR, KOL_DOD = (1.0, 1.0, 1.0), (1.0, 0.82, 0.40)
KOL_SILA, KOL_ZGODNA, KOL_PRZECIW = (1.0, 0.62, 0.20), (1.0, 0.78, 0.35), (0.35, 0.68, 1.0)
KOL_BAZ = {'P3': '#e8e8e8', 'A': '#ffb000', 'A_t': '#ff7a45', 'S': '#4af0ff', 'W': '#c264ff', 'D': '#7ee081'}
OKIENKA = (('R0', 'zdjęcie', 'R0. Zdjęcie: ramka GT, ramka modelu i IoU, patche narzędzia (obrys) i grupa wykryta bez etykiet'),
           ('R1', 'A', 'R1. A — mapa klas: oś narzędzie–tło (100 % separacji) + reszta chmury'),
           ('R2', 'S', 'R2. S — siły: 3 kierunki największej energii sił netto bloku 6'),
           ('R3', 'W', 'R3. W — wiązki ważone aktywacją (następca P3)'),
           ('R4', 'B', 'R4. B — rama siły wybranej cząsteczki (lokalna)'),
           ('R5', 'C', 'R5. C — rama Schura: podprzestrzeń niezmiennicza jakobianu (lokalna)'),
           ('R6', 'D', 'R6. D — wędrówka w czasie uczenia'),
           ('R7', 'F', 'R7. F — 7 widoków Fano: każda para kierunków razem raz'),
           ('R8', 'bez rzutu', 'R8. Bez rzutu: miary 192D i wierność widoków'),
           ('R9', 'L', 'R9. L — separacja narzędzia od tła: oś LDA ze skurczem + własne parametry'),
           ('R12', 'kNN 3D', 'R12. kNN na chmurze 3D: kolor tokenu narzędzia = ułamek sąsiadów, którzy też są narzędziem, z liniami do sąsiadów'),
           ('R11', 'miary w czasie', 'R11. Miary separacji w czasie uczenia: Fisher LDA i 192D, kNN PCA3, sylwetka 192D z zerem przestrzennym'),
           ('R10', 'LDA na żywo', 'R10. LDA na żywo: narzędzie vs tło oraz ta sama maska przesunięta w tło (zero przestrzenne)'))
pl = lambda x, n=2: '—' if x is None or not np.isfinite(x) else f'{x:.{n}f}'.replace('.', ',')


def proc(x):
    if x is None or not np.isfinite(x):
        return '—'
    v = 100 * float(x)
    return (f'{v:.1f}' if abs(v) < 10 else f'{v:.0f}').replace('.', ',') + ' %'


def strzalki(O, W, kol):
    """Odcinki O → O + W (n, 3) dla GLLinePlotItem (mode='lines'), kolory wierzchołków i końce (groty)."""
    O = np.asarray(O, np.float32); W = np.asarray(W, np.float32)
    seg = np.empty((2 * len(O), 3), np.float32)
    seg[0::2] = O; seg[1::2] = O + W
    return seg, np.repeat(np.asarray(kol, np.float32), 2, 0), O + W


class ProjectionWorker(QThread):
    postep = pyqtSignal(int, int, str)
    gotowe = pyqtSignal(object, object)
    blad = pyqtSignal(str)

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.stop = False

    def run(self):
        try:
            d = projections.policz(self.h, postep=lambda i, n, o: self.postep.emit(i, n, o), przerwij=lambda: self.stop)
            self.gotowe.emit(str(self.h.katalog), d)
        except projections.Interrupted:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class MetricsWorker(QThread):
    """Miary separacji w czasie uczenia (projections.miary_w_czasie_gpu) w tle; wynik zapisywany obok stanów 192D."""
    gotowe = pyqtSignal(object, object)
    postep = pyqtSignal(int, int)

    def __init__(self, dane, rdz, mies, wyklucz, plik, klucz):
        super().__init__()
        self.a = (dane, rdz, mies, wyklucz); self.plik = plik; self.klucz = klucz

    def run(self):
        try:
            w = projections.miary_w_czasie_gpu(*self.a, postep=lambda a, b: self.postep.emit(a, b))
            if w is not None:
                np.savez(self.plik, **w)
            self.gotowe.emit(self.klucz, w)
        except Exception:                                                # noqa: BLE001
            traceback.print_exc()
            self.gotowe.emit(self.klucz, None)


class FrameCWorker(QThread):
    """Rama Schura (projections.policz_rame_C) w tle. Zlecenia nadpisują się: liczy się tylko ostatnie."""
    gotowe = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._zlecenie = None

    def zlec(self, klucz, kw):
        self._zlecenie = (klucz, kw)
        if not self.isRunning():
            self.start()

    def run(self):
        while self._zlecenie is not None:
            klucz, kw = self._zlecenie
            self._zlecenie = None
            try:
                self.gotowe.emit(klucz, projections.policz_rame_C(**kw))
            except Exception as e:                                       # noqa: BLE001
                traceback.print_exc()
                self.gotowe.emit(klucz, dict(blad_watku=f'{type(e).__name__}: {e}'))


class LocalSections:
    """Elementy lokalne ProjectionGPU (S, SV, SK: (N, 4)): strzałki z grotem, odcinki i kropki; gruba = druga grubość linii."""

    def __init__(self):
        self.S, self.SV, self.SK = [], [], []

    def strzalki(self, p0, d, kol, gruba=False, grot=True):
        d = np.atleast_2d(np.asarray(d, np.float32))
        n = len(d)
        if n == 0:
            return self
        p0 = np.broadcast_to(np.asarray(p0, np.float32), (n, 3))
        self.S.append(np.c_[p0, np.full(n, 1.0 if grot else 0.0, np.float32)])
        self.SV.append(np.c_[d, np.full(n, float(gruba), np.float32)])
        self.SK.append(np.broadcast_to(np.asarray(kol, np.float32), (n, 4)))
        return self

    def odcinki(self, a, b, kol, gruba=False):
        return self.strzalki(a, np.asarray(b, np.float32) - np.asarray(a, np.float32), kol, gruba, grot=False)

    def kropki(self, p, kol):
        p = np.atleast_2d(np.asarray(p, np.float32))
        n = len(p)
        self.S.append(np.c_[p, np.full(n, 2.0, np.float32)])
        self.SV.append(np.zeros((n, 4), np.float32))
        self.SK.append(np.broadcast_to(np.asarray(kol, np.float32), (n, 4)))
        return self

    def tablice(self):
        if not self.S:
            return None
        return tuple(np.ascontiguousarray(np.concatenate(x), np.float32) for x in (self.S, self.SV, self.SK))


class Scene:
    """Widok 3D jednego rzutu. Cząsteczki, ślady, strzałki sił i elementy lokalne rysują shadery (projections_gl.ProjectionGPU: historia
    na karcie raz, w klatce tylko czas); osie, napisy i tory zerowe to lekkie elementy pyqtgraph."""

    def __init__(self):
        from laser_app.gpu.projections_gl import ProjectionGPU
        self.v = GLView()
        self.v.setBackgroundColor(TLO)
        self.v.setCameraPosition(distance=48, elevation=18, azimuth=-55)
        self.linie = {n: self._linie(w) for n, w in (('osie', 1.0), ('zero', 1.2))}
        self.g = ProjectionGPU()
        self.g.opcje.update(osie=0.0, grot=0.7)
        self.g.setDepthValue(5)
        self.v.addItem(self.g)
        self.napisy = []
        self.kadr = None
        self._kadr_arg = None
        self.v.reset_widoku = self._reset_widoku
        self._klucz_g = self._klucz_l = None
        self._maska = None

    def _linie(self, szer):
        it = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1, 1, 1, 0), width=szer, antialias=True, mode='lines',
                               glOptions='translucent')
        it.setVisible(False)
        self.v.addItem(it)
        return it

    def odcinki(self, nazwa, seg, kol):
        it = self.linie[nazwa]
        if seg is None or len(seg) < 2:
            it.setVisible(False)
            return
        it.setData(pos=np.ascontiguousarray(seg, np.float32), color=np.ascontiguousarray(kol, np.float32), mode='lines')
        it.setVisible(True)

    def wgraj(self, klucz, P, PS=None, F=None, rho=None):
        """Historia położeń (i sił) w jednostkach ekranu na kartę, tylko przy zmianie klucza (baza, środek, skala, wygładzanie)."""
        if klucz != self._klucz_g:
            self._klucz_g = klucz
            self.g.ustaw_polozenia(P, F, rho, PS)

    def maska(self, M):
        if self._maska is None or not np.array_equal(self._maska, M):
            self._maska = M.copy()
            self.g.ustaw_maske(M)

    def lokalne(self, klucz, budowa, kotwica=-1):
        """Elementy lokalne: budowa() → Lokalne, wołane tylko przy zmianie klucza (None = brak)."""
        self.g.opcje['kotwica'] = int(kotwica)
        if klucz != self._klucz_l:
            self._klucz_l = klucz
            el = None if budowa is None else budowa().tablice()
            self.g.ustaw_lokalne(*(el if el is not None else (None,)))
        self.g.update()

    def podpisy(self, lista):
        """lista: [(pozycja (3,), tekst, kolor (r, g, b, a) 0–255)]."""
        while len(self.napisy) < len(lista):
            t = gl.GLTextItem()
            self.v.addItem(t)
            self.napisy.append(t)
        for i, t in enumerate(self.napisy):
            if i < len(lista):
                p, s, c = lista[i]
                t.setData(pos=tuple(float(v) for v in p), text=s, color=c)
                t.setVisible(True)
            else:
                t.setVisible(False)

    def osie(self, nazwy, dl=1.25 * S_EKR, o=(0, 0, 0)):
        """Trzy osie widoku od punktu o z nazwami (nazwy = None: bez osi)."""
        if not nazwy:
            self.odcinki('osie', None, None)
            return []
        o = np.asarray(o, np.float32)
        E = np.eye(3, dtype=np.float32) * dl
        kol = np.array([[1, .45, .45, .55], [.45, 1, .55, .55], [.5, .65, 1, .55]], np.float32)
        seg, k, _ = strzalki(np.repeat(o[None], 3, 0), E, kol)
        self.odcinki('osie', seg, k)
        return [(o + E[a] * 1.06, nazwy[a], (210, 210, 210, 220)) for a in range(3)]

    def kadruj(self, P, klucz, kwantyl=99.5, zapas=2.4):
        """Kamera na środek, odległość = zapas × promień (kwantyl odległości punktów); tylko przy zmianie klucza."""
        if klucz == self.kadr or P is None or len(P) == 0:
            return
        self.kadr = klucz
        r = float(np.percentile(np.linalg.norm(P, axis=1), kwantyl))
        self._kadr_arg = max(zapas * r, 6.0)
        self._reset_widoku()

    def _reset_widoku(self):
        """Podwójny ŚPM: środek na początek układu i odległość z ostatniego kadru (kąty kamery zostają)."""
        self.v.opts['center'] = QVector3D(0, 0, 0)
        if self._kadr_arg is not None:
            self.v.setCameraPosition(distance=self._kadr_arg)
        self.v.update()

    def srodek(self, p):
        self.v.opts['center'] = QVector3D(float(p[0]), float(p[1]), float(p[2]))
        self.v.update()


class ProjectionWindows(QWidget):
    """Sekcja „Rzuty z 192 wymiarów” w lewym panelu zakładki i okienka R1–R8 (budowane przez buduj_okienka)."""

    def __init__(self, zakladka):
        super().__init__()
        from laser_app.ui.widgets import CollapsibleSection, _spin
        self._z = zakladka
        self.dane = None
        self._h = None
        self.t = 0.0
        self._watek = None
        self._sceny, self._lb, self._ostatnie = {}, {}, {}
        self._pam, self._u, self._frB = {}, {}, {}
        self._C = None; self._C_zlecony = None
        self._B = None
        self._blok = False
        self._i_panel = None
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        g = CollapsibleSection('Rzuty z 192 wymiarów: okienka R1–R8'); f = QFormLayout(g); lay.addWidget(g)
        self.sekcja = g
        opis = QLabel('Liczone w pełnych 192 wymiarach, rysowane w 3D. Pod każdym okienkiem: ile czego ten rzut pokazuje; losowy rzut '
                      '3D pokazuje średnio 1,6 % energii i 12,5 % długości wektora. Ślady = wędrówka cząsteczek przez checkpointy.')
        opis.setWordWrap(True); f.addRow(opis)
        hb = QHBoxLayout()
        self.bt_rz = QPushButton('Policz / wczytaj stany 192D'); self.bt_rz_stop = QPushButton('Przerwij')
        self.bt_rz.setToolTip('Jedno przejście w przód na każdy checkpoint wybranej historii (GPU): stany 192D przed MLP bloku 6, siły,\n'
                              'kolumny fc2, bazy rzutów i mierniki wierności. Wynik zostaje w katalogu historii (rzuty/), potem wczytuje się od razu.')
        self.bt_rz.clicked.connect(self._licz); self.bt_rz_stop.clicked.connect(self._przerwij); self.bt_rz_stop.setEnabled(False)
        hb.addWidget(self.bt_rz); hb.addWidget(self.bt_rz_stop); f.addRow(hb)
        self.pasek_rz = QProgressBar(); self.pasek_rz.setTextVisible(True); f.addRow(self.pasek_rz)
        self.lb_rz = QLabel('Najpierw „Policz / wczytaj” historię w sekcji Dane.'); self.lb_rz.setWordWrap(True); f.addRow(self.lb_rz)
        w_ = QWidget(); s_ = QGridLayout(w_); s_.setContentsMargins(0, 0, 0, 0); s_.setSpacing(2)
        for n, (k, sk, tyt) in enumerate(OKIENKA):
            b = QPushButton(f'{k} {sk}'); b.setToolTip(tyt + '\nKlik: pokaż to okienko na wierzchu.')
            b.clicked.connect(lambda _=False, k=k: self.pokaz(k))
            s_.addWidget(b, n // 4, n % 4)
        f.addRow('Okienka', w_)

        gw = CollapsibleSection('Wspólne: cząsteczki, ślady, strzałki (R1–R3, R6, R7)'); fw = QFormLayout(gw); f.addRow(gw)
        self.cb_kogo = NarrowCombo(); self.cb_kogo.addItems(['wszystkie cząsteczki', 'tylko narzędzie', 'tylko wybrane', 'narzędzie i wybrane'])
        self.cb_kogo.setToolTip('Które cząsteczki rysować. Wybrane = „Cząsteczka” i „+ cząsteczki” z sekcji sfery (biała = pierwsza).')
        self.cb_slad = NarrowCombo()
        self.cb_slad.addItems(['ślady: brak', 'ślady wybranych', 'ślady narzędzia', 'ślady narzędzia i wybranych', 'ślady wszystkich'])
        self.cb_slad.setCurrentIndex(3)
        self.cb_slad.setToolTip('Wędrówka: droga cząsteczki przez checkpointy uczenia w tym samym rzucie (rzut liniowy, więc droga rzutu =\n'
                                'rzut drogi). Jaśniejsza = nowsza.')
        self.sp_slad = _spin(0, 5000, 60, 5, sufiks=' ckpt'); self.sp_slad.setSpecialValueText('od początku uczenia')
        self.sp_slad.setToolTip('Długość śladu w checkpointach (0 = od pierwszego checkpointu).')
        self.sp_wyg = _spin(1, 200, 5, 1, sufiks=' ckpt'); self.sp_wyg.setSpecialValueText('bez wygładzania')
        self.sp_wyg.setToolTip('Ślady i tory zerowe jako średnia krocząca z tylu ostatnich checkpointów (tylko przeszłość). Szum SGD\n'
                               'między gęstymi checkpointami (co 10 kroków na początku) zamienia surowy ślad w gąszcz; wygładzony ślad\n'
                               'pokazuje dryf. Punkty cząsteczek zostają surowe. Odstępy checkpointów: 10 kroków do kroku 1000, potem 100.')
        self.cb_strz = NarrowCombo()
        self.cb_strz.addItems(['strzałki: brak', 'siła netto bloku 6: pokazane cząsteczki', 'siła netto bloku 6: tylko wybrane'])
        self.cb_strz.setCurrentIndex(2)
        self.cb_strz.setToolTip('Siła = przesunięcie przez MLP bloku 6 względem średniej po tokenach (suma 768 wiązek), dokładny rzut.')
        self.sp_strz = _spin(0.02, 50, 1.0, 0.1, 2, ' ×'); self.sp_strz.setToolTip('Długość strzałek sił i wiązek (te same jednostki co położenia).')
        self.ch_rho = QCheckBox('krycie strzałki = widoczna część długości ρ = |Pu|/|u|'); self.ch_rho.setChecked(True)
        self.ch_rho.setToolTip('Strzałka prawie niewidoczna = wektor leży prawie cały poza tym rzutem (jego długość na ekranie kłamie).')
        self.cb_srodek = NarrowCombo()
        self.cb_srodek.addItems(['środek: środek chmury w checkpoincie', 'środek: środek tła w checkpoincie (ruch względem tła)'])
        self.cb_srodek.setToolTip('Środek tła: widać, jak narzędzie odsuwa się od tła, a nie jak dryfuje cała chmura.')
        self.cb_skala = NarrowCombo()
        self.cb_skala.addItems(['skala: promień chmury w każdym checkpoincie (kształt)', 'skala: stała z końca uczenia (widać wzrost chmury)'])
        self.cb_skala.setToolTip('R = promień RMS chmury (średnia odległość tokenu od środka) w 192D.')
        self.sp_rozm = _spin(1, 20, 6, 0.5, 1, ' px')
        self.ch_kamera = QCheckBox('kamera śledzi wybraną cząsteczkę'); self.ch_kamera.setChecked(True)
        self.ch_kamera.setToolTip('Gdy jest wybrana cząsteczka, środek widoku jest na niej: LPM obraca wokół niej, kółko zbliża do niej.\n'
                                  'Bez wybranej cząsteczki (albo po wyłączeniu) kamera stoi na środku chmury.')
        self.ch_osie = QCheckBox('osie z nazwami'); self.ch_osie.setChecked(True)
        for nazwa, w in (('Pokazuj', self.cb_kogo), ('Wędrówka', self.cb_slad), ('Długość śladu', self.sp_slad),
                         ('Wygładzanie śladu', self.sp_wyg), ('Strzałki', self.cb_strz),
                         ('Długość strzałek', self.sp_strz), ('Środek', self.cb_srodek), ('Skala', self.cb_skala),
                         ('Wielkość punktów', self.sp_rozm)):
            fw.addRow(nazwa, w)
        fw.addRow(self.ch_rho); fw.addRow(self.ch_kamera); fw.addRow(self.ch_osie)

        ga = CollapsibleSection('R1 A, R3 W, R7 F'); fa = QFormLayout(ga); f.addRow(ga)
        self.cb_A = NarrowCombo()
        self.cb_A.addItems(['A: baza wspólna dla wszystkich checkpointów (stabilna)', 'A: baza w każdym checkpoincie (osie 2–3 wyrównane Procrustesem)'])
        self.cb_A.setToolTip('Wspólna: jedna kamera przez całe uczenie (najstabilniejsza wg benchmarku Verniera i in. 2020).\n'
                             'W każdym checkpoincie: oś 1 = Δ tego checkpointu, osie 2–3 obrócone do osi wspólnych; różnica między nimi\n'
                             'pokazuje, ile ruchu wnosi sama kamera.')
        self.ch_W = QCheckBox('R3: wiązki wybranej cząsteczki (sieć silna jak w R4)'); self.ch_W.setChecked(True)
        self.cb_F = NarrowCombo()
        self.cb_F.addItems([f'F {n + 1}: ' + ' · '.join(projections.NAZWY_F7[a] for a in tr) for n, tr in enumerate(projections.FANO)])
        self.cb_F.setToolTip('7 kierunków (Δ, 2 × reszta chmury, 2 × siły, 2 × wiązki), ortonormalizowanych w tej kolejności;\n'
                             'w 7 widokach płaszczyzny Fano każda para tych kierunków jest razem w dokładnie jednym widoku.')
        fa.addRow(self.cb_A); fa.addRow(self.ch_W); fa.addRow('R7 widok', self.cb_F)

        gb = CollapsibleSection('R4 B: rama siły'); fb = QFormLayout(gb); f.addRow(gb)
        self.sp_B_prog = _spin(0.2, 1.0, 0.8, 0.05, 2)
        self.sp_B_prog.setToolTip('Sieć silna: najsilniejsze wiązki, które razem niosą ten ułamek energii Σ|u_c|²; reszta jako jedna\n'
                                  'szara strzałka. Suma narysowanych = siła netto dokładnie (Radjai i in. 1998).')
        self.sp_B_skala = _spin(0.1, 20, 1.0, 0.1, 1, ' ×')
        fb.addRow('Sieć silna: ułamek energii', self.sp_B_prog); fb.addRow('Długość wiązek', self.sp_B_skala)

        gc = CollapsibleSection('R5 C: rama Schura'); fc = QFormLayout(gc); f.addRow(gc)
        self.ch_C = QCheckBox('licz przy zmianie checkpointu i cząsteczki (gdy R5 albo R8 widać)'); self.ch_C.setChecked(True)
        self.cb_C_para = NarrowCombo()
        self.cb_C_para.addItems(['para zespolona: najszybszy obrót |Im λ|', 'para zespolona: najmniej stabilna (max Re λ)',
                                 'para zespolona: najbliżej siły F̂'])
        self.cb_C_rzecz = NarrowCombo()
        self.cb_C_rzecz.addItems(['mod rzeczywisty: najsilniej wypycha (max Re λ)', 'mod rzeczywisty: najsilniej ściąga (min Re λ)',
                                  'mod rzeczywisty: najbliżej siły F̂'])
        self.sp_C_G = _spin(3, 15, 7, 2); self.sp_C_G.setToolTip('Węzłów siatki pola na bok (G³ punktów).')
        self.sp_C_zas = _spin(0.02, 3.0, 0.5, 0.05, 2, ' × odl. sąsiada')
        self.sp_C_zas.setToolTip('Pół boku sześcianu pola jako ułamek odległości do najbliższej innej cząsteczki w 192D.')
        self.cb_C_pole = NarrowCombo()
        self.cb_C_pole.addItems(['pole dokładne: F(x + Qu) − F(x)', 'część liniowa: (QᵀJQ) u'])
        self.sp_C_strz = _spin(0.1, 5, 1.0, 0.1, 1, ' ×')
        fc.addRow(self.ch_C); fc.addRow(self.cb_C_para); fc.addRow(self.cb_C_rzecz); fc.addRow('Siatka', self.sp_C_G)
        fc.addRow('Pół boku', self.sp_C_zas); fc.addRow('Pole', self.cb_C_pole); fc.addRow('Długość strzałek', self.sp_C_strz)

        gd = CollapsibleSection('R6 D: wędrówka'); fd = QFormLayout(gd); f.addRow(gd)
        self.cb_D = NarrowCombo()
        self.cb_D.addItems(['PCA przemieszczeń wszystkich cząsteczek (wspólna)', 'PCA przemieszczeń pokazanych cząsteczek',
                            'rama wybranej cząsteczki: start → koniec + odchylenia', 'mapa klas A (wspólna)'])
        self.cb_D.setToolTip('Kierunki ruchu (przemieszczeń między checkpointami), a nie rozrzutu chmury. Rama cząsteczki: oś 1 = jej\n'
                             'przesunięcie netto (widoczne w 100 %), osie 2–3 = główne zboczenia toru z tej prostej.')
        self.ch_D_zero = QCheckBox('tory zerowe wybranej: błądzenie losowe z tymi samymi krokami'); self.ch_D_zero.setChecked(True)
        self.ch_D_zero.setToolTip('Szare tory: te same długości kroków, losowe kierunki w 192D. W ramie cząsteczki każdy tor zerowy ma własną\n'
                                  'ramę (tę samą przewagę). PCA samego błądzenia daje łuki i pętle (Antognini i Sohl-Dickstein 2018):\n'
                                  'kształt toru znaczy coś tylko wtedy, gdy różni się od torów zerowych.')
        self.sp_D_zero = _spin(1, 30, 6, 1)
        fd.addRow(self.cb_D); fd.addRow(self.ch_D_zero); fd.addRow('Torów zerowych', self.sp_D_zero)

        gp = CollapsibleSection('R8: bez rzutu'); fp = QFormLayout(gp); f.addRow(gp)
        self.cb_P = NarrowCombo()
        self.cb_P.addItems(['wierność: rozrzut chmury', 'wierność: energia sił netto', 'wierność: energia wiązek',
                            'wierność: długość separacji Δ', 'wierność: ruch między checkpointami'])
        fp.addRow('Wykres wierności', self.cb_P)

        gl_ = CollapsibleSection('R9 L: separacja narzędzia od tła'); fl = QFormLayout(gl_); f.addRow(gl_)
        self.sp_L_lam = _spin(0.0, 10.0, 0.1, 0.05, 2)
        self.sp_L_lam.setToolTip('Skurcz LDA: w ∝ (S_w + λ·tr(S_w)/192·I)⁻¹ Δ. 0 = czysta LDA (przy 18 tokenach narzędzia przeucza),\n'
                                 'duże λ → sam kierunek Δ (jak oś 1 widoku A).')
        self.sp_L_N = _spin(1, 2000, 100, 10, sufiks=' ckpt')
        self.sp_L_N.setToolTip('Pula dopasowania: tyle ostatnich checkpointów (do checkpointu dopasowania) razem; więcej = stabilniejsza oś,\n'
                               'mniej = lepiej dopasowana do jednej chwili. 1 = tylko checkpoint dopasowania.')
        self.cb_L_kiedy = NarrowCombo(); self.cb_L_kiedy.addItems(['dopasuj na końcu uczenia (stała kamera)', 'dopasuj na bieżącym checkpoincie'])
        self.cb_L_osie = NarrowCombo()
        self.cb_L_osie.addItems(['osie 2–3: największy rozrzut wewnątrz klas', 'osie 2–3: najmniejszy rozrzut wewnątrz klas (klasy jak kropki)',
                                 'osie 2–3: PCA tła', 'osie 2–3: PCA narzędzia'])
        self.cb_L_etyk = NarrowCombo(); self.cb_L_etyk.addItems(['obrys narzędzia (udział pikseli w łacie)', 'ramka GT (stara etykieta)',
                                 'grupa wykryta bez etykiet (TokenCut na latencie MLP 768D)'])
        self.cb_L_etyk.setToolTip('Obrys: wielokąty narzędzia z adnotacji Cataract-LMM; narzędzie = łaty z udziałem pikseli ≥ próg,\n'
                                  'mieszane (0 < udział < próg) na żółto, poza dopasowaniem i miarami. Ramka GT obejmuje głównie tło\n'
                                  '(cienki cystotom: z 18 łat ramki 3 są w co najmniej połowie narzędziem).\n'
                                  'Grupa wykryta bez etykiet: TokenCut na aktywacjach MLP bloku 6 w danym checkpoincie (bez adnotacji);\n'
                                  'oś LDA dopasowana do tej grupy, obrys tylko do sprawdzenia (Dice, precyzja; żółte = narzędzie pominięte).')
        self.cb_L_os = NarrowCombo(); self.cb_L_os.addItems(['oś LDA z etykiety (obrys)', 'oś LDA z wykrytej grupy (bez etykiet)'])
        self.cb_L_os.setToolTip('Dla etykiety „grupa wykryta bez etykiet”: do czego dopasować oś LDA. Z obrysu: widok jak przy etykiecie,\n'
                                'a kolor narzędzia = grupa wykryta bez etykiet, więc widać, czy grupa leży tam, gdzie oznaczone narzędzie.\n'
                                'Z grupy: widok całkowicie bez etykiet. Przy innych etykietach oś zawsze z wybranej etykiety.')
        self.sp_L_prog = _spin(0.05, 1.0, 0.5, 0.05, 2)
        self.sp_L_prog.setToolTip('Próg udziału pikseli narzędzia w łacie (etykieta z obrysu).')
        self.ch_L_przestrz = QCheckBox('zero przestrzenne (ta sama maska w innych miejscach tła)'); self.ch_L_przestrz.setChecked(True)
        self.ch_L_przestrz.setToolTip('Maska narzędzia przesunięta w każde położenie siatki rozłączne z narzędziem, LDA od nowa dla każdego.\n'
                                      'Dowolny spójny kawałek tła też się „oddziela” (podobne sąsiednie łaty), więc to jest właściwe zero;\n'
                                      'p = odsetek położeń tła z miarą ≥ narzędzia. Liczone na pauzie, LDA dopasowana na bieżącym checkpoincie.')
        self.sp_L_brzeg = _spin(0, 3, 0, 1, sufiks=' kom.')
        self.sp_L_brzeg.setToolTip('Pomiń tyle skrajnych kolumn i wierszy ramki przy dopasowaniu i w miarach: prostokątna ramka obejmuje\n'
                                   'na brzegach piksele tła. Pominięte tokeny są rysowane na żółto.')
        self.sp_L_perm = _spin(0, 500, 100, 50); self.sp_L_perm.setToolTip('Losowych etykiet do zera miar (95. percentyl); 0 = bez zera.')
        self.ch_L_refit = QCheckBox('zero z ponownym dopasowaniem osi (DiProPerm)'); self.ch_L_refit.setChecked(True)
        self.ch_L_refit.setToolTip('Dla każdej losowej etykiety oś jest dopasowywana od nowa (na bieżącym checkpoincie), więc zero obejmuje\n'
                                   'przeuczenie 18 punktów w 192D. Uczciwsze od zera przy stałych osiach (dolne ograniczenie).')
        self.ch_L_cv = QCheckBox('walidacja blokowana po checkpointach (oś z innych bloków)'); self.ch_L_cv.setChecked(True)
        self.ch_L_cv.setToolTip('Pula checkpointów dzielona na bloki ciągłe; oś dopasowana bez bloku jest oceniana na jego środkowym checkpoincie.\n'
                                'Chroni przed korelacją czasową; nie chroni przed tymi samymi tokenami (te są w każdym checkpoincie).')
        self.sp_L_bl = _spin(2, 20, 5, 1, sufiks=' bloków')
        for nazwa, w in (('Skurcz λ', self.sp_L_lam), ('Pula dopasowania', self.sp_L_N), ('Kiedy', self.cb_L_kiedy),
                         ('Osie 2–3', self.cb_L_osie), ('Etykieta', self.cb_L_etyk), ('Oś LDA', self.cb_L_os), ('Próg udziału', self.sp_L_prog),
                         ('Brzeg ramki', self.sp_L_brzeg), ('Permutacji', self.sp_L_perm)):
            fl.addRow(nazwa, w)
        fl.addRow(self.ch_L_przestrz); fl.addRow(self.ch_L_refit); fl.addRow(self.ch_L_cv); fl.addRow('Bloków CV', self.sp_L_bl)
        g0 = CollapsibleSection('R0 i R10: zdjęcie z patchami i LDA na żywo'); f0 = QFormLayout(g0); f.addRow(g0)
        self.ch_R0_grupa = QCheckBox('R0: pokaż grupę wykrytą bez etykiet (czerwone patche)'); self.ch_R0_grupa.setChecked(True)
        self.ch_R0_grupa.setToolTip('TokenCut na aktywacjach MLP bloku 6 w bieżącym checkpoincie. Podczas odtwarzania liczy się tylko to, co już\n'
                                    'jest w pamięci; reszta na pauzie.')
        self.ch_R0_siatka = QCheckBox('R0: siatka patchy 16 × 16'); self.ch_R0_siatka.setChecked(False)
        self.cb_L10 = NarrowCombo(); self.cb_L10.addItems(['1. panel: oś jak w R9 (stała kamera)', '1. panel: oś dopasowana na tym checkpoincie',
                              '1. panel: PCA3 chmury (bez etykiet)'])
        self.cb_L10.setToolTip('Osie 2. i 3. panelu zawsze dopasowane od nowa do przesuniętej maski (zero przestrzenne).')
        self.ch_L10_tlo = QCheckBox('R10: ta sama maska w tle (mediana i najlepsze położenie)'); self.ch_L10_tlo.setChecked(True)
        self.ch_L10_tlo.setToolTip('Maska narzędzia przesunięta w każde położenie tła, LDA od nowa. Liczy się na pauzie (ok. 1 s).')
        self.cb_R12_widok = NarrowCombo(); self.cb_R12_widok.addItems(['PCA3 chmury (bez etykiet)', 'oś LDA jak w R9'])
        self.cb_R12_przestrzen = NarrowCombo(); self.cb_R12_przestrzen.addItems(['sąsiedzi w widoku 3D', 'sąsiedzi w pełnych 192D (rysunek w 3D)'])
        self.cb_R12_przestrzen.setToolTip('Gdzie szukać najbliższych sąsiadów: w tych 3 wymiarach, które widać, czy w pełnych 192D (wtedy linie\n'
                                          'mogą prowadzić do tokenów, które w rzucie są daleko).')
        self.sp_R12_k = _spin(0, 50, 0, 1); self.sp_R12_k.setSpecialValueText('auto (min(10, n − 1))')
        self.ch_R12_kr = QCheckBox('R12: linie do sąsiadów'); self.ch_R12_kr.setChecked(True)
        self.cb_R12_baza = NarrowCombo()
        self.cb_R12_baza.addItems(['baza PCA: wspólna dla całej historii (stabilna)',
                                   'baza PCA: w każdym checkpoincie, wyrównana do wspólnej (Procrustes 3 × 3)',
                                   'baza PCA: w każdym checkpoincie surowa (skacze)'])
        self.cb_R12_baza.setToolTip('Dotyczy widoku „PCA3 chmury”. Wspólna: osie z kowariancji zebranej po próbce checkpointów całej historii, liczone\n'
                                    'raz; kamera nie obraca się w czasie (domyślna, najstabilniejsza). W każdym checkpoincie wyrównana: osie PCA tego\n'
                                    'checkpointu obrócone/odbite Procrustesem 3 × 3 do osi wspólnych (pokazuje kształt dopasowany na bieżąco, bez\n'
                                    'przeskoków znaku i kolejności osi). Surowa: PCA od nowa w każdym checkpoincie, bez wyrównania; znaki osi i ich\n'
                                    'kolejność przeskakują, gdy wartości własne się mijają, więc chmura skacze (stare zachowanie; bez interpolacji).')
        self.sp_R12_bwyg = _spin(0, 50, 0, 1, sufiks=' ckpt'); self.sp_R12_bwyg.setSpecialValueText('wyłączone')
        self.sp_R12_bwyg.setToolTip('Tylko dla bazy „w każdym checkpoincie” (wyrównanej albo surowej): kowariancja przed liczeniem osi uśredniana po\n'
                                    'checkpointach od t − n do t + n, więc osie zmieniają się powoli. 0 = bez uśredniania.')
        self.ch_R12_interp = QCheckBox('R12: interpolacja między checkpointami'); self.ch_R12_interp.setChecked(True)
        self.ch_R12_interp.setToolTip('Położenie tokenu = (1 − f)·P_i + f·P_j między sąsiednimi checkpointami i i j (każdy w swojej bazie, więc bez\n'
                                      'skoku przy zmianie checkpointu), a kolor mieszany tak samo z ułamków sąsiadów. Wyłączona: ruch skokowy co\n'
                                      'checkpoint (najbliższy). Linie do sąsiadów zawsze z najbliższego checkpointu. Nie dotyczy bazy surowej.')
        self.cb_R12_metryka = NarrowCombo(); self.cb_R12_metryka.addItems(['metryka sąsiadów: euklidesowa', 'metryka sąsiadów: kosinusowa'])
        self.cb_R12_metryka.setToolTip('Jak mierzyć bliskość tokenów przy szukaniu sąsiadów: odległością euklidesową albo kosinusem kąta między\n'
                                       'wektorami (w widoku 3D: współrzędnych wyśrodkowanych, w 192D: stanów tokenów; długość nie ma znaczenia).')
        self.sp_R12_fwyg = _spin(0, 50, 0, 1, sufiks=' ckpt'); self.sp_R12_fwyg.setSpecialValueText('wyłączone')
        self.sp_R12_fwyg.setToolTip('Ułamek sąsiadów-narzędzi (kolor tokenu) uśredniony po checkpointach od t − n do t + n: sąsiedzi liczeni w każdym\n'
                                    'z nich osobno (w bieżącej bazie rzutu), potem średnia. Uspokaja migotanie koloru. 0 = bez uśredniania.')
        f0.addRow(self.ch_R0_grupa); f0.addRow(self.ch_R0_siatka); f0.addRow('R10', self.cb_L10); f0.addRow(self.ch_L10_tlo)
        f0.addRow('R12 widok', self.cb_R12_widok); f0.addRow('R12 baza PCA', self.cb_R12_baza); f0.addRow('R12 wygładzanie bazy ±', self.sp_R12_bwyg)
        f0.addRow(self.ch_R12_interp); f0.addRow('R12 sąsiedzi', self.cb_R12_przestrzen); f0.addRow('R12 metryka', self.cb_R12_metryka)
        f0.addRow('R12 k', self.sp_R12_k); f0.addRow('R12 wygładzanie ułamka ±', self.sp_R12_fwyg); f0.addRow(self.ch_R12_kr)
        for w_ in (self.cb_R12_widok, self.cb_R12_przestrzen, self.cb_R12_baza, self.cb_R12_metryka):
            w_.currentIndexChanged.connect(self._opcje)
        for w_ in (self.sp_R12_k, self.sp_R12_bwyg, self.sp_R12_fwyg):
            w_.valueChanged.connect(self._opcje)
        self.ch_R12_kr.toggled.connect(self._opcje); self.ch_R12_interp.toggled.connect(self._opcje)
        for w_ in (self.ch_R0_grupa, self.ch_R0_siatka, self.ch_L10_tlo):
            w_.toggled.connect(self._opcje)
        self.cb_L10.currentIndexChanged.connect(self._opcje)
        for z_ in (gw, ga, gb, gc, gd, gp, gl_, g0):
            z_.naglowek.setChecked(False)                                  # podsekcje zwinięte: panel nie rośnie
        # okienko → podsekcje z jego ustawieniami (prawy klik w okienko R otwiera je w popupie, tab.menu_okienka)
        self._sekcje_R = {'R0': (g0,), 'R1': (ga, gw), 'R2': (gw,), 'R3': (ga, gw), 'R4': (gb,), 'R5': (gc,), 'R6': (gd, gw),
                          'R7': (ga, gw), 'R8': (gp,), 'R9': (gl_,), 'R10': (g0, gl_), 'R11': (gl_,), 'R12': (g0,)}
        for w in (self.cb_kogo, self.cb_slad, self.cb_strz, self.cb_srodek, self.cb_skala, self.cb_A, self.cb_F, self.cb_D,
                  self.cb_L_kiedy, self.cb_L_osie, self.cb_L_etyk, self.cb_L_os):
            w.currentIndexChanged.connect(self._opcje)
        for w in (self.sp_slad, self.sp_wyg, self.sp_strz, self.sp_rozm, self.sp_B_prog, self.sp_B_skala, self.sp_C_strz, self.sp_D_zero,
                  self.sp_L_lam, self.sp_L_N, self.sp_L_brzeg, self.sp_L_perm, self.sp_L_bl, self.sp_L_prog):
            w.valueChanged.connect(self._opcje)
        for w in (self.ch_rho, self.ch_kamera, self.ch_osie, self.ch_W, self.ch_D_zero, self.ch_L_refit, self.ch_L_cv, self.ch_L_przestrz):
            w.toggled.connect(self._opcje)
        for w in (self.cb_C_para, self.cb_C_rzecz, self.cb_C_pole):
            w.currentIndexChanged.connect(lambda *_: self._rysuj_widoczne())
        for w in (self.sp_C_G, self.sp_C_zas):
            w.valueChanged.connect(lambda *_: self._rysuj_widoczne())
        self.ch_C.toggled.connect(lambda *_: self._rysuj_widoczne())
        self.cb_P.currentIndexChanged.connect(lambda *_: self._panel_wiernosc())

    def sekcje_okienka(self, k):
        """Podsekcje panelu z ustawieniami okienka k (R0–R12); pusta lista dla nieznanego."""
        return list(self._sekcje_R.get(k, ()))

    # ------------------------------------------------------------------ budowa okienek
    def buduj_okienka(self):
        z = self._z
        for k, _, tyt in OKIENKA:
            if k == 'R8':
                w = self._buduj_panel()
            elif k == 'R0':
                w = self._buduj_zdjecie()
            elif k == 'R10':
                w = self._buduj_lda()
            elif k == 'R11':
                w = self._buduj_miary()
            elif k == 'R12':
                w = self._buduj_knn()
            else:
                sc = Scene()
                self._sceny[k] = sc
                sc.v.klik_px.connect(lambda x, y, k=k: self._klik(k, x, y))
                w = self._z_napisem(sc.v, k)
            d = z._dok(k, tyt, w)
            d.visibilityChanged.connect(lambda on, k=k: self._rysuj(k) if on else None)
        z.sp_tok.valueChanged.connect(lambda *_: self._wybor())
        z.ed_tok.editingFinished.connect(self._wybor)
        self._watek_C = FrameCWorker()
        self._watek_C.gotowe.connect(self._C_gotowe)

    def uklad_domyslny(self):
        """R1–R7 w kartach obok sfery reflektorów, R8 w karcie obok wykresu 6."""
        d, okna = self._z.doki, self._z.okna
        prev = d['s']
        for k in ('R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R9', 'R10', 'R11', 'R12'):
            okna.tabifyDockWidget(prev, d[k])
            prev = d[k]
        okna.tabifyDockWidget(d['6'], d['R8'])
        d['s'].raise_(); d['6'].raise_()

    def _z_napisem(self, widok, k):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(widok, 1)
        lb = QLabel('')
        lb.setStyleSheet('color:#ffd166; padding:2px 6px; background:#0b1014')
        lb.setWordWrap(True)
        lb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        lb.setFixedHeight(4 * lb.fontMetrics().height() + 10)              # stała wysokość: zmiana napisu nie przelicza doków
        lb.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        from laser_app.ui.dock_windows import bez_ukladu
        v.addWidget(bez_ukladu(lb))                                         # setText nie przelicza układu okna (dock_windows._LayoutShield)
        self._lb[k] = lb
        return w

    # ------------------------------------------------------------------ R0: zdjęcie z ramkami i patchami
    def _buduj_zdjecie(self):
        w = pg.GraphicsLayoutWidget(); w.setBackground(TLO)
        self.vb0 = w.addViewBox(lockAspect=True, enableMenu=False)
        self.vb0.invertY(True)
        self.im0 = pg.ImageItem(axisOrder='row-major'); self.vb0.addItem(self.im0)
        self._r0_klucz = None
        self.kr0 = {}
        for n, (kol, szer, styl) in dict(siatka=((255, 255, 255, 40), 1, None), mies=((255, 209, 102, 255), 2, None),
                                           narz=((90, 255, 90, 255), 2, None), grupa=((255, 70, 70, 255), 2, Qt.PenStyle.DashLine),
                                           wyb=((255, 255, 255, 255), 3, None), gt=((255, 255, 255, 230), 2, Qt.PenStyle.DotLine),
                                           model=((0, 230, 255, 255), 2, None)).items():
            c = pg.PlotCurveItem(pen=pg.mkPen(kol, width=szer, style=styl) if styl is not None else pg.mkPen(kol, width=szer),
                                 connect='pairs' if n in ('siatka', 'mies', 'narz', 'grupa', 'wyb') else 'all')
            self.vb0.addItem(c); self.kr0[n] = c
        self.tx0 = pg.TextItem('', color=(255, 255, 255), anchor=(0, 0), fill=(0, 0, 0, 160))
        self.vb0.addItem(self.tx0); self.tx0.setPos(4, 4)
        return self._z_napisem(w, 'R0')

    @staticmethod
    def _krawedzie(idx):
        """(n · 8, 2): odcinki obrysów patchy 16 × 16 (pary punktów) dla tokenów idx."""
        out = []
        for t in idx:
            r, c = divmod(int(t), hist.W)
            x0, y0 = c * 16, r * 16; x1, y1 = x0 + 16, y0 + 16
            out += [(x0, y0), (x1, y0), (x1, y0), (x1, y1), (x1, y1), (x0, y1), (x0, y1), (x0, y0)]
        return np.array(out, float).reshape(-1, 2)

    def _rysuj_zdjecie(self):
        h = self._h
        K = h.K
        i = int(np.clip(round(self.t), 0, K - 1))
        if self._r0_klucz != id(h):
            self._r0_klucz = id(h)
            self.im0.setImage(np.asarray(h.rgb), axisOrder='row-major')
            self.vb0.setRange(QRectF(0, 0, hist.W * 16, hist.H * 16), padding=0.01)
            self.kr0['siatka'].setData(self._krawedzie(range(hist.TOK)).T[0], self._krawedzie(range(hist.TOK)).T[1])
        narz = np.asarray(h.narz, bool); mies = np.asarray(getattr(h, 'mieszane', np.zeros(hist.TOK, bool)), bool)
        zbior = lambda n, idx: self.kr0[n].setData(*self._krawedzie(idx).T) if len(idx) else self.kr0[n].setData([], [])
        zbior('narz', np.flatnonzero(narz)); zbior('mies', np.flatnonzero(mies))
        self.kr0['siatka'].setVisible(self.ch_R0_siatka.isChecked())
        wyb = self._wybrane()
        zbior('wyb', list(wyb)[:1] if wyb else [])
        grupa_txt = ''
        G = None
        if self.ch_R0_grupa.isChecked() and (('grupa', i) in self._pam or not self._z.bt_play.isChecked()):
            G = self._grupa(i)
        zbior('grupa', np.flatnonzero(G) if G is not None else [])
        if G is not None:
            u = np.asarray(getattr(h, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
            if np.isfinite(u).all():
                rdz = u >= 0.25
                dice = 2 * (G & rdz).sum() / max(G.sum() + rdz.sum(), 1)
                prec = (u[G] > 0).mean() if G.any() else float('nan')
                grupa_txt = f' Grupa bez etykiet (czerwona): {int(G.sum())} patchy, Dice z obrysem {pl(float(dice))}, precyzja {pl(float(prec))}.'
            else:
                grupa_txt = f' Grupa bez etykiet (czerwona): {int(G.sum())} patchy.'
        gt = h.gt
        if gt is not None:
            self.kr0['gt'].setData([gt[0], gt[2], gt[2], gt[0], gt[0]], [gt[1], gt[1], gt[3], gt[3], gt[1]])
        rb = np.asarray(h.ramka[i], float)
        self.kr0['model'].setData([rb[0], rb[2], rb[2], rb[0], rb[0]], [rb[1], rb[1], rb[3], rb[3], rb[1]])
        iou, pew = float(h.iou[i]), float(h.pew[i])
        self.tx0.setText(f'krok {int(h.kroki[i])}  IoU {pl(iou, 3)}  pewność {pl(pew)}')
        self._lb['R0'].setText(f'Checkpoint {i + 1}/{K} (krok {int(h.kroki[i])}). Biała kropkowana: ramka GT, cyjan: ramka modelu, IoU {pl(iou, 3)}, '
                               f'pewność {pl(pew)}. Zielone patche: narzędzie z obrysu ({int(narz.sum())}), żółte: mieszane ({int(mies.sum())}).'
                               f'{grupa_txt}  [{getattr(h, "etykieta", "ramka GT")}]')

    # ------------------------------------------------------------------ R10: LDA na żywo
    def _buduj_lda(self):
        w = pg.GraphicsLayoutWidget(); w.setBackground('#0b1014')
        self.p10, self.s10 = [], []
        for c in range(3):
            p = w.addPlot(row=0, col=c)
            p.showGrid(x=True, y=True, alpha=0.2); p.setLabel('bottom', 'oś 1 (LDA)'); p.setLabel('left', 'oś 2')
            tlo = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(150, 160, 175, 150))
            nar = pg.ScatterPlotItem(size=9, pen=pg.mkPen('k', width=0.5), brush=pg.mkBrush(90, 255, 90))
            p.addItem(tlo); p.addItem(nar)
            self.p10.append(p); self.s10.append((tlo, nar))
        return self._z_napisem(w, 'R10')

    def _wyklucz(self, rdz, mies):
        wyk = rdz | mies
        u = np.asarray(getattr(self.dane, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
        if np.isfinite(u).all():
            wyk = wyk | (u > 0)                                   # żadnego piksela narzędzia w położeniach tła
        return wyk

    def _rysuj_lda(self):
        import torch
        d = self.dane
        _, _, _, i = self._indeksy()
        rdz, mies = self._etykieta(i)
        keep = ~mies
        X = np.asarray(d.X[i], np.float32)
        Xk, m = X[keep], rdz[keep]
        if m.sum() < 2:
            self._lb['R10'].setText('Za mało tokenów narzędzia w tej etykiecie.')
            return
        kk = projections.k_sasiadow(m.sum())
        lam, osie = float(self.sp_L_lam.value()), self.cb_L_osie.currentIndex()
        if self.cb_L10.currentIndex() == 0:
            B, _ = self._baza_L()
            zrodlo = 'oś jak w R9'
        elif self.cb_L10.currentIndex() == 2:
            dev = projections._dev()
            Xt = torch.as_tensor(Xk, device=dev)
            Xc = Xt - Xt.mean(0, keepdim=True)
            B = torch.linalg.eigh(Xc.T @ Xc)[1][:, -3:].flip(1).cpu().numpy()
            zrodlo = 'PCA3 chmury (bez etykiet)'
        else:
            dev = projections._dev()
            B = projections.baza_L_t(torch.as_tensor(Xk, device=dev), torch.as_tensor(m, device=dev)[None], lam, osie)[0].cpu().numpy()
            zrodlo = 'oś dopasowana na tym checkpoincie'
        P = Xk @ np.asarray(B, np.float32)
        mi, _ = projections.separacja_gpu(P, m, 0, k=kk)
        fp = lambda v: f'{v:.3f}'.replace('.', ',')

        def rysuj(n, P_, m_, tytul):
            tlo, nar = self.s10[n]
            tlo.setData(P_[~m_, 0], P_[~m_, 1]); nar.setData(P_[m_, 0], P_[m_, 1])
            self.p10[n].setTitle(tytul, size='9pt')
        rysuj(0, P, m, f'narzędzie ({self._opis_zr(rdz)}): Fisher {pl(mi["fisher"])}, sylw. {pl(mi["sil"])}, {kk}NN {pl(mi["hit"])}')
        klucz = ('R10', int(i), self._klucz_etyk(), round(lam, 3), osie, kk)
        gra = self._z.bt_play.isChecked()
        tekst = f'Checkpoint {i + 1}, {zrodlo}. Etykieta: {self._opis_etyk(rdz, mies)}. Narzędzie: Fisher {pl(mi["fisher"])}, sylwetka {pl(mi["sil"])}.'
        if self.ch_L10_tlo.isChecked():
            if klucz not in self._pam and not gra:
                self._pam[klucz] = projections.widoki_zera_przestrzennego_gpu(X, rdz, self._wyklucz(rdz, mies), keep, lam, osie, kk)
            wy = self._pam.get(klucz, 'brak')
            if wy is None:
                tekst += ' Zero przestrzenne: brak położeń maski w tle.'
            elif wy == 'brak':
                tekst += ' Maska w tle liczy się na pauzie.'
            else:
                for n, nazwa in ((1, 'mediana'), (2, 'najlepsza')):
                    P_, m_, mm, j = wy['widoki'][nazwa]
                    rysuj(n, P_, m_, f'ta sama maska w tle ({nazwa} z {wy["n"]}): Fisher {pl(mm["fisher"])}, sylw. {pl(mm["sil"])}, {kk}NN {pl(mm["hit"])}')
                    self.s10[n][1].setBrush(pg.mkBrush(255, 150, 40) if n == 1 else pg.mkBrush(255, 70, 70))
                tekst += (f' Zero przestrzenne ({wy["n"]} położeń, LDA od nowa): p Fisher = {fp(wy["p"]["fisher"])}, p sylwetka = '
                          f'{fp(wy["p"]["sil"])}, p {kk}NN = {fp(wy["p"]["hit"])}.')
        else:
            for n in (1, 2):
                self.s10[n][0].setData([], []); self.s10[n][1].setData([], []); self.p10[n].setTitle('')
        self._lb['R10'].setText(tekst)

    def _opis_zr(self, rdz):
        return 'grupa bez etykiet' if self.cb_L_etyk.currentIndex() == 2 else ('obrys' if self.cb_L_etyk.currentIndex() == 0 else 'ramka GT')

    # ------------------------------------------------------------------ R11: miary separacji w czasie
    def _buduj_miary(self):
        w = pg.GraphicsLayoutWidget(); w.setBackground('#0b1014')
        self.p11 = {}
        for n, (klucz, tyt) in enumerate((('LDA_fisher', 'Fisher, LDA'), ('192D_fisher', 'Fisher, pełne 192D'),
                                          ('PCA3_hit', 'kNN, PCA3 chmury'), ('192D_sil', 'Sylwetka, 192D'))):
            p = w.addPlot(row=n // 2, col=n % 2, title=tyt)
            p.showGrid(x=True, y=True, alpha=0.2); p.setLabel('bottom', 'checkpoint'); p.setMouseEnabled(x=True, y=False)
            p.addLegend(offset=(5, 5))
            zero = p.plot(pen=pg.mkPen((170, 170, 170), width=1.5, style=Qt.PenStyle.DashLine), name='95 % położeń tła (zero przestrzenne)')
            nar = p.plot(pen=pg.mkPen((90, 255, 90), width=2), name='narzędzie')
            kur = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('w', width=1.5))
            p.addItem(kur); kur.sigPositionChanged.connect(self._kursor); p.kursor = kur
            self.p11[klucz] = (p, nar, zero)
        self._miary_czas = None
        self._miary_klucz = None
        self._watek_m = None
        return self._z_napisem(w, 'R11')

    def _rysuj_miary(self):
        d, h = self.dane, self._h
        klucz = str(h.katalog)
        if self._miary_klucz != klucz:
            self._miary_klucz = klucz
            self._miary_czas = None
            plik = projections.katalog(h) / 'miary_czas_v1.npz'
            if plik.exists():
                try:
                    z = np.load(plik, allow_pickle=False)
                    self._miary_czas = {k: z[k] for k in z.files}
                except Exception:                                        # noqa: BLE001
                    self._miary_czas = None
        if self._miary_czas is None:
            if self._watek_m is None:
                rdz = np.asarray(d.narz, bool); mies = np.asarray(getattr(d, 'mieszane', np.zeros(hist.TOK, bool)), bool)
                u = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
                wyk = rdz | mies | ((u > 0) if np.isfinite(u).all() else False)
                self._watek_m = MetricsWorker(d, rdz, mies, wyk, projections.katalog(h) / 'miary_czas_v1.npz', klucz)
                self._watek_m.gotowe.connect(self._miary_gotowe)
                self._watek_m.postep.connect(lambda a, b: self._lb['R11'].setText(f'Liczę miary w czasie na GPU: {a}/{b} checkpointów…'))
                self._watek_m.start()
            return
        m = self._miary_czas
        x = m['idx'].astype(float)
        for kl, (p, nar, zero) in self.p11.items():
            v, q = kl.split('_')
            vi, qi = projections.MIARY_CZAS_WIDOKI.index(v), projections.MIARY_CZAS_NAZWY.index(q)
            nar.setData(x, m['narz'][:, vi, qi]); zero.setData(x, m['zero95'][:, vi, qi])
            self._blok = True; p.kursor.setValue(self.t); self._blok = False
        self._lb['R11'].setText(
            f'Narzędzie: {int(m["n_narz"])} patchy, zero przestrzenne z {int(m["n_polozen"])} losowych położeń tej samej maski w tle (95. percentyl; '
            f'LDA dopasowana od nowa na każdym checkpoincie i położeniu), kNN z k = {int(m["k"])}. Narzędzie nad linią przerywaną = wyróżnia się. '
            f'Kursor = bieżący checkpoint. Wynik zapisany w katalogu rzutów.')

    def _miary_gotowe(self, klucz, wyn):
        self._watek_m = None
        if wyn is not None and klucz == self._miary_klucz:
            self._miary_czas = wyn
            self._rysuj_miary()

    # ------------------------------------------------------------------ R12: kNN na chmurze 3D
    def _buduj_knn(self):
        self.v12 = GLView(); self.v12.setBackgroundColor(TLO)
        self.v12.setCameraPosition(distance=40, elevation=18, azimuth=-55)
        self.sc12_tlo = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=3, color=(0.55, 0.62, 0.7, 0.55), pxMode=True)
        self.sc12_nar = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=13, color=(0.4, 1, 0.4, 1), pxMode=True)
        self.kr12 = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1, 1, 1, 0.5), width=1, mode='lines', antialias=True)
        for it in (self.sc12_tlo, self.kr12, self.sc12_nar):
            self.v12.addItem(it)
        self._kadr12 = None
        self._pam12 = {}                                                     # bazy PCA R12 (wspólna, wyrównane): osobno od _pam, które _ekran czyści
        return self._z_napisem(self.v12, 'R12')

    def _osie12(self, c, keep_t):
        """Baza rzutu R12 (192, 3) torch dla checkpointu c i informacja, czy chmurę przed rzutem wyśrodkować (PCA tak, oś LDA nie).
        Wspólna baza PCA liczona raz na historię (próbka 32 checkpointów), wyrównane i surowe osie raz na checkpoint, wszystko w `_pam`."""
        import torch
        d = self.dane
        dev = projections._dev()
        if self.cb_R12_widok.currentIndex() == 1:
            B, _ = self._baza_L()
            return torch.as_tensor(np.asarray(B, np.float32), device=dev), False
        tryb = self.cb_R12_baza.currentIndex()
        nb = 0 if tryb == 0 else int(self.sp_R12_bwyg.value())
        ke = (str(self._h.katalog), d.K, self._klucz_etyk(), int(keep_t.sum()))
        kU = ('R12U',) + ke
        if kU not in self._pam12:
            ks = np.unique(np.linspace(0, d.K - 1, min(d.K, 32)).round().astype(int))
            Xs = torch.as_tensor(np.asarray(d.X[ks], np.float32), device=dev)[:, keep_t]
            self._pam12[kU] = projections.baza_pca3_wspolna_t(Xs)
        U = self._pam12[kU]
        if tryb == 0:
            return U, True
        kV = ('R12V', tryb, nb, c) + ke
        if kV not in self._pam12:
            if len(self._pam12) > 3000:
                self._pam12 = {k_: v_ for k_, v_ in self._pam12.items() if k_[0] == 'R12U'}   # ograniczenie pamięci: zostaje tylko baza wspólna
            ks = np.arange(max(0, c - nb), min(d.K - 1, c + nb) + 1)               # okno ± nb checkpointów, obcięte na brzegach historii
            Xw = torch.as_tensor(np.asarray(d.X[ks], np.float32), device=dev)[:, keep_t]
            self._pam12[kV] = projections.baza_pca3_ckpt_t(Xw, U if tryb == 1 else None)
        return self._pam12[kV], True

    def _sasiedzi12(self, c, V, cen, keep_t, m_t, idx_t, k):
        """(ułamek sąsiadów-narzędzi tokenów narzędzia, indeksy sąsiadów (n, k)) dla checkpointu c: sąsiedzi w każdym checkpoincie okna
        ± „wygładzanie ułamka” osobno (w bazie V checkpointu c), ułamek uśredniony po oknie, indeksy z checkpointu c."""
        import torch
        d = self.dane
        nsm = int(self.sp_R12_fwyg.value())
        ks = np.arange(max(0, c - nsm), min(d.K - 1, c + nsm) + 1)
        Xw = torch.as_tensor(np.asarray(d.X[ks], np.float32), device=projections._dev())[:, keep_t]
        if self.cb_R12_przestrzen.currentIndex() == 0:
            Q = (Xw - Xw.mean(1, keepdim=True) if cen else Xw) @ V
        else:
            Q = Xw
        nn = projections.knn_t(Q, k, self.cb_R12_metryka.currentIndex())                  # (b, n, k)
        fr = m_t[nn[:, idx_t]].float().mean(-1).mean(0)                          # (nt,) średnia po oknie
        return fr.cpu().numpy(), nn[c - int(ks[0])].cpu().numpy()

    def _rysuj_knn(self):
        import torch
        d = self.dane
        ia, ja, f, wi = self._indeksy()
        rdz, mies = self._etykieta(wi)
        keep = ~mies
        m = rdz[keep]
        nt = int(m.sum())
        nk = int(keep.sum())
        if nt < 2:
            self._lb['R12'].setText('Za mało tokenów narzędzia w tej etykiecie.')
            return
        dev = projections._dev()
        pca = self.cb_R12_widok.currentIndex() == 0
        tryb = self.cb_R12_baza.currentIndex()
        # interpolacja między checkpointami; surowa baza PCA skacze, więc jej środek (średnia przeskoczonych osi) nie ma sensu
        interp = self.ch_R12_interp.isChecked() and not (pca and tryb == 2) and ia != ja and f > 0
        if not interp:
            ia = ja = wi; f = 0.0
        keep_t = torch.as_tensor(keep, device=dev)
        m_t = torch.as_tensor(m, device=dev)
        idx_n = np.flatnonzero(m)
        idx_t = torch.as_tensor(idx_n, device=dev)
        k = int(self.sp_R12_k.value()) or projections.k_sasiadow(nt)
        k = int(min(k, nk - 1))
        P_c, fr_c, nn_c = {}, {}, {}
        for c in sorted({ia, ja, wi}):
            V, cen = self._osie12(c, keep_t)
            X = torch.as_tensor(np.asarray(d.X[c], np.float32), device=dev)[keep_t]
            P_c[c] = (X - X.mean(0, keepdim=True) if cen else X) @ V              # każdy checkpoint w swojej (wyrównanej) bazie: ciągłość przy zmianie i → j
            fr_c[c], nn_c[c] = self._sasiedzi12(c, V, cen, keep_t, m_t, idx_t, k)
        P = ((1 - f) * P_c[ia] + f * P_c[ja]).cpu().numpy().astype(np.float32)
        frac = ((1 - f) * fr_c[ia] + f * fr_c[ja]).astype(np.float32)             # kolor też mieszany; linie z najbliższego checkpointu
        nn = nn_c[wi]
        traf = m[nn[idx_n]]                                                    # (nt, k) czy sąsiad jest narzędziem
        # kolor tokenu narzędzia: czerwony (0) → żółty → zielony (1) wg ułamka sąsiadów będących narzędziem
        kol = np.stack([np.clip(2 - 2 * frac, 0, 1), np.clip(2 * frac, 0, 1), 0.15 * np.ones_like(frac), np.ones_like(frac)], 1).astype(np.float32)
        self.sc12_tlo.setData(pos=P[~m], size=3, color=(0.55, 0.62, 0.7, 0.55))
        self.sc12_nar.setData(pos=P[m], size=13, color=kol)
        if self.ch_R12_kr.isChecked():
            a = np.repeat(P[idx_n], k, axis=0)
            b = P[nn[idx_n].reshape(-1)]
            seg = np.empty((2 * len(a), 3), np.float32); seg[0::2], seg[1::2] = a, b
            kk = np.where(traf.reshape(-1, 1), [[0.3, 1.0, 0.3, 0.7]], [[1.0, 0.3, 0.25, 0.7]]).astype(np.float32)
            self.kr12.setData(pos=seg, color=np.repeat(kk, 2, axis=0))
            self.kr12.setVisible(True)
        else:
            self.kr12.setVisible(False)
        r_ = float(np.linalg.norm(P, axis=1).max())                         # kadr mieści każdy token, także daleko leżące narzędzie
        klucz = (str(self._h.katalog), self.cb_R12_widok.currentIndex(), tryb)
        if self._kadr12 is None or self._kadr12[0] != klucz or r_ > self._kadr12[1] * 0.95 or r_ < self._kadr12[1] * 0.4:
            self._kadr12 = (klucz, r_)                                       # zmiana skali chmury w czasie: kadr z histerezą, bez skakania
            self.v12.opts['center'] = QVector3D(0, 0, 0)
            self.v12.setCameraPosition(distance=max(2.9 * r_, 3.0))
        losowo = (nt - 1) / max(nk - 1, 1)
        nb, nf = int(self.sp_R12_bwyg.value()), int(self.sp_R12_fwyg.value())
        if pca:
            widok = 'PCA3 chmury (bez etykiet), ' + ['baza wspólna dla całej historii',
                                                    'baza w każdym checkpoincie wyrównana Procrustesem do wspólnej',
                                                    'baza w każdym checkpoincie surowa (skacze)'][tryb]
            if tryb and nb:
                widok += f', kowariancja uśredniona ± {nb} ckpt'
        else:
            widok = 'oś LDA jak w R9'
        czas = f'{pl(ia + f + 1)} (interpolacja)' if interp else f'{wi + 1}'
        self._lb['R12'].setText(
            f'Checkpoint {czas}. Widok 3D: {widok}; sąsiedzi (k = {k}, metryka {["euklidesowa", "kosinusowa"][self.cb_R12_metryka.currentIndex()]}) liczeni '
            f'{"w tym widoku 3D" if self.cb_R12_przestrzen.currentIndex() == 0 else "w pełnych 192D"}. '
            f'Duże kule = tokeny narzędzia ({nt}), kolor = ułamek ich sąsiadów, którzy też są narzędziem{f" (uśredniony ± {nf} ckpt)" if nf else ""} '
            f'(zielony 1, czerwony 0); średnio {pl(float(frac.mean()))} (losowo {pl(losowo, 3)}). Linie: zielone = sąsiad narzędzie, czerwone = sąsiad tło. '
            f'Szare = tło ({int((~m).sum())}); łaty mieszane pominięte ({int(mies.sum())}).')

    def _buduj_panel(self):
        w = pg.GraphicsLayoutWidget(); w.setBackground('#0b1014')
        self.p_w = w.addPlot(row=0, col=0, colspan=2, title='Wierność widoków w czasie: jaka część jest w rzucie 3D')
        self.p_sep = w.addPlot(row=1, col=0, title='Separacja narzędzie–tło w 192D [R] i losowe etykiety')
        self.p_cos = w.addPlot(row=1, col=1, title='cos(x_i, x_j) w 192D w bieżącym checkpoincie')
        self.p_wz = w.addPlot(row=2, col=0, title='Pary wiązek wybranej cząsteczki: cos w 192D i w P3')
        self.p_J = w.addPlot(row=2, col=1, title='Widmo J wybranej cząsteczki (z R5) i koło Ginibre’a')
        for p in (self.p_w, self.p_sep, self.p_cos, self.p_wz, self.p_J):
            p.showGrid(x=True, y=True, alpha=0.2)
        for p in (self.p_w, self.p_sep):
            kur = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('w', width=1.5))
            p.addItem(kur)
            kur.sigPositionChanged.connect(self._kursor)
            p.kursor = kur
            p.setMouseEnabled(x=True, y=False)
        self.p_w.addLegend(offset=(5, 5)); self.p_cos.addLegend(offset=(5, 5)); self.p_wz.addLegend(offset=(-5, 5))
        self.tx_wz = pg.TextItem('', color='#ffd166', anchor=(0, 0)); self.p_wz.addItem(self.tx_wz)
        self.tx_J = pg.TextItem('', color='#ffd166', anchor=(0, 0)); self.p_J.addItem(self.tx_J)
        self.p_J.setAspectLocked(True)
        return w

    # ------------------------------------------------------------------ dane
    def ustaw_historie(self, h):
        self._h = h
        self.dane = None
        self._pam.clear(); self._u.clear(); self._frB.clear(); self._C = None; self._C_zlecony = None; self._B = None; self._i_panel = None
        for sc in self._sceny.values():
            sc.kadr = None
        try:
            self.dane = projections.wczytaj(h)
        except Exception:                                                # noqa: BLE001
            traceback.print_exc()
        if self.dane is None:
            self.lb_rz.setText('Stany 192D tej historii nie są jeszcze policzone: „Policz / wczytaj stany 192D” '
                               f'({h.K} przejść w przód na GPU, potem z dysku).')
            for lb in self._lb.values():
                lb.setText('Brak stanów 192D: sekcja „Rzuty z 192 wymiarów” → „Policz / wczytaj stany 192D”.')
            return
        self._po_wczytaniu()

    def _po_wczytaniu(self):
        d = self.dane
        self.lb_rz.setText(f'Stany 192D: {d.K} checkpointów. Zgodność z rzutem sfery P3: błąd {pl(float(d.zgodnosc_P3) * 100, 2)} % '
                           f'(float16). Wymiar efektywny chmury (współczynnik uczestnictwa) {pl(float(d.PR[0]), 1)} → '
                           f'{pl(float(d.PR[-1]), 1)}.')
        self._panel_wiernosc()
        self._rysuj_widoczne()

    def _licz(self):
        h = self._z.h
        if h is None or self._watek is not None:
            return
        self._watek = ProjectionWorker(h)
        self._watek.postep.connect(lambda i, n, o: (self.pasek_rz.setRange(0, n), self.pasek_rz.setValue(i), self.lb_rz.setText(o)))
        self._watek.gotowe.connect(self._policzone)
        self._watek.blad.connect(lambda s: (self.lb_rz.setText(f'Nie policzono: {s}'), self._koniec_liczenia()))
        self._watek.finished.connect(lambda: setattr(self, '_watek', None))
        self.bt_rz.setEnabled(False); self.bt_rz_stop.setEnabled(True)
        self.lb_rz.setText('Liczenie stanów 192D na GPU…')
        self._watek.start()

    def _koniec_liczenia(self):
        self.bt_rz.setEnabled(True); self.bt_rz_stop.setEnabled(False)

    def _przerwij(self):
        if self._watek is not None:
            self._watek.stop = True

    def _policzone(self, katalog, d):
        self._koniec_liczenia()
        if self._h is None or str(self._h.katalog) != katalog:
            return                                                       # w międzyczasie wczytano inną historię
        self.dane = d
        self._pam.clear(); self._u.clear(); self._frB.clear(); self._i_panel = None
        for sc in self._sceny.values():
            sc.kadr = None
        self._po_wczytaniu()

    # ------------------------------------------------------------------ czas, wybór, opcje
    def ustaw_czas(self, t):
        self.t = float(t)
        if self.dane is None:
            return
        self._rysuj_widoczne()

    def _rysuj_widoczne(self):
        for k, _, _ in OKIENKA:                                          # okienka w nieaktywnych kartach nie liczą (widac)
            if self._z.widac(k):
                self._rysuj(k)

    def _wybor(self):
        for k in [k for k in self._pam if k[0] in ('D1', 'Dr', 'zero', 'zero3', 'ruch', 'ruchj')]:
            del self._pam[k]
        self._B = None
        self._rysuj_widoczne()

    def _opcje(self, *_):
        for k in [k for k in self._pam if k[0] in ('D1', 'Dr', 'zero', 'zero3', 'ruch', 'ruchj', 'L', 'Lm')]:
            del self._pam[k]
        self._rysuj_widoczne()

    def pokaz(self, k):
        d = self._z.doki.get(k)
        if d is not None:
            d.show(); d.raise_()
            self._rysuj(k)

    def _indeksy(self):
        K = self.dane.K
        t = float(np.clip(self.t, 0, K - 1))
        i = int(np.floor(t)); j = min(i + 1, K - 1)
        return i, j, t - i, int(np.clip(round(t), 0, K - 1))

    def _wybrane(self):
        return self._z._wybrane_tok()

    def _maski(self):
        d = self.dane
        narz = np.asarray(d.narz, bool)
        wyb = np.zeros(hist.TOK, bool); wyb[self._wybrane()] = True
        pok = [np.ones(hist.TOK, bool), narz, wyb, narz | wyb][self.cb_kogo.currentIndex()]
        sl = [np.zeros(hist.TOK, bool), wyb, narz, narz | wyb, np.ones(hist.TOK, bool)][self.cb_slad.currentIndex()]
        return pok, wyb, sl & pok

    def _skala_k(self):
        R = np.asarray(self.dane.R, np.float64)
        return S_EKR / np.maximum(R if self.cb_skala.currentIndex() == 0 else np.full_like(R, R[-1]), 1e-12)

    def _ekran(self, C):
        """Współrzędne ekranu (K, 600, 3) dla współrzędnych C (K, 600, 3) w 192D: środek i skala według opcji (raz na opcje)."""
        klucz = ('ekr', id(C), self.cb_srodek.currentIndex(), self.cb_skala.currentIndex())
        if klucz not in self._pam:
            if len(self._pam) > 48:
                self._pam.clear()
            cen = 0.0
            if self.cb_srodek.currentIndex() == 1:
                m = ~np.asarray(self.dane.narz, bool)
                cen = C[:, m].mean(1, keepdims=True) if m.any() else 0.0
            self._pam[klucz] = (C, ((C - cen) * self._skala_k()[:, None, None]).astype(np.float32))
        return self._pam[klucz][1]

    def _wygladz(self, E, os_=0):
        """Średnia krocząca z ostatnich `Wygładzanie śladu` checkpointów (tylko przeszłość) wzdłuż osi czasu os_ (raz na tablicę)."""
        w = int(self.sp_wyg.value())
        if w <= 1:
            return E
        klucz = ('wyg', id(E), w, os_)
        if klucz not in self._pam:
            K = E.shape[os_]
            ksz = [1] * E.ndim; ksz[os_] = -1
            c0 = np.concatenate([np.zeros_like(np.take(E, [0], axis=os_), dtype=np.float64),
                                 np.cumsum(np.asarray(E, np.float64), axis=os_)], axis=os_)
            hi = np.arange(1, K + 1); lo = np.maximum(hi - w, 0)
            S = (np.take(c0, hi, axis=os_) - np.take(c0, lo, axis=os_)) / (hi - lo).reshape(ksz)
            self._pam[klucz] = (E, S.astype(np.float32))
        return self._pam[klucz][1]

    def _tor_wzgl(self, j):
        """Tor cząsteczki j (K, 192) względem wybranego środka (chmury albo tła)."""
        tor = self.dane.tor(j)
        return tor - self.dane.srodek_tla() if self.cb_srodek.currentIndex() == 1 else tor

    def _kursor(self, l):
        if self._blok or self._z.h is None:
            return
        self._z.ustaw_czas(float(l.value()))

    def _klik(self, k, x, y):
        P = self._ostatnie.get(k)
        if P is None:
            return
        P, pok = P
        px, gl_ = self._sceny[k].v.na_ekran(P)
        dd = np.hypot(px[:, 0] - x, px[:, 1] - y)
        dd[~pok | ~np.isfinite(dd) | ~np.isfinite(gl_)] = np.inf
        j = int(np.argmin(dd))
        if dd[j] < 25:
            self._z.sp_tok.setValue(j)

    # ------------------------------------------------------------------ rysowanie
    def _rysuj(self, k):
        if self._h is None or (self.dane is None and k != 'R0'):
            return
        try:
            if k == 'R0':
                self._rysuj_zdjecie()
            elif k == 'R10':
                self._rysuj_lda()
            elif k == 'R11':
                self._rysuj_miary()
            elif k == 'R12':
                self._rysuj_knn()
            elif k == 'R8':
                self._rysuj_panel()
            elif k == 'R4':
                self._rysuj_B()
            elif k == 'R5':
                self._rysuj_C()
            else:
                self._rysuj_globalny(k)
        except Exception as e:                                           # noqa: BLE001 (błąd jednego okienka nie zatrzymuje zakładki)
            traceback.print_exc()
            if k in self._lb:
                self._lb[k].setText(f'Błąd rysowania: {type(e).__name__}: {e}')

    def _baza_okienka(self, k):
        """(współrzędne położeń (K, 600, 3), sił (K, 600, 3), nazwy osi, indeks bazy w miernikach albo None, opis)."""
        d = self.dane
        if k == 'R1':
            if self.cb_A.currentIndex() == 1:
                return d.wsp(d.A_t, 'X'), d.wsp(d.A_t, 'F'), ['Δ narzędzie–tło (checkpoint)', 'reszta 1', 'reszta 2'], 2, 'A w każdym checkpoincie'
            return d.wsp(d.baza_A, 'X'), d.wsp(d.baza_A, 'F'), ['Δ narzędzie–tło', 'reszta chmury 1', 'reszta chmury 2'], 1, 'A wspólna'
        if k == 'R2':
            return d.wsp(d.baza_S, 'X'), d.wsp(d.baza_S, 'F'), ['siły 1', 'siły 2', 'siły 3'], 3, 'S'
        if k == 'R3':
            return d.wsp(d.baza_W, 'X'), d.wsp(d.baza_W, 'F'), ['wiązki 1', 'wiązki 2', 'wiązki 3'], 4, 'W'
        if k == 'R7':
            tr = projections.FANO[self.cb_F.currentIndex()]
            klucz = ('F7', tr)
            if klucz not in self._pam:
                self._pam[klucz] = (np.ascontiguousarray(d.wsp(d.baza_F7, 'X')[..., list(tr)]),
                                    np.ascontiguousarray(d.wsp(d.baza_F7, 'F')[..., list(tr)]))
            X3, F3 = self._pam[klucz]
            return X3, F3, [d.nazwy_F7[a] for a in tr], None, f'Fano {self.cb_F.currentIndex() + 1}'
        if k == 'R9':
            B, opis = self._baza_L()
            return d.wsp(B, 'X'), d.wsp(B, 'F'), ['LDA narzędzie–tło', 'oś 2', 'oś 3'], None, opis
        B, nazwy, b, opis = self._baza_D()
        return d.wsp(B, 'X'), d.wsp(B, 'F'), nazwy, b, opis

    def _grupa(self, i):
        """Grupa tokenów wykryta bez etykiet w checkpoincie i (TokenCut na h.Z[i], aktywacje MLP bloku 6), raz na checkpoint."""
        klucz = ('grupa', int(i))
        if klucz not in self._pam:
            self._pam[klucz] = projections.wykryj_grupe(np.asarray(self._h.Z[int(i)], np.float32))
        return self._pam[klucz]

    def _etykieta(self, i=None):
        """(narzędzie, mieszane) bool (600,) dla R9: z obrysu (udział pikseli ≥ próg) albo ze starej ramki GT bez `brzeg` skrajnych
        kolumn i wierszy; mieszane nie wchodzą ani do narzędzia, ani do tła."""
        d = self.dane
        if self.cb_L_etyk.currentIndex() == 2:
            G = self._grupa(self._indeksy()[3] if i is None else i)
            if G.sum() >= 2:
                return G, np.zeros(hist.TOK, bool)
        u = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
        if self.cb_L_etyk.currentIndex() == 0 and np.isfinite(u).all():
            narz = u >= float(self.sp_L_prog.value())
            if narz.sum() >= 2:
                return narz, (u > 0) & ~narz
        ramka = np.asarray(getattr(d, 'narz_ramka', d.narz), bool)
        rdz = projections.rdzen_narzedzia(ramka, int(self.sp_L_brzeg.value()))
        return rdz, ramka & ~rdz

    def _klucz_etyk(self):
        return (self.cb_L_etyk.currentIndex(), self.cb_L_os.currentIndex(), round(float(self.sp_L_prog.value()), 3), int(self.sp_L_brzeg.value()))

    def _etykieta_osi(self, kf):
        """Etykieta do dopasowania osi LDA: przy grupie wykrytej bez etykiet i „oś z etykiety” – obrys (udział ≥ próg),
        w pozostałych przypadkach ta sama co etykieta widoku."""
        if self.cb_L_etyk.currentIndex() == 2 and self.cb_L_os.currentIndex() == 0:
            u = np.asarray(getattr(self.dane, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
            if np.isfinite(u).all() and (u >= float(self.sp_L_prog.value())).sum() >= 2:
                narz = u >= float(self.sp_L_prog.value())
                return narz, (u > 0) & ~narz
        return self._etykieta(kf)

    def _opis_etyk(self, rdz, mies):
        d = self.dane
        if self.cb_L_etyk.currentIndex() == 2 and rdz.sum() >= 2 and not mies.any():
            u = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
            s_ = f'grupa wykryta bez etykiet (TokenCut, latent MLP), {int(rdz.sum())} tokenów'
            if np.isfinite(u).all():
                z = projections.zgodnosc_z_obrysem(rdz, u)
                fp = lambda v: f'{v:.3f}'.replace('.', ',')
                s_ += (f'; sprawdzenie z obrysem (nieużytym do wykrycia): Dice {pl(z["dice"])}, precyzja {pl(z["precyzja"])}, '
                       f'czułość {pl(z["czulosc"])}, p przesunięć grupy = {fp(z["p"])}; żółte = narzędzie pominięte')
            return s_
        u = np.asarray(getattr(d, 'udzial', [np.nan]), np.float32)
        obrys = self.cb_L_etyk.currentIndex() == 0 and np.isfinite(u).all()
        zrodlo = str(getattr(d, 'etykieta', 'obrys')) if obrys else 'ramka GT'
        if obrys and (u >= float(self.sp_L_prog.value())).sum() < 2:
            zrodlo = 'ramka GT (przy tym progu < 2 łaty narzędzia)'
        elif self.cb_L_etyk.currentIndex() == 0 and not obrys:
            zrodlo = 'ramka GT (brak obrysu dla tej klatki)'
        return f'{zrodlo}, {int(rdz.sum())} tokenów narzędzia, {int(mies.sum())} mieszanych (żółte, pominięte)'

    def _baza_L(self):
        """Baza R9 (LDA ze skurczem na puli checkpointów), raz na parametry i checkpoint dopasowania."""
        d = self.dane
        _, _, _, ir = self._indeksy()
        kf = ir if self.cb_L_kiedy.currentIndex() == 1 else d.K - 1
        N = int(self.sp_L_N.value())
        klucz = ('L', kf, N, round(float(self.sp_L_lam.value()), 3), self.cb_L_osie.currentIndex(), self._klucz_etyk())
        if klucz not in self._pam:
            ks = np.arange(max(0, kf - N + 1), kf + 1)
            if len(ks) > 200:
                ks = np.unique(np.concatenate([np.linspace(ks[0], ks[-1], 200).astype(int), [kf]]))   # pula: najwyżej 200 checkpointów
            Xs = np.asarray(d.X[ks], np.float64).reshape(-1, projections.D)
            rdz, mies = self._etykieta_osi(kf)
            keep = np.tile(~mies, len(ks))
            B, cosd = projections.baza_L(Xs[keep], np.tile(rdz, len(ks))[keep], float(self.sp_L_lam.value()), self.cb_L_osie.currentIndex())
            self._pam[klucz] = (B.astype(np.float32), f'L: LDA λ={pl(float(self.sp_L_lam.value()))}, pula {len(ks)} ckpt do {kf + 1}, '
                                                       f'cos(w, Δ) = {pl(cosd)}')
        return self._pam[klucz]

    def _baza_D(self):
        d = self.dane
        o = self.cb_D.currentIndex()
        if o == 0:
            return d.baza_D, ['ruch 1', 'ruch 2', 'ruch 3'], 5, 'D: PCA przemieszczeń wszystkich'
        if o == 3:
            return d.baza_A, ['Δ narzędzie–tło', 'reszta chmury 1', 'reszta chmury 2'], 1, 'A wspólna'
        if o == 1:
            pok, _, _ = self._maski()
            idx = np.flatnonzero(pok)
            klucz = ('D1', tuple(idx.tolist()), self.cb_srodek.currentIndex())
            if klucz not in self._pam:
                S = np.zeros((projections.D, projections.D))
                tlo = d.srodek_tla() if self.cb_srodek.currentIndex() == 1 else None
                for a in range(0, d.K, 64):
                    blok = np.asarray(d.X[a:min(a + 65, d.K)], np.float64)[:, idx]
                    if tlo is not None:
                        blok = blok - tlo[a:a + len(blok)][:, None]
                    dd = np.diff(blok, axis=0).reshape(-1, projections.D)
                    S += dd.T @ dd
                self._pam[klucz] = projections.glowne(S, 3)
            return self._pam[klucz], ['ruch pokazanych 1', 'ruch pokazanych 2', 'ruch pokazanych 3'], None, 'D: PCA przemieszczeń pokazanych'
        j = self._wybrane()[0]
        klucz = ('Dr', j, self.cb_srodek.currentIndex())
        if klucz not in self._pam:
            self._pam[klucz] = projections.rama_toru(self._tor_wzgl(j))
        return self._pam[klucz], [f'start → koniec ({j})', 'odchylenie 1', 'odchylenie 2'], None, f'D: rama cząsteczki {j}'

    def _rysuj_globalny(self, k):
        """Okienka R1–R3, R6, R7: historia w bazie okienka na karcie (raz na bazę i opcje), w klatce tylko czas, maska i napisy."""
        d, sc = self.dane, self._sceny[k]
        Cx, Cf, nazwy, b, opis_bazy = self._baza_okienka(k)
        E = self._ekran(Cx)
        i, j, u, ir = self._indeksy()
        P = E[i] * (1 - u) + E[j] * u                                    # tylko kamera, klik i kadr: cząsteczki rysuje karta
        pok, wyb, sl = self._maski()
        wybrane = self._wybrane()
        s = self._skala_k()
        kg = (id(Cx), id(Cf), self.cb_srodek.currentIndex(), self.cb_skala.currentIndex(), int(self.sp_wyg.value()))
        if kg != sc._klucz_g:
            rho = np.clip(np.linalg.norm(Cf, axis=2) / np.maximum(d.FN, 1e-30), 0, 1)   # widoczna część długości siły
            sc.wgraj(kg, E, self._wygladz(E), Cf * s[:, None, None], rho)
        o_ = self.cb_strz.currentIndex()
        M = np.zeros((hist.TOK, 4), np.float32)
        M[:, 0] = np.asarray(self._etykieta()[0] if k == 'R9' else d.narz, bool)
        M[~pok & ~wyb, 0] = 2.0                                           # ukryte; wybrane widać zawsze
        M[wyb, 1] = 1.0
        if k == 'R9':
            M[self._etykieta()[1], 1] = 1.0                                  # tokeny mieszane (albo pominięty brzeg ramki) na żółto
            u_ = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
            if self.cb_L_etyk.currentIndex() == 2 and np.isfinite(u_).all():
                M[(u_ >= 0.25) & ~self._etykieta()[0], 1] = 1.0              # narzędzie z obrysu pominięte przez grupę: na żółto
        if wybrane:
            M[wybrane[0], 1] = 2.0
        M[:, 2] = sl
        M[:, 3] = (pok if o_ == 1 else wyb) if o_ else False
        sc.maska(M)
        L = int(self.sp_slad.value())
        sc.g.opcje.update(czastki=True, tlo=True, rozm=float(self.sp_rozm.value()), L=L, od0=L == 0, strzalki=o_ > 0,
                          strz=float(self.sp_strz.value()), rho=self.ch_rho.isChecked())
        if k == 'R3' and self.ch_W.isChecked() and wybrane:
            jt, sk = wybrane[0], float(s[ir] * self.sp_strz.value())
            sc.lokalne(('W', ir, jt, round(float(self.sp_B_prog.value()), 3), sk, self.ch_rho.isChecked()),
                       lambda: self._wiazki_lokalne(d.baza_W, ir, jt, sk), kotwica=jt)
        else:
            sc.lokalne(None, None)
        sc.g.ustaw_czas(self.t)
        self._ostatnie[k] = (P, pok | wyb)
        if k == 'R6':
            self._tory_zerowe(sc, E, i, u)
        else:
            sc.odcinki('zero', None, None)
        sc.podpisy(sc.osie(nazwy if self.ch_osie.isChecked() else None))
        # kadr z całej historii (co ~50. checkpoint): kamera mieści każdy checkpoint i nie skacze w czasie
        sc.kadruj(E[::max(1, d.K // 50)][:, pok | wyb].reshape(-1, 3), (k, id(Cx), self.cb_skala.currentIndex(), self.cb_srodek.currentIndex()))
        if self.ch_kamera.isChecked() and wybrane:
            sc.srodek(P[wybrane[0]])
        self._lb[k].setText(self._opis(k, b, ir, opis_bazy, pok))

    def _rama_B(self, ir, jt):
        """Rama siły i sieć silna cząsteczki jt w checkpoincie ir (raz na checkpoint, cząsteczkę i próg; kilka ostatnich)."""
        klucz = (ir, jt, round(float(self.sp_B_prog.value()), 3))
        w = self._frB.get(klucz)
        if w is None:
            u = self._wklady(ir, jt)
            silne, reszta = projections.siec_silna(u, klucz[2])
            w = self._frB[klucz] = dict(u=u, fr=projections.rama_sily(u), silne=silne, reszta=reszta)
            while len(self._frB) > 6:
                self._frB.pop(next(iter(self._frB)))
        return w

    def _kolory_wiazek(self, U, W3, cos_):
        """Kolor wiązki = znak cos(u_c, F̂) w 192D (pomarańcz pcha wzdłuż siły, błękit przeciw), krycie = ρ = |Pu|/|u|."""
        kk = np.empty((len(U), 4), np.float32)
        kk[:, :3] = np.where(np.asarray(cos_)[:, None] > 0, KOL_ZGODNA, KOL_PRZECIW)
        rho = np.linalg.norm(W3, axis=1) / np.maximum(np.linalg.norm(U, axis=1), 1e-30)
        kk[:, 3] = 0.15 + 0.85 * np.clip(rho, 0, 1) if self.ch_rho.isChecked() else 0.85
        return kk

    def _wiazki_lokalne(self, B, ir, jt, s):
        """R3: wiązki cząsteczki jt w bazie B jako elementy lokalne zakotwiczone w cząsteczce: sieć silna i gruba szara reszta."""
        w = self._rama_B(ir, jt)
        u, fr, silne, reszta = w['u'], w['fr'], w['silne'], w['reszta']
        W3 = u[silne] @ B
        el = LocalSections().strzalki(np.zeros(3), W3 * s, self._kolory_wiazek(u[silne], W3, fr['cos'][silne]))
        return el.strzalki(np.zeros(3), (reszta @ B * s)[None], (0.62, 0.62, 0.62, 0.9), gruba=True)

    def _wklady(self, ir, jt):
        """Wkłady 768 wiązek cząsteczki jt w checkpoincie ir (768, 192); pamięć na kilka ostatnich (każdy ma 1,2 MB)."""
        klucz = (ir, jt)
        u = self._u.get(klucz)
        if u is None:
            z = np.asarray(self._h.Z[ir], np.float64)
            u = self._u[klucz] = projections.wklady(z[jt], z.mean(0), self.dane.V[ir])
            while len(self._u) > 6:
                self._u.pop(next(iter(self._u)))
        return u

    def _zerowe(self, j):
        """Tory zerowe cząsteczki j (n, K, 192) względem wybranego środka (raz na wybór)."""
        klucz = ('zero', j, self.sp_D_zero.value(), self.cb_srodek.currentIndex())
        if klucz not in self._pam:
            self._pam[klucz] = projections.tory_zerowe(self._tor_wzgl(j), self.sp_D_zero.value(), ziarno=j)
        return self._pam[klucz]

    def _tory_zerowe(self, sc, E, i, u):
        """R6: szare tory zerowe pierwszej wybranej cząsteczki do bieżącego checkpointu."""
        wybrane = self._wybrane()
        if not self.ch_D_zero.isChecked() or not wybrane:
            sc.odcinki('zero', None, None)
            return
        j = wybrane[0]
        o = self.cb_D.currentIndex()
        B = None if o == 2 else self._baza_D()[0]
        klucz = ('zero3', j, self.sp_D_zero.value(), self.cb_srodek.currentIndex(), o, None if B is None else B.tobytes())
        if klucz not in self._pam:
            Z0 = self._zerowe(j)                                            # (n, K, 192)
            if B is None:                                                   # każdy tor zerowy we własnej ramie (ta sama przewaga)
                Z3 = np.stack([(z_ - z_[0]) @ projections.rama_toru(z_) for z_ in Z0])
            else:
                Z3 = Z0 @ B
            self._pam[klucz] = Z3.astype(np.float32)
        Z3 = self._wygladz(self._pam[klucz], os_=1)                         # to samo wygładzanie co ślady cząsteczek
        s = self._skala_k()
        T3 = (E[0, j] + Z3 * s[None, :, None]) if B is None else Z3 * s[None, :, None]
        T3 = T3[:, :i + 2]
        if T3.shape[1] > i + 1:                                             # ostatni odcinek do chwili t
            T3[:, i + 1] = T3[:, i] * (1 - u) + T3[:, i + 1] * u
        if T3.shape[1] < 2:
            sc.odcinki('zero', None, None)
            return
        seg = np.stack([T3[:, :-1], T3[:, 1:]], 2).reshape(-1, 3)
        sc.odcinki('zero', seg, np.tile([0.62, 0.62, 0.62, 0.45], (len(seg), 1)))

    def _opis(self, k, b, i, opis_bazy, pok):
        d = self.dane
        kr = int(d.kroki[i])
        if b is not None:
            s1 = (f'{opis_bazy}, checkpoint {i + 1}/{d.K} (krok {kr}). Widać: rozrzut chmury {proc(d.WX[i, b])} '
                  f'(najlepsze 3D {proc(d.own3[i])}), energia sił {proc(d.WF[i, b])}, energia wiązek {proc(d.WW[i, b])}, '
                  f'długość Δ {proc(d.WS[i, b])}, ruch od poprzedniego {proc(d.WD[i, b])}.')
        elif k == 'R7':
            tr = projections.FANO[self.cb_F.currentIndex()]
            os_ = '; '.join(f'{d.nazwy_F7[a]}: rozrzut {proc(d.E7[i, 0, a])}, siły {proc(d.E7[i, 1, a])}, wiązki {proc(d.E7[i, 2, a])}'
                            for a in tr)
            s1 = f'{opis_bazy}, checkpoint {i + 1}/{d.K} (krok {kr}). Osie: {os_}.'
        else:
            s1 = f'{opis_bazy}, checkpoint {i + 1}/{d.K} (krok {kr}).'
        if k == 'R6':
            s1 += ' ' + self._opis_ruchu(pok)
        if k == 'R9':
            s1 += ' ' + self._opis_separacji(i)
        sep = ''
        if np.isfinite(d.sep[i]):
            sep = (f' Separacja w 192D: |Δ| = {pl(float(d.sep[i]))} R; losowe etykiety {pl(float(d.sep0[i, 0]))} R '
                   f'(95 %: {pl(float(d.sep0[i, 1]))} R).')
        return s1 + ' Losowe 3D: 1,6 % energii, 12,5 % długości.' + sep

    def _opis_separacji_podst(self, i, B):
        """Podstawowe miary R9 (bez zera z refitem i walidacji blokowanej): szybkie, także podczas odtwarzania."""
        d = self.dane
        rdz, mies = self._etykieta(i); keep = ~mies
        X = np.asarray(d.X[i], np.float64)[keep]; m = rdz[keep]
        kk = projections.k_sasiadow(m.sum())
        m3, z = projections.separacja_gpu(X @ B, m, int(self.sp_L_perm.value()), k=kk)
        m192, _ = projections.separacja_gpu(X, m, 0, k=kk)
        f = lambda n_: f'{pl(m3[n_])}' + ('' if z is None else f' (losowe etykiety {pl(z[n_])})')
        return (f'Etykieta: {self._opis_etyk(rdz, mies)}. Separacja w tym rzucie: Fisher {f("fisher")}, sylwetka {f("sil")}, '
                f'trafienia {kk}NN {f("hit")}, margines {f("marg")} odstępu tła; '
                f'w 192D: Fisher {pl(m192["fisher"])}, sylwetka {pl(m192["sil"])}, kNN {pl(m192["hit"])}.')

    def _opis_separacji(self, i):
        """R9: miary separacji rdzenia narzędzia od tła w tym rzucie i w 192D, z zerem dla losowych etykiet (raz na checkpoint)."""
        d = self.dane
        B, _ = self._baza_L()
        ciezkie = (self.ch_L_refit.isChecked() and self.sp_L_perm.value() > 0) or self.ch_L_cv.isChecked() or self.ch_L_przestrz.isChecked()
        gra = self._z.bt_play.isChecked()
        if ciezkie and gra and ('Lm', i, 'ciezkie') not in self._pam:
            klucz_l = ('Lm', i, B.tobytes(), self._klucz_etyk(), int(self.sp_L_perm.value()), False, False, False, 0, 0, 0, 0)
            if klucz_l not in self._pam:
                self._pam[klucz_l] = self._opis_separacji_podst(i, B)
            return self._pam[klucz_l] + ' Zero przestrzenne, zero z refitem i walidacja blokowana liczą się na pauzie.'
        klucz = ('Lm', i, B.tobytes(), self._klucz_etyk(), int(self.sp_L_perm.value()), self.ch_L_refit.isChecked(),
                 self.ch_L_cv.isChecked(), self.ch_L_przestrz.isChecked(), int(self.sp_L_bl.value()), int(self.sp_L_N.value()), round(float(self.sp_L_lam.value()), 3),
                 self.cb_L_osie.currentIndex())
        if klucz not in self._pam:
            rdz, mies = self._etykieta(i); keep = ~mies
            X = np.asarray(d.X[i], np.float64)[keep]; m = rdz[keep]
            kk = projections.k_sasiadow(m.sum())
            tekst = self._opis_separacji_podst(i, B)
            lam, osie = float(self.sp_L_lam.value()), self.cb_L_osie.currentIndex()
            if self.ch_L_przestrz.isChecked():
                wyk = rdz | mies
                u = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
                if np.isfinite(u).all():
                    wyk |= u > 0                                          # żadnego piksela narzędzia w położeniach tła
                pr, zp, n = projections.zero_przestrzenne_gpu(np.asarray(d.X[i], np.float32), rdz, wyk, keep, lam, osie, kk)
                fp = lambda v: f'{v:.3f}'.replace('.', ',')
                if zp:
                    tekst += (f' Zero przestrzenne ({n} położeń tej maski w tle, LDA od nowa na tym checkpoincie): narzędzie Fisher '
                              f'{pl(pr["fisher"])} (95 % tła {pl(zp["fisher"][0])}, p = {fp(zp["fisher"][1])}), sylwetka {pl(pr["sil"])} '
                              f'({pl(zp["sil"][0])}, p = {fp(zp["sil"][1])}), {kk}NN {pl(pr["hit"])} ({pl(zp["hit"][0])}, p = {fp(zp["hit"][1])}).')
                else:
                    tekst += f' Zero przestrzenne: za mało położeń maski w tle ({n}).'
            if self.ch_L_refit.isChecked() and self.sp_L_perm.value() > 0:
                zr = projections.zero_z_refitem_gpu(X, m, lam, osie, int(self.sp_L_perm.value()), k=kk)
                tekst += (f' Zero z ponownym dopasowaniem osi ({int(self.sp_L_perm.value())} perm.): sylwetka {pl(zr["sil"])}, '
                          f'kNN {pl(zr["hit"])}, margines {pl(zr["marg"])}, Fisher {pl(zr["fisher"])}.')
            if self.ch_L_cv.isChecked():
                ks = np.arange(max(0, i - int(self.sp_L_N.value()) + 1), i + 1)
                if len(ks) >= 4:
                    cv, _ = projections.cv_blokowane_gpu(np.asarray(d.X[ks], np.float32)[:, keep], m, lam, osie, int(self.sp_L_bl.value()), k=kk)
                    tekst += (f' Walidacja blokowana po checkpointach ({int(self.sp_L_bl.value())} bloków z puli {len(ks)}): sylwetka '
                              f'{pl(cv["sil"])}, kNN {pl(cv["hit"])}, margines {pl(cv["marg"])}, Fisher {pl(cv["fisher"])}.')
                else:
                    tekst += ' Walidacja blokowana: pula za krótka (zwiększ „Pulę dopasowania”).'
            self._pam[klucz] = tekst
        return self._pam[klucz]

    def _opis_ruchu(self, pok):
        """R6: ile ruchu pokazanych cząsteczek widać w tym rzucie i porównanie toru wybranej z torami zerowymi (192D)."""
        d = self.dane
        B, _, _, _ = self._baza_D()
        idx = np.flatnonzero(pok)
        klucz = ('ruch', tuple(idx.tolist()), B.tobytes(), self.cb_srodek.currentIndex())
        if klucz not in self._pam:
            if len(idx) > 120:
                idx = idx[np.linspace(0, len(idx) - 1, 120).astype(int)]    # próbka: miara, nie rysunek
            tory = np.asarray(d.X[:, idx], np.float64)                      # (K, n, 192), jeden odczyt
            if self.cb_srodek.currentIndex() == 1:
                tory = tory - d.srodek_tla()[:, None]
            self._pam[klucz] = projections.udzial_ruchu(tory.transpose(1, 0, 2), B)
        en, dl = self._pam[klucz]
        s = f'Ruch pokazanych w tym rzucie: {proc(en)} energii przemieszczeń, {proc(dl)} długości drogi.'
        wybrane = self._wybrane()
        if wybrane:
            j = wybrane[0]
            klucz = ('ruchj', j, self.sp_D_zero.value(), self.cb_srodek.currentIndex(), self.ch_D_zero.isChecked())
            if klucz not in self._pam:
                tor = self._tor_wzgl(j)
                R = float(d.R[-1])
                droga = float(np.linalg.norm(np.diff(tor, axis=0), axis=1).sum()) / R
                netto = float(np.linalg.norm(tor[-1] - tor[0])) / R
                p1, p3 = projections.udzialy_pca(tor)
                t = (f' Cząsteczka {j}: droga {pl(droga)} R, przesunięcie netto {pl(netto)} R; kąt skrętu (mediana cos) '
                     f'{pl(projections.kat_skretu(tor))}; PCA własnego toru PC1 {proc(p1)}, PC1–3 {proc(p3)}')
                if self.ch_D_zero.isChecked():
                    Z0 = self._zerowe(j)
                    u0 = np.array([projections.udzialy_pca(z_) for z_ in Z0])
                    t += (f'; tory zerowe: PC1 {proc(u0[:, 0].mean())}, PC1–3 {proc(u0[:, 1].mean())}, kąt skrętu '
                          f'{pl(float(np.median([projections.kat_skretu(z_) for z_ in Z0])))}')
                self._pam[klucz] = t + '.'
            s += self._pam[klucz]
        return s

    # ------------------------------------------------------------------ R4 B: rama siły
    def _rysuj_B(self):
        d, sc = self.dane, self._sceny['R4']
        _, _, _, ir = self._indeksy()
        wybrane = self._wybrane()
        if not wybrane:
            return
        j = wybrane[0]
        wB = self._rama_B(ir, j)
        u, fr, silne, reszta = wB['u'], wB['fr'], wB['silne'], wB['reszta']
        B = fr['B']
        u3, r3, F3 = u[silne] @ B, reszta @ B, fr['F'] @ B
        L = max(float(np.linalg.norm(u3, axis=1).max()), float(np.linalg.norm(F3)), float(np.linalg.norm(r3)), 1e-30)
        s = 1.2 * S_EKR * self.sp_B_skala.value() / L

        def budowa():
            el = LocalSections().strzalki(np.zeros(3), u3 * s, self._kolory_wiazek(u[silne], u3, fr['cos'][silne]))
            el.strzalki(np.zeros(3), (F3 * s)[None], KOL_SILA + (1.0,), gruba=True)
            el.strzalki(np.zeros(3), (r3 * s)[None], (0.62, 0.62, 0.62, 0.9), gruba=True)
            return el.kropki(np.zeros((1, 3)), KOL_WYBR + (1.0,))

        sc.g.opcje.update(czastki=False, strzalki=False, rozm=9.0)
        sc.lokalne(('B', ir, j, round(float(self.sp_B_prog.value()), 3), float(self.sp_B_skala.value()), self.ch_rho.isChecked()),
                   budowa)
        sc.odcinki('zero', None, None)
        nazwy = ['F̂ (100 % siły)', 'poprzeczna 1', 'poprzeczna 2'] if self.ch_osie.isChecked() else None
        napisy = [((F3 * s) * 1.05, 'siła netto', (255, 190, 110, 230)), ((r3 * s) * 1.05, f'reszta {hist.NK - len(silne)} wiązek',
                                                                            (200, 200, 200, 220))]
        sc.podpisy(napisy + sc.osie(nazwy))
        sc.kadruj(np.concatenate([u3 * s, (F3 * s)[None], (r3 * s)[None]]), ('R4', 'stala'), kwantyl=100, zapas=2.0)
        if 'pary' not in wB:
            wB['pary'] = tuple(projections.pary_przeciwne(u[silne], B_) if len(silne) <= 400 else np.nan for B_ in (None, B, d.baza_P3))
        p192, p3, pP3 = wB['pary']
        self._B = dict(ir=ir, j=j, u=u, fr=fr)
        self._lb['R4'].setText(
            f'Cząsteczka {j}, checkpoint {ir + 1}/{d.K} (krok {int(d.kroki[ir])}): |F| = {pl(fr["nF"] / float(d.R[ir]), 3)} R; '
            f'oś 1 = F̂ (100 % siły), osie 2–3 niosą {proc(fr["poprz"])} energii składowych poprzecznych (zero: 2/191 = 1,0 %). '
            f'Sieć silna: {len(silne)} wiązek = {proc(self.sp_B_prog.value())} energii, reszta {hist.NK - len(silne)} jako szara strzałka '
            f'(suma = F dokładnie). Pomarańczowe pchają wzdłuż F̂, niebieskie przeciw (cos w 192D). Pary ≥ 150° w sieci silnej: '
            f'192D {proc(p192)}, ta rama {proc(p3)}, P3 {proc(pP3)} (losowe 3D 6,7 %). Rama lokalna: strzałek różnych cząsteczek nie porównywać.')

    # ------------------------------------------------------------------ R5 C: rama Schura
    def _klucz_C(self):
        _, _, _, ir = self._indeksy()
        wybrane = self._wybrane()
        return (str(self._h.katalog), ir, wybrane[0] if wybrane else 0, self.cb_C_para.currentIndex(), self.cb_C_rzecz.currentIndex(),
                int(self.sp_C_G.value()), round(float(self.sp_C_zas.value()), 3))

    def _zlec_C(self):
        if not self.ch_C.isChecked() or self._h is None:
            return
        k = self._klucz_C()
        if k == self._C_zlecony:
            return
        self._C_zlecony = k
        h = self._h
        self._watek_C.zlec(k, dict(wagi=str(projections._katalog_wag(h)), krok=int(h.kroki[k[1]]), rgb=h.rgb, tok=k[2], para=k[3], rzecz=k[4],
                                   G=k[5], zasieg=k[6], krok0=int(h.kroki[0])))

    def _C_gotowe(self, klucz, w):
        if 'blad_watku' in w:
            self._lb['R5'].setText(f'Rama C: {w["blad_watku"]}')
            return
        if self._h is None or klucz[0] != str(self._h.katalog):
            return
        self._C = (klucz, w)
        if self._z.widac('R5'):
            self._rysuj_C(zlec=False)
        if self._z.widac('R8'):
            self._panel_widmo()

    def _rysuj_C(self, zlec=True):
        if zlec:
            self._zlec_C()
        sc = self._sceny['R5']
        sc.g.opcje.update(czastki=False, strzalki=False, rozm=9.0)
        if self._C is None:
            sc.lokalne(None, None)
            self._lb['R5'].setText('Rama C: liczenie (przejście w przód, jakobian 192 × 192, siatka pola)…' if self.ch_C.isChecked()
                                   else 'Rama C: włącz liczenie w sekcji „R5 C: rama Schura”.')
            return
        klucz, w = self._C
        G = int(round(len(w['siatka']) ** (1 / 3)))
        g = w['siatka'] * (S_EKR / max(w['r'], 1e-30))
        lin = self.cb_C_pole.currentIndex() == 1 and w.get('liniowe') is not None
        pole = w['liniowe'] if lin else w['pole']
        nrm = np.linalg.norm(pole, axis=1)
        dl = 0.9 * (2 * S_EKR / max(G - 1, 1)) * self.sp_C_strz.value() / max(float(nrm.max()), 1e-30)
        gn = np.linalg.norm(g, axis=1)
        rad = np.clip(np.where((nrm > 1e-30) & (gn > 1e-30), (pole * g).sum(1) / np.maximum(nrm * gn, 1e-30), 0.0), -1, 1)[:, None]
        styczny, wyplyw, wplyw = np.array([0.78, 0.62, 0.98]), np.array([1.0, 0.38, 0.3]), np.array([0.32, 0.62, 1.0])
        kk = np.c_[np.where(rad > 0, styczny + (wyplyw - styczny) * rad, styczny + (wplyw - styczny) * (-rad)), 0.4 + 0.55 * np.abs(rad)]
        napisy = []
        o_ = None
        if w['os'] is not None:                                             # kierunek własny rzeczywisty
            lr = float(w['lam_r'])
            kr = (1.0, 0.35, 0.3, 0.95) if lr > 0 else (0.35, 0.65, 1.0, 0.95)
            o_ = np.asarray(w['os']) * 1.3 * S_EKR
            napisy.append((o_ * 1.05, f'λ = {pl(lr, 4)} ({"wypycha" if lr > 0 else "ściąga"})', (230, 230, 230, 230)))
        pk = None
        if w['plaszczyzna'] is not None:                                    # płaszczyzna obrotu (para zespolona), jaśniej = dokąd
            th = np.linspace(0, 2 * np.pi, 97)
            th = th if w['kierunek'] >= 0 else th[::-1]
            pk = 0.85 * S_EKR * (np.cos(th)[:, None] * w['plaszczyzna'][:, 0] + np.sin(th)[:, None] * w['plaszczyzna'][:, 1])
            lp = w['lam_p']
            napisy.append((pk[len(pk) // 4] * 1.12, f'λ = {pl(lp.real, 4)} ± {pl(abs(lp.imag), 4)} i (obrót, jaśniej = dokąd)',
                           (215, 170, 255, 230)))
        QF = np.asarray(w['QF'])
        v_ = None
        if np.linalg.norm(QF) > 1e-30:                                      # skok cząsteczki: kierunek w ramie
            v_ = QF / np.linalg.norm(QF) * 1.4 * S_EKR
            napisy.append((v_ * 1.04, f'skok: {proc(w["skok_w_ramie"])} długości w ramie', (255, 190, 110, 230)))

        def budowa():
            el = LocalSections().strzalki(g, pole * dl, kk)
            if o_ is not None:
                el.odcinki(-o_, o_, kr, gruba=True)
            if pk is not None:
                al = np.linspace(0.15, 0.95, len(pk) - 1)
                el.odcinki(pk[:-1], pk[1:], np.c_[np.tile([0.78, 0.45, 1.0], (len(al), 1)), al], gruba=True)
            if v_ is not None:
                el.strzalki(np.zeros(3), v_[None], KOL_SILA + (1.0,), gruba=True)
            return el.kropki(np.zeros((1, 3)), KOL_WYBR + (1.0,))

        sc.lokalne((klucz, lin, round(float(self.sp_C_strz.value()), 3)), budowa)
        sc.odcinki('zero', None, None)
        osie = sc.osie(['oś ramy 1', 'oś ramy 2', 'oś ramy 3'] if self.ch_osie.isChecked() else None, dl=1.1 * S_EKR)
        sc.podpisy(napisy + osie)
        sc.kadruj(g, ('R5', 'stala'), kwantyl=100, zapas=2.0)
        wid, w0 = w['widmo'], w['widmo0']
        teraz = klucz == self._klucz_C()
        porown = '' if w0 is None else f' (w pierwszym checkpoincie {proc(w0["zesp"])}, Re λ > 0: {w0["dodatnie"]})'
        self._lb['R5'].setText(
            f'Cząsteczka {klucz[2]}, checkpoint {klucz[1] + 1} (krok {w["krok"]})' + ('' if teraz else ' — liczę bieżący…') +
            f'. Rama = płaszczyzna pary zespolonej i kierunek rzeczywisty: niezmiennicza dla J (resztka {w["reszta"]:.1e}); '
            f'pole dokładne poza ramą: mediana {proc(float(np.median(w["poza"])))}. Widmo J: {proc(wid["zesp"])} zespolonych '
            f'(losowa macierz ≈ {proc(projections.GINIBRE_ZESPOLONE)}){porown}, Re λ > 0: {wid["dodatnie"]}, Tr J/192 = {pl(wid["div"], 4)}, '
            f'odchylenie od normalności {pl(wid["henrici"], 2)}. Tokeny nie są punktami stałymi (F ≠ 0): to rama lokalnej deformacji, '
            f'pole = zmiana siły względem skoku samej cząsteczki; czerwone wypływa, niebieskie wpływa, fiolet krąży.')

    # ------------------------------------------------------------------ R8: bez rzutu
    def _panel_wiernosc(self):
        if self.dane is None or not hasattr(self, 'p_w'):
            return
        d, h = self.dane, self._h
        m = self.cb_P.currentIndex()
        Y = (d.WX, d.WF, d.WW, d.WS, d.WD)[m]
        p = self.p_w
        for it in list(p.listDataItems()):
            p.removeItem(it)
        if p.legend is not None:
            p.legend.clear()
        x = np.arange(d.K)
        for b, nazwa in enumerate(projections.BAZY):
            styl = Qt.PenStyle.DashLine if nazwa == 'A_t' else Qt.PenStyle.SolidLine
            p.plot(x, Y[:, b], pen=pg.mkPen(KOL_BAZ[nazwa], width=1.4, style=styl), name=nazwa, connect='finite')
        zero = projections.LOS_DLUGOSC if m == 3 else projections.LOS_ENERGIA
        p.plot(x, np.full(d.K, zero), pen=pg.mkPen('#888888', width=1, style=Qt.PenStyle.DotLine), name='losowe 3D')
        if m == 0:
            p.plot(x, d.own3, pen=pg.mkPen('#ffd166', width=1, style=Qt.PenStyle.DotLine), name='najlepsze 3D (PCA chmury)')
        p.setYRange(0, 1); p.setXRange(0, d.K - 1)
        q = self.p_sep
        for it in list(q.listDataItems()):
            q.removeItem(it)
        q.plot(x, d.sep, pen=pg.mkPen('#ffb000', width=1.5), connect='finite')
        q.plot(x, d.sep0[:, 0], pen=pg.mkPen('#888888', width=1), connect='finite')
        q.plot(x, d.sep0[:, 1], pen=pg.mkPen('#888888', width=1, style=Qt.PenStyle.DashLine), connect='finite')
        q.setXRange(0, d.K - 1)
        EP = hist.NA_EPOKE
        kr = np.asarray(d.kroki)
        tk = [k for k in (0, 1000, EP * 5, EP * 10, EP * 20, EP * 40, EP * 60, EP * 100, EP * 150, EP * 200)   # bez 300 i e1: nachodzą
              if kr[0] <= k <= kr[-1]]
        nazwy = lambda k: {0: '0', 300: '300', 1000: '1k'}.get(k, f'e{int(round(k / EP))}')
        ticks = [[(float(np.interp(k, kr, np.arange(d.K))), nazwy(k)) for k in tk]]
        for p_ in (p, q):
            p_.getAxis('bottom').setTicks(ticks)
        self._i_panel = None

    def _rysuj_panel(self):
        d = self.dane
        _, _, _, ir = self._indeksy()
        self._blok = True
        for p in (self.p_w, self.p_sep):
            p.kursor.setValue(self.t)
        self._blok = False
        wybrane = self._wybrane()
        klucz = (ir, wybrane[0] if wybrane else None)
        if klucz == self._i_panel:
            return
        self._i_panel = klucz
        X = np.asarray(d.X[ir], np.float64)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
        Cm = Xn @ Xn.T
        narz = np.asarray(d.narz, bool)
        iu = np.triu_indices(hist.TOK, 1)
        grupy = (('narzędzie–narzędzie', narz[iu[0]] & narz[iu[1]], KOL_NARZ),
                 ('narzędzie–tło', narz[iu[0]] ^ narz[iu[1]], (1.0, 0.82, 0.4)),
                 ('tło–tło', ~narz[iu[0]] & ~narz[iu[1]], KOL_TLO))
        p = self.p_cos
        for it in list(p.listDataItems()):
            p.removeItem(it)
        if p.legend is not None:
            p.legend.clear()
        cv = Cm[iu]
        for nazwa, m, kol in grupy:
            if m.any():
                y, e = np.histogram(cv[m], 60, (-1, 1), density=True)
                p.plot(e, y, stepMode='center', pen=pg.mkPen(tuple(int(255 * c) for c in kol), width=1.4), name=nazwa)
        if wybrane:
            u = self._wklady(ir, wybrane[0])
            top = np.argsort(-np.linalg.norm(u, axis=1))[:100]
            q = self.p_wz
            for it in list(q.listDataItems()):
                q.removeItem(it)
            if q.legend is not None:
                q.legend.clear()
            iu2 = np.triu_indices(len(top), 1)
            ymax = 0.0
            for nazwa, v, kol in (('192D', u[top], '#ffd166'), ('rzut P3', u[top] @ d.baza_P3, '#e8e8e8')):
                vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)
                c = (vn @ vn.T)[iu2]
                y, e = np.histogram(c, 50, (-1, 1), density=True)
                ymax = max(ymax, float(y.max()))
                q.plot(e, y, stepMode='center', pen=pg.mkPen(kol, width=1.4), name=nazwa)
            self.tx_wz.setText(f'100 najsilniejszych wiązek cząsteczki {wybrane[0]}:\npary ≥ 150° w 192D '
                               f'{proc(projections.pary_przeciwne(u[top]))}, w P3 {proc(projections.pary_przeciwne(u[top], d.baza_P3))} '
                               f'(losowe 3D {proc(projections.LOS_PRZECIWNE_3D)})')
            self.tx_wz.setPos(-1, ymax * 1.15)
        if self.ch_C.isChecked():
            self._zlec_C()
        self._panel_widmo()

    def _panel_widmo(self):
        p = self.p_J
        for it in list(p.listDataItems()):
            p.removeItem(it)
        if self._C is None:
            self.tx_J.setText('widmo pojawi się po policzeniu ramy C (R5)')
            return
        klucz, w = self._C
        lam = w['widmo']['lam']
        p.plot(lam.real, lam.imag, pen=None, symbol='o', symbolSize=4, symbolBrush='#c264ff', symbolPen=None)
        g = w['ginibre']
        th = np.linspace(0, 2 * np.pi, 200)
        p.plot(g['srodek'] + g['promien'] * np.cos(th), g['promien'] * np.sin(th), pen=pg.mkPen('#888888', width=1, style=Qt.PenStyle.DashLine))
        p.plot([0, 0], [-g['promien'], g['promien']], pen=pg.mkPen('#555555', width=1))
        sel = np.asarray(w['wybrane'])
        if len(sel):
            p.plot(np.concatenate([sel.real, sel.real]), np.concatenate([sel.imag, -sel.imag]), pen=None, symbol='x', symbolSize=11,
                   symbolPen=pg.mkPen('#ffd166', width=2))
        self.tx_J.setText(f'cząsteczka {klucz[2]}, checkpoint {klucz[1] + 1}:\n{proc(w["widmo"]["zesp"])} zespolonych '
                          f'(losowa ≈ {proc(projections.GINIBRE_ZESPOLONE)})\n× = rama R5')
        self.tx_J.setPos(float(lam.real.min()), float(np.abs(lam.imag).max()))
