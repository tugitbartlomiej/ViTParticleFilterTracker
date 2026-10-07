"""Zakładka „Film: sekwencja klatek”: zakres filmu (od mm:ss do mm:ss) -> N klatek -> latent ViT każdej
klatki, etykiety i obrysy narzędzia, ręczne obwody, odtwarzanie w cząsteczkach 3D nad obrazem i
w przestrzeni makrowektorów.

Klatek nie zapisuje się na dysk; w pamięci jest sekwencja (obrazy 480x320 i latent float16).
„Zapisz sekwencję” zapisuje tylko latent, wyniki ViT, etykiety i ręczne obwody (NPZ + JSON).
"""
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                             QProgressBar, QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import detekcja as D
import film as F
import silnik
from styl import rola
from uklad import Przeplyw, dodaj

KOLOR_GT = (0, 1, 1, 1)           # obrys z etykiety
KOLOR_RECZNY = (1, 1, 0, 1)       # obwód narysowany
KOLOR_VIT = (0.2, 1, 0.2, 1)      # ramka ViT
KOLOR_RAMKA_GT = (0, .6, 1, 1)
TYTUL_KLATKI = .07                # część wysokości płótna klatki na tytuł nad obrazem


class ProporcjaKlatki(QObject):
    """Trzyma wysokość płótna klatki równą szerokości panelu razy 2/3 (obraz 480x320) plus tytuł,
    żeby nad i pod obrazem nie zostawały puste pasy; resztę wysokości panelu dostaje wykres przebiegu."""

    def __init__(self, plotno, panel):
        super().__init__(panel)
        self.plotno = plotno
        panel.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Resize:
            wys = int(round(obj.width() * 320 / 480 / (1 - TYTUL_KLATKI)))
            wys = max(200, min(wys, int(obj.height() * .95)))     # bez wykresu przebiegu klatka może zająć prawie całość
            if self.plotno.height() != wys:
                self.plotno.setFixedHeight(wys)
        return False


class WatekFilmu(QThread):
    postep = pyqtSignal(int, str)
    gotowy = pyqtSignal(object)
    blad = pyqtSignal(str)

    def __init__(self, funkcja, parent=None):
        super().__init__(parent)
        self.funkcja, self.stop = funkcja, False

    def run(self):
        try:
            self.gotowy.emit(self.funkcja(self))
        except Exception:
            self.blad.emit(traceback.format_exc())


def prostokat(r):
    x0, y0, x1, y1 = r
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], np.float32)


class ZakladkaFilmu(QWidget):
    def __init__(self, okno):
        super().__init__()
        self.okno = okno
        self.j128 = [w['kanal'] for w in okno.narzedzie]
        self.filmy = F.lista_filmow()
        self.mp4 = None; self.info = None; self.film = None
        self.folder = None                 # sekwencja z folderu zdjęć (F.folder_zdjec)
        self.ckpt, self.det = silnik.VIT_CHECKPOINT, None
        self.sek = None                    # słownik sekwencji
        self.obwody = {}                   # t -> lista wielokątów narysowanych ręcznie (480x320)
        self.rysowany = []                 # punkty bieżącego obwodu
        self.t, self.token, self.watek = 0, None, None
        lay = QVBoxLayout(self)

        # ---------------- góra: dwie grupy obok siebie, źródło klatek | zakres i konwersja
        gora = QHBoxLayout(); lay.addLayout(gora)
        g1 = QGroupBox('1. Źródło klatek'); l1 = QVBoxLayout(g1); l1.setSpacing(4); gora.addWidget(g1, 2)
        w = Przeplyw(); l1.addLayout(w)                                  # wiersz 1: wybór filmu
        self.combo_film = QComboBox(); self.combo_film.setMinimumWidth(340)
        for f in self.filmy:
            n = f'{f["z_narzedziem"]}/{f["klatek"]}' if f['klatek'] else '?'
            self.combo_film.addItem(f'{f["id"]}  [{f["split"]}]  ({n} z narzędziem)', f['id'])
        dodaj(w, 'Film:', self.combo_film, 'Filmy Cataract-LMM 3_Object_Tracking (170 zipów adnotacji).\n'
              'train = zbiór uczący ViT (118 filmów), val = walidacyjny (25), poza manifestem = pozostałe.\n'
              'W nawiasie: klatek z oznaczonym narzędziem / wszystkich klatek.')
        self.combo_split = QComboBox(); self.combo_split.addItems(['wszystkie', 'tylko train', 'tylko val'])
        self.combo_split.setToolTip('Filtr listy filmów po podziale na zbiory.')
        w.addWidget(self.combo_split)
        self.btn_otworz = QPushButton('Otwórz film')
        self.btn_otworz.setToolTip(f'Wypakowuje MP4 wybranego filmu (raz) do {F.KATALOG_MP4} i czyta jego długość.')
        w.addWidget(self.btn_otworz)
        w = Przeplyw(); l1.addLayout(w)                                  # wiersz 2: inne źródła i checkpoint
        e = rola(QLabel('albo:'), 'opis'); w.addWidget(e)
        self.btn_plik = rola(QPushButton('Plik MP4…'), 'narzedzie')
        self.btn_plik.setToolTip('Dowolny plik MP4. Etykiety są wczytywane,\n'
                                 'jeśli nazwa pliku zawiera identyfikator TR_xxxx_Sx_Pxx.')
        w.addWidget(self.btn_plik)
        self.btn_folder = rola(QPushButton('Folder zdjęć…'), 'narzedzie')
        self.btn_folder.setToolTip('Sekwencja z folderu zdjęć (PNG, JPG, BMP, TIF), w kolejności nazw plików,\n'
                                   'np. app/narzedzie_czarne_tlo/<sekwencja> (samo narzędzie na czarnym tle).\n'
                                   'Gdy folder ma MANIFEST.json (wytnij_narzedzie.py), etykiety i obrysy narzędzia\n'
                                   'są brane z annotation.csv dla właściwych numerów klatek. Czas = numer zdjęcia / kl./s.')
        w.addWidget(self.btn_folder)
        self.btn_ckpt = rola(QPushButton('Checkpoint ViT…'), 'narzedzie')
        self.btn_ckpt.setToolTip('Checkpoint małego ViT do latentu i detekcji.')
        w.addWidget(self.btn_ckpt)

        g2 = QGroupBox('2. Zakres i konwersja na sekwencję'); l2 = QVBoxLayout(g2); l2.setSpacing(4); gora.addWidget(g2, 3)
        w = Przeplyw(); l2.addLayout(w)                                  # wiersz 1: zakres czasu i liczba klatek

        def czas(nazwa, opis):
            m = QSpinBox(); m.setRange(0, 600); m.setSuffix(' min'); m.setToolTip(opis)
            s = QDoubleSpinBox(); s.setRange(0, 59.9); s.setDecimals(1); s.setSingleStep(1); s.setSuffix(' s'); s.setToolTip(opis)
            para = QWidget(); l = QHBoxLayout(para); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(2)
            l.addWidget(m); l.addWidget(s)
            dodaj(w, nazwa, para, opis)
            return m, s

        self.od_min, self.od_s = czas('Od:', 'Początek zakresu: minuta i sekunda filmu.')
        self.do_min, self.do_s = czas('Do:', 'Koniec zakresu: minuta i sekunda filmu (włącznie).')
        self.do_s.setValue(10)
        self.spin_n = QSpinBox(); self.spin_n.setRange(2, 1000); self.spin_n.setValue(100)
        dodaj(w, 'Klatek:', self.spin_n,
              'Ile klatek wziąć z zakresu, równo rozłożonych. Gdy zakres ma mniej klatek, bierze wszystkie.\n'
              '100 klatek to około 18 MB latentu (float16) i 46 MB obrazów w pamięci.')
        self.etykieta_zakresu = rola(QLabel(''), 'opis')
        self.etykieta_zakresu.setToolTip('Zakres czasu, numery klatek filmu i co ile klatek bierze sekwencja.')
        w.addWidget(self.etykieta_zakresu)
        w = Przeplyw(); l2.addLayout(w)                                  # wiersz 2: skąd czytać i konwersja
        self.combo_zrodlo = QComboBox(); self.combo_zrodlo.addItems(['MP4', 'JPG z zipa adnotacji', 'folder zdjęć'])
        dodaj(w, 'Czytaj z:', self.combo_zrodlo, 'Skąd czytać klatki. MP4: film. JPG: te same klatki zapisane w zipie adnotacji\n'
              '(na nich robiono etykiety; nie trzeba wypakowywać MP4).')
        self.chk_z_narz = QCheckBox('tylko klatki z etykietą narzędzia')
        self.chk_z_narz.setToolTip('Bierze tylko klatki, na których w annotation.csv oznaczono cystotom.')
        w.addWidget(self.chk_z_narz)
        self.btn_konwertuj = rola(QPushButton('Konwertuj na sekwencję'), 'glowny')
        self.btn_konwertuj.setToolTip('Czyta klatki zakresu, liczy latent ViT (600 x 768) i detekcję każdej klatki,\n'
                                      'wczytuje etykiety i obrysy narzędzia.')
        w.addWidget(self.btn_konwertuj)
        self.btn_przerwij = QPushButton('Przerwij'); self.btn_przerwij.setEnabled(False); w.addWidget(self.btn_przerwij)
        self.pasek = QProgressBar(); self.pasek.setMinimumWidth(160); self.pasek.setMaximumWidth(220)
        self.pasek.setToolTip('Postęp konwersji (klatki wczytane i przeliczone przez ViT).')
        w.addWidget(self.pasek)
        self.btn_zapisz = QPushButton('Zapisz sekwencję')
        self.btn_zapisz.setToolTip(f'NPZ (latent float16, ramki ViT, etykiety) i JSON (ręczne obwody) do {silnik.WYJSCIE}.\n'
                                   'Klatki obrazu nie są zapisywane.')
        w.addWidget(self.btn_zapisz)
        w.addStretch(1)
        self.etykieta_filmu = rola(QLabel(''), 'opis'); self.etykieta_filmu.setWordWrap(True)
        lay.addWidget(self.etykieta_filmu)

        # ---------------- środek: klatka | 3D nad obrazem | przestrzeń makrowektorów
        podzial = QSplitter(Qt.Orientation.Horizontal)
        lewa = QWidget(); ll = QVBoxLayout(lewa); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(4)

        def grupa(tytul):
            g = QGroupBox(tytul); w = Przeplyw(); g.setLayout(w); w.setContentsMargins(8, 4, 8, 4); ll.addWidget(g)
            return w

        w = grupa('Nakładki na klatce')
        self.chk = {}
        for k, nazwa, opis, wl in (('obrys', 'obrys narzędzia', 'Obrys cystotomu z etykiety (cystotome_seg), turkusowy.', True),
                                   ('ramka', 'ramka GT', 'Ramka narzędzia z etykiety (cystotome_bbox), niebieska.', False),
                                   ('vit', 'ramka ViT', 'Najpewniejsza ramka ViT (zielona) z pewnością.', True),
                                   ('czubek', 'czubek', 'Czubek narzędzia z etykiety (cystotome_tip).', True),
                                   ('rogowka', 'rogówka', 'Obrys rogówki z etykiety.', False),
                                   ('zrenica', 'źrenica', 'Obrys źrenicy z etykiety.', False),
                                   ('siatka', 'siatka patchy', 'Siatka 20x30 i patche maski narzędzia.', False)):
            c = QCheckBox(nazwa); c.setChecked(wl); c.setToolTip(opis); w.addWidget(c); self.chk[k] = c
        w = grupa('Obwód narzędzia ręcznie')
        self.btn_rysuj = rola(QPushButton('✎ Rysuj obwód'), 'narzedzie'); self.btn_rysuj.setCheckable(True)
        self.btn_rysuj.setToolTip('Lewy klik dodaje punkt obwodu narzędzia, prawy klik zamyka obwód.\n'
                                  'Obwód należy do bieżącej klatki (żółty). Można narysować kilka.')
        w.addWidget(self.btn_rysuj)
        self.btn_cofnij = rola(QPushButton('Cofnij punkt'), 'narzedzie')
        self.btn_cofnij.setToolTip('Usuwa ostatni punkt rysowanego obwodu, a gdy go nie ma, ostatni obwód klatki.')
        w.addWidget(self.btn_cofnij)
        self.btn_wyczysc = rola(QPushButton('Wyczyść klatkę'), 'narzedzie')
        self.btn_wyczysc.setToolTip('Usuwa wszystkie obwody narysowane na bieżącej klatce.')
        w.addWidget(self.btn_wyczysc)
        self.btn_kopiuj = rola(QPushButton('Kopiuj na następne'), 'narzedzie')
        self.btn_kopiuj.setToolTip('Kopiuje obwody bieżącej klatki na wszystkie następne klatki bez obwodu.')
        w.addWidget(self.btn_kopiuj)
        w = grupa('Maska patchy narzędzia')
        self.combo_maska = QComboBox(); self.combo_maska.addItems(['obrys z etykiety', 'obwód narysowany', 'oba (suma)'])
        dodaj(w, 'Z:', self.combo_maska, 'Z którego obrysu liczyć, które patche należą do narzędzia\n'
              '(wybór cząstek „tylko w obrysie” w 3D i kolor „narzędzie” w przestrzeni makrowektorów).')
        self.spin_prog = QDoubleSpinBox(); self.spin_prog.setRange(1, 100); self.spin_prog.setValue(10); self.spin_prog.setSuffix(' %')
        dodaj(w, 'Pokrycie ≥', self.spin_prog,
              'Patch należy do narzędzia, gdy obrys pokrywa co najmniej tyle procent jego pikseli.')
        w = grupa('Wycinanie z obrazu (przed analizą ViT)')
        self.chk_wyc = {}
        for k, nazwa in F.OBSZARY_WYCINANIA:
            c = QCheckBox(nazwa); w.addWidget(c); self.chk_wyc[k] = c
            c.setToolTip(f'Obszar „{nazwa}” z etykiety klatki (annotation.csv) bierze udział w wycinaniu.\n'
                         'Można zaznaczyć jeden albo kilka (suma obszarów).')
        self.combo_wyc = QComboBox(); self.combo_wyc.addItems(['zostaw tylko zaznaczone (reszta czarna)',
                                                               'zamaluj zaznaczone na czarno'])
        dodaj(w, 'Tryb:', self.combo_wyc,
              'Jak zmienić klatkę przed ViT: zostawić tylko wybrane obszary na czarnym tle albo je zaczernić.\n'
              'Tak zmieniona klatka idzie do latentu, detekcji, atencji i widoków 3D. Zmiana wymaga ponownej\n'
              'konwersji („Konwertuj na sekwencję”). Klatki bez etykiety danego obszaru: brak tego obszaru.')
        self.spin_wyc_marg = QSpinBox(); self.spin_wyc_marg.setRange(0, 60); self.spin_wyc_marg.setSuffix(' px')
        dodaj(w, 'Margines:', self.spin_wyc_marg, 'Poszerzenie wycinanych obszarów o tyle pikseli (obraz 480 x 320).')
        self.chk_wyc_zywo = QCheckBox('na żywo'); self.chk_wyc_zywo.setChecked(True)
        self.chk_wyc_zywo.setToolTip('Zmiana obszarów, trybu albo marginesu od razu przelicza ViT na klatkach wczytanej\n'
                                     'sekwencji (trzymanych w pamięci, bez ponownego czytania filmu): latent, detekcja,\n'
                                     'atencja, widok 3D i przestrzeń makrowektorów. Wyłączone: zmiana działa dopiero\n'
                                     'przy „Konwertuj na sekwencję”.')
        w.addWidget(self.chk_wyc_zywo)
        self._timer_wyc = QTimer(self); self._timer_wyc.setSingleShot(True); self._timer_wyc.setInterval(350)
        self._timer_wyc.timeout.connect(self._wycinanie_na_zywo)
        # klatka 480x320 (3:2) plus pasek tytułu: wysokość płótna idzie za szerokością panelu, bez pustych pasów
        self.fig = Figure(figsize=(6, 4.3), facecolor='#1e1f22'); self.ax = self.fig.add_axes([0, 0, 1, 1 - TYTUL_KLATKI])
        self.canvas = FigureCanvasQTAgg(self.fig); self.canvas.setMinimumHeight(200)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ll.addWidget(self.canvas)
        self.fig_os = Figure(figsize=(6, 1.8), facecolor='#1e1f22'); self.ax_os = self.fig_os.add_axes([.08, .22, .9, .7])
        self.canvas_os = FigureCanvasQTAgg(self.fig_os); self.canvas_os.setMinimumHeight(140)
        self.canvas_os.setToolTip('Przebieg sekwencji: pewność ViT, IoU z ramką etykiety i zmiana latentu względem\n'
                                  'poprzedniej klatki. Kliknięcie przechodzi do klatki.')
        ll.addWidget(self.canvas_os, 1)
        self.canvas_os.setVisible(False)                   # przebieg sekwencji niepotrzebny: klatka dostaje miejsce
        ll.addStretch(1)
        self._proporcja = ProporcjaKlatki(self.canvas, lewa)
        podzial.addWidget(lewa)

        from czasteczki3d import WidokCzasteczek3D
        from przestrzen_makro import WidokPrzestrzeniMakro
        self.w3d = WidokCzasteczek3D(self.j128)
        self.makro = WidokPrzestrzeniMakro(self.j128)
        self.makro.ustaw_wzor(self.w3d)                    # te same kierunki mikrowektorów i orientacja kamery
        podzial.addWidget(self.w3d); podzial.addWidget(self.makro)
        podzial.setStretchFactor(0, 0); podzial.setStretchFactor(1, 1); podzial.setStretchFactor(2, 1)
        podzial.setSizes([480, 700, 700])          # lewa kolumna stała, oba widoki 3D dzielą resztę po równo
        self._podzial = podzial
        self._podzielono = False                   # podział kolumn ustawiany przy pierwszym pokazaniu zakładki
        lay.addWidget(podzial, 1)

        # ---------------- odtwarzanie
        w = QHBoxLayout(); lay.addLayout(w)
        self.btn_pocz = QPushButton('⏮ Początek'); self.btn_wstecz = QPushButton('◀ Klatka')
        self.btn_graj = rola(QPushButton('▶ Odtwarzaj'), 'glowny'); self.btn_graj.setMinimumWidth(130)
        self.btn_graj.setCheckable(True); self.btn_dalej = QPushButton('Klatka ▶'); self.btn_kon = QPushButton('Koniec ⏭')
        for b, opis in ((self.btn_pocz, 'Pierwsza klatka sekwencji.'), (self.btn_wstecz, 'Poprzednia klatka.'),
                        (self.btn_graj, 'Odtwarzanie sekwencji (spacja w polu suwaka też przesuwa).'),
                        (self.btn_dalej, 'Następna klatka.'), (self.btn_kon, 'Ostatnia klatka sekwencji.')):
            b.setToolTip(opis); w.addWidget(b)
        self.suwak = QSlider(Qt.Orientation.Horizontal); self.suwak.setRange(0, 0); w.addWidget(self.suwak, 1)
        self.spin_fps = QDoubleSpinBox(); self.spin_fps.setRange(.5, 30); self.spin_fps.setValue(6)
        dodaj(w, 'kl./s', self.spin_fps,
              'Szybkość odtwarzania sekwencji. Przy 768 wektorach x 600 patchy jedna klatka 3D to około 60 ms.')
        self.chk_petla = QCheckBox('pętla'); self.chk_petla.setChecked(True)
        self.chk_petla.setToolTip('Po ostatniej klatce odtwarzanie wraca do pierwszej.')
        w.addWidget(self.chk_petla)
        # drugi wiersz: co pokazują widoki obok (porównanie, śledzony patch) i opis bieżącej klatki
        w = QHBoxLayout(); lay.addLayout(w)
        self.combo_por = QComboBox(); self.combo_por.addItems(['brak', 'z poprzednią klatką', 'z pierwszą klatką'])
        dodaj(w, 'Porównanie w 3D:', self.combo_por,
              'Co jest „PRZED” w widoku 3D: brak (tylko bieżąca klatka), poprzednia klatka\n'
              'albo pierwsza klatka sekwencji; „PO” = bieżąca. Tryb „różnica” w 3D pokaże zmianę.')
        w.addSpacing(12)
        self.combo_czubek = QComboBox()
        self.combo_czubek.addItems(['wyłączone', 'czubek z etykiety (ViT, gdy brak)', 'najpewniejszy patch ViT'])
        dodaj(w, 'Śledź czubek:', self.combo_czubek,
              'W każdej klatce automatycznie wybiera patch z czubkiem igły (cząstka w 3D,\n'
              'cząstka i jej mikrowektory w przestrzeni makrowektorów, ramka na klatce):\n'
              '- czubek z etykiety: patch, w którym leży cystotome_tip z annotation.csv;\n'
              '  w klatce bez oznaczonego czubka bierze najpewniejszy patch ViT;\n'
              '- najpewniejszy patch ViT: patch o największej pewności środka na mapie ViT\n'
              '  (środek wykrytego narzędzia, nie zawsze sam czubek).\n'
              'Ręczny klik w patch wyłącza śledzenie.')
        w.addSpacing(16)
        self.chk_langevin = QCheckBox('Langevin'); self.chk_langevin.setChecked(True)
        self.chk_langevin.setToolTip('Liniowy model ruchu każdej cząstki po klatkach filmu (dyskretny Langevin z bezwładnością):\n'
                                     '  v(t+1) = beta v(t) − kappa (z(t) − m(t)),  z(t+1) = z(t) + v(t+1),\n'
                                     'm = średnia ostatnich klatek; beta i kappa każdej cząstki dopasowane na pierwszych 70 % klatek.\n'
                                     'W przestrzeni makrowektorów dla wybranej cząstki: pomarańczowa przerywana = tor z modelu\n'
                                     '(tylko ze stanów do bieżącej klatki), turkusowa = rzeczywisty tor w tych klatkach.\n'
                                     'Ocena na ostatnich 30 % klatek w opisie pod spodem (błąd / błąd „bez ruchu”).')
        w.addWidget(self.chk_langevin)
        self.spin_lang_h = QSpinBox(); self.spin_lang_h.setRange(2, 200); self.spin_lang_h.setValue(10)
        dodaj(w, 'na', self.spin_lang_h, 'Na ile klatek do przodu tor z modelu Langevina.')
        self.chk_langevin.toggled.connect(lambda *_: self.idz(self.t))
        self.spin_lang_h.valueChanged.connect(lambda *_: self.idz(self.t))
        self.osc, self._opis_lang = None, ''
        w.addSpacing(12)
        self.chk_widok3d = QCheckBox('widok 3D'); self.chk_widok3d.setChecked(True)
        self.chk_widok_makro = QCheckBox('makrowektory'); self.chk_widok_makro.setChecked(True)
        for c, opis in ((self.chk_widok3d, 'Widok cząsteczek 3D nad obrazem (środkowa kolumna).'),
                        (self.chk_widok_makro, 'Przestrzeń makrowektorów (prawa kolumna).')):
            c.setToolTip(opis + '\nWyłączony widok znika i nie jest przeliczany, więc odtwarzanie jest szybsze.')
        e = QLabel('Widoki:'); e.setToolTip('Które widoki obok klatki pokazywać i liczyć.')
        w.addWidget(e); w.addWidget(self.chk_widok3d); w.addWidget(self.chk_widok_makro)
        w.addSpacing(16)
        self.etykieta_klatki = QLabel(''); self.etykieta_klatki.setWordWrap(True)
        w.addWidget(self.etykieta_klatki, 1)
        self.timer = QTimer(self); self.timer.timeout.connect(self._nastepna)

        # ---------------- sygnały
        self.combo_split.currentIndexChanged.connect(self._filtruj)
        self.btn_otworz.clicked.connect(self._otworz)
        self.btn_plik.clicked.connect(self._plik)
        self.btn_folder.clicked.connect(self._wybierz_folder)
        self.btn_ckpt.clicked.connect(self._wybierz_ckpt)
        for s in (self.od_min, self.od_s, self.do_min, self.do_s, self.spin_n):
            s.valueChanged.connect(self._opis_zakresu)
        self.btn_konwertuj.clicked.connect(self._konwertuj)
        for c in self.chk_wyc.values():
            c.toggled.connect(lambda *_: self._timer_wyc.start() if self.chk_wyc_zywo.isChecked() else None)
        for s_ in (self.combo_wyc.currentIndexChanged, self.spin_wyc_marg.valueChanged):
            s_.connect(lambda *_: self._timer_wyc.start() if self.chk_wyc_zywo.isChecked() else None)
        self.chk_wyc_zywo.toggled.connect(lambda v: self._timer_wyc.start() if v else None)
        self.btn_przerwij.clicked.connect(lambda: setattr(self.watek, 'stop', True) if self.watek else None)
        self.btn_zapisz.clicked.connect(self._zapisz)
        for c in self.chk.values():
            c.toggled.connect(lambda *_: self._rysuj_klatke())
        self.btn_rysuj.toggled.connect(self._tryb_rysowania)
        self.btn_cofnij.clicked.connect(self._cofnij)
        self.btn_wyczysc.clicked.connect(self._wyczysc)
        self.btn_kopiuj.clicked.connect(self._kopiuj)
        self.combo_maska.currentIndexChanged.connect(lambda *_: self._zmiana_maski())
        self.spin_prog.valueChanged.connect(lambda *_: self._zmiana_maski())
        self.canvas.mpl_connect('button_press_event', self._klik_klatki)
        self.canvas_os.mpl_connect('button_press_event', self._klik_osi)
        self.suwak.valueChanged.connect(self.idz)
        self.btn_pocz.clicked.connect(lambda: self.suwak.setValue(0))
        self.btn_kon.clicked.connect(lambda: self.suwak.setValue(self.suwak.maximum()))
        self.btn_wstecz.clicked.connect(lambda: self.suwak.setValue(self.suwak.value() - 1))
        self.btn_dalej.clicked.connect(lambda: self.suwak.setValue(self.suwak.value() + 1))
        self.btn_graj.toggled.connect(self._graj)
        self.spin_fps.valueChanged.connect(lambda v: self.timer.setInterval(int(1000 / v)))
        self.combo_por.currentIndexChanged.connect(lambda *_: self.idz(self.t))
        self.chk_widok3d.toggled.connect(lambda v: self._pokaz_widok(self.w3d, v))
        self.chk_widok_makro.toggled.connect(lambda v: self._pokaz_widok(self.makro, v))
        self.combo_czubek.currentIndexChanged.connect(lambda *_: self.idz(self.t))
        self.w3d.widok.klik.connect(lambda *_: self._wybierz_token(self.w3d.token))
        self.makro.token_wybrany.connect(self._wybierz_token)     # klik w cząstkę: ten sam patch na obrazie i w 3D
        self._opis_zakresu()

    def showEvent(self, e):
        super().showEvent(e)
        if not self._podzielono:                   # dopiero teraz splitter ma prawdziwą szerokość
            self._podzielono = True
            QTimer.singleShot(0, self._podziel_kolumny)

    def _podziel_kolumny(self):
        szer = self._podzial.width()
        if szer > 900:
            lewa = min(460, szer // 3)
            self._podzial.setSizes([lewa, (szer - lewa) // 2, (szer - lewa) // 2])

    # ------------------------------------------------------------ film
    def _status(self, t, ms=8000):
        self.okno.statusBar().showMessage(t, ms)

    def _filtruj(self):
        i = self.combo_split.currentIndex()
        biezacy = self.combo_film.currentData()
        self.combo_film.clear()
        for f in self.filmy:
            if i == 0 or f['split'] == ('train', 'val')[i - 1]:
                n = f'{f["z_narzedziem"]}/{f["klatek"]}' if f['klatek'] else '?'
                self.combo_film.addItem(f'{f["id"]}  [{f["split"]}]  ({n} z narzędziem)', f['id'])
        j = self.combo_film.findData(biezacy)
        if j >= 0:
            self.combo_film.setCurrentIndex(j)

    def _wybierz_folder(self):
        start = self.folder['katalog'] if self.folder else str(Path(__file__).resolve().parent / 'narzedzie_czarne_tlo')
        s = QFileDialog.getExistingDirectory(self, 'Folder z sekwencją zdjęć', start)
        if s:
            self.otworz_folder(s)

    def otworz_folder(self, katalog):
        """Wczytuje sekwencję z folderu zdjęć; zakres ustawia na cały folder, liczbę klatek na wszystkie."""
        if self.watek:
            return
        f = F.folder_zdjec(katalog)
        if not f['pliki']:
            QMessageBox.information(self, 'Folder zdjęć', f'W folderze nie ma zdjęć:\n{katalog}')
            return
        from PIL import Image
        with Image.open(f['pliki'][0]) as im:
            szer, wys = im.size
        wiersze = {}
        if f['film']:
            et = F.etykiety(f['film'])
            nr = f['frame_numbers']
            if nr:                                                        # plik k <-> frame_number z MANIFEST
                wiersze = {k + 1: et[n] for k, n in enumerate(nr) if n in et}
        n = len(f['pliki'])
        self.folder, self.mp4 = f, None
        self.combo_zrodlo.blockSignals(True); self.combo_zrodlo.setCurrentIndex(2); self.combo_zrodlo.blockSignals(False)
        info = dict(fps=f['fps'], klatek=n, sekund=n / f['fps'], szer=szer, wys=wys)
        self._po_otwarciu((f['film'] or Path(katalog).name, None, info, wiersze))
        for m, s, t in ((self.od_min, self.od_s, 0.0), (self.do_min, self.do_s, (n - 1) / f['fps'])):
            m.setValue(int(t // 60)); s.setValue(round(t % 60, 1))
        self.spin_n.setValue(min(n, self.spin_n.maximum()))
        self.etykieta_filmu.setText(f'Folder {Path(katalog).name}: {n} zdjęć {szer}x{wys}, {f["fps"]:.2f} kl./s'
                                    + (f'; {f["opis"]}' if f['opis'] else '')
                                    + (f'; etykiety z annotation.csv dla {len(wiersze)} zdjęć' if wiersze else '; bez etykiet'))
        self._opis_zakresu()
        self._konwertuj()                                  # folder = od razu cała sekwencja (latent, detekcja, widoki)

    def _otworz(self):
        if self.combo_zrodlo.currentIndex() == 2:
            self._wybierz_folder()
            return
        film = self.combo_film.currentData()
        if not film or self.watek:
            return
        self.folder = None

        def zadanie(wt):
            if self.combo_zrodlo.currentIndex() == 0:
                p = F.sciezka_mp4(film, lambda t: wt.postep.emit(0, t))
                return film, p, F.info_mp4(p), F.etykiety(film)
            et = F.etykiety(film)
            n = len(et)
            return film, None, dict(fps=29.97, klatek=n, sekund=n / 29.97, szer=720, wys=480), et
        self._uruchom(zadanie, self._po_otwarciu, f'otwieram {film}...')

    def _plik(self):
        s, _ = QFileDialog.getOpenFileName(self, 'Film MP4', str(F.KATALOG_MP4), 'Filmy (*.mp4 *.avi *.mov *.mkv)')
        if not s:
            return
        film = F.id_filmu(s)
        self.folder = None
        self.combo_zrodlo.setCurrentIndex(0)
        self._uruchom(lambda wt: (film, s, F.info_mp4(s), F.etykiety(film) if film else {}), self._po_otwarciu)

    def _po_otwarciu(self, w):
        self.film, self.mp4, self.info, self.wiersze = w
        i = self.info
        n_narz = sum(1 for r in self.wiersze.values() if F._json(r.get('cystotome_bbox')))
        self.etykieta_filmu.setText(f'{self.film or "film bez etykiet"}: {F.mmss(i["sekund"])} '
                                    f'({i["klatek"]} kl., {i["fps"]:.2f} kl./s, {i["szer"]}x{i["wys"]}); '
                                    f'etykiet narzędzia: {n_narz}')
        for m in (self.od_min, self.do_min):
            m.setMaximum(int(i['sekund'] // 60))
        self._opis_zakresu()
        self._status(f'Otwarto {self.film}.')

    def _wybierz_ckpt(self):
        s, _ = QFileDialog.getOpenFileName(self, 'Checkpoint ViT', str(self.ckpt), 'Checkpoint (*.pt *.pth)')
        if s:
            self.ckpt, self.det = s, None
            self._status(f'Checkpoint ViT: {s}')

    def _detektor(self):
        if self.det is None:
            self.det = D.Detektor(self.ckpt)
        return self.det

    def _czasy(self):
        return (self.od_min.value() * 60 + self.od_s.value(), self.do_min.value() * 60 + self.do_s.value())

    def _opis_zakresu(self):
        a, b = self._czasy()
        if self.info:
            fps = self.info['fps']
            i0, i1 = int(round(a * fps)), int(round(b * fps))
            n = max(0, min(i1, self.info['klatek'] - 1) - i0 + 1)
            krok = max(1.0, n / max(1, self.spin_n.value()))
            self.etykieta_zakresu.setText(f'{F.mmss(a)} – {F.mmss(b)} = kl. {i0}..{i1} ({n} kl.), co ~{krok:.1f}.')
        else:
            self.etykieta_zakresu.setText(f'{F.mmss(a)} – {F.mmss(b)} (otwórz film)')

    # ------------------------------------------------------------ konwersja
    def _uruchom(self, zadanie, gotowy, opis=''):
        if self.watek:
            return
        self.watek = WatekFilmu(zadanie, self)
        self.watek.postep.connect(lambda k, t: (self.pasek.setValue(k), self._status(t, 3000)))
        self.watek.gotowy.connect(lambda w: (self._koniec_watku(), gotowy(w)))
        self.watek.blad.connect(lambda t: (self._koniec_watku(), QMessageBox.critical(self, 'Błąd', t[-2000:])))
        self.btn_przerwij.setEnabled(True)
        if opis:
            self._status(opis)
        self.watek.start()

    def _koniec_watku(self):
        self.watek = None
        self.btn_przerwij.setEnabled(False)

    def _konwertuj(self):
        if not self.info:
            self._otworz_i_konwertuj = True
            QMessageBox.information(self, 'Film', 'Najpierw „Otwórz film” (albo „Plik MP4…”).')
            return
        a, b = self._czasy()
        idx = F.indeksy_zakresu(a, b, self.spin_n.value(), self.info['fps'], self.info['klatek'], self.wiersze,
                                self.chk_z_narz.isChecked())
        if len(idx) == 0:
            QMessageBox.information(self, 'Film', 'W tym zakresie nie ma klatek (albo klatek z etykietą narzędzia).')
            return
        film, mp4, wiersze, jpg = self.film, self.mp4, self.wiersze, self.combo_zrodlo.currentIndex() == 1 or self.mp4 is None
        folder = self.folder if self.folder and (self.combo_zrodlo.currentIndex() == 2 or self.mp4 is None) else None
        det = self._detektor()
        self.pasek.setRange(0, len(idx))
        wyc = dict(obszary=[k for k, c in self.chk_wyc.items() if c.isChecked()], tryb=self.combo_wyc.currentIndex(),
                   margines=self.spin_wyc_marg.value())

        def zadanie(wt):
            if folder:
                czytnik = F.czytaj_klatki_folder(folder['pliki'], idx, przerwij=lambda: wt.stop)
            elif jpg:
                czytnik = F.czytaj_klatki_jpg(film, idx, przerwij=lambda: wt.stop)
            else:
                czytnik = F.czytaj_klatki(mp4, idx, przerwij=lambda: wt.stop)
            I, Z, ramki, pew, mapy_pew, obrazy, et, oryg = [], [], [], [], [], [], [], []
            t0 = time.perf_counter()
            for k, (i, rgb) in enumerate(czytnik):
                r480 = D.do_480(rgb); r480_oryg = r480
                skala = (480 / rgb.shape[1], 320 / rgb.shape[0])
                e_ = F.etykieta_klatki(wiersze.get(i + 1), skala) if wiersze else F.etykieta_klatki(None)
                if wyc['obszary']:                          # wycinanie obszarów z etykiet przed ViT
                    r480 = F.wytnij(r480, F.maska_wycinania(e_, wyc['obszary'], r480.shape[:2], wyc['margines']), wyc['tryb'])
                a_ = det.analizuj(r480)
                I.append(i); obrazy.append(r480); oryg.append(r480_oryg)
                Z.append(a_['z'][0].cpu().numpy().astype(np.float16))
                ramki.append(a_['boxes'][0]); pew.append(float(a_['scores'][0]))
                mapy_pew.append(torch_sigmoid(a_['mapa'][0, ..., 0]).reshape(-1))     # pewność środka w patchu [600]
                et.append(e_)
                wt.postep.emit(k + 1, f'klatka {k + 1}/{len(idx)} (film kl. {i}), {(time.perf_counter() - t0) / (k + 1) * 1000:.0f} ms/kl.')
            if not I:
                raise RuntimeError('nie wczytano żadnej klatki')
            Z = np.stack(Z)
            iou = [float(D.iou(np.asarray(r)[None], e['ramka'])[0]) if e['ramka'] is not None else np.nan
                   for r, e in zip(ramki, et)]
            zm = [np.nan] + [float(np.linalg.norm(Z[t].astype(np.float32) - Z[t - 1]) / max(np.linalg.norm(Z[t - 1].astype(np.float32)), 1e-9))
                             for t in range(1, len(Z))]
            return dict(film=film, idx=np.array(I), obrazy=obrazy, obrazy_oryg=oryg, Z=Z, ramki=np.array(ramki), pewnosc=np.array(pew),
                        mapy_pew=np.stack(mapy_pew), etykiety=et, iou=np.array(iou), zmiana=np.array(zm),
                        fps=self.info['fps'], zrodlo=('folder ' + folder['katalog']) if folder else ('JPG' if jpg else 'MP4'),
                        wycinanie=wyc)
        self.btn_graj.setChecked(False)
        self._uruchom(zadanie, self._po_konwersji, f'konwertuję {len(idx)} klatek...')

    def _wycinanie_na_zywo(self):
        """Przelicza ViT na klatkach wczytanej sekwencji z bieżącym wycinaniem (bez ponownego czytania filmu)."""
        s = self.sek
        if s is None or 'obrazy_oryg' not in s:
            return
        if self.watek:                                         # trwa inne przeliczenie: spróbuj za chwilę
            self._timer_wyc.start(); return
        wyc = dict(obszary=[k for k, c in self.chk_wyc.items() if c.isChecked()], tryb=self.combo_wyc.currentIndex(),
                   margines=self.spin_wyc_marg.value())
        det = self._detektor()
        oryg, et = s['obrazy_oryg'], s['etykiety']
        self.pasek.setRange(0, len(oryg))

        def zadanie(wt):
            obrazy, Z, ramki, pew, mapy = [], [], [], [], []
            t0 = time.perf_counter()
            for k, (r480, e_) in enumerate(zip(oryg, et)):
                if wt.stop:
                    raise RuntimeError('przerwano przeliczanie wycinania')
                if wyc['obszary']:
                    r480 = F.wytnij(r480, F.maska_wycinania(e_, wyc['obszary'], r480.shape[:2], wyc['margines']), wyc['tryb'])
                a_ = det.analizuj(r480)
                obrazy.append(r480); Z.append(a_['z'][0].cpu().numpy().astype(np.float16))
                ramki.append(a_['boxes'][0]); pew.append(float(a_['scores'][0]))
                mapy.append(torch_sigmoid(a_['mapa'][0, ..., 0]).reshape(-1))
                if k % 5 == 0:
                    wt.postep.emit(k + 1, f'wycinanie na żywo: klatka {k + 1}/{len(oryg)}, '
                                          f'{(time.perf_counter() - t0) / (k + 1) * 1000:.0f} ms/kl.')
            Z = np.stack(Z)
            iou = [float(D.iou(np.asarray(r)[None], e['ramka'])[0]) if e['ramka'] is not None else np.nan
                   for r, e in zip(ramki, et)]
            zm = [np.nan] + [float(np.linalg.norm(Z[t].astype(np.float32) - Z[t - 1]) / max(np.linalg.norm(Z[t - 1].astype(np.float32)), 1e-9))
                             for t in range(1, len(Z))]
            return dict(obrazy=obrazy, Z=Z, ramki=np.array(ramki), pewnosc=np.array(pew), mapy_pew=np.stack(mapy),
                        iou=np.array(iou), zmiana=np.array(zm), wycinanie=wyc)

        def gotowe(w):
            s.update(w)
            self._po_konwersji(s, zachowaj=True)

        self._uruchom(zadanie, gotowe, 'przeliczam ViT z nowym wycinaniem...')

    def _po_konwersji(self, s, zachowaj=False):
        if zachowaj:                                           # przeliczenie tej samej sekwencji: obwody, klatka, patch zostają
            t_stary = self.t
            self.sek = s
        else:
            self.sek, self.obwody, self.rysowany, self.t = s, {}, [], 0
            t_stary = 0
        T = len(s['idx'])
        maski = self._maski()
        self.makro.ckpt = self.ckpt
        self.w3d.ustaw_baze_pca(s['Z'])            # PCA w 3D: jedna baza dla całej sekwencji (przed makro: wspólna baza)
        self.makro.ustaw_sekwencje(s['Z'], maski, s['mapy_pew'])
        self._licz_langevin()
        self.suwak.blockSignals(True); self.suwak.setRange(0, T - 1); self.suwak.setValue(min(t_stary, T - 1))
        self.suwak.blockSignals(False)
        self._rysuj_os()
        self.idz(min(t_stary, T - 1))
        iou = s['iou'][~np.isnan(s['iou'])]
        self._status(f'Sekwencja: {T} klatek {s["film"]} ({s["zrodlo"]}); średnia pewność {s["pewnosc"].mean() * 100:.1f} %'
                     + (f', średnie IoU {iou.mean() * 100:.1f} % na {len(iou)} kl. z etykietą' if len(iou) else '')
                     + self._opis_wycinania(s.get('wycinanie')), 15000)

    @staticmethod
    def _opis_wycinania(w):
        if not w or not w['obszary']:
            return ''
        return (f'; wycinanie: {" + ".join(dict(F.OBSZARY_WYCINANIA)[o] for o in w["obszary"])}, '
                f'{("zostaw tylko", "zaczernione")[w["tryb"]]}' + (f', margines {w["margines"]} px' if w['margines'] else ''))

    # ------------------------------------------------------------ maski i obwody
    def _wielokaty_maski(self, t):
        e = self.sek['etykiety'][t]
        i = self.combo_maska.currentIndex()
        w = []
        if i in (0, 2):
            w += list(e['obrys'])
        if i in (1, 2):
            w += list(self.obwody.get(t, []))
        return w

    def _maski(self):
        if self.sek is None:
            return None
        prog = self.spin_prog.value() / 100
        return np.stack([F.maska_patchy(self._wielokaty_maski(t), prog) for t in range(len(self.sek['idx']))])

    def _zmiana_maski(self):
        if self.sek is None:
            return
        self.makro.ustaw_maski(self._maski())
        self.idz(self.t)

    def _tryb_rysowania(self, wl):
        self.rysowany = []
        if wl:
            self.btn_graj.setChecked(False)
        self._rysuj_klatke()

    def _klik_klatki(self, e):
        if self.sek is None or e.xdata is None:
            return
        if self.btn_rysuj.isChecked():
            if e.button == 1:
                self.rysowany.append((e.xdata, e.ydata))
            elif e.button == 3 and len(self.rysowany) >= 3:
                self.obwody.setdefault(self.t, []).append(np.array(self.rysowany, np.float32))
                self.rysowany = []
                self._zmiana_maski()
                return
            self._rysuj_klatke()
        else:
            k = int(np.clip(e.xdata // 16, 0, 29)); r = int(np.clip(e.ydata // 16, 0, 19))
            self._wybierz_token(r * 30 + k)

    def _token_czubka(self, t):
        """Patch czubka w klatce t (wg combo_czubek) i skąd wzięty; (None, '') gdy śledzenie wyłączone."""
        tryb = self.combo_czubek.currentIndex()
        if tryb == 0 or self.sek is None:
            return None, ''
        if tryb == 1:
            c = self.sek['etykiety'][t]['czubek']
            if c is not None:
                k = int(np.clip(c[0] // 16, 0, 29)); r = int(np.clip(c[1] // 16, 0, 19))
                return r * 30 + k, 'czubek z etykiety'
        return int(np.argmax(self.sek['mapy_pew'][t])), 'najpewniejszy patch ViT'

    def _wybierz_token(self, tok, reczny=True):
        if tok is None:
            return
        if reczny and self.combo_czubek.currentIndex():
            self.combo_czubek.blockSignals(True); self.combo_czubek.setCurrentIndex(0); self.combo_czubek.blockSignals(False)
        self.token = tok
        self.w3d.token = tok
        self.idz(self.t)

    def _cofnij(self):
        if self.rysowany:
            self.rysowany.pop()
        elif self.obwody.get(self.t):
            self.obwody[self.t].pop()
            self._zmiana_maski(); return
        self._rysuj_klatke()

    def _wyczysc(self):
        self.obwody.pop(self.t, None); self.rysowany = []
        self._zmiana_maski()

    def _kopiuj(self):
        if self.sek is None or not self.obwody.get(self.t):
            return
        for t in range(self.t + 1, len(self.sek['idx'])):
            if not self.obwody.get(t):
                self.obwody[t] = [p.copy() for p in self.obwody[self.t]]
        self._zmiana_maski()

    # ------------------------------------------------------------ odtwarzanie
    def _graj(self, wl):
        if wl and self.sek is not None:
            self.btn_graj.setText('⏸ Pauza')
            self.timer.setInterval(int(1000 / self.spin_fps.value())); self.timer.start()
        else:
            self.btn_graj.setText('▶ Odtwarzaj'); self.timer.stop()
            if wl:
                self.btn_graj.setChecked(False)

    def _nastepna(self):
        if self.sek is None:
            return
        t = self.t + 1
        if t >= len(self.sek['idx']):
            if not self.chk_petla.isChecked():
                self.btn_graj.setChecked(False); return
            t = 0
        self.suwak.setValue(t)

    def _pokaz_widok(self, widok, wl):
        """Włącza albo wyłącza widok obok klatki; odpięty widok najpierw wraca na miejsce."""
        if not wl and getattr(widok, 'odpinanie', None) is not None and widok.odpinanie.okno is not None:
            widok.odpinanie.przycisk.setChecked(False)
        widok.setVisible(wl)
        if wl:
            self.idz(self.t)

    def idz(self, t):
        if self.sek is None:
            return
        s = self.sek
        self.t = t = int(np.clip(t, 0, len(s['idx']) - 1))
        if self.suwak.value() != t:
            self.suwak.blockSignals(True); self.suwak.setValue(t); self.suwak.blockSignals(False)
        tok, zrodlo_czubka = self._token_czubka(t)
        if tok is not None:                                 # śledzenie czubka: patch wybierany automatycznie
            self.token = tok; self.w3d.token = tok
        rgb = s['obrazy'][t]
        z = s['Z'][t].astype(np.float32)
        por = self.combo_por.currentIndex()
        przed = None if por == 0 else s['Z'][max(0, t - 1) if por == 1 else 0].astype(np.float32)
        det = self.det
        obrysy = [(p, KOLOR_GT) for p in s['etykiety'][t]['obrys']] if self.chk['obrys'].isChecked() else []
        obrysy += [(p, KOLOR_RECZNY) for p in self.obwody.get(t, [])]
        if self.chk['vit'].isChecked():
            obrysy.append((prostokat(s['ramki'][t]), KOLOR_VIT))
        if self.chk['ramka'].isChecked() and s['etykiety'][t]['ramka'] is not None:
            obrysy.append((prostokat(s['etykiety'][t]['ramka']), KOLOR_RAMKA_GT))
        maska = F.maska_patchy(self._wielokaty_maski(t), self.spin_prog.value() / 100)
        fn = (lambda b, rgb=rgb: det.atencja(rgb, b)) if det is not None else None
        if self.chk_widok3d.isChecked():                    # wyłączony widok nie jest przeliczany
            self.w3d.ustaw_obrysy(obrysy, maska)
            if przed is None:
                self.w3d.pokaz(z, None, rgb, fn)
            else:
                if self.w3d.combo_tryb.currentIndex() == 0:
                    self.w3d.combo_tryb.blockSignals(True); self.w3d.combo_tryb.setCurrentIndex(1); self.w3d.combo_tryb.blockSignals(False)
                self.w3d.pokaz(przed, z, rgb, fn)
        if self.chk_widok_makro.isChecked():
            self.makro.ustaw_atencje(fn)                    # mapa i strzałki atencji tej klatki
            self.makro.ustaw_obraz(rgb)                     # zdjęcie zdeformowane do położeń cząstek
            self.makro.pokaz_klatke(t, self.token)
            self._tor_langevina(t)
        self._rysuj_klatke()
        self._znacznik_osi()
        i = int(s['idx'][t])
        iou = s['iou'][t]
        self.etykieta_klatki.setText(f'klatka {t + 1}/{len(s["idx"])}: film kl. {i} (t = {F.mmss(i / s["fps"])}), '
                                     f'pewność {s["pewnosc"][t] * 100:.1f} %'
                                     + ('' if np.isnan(iou) else f', IoU {iou * 100:.1f} %')
                                     + f', patchy narzędzia {int(maska.sum())}'
                                     + (f'; czubek: patch {tok} (wiersz {tok // 30}, kolumna {tok % 30}, {zrodlo_czubka})'
                                        if tok is not None else '') + self._opis_lang)

    # ------------------------------------------------------------ Langevin po klatkach filmu
    def _licz_langevin(self):
        """Parametry oscylatora każdej cząstki (pierwsze 70 % klatek) i ocena prognozy na pozostałych."""
        import analiza_uczenia as AU
        Z = self.sek['Z']; T = len(Z)
        self.osc, self._opis_lang = None, ''
        if T < 12:
            self._opis_lang = '; Langevin: za mało klatek (min. 12)'
            return
        do = max(4, int(T * .7)); okno = int(np.clip(T // 8, 3, 50))
        self.osc = AU.oscylator(Z, od=1, do=do, okno=okno)
        oc = AU.ocena_oscylatora(Z, self.osc, do, hs=(1, 5))
        self._opis_lang = ('; Langevin (dopasowany na kl. 1-' + str(do) + f', beta {np.median(self.osc["beta"]):.2f}, '
                           f'kappa {np.median(self.osc["kappa"]):.3f}): błąd prognozy / bez ruchu na kl. {do}-{T}: '
                           + ', '.join(f'{h} kl. {v:.2f}' for h, v in oc.items()))

    def _tor_langevina(self, t):
        if not self.chk_langevin.isChecked() or self.osc is None or self.token is None:
            self.makro.rysuj_przyczyny(None); return
        import analiza_uczenia as AU
        Z, h, tok = self.sek['Z'], self.spin_lang_h.value(), self.token
        tor = AU.prognoza_oscylatora(Z, t, tok, self.osc, h)
        o = self.makro._biezace()[tok]
        P = o + self.makro.rzut_delty(tor - tor[0])
        seg, kol = [], []
        for a in range(0, h, 2):
            seg += [P[a], P[a + 1]]; kol += [(1, .55, .1, 1)] * 2
        kon = min(len(Z) - 1, t + h)
        if kon > t:
            pr = np.asarray(Z[t:kon + 1, tok], np.float32)
            Q = o + self.makro.rzut_delty(pr - pr[0])
            for a in range(len(Q) - 1):
                seg += [Q[a], Q[a + 1]]; kol += [(.2, .9, 1, .8)] * 2
        self.makro.rysuj_przyczyny(np.array(seg), np.array(kol))

    # ------------------------------------------------------------ rysunki 2D
    def _rysuj_klatke(self):
        if self.sek is None:
            return
        s, t, ax = self.sek, self.t, self.ax
        ax.clear(); ax.imshow(s['obrazy'][t], extent=(0, 480, 320, 0))
        e = s['etykiety'][t]

        def linia(p, kol, lw=1.4, zamk=True, styl='-'):
            p = np.asarray(p)
            if zamk:
                p = np.vstack([p, p[:1]])
            ax.plot(p[:, 0], p[:, 1], styl, color=kol[:3], lw=lw)

        if self.chk['siatka'].isChecked():
            for x in range(0, 481, 16):
                ax.plot([x, x], [0, 320], color='w', lw=.2, alpha=.4)
            for y in range(0, 321, 16):
                ax.plot([0, 480], [y, y], color='w', lw=.2, alpha=.4)
            m = F.maska_patchy(self._wielokaty_maski(t), self.spin_prog.value() / 100).reshape(20, 30)
            ax.imshow(np.ma.masked_where(~m, m), extent=(0, 480, 320, 0), cmap='autumn', alpha=.35, interpolation='nearest')
        if self.chk['obrys'].isChecked():
            for p in e['obrys']:
                linia(p, KOLOR_GT)
        for k, kol in (('rogowka', (1, .5, 1)), ('zrenica', (1, .6, .2))):
            if self.chk[k].isChecked():
                for p in e[k]:
                    linia(p, kol, 1)
        if self.chk['ramka'].isChecked() and e['ramka'] is not None:
            linia(prostokat(e['ramka']), KOLOR_RAMKA_GT, 1.2)
        if self.chk['vit'].isChecked():
            r = s['ramki'][t]
            linia(prostokat(r), KOLOR_VIT, 1.6)
            ax.text(r[0], r[1] - 3, f'ViT {s["pewnosc"][t] * 100:.0f} %', color='lime', fontsize=8)
        if self.chk['czubek'].isChecked() and e['czubek'] is not None:
            ax.plot(*e['czubek'], 'o', ms=5, mfc='none', mec='r', mew=1.5)
        for p in self.obwody.get(t, []):
            linia(p, KOLOR_RECZNY, 1.8)
        if self.rysowany:
            linia(np.array(self.rysowany), KOLOR_RECZNY, 1.2, zamk=False, styl='--')
            ax.plot(*np.array(self.rysowany).T, 'o', color='y', ms=3)
        if self.token is not None:
            r_, k_ = divmod(self.token, 30)
            linia(prostokat([k_ * 16, r_ * 16, k_ * 16 + 16, r_ * 16 + 16]), (1, 1, 1), 1.2)
        ax.set_xlim(0, 480); ax.set_ylim(320, 0); ax.axis('off')
        tryb = ' | RYSOWANIE: lewy klik = punkt, prawy = zamknij' if self.btn_rysuj.isChecked() else ' | klik = wybór patcha'
        ax.set_title(f'{s["film"]} kl. {int(s["idx"][t])}{tryb}', fontsize=8, color='#dddddd')
        self.canvas.draw_idle()

    def _rysuj_os(self):
        s, ax = self.sek, self.ax_os
        ax.clear()
        x = np.arange(len(s['idx']))
        ax.plot(x, s['pewnosc'] * 100, color='tab:green', lw=1.2, label='pewność ViT %')
        ax.plot(x, s['iou'] * 100, color='tab:blue', lw=1.2, label='IoU z etykietą %')
        zm = s['zmiana']
        if np.isfinite(zm).any():
            ax.plot(x, zm / np.nanmax(zm) * 100, color='tab:orange', lw=1, alpha=.8, label='zmiana latentu (wzgl. maks.)')
        ax.set_ylim(0, 105); ax.set_xlim(0, max(1, len(x) - 1)); ax.tick_params(labelsize=7, colors='#cccccc')
        ax.set_facecolor('#26282c')
        for b in ax.spines.values():
            b.set_color('#55585e')
        ax.set_xlabel('klatka sekwencji', fontsize=7, color='#cccccc')
        leg = ax.legend(fontsize=6, loc='lower left', ncol=3, facecolor='#26282c', edgecolor='#55585e')
        for t in leg.get_texts():
            t.set_color('#dddddd')
        self._linia_os = ax.axvline(0, color='white', lw=1)
        self.canvas_os.draw_idle()

    def _znacznik_osi(self):
        if getattr(self, '_linia_os', None) is not None and self.canvas_os.isVisible():
            self._linia_os.set_xdata([self.t, self.t])
            self.canvas_os.draw_idle()

    def _klik_osi(self, e):
        if self.sek is not None and e.xdata is not None:
            self.suwak.setValue(int(round(e.xdata)))

    # ------------------------------------------------------------ zapis
    def _zapisz(self):
        if self.sek is None:
            return
        s = self.sek
        silnik.WYJSCIE.mkdir(parents=True, exist_ok=True)
        i0, i1 = int(s['idx'][0]), int(s['idx'][-1])
        tag = f'film_{s["film"]}_{F.mmss(i0 / s["fps"]).replace(":", "m")}-{F.mmss(i1 / s["fps"]).replace(":", "m")}_{len(s["idx"])}kl'
        np.savez_compressed(silnik.WYJSCIE / f'{tag}.npz', idx=s['idx'], Z=s['Z'], ramki_vit=s['ramki'],
                            pewnosc=s['pewnosc'], iou=s['iou'], zmiana=s['zmiana'],
                            ramki_gt=np.array([e['ramka'] if e['ramka'] is not None else [np.nan] * 4 for e in s['etykiety']], np.float32),
                            maski=self._maski())
        meta = dict(film=s['film'], zrodlo=s['zrodlo'], fps=s['fps'], checkpoint=str(self.ckpt), wycinanie=s.get('wycinanie'),
                    klatki=[int(i) for i in s['idx']], uwaga='współrzędne w pikselach obrazu 480x320 (wejście ViT)',
                    maska_z=self.combo_maska.currentText(), prog_pokrycia=self.spin_prog.value() / 100,
                    obwody_reczne={int(s['idx'][t]): [p.tolist() for p in v] for t, v in self.obwody.items() if v})
        (silnik.WYJSCIE / f'{tag}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
        self._status(f'Zapisano {silnik.WYJSCIE / tag}.npz i .json', 15000)


def torch_sigmoid(m):
    return m.detach().float().sigmoid().cpu().numpy()
