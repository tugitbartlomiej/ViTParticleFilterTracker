"""Widok „Przestrzeń makrowektorów”: każdy patch (token) jako cząstka pchana przez swoje 768 mikrowektorów.

Kanał c ma stały kierunek d_c w 3D (makrowektor, ten sam dla wszystkich patchy i klatek). Mikrowektor
patcha i w klatce t to (z_tic − μ_c) · d_c: siła = wartość kanału (względem średniej), kierunek = d_c.
Położenie cząstki = suma jej 768 mikrowektorów, przemnożona przez stałą skalę sekwencji:

    p_ti = s · Σ_c (z_tic − μ_c) d_c = s · (z_ti − μ) D,   D: [768, 3]

więc między klatkami cząstka przelatuje tam, dokąd pchają ją zmienione siły kanałów. μ = średni latent
całej sekwencji (opcjonalnie 0); s dobrane raz na sekwencję (99. percentyl promienia = 100 jednostek),
żeby ruch był porównywalny między klatkami. Kierunki: spirala Fibonacciego na sferze (jak w widoku 3D),
losowe, albo trzy główne składowe PCA latentu sekwencji (największy rozrzut; kierunek kanału = jego
ładunki PCA, wciąż stały i liniowy).
Rysowanie: pyqtgraph.opengl (VBO na karcie).
"""
import time

import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QLineEdit, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from uklad import WaskiCombo, etykieta_statusu
from styl import rola
from uklad import Odpinanie, PanelOpcji, dodaj, dodaj_nawigacje
from widok_gl import LinieGrube, WidokGL
from wspolne import CZASTECZKI_CO, KOLUMNY, PCA_SKLADOWE, kierunki, kolory_kanalow, opisy_czasteczek, pca, tokeny_co


class WidokPrzestrzeniMakro(QWidget):
    token_wybrany = pyqtSignal(int)
    KIERUNKI = ['sfera Fibonacciego (jak widok 3D)', 'losowe (stałe ziarno)', 'PCA latentu sekwencji (3 składowe)',
                'półsfera Fibonacciego (z > 0)', 'jak widok 3D nad obrazem (te same kierunki, PCA i kanały)']
    KOLORY = ['kolor najsilniejszego mikrowektora (jak w widoku 3D)', 'mieszanka kolorów mikrowektorów (ważona długością)',
              'wg położenia patcha w siatce (kolumna = barwa, wiersz = jasność)', 'narzędzie (czerwone) i tło (szare)',
              'wg pewności ViT w patchu']

    def __init__(self, kanaly_j128=None):
        super().__init__()
        self.j128 = None if kanaly_j128 is None else np.asarray(kanaly_j128)
        self.Z = None                     # [T, 600, 768] float16
        self.maski = None                 # [T, 600] bool albo None
        self.pewnosci = None              # [T, 600] albo None
        self.t, self.token = 0, None
        self.P = None                     # [T, 600, 3] położenia
        self.kolory_kan = kolory_kanalow()
        self._pca, self._opis_pca = None, ''
        self._wb, self.wzor = None, None             # baza wektorów z widoku 3D (tryb „jak widok 3D”)
        self.odniesienie = None                      # [600, 768] odejmowane od latentu każdej klatki (albo None)
        self._poz, self._opis_poz = None, ''
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)

        self.panel = PanelOpcji(lay, ['Siły i kierunki', 'Wygląd', 'Mikrowektory', 'Atencja'])

        # Siły i kierunki: skąd siły | co odjąć z latentu
        w = self.panel.wiersz('Siły i kierunki')
        self.combo_kier = dodaj(w, 'Kierunki kanałów:', WaskiCombo(13),
                                'Stały kierunek d_c każdego z 768 kanałów (makrowektor):\n'
                                '- sfera Fibonacciego: te same kierunki co w widoku 3D nad obrazem (pełna sfera),\n'
                                '- losowe: 768 losowych kierunków jednostkowych (projekcja losowa),\n'
                                '- PCA: kierunek kanału = jego ładunki w 3 głównych składowych latentu sekwencji;\n'
                                '- półsfera Fibonacciego: jak sfera, ale wszystkie kierunki mają z > 0 (w górę);\n'
                                '- jak widok 3D nad obrazem: dokładnie te kierunki, kolory, kanały i baza PCA, którymi\n'
                                '  rysowane są mikrowektory cząsteczek nad obrazem (sfera albo półsfera stamtąd); mikrowektor\n'
                                '  ma w obu widokach ten sam kierunek, różni się tylko długością. Domyślne w zakładce filmu.\n'
                                '  największy rozrzut cząstek, ale kierunki zależą od sekwencji.')
        self.combo_kier.addItems(self.KIERUNKI); self.combo_kier.setCurrentIndex(2)
        self.combo_kan = dodaj(w, 'Siły z kanałów:', WaskiCombo(9), 'Które kanały pchają cząstkę (pozostałe mają siłę 0).')
        self.combo_kan.addItems(['wszystkie 768', 'tylko J128 (narzędzie)', 'wszystkie poza J128'])
        self.combo_pca = dodaj(w, 'PCA wektorów:', WaskiCombo(9),
                               'Zamiast 768 mikrowektorów kanałów cząstka ma k mikrowektorów = k głównych składowych\n'
                               '(PCA) latentu sekwencji, liczonych z kanałów wybranych w „Siły z kanałów”.\n'
                               'Siła składowej j = (z − μ) · w_j (μ = średni latent sekwencji, zawsze odejmowany),\n'
                               'kierunek = d_j z „Kierunki kanałów” policzony dla k wektorów.\n'
                               'Przy kierunkach PCA położenie to po prostu 3 pierwsze składowe.')
        self.combo_pca.addItems(['bez PCA (768 kanałów)'] + [f'{k} składowych' for k in PCA_SKLADOWE])
        w = self.panel.wiersz('Siły i kierunki')
        self.combo_poz = dodaj(w, 'Położenie patcha:', WaskiCombo(13),
                               'Odjęcie z latentu tego, co wynika z samego miejsca patcha na obrazie (przed PCA i kierunkami):\n'
                               '- średnia tego patcha w sekwencji: z − M_i(sekwencja) + g; usuwa położenie i wszystko, co\n'
                               '  w tym miejscu stoi przez całą sekwencję (nieruchome tło); zostaje to, co się zmienia;\n'
                               '- średnia tego patcha z 3000 klatek walidacji: z − M_i + g; M_i to latent „samego miejsca”\n'
                               '  uśredniony po różnych obrazach (zawiera też typowy układ sceny, np. źrenica w środku);\n'
                               '- rzut: usuwa z każdego tokenu tylko kierunki, w których różnią się miejsca (13 kierunków\n'
                               '  = 95 % wariancji położenia dla K1000), resztę latentu zostawia.\n'
                               'Dla checkpointu K1000 położenie to 39 % wariancji latentu. Liczone raz (około 15 s), potem z pliku.')
        self.combo_poz.addItems(['nie odejmuj', 'odejmij średnią patcha z sekwencji', 'odejmij średnią patcha z 3000 klatek',
                                 'usuń kierunki położenia (rzut)'])
        self.ckpt = None                     # checkpoint ViT sekwencji (ustawia zakładka filmu), do średniej położenia
        self.chk_srodek = QCheckBox('Odejmij średni latent'); self.chk_srodek.setChecked(True)
        self.chk_srodek.setToolTip('Siła kanału = z − μ (średnia sekwencji). Bez tego wszystkie cząstki odlatują razem\n'
                                   'w stronę średniej, bo po GELU prawie wszystkie wartości są dodatnie.')
        w.addWidget(self.chk_srodek)

        # Wygląd: które cząstki | jak wyglądają | ruch między klatkami
        w = self.panel.wiersz('Wygląd')
        self.combo_ile = dodaj(w, 'Cząstek:', WaskiCombo(10),
                               'Ile cząstek rysować: co n-ty patch w wierszach i w kolumnach siatki 20 x 30 (równo po obrazie).\n'
                               'Łączy się z „tylko cząstki narzędzia”. PCA i skala liczą się dalej ze wszystkich 600.')
        self.combo_ile.addItems(opisy_czasteczek())
        self.chk_tylko_maska = QCheckBox('Tylko cząstki narzędzia')
        self.chk_tylko_maska.setToolTip('Pokazuje tylko patche w obrysie narzędzia bieżącej klatki.')
        w.addWidget(self.chk_tylko_maska)
        w = self.panel.wiersz('Wygląd')
        self.combo_kolor = dodaj(w, 'Kolor cząstek:', WaskiCombo(11),
                                 'Jak barwić 600 cząstek (i ich ślady).\n'
                                 '- najsilniejszy mikrowektor: kolor kanału (składowej PCA) o największej długości\n'
                                 '  mikrowektora tej cząstki w bieżącej klatce; te same kolory co wektory w widoku 3D;\n'
                                 '- mieszanka: średnia kolorów wszystkich mikrowektorów ważona ich długością;\n'
                                 '- położenie w siatce, narzędzie/tło, pewność ViT: kolory niezależne od wektorów.')
        self.combo_kolor.addItems(self.KOLORY)
        self.spin_rozmiar = dodaj(w, 'Rozmiar:', QDoubleSpinBox(), 'Średnica cząstek w pikselach ekranu.')
        self.spin_rozmiar.setRange(1, 30); self.spin_rozmiar.setValue(6)
        self.chk_czastki = QCheckBox('Pokaż cząstki'); self.chk_czastki.setChecked(True)
        self.chk_czastki.setToolTip('Wyłączone: bez 600 cząstek i ich śladów; zostają mikrowektory wybranego patcha,\n'
                                    'jego cząstka, gwiazda kierunków i strzałki atencji.')
        w.addWidget(self.chk_czastki)
        self.chk_gwiazda = QCheckBox('Gwiazda kierunków'); self.chk_gwiazda.setChecked(True)
        self.chk_gwiazda.setToolTip('Gwiazda 768 kierunków d_c w środku układu (kolor kanału jak w widoku 3D).')
        w.addWidget(self.chk_gwiazda)
        w = self.panel.wiersz('Wygląd')
        self.combo_mies_waga = dodaj(w, 'Mieszanka kolorów, waga:', WaskiCombo(12),
                                     'Dla koloru „mieszanka kolorów mikrowektorów”: kolor cząstki = średnia kolorów jej\n'
                                     'mikrowektorów (te same kolory co wektory w widoku 3D) ważona siłą mikrowektora.\n'
                                     'Liczona w każdej klatce, więc kolor zmienia się, gdy cząstka się rusza.\n'
                                     '- |wartość|: długość mikrowektora;\n'
                                     '- wartość²: mocniej faworyzuje najsilniejsze;\n'
                                     '- wartość⁴: prawie sam najsilniejszy, ale płynnie;\n'
                                     '- tylko dodatnie: mikrowektory skierowane zgodnie ze swoim kierunkiem (siła > 0).')
        self.combo_mies_waga.addItems(['|wartość| (długość)', 'wartość²', 'wartość⁴', 'tylko dodatnie'])
        self.spin_mies_nasyc = dodaj(w, 'Nasycenie:', QDoubleSpinBox(),
                                     'Mieszanka wielu kolorów szarzeje, więc kolor jest automatycznie oddalany od szarości\n'
                                     '(98 % cząstek w pełnej palecie); ten mnożnik wzmacnia (> 1) albo osłabia (< 1).')
        self.spin_mies_nasyc.setRange(.1, 10); self.spin_mies_nasyc.setSingleStep(.1); self.spin_mies_nasyc.setValue(1)
        self.chk_mies_jasn = QCheckBox('jasność wg siły')
        self.chk_mies_jasn.setToolTip('Cząstki o większej łącznej sile mikrowektorów jaśniejsze (względem 99. centyla w klatce);\n'
                                      'wyłączone: każda cząstka w pełnej jasności swojej barwy.')
        w.addWidget(self.chk_mies_jasn)
        w = self.panel.wiersz('Wygląd')
        self.spin_slad = dodaj(w, 'Ślad (klatek):', QSpinBox(), 'Ile poprzednich klatek pokazuje linia toru każdej cząstki. 0 = bez śladu.')
        self.spin_slad.setRange(0, 200); self.spin_slad.setValue(8)
        self.spin_ema = dodaj(w, 'Wygładzanie EMA α:', QDoubleSpinBox(),
                              'Wygładza tory cząstek średnią wykładniczą po klatkach: wygładzone(t) = α · bieżące(t)\n'
                              '+ (1 − α) · wygładzone(t − 1), od pierwszej klatki sekwencji. 1 = bez wygładzania;\n'
                              '0,3 = mocno (pamięć około 3 klatek), 0,1 = bardzo mocno (około 10 klatek).\n'
                              'Wygładzane są siły mikrowektorów (liniowo), więc mikrowektory wybranej cząstki i ich\n'
                              'suma (wypadkowa) nadal kończą się dokładnie w wygładzonej cząstce. Opóźnia ruch.')
        self.spin_ema.setRange(.02, 1); self.spin_ema.setSingleStep(.05); self.spin_ema.setDecimals(2); self.spin_ema.setValue(1)
        self.spin_przejscie = dodaj(w, 'Przelot:', QSpinBox(), 'Czas płynnego przelotu cząstek do nowej klatki (ms). 0 = skok.')
        self.spin_przejscie.setRange(0, 3000); self.spin_przejscie.setValue(250); self.spin_przejscie.setSuffix(' ms')
        w = self.panel.wiersz('Wygląd')
        self.combo_zdj = dodaj(w, 'Zdjęcie w przestrzeni:', WaskiCombo(12),
                               'Zdjęcie zdeformowane do położeń cząstek, żeby wiedzieć, co jest czym:\n'
                               'siatka 20 x 30 patchy, w której każdy węzeł leży tam, gdzie jest cząstka tego patcha,\n'
                               'a sąsiedzi na obrazie są połączeni. Między węzłami położenie i kolor są interpolowane.\n'
                               'Gdzie siatka jest gładka, sąsiednie patche mają podobny latent; gdzie się rozdziera albo\n'
                               'zawija, latent zmienia się skokowo (np. na krawędzi narzędzia).\n'
                               '- powierzchnia: półprzezroczyste zdjęcie rozpięte na siatce;\n'
                               '- siatka linii: same linie wierszy i kolumn patchy (lżejsze, cząstki lepiej widać).')
        self.combo_zdj.addItems(['brak', 'powierzchnia', 'siatka linii', 'powierzchnia i siatka'])
        self.combo_zdj.setCurrentIndex(3)
        self.combo_zdj_kolor = dodaj(w, 'Kolor:', WaskiCombo(11),
                                     'Kolory zdjęcia albo kolory cząstek (jak w „Kolor cząstek”).')
        self.combo_zdj_kolor.addItems(['kolory zdjęcia', 'kolory cząstek', 'zdjęcie, narzędzie na czerwono'])
        self.spin_zdj_krycie = dodaj(w, 'Krycie:', QDoubleSpinBox(), 'Krycie zdjęcia w przestrzeni (0..1).')
        self.spin_zdj_krycie.setRange(.02, 1); self.spin_zdj_krycie.setSingleStep(.05); self.spin_zdj_krycie.setValue(.45)
        self.spin_zdj_gest = dodaj(w, 'Gęstość:', QSpinBox(),
                                   'Ile punktów siatki na bok patcha (1 = tylko środki patchy; 4 = gładsze zdjęcie).')
        self.spin_zdj_gest.setRange(1, 8); self.spin_zdj_gest.setValue(4)
        self.chk_podpisy = QCheckBox('Podpisy'); self.chk_podpisy.setChecked(True)
        self.chk_podpisy.setToolTip('Podpisy przy cząstkach rogów obrazu (lewy górny itd.), środka obrazu i środka\n'
                                    'narzędzia, oraz ramka wokół brzegu obrazu (niebieska: góra, żółta: lewa strona).')
        w.addWidget(self.chk_podpisy)
        self.rgb = None

        # Mikrowektory wybranego patcha
        w = self.panel.wiersz('Mikrowektory')
        self.combo_mikro = dodaj(w, 'Wybrany patch:', WaskiCombo(15),
                                 'Jak pokazać 768 mikrowektorów (z_c − μ_c) · s · d_c wybranej cząstki. Oba sposoby\n'
                                 'to to samo dodawanie wektorów, wynik jest ten sam: położenie cząstki.\n'
                                 '- ze wspólnego środka: każdy mikrowektor wychodzi z tego samego punktu (środek układu),\n'
                                 '  biała strzałka = ich suma (wypadkowa), kończy się w cząstce;\n'
                                 '- łańcuch: wektory dokładane koniec do początku (od najsilniejszego), ostatni kończy się w cząstce;\n'
                                 '- oba naraz.')
        self.combo_mikro.addItems(['ze wspólnego środka + wypadkowa', 'łańcuch (koniec do początku)', 'oba', 'brak'])
        self.spin_mikro = dodaj(w, 'Powiększ:', QDoubleSpinBox(),
                                'Mnożnik długości mikrowektorów ze wspólnego środka (tylko rysunek, wypadkowa bez zmian).\n'
                                'Pojedynczy mikrowektor jest zwykle dużo krótszy od wypadkowej.')
        self.spin_mikro.setRange(1, 500); self.spin_mikro.setValue(1); self.spin_mikro.setPrefix('× ')
        self.chk_atencja = QCheckBox('Atencja na cząstkach')
        self.chk_atencja.setToolTip('Mapa atencji ViT bieżącej klatki na cząstkach: kolor (inferno) i rozmiar według\n'
                                    'wartości mapy patcha (0..1, jak na obrazie w widoku nad obrazem), oraz strzałki\n'
                                    'źródło -> cel jako linie między cząstkami w przestrzeni makrowektorów\n'
                                    '(źródło patrzy na cel, czyli ciągnie z niego). Opcje mapy i strzałek niżej, jak w 3D.\n'
                                    'Działa w zakładce filmu (atencja liczona z obrazu każdej klatki).')
        self.spin_wzm = dodaj(self.panel.wiersz('Atencja'), 'Powiększ wg mapy:', QDoubleSpinBox(),
                              'Rozmiar cząstki = rozmiar · (0,4 + ta wartość · mapa). 0 = sam kolor.')
        self.spin_wzm.setRange(0, 10); self.spin_wzm.setSingleStep(.5); self.spin_wzm.setValue(2.5)
        w = self.panel.wiersz('Atencja')
        self.combo_zbior = dodaj(w, 'Mapa z patchy:', WaskiCombo(14),
                                 'Dla map „z wybranego patcha” i „do wybranego patcha”: z którego zbioru patchy (średnia):\n'
                                 '- wybrany patch: kliknięty albo śledzony czubek,\n'
                                 '- wszystkie widoczne cząstki (z „Cząstek” i „tylko cząstki narzędzia”),\n'
                                 '- patche narzędzia z maski bieżącej klatki,\n'
                                 '- lista: numery patchy w polu obok, np. 315, 316, 340-345 (wiersz · 30 + kolumna).')
        self.combo_zbior.addItems(['wybrany patch', 'wszystkie widoczne cząstki', 'patche narzędzia (maska klatki)',
                                   'lista patchy (pole obok)'])
        self.edit_lista = QLineEdit(); self.edit_lista.setPlaceholderText('np. 315, 316, 340-345'); self.edit_lista.setMaximumWidth(170)
        self.edit_lista.setToolTip(self.combo_zbior.toolTip()); w.addWidget(self.edit_lista)
        w = self.panel.wiersz('Atencja')
        self.chk_objetosc = QCheckBox('Heatmapa całej przestrzeni')
        self.chk_objetosc.setToolTip('Mapa atencji rozlana gradientem po przestrzeni makrowektorów: na siatce 48 x 48 x 48\n'
                                     'wartość = gładka średnia mapy pobliskich cząstek (rozmycie Gaussa z normalizacją),\n'
                                     'barwa inferno, rysowana jako świecąca mgła (nie zasłania cząstek).\n'
                                     'Mapę i zbiór patchy wybiera się jak dla cząstek; działa też bez „Atencja na cząstkach”.')
        w.addWidget(self.chk_objetosc)
        self.combo_zasieg = dodaj(w, 'Zasięg:', WaskiCombo(12),
                                  'Przy cząstkach: mgła tylko tam, gdzie są cząstki (krycie rośnie z ich gęstością).\n'
                                  'Cała przestrzeń: cały sześcian zamalowany, dalej od cząstek wartość najbliższych.')
        self.combo_zasieg.addItems(['przy cząstkach', 'cała przestrzeń'])
        self.spin_rozmycie = dodaj(w, 'Rozmycie:', QDoubleSpinBox(), 'Szerokość gradientu: odchylenie Gaussa w % boku sześcianu.')
        self.spin_rozmycie.setRange(1, 40); self.spin_rozmycie.setValue(6); self.spin_rozmycie.setSuffix(' %')
        self.spin_krycie_obj = dodaj(w, 'Krycie mgły:', QDoubleSpinBox(), 'Jasność mgły (0..1).')
        self.spin_krycie_obj.setRange(0, 1); self.spin_krycie_obj.setSingleStep(.05); self.spin_krycie_obj.setValue(.35)
        from opcje_atencji import OpcjeAtencji
        self.opcje = OpcjeAtencji(na_poczatku=(self.chk_atencja,))
        self.panel.strona('Atencja').addWidget(self.opcje)
        self.atencja_fn = None
        b = rola(QPushButton('Kamera od nowa'), 'narzedzie'); b.clicked.connect(self._kamera); self.panel.pasek.addWidget(b)
        self.btn_odepnij = rola(QPushButton('Odepnij'), 'narzedzie'); self.panel.pasek.addWidget(self.btn_odepnij)
        self.odpinanie = Odpinanie(self, 'Przestrzeń makrowektorów', self.btn_odepnij)
        b.setToolTip('Widok całego układu z ukosa.')
        self.panel.zakoncz()

        self.widok = WidokGL()
        dodaj_nawigacje(self, self.panel, self.widok)       # obrót 360, kroki, ujęcia, lustro
        self.widok.setBackgroundColor(pg.mkColor(10, 10, 14))
        self.widok.setMinimumHeight(400)
        lay.addWidget(self.widok, 1)
        self.etykieta = etykieta_statusu(QLabel('Wczytaj sekwencję w zakładce filmu.'))     # stan widoku, pod nim
        lay.addWidget(self.etykieta)
        v = self.widok
        osie = gl.GLAxisItem(); osie.setSize(60, 60, 60); v.addItem(osie)
        self.siatka = gl.GLGridItem(); self.siatka.setSize(300, 300); self.siatka.setSpacing(25, 25); self.siatka.setColor((255, 255, 255, 25))
        v.addItem(self.siatka)
        self.gwiazda = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', glOptions='translucent')
        v.addItem(self.gwiazda)
        self.slady = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', glOptions='translucent')
        v.addItem(self.slady)
        self.lancuch = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='line_strip', width=2, glOptions='translucent')
        v.addItem(self.lancuch)
        self.mikro = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=1.5, glOptions='translucent')
        self.mikro.setVisible(False)
        v.addItem(self.mikro)
        self.wypadkowa = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=4, color=(1, 1, 1, .95),
                                           glOptions='translucent')
        self.wypadkowa.setVisible(False)
        v.addItem(self.wypadkowa)
        self.luki = LinieGrube(v)                         # strzałki atencji, grubość według siły
        self.groty = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=7, pxMode=True, glOptions='translucent')
        self.groty.setVisible(False); v.addItem(self.groty)
        self.objetosc = gl.GLVolumeItem(np.zeros((2, 2, 2, 4), np.uint8), sliceDensity=2, smooth=True, glOptions='additive')
        self.objetosc.setVisible(False); v.addItem(self.objetosc)
        self.punkty = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=6, pxMode=True, glOptions='translucent')
        v.addItem(self.punkty)
        self.wybrany = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=16, color=(1, 1, 1, 1), pxMode=True)
        self.wybrany.setVisible(False)
        v.addItem(self.wybrany)
        self.przyczyny = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=3, glOptions='translucent')
        self.przyczyny.setVisible(False); v.addItem(self.przyczyny)      # strzałki przyczyn ruchu (zakładka uczenia)
        self.zdj_pow = gl.GLMeshItem(vertexes=np.zeros((3, 3), np.float32), faces=np.array([[0, 1, 2]]), smooth=False,
                                     glOptions='translucent')
        self.zdj_pow.setVisible(False); v.addItem(self.zdj_pow)         # zdjęcie zdeformowane do położeń cząstek
        self.zdj_lin = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=1, glOptions='translucent')
        self.zdj_lin.setVisible(False); v.addItem(self.zdj_lin)
        self.brzeg = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=2.5, glOptions='translucent')
        self.brzeg.setVisible(False); v.addItem(self.brzeg)
        self.podpisy = []
        self.znaczniki = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=10, pxMode=True)
        self.znaczniki.setVisible(False); v.addItem(self.znaczniki)      # dodatkowe kropki (np. średnia narzędzia)
        self._kamera()

        self._timer = QTimer(self); self._timer.setInterval(16); self._timer.timeout.connect(self._krok_przelotu)
        self._od = self._do = None; self._u = 1.0
        self.combo_kier.currentIndexChanged.connect(lambda *_: self._tryb_kierunkow())
        for s in (self.combo_kier.currentIndexChanged, self.combo_kan.currentIndexChanged, self.chk_srodek.toggled,
                  self.combo_pca.currentIndexChanged, self.combo_poz.currentIndexChanged, self.spin_ema.valueChanged):
            s.connect(lambda *_: self._przelicz())
        for s in (self.combo_mies_waga.currentIndexChanged, self.spin_mies_nasyc.valueChanged, self.chk_mies_jasn.toggled):
            s.connect(lambda *_: self._rysuj(skok=True))
        for s in (self.combo_kolor.currentIndexChanged, self.spin_slad.valueChanged, self.spin_rozmiar.valueChanged,
                  self.combo_mikro.currentIndexChanged, self.spin_mikro.valueChanged,
                  self.chk_atencja.toggled, self.opcje.zmiana, self.spin_wzm.valueChanged, self.chk_czastki.toggled,
                  self.combo_zbior.currentIndexChanged, self.edit_lista.editingFinished, self.chk_objetosc.toggled,
                  self.combo_zasieg.currentIndexChanged, self.spin_rozmycie.valueChanged, self.spin_krycie_obj.valueChanged,
                  self.chk_gwiazda.toggled, self.chk_tylko_maska.toggled, self.combo_ile.currentIndexChanged,
                  self.combo_zdj.currentIndexChanged, self.combo_zdj_kolor.currentIndexChanged,
                  self.spin_zdj_krycie.valueChanged, self.spin_zdj_gest.valueChanged, self.chk_podpisy.toggled):
            s.connect(lambda *_: self._rysuj(skok=True))

    def _kamera(self):
        # ta sama orientacja co widok 3D nad obrazem: x w prawo, y w głąb, z w górę (układ prawoskrętny)
        self.widok.setCameraPosition(pos=QVector3D(0, 0, 0), distance=420, elevation=38, azimuth=-90)

    def ustaw_wzor(self, widok3d):
        """Widok 3D nad obrazem jako wzór kierunków (tryb „jak widok 3D”, domyślny w zakładce filmu) i kamery."""
        self.wzor = widok3d
        for s in (widok3d.combo_ksztalt.currentIndexChanged, widok3d.combo_pca.currentIndexChanged,
                  widok3d.combo_kan.currentIndexChanged):
            s.connect(lambda *_: self._przelicz() if self.combo_kier.currentIndex() == 4 else None)
        b = rola(QPushButton('Kamera jak 3D'), 'narzedzie')
        b.setToolTip('Ustawia kamerę pod tym samym kątem co widok 3D nad obrazem (te same osie x, y, z).')
        b.clicked.connect(lambda: self.widok.setCameraPosition(elevation=widok3d.widok.opts['elevation'],
                                                               azimuth=widok3d.widok.opts['azimuth']))
        self.panel.pasek.addWidget(b)
        self.combo_kier.setCurrentIndex(4)

    def _tryb_kierunkow(self):
        jak3d = self.combo_kier.currentIndex() == 4 and self.wzor is not None
        for w in (self.combo_kan, self.combo_pca):              # w trybie „jak 3D” rządzą opcje widoku 3D
            w.setEnabled(not jak3d)

    # ------------------------------------------------------------ dane
    def ustaw_sekwencje(self, Z, maski=None, pewnosci=None):
        """Z: [T, 600, 768] (float16 albo float32); maski: [T, 600] bool; pewnosci: [T, 600] w 0..1."""
        self.Z = Z
        self.maski = None if maski is None else np.asarray(maski, bool)
        self.pewnosci = None if pewnosci is None else np.asarray(pewnosci, np.float32)
        self.t = 0
        self._przelicz()

    def ustaw_maski(self, maski):
        self.maski = None if maski is None else np.asarray(maski, bool)
        self._rysuj(skok=True)

    def _kanaly_maska(self):
        m = np.ones(768, bool)
        if self.j128 is not None and self.combo_kan.currentIndex():
            m[:] = self.combo_kan.currentIndex() == 2
            m[self.j128] = self.combo_kan.currentIndex() == 1
        return m

    def _k_pca(self):
        i = self.combo_pca.currentIndex()
        return PCA_SKLADOWE[i - 1] if i > 0 else 0

    def _z(self, t):
        """Latent klatki t [600, 768] float32, z odjętym położeniem patcha, jeśli wybrano."""
        z = self.Z[t].astype(np.float32)
        if self.odniesienie is not None:                   # np. stan końcowy uczenia (zakładka „Uczenie ViT”)
            z = z - self.odniesienie
        p = self._poz
        if p is None:
            return z
        if p[0] == 'rzut':
            _, g, U = p
            return z - ((z - g) @ U) @ U.T
        _, M, g = p
        return z - M + g

    def _przygotuj_polozenie(self):
        """Wybrany sposób odjęcia położenia -> self._poz (None, ('srednia', M, g) albo ('rzut', g, U))."""
        i = self.combo_poz.currentIndex()
        self._poz, self._opis_poz = None, ''
        if i == 0:
            return
        if i == 1:
            M = np.mean([self.Z[t].astype(np.float32) for t in range(len(self.Z))], 0)       # [600, 768]
            self._poz = ('srednia', M, M.mean(0))
            self._opis_poz = '; odjęta średnia każdego patcha z sekwencji'
            return
        import polozenie
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        import silnik
        ckpt = self.ckpt or silnik.VIT_CHECKPOINT
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            w = polozenie.srednia_pozycji(ckpt, postep=lambda k, n: (self.etykieta.setText(
                f'liczę latent położenia: {k}/{n} klatek walidacji...'), QApplication.processEvents()))
        finally:
            QApplication.restoreOverrideCursor()
        if i == 2:
            self._poz = ('srednia', w['M'], w['g'])
            self._opis_poz = f'; odjęta średnia każdego patcha z {int(w["n"])} klatek (położenie = {100 * float(w["udzial"]):.0f} % wariancji)'
        else:
            self._poz = ('rzut', w['g'], w['U'])
            self._opis_poz = f'; usunięte {int(w["r"])} kierunków położenia (95 % jego wariancji)'

    def rzut_delty(self, dz):
        """Zmiana latentu dz [..., 768] -> przesunięcie cząstki w przestrzeni makro [..., 3] (ta sama baza, kierunki
        i skala co położenia; średnia i położenie patcha nie grają roli, bo to różnica)."""
        dz = np.asarray(dz, np.float32)
        if self._wb is not None:
            kan, p3 = self._wb
            F = dz[..., kan] if p3 is None else dz[..., kan] @ p3[1]
        elif self._pca is not None:
            mu, W, maska = self._pca
            F = dz[..., maska] @ W
        else:
            F = dz
        return F @ self.D * self.s

    def rysuj_przyczyny(self, odcinki=None, kolory=None):
        """Dodatkowe odcinki [2n, 3] z kolorami [2n, 4] (strzałki przyczyn ruchu); None chowa."""
        if odcinki is None or len(odcinki) == 0:
            self.przyczyny.setVisible(False); return
        self.przyczyny.setData(pos=np.ascontiguousarray(odcinki, np.float32), color=np.ascontiguousarray(kolory, np.float32))
        self.przyczyny.setVisible(True)

    def ustaw_obraz(self, rgb480):
        """Zdjęcie bieżącej klatki [320, 480, 3] uint8 (do zdeformowanego zdjęcia w przestrzeni); wołać przed pokaz_klatke."""
        self.rgb = None if rgb480 is None else np.asarray(rgb480)[:320, :480, :3]

    def _rysuj_zdjecie(self, P, kol):
        """Zdjęcie rozpięte na siatce 20 x 30 położeń cząstek P [600, 3] (interpolacja dwuliniowa między węzłami)."""
        tryb = self.combo_zdj.currentIndex()
        if tryb == 0 or (self.rgb is None and self.combo_zdj_kolor.currentIndex() != 1):
            self.zdj_pow.setVisible(False); self.zdj_lin.setVisible(False)
        else:
            from scipy import ndimage
            n = self.spin_zdj_gest.value()
            R, C = 19 * n + 1, 29 * n + 1
            r, c = np.linspace(0, 19, R), np.linspace(0, 29, C)
            rr, cc = np.meshgrid(r, c, indexing='ij')
            G = P.reshape(20, 30, 3)
            V = np.stack([ndimage.map_coordinates(G[..., i], [rr, cc], order=1) for i in range(3)], -1)   # [R, C, 3]
            k = self.combo_zdj_kolor.currentIndex()
            if k == 1:
                K = kol[:, :3].reshape(20, 30, 3)
                K = np.stack([ndimage.map_coordinates(K[..., i], [rr, cc], order=1) for i in range(3)], -1)
            else:
                y = np.clip((8 + rr * 16).round().astype(int), 0, 319); x = np.clip((8 + cc * 16).round().astype(int), 0, 479)
                K = self.rgb[y, x].astype(np.float32) / 255
                if k == 2 and self.maski is not None:
                    m = self.maski[self.t].reshape(20, 30).astype(np.float32)
                    mm = ndimage.map_coordinates(m, [rr, cc], order=1)[..., None]
                    K = K * (1 - .7 * mm) + np.array([1, .15, .1]) * .7 * mm
            a = self.spin_zdj_krycie.value()
            if tryb in (1, 3):
                if getattr(self, '_twarze_n', None) != (R, C):
                    i = np.arange(R * C).reshape(R, C)
                    q = np.stack([i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1]], -1).reshape(-1, 4)
                    self._twarze = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int32)
                    self._twarze_n = (R, C)
                kv = np.concatenate([K.reshape(-1, 3), np.full((R * C, 1), a)], 1).astype(np.float32)
                self.zdj_pow.setMeshData(vertexes=V.reshape(-1, 3).astype(np.float32), faces=self._twarze, vertexColors=kv)
                self.zdj_pow.setVisible(True)
            else:
                self.zdj_pow.setVisible(False)
            if tryb in (2, 3):                                           # linie wierszy i kolumn patchy
                Vp, Kp = V[::n, ::n], K[::n, ::n]                          # środki patchy [20, 30]
                seg = np.concatenate([np.stack([Vp[:, :-1], Vp[:, 1:]], 2).reshape(-1, 3),
                                      np.stack([Vp[:-1], Vp[1:]], 2).reshape(-1, 3)])
                ks = np.concatenate([np.stack([Kp[:, :-1], Kp[:, 1:]], 2).reshape(-1, 3),
                                     np.stack([Kp[:-1], Kp[1:]], 2).reshape(-1, 3)])
                al = min(1., a * 1.6) if tryb == 2 else min(1., a + .25)
                ks = np.concatenate([np.clip(ks * 1.3, 0, 1), np.full((len(ks), 1), al)], 1)
                self.zdj_lin.setData(pos=seg.astype(np.float32), color=ks.astype(np.float32))
                self.zdj_lin.setVisible(True)
            else:
                self.zdj_lin.setVisible(False)
        self._rysuj_podpisy(P)

    def _rysuj_podpisy(self, P):
        if not self.chk_podpisy.isChecked():
            self.brzeg.setVisible(False)
            for it in self.podpisy:
                it.setVisible(False)
            return
        G = P.reshape(20, 30, 3)
        gora, dol, lewa, prawa = G[0], G[-1], G[:, 0], G[:, -1]
        seg, kol = [], []
        for L, c in ((gora, (.3, .6, 1, .9)), (lewa, (1, .85, .2, .9)), (dol, (.6, .6, .6, .5)), (prawa, (.6, .6, .6, .5))):
            seg.append(np.stack([L[:-1], L[1:]], 1).reshape(-1, 3)); kol += [c] * (2 * (len(L) - 1))
        self.brzeg.setData(pos=np.concatenate(seg).astype(np.float32), color=np.array(kol, np.float32))
        self.brzeg.setVisible(True)
        napisy = [(0, 'lewy górny róg'), (29, 'prawy górny'), (570, 'lewy dolny'), (599, 'prawy dolny'), (315, 'środek obrazu')]
        pozycje = [(P[t], tekst) for t, tekst in napisy]
        if self.maski is not None and self.maski[self.t].any():
            pozycje.append((P[self.maski[self.t]].mean(0), 'narzędzie (średnia)'))
        while len(self.podpisy) < len(pozycje):                          # napisy tworzone raz, potem tylko przesuwane
            it = gl.GLTextItem(color=(235, 235, 235, 230)); self.widok.addItem(it); self.podpisy.append(it)
        for it, (pos, tekst) in zip(self.podpisy, pozycje):
            it.setData(pos=np.asarray(pos, float), text=tekst); it.setVisible(True)
        for it in self.podpisy[len(pozycje):]:
            it.setVisible(False)

    def rysuj_znaczniki(self, pos=None, kolory=None, rozmiary=None):
        """Dodatkowe kropki [n, 3] z kolorami [n, 4] i rozmiarami w pikselach [n]; None chowa."""
        if pos is None or len(pos) == 0:
            self.znaczniki.setVisible(False); return
        self.znaczniki.setData(pos=np.ascontiguousarray(pos, np.float32), color=np.ascontiguousarray(kolory, np.float32),
                               size=np.ascontiguousarray(rozmiary, np.float32))
        self.znaczniki.setVisible(True)

    def _z_na_sily(self, z):
        """Latent [..., 768] po korekcie położenia -> siły [..., K] w tej samej bazie co widok 3D (tryb „jak 3D”)."""
        kan, p3 = self._wb
        zk = z[..., kan]
        return zk - self.mu[kan] if p3 is None else (zk - p3[0]) @ p3[1]

    def sily(self, t):
        """Siły mikrowektorów wszystkich cząstek w klatce t: [600, K] (K = 768 kanałów albo k składowych PCA)."""
        z = self._z(t)
        if self._wb is not None:
            return self._z_na_sily(z)
        if self._pca is None:
            return z - self.mu
        mu, W, maska = self._pca
        return (z[:, maska] - mu) @ W

    def _sily_wierszy(self, t0, t1, tok):
        """Siły mikrowektorów jednej cząstki tok w klatkach t0..t1 włącznie: [t1 - t0 + 1, K] (jak sily(t)[tok])."""
        z = self.Z[t0:t1 + 1, tok].astype(np.float32)                          # [n, 768]
        p = self._poz
        if p is not None and p[0] == 'rzut':
            _, g, U = p
            z = z - ((z - g) @ U) @ U.T
        elif p is not None:
            _, M, g = p
            z = z - M[tok] + g
        if self._wb is not None:
            return self._z_na_sily(z)
        if self._pca is None:
            return z - self.mu
        mu, W, maska = self._pca
        return (z[:, maska] - mu) @ W

    def sily_tokenu(self, t, tok):
        """Siły mikrowektorów cząstki tok w klatce t [K], po wygładzeniu EMA (gdy α < 1), dokładnie jak w self.P."""
        a = self.spin_ema.value()
        if a >= 1:
            return self._sily_wierszy(t, t, tok)[0]
        X = self._sily_wierszy(0, t, tok)                                       # EMA od pierwszej klatki
        w = a * (1 - a) ** np.arange(t, -1, -1, dtype=np.float64)
        w[0] = (1 - a) ** t
        return (w[:, None] * X).sum(0).astype(np.float32)

    def _przelicz(self):
        """Siły (kanały albo składowe PCA), kierunki D [K, 3], skala s i położenia cząstek we wszystkich klatkach."""
        if self.Z is None:
            return
        T = len(self.Z)
        krok = max(1, T // 50)
        self._przygotuj_polozenie()
        maska = self._kanaly_maska()
        kp = self._k_pca()
        self._opis_pca = ''
        self.mu = np.zeros(768, np.float32)
        k = self.combo_kier.currentIndex()
        wb = self.wzor.baza_wektorow() if (k == 4 and self.wzor is not None) else None
        self._wb = None
        if wb is not None:                                  # te same kanały, PCA i kierunki co widok 3D nad obrazem
            kan, p3, D, kol = wb
            self._pca = None
            if p3 is None and self.chk_srodek.isChecked():
                self.mu = np.mean([self._z(t).mean(0) for t in range(T)], 0)
            self._wb = (np.asarray(kan), p3)
            self._opis_pca = ('; kierunki, kanały i PCA jak w widoku 3D nad obrazem ('
                              + (f'{len(D)} składowych PCA' if p3 is not None else f'{len(D)} kanałów') + ')')
            self.D, self.kolory_kan = np.asarray(D, np.float32).copy(), np.asarray(kol, np.float32).copy()
        elif kp:
            X = np.concatenate([self._z(t)[:, maska] for t in range(0, T, krok)])
            mu, W, udz = pca(X, kp)
            self._pca = (mu, W, maska)
            K = W.shape[1]
            self._opis_pca = (f'; PCA {K} składowych z {int(maska.sum())} kanałów wyjaśnia '
                              f'{100 * float(udz.sum()):.1f} % wariancji')
        else:
            self._pca = None
            K = 768
            if self.chk_srodek.isChecked():
                self.mu = np.mean([self._z(t).mean(0) for t in range(T)], 0)
        if wb is None:
            if k in (0, 3, 4):
                D = kierunki(k == 3, K)                                      # 3: półsfera; 4 bez widoku 3D: sfera
            elif k == 1:
                D = np.random.default_rng(0).standard_normal((K, 3)).astype(np.float32)
                D /= np.linalg.norm(D, axis=1, keepdims=True)
            else:
                X = np.concatenate([self.sily(t) for t in range(0, T, krok)])
                _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
                D = Vt[:3].T.astype(np.float32)                                # ładunki: [K, 3]
            if not kp:
                D = D * maska[:, None]                                         # kanały spoza wyboru: siła 0
            self.D = D
            self.kolory_kan = kolory_kanalow(K)
        a = self.spin_ema.value()
        dl_d = np.linalg.norm(D, axis=1)                                        # [K]: 0 dla kanałów spoza wyboru
        surowe = np.zeros((T, 600, 3), np.float32)
        self._dom = np.zeros((T, 600), np.int32)                                # najsilniejszy mikrowektor cząstki
        F = None
        for t in range(T):
            Ft = self.sily(t)                                                   # [600, K]
            F = Ft if (F is None or a >= 1) else a * Ft + (1 - a) * F           # EMA sił (liniowa) = EMA torów
            surowe[t] = F @ D
            dl = np.abs(F) * dl_d                                               # długości mikrowektorów [600, K]
            self._dom[t] = dl.argmax(1)
        r = np.linalg.norm(surowe, axis=-1)
        self.s = 100 / max(float(np.percentile(r, 99)), 1e-9)
        self.P = surowe * self.s
        self._rysuj(skok=True)

    def pokaz_klatke(self, t, token=None):
        if self.P is None:
            return
        stary_token = self.token
        self.token = token if token is not None else self.token
        stare = self._biezace() if self._od is not None else None
        v_stare = self._v_biezace() if stary_token == self.token else None      # mikrowektory z chwili startu przelotu
        self.t = int(np.clip(t, 0, len(self.P) - 1))
        if stare is not None and self.spin_przejscie.value() > 0:
            self._od, self._u = stare, 0.0
            self._v_od = v_stare
            self._t0 = time.perf_counter()
            self._timer.start()
            self._rysuj()
        else:
            self._rysuj(skok=True)

    def _gladkie_u(self):
        return self._u * self._u * (3 - 2 * self._u)                           # łagodny start i koniec

    def _biezace(self):
        if self._od is None or self._u >= 1:
            return self.P[self.t]
        return self._od + (self.P[self.t] - self._od) * self._gladkie_u()

    def _v_biezace(self):
        """Mikrowektory wybranej cząstki w tej chwili [K, 3]. W czasie przelotu przechodzą płynnie od tych z chwili
        startu do tych z nowej klatki, tym samym tempem co cząstka, więc ich suma (wypadkowa ze wspólnego środka)
        trafia w lecącą cząstkę w każdej chwili, a początek zostaje w środku układu."""
        if self.token is None or self.Z is None or self.P is None:
            return None
        v = self.sily_tokenu(self.t, self.token)[:, None] * self.D * self.s
        vo = getattr(self, '_v_od', None)
        if self._od is None or self._u >= 1:
            return v
        if vo is not None and vo.shape == v.shape:
            v = vo + (v - vo) * self._gladkie_u()
        # reszta do położenia lecącej cząstki (0 przy spójnym przelocie; inna, gdy cząstkę wybrano w trakcie
        # przelotu) rozłożona po równo na mikrowektory: suma zawsze trafia dokładnie w cząstkę
        return v + (self._biezace()[self.token] - v.sum(0)) / len(v)

    def _krok_przelotu(self):
        self._u = min(1.0, (time.perf_counter() - self._t0) * 1000 / max(1, self.spin_przejscie.value()))
        self._rysuj(tylko_punkty=True)
        if self._u >= 1:
            self._timer.stop()

    # ------------------------------------------------------------ rysowanie
    def _mieszanka_klatki(self):
        """Kolor cząstek bieżącej klatki = kolory mikrowektorów ważone siłą (combo_mies_waga): ([600, 3], siła [600]).
        Liczona przy rysowaniu (około 3 ms), z sił tej klatki; przy wygładzaniu EMA z sił niewygładzonych."""
        j = self.combo_mies_waga.currentIndex()
        klucz = (self.t, j, id(self.D), id(self.Z))
        if getattr(self, '_mies_klucz', None) != klucz:
            F = self.sily(self.t)
            dl_d = np.linalg.norm(self.D, axis=1)
            dl = np.abs(F) * dl_d
            if j in (1, 2):
                dn = dl / np.maximum(dl.max(1, keepdims=True), 1e-12)
                W = dn * dn if j == 1 else (dn * dn) ** 2
            else:
                W = dl if j == 0 else np.clip(F, 0, None) * dl_d
            sw = W.sum(1)
            m = W @ self.kolory_kan[:, :3] / np.maximum(sw, 1e-12)[:, None]
            self._mies_klucz, self._mies_wynik = klucz, (m.astype(np.float32), (dl.sum(1) if j in (1, 2) else sw))
        return self._mies_wynik

    def _kolory(self):
        i = self.combo_kolor.currentIndex()
        t = np.arange(600)
        if i in (0, 1) and getattr(self, '_dom', None) is not None and len(self._dom) == len(self.P):
            if i == 0:                                                          # kolor kanału, jak wektor w 3D
                c = self.kolory_kan[self._dom[self.t]].copy()
            else:
                m, sw = self._mieszanka_klatki()
                g = m.mean(1, keepdims=True)
                ref = max(float(np.percentile(np.abs(m - g).max(1), 98)), 1e-6)   # różnica od szarości w tej klatce
                m = np.clip(g + (m - g) * (.45 / ref) * self.spin_mies_nasyc.value(), 0, 1)
                if self.chk_mies_jasn.isChecked():
                    m = m * np.clip(sw / max(float(np.percentile(sw, 99)), 1e-12), .08, 1)[:, None]
                else:
                    m = m / np.maximum(m.max(1, keepdims=True), 1e-6)           # pełna jasność, ta sama barwa
                c = np.concatenate([m, np.ones((600, 1), np.float32)], 1)
            c[:, 3] = .9
            return c.astype(np.float32)
        i = max(0, i - 2)
        if i == 1 and self.maski is not None:
            m = self.maski[self.t]
            c = np.tile(np.array([.55, .55, .6, .55], np.float32), (600, 1)); c[m] = (1, .2, .15, 1)
            return c
        if i == 2 and self.pewnosci is not None:
            import matplotlib
            return matplotlib.colormaps['viridis'](self.pewnosci[self.t]).astype(np.float32)
        import colorsys
        return np.array([(*colorsys.hsv_to_rgb((k % KOLUMNY) / KOLUMNY, .85, .45 + .55 * (k // KOLUMNY) / 19), .9)
                         for k in t], np.float32)

    def ustaw_atencje(self, atencja_fn):
        """Źródło atencji bieżącej klatki: atencja_fn(blok_od_0) -> [głowy, 600, 600] (zakładka filmu, przed
        pokaz_klatke). Liczona dopiero, gdy włączono „Atencja na cząstkach”."""
        self.atencja_fn = atencja_fn
        self.opcje.ustaw_zrodlo(atencja_fn)

    def _zbior(self):
        """Patch albo zbiór patchy dla map „z / do wybranego” (combo_zbior); None = brak."""
        i = self.combo_zbior.currentIndex()
        if i == 1:
            return self._widoczne()
        if i == 2:
            return np.flatnonzero(self.maski[self.t]) if self.maski is not None else None
        if i == 3:
            tok = set()
            for cz in self.edit_lista.text().replace(';', ',').split(','):
                cz = cz.strip()
                try:
                    a, _, b = cz.partition('-')
                    tok.update(range(int(a), int(b or a) + 1))
                except ValueError:
                    continue
            tok = np.array(sorted(t for t in tok if 0 <= t < 600), int)
            return tok if len(tok) else None
        return self.token

    def _atencja(self):
        """(mapa [600] 0..1 albo None, strzałki (src, dst, w) albo None, opis). Liczona, gdy włączono atencję na
        cząstkach albo heatmapę przestrzeni."""
        if not (self.chk_atencja.isChecked() or self.chk_objetosc.isChecked()) or self.atencja_fn is None:
            return None, None, ''
        zb = self._zbior()
        if zb is not None and np.ndim(zb) and len(zb) == 0:
            zb = None
        A, opis = self.opcje.mapa(zb)
        strz = self.opcje.strzalki(self.token) if self.chk_atencja.isChecked() else None
        return (None if A is None else A.reshape(-1)), strz, opis

    def _rysuj_objetosc(self, P, A, tok):
        """Heatmapa objętościowa: gładka średnia mapy cząstek na siatce n^3 (Nadaraya-Watson z jądrem Gaussa)."""
        from scipy import ndimage
        import matplotlib
        n = 48
        L = max(float(np.percentile(np.abs(self.P), 99.5)) * 1.15, 1e-6)
        ix = np.clip(((P[tok] + L) / (2 * L) * n).astype(int), 0, n - 1)
        waga = np.zeros((n, n, n), np.float32); suma = np.zeros((n, n, n), np.float32)
        np.add.at(waga, tuple(ix.T), 1.0); np.add.at(suma, tuple(ix.T), A[tok].astype(np.float32))
        sig = self.spin_rozmycie.value() / 100 * n
        waga = ndimage.gaussian_filter(waga, sig, mode='constant'); suma = ndimage.gaussian_filter(suma, sig, mode='constant')
        jest = waga > waga.max() * 1e-4
        wart = np.where(jest, suma / np.maximum(waga, 1e-12), 0)
        if self.combo_zasieg.currentIndex() == 1:            # cała przestrzeń: puste voksele = wartość najbliższych
            _, ind = ndimage.distance_transform_edt(~jest, return_indices=True)
            wart = wart[tuple(ind)]
            alfa = .25 + .75 * wart
        else:
            gest = np.sqrt(waga / max(float(waga.max()), 1e-12))
            alfa = gest * (.3 + .7 * wart)
        rgba = matplotlib.colormaps['inferno'](np.clip(wart, 0, 1))
        rgba[..., 3] = np.clip(alfa * self.spin_krycie_obj.value(), 0, 1)
        self.objetosc.setData((rgba * 255).astype(np.uint8))
        self.objetosc.resetTransform()
        self.objetosc.scale(2 * L / n, 2 * L / n, 2 * L / n)
        self.objetosc.translate(-L, -L, -L)
        self.objetosc.setVisible(True)

    def _widoczne(self):
        tok = np.arange(600)
        if self.chk_tylko_maska.isChecked() and self.maski is not None:
            tok = np.flatnonzero(self.maski[self.t])
        if self.combo_ile.currentIndex():
            tok = np.intersect1d(tok, tokeny_co(CZASTECZKI_CO[self.combo_ile.currentIndex()]))
        return tok

    def _rysuj(self, skok=False, tylko_punkty=False):
        if self.P is None:
            return
        if skok:
            self._od, self._u, self._v_od = self.P[self.t], 1.0, None
            self._timer.stop()
        P = self._biezace()
        tok = self._widoczne()
        kol = self._kolory()
        rozm = self.spin_rozmiar.value()
        A, strz, self._opis_at = self._atencja()
        if self.chk_objetosc.isChecked() and A is not None:
            if not tylko_punkty or self._u >= 1:             # w czasie przelotu mgła stoi, liczona na jego końcu
                self._rysuj_objetosc(P, A, tok)
        else:
            self.objetosc.setVisible(False)
        if A is not None and self.chk_atencja.isChecked():   # mapa atencji: kolor inferno, rozmiar według wartości
            import matplotlib
            kol = matplotlib.colormaps['inferno'](A).astype(np.float32); kol[:, 3] = .95
            rozm = (rozm * (.4 + self.spin_wzm.value() * A)).astype(np.float32)[tok]
        self.punkty.setData(pos=P[tok], color=kol[tok], size=rozm)
        self.punkty.setVisible(self.chk_czastki.isChecked())
        self._rysuj_zdjecie(P, kol)
        if strz is not None:                                 # strzałki źródło -> cel między cząstkami
            import matplotlib
            src, dst, w = strz
            c = matplotlib.colormaps['cool'](w).astype(np.float32); c[:, 3] = .25 + .75 * w
            self.luki.setData(pos=np.stack([P[src], P[dst]], 1).reshape(-1, 3).astype(np.float32),
                              color=np.repeat(c, 2, 0), wagi=w, grubosc_max=self.opcje.spin_grubosc.value())
            self.groty.setData(pos=P[dst].astype(np.float32), color=c)
            self.luki.setVisible(True); self.groty.setVisible(True)
        else:
            self.luki.setVisible(False); self.groty.setVisible(False)
        if self.token is not None:
            self.wybrany.setData(pos=P[self.token][None], size=self.spin_rozmiar.value() + 10, color=(1, 1, 1, .9))
            self.wybrany.setVisible(True)
        else:
            self.wybrany.setVisible(False)
        tryb_m = self.combo_mikro.currentIndex()
        jest = self.token is not None and self.Z is not None
        if jest and tryb_m != 3:
            v = self._v_biezace()                                           # [K, 3]: mikrowektory cząstki
            kol_v = self.kolory_kan
            o = np.zeros(3, np.float32)                                     # wspólny początek: zawsze środek układu
        if jest and tryb_m in (1, 2):
            kolej = np.argsort(-np.linalg.norm(v, axis=1))                       # od najsilniejszego
            lan = o + np.concatenate([np.zeros((1, 3), np.float32), np.cumsum(v[kolej], 0)])
            self.lancuch.setData(pos=lan, color=np.concatenate([kol_v[kolej[:1]], kol_v[kolej]]))
            self.lancuch.setVisible(True)
        else:
            self.lancuch.setVisible(False)
        if jest and tryb_m in (0, 2):
            seg = np.stack([np.broadcast_to(o, v.shape), o + v * self.spin_mikro.value()], 1).reshape(-1, 3)
            dl = np.linalg.norm(v, axis=1)
            c = np.repeat(kol_v, 2, 0).copy()
            c[:, 3] = 1                                                         # pełne kolory, jak wektory w 3D
            self.mikro.setData(pos=np.ascontiguousarray(seg, np.float32), color=c)
            self.wypadkowa.setData(pos=np.stack([o, o + v.sum(0)]).astype(np.float32))   # = położenie cząstki
            self.mikro.setVisible(True); self.wypadkowa.setVisible(True)
        else:
            self.mikro.setVisible(False); self.wypadkowa.setVisible(False)
        if tylko_punkty:
            return
        L = self.spin_slad.value()
        if L > 0 and self.t > 0:
            t0 = max(0, self.t - L)
            S = self.P[t0:self.t + 1][:, tok]                                   # [L+1, n, 3]
            seg = np.stack([S[:-1], S[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 3)
            a = np.linspace(.08, .6, len(S) - 1, dtype=np.float32)
            c = np.repeat(kol[tok][:, None, :], len(S) - 1, 1).copy(); c[..., 3] = a
            self.slady.setData(pos=seg, color=np.repeat(c, 2, 1).reshape(-1, 4))
            self.slady.setVisible(self.chk_czastki.isChecked())
        else:
            self.slady.setVisible(False)
        if self.chk_gwiazda.isChecked():
            d = self.D / max(float(np.linalg.norm(self.D, axis=1).max()), 1e-9) * 40
            seg = np.stack([np.zeros_like(d), d], 1).reshape(-1, 3)
            c = np.repeat(self.kolory_kan, 2, 0).copy(); c[:, 3] = .25
            self.gwiazda.setData(pos=seg, color=c); self.gwiazda.setVisible(True)
        else:
            self.gwiazda.setVisible(False)
        T = len(self.P)
        przes = float(np.linalg.norm(self.P[self.t] - self.P[self.t - 1], axis=1).mean()) if self.t > 0 else 0.0
        self.etykieta.setText(f'klatka {self.t + 1}/{T}: {len(tok)} cząstek; położenie = suma mikrowektorów '
                              f'(skala s = {self.s:.3g}); średni przelot od poprzedniej klatki {przes:.1f} jedn.'
                              + ('' if self.token is None else f'; wybrany patch {self.token}, '
                                 f'odległość od środka {float(np.linalg.norm(self.P[self.t, self.token])):.1f}') + self._opis_pca + self._opis_poz
                              + (f'; atencja: {self._opis_at}' if getattr(self, '_opis_at', '') else ''))
