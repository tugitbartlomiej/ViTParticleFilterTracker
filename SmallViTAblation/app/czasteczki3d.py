"""Widok „Cząsteczki 3D”: 600 tokenów jako cząsteczki w jednym globalnym układzie 3D, liczone i rysowane na GPU.

Układ globalny: x = kolumna obrazu w pikselach wejścia ViT (0..480), y = wiersz liczony od dołu (0..320),
z = wysokość nad obrazem. Środek tokenu leży w środku jego patcha 16x16 (opcjonalnie podniesiony o atencję).
Kanał c ma stały kierunek w 3D (spirala Fibonacciego na sferze albo półsferze nad obrazem) i stały kolor
z koła barw (768 kolorów), te same dla wszystkich tokenów. Wektor ma długość promień zera + skala * wartość.

Liczenie i rysowanie: czasteczki3d_gpu.py (OpenGL 4.3, compute shadery GLSL i SSBO, natywnie pod Windows).
Na kartę idzie latent PRZED i PO tylko przy nowej klatce, a przy zmianie opcji same małe tablice albo liczby;
mikrowektory, skala (99,5. centyl), przejście PRZED → PO, łuki atencji, mieszanka kolorów i tekstura obrazu
liczą się w shaderach. Animacja przejścia wysyła na kartę jedną liczbę t na klatkę; obrót kamery nic nie liczy.
"""
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QVector3D
from PyQt6.QtWidgets import (QCheckBox, QDoubleSpinBox, QLabel, QPushButton, QSlider, QVBoxLayout,
                             QWidget)

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from opcje_atencji import OpcjeAtencji
from uklad import WaskiCombo, etykieta_statusu
from styl import rola
from uklad import Odpinanie, PanelOpcji, dodaj, dodaj_nawigacje
from czasteczki3d_gpu import CzasteczkiGPU, PlaszczyznaGPU
from widok_gl import Widok3D
from wspolne import baza_pca_rgb
from wspolne import (CZASTECZKI_CO, KOLUMNY, KOMORKA, PCA_SKLADOWE, SZER, WIERSZE, WYNIKI, WYS, kierunki,
                     kolory_kanalow, opisy_czasteczek, pca, tokeny_co)


class WidokCzasteczek3D(QWidget):
    TRYBY = ['PRZED', 'PO', 'różnica po − przed', 'przed (szary) i po (kolor) nałożone',
             'przejście PRZED → PO (suwak, animacja)']
    KANALY = ['wszystkie 768', 'co 2. kanał (384)', 'co 4. kanał (192)', 'co 8. kanał (96)', 'tylko J128 (128)']

    def __init__(self, kanaly_j128):
        super().__init__()
        self.j128 = np.asarray(kanaly_j128)
        self.przed = self.po = self.rgb = None
        self.atencja_fn = None
        self.token = None
        self.maska_tokenow = None          # bool [600]: patche narzędzia (z obrysu), ustawiane z zakładki filmu
        self.zaznaczenie = None            # zaznaczone patche (Ctrl + klik, prostokąt), ustawiane z zakładki
        self.kolory = kolory_kanalow()
        self._okna = []
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)
        self.panel = PanelOpcji(lay, ['Wektory', 'Obraz', 'Atencja', 'Cząsteczki'])
        pasek = self.panel.pasek

        # Wektory: co rysować | jak wyglądają | jak długie
        w = self.panel.wiersz('Wektory')
        self.combo_tryb = dodaj(w, 'Pokaż:', WaskiCombo(10),
                                'PRZED / PO: wartości latentu tokenu.\n'
                                'Różnica: po − przed (wektor dłuższy od sfery zera = kanał wzmocniony, krótszy = osłabiony).\n'
                                'Nałożone: przed na szaro, po w kolorach kanałów.\n'
                                'Przejście: latent przed + t · (po − przed), t z suwaka; „Animuj” przesuwa t w pętli.')
        self.combo_tryb.addItems(self.TRYBY)
        self.combo_kan = dodaj(w, 'Kanały:', WaskiCombo(9), 'Ile z 768 wektorów rysować w każdej cząsteczce.\n'
                               'Kierunek i kolor kanału są zawsze te same.')
        self.combo_kan.addItems(self.KANALY)
        self.combo_pca = dodaj(w, 'PCA wektorów:', WaskiCombo(9),
                               'Zamiast 768 kanałów każda cząsteczka dostaje k wektorów = k głównych składowych (PCA)\n'
                               'latentu: wektor j ma długość sfera zera + skala · (z − μ) · w_j, gdzie w_j to j-ta składowa.\n'
                               'Sfera zera = średni token (μ), krótszy wektor = wynik ujemny. Składowe liczone z kanałów\n'
                               'wybranych w „Kanały” (np. tylko J128), z latentu przed i po; w zakładce filmu z całej\n'
                               'sekwencji, więc baza jest ta sama we wszystkich klatkach. Kierunki i kolory: k wektorów\n'
                               'równo na sferze (jak dla 768), składowa 1 u góry.')
        self.combo_pca.addItems(['bez PCA (kanały)'] + [f'{k} składowych' for k in PCA_SKLADOWE])
        w = self.panel.wiersz('Wektory')
        self.combo_ksztalt = dodaj(w, 'Kierunki:', WaskiCombo(12),
                                   'Pełna sfera (domyślnie): 768 kierunków równo we wszystkie strony, także w dół.\n'
                                   'Środki cząsteczek leżą na obrazie, dolna połowa kuli jest pod nim i widać ją\n'
                                   'przez półprzezroczyste zdjęcie (pole „krycie”).\n'
                                   'Półsfera: wszystkie 768 kierunków nad obrazem (z > 0), cząsteczki leżą na obrazie.\n'
                                   'Kanał 0 wskazuje w górę, kolejne schodzą spiralą w dół (kolor zmienia się razem z wysokością).')
        self.combo_ksztalt.addItems(['pełna sfera (kula)', 'półsfera nad obrazem'])
        self.combo_styl = dodaj(w, 'Styl:', WaskiCombo(10), 'Wektory: odcinki od środka cząsteczki.\n'
                                'Końce: tylko punkty na końcach wektorów (chmura cząstek).')
        self.combo_styl.addItems(['wektory', 'końce (punkty)', 'wektory i końce'])
        self.spin_gr = dodaj(w, 'Grubość:', QDoubleSpinBox(), 'Grubość linii i punktów w pikselach ekranu.')
        self.spin_gr.setRange(1, 10); self.spin_gr.setSingleStep(.5); self.spin_gr.setValue(1)
        w = self.panel.wiersz('Wektory')
        self.spin_r0 = dodaj(w, 'Sfera zera:', QDoubleSpinBox(), 'Promień sfery oznaczającej wartość 0, w % połowy patcha.\n'
                             'Wektor krótszy od niej = wartość ujemna.')
        self.spin_r0.setRange(0, 90); self.spin_r0.setDecimals(0); self.spin_r0.setValue(25); self.spin_r0.setSuffix(' %')
        self.chk_zero = QCheckBox('Okręgi zera'); self.chk_zero.setChecked(False)
        self.chk_zero.setToolTip('Szary okrąg (przekrój sfery zera) wokół środka każdej cząsteczki.')
        w.addWidget(self.chk_zero)
        self.spin_skala = dodaj(w, 'Skala długości:', QDoubleSpinBox(),
                                'Mnożnik długości. 100 % = wartość większa od 99,5 % wszystkich na rysunku sięga połowy patcha.\n'
                                'W 3D można dać więcej, bo wektory idą w górę i mniej się zasłaniają.')
        self.spin_skala.setRange(5, 5000); self.spin_skala.setDecimals(0); self.spin_skala.setValue(200); self.spin_skala.setSuffix(' %')
        self.spin_maks = dodaj(w, 'Maks. długość:', QDoubleSpinBox(),
                               'Przycina wektory dłuższe niż ta wartość (w % szerokości patcha), żeby nieliczne bardzo duże\n'
                               'wartości nie przecinały całego obrazu. 0 = bez przycinania.')
        self.spin_maks.setRange(0, 5000); self.spin_maks.setDecimals(0); self.spin_maks.setValue(0); self.spin_maks.setSuffix(' %')

        # Obraz: płaszczyzna zdjęcia pod cząsteczkami
        w = self.panel.wiersz('Obraz')
        self.chk_obraz = QCheckBox('Obraz pod spodem'); self.chk_obraz.setChecked(True)
        self.chk_obraz.setToolTip('Zdjęcie 480x320 (wejście ViT) jako płaszczyzna z = 0.')
        w.addWidget(self.chk_obraz)
        self.spin_alfa = dodaj(w, 'Jasność:', QDoubleSpinBox(), '0 = obraz czarny, 1 = pełny.')
        self.spin_alfa.setRange(0, 1); self.spin_alfa.setSingleStep(.1); self.spin_alfa.setValue(.7)
        self.spin_krycie = dodaj(w, 'Krycie:', QDoubleSpinBox(),
                                 'Krycie płaszczyzny obrazu: 1 = nieprzezroczysty (wektory pod obrazem niewidoczne),\n'
                                 '0,5 = półprzezroczysty (wektory pod spodem widać przez zdjęcie), 0 = sam obraz znika.\n'
                                 'Środki cząsteczek leżą na płaszczyźnie obrazu; przy pełnej sferze połowa wektorów idzie pod nią.')
        self.spin_krycie.setRange(0, 1); self.spin_krycie.setSingleStep(.1); self.spin_krycie.setValue(.5)
        self.chk_siatka = QCheckBox('Siatka patchy'); self.chk_siatka.setChecked(True)
        self.chk_siatka.setToolTip('Linie siatki 20x30 na płaszczyźnie obrazu.')
        w.addWidget(self.chk_siatka)
        w = self.panel.wiersz('Obraz')
        self.chk_mies = QCheckBox('Koloruj obraz mieszanką kolorów wektorów')
        self.chk_mies.setToolTip(
            'Każdy patch obrazu dostaje kolor = średnia kolorów jego wektorów (kanałów albo składowych PCA,\n'
            'te same kolory co wektory) ważona siłą wektora. Silniejszy wektor ma większy udział w kolorze.\n'
            'Liczone z tego, co pokazuje widok (PRZED, PO, różnica, przejście), dla wszystkich 600 patchy.')
        w.addWidget(self.chk_mies)
        self.combo_waga = dodaj(w, 'Kolor / waga:', WaskiCombo(12),
                                'Waga wektora c w mieszance:\n'
                                '- |wartość|: długość wektora (siła kanału albo składowej);\n'
                                '- wartość²: mocniej faworyzuje najsilniejsze wektory;\n'
                                '- wartość⁴: prawie sam najsilniejszy wektor, ale płynnie;\n'
                                '- tylko dodatnie: wektory skierowane na zewnątrz (wartość > 0), waga = wartość.\n'
                                'Inne kolorowania (nie mieszanka):\n'
                                '- PCA jako RGB: 3 główne składowe pełnego latentu (768) jako czerwony, zielony, niebieski\n'
                                '  (jak w DINOv2 i FeatUp); baza wspólna dla całej sekwencji / historii, ta sama co w przestrzeni\n'
                                '  makro przy „Kolor cząstek: PCA jako RGB”, więc ten sam kolor = ta sama cząstka w obu widokach;\n'
                                '- podobieństwo do wybranego patcha: kosinus latentu (po odjęciu średniej klatki) każdego patcha\n'
                                '  z wybranym; czerwony = podobny, niebieski = przeciwny, biały = obojętny.')
        self.combo_waga.addItems(['|wartość| (długość)', 'wartość²', 'wartość⁴', 'tylko dodatnie',
                                  'PCA: 3 składowe jako RGB', 'podobieństwo do wybranego patcha (cos)'])
        self.spin_mies_krycie = dodaj(w, 'Krycie koloru:', QDoubleSpinBox(),
                                      '0 = sam obraz, 1 = sam kolor mieszanki (bez zdjęcia).')
        self.spin_mies_krycie.setRange(0, 1); self.spin_mies_krycie.setSingleStep(.1); self.spin_mies_krycie.setValue(.6)
        self.spin_nasyc = dodaj(w, 'Nasycenie:', QDoubleSpinBox(),
                                'Mieszanka wielu kolorów szarzeje, więc kolor jest automatycznie oddalany od szarości tak,\n'
                                'żeby 98 % patchy mieściło się w pełnej palecie; ten mnożnik wzmacnia (> 1) albo osłabia (< 1).')
        self.spin_nasyc.setRange(.1, 10); self.spin_nasyc.setSingleStep(.1); self.spin_nasyc.setValue(1)
        self.chk_mies_jasn = QCheckBox('jasność wg siły')
        self.chk_mies_jasn.setToolTip('Patche o większej łącznej sile wektorów jaśniejsze (względem 99. centyla w klatce).')
        w.addWidget(self.chk_mies_jasn)
        self.chk_mies_gladko = QCheckBox('gładko')
        self.chk_mies_gladko.setToolTip('Płynne przejścia między patchami (interpolacja); wyłączone = kwadraty patchy.')
        w.addWidget(self.chk_mies_gladko)
        self._mies = None

        # Atencja: włącznik w pierwszym wierszu wspólnego panelu opcji atencji
        self.chk_atencja = QCheckBox('Atencja na obrazie')
        self.chk_atencja.setToolTip('Mapa cieplna atencji ViT wtopiona w płaszczyznę obrazu i łuki źródło → cel\n'
                                    '(opcje obok). Atencja liczy się z obrazu; auto-szukanie zmienia latent\n'
                                    'za atencją, więc jej nie zmienia.')
        self.opcje = OpcjeAtencji(na_poczatku=(self.chk_atencja,))
        self.panel.strona('Atencja').addWidget(self.opcje)

        # Cząsteczki: które rysować i gdzie leżą ich środki
        w = self.panel.wiersz('Cząsteczki')
        self.combo_ile = dodaj(w, 'Cząsteczek:', WaskiCombo(10),
                               'Ile cząsteczek rysować: co n-ty patch w wierszach i w kolumnach siatki 20 x 30,\n'
                               'od środka pierwszego bloku n x n, więc wybrane leżą równo po całym obrazie.\n'
                               'Łączy się z „tylko w obrysie” (część wspólna). PCA i atencja liczą się dalej ze wszystkich 600.')
        self.combo_ile.addItems(opisy_czasteczek())
        self.combo_patche = dodaj(w, 'Obrys:', WaskiCombo(12),
                                  'Które cząsteczki rysować, gdy znany jest obrys narzędzia (zakładka filmu).\n'
                                  'Patch należy do obrysu, gdy obrys pokrywa co najmniej ustawiony procent jego pikseli.')
        self.combo_patche.addItems(['wszystkie patche', 'tylko patche w obrysie narzędzia', 'tylko patche poza obrysem'])
        self.combo_patche.setEnabled(False)
        self.chk_czasteczki = QCheckBox('Pokaż cząsteczki'); self.chk_czasteczki.setChecked(True)
        self.chk_czasteczki.setToolTip('Wyłączone: bez wektorów i punktów cząsteczek (zostaje obraz, mapa atencji i łuki);\n'
                                       'widok liczy się wtedy dużo szybciej.')
        w.addWidget(self.chk_czasteczki)
        self.chk_tylko = QCheckBox('Tylko wybrany patch')
        self.chk_tylko.setToolTip('Rysuje tylko cząsteczkę klikniętego patcha (kliknij na obrazie) i ustawia na niej kamerę.')
        w.addWidget(self.chk_tylko)
        self.chk_zazn = QCheckBox('Tylko zaznaczone patche')
        self.chk_zazn.setToolTip('Rysuje tylko zaznaczone patche (Ctrl + klik albo prostokąt prawym przyciskiem na obrazie,\n'
                                 'Shift + przeciągnij albo Ctrl + klik w przestrzeni makro). Bez zaznaczenia: wszystkie.')
        w.addWidget(self.chk_zazn)
        w = self.panel.wiersz('Cząsteczki')
        self.spin_wys = dodaj(w, 'Podnieś o atencję:', QDoubleSpinBox(),
                              'Środek cząsteczki leży na wysokości z = ta wartość · mapa atencji (0..1, jak na obrazie).\n'
                              'Patche o największej wartości mapy unoszą się najwyżej. 0 = wszystkie na obrazie.\n'
                              'Mapę wybiera się w zakładce „Atencja”.')
        self.spin_wys.setRange(0, 1000); self.spin_wys.setDecimals(0); self.spin_wys.setValue(0); self.spin_wys.setSuffix(' px')

        w = pasek                                   # pasek nad widokiem: przejście, kamera, zapis
        w.addWidget(QLabel('PRZED'))
        self.suwak = QSlider(Qt.Orientation.Horizontal); self.suwak.setRange(0, 100); self.suwak.setValue(100)
        self.suwak.setToolTip('t w trybie przejścia: 0 = latent PRZED, 100 = latent PO.')
        self.suwak.setFixedWidth(80)
        w.addWidget(self.suwak); w.addWidget(QLabel('PO'))
        self.btn_anim = rola(QPushButton('Animuj'), 'narzedzie'); self.btn_anim.setCheckable(True)
        self.btn_anim.setToolTip('Przesuwa t tam i z powrotem (tryb przejścia PRZED → PO, przełącza na niego).\n'
                                 'Na kartę idzie tylko t; położenia wektorów liczy shader.')
        w.addWidget(self.btn_anim)
        self.btn_kamera = rola(QPushButton('Kamera od nowa'), 'narzedzie')
        self.btn_kamera.setToolTip('Ustawia kamerę od nowa: widok całego obrazu z ukosa (elewacja 38 st.).')
        w.addWidget(self.btn_kamera)
        self.btn_gora = rola(QPushButton('Z góry'), 'narzedzie')
        self.btn_gora.setToolTip('Kamera prosto z góry na środek obrazu, jak na zdjęciu.')
        w.addWidget(self.btn_gora)
        self.btn_png = rola(QPushButton('Zapisz PNG'), 'narzedzie')
        self.btn_png.setToolTip(f'Zapisuje bieżący widok (to, co widać, z kamerą) jako PNG do {WYNIKI}')
        w.addWidget(self.btn_png)
        self.btn_odepnij = rola(QPushButton('Odepnij'), 'narzedzie'); w.addWidget(self.btn_odepnij)
        self.odpinanie = Odpinanie(self, 'Cząsteczki 3D', self.btn_odepnij)
        self.btn_okno = rola(QPushButton('Kopia w oknie'), 'narzedzie')
        self.btn_okno.setToolTip('Nieruchoma kopia bieżącego widoku w nowym oknie (do porównań; nie odtwarza dalej).')
        w.addWidget(self.btn_okno)
        self.panel.zakoncz()

        self.widok = Widok3D()
        dodaj_nawigacje(self, self.panel, self.widok)       # obrót 360, kroki, ujęcia, lustro
        self.widok.setBackgroundColor(pg.mkColor(12, 12, 16))
        self.widok.setMinimumHeight(500)
        lay.addWidget(self.widok, 1)
        # stan widoku pod nim, małą szarą czcionką: widok zaczyna się zawsze na tej samej wysokości
        self.etykieta = etykieta_statusu(QLabel('Obrót: lewy przycisk. Przesuwanie: środkowy albo Ctrl + lewy. '
                                                'Zoom: kółko. Klik bez ruchu = wybór patcha.'))
        self.etykieta.setToolTip('Stan widoku. Obrót: lewy przycisk, przesuwanie: środkowy albo Ctrl + lewy,\n'
                                 'zoom: kółko, klik bez ruchu = wybór patcha.')
        lay.addWidget(self.etykieta)
        self._buduj_scene()
        self._kamera()

        self.timer = QTimer(self); self.timer.setInterval(33); self.timer.timeout.connect(self._klatka)
        self._kierunek_anim = -1
        self.btn_anim.toggled.connect(self._animuj)
        self.btn_kamera.clicked.connect(self._kamera)
        self.btn_gora.clicked.connect(lambda: self.widok.setCameraPosition(pos=QVector3D(SZER / 2, WYS / 2, 0), distance=520,
                                                                            elevation=90, azimuth=-90))
        self.chk_tylko.toggled.connect(lambda v: None if v else self._kamera())
        self.btn_png.clicked.connect(self.zapisz_png)
        self.btn_okno.clicked.connect(self._osobne)
        self.widok.klik.connect(self._klik)
        self._pca_zrodlo = None                  # latent sekwencji [N, 768] do bazy PCA (zakładka filmu)
        self._pca_cache = {}
        self._pca_wersja = 0
        # opcje, które zmieniają zestaw wektorów, widoczne cząsteczki albo mapę: małe tablice na kartę
        for k in (self.combo_tryb.currentIndexChanged, self.combo_kan.currentIndexChanged,
                  self.combo_ksztalt.currentIndexChanged, self.combo_pca.currentIndexChanged,
                  self.spin_wys.valueChanged, self.chk_tylko.toggled, self.combo_patche.currentIndexChanged,
                  self.combo_ile.currentIndexChanged, self.chk_czasteczki.toggled, self.chk_zazn.toggled):
            k.connect(lambda *_: self._rysuj())
        for k in (self.chk_atencja.toggled, self.opcje.zmiana):
            k.connect(lambda *_: self._atencja_od_nowa())
        # opcje, które zmieniają tylko liczby w shaderach (bez żadnego przeliczenia na CPU)
        for k in (self.combo_styl.currentIndexChanged, self.spin_gr.valueChanged, self.spin_r0.valueChanged,
                  self.spin_skala.valueChanged, self.spin_maks.valueChanged, self.chk_zero.toggled,
                  self.suwak.valueChanged, self.chk_obraz.toggled, self.spin_alfa.valueChanged, self.spin_krycie.valueChanged,
                  self.chk_mies.toggled, self.combo_waga.currentIndexChanged, self.spin_mies_krycie.valueChanged,
                  self.spin_nasyc.valueChanged, self.chk_mies_jasn.toggled, self.chk_mies_gladko.toggled):
            k.connect(lambda *_: self._parametry())
        self.chk_siatka.toggled.connect(lambda v: self.siatka.setVisible(v))

    # ------------------------------------------------------------ scena
    def _buduj_scene(self):
        v = self.widok
        lin = []
        for x in range(0, SZER + 1, KOMORKA):
            lin += [(x, 0, .05), (x, WYS, .05)]
        for y in range(0, WYS + 1, KOMORKA):
            lin += [(0, y, .05), (SZER, y, .05)]
        self.siatka = gl.GLLinePlotItem(pos=np.array(lin, np.float32), color=(1, 1, 1, .18), mode='lines',
                                        glOptions='translucent')
        v.addItem(self.siatka)
        # mikrowektory, końce, okręgi zera, szare PRZED, łuki i groty atencji: liczone i rysowane na GPU
        self.gpu = CzasteczkiGPU()
        self.gpu.po_skali = lambda m: QTimer.singleShot(0, self._etykieta)
        self.gpu.po_bledzie = lambda b: QTimer.singleShot(0, self._etykieta)
        v.addItem(self.gpu)
        self.podloze = PlaszczyznaGPU(self.gpu)          # płaszczyzna obrazu z teksturą policzoną na GPU (głębia 10)
        v.addItem(self.podloze)
        self.obrys_item = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1, 1, 0, 1), width=2.5,
                                            mode='lines', glOptions='translucent')
        self.obrys_item.setVisible(False)
        v.addItem(self.obrys_item)
        self.ramka = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1, 1, 1, 1), width=2, mode='line_strip')
        self.ramka.setVisible(False)
        v.addItem(self.ramka)
        # legenda: 768 kierunków i kolorów kanałów na kuli obok obrazu
        self.legenda = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=4, pxMode=True, glOptions='translucent')
        v.addItem(self.legenda)
        self.legenda_tekst = gl.GLTextItem(pos=(-70, WYS - 10, 75), text='kanał c: kierunek i kolor',
                                           color=(220, 220, 220, 255))
        v.addItem(self.legenda_tekst)
        for pos, txt in (((SZER / 2, -18, 0), 'kolumna patcha →'), ((-18, WYS / 2, 0), 'wiersz ↓'),
                         ((0, WYS + 4, 0), 'wiersz 0, kolumna 0')):
            v.addItem(gl.GLTextItem(pos=pos, text=txt, color=(200, 200, 200, 255)))
        self._legenda()

    def _legenda(self):
        polsfera = self.combo_ksztalt.currentIndex() == 1
        k = self._k_pca()
        if getattr(self, '_klucz_legendy', None) == (polsfera, k):
            return
        self._klucz_legendy = (polsfera, k)
        d = kierunki(polsfera, k or 768)
        kol = kolory_kanalow(k) if k else self.kolory
        self.legenda.setData(pos=np.array([-60, WYS - 60, 0], np.float32) + 40 * d, color=kol,
                             size=4 if not k else (10 if k <= 32 else 6))

    def _kamera(self):
        self.widok.setCameraPosition(pos=QVector3D(SZER / 2, WYS / 2, 0), distance=720, elevation=38, azimuth=-90)

    # ------------------------------------------------------------ dane
    def pokaz(self, przed, po=None, rgb480=None, atencja_fn=None):
        self.przed = np.asarray(przed, np.float32)
        self.po = None if po is None else np.asarray(po, np.float32)
        self._wersja = getattr(self, '_wersja', 0) + 1     # nowe dane = nowa baza PCA (gdy nie z sekwencji)
        self.rgb, self.atencja_fn = rgb480, atencja_fn
        self.opcje.ustaw_zrodlo(atencja_fn)
        self._at_klucz = None                             # nowa klatka: atencja od nowa
        if self.po is None and self.combo_tryb.currentIndex() != 0:
            self.combo_tryb.blockSignals(True); self.combo_tryb.setCurrentIndex(0); self.combo_tryb.blockSignals(False)
        self.gpu.ustaw_latent(self.przed, self.po)        # jedyne duże dane na kartę: latent tej klatki
        self.gpu.ustaw_obraz(rgb480)
        self._rysuj()

    def _osobne(self):
        if self.przed is None:
            return
        w = WidokCzasteczek3D(self.j128)
        w._pca_zrodlo = self._pca_zrodlo
        for a, b in ((w.combo_tryb, self.combo_tryb), (w.combo_kan, self.combo_kan), (w.combo_ksztalt, self.combo_ksztalt),
                     (w.combo_styl, self.combo_styl), (w.combo_pca, self.combo_pca), (w.combo_ile, self.combo_ile)):
            a.setCurrentIndex(b.currentIndex())
        for a, b in ((w.spin_r0, self.spin_r0), (w.spin_skala, self.spin_skala), (w.spin_gr, self.spin_gr), (w.spin_krycie, self.spin_krycie),
                     (w.spin_alfa, self.spin_alfa), (w.spin_wys, self.spin_wys), (w.spin_maks, self.spin_maks)):
            a.setValue(b.value())
        w.combo_waga.setCurrentIndex(self.combo_waga.currentIndex())
        w.spin_mies_krycie.setValue(self.spin_mies_krycie.value()); w.spin_nasyc.setValue(self.spin_nasyc.value())
        for a, b in ((w.chk_obraz, self.chk_obraz), (w.chk_atencja, self.chk_atencja), (w.chk_siatka, self.chk_siatka),
                     (w.chk_zero, self.chk_zero), (w.chk_tylko, self.chk_tylko), (w.chk_mies, self.chk_mies),
                     (w.chk_zazn, self.chk_zazn),
                     (w.chk_mies_jasn, self.chk_mies_jasn), (w.chk_mies_gladko, self.chk_mies_gladko)):
            a.setChecked(b.isChecked())
        w.suwak.setValue(self.suwak.value())
        w.opcje.kopiuj_z(self.opcje)
        w.token = self.token
        w.zaznaczenie = self.zaznaczenie
        w.setWindowTitle('Cząsteczki 3D: globalny układ, 768 wektorów na token')
        w.resize(1600, 1000)
        w.pokaz(self.przed, self.po, self.rgb, self.atencja_fn)
        from wygoda import ulatw
        ulatw(w)
        w.show()
        self._okna.append(w)

    def _klik(self, x, y):
        k = int(np.clip(x // KOMORKA, 0, KOLUMNY - 1))
        r = int(np.clip((WYS - y) // KOMORKA, 0, WIERSZE - 1))
        self.token = r * KOLUMNY + k
        self._rysuj()

    def _atencja_od_nowa(self):
        """Zmiana opcji atencji: mapa i łuki liczone od nowa (na CPU z macierzy atencji ViT), potem na kartę."""
        self._at_klucz = None
        self._rysuj()

    def _atencja(self):
        """(mapa [20, 30] albo None, łuki (src, dst, w) albo None, opis łuków). Liczone tylko, gdy zmieniły się
        dane, opcje atencji albo (dla opcji „wybrany patch”) wybrany patch."""
        at = self.chk_atencja.isChecked() and self.atencja_fn is not None
        mapa = self.atencja_fn is not None and (at or self.spin_wys.value() > 0)
        klucz = (at, mapa, self.token if self.opcje.potrzebuje_tokenu() else None)
        if getattr(self, '_at_klucz', None) != klucz:
            self._at_klucz = klucz
            A = self.opcje.mapa(self.token)[0] if mapa else None
            strz = self.opcje.strzalki(self.token) if at else None
            if strz is not None:
                opis = f'; łuki: {self.opcje.combo_strz.currentText()} ({len(strz[0])})'
            elif self.chk_atencja.isChecked() and self.opcje.combo_strz.currentIndex() in (1, 2) and self.token is None:
                opis = '; łuki: kliknij patch'
            else:
                opis = ''
            self._at = (A, strz, opis)
        return self._at

    def baza_rgb(self):
        """Baza PCA 3 składowych pełnego latentu do kolorów RGB, z sekwencji (ustaw_baze_pca) albo z przed i po:
        (mu [768], W [768, 3], dolny i górny 1. / 99. centyl wyników [3]); pamiętana do zmiany danych."""
        zr = self._pca_zrodlo
        klucz = self._klucz_pca()
        if getattr(self, '_rgb_klucz', None) != klucz:
            if zr is not None:
                X = zr[::max(1, len(zr) // 12000)]
            elif self.przed is not None:
                X = np.concatenate([self.przed] + ([self.po] if self.po is not None else []))
            else:
                return None
            self._rgb, self._rgb_klucz = baza_pca_rgb(X), klucz
        return self._rgb

    # ------------------------------------------------------------ geometria (CPU: tylko małe tablice)
    def _kanaly(self):
        i = self.combo_kan.currentIndex()
        return self.j128 if i == 4 else np.arange(0, 768, (1, 2, 4, 8)[i])

    def _srodki(self):
        """Środki 600 cząsteczek [600, 3] (te same wzory co shader `srodek`): do ustawienia kamery."""
        t = np.arange(600)
        s = np.stack([(t % KOLUMNY) * KOMORKA + KOMORKA / 2, WYS - ((t // KOLUMNY) * KOMORKA + KOMORKA / 2),
                      np.zeros(600)], 1).astype(np.float32)
        h = self.spin_wys.value()
        if h > 0:
            A = self._atencja()[0]
            if A is not None:
                s[:, 2] = h * A.reshape(-1)
        return s

    def ustaw_baze_pca(self, Z):
        """Latent, z którego liczyć bazę PCA (np. cała sekwencja filmu [T, 600, 768] albo [N, 768]); None = z przed i po."""
        self._pca_zrodlo = None if Z is None else np.asarray(Z).reshape(-1, np.shape(Z)[-1])
        self._pca_cache = {}
        self._pca_wersja += 1                     # nowa sekwencja: nowa baza PCA wektorów i baza PCA RGB
        if self._k_pca() or (self.chk_mies.isChecked() and self.combo_waga.currentIndex() == 4):
            self._rysuj()

    def _k_pca(self):
        i = self.combo_pca.currentIndex()
        return PCA_SKLADOWE[i - 1] if i > 0 else 0

    def _klucz_pca(self):
        return ('sekwencja', self._pca_wersja) if self._pca_zrodlo is not None else getattr(self, '_wersja', 0)

    def _baza_pca(self, kan):
        """(mu, W, udział wariancji) PCA latentu na kanałach kan: z całej sekwencji (ustaw_baze_pca) albo z przed i po;
        pamiętana, dopóki nie zmienią się kanały ani dane. None, gdy nie ma z czego liczyć."""
        zr = self._pca_zrodlo
        klucz = (self.combo_kan.currentIndex(), self._klucz_pca())
        if klucz not in self._pca_cache:
            if zr is not None:
                X = zr[::max(1, len(zr) // 12000)][:, kan]
            elif self.przed is not None:
                X = np.concatenate([self.przed[:, kan]] + ([self.po[:, kan]] if self.po is not None else []))
            else:
                return None
            self._pca_cache = {klucz: pca(X)}
        return self._pca_cache[klucz]

    def baza_wektorow(self):
        """Dokładnie te wektory, którymi rysowane są mikrowektory cząsteczek: (kanały, PCA (mu, W) albo None,
        kierunki [K, 3], kolory [K, 4]). Przestrzeń makrowektorów w trybie „jak widok 3D” używa tych samych
        kierunków, więc mikrowektor kanału (składowej) c ma w obu widokach ten sam kierunek i kolor."""
        polsfera = self.combo_ksztalt.currentIndex() == 1
        kan = self._kanaly()
        k = self._k_pca()
        if not k:
            return kan, None, kierunki(polsfera)[kan], self.kolory[kan]
        b = self._baza_pca(kan)
        if b is None:
            return None
        mu, W, _ = b
        k = min(k, W.shape[1])
        return kan, (mu, W[:, :k]), kierunki(polsfera, k), kolory_kanalow(k)

    def _wektory_na_karte(self):
        """Przestrzeń wektorów (kanały albo baza PCA, kierunki, kolory) na kartę, tylko gdy się zmieniła.
        Zwraca (K, opis PCA) albo None."""
        bw = self.baza_wektorow()
        if bw is None:
            return None
        kan, pc, d, kol = bw
        k = self._k_pca()
        klucz = (self.combo_ksztalt.currentIndex(), self.combo_kan.currentIndex(), k, self._klucz_pca() if k else None)
        self.gpu.ustaw_wektory(klucz, kan, pc, d, kol)
        if pc is None:
            return len(kan), ''
        udz = self._baza_pca(kan)[2]
        K = pc[1].shape[1]
        zrodlo = 'sekwencji' if self._pca_zrodlo is not None else ('przed i po' if self.po is not None else 'klatki')
        return K, f'; PCA {K} składowych z {len(kan)} kanałów ({zrodlo}) wyjaśnia {100 * float(udz[:K].sum()):.1f} % wariancji'

    def _tokeny(self):
        """Rysowane cząsteczki: wybrany patch, zaznaczone, obrys narzędzia, co n-ty patch."""
        tok = np.arange(600)
        zazn = self.chk_zazn.isChecked() and self.zaznaczenie is not None and len(self.zaznaczenie) > 0
        tylko = self.chk_tylko.isChecked() and self.token is not None
        if tylko:
            tok = np.array([self.token])
        elif zazn:                                                # tylko zaznaczone: bez innych filtrów
            tok = np.sort(np.asarray(self.zaznaczenie, int))
        elif self.maska_tokenow is not None and self.combo_patche.currentIndex():
            m = self.maska_tokenow if self.combo_patche.currentIndex() == 1 else ~self.maska_tokenow
            tok = tok[m]
        if not tylko and not zazn and self.combo_ile.currentIndex():
            tok = np.intersect1d(tok, tokeny_co(CZASTECZKI_CO[self.combo_ile.currentIndex()]))
        return tok

    def _rysuj(self, tylko_pozycje=False):
        """Zmiana danych albo opcji: małe tablice (widoczne cząsteczki, mapa i łuki atencji, zestaw wektorów) na
        kartę, gdy się zmieniły, i parametry. Geometrię liczą shadery przy najbliższym rysowaniu."""
        if self.przed is None:
            return
        if tylko_pozycje:                                         # zgodność: animacja to same parametry
            self._parametry()
            return
        t0 = time.perf_counter()
        g = self.gpu
        A, strz, self._tekst_at = self._atencja()
        g.ustaw_mape(A)
        wek = self._wektory_na_karte()
        self._K, self._opis_pca = wek if wek is not None else (0, '')
        tok = self._tokeny() if self.chk_czasteczki.isChecked() else np.arange(0)
        self._n_tok = len(tok)
        self._pusto = self.chk_czasteczki.isChecked() and len(tok) == 0
        g.ustaw_widoczne(tok)
        g.ustaw_luki(None if self._pusto else strz)
        if self.token is not None:
            r, k = divmod(self.token, KOLUMNY)
            x0, y0 = k * KOMORKA, WYS - (r + 1) * KOMORKA
            z = .2
            self.ramka.setData(pos=np.array([(x0, y0, z), (x0 + KOMORKA, y0, z), (x0 + KOMORKA, y0 + KOMORKA, z),
                                             (x0, y0 + KOMORKA, z), (x0, y0, z)], np.float32))
            self.ramka.setVisible(True)
        self._legenda()
        if self.chk_tylko.isChecked() and self.token is not None:   # kamera na cząsteczce, gdy ta się przesunęła
            c = self._srodki()[self.token]
            if getattr(self, '_klucz_kamery', None) != (self.token, *c.tolist()):
                self._klucz_kamery = (self.token, *c.tolist())
                self.widok.setCameraPosition(pos=QVector3D(*map(float, c)), distance=60)
        else:
            self._klucz_kamery = None
        self._ms = (time.perf_counter() - t0) * 1000
        self._parametry()

    def _parametry(self):
        """Liczby i przełączniki widoku do shaderów (bez przeliczania czegokolwiek na CPU) i opis pod widokiem."""
        if self.przed is None:
            return
        tryb = self.combo_tryb.currentIndex() if self.po is not None else 0
        waga = self.combo_waga.currentIndex()
        if self.chk_mies.isChecked() and waga == 4:
            self.gpu.ustaw_baze_rgb(self.baza_rgb())
        dowolny = self.chk_obraz.isChecked() or self.chk_atencja.isChecked() or self.chk_mies.isChecked()
        self.gpu.ustaw(wtryb=tryb, t=self.suwak.value() / 100, r0=KOMORKA / 2 * self.spin_r0.value() / 100,
                       proc=self.spin_skala.value() / 100, maks=KOMORKA * self.spin_maks.value() / 100,
                       h=float(self.spin_wys.value()), styl=self.combo_styl.currentIndex(), gr=self.spin_gr.value(),
                       szare=tryb == 3, zera=self.chk_zero.isChecked(),
                       czastki=self.chk_czasteczki.isChecked() and not self._pusto,
                       grubosc_luk=self.opcje.spin_grubosc.value(), waga=waga, jasn=self.chk_mies_jasn.isChecked(),
                       nasyc=self.spin_nasyc.value(), tok=-1 if self.token is None else int(self.token),
                       mies=self.chk_mies.isChecked(), gladko=self.chk_mies_gladko.isChecked(),
                       obraz=self.chk_obraz.isChecked() and self.rgb is not None, jasnosc=self.spin_alfa.value(),
                       atencja=self.chk_atencja.isChecked(), kr_at=self.opcje.krycie(),
                       kr_mies=self.spin_mies_krycie.value(),
                       alfa=int(round(255 * self.spin_krycie.value())) if dowolny else 0)
        self._etykieta()

    def _etykieta(self):
        """Opis stanu pod widokiem; wartość połowy patcha przychodzi z karty (99,5. centyl liczony w shaderze)."""
        if self.przed is None:
            return
        blad = f'; BŁĄD GPU (wymaga OpenGL 4.3): {self.gpu.blad}' if self.gpu.blad else ''
        if not self.chk_czasteczki.isChecked():
            self.etykieta.setText('cząsteczki wyłączone („Pokaż cząsteczki” w zakładce Cząsteczki)' + self._tekst_at + blad)
            return
        if self._pusto:
            self.etykieta.setText('brak patchy do narysowania (obrys nie pokrywa żadnego patcha)' + blad)
            return
        tryb = self.combo_tryb.currentIndex() if self.po is not None else 0
        opis = self.TRYBY[tryb] + (f' t = {self.suwak.value()} %' if tryb == 4 else '')
        m = self.gpu.m
        pol = '...' if m is None else f'{m * self.spin_skala.value() / 100:.3g}'
        n, K = self._n_tok, self._K
        self.etykieta.setText(f'{opis}: {n} cząsteczek x {K} wektorów = {n * K:,} (CPU {self._ms:.1f} ms, '
                              f'geometria na GPU); połowa patcha = wartość {pol}'
                              + ('' if self.token is None else f'; patch: token {self.token}') + self._opis_pca
                              + self._tekst_at + blad)

    def ustaw_obrysy(self, obrysy, maska=None):
        """obrysy: lista (wielokąt [n, 2] w pikselach obrazu 480x320, kolor RGBA 0..1), rysowane jako zamknięte
        linie tuż nad obrazem; maska: bool [600] patchy narzędzia albo None."""
        seg, kol = [], []
        for p, c in obrysy:
            p = np.asarray(p, np.float32).reshape(-1, 2)
            if len(p) < 2:
                continue
            q = np.stack([p[:, 0], WYS - p[:, 1], np.full(len(p), .4, np.float32)], 1)
            seg.append(np.stack([q, np.roll(q, -1, 0)], 1).reshape(-1, 3))
            kol.append(np.tile(np.asarray(c, np.float32), (2 * len(p), 1)))
        if seg:
            self.obrys_item.setData(pos=np.concatenate(seg), color=np.concatenate(kol))
        self.obrys_item.setVisible(bool(seg))
        self.maska_tokenow = None if maska is None else np.asarray(maska, bool)
        self.combo_patche.setEnabled(maska is not None)

    # ------------------------------------------------------------ animacja i zapis
    def _animuj(self, wl):
        if wl:
            if self.po is None:
                self.btn_anim.setChecked(False)
                self.etykieta.setText('Animuj: potrzebne dwa stany, PRZED i PO (porównanie); teraz jest tylko jeden.')
                return
            if self.combo_tryb.currentIndex() != 4:
                self.combo_tryb.setCurrentIndex(4)
            self.timer.start()
        else:
            self.timer.stop()

    def _klatka(self):
        """Krok animacji: tylko nowe t (suwak) -> uniform w shaderach; CPU nie liczy żadnych położeń."""
        v = self.suwak.value()
        if v <= 0 or v >= 100:
            self._kierunek_anim *= -1
        self.suwak.setValue(int(np.clip(v + 2 * self._kierunek_anim, 0, 100)))

    def zapisz_png(self, sciezka=None):
        WYNIKI.mkdir(parents=True, exist_ok=True)
        sciezka = Path(sciezka) if sciezka else WYNIKI / f'czasteczki3d_{time.strftime("%Y%m%d_%H%M%S")}.png'
        self.widok.grabFramebuffer().save(str(sciezka))
        self.etykieta.setText(f'zapisano {sciezka}')
        return sciezka
