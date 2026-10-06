"""Widok „Cząsteczki 3D”: 600 tokenów jako cząsteczki w jednym globalnym układzie 3D, rysowane na GPU.

Układ globalny: x = kolumna obrazu w pikselach wejścia ViT (0..480), y = wiersz liczony od dołu (0..320),
z = wysokość nad obrazem. Środek tokenu leży w środku jego patcha 16x16 (opcjonalnie podniesiony o atencję).
Kanał c ma stały kierunek w 3D (spirala Fibonacciego na sferze albo półsferze nad obrazem) i stały kolor
z koła barw (768 kolorów), te same dla wszystkich tokenów. Wektor ma długość promień zera + skala * wartość.

Rysowanie: pyqtgraph.opengl (OpenGL, bufory VBO na karcie graficznej). Obrót kamery nie przelicza danych,
a zmiana wartości wysyła na kartę jedną tablicę pozycji (do 600 x 768 x 2 wierzchołków).
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
from uklad import PanelOpcji, dodaj
from widok_gl import ObrazPrzezroczysty, Widok3D
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
        self.chk_tylko = QCheckBox('Tylko wybrany patch')
        self.chk_tylko.setToolTip('Rysuje tylko cząsteczkę klikniętego patcha (kliknij na obrazie) i ustawia na niej kamerę.')
        w.addWidget(self.chk_tylko)
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
        self.btn_anim.setToolTip('Przesuwa t tam i z powrotem (tryb przejścia). Każda klatka to nowe pozycje na karcie.')
        w.addWidget(self.btn_anim)
        self.btn_kamera = rola(QPushButton('Kamera od nowa'), 'narzedzie'); self.btn_kamera.setToolTip('Widok całego obrazu z ukosa.')
        w.addWidget(self.btn_kamera)
        self.btn_gora = rola(QPushButton('Z góry'), 'narzedzie'); self.btn_gora.setToolTip('Widok prosto z góry, jak na zdjęciu.')
        w.addWidget(self.btn_gora)
        self.btn_png = rola(QPushButton('Zapisz PNG'), 'narzedzie')
        self.btn_png.setToolTip(f'Zapis bieżącego widoku do {WYNIKI}')
        w.addWidget(self.btn_png)
        self.btn_okno = rola(QPushButton('Osobne okno'), 'narzedzie'); self.btn_okno.setToolTip('Ten sam widok w dużym, osobnym oknie.')
        w.addWidget(self.btn_okno)
        self.panel.zakoncz()

        self.widok = Widok3D()
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
        self.combo_pca.currentIndexChanged.connect(lambda *_: (self._legenda(), self._rysuj()))
        self._pca_zrodlo = None                  # latent sekwencji [N, 768] do bazy PCA (zakładka filmu)
        self._pca_cache = {}
        for k in (self.combo_tryb.currentIndexChanged, self.combo_kan.currentIndexChanged,
                  self.combo_ksztalt.currentIndexChanged, self.combo_styl.currentIndexChanged,
                  self.spin_r0.valueChanged, self.spin_skala.valueChanged, self.spin_wys.valueChanged,
                  self.spin_maks.valueChanged, self.chk_tylko.toggled, self.combo_patche.currentIndexChanged,
                  self.combo_ile.currentIndexChanged):
            k.connect(lambda *_: self._rysuj())
        self.suwak.valueChanged.connect(lambda *_: self._rysuj(tylko_pozycje=True))
        self.spin_gr.valueChanged.connect(self._grubosc)
        for k in (self.chk_obraz.toggled, self.spin_alfa.valueChanged, self.chk_atencja.toggled, self.opcje.zmiana,
                  self.spin_krycie.valueChanged):
            k.connect(lambda *_: (self._podloze(), self._rysuj()))
        self.chk_siatka.toggled.connect(lambda v: self.siatka.setVisible(v))
        self.chk_zero.toggled.connect(lambda *_: self._rysuj())

    # ------------------------------------------------------------ scena
    def _buduj_scene(self):
        v = self.widok
        self.podloze = ObrazPrzezroczysty(np.zeros((SZER, WYS, 4), np.uint8))
        v.addItem(self.podloze)
        lin = []
        for x in range(0, SZER + 1, KOMORKA):
            lin += [(x, 0, .05), (x, WYS, .05)]
        for y in range(0, WYS + 1, KOMORKA):
            lin += [(0, y, .05), (SZER, y, .05)]
        self.siatka = gl.GLLinePlotItem(pos=np.array(lin, np.float32), color=(1, 1, 1, .18), mode='lines',
                                        glOptions='translucent')
        v.addItem(self.siatka)
        self.zera = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(.6, .6, .6, .7), mode='lines',
                                      glOptions='translucent')
        v.addItem(self.zera)
        self.szare = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(.55, .55, .55, .5), mode='lines',
                                       glOptions='translucent')
        v.addItem(self.szare)
        self.linie = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1, 1, 1, 1), mode='lines',
                                       glOptions='translucent')
        v.addItem(self.linie)
        self.punkty = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=2, pxMode=True, glOptions='translucent')
        v.addItem(self.punkty)
        self.luki = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1, 1, 1, 1), width=2, mode='lines',
                                      glOptions='translucent')
        self.luki.setVisible(False)
        v.addItem(self.luki)
        self.groty = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=6, pxMode=True, glOptions='translucent')
        self.groty.setVisible(False)
        v.addItem(self.groty)
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
        if self.po is None and self.combo_tryb.currentIndex() != 0:
            self.combo_tryb.blockSignals(True); self.combo_tryb.setCurrentIndex(0); self.combo_tryb.blockSignals(False)
        self._podloze()
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
        for a, b in ((w.chk_obraz, self.chk_obraz), (w.chk_atencja, self.chk_atencja), (w.chk_siatka, self.chk_siatka),
                     (w.chk_zero, self.chk_zero), (w.chk_tylko, self.chk_tylko)):
            a.setChecked(b.isChecked())
        w.suwak.setValue(self.suwak.value())
        w.opcje.kopiuj_z(self.opcje)
        w.token = self.token
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
        self.etykieta.setText(f'wybrany patch: token {self.token} (wiersz {r}, kolumna {k})')
        if self.chk_atencja.isChecked() and self.opcje.potrzebuje_tokenu():
            self._podloze()
        self._rysuj()

    def _mapa_atencji(self):
        return self.opcje.mapa(self.token)[0] if self.atencja_fn is not None else None

    def _podloze(self):
        """Tekstura płaszczyzny z = 0: obraz (jasność) z wtopioną mapą atencji."""
        img = np.zeros((WYS, SZER, 3), np.float32)
        if self.rgb is not None and self.chk_obraz.isChecked():
            img = np.asarray(self.rgb, np.float32)[:WYS, :SZER, :3] / 255 * self.spin_alfa.value()
        if self.chk_atencja.isChecked():
            A = self._mapa_atencji()
            if A is not None:
                import matplotlib
                kol = matplotlib.colormaps['inferno'](A)[..., :3].astype(np.float32)
                kol = np.repeat(np.repeat(kol, KOMORKA, 0), KOMORKA, 1)
                a = self.opcje.krycie()
                img = img * (1 - a) + kol * a
        alfa = int(round(255 * self.spin_krycie.value())) if (self.chk_obraz.isChecked() or self.chk_atencja.isChecked()) else 0
        rgba = np.concatenate([np.clip(img * 255, 0, 255).astype(np.uint8), np.full((WYS, SZER, 1), alfa, np.uint8)], 2)
        self.podloze.setData(np.ascontiguousarray(np.flipud(rgba).transpose(1, 0, 2)))   # [x, y, RGBA], y w górę

    # ------------------------------------------------------------ geometria
    def _kanaly(self):
        i = self.combo_kan.currentIndex()
        return self.j128 if i == 4 else np.arange(0, 768, (1, 2, 4, 8)[i])

    def _srodki(self):
        t = np.arange(600)
        s = np.stack([(t % KOLUMNY) * KOMORKA + KOMORKA / 2, WYS - ((t // KOLUMNY) * KOMORKA + KOMORKA / 2),
                      np.zeros(600)], 1).astype(np.float32)
        h = self.spin_wys.value()
        if h > 0:
            A = self._mapa_atencji()
            if A is not None:
                s[:, 2] = h * A.reshape(-1)
        return s

    def ustaw_baze_pca(self, Z):
        """Latent, z którego liczyć bazę PCA (np. cała sekwencja filmu [T, 600, 768] albo [N, 768]); None = z przed i po."""
        self._pca_zrodlo = None if Z is None else np.asarray(Z).reshape(-1, np.shape(Z)[-1])
        self._pca_cache = {}
        if self._k_pca():
            self._rysuj()

    def _k_pca(self):
        i = self.combo_pca.currentIndex()
        return PCA_SKLADOWE[i - 1] if i > 0 else 0

    def _dane(self, polsfera):
        """Przestrzeń wektorów cząsteczki: (przed, po, indeksy wektorów, kolory [n, 4], kierunki [n, 3], opis).
        Bez PCA to 768 kanałów; z PCA k wyników składowych (z − μ) W, liczonych na wybranych kanałach."""
        kan = self._kanaly()
        k = self._k_pca()
        if not k:
            return self.przed, self.po, kan, self.kolory, kierunki(polsfera), ''
        zr = self._pca_zrodlo
        klucz = (self.combo_kan.currentIndex(), 'sekwencja' if zr is not None else self._wersja)
        if klucz not in self._pca_cache:
            if zr is not None:
                X = zr[::max(1, len(zr) // 12000)][:, kan]
            else:
                X = np.concatenate([self.przed[:, kan]] + ([self.po[:, kan]] if self.po is not None else []))
            self._pca_cache = {klucz: pca(X)}
        mu, W, udz = self._pca_cache[klucz]
        k = min(k, W.shape[1])
        W = W[:, :k]
        przed = (self.przed[:, kan] - mu) @ W
        po = None if self.po is None else (self.po[:, kan] - mu) @ W
        zrodlo = 'sekwencji' if zr is not None else ('przed i po' if self.po is not None else 'klatki')
        opis = f'; PCA {k} składowych z {len(kan)} kanałów ({zrodlo}) wyjaśnia {100 * float(udz[:k].sum()):.1f} % wariancji'
        return przed, po, np.arange(k), kolory_kanalow(k), kierunki(polsfera, k), opis

    def _latent(self, przed=None, po=None):
        przed = self.przed if przed is None else przed
        po = self.po if po is None else po
        tryb = self.combo_tryb.currentIndex() if po is not None else 0
        if tryb == 0:
            return przed, tryb
        if tryb == 2:
            return po - przed, tryb
        if tryb == 4:
            t = self.suwak.value() / 100
            return przed + t * (po - przed), tryb
        return po, tryb

    def _wektory(self, L, kan, tok, srodki, d, r0, skala):
        """Pozycje [len(tok)*K*2, 3] odcinków (środek, koniec) i końce [len(tok)*K, 3]."""
        maks = KOMORKA * self.spin_maks.value() / 100 or None
        dl = np.clip(r0 + skala * L[np.ix_(tok, kan)], 0, maks)                 # [T, K]
        konce = srodki[tok, None, :] + dl[..., None] * d[None, kan, :]           # [T, K, 3]
        odc = np.empty((len(tok), len(kan), 2, 3), np.float32)
        odc[:, :, 0] = srodki[tok, None, :]
        odc[:, :, 1] = konce
        return odc.reshape(-1, 3), konce.reshape(-1, 3)

    def _rysuj(self, tylko_pozycje=False):
        if self.przed is None:
            return
        t0 = time.perf_counter()
        polsfera = self.combo_ksztalt.currentIndex() == 1
        przed, po, kan, kolory, d, opis_pca = self._dane(polsfera)
        L, tryb = self._latent(przed, po)
        tok = np.arange(600)
        if self.chk_tylko.isChecked() and self.token is not None:
            tok = np.array([self.token])
        elif self.maska_tokenow is not None and self.combo_patche.currentIndex():
            m = self.maska_tokenow if self.combo_patche.currentIndex() == 1 else ~self.maska_tokenow
            tok = tok[m]
        if not (self.chk_tylko.isChecked() and self.token is not None) and self.combo_ile.currentIndex():
            tok = np.intersect1d(tok, tokeny_co(CZASTECZKI_CO[self.combo_ile.currentIndex()]))
        if len(tok) == 0:
            for it in (self.linie, self.punkty, self.szare, self.zera, self.luki, self.groty):
                it.setVisible(False)
            self.etykieta.setText('brak patchy do narysowania (obrys nie pokrywa żadnego patcha)')
            return
        polowa = KOMORKA / 2
        r0 = polowa * self.spin_r0.value() / 100
        if tryb == 2:
            zakres = np.abs(L[:, kan])
        else:
            zakres = np.abs(np.concatenate([przed[:, kan]] + ([po[:, kan]] if po is not None and tryb in (1, 3, 4) else [])))
        m = max(float(np.percentile(zakres, 99.5)), 1e-6)            # stała skala w czasie animacji
        skala = (polowa - r0) / m * self.spin_skala.value() / 100 if polowa > r0 else polowa / m
        srodki = self._srodki()                                       # środki na płaszczyźnie obrazu (z = 0)
        pos, konce = self._wektory(L, kan, tok, srodki, d, r0, skala)
        styl = self.combo_styl.currentIndex()
        kol = kolory[kan]
        if not tylko_pozycje or not hasattr(self, '_kol_n') or self._kol_n != (len(tok), len(kan), len(kolory)):
            self._kol_n = (len(tok), len(kan), len(kolory))
            self._kol_odc = np.ascontiguousarray(np.broadcast_to(kol[None, :, None, :], (len(tok), len(kan), 2, 4)).reshape(-1, 4))
            self._kol_pkt = np.ascontiguousarray(np.broadcast_to(kol[None], (len(tok), len(kan), 4)).reshape(-1, 4))
            self.linie.setData(color=self._kol_odc)
            self.punkty.setData(color=self._kol_pkt)
        gr = self.spin_gr.value()
        self.linie.setVisible(styl in (0, 2))
        self.punkty.setVisible(styl in (1, 2))
        if styl in (0, 2):
            self.linie.setData(pos=pos, width=gr)
        if styl in (1, 2):
            self.punkty.setData(pos=konce, size=gr + 1.5)
        if tryb == 3:
            szare, _ = self._wektory(przed, kan, tok, srodki, d, r0, skala)
            self.szare.setData(pos=szare, width=gr); self.szare.setVisible(True)
        else:
            self.szare.setVisible(False)
        if self.chk_zero.isChecked() and r0 > 0:
            th = np.linspace(0, 2 * np.pi, 33)
            o = srodki[tok, None, :] + r0 * np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)[None]
            self.zera.setData(pos=np.stack([o[:, :-1], o[:, 1:]], 2).reshape(-1, 3)); self.zera.setVisible(True)
        else:
            self.zera.setVisible(False)
        if self.token is not None:
            r, k = divmod(self.token, KOLUMNY)
            x0, y0 = k * KOMORKA, WYS - (r + 1) * KOMORKA
            z = .2
            self.ramka.setData(pos=np.array([(x0, y0, z), (x0 + KOMORKA, y0, z), (x0 + KOMORKA, y0 + KOMORKA, z),
                                             (x0, y0 + KOMORKA, z), (x0, y0, z)], np.float32))
            self.ramka.setVisible(True)
        tekst_at = self._luki(srodki) if not tylko_pozycje else getattr(self, '_tekst_at', '')
        self._tekst_at = tekst_at
        if not tylko_pozycje:
            self._legenda()
            if self.chk_tylko.isChecked() and self.token is not None:
                c = srodki[self.token]
                self.widok.setCameraPosition(pos=QVector3D(*map(float, c)), distance=60)
        ms = (time.perf_counter() - t0) * 1000
        opis = self.TRYBY[tryb] + (f' t = {self.suwak.value()} %' if tryb == 4 else '')
        self.etykieta.setText(f'{opis}: {len(tok)} cząsteczek x {len(kan)} wektorów = {len(tok) * len(kan):,} '
                              f'(przeliczenie {ms:.0f} ms); połowa patcha = wartość {m * self.spin_skala.value() / 100:.3g}'
                              + ('' if self.token is None else f'; patch: token {self.token}') + opis_pca + tekst_at)

    def _luki(self, srodki):
        """Łuki źródło -> cel nad obrazem: patch u nasady patrzy na patch na końcu (ciągnie z niego)."""
        strz = self.opcje.strzalki(self.token) if self.chk_atencja.isChecked() and self.atencja_fn is not None else None
        if strz is None:
            self.luki.setVisible(False); self.groty.setVisible(False)
            if self.chk_atencja.isChecked() and self.opcje.combo_strz.currentIndex() in (1, 2) and self.token is None:
                return '; łuki: kliknij patch'
            return ''
        import matplotlib
        src, dst, wg = strz
        a, b = srodki[src], srodki[dst]
        n = 24
        u = np.linspace(0, 1, n + 1, dtype=np.float32)
        h = .35 * np.linalg.norm(b - a, axis=1) + 8                                          # wyższy łuk = dalszy cel
        p = a[:, None] * (1 - u)[None, :, None] + b[:, None] * u[None, :, None]
        p[..., 2] += 4 * h[:, None] * u * (1 - u)
        odc = np.stack([p[:, :-1], p[:, 1:]], 2).reshape(-1, 3)
        kol = matplotlib.colormaps['cool'](wg).astype(np.float32)                           # [E, 4]
        alfa = .15 + .85 * np.repeat(np.stack([u[:-1], u[1:]], 1)[None], len(src), 0)      # jaśniej przy celu
        c = np.repeat(np.repeat(kol[:, None, None, :], n, 1), 2, 2).copy()
        c[..., 3] = alfa * (.35 + .65 * wg[:, None, None])
        self.luki.setData(pos=odc, color=c.reshape(-1, 4), width=1.5 if len(src) > 100 else 2.5)
        kg = kol.copy(); kg[:, 3] = 1
        self.groty.setData(pos=b + np.array([0, 0, .5], np.float32), color=kg, size=4 + 8 * wg)
        self.luki.setVisible(True); self.groty.setVisible(True)
        return f'; łuki: {self.opcje.combo_strz.currentText()} ({len(src)})'

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

    def _grubosc(self, v):
        self.linie.setData(width=v); self.szare.setData(width=v); self.punkty.setData(size=v + 1.5)

    # ------------------------------------------------------------ animacja i zapis
    def _animuj(self, wl):
        if wl:
            if self.po is None:
                self.btn_anim.setChecked(False)
                return
            if self.combo_tryb.currentIndex() != 4:
                self.combo_tryb.setCurrentIndex(4)
            self.timer.start()
        else:
            self.timer.stop()

    def _klatka(self):
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
