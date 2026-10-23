"""Podkroki sieci: przejście sygnału przez 6 bloków (okienko P, pasek podkroków), zapis przejść na dysk, animacja i opis podkroków."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy,
                             QSlider, QVBoxLayout, QWidget)

from laser_app.compute import history as hist, passage_store
from laser_app.gpu.network_passage_gl import NetworkPassageGPU
from laser_app.lasers_tab.shared import pl, TLO
from laser_app.lasers_tab.widgets import NetworkDiagram
from laser_app.lasers_tab.workers import AllPassagesWorker, plaski_podkrokow, statystyki_podkrokow, SubstepsWorker
from laser_app.ui.dock_windows import bez_ukladu
from laser_app.ui.gl_view import GLView
from laser_app.ui.widgets import CollapsibleSection, NarrowCombo, _spin, _staly_uklad


class SubstepsMixin:
    """Podkroki sieci: przejście sygnału przez 6 bloków (okienko P, pasek podkroków), zapis przejść na dysk, animacja i opis podkroków.
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_sek_zapis(self):
        """Sekcja zapisu przejść przez sieć na dysk."""
        # --- zapis przejść przez sieć (podkroki) na dysk
        g = CollapsibleSection('Zapis przejść przez sieć (podkroki) na dysk'); f = QFormLayout(g)
        self.ch_prz_zapis = QCheckBox('zapisuj policzone przejścia (potem wczytują się zamiast liczenia)'); self.ch_prz_zapis.setChecked(True)
        self.ch_prz_zapis.setToolTip('Każde policzone przejście przez sieć (13 stanów, uwaga, jasności laserów) trafia do katalogu historii '
                                     '(cache_uczenia/lasery/<historia>/przejscia). Zapisane wczytują się zawsze, także gdy to pole jest wyłączone. '
                                     'Wspólny rzut 3D jest w baza.npz, więc zapisane przejścia pasują do siebie także między sesjami. '
                                     'Ogniskowe w osi uczenia (okienko 11) zapisują się osobno i zawsze.')
        self.cb_prz_dokl = NarrowCombo(); self.cb_prz_dokl.addItems(['dokładnie: float32, około 44 MB na checkpoint',
                                                                   'oszczędnie: float16, około 22 MB na checkpoint'])
        self.cb_prz_dokl.setToolTip('Dokładnie: kontrola „suma wkładów = ruch” jak przy liczeniu (~10⁻⁶). Oszczędnie: atencja, jasności i stany '
                                    'w float16, kontrola ~10⁻³.')
        self.sp_prz_limit = _spin(1, 2000, 40, 5, 0, ' GB'); self.sp_prz_limit.setToolTip('Po przekroczeniu tej wielkości katalogu nowe przejścia się nie zapisują.')
        self.sp_prz_co = _spin(1, 200, 1, 1, sufiks=' ckpt'); self.sp_prz_co.setToolTip('Zapis wszystkich: co ile checkpointów (ostatni zawsze).')
        hb_ = QHBoxLayout(); self.bt_prz = QPushButton('Zapisz przejścia wszystkich checkpointów'); self.bt_prz_stop = QPushButton('Przerwij')
        self.bt_prz.clicked.connect(self._prz_wszystkie); self.bt_prz_stop.clicked.connect(self._prz_przerwij); self.bt_prz_stop.setEnabled(False)
        hb_.addWidget(self.bt_prz); hb_.addWidget(self.bt_prz_stop)
        self.pasek_prz = QProgressBar(); self.pasek_prz.setTextVisible(True)
        self.lb_prz = QLabel('—'); self.lb_prz.setWordWrap(True)
        f.addRow(self.ch_prz_zapis); f.addRow('Zapis', self.cb_prz_dokl); f.addRow('Limit katalogu', self.sp_prz_limit)
        f.addRow('Wszystkie: co', self.sp_prz_co); f.addRow(hb_); f.addRow(self.pasek_prz); f.addRow(self.lb_prz)
        self._prz_watek = None; self._prz_bajty = 0
        self._sek['zapis'] = g

    # ------------------------------------------------------------------ okienko „Przejście przez sieć” (podkroki w głębokości)
    def _buduj_przejscie(self):
        pasek = QWidget(); pv = QVBoxLayout(pasek); pv.setContentsMargins(4, 0, 4, 2); pv.setSpacing(2)
        rp = QHBoxLayout()
        self.ch_pod = QCheckBox('Podkroki sieci we wszystkich widokach')
        self.ch_pod.setToolTip('Sfera reflektorów i rzuty 1–3 pokazują jedno przejście przez sieć bieżącego checkpointu: łodyga + PE, '
                               'potem w każdym z 6 bloków uwaga (siły między cząsteczkami) i MLP (768 laserów bloku). Czas cząsteczek = głębokość.')
        self.bt_pod = QPushButton('▶ podkroki'); self.bt_pod.setCheckable(True); self.bt_pod.toggled.connect(self._graj_podkroki)
        self.bt_pod.setToolTip('Animacja przejścia sygnału przez sieć dla bieżącego checkpointu (w pętli).')
        self.suwak_pod = QSlider(Qt.Orientation.Horizontal); self.suwak_pod.setRange(0, 1200)
        self.suwak_pod.valueChanged.connect(lambda x: None if self._blokada else self.ustaw_podkrok(x / 100.0))
        self.sp_pod_tempo = _spin(0.1, 600, 1.0, 0.1, 1, ' podkr./s')   # 600 = 50 przejść sieci/s; krok rośnie z wartością
        self.sp_pod_tempo.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self.lb_podkrok = QLabel('podkrok 0/12: wejście bloków (łodyga + PE)'); _staly_uklad(self.lb_podkrok)
        self.ch_sync = QCheckBox('synchronizuj kroki z podkrokami')
        self.ch_sync.setToolTip('Odtwarzanie osi czasu (▶ na górze): w każdym checkpoincie przechodzą wszystkie 12 podkroków sieci '
                                '(tempo: podkr./s), dopiero potem następny checkpoint (co n).')
        self.sp_sync_co = _spin(1, 200, 1, 1, sufiks=' ckpt'); self.sp_sync_co.setToolTip('Co ile checkpointów skakać po pełnym przejściu sieci.')
        self.sp_slad_pod = _spin(0, 500, 50, 5, sufiks=' ckpt'); self.sp_slad_pod.setSpecialValueText('od nowa')
        self.sp_slad_pod.setToolTip('Przy synchronizacji kroków z podkrokami ścieżki koloidów na sferze („cała historia”) i tory okienka P\n'
                                    'nie zaczynają się od nowa w kolejnym checkpoincie: zostają przejścia tylu ostatnich checkpointów\n'
                                    '(ze stanami powrotu). „od nowa” = jak dawniej, ścieżka tylko bieżącego przejścia.')
        self.sp_slad_pod.valueChanged.connect(self._slad_podkrokow)
        self.sp_przejscie = _spin(0, 6, 0.5, 0.25, 2, ' podkr.')
        self.sp_przejscie.setToolTip('Długość płynnego powrotu z wyjścia sieci do wejścia następnego checkpointu (w podkrokach); 0 = natychmiast.')
        self.ch_wyp = QCheckBox('wypadkowe wektory'); self.ch_wyp.setChecked(True)
        self.ch_wyp.setToolTip('Wypadkowe przesunięcie każdej cząsteczki w bieżącym podkroku (suma sił uwagi albo laserów): '
                               'niebieskie = uwaga, pomarańczowe = MLP; wybrana cząsteczka grubo z grotem.')
        self.sp_skala_wyp = _spin(0.1, 50, 3.0, 0.5, 1, ' ×'); self.sp_skala_wyp.setToolTip('Długość wypadkowych wektorów na ekranie.')
        self.cb_plynnie = QComboBox(); self.cb_plynnie.addItems(['płynne przejście uwaga ↔ MLP', 'osobne podkroki (skokowo)'])
        self.cb_plynnie.setToolTip('Jak wiązki zmieniają się na granicy podkroków (sfera, okienko 11, „Przejście przez sieć”):\n'
                                   '• płynne: wiązki MLP gasną, a linie i wkłady uwagi następnego podkroku rozjaśniają się (przenikanie);\n'
                                   '  reflektory stoją, w nowym bloku z tych samych miejsc świecą neurony tego bloku pod innymi kątami;\n'
                                   '• osobne podkroki: każdy podkrok osobno, wiązki przełączają się od razu na granicy (jak dawniej).')
        self.sp_plynnie = _spin(0.1, 1.0, 0.6, 0.1, 2, ' podkr.')
        self.sp_plynnie.setToolTip('Szerokość przenikania na granicy podkroków (w podkrokach): 1 = przez cały podkrok, 0,2 = krótko '
                                   'przy granicy. W połowie przenikania obie warstwy wiązek świecą po połowie.')
        # dwa wiersze, żeby nic nie wychodziło za prawą krawędź: [podkroki ▶ suwak tempo opis] i [synchronizacja, powrót, wypadkowe]
        for x in (self.ch_pod, self.bt_pod, self.suwak_pod, self.sp_pod_tempo, self.lb_podkrok):
            rp.addWidget(bez_ukladu(x) if x is self.lb_podkrok else x, 3 if x is self.suwak_pod else 2 if x is self.lb_podkrok else 0)
        rp2 = QHBoxLayout()
        for x in (self.ch_sync, self.sp_sync_co, QLabel('ścieżki przez'), self.sp_slad_pod, QLabel('powrót'), self.sp_przejscie, self.ch_wyp, QLabel('długość'), self.sp_skala_wyp,
                  self.cb_plynnie, self.sp_plynnie):
            rp2.addWidget(x)
        rp2.addStretch(1)
        self._rp2 = QWidget(); self._rp2.setLayout(rp2); rp2.setContentsMargins(0, 0, 0, 0)
        for x in (self.ch_wyp,):
            x.toggled.connect(lambda *_: self.ustaw_podkrok(self.s_pod))
        self.sp_skala_wyp.valueChanged.connect(lambda *_: self.ustaw_podkrok(self.s_pod))
        self.cb_plynnie.currentIndexChanged.connect(lambda i: (self.sp_plynnie.setEnabled(i == 0), self.ustaw_podkrok(self.s_pod)))
        self.sp_plynnie.valueChanged.connect(lambda *_: self.ustaw_podkrok(self.s_pod))
        pv.addLayout(rp); pv.addWidget(self._rp2); self._rp2.setVisible(False)
        self.diagram = NetworkDiagram(); self.diagram.klik.connect(lambda s: self.ustaw_podkrok(s))
        self.diagram.setToolTip('Schemat MałegoViT (diagram_maly_vit.pdf). Klik w uwagę albo MLP bloku przechodzi do tego podkroku.')
        self.diagram.setVisible(False)
        pv.addWidget(self.diagram)
        # podkroki sieci: osobne okienko (dok „Podkroki sieci”), które można odpiąć i postawić gdziekolwiek, także na drugim ekranie
        pasek.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.dok_podkroki = self._dok_sterowania('podkroki', 'Podkroki sieci: przejście sygnału przez 6 bloków', pasek)
        self.ch_pod.toggled.connect(self._tryb_podkrokow)
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)
        g_p = CollapsibleSection('Okienko P: przejście przez sieć (siły uwagi, lasery MLP, tory)'); f_p = QFormLayout(g_p)
        self._sek['przejscie'] = g_p
        r = QHBoxLayout(); r.setContentsMargins(4, 0, 4, 0)
        self.cb_sily = NarrowCombo(); self.cb_sily.addItems(['siły uwagi: brak', 'siły na wybraną cząsteczkę', 'wszystkie pary (linie)'])
        self.cb_sily.setCurrentIndex(1)
        self.cb_sily.setToolTip('Podkrok uwagi: f_ij = Σ_h A^h_ij W_o v^h_j, wkład cząsteczki j w przesunięcie i. Linie = kogo słucha i, '
                                'wektory = siły (kolor = kolor cząsteczki j).')
        self.sp_gain_a = _spin(0.01, 100, 1.0, 0.1, 2); self.sp_gain_a.setToolTip('Jasność linii uwagi (× średnia atencja głów).')
        self.sp_skala_sil = _spin(0.01, 1000, 10.0, 1.0, 2); self.sp_skala_sil.setToolTip('Długość wektorów sił f_ij na ekranie (mnożnik).')
        self.ch_lasery = QCheckBox('lasery MLP'); self.ch_lasery.setChecked(True)
        self.ch_lasery.setToolTip('Podkrok MLP: 768 wektorów z_jc · v_c na wybraną cząsteczkę (kolor = laser, błękit = ujemna jasność z < 0).')
        self.sp_skala_l = _spin(0.01, 1000, 3.0, 0.5, 2); self.sp_skala_l.setToolTip('Długość wektorów laserów na ekranie (mnożnik).')
        self.cb_kolor_cz = NarrowCombo(); self.cb_kolor_cz.addItems(['kolor patcha ze zdjęcia', 'położenie patcha na obrazie (siatka 30 × 20: barwa = kolumna, jasność = wiersz)', 'narzędzie',
                                                                    'obszary oka: narzędzie, źrenica, rogówka, tło'])
        self.cb_tory = NarrowCombo(); self.cb_tory.addItems(['tory: brak', 'tor wybranej cząsteczki', 'tory wszystkich'])
        self.cb_tory.setCurrentIndex(1); self.cb_tory.setToolTip('Droga cząsteczki przez kolejne podkroki (głębokość sieci).')
        self.cb_uklad = NarrowCombo(); self.cb_uklad.addItems(['względem środka, jedna skala na checkpoint (nie LayerNorm)', 'względem środka chmury',
                                                              'surowy strumień rezydualny'])
        self.cb_uklad.setToolTip('Jedna skala: x − średnia tokenów w każdym stanie, podzielone przez JEDNĄ liczbę na cały checkpoint '
                                 '(średnia norma tokenu po 13 stanach), więc ruch bierze się tylko z sił, nie ze zmiany skali. To nie jest '
                                 'LayerNorm (ten normuje każdy token osobno). Względem środka: bez dzielenia (norma rośnie z głębokością). '
                                 'W obu układach względem środka siły są wycentrowane tak samo (A_ij − ā_j, z_jc − z̄_c, biasy znikają), '
                                 'więc suma wektorów = przesunięcie cząsteczki. Surowy: pełny stan z biasami, widać wspólny dryf tokenów. '
                                 'Każdy układ ma własny rzut PCA 3D.')
        self._buduj_widocznosc_p()
        # opcje okienka P są w panelu (strona „Podkroki”) i pod prawym klikiem w okienko; samo okienko to widok i opis
        f_p.addRow(self.ch_p_wiazki); f_p.addRow(self._p_przyciski); f_p.addRow(self._p_presety)
        f_p.addRow(self._p_punkty); f_p.addRow('  grubość linii', self.sp_p_grub)
        f_p.addRow('Siły uwagi', self.cb_sily); f_p.addRow(self._p_uwaga_wiersz); f_p.addRow('  głowa', self.cb_p_glowa)
        f_p.addRow('  jasność linii uwagi', self.sp_gain_a); f_p.addRow('  próg atencji', self.sp_p_prog)
        f_p.addRow('  długość sił', self.sp_skala_sil)
        f_p.addRow(self.ch_lasery); f_p.addRow('  znak', self.cb_p_znak_l); f_p.addRow('  najsilniejsze', self.sp_p_top_l)
        f_p.addRow('  próg |z|', self.sp_p_prog_l); f_p.addRow('  długość laserów', self.sp_skala_l); f_p.addRow('  krycie', self.sp_p_krycie_l)
        f_p.addRow('Tory', self.cb_tory); f_p.addRow(self.ch_p_przes)
        f_p.addRow('Układ', self.cb_uklad); f_p.addRow('Kolor cząsteczek', self.cb_kolor_cz)
        self.vp = GLView(); self.vp.setBackgroundColor(TLO); self.vp.setCameraPosition(distance=250, elevation=20, azimuth=-60)
        self.ep = NetworkPassageGPU(); self.vp.addItem(self.ep); self.vp.klik_px.connect(self._klik_przejscie)
        v.addWidget(self.vp, 1)
        self.lb_pod = QLabel('Przejście przez sieć liczy się dla bieżącego checkpointu osi czasu (po wczytaniu historii).')
        self.lb_pod.setWordWrap(True); self.lb_pod.setStyleSheet('color:#ffd166; padding:2px 6px; background:#0b1014')
        self.lb_pod.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        v.addWidget(bez_ukladu(self.lb_pod, 3 * self.lb_pod.fontMetrics().height() + 10))   # 3 linie, bez przeliczania układu
        self._dok('p', 'Przejście przez sieć: podkroki (uwaga = siły między cząsteczkami, MLP = lasery)', w)
        self.d_pod = None; self.s_pod = 0.0; self._kamera_pod = False
        self._kol_cz = None
        self.watek_pod = SubstepsWorker(); self.watek_pod.gotowe.connect(self._podkroki_gotowe)
        self.timer_pod = QTimer(self); self.timer_pod.setInterval(16); self.timer_pod.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer_pod.timeout.connect(self._klatka_podkrokow)
        for x in (self.cb_sily, self.cb_kolor_cz, self.cb_uklad, self.cb_tory):
            x.currentIndexChanged.connect(self._opcje_przejscia)
        for x in (self.cb_p_glowa, self.cb_p_znak_l):
            x.currentIndexChanged.connect(self._opcje_przejscia)
        for x in (self.sp_gain_a, self.sp_skala_sil, self.sp_skala_l, self.sp_p_grub, self.sp_p_prog, self.sp_p_top_l, self.sp_p_prog_l,
                  self.sp_p_krycie_l):
            x.valueChanged.connect(self._opcje_przejscia)
        for x in (self.ch_lasery, self.ch_p_wiazki, self.ch_p_linie, self.ch_p_wektory, self.ch_p_przes, self.ch_p_czastki, self.ch_p_wybrana):
            x.toggled.connect(self._opcje_przejscia)
        self.doki['p'].visibilityChanged.connect(self._p_widoczne)

    def _buduj_widocznosc_p(self):
        """Przełączniki widoczności okienka P: wszystkie wiązki naraz, gotowe zestawy, osobno każdy rodzaj wiązek i punktów."""
        self.ch_p_wiazki = QCheckBox('wszystkie wiązki i linie (odznacz: chowa wszystkie naraz, ustawienia zostają)')
        self.ch_p_wiazki.setChecked(True)
        self.ch_p_wiazki.setToolTip('Jednym kliknięciem chowa albo pokazuje wszystkie wiązki okienka P: linie i wektory uwagi, lasery MLP, '
                                    'przesunięcia w podkroku i tory. Pojedyncze przełączniki poniżej zostają takie, jakie były.')
        przyciski = [('Włącz wszystkie', 'Wszystkie wiązki: linie i wektory uwagi, lasery, przesunięcia, tory.', self._p_wlacz_wszystkie),
                     ('Wyłącz wszystkie', 'Chowa wszystkie wiązki (to samo co odznaczenie pola wyżej).', lambda: self.ch_p_wiazki.setChecked(False)),
                     ('Tylko uwaga', 'Linie i wektory sił uwagi; bez laserów, przesunięć i torów.', lambda: self._p_zestaw(uwaga=True)),
                     ('Tylko lasery', 'Lasery MLP; bez uwagi, przesunięć i torów.', lambda: self._p_zestaw(lasery=True)),
                     ('Tylko ruch', 'Przesunięcia w podkroku i tory; bez uwagi i laserów.', lambda: self._p_zestaw(ruch=True))]
        self._p_przyciski = QWidget(); self._p_presety = QWidget()
        for kontener, lista in ((self._p_przyciski, przyciski[:2]), (self._p_presety, przyciski[2:])):
            hb = QHBoxLayout(kontener); hb.setContentsMargins(0, 0, 0, 0)
            for tekst, dymek, f in lista:
                b = QPushButton(tekst); b.setToolTip(dymek); b.clicked.connect(f); hb.addWidget(b)
        self.ch_p_czastki = QCheckBox('cząsteczki'); self.ch_p_czastki.setChecked(True)
        self.ch_p_czastki.setToolTip('Punkty 600 cząsteczek w bieżącej chwili podkroku.')
        self.ch_p_wybrana = QCheckBox('wybrana cząsteczka (duży punkt)'); self.ch_p_wybrana.setChecked(True)
        self.ch_p_wybrana.setToolTip('Duży żółty punkt wybranej cząsteczki.')
        self._p_punkty = QWidget(); hb = QHBoxLayout(self._p_punkty); hb.setContentsMargins(0, 0, 0, 0)
        hb.addWidget(self.ch_p_czastki); hb.addWidget(self.ch_p_wybrana); hb.addStretch(1)
        self.sp_p_grub = _spin(0.5, 8, 1.0, 0.5, 1, ' ×'); self.sp_p_grub.setToolTip('Grubość wszystkich linii okienka P (mnożnik).')
        self.ch_p_linie = QCheckBox('linie uwagi (kogo słucha)'); self.ch_p_linie.setChecked(True)
        self.ch_p_linie.setToolTip('Linie od wybranej cząsteczki do tych, których słucha (przy „wszystkie pary” także linie wszystkich par).')
        self.ch_p_wektory = QCheckBox('wektory sił f_ij'); self.ch_p_wektory.setChecked(True)
        self.ch_p_wektory.setToolTip('Wektory sił f_ij na wybraną cząsteczkę (kolor = kolor cząsteczki j, która ciągnie).')
        self._p_uwaga_wiersz = QWidget(); hb = QHBoxLayout(self._p_uwaga_wiersz); hb.setContentsMargins(12, 0, 0, 0)
        hb.addWidget(self.ch_p_linie); hb.addWidget(self.ch_p_wektory); hb.addStretch(1)
        self.cb_p_glowa = NarrowCombo(); self.cb_p_glowa.addItems(['średnia 3 głów', 'głowa 1', 'głowa 2', 'głowa 3'])
        self.cb_p_glowa.setToolTip('Linie i siły uwagi z jednej głowy albo wszystkich trzech (linie: średnia atencja, siły: suma głów).')
        self.sp_p_prog = _spin(0.0, 1.0, 0.02, 0.005, 3)
        self.sp_p_prog.setToolTip('Linie i siły słabsze niż ten próg (atencja × jasność) się nie rysują. 0 = wszystkie.')
        self.cb_p_znak_l = NarrowCombo(); self.cb_p_znak_l.addItems(['wszystkie lasery', 'tylko dodatnie (z > 0, kolor lasera)',
                                                                  'tylko ujemne (z < 0, błękitne)'])
        self.cb_p_znak_l.setToolTip('Które lasery MLP rysować według znaku jasności z_jc: dodatnie pchają wzdłuż v_c, ujemne w stronę −v_c.')
        self.sp_p_top_l = _spin(0, 768, 0, 5, sufiks=' laserów'); self.sp_p_top_l.setSpecialValueText('wszystkie ponad próg')
        self.sp_p_top_l.setToolTip('Rysuje tylko tyle najsilniejszych laserów (największe |z|) wybranej cząsteczki; 0 = wszystkie ponad próg.')
        self.sp_p_prog_l = _spin(0.0, 10.0, 0.05, 0.01, 3)
        self.sp_p_prog_l.setToolTip('Lasery o |z| poniżej tego progu się nie rysują. 0 = wszystkie 768.')
        self.sp_p_krycie_l = _spin(0.05, 1.0, 0.85, 0.05, 2); self.sp_p_krycie_l.setToolTip('Krycie (nieprzezroczystość) wiązek laserów.')
        self.ch_p_przes = QCheckBox('przesunięcia w podkroku (stan k → k+1)'); self.ch_p_przes.setChecked(True)
        self.ch_p_przes.setToolTip('Krótkie kreski: skąd i dokąd przesuwa się każda cząsteczka w bieżącym podkroku.')
        self._p_tory = 1

    def _p_wlacz_wszystkie(self):
        """Wszystkie wiązki okienka P: uwaga, lasery, przesunięcia i tory."""
        self._p_zestaw(uwaga=True, lasery=True, ruch=True)

    def _p_zestaw(self, uwaga=False, lasery=False, ruch=False):
        """Gotowy zestaw wiązek okienka P; włącza też bramkę „wszystkie wiązki”. Uwaga przy „siły uwagi: brak” przechodzi na
        siły na wybraną cząsteczkę; tory wracają do ostatniego wybranego trybu."""
        if uwaga and self.cb_sily.currentIndex() == 0:
            self.cb_sily.setCurrentIndex(1)
        if self.cb_tory.currentIndex() > 0:
            self._p_tory = self.cb_tory.currentIndex()
        for ch, on in ((self.ch_p_linie, uwaga), (self.ch_p_wektory, uwaga), (self.ch_lasery, lasery), (self.ch_p_przes, ruch),
                       (self.ch_p_wiazki, True)):
            ch.setChecked(on)
        self.cb_tory.setCurrentIndex(self._p_tory if ruch else 0)

    def _widac_p(self):
        d = self.doki['p']
        return d.isVisible() and (d.isFloating() or not d.visibleRegion().isEmpty())

    def _p_widoczne(self, wid):
        """Okienko szczegółów podkroków odsłonięte: dosyła dane bieżącego checkpointu, jeśli są nowsze niż w nim."""
        if not wid:
            return
        d = self.d_pod
        if d is not None and getattr(self, '_ep_krok', None) != d['krok']:
            if getattr(self, '_kol_cz', None) is None:
                self._kolory_czastek()
            self.ep.ustaw_podkroki(d, self._kol_cz, self._uklad()); self._ep_krok = d['krok']
            if not self._kamera_pod:
                self._kadruj_przejscie(); self._kamera_pod = True
        self._zlec_podkroki()

    def _opcje_przejscia(self, *_):
        o = self.ep.opcje
        o.update(sily=self.cb_sily.currentIndex(), gain=self.sp_gain_a.value(), skala_sil=self.sp_skala_sil.value(),
                 lasery=self.ch_lasery.isChecked(), skala_l=self.sp_skala_l.value(), tory=self.cb_tory.currentIndex(),
                 tok=self.sp_tok.value(), rozm=self.sp_rozm3.value(), wiazki=self.ch_p_wiazki.isChecked(),
                 linie=self.ch_p_linie.isChecked(), wektory=self.ch_p_wektory.isChecked(), przes=self.ch_p_przes.isChecked(),
                 czastki=self.ch_p_czastki.isChecked(), wybrana=self.ch_p_wybrana.isChecked(), grub=self.sp_p_grub.value(),
                 glowa=self.cb_p_glowa.currentIndex(), prog=self.sp_p_prog.value(), znak_l=(0, 1, -1)[self.cb_p_znak_l.currentIndex()],
                 top_l=self.sp_p_top_l.value(), prog_l=self.sp_p_prog_l.value(), alfa_l=self.sp_p_krycie_l.value())
        if self.cb_tory.currentIndex() > 0:
            self._p_tory = self.cb_tory.currentIndex()
        if self.sender() in (self.cb_kolor_cz, self.cb_uklad) and self.d_pod is not None:
            self._kolory_czastek(); self.ep.ustaw_podkroki(self.d_pod, self._kol_cz, self._uklad())
            if self.sender() is self.cb_uklad:
                self._kadruj_przejscie()
        self.ep.update()
        self._opis_podkroku()

    def _zlec_podkroki(self):
        h = self.h
        widac = self.doki['p'].isVisible() and (not self.doki['p'].visibleRegion().isEmpty() or self.doki['p'].isFloating())
        if h is None or not (widac or self.ch_pod.isChecked() or self._og_tryb_pod()):
            return
        krok = int(h.kroki[int(np.clip(round(self.t), 0, h.K - 1))])
        if self.d_pod is not None and self.d_pod.get('krok') == krok and self.d_pod.get('_wagi') == h.wagi:
            return
        pam = self._pamiec_pod()
        if krok in pam:                                              # policzony wcześniej (np. z wyprzedzeniem)
            self._podkroki_gotowe(pam[krok]); return
        if getattr(self, '_zlecony', None) == (h.wagi, krok) and (self.watek_pod.isRunning() or self.watek_pod.w_kolejce(krok)):
            return                                                   # ten checkpoint już się liczy
        self._zlecony = (h.wagi, krok)
        self.watek_pod.zlec(h.wagi, krok, h.rgb, getattr(self, '_baza_pod', None), g=h.g, narz=h.narz, **self._prz_arg())

    def _pamiec_pod(self):
        """Policzone podkroki bieżącej historii i zdjęcia: krok → słownik (najwyżej 8, najstarsze wypadają)."""
        klucz = (self.h.wagi, id(self.h)) if self.h is not None else None
        if getattr(self, '_pam_klucz', None) != klucz:
            from collections import OrderedDict
            baza = passage_store.wczytaj_baze(passage_store.katalog(self.h)) if self.h is not None else None   # rzut zapisanych przejść
            self._pam_klucz, self._pam_pod, self._baza_pod = klucz, OrderedDict(), baza
        return self._pam_pod

    # ------------------------------------------------------------------ zapis przejść na dysk
    def _prz_arg(self):
        if self.h is None:
            return {}
        zapis = self.ch_prz_zapis.isChecked() and self._prz_bajty < self.sp_prz_limit.value() * 1e9
        return dict(kat=passage_store.katalog(self.h), zapis=zapis, f16=self.cb_prz_dokl.currentIndex() == 1)

    def _prz_opis(self, dopisek=''):
        if self.h is None:
            return
        kat = passage_store.katalog(self.h)
        n, b = passage_store.spis(kat); self._prz_bajty = b
        self.lb_prz.setText(f'Zapisane przejścia: {n} z {self.h.K} checkpointów, {b / 1e9:.2f} GB w {kat}. '
                            + ('Wspólny rzut 3D: baza.npz. ' if (kat / 'baza.npz').exists() else '') + dopisek)

    def _prz_wszystkie(self):
        h = self.h
        if h is None or (self._prz_watek is not None and self._prz_watek.isRunning()):
            return
        self._pamiec_pod()
        w = AllPassagesWorker(h, passage_store.katalog(h), self.cb_prz_dokl.currentIndex() == 1, self._baza_pod,
                                    self.sp_prz_co.value(), self.sp_prz_limit.value() * 1e9, self._prz_bajty)
        w.postep.connect(lambda i, n, o: (self.pasek_prz.setMaximum(n), self.pasek_prz.setValue(i), self.pasek_prz.setFormat(o)))
        w.baza.connect(self._prz_baza)
        w.koniec.connect(lambda o: (self.bt_prz.setEnabled(True), self.bt_prz_stop.setEnabled(False), self._prz_opis(o)))
        self._prz_watek = w; self.bt_prz.setEnabled(False); self.bt_prz_stop.setEnabled(True); self.pasek_prz.setValue(0)
        w.start()

    def _prz_baza(self, baza):
        if getattr(self, '_baza_pod', None) is None:
            self._baza_pod = baza

    def _prz_przerwij(self):
        if self._prz_watek is not None:
            self._prz_watek.stop = True

    def _z_wyprzedzeniem(self, j):
        """Zlecenie podkroków checkpointu j w tle (bez wyświetlania), żeby następny krok był gotowy bez czekania."""
        h = self.h
        if h is None or j is None or getattr(self, '_baza_pod', None) is None:
            return
        krok = int(h.kroki[j])
        if krok not in self._pamiec_pod() and not self.watek_pod.w_kolejce(krok):
            self.watek_pod.zlec(h.wagi, krok, h.rgb, self._baza_pod, pilne=False, g=h.g, narz=h.narz, **self._prz_arg())

    def _podkroki_gotowe(self, d):
        if 'blad' in d:
            self.lb_pod.setText(f'Przejście przez sieć: błąd {d["blad"]}'); return
        if self.h is None:
            return
        d['_wagi'] = self.h.wagi
        if d.get('_zapisano'):
            self._prz_bajty += int(d['_zapisano'])
            self._prz_opis()
        pam = self._pamiec_pod()
        pam[d['krok']] = d; pam.move_to_end(d['krok'])
        while len(pam) > 8:
            pam.popitem(last=False)
        if self._baza_pod is None:
            self._baza_pod = d.get('baza')                           # wspólny rzut 3D dla kolejnych checkpointów
        krok_teraz = int(self.h.kroki[int(np.clip(round(self.t), 0, self.h.K - 1))])
        if d['krok'] != krok_teraz or getattr(self, '_w_przejsciu', False):
            return                                                   # policzony z wyprzedzeniem: czeka w pamięci
        self.d_pod = d
        if getattr(self, '_kol_cz', None) is None:
            self._kolory_czastek()
        if self._widac_p():
            self.ep.ustaw_podkroki(d, self._kol_cz, self._uklad()); self._ep_krok = d['krok']
            if not self._kamera_pod:
                self._kadruj_przejscie(); self._kamera_pod = True
        self._opcje_przejscia()
        self.diagram.ustaw_lodyge(self.h.rgb, d)
        self._stat_pod = self._statystyki_podkrokow(d)
        if self.ch_pod.isChecked():
            self._zastosuj_podkroki()
        self.ustaw_podkrok(self.s_pod)

    # ------------------------------------------------------------------ tryb „podkroki sieci we wszystkich widokach”
    def _plaski_podkrokow(self, d):
        if '_pp' not in d:
            d['_pp'] = plaski_podkrokow(d, self.h.g, self.h.narz)
        return d['_pp']

    def _zastosuj_podkroki(self):
        d = self.d_pod
        if d is None or self.h is None:
            return
        self._pp = pp = self._plaski_podkrokow(d)
        for e in (self.e1, self.e2, self.e3):
            e.ustaw_historie(pp)
        self.es.ustaw_podkroki(d, self.h.narz)
        zakres = np.abs(np.concatenate([pp.wezly[..., :2], pp.glb], 1)).max((0, 1)) + 1.5
        self.v1.kadruj(0, 0, 2 * zakres[0], 2 * zakres[1])
        self.ustaw_podkrok(self.s_pod)

    def _dopasuj_podkroki(self):
        """Okienko podkroków przypięte u góry albo u dołu dostaje wysokość treści (schemat sieci i drugi wiersz opcji pokazują się
        dopiero w trybie podkroków); szerokości nie rusza, a gdy jej brakuje, treść się przewija zamiast rozpychać okno."""
        d = self.dok_podkroki
        if d.isFloating() or not d.isVisible() or self.okna.dockWidgetArea(d) not in (Qt.DockWidgetArea.TopDockWidgetArea,
                                                                                      Qt.DockWidgetArea.BottomDockWidgetArea):
            return
        wys = d.widget().sizeHint().height() + (d.titleBarWidget().sizeHint().height() if d.titleBarWidget() else 0) + 4
        self.okna.resizeDocks([d], [min(wys, max(120, self.okna.height() // 2))], Qt.Orientation.Vertical)

    def _tryb_podkrokow(self, on):
        self.diagram.setVisible(on); self._rp2.setVisible(on)            # drugi wiersz opcji podkroków tylko w trybie podkroków
        QTimer.singleShot(0, self._dopasuj_podkroki)
        h = self.h
        if h is None:
            return
        if on:
            if self.bt_play.isChecked() and not self.ch_sync.isChecked():
                self.bt_play.setChecked(False)
            if self.d_pod is not None and self.d_pod.get('krok') == int(h.kroki[int(np.clip(round(self.t), 0, h.K - 1))]):
                self._zastosuj_podkroki()
            else:
                self._zlec_podkroki()
        else:
            self._pp = None
            for e in (self.e1, self.e2, self.e3):
                e.ustaw_historie(h)
            self.es.ustaw_historie(self._h_sfery(), self.cb_skala.currentIndex() == 1)
            zakres = np.abs(np.concatenate([h.wezly[..., :2], h.atomy[..., :2], h.glb], 1)).max((0, 1)) + 1.5
            self.v1.kadruj(0, 0, 2 * zakres[0], 2 * zakres[1])
            for e in (self.e1, self.e2, self.e3):
                e.opcje['pod_kol'] = None
            self.ustaw_czas(self.t)

    def _statystyki_podkrokow(self, d):
        if '_stat' not in d:
            d['_stat'] = statystyki_podkrokow(d)
        return d['_stat']

    def _uklad(self):
        return ('_n', '_c', '')[self.cb_uklad.currentIndex()]

    def _kadruj_przejscie(self):
        from PyQt6.QtGui import QVector3D
        P = self.d_pod['pos' + self._uklad()].reshape(-1, 3)
        lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
        c = (lo + hi) / 2
        self.vp.opts['center'] = QVector3D(*[float(x) for x in c])
        self.vp.setCameraPosition(distance=2.4 * float(np.linalg.norm(hi - lo)), elevation=20, azimuth=-60)

    def _plynnie(self):
        """Szerokość przenikania uwaga ↔ MLP w podkrokach (0 = osobne podkroki, skokowo)."""
        return self.sp_plynnie.value() if hasattr(self, 'cb_plynnie') and self.cb_plynnie.currentIndex() == 0 else 0.0

    def ustaw_podkrok(self, s):
        self.s_pod = s = float(np.clip(s, 0, 12))
        self.es.opcje['plynnie'] = self.ep.opcje['plynnie'] = self._plynnie()
        self.ep.ustaw_podkrok(s); self.diagram.ustaw(s)
        self._blokada = True; self.suwak_pod.setValue(int(round(s * 100))); self._blokada = False
        k = int(np.clip(np.floor(s), 0, 11))
        nazwa = ('wyjście sieci → LayerNorm → głowa' if s >= 12 else 'wejście bloków (łodyga + PE)' if s == 0 else
                 f'blok {k // 2 + 1}: ' + ('uwaga — siły między cząsteczkami' if k % 2 == 0 else 'MLP — 768 laserów bloku'))
        st = getattr(self, '_stat_pod', None)
        if st is not None and s < 12:
            sr, pr = float(st['stos'][k]), float(st['prom'][k])
            nazwa += (f';  rozrzut chmury ×{pl(sr, 3)}, promieniowo {pr:+.3f} r ('
                      + ('do środka' if pr < 0 else 'na zewnątrz') + ')')
        self.lb_podkrok.setText(f'podkrok {pl(s, 2)}/12: {nazwa}')
        self.es.opcje.update(wypadkowe=self.ch_wyp.isChecked(), skala_wyp=self.sp_skala_wyp.value())
        kol_pod = None if (s >= 12 or not self.ch_wyp.isChecked()) else ((0.35, 0.8, 1.0) if k % 2 == 0 else (1.0, 0.62, 0.25))
        for e in (self.e1, self.e2, self.e3):
            e.opcje['pod_kol'] = kol_pod if self.ch_pod.isChecked() else None
            e.opcje['pod_skala'] = self.sp_skala_wyp.value()
        pp = getattr(self, '_pp', None)
        if self.ch_pod.isChecked() and pp is not None:
            for e in (self.e1, self.e2, self.e3):
                e.opcje['L'] = int(min(np.floor(s), 3)); e.ustaw_czas(s)
            self.es.ustaw_podkrok(s)
            u = s - np.floor(s); a = int(np.clip(np.floor(s), 0, 12)); b = min(a + 1, 12)
            par = pp.par[a] * (1 - u) + pp.par[b] * u
            self.lb1.setText(f'podkroki sieci: przesunięcie {pl(float(np.hypot(*par[:2])))} kom.   obrót {par[2]:+.1f}°   skala ×{pl(par[3], 3)}'
                             f'   (mapa liniowa stanów na adresy patchy, R² {pl(pp.r2)})')
            self.lb2.setText(f'odkształcenie siatki w głębokości: średnio {pl(float(pp.lok[a].mean() * (1 - u) + pp.lok[b].mean() * u))} kom.;  '
                             f'{nazwa}')
            self.lb3.setText(f'droga tokenów przez ostatnie podkroki (ślad); przesunięcie w podkroku średnio {pl(float(pp.vib[b].mean()))} kom.')
            cz = pp.cz[a] * (1 - u) + pp.cz[b] * u
            if self.ch_kamera.isChecked():
                self.v3.kadruj(cz[0], cz[1], self.sp_zbl.value(), self.sp_zbl.value() * 2 / 3)
            zb = self.sp_zbl.value()
            self.e2.opcje['box'] = (cz[0] - zb / 2, cz[1] - zb / 3, cz[0] + zb / 2, cz[1] + zb / 3) if self.ch_kamera.isChecked() else None
            self.lbs.setText(f'podkroki: {nazwa};  sfera = stany względem środka, jedna skala na checkpoint, reflektory bloku {k // 2 + 1} '
                             f'(błękit = ujemna jasność)')
        self._opis_podkroku()
        if self._og_tryb_pod():
            self._og_podkrok(s)

    def _slad_podkrokow(self, n):
        """Ile ostatnich przejść (checkpointów) pamiętają ścieżki sfery i tory okienka P przy synchronizacji; 0 czyści ślad."""
        self.es.slad_max = self.ep.slad_max = int(n)
        if not n:
            self.es.czysc_slad()
            self.ep.n_slad = 0; self.ep._slad = []; self.ep.update()

    def _graj_podkroki(self, on):
        self.bt_pod.setText('⏸ podkroki' if on else '▶ podkroki')
        self._zegar_pod = None
        self._s_anim = self.s_pod
        self.timer_pod.start() if on else self.timer_pod.stop()

    def _klatka_podkrokow(self):
        """Pętla 0 → 12 podkroków i krótka pauza na wyjściu sieci."""
        a = getattr(self, '_s_anim', self.s_pod) + self.sp_pod_tempo.value() * self._dt('pod')
        if a > 12.8:                                                 # reszta przechodzi na następną pętlę (stałe tempo także przy setkach podkr./s)
            a %= 12.8
        self._s_anim = a
        self.ustaw_podkrok(min(a, 12.0))

    def _opis_podkroku(self):
        d = self.d_pod
        if d is None:
            return
        s = self.s_pod; k = int(np.clip(np.floor(s), 0, 11)); j = self.sp_tok.value()
        rr, cc = divmod(j, hist.W)
        pos = d['pos']
        if s >= 12:
            self.lb_pod.setText(f'krok {d["krok"]}: koniec przejścia → LayerNorm → głowa: pewność {pl(d["pewnosc"])}, '
                                f'ramka [{", ".join(f"{x:.0f}" for x in d["ramka"])}] px;  patch {j} (wiersz {rr}, kolumna {cc}): '
                                f'mapa środka {pl(float(d["mapa"][j]))}.  Błąd względny „suma sił = ruch”: surowy {d["sprawdzenie"]:.1e}, '
                                f'środek {d.get("sprawdzenie_c", float("nan")):.1e}, jedna skala {d.get("sprawdzenie_n", float("nan")):.1e}')
            return
        b = k // 2
        dx = np.linalg.norm(pos[k + 1] - pos[k], axis=1)
        if k % 2 == 0:
            A = d['ah'][b, :, j].mean(0); A[j] = 0
            top = np.argsort(A)[::-1][:3]
            f = np.einsum('hj,hjd->jd', d['ah'][b, :, j], d['uh'][b])
            mf = np.linalg.norm(f, axis=1); mf[j] = 0
            gr = np.stack(np.divmod(np.arange(hist.TOK), hist.W), 1)
            sas = np.abs(gr - gr[j]).max(1) <= 1
            udz = float(mf[sas].sum() / max(mf.sum(), 1e-12))
            self.lb_pod.setText(f'krok {d["krok"]}, blok {b + 1}, UWAGA: Δx_i = Σ_j f_ij + b_o, f_ij = Σ_h A^h_ij W_o v^h_j.  '
                                f'Średnie |Δx| {pl(float(dx.mean()))};  patch {j} (w{rr}, k{cc}): |Δx| {pl(float(dx[j]))}, słucha najbardziej '
                                + ', '.join(f'{t} (w{t // hist.W}, k{t % hist.W}) {A[t]:.0%}' for t in top)
                                + f';  udział sąsiadów na siatce w sile {udz:.0%} (8 z 599 patchy).')
        else:
            z = d['z'][b, j]; L = z[:, None] * d['pv'][b]
            m = np.linalg.norm(L, axis=1); top = np.argsort(m)[::-1][:3]
            self.lb_pod.setText(f'krok {d["krok"]}, blok {b + 1}, MLP: Δx_j = Σ_c z_jc v_c + b₂ (768 laserów).  Średnie |Δx| {pl(float(dx.mean()))};  '
                                f'patch {j} (w{rr}, k{cc}): |Δx| {pl(float(dx[j]))}, świeci {(z > 0).sum()} laserów, najsilniejsze '
                                + ', '.join(f'c{c} (z {pl(float(z[c]))})' for c in top) + '.')

    def _klik_przejscie(self, x, y):
        if self.d_pod is None:
            return
        s = self.s_pod; a = int(np.clip(np.floor(s), 0, 11)); u = s - a
        X = self.d_pod['pos' + self._uklad()]
        P = X[a] * (1 - u) + X[min(a + 1, 12)] * u
        self._wybierz_tok(self.vp, P, x, y)
