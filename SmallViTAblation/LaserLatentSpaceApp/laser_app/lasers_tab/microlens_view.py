"""Okienko M: mikrosoczewki (microlens_gl.MicrolensScene) razem z drugą sferą reflektorów w jednym widoku 3D.

Zdjęcie na dole naświetla przez płyty łodygi 600 półfabrykatów soczewek (stan 0), w sferze lasery bloków i gwiazda uwagi
formują ich kształt (12 podkroków), na górze lampa prześwietla matrycę gotowych soczewek na ekran: każda soczewka tylko na
własną komórkę (głowa jest na token), jasność = ciepło (pewność), ramka = prostokąt plamki najjaśniejszej soczewki.
Dane: przejście bieżącego checkpointu (history.podkroki, klucz 'glowa' = głowa na każdym z 13 stanów), to samo co okienko P."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QFormLayout

from laser_app.compute import history as hist
from laser_app.gpu.microlens_gl import MicrolensScene
from laser_app.gpu.sphere_gl import ReflectorSphere
from laser_app.lasers_tab.shared import TLO
from laser_app.ui.gl_view import GLView
from laser_app.ui.widgets import CollapsibleSection, NarrowCombo, _spin

FILTRY = ['brak', 'tokeny narzędzia', 'wybrane cząsteczki', 'wszystkie 600']


class MicrolensMixin:
    """Okienko M: mikrosoczewki. Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_sek_mikro(self):
        """Sekcja panelu „Mikrosoczewki” (okienko M)."""
        g = CollapsibleSection('Mikrosoczewki (okienko M): zdjęcie → łodyga → formowanie w sferze → lampa i ekran'); f = QFormLayout(g)
        self.ch_mk_obraz = QCheckBox('zdjęcie na dole: naświetla półfabrykaty soczewek'); self.ch_mk_obraz.setChecked(True)
        self.ch_mk_obraz.setToolTip('Klatka wideo pod sferą. W alegorii światło zdjęcia nie przechodzi na ekran: przez filtry łodygi\n'
                                    'odciska początkowy kształt każdej mikrosoczewki (stan 0 = łodyga + kodowanie pozycji).')
        self.ch_mk_plyty = QCheckBox('płyty łodygi: 4 konwolucje (3 główne składowe cech jako RGB)'); self.ch_mk_plyty.setChecked(True)
        self.ch_mk_plyty.setToolTip('Cztery płyty między zdjęciem a sferą: wyjścia kolejnych konwolucji łodygi (240×160 → 30×20), każde\n'
                                    'jako obraz PCA-RGB. Pokazują, co z obrazu przechodzi do kształtu soczewek.')
        self.sp_mk_krycie_plyt = _spin(0.05, 1.0, 0.55, 0.05, 2); self.sp_mk_krycie_plyt.setToolTip('Krycie płyt łodygi (1 = nieprzezroczyste).')
        self.cb_mk_prom = NarrowCombo(); self.cb_mk_prom.addItems([f'promienie naświetlenia: {x}' for x in FILTRY]); self.cb_mk_prom.setCurrentIndex(1)
        self.cb_mk_prom.setToolTip('Linie od komórki zdjęcia do jej koloidu w sferze (kolor = kolor łaty): które soczewki pokazać.\n'
                                   'Wszystkie 600 naraz zasłaniają sferę; narzędzie albo wybrane cząsteczki odsłaniają szczegóły.')
        self.cb_mk_linki = NarrowCombo(); self.cb_mk_linki.addItems([f'linki koloid → soczewka: {x}' for x in FILTRY]); self.cb_mk_linki.setCurrentIndex(1)
        self.cb_mk_linki.setToolTip('Linie od koloidu w sferze do jego mikrosoczewki w matrycy nad sferą (jaśniejsze = większa transmisja).\n'
                                    'Soczewka przyjmuje kształt koloidu: jego położenie w 192 wymiarach.')
        self.sp_mk_jas_prom = _spin(0.1, 10.0, 1.0, 0.1, 2); self.sp_mk_jas_prom.setToolTip('Jasność promieni naświetlenia i linków (mnożnik).')
        self.sp_mk_krycie_prom = _spin(0.05, 1.0, 0.6, 0.05, 2); self.sp_mk_krycie_prom.setToolTip('Krycie promieni naświetlenia i linków.')
        self.ch_mk_sfera = QCheckBox('sfera reflektorów w środku: koloidy, lasery bloku, linie uwagi'); self.ch_mk_sfera.setChecked(True)
        self.ch_mk_sfera.setToolTip('Druga sfera reflektorów (te same ustawienia co duża sfera, sekcja „Sfera”) w trybie przejścia przez sieć:\n'
                                    'w podkroku MLP świecą lasery bloku, w podkroku uwagi linie między cząsteczkami. Formowanie soczewek.')
        self.ch_mk_socz = QCheckBox('matryca mikrosoczewek nad sferą: jasność = transmisja (ciepło)'); self.ch_mk_socz.setChecked(True)
        self.ch_mk_socz.setToolTip('600 kółek w siatce 30 × 20 nad sferą, jedna soczewka na łatę zdjęcia. Jasność = ile światła lampy\n'
                                   'przepuszcza (sigmoid ciepła z głowy). Żółty obrys = wybrane cząsteczki.')
        self.ch_mk_lampa = QCheckBox('lampa pod matrycą i wiązki przez soczewki na ekran'); self.ch_mk_lampa.setChecked(True)
        self.ch_mk_lampa.setToolTip('Jednostajna lampa świeci przez matrycę; każda soczewka przepuszcza wiązkę tylko na własną komórkę ekranu\n'
                                    '(głowa działa na każdy token osobno, bez mieszania). Jasność wiązki = ciepło.')
        self.sp_mk_jas_lampy = _spin(0.1, 10.0, 1.0, 0.1, 2); self.sp_mk_jas_lampy.setToolTip('Jasność wiązek lampy (mnożnik ciepła).')
        self.sp_mk_prog = _spin(0.0, 1.0, 0.02, 0.01, 3); self.sp_mk_prog.setToolTip('Wiązki lampy słabsze niż ten próg ciepła nie są rysowane.')
        self.cb_mk_ekran = NarrowCombo()
        self.cb_mk_ekran.addItems(['ekran: cień soczewek (ciepło = pewność komórki)', 'ekran: różnica cień − kalka (czerwony za jasno, niebieski za ciemno)',
                                   'ekran: kalka (cel ciepła z etykiety, Gauss wokół środka narzędzia)'])
        self.cb_mk_ekran.setToolTip('Co pokazuje ekran na górze: cień (mapa ciepła 20 × 30 z głowy), różnicę między cieniem a kalką\n'
                                    '(to prowadzi uczenie: gdzie soczewki mają się jeszcze zmienić) albo samą kalkę z etykiety.')
        self.cb_mk_sonda = NarrowCombo()
        self.cb_mk_sonda.addItems(['głowa czyta stan po bieżącym podkroku (sonda; sieć tak nie robi)', 'głowa czyta tylko stan końcowy (jak sieć)'])
        self.cb_mk_sonda.setToolTip('Sieć czyta głową wyłącznie stan po 12. podkroku. Sonda przykłada tę samą głowę do stanu po bieżącym\n'
                                    'podkroku, żeby zobaczyć, jak cień powstaje w głąb sieci (zabieg diagnostyczny, nie działanie sieci).')
        self.sp_mk_gamma = _spin(0.2, 3.0, 1.0, 0.1, 2); self.sp_mk_gamma.setToolTip('Wykładnik jasności ciepła na ekranie, soczewkach i wiązkach (0,5 rozjaśnia słabe).')
        self.ch_mk_ramki = QCheckBox('ramki na ekranie: błękitna z głowy (komórka szczytu ciepła), biała GT'); self.ch_mk_ramki.setChecked(True)
        self.ch_mk_ramki.setToolTip('Ramka z głowy: prostokąt plamki najjaśniejszej soczewki (l, t, r, b z jej komórki). Biała: ramka z etykiety.')
        f.addRow(self.ch_mk_obraz); f.addRow(self.ch_mk_plyty); f.addRow('  krycie płyt', self.sp_mk_krycie_plyt)
        f.addRow(self.cb_mk_prom); f.addRow(self.cb_mk_linki); f.addRow('  jasność promieni', self.sp_mk_jas_prom)
        f.addRow('  krycie promieni', self.sp_mk_krycie_prom); f.addRow(self.ch_mk_sfera)
        f.addRow(self.ch_mk_socz); f.addRow(self.ch_mk_lampa); f.addRow('  jasność lampy', self.sp_mk_jas_lampy); f.addRow('  próg wiązek', self.sp_mk_prog)
        f.addRow(self.cb_mk_ekran); f.addRow(self.cb_mk_sonda); f.addRow('  wykładnik jasności', self.sp_mk_gamma); f.addRow(self.ch_mk_ramki)
        for w in (self.ch_mk_obraz, self.ch_mk_plyty, self.ch_mk_sfera, self.ch_mk_socz, self.ch_mk_lampa, self.ch_mk_ramki):
            w.toggled.connect(self._opcje_mikro)
        for w in (self.cb_mk_prom, self.cb_mk_linki, self.cb_mk_ekran, self.cb_mk_sonda):
            w.currentIndexChanged.connect(self._opcje_mikro)
        for w in (self.sp_mk_krycie_plyt, self.sp_mk_jas_prom, self.sp_mk_krycie_prom, self.sp_mk_jas_lampy, self.sp_mk_prog, self.sp_mk_gamma):
            w.valueChanged.connect(self._opcje_mikro)
        self._sek['mikro'] = g

    def _buduj_okienko_mikro(self):
        """Okienko M: widok 3D ze sceną mikrosoczewek i drugą sferą reflektorów (wspólne opcje z dużą sferą)."""
        self.vm = GLView(); self.vm.setBackgroundColor(TLO); self.vm.setCameraPosition(distance=330, elevation=30, azimuth=-55)
        self.emk = MicrolensScene(); self.vm.addItem(self.emk)
        self.em = ReflectorSphere(hist.R_SFERY); self.em.opcje = self.es.opcje; self.vm.addItem(self.em)
        self._dok('m', 'M. Mikrosoczewki: zdjęcie naświetla, lasery i gwiazda formują, lampa rzuca cień na ekran',
                  self._z_napisem(self.vm, 'lbm'))
        self.doki['m'].visibilityChanged.connect(self._mk_widoczne)
        self._mk_klucz = None; self._mk_zdjecie = None

    # ------------------------------------------------------------------ zasilanie: przejście bieżącego checkpointu (jak okienko P)
    def _mk_widoczne(self, on):
        """Okienko M odsłonięte: po chwili (gdy widać już jego obszar) bierze gotowe przejście albo zleca policzenie."""
        if not on or self.h is None:
            return
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._mk_odsloniete)

    def _mk_odsloniete(self):
        h = self.h
        if h is None or not self.widac('m'):
            return
        krok = int(h.kroki[int(np.clip(round(self.t), 0, h.K - 1))])
        if self.d_pod is not None and self.d_pod.get('krok') == krok and self.d_pod.get('_wagi') == h.wagi:
            self._mk_przejscie(self.d_pod)
        else:
            self._zlec_podkroki()

    def _mk_przejscie(self, d):
        """Nowe przejście (history.podkroki) bieżącego checkpointu: sfera okienka M w trybie podkroków i scena mikrosoczewek."""
        h = self.h
        if h is None or d is None or not self.widac('m'):
            return
        if self._mk_zdjecie != id(h):
            self.emk.ustaw_zdjecie(h.rgb, h.gt, h.narz); self._mk_zdjecie = id(h)
        if self._mk_klucz == (id(d), id(h)):
            return
        self._mk_klucz = (id(d), id(h))
        self.em.ustaw_podkroki(d, h.narz)
        self.emk.ustaw_przejscie(d, self.em._P_surowe[..., :3])
        self._opcje_mikro()

    def _mk_podkrok(self, s):
        """Chwila podkroku 0..12 (ułamkowa) w okienku M: sfera i scena; napis pod widokiem."""
        if not hasattr(self, 'emk') or not self.widac('m'):
            return
        self.em.ustaw_podkrok(s); self.emk.ustaw_czas(s)
        self.lbm.setText(self.emk.opis(s, hist.iou))

    def _opcje_mikro(self, *_):
        if not hasattr(self, 'emk'):
            return
        o = self.emk.opcje
        o.update(obraz=self.ch_mk_obraz.isChecked(), plyty=self.ch_mk_plyty.isChecked(), krycie_plyt=self.sp_mk_krycie_plyt.value(),
                 promienie=self.cb_mk_prom.currentIndex(), linki=self.cb_mk_linki.currentIndex(), jas_prom=self.sp_mk_jas_prom.value(),
                 krycie_prom=self.sp_mk_krycie_prom.value(), soczewki=self.ch_mk_socz.isChecked(), lampa=self.ch_mk_lampa.isChecked(),
                 jas_lampy=self.sp_mk_jas_lampy.value(), prog=self.sp_mk_prog.value(), ekran=self.cb_mk_ekran.currentIndex(),
                 sonda=self.cb_mk_sonda.currentIndex(), gamma=self.sp_mk_gamma.value(), ramki=self.ch_mk_ramki.isChecked(),
                 skala=float(self.es.opcje.get('skala_chmury', 1.0)))
        self.em.setVisible(self.ch_mk_sfera.isChecked())
        if self.h is not None:
            tok = self._wybrane_tok()
            self.emk.ustaw_wybor(tok); self.em.ustaw_wybor(self._wybrane_refl(), tok)
            ks = self.cb_kol_sciezki.currentIndex()
            if ks in (1, 2, 3, 5) and getattr(self, '_mk_ks', None) != (ks, id(self.h)):
                self._mk_ks = (ks, id(self.h)); self.em.ustaw_kolory_sciezek(self._kolory_sfery(ks - 1 if ks < 4 else 3))
        self.emk.update(); self.em.update()
        if self.widac('m') and self.emk.G is not None:
            self.lbm.setText(self.emk.opis(self.s_pod, hist.iou))
