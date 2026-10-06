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
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from uklad import Przeplyw, dodaj
from widok_gl import WidokGL
from wspolne import KOLUMNY, kierunki, kolory_kanalow


class WidokPrzestrzeniMakro(QWidget):
    token_wybrany = pyqtSignal(int)
    KIERUNKI = ['sfera Fibonacciego (jak widok 3D)', 'losowe (stałe ziarno)', 'PCA latentu sekwencji (3 składowe)']
    KOLORY = ['wg położenia patcha (kolumna = barwa, wiersz = jasność)', 'narzędzie (czerwone) i tło (szare)',
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
        lay = QVBoxLayout(self)

        def wiersz():
            w = Przeplyw(); lay.addLayout(w)
            return w

        w = wiersz()
        self.combo_kier = dodaj(w, 'Kierunki kanałów:', QComboBox(),
                                'Stały kierunek d_c każdego z 768 kanałów (makrowektor):\n'
                                '- sfera Fibonacciego: te same kierunki co w widoku 3D nad obrazem (pełna sfera),\n'
                                '- losowe: 768 losowych kierunków jednostkowych (projekcja losowa),\n'
                                '- PCA: kierunek kanału = jego ładunki w 3 głównych składowych latentu sekwencji;\n'
                                '  największy rozrzut cząstek, ale kierunki zależą od sekwencji.')
        self.combo_kier.addItems(self.KIERUNKI); self.combo_kier.setCurrentIndex(2)
        self.combo_kan = dodaj(w, 'Siły z kanałów:', QComboBox(), 'Które kanały pchają cząstkę (pozostałe mają siłę 0).')
        self.combo_kan.addItems(['wszystkie 768', 'tylko J128 (narzędzie)', 'wszystkie poza J128'])
        self.chk_srodek = QCheckBox('odejmij średni latent'); self.chk_srodek.setChecked(True)
        self.chk_srodek.setToolTip('Siła kanału = z − μ (średnia sekwencji). Bez tego wszystkie cząstki odlatują razem\n'
                                   'w stronę średniej, bo po GELU prawie wszystkie wartości są dodatnie.')
        w.addWidget(self.chk_srodek)
        self.combo_kolor = dodaj(w, 'Kolor cząstek:', QComboBox(), 'Jak barwić 600 cząstek.')
        self.combo_kolor.addItems(self.KOLORY)
        w.addStretch(1)

        w = wiersz()
        self.spin_slad = dodaj(w, 'Ślad (klatek):', QSpinBox(), 'Ile poprzednich klatek pokazuje linia toru każdej cząstki. 0 = bez śladu.')
        self.spin_slad.setRange(0, 200); self.spin_slad.setValue(8)
        self.spin_rozmiar = dodaj(w, 'Rozmiar:', QDoubleSpinBox(), 'Średnica cząstek w pikselach ekranu.')
        self.spin_rozmiar.setRange(1, 30); self.spin_rozmiar.setValue(6)
        self.combo_mikro = dodaj(w, 'Mikrowektory wybranego patcha:', QComboBox(),
                                 'Jak pokazać 768 mikrowektorów (z_c − μ_c) · s · d_c wybranej cząstki. Oba sposoby\n'
                                 'to to samo dodawanie wektorów, wynik jest ten sam: położenie cząstki.\n'
                                 '- ze wspólnego środka: każdy mikrowektor wychodzi z tego samego punktu (środek układu),\n'
                                 '  biała strzałka = ich suma (wypadkowa), kończy się w cząstce;\n'
                                 '- łańcuch: wektory dokładane koniec do początku (od najsilniejszego), ostatni kończy się w cząstce;\n'
                                 '- oba naraz.')
        self.combo_mikro.addItems(['ze wspólnego środka + wypadkowa', 'łańcuch (koniec do początku)', 'oba', 'brak'])
        self.spin_mikro = dodaj(w, 'Powiększ mikrowektory:', QDoubleSpinBox(),
                                'Mnożnik długości mikrowektorów ze wspólnego środka (tylko rysunek, wypadkowa bez zmian).\n'
                                'Pojedynczy mikrowektor jest zwykle dużo krótszy od wypadkowej.')
        self.spin_mikro.setRange(1, 500); self.spin_mikro.setValue(1); self.spin_mikro.setPrefix('× ')
        self.chk_gwiazda = QCheckBox('kierunki kanałów'); self.chk_gwiazda.setChecked(True)
        self.chk_gwiazda.setToolTip('Gwiazda 768 kierunków d_c w środku układu (kolor kanału jak w widoku 3D).')
        w.addWidget(self.chk_gwiazda)
        self.chk_tylko_maska = QCheckBox('tylko cząstki narzędzia')
        self.chk_tylko_maska.setToolTip('Pokazuje tylko patche w obrysie narzędzia bieżącej klatki.')
        w.addWidget(self.chk_tylko_maska)
        self.spin_przejscie = dodaj(w, 'Przelot:', QSpinBox(), 'Czas płynnego przelotu cząstek do nowej klatki (ms). 0 = skok.')
        self.spin_przejscie.setRange(0, 3000); self.spin_przejscie.setValue(250); self.spin_przejscie.setSuffix(' ms')
        b = QPushButton('Kamera od nowa'); b.clicked.connect(self._kamera); w.addWidget(b)
        w.addStretch(1)
        self.etykieta = QLabel('Wczytaj sekwencję w zakładce filmu.'); self.etykieta.setWordWrap(True)
        lay.addWidget(self.etykieta)

        self.widok = WidokGL()
        self.widok.setBackgroundColor(pg.mkColor(10, 10, 14))
        self.widok.setMinimumHeight(400)
        lay.addWidget(self.widok, 1)
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
        self.punkty = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=6, pxMode=True, glOptions='translucent')
        v.addItem(self.punkty)
        self.wybrany = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=16, color=(1, 1, 1, 1), pxMode=True)
        self.wybrany.setVisible(False)
        v.addItem(self.wybrany)
        self._kamera()

        self._timer = QTimer(self); self._timer.setInterval(16); self._timer.timeout.connect(self._krok_przelotu)
        self._od = self._do = None; self._u = 1.0
        for s in (self.combo_kier.currentIndexChanged, self.combo_kan.currentIndexChanged, self.chk_srodek.toggled):
            s.connect(lambda *_: self._przelicz())
        for s in (self.combo_kolor.currentIndexChanged, self.spin_slad.valueChanged, self.spin_rozmiar.valueChanged,
                  self.combo_mikro.currentIndexChanged, self.spin_mikro.valueChanged,
                  self.chk_gwiazda.toggled, self.chk_tylko_maska.toggled):
            s.connect(lambda *_: self._rysuj(skok=True))

    def _kamera(self):
        self.widok.setCameraPosition(pos=QVector3D(0, 0, 0), distance=420, elevation=25, azimuth=45)

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

    def _przelicz(self):
        """Kierunki D [768, 3], środek μ, skala s i położenia wszystkich cząstek we wszystkich klatkach."""
        if self.Z is None:
            return
        T = len(self.Z)
        mu = np.zeros(768, np.float32)
        if self.chk_srodek.isChecked():
            mu = np.mean([self.Z[t].astype(np.float32).mean(0) for t in range(T)], 0)
        k = self.combo_kier.currentIndex()
        if k == 0:
            D = kierunki(False)
        elif k == 1:
            D = np.random.default_rng(0).standard_normal((768, 3)).astype(np.float32)
            D /= np.linalg.norm(D, axis=1, keepdims=True)
        else:
            krok = max(1, T // 50)
            X = np.concatenate([self.Z[t].astype(np.float32) for t in range(0, T, krok)]) - mu
            _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
            D = Vt[:3].T.astype(np.float32)                                    # ładunki: [768, 3]
        D = D * self._kanaly_maska()[:, None]
        self.D, self.mu = D, mu
        surowe = np.stack([(self.Z[t].astype(np.float32) - mu) @ D for t in range(T)])   # [T, 600, 3]
        r = np.linalg.norm(surowe, axis=-1)
        self.s = 100 / max(float(np.percentile(r, 99)), 1e-9)
        self.P = surowe * self.s
        self._rysuj(skok=True)

    def pokaz_klatke(self, t, token=None):
        if self.P is None:
            return
        self.token = token if token is not None else self.token
        stare = self._biezace() if self._od is not None else None
        self.t = int(np.clip(t, 0, len(self.P) - 1))
        if stare is not None and self.spin_przejscie.value() > 0:
            self._od, self._u = stare, 0.0
            self._t0 = time.perf_counter()
            self._timer.start()
            self._rysuj()
        else:
            self._rysuj(skok=True)

    def _biezace(self):
        if self._od is None or self._u >= 1:
            return self.P[self.t]
        u = self._u * self._u * (3 - 2 * self._u)                              # łagodny start i koniec
        return self._od + (self.P[self.t] - self._od) * u

    def _krok_przelotu(self):
        self._u = min(1.0, (time.perf_counter() - self._t0) * 1000 / max(1, self.spin_przejscie.value()))
        self._rysuj(tylko_punkty=True)
        if self._u >= 1:
            self._timer.stop()

    # ------------------------------------------------------------ rysowanie
    def _kolory(self):
        i = self.combo_kolor.currentIndex()
        t = np.arange(600)
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

    def _widoczne(self):
        if self.chk_tylko_maska.isChecked() and self.maski is not None:
            return np.flatnonzero(self.maski[self.t])
        return np.arange(600)

    def _rysuj(self, skok=False, tylko_punkty=False):
        if self.P is None:
            return
        if skok:
            self._od, self._u = self.P[self.t], 1.0
            self._timer.stop()
        P = self._biezace()
        tok = self._widoczne()
        kol = self._kolory()
        self.punkty.setData(pos=P[tok], color=kol[tok], size=self.spin_rozmiar.value())
        if self.token is not None:
            self.wybrany.setData(pos=P[self.token][None], size=self.spin_rozmiar.value() + 10, color=(1, 1, 1, .9))
            self.wybrany.setVisible(True)
        else:
            self.wybrany.setVisible(False)
        tryb_m = self.combo_mikro.currentIndex()
        jest = self.token is not None and self.Z is not None
        if jest and tryb_m != 3:
            v = (self.Z[self.t, self.token].astype(np.float32) - self.mu)[:, None] * self.D * self.s   # [768, 3]
            kol_v = self.kolory_kan
            # przelot: w trakcie przelotu wszystko jedzie razem z cząstką, w spoczynku początek = (0, 0, 0)
            o = (P[self.token] - v.sum(0)) if self._u < 1 else np.zeros(3, np.float32)
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
            c[:, 3] = np.repeat(.25 + .75 * dl / max(float(dl.max()), 1e-9), 2)  # silniejszy = mniej przezroczysty
            self.mikro.setData(pos=np.ascontiguousarray(seg, np.float32), color=c)
            self.wypadkowa.setData(pos=np.stack([o, P[self.token]]).astype(np.float32))
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
            self.slady.setVisible(True)
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
                                 f'odległość od środka {float(np.linalg.norm(self.P[self.t, self.token])):.1f}'))
