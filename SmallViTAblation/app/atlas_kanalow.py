"""Atlas kanałów: mapa każdego kanału (albo składowej PCA) przestrzeni makrowektorów na siatce patchy 20 x 30.

Dla bieżącej klatki przestrzeni makro siły F = sily(t) [600, K] (te same, które pchają cząstki) rozkładane są na K
małych map 20 x 30, posortowanych według zmienności po patchach. Nad każdą mapą pasek w kolorze kanału (ten sam
kolor co mikrowektor kanału w obu widokach 3D). Klik w mapę podświetla kierunek d_c tego kanału w przestrzeni makro.
Mozaika to jeden obraz, więc odświeżenie trwa milisekundy; okno śledzi klatkę przestrzeni makro.
"""
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget

import pyqtgraph as pg

from uklad import dodaj

WIER, KOL, PAS, ODST = 20, 30, 3, 2              # wymiary mapy, pasek koloru kanału, odstęp między mapami


class OknoAtlasu(QWidget):
    def __init__(self, makro):
        super().__init__()
        self.makro = makro
        self.setWindowTitle('Atlas kanałów przestrzeni makrowektorów')
        self.resize(1300, 900)
        lay = QVBoxLayout(self)
        w = QHBoxLayout(); lay.addLayout(w)
        self.combo_sort = QComboBox(); self.combo_sort.addItems(['wg zmienności po patchach (od największej)', 'wg numeru kanału'])
        dodaj(w, 'Kolejność:', self.combo_sort, 'Kolejność map w mozaice.')
        self.combo_skala = QComboBox(); self.combo_skala.addItems(['każda mapa osobno', 'wspólna skala'])
        dodaj(w, 'Skala:', self.combo_skala, 'Każda mapa osobno: kolor do największej wartości tej mapy (widać wzór nawet\n'
              'słabych kanałów). Wspólna: jedna skala dla wszystkich (widać, które kanały są silne).')
        self.spin_ile = QSpinBox(); self.spin_ile.setRange(4, 768); self.spin_ile.setValue(192)
        dodaj(w, 'Map:', self.spin_ile, 'Ile map pokazać (pierwsze w wybranej kolejności).')
        w.addStretch(1)
        self.etykieta = QLabel(''); self.etykieta.setWordWrap(True); lay.addWidget(self.etykieta)
        self.glw = pg.GraphicsLayoutWidget(); lay.addWidget(self.glw, 1)
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.obraz = pg.ImageItem(); self.vb.addItem(self.obraz)
        self.glw.scene().sigMouseClicked.connect(self._klik)
        self.wybrany = None
        self._stan = None
        for s in (self.combo_sort.currentIndexChanged, self.combo_skala.currentIndexChanged, self.spin_ile.valueChanged):
            s.connect(lambda *_: self.odswiez(True))
        self.timer = QTimer(self); self.timer.setInterval(250); self.timer.timeout.connect(self.odswiez)
        self.timer.start()

    def closeEvent(self, e):
        self.makro.podswietl_kanal(None)
        super().closeEvent(e)

    def odswiez(self, sila=False):
        m = self.makro
        if not self.isVisible() or m.P is None:
            return
        stan = (m.t, id(m.D), m.D.shape, m.token)
        if stan == self._stan and not sila:
            return
        self._stan = stan
        F = np.asarray(m.sily(m.t), np.float32)                                     # [600, K]
        K = F.shape[1]
        war = F.var(0)
        kolejnosc = np.argsort(-war) if self.combo_sort.currentIndex() == 0 else np.arange(K)
        kolejnosc = kolejnosc[:min(self.spin_ile.value(), K)]
        n = len(kolejnosc)
        nk = int(np.ceil(np.sqrt(n * 1.6)))                                          # map w wierszu
        nw = int(np.ceil(n / nk))
        H, W = WIER + PAS + ODST, KOL + ODST
        img = np.full((nw * H, nk * W, 3), 25, np.uint8)
        import matplotlib
        cm = matplotlib.colormaps['coolwarm']
        wsp = max(float(np.percentile(np.abs(F[:, kolejnosc]), 99.5)), 1e-9)
        for j, c in enumerate(kolejnosc):
            r, k = divmod(j, nk)
            mapa = F[:, c].reshape(WIER, KOL)
            s = wsp if self.combo_skala.currentIndex() else max(float(np.abs(mapa).max()), 1e-9)
            y0, x0 = r * H, k * W
            img[y0:y0 + PAS, x0:x0 + KOL] = (np.asarray(m.kolory_kan[c][:3]) * 255).astype(np.uint8)
            img[y0 + PAS:y0 + PAS + WIER, x0:x0 + KOL] = (cm((mapa / s + 1) / 2)[..., :3] * 255).astype(np.uint8)
            if m.token is not None:                                                  # wybrany patch: czarna kropka
                tr, tk = divmod(m.token, KOL)
                img[y0 + PAS + tr, x0 + tk] = 0
            if c == self.wybrany:
                img[y0:y0 + H - ODST, x0] = img[y0:y0 + H - ODST, x0 + KOL - 1] = 255
                img[y0, x0:x0 + KOL] = img[y0 + H - ODST - 1, x0:x0 + KOL] = 255
        self._uklad = (kolejnosc, nk, H, W)
        self.obraz.setImage(img.transpose(1, 0, 2), levels=(0, 255))
        opis = (f'klatka {m.t + 1}: {K} {"składowych PCA" if K < 768 else "kanałów"} jako mapy 20 x 30 (czerwony = siła dodatnia, '
                f'niebieski = ujemna, pasek = kolor kanału, czarna kropka = wybrany patch)')
        if self.wybrany is not None and self.wybrany < K:
            v = F[:, self.wybrany]
            opis += (f'. Wybrany kanał {self.wybrany}: zmienność {war[self.wybrany]:.3g} (miejsce {int((war > war[self.wybrany]).sum()) + 1}'
                     f' z {K}), najsilniej w patchu {int(np.abs(v).argmax())}')
        self.etykieta.setText(opis + '. Klik w mapę: kierunek kanału w przestrzeni makro.')

    def _klik(self, ev):
        if getattr(self, '_uklad', None) is None:
            return
        p = self.vb.mapSceneToView(ev.scenePos())
        kolejnosc, nk, H, W = self._uklad
        k, r = int(p.x() // W), int(p.y() // H)
        j = r * nk + k
        if 0 <= k < nk and 0 <= j < len(kolejnosc) and p.x() >= 0 and p.y() >= 0:
            c = int(kolejnosc[j])
            self.wybrany = None if c == self.wybrany else c
            self.makro.podswietl_kanal(self.wybrany)
            self.odswiez(True)
