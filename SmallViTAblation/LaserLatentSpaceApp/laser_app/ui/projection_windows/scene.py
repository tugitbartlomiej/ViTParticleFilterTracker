"""Scena 3D jednego rzutu (Scene) i lokalne sekcje okienek (LocalSections)."""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QVector3D

from laser_app.ui.gl_view import GLView
from laser_app.ui.projection_windows.shared import S_EKR, strzalki, TLO


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
