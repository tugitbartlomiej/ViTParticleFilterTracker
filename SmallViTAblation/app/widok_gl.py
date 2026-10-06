"""Widoki OpenGL (pyqtgraph.opengl) wspólne dla cząsteczek 3D i przestrzeni makrowektorów."""
import numpy as np
from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QVector4D

import pyqtgraph.opengl as gl

from wspolne import SZER, WYS


class WidokGL(gl.GLViewWidget):
    """GLViewWidget odporny na pierwsze klatki: zaraz po pokazaniu widżetu w zakładce Qt woła paintGL, zanim
    kontekst jest gotowy, i OpenGL zgłasza GL_INVALID_OPERATION (1282) przy glClearColor (tak samo pusty
    GLViewWidget). Taką klatkę pomijamy i prosimy o następną; potem rysowanie działa normalnie.
    Przy rozłączonej sesji Windows (pulpit zdalny rozłączony) błąd jest w każdej klatce i widok zostaje czarny."""

    def paintGL(self):
        from OpenGL import GL
        from OpenGL.error import GLError
        try:
            for _ in range(16):
                if GL.glGetError() == GL.GL_NO_ERROR:
                    break
            super().paintGL()
        except GLError:
            self._pominiete = getattr(self, '_pominiete', 0) + 1
            if self._pominiete < 50:
                QTimer.singleShot(30, self.update)


class Widok3D(WidokGL):
    """GLViewWidget, który zgłasza kliknięcie (bez przeciągania) jako punkt na płaszczyźnie obrazu z = 0."""
    klik = pyqtSignal(float, float)

    def mousePressEvent(self, ev):
        self._start = ev.position().toPoint()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        p = ev.position().toPoint()
        if ev.button() == Qt.MouseButton.LeftButton and (p - getattr(self, '_start', QPoint())).manhattanLength() < 4:
            xy = self.punkt_na_obrazie(p.x(), p.y())
            if xy is not None:
                self.klik.emit(*xy)

    def punkt_na_obrazie(self, px, py):
        """Promień z kamery przez piksel (px, py) przecięty z płaszczyzną z = 0; None poza obrazem."""
        w, h = self.width(), self.height()
        m = self.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * self.viewMatrix()
        inv, ok = m.inverted()
        if not ok:
            return None
        nx, ny = 2 * px / w - 1, 1 - 2 * py / h
        a = inv.map(QVector4D(nx, ny, -1, 1)); b = inv.map(QVector4D(nx, ny, 1, 1))
        a = np.array([a.x(), a.y(), a.z()]) / a.w(); b = np.array([b.x(), b.y(), b.z()]) / b.w()
        if abs(b[2] - a[2]) < 1e-9:
            return None
        t = -a[2] / (b[2] - a[2])
        x, y = (a + t * (b - a))[:2]
        return (x, y) if 0 <= x < SZER and 0 <= y < WYS else None
