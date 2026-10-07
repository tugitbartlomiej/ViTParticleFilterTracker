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
    Lustro: self.lustro = (sx, sy, sz) z wartościami ±1 odbija całą scenę względem środka widoku w macierzy widoku
    (obraz, wektory, napisy); klikanie w obraz liczy się z tej samej macierzy, więc dalej trafia.
    Przy rozłączonej sesji Windows (pulpit zdalny rozłączony) błąd jest w każdej klatce i widok zostaje czarny."""

    lustro = (1, 1, 1)

    def viewMatrix(self):
        tr = super().viewMatrix()
        sx, sy, sz = self.lustro
        if (sx, sy, sz) != (1, 1, 1):                           # odbicie względem środka widoku
            c = self.opts['center']
            tr.translate(c.x(), c.y(), c.z()); tr.scale(sx, sy, sz); tr.translate(-c.x(), -c.y(), -c.z())
        return tr

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


class ObrazPrzezroczysty(gl.GLImageItem):
    """Płaszczyzna obrazu z przezroczystością (kanał alfa tekstury), która nie zasłania tego, co leży nad nią,
    i przepuszcza to, co leży pod nią. Rysowana po liniach (wysoka wartość głębi), z testem głębi, ale bez zapisu
    do bufora głębi: wektory nad obrazem zostają ostre, wektory pod obrazem są przyciemnione o krycie obrazu."""

    def __init__(self, dane):
        super().__init__(dane, glOptions='translucent')
        self.setDepthValue(10)

    def paint(self):
        from OpenGL import GL
        GL.glDepthMask(GL.GL_FALSE)
        try:
            super().paint()
        finally:
            GL.glDepthMask(GL.GL_TRUE)


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


class LinieGrube:
    """Odcinki o grubości zależnej od wagi (0..1): OpenGL ma jedną grubość na element, więc odcinki trafiają do
    KLAS elementów GLLinePlotItem o grubościach od 1 px do `grubosc_max` (najsilniejsze najgrubsze).
    API jak GLLinePlotItem w trybie 'lines': setData(pos [2E, 3], color [2E, 4], wagi [E], grubosc_max),
    setVisible, visible, pos (wszystkie odcinki razem)."""
    KLAS = 5

    def __init__(self, widok):
        self.elementy = [gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', glOptions='translucent')
                         for _ in range(self.KLAS)]
        for e in self.elementy:
            e.setVisible(False); widok.addItem(e)
        self.pos = np.zeros((0, 3), np.float32)
        self._widoczne = False

    def setData(self, pos, color, wagi, grubosc_max=6.0):
        pos = np.asarray(pos, np.float32); color = np.asarray(color, np.float32); wagi = np.asarray(wagi, np.float32)
        self.pos = pos
        klasa = np.minimum((np.clip(wagi, 0, 1) * self.KLAS).astype(int), self.KLAS - 1)       # [E]
        grub = np.linspace(1.0, max(1.0, float(grubosc_max)), self.KLAS)
        for k, e in enumerate(self.elementy):
            m = np.repeat(klasa == k, 2)
            if m.any():
                e.setData(pos=np.ascontiguousarray(pos[m]), color=np.ascontiguousarray(color[m]), width=float(grub[k]))
            e.setVisible(bool(m.any()) and self._widoczne)
            e._ma_dane = bool(m.any())

    def setVisible(self, v):
        self._widoczne = bool(v)
        for e in self.elementy:
            e.setVisible(self._widoczne and getattr(e, '_ma_dane', False))

    def visible(self):
        return self._widoczne

