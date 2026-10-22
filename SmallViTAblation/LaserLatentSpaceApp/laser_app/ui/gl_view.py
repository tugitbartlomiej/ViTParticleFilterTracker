"""Widoki OpenGL (pyqtgraph.opengl) wspólne dla cząsteczek 3D i przestrzeni makrowektorów."""
import weakref

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtCore import pyqtSignal, QPoint, QRect, QSize, Qt, QTimer
from PyQt6.QtGui import QGuiApplication, QMatrix4x4



class GLView(gl.GLViewWidget):
    """GLViewWidget odporny na pierwsze klatki: zaraz po pokazaniu widżetu w zakładce Qt woła paintGL, zanim
    kontekst jest gotowy, i OpenGL zgłasza GL_INVALID_OPERATION (1282) przy glClearColor (tak samo pusty
    GLViewWidget). Taką klatkę pomijamy i prosimy o następną; potem rysowanie działa normalnie.
    Lustro: self.lustro = (sx, sy, sz) z wartościami ±1 odbija całą scenę względem środka widoku w macierzy widoku
    (obraz, wektory, napisy); klikanie w obraz liczy się z tej samej macierzy, więc dalej trafia.
    Przy rozłączonej sesji Windows (pulpit zdalny rozłączony) błąd jest w każdej klatce i widok zostaje czarny."""

    lustro = (1, 1, 1)
    # Stereo 3D „side by side” (monitory 3D typu Odyssey 3D): lewe oko w lewej, prawe w prawej połowie ekranu, każde zgniecione do
    # połowy szerokości (half-SBS, wyświetlacz je rozciąga). Ustawienia wspólne dla wszystkich widoków (klasowe).
    stereo = False
    stereo_odstep = 0.035                        # odstęp oczu jako ułamek odległości kamery od środka (płaszczyzna zerowego rozrzutu)
    stereo_zamiana = False                       # zamiana oczu (L ↔ P)
    _instancje = weakref.WeakSet()
    _oko = 0.0                                   # przesunięcie kamery w układzie widoku w bieżącym przebiegu (0 = obraz zwykły)
    klik_px = pyqtSignal(float, float)            # klik bez przeciągania: piksel w widżecie
    prostokat_px = pyqtSignal(float, float, float, float)   # Shift + przeciągnięcie: prostokąt w pikselach

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        GLView._instancje.add(self)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._klawisze = set()
        self._czas_kl = QTimer(self); self._czas_kl.setInterval(16); self._czas_kl.timeout.connect(self._tik_klawiszy)
        self._t_kl = 0.0

    # ------------------------------------------------------------------ klawiatura w aktywnym widoku (pod kursorem myszy albo po kliknięciu)
    # W/S obrót w górę/dół, A/D obrót w lewo/prawo, Q/E oddal/zbliż (też −/+), I/K/J/L przesuwanie w płaszczyźnie widoku,
    # Home albo 0: powrót do kadru, Shift = szybciej (×3), Ctrl = wolniej (×0,3). W widoku z góry WASD przesuwa. Strzałki i spacja
    # zostają dla osi czasu zakładki.
    _KLAWISZE = {Qt.Key.Key_W: 'w', Qt.Key.Key_S: 's', Qt.Key.Key_A: 'a', Qt.Key.Key_D: 'd', Qt.Key.Key_Q: 'q', Qt.Key.Key_E: 'e',
                 Qt.Key.Key_Minus: 'q', Qt.Key.Key_Plus: 'e', Qt.Key.Key_Equal: 'e', Qt.Key.Key_I: 'i', Qt.Key.Key_K: 'k',
                 Qt.Key.Key_J: 'j', Qt.Key.Key_L: 'l'}

    def enterEvent(self, ev):
        """Aktywne jest okno pod kursorem; nie odbieramy fokusu polom tekstowym i listom rozwijanym."""
        from PyQt6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QTextEdit
        f = QApplication.focusWidget()
        if not isinstance(f, (QLineEdit, QTextEdit, QAbstractSpinBox, QComboBox)):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().enterEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Home, Qt.Key.Key_0):
            if getattr(self, '_kadr', None) is not None and hasattr(self, 'kadruj'):
                self.kadruj(*self._kadr)
            elif callable(getattr(self, 'reset_widoku', None)):
                self.reset_widoku()
            ev.accept()
            return
        k = self._KLAWISZE.get(ev.key())
        if k is None or (ev.modifiers() & Qt.KeyboardModifier.AltModifier):
            super().keyPressEvent(ev)
            return
        self._klawisze.add(k)
        if not self._czas_kl.isActive():
            import time
            self._t_kl = time.perf_counter()
            self._czas_kl.start()
        ev.accept()

    def keyReleaseEvent(self, ev):
        k = self._KLAWISZE.get(ev.key())
        if k is not None and not ev.isAutoRepeat():
            self._klawisze.discard(k)
            if not self._klawisze:
                self._czas_kl.stop()
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    def focusOutEvent(self, ev):
        self._klawisze.clear(); self._czas_kl.stop()
        super().focusOutEvent(ev)

    def _tik_klawiszy(self):
        import time
        now = time.perf_counter()
        dt = min(now - self._t_kl, 0.1); self._t_kl = now
        mod = QGuiApplication.keyboardModifiers()
        v = 3.0 if mod & Qt.KeyboardModifier.ShiftModifier else (0.3 if mod & Qt.KeyboardModifier.ControlModifier else 1.0)
        kl = self._klawisze
        h = (('d' in kl) - ('a' in kl)); u = (('w' in kl) - ('s' in kl))
        if h or u:
            if self.obrot:
                self.orbit(-h * 90.0 * v * dt, u * 60.0 * v * dt)           # 90°/s w poziomie, 60°/s w pionie
            else:
                self.pan(-h * 400 * v * dt, u * 400 * v * dt, 0, relative='view')
        z = ('e' in kl) - ('q' in kl)
        if z:
            self.opts['distance'] = float(np.clip(self.opts['distance'] * np.exp(-1.2 * v * dt * z), 1e-3, 1e6))
        ph, pv = ('l' in kl) - ('j' in kl), ('i' in kl) - ('k' in kl)
        if ph or pv:
            self.pan(-ph * 400 * v * dt, pv * 400 * v * dt, 0, relative='view')
        self.update()

    @classmethod
    def ustaw_stereo(cls, wlacz=None, odstep=None, zamiana=None):
        """Przełącznik 2D ↔ 3D dla wszystkich widoków OpenGL naraz; odświeża je."""
        if wlacz is not None:
            GLView.stereo = bool(wlacz)
        if odstep is not None:
            GLView.stereo_odstep = float(odstep)
        if zamiana is not None:
            GLView.stereo_zamiana = bool(zamiana)
        for w in list(GLView._instancje):
            try:
                w.update()
            except RuntimeError:                         # widok już usunięty
                pass

    def mousePressEvent(self, ev):
        self._start_px = ev.position().toPoint()
        if ev.button() == Qt.MouseButton.LeftButton and ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            from PyQt6.QtWidgets import QRubberBand
            if getattr(self, '_gumka', None) is None:
                self._gumka = QRubberBand(QRubberBand.Shape.Rectangle, self)
            self._gumka_start = self._start_px
            self._gumka.setGeometry(QRect(self._gumka_start, QSize())); self._gumka.show()
            return
        super().mousePressEvent(ev)

    obrot = True                                 # False: widok z góry (lewy przycisk przesuwa zamiast obracać)
    prawy_px = pyqtSignal(float, float)          # prawy klik bez przeciągania: piksel w widżecie

    def mouseMoveEvent(self, ev):
        """Nawigacja: LPM obraca (Ctrl+LPM albo ŚPM przesuwa w płaszczyźnie widoku), PPM + przeciągnięcie góra/dół zbliża i oddala,
        Ctrl+ŚPM przesuwa w pionie, Shift+LPM rysuje prostokąt (zaznaczanie). Kółko zbliża do kursora, Ctrl+kółko zmienia pole widzenia."""
        if getattr(self, '_gumka_start', None) is not None:
            self._gumka.setGeometry(QRect(self._gumka_start, ev.position().toPoint()).normalized())
            return
        p = ev.position()
        poprz = getattr(self, 'mousePos', p)
        self.mousePos = p
        d = p - poprz
        b, ctrl = ev.buttons(), bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if b == Qt.MouseButton.LeftButton:
            if ctrl or not self.obrot:
                self.pan(d.x(), d.y(), 0, relative='view')
            else:
                self.orbit(-d.x(), d.y())
        elif b == Qt.MouseButton.MiddleButton:
            if ctrl:
                self.pan(0, 0, d.y(), relative='view-upright')
            else:
                self.pan(d.x(), d.y(), 0, relative='view')
        elif b == Qt.MouseButton.RightButton:
            self.opts['distance'] *= 1.006 ** d.y()          # w dół oddala, w górę zbliża
            self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y() or ev.angleDelta().x()
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.opts['fov'] = float(np.clip(self.opts['fov'] * 0.999 ** delta, 5, 120))
            self.update()
            return
        f = 0.999 ** delta
        c = ev.position()
        # punkt pod kursorem zostaje w miejscu: środek widoku przesuwamy w stronę kursora o (1 − f) odstępu (w skali sprzed zoomu)
        self.pan(-(c.x() - self.width() / 2) * (1 - f), -(c.y() - self.height() / 2) * (1 - f), 0, relative='view')
        self.opts['distance'] = float(np.clip(self.opts['distance'] * f, 1e-3, 1e6))
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:       # podwójny ŚPM: wróć do kadru (widok z góry albo funkcja sceny)
            if getattr(self, '_kadr', None) is not None and hasattr(self, 'kadruj'):
                self.kadruj(*self._kadr)
                return
            if callable(getattr(self, 'reset_widoku', None)):
                self.reset_widoku()
                return
        super().mouseDoubleClickEvent(ev)

    def mouseReleaseEvent(self, ev):
        if getattr(self, '_gumka_start', None) is not None:
            a, b = self._gumka_start, ev.position().toPoint()
            self._gumka_start = None; self._gumka.hide()
            self.prostokat_px.emit(float(min(a.x(), b.x())), float(min(a.y(), b.y())),
                                   float(max(a.x(), b.x())), float(max(a.y(), b.y())))
            return
        super().mouseReleaseEvent(ev)
        p = ev.position().toPoint()
        if ev.button() == Qt.MouseButton.LeftButton and (p - getattr(self, '_start_px', QPoint())).manhattanLength() < 4:
            self.klik_px.emit(float(p.x()), float(p.y()))
        elif ev.button() == Qt.MouseButton.RightButton and (p - getattr(self, '_start_px', QPoint())).manhattanLength() < 4:
            self.prawy_px.emit(float(p.x()), float(p.y()))

    def na_ekran(self, P):
        """Punkty sceny P [n, 3] -> piksele widżetu [n, 2] i głębia [n] (inf za kamerą)."""
        w, h = self.width(), self.height()
        m = self.projectionMatrix((0, 0, w, h), (0, 0, w, h)) * self.viewMatrix()
        M = np.array(m.data(), np.float64).reshape(4, 4).T                 # Qt trzyma macierz kolumnami
        X = np.concatenate([np.asarray(P, np.float64), np.ones((len(P), 1))], 1) @ M.T
        wsp = X[:, 3:4]
        przod = wsp[:, 0] > 1e-9
        n = X[:, :3] / np.where(przod[:, None], wsp, 1)
        px = np.stack([(n[:, 0] + 1) / 2 * w, (1 - n[:, 1]) / 2 * h], 1)
        return px, np.where(przod, n[:, 2], np.inf)

    def viewMatrix(self):
        tr = super().viewMatrix()
        sx, sy, sz = self.lustro
        if (sx, sy, sz) != (1, 1, 1):                           # odbicie względem środka widoku
            c = self.opts['center']
            tr.translate(c.x(), c.y(), c.z()); tr.scale(sx, sy, sz); tr.translate(-c.x(), -c.y(), -c.z())
        if self._oko:
            m = QMatrix4x4(); m.translate(-self._oko, 0.0, 0.0)          # oko przesunięte w prawo/lewo w układzie widoku
            tr = m * tr
        return tr

    def projectionMatrix(self, region, viewport):
        tr = super().projectionMatrix(region, viewport)
        if self._oko:                                                     # frustum asymetryczne: zerowy rozrzut na odległości środka
            r0 = tr.row(0)
            r0.setZ(r0.z() - r0.x() * self._oko / self.opts['distance'])
            tr.setRow(0, r0)
        return tr

    def _paint_stereo(self):
        """Dwa przebiegi (oko lewe i prawe) do połówek bieżącego viewportu, z nożyczkami, żeby czyszczenie nie zmazywało drugiego oka.
        Napisy GLTextItem są rysowane w pikselach całego widżetu, więc w trybie stereo je pomijamy."""
        from OpenGL import GL
        x, y, W, H = [int(v) for v in GL.glGetIntegerv(GL.GL_VIEWPORT)]
        pol = W // 2
        reg = self.getViewport()                                          # cała geometria widżetu: oko ma pełne proporcje i jest zgniatane
        sep = self.stereo_odstep * self.opts['distance']
        znak = -1.0 if GLView.stereo_zamiana else 1.0
        ukryte = [it for it in self.items if isinstance(it, gl.GLTextItem) and it.visible()]
        for it in ukryte:
            it.setVisible(False)
        try:
            GL.glEnable(GL.GL_SCISSOR_TEST)
            for oko, (x0, szer) in ((-0.5 * znak * sep, (x, pol)), (0.5 * znak * sep, (x + pol, W - pol))):
                GL.glViewport(x0, y, szer, H); GL.glScissor(x0, y, szer, H)
                self._oko = oko
                self.paint(region=reg, viewport=reg)
        finally:
            self._oko = 0.0
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glViewport(x, y, W, H)
            for it in ukryte:
                it.setVisible(True)

    def paintGL(self):
        from OpenGL import GL
        from OpenGL.error import GLError
        try:
            for _ in range(16):
                if GL.glGetError() == GL.GL_NO_ERROR:
                    break
            if GLView.stereo:
                self._paint_stereo()
            else:
                super().paintGL()
        except GLError:
            self._pominiete = getattr(self, '_pominiete', 0) + 1
            if self._pominiete < 50:
                QTimer.singleShot(30, self.update)


