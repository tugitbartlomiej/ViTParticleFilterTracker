"""Widżety zakładki: widok 2D okienek 1–3 (View2D), schemat sieci (NetworkDiagram), suwak osi czasu z punktami A–B (ABSlider)."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal, QEvent, Qt
from PyQt6.QtWidgets import QSlider, QWidget

from laser_app.lasers_tab.shared import TLO
from laser_app.ui.gl_view import GLView


class View2D(GLView):
    """Widok z góry (rzut płaski): lewy przycisk przesuwa, kółko przybliża; brak obrotu."""

    obrot = False                                # widok z góry: LPM i ŚPM przesuwają, PPM + przeciągnięcie zbliża/oddala

    def __init__(self):
        super().__init__()
        self.setBackgroundColor(TLO)
        self.opts['fov'] = 30
        self.setCameraPosition(elevation=90, azimuth=-90, distance=60)

    def kadruj(self, cx, cy, szer, wys=None):
        """Środek (cx, cy) w komórkach (y obrazu w dół); w kadrze mieści się szer × wys komórek (proporcje widżetu)."""
        from PyQt6.QtGui import QVector3D
        self._kadr = (cx, cy, szer, wys)
        if wys is not None:
            szer = max(float(szer), float(wys) * self.width() / max(self.height(), 1))
        self.opts['center'] = QVector3D(float(cx), float(-cy), 0.0)
        self.opts['distance'] = float(szer) / 2 / np.tan(np.radians(self.opts['fov'] / 2))   # pole widzenia poziome
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, '_kadr', None) is not None:
            self.kadruj(*self._kadr)

    def px_na_komorke(self):
        return self.width() / (2 * self.opts['distance'] * np.tan(np.radians(self.opts['fov'] / 2)))


class NetworkDiagram(QWidget):
    """Schemat MałegoViT jak w diagram_maly_vit.pdf: obraz → łodyga + kodowanie pozycji → 6 bloków (uwaga, MLP) → LayerNorm → głowa.
    Podświetla bieżący podkrok s (0..12: stan 0 = wejście bloków, podkrok k = uwaga (k parzyste) albo MLP bloku k // 2), klik wybiera podkrok."""
    klik = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.s = 0.0
        self.setMinimumHeight(118); self.setMaximumHeight(150)
        self._pola = []
        self._miniatury = []                       # QImage: obraz, 4 etapy łodygi, kodowanie pozycji, tokeny + PE
        self._opisy = []

    def ustaw_lodyge(self, rgb, d):
        """Miniatury etapów wejścia dla bieżącego checkpointu (mapy cech łodygi jako 3 główne składowe kanałów)."""
        from PyQt6.QtGui import QImage
        obrazy = [rgb] + list(d.get('lodyga', []))
        self._miniatury = []
        for a in obrazy:
            a = np.ascontiguousarray(a)
            q = QImage(a.data, a.shape[1], a.shape[0], 3 * a.shape[1], QImage.Format.Format_RGB888).copy()
            self._miniatury.append(q)
        ks = d.get('lodyga_ksztalty', [])
        self._opisy = ['obraz 480×320×3'] + [f'konw. {i + 1}: {w_}×{h_}×{c_}' for i, (c_, h_, w_) in enumerate(ks[:4])] + \
                      ['kodowanie pozycji sin-cos 30×20×192', 'tokeny = konw. 4 + PE (600 × 192)']
        self.update()

    def ustaw(self, s):
        self.s = float(s); self.update()

    def paintEvent(self, ev):
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QColor, QFont, QPainter, QPen
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#0b1014'))
        W_, H_ = self.width(), self.height()
        n_kol = 6 + 6 + 2 + 0.4                               # obraz, 4 konwolucje, +PE, 6 bloków, LN, głowa
        szer = (W_ - 20) / (n_kol + 0.6)
        y0, h = 22, H_ - 34
        self._pola = []
        f = QFont(self.font()); f.setPointSizeF(max(7.0, min(10.0, szer / 9))); p.setFont(f)
        s = self.s; k = int(np.clip(np.floor(s), 0, 11))

        def pudelko(x, y, w, hh, tekst, kolor, aktywne=0.0, gotowe=False):
            r = QRectF(x, y, w, hh)
            c = QColor(kolor)
            if gotowe:
                c = c.darker(135)
            p.setBrush(c); p.setPen(QPen(QColor('#ffb000') if aktywne > 0 else QColor('#2a333a'), 3 if aktywne > 0 else 1))
            p.drawRoundedRect(r, 6, 6)
            if aktywne > 0:                                    # postęp sygnału w podkroku
                p.fillRect(QRectF(x + 2, y + hh - 6, (w - 4) * min(aktywne, 1.0), 4), QColor('#ffb000'))
            p.setPen(QColor('#101418')); p.drawText(r, int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), tekst)
            return r

        def strzalka(x0, x1, y):
            p.setPen(QPen(QColor('#cfd3d8'), 1.5)); p.drawLine(int(x0), int(y), int(x1), int(y))
            p.drawLine(int(x1), int(y), int(x1 - 5), int(y - 4)); p.drawLine(int(x1), int(y), int(x1 - 5), int(y + 4))

        x = 10; ym = y0 + h / 2
        # wejście: obraz → 4 konwolucje łodygi (krok 2) → + kodowanie pozycji; miniatury = mapy cech bieżącego checkpointu
        nazwy = ['obraz', 'konw. 1', 'konw. 2', 'konw. 3', 'konw. 4', '+ PE']
        p.setPen(QColor('#f5c6c6')); p.drawText(QRectF(x + szer, y0 - 18, 4 * szer * 0.95, 16), int(Qt.AlignmentFlag.AlignCenter),
                                                 'łodyga CNN (w każdym przejściu od pikseli)')
        for i, nz in enumerate(nazwy):
            r = QRectF(x, y0 + 4, szer * 0.86, h - 8)
            aktyw = (s == 0 and i == 5)
            p.setBrush(QColor('#1a2228')); p.setPen(QPen(QColor('#ffb000') if aktyw else QColor('#55606a'), 3 if aktyw else 1))
            p.drawRoundedRect(r, 5, 5)
            p.setPen(QColor('#e8e8e8')); p.drawText(QRectF(x, y0 + 4, szer * 0.86, 14), int(Qt.AlignmentFlag.AlignCenter), nz)
            if i < len(self._miniatury):
                if i == 5 and len(self._miniatury) > 6:
                    im = self._miniatury[6]
                else:
                    im = self._miniatury[i]
                obsz = QRectF(x + 3, y0 + 20, szer * 0.86 - 6, h - 30)
                sk = min(obsz.width() / im.width(), obsz.height() / im.height())
                ww, hh = im.width() * sk, im.height() * sk
                p.drawImage(QRectF(obsz.x() + (obsz.width() - ww) / 2, obsz.y() + (obsz.height() - hh) / 2, ww, hh), im)
            self._pola.append((r, 0.0, self._opisy[i] if i < len(self._opisy) else nz))
            x += szer
            if i < 5:
                strzalka(x - szer * 0.14, x - 1, ym)
        for b in range(6):
            strzalka(x - szer * 0.1, x, ym)
            p.setPen(QPen(QColor('#cfd3d8'), 1)); p.setBrush(QColor('#1a2228'))
            p.drawRoundedRect(QRectF(x, y0, szer * 0.9, h), 6, 6)
            p.setPen(QColor('#e8e8e8')); p.drawText(QRectF(x, y0 - 18, szer * 0.9, 16), int(Qt.AlignmentFlag.AlignCenter), f'blok {b + 1}')
            ku, km = 2 * b, 2 * b + 1
            am = (s - km) if k == km and s < 12 else 0
            au = (s - ku) if k == ku and s < 12 else 0
            # kolejność w czasie z góry na dół: najpierw uwaga, potem MLP; napis skrócony, gdy nie mieści się w jednej linii
            wn = szer * 0.9 - 8
            t_m = 'MLP\n768 laserów' if p.fontMetrics().horizontalAdvance('768 laserów') < wn - 4 else 'MLP\n768'
            ru = pudelko(x + 4, y0 + 4, wn, h / 2 - 6, 'uwaga\nsiły f_ij', '#c8f0d0', aktywne=max(au, 1e-3) if k == ku and s < 12 else 0,
                         gotowe=s >= ku + 1)
            rm = pudelko(x + 4, y0 + h / 2 + 2, wn, h / 2 - 6, t_m, '#bfe0f5', aktywne=max(am, 1e-3) if k == km and s < 12 else 0,
                         gotowe=s >= km + 1)
            self._pola += [(ru, float(ku), f'blok {b + 1}: uwaga'), (rm, float(km), f'blok {b + 1}: MLP')]
            x += szer
        strzalka(x - szer * 0.1, x, ym)
        r = pudelko(x, y0 + h * 0.2, szer * 0.8, h * 0.6, 'Layer-\nNorm', '#fde3c3', aktywne=1.0 if s >= 12 else 0)
        self._pola.append((r, 12.0, 'LayerNorm i głowa')); x += szer
        strzalka(x - szer * 0.2, x, ym)
        pudelko(x, y0 + h * 0.1, szer * 0.9, h * 0.8, 'głowa\nmapa +\nramka', '#fde3c3', aktywne=1.0 if s >= 12 else 0)
        p.end()

    def mousePressEvent(self, ev):
        for r, s, _ in self._pola:
            if r.contains(ev.position()):
                self.klik.emit(s); return

    def event(self, ev):
        from PyQt6.QtCore import QEvent, QPointF
        from PyQt6.QtWidgets import QToolTip
        if ev.type() == QEvent.Type.ToolTip:
            for r, _, opis in self._pola:
                if r.contains(QPointF(ev.pos())):
                    QToolTip.showText(ev.globalPos(), opis, self); return True
        return super().event(ev)


class ABSlider(QSlider):
    """Suwak osi czasu z zaznaczonym odcinkiem A–B (wartości suwaka; None = brak punktu)."""
    def __init__(self):
        super().__init__(Qt.Orientation.Horizontal)
        self.ab = [None, None]

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self.maximum() <= self.minimum() or self.ab == [None, None]:
            return
        from PyQt6.QtGui import QColor, QPainter
        from PyQt6.QtWidgets import QStyle, QStyleOptionSlider
        o = QStyleOptionSlider(); self.initStyleOption(o)
        uchwyt = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, o, QStyle.SubControl.SC_SliderHandle, self)
        x0, szer = uchwyt.width() / 2, self.width() - uchwyt.width()
        xx = lambda v: x0 + szer * (v - self.minimum()) / (self.maximum() - self.minimum())
        a = self.minimum() if self.ab[0] is None else self.ab[0]
        b = self.maximum() if self.ab[1] is None else self.ab[1]
        q = QPainter(self); h = self.height()
        q.fillRect(int(xx(min(a, b))), h // 2 - 5, max(1, int(xx(max(a, b)) - xx(min(a, b)))), 10, QColor(80, 200, 255, 70))
        for v, kol in ((self.ab[0], QColor(80, 200, 255)), (self.ab[1], QColor(255, 120, 90))):
            if v is not None:
                q.fillRect(int(xx(v)) - 1, 2, 3, h - 4, kol)
        q.end()
