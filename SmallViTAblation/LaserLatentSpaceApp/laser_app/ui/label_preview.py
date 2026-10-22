"""Podgląd etykiety narzędzia na zdjęciu: siatka 30 × 20 łat (patchy) na zdjęciu, łaty narzędzia w bieżącym trybie etykiety
wypełnione na zielono, mieszane (dotknięte obrysem, ale nie narzędzie w tym trybie) obwiedzione na żółto, jasność wypełnienia
= udział pikseli obrysu w łacie. Najechanie myszą pokazuje numer łaty i jej udział, klik wybiera tę cząsteczkę (sygnał `klik`).
Porównanie trybów („dotknięte”, „≥ 50 % jak dawniej”, własny próg) daje `opis_trybow`."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

KOLUMNY, WIERSZE = 30, 20


class LabelPreview(QWidget):
    klik = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._obraz = None
        self.udzial = self.narz = self.mies = None
        self.wybrana = None
        self.setMouseTracking(True)
        sp = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred); sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setMinimumHeight(120)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return int(w * WIERSZE / KOLUMNY)

    def sizeHint(self):
        return QSize(360, 240)

    def ustaw(self, rgb, udzial, narz, mies, wybrana=None):
        """rgb (H, W, 3) uint8; udzial (600,) udział pikseli obrysu; narz, mies (600,) bool z bieżącego trybu etykiety."""
        if rgb is not None:
            a = np.ascontiguousarray(np.asarray(rgb, np.uint8))
            self._obraz = QImage(a.data, a.shape[1], a.shape[0], 3 * a.shape[1], QImage.Format.Format_RGB888).copy()
        self.udzial = None if udzial is None else np.asarray(udzial, np.float32)
        self.narz = None if narz is None else np.asarray(narz, bool)
        self.mies = None if mies is None else np.asarray(mies, bool)
        self.wybrana = wybrana
        self.update()

    def _kadr(self):
        w, h = self.width(), self.height()
        s = min(w / KOLUMNY, h / WIERSZE)
        return QRectF((w - s * KOLUMNY) / 2, (h - s * WIERSZE) / 2, s * KOLUMNY, s * WIERSZE), s

    def _lata(self, pos):
        r, s = self._kadr()
        c, w = int((pos.x() - r.x()) // s), int((pos.y() - r.y()) // s)
        return w * KOLUMNY + c if 0 <= c < KOLUMNY and 0 <= w < WIERSZE else None

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#0b1014'))
        r, s = self._kadr()
        if self._obraz is not None:
            p.drawImage(r, self._obraz)
        if self.narz is not None:
            u = self.udzial if self.udzial is not None else self.narz.astype(np.float32)
            for t in np.flatnonzero(self.narz | (self.mies if self.mies is not None else False)):
                x, y = r.x() + (t % KOLUMNY) * s, r.y() + (t // KOLUMNY) * s
                if self.narz[t]:
                    p.fillRect(QRectF(x, y, s, s), QColor(60, 255, 110, int(70 + 130 * min(float(u[t]), 1.0))))
                else:
                    p.setPen(QPen(QColor(255, 210, 60, 230), 1.5)); p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(QRectF(x + 1, y + 1, s - 2, s - 2))
        p.setPen(QPen(QColor(255, 255, 255, 28), 1))
        for c in range(KOLUMNY + 1):
            p.drawLine(int(r.x() + c * s), int(r.y()), int(r.x() + c * s), int(r.bottom()))
        for w in range(WIERSZE + 1):
            p.drawLine(int(r.x()), int(r.y() + w * s), int(r.right()), int(r.y() + w * s))
        if self.wybrana is not None and 0 <= self.wybrana < KOLUMNY * WIERSZE:
            t = int(self.wybrana)
            p.setPen(QPen(QColor(255, 255, 255), 2)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(r.x() + (t % KOLUMNY) * s, r.y() + (t // KOLUMNY) * s, s, s))
        p.end()

    def mouseMoveEvent(self, e):
        t = self._lata(e.position())
        if t is None:
            QToolTip.hideText(); return
        u = '' if self.udzial is None else f', obrys narzędzia {float(self.udzial[t]):.0%} pikseli'
        rola = 'narzędzie' if self.narz is not None and self.narz[t] else ('mieszana' if self.mies is not None and self.mies[t] else 'tło')
        QToolTip.showText(e.globalPosition().toPoint(), f'łata {t} (wiersz {t // KOLUMNY}, kolumna {t % KOLUMNY}){u}: {rola}. Klik wybiera cząsteczkę.', self)

    def mousePressEvent(self, e):
        t = self._lata(e.position())
        if t is not None and e.button() == Qt.MouseButton.LeftButton:
            self.klik.emit(int(t))


def opis_trybow(udzial, prog, srodek=None, ramka=None):
    """Ile łat narzędzia daje każdy tryb etykiety (do porównania w panelu)."""
    from laser_app.compute import tool_labels
    out = []
    for k, _ in tool_labels.TRYBY:
        try:
            n, _m = tool_labels.maski(udzial, prog, k, srodek=srodek, ramka=ramka)
            out.append(f'{tool_labels.KROTKO.get(k, k).format(prog=prog)} {int(n.sum())}')
        except ValueError:
            pass
    return ' · '.join(out)
