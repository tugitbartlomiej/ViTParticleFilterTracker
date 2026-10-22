"""Okienko R8: panel bez rzutu (wierność widoków w czasie, separacja, cos, pary wiązek, widmo J)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from laser_app.compute import history as hist, projections
from laser_app.ui.projection_windows.shared import KOL_BAZ, KOL_NARZ, KOL_TLO, proc


class PanelMixin:
    """Okienko R8: panel bez rzutu (wierność widoków w czasie, separacja, cos, pary wiązek, widmo J).
    Część ProjectionWindows (ui/projections_view.py); korzysta z jej atrybutów przez self."""

    def _buduj_panel(self):
        w = pg.GraphicsLayoutWidget(); w.setBackground('#0b1014')
        self.p_w = w.addPlot(row=0, col=0, colspan=2, title='Wierność widoków w czasie: jaka część jest w rzucie 3D')
        self.p_sep = w.addPlot(row=1, col=0, title='Separacja narzędzie–tło w 192D [R] i losowe etykiety')
        self.p_cos = w.addPlot(row=1, col=1, title='cos(x_i, x_j) w 192D w bieżącym checkpoincie')
        self.p_wz = w.addPlot(row=2, col=0, title='Pary wiązek wybranej cząsteczki: cos w 192D i w P3')
        self.p_J = w.addPlot(row=2, col=1, title='Widmo J wybranej cząsteczki (z R5) i koło Ginibre’a')
        for p in (self.p_w, self.p_sep, self.p_cos, self.p_wz, self.p_J):
            p.showGrid(x=True, y=True, alpha=0.2)
        for p in (self.p_w, self.p_sep):
            kur = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('w', width=1.5))
            p.addItem(kur)
            kur.sigPositionChanged.connect(self._kursor)
            p.kursor = kur
            p.setMouseEnabled(x=True, y=False)
        self.p_w.addLegend(offset=(5, 5)); self.p_cos.addLegend(offset=(5, 5)); self.p_wz.addLegend(offset=(-5, 5))
        self.tx_wz = pg.TextItem('', color='#ffd166', anchor=(0, 0)); self.p_wz.addItem(self.tx_wz)
        self.tx_J = pg.TextItem('', color='#ffd166', anchor=(0, 0)); self.p_J.addItem(self.tx_J)
        self.p_J.setAspectLocked(True)
        return w

    # ------------------------------------------------------------------ R8: bez rzutu
    def _panel_wiernosc(self):
        if self.dane is None or not hasattr(self, 'p_w'):
            return
        d, h = self.dane, self._h
        m = self.cb_P.currentIndex()
        Y = (d.WX, d.WF, d.WW, d.WS, d.WD)[m]
        p = self.p_w
        for it in list(p.listDataItems()):
            p.removeItem(it)
        if p.legend is not None:
            p.legend.clear()
        x = np.arange(d.K)
        for b, nazwa in enumerate(projections.BAZY):
            styl = Qt.PenStyle.DashLine if nazwa == 'A_t' else Qt.PenStyle.SolidLine
            p.plot(x, Y[:, b], pen=pg.mkPen(KOL_BAZ[nazwa], width=1.4, style=styl), name=nazwa, connect='finite')
        zero = projections.LOS_DLUGOSC if m == 3 else projections.LOS_ENERGIA
        p.plot(x, np.full(d.K, zero), pen=pg.mkPen('#888888', width=1, style=Qt.PenStyle.DotLine), name='losowe 3D')
        if m == 0:
            p.plot(x, d.own3, pen=pg.mkPen('#ffd166', width=1, style=Qt.PenStyle.DotLine), name='najlepsze 3D (PCA chmury)')
        p.setYRange(0, 1); p.setXRange(0, d.K - 1)
        q = self.p_sep
        for it in list(q.listDataItems()):
            q.removeItem(it)
        q.plot(x, d.sep, pen=pg.mkPen('#ffb000', width=1.5), connect='finite')
        q.plot(x, d.sep0[:, 0], pen=pg.mkPen('#888888', width=1), connect='finite')
        q.plot(x, d.sep0[:, 1], pen=pg.mkPen('#888888', width=1, style=Qt.PenStyle.DashLine), connect='finite')
        q.setXRange(0, d.K - 1)
        EP = hist.NA_EPOKE
        kr = np.asarray(d.kroki)
        tk = [k for k in (0, 1000, EP * 5, EP * 10, EP * 20, EP * 40, EP * 60, EP * 100, EP * 150, EP * 200)   # bez 300 i e1: nachodzą
              if kr[0] <= k <= kr[-1]]
        nazwy = lambda k: {0: '0', 300: '300', 1000: '1k'}.get(k, f'e{int(round(k / EP))}')
        ticks = [[(float(np.interp(k, kr, np.arange(d.K))), nazwy(k)) for k in tk]]
        for p_ in (p, q):
            p_.getAxis('bottom').setTicks(ticks)
        self._i_panel = None

    def _rysuj_panel(self):
        d = self.dane
        _, _, _, ir = self._indeksy()
        self._blok = True
        for p in (self.p_w, self.p_sep):
            p.kursor.setValue(self.t)
        self._blok = False
        wybrane = self._wybrane()
        klucz = (ir, wybrane[0] if wybrane else None)
        if klucz == self._i_panel:
            return
        self._i_panel = klucz
        X = np.asarray(d.X[ir], np.float64)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
        Cm = Xn @ Xn.T
        narz = np.asarray(d.narz, bool)
        iu = np.triu_indices(hist.TOK, 1)
        grupy = (('narzędzie–narzędzie', narz[iu[0]] & narz[iu[1]], KOL_NARZ),
                 ('narzędzie–tło', narz[iu[0]] ^ narz[iu[1]], (1.0, 0.82, 0.4)),
                 ('tło–tło', ~narz[iu[0]] & ~narz[iu[1]], KOL_TLO))
        p = self.p_cos
        for it in list(p.listDataItems()):
            p.removeItem(it)
        if p.legend is not None:
            p.legend.clear()
        cv = Cm[iu]
        for nazwa, m, kol in grupy:
            if m.any():
                y, e = np.histogram(cv[m], 60, (-1, 1), density=True)
                p.plot(e, y, stepMode='center', pen=pg.mkPen(tuple(int(255 * c) for c in kol), width=1.4), name=nazwa)
        if wybrane:
            u = self._wklady(ir, wybrane[0])
            top = np.argsort(-np.linalg.norm(u, axis=1))[:100]
            q = self.p_wz
            for it in list(q.listDataItems()):
                q.removeItem(it)
            if q.legend is not None:
                q.legend.clear()
            iu2 = np.triu_indices(len(top), 1)
            ymax = 0.0
            for nazwa, v, kol in (('192D', u[top], '#ffd166'), ('rzut P3', u[top] @ d.baza_P3, '#e8e8e8')):
                vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)
                c = (vn @ vn.T)[iu2]
                y, e = np.histogram(c, 50, (-1, 1), density=True)
                ymax = max(ymax, float(y.max()))
                q.plot(e, y, stepMode='center', pen=pg.mkPen(kol, width=1.4), name=nazwa)
            self.tx_wz.setText(f'100 najsilniejszych wiązek cząsteczki {wybrane[0]}:\npary ≥ 150° w 192D '
                               f'{proc(projections.pary_przeciwne(u[top]))}, w P3 {proc(projections.pary_przeciwne(u[top], d.baza_P3))} '
                               f'(losowe 3D {proc(projections.LOS_PRZECIWNE_3D)})')
            self.tx_wz.setPos(-1, ymax * 1.15)
        if self.ch_C.isChecked():
            self._zlec_C()
        self._panel_widmo()

    def _panel_widmo(self):
        p = self.p_J
        for it in list(p.listDataItems()):
            p.removeItem(it)
        if self._C is None:
            self.tx_J.setText('widmo pojawi się po policzeniu ramy C (R5)')
            return
        klucz, w = self._C
        lam = w['widmo']['lam']
        p.plot(lam.real, lam.imag, pen=None, symbol='o', symbolSize=4, symbolBrush='#c264ff', symbolPen=None)
        g = w['ginibre']
        th = np.linspace(0, 2 * np.pi, 200)
        p.plot(g['srodek'] + g['promien'] * np.cos(th), g['promien'] * np.sin(th), pen=pg.mkPen('#888888', width=1, style=Qt.PenStyle.DashLine))
        p.plot([0, 0], [-g['promien'], g['promien']], pen=pg.mkPen('#555555', width=1))
        sel = np.asarray(w['wybrane'])
        if len(sel):
            p.plot(np.concatenate([sel.real, sel.real]), np.concatenate([sel.imag, -sel.imag]), pen=None, symbol='x', symbolSize=11,
                   symbolPen=pg.mkPen('#ffd166', width=2))
        self.tx_J.setText(f'cząsteczka {klucz[2]}, checkpoint {klucz[1] + 1}:\n{proc(w["widmo"]["zesp"])} zespolonych '
                          f'(losowa ≈ {proc(projections.GINIBRE_ZESPOLONE)})\n× = rama R5')
        self.tx_J.setPos(float(lam.real.min()), float(np.abs(lam.imag).max()))
