"""Okienko R11: miary separacji w czasie uczenia (liczone w tle, zapisywane obok stanów 192D)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt

from laser_app.compute import history as hist, projections
from laser_app.ui.projection_windows.workers import MetricsWorker


class MetricsMixin:
    """Okienko R11: miary separacji w czasie uczenia (liczone w tle, zapisywane obok stanów 192D).
    Część ProjectionWindows (ui/projections_view.py); korzysta z jej atrybutów przez self."""

    # ------------------------------------------------------------------ R11: miary separacji w czasie
    def _buduj_miary(self):
        w = pg.GraphicsLayoutWidget(); w.setBackground('#0b1014')
        self.p11 = {}
        for n, (klucz, tyt) in enumerate((('LDA_fisher', 'Fisher, LDA'), ('192D_fisher', 'Fisher, pełne 192D'),
                                          ('PCA3_hit', 'kNN, PCA3 chmury'), ('192D_sil', 'Sylwetka, 192D'))):
            p = w.addPlot(row=n // 2, col=n % 2, title=tyt)
            p.showGrid(x=True, y=True, alpha=0.2); p.setLabel('bottom', 'checkpoint'); p.setMouseEnabled(x=True, y=False)
            p.addLegend(offset=(5, 5))
            zero = p.plot(pen=pg.mkPen((170, 170, 170), width=1.5, style=Qt.PenStyle.DashLine), name='95 % położeń tła (zero przestrzenne)')
            nar = p.plot(pen=pg.mkPen((90, 255, 90), width=2), name='narzędzie')
            kur = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('w', width=1.5))
            p.addItem(kur); kur.sigPositionChanged.connect(self._kursor); p.kursor = kur
            self.p11[klucz] = (p, nar, zero)
        self._miary_czas = None
        self._miary_klucz = None
        self._watek_m = None
        return self._z_napisem(w, 'R11')

    def _rysuj_miary(self):
        d, h = self.dane, self._h
        klucz = str(h.katalog)
        if self._miary_klucz != klucz:
            self._miary_klucz = klucz
            self._miary_czas = None
            plik = projections.katalog(h) / 'miary_czas_v1.npz'
            if plik.exists():
                try:
                    z = np.load(plik, allow_pickle=False)
                    self._miary_czas = {k: z[k] for k in z.files}
                except Exception:                                        # noqa: BLE001
                    self._miary_czas = None
        if self._miary_czas is None:
            if self._watek_m is None:
                rdz = np.asarray(d.narz, bool); mies = np.asarray(getattr(d, 'mieszane', np.zeros(hist.TOK, bool)), bool)
                u = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
                wyk = rdz | mies | ((u > 0) if np.isfinite(u).all() else False)
                self._watek_m = MetricsWorker(d, rdz, mies, wyk, projections.katalog(h) / 'miary_czas_v1.npz', klucz)
                self._watek_m.gotowe.connect(self._miary_gotowe)
                self._watek_m.postep.connect(lambda a, b: self._lb['R11'].setText(f'Liczę miary w czasie na GPU: {a}/{b} checkpointów…'))
                self._watek_m.start()
            return
        m = self._miary_czas
        x = m['idx'].astype(float)
        for kl, (p, nar, zero) in self.p11.items():
            v, q = kl.split('_')
            vi, qi = projections.MIARY_CZAS_WIDOKI.index(v), projections.MIARY_CZAS_NAZWY.index(q)
            nar.setData(x, m['narz'][:, vi, qi]); zero.setData(x, m['zero95'][:, vi, qi])
            self._blok = True; p.kursor.setValue(self.t); self._blok = False
        self._lb['R11'].setText(
            f'Narzędzie: {int(m["n_narz"])} patchy, zero przestrzenne z {int(m["n_polozen"])} losowych położeń tej samej maski w tle (95. percentyl; '
            f'LDA dopasowana od nowa na każdym checkpoincie i położeniu), kNN z k = {int(m["k"])}. Narzędzie nad linią przerywaną = wyróżnia się. '
            f'Kursor = bieżący checkpoint. Wynik zapisany w katalogu rzutów.')

    def _miary_gotowe(self, klucz, wyn):
        self._watek_m = None
        if wyn is not None and klucz == self._miary_klucz:
            self._miary_czas = wyn
            self._rysuj_miary()
