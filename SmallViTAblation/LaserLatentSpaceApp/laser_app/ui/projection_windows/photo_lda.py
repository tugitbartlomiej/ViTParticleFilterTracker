"""Okienka R0 (zdjęcie z patchami, ramki GT i modelu) oraz R9–R10 (separacja narzędzia od tła, LDA na żywo)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt

from laser_app.compute import history as hist, projections
from laser_app.ui.projection_windows.shared import pl, TLO


class PhotoLDAMixin:
    """Okienka R0 (zdjęcie z patchami, ramki GT i modelu) oraz R9–R10 (separacja narzędzia od tła, LDA na żywo).
    Część ProjectionWindows (ui/projections_view.py); korzysta z jej atrybutów przez self."""

    # ------------------------------------------------------------------ R0: zdjęcie z ramkami i patchami
    def _buduj_zdjecie(self):
        w = pg.GraphicsLayoutWidget(); w.setBackground(TLO)
        self.vb0 = w.addViewBox(lockAspect=True, enableMenu=False)
        self.vb0.invertY(True)
        self.im0 = pg.ImageItem(axisOrder='row-major'); self.vb0.addItem(self.im0)
        self._r0_klucz = None
        self.kr0 = {}
        for n, (kol, szer, styl) in dict(siatka=((255, 255, 255, 40), 1, None), mies=((255, 209, 102, 255), 2, None),
                                           narz=((90, 255, 90, 255), 2, None), grupa=((255, 70, 70, 255), 2, Qt.PenStyle.DashLine),
                                           wyb=((255, 255, 255, 255), 3, None), gt=((255, 255, 255, 230), 2, Qt.PenStyle.DotLine),
                                           model=((0, 230, 255, 255), 2, None)).items():
            c = pg.PlotCurveItem(pen=pg.mkPen(kol, width=szer, style=styl) if styl is not None else pg.mkPen(kol, width=szer),
                                 connect='pairs' if n in ('siatka', 'mies', 'narz', 'grupa', 'wyb') else 'all')
            self.vb0.addItem(c); self.kr0[n] = c
        self.tx0 = pg.TextItem('', color=(255, 255, 255), anchor=(0, 0), fill=(0, 0, 0, 160))
        self.vb0.addItem(self.tx0); self.tx0.setPos(4, 4)
        return self._z_napisem(w, 'R0')

    @staticmethod
    def _krawedzie(idx):
        """(n · 8, 2): odcinki obrysów patchy 16 × 16 (pary punktów) dla tokenów idx."""
        out = []
        for t in idx:
            r, c = divmod(int(t), hist.W)
            x0, y0 = c * 16, r * 16; x1, y1 = x0 + 16, y0 + 16
            out += [(x0, y0), (x1, y0), (x1, y0), (x1, y1), (x1, y1), (x0, y1), (x0, y1), (x0, y0)]
        return np.array(out, float).reshape(-1, 2)

    def _rysuj_zdjecie(self):
        h = self._h
        K = h.K
        i = int(np.clip(round(self.t), 0, K - 1))
        if self._r0_klucz != id(h):
            self._r0_klucz = id(h)
            self.im0.setImage(np.asarray(h.rgb), axisOrder='row-major')
            self.vb0.setRange(QRectF(0, 0, hist.W * 16, hist.H * 16), padding=0.01)
            self.kr0['siatka'].setData(self._krawedzie(range(hist.TOK)).T[0], self._krawedzie(range(hist.TOK)).T[1])
        narz = np.asarray(h.narz, bool); mies = np.asarray(getattr(h, 'mieszane', np.zeros(hist.TOK, bool)), bool)
        zbior = lambda n, idx: self.kr0[n].setData(*self._krawedzie(idx).T) if len(idx) else self.kr0[n].setData([], [])
        zbior('narz', np.flatnonzero(narz)); zbior('mies', np.flatnonzero(mies))
        self.kr0['siatka'].setVisible(self.ch_R0_siatka.isChecked())
        wyb = self._wybrane()
        zbior('wyb', list(wyb)[:1] if wyb else [])
        grupa_txt = ''
        G = None
        if self.ch_R0_grupa.isChecked() and (('grupa', i) in self._pam or not self._z.bt_play.isChecked()):
            G = self._grupa(i)
        zbior('grupa', np.flatnonzero(G) if G is not None else [])
        if G is not None:
            u = np.asarray(getattr(h, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
            if np.isfinite(u).all():
                rdz = u >= 0.25
                dice = 2 * (G & rdz).sum() / max(G.sum() + rdz.sum(), 1)
                prec = (u[G] > 0).mean() if G.any() else float('nan')
                grupa_txt = f' Grupa bez etykiet (czerwona): {int(G.sum())} patchy, Dice z obrysem {pl(float(dice))}, precyzja {pl(float(prec))}.'
            else:
                grupa_txt = f' Grupa bez etykiet (czerwona): {int(G.sum())} patchy.'
        gt = h.gt
        if gt is not None:
            self.kr0['gt'].setData([gt[0], gt[2], gt[2], gt[0], gt[0]], [gt[1], gt[1], gt[3], gt[3], gt[1]])
        rb = np.asarray(h.ramka[i], float)
        self.kr0['model'].setData([rb[0], rb[2], rb[2], rb[0], rb[0]], [rb[1], rb[1], rb[3], rb[3], rb[1]])
        iou, pew = float(h.iou[i]), float(h.pew[i])
        self.tx0.setText(f'krok {int(h.kroki[i])}  IoU {pl(iou, 3)}  pewność {pl(pew)}')
        self._lb['R0'].setText(f'Checkpoint {i + 1}/{K} (krok {int(h.kroki[i])}). Biała kropkowana: ramka GT, cyjan: ramka modelu, IoU {pl(iou, 3)}, '
                               f'pewność {pl(pew)}. Zielone patche: narzędzie z obrysu ({int(narz.sum())}), żółte: mieszane ({int(mies.sum())}).'
                               f'{grupa_txt}  [{getattr(h, "etykieta", "ramka GT")}]')

    # ------------------------------------------------------------------ R10: LDA na żywo
    def _buduj_lda(self):
        w = pg.GraphicsLayoutWidget(); w.setBackground('#0b1014')
        self.p10, self.s10 = [], []
        for c in range(3):
            p = w.addPlot(row=0, col=c)
            p.showGrid(x=True, y=True, alpha=0.2); p.setLabel('bottom', 'oś 1 (LDA)'); p.setLabel('left', 'oś 2')
            tlo = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(150, 160, 175, 150))
            nar = pg.ScatterPlotItem(size=9, pen=pg.mkPen('k', width=0.5), brush=pg.mkBrush(90, 255, 90))
            p.addItem(tlo); p.addItem(nar)
            self.p10.append(p); self.s10.append((tlo, nar))
        return self._z_napisem(w, 'R10')

    def _wyklucz(self, rdz, mies):
        wyk = rdz | mies
        u = np.asarray(getattr(self.dane, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
        if np.isfinite(u).all():
            wyk = wyk | (u > 0)                                   # żadnego piksela narzędzia w położeniach tła
        return wyk

    def _rysuj_lda(self):
        import torch
        d = self.dane
        _, _, _, i = self._indeksy()
        rdz, mies = self._etykieta(i)
        keep = ~mies
        X = np.asarray(d.X[i], np.float32)
        Xk, m = X[keep], rdz[keep]
        if m.sum() < 2:
            self._lb['R10'].setText('Za mało tokenów narzędzia w tej etykiecie.')
            return
        kk = projections.k_sasiadow(m.sum())
        lam, osie = float(self.sp_L_lam.value()), self.cb_L_osie.currentIndex()
        if self.cb_L10.currentIndex() == 0:
            B, _ = self._baza_L()
            zrodlo = 'oś jak w R9'
        elif self.cb_L10.currentIndex() == 2:
            dev = projections._dev()
            Xt = torch.as_tensor(Xk, device=dev)
            Xc = Xt - Xt.mean(0, keepdim=True)
            B = torch.linalg.eigh(Xc.T @ Xc)[1][:, -3:].flip(1).cpu().numpy()
            zrodlo = 'PCA3 chmury (bez etykiet)'
        else:
            dev = projections._dev()
            B = projections.baza_L_t(torch.as_tensor(Xk, device=dev), torch.as_tensor(m, device=dev)[None], lam, osie)[0].cpu().numpy()
            zrodlo = 'oś dopasowana na tym checkpoincie'
        P = Xk @ np.asarray(B, np.float32)
        mi, _ = projections.separacja_gpu(P, m, 0, k=kk)
        fp = lambda v: f'{v:.3f}'.replace('.', ',')

        def rysuj(n, P_, m_, tytul):
            tlo, nar = self.s10[n]
            tlo.setData(P_[~m_, 0], P_[~m_, 1]); nar.setData(P_[m_, 0], P_[m_, 1])
            self.p10[n].setTitle(tytul, size='9pt')
        rysuj(0, P, m, f'narzędzie ({self._opis_zr(rdz)}): Fisher {pl(mi["fisher"])}, sylw. {pl(mi["sil"])}, {kk}NN {pl(mi["hit"])}')
        klucz = ('R10', int(i), self._klucz_etyk(), round(lam, 3), osie, kk)
        gra = self._z.bt_play.isChecked()
        tekst = f'Checkpoint {i + 1}, {zrodlo}. Etykieta: {self._opis_etyk(rdz, mies)}. Narzędzie: Fisher {pl(mi["fisher"])}, sylwetka {pl(mi["sil"])}.'
        if self.ch_L10_tlo.isChecked():
            if klucz not in self._pam and not gra:
                self._pam[klucz] = projections.widoki_zera_przestrzennego_gpu(X, rdz, self._wyklucz(rdz, mies), keep, lam, osie, kk)
            wy = self._pam.get(klucz, 'brak')
            if wy is None:
                tekst += ' Zero przestrzenne: brak położeń maski w tle.'
            elif wy == 'brak':
                tekst += ' Maska w tle liczy się na pauzie.'
            else:
                for n, nazwa in ((1, 'mediana'), (2, 'najlepsza')):
                    P_, m_, mm, j = wy['widoki'][nazwa]
                    rysuj(n, P_, m_, f'ta sama maska w tle ({nazwa} z {wy["n"]}): Fisher {pl(mm["fisher"])}, sylw. {pl(mm["sil"])}, {kk}NN {pl(mm["hit"])}')
                    self.s10[n][1].setBrush(pg.mkBrush(255, 150, 40) if n == 1 else pg.mkBrush(255, 70, 70))
                tekst += (f' Zero przestrzenne ({wy["n"]} położeń, LDA od nowa): p Fisher = {fp(wy["p"]["fisher"])}, p sylwetka = '
                          f'{fp(wy["p"]["sil"])}, p {kk}NN = {fp(wy["p"]["hit"])}.')
        else:
            for n in (1, 2):
                self.s10[n][0].setData([], []); self.s10[n][1].setData([], []); self.p10[n].setTitle('')
        self._lb['R10'].setText(tekst)

    def _opis_zr(self, rdz):
        return 'grupa bez etykiet' if self.cb_L_etyk.currentIndex() == 2 else ('obrys' if self.cb_L_etyk.currentIndex() == 0 else 'ramka GT')
