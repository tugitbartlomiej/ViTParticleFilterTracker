"""Okienko R12: kNN na chmurze 3D (wspólna baza PCA albo wyrównana Procrustesem, interpolacja, metryka)."""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PyQt6.QtGui import QVector3D

from laser_app.compute import projections
from laser_app.ui.gl_view import GLView
from laser_app.ui.projection_windows.shared import pl, TLO


class KnnMixin:
    """Okienko R12: kNN na chmurze 3D (wspólna baza PCA albo wyrównana Procrustesem, interpolacja, metryka).
    Część ProjectionWindows (ui/projections_view.py); korzysta z jej atrybutów przez self."""

    # ------------------------------------------------------------------ R12: kNN na chmurze 3D
    def _buduj_knn(self):
        self.v12 = GLView(); self.v12.setBackgroundColor(TLO)
        self.v12.setCameraPosition(distance=40, elevation=18, azimuth=-55)
        self.sc12_tlo = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=3, color=(0.55, 0.62, 0.7, 0.55), pxMode=True)
        self.sc12_nar = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=13, color=(0.4, 1, 0.4, 1), pxMode=True)
        self.kr12 = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), color=(1, 1, 1, 0.5), width=1, mode='lines', antialias=True)
        for it in (self.sc12_tlo, self.kr12, self.sc12_nar):
            self.v12.addItem(it)
        self._kadr12 = None
        self._pam12 = {}                                                     # bazy PCA R12 (wspólna, wyrównane): osobno od _pam, które _ekran czyści
        return self._z_napisem(self.v12, 'R12')

    def _osie12(self, c, keep_t):
        """Baza rzutu R12 (192, 3) torch dla checkpointu c i informacja, czy chmurę przed rzutem wyśrodkować (PCA tak, oś LDA nie).
        Wspólna baza PCA liczona raz na historię (próbka 32 checkpointów), wyrównane i surowe osie raz na checkpoint, wszystko w `_pam`."""
        import torch
        d = self.dane
        dev = projections._dev()
        if self.cb_R12_widok.currentIndex() == 1:
            B, _ = self._baza_L()
            return torch.as_tensor(np.asarray(B, np.float32), device=dev), False
        tryb = self.cb_R12_baza.currentIndex()
        nb = 0 if tryb == 0 else int(self.sp_R12_bwyg.value())
        ke = (str(self._h.katalog), d.K, self._klucz_etyk(), int(keep_t.sum()))
        kU = ('R12U',) + ke
        if kU not in self._pam12:
            ks = np.unique(np.linspace(0, d.K - 1, min(d.K, 32)).round().astype(int))
            Xs = torch.as_tensor(np.asarray(d.X[ks], np.float32), device=dev)[:, keep_t]
            self._pam12[kU] = projections.baza_pca3_wspolna_t(Xs)
        U = self._pam12[kU]
        if tryb == 0:
            return U, True
        kV = ('R12V', tryb, nb, c) + ke
        if kV not in self._pam12:
            if len(self._pam12) > 3000:
                self._pam12 = {k_: v_ for k_, v_ in self._pam12.items() if k_[0] == 'R12U'}   # ograniczenie pamięci: zostaje tylko baza wspólna
            ks = np.arange(max(0, c - nb), min(d.K - 1, c + nb) + 1)               # okno ± nb checkpointów, obcięte na brzegach historii
            Xw = torch.as_tensor(np.asarray(d.X[ks], np.float32), device=dev)[:, keep_t]
            self._pam12[kV] = projections.baza_pca3_ckpt_t(Xw, U if tryb == 1 else None)
        return self._pam12[kV], True

    def _sasiedzi12(self, c, V, cen, keep_t, m_t, idx_t, k):
        """(ułamek sąsiadów-narzędzi tokenów narzędzia, indeksy sąsiadów (n, k)) dla checkpointu c: sąsiedzi w każdym checkpoincie okna
        ± „wygładzanie ułamka” osobno (w bazie V checkpointu c), ułamek uśredniony po oknie, indeksy z checkpointu c."""
        import torch
        d = self.dane
        nsm = int(self.sp_R12_fwyg.value())
        ks = np.arange(max(0, c - nsm), min(d.K - 1, c + nsm) + 1)
        Xw = torch.as_tensor(np.asarray(d.X[ks], np.float32), device=projections._dev())[:, keep_t]
        if self.cb_R12_przestrzen.currentIndex() == 0:
            Q = (Xw - Xw.mean(1, keepdim=True) if cen else Xw) @ V
        else:
            Q = Xw
        nn = projections.knn_t(Q, k, self.cb_R12_metryka.currentIndex())                  # (b, n, k)
        fr = m_t[nn[:, idx_t]].float().mean(-1).mean(0)                          # (nt,) średnia po oknie
        return fr.cpu().numpy(), nn[c - int(ks[0])].cpu().numpy()

    def _rysuj_knn(self):
        import torch
        d = self.dane
        ia, ja, f, wi = self._indeksy()
        rdz, mies = self._etykieta(wi)
        keep = ~mies
        m = rdz[keep]
        nt = int(m.sum())
        nk = int(keep.sum())
        if nt < 2:
            self._lb['R12'].setText('Za mało tokenów narzędzia w tej etykiecie.')
            return
        dev = projections._dev()
        pca = self.cb_R12_widok.currentIndex() == 0
        tryb = self.cb_R12_baza.currentIndex()
        # interpolacja między checkpointami; surowa baza PCA skacze, więc jej środek (średnia przeskoczonych osi) nie ma sensu
        interp = self.ch_R12_interp.isChecked() and not (pca and tryb == 2) and ia != ja and f > 0
        if not interp:
            ia = ja = wi; f = 0.0
        keep_t = torch.as_tensor(keep, device=dev)
        m_t = torch.as_tensor(m, device=dev)
        idx_n = np.flatnonzero(m)
        idx_t = torch.as_tensor(idx_n, device=dev)
        k = int(self.sp_R12_k.value()) or projections.k_sasiadow(nt)
        k = int(min(k, nk - 1))
        P_c, fr_c, nn_c = {}, {}, {}
        for c in sorted({ia, ja, wi}):
            V, cen = self._osie12(c, keep_t)
            X = torch.as_tensor(np.asarray(d.X[c], np.float32), device=dev)[keep_t]
            P_c[c] = (X - X.mean(0, keepdim=True) if cen else X) @ V              # każdy checkpoint w swojej (wyrównanej) bazie: ciągłość przy zmianie i → j
            fr_c[c], nn_c[c] = self._sasiedzi12(c, V, cen, keep_t, m_t, idx_t, k)
        P = ((1 - f) * P_c[ia] + f * P_c[ja]).cpu().numpy().astype(np.float32)
        frac = ((1 - f) * fr_c[ia] + f * fr_c[ja]).astype(np.float32)             # kolor też mieszany; linie z najbliższego checkpointu
        nn = nn_c[wi]
        traf = m[nn[idx_n]]                                                    # (nt, k) czy sąsiad jest narzędziem
        # kolor tokenu narzędzia: czerwony (0) → żółty → zielony (1) wg ułamka sąsiadów będących narzędziem
        kol = np.stack([np.clip(2 - 2 * frac, 0, 1), np.clip(2 * frac, 0, 1), 0.15 * np.ones_like(frac), np.ones_like(frac)], 1).astype(np.float32)
        self.sc12_tlo.setData(pos=P[~m], size=3, color=(0.55, 0.62, 0.7, 0.55))
        self.sc12_nar.setData(pos=P[m], size=13, color=kol)
        if self.ch_R12_kr.isChecked():
            a = np.repeat(P[idx_n], k, axis=0)
            b = P[nn[idx_n].reshape(-1)]
            seg = np.empty((2 * len(a), 3), np.float32); seg[0::2], seg[1::2] = a, b
            kk = np.where(traf.reshape(-1, 1), [[0.3, 1.0, 0.3, 0.7]], [[1.0, 0.3, 0.25, 0.7]]).astype(np.float32)
            self.kr12.setData(pos=seg, color=np.repeat(kk, 2, axis=0))
            self.kr12.setVisible(True)
        else:
            self.kr12.setVisible(False)
        r_ = float(np.linalg.norm(P, axis=1).max())                         # kadr mieści każdy token, także daleko leżące narzędzie
        klucz = (str(self._h.katalog), self.cb_R12_widok.currentIndex(), tryb)
        if self._kadr12 is None or self._kadr12[0] != klucz or r_ > self._kadr12[1] * 0.95 or r_ < self._kadr12[1] * 0.4:
            self._kadr12 = (klucz, r_)                                       # zmiana skali chmury w czasie: kadr z histerezą, bez skakania
            self.v12.opts['center'] = QVector3D(0, 0, 0)
            self.v12.setCameraPosition(distance=max(2.9 * r_, 3.0))
        losowo = (nt - 1) / max(nk - 1, 1)
        nb, nf = int(self.sp_R12_bwyg.value()), int(self.sp_R12_fwyg.value())
        if pca:
            widok = 'PCA3 chmury (bez etykiet), ' + ['baza wspólna dla całej historii',
                                                    'baza w każdym checkpoincie wyrównana Procrustesem do wspólnej',
                                                    'baza w każdym checkpoincie surowa (skacze)'][tryb]
            if tryb and nb:
                widok += f', kowariancja uśredniona ± {nb} ckpt'
        else:
            widok = 'oś LDA jak w R9'
        czas = f'{pl(ia + f + 1)} (interpolacja)' if interp else f'{wi + 1}'
        self._lb['R12'].setText(
            f'Checkpoint {czas}. Widok 3D: {widok}; sąsiedzi (k = {k}, metryka {["euklidesowa", "kosinusowa"][self.cb_R12_metryka.currentIndex()]}) liczeni '
            f'{"w tym widoku 3D" if self.cb_R12_przestrzen.currentIndex() == 0 else "w pełnych 192D"}. '
            f'Duże kule = tokeny narzędzia ({nt}), kolor = ułamek ich sąsiadów, którzy też są narzędziem{f" (uśredniony ± {nf} ckpt)" if nf else ""} '
            f'(zielony 1, czerwony 0); średnio {pl(float(frac.mean()))} (losowo {pl(losowo, 3)}). Linie: zielone = sąsiad narzędzie, czerwone = sąsiad tło. '
            f'Szare = tło ({int((~m).sum())}); łaty mieszane pominięte ({int(mies.sum())}).')
