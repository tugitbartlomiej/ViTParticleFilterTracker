"""Rysunek mapy pułapek (okienko 14 i nakładka na sferze): kolory cząsteczek wg charakteru pola, wyróżnianie i elementy GL.
Obliczenia mapy są w compute/trap_map.py (bez Qt)."""
import numpy as np
import pyqtgraph.opengl as _gl

from laser_app.compute.trap_map import charakter

KOLORY_TYPOW = np.array([[0.45, 0.48, 0.55, 0.55],                    # brak zera: szary
                         [0.30, 0.95, 0.45, 1.0],                     # ściąga: zielony
                         [1.00, 0.78, 0.25, 1.0],                     # siodło: żółty
                         [1.00, 0.30, 0.30, 1.0]], np.float32)        # wypycha: czerwony
KOLOR_WIRU = np.array([0.35, 0.85, 1.0], np.float32)                   # domieszka błękitu: wir (λ zespolone)


TRYBY_KOLORU = ('kształt pola w cząsteczce (λ₀): ściąga / siodło / wypycha, błękit = wir',
                'pułapka w komórce (λ w zerze pola; szare = brak zera w komórce)',
                'siła skoku |f(0)| w bloku 6',
                'odległość pułapki od cząsteczki / r_kom (szare = brak zera)')


def kolory(d, k, tryb=0, pos=None):
    """Kolory (n, 4) cząsteczek w checkpoincie k wg TRYBY_KOLORU; pos = położenia koloidów (n, 3) dla trybu 3."""
    if tryb in (0, 1) or (tryb == 3 and pos is None):
        lam, zb = (d['lam0'][k], np.ones(d['zb'][k].shape, bool)) if tryb == 0 else (d['lam'][k], d['zb'][k])
        typ, wir = charakter(lam[None], zb[None])
        typ = typ[0]; wir = wir[0] & (typ > 0)
        c = KOLORY_TYPOW[typ].copy()
        c[wir, :3] = 0.55 * c[wir, :3] + 0.45 * KOLOR_WIRU
        return c
    if tryb == 2:
        v = np.linalg.norm(d['skok'][k], axis=1); v = v / max(float(np.percentile(v, 95)), 1e-30)
    else:
        v = np.linalg.norm(d['ogn'][k] - np.asarray(pos, np.float32), axis=1) / np.maximum(d['r_kom'][k], 1e-30)
    import pyqtgraph as pg
    c = pg.colormap.get('plasma').map(np.clip(v, 0, 1), mode='float').astype(np.float32)
    if c.shape[1] == 3:
        c = np.concatenate([c, np.ones((len(c), 1), np.float32)], 1)
    if tryb == 3:
        c[~d['zb'][k]] = KOLORY_TYPOW[0]
    return c


def maska_filtru(d, k, filtr):
    """Które cząsteczki wyróżnić: 0 wszystkie, 1 siodła (λ₀), 2 ściągające (λ₀), 3 wypychające (λ₀), 4 z wirem (λ₀), 5 z zerem w komórce."""
    n = d['zb'].shape[1]
    if filtr == 0:
        return np.ones(n, bool)
    if filtr == 5:
        return np.asarray(d['zb'][k], bool)
    typ, wir = charakter(d['lam0'][k][None], np.ones((1, n), bool))
    return wir[0] if filtr == 4 else typ[0] == {1: 2, 2: 1, 3: 3}[filtr]


class TrapMapGL:
    """Rysunek mapy w jednym widoku GL: kropki cząsteczek (kolor wg trybu), linie cząsteczka → pułapka (lupa) i wektory skoku.
    Dwa egzemplarze: okienko 14 i nakładka na dużej sferze (element GL należy do jednego widoku)."""

    def __init__(self):
        self.kropki = _gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=6, pxMode=True)
        self.pulapki = _gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=1.5, antialias=True)
        self.skoki = _gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=1, antialias=True)
        self.wybrana = _gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=16, color=(1, 1, 1, 0.9), pxMode=True)
        self.elementy = (self.skoki, self.pulapki, self.kropki, self.wybrana)
        for e in self.elementy:
            e.setGLOptions('translucent')
        self.wybrana.setDepthValue(3)

    def dodaj_do(self, widok):
        for e in self.elementy:
            widok.addItem(e)

    def widoczna(self, on):
        for e in self.elementy:
            e.setVisible(bool(on))

    def ustaw(self, P, kol, rozm=6.0, Q=None, kol_q=None, S=None, wybrana=None):
        """P (n, 3) kropki, kol (n, 4); Q (m, 2, 3) odcinki do pułapek z kolorami kol_q (m, 4); S (n, 2, 3) wektory skoku."""
        self.kropki.setData(pos=np.asarray(P, np.float32), color=np.asarray(kol, np.float32), size=float(rozm))
        if Q is not None and len(Q):
            self.pulapki.setData(pos=np.asarray(Q, np.float32).reshape(-1, 3), color=np.repeat(np.asarray(kol_q, np.float32), 2, axis=0))
            self.pulapki.setVisible(True)
        else:
            self.pulapki.setVisible(False)
        if S is not None and len(S):
            self.skoki.setData(pos=np.asarray(S, np.float32).reshape(-1, 3), color=(1.0, 1.0, 1.0, 0.35))
            self.skoki.setVisible(True)
        else:
            self.skoki.setVisible(False)
        if wybrana is not None:
            self.wybrana.setData(pos=np.asarray(wybrana, np.float32).reshape(1, 3), size=float(rozm) * 2.6, color=(1, 1, 1, 0.9))
            self.wybrana.setVisible(True)
        else:
            self.wybrana.setVisible(False)
