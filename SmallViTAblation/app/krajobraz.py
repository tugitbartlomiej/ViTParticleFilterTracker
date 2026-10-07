"""Okno „Krajobraz i tunele”: po czym biegają cząstki w przestrzeni makrowektorów.

Położenia cząstek P [T, 600, 3] z przestrzeni makro (ta sama projekcja, środek i wygładzenie). Trzy obrazy
(przegląd literatury 2026-10-07; generator „studni” TFT: Research/DiagnosticTool/koopman_m5/plotly_studnie.py):
- teren energii swobodnej: U = −ln gęstość położeń na płaszczyźnie dwóch głównych osi chmury (jak powierzchnia
  energii swobodnej z trajektorii w dynamice molekularnej, Laio i Parrinello 2002); dołki = miejsca, w których
  cząstki przebywają długo (studnie, klatki), grzbiety = rzadko odwiedzane przejścia;
- wosk z tunelami: siatka wokseli 3D, w której każde przejście cząstki zostawia ślad; izopowierzchnia odwiedzin
  to wydrążone tunele, kolor = krok pierwszego przejścia (wiek tunelu). Odpowiednik kanałów i „sznurów” ruchu
  w szkle (Donati i in. 1998) tylko wtedy, gdy cząstki wchodzą w ślady poprzedników: mierzy to liczba pod oknem
  (udział odwiedzin w wokselach wcześniej przebytych przez inną cząstkę wobec kontroli z losowym przesunięciem
  czasu każdej cząstki);
- otoczka: izopowierzchnia gęstości z całej historii obejmująca wybrany odsetek masy, czyli „powierzchnia”,
  po której i wewnątrz której biegają cząstki.
Analogia ma granice: tokeny nie oddziałują ze sobą jak cząstki szkła, wszystkie przesuwa ta sama zmiana wag,
a −ln gęstość nie jest prawdziwym potencjałem (brak równowagi, pęd, wspólny zegar uczenia). Rzut z 768 wymiarów
do 3 może też tworzyć pozorne dołki i kanały.
"""
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QSpinBox, QVBoxLayout, QWidget

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from uklad import Przeplyw, dodaj
from widok_gl import WidokGL


class OknoKrajobrazu(QWidget):
    TRYBY = ['teren energii U = −ln gęstość (studnie)', 'wosk z wydrążonymi tunelami', 'otoczka: powierzchnia ruchu cząstek']

    def __init__(self, makro):
        super().__init__()
        self.makro = makro
        self.setWindowTitle('Krajobraz i tunele: po czym biegają cząstki')
        self.resize(1300, 950)
        lay = QVBoxLayout(self)
        w = Przeplyw(); lay.addLayout(w)
        self.combo_tryb = QComboBox(); self.combo_tryb.addItems(self.TRYBY)
        dodaj(w, 'Obraz:', self.combo_tryb, __doc__.split('\n\n')[1])
        self.combo_gest = QComboBox(); self.combo_gest.addItems(['cała historia', 'do bieżącego kroku (drążenie)',
                                                                 'ostatnie N kroków (okno)'])
        self.combo_gest.setCurrentIndex(1)
        dodaj(w, 'Gęstość z:', self.combo_gest, 'Z których kroków liczyć gęstość / odwiedziny:\n'
              '- cała historia: jeden stały krajobraz, cząstki poruszają się po nim;\n'
              '- do bieżącego kroku: tunele i dołki powstają w miarę uczenia (drążenie);\n'
              '- okno: tylko ostatnie N kroków (krajobraz chwili, ślady starsze znikają).')
        self.spin_okno = QSpinBox(); self.spin_okno.setRange(5, 1000); self.spin_okno.setValue(100)
        dodaj(w, 'N', self.spin_okno, 'Długość okna kroków dla „ostatnie N kroków”.')
        self.spin_roz = QSpinBox(); self.spin_roz.setRange(24, 128); self.spin_roz.setValue(64)
        dodaj(w, 'Siatka:', self.spin_roz, 'Rozdzielczość siatki (wokseli na bok w 3D, komórek na bok terenu x 2).')
        self.spin_rozmycie = QDoubleSpinBox(); self.spin_rozmycie.setRange(0, 6); self.spin_rozmycie.setSingleStep(.5)
        self.spin_rozmycie.setValue(1.0)
        dodaj(w, 'Rozmycie:', self.spin_rozmycie, 'Odchylenie Gaussa wygładzania gęstości (w komórkach siatki).')
        self.spin_prog = QDoubleSpinBox(); self.spin_prog.setRange(.05, .99); self.spin_prog.setSingleStep(.05)
        self.spin_prog.setValue(.85)
        dodaj(w, 'Masa / próg:', self.spin_prog, 'Wosk: próg izopowierzchni jako kwantyl odwiedzonych wokseli (wyżej = węższe\n'
              'tunele, niżej = szersze). Otoczka: jaki odsetek wszystkich odwiedzin ma być wewnątrz powierzchni.')
        self.spin_wys = QDoubleSpinBox(); self.spin_wys.setRange(.05, 3); self.spin_wys.setSingleStep(.05); self.spin_wys.setValue(.4)
        dodaj(w, 'Wysokość terenu:', self.spin_wys, 'Wysokość terenu względem jego szerokości.')
        self.spin_krycie = QDoubleSpinBox(); self.spin_krycie.setRange(.05, 1); self.spin_krycie.setSingleStep(.05)
        self.spin_krycie.setValue(.55)
        dodaj(w, 'Krycie:', self.spin_krycie, 'Krycie terenu albo powierzchni.')
        self.chk_lity = QCheckBox('lity wosk (odwrotnie)')
        self.chk_lity.setToolTip('Wosk: zamiast powierzchni tuneli półprzezroczysty blok wosku, przezroczysty tam, gdzie\n'
                                 'cząstki go przebyły (odwiedziny wydrążają wosk).')
        w.addWidget(self.chk_lity)
        self.chk_czastki = QCheckBox('cząstki'); self.chk_czastki.setChecked(True); w.addWidget(self.chk_czastki)
        self.spin_slad = QSpinBox(); self.spin_slad.setRange(0, 500); self.spin_slad.setValue(20)
        dodaj(w, 'ślad', self.spin_slad, 'Ślad cząstek (kroków); 0 = bez.')
        self.etykieta = QLabel(''); self.etykieta.setWordWrap(True); lay.addWidget(self.etykieta)
        self.widok = WidokGL(); self.widok.setBackgroundColor(pg.mkColor(10, 10, 14)); lay.addWidget(self.widok, 1)
        self.widok.setCameraPosition(distance=560, elevation=30, azimuth=-60)
        v = self.widok
        self.pow = gl.GLSurfacePlotItem(x=np.arange(2), y=np.arange(2), z=np.zeros((2, 2)), glOptions='translucent')
        self.siatka3 = gl.GLMeshItem(vertexes=np.zeros((3, 3), np.float32), faces=np.array([[0, 1, 2]]), smooth=True,
                                     shader='shaded', glOptions='translucent')
        self.obj = gl.GLVolumeItem(np.zeros((2, 2, 2, 4), np.uint8), sliceDensity=2, smooth=True, glOptions='translucent')
        self.slady = gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', glOptions='translucent')
        self.punkty = gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=6, pxMode=True, glOptions='translucent')
        for it in (self.pow, self.siatka3, self.obj, self.slady, self.punkty):
            it.setVisible(False); v.addItem(it)
        self._dane = None; self._stan = None
        self.combo_tryb.currentIndexChanged.connect(lambda i: self.spin_prog.setValue(.4 if i == 1 else .85))
        for s in (self.combo_tryb.currentIndexChanged, self.combo_gest.currentIndexChanged, self.spin_okno.valueChanged,
                  self.spin_roz.valueChanged, self.spin_rozmycie.valueChanged, self.spin_prog.valueChanged,
                  self.spin_wys.valueChanged, self.spin_krycie.valueChanged, self.chk_lity.toggled, self.chk_czastki.toggled,
                  self.spin_slad.valueChanged):
            s.connect(lambda *_: self.odswiez(True))
        self.timer = QTimer(self); self.timer.setInterval(200); self.timer.timeout.connect(self.odswiez); self.timer.start()

    # ------------------------------------------------------------ dane wspólne dla trybów
    def _przygotuj(self):
        """Siatki indeksów (raz na dane makro): położenia -> komórki terenu 2D i woksele 3D, pierwsze odwiedziny."""
        m = self.makro
        P = np.asarray(m.P, np.float32)
        klucz = (id(m.P), P.shape, float(P[-1].sum()), self.spin_roz.value())
        if self._dane is not None and self._dane['klucz'] == klucz:
            return self._dane
        T = len(P); n = self.spin_roz.value()
        X = P.reshape(-1, 3)
        lo, hi = np.percentile(X, .5, 0), np.percentile(X, 99.5, 0)
        marg = (hi - lo) * .08; lo, hi = lo - marg, hi + marg
        idx3 = np.clip(((P - lo) / (hi - lo) * n).astype(np.int32), 0, n - 1)        # [T, 600, 3]
        lin3 = (idx3[..., 0] * n + idx3[..., 1]) * n + idx3[..., 2]                    # [T, 600]
        # teren: 2 główne osie chmury położeń
        probka = X[::max(1, len(X) // 60000)]
        mu = probka.mean(0)
        _, _, Vt = np.linalg.svd(probka - mu, full_matrices=False)
        Q = (P - mu) @ Vt[:2].T                                                        # [T, 600, 2]
        n2 = 2 * n
        qlo, qhi = np.percentile(Q.reshape(-1, 2), .5, 0), np.percentile(Q.reshape(-1, 2), 99.5, 0)
        qm = (qhi - qlo) * .08; qlo, qhi = qlo - qm, qhi + qm
        idx2 = np.clip(((Q - qlo) / (qhi - qlo) * n2).astype(np.int32), 0, n2 - 1)
        lin2 = idx2[..., 0] * n2 + idx2[..., 1]
        # pierwsze odwiedziny woksela: krok i cząstka
        klucz_pierw = np.full(n ** 3, np.iinfo(np.int64).max, np.int64)
        np.minimum.at(klucz_pierw, lin3.reshape(-1), (np.arange(T)[:, None] * 600 + np.arange(600)[None]).reshape(-1))
        self._dane = dict(klucz=klucz, P=P, T=T, n=n, lo=lo, hi=hi, lin3=lin3, Q=Q, n2=n2, qlo=qlo, qhi=qhi, lin2=lin2,
                          pierw=klucz_pierw, slady=self._slady_poprzednikow(lin3, n))
        return self._dane

    @staticmethod
    def _slady_poprzednikow(lin3, n):
        """Udział odwiedzin w wokselach, które wcześniej przebyła INNA cząstka, i to samo w kontroli z losowym
        przesunięciem czasu każdej cząstki (te same tory, zerwane zgranie w czasie)."""
        def udzial(L):
            T = len(L)
            k = (np.arange(T)[:, None] * 600 + np.arange(600)[None]).reshape(-1)
            pierw = np.full(n ** 3, np.iinfo(np.int64).max, np.int64)
            np.minimum.at(pierw, L.reshape(-1), k)
            pk = pierw[L.reshape(-1)]
            krok_p, czast_p = pk // 600, pk % 600
            krok, czast = k // 600, k % 600
            wczesniej_inna = (krok_p < krok) & (czast_p != czast)
            nowe = krok > 0
            return float(wczesniej_inna[nowe].mean())
        rng = np.random.default_rng(0)
        T = len(lin3)
        przes = np.stack([np.roll(lin3[:, i], rng.integers(T)) for i in range(lin3.shape[1])], 1)
        return udzial(lin3), udzial(przes)

    def _zakres_krokow(self, t, T):
        g = self.combo_gest.currentIndex()
        if g == 0:
            return 0, T - 1
        if g == 1:
            return 0, t
        return max(0, t - self.spin_okno.value() + 1), t

    # ------------------------------------------------------------ rysowanie
    def odswiez(self, sila=False):
        m = self.makro
        if not self.isVisible() or m.P is None:
            return
        stan = (m.t, id(m.P), self.combo_tryb.currentIndex())
        if stan == self._stan and not sila:
            return
        self._stan = stan
        d = self._przygotuj()
        t = m.t; a, b = self._zakres_krokow(t, d['T'])
        tryb = self.combo_tryb.currentIndex()
        for it in (self.pow, self.siatka3, self.obj):
            it.setVisible(False)
        if tryb == 0:
            self._teren(d, a, b, t)
        else:
            self._wosk(d, a, b, t, otoczka=tryb == 2)
        s1, s0 = d['slady']
        self.etykieta.setText(
            f'krok {t + 1}/{d["T"]}, gęstość z kroków {a}–{b}. '
            f'Wchodzenie w ślady poprzedników (siatka {d["n"]}³): {s1 * 100:.1f} % odwiedzin trafia w woksel przebyty '
            f'wcześniej przez inną cząstkę; kontrola z losowo przesuniętym czasem każdej cząstki: {s0 * 100:.1f} %. '
            + ('Więcej niż w kontroli = cząstki idą wspólnymi kanałami (jak sznury ruchu w szkle).' if s1 > s0 * 1.05 else
               'Nie więcej niż w kontroli = brak wspólnych kanałów ponad to, co daje sam kształt chmury.'))

    def _gestosc(self, lin, rozmiar, a, b):
        return np.bincount(lin[a:b + 1].reshape(-1), minlength=rozmiar).astype(np.float32)

    def _czastki(self, pos, t, d, z_fun=None):
        m = self.makro
        tok = m._widoczne()
        if not self.chk_czastki.isChecked() or len(tok) == 0:
            self.punkty.setVisible(False); self.slady.setVisible(False); return
        kol = m._kolory()[tok].copy(); kol[:, 3] = 1
        self.punkty.setData(pos=pos[t][tok].astype(np.float32), color=kol, size=m.spin_rozmiar.value())
        self.punkty.setVisible(True)
        L = min(self.spin_slad.value(), t)
        if L > 0:
            S = pos[t - L:t + 1][:, tok]                                                # [L+1, n, 3]
            seg = np.stack([S[:-1], S[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 3)
            c = np.repeat(kol[:, None, :], L, 1).copy(); c[..., 3] = np.linspace(.1, .8, L)[None]
            self.slady.setData(pos=seg.astype(np.float32), color=np.repeat(c, 2, 1).reshape(-1, 4))
            self.slady.setVisible(True)
        else:
            self.slady.setVisible(False)

    def _teren(self, d, a, b, t):
        from scipy import ndimage
        import matplotlib
        n2 = d['n2']
        rho = self._gestosc(d['lin2'], n2 * n2, a, b).reshape(n2, n2)
        rho = ndimage.gaussian_filter(rho, max(self.spin_rozmycie.value(), .01) * 1.5)
        jest = rho > rho.max() * 1e-4
        U = np.where(jest, -np.log(rho / rho.sum() + 1e-12), 0)
        U = U - U[jest].min() if jest.any() else U
        sufit = float(np.percentile(U[jest], 98)) if jest.any() else 1
        U = np.where(jest, np.minimum(U, sufit), sufit)
        szer = 300.0
        xs = np.linspace(-szer / 2, szer / 2, n2); ys = np.linspace(-szer / 2, szer / 2, n2)
        skala = szer * self.spin_wys.value() / max(sufit, 1e-9)
        Zs = U * skala
        kol = matplotlib.colormaps['viridis'](1 - U / max(sufit, 1e-9)).astype(np.float32)
        kol[..., 3] = np.where(jest, self.spin_krycie.value(), .12)
        self.pow.setData(x=xs, y=ys, z=Zs, colors=kol)
        self.pow.setVisible(True)
        # cząstki na terenie: (x, y) z osi chmury, z = U komórki
        Q, qlo, qhi = d['Q'], d['qlo'], d['qhi']
        L = min(self.spin_slad.value(), t)
        kroki = slice(t - L, t + 1)
        q = Q[kroki]
        ix = np.clip(((q - qlo) / (qhi - qlo) * n2).astype(int), 0, n2 - 1)
        xy = (q - qlo) / (qhi - qlo) * szer - szer / 2
        z = Zs[ix[..., 0], ix[..., 1]] + 2
        pos = np.full((t + 1, 600, 3), 0, np.float32)
        pos[kroki] = np.concatenate([xy, z[..., None]], -1)
        self._czastki(pos, t, d)

    def _wosk(self, d, a, b, t, otoczka=False):
        from scipy import ndimage
        import matplotlib
        n = d['n']
        V = self._gestosc(d['lin3'], n ** 3, 0 if otoczka and self.combo_gest.currentIndex() == 0 else a, b).reshape(n, n, n)
        Vs = ndimage.gaussian_filter(V, self.spin_rozmycie.value()) if self.spin_rozmycie.value() > 0 else V
        lo, hi = d['lo'], d['hi']
        skala = (hi - lo) / n
        if not Vs.any():
            return
        if self.chk_lity.isChecked() and not otoczka:                                # blok wosku z wydrążeniami
            w = np.clip(Vs / max(float(np.percentile(Vs[Vs > 0], 90)), 1e-9), 0, 1)
            rgba = np.zeros((n, n, n, 4), np.uint8)
            rgba[..., :3] = (235, 205, 140)
            rgba[..., 3] = (np.clip(1 - w, 0, 1) * 255 * self.spin_krycie.value() * .25).astype(np.uint8)
            self.obj.setData(rgba); self.obj.resetTransform()
            self.obj.scale(*skala); self.obj.translate(*lo); self.obj.setVisible(True)
        else:
            from skimage import measure
            dodatnie = np.sort(Vs[Vs > 0])
            if otoczka:                                                               # poziom obejmujący masę
                c = np.cumsum(dodatnie[::-1]); poziom = dodatnie[::-1][np.searchsorted(c, c[-1] * self.spin_prog.value())]
            else:
                poziom = dodatnie[int(self.spin_prog.value() * (len(dodatnie) - 1))]
            poziom = float(min(max(poziom, dodatnie[0] * 1.0001), dodatnie[-1] * .999))
            try:
                ver, fac, _, _ = measure.marching_cubes(Vs, poziom)
            except (ValueError, RuntimeError):
                return
            idx = np.clip(np.round(ver).astype(int), 0, n - 1)
            if otoczka:
                kol = np.tile(np.array([.55, .75, 1, self.spin_krycie.value()], np.float32), (len(ver), 1))
            else:                                                                     # wiek tunelu: krok pierwszego przejścia
                pk = d['pierw'][(idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2]]
                wiek = np.where(pk < np.iinfo(np.int64).max, pk // 600, b).astype(np.float32)
                kol = matplotlib.colormaps['plasma'](np.clip(wiek / max(d['T'] - 1, 1), 0, 1)).astype(np.float32)
                kol[:, 3] = self.spin_krycie.value()
            md = gl.MeshData(vertexes=(ver * skala + lo).astype(np.float32), faces=fac, vertexColors=kol)
            self.siatka3.setMeshData(meshdata=md); self.siatka3.setVisible(True)
        self._czastki(d['P'], t, d)
