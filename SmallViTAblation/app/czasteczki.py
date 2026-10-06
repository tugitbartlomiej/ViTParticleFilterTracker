"""Widok „Cząsteczki”: każdy z 600 tokenów (patch 16x16 na siatce 20x30) jako okrąg 768 wektorów.

Kanał c ma stały kąt 2*pi*c/768 i własny kolor z koła barw (768 kolorów). Wektor wychodzi ze środka
patcha; jego długość to promień zera + skala * wartość kanału w tym tokenie. Cienki szary okrąg = zero:
wektor krótszy od okręgu = wartość ujemna (po GELU najwyżej −0,17), dłuższy = dodatnia.
Opcje: przed / po / różnica / przed i po nałożone, przerzedzenie kanałów, obraz pod spodem,
mapa atencji ViT i strzałki „który patch ciągnie z którego” (opcje w opcje_atencji.py).
"""
import numpy as np
from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel, QVBoxLayout, QWidget
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from opcje_atencji import OpcjeAtencji
from uklad import WaskiCombo, etykieta_statusu
from uklad import PanelOpcji, dodaj
from wspolne import KOLUMNY, KOMORKA, WIERSZE, kolory_kanalow


class WidokCzasteczek(QWidget):
    TRYBY = ['PRZED', 'PO', 'różnica po − przed', 'przed (szary) i po (kolor) nałożone']
    KANALY = ['wszystkie 768', 'co 2. kanał (384)', 'co 4. kanał (192)', 'co 8. kanał (96)', 'tylko J128 (128)']

    def __init__(self, kanaly_j128, PanelWykresu, osobne_okno):
        super().__init__()
        self.j128 = np.asarray(kanaly_j128)
        self._osobne_okno = osobne_okno
        self._Panel = PanelWykresu
        self.przed = self.po = self.rgb = None
        self.atencja_fn = None
        self._atencja = {}
        self.token = None
        lay = QVBoxLayout(self)
        # opcje jak w widoku 3D: pasek (przycisk „Opcje” i stan) oraz zakładki z grupami opcji;
        # self.panel to dalej panel wykresu matplotlib (pasek nawigacji, rozmiar, osobne okno)
        self.panel_opcji = PanelOpcji(lay, ['Wektory', 'Obraz', 'Atencja'])

        # Wektory: co rysować | jak długie
        w = self.panel_opcji.wiersz('Wektory')
        self.combo_tryb = dodaj(w, 'Pokaż:', WaskiCombo(10), 'PRZED / PO: wartości latentu tokenu.\n'
                                'Różnica: po − przed (wektor dłuższy od okręgu = kanał wzmocniony, krótszy = osłabiony).\n'
                                'Nałożone: przed na szaro, po w kolorach kanałów.')
        self.combo_tryb.addItems(self.TRYBY)
        self.combo_kan = dodaj(w, 'Kanały:', WaskiCombo(9), 'Ile z 768 wektorów rysować w każdym okręgu (mniej = szybciej i czytelniej).\n'
                               'Kąt i kolor kanału są zawsze te same: kanał c pod kątem 360° · c / 768.')
        self.combo_kan.addItems(self.KANALY)
        self.spin_gr = dodaj(w, 'Grubość:', QDoubleSpinBox(), 'Grubość linii wektorów.')
        self.spin_gr.setRange(.05, 3); self.spin_gr.setSingleStep(.05); self.spin_gr.setValue(.3)
        w = self.panel_opcji.wiersz('Wektory')
        self.spin_r0 = dodaj(w, 'Okrąg zera:', QDoubleSpinBox(), 'Promień szarego okręgu oznaczającego wartość 0, w % połowy patcha.\n'
                             'Wektor krótszy od okręgu = wartość ujemna.')
        self.spin_r0.setRange(0, 90); self.spin_r0.setDecimals(0); self.spin_r0.setValue(25); self.spin_r0.setSuffix(' %')
        self.spin_skala = dodaj(w, 'Skala długości:', QDoubleSpinBox(), 'Mnożnik długości wektorów. 100 % = wartość większa od 99,5 % wszystkich na rysunku\n'
                                'sięga brzegu patcha (nieliczne dłuższe wchodzą na sąsiednie patche);\n'
                                'więcej = dłuższe wektory. Wartość brzegu patcha jest w tytule wykresu.')
        self.spin_skala.setRange(5, 2000); self.spin_skala.setDecimals(0); self.spin_skala.setValue(100); self.spin_skala.setSuffix(' %')

        # Obraz pod okręgami
        w = self.panel_opcji.wiersz('Obraz')
        self.chk_obraz = QCheckBox('Obraz pod spodem'); self.chk_obraz.setChecked(True)
        self.chk_obraz.setToolTip('Zdjęcie 480x320 (wejście ViT) pod okręgami.')
        w.addWidget(self.chk_obraz)
        self.spin_alfa = dodaj(w, 'Krycie:', QDoubleSpinBox(), 'Krycie obrazu: 0 = obraz niewidoczny, 1 = pełny.')
        self.spin_alfa.setRange(0, 1); self.spin_alfa.setSingleStep(.1); self.spin_alfa.setValue(.55)

        # Atencja: włącznik w pierwszym wierszu wspólnego panelu opcji atencji
        self.chk_atencja = QCheckBox('Atencja na obrazie')
        self.chk_atencja.setToolTip('Mapa cieplna atencji ViT na siatce 20x30 i strzałki źródło → cel (opcje obok).\n'
                                    'Atencja liczy się z obrazu; auto-szukanie zmienia latent za atencją, więc jej nie zmienia.')
        self.opcje = OpcjeAtencji(na_poczatku=(self.chk_atencja,))
        self.panel_opcji.strona('Atencja').addWidget(self.opcje)

        # stan (wybrany patch) w pasku obok przycisku „Opcje”: krótki, więc mieści się w jednej linii
        self.etykieta = etykieta_statusu(QLabel(''))
        self.etykieta.setWordWrap(False)
        self.etykieta.setMinimumHeight(self.panel_opcji.btn_opcje.sizeHint().height())
        self.panel_opcji.pasek.addWidget(self.etykieta)
        self.panel_opcji.zakoncz()

        self.fig = Figure(figsize=(12, 8.4))
        self.panel = PanelWykresu(self.fig, 900, self._osobne)
        lay.addWidget(self.panel, 1)
        self.panel.canvas.mpl_connect('button_press_event', self._klik)
        for k in (self.combo_tryb.currentIndexChanged, self.combo_kan.currentIndexChanged, self.spin_r0.valueChanged,
                  self.spin_skala.valueChanged, self.spin_gr.valueChanged, self.chk_obraz.toggled,
                  self.spin_alfa.valueChanged, self.chk_atencja.toggled, self.opcje.zmiana):
            k.connect(lambda *_: self._rysuj())

    # ------------------------------------------------------------ dane
    def pokaz(self, przed, po=None, rgb480=None, atencja_fn=None):
        self.przed, self.po, self.rgb = przed, po, rgb480
        self.atencja_fn = atencja_fn
        self.opcje.ustaw_zrodlo(atencja_fn)
        if po is None and self.combo_tryb.currentIndex() != 0:
            self.combo_tryb.blockSignals(True); self.combo_tryb.setCurrentIndex(0); self.combo_tryb.blockSignals(False)
        self._rysuj()

    def _osobne(self):
        if self.przed is None:
            return

        def nowy():
            w = WidokCzasteczek(self.j128, self._Panel, self._osobne_okno)
            w.opcje.kopiuj_z(self.opcje)
            for a, b in ((w.combo_tryb, self.combo_tryb), (w.combo_kan, self.combo_kan)):
                a.setCurrentIndex(b.currentIndex())
            for a, b in ((w.spin_r0, self.spin_r0), (w.spin_skala, self.spin_skala), (w.spin_gr, self.spin_gr),
                         (w.spin_alfa, self.spin_alfa)):
                a.setValue(b.value())
            w.chk_obraz.setChecked(self.chk_obraz.isChecked()); w.chk_atencja.setChecked(self.chk_atencja.isChecked())
            w.token = self.token
            w.panel.wys_baz = 1300
            w.pokaz(self.przed, self.po, self.rgb, self.atencja_fn)
            from wygoda import ulatw
            ulatw(w)
            return w
        self._osobne_okno(self, 'Cząsteczki: okręgi 768 wektorów na patchach', nowy)

    def _klik(self, e):
        if e.inaxes is None or e.xdata is None or self.panel.pasek.mode:
            return
        k = int(np.clip(e.xdata // KOMORKA, 0, KOLUMNY - 1)); r = int(np.clip(e.ydata // KOMORKA, 0, WIERSZE - 1))
        self.token = r * KOLUMNY + k
        if self.chk_atencja.isChecked() and self.opcje.potrzebuje_tokenu():
            self._rysuj()
        else:
            self.etykieta.setText(f'wybrany patch: token {self.token} (wiersz {r}, kolumna {k})')

    # ------------------------------------------------------------ rysowanie
    def _kanaly(self):
        i = self.combo_kan.currentIndex()
        return self.j128 if i == 4 else np.arange(0, 768, (1, 2, 4, 8)[i])

    def _segmenty(self, L, kan, r0, skala):
        """L: [600, 768] wartości; zwraca segmenty [600*K, 2, 2] od środka patcha wzdłuż kąta kanału."""
        theta = 2 * np.pi * kan / 768
        kier = np.stack([np.cos(theta), -np.sin(theta)], 1)                 # oś y obrazu w dół
        t = np.arange(600)
        srodki = np.stack([(t % KOLUMNY) * KOMORKA + KOMORKA / 2, (t // KOLUMNY) * KOMORKA + KOMORKA / 2], 1)
        dl = np.clip(r0 + skala * L[:, kan], 0, None)                        # [600, K]
        konce = srodki[:, None, :] + dl[..., None] * kier[None]
        poczatki = np.broadcast_to(srodki[:, None, :], konce.shape)
        return np.stack([poczatki, konce], 2).reshape(-1, 2, 2)

    def _rysuj(self):
        if self.przed is None:
            return
        self.fig.clear()
        ax = self.fig.add_axes([0.01, 0.06, 0.88, 0.9])
        kan = self._kanaly()
        tryb = self.combo_tryb.currentIndex()
        if self.po is None:
            tryb = 0
        if tryb == 0:
            L = self.przed
        elif tryb == 1:
            L = self.po
        elif tryb == 2:
            L = self.po - self.przed
        else:
            L = self.po
        polowa = KOMORKA / 2
        r0 = polowa * self.spin_r0.value() / 100
        zakres = [self.przed[:, kan]] + ([self.po[:, kan]] if self.po is not None and tryb in (1, 3) else [])
        if tryb == 2:
            zakres = [L[:, kan]]
        # brzeg patcha = 99,5. percentyl |wartości| (nie maksimum), żeby typowe wektory były widoczne;
        # pojedyncze bardzo długie mogą wejść na sąsiedni patch
        m = max(float(np.percentile(np.abs(np.concatenate(zakres)), 99.5)), 1e-6)
        skala = (polowa - r0) / m * self.spin_skala.value() / 100 if polowa > r0 else polowa / m

        if self.rgb is not None and self.chk_obraz.isChecked():
            ax.imshow(self.rgb, extent=(0, 480, 320, 0), alpha=self.spin_alfa.value(), zorder=0)
        tekst_at = ''
        if self.chk_atencja.isChecked():
            A, tekst = self.opcje.mapa(self.token)
            if A is not None:
                ax.imshow(A, extent=(0, 480, 320, 0), cmap='inferno', alpha=self.opcje.krycie(),
                          interpolation='nearest', zorder=1, vmin=0, vmax=1)
            tekst_at = f' | {tekst}' if tekst else ''
            strz = self.opcje.strzalki(self.token)
            if strz is not None:
                src, dst, wg = strz
                t = np.arange(600)
                c = np.stack([(t % KOLUMNY) * KOMORKA + KOMORKA / 2, (t // KOLUMNY) * KOMORKA + KOMORKA / 2], 1)
                d = c[dst] - c[src]
                klasa = np.minimum((wg * 5).astype(int), 4)          # grubość według siły atencji (5 klas)
                g_max = self.opcje.spin_grubosc.value()
                for k in range(5):
                    m = klasa == k
                    if m.any():
                        sz = (.0018 if len(src) > 100 else .0028) * (1 + (g_max - 1) * k / 4) / 2
                        ax.quiver(c[src[m], 0], c[src[m], 1], d[m, 0], d[m, 1], wg[m], cmap='cool', clim=(0, 1),
                                  angles='xy', scale_units='xy', scale=1, width=sz, headwidth=4, headlength=5,
                                  alpha=.9, zorder=7)
                ax.scatter(c[dst, 0], c[dst, 1], s=8 + 40 * wg, facecolors='none', edgecolors='w', lw=.6, zorder=7)
                tekst_at += f' | strzałki: {self.opcje.combo_strz.currentText()} ({len(src)})'
            elif self.opcje.combo_strz.currentIndex() in (1, 2) and self.token is None:
                tekst_at += ' | strzałki: kliknij patch'
        for x in range(0, 481, KOMORKA):
            ax.plot([x, x], [0, 320], color='w', lw=.25, alpha=.5, zorder=2)
        for y in range(0, 321, KOMORKA):
            ax.plot([0, 480], [y, y], color='w', lw=.25, alpha=.5, zorder=2)
        th = np.linspace(0, 2 * np.pi, 33)
        t = np.arange(600)
        cx, cy = (t % KOLUMNY) * KOMORKA + polowa, (t // KOLUMNY) * KOMORKA + polowa
        okregi = np.stack([cx[:, None] + r0 * np.cos(th), cy[:, None] + r0 * np.sin(th)], -1)
        ax.add_collection(LineCollection(okregi, colors=(.6, .6, .6, .8), linewidths=.3, zorder=3))
        kolory = kolory_kanalow()[kan]
        if tryb == 3:
            ax.add_collection(LineCollection(self._segmenty(self.przed, kan, r0, skala), colors=(.55, .55, .55, .6),
                                             linewidths=self.spin_gr.value(), zorder=4))
        ax.add_collection(LineCollection(self._segmenty(L, kan, r0, skala), colors=np.tile(kolory, (600, 1)),
                                         linewidths=self.spin_gr.value(), zorder=5))
        if self.token is not None:
            r, k = divmod(self.token, KOLUMNY)
            ax.add_patch(Rectangle((k * KOMORKA, r * KOMORKA), KOMORKA, KOMORKA, fill=False, ec='w', lw=1.2, zorder=6))
        ax.set_xlim(0, 480); ax.set_ylim(320, 0); ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, 481, 48), [str(i) for i in range(0, 31, 3)], fontsize=7)
        ax.set_yticks(np.arange(0, 321, 32), [str(i) for i in range(0, 21, 2)], fontsize=7)
        ax.set_xlabel('kolumna patcha', fontsize=8); ax.set_ylabel('wiersz patcha', fontsize=8)
        ax.set_title(f'Cząsteczki: {self.TRYBY[tryb]}, {len(kan)} wektorów na okrąg; okrąg = 0, '
                     f'brzeg patcha = {m * self.spin_skala.value() / 100:.3g}{tekst_at}', fontsize=9)
        # koło barw kanałów
        kolo = self.fig.add_axes([0.9, 0.55, 0.09, 0.3], projection='polar')
        kk = np.arange(768)
        kolo.bar(2 * np.pi * kk / 768, np.ones(768), width=2 * np.pi / 768, color=kolory_kanalow(), bottom=.4)
        if self.combo_kan.currentIndex() == 4:
            kolo.bar(2 * np.pi * self.j128 / 768, np.full(len(self.j128), .5), width=2 * np.pi / 768, color='k', bottom=1.45)
        kolo.set_yticks([]); kolo.set_xticks(2 * np.pi * np.array([0, 192, 384, 576]) / 768, ['0', '192', '384', '576'], fontsize=7)
        kolo.set_title('kolor i kąt kanału', fontsize=8)
        self.etykieta.setText('' if self.token is None else f'wybrany patch: token {self.token}')
        self.panel.canvas.draw_idle()
