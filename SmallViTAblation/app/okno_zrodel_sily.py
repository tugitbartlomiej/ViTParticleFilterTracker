"""Okno „Źródła siły”: skąd pochodzi wektor (siła) c patcha i. Sygnał płynie od obrazu przez łodygę, kolejne bloki
i głowy atencji aż do neuronu c ukrytej warstwy MLP ostatniego bloku (rozkład ścisły, przeplyw_sily.py).

Karty:
- „Przepływ”: obraz z mapą sygnału na kolejnych głębokościach (płynna animacja), strzałki przeniesień głów atencji
  z kropkami płynącymi od patcha źródłowego do docelowego, wodospad wkładów (obraz, położenie, MLP bloków, stałe);
- „Warstwy × głowy”: siatka map 20 x 30: sygnał na wejściu bloku, źródła każdej głowy, wstrzyknięcie MLP;
- „Graf przepływu”: rzeka sygnału przez bloki z głowami i MLP jako węzłami (kolor i liczba = wkład);
- „Wektor patcha”: 768 wartości latentu patcha; klik wybiera wektor.
"""
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
                             QTabWidget, QVBoxLayout, QWidget)
from matplotlib.figure import Figure

from styl import rola
from uklad import Przeplyw, dodaj
from zakladka_detekcji import PlotnoCiemne

KOL_GLOW = ['#ff7f0e', '#2ca02c', '#1f9bff']                 # głowy 0, 1, 2
SRODKI = np.stack(np.divmod(np.arange(600), 30), 1)[:, ::-1] * 16 + 8      # środki patchy (x, y) w pikselach


def _etapy(r):
    """Kolejne głębokości: (nazwa, mapa [600] albo [320, 480], blok przeniesień albo None, opis)."""
    e = [('obraz: piksele (Integrated Gradients przez łodygę)', r['ig'], None,
          'Które piksele obrazu tworzą tę siłę (wkład łodygi; czerwone dodają, niebieskie odejmują).'),
         ('łodyga + położenie: osadzenie tokenów', r['S'][0], None,
          'Sygnał w osadzeniu każdego patcha: obraz po łodydze plus położenie sin-cos.')]
    for l in range(6):
        e.append((f'blok {l}: po atencji', r['Sm'][l], l,
                  f'Głowy atencji bloku {l} przenoszą sygnał między patchami (strzałki: skąd -> dokąd, kolor = głowa).'))
        if l < 5:
            e.append((f'blok {l}: po MLP', r['S'][l + 1], None,
                      f'MLP bloku {l} wstrzykuje nowy sygnał (w sumie {r["wklad_mlp"][l]:+.3f}); atencja go tylko przenosi.'))
    m = np.zeros(600, np.float32); m[r['tok']] = r['przed']
    e.append((f'wektor {r["kan"]} patcha {r["tok"]}', m, None,
              f'Przed GELU {r["przed"]:+.3f}, po GELU (wartość latentu) {r["wartosc"]:+.3f}.'))
    return e


def _piksele(m):
    m = np.asarray(m, np.float32)
    return m if m.ndim == 2 else np.kron(m.reshape(20, 30), np.ones((16, 16), np.float32))


class OknoZrodelSily(QWidget):
    """dostawca() -> dict(wagi=ścieżka checkpointu, rgb=obraz 480 x 320, tok=patch albo None, opis=tekst kroku)."""

    def __init__(self, dostawca, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle('Źródła siły: skąd pochodzi wektor patcha (przepływ sygnału w ViT)')
        self.resize(1500, 950)
        self.dostawca = dostawca
        self.r = None; self.etapy = []; self._det = (None, None); self.watek = None; self._zlecone = None
        lay = QVBoxLayout(self)
        w = Przeplyw(); lay.addLayout(w)
        self.chk_sledz = QCheckBox('śledź zakładkę'); self.chk_sledz.setChecked(True)
        self.chk_sledz.setToolTip('Patch i krok uczenia (wagi) z zakładki: zmiana kroku albo wybranego patcha liczy od nowa.')
        w.addWidget(self.chk_sledz)
        self.spin_tok = QSpinBox(); self.spin_tok.setRange(0, 599); self.spin_tok.setValue(255)
        dodaj(w, 'patch', self.spin_tok, 'Numer patcha (wiersz · 30 + kolumna). Klik w obraz na karcie „Przepływ” też wybiera.')
        self.combo_wyb = QComboBox(); self.combo_wyb.addItems(['najsilniejszy wektor patcha', 'ręcznie'])
        dodaj(w, 'wektor', self.combo_wyb, 'Który wektor (neuron 0..767) rozkładać: największa |wartość| w patchu albo ręcznie.')
        self.spin_kan = QSpinBox(); self.spin_kan.setRange(0, 767)
        dodaj(w, 'nr', self.spin_kan, 'Numer wektora (kanału) latentu. Klik na karcie „Wektor patcha” też wybiera.')
        self.btn = rola(QPushButton('Licz'), 'glowny'); w.addWidget(self.btn)
        self.btn.setToolTip('Liczy rozkład wybranego wektora wybranego patcha na źródła, przy wagach bieżącego kroku zakładki Uczenie.\n'
                            'Wynik pojawia się na kartach poniżej.')
        self.etykieta = QLabel(''); self.etykieta.setWordWrap(True); lay.addWidget(self.etykieta)

        self.zakl = QTabWidget(); lay.addWidget(self.zakl, 1)
        # --- Przepływ
        s = QWidget(); ls = QVBoxLayout(s); ls.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(12, 7)); self.canvas = PlotnoCiemne(self.fig); ls.addWidget(self.canvas, 1)
        w2 = QHBoxLayout(); ls.addLayout(w2)
        self.btn_graj = rola(QPushButton('▶ Płyń'), 'glowny'); self.btn_graj.setCheckable(True); w2.addWidget(self.btn_graj)
        self.btn_graj.setToolTip('Animuje przepływ sygnału przez kolejne etapy sieci: obraz, łodyga, 6 bloków, wektor wynikowy.\n'
                                 'Szybkość ustawia pole „etapów/s”; ponowny klik zatrzymuje.')
        self.suwak = QSlider(Qt.Orientation.Horizontal); self.suwak.setRange(0, 1000); w2.addWidget(self.suwak, 1)
        self.suwak.setToolTip('Pozycja w przepływie przez etapy sieci (od obrazu po wektor wynikowy). Przeciągnij, żeby przejść ręcznie.')
        self.spin_tempo = QDoubleSpinBox(); self.spin_tempo.setRange(.1, 10); self.spin_tempo.setValue(1)
        dodaj(w2, 'etapów/s', self.spin_tempo, 'Szybkość płynięcia przez głębokości sieci.')
        self.spin_strz = QSpinBox(); self.spin_strz.setRange(0, 300); self.spin_strz.setValue(40)
        dodaj(w2, 'strzałek', self.spin_strz, 'Ile najsilniejszych przeniesień głów (patch -> inny patch) rysować w etapie atencji.')
        self.combo_skala = QComboBox(); self.combo_skala.addItems(['skala na etap', 'wspólna skala'])
        dodaj(w2, '', self.combo_skala, 'Kolor mapy: osobna skala w każdym etapie (widać kształt) albo wspólna (widać wzrost sygnału).')
        self.zakl.addTab(s, 'Przepływ')
        # --- siatka, graf, wektor
        self.fig_s = Figure(figsize=(12, 13)); self.canvas_s = PlotnoCiemne(self.fig_s)
        from PyQt6.QtWidgets import QScrollArea
        sc = QScrollArea(); self.canvas_s.setMinimumHeight(1300); sc.setWidget(self.canvas_s); sc.setWidgetResizable(True)
        self.zakl.addTab(sc, 'Warstwy × głowy')
        self.fig_g = Figure(figsize=(12, 7)); self.canvas_g = PlotnoCiemne(self.fig_g); self.zakl.addTab(self.canvas_g, 'Graf przepływu')
        self.fig_w = Figure(figsize=(12, 7)); self.canvas_w = PlotnoCiemne(self.fig_w); self.zakl.addTab(self.canvas_w, 'Wektor patcha')

        self.timer = QTimer(self); self.timer.setInterval(30); self.timer.timeout.connect(self._klatka)
        self._odlicz = QTimer(self); self._odlicz.setSingleShot(True); self._odlicz.setInterval(350)
        self._odlicz.timeout.connect(self.licz)
        self.btn.clicked.connect(self.licz)
        self.btn_graj.toggled.connect(self._graj)
        self.suwak.valueChanged.connect(lambda *_: self._rysuj_przeplyw())
        for sg in (self.spin_strz.valueChanged, self.combo_skala.currentIndexChanged):
            sg.connect(lambda *_: self._rysuj_przeplyw(nowy=True))
        self.spin_tok.valueChanged.connect(lambda *_: self._odlicz.start())
        self.spin_kan.valueChanged.connect(lambda *_: self._odlicz.start() if self.combo_wyb.currentIndex() == 1 else None)
        self.combo_wyb.currentIndexChanged.connect(lambda *_: self._odlicz.start())
        self.canvas.mpl_connect('button_press_event', self._klik_obrazu)
        self.canvas_w.mpl_connect('button_press_event', self._klik_wektora)

    # ------------------------------------------------------------ liczenie
    def przy_zmianie(self):
        """Wołane przez zakładkę po zmianie kroku albo patcha."""
        if self.isVisible() and self.chk_sledz.isChecked():
            d = self.dostawca()
            if d is None:
                return
            klucz = (str(d['wagi']), d.get('tok'), id(d['rgb']))
            if klucz == getattr(self, '_klucz_ost', None):             # te same wagi i patch: nic nowego
                return
            self._klucz_ost = klucz
            if d.get('tok') is not None and d['tok'] != self.spin_tok.value():
                self.spin_tok.blockSignals(True); self.spin_tok.setValue(int(d['tok'])); self.spin_tok.blockSignals(False)
            self._odlicz.start()

    def licz(self):
        d = self.dostawca()
        if d is None:
            self.etykieta.setText('Wczytaj historię uczenia w zakładce „Uczenie ViT”.'); return
        tok = self.spin_tok.value()
        kan = None if self.combo_wyb.currentIndex() == 0 else self.spin_kan.value()
        if self.watek is not None:
            self._zlecone = True; return
        from zakladka_filmu import WatekFilmu
        rgb, sciezka = d['rgb'], d['wagi']
        pam = self._det

        def zadanie(wt):
            import detekcja as D
            import przeplyw_sily as PS
            det = pam[1] if pam[0] == str(sciezka) else D.Detektor(sciezka)
            k = kan
            if k is None:
                z = det.analizuj(rgb)['z']
                z = (z[0] if z.ndim == 3 else z).cpu().numpy()
                k = int(np.abs(z[tok]).argmax())
            r = PS.licz(det.net, rgb, tok, k)
            return det, r

        def gotowe(w):
            self.watek = None
            det, r = w
            self._det = (str(sciezka), det)
            r['opis'] = d.get('opis', ''); r['rgb'] = rgb
            self.pokaz(r)
            if self._zlecone:
                self._zlecone = None; self.licz()

        self.etykieta.setText(f'Liczę rozkład siły patcha {tok}...')
        self.watek = WatekFilmu(zadanie, self)
        self.watek.gotowy.connect(gotowe)
        self.watek.blad.connect(lambda t: (setattr(self, 'watek', None), self.etykieta.setText('Błąd: ' + t[-500:])))
        self.watek.start()

    def pokaz(self, r):
        self.r = r
        self.etapy = _etapy(r)
        self.spin_kan.blockSignals(True); self.spin_kan.setValue(r['kan']); self.spin_kan.blockSignals(False)
        suma = r['wklad_obraz'] + r['wklad_poz'] + r['wklad_mlp'].sum() + r['stale']
        self.etykieta.setText(
            f'{r["opis"]}; patch {r["tok"]}, wektor {r["kan"]}: przed GELU {r["przed"]:+.4f} = obraz {r["wklad_obraz"]:+.3f} + '
            f'położenie {r["wklad_poz"]:+.3f} + MLP bloków 0-4 {" ".join(f"{v:+.3f}" for v in r["wklad_mlp"])} + stałe '
            f'{r["stale"]:+.3f} (suma {suma:+.4f}; kontrola stałych {r["stale_kontrola"]:+.4f}); po GELU {r["wartosc"]:+.4f}. '
            f'Integrated Gradients łodygi: suma pikseli {r["ig_suma"]:+.3f} wobec celu {r["ig_cel"]:+.3f}.')
        self._rysuj_przeplyw(nowy=True); self._rysuj_siatke(); self._rysuj_graf(); self._rysuj_wektor()

    # ------------------------------------------------------------ karta „Przepływ”
    def _graj(self, wl):
        self.btn_graj.setText('⏸ Stop' if wl else '▶ Płyń')
        if wl:
            if self.suwak.value() >= 1000:
                self.suwak.setValue(0)
            import time
            self._start = (time.perf_counter(), self.suwak.value())
            self.timer.start()
        else:
            self.timer.stop()

    def _klatka(self):
        if not self.etapy:
            return
        import time
        t0, v0 = self._start                                              # pozycja wg zegara: tempo niezależne od rysowania
        v = int(v0 + (time.perf_counter() - t0) * self.spin_tempo.value() * 1000 / (len(self.etapy) - 1))
        if v >= 1000:
            v = 1000; self.btn_graj.setChecked(False)
        self.suwak.setValue(v)

    def _t(self):
        return self.suwak.value() / 1000 * (len(self.etapy) - 1)

    def _rysuj_przeplyw(self, nowy=False):
        r = self.r
        if r is None:
            return
        f = self.fig
        t = self._t(); a = int(np.floor(t)); b = min(a + 1, len(self.etapy) - 1); u = t - a
        ma, mb = _piksele(self.etapy[a][1]), _piksele(self.etapy[b][1])
        mapa = (1 - u) * ma + u * mb
        if nowy or not hasattr(self, '_ax'):
            f.clear()
            self._ax = f.add_axes([.01, .08, .6, .84]); self._axw = f.add_axes([.68, .12, .3, .78])
            self._ax.imshow(r['rgb'], extent=(0, 480, 320, 0))
            self._im = self._ax.imshow(np.zeros((320, 480, 4)), extent=(0, 480, 320, 0), interpolation='bilinear')
            rr, cc = divmod(r['tok'], 30)
            from matplotlib.patches import Rectangle
            self._ax.add_patch(Rectangle((cc * 16, rr * 16), 16, 16, fill=False, ec='white', lw=1.8))
            self._ax.set_xlim(0, 480); self._ax.set_ylim(320, 0); self._ax.axis('off')
            from matplotlib.collections import LineCollection
            self._lc = LineCollection([], linewidths=1); self._ax.add_collection(self._lc)
            self._kropki = self._ax.scatter([], [], s=14, c=[], zorder=6)
            self._tytul = self._ax.set_title('', fontsize=10, color='white')
            self._opis = f.text(.01, .02, '', fontsize=9, color='#d0d0d0')
            self._wodospad()
            if self.combo_skala.currentIndex() == 1:
                self._vmax = max(float(np.abs(_piksele(e[1])).max()) for e in self.etapy[1:])
        vm = getattr(self, '_vmax', None) if self.combo_skala.currentIndex() == 1 else None
        if a == 0 and vm is not None and u < 1:          # piksele mają inną skalę niż tokeny: osobno
            vm = None
        vm = vm or max(float(np.abs(mapa).max()), 1e-9)
        import matplotlib
        rgba = matplotlib.colormaps['RdBu_r'](np.clip(.5 + .5 * mapa / vm, 0, 1))
        rgba[..., 3] = np.clip(np.abs(mapa) / vm, 0, 1) ** .55 * .9      # słaby sygnał prawie przezroczysty
        self._im.set_data(rgba)
        nazwa = self.etapy[a][0] if u < .5 else self.etapy[b][0]
        self._tytul.set_text(f'{nazwa}   (sygnał w tym etapie: suma {float(np.sum(self.etapy[a if u < .5 else b][1])):+.3f})'
                             if (a if u < .5 else b) else nazwa)
        self._opis.set_text(self.etapy[b if u >= .5 else a][3])
        # przeniesienia głów w etapie atencji (przejście do etapu „po atencji”)
        l = self.etapy[b][2]
        segs, kol, pkt, pk = [], [], [], []
        if l is not None and 0 < u and self.spin_strz.value():
            Fl = r['F'][l]                                                 # [H, k, j]
            Fs = Fl.sum(0).copy(); np.fill_diagonal(Fs, 0)
            n = self.spin_strz.value()
            idx = np.argsort(-np.abs(Fs), axis=None)[:n]
            ks, js = np.unravel_index(idx, Fs.shape)
            mx = max(float(np.abs(Fs[ks, js]).max()), 1e-12)
            from matplotlib.colors import to_rgb
            for k, j in zip(ks, js):
                wart = Fs[k, j]
                h = int(np.argmax(np.abs(Fl[:, k, j])))
                c = to_rgb(KOL_GLOW[h]); al = .25 + .75 * abs(wart) / mx
                p0, p1 = SRODKI[j], SRODKI[k]
                segs.append([p0, p1]); kol.append((*c, al * (1 if wart > 0 else .6)))
                pkt.append(p0 + (p1 - p0) * u); pk.append((*c, 1))               # kropka płynie z j do k
            self._lc.set_linewidths([.6 + 3.4 * abs(Fs[k, j]) / mx for k, j in zip(ks, js)])
            self._lc.set_linestyles(['solid' if Fs[k, j] > 0 else 'dashed' for k, j in zip(ks, js)])
        self._lc.set_segments(segs); self._lc.set_colors(kol if kol else 'none')
        self._kropki.set_offsets(np.array(pkt) if pkt else np.zeros((0, 2)))
        if pk:
            self._kropki.set_color(pk)
        self._podswietl_wodospad(t)
        self.canvas.draw_idle()

    def _wodospad(self):
        r = self.r; ax = self._axw
        ax.clear()
        nazwy = ['obraz (łodyga)', 'położenie'] + [f'MLP bloku {l}' for l in range(5)] + ['stałe (biasy)']
        wart = [r['wklad_obraz'], r['wklad_poz']] + list(r['wklad_mlp']) + [r['stale']]
        dol = np.concatenate([[0], np.cumsum(wart)[:-1]])
        self._slupki = ax.barh(range(len(wart)), wart, left=dol, color=['#e45756' if v > 0 else '#4c78a8' for v in wart],
                               edgecolor='white', linewidth=.5)
        for i, v in enumerate(wart):
            ax.text(dol[i] + v + (.01 if v >= 0 else -.01) * max(1, abs(r['przed'])), i, f'{v:+.3f}', va='center',
                    ha='left' if v >= 0 else 'right', fontsize=8, color='white')
        ax.axvline(r['przed'], color='#ffd030', lw=1.2, ls='--')
        ax.text(r['przed'], len(wart) - .3, f' przed GELU {r["przed"]:+.3f}\n po GELU {r["wartosc"]:+.3f}', color='#ffd030',
                fontsize=8, va='bottom')
        kon = np.cumsum(wart); lo, hi = min(0., float(kon.min()), float(dol.min())), max(0., float(kon.max()), float(dol.max()))
        ax.set_xlim(lo - .08 * (hi - lo) - 1e-6, hi + .3 * (hi - lo) + 1e-6)
        ax.set_yticks(range(len(wart))); ax.set_yticklabels(nazwy, fontsize=8); ax.invert_yaxis()
        ax.set_title(f'skąd jest wektor {r["kan"]} patcha {r["tok"]}: wkłady (suma = przed GELU)', fontsize=9)
        ax.tick_params(labelsize=7)
        # etap, od którego wkład jest już w sygnale: obraz i położenie od 1, MLP l od etapu „blok l: po MLP”
        self._etap_wkladu = [1, 1] + [3 + 2 * l for l in range(5)] + [len(self.etapy) - 1]

    def _podswietl_wodospad(self, t):
        for s, e in zip(self._slupki, self._etap_wkladu):
            s.set_alpha(1 if t >= e - .5 else .25)

    def _klik_obrazu(self, e):
        if e.inaxes is getattr(self, '_ax', None) and e.xdata is not None:
            tok = int(np.clip(e.ydata // 16, 0, 19)) * 30 + int(np.clip(e.xdata // 16, 0, 29))
            self.chk_sledz.setChecked(False)
            self.spin_tok.setValue(tok)

    # ------------------------------------------------------------ karta „Warstwy × głowy”
    def _rysuj_siatke(self):
        r = self.r; f = self.fig_s; f.clear()
        osie = f.subplots(7, 5)
        f.subplots_adjust(left=.07, right=.99, top=.96, bottom=.01, hspace=.35, wspace=.06)
        rr, cc = divmod(r['tok'], 30)

        def mapa(ax, m, tytul):
            m = np.asarray(m).reshape(20, 30); v = max(float(np.abs(m).max()), 1e-9)
            ax.imshow(m, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
            ax.plot(cc, rr, 's', mfc='none', mec='#ffd030', ms=5, mew=1)
            ax.set_title(tytul, fontsize=7); ax.set_xticks([]); ax.set_yticks([])

        for ax in osie[0]:
            ax.axis('off')
        ax = osie[0][0]; ax.axis('on')
        v = max(float(np.abs(r['ig']).max()), 1e-9)
        ax.imshow(r['rgb']); ax.imshow(r['ig'], cmap='RdBu_r', vmin=-v, vmax=v, alpha=.65)
        ax.set_title(f'obraz: piksele (IG) {r["ig_suma"]:+.3f}', fontsize=7); ax.set_xticks([]); ax.set_yticks([])
        mapa(osie[0][1], r['wklad_obraz_tokeny'], f'obraz po łodydze {r["wklad_obraz"]:+.3f}'); osie[0][1].axis('on')
        mapa(osie[0][2], r['wklad_poz_tokeny'], f'położenie {r["wklad_poz"]:+.3f}'); osie[0][2].axis('on')
        for l in range(6):
            w = osie[l + 1]
            mapa(w[0], r['S'][l], f'blok {l}: sygnał na wejściu {r["S"][l].sum():+.3f}')
            for h in range(3):
                zr = r['F'][l][h].sum(0)                                   # skąd głowa h bierze (po patchach źródłowych)
                mapa(w[1 + h], zr, f'głowa {h}: źródła {zr.sum():+.3f}')
                w[1 + h].title.set_color(KOL_GLOW[h])
            if l < 5:
                mapa(w[4], r['M'][l], f'MLP {l}: nowy sygnał {r["M"][l].sum():+.3f}')
            else:
                m = np.zeros(600); m[r['tok']] = r['przed']
                mapa(w[4], m, f'wektor {r["kan"]}: {r["przed"]:+.3f}')
        f.suptitle(f'Wektor {r["kan"]} patcha {r["tok"]} (żółty kwadrat): skąd przychodzi sygnał w każdym bloku i głowie '
                   f'(czerwone dodaje, niebieskie odejmuje; skala osobna w każdej mapie)', fontsize=9, color='white')
        self.canvas_s.draw_idle()

    # ------------------------------------------------------------ karta „Graf przepływu”
    def _rysuj_graf(self):
        r = self.r; f = self.fig_g; f.clear(); ax = f.add_axes([.02, .05, .96, .88]); ax.axis('off')
        import matplotlib
        from matplotlib.patches import FancyBboxPatch
        cm = matplotlib.colormaps['RdBu_r']
        wart = [r['F'][l][h].sum() for l in range(6) for h in range(3)] + list(r['M'].sum(1)) + [r['wklad_obraz'], r['wklad_poz']]
        vm = max(abs(float(v)) for v in wart) or 1
        S = [r['S'][l].sum() for l in range(6)] + [r['przed']]
        smax = max(abs(float(v)) for v in S) or 1
        xs = np.arange(8) * 1.6
        # rzeka sygnału: grubość = |sygnał| na wejściu bloku
        for i in range(7):
            ax.plot([xs[i] + .55, xs[i + 1] - .55], [3.4, 3.4], color='#ffd030', lw=2 + 16 * abs(S[i]) / smax, alpha=.55,
                    solid_capstyle='butt')
            ax.text((xs[i] + xs[i + 1]) / 2, 3.95, f'{S[i]:+.2f}', ha='center', fontsize=7, color='#ffd030')

        def wezel(x, y, tekst, v, kolor_ramki='white'):
            c = cm(.5 + .5 * float(v) / vm)
            ax.add_patch(FancyBboxPatch((x - .5, y - .28), 1, .56, boxstyle='round,pad=.02', fc=c, ec=kolor_ramki, lw=1.2))
            ax.text(x, y, f'{tekst}\n{float(v):+.3f}', ha='center', va='center', fontsize=7,
                    color='#111111' if abs(float(v)) / vm < .6 else 'white')

        wezel(xs[0], 3.4, 'obraz + położenie', r['wklad_obraz'] + r['wklad_poz'])
        for l in range(6):
            x = xs[l + 1]
            ax.text(x, 4.5, f'blok {l}', ha='center', fontsize=9, color='white')
            for h in range(3):
                wezel(x, 2.3 - .75 * h, f'głowa {h}', r['F'][l][h].sum(), KOL_GLOW[h])
                ax.annotate('', xy=(x, 3.1), xytext=(x, 2.3 - .75 * h + .28),
                            arrowprops=dict(arrowstyle='->', color=KOL_GLOW[h], lw=.8, alpha=.6))
            if l < 5:
                wezel(x + .8, 0.0, f'MLP {l}', r['M'][l].sum())
                ax.annotate('', xy=(x + .8, 3.1), xytext=(x + .8, .28), arrowprops=dict(arrowstyle='->', color='#e0e0e0', lw=1))
        wezel(xs[7], 3.4, f'wektor {r["kan"]}', r['przed'], '#ffd030')
        ax.text(xs[7], 2.7, f'po GELU {r["wartosc"]:+.3f}', ha='center', fontsize=8, color='#ffd030')
        ax.set_xlim(-.8, xs[-1] + .8); ax.set_ylim(-.6, 4.8)
        ax.set_title('Rzeka sygnału (żółta, grubość = sygnał na wejściu bloku): głowy atencji przenoszą go między patchami '
                     '(liczba = ile przeniosły), MLP dodają nowy', fontsize=9, color='white')
        self.canvas_g.draw_idle()

    # ------------------------------------------------------------ karta „Wektor patcha”
    def _rysuj_wektor(self):
        r = self.r; f = self.fig_w; f.clear(); ax = f.add_axes([.06, .12, .92, .8])
        z = r['latent_patcha']
        ax.bar(np.arange(768), z, width=1, color=['#e45756' if v > 0 else '#4c78a8' for v in z])
        top = np.argsort(-np.abs(z))[:10]
        for c in top:
            ax.text(c, z[c], f' {c}', fontsize=7, color='white', rotation=90, va='bottom' if z[c] >= 0 else 'top')
        ax.axvline(r['kan'], color='#ffd030', lw=1)
        ax.set_title(f'768 wektorów (sił) latentu patcha {r["tok"]}; żółta linia = rozkładany wektor {r["kan"]}. '
                     f'Klik wybiera wektor do rozkładu.', fontsize=9)
        ax.set_xlabel('numer wektora (neuronu ukrytej warstwy MLP ostatniego bloku)', fontsize=8); ax.tick_params(labelsize=7)
        self._ax_w = ax
        self.canvas_w.draw_idle()

    def _klik_wektora(self, e):
        if e.inaxes is getattr(self, '_ax_w', None) and e.xdata is not None:
            self.combo_wyb.blockSignals(True); self.combo_wyb.setCurrentIndex(1); self.combo_wyb.blockSignals(False)
            self.spin_kan.setValue(int(np.clip(round(e.xdata), 0, 767)))
            self._odlicz.start()
