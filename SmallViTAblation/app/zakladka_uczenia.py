"""Zakładka „Uczenie ViT: ruch cząsteczek po krokach”: oś czasu to krok uczenia, nie klatka filmu.

Ta sama klatka próbna, latent z każdego kroku optymalizatora (gęsta historia, analiza_uczenia), widoki
3D nad obrazem i przestrzeń makrowektorów (te same kierunki i kolory co w zakładce filmu) oraz wykresy
ruchu cząsteczek pod osią kroków: strata i mAP, prędkość tokenów w latencie i cząstek w przestrzeni makro,
odległość od stanu końcowego, rozdzielenie narzędzie / tło, efektywny rząd i mapa CKA krok x krok.
Detekcja i atencja pochodzą z najbliższego zapisanego checkpointu wag (co 10 kroków), co jest podpisane.

Baza PCA i środek układu liczone są raz z puli wszystkich kroków, a nie osobno w każdym kroku:
w jednym biegu kanał c w kroku t i t+1 to ten sam neuron, a PCA na krok dawałaby przeskoki osi,
które wyglądają jak ruch cząstek, a nim nie są (Vernier i in. 2020).
"""
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QProgressBar, QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget,
                             QVBoxLayout, QWidget)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import analiza_uczenia as AU
import przyczyny_ruchu as PR
from styl import rola
from uklad import Przeplyw, dodaj
from zakladka_detekcji import PlotnoCiemne
from zakladka_filmu import KOLOR_VIT, WatekFilmu, prostokat

KOL_N, KOL_T, KOL_P = '#ff5a4e', '#8fa3b8', '#ffffff'      # narzędzie, tło, wybrany patch


class ZakladkaUczenia(QWidget):
    def __init__(self, okno):
        super().__init__()
        self.okno = okno
        self.j128 = [w['kanal'] for w in okno.narzedzie]
        self.biegi = AU.znajdz_biegi()
        self.opis = self.Z = self.an = self.dety = None
        self.i_kl, self.t, self.token, self.watek = 0, 0, None, None
        self._ckpt_det = None
        lay = QVBoxLayout(self)

        # ---------------- góra: źródło historii
        g = QGroupBox('Historia uczenia (latent zapisany w każdym kroku optymalizatora)')
        w = Przeplyw(); g.setLayout(w); lay.addWidget(g)
        self.combo_bieg = QComboBox(); self.combo_bieg.setMinimumWidth(320)
        for b in self.biegi:
            K, N = b['ksztalt'][:2]
            self.combo_bieg.addItem(f"{b['nazwa']}  ({K} kroków, {N} klatek)")
        dodaj(w, 'Bieg:', self.combo_bieg, 'Biegi z plikiem LATENT_GESTY.npy [kroki, klatki, 600, 768] w experiments/*/wyniki/*/dane.\n'
              'Latent ostatniego bloku (MLP po GELU) w trybie eval, ten sam co w reszcie aplikacji.')
        self.combo_klatka = QComboBox(); self.combo_klatka.setMinimumWidth(300)
        dodaj(w, 'Klatka próbna:', self.combo_klatka, 'Stała klatka, na której zapisano latent w każdym kroku uczenia.\n'
              'W nawiasie: zbiór; obok liczba patchy narzędzia.')
        self.spin_co = QSpinBox(); self.spin_co.setRange(1, 50); self.spin_co.setValue(1)
        dodaj(w, 'Co ile kroków:', self.spin_co, 'Animacja i przestrzeń makro co tyle kroków (1 = każdy krok).\n'
              'Wykresy zawsze pokazują wszystkie kroki.')
        self.btn_wczytaj = rola(QPushButton('Wczytaj historię'), 'glowny')
        self.btn_wczytaj.setToolTip('Wczytuje latent klatki ze wszystkich kroków, liczy miary ruchu i detekcję z wag.\n'
                                    'Pierwszy raz około 30-50 s (kopia float16 trafia do app/cache_uczenia), potem kilka sekund.')
        w.addWidget(self.btn_wczytaj)
        self.pasek = QProgressBar(); self.pasek.setMaximumWidth(220); self.pasek.setTextVisible(False); w.addWidget(self.pasek)
        self.btn_przyczyny = rola(QPushButton('Licz przyczyny ruchu'), 'narzedzie')
        self.btn_przyczyny.setToolTip('Rozkłada ruch cząsteczek tej klatki w każdym przedziale między checkpointami wag\n'
                                      '(co 10 kroków) na przyczyny: grupy wag (łodyga, atencja i MLP każdego bloku),\n'
                                      'gradient straty względem latentu i atencję. Około 30 s. Wyniki w zakładce\n'
                                      '„Przyczyny ruchu” pod klatką i jako strzałki w przestrzeni makro.')
        w.addWidget(self.btn_przyczyny)
        self.etykieta_biegu = rola(QLabel('' if self.biegi else 'Brak biegów z gęstą historią w experiments/*/wyniki/*/dane.'),
                                   'opis')
        w.addWidget(self.etykieta_biegu)

        # ---------------- środek: klatka i wykresy | 3D | makro
        podzial = QSplitter(Qt.Orientation.Horizontal)
        lewa = QWidget(); ll = QVBoxLayout(lewa); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(4)
        self.fig = Figure(figsize=(5, 3.6)); self.ax = self.fig.add_axes([0, 0, 1, .9])
        self.canvas = PlotnoCiemne(self.fig); self.canvas.setFixedHeight(300)
        self.canvas.setToolTip('Klatka próbna: patche narzędzia (czerwone kropki), ramka ViT z wag najbliższego\n'
                               'checkpointu (zielona) i wybrany patch (biały). Klik wybiera patch.')
        ll.addWidget(self.canvas)
        self.zakl_wyk = QTabWidget(); ll.addWidget(self.zakl_wyk, 1)
        self.fig_r = Figure(figsize=(5, 7)); self.canvas_r = PlotnoCiemne(self.fig_r)
        self.canvas_r.setToolTip('Przebieg uczenia; pionowa linia = bieżący krok. Klik przechodzi do kroku.')
        self.zakl_wyk.addTab(self.canvas_r, 'Ruch cząsteczek')
        self.fig_c = Figure(figsize=(5, 5)); self.canvas_c = PlotnoCiemne(self.fig_c)
        self.canvas_c.setToolTip('Liniowe CKA tokenów klatki między krokami (1 = ta sama geometria 600 tokenów,\n'
                                 'niezależnie od obrotu i skali). Kwadratowe bloki = fazy uczenia. Klik: krok z osi x.')
        self.zakl_wyk.addTab(self.canvas_c, 'Mapa CKA krok × krok')
        strona = QWidget(); ls = QVBoxLayout(strona); ls.setContentsMargins(0, 0, 0, 0)
        wp = Przeplyw(); ls.addLayout(wp)
        self.combo_zbior_p = QComboBox(); self.combo_zbior_p.addItems(['patche narzędzia', 'patche tła', 'wybrany patch'])
        dodaj(wp, 'Wykresy dla:', self.combo_zbior_p, 'Po których cząstkach uśredniać wykresy przyczyn ruchu.')
        self.fig_p = Figure(figsize=(5, 7)); self.canvas_p = PlotnoCiemne(self.fig_p)
        self.canvas_p.setToolTip('Przyczyny ruchu cząsteczek w przedziałach między checkpointami wag. Klik: krok.')
        ls.addWidget(self.canvas_p, 1)
        self.zakl_wyk.addTab(strona, 'Przyczyny ruchu')
        podzial.addWidget(lewa)

        from czasteczki3d import WidokCzasteczek3D
        from przestrzen_makro import WidokPrzestrzeniMakro
        self.w3d = WidokCzasteczek3D(self.j128)
        self.makro = WidokPrzestrzeniMakro(self.j128)
        self.makro.ustaw_wzor(self.w3d)
        podzial.addWidget(self.w3d); podzial.addWidget(self.makro)
        podzial.setStretchFactor(0, 0); podzial.setStretchFactor(1, 1); podzial.setStretchFactor(2, 1)
        podzial.setSizes([460, 700, 700])
        self._podzial, self._podzielono = podzial, False
        lay.addWidget(podzial, 1)

        # ---------------- odtwarzanie po krokach
        w = QHBoxLayout(); lay.addLayout(w)
        self.btn_pocz = QPushButton('⏮ Krok 0'); self.btn_wstecz = QPushButton('◀ Krok')
        self.btn_graj = rola(QPushButton('▶ Ucz (odtwarzaj)'), 'glowny'); self.btn_graj.setCheckable(True)
        self.btn_graj.setMinimumWidth(150)
        self.btn_dalej = QPushButton('Krok ▶'); self.btn_kon = QPushButton('Koniec ⏭')
        for b, opis in ((self.btn_pocz, 'Pierwszy krok (wagi losowe).'), (self.btn_wstecz, 'Poprzedni krok.'),
                        (self.btn_graj, 'Odtwarza uczenie krok po kroku.'), (self.btn_dalej, 'Następny krok.'),
                        (self.btn_kon, 'Ostatni krok uczenia.')):
            b.setToolTip(opis); w.addWidget(b)
        self.suwak = QSlider(Qt.Orientation.Horizontal); self.suwak.setRange(0, 0); w.addWidget(self.suwak, 1)
        self.spin_fps = QDoubleSpinBox(); self.spin_fps.setRange(.5, 60); self.spin_fps.setValue(10)
        dodaj(w, 'kroków/s', self.spin_fps, 'Szybkość odtwarzania uczenia (pozycji suwaka na sekundę).')
        self.chk_petla = QCheckBox('pętla'); self.chk_petla.setChecked(True); w.addWidget(self.chk_petla)

        w = QHBoxLayout(); lay.addLayout(w)
        self.combo_por = QComboBox(); self.combo_por.addItems(['brak', 'z poprzednim krokiem', 'z krokiem 0', 'ze stanem końcowym'])
        dodaj(w, 'Porównanie w 3D:', self.combo_por, 'Co jest „PRZED” w widoku 3D; „PO” = bieżący krok.')
        self.combo_srodek = QComboBox()
        self.combo_srodek.addItems(['średnia ze wszystkich kroków', 'stan końcowy (cząstki zbiegają do środka)', 'krok 0'])
        dodaj(w, 'Środek makro:', self.combo_srodek,
              'Względem czego liczone jest położenie cząstki w przestrzeni makrowektorów:\n'
              '- średnia ze wszystkich kroków (i tokenów): zwykły widok;\n'
              '- stan końcowy: cząstka = s · (z_t − z_końcowy tego patcha) · D, więc na końcu uczenia wszystkie\n'
              '  cząstki są w środku układu; widać, jak każdy patch dochodzi do swojego stanu końcowego;\n'
              '- krok 0: cząstki startują ze środka i rozchodzą się w miarę uczenia.')
        self.combo_sledz = QComboBox(); self.combo_sledz.addItems(['wyłączone', 'najpewniejszy patch ViT', 'najszybszy patch w kroku'])
        dodaj(w, 'Śledź patch:', self.combo_sledz, 'Automatyczny wybór patcha w każdym kroku; ręczny klik wyłącza.')
        self.chk_strzalki = QCheckBox('Strzałki przyczyn w makro'); self.chk_strzalki.setChecked(True)
        self.chk_strzalki.setToolTip('Dla wybranej cząstki w przestrzeni makro, z jej bieżącego położenia (po „Licz przyczyny”):\n'
                                     '- biała: rzeczywisty ruch w bieżącym przedziale wag (np. krok 500 -> 510);\n'
                                     '- kolorowe: wkład każdej grupy wag (łodyga, atencja i MLP bloków; kolory jak w\n'
                                     '  wykresie przyczyn), ich suma ≈ biała strzałka;\n'
                                     '- żółta przerywana: kierunek −gradient straty względem latentu tej klatki, tej\n'
                                     '  samej długości co biała (dokąd strata „chciałaby” przesunąć cząstkę).')
        w.addWidget(self.chk_strzalki)
        self.spin_strz = QDoubleSpinBox(); self.spin_strz.setRange(.1, 100); self.spin_strz.setValue(3)
        dodaj(w, 'x', self.spin_strz, 'Mnożnik długości strzałek przyczyn (ruch w przedziale 10 kroków bywa krótki).')
        self.chk_prognoza = QCheckBox('Prognoza liniowa'); self.chk_prognoza.setChecked(True)
        self.chk_prognoza.setToolTip('Dla wybranej cząstki w przestrzeni makro:\n'
                                     '- pomarańczowa przerywana: tor z liniowego modelu ruchu tej cząstki (oscylator:\n'
                                     '  v(t+1) = beta v(t) − kappa (z(t) − m(t)), z(t+1) = z(t) + v(t+1), m = średnia\n'
                                     '  ostatnich 50 kroków), liczony TYLKO ze stanów do bieżącego kroku;\n'
                                     '- turkusowa cienka: rzeczywisty tor tej cząstki w tych samych krokach (do porównania).\n'
                                     'beta i kappa każdej cząstki dopasowane na krokach 300-700 tej historii.')
        w.addWidget(self.chk_prognoza)
        self.spin_h = QSpinBox(); self.spin_h.setRange(2, 300); self.spin_h.setValue(30)
        dodaj(w, 'na', self.spin_h, 'Na ile kroków do przodu prognoza liniowa.')
        self.chk_widok3d = QCheckBox('widok 3D'); self.chk_widok3d.setChecked(True)
        self.chk_widok_makro = QCheckBox('makrowektory'); self.chk_widok_makro.setChecked(True)
        w.addWidget(QLabel('Widoki:')); w.addWidget(self.chk_widok3d); w.addWidget(self.chk_widok_makro)
        self.etykieta_kroku = QLabel(''); self.etykieta_kroku.setWordWrap(True); w.addWidget(self.etykieta_kroku, 1)
        self.timer = QTimer(self); self.timer.timeout.connect(self._nastepny)

        # ---------------- sygnały
        self.combo_bieg.currentIndexChanged.connect(self._zmiana_biegu)
        self.btn_wczytaj.clicked.connect(self._wczytaj)
        self.suwak.valueChanged.connect(self.idz)
        self.btn_pocz.clicked.connect(lambda: self.suwak.setValue(0))
        self.btn_kon.clicked.connect(lambda: self.suwak.setValue(self.suwak.maximum()))
        self.btn_wstecz.clicked.connect(lambda: self.suwak.setValue(self.suwak.value() - 1))
        self.btn_dalej.clicked.connect(lambda: self.suwak.setValue(self.suwak.value() + 1))
        self.btn_graj.toggled.connect(self._graj)
        self.spin_fps.valueChanged.connect(lambda v: self.timer.setInterval(int(1000 / v)))
        self.combo_por.currentIndexChanged.connect(lambda *_: self.idz(self.t))
        self.combo_sledz.currentIndexChanged.connect(lambda *_: self.idz(self.t))
        self.combo_srodek.currentIndexChanged.connect(lambda *_: self._ustaw_makro())
        self.spin_co.valueChanged.connect(lambda *_: self._po_wczytaniu(None) if self.Z is not None else None)
        self.chk_widok3d.toggled.connect(lambda v: (self.w3d.setVisible(v), self.idz(self.t)))
        self.chk_widok_makro.toggled.connect(lambda v: (self.makro.setVisible(v), self.idz(self.t)))
        self.w3d.widok.klik.connect(lambda *_: self._wybierz_token(self.w3d.token))
        self.canvas.mpl_connect('button_press_event', self._klik_klatki)
        self.canvas_r.mpl_connect('button_press_event', self._klik_wykresu)
        self.canvas_c.mpl_connect('button_press_event', self._klik_cka)
        self.canvas_p.mpl_connect('button_press_event', self._klik_cka)
        self.btn_przyczyny.clicked.connect(self._licz_przyczyny)
        self.combo_zbior_p.currentIndexChanged.connect(lambda *_: self._rysuj_przyczyny())
        self.chk_strzalki.toggled.connect(lambda *_: self.idz(self.t))
        self.spin_strz.valueChanged.connect(lambda *_: self.idz(self.t))
        self.chk_prognoza.toggled.connect(lambda *_: self.idz(self.t))
        self.spin_h.valueChanged.connect(lambda *_: self.idz(self.t))
        self.prz = self.P_prz = None; self._delty = (None, None)
        self._zmiana_biegu()

    # ------------------------------------------------------------ wczytanie
    def showEvent(self, e):
        super().showEvent(e)
        if not self._podzielono:
            self._podzielono = True
            QTimer.singleShot(0, lambda: self._podzial.setSizes(
                [460, (self._podzial.width() - 460) // 2, (self._podzial.width() - 460) // 2]))

    def _status(self, t, ms=8000):
        self.okno.statusBar().showMessage(t, ms)

    def _bieg(self):
        i = self.combo_bieg.currentIndex()
        return self.biegi[i] if 0 <= i < len(self.biegi) else None

    def _zmiana_biegu(self, *_):
        b = self._bieg()
        self.combo_klatka.clear()
        if b is None:
            return
        o = AU.opis_biegu(b)
        for i, op in enumerate(o['opisy']):
            self.combo_klatka.addItem(f'{i}: {op}, {int(o["narzedzie"][i].sum())} patchy narzędzia')
        self.etykieta_biegu.setText(f'kroki {int(o["kroki"][0])}..{int(o["kroki"][-1])}, wagi co '
                                    f'{int(np.diff(o["kroki_wag"]).min()) if len(o["kroki_wag"]) > 1 else "?"} kroków '
                                    f'({len(o["wagi"])} checkpointów), ziarno {o["ziarno"]}')

    def _wczytaj(self):
        b = self._bieg()
        if b is None or self.watek:
            return
        i = max(0, self.combo_klatka.currentIndex())
        self.btn_graj.setChecked(False)

        def zadanie(wt):
            o = AU.opis_biegu(b)
            Z = AU.wczytaj_klatke(b, i, lambda k, n: wt.postep.emit(int(40 * k / n), f'latent: {k}/{n} kroków'))
            an = AU.analiza(Z, o['narzedzie'][i], o['tlo'][i],
                            lambda k, n: wt.postep.emit(40 + int(30 * k / n), f'miary ruchu: {k}/{n} kroków'))
            rgb = np.ascontiguousarray(o['obrazy'][i])
            d = AU.detekcje(o, rgb, lambda k, n: wt.postep.emit(70 + int(30 * k / n), f'detekcja z wag: {k}/{n}'))
            return dict(opis=o, Z=Z, an=an, dety=d, i=i, rgb=rgb, osc=AU.oscylator(Z))

        self.watek = WatekFilmu(zadanie, self)
        self.pasek.setRange(0, 100)
        self.watek.postep.connect(lambda k, t: (self.pasek.setValue(k), self._status(t, 3000)))
        self.watek.gotowy.connect(lambda w: (setattr(self, 'watek', None), self._po_wczytaniu(w)))
        self.watek.blad.connect(lambda t: (setattr(self, 'watek', None), QMessageBox.critical(self, 'Błąd', t[-2000:])))
        self._status(f'wczytuję historię uczenia klatki {i}...')
        self.watek.start()

    def _po_wczytaniu(self, w):
        if w is not None:
            self.opis, self.Z, self.an, self.dety, self.i_kl, self.rgb = w['opis'], w['Z'], w['an'], w['dety'], w['i'], w['rgb']
            self.osc = w['osc']
            self.det = self.dety.pop('det'); self._ckpt_det = len(self.opis['wagi']) - 1
            self.narz = self.opis['narzedzie'][self.i_kl]
            self.token = None
            self.prz = self.P_prz = None; self._delty = (None, None); self.fig_p.clear(); self.canvas_p.draw_idle()
        co = self.spin_co.value()
        self.idx = np.arange(0, len(self.Z), co)
        if self.idx[-1] != len(self.Z) - 1:
            self.idx = np.append(self.idx, len(self.Z) - 1)                  # zawsze z krokiem końcowym
        self.kroki = self.opis['kroki']
        Zs = self.Z[self.idx]
        self.w3d.ustaw_baze_pca(Zs[::max(1, len(Zs) // 200)])                 # wspólna baza PCA z puli kroków
        self.makro.ckpt = self.opis['wagi'][-1] if self.opis['wagi'] else None
        self._Zs = Zs
        self._ustaw_makro(rysuj=False)
        self.suwak.blockSignals(True); self.suwak.setRange(0, len(self.idx) - 1); self.suwak.setValue(0)
        self.suwak.blockSignals(False)
        self._rysuj_wykresy(); self._rysuj_cka()
        self.idz(0)
        a = self.an
        self._status(f'Historia: {len(self.Z)} kroków, klatka {self.i_kl}; animacja co {co} krok(ów). '
                     f'Względna zmiana latentu na krok: mediana {np.median(a["wzgl"][1:]) * 100:.2f} %.', 15000)

    def _ustaw_makro(self, rysuj=True):
        if self.Z is None:
            return
        i = self.combo_srodek.currentIndex()
        m = self.makro
        m.odniesienie = None if i == 0 else np.asarray(self.Z[-1 if i == 1 else 0], np.float32)
        if i and m.chk_srodek.isChecked():
            m.chk_srodek.blockSignals(True); m.chk_srodek.setChecked(False); m.chk_srodek.blockSignals(False)
        maski = np.repeat(self.narz[None], len(self.idx), 0)
        pew = self.dety['mapy_pew'][self._ckpt_kroku(self.idx)]
        m.ustaw_sekwencje(self._Zs, maski, pew)
        dP = np.linalg.norm(np.diff(m.P, axis=0), axis=-1) / np.maximum(np.diff(self.idx), 1)[:, None]
        self.v_makro = np.concatenate([np.zeros((1, 600), np.float32), dP]).astype(np.float32)   # [len(idx), 600]
        if rysuj:
            self._rysuj_wykresy(); self.idz(self.t)

    def _ckpt_kroku(self, k):
        """Indeks najbliższego checkpointu wag dla kroku (albo tablicy kroków) k."""
        kw = self.opis['kroki_wag']
        k = np.asarray(self.kroki)[np.asarray(k)]
        return np.abs(kw[None, :] - np.atleast_1d(k)[:, None]).argmin(1) if np.ndim(k) else int(np.abs(kw - k).argmin())

    def _atencja(self, k_ckpt, blok):
        """Atencja klatki próbnej z wag checkpointu k_ckpt (ładowanie wag tylko przy zmianie checkpointu)."""
        import torch
        if self._ckpt_det != k_ckpt:
            ck = torch.load(self.opis['wagi'][k_ckpt], map_location='cpu', weights_only=True)
            self.det.net.load_state_dict(ck.get('net', ck), strict=True)
            self._ckpt_det = k_ckpt
        return self.det.atencja(self.rgb, blok)

    # ------------------------------------------------------------ odtwarzanie
    def _graj(self, wl):
        if wl and self.Z is not None:
            self.btn_graj.setText('⏸ Pauza')
            self.timer.setInterval(int(1000 / self.spin_fps.value())); self.timer.start()
        else:
            self.btn_graj.setText('▶ Ucz (odtwarzaj)'); self.timer.stop()
            if wl:
                self.btn_graj.setChecked(False)

    def _nastepny(self):
        t = self.t + 1
        if t >= len(self.idx):
            if not self.chk_petla.isChecked():
                self.btn_graj.setChecked(False); return
            t = 0
        self.suwak.setValue(t)

    def _wybierz_token(self, tok):
        if tok is None:
            return
        if self.combo_sledz.currentIndex():
            self.combo_sledz.blockSignals(True); self.combo_sledz.setCurrentIndex(0); self.combo_sledz.blockSignals(False)
        self.token = int(tok); self.w3d.token = self.token
        self._rysuj_wykresy()
        if self.prz is not None and self.combo_zbior_p.currentIndex() == 2:
            self._rysuj_przyczyny()
        self.idz(self.t)

    def idz(self, t):
        if self.Z is None:
            return
        self.t = t = int(np.clip(t, 0, len(self.idx) - 1))
        if self.suwak.value() != t:
            self.suwak.blockSignals(True); self.suwak.setValue(t); self.suwak.blockSignals(False)
        k = int(self.idx[t])                                     # indeks kroku w historii
        kc = self._ckpt_kroku(k)
        s = self.combo_sledz.currentIndex()
        if s == 1:
            self.token = int(np.argmax(self.dety['mapy_pew'][kc]))
        elif s == 2:
            self.token = int(np.argmax(self.an['v'][k]))
        self.w3d.token = self.token
        z = np.asarray(self.Z[k], np.float32)
        por = self.combo_por.currentIndex()
        przed = None if por == 0 else np.asarray(self.Z[(int(self.idx[max(0, t - 1)]), 0, -1)[por - 1]], np.float32)
        fn = lambda b, kc=kc: self._atencja(kc, b)
        if self.chk_widok3d.isChecked():
            obrysy = [(prostokat(self.dety['ramka'][kc]), KOLOR_VIT)]
            self.w3d.ustaw_obrysy(obrysy, self.narz)
            if przed is None:
                self.w3d.pokaz(z, None, self.rgb, fn)
            else:
                if self.w3d.combo_tryb.currentIndex() == 0:
                    self.w3d.combo_tryb.blockSignals(True); self.w3d.combo_tryb.setCurrentIndex(1)
                    self.w3d.combo_tryb.blockSignals(False)
                self.w3d.pokaz(przed, z, self.rgb, fn)
        if self.chk_widok_makro.isChecked():
            self.makro.ustaw_atencje(fn)
            self.makro.pokaz_klatke(t, self.token)
            self._strzalki_przyczyn(k)
        self._rysuj_klatke(k, kc)
        self._kursor(k)
        a = self.an
        kr = int(self.kroki[k])
        tekst = (f'krok {kr} ({t + 1}/{len(self.idx)}); detekcja i atencja z wag kroku {int(self.opis["kroki_wag"][kc])}: '
                 f'pewność {self.dety["pewnosc"][kc] * 100:.1f} %; zmiana latentu w tym kroku {a["wzgl"][k] * 100:.2f} %, '
                 f'prędkość narzędzie / tło {a["v"][k][self.narz].mean():.3f} / {a["v"][k][~self.narz].mean():.3f}')
        if self.token is not None:
            tekst += f'; patch {self.token}: prędkość {a["v"][k][self.token]:.3f}, do stanu końcowego {a["do_konca"][k][self.token]:.2f}'
        self.etykieta_kroku.setText(tekst)

    # ------------------------------------------------------------ rysunki
    def _rysuj_klatke(self, k, kc):
        ax = self.ax
        ax.clear(); ax.imshow(self.rgb, extent=(0, 480, 320, 0))
        xy = np.stack(np.divmod(np.arange(600), 30), 1)[:, ::-1] * 16 + 8                # środki patchy (x, y)
        n = self.narz
        ax.scatter(xy[n, 0], xy[n, 1], s=6, c=KOL_N, alpha=.8)
        x0, y0, x1, y1 = self.dety['ramka'][kc]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=KOLOR_VIT[:3], lw=1.6))
        if self.token is not None:
            r, c = divmod(self.token, 30)
            ax.add_patch(Rectangle(    (c * 16, r * 16), 16, 16, fill=False, ec='white', lw=1.5))
        ax.set_xlim(0, 480); ax.set_ylim(320, 0); ax.axis('off')
        ax.set_title(f'krok {int(self.kroki[k])}: wagi kroku {int(self.opis["kroki_wag"][kc])}, pewność '
                     f'{self.dety["pewnosc"][kc] * 100:.0f} %', fontsize=9, color='white')
        self.canvas.draw_idle()

    def _rysuj_wykresy(self):
        if self.an is None:
            return
        a, kr, n = self.an, self.kroki, self.narz
        f = self.fig_r; f.clear()
        osie = f.subplots(5, 1, sharex=True)
        f.subplots_adjust(left=.13, right=.86, top=.97, bottom=.06, hspace=.28)
        ax = osie[0]
        ks, st = self.opis['strata']
        ax.plot(ks, st, color='#f0c040', lw=1.2, label='strata'); ax.set_ylabel('strata', fontsize=8)
        ax2 = ax.twinx(); km, mp = self.opis['mAP']
        ax2.plot(km, mp * 100, 'o-', color='#40d0ff', ms=3, lw=1, label='mAP@0,5'); ax2.set_ylabel('mAP %', fontsize=8)
        ax.set_title('strata uczenia (żółta) i mAP walidacji (niebieska)', fontsize=8)

        def trzy(ax, Y, x, tytul, log=True):
            ax.plot(x, Y[:, n].mean(1), color=KOL_N, lw=.9, label='narzędzie')
            ax.plot(x, Y[:, ~n].mean(1), color=KOL_T, lw=.9, label='tło')
            if self.token is not None:
                ax.plot(x, Y[:, self.token], color=KOL_P, lw=.7, alpha=.8, label=f'patch {self.token}')
            if log:
                ax.set_yscale('log')
            ax.set_title(tytul, fontsize=8)
            ax.legend(fontsize=7, loc='upper right', ncol=3, framealpha=.3)

        v = a['v'].copy(); v[0] = np.nan
        trzy(osie[1], v, kr, 'prędkość tokenu w latencie ||z_t − z_(t−1)|| (na krok)')
        vm = self.v_makro.copy(); vm[0] = np.nan
        trzy(osie[2], vm, kr[self.idx], 'prędkość cząstki w przestrzeni makro ||P_t − P_(t−1)|| (na krok)')
        trzy(osie[3], a['do_konca'], kr, 'odległość od stanu końcowego ||z_t − z_końcowy||', log=False)
        ax = osie[4]
        ax.plot(kr, a['separacja'], color='#7CFC00', lw=1, label='rozdzielenie narzędzie / tło')
        ax.set_ylabel('rozdz.', fontsize=8)
        ax2 = ax.twinx(); ax2.plot(kr, a['rzad'], color='#d080ff', lw=1); ax2.set_ylabel('rząd ef.', fontsize=8)
        ax.set_title('rozdzielenie narzędzie / tło (zielone) i efektywny rząd tokenów (fioletowy)', fontsize=8)
        ax.set_xlabel('krok uczenia', fontsize=8)
        for ax in f.axes:
            ax.tick_params(labelsize=7)
        self._linie = [ax.axvline(0, color='white', lw=.8, alpha=.7) for ax in osie]
        self.canvas_r.draw_idle()

    def _rysuj_cka(self):
        a = self.an
        f = self.fig_c; f.clear(); ax = f.add_axes([.12, .1, .8, .82])
        kr = self.kroki[a['cka_kroki']]
        ax.imshow(a['cka'], origin='lower', cmap='magma', vmin=max(0., float(np.percentile(a['cka'], 2))), vmax=1,
                  extent=(kr[0], kr[-1], kr[0], kr[-1]), aspect='auto')
        ax.set_title('CKA tokenów między krokami (jasne = podobna geometria)', fontsize=8)
        ax.set_xlabel('krok', fontsize=8); ax.set_ylabel('krok', fontsize=8); ax.tick_params(labelsize=7)
        self._linie_c = [ax.axvline(kr[0], color='cyan', lw=.8), ax.axhline(kr[0], color='cyan', lw=.8)]
        self.canvas_c.draw_idle()

    # ------------------------------------------------------------ przyczyny ruchu
    def _licz_przyczyny(self):
        if self.Z is None or self.watek:
            return
        b, o, i, rgb = self._bieg(), self.opis, self.i_kl, self.rgb

        def zadanie(wt):
            P = PR.Przyczyny(b, o, i, rgb)
            return P, P.licz(lambda k, n: wt.postep.emit(int(100 * k / n), f'przyczyny ruchu: checkpoint {k}/{n}'))

        def gotowe(w):
            self.P_prz, self.prz = w
            self._delty = (None, None)
            self._rysuj_przyczyny(); self.zakl_wyk.setCurrentIndex(2); self.idz(self.t)
            r = self.prz
            self._status(f'Przyczyny ruchu: {len(r["kroki_wag"]) - 1} przedziałów; średni cos(−gradient, ruch) '
                         f'{np.nanmean(r["cos_grad"]):.3f}; ruch razem z atencją {np.nanmean(r["cos_atencji"]):.2f} '
                         f'(kontrola: ze średnią {np.nanmean(r["cos_rowno"]):.2f})', 20000)

        self.watek = WatekFilmu(zadanie, self)
        self.watek.postep.connect(lambda k, t: (self.pasek.setValue(k), self._status(t, 3000)))
        self.watek.gotowy.connect(lambda w: (setattr(self, 'watek', None), gotowe(w)))
        self.watek.blad.connect(lambda t: (setattr(self, 'watek', None), QMessageBox.critical(self, 'Błąd', t[-2000:])))
        self.watek.start()

    def _maska_p(self):
        i = self.combo_zbior_p.currentIndex()
        if i == 2 and self.token is not None:
            m = np.zeros(600, bool); m[self.token] = True
            return m, f'patch {self.token}'
        return (self.narz, 'narzędzie') if i == 0 else (~self.narz, 'tło')

    def _rysuj_przyczyny(self):
        r = self.prz
        f = self.fig_p; f.clear()
        if r is None:
            self.canvas_p.draw_idle(); return
        import matplotlib
        from matplotlib.colors import TwoSlopeNorm
        m, nazwa = self._maska_p()
        kw = r['kroki_wag']; x0, x1 = float(kw[0]), float(kw[-1])
        osie = f.subplots(4, 1, sharex=True)
        f.subplots_adjust(left=.2, right=.86, top=.96, bottom=.06, hspace=.35)
        U = r['udzialy'][:, :, m].mean(-1).T                                        # [G, przedziały]
        v = max(.05, float(np.percentile(np.abs(U), 98)))
        ax = osie[0]
        ax.imshow(U, aspect='auto', cmap='coolwarm', norm=TwoSlopeNorm(0, -v, v), extent=(x0, x1, len(PR.GRUPY) - .5, -.5),
                  interpolation='nearest')
        ax.set_yticks(range(len(PR.GRUPY))); ax.set_yticklabels(PR.GRUPY, fontsize=6)
        ax.set_title(f'udział grup wag w ruchu cząstek ({nazwa}); czerwone = pcha w stronę ruchu', fontsize=8)
        kol = matplotlib.colormaps['tab20'](np.arange(len(PR.GRUPY)) % 20)
        for g in range(len(PR.GRUPY)):
            ax.add_patch(matplotlib.patches.Rectangle((x0 - (x1 - x0) * .03, g - .5), (x1 - x0) * .025, 1, color=kol[g],
                                                      clip_on=False))
        ax = osie[1]
        ax.imshow(np.log10(np.maximum(r['zmiana_wag'].T, 1e-6)), aspect='auto', cmap='viridis',
                  extent=(x0, x1, len(PR.GRUPY) - .5, -.5), interpolation='nearest')
        ax.set_yticks(range(len(PR.GRUPY))); ax.set_yticklabels(PR.GRUPY, fontsize=6)
        ax.set_title('zmiana wag grupy w przedziale log10(||ΔW|| / ||W||)', fontsize=8)
        ax = osie[2]
        xs = kw[:-1]
        ax.plot(xs, r['cos_grad'][:, m].mean(1), color='#ffd84a', lw=1, label='cos(−∂L/∂z, Δz)')
        ax.axhline(0, color='gray', lw=.6)
        ax.set_ylabel('cos', fontsize=8)
        ax2 = ax.twinx(); ax2.plot(kw, r['grad_norma'][:, m].mean(1), color='#ff8c40', lw=.9); ax2.set_ylabel('||∂L/∂z||', fontsize=8)
        ax.set_title('czy cząstki ruszają się wzdłuż −gradientu straty tej klatki (żółta) i jego siła (pomarańczowa)', fontsize=8)
        ax = osie[3]
        ax.plot(xs, r['cos_atencji'][:, m].mean(1), color='#40d0ff', lw=1, label='z tym, na co patrzy (atencja)')
        ax.plot(xs, r['cos_rowno'][:, m].mean(1), color='#8fa3b8', lw=1, ls='--', label='kontrola: ze średnią wszystkich')
        ax.set_ylabel('cos', fontsize=8)
        ax2 = ax.twinx(); ax2.plot(kw[1:], r['zmiana_atencji'][:, m].mean(1), color='#d080ff', lw=.9)
        ax2.set_ylabel('zmiana atencji', fontsize=8)
        ax.legend(fontsize=6, loc='lower left', framealpha=.3)
        ax.set_title('ruch razem z tokenami, na które patrzy (niebieska) i zmiana atencji (fioletowa)', fontsize=8)
        ax.set_xlabel('krok uczenia (początek przedziału wag)', fontsize=8)
        for ax in f.axes:
            ax.tick_params(labelsize=7)
        self._linie_p = [ax.axvline(0, color='white', lw=.8, alpha=.7) for ax in osie]
        self.canvas_p.draw_idle()

    def _strzalki_przyczyn(self, k):
        seg, kolory = self._odcinki_przyczyn(k)
        s2, k2 = self._odcinki_prognozy(k)
        seg, kolory = seg + s2, kolory + k2
        self.makro.rysuj_przyczyny(np.array(seg) if seg else None, np.array(kolory) if kolory else None)

    def _odcinki_prognozy(self, k):
        if not self.chk_prognoza.isChecked() or self.token is None or getattr(self, 'osc', None) is None:
            return [], []
        h = self.spin_h.value(); tok = self.token
        tor = AU.prognoza_oscylatora(self.Z, k, tok, self.osc, h)                       # [h+1, 768]
        o = self.makro._biezace()[tok]
        P = o + self.makro.rzut_delty(tor - tor[0])
        seg, kol = [], []
        for a in range(0, h, 1):
            if a % 2 == 0:                                                             # przerywana
                seg += [P[a], P[a + 1]]; kol += [(1, .55, .1, 1)] * 2
        kon = min(len(self.Z) - 1, k + h)
        if kon > k:
            prawdziwy = np.asarray(self.Z[k:kon + 1, tok], np.float32)
            Q = o + self.makro.rzut_delty(prawdziwy - prawdziwy[0])
            for a in range(len(Q) - 1):
                seg += [Q[a], Q[a + 1]]; kol += [(.2, .9, 1, .8)] * 2
        return seg, kol

    def _odcinki_przyczyn(self, k):
        if self.prz is None or self.token is None or not self.chk_strzalki.isChecked():
            return [], []
        import matplotlib
        kw = self.prz['kroki_wag']
        i = int(np.clip(np.searchsorted(kw, self.kroki[k], 'right') - 1, 0, len(kw) - 2))
        if self._delty[0] != i:
            z0, dz, D = self.P_prz.delty(i)
            g = self.P_prz.gradient(i)[0]
            self._delty = (i, (dz.cpu().numpy(), D.cpu().numpy(), g.cpu().numpy()))
        dz, D, g = self._delty[1]
        tok = self.token
        mn = self.spin_strz.value()
        o = self.makro._biezace()[tok]
        wek = self.makro.rzut_delty(np.concatenate([dz[tok][None], D[:, tok]], 0)) * mn          # [1 + G, 3]
        gn = self.makro.rzut_delty(-g[tok][None])[0]
        gn = gn / max(float(np.linalg.norm(gn)), 1e-12) * float(np.linalg.norm(wek[0]))
        kol = np.concatenate([[[1, 1, 1, 1]], matplotlib.colormaps['tab20'](np.arange(len(PR.GRUPY)) % 20)]).astype(np.float32)
        seg = [o, o + wek[0]] + sum(([o, o + v] for v in wek[1:]), [])
        kolory = [kol[0], kol[0]] + sum(([c, c] for c in kol[1:]), [])
        for a in range(6):                                                          # przerywana żółta: −gradient
            seg += [o + gn * a / 6, o + gn * (a + .5) / 6]; kolory += [(1, .85, .2, 1)] * 2
        return seg, [tuple(c) for c in kolory]

    def _kursor(self, k):
        x = float(self.kroki[k])
        for l in getattr(self, '_linie_p', []):
            l.set_xdata([x, x])
        if getattr(self, '_linie_p', None) and self.zakl_wyk.currentIndex() == 2:
            self.canvas_p.draw_idle()
        for l in getattr(self, '_linie', []):
            l.set_xdata([x, x])
        self.canvas_r.draw_idle()
        if getattr(self, '_linie_c', None):
            self._linie_c[0].set_xdata([x, x]); self._linie_c[1].set_ydata([x, x])
            if self.zakl_wyk.currentIndex() == 1:
                self.canvas_c.draw_idle()

    def _idz_do_kroku(self, x):
        if self.Z is None or x is None:
            return
        k = int(np.abs(self.kroki - x).argmin())
        self.suwak.setValue(int(np.abs(self.idx - k).argmin()))

    def _klik_wykresu(self, e):
        if e.inaxes is not None:
            self._idz_do_kroku(e.xdata)

    def _klik_cka(self, e):
        if e.inaxes is not None:
            self._idz_do_kroku(e.xdata)

    def _klik_klatki(self, e):
        if e.inaxes is None or self.Z is None or e.xdata is None:
            return
        c = int(np.clip(e.xdata // 16, 0, 29)); r = int(np.clip(e.ydata // 16, 0, 19))
        self._wybierz_token(r * 30 + c)
