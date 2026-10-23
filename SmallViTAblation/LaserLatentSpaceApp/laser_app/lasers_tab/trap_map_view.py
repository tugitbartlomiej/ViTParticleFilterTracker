"""Okienko 14: mapa pułapek wszystkich 600 cząsteczek (trap_map) i nakładka na sferze."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton

from laser_app.compute import trap_map
from laser_app.gpu import trap_map_gl
from laser_app.lasers_tab.workers import TrapMapWorker
from laser_app.ui.widgets import CollapsibleSection, NarrowCombo, _spin


class TrapMapMixin:
    """Okienko 14: mapa pułapek wszystkich 600 cząsteczek (trap_map) i nakładka na sferze.
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_sek_mapa_pul(self):
        """Sekcja mapy pułapek wszystkich cząsteczek (okienko 14)."""
        # --- mapa pułapek wszystkich cząsteczek (okienko 14, trap_map.py)
        g = CollapsibleSection('Mapa pułapek: wszystkie 600 cząsteczek (okienko 14)'); f = QFormLayout(g)
        self.ch_mp = QCheckBox('pokaż mapę pułapek (wczytaj albo policz)'); self.ch_mp.setChecked(False)
        self.ch_mp.setToolTip('Pułapka optyczna KAŻDEJ z 600 cząsteczek w każdym checkpoincie (bez siatki pola, ta jest w okienku 11): skok\n'
                              'w bloku 6, zero pola w komórce (Newton dla wszystkich cząsteczek naraz na GPU) i kształt pola w cząsteczce\n'
                              'z wartości własnych jakobianu: ściąga, siodło, wypycha, wir. Liczy się raz (ok. 1–2 min), potem z pliku.')
        self.bt_mp = QPushButton('Wczytaj / policz'); self.bt_mp_stop = QPushButton('Przerwij'); self.bt_mp_stop.setEnabled(False)
        self.bt_mp.setToolTip('Wczytuje mapę z pliku obok historii albo liczy ją w tle (zasięg komórki jak w „Pułapka optyczna”).')
        self.bt_mp_stop.setToolTip('Przerywa liczenie mapy (nic się nie zapisuje).')
        self.bt_mp.clicked.connect(lambda: self._mp_zlec(wymus=True) if self.ch_mp.isChecked() else self.ch_mp.setChecked(True))
        self.bt_mp_stop.clicked.connect(lambda: setattr(self._mp_watek, 'stop', True) if self._mp_watek is not None else None)
        self.pasek_mp = QProgressBar(); self.pasek_mp.setTextVisible(True)
        self.cb_mp_kolor = NarrowCombo(); self.cb_mp_kolor.addItems(trap_map_gl.TRYBY_KOLORU)
        self.cb_mp_kolor.setToolTip('Kolor cząsteczki: kształt pola w niej (wartości własne λ₀ jakobianu ∂f/∂u: zielony ściąga, żółty siodło,\n'
                                    'czerwony wypycha, domieszka błękitu = wir, czyli para zespolona λ), charakter zera pola w komórce,\n'
                                    'siła skoku albo odległość zera od cząsteczki w promieniach komórki.')
        self.cb_mp_filtr = NarrowCombo(); self.cb_mp_filtr.addItems(['wyróżnij: wszystkie', 'wyróżnij: siodła', 'wyróżnij: ściągające',
                                                                    'wyróżnij: wypychające', 'wyróżnij: z wirem', 'wyróżnij: z zerem w komórce'])
        self.cb_mp_filtr.setToolTip('Pozostałe cząsteczki są przyciemnione (kształt pola w cząsteczce, λ₀).')
        self.sp_mp_rozm = _spin(1, 30, 6, 1, 0, ' px'); self.sp_mp_rozm.setToolTip('Wielkość kropek cząsteczek na ekranie.')
        self.ch_mp_pul = QCheckBox('linie od cząsteczek do zer pola w komórce'); self.ch_mp_pul.setChecked(True)
        self.ch_mp_pul.setToolTip('Odcinek od cząsteczki do jej pułapki (zera pola), gdy zero leży w komórce; długość × lupa.')
        self.sp_mp_lupa = _spin(1, 200, 5, 1, 0, ' ×'); self.sp_mp_lupa.setToolTip('Powiększenie odcinków do pułapek (komórka jest mała).')
        self.ch_mp_skok = QCheckBox('wektory skoku w bloku 6'); self.ch_mp_skok.setChecked(False)
        self.ch_mp_skok.setToolTip('Pchnięcie 768 wiązek bloku 6 na każdą cząsteczkę (f(0), względem środka chmury).')
        self.sp_mp_skok = _spin(0.1, 100, 3, 0.5, 1, ' ×'); self.sp_mp_skok.setToolTip('Długość wektorów skoku na ekranie.')
        self.ch_mp_sfera = QCheckBox('pokaż mapę także na dużej sferze'); self.ch_mp_sfera.setChecked(False)
        self.ch_mp_sfera.setToolTip('Te same kropki, linie i wektory na sferze reflektorów (w jej skali i skali chmury).')
        self.lb_mp = QLabel('—'); self.lb_mp.setWordWrap(True)
        # strzałki pola: siatka G³ w komórce każdej z 600 cząsteczek, pole liniowe f(0) + J₀ u (jak strzałki okienka 11, ale wszędzie)
        self.cb_mp_strz = NarrowCombo(); self.cb_mp_strz.addItems(['brak', 'pole wiązek (pełne pchnięcie)',
                                                                   'kształt pułapki (pole względem skoku cząsteczki)'])
        self.cb_mp_strz.setCurrentIndex(2)
        self.cb_mp_strz.setToolTip('Strzałki pola bloku 6 w komórkach wszystkich 600 cząsteczek naraz (pole liniowe f(0) + J₀ u z jakobianu\n'
                                   'w cząsteczce). Pełne pchnięcie: dokąd wiązki pchają cząstkę postawioną w punkcie (prawie równoległe,\n'
                                   'bo zera pola w komórkach prawie nie ma). Kształt pułapki: to samo minus skok f(0), czyli J₀ u: zbiega się\n'
                                   '(ściąga), rozbiega (wypycha), siodło albo wir wokół każdej cząsteczki.')
        self.sp_mp_G = _spin(1, 5, 3, 1, sufiks=' na bok')
        self.sp_mp_G.setToolTip('Strzałek na bok komórki (G³ na cząsteczkę; 1 = jedna w cząsteczce).')
        self.cb_mp_strz_dl = NarrowCombo()
        self.cb_mp_strz_dl.addItems(['długość ∝ |f| (wspólna skala w checkpoincie)', 'jednakowa długość (sam kierunek)'])
        self.cb_mp_strz_dl.setToolTip('∝ |f|: 95. percentyl długości = odstęp strzałek, więc silne pole widać jako dłuższe strzałki.\n'
                                      'Jednakowa: każda strzałka ma długość odstępu, widać tylko kierunek pola.')
        self.sp_mp_strz = _spin(0.1, 10, 0.8, 0.1, 1, ' ×')
        self.sp_mp_strz.setToolTip('Długość strzałek (1 = odstęp między strzałkami w komórce).')
        self.sp_mp_krycie = _spin(0.05, 1.0, 0.8, 0.05, 2); self.sp_mp_krycie.setToolTip('Krycie strzałek pola (gęste pole: mniejsze).')
        f.addRow(self.ch_mp)
        hb = QHBoxLayout(); hb.addWidget(self.bt_mp); hb.addWidget(self.bt_mp_stop); f.addRow(hb); f.addRow(self.pasek_mp)
        f.addRow('Strzałki', self.cb_mp_strz); f.addRow('  siatka', self.sp_mp_G); f.addRow('  długość', self.cb_mp_strz_dl)
        f.addRow('  mnożnik', self.sp_mp_strz); f.addRow('  krycie', self.sp_mp_krycie)
        f.addRow('Kolor', self.cb_mp_kolor); f.addRow('Wyróżnij', self.cb_mp_filtr); f.addRow('Kropki', self.sp_mp_rozm)
        f.addRow(self.ch_mp_pul); f.addRow('  lupa', self.sp_mp_lupa); f.addRow(self.ch_mp_skok); f.addRow('  długość', self.sp_mp_skok)
        f.addRow(self.ch_mp_sfera); f.addRow(self.lb_mp)
        self.ch_mp.toggled.connect(lambda on: (self._mp_zlec(), self._mp_czas(self.t)) if self.h is not None else None)
        for w_ in (self.cb_mp_kolor, self.cb_mp_filtr, self.cb_mp_strz, self.cb_mp_strz_dl):
            w_.currentIndexChanged.connect(lambda *_: self._mp_czas(self.t) if self.h is not None else None)
        for w_ in (self.sp_mp_rozm, self.sp_mp_lupa, self.sp_mp_skok, self.sp_mp_G, self.sp_mp_strz, self.sp_mp_krycie):
            w_.valueChanged.connect(lambda *_: self._mp_czas(self.t) if self.h is not None else None)
        for w_ in (self.ch_mp_pul, self.ch_mp_skok, self.ch_mp_sfera):
            w_.toggled.connect(lambda *_: (self._mp_zlec(), self._mp_czas(self.t)) if self.h is not None else None)
        self._mp = None; self._mp_watek = None
        self._sek['mapa_pul'] = g

    # ------------------------------------------------------------------ pułapka optyczna (ogniskowa wiązek, okienko 11)
    def _mp_widac(self):
        return self.widac('14') or (self.ch_mp_sfera.isChecked() and self.widac('s'))

    def _mp_zlec(self, wymus=False):
        """Mapa pułapek: z pliku od razu, inaczej liczenie w tle, gdy okienko 14 (albo mapa na sferze) jest widoczne."""
        h = self.h
        if h is None or not self.ch_mp.isChecked() or (self._mp is not None and not wymus):
            return
        if self._mp_watek is not None and self._mp_watek.isRunning():
            return
        zas = round(float(self.sp_og_zas.value()), 2)
        d = trap_map.wczytaj(h, zas)
        if d is not None:
            self._mp_gotowe(d)
            return
        if not wymus and not self._mp_widac():
            self.lb_mp.setText('Mapa policzy się, gdy okienko 14 (albo mapa na sferze) będzie widoczne.')
            return
        w = TrapMapWorker(h, zas, self._dokladnosc())
        w.postep.connect(lambda i, n, o: (self.pasek_mp.setMaximum(n), self.pasek_mp.setValue(i), self.pasek_mp.setFormat(o)))
        w.gotowe.connect(self._mp_gotowe)
        w.blad.connect(lambda t: (self.bt_mp_stop.setEnabled(False), self.lb_mp.setText(f'Mapa pułapek: {t}')))
        self._mp_watek = w; self.bt_mp_stop.setEnabled(True); self.pasek_mp.setValue(0)
        self.lb_mp.setText(f'Liczę pułapki 600 cząsteczek we wszystkich {h.K} checkpointach (w tle, GPU)…')
        w.start()

    def _mp_gotowe(self, d):
        self.bt_mp_stop.setEnabled(False)
        h = self.h
        if h is None or len(d['kroki']) != h.K:
            return
        self._mp = d
        R_ = 0.55 * self.es.R * self.sp_skala_chm.value()                # chmura ma promień ~0,55 R sfery (r99 → 0,55 R)
        self.vmp.opts['distance'] = 2.4 * R_ / np.tan(np.radians(self.vmp.opts['fov']) / 2) * 0.5
        zb = np.asarray(d['zb'])
        self.lb_mp.setText(f'Mapa pułapek: {h.K} checkpointów × {zb.shape[1]} cząsteczek ({d.get("dokladnosc", "?")}); zero pola w komórce '
                           f'w {zb.mean():.2%} par checkpoint–cząsteczka.')
        self._mp_czas(self.t)

    def _mp_czas(self, t):
        """Rysunek mapy w chwili t (położenia, odcinki do zer i skoki interpolowane między checkpointami; kolory przenikają)."""
        d, h = self._mp, self.h
        pokaz14 = d is not None and h is not None and self.ch_mp.isChecked()
        self.gl_mp.widoczna(pokaz14)
        self.gl_mp_s.widoczna(pokaz14 and self.ch_mp_sfera.isChecked())
        if not pokaz14:
            if h is not None and self.ch_mp.isChecked() and d is None:
                self.lb14.setText('Mapa pułapek: kliknij „Wczytaj / policz” (sekcja „Mapa pułapek”).')
            elif h is not None and not self.ch_mp.isChecked():
                self.lb14.setText('Mapa pułapek wyłączona: zaznacz „pokaż mapę pułapek” (sekcja „Mapa pułapek”).')
            return
        if not self._mp_widac():
            return
        K = h.K
        t = float(np.clip(t, 0, K - 1)); a = int(np.floor(t)); b = min(a + 1, K - 1); f = t - a
        r = h.r99[-1] * np.ones(K) if self.cb_skala.currentIndex() == 1 else h.r99
        s = 0.55 * self.es.R / np.maximum(np.asarray(r, np.float64), 1e-9) * self.sp_skala_chm.value()
        pos = lambda k: np.asarray(h.pos[k], np.float64)
        P = (1 - f) * pos(a) * s[a] + f * pos(b) * s[b]
        tryb = self.cb_mp_kolor.currentIndex()
        kol = (1 - f) * trap_map_gl.kolory(d, a, tryb, pos(a)) + f * trap_map_gl.kolory(d, b, tryb, pos(b))
        filtr = self.cb_mp_filtr.currentIndex()
        if filtr:
            m = trap_map_gl.maska_filtru(d, a if f < 0.5 else b, filtr)
            kol[~m, :3] *= 0.25; kol[~m, 3] *= 0.35
        Q = kq = None
        if self.ch_mp_pul.isChecked():
            zb = np.asarray(d['zb'][a if f < 0.5 else b], bool)
            lu = self.sp_mp_lupa.value()
            off = (1 - f) * (d['ogn'][a] - pos(a)) * s[a] + f * (d['ogn'][b] - pos(b)) * s[b]
            Q = np.stack([P[zb], P[zb] + lu * off[zb]], 1); kq = kol[zb]
        S = None
        if self.ch_mp_skok.isChecked():
            dl = self.sp_mp_skok.value()
            sk = (1 - f) * d['skok'][a] * s[a] + f * d['skok'][b] * s[b]
            S = np.stack([P, P + dl * sk], 1)
        pole, opis_pola = self._mp_strzalki(d, a, b, f, s, P, kol)
        tok = int(self.sp_tok.value())
        wyb = P[tok] if 0 <= tok < len(P) else None
        rozm = self.sp_mp_rozm.value()
        if self.widac('14'):
            self.gl_mp.ustaw(P, kol, rozm, Q, kq, S, wyb, pole)
        if self.ch_mp_sfera.isChecked() and self.widac('s'):
            self.gl_mp_s.ustaw(P, kol, rozm, Q, kq, S, wyb, pole)
        k = int(round(t))
        self.lb14.setText(f'Checkpoint {k + 1}/{K} (krok {int(h.kroki[k])}): {trap_map.statystyki(d, k)}.  Zielony ściąga, żółty siodło, '
                          f'czerwony wypycha, błękitny odcień = wir; biała obwódka = wybrana cząsteczka {tok}.{opis_pola}')

    def _mp_strzalki(self, d, a, b, f, s, P, kol):
        """Strzałki pola w komórkach wszystkich cząsteczek (G³ na cząsteczkę) w chwili między checkpointami a i b (waga f):
        punkty siatki i wektory na ekranie interpolowane jak położenia, kolor = kolor cząsteczki. Zwraca (pole dla TrapMapGL, opis)."""
        tryb = self.cb_mp_strz.currentIndex()
        if not tryb or 'J0' not in d:
            return None, ''
        G = int(self.sp_mp_G.value()); wzgl = tryb == 2; jedn = self.cb_mp_strz_dl.currentIndex() == 1
        mn = self.sp_mp_strz.value()

        def ekran(k):
            U, F, odst = trap_map.pole_w_komorkach(d, k, G, wzgl)
            odst_e = odst * s[k]                                                   # odstęp strzałek na ekranie, (n,)
            m = np.linalg.norm(F, axis=2)
            if jedn:
                V = F / np.maximum(m, 1e-30)[..., None] * odst_e[:, None, None]
            else:
                V = F * (float(np.median(odst_e)) / max(float(np.percentile(m, 95)), 1e-30))
            return U * s[k], mn * V, m

        Ua, Va, ma = ekran(a)
        Ub, Vb, _ = ekran(b) if b != a and f > 0 else (Ua, Va, ma)
        P0 = P[:, None] + (1 - f) * Ua + f * Ub
        V = (1 - f) * Va + f * Vb
        c = np.repeat(np.asarray(kol, np.float32), G ** 3, axis=0); c[:, 3] *= self.sp_mp_krycie.value()
        pole = trap_map_gl.odcinki_strzalek(P0.reshape(-1, 3), V.reshape(-1, 3), c)
        k = a if f < 0.5 else b
        rogi = 2 * trap_map.siatka_komorki(2)[None] * (float(d.get('zasieg', 1.0)) * d['r_kom'][k])[:, None, None]
        Jr = np.einsum('nab,ngb->nga', d['J0'][k], rogi)                          # J₀ u w rogach komórki wobec skoku
        ksz = np.linalg.norm(Jr, axis=2).max(1) / np.maximum(np.linalg.norm(d['skok'][k], axis=1), 1e-30)
        return pole, (f'  Strzałki pola: {len(P0) * G ** 3} ({G}³ na cząsteczkę); kształt pola J₀u w rogu komórki to mediana '
                      f'{float(np.median(ksz)):.1%} skoku f(0).')
