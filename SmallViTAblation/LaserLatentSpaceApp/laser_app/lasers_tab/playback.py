"""Oś czasu: odtwarzanie, punkty A–B, tempo, synchronizacja z podkrokami, klawisze i kliknięcia w widoki (wybór cząsteczki)."""
from __future__ import annotations

from PyQt6.QtCore import Qt
import numpy as np

from laser_app.compute import history as hist


class PlaybackMixin:
    """Oś czasu: odtwarzanie, punkty A–B, tempo, synchronizacja z podkrokami, klawisze i kliknięcia w widoki (wybór cząsteczki).
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _ustaw_ab(self, i):
        """i = 0 (A) albo 1 (B) na bieżącym checkpoincie (ponowny klik w tym samym miejscu usuwa punkt); None = usuń oba;
        -1 = tylko odśwież opis."""
        if i == -1:
            pass
        elif i is None:
            self.ab = [None, None]
        elif self.h is not None:
            t = float(np.clip(round(self.t), 0, self.h.K - 1))
            self.ab[i] = None if self.ab[i] == t else t
            if None not in self.ab and self.ab[0] > self.ab[1]:
                self.ab.reverse()
        h = self.h
        for k, (b_, n) in enumerate(((self.bt_a, 'A'), (self.bt_b, 'B'))):
            v = self.ab[k]
            b_.setText(n + ('•' if v is not None else ''))
            b_.setStatusTip('' if v is None or h is None else f'{n} = krok {int(h.kroki[int(v)])}')
        self.suwak.ab = [None if v is None else int(v * 100) for v in self.ab]; self.suwak.update()
        if h is not None and None not in self.ab:
            self.lb_stan.setText(f'Odcinek A–B: kroki {int(h.kroki[int(self.ab[0])])}–{int(h.kroki[int(self.ab[1])])}'
                                 + (' (w pętli).' if self.ch_petla.isChecked() else ' (bez pętli: stop w B).'))

    def _zakres_ab(self):
        """(początek, koniec) odtwarzania w indeksach checkpointów: A–B, a bez punktu brzeg osi czasu."""
        K1 = 0.0 if self.h is None else float(self.h.K - 1)
        a = 0.0 if self.ab[0] is None else self.ab[0]
        b = K1 if self.ab[1] is None else self.ab[1]
        return (a, b) if a < b else (0.0, K1)

    def _graj(self, on):
        self.bt_play.setText('⏸' if on else '▶')
        self._zegar_sync = self._zegar_os = None
        if not on and getattr(self, '_w_przejsciu', False):            # pauza w czasie powrotu: wróć do pełnych podkroków
            self._w_przejsciu = False; self._sync_stan = 'start'
            self._zastosuj_podkroki()
        if on and self.h is not None and self.ch_sync.isChecked():
            if self.bt_pod.isChecked():
                self.bt_pod.setChecked(False)
            self._s_sync = 0.0; self._sync_stan = 'start'; self._w_przejsciu = False
            a, b = self._zakres_ab()
            self.ustaw_czas(float(round(self.t)) if a <= round(self.t) < b else a)
            if not self.ch_pod.isChecked():
                self.ch_pod.setChecked(True)
            self.timer.start(); return
        if on and self.h is not None:
            a, b = self._zakres_ab()
            if not a <= self.t < b:                                  # poza odcinkiem A–B albo w B: start od A
                self.ustaw_czas(a)
            self.krok_odtw = self.h.krok(self.t)
            self.timer.start()
        else:
            self.timer.stop()

    def _dt(self, zegar):
        """Sekundy od poprzedniej klatki danego zegara (najwyżej 0,1 s, żeby po przestoju okna nie przeskakiwać)."""
        import time
        teraz = time.perf_counter(); poprz = getattr(self, '_zegar_' + zegar, None)
        setattr(self, '_zegar_' + zegar, teraz)
        return 0.0 if poprz is None or teraz - poprz > 0.5 else min(teraz - poprz, 0.1)

    def _nastepny(self, i):
        a, b = self._zakres_ab()
        j = i + self.sp_sync_co.value()
        if j > b:
            return int(a) if self.ch_petla.isChecked() else None
        return j

    def _klatka_sync(self):
        """Synchronizacja bez zatrzymań: w checkpoincie podkroki 0 → 12, potem płynny powrót (wyjście sieci → wejście następnego
        checkpointu, długość w podkrokach) i od razu podkroki następnego. Następny checkpoint liczy się w tle z wyprzedzeniem,
        wszystkie we wspólnym rzucie 3D."""
        h = self.h
        i = int(np.clip(round(self.t), 0, h.K - 1)); j = self._nastepny(i)
        if not self.ch_pod.isChecked():
            self.ch_pod.setChecked(True)
        dt = self.sp_pod_tempo.value() * self._dt('sync')
        stan = getattr(self, '_sync_stan', 'start')
        if stan == 'start':
            gotowe = self.d_pod is not None and self.d_pod.get('krok') == int(h.kroki[i]) and getattr(self, '_pp', None) is not None
            if not gotowe:
                self._zlec_podkroki(); return
            self._sync_stan = 'pod'; self.ustaw_podkrok(0.0); return
        if stan == 'pod':
            s = self.s_pod + dt
            if s >= 2.0 and getattr(self, '_wyprzedzone', None) != i:    # liczenie w tle dopiero, gdy klatki już płyną
                self._wyprzedzone = i
                self._z_wyprzedzeniem(j)
                if j is not None:
                    self._z_wyprzedzeniem(self._nastepny(j))
            if s < 12.0:
                self.ustaw_podkrok(s); return
            if self.s_pod < 12.0:
                self.ustaw_podkrok(12.0)
            if j is None:
                self.bt_play.setChecked(False); return
            dn = self._pamiec_pod().get(int(h.kroki[j]))
            if dn is None:                                           # rzadko: tło nie zdążyło — dolicz pilnie
                if not self.watek_pod.w_kolejce(int(h.kroki[j])):
                    self.watek_pod.zlec(h.wagi, int(h.kroki[j]), h.rgb, self._baza_pod, g=h.g, narz=h.narz, **self._prz_arg())
                return
            if self.sp_przejscie.value() <= 0:
                self._nowy_checkpoint(j); return
            self._start_przejscia(dn); self._sync_stan = 'przejscie'; self._u_prz = 0.0; return
        if stan == 'przejscie':
            self._u_prz += dt / self.sp_przejscie.value()
            if self._u_prz < 1.0:
                self._ustaw_przejscie(self._u_prz); return
            self._nowy_checkpoint(j)

    def _nowy_checkpoint(self, j):
        self._w_przejsciu = False; self._sync_stan = 'pod'
        self.s_pod = 0.0
        self.ustaw_czas(float(j))                                    # weźmie podkroki z pamięci i ustawi podkrok 0
        self.ustaw_podkrok(0.0)

    def _start_przejscia(self, dn):
        """Wyjście sieci bieżącego checkpointu (stan 12) → wejście następnego (stan 0): sfera i rzuty 1–3 jako 2 stany, czas 0..1."""
        from types import SimpleNamespace
        self._w_przejsciu = True
        Pa = self.es.pozycje_podkrokow(self.d_pod)[12]; Pb = self.es.pozycje_podkrokow(dn)[0]
        self.es.ustaw_przejscie(Pa, Pb, self.h.narz)
        stare, nowe = self._pp, self._plaski_podkrokow(dn)
        st = lambda a, b: np.stack([a, b])
        pp2 = SimpleNamespace(K=2, g=stare.g, narz=stare.narz, qc=st(stare.qc[12], nowe.qc[0]), ac=st(stare.qc[12], nowe.qc[0]),
                              wezly=st(stare.wezly[12], nowe.wezly[0]), atomy=st(stare.wezly[12], nowe.wezly[0]),
                              glb=st(stare.glb[12], nowe.glb[0]), par=st(stare.par[12], nowe.par[0]), lok=st(stare.lok[12], nowe.lok[0]),
                              vib=np.zeros((2, hist.TOK), np.float32), LOK99=nowe.LOK99, VIB97=nowe.VIB97, r2=nowe.r2, cz=st(stare.cz[12], nowe.cz[0]))
        self._pp_prz = pp2
        for e in (self.e1, self.e2, self.e3):
            e.ustaw_historie(pp2); e.opcje['pod_kol'] = None; e.opcje['L'] = 0
        self._krok_prz = (int(self.d_pod['krok']), int(dn['krok']))

    def _ustaw_przejscie(self, u):
        u = float(np.clip(u, 0, 1)); g = u * u * (3 - 2 * u)                # łagodny start i koniec
        self.es.ustaw_czas(g)
        for e in (self.e1, self.e2, self.e3):
            e.ustaw_czas(g)
        a, b = self._krok_prz
        self.lb_podkrok.setText(f'przejście: wyjście sieci w kroku {a} → wejście sieci w kroku {b}  ({u:.0%})')
        cz = self._pp_prz.cz[0] * (1 - g) + self._pp_prz.cz[1] * g
        if self.ch_kamera.isChecked():
            self.v3.kadruj(cz[0], cz[1], self.sp_zbl.value(), self.sp_zbl.value() * 2 / 3)

    def _klatka(self):
        h = self.h
        if h is None:
            return
        if self.ch_sync.isChecked():
            self._klatka_sync(); return
        if self.cb_tempo.currentIndex() == 0:
            v = min(0.8 + self.krok_odtw / 250, 150) * self.sp_tempo.value() * 30.0 * self._dt('os')   # film: v kroków na klatkę 30 kl/s
            self.krok_odtw = min(self.krok_odtw + v, float(h.kroki[-1]))
            t = h.indeks(self.krok_odtw)
        else:
            t = self.t + self.sp_tempo.value() * self._dt('os')
        a, b = self._zakres_ab()
        if t >= b and self.ch_petla.isChecked() and self.t >= b:     # pętla: B (albo koniec uczenia) → A (albo krok 0)
            t = a; self.krok_odtw = h.krok(a)
        t = min(t, b)
        self.ustaw_czas(t)
        if t >= b and not self.ch_petla.isChecked():
            self.bt_play.setChecked(False)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self.bt_play.toggle(); return
        if ev.key() in (Qt.Key.Key_Right, Qt.Key.Key_Left) and self.h is not None:
            self.ustaw_czas(round(self.t) + (1 if ev.key() == Qt.Key.Key_Right else -1)); return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------ kliknięcia
    def _klik_wykres(self, p, ev):
        if self.h is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        x = p.plotItem.vb.mapSceneToView(ev.scenePos()).x()
        self.ustaw_czas(x)

    def _wybierz_tok(self, v, P, x, y):
        px, gl_ = v.na_ekran(P)
        d = np.hypot(px[:, 0] - x, px[:, 1] - y); d[~np.isfinite(gl_)] = np.inf
        j = int(np.argmin(d))
        if d[j] < 25:
            self.sp_tok.setValue(j)

    def _klik_2d(self, v, e, x, y):
        h = self.h
        if h is None:
            return
        X = {'globalny': h.wezly[..., :2], 'lokalny': h.qc, 'drgania': h.ac}[e.tryb]
        P = h.interp(X, self.t)
        self._wybierz_tok(v, np.concatenate([P[:, :1], -P[:, 1:2], np.zeros((len(P), 1))], 1), x, y)

    def _klik_sfera(self, x, y):
        h = self.h
        if h is None:
            return
        hs = self._h_sfery()
        r = hs.r99[-1] * np.ones(hs.K) if self.cb_skala.currentIndex() == 1 else hs.r99
        P = h.interp(hs.pos, self.t) / max(float(h.interp(r, self.t)), 1e-9) * 0.55 * self.es.R * self.sp_skala_chm.value()
        if hs is self.h:
            self._wybierz_tok(self.vs, P, x, y); return
        px, gl_ = self.vs.na_ekran(P)                                    # zestaw: koloid któregokolwiek zdjęcia
        d = np.hypot(px[:, 0] - x, px[:, 1] - y); d[~np.isfinite(gl_)] = np.inf
        j = int(np.argmin(d))
        if d[j] < 25:
            i, tok = divmod(j, hist.TOK)
            if i != self._zestaw_idx():
                self.cb_zestaw_akt.setCurrentIndex(i)                    # wczytuje historię tego zdjęcia
            self.sp_tok.setValue(tok)
