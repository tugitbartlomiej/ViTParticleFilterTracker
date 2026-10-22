"""Zakładka „Lasery i koloidy”: układ z filmów 2, 12 i 16 (`outputs/porownanie_szklo_20261010/film_krysztal`) na żywo, na GPU.

Wybierasz historię checkpointów (katalog k<krok>.pt), zakres kroków i zdjęcie; `lasery_dane.policz` liczy na GPU (raz, potem z cache)
latent 600 × 768, kierunki fc2, atencję 6 bloków, ramkę i pewność w każdym checkpoincie oraz rozkład ruchu na trzy składowe.
Okienka (każde można zamknąć, odpiąć jako osobne okno albo przesunąć; menu „Okienka”):
  1. Ruch globalny (układ inercjalny)          2. Odkształcenie lokalne: studnie i strzałki od adresów
  3. Drgania wokół centrów (zbliżenie)        Sfera reflektorów (768 reflektorów, wiązki, załamanie na sferze — film 12)
  4. Atencja 6 bloków (mozaika)               5. Mapa na zdjęciu (centra, drgania, atencja, natężenie reflektora)
  6. Wielkość trzech składowych ruchu         7. IoU    8. Pewność    9. Pary przeciwnych reflektorów
  10. Obrót igieł reflektorów (gradient, film 23); na sferze strzałki obrotu („Strzałki obrotu” w sekcji sfery)
  11. Pułapka optyczna (ogniskowa wiązek)     12. Dolina pewności i IoU wybranej cząsteczki (dolina.py; też na sferze)
  13. Reflektor: mechanika (reflektor_mechanika.py): odrzut wiązek i wykonany ruch igły, celownika i progu
  R1–R8. Rzuty z 192 wymiarów z miernikiem wierności (sekcja „Rzuty z 192 wymiarów”, rzuty_okienka.py, rzuty_dane.py):
      mapa klas, siły, wiązki ważone, rama siły, rama Schura, wędrówka w czasie uczenia, 7 widoków Fano i panel bez rzutu
Wszystkie okienka pokazują tę samą chwilę osi czasu (suwak, odtwarzanie, klik w wykres). Rysują shadery (lasery_gpu.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QByteArray, QEvent, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QProgressBar, QPushButton, QSizePolicy, QSlider,
                             QSpinBox, QToolButton, QVBoxLayout, QWidget)

import lasery_dane as LD
import ogniskowa as OGN
import przejscia_zapis as PZ
import rzuty_okienka as RZO
from lasery_gpu import DolinaGPU, MapyNaZdjeciu, PoleOgniskowej, PrzejscieGPU, RuchPlaski, SchematGPU, SferaReflektorow
from okna import bez_ukladu
from uklad import WaskiCombo
from widok_gl import WidokGL

TLO = '#03060a'
EP = LD.NA_EPOKE
pl = lambda x, n=2: f'{x:.{n}f}'.replace('.', ',')


class Widok2D(WidokGL):
    """Widok z góry (rzut płaski): lewy przycisk przesuwa, kółko przybliża; brak obrotu."""

    obrot = False                                # widok z góry: LPM i ŚPM przesuwają, PPM + przeciągnięcie zbliża/oddala

    def __init__(self):
        super().__init__()
        self.setBackgroundColor(TLO)
        self.opts['fov'] = 30
        self.setCameraPosition(elevation=90, azimuth=-90, distance=60)

    def kadruj(self, cx, cy, szer, wys=None):
        """Środek (cx, cy) w komórkach (y obrazu w dół); w kadrze mieści się szer × wys komórek (proporcje widżetu)."""
        from PyQt6.QtGui import QVector3D
        self._kadr = (cx, cy, szer, wys)
        if wys is not None:
            szer = max(float(szer), float(wys) * self.width() / max(self.height(), 1))
        self.opts['center'] = QVector3D(float(cx), float(-cy), 0.0)
        self.opts['distance'] = float(szer) / 2 / np.tan(np.radians(self.opts['fov'] / 2))   # pole widzenia poziome
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, '_kadr', None) is not None:
            self.kadruj(*self._kadr)

    def px_na_komorke(self):
        return self.width() / (2 * self.opts['distance'] * np.tan(np.radians(self.opts['fov'] / 2)))


class Liczenie(QThread):
    postep = pyqtSignal(int, int, str)
    gotowe = pyqtSignal(str)
    blad = pyqtSignal(str)

    def __init__(self, wagi, kroki, rgb, gt, okno):
        super().__init__()
        self.args = (wagi, kroki, rgb, gt, okno)
        self.stop = False

    def run(self):
        try:
            out = LD.policz(*self.args, postep=lambda i, n, o: self.postep.emit(i, n, o), przerwij=lambda: self.stop)
            self.gotowe.emit(str(out))
        except LD.Przerwano:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class ZestawLiczenie(QThread):
    """Historie kilku zdjęć tej samej historii wag (te same checkpointy): z cache od razu, inaczej `lasery_dane.policz` po kolei."""
    postep = pyqtSignal(int, int, str)
    gotowe = pyqtSignal(object)
    blad = pyqtSignal(str)

    def __init__(self, wagi, kroki, zdjecia, okno):
        super().__init__()
        self.args = (wagi, kroki, list(zdjecia), okno)
        self.stop = False

    def run(self):
        wagi, kroki, zdjecia, okno = self.args
        katalogi = []
        try:
            for i, zd in enumerate(zdjecia):
                rgb, roz = LD.wczytaj_zdjecie(zd)
                gt, _ = LD.ramka_prawdziwa(zd, roz)
                out = LD.policz(wagi, kroki, rgb, gt, okno, przerwij=lambda: self.stop,
                                postep=lambda a, b, o, i=i: self.postep.emit(a, b, f'zdjęcie {i + 1}/{len(zdjecia)}: {o}'))
                katalogi.append(str(out))
            self.gotowe.emit(katalogi)
        except LD.Przerwano:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class OgniskowaWatek(QThread):
    """Pułapka optyczna we wszystkich checkpointach historii, w tle: wybrana cząsteczka i (opcjonalnie) cząsteczki „razem”
    w jednym przejściu (ogniskowa.policz_wiele: wagi i sieć wspólne). gotowe(args, {cząsteczka: wynik})."""
    postep = pyqtSignal(int, int, str)
    gotowe = pyqtSignal(object, object)
    blad = pyqtSignal(str)

    def __init__(self, h, tok, G, zasieg, razem=()):
        super().__init__()
        self.h, self.args = h, (int(tok), int(G), float(zasieg))
        self.toki = list(dict.fromkeys([int(tok)] + [int(t) for t in razem]))
        self.stop = False

    def run(self):
        try:
            W = OGN.policz_wiele(self.h, self.toki, *self.args[1:], postep=lambda i, n, o: self.postep.emit(i, n, o),
                                 przerwij=lambda: self.stop)
            self.gotowe.emit(self.args, W)
        except OGN.Przerwano:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class PolePodkrokowWatek(QThread):
    """Pole sił w kuli komórki wybranej cząsteczki w 12 podkrokach jednego checkpointu (pole_podkrokow.policz), w tle.
    Zlecenia nadpisują się: liczy się tylko ostatnie (przy przewijaniu checkpointów stare są pomijane)."""
    gotowe = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._zlecenie = None

    def zlec(self, klucz, *args, **kw):
        self._zlecenie = (klucz, args, kw)
        if not self.isRunning():
            self.start()

    def run(self):
        import pole_podkrokow as PP
        while self._zlecenie is not None:
            klucz, args, kw = self._zlecenie; self._zlecenie = None
            try:
                self.gotowe.emit(klucz, PP.policz(*args, **kw))
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.gotowe.emit(klucz, dict(blad_watku=f'{type(e).__name__}: {e}'))


class DolinaWatek(QThread):
    """Przekrój doliny pewności i IoU (dolina.Liczydlo.policz) w tle. Zlecenia nadpisują się: liczy się tylko ostatnie."""
    gotowe = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._zlecenie = None
        self._liczydlo = None

    def zlec(self, klucz, h, *args):
        self._zlecenie = (klucz, h, args)
        if not self.isRunning():
            self.start()

    def run(self):
        import dolina as DL_
        while self._zlecenie is not None:
            klucz, h, args = self._zlecenie; self._zlecenie = None
            try:
                if self._liczydlo is None or self._liczydlo.h is not h:
                    self._liczydlo = DL_.Liczydlo(h)
                self.gotowe.emit(klucz, self._liczydlo.policz(*args))
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.gotowe.emit(klucz, dict(blad_watku=f'{type(e).__name__}: {e}'))


class MechanikaWatek(QThread):
    """Mechanika reflektora (reflektor_mechanika.Liczydlo.policz) w tle. Zlecenia nadpisują się: liczy się tylko ostatnie."""
    gotowe = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._zlecenie = None
        self._liczydlo = None

    def zlec(self, klucz, h, *args):
        self._zlecenie = (klucz, h, args)
        if not self.isRunning():
            self.start()

    def run(self):
        import reflektor_mechanika as RM
        while self._zlecenie is not None:
            klucz, h, args = self._zlecenie; self._zlecenie = None
            try:
                if self._liczydlo is None or self._liczydlo.h is not h:
                    self._liczydlo = RM.Liczydlo(h)
                self.gotowe.emit(klucz, self._liczydlo.policz(*args))
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.gotowe.emit(klucz, dict(blad_watku=f'{type(e).__name__}: {e}'))


class PomiaryPulapkiWatek(QThread):
    """Pomiary pułapki w podkrokach dla całej historii (pulapka_obliczenia.policz_historie), w tle; zapis npz + csv."""
    postep = pyqtSignal(int, int, str)
    koniec = pyqtSignal(str)

    def __init__(self, h, tok, co, G):
        super().__init__()
        self.h, self.tok, self.co, self.G = h, int(tok), int(co), int(G)
        self.stop = False

    def run(self):
        import pulapka_obliczenia as PO
        try:
            p = PO.policz_historie(self.h, self.tok, self.co, self.G, postep=lambda i, n, t: self.postep.emit(i, n, t),
                                   przerwij=lambda: self.stop)
            self.koniec.emit(f'zapisano: {p} (i .csv obok)' if p else 'przerwano przed pierwszym checkpointem')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.koniec.emit(f'błąd {type(e).__name__}: {e}')


class Zwijana(QGroupBox):
    """Sekcja panelu zwijana strzałką w nagłówku: ▾ rozwinięta, ▸ zwinięta (klik w cały nagłówek). Zwinięta pokazuje tylko nagłówek.
    Nagłówek leży nad treścią (górny margines treści = jego wysokość), więc sekcje budują się jak zwykły QGroupBox: QFormLayout(g)."""

    def __init__(self, tytul):
        super().__init__()
        self.naglowek = QToolButton(self)
        self.naglowek.setText(tytul); self.naglowek.setCheckable(True); self.naglowek.setChecked(True)
        self.naglowek.setArrowType(Qt.ArrowType.DownArrow)
        self.naglowek.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.naglowek.setAutoRaise(True)
        self.naglowek.setStyleSheet('QToolButton { border: none; font-weight: bold; padding: 3px 2px; }'
                                    'QToolButton:hover { background: rgba(255, 255, 255, 0.07); }')
        self.naglowek.setToolTip('Kliknij, żeby zwinąć (▸) albo rozwinąć (▾) sekcję.')
        self.naglowek.toggled.connect(self._zwin)
        self._wys = 0
        self._przelicz()

    def _przelicz(self):
        """Wysokość nagłówka z jego faktycznego rozmiaru (pogrubiona czcionka, arkusz stylów, skalowanie DPI) i margines treści równy
        tej wysokości. Liczona po nałożeniu stylu, nie w konstruktorze: tam sizeHint jest za mały i treść wchodziła pod nagłówek."""
        self.naglowek.ensurePolished()
        wys = self.naglowek.sizeHint().height() + 6
        if wys != self._wys:
            self._wys = wys
            self.naglowek.setGeometry(3, 1, max(self.width() - 6, 10), wys - 2)
            if self.naglowek.isChecked():
                self.setContentsMargins(0, wys, 0, 0)
            else:
                self.setMaximumHeight(wys + 3)
        elif self.contentsMargins().top() != wys and self.naglowek.isChecked():
            self.setContentsMargins(0, wys, 0, 0)       # QGroupBox sam przelicza ramkę przy zmianie stylu i zeruje nasz margines

    def event(self, ev):
        wynik = super().event(ev)
        if ev.type() in (QEvent.Type.Polish, QEvent.Type.StyleChange, QEvent.Type.FontChange, QEvent.Type.Show):
            self._przelicz()
        return wynik

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._przelicz()
        self.naglowek.setGeometry(3, 1, max(self.width() - 6, 10), self._wys - 2)
        self.naglowek.raise_()

    def zwinieta(self):
        return not self.naglowek.isChecked()

    def _zwin(self, on):
        self.naglowek.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        for w in self.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            if w is not self.naglowek:
                w.setVisible(on)
        self.setMaximumHeight(16777215 if on else self._wys + 3)       # zwinięta: sam nagłówek (bez marginesów pustej treści)


def plaski_podkrokow(d, g_, narz):
    """Rzuty płaskie stanów strumienia rezydualnego jak w filmie 2: mapa liniowa 192 → 2 dopasowana (najmniejsze kwadraty) do adresów
    patchy na wszystkich 13 stanach naraz (część położenia, która koduje adres), potem Prokrustes do siatki w każdym stanie:
    panel 1 = ruch globalny siatki, panel 2 = odkształcenie lokalne (strzałka adres → token), panel 3 = droga tokenów przez podkroki."""
    from types import SimpleNamespace
    g = np.asarray(g_, np.float64)
    X = d['xn'].astype(np.float64)                                              # (13, 600, 192) znormowane stany
    A = np.concatenate([X.reshape(-1, X.shape[2]), np.ones((X.shape[0] * X.shape[1], 1))], 1)
    L, *_ = np.linalg.lstsq(A, np.tile(g, (X.shape[0], 1)), rcond=None)
    S = (A @ L).reshape(X.shape[0], X.shape[1], 2)
    par, mapy = LD.prokrustes2d(S, g)
    QC = np.stack([LD.zastosuj(S[t], mapy[t]) for t in range(len(S))]).astype(np.float32)
    GLB = np.stack([(g - mapy[t][3]) @ mapy[t][1].T / mapy[t][2] + mapy[t][0] for t in range(len(S))]).astype(np.float32)
    lok = np.linalg.norm(QC - g, axis=-1).astype(np.float32)
    vib = np.concatenate([np.zeros((1, LD.TOK)), np.linalg.norm(np.diff(QC, axis=0), axis=-1)]).astype(np.float32)
    S3 = np.concatenate([S, np.zeros(S.shape[:2] + (1,))], 2).astype(np.float32)
    r2 = float(1 - ((A @ L - np.tile(g, (X.shape[0], 1))) ** 2).sum() / ((g - g.mean(0)) ** 2).sum() / X.shape[0])
    return SimpleNamespace(K=len(S), g=np.asarray(g_), narz=narz, qc=QC, ac=QC, wezly=S3, atomy=S3, glb=GLB, par=par, lok=lok, vib=vib,
                           LOK99=float(np.percentile(lok, 99)), VIB97=float(max(np.percentile(vib, 97), 1e-6)), r2=r2,
                           cz=QC[:, narz].mean(1) if narz.any() else np.zeros((len(S), 2)))


def statystyki_podkrokow(d):
    """Rozrzut chmury r = średnia |x_i − x̄| w każdym stanie (surowo) i składowa promieniowa przesunięcia w podkroku względem
    środka chmury (< 0: ciągnie do środka), w jednostkach r przed podkrokiem."""
    Xc = d['xn'] * d['rms'][:, None, None]
    U = Xc / np.maximum(np.linalg.norm(Xc, axis=2, keepdims=True), 1e-12)
    prom = np.array([((Xc[s + 1] - Xc[s]) * U[s]).sum(1).mean() / d['rms'][s] for s in range(12)])
    return dict(stos=d['rms'][1:] / d['rms'][:-1], prom=prom)


class PodkrokiWatek(QThread):
    """Liczy `lasery_dane.podkroki` po kolei dla zleconych checkpointów (kolejka bez powtórzeń; `pilne` wskakuje na początek)."""
    gotowe = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        import threading
        self._zamek = threading.Lock()
        self._kolejka = []
        self.finished.connect(lambda: self.start() if self._kolejka else None)

    def zlec(self, wagi, krok, rgb, baza=None, pilne=True, g=None, narz=None, kat=None, zapis=False, f16=False):
        """kat: katalog zapisanych przejść (przejscia_zapis); zapis: zapisz po policzeniu; f16: zapis oszczędny."""
        with self._zamek:
            self._kolejka = [z for z in self._kolejka if z[1] != krok]
            z = (wagi, krok, rgb, baza, g, narz, kat, zapis, f16)
            self._kolejka.insert(0, z) if pilne else self._kolejka.append(z)
            del self._kolejka[6:]
        if not self.isRunning():
            self.start()

    def w_kolejce(self, krok):
        with self._zamek:
            return any(z[1] == krok for z in self._kolejka)

    def run(self):
        while True:
            with self._zamek:
                if not self._kolejka:
                    return
                wagi, krok, rgb, baza, g, narz, kat, zapis, f16 = self._kolejka.pop(0)
            try:
                d = PZ.wczytaj(kat, krok) if kat is not None else None
                if d is not None:                                    # zapisane wcześniej: bez liczenia
                    d['baza'] = PZ.wczytaj_baze(kat) or baza; d['_z_dysku'] = True
                else:
                    with PZ.ZAMEK_SIECI:
                        d = LD.podkroki(wagi, krok, rgb, baza=baza)
                    if kat is not None and zapis:
                        PZ.zapisz_baze(kat, d['baza'])
                        bz = PZ.wczytaj_baze(kat)
                        if bz is not None and np.array_equal(bz['P_n'], d['baza']['P_n']):   # tylko w rzucie z baza.npz
                            d['_zapisano'] = PZ.zapisz(kat, d, f16)
                if g is not None:                                    # cięższe rachunki CPU też tutaj, nie w wątku okna
                    d['_pp'] = plaski_podkrokow(d, g, narz)
                    d['_stat'] = statystyki_podkrokow(d)
                self.gotowe.emit(d)
            except Exception as e:                                       # noqa: BLE001
                self.gotowe.emit({'blad': f'{type(e).__name__}: {e}'})


class WszystkiePrzejsciaWatek(QThread):
    """Zapis przejść przez sieć dla checkpointów historii (co `co`, zawsze z ostatnim), w tle; już zapisane pomija."""
    postep = pyqtSignal(int, int, str)
    baza = pyqtSignal(object)
    koniec = pyqtSignal(str)

    def __init__(self, h, kat, f16, baza, co, limit_b, juz_b):
        super().__init__()
        self.h, self.kat, self.f16, self.baza_, self.co, self.limit_b, self.juz_b = h, kat, f16, baza, max(int(co), 1), limit_b, juz_b
        self.stop = False

    def run(self):
        import time
        h = self.h
        idx = list(range(0, h.K, self.co))
        if idx[-1] != h.K - 1:
            idx.append(h.K - 1)
        try:
            baza = PZ.wczytaj_baze(self.kat) or self.baza_
            nowe = pom = 0; bajty = self.juz_b; t0 = time.time()
            for i, j in enumerate(idx):
                if self.stop:
                    self.koniec.emit(f'przerwano: zapisano {nowe} nowych, {pom} było już zapisanych'); return
                krok = int(h.kroki[j])
                if PZ.plik(self.kat, krok).exists():
                    pom += 1; continue
                if bajty > self.limit_b:
                    self.koniec.emit(f'limit miejsca ({bajty / 1e9:.1f} GB): zapisano {nowe} nowych'); return
                with PZ.ZAMEK_SIECI:
                    d = LD.podkroki(h.wagi, krok, h.rgb, baza=baza)
                if baza is None:
                    baza = d['baza']; self.baza.emit(baza)
                PZ.zapisz_baze(self.kat, baza)
                bajty += PZ.zapisz(self.kat, d, self.f16); nowe += 1
                tt = (time.time() - t0) / nowe * (len(idx) - i - 1)
                self.postep.emit(i + 1, len(idx), f'krok {krok}: {i + 1}/{len(idx)}, {bajty / 1e9:.1f} GB, zostało ~{tt / 60:.0f} min')
            self.koniec.emit(f'gotowe: zapisano {nowe} nowych, {pom} było już zapisanych ({bajty / 1e9:.1f} GB)')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.koniec.emit(f'błąd {type(e).__name__}: {e}')


class DiagramSieci(QWidget):
    """Schemat MałegoViT jak w diagram_maly_vit.pdf: obraz → łodyga + kodowanie pozycji → 6 bloków (uwaga, MLP) → LayerNorm → głowa.
    Podświetla bieżący podkrok s (0..12: stan 0 = wejście bloków, podkrok k = uwaga (k parzyste) albo MLP bloku k // 2), klik wybiera podkrok."""
    klik = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.s = 0.0
        self.setMinimumHeight(118); self.setMaximumHeight(150)
        self._pola = []
        self._miniatury = []                       # QImage: obraz, 4 etapy łodygi, kodowanie pozycji, tokeny + PE
        self._opisy = []

    def ustaw_lodyge(self, rgb, d):
        """Miniatury etapów wejścia dla bieżącego checkpointu (mapy cech łodygi jako 3 główne składowe kanałów)."""
        from PyQt6.QtGui import QImage
        obrazy = [rgb] + list(d.get('lodyga', []))
        self._miniatury = []
        for a in obrazy:
            a = np.ascontiguousarray(a)
            q = QImage(a.data, a.shape[1], a.shape[0], 3 * a.shape[1], QImage.Format.Format_RGB888).copy()
            self._miniatury.append(q)
        ks = d.get('lodyga_ksztalty', [])
        self._opisy = ['obraz 480×320×3'] + [f'konw. {i + 1}: {w_}×{h_}×{c_}' for i, (c_, h_, w_) in enumerate(ks[:4])] + \
                      ['kodowanie pozycji sin-cos 30×20×192', 'tokeny = konw. 4 + PE (600 × 192)']
        self.update()

    def ustaw(self, s):
        self.s = float(s); self.update()

    def paintEvent(self, ev):
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QColor, QFont, QPainter, QPen
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#0b1014'))
        W_, H_ = self.width(), self.height()
        n_kol = 6 + 6 + 2 + 0.4                               # obraz, 4 konwolucje, +PE, 6 bloków, LN, głowa
        szer = (W_ - 20) / (n_kol + 0.6)
        y0, h = 22, H_ - 34
        self._pola = []
        f = QFont(self.font()); f.setPointSizeF(max(7.0, min(10.0, szer / 9))); p.setFont(f)
        s = self.s; k = int(np.clip(np.floor(s), 0, 11))

        def pudelko(x, y, w, hh, tekst, kolor, aktywne=0.0, gotowe=False):
            r = QRectF(x, y, w, hh)
            c = QColor(kolor)
            if gotowe:
                c = c.darker(135)
            p.setBrush(c); p.setPen(QPen(QColor('#ffb000') if aktywne > 0 else QColor('#2a333a'), 3 if aktywne > 0 else 1))
            p.drawRoundedRect(r, 6, 6)
            if aktywne > 0:                                    # postęp sygnału w podkroku
                p.fillRect(QRectF(x + 2, y + hh - 6, (w - 4) * min(aktywne, 1.0), 4), QColor('#ffb000'))
            p.setPen(QColor('#101418')); p.drawText(r, int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), tekst)
            return r

        def strzalka(x0, x1, y):
            p.setPen(QPen(QColor('#cfd3d8'), 1.5)); p.drawLine(int(x0), int(y), int(x1), int(y))
            p.drawLine(int(x1), int(y), int(x1 - 5), int(y - 4)); p.drawLine(int(x1), int(y), int(x1 - 5), int(y + 4))

        x = 10; ym = y0 + h / 2
        # wejście: obraz → 4 konwolucje łodygi (krok 2) → + kodowanie pozycji; miniatury = mapy cech bieżącego checkpointu
        nazwy = ['obraz', 'konw. 1', 'konw. 2', 'konw. 3', 'konw. 4', '+ PE']
        p.setPen(QColor('#f5c6c6')); p.drawText(QRectF(x + szer, y0 - 18, 4 * szer * 0.95, 16), int(Qt.AlignmentFlag.AlignCenter),
                                                 'łodyga CNN (w każdym przejściu od pikseli)')
        for i, nz in enumerate(nazwy):
            r = QRectF(x, y0 + 4, szer * 0.86, h - 8)
            aktyw = (s == 0 and i == 5)
            p.setBrush(QColor('#1a2228')); p.setPen(QPen(QColor('#ffb000') if aktyw else QColor('#55606a'), 3 if aktyw else 1))
            p.drawRoundedRect(r, 5, 5)
            p.setPen(QColor('#e8e8e8')); p.drawText(QRectF(x, y0 + 4, szer * 0.86, 14), int(Qt.AlignmentFlag.AlignCenter), nz)
            if i < len(self._miniatury):
                if i == 5 and len(self._miniatury) > 6:
                    im = self._miniatury[6]
                else:
                    im = self._miniatury[i]
                obsz = QRectF(x + 3, y0 + 20, szer * 0.86 - 6, h - 30)
                sk = min(obsz.width() / im.width(), obsz.height() / im.height())
                ww, hh = im.width() * sk, im.height() * sk
                p.drawImage(QRectF(obsz.x() + (obsz.width() - ww) / 2, obsz.y() + (obsz.height() - hh) / 2, ww, hh), im)
            self._pola.append((r, 0.0, self._opisy[i] if i < len(self._opisy) else nz))
            x += szer
            if i < 5:
                strzalka(x - szer * 0.14, x - 1, ym)
        for b in range(6):
            strzalka(x - szer * 0.1, x, ym)
            p.setPen(QPen(QColor('#cfd3d8'), 1)); p.setBrush(QColor('#1a2228'))
            p.drawRoundedRect(QRectF(x, y0, szer * 0.9, h), 6, 6)
            p.setPen(QColor('#e8e8e8')); p.drawText(QRectF(x, y0 - 18, szer * 0.9, 16), int(Qt.AlignmentFlag.AlignCenter), f'blok {b + 1}')
            ku, km = 2 * b, 2 * b + 1
            am = (s - km) if k == km and s < 12 else 0
            au = (s - ku) if k == ku and s < 12 else 0
            # kolejność w czasie z góry na dół: najpierw uwaga, potem MLP; napis skrócony, gdy nie mieści się w jednej linii
            wn = szer * 0.9 - 8
            t_m = 'MLP\n768 laserów' if p.fontMetrics().horizontalAdvance('768 laserów') < wn - 4 else 'MLP\n768'
            ru = pudelko(x + 4, y0 + 4, wn, h / 2 - 6, 'uwaga\nsiły f_ij', '#c8f0d0', aktywne=max(au, 1e-3) if k == ku and s < 12 else 0,
                         gotowe=s >= ku + 1)
            rm = pudelko(x + 4, y0 + h / 2 + 2, wn, h / 2 - 6, t_m, '#bfe0f5', aktywne=max(am, 1e-3) if k == km and s < 12 else 0,
                         gotowe=s >= km + 1)
            self._pola += [(ru, float(ku), f'blok {b + 1}: uwaga'), (rm, float(km), f'blok {b + 1}: MLP')]
            x += szer
        strzalka(x - szer * 0.1, x, ym)
        r = pudelko(x, y0 + h * 0.2, szer * 0.8, h * 0.6, 'Layer-\nNorm', '#fde3c3', aktywne=1.0 if s >= 12 else 0)
        self._pola.append((r, 12.0, 'LayerNorm i głowa')); x += szer
        strzalka(x - szer * 0.2, x, ym)
        pudelko(x, y0 + h * 0.1, szer * 0.9, h * 0.8, 'głowa\nmapa +\nramka', '#fde3c3', aktywne=1.0 if s >= 12 else 0)
        p.end()

    def mousePressEvent(self, ev):
        for r, s, _ in self._pola:
            if r.contains(ev.position()):
                self.klik.emit(s); return

    def event(self, ev):
        from PyQt6.QtCore import QEvent, QPointF
        from PyQt6.QtWidgets import QToolTip
        if ev.type() == QEvent.Type.ToolTip:
            for r, _, opis in self._pola:
                if r.contains(QPointF(ev.pos())):
                    QToolTip.showText(ev.globalPos(), opis, self); return True
        return super().event(ev)


def _staly_uklad(lb):
    """Napis zmieniany w każdej klatce nie może zmieniać układu okna: inaczej Qt przelicza doki i zmienia rozmiar widoków OpenGL
    (ponowne bufory, przerwy ~100 ms). Szerokość z układu, nie z tekstu; wysokość stała (jedna linia)."""
    from PyQt6.QtWidgets import QSizePolicy
    lb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    lb.setFixedHeight(lb.fontMetrics().height() + 8)
    lb.setWordWrap(False)


class SuwakAB(QSlider):
    """Suwak osi czasu z zaznaczonym odcinkiem A–B (wartości suwaka; None = brak punktu)."""
    def __init__(self):
        super().__init__(Qt.Orientation.Horizontal)
        self.ab = [None, None]

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self.maximum() <= self.minimum() or self.ab == [None, None]:
            return
        from PyQt6.QtGui import QColor, QPainter
        from PyQt6.QtWidgets import QStyle, QStyleOptionSlider
        o = QStyleOptionSlider(); self.initStyleOption(o)
        uchwyt = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, o, QStyle.SubControl.SC_SliderHandle, self)
        x0, szer = uchwyt.width() / 2, self.width() - uchwyt.width()
        xx = lambda v: x0 + szer * (v - self.minimum()) / (self.maximum() - self.minimum())
        a = self.minimum() if self.ab[0] is None else self.ab[0]
        b = self.maximum() if self.ab[1] is None else self.ab[1]
        q = QPainter(self); h = self.height()
        q.fillRect(int(xx(min(a, b))), h // 2 - 5, max(1, int(xx(max(a, b)) - xx(min(a, b)))), 10, QColor(80, 200, 255, 70))
        for v, kol in ((self.ab[0], QColor(80, 200, 255)), (self.ab[1], QColor(255, 120, 90))):
            if v is not None:
                q.fillRect(int(xx(v)) - 1, 2, 3, h - 4, kol)
        q.end()


def lista_liczb(tekst, n):
    """'12, 40 100-105' → [12, 40, 100, …, 105] (bez powtórzeń, kolejność wpisania, tylko 0..n−1)."""
    out, byly = [], set()
    for kaw in tekst.replace(';', ',').replace(',', ' ').split():
        a, _, b = kaw.partition('-')
        try:
            lo = int(a); hi = int(b) if b else lo
        except ValueError:
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        for x in range(max(lo, 0), min(hi, n - 1) + 1):                 # zakres przycięty do 0..n−1 (np. „0-100000000” bez zawieszenia)
            if x not in byly:
                byly.add(x); out.append(x)
    return out


def _spin(lo, hi, v, krok=1, dec=None, sufiks=''):
    s = QDoubleSpinBox() if dec is not None else QSpinBox()
    if dec is not None:
        s.setDecimals(dec)
    s.setRange(lo, hi); s.setSingleStep(krok); s.setValue(v)
    if sufiks:
        s.setSuffix(sufiks)
    return s


class ZakladkaLaserow(QWidget):
    def __init__(self, okno=None):
        super().__init__()
        self.okno = okno
        self.h: LD.Historia | None = None
        self.t = 0.0
        self.krok_odtw = 0.0
        self.watek = None
        self._blokada = False
        self._buduj()
        self._lista_historii()
        self._lista_zdjec()
        self.timer = QTimer(self); self.timer.setInterval(16); self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._klatka)
        QTimer.singleShot(50, self._wczytaj_z_cache)

    # ------------------------------------------------------------------ UI
    def _buduj(self):
        """Układ: u góry pasek osi czasu, pod nim okienka (doki). Parametry nie zajmują już stałej kolumny: są w okienku
        „Parametry” (lista sekcji + jedna strona naraz), które można odpiąć jak każde okienko albo schować (F2), a ustawienia
        konkretnego okienka otwiera prawy klik w nie (panel_ustawien.PopupUstawien). Sekcje budują się jak dawniej
        (Zwijana + QFormLayout) i trafiają do self._sek; strony panelu składa _buduj_panel."""
        lay = QVBoxLayout(self); lay.setContentsMargins(2, 2, 2, 2); lay.setSpacing(2)
        pl_ = lay
        self._sek = {}

        wiersz_st = QHBoxLayout()
        self.ch_stereo = QCheckBox('Stereo 3D (SBS)')
        self.ch_stereo.setToolTip('Przełącznik 2D ↔ 3D dla wszystkich widoków OpenGL (sfera, R1–R7, dolina…): lewe i prawe oko obok siebie,\n'
                                  'każde zgniecione do połowy szerokości (half side-by-side) dla monitora 3D, np. Samsung Odyssey 3D.\n'
                                  'Okno z widokiem rozwiń na cały ekran monitora 3D (⛶ albo F11). Napisy 3D są w stereo ukryte, a klikanie\n'
                                  'w cząsteczki w trybie stereo nie jest dokładne (wyłącz stereo do zaznaczania).')
        self.sp_stereo = QDoubleSpinBox(); self.sp_stereo.setRange(0.0, 0.2); self.sp_stereo.setSingleStep(0.005); self.sp_stereo.setDecimals(3)
        self.sp_stereo.setValue(WidokGL.stereo_odstep); self.sp_stereo.setPrefix('odstęp ')
        self.sp_stereo.setToolTip('Odstęp oczu jako ułamek odległości kamery od środka widoku (tam jest zerowy rozrzut). Większy = głębsza scena.')
        self.ch_stereo_zam = QCheckBox('zamień oczy'); self.ch_stereo_zam.setToolTip('Gdy głębia jest odwrócona (przód i tył zamienione).')
        self.ch_stereo.toggled.connect(lambda on: WidokGL.ustaw_stereo(wlacz=on))
        self.sp_stereo.valueChanged.connect(lambda v: WidokGL.ustaw_stereo(odstep=v))
        self.ch_stereo_zam.toggled.connect(lambda on: WidokGL.ustaw_stereo(zamiana=on))
        for w_ in (self.ch_stereo, self.sp_stereo, self.ch_stereo_zam):
            wiersz_st.addWidget(w_)
        wiersz_st.addStretch(1)
        g = Zwijana('Stereo 3D (monitor 3D, wszystkie widoki)'); f = QFormLayout(g); f.addRow(wiersz_st)
        self._sek['stereo'] = g

        # --- dane
        g = Zwijana('Dane: historia uczenia i zdjęcie'); f = QFormLayout(g)
        self.cb_historia = WaskiCombo(); self.cb_historia.setToolTip('Katalog checkpointów k<krok>.pt (historia uczenia).')
        b = QPushButton('Inny katalog wag…'); b.clicked.connect(self._inny_katalog)
        f.addRow('Historia', self.cb_historia); f.addRow('', b)
        self.sp_od = _spin(0, 10 ** 7, 0, 100); self.sp_do = _spin(0, 10 ** 7, 10 ** 7, 100); self.sp_co = _spin(1, 100, 1)
        self.sp_od.setToolTip('Pierwszy krok uczenia (checkpointy z mniejszym krokiem są pomijane).')
        self.sp_do.setToolTip('Ostatni krok uczenia.')
        self.sp_co.setToolTip('Co który checkpoint brać (1 = wszystkie). Drgania wymagają gęstych checkpointów (okno ± kroków).')
        f.addRow('Od kroku', self.sp_od); f.addRow('Do kroku', self.sp_do); f.addRow('Co n-ty checkpoint', self.sp_co)
        self.cb_film = WaskiCombo()
        self.cb_film.setToolTip('Film zbioru uczącego biegu 35 000 (MANIFEST_35000: 117 filmów, 35 000 klatek z ramką narzędzia)\n'
                                'albo pliki z app/zdjeciatestowe (większość z nich NIE była w zbiorze uczącym).')
        self.cb_zdjecie = WaskiCombo(); b2 = QPushButton('Zdjęcie z pliku…'); b2.clicked.connect(self._plik_zdjecia)
        self.cb_zdjecie.setToolTip('Klatka wybranego filmu: obraz z archiwum zip (SHA-256 z manifestu), rozciągnięty do 480 × 320\n'
                                   'i ramka z manifestu, dokładnie jak w treningu. „nr” = wiersz manifestu (0–34 999).')
        f.addRow('Film', self.cb_film); f.addRow('Zdjęcie', self.cb_zdjecie); f.addRow('', b2)
        self.cb_film.currentIndexChanged.connect(self._klatki_filmu)
        self.lb_gt = QLabel('—'); self.lb_gt.setWordWrap(True); f.addRow('Ramka', self.lb_gt)
        self.sp_okno = _spin(10, 20000, 300, 50, sufiks=' kroków')
        self.sp_okno.setToolTip('Centrum studni = średnie położenie tokenu z ± tylu kroków (film 2: ±300).')
        f.addRow('Okno centrów ±', self.sp_okno)
        hb = QHBoxLayout(); self.bt_licz = QPushButton('Policz / wczytaj'); self.bt_stop = QPushButton('Przerwij')
        self.bt_licz.clicked.connect(self._licz); self.bt_stop.clicked.connect(self._przerwij); self.bt_stop.setEnabled(False)
        hb.addWidget(self.bt_licz); hb.addWidget(self.bt_stop); f.addRow(hb)
        self.pasek = QProgressBar(); self.pasek.setTextVisible(True); f.addRow(self.pasek)
        self.lb_stan = QLabel('Wybierz historię i zdjęcie, potem „Policz / wczytaj”. Wynik zostaje w cache_uczenia/lasery.')
        self.lb_stan.setWordWrap(True); f.addRow(self.lb_stan)
        self._sek['dane'] = g
        for w in (self.cb_historia, self.cb_zdjecie):
            w.currentIndexChanged.connect(self._po_wyborze)
        for w in (self.sp_od, self.sp_do, self.sp_co, self.sp_okno):
            w.valueChanged.connect(self._po_wyborze)

        # --- rzuty z 192 wymiarów (okienka R1–R8, rzuty_okienka.py)
        self.rzuty = RZO.Rzuty(self)
        self._sek['rzuty'] = self.rzuty

        # --- rzuty 2D
        g = Zwijana('Okienka 1–3: rzut płaski (film 2)'); f = QFormLayout(g)
        self.sp_rozm = _spin(1, 20, 3.0, 0.5, 1); self.sp_jas = _spin(0.05, 5, 1.0, 0.1, 2); self.sp_sig = _spin(0.05, 2, 0.26, 0.02, 2)
        self.sp_strz = _spin(0.1, 5, 1.0, 0.1, 1); self.sp_slad = _spin(0, 20000, 200, 50); self.sp_zbl = _spin(2, 30, 7.2, 0.5, 1)
        self.ch_kamera = QCheckBox('kamera okienka 3 idzie za narzędziem'); self.ch_kamera.setChecked(True)
        self.ch_narz2 = QCheckBox('pierścienie na narzędziu'); self.ch_narz2.setChecked(True)
        for nazwa, w, dymek in (('Wielkość tokenów [px]', self.sp_rozm, ''), ('Jasność studni', self.sp_jas, ''),
                                ('Szerokość studni σ [kom.]', self.sp_sig, 'Gauss studni w komórkach siatki (film: 0,26).'),
                                ('Skala strzałek', self.sp_strz, 'Strzałka adres → centrum mnożona przez tę liczbę.'),
                                ('Ślad drgań [kroki]', self.sp_slad, 'Długość śladu atomu w okienku 3 w krokach uczenia (film 16: 200).'),
                                ('Zbliżenie okienka 3 [kom.]', self.sp_zbl, 'Szerokość kadru okienka 3 w komórkach siatki.')):
            w.setToolTip(dymek); f.addRow(nazwa, w); w.valueChanged.connect(self._opcje)
        f.addRow(self.ch_kamera); f.addRow(self.ch_narz2)
        self.ch_kamera.toggled.connect(self._opcje); self.ch_narz2.toggled.connect(self._opcje)
        self._sek['plaskie'] = g

        # --- sfera
        g = Zwijana('Sfera reflektorów (film 12)'); f = QFormLayout(g)
        self.ch_swiatlo = QCheckBox('reflektory świecą (wiązki i linie uwagi)'); self.ch_swiatlo.setChecked(True)
        self.ch_swiatlo.setToolTip('Odznaczone: gasną wszystkie wiązki laserów i linie uwagi na sferze; wybór „Wiązki” zostaje zapamiętany.')
        self.ch_swiatlo.toggled.connect(self._opcje)
        f.addRow(self.ch_swiatlo)
        self.cb_wiazki = WaskiCombo(); self.cb_wiazki.addItems(['brak', 'wszystkie (768 × 600)', 'do wybranych cząsteczek', 'z wybranych reflektorów'])
        self.cb_wiazki.setCurrentIndex(1)
        self.sp_gain = _spin(0.0005, 20, 0.05, 0.01, 4); self.sp_prog = _spin(0, 5, 0.02, 0.01, 3)
        self.sp_gain.setToolTip('Jasność wiązki = z_jc / 2 × wzmocnienie (film 12: 0,05).')
        self.sp_eksp = _spin(0.05, 20, 1.0, 0.1, 2)
        self.sp_grub_w = _spin(0.5, 10, 1.0, 0.5, 1, ' px'); self.sp_grub_w.setToolTip('Grubość linii wiązek laserów i linii uwagi na sferze.')
        self.ch_kier = QCheckBox('wypadkowy kierunek wybranych reflektorów (średnica d_c)'); self.ch_kier.setChecked(True)
        self.ch_kier.setToolTip('Linia przez całą sferę wzdłuż d_c: kierunek, w który pchają wszystkie wiązki wybranych reflektorów '
                                '(tylko przy „Wiązki: z wybranych reflektorów”).')
        self.ch_sfera = QCheckBox(); self.ch_sfera.setChecked(True); self.ch_sfera.setToolTip('Pokaż siatkę szklanej sfery.')
        self.sp_eksp.setToolTip('Światło wiązek sumowane w buforze float i tonowane 1 − exp(−ekspozycja · światło), jak w filmie 12.')
        self.sp_prog.setToolTip('Wiązki ciemniejsze niż próg nie są rysowane (mniej szumu, szybciej).')
        self.cb_skala = WaskiCombo(); self.cb_skala.addItems(['chmura dopasowana do kroku', 'stała skala (z końca)'])
        self.sp_tok = _spin(0, 599, 0); self.sp_refl = _spin(0, 767, 0)
        bt_tok = QPushButton('narzędzie'); bt_tok.setToolTip('Cząsteczka w środku ramki narzędzia.'); bt_tok.clicked.connect(self._tok_narzedzia)
        bt_refl = QPushButton('najzmienniejszy'); bt_refl.setToolTip('Reflektor o największej wariancji natężeń na końcu.')
        bt_refl.clicked.connect(self._refl_pokaz)
        h1 = QHBoxLayout(); h1.addWidget(self.sp_tok); h1.addWidget(bt_tok)
        h2 = QHBoxLayout(); h2.addWidget(self.sp_refl); h2.addWidget(bt_refl)
        self.ed_tok = QLineEdit(); self.ed_refl = QLineEdit()
        self.ed_tok.setPlaceholderText('np. 12, 40, 100-105')
        self.ed_refl.setPlaceholderText('np. 51, 121, 300-310')
        self.ed_tok.setToolTip('Dodatkowe wybrane cząsteczki (oprócz tej powyżej): liczby po przecinku albo spacji, zakres a-b. '
                               'Działa w „Wiązki: do wybranych cząsteczek”, „Wektory dla: wybranych cząsteczek” i liniach uwagi.')
        self.ed_refl.setToolTip('Dodatkowe wybrane reflektory (oprócz tego powyżej): liczby po przecinku albo spacji, zakres a-b. '
                                'Wiązki „z wybranych reflektorów” świecą ze wszystkich, a mapa „natężenie wybranych reflektorów” '
                                'w okienku 5 pokazuje łączne oświetlenie Σ_c max(z_jc, 0).')
        for e_ in (self.ed_tok, self.ed_refl):
            e_.editingFinished.connect(self._opcje)
        self.sp_sfera = _spin(0, 1, 0.10, 0.02, 2); self.ch_refl = QCheckBox('pokaż reflektory'); self.ch_refl.setChecked(True)
        self.sp_rozm3 = _spin(1, 30, 5, 0.5, 1, ' px'); self.sp_rozm3.setToolTip('Wielkość punktu koloidu (tokenu) na ekranie.')
        self.sp_rozm_pul = _spin(1, 80, 13, 1, 1, ' px'); self.sp_rozm_pul.setToolTip('Średnica poświaty średniej skoków koloidu na ekranie.')
        self.sp_jas_pul = _spin(0.05, 3, 0.55, 0.05, 2); self.sp_jas_pul.setToolTip('Jasność poświaty średnich skoków.')
        self.sp_skala_chm = _spin(0.1, 1.8, 1.0, 0.05, 2, ' ×')
        self.sp_skala_chm.setToolTip('Rozmiar całej chmury koloidów i średnich skoków względem szklanej sfery (1 = jak film 12: 99 % chmury w 0,55 promienia). '
                                     'Wiązki, punkty załamania, wektory i drogi przeliczają się razem z chmurą.')
        self.cb_widok_s = WaskiCombo(); self.cb_widok_s.addItems(['koloidy (tokeny)', 'średnie skoków koloidów (±okno)', 'koloidy i średnie skoków'])
        self.cb_widok_s.setToolTip('Średnia skoków koloidu = jego średnie położenie z ± okna kroków (jak centra studni w filmie 2; dawniej „pułapka”), '
                                   'w tych samych współrzędnych sfery. Liczona z ruchu koloidu i z przyszłych kroków, więc NIE jest przyczyną ruchu. '
                                   'Pułapka optyczna z laserów: sekcja „Pułapka optyczna” i okienko 11. W trybie podkroków średnia skoków = stan cząsteczki.')
        WEK = ['brak', 'z kroku na krok', 'od kroku 0 (adres startu)', 'cała historia: droga od kroku 0']
        self.cb_wek_kol = WaskiCombo(); self.cb_wek_kol.addItems(WEK)
        self.cb_wek_pul = WaskiCombo(); self.cb_wek_pul.addItems(WEK)
        for c_ in (self.cb_wek_kol, self.cb_wek_pul):
            c_.setToolTip('Z kroku na krok: od położenia w poprzednim checkpoincie (w podkrokach: poprzednim podkroku) do teraz. '
                          'Od kroku 0: od położenia w pierwszym checkpoincie (adres startu) do teraz. Cała historia: droga przez '
                          'wszystkie checkpointy od 0 do teraz. Koloidy żółte, średnie skoków zielone.')
        self.cb_wek_dla = WaskiCombo(); self.cb_wek_dla.addItems(['wszystkich cząsteczek', 'narzędzia', 'wybranych cząsteczek'])
        self.cb_wek_dla.setCurrentIndex(1)
        self.sp_skala_wek = _spin(0.1, 50, 1.0, 0.5, 1, ' ×'); self.sp_skala_wek.setToolTip('Długość wektorów (droga zawsze w skali 1).')
        for nazwa, w in (('Wiązki', self.cb_wiazki), ('Grubość wiązek', self.sp_grub_w), ('Wzmocnienie', self.sp_gain), ('Ekspozycja światła', self.sp_eksp), ('Próg', self.sp_prog),
                         ('Skala chmury', self.cb_skala)):
            f.addRow(nazwa, w)
        self._wiersze_wyboru = (('Cząsteczka (patch)', h1), ('  + cząsteczki', self.ed_tok), ('Reflektor (neuron)', h2),
                                ('  + reflektory', self.ed_refl))                # osobna sekcja „wybor” (_buduj_wybor)
        self.cb_obrot = WaskiCombo(); self.cb_obrot.addItems(['brak', 'dryf netto (2000 kroków)', 'drganie (100 kroków)', 'dryf i drganie'])
        self.cb_obrot.setToolTip('Gdzie w układzie jest gradient (film 23). Przy każdym reflektorze igła = kierunek d_c, w który pchają jego wiązki; '
                                 'strzałka na końcu igły = dokąd i jak szybko igła się obraca w czasie uczenia. Kierunek = rzut obrotu na 3D, '
                                 'długość = prędkość kątowa v̂_c w 192 wymiarach (°/100 kroków, skala log). Dryf netto: z checkpointów odległych '
                                 'o ≥ 2000 kroków (kolor plasma, grot); drganie: o ≥ 100 kroków (biała kreska). To rzeczywisty krok uczenia '
                                 '(gradient Σ_j z_jc δ_j przez AdamW); weight decay tylko skraca v_c, nie obraca go. Wybrany reflektor grubo.')
        self.sp_dl_obr = _spin(0.2, 5, 1.0, 0.1, 1, ' ×'); self.sp_dl_obr.setToolTip('Długość igieł i strzałek obrotu.')
        h3 = QHBoxLayout(); h3.addWidget(self.cb_obrot, 1); h3.addWidget(self.sp_dl_obr)
        f.addRow('Strzałki obrotu', h3); f.addRow(self.ch_kier)
        self.cb_obrot.currentIndexChanged.connect(self._opcje); self.sp_dl_obr.valueChanged.connect(self._opcje)
        h4 = QHBoxLayout(); h4.addWidget(self.ch_sfera); h4.addWidget(self.sp_sfera, 1)
        self.ch_sfera.toggled.connect(self.sp_sfera.setEnabled)
        f.addRow('Sfera (krycie)', h4); f.addRow('Wielkość koloidów', self.sp_rozm3); f.addRow('Wielkość średnich skoków', self.sp_rozm_pul)
        f.addRow('Jasność średnich skoków', self.sp_jas_pul); f.addRow('Skala chmury w sferze', self.sp_skala_chm); f.addRow(self.ch_refl)
        for w_ in (self.sp_rozm_pul, self.sp_jas_pul, self.sp_skala_chm):
            w_.valueChanged.connect(self._opcje)
        f.addRow('Pokazuj', self.cb_widok_s); f.addRow('Wektory koloidów', self.cb_wek_kol); f.addRow('Wektory średnich skoków', self.cb_wek_pul)
        f.addRow('Wektory dla', self.cb_wek_dla); f.addRow('Skala wektorów', self.sp_skala_wek)
        self.cb_kol_sciezki = WaskiCombo()
        self.cb_kol_sciezki.addItems(['jednolity (koloidy żółte, średnie skoków zielone)', 'kolor patcha ze zdjęcia', 'położenie patcha na obrazie (siatka 30 × 20: barwa = kolumna, jasność = wiersz)', 'narzędzie',
                                      'odpływ od adresu (jak okienko 2, plasma)', 'obszary oka: narzędzie, źrenica, rogówka, tło'])
        self.cb_kol_sciezki.setToolTip('Kolor drogi i wektorów koloidów i średnich skoków. „Odpływ od adresu”: każdy punkt drogi i koniec wektora ma kolor '
                                       'według odległości od położenia w kroku 0 (adres), skala plasma jak strzałki w okienku 2: '
                                       'ciemny fiolet = przy adresie, żółty = 99. percentyl odpływu.')
        self.cb_wyg = WaskiCombo(); self.cb_wyg.addItems(['brak', 'średnia ważona krocząca', 'EMA', 'średnia krocząca + EMA'])
        self.cb_wyg.setToolTip('Wygładzanie średnich skoków wzdłuż checkpointów (tylko przeszłość, wagi rosnące ku teraźniejszości / wykładnicze). '
                               'Dotyczy drogi, wektorów i poświaty średnich skoków. Nakłada się na już symetryczną średnią ±okno, więc przyszłe kroki zostają w środku. Uwaga: odstępy checkpointów to 10 kroków do kroku 1000, potem 100.')
        self.sp_wyg_okno = _spin(2, 300, 10, 1, sufiks=' ckpt'); self.sp_wyg_okno.setToolTip('Okno średniej ważonej (wagi okno, okno−1, …, 1).')
        self.sp_wyg_alfa = _spin(0.005, 1.0, 0.2, 0.05, 3); self.sp_wyg_alfa.setToolTip('EMA: y = α·x + (1 − α)·y_poprzednie (na checkpoint).')
        self.sp_zanik = _spin(0, 70000, 0, 100, sufiks=' kroków'); self.sp_zanik.setSpecialValueText('bez zaniku')
        self.sp_zanik.setToolTip('Droga gaśnie z wiekiem: jasność exp(−3·wiek/zanik), wiek w krokach uczenia (w podkrokach: w podkrokach). 0 = bez zaniku.')
        f.addRow('Kolor ścieżek', self.cb_kol_sciezki); f.addRow('Wygładzanie średnich skoków', self.cb_wyg)
        f.addRow('Okno średniej', self.sp_wyg_okno); f.addRow('EMA α', self.sp_wyg_alfa); f.addRow('Zanik historii', self.sp_zanik)
        for w_ in (self.cb_kol_sciezki, self.cb_wyg):
            w_.currentIndexChanged.connect(self._opcje)
        for w_ in (self.sp_wyg_okno, self.sp_wyg_alfa, self.sp_zanik):
            w_.valueChanged.connect(self._opcje)
        for w_ in (self.cb_widok_s, self.cb_wek_kol, self.cb_wek_pul, self.cb_wek_dla):
            w_.currentIndexChanged.connect(self._opcje)
        self.sp_skala_wek.valueChanged.connect(self._opcje)
        for w in (self.cb_wiazki, self.cb_skala):
            w.currentIndexChanged.connect(self._opcje)
        for w in (self.sp_gain, self.sp_eksp, self.sp_prog, self.sp_tok, self.sp_refl, self.sp_sfera, self.sp_rozm3, self.sp_grub_w):
            w.valueChanged.connect(self._opcje)
        for w in (self.ch_kier, self.ch_sfera):
            w.toggled.connect(self._opcje)
        self.ch_refl.toggled.connect(self._opcje)
        self._sek['sfera'] = g

        # --- zapis przejść przez sieć (podkroki) na dysk
        g = Zwijana('Zapis przejść przez sieć (podkroki) na dysk'); f = QFormLayout(g)
        self.ch_prz_zapis = QCheckBox('zapisuj policzone przejścia (potem wczytują się zamiast liczenia)'); self.ch_prz_zapis.setChecked(True)
        self.ch_prz_zapis.setToolTip('Każde policzone przejście przez sieć (13 stanów, uwaga, jasności laserów) trafia do katalogu historii '
                                     '(cache_uczenia/lasery/<historia>/przejscia). Zapisane wczytują się zawsze, także gdy to pole jest wyłączone. '
                                     'Wspólny rzut 3D jest w baza.npz, więc zapisane przejścia pasują do siebie także między sesjami. '
                                     'Ogniskowe w osi uczenia (okienko 11) zapisują się osobno i zawsze.')
        self.cb_prz_dokl = WaskiCombo(); self.cb_prz_dokl.addItems(['dokładnie: float32, około 44 MB na checkpoint',
                                                                   'oszczędnie: float16, około 22 MB na checkpoint'])
        self.cb_prz_dokl.setToolTip('Dokładnie: kontrola „suma wkładów = ruch” jak przy liczeniu (~10⁻⁶). Oszczędnie: atencja, jasności i stany '
                                    'w float16, kontrola ~10⁻³.')
        self.sp_prz_limit = _spin(1, 2000, 40, 5, 0, ' GB'); self.sp_prz_limit.setToolTip('Po przekroczeniu tej wielkości katalogu nowe przejścia się nie zapisują.')
        self.sp_prz_co = _spin(1, 200, 1, 1, sufiks=' ckpt'); self.sp_prz_co.setToolTip('Zapis wszystkich: co ile checkpointów (ostatni zawsze).')
        hb_ = QHBoxLayout(); self.bt_prz = QPushButton('Zapisz przejścia wszystkich checkpointów'); self.bt_prz_stop = QPushButton('Przerwij')
        self.bt_prz.clicked.connect(self._prz_wszystkie); self.bt_prz_stop.clicked.connect(self._prz_przerwij); self.bt_prz_stop.setEnabled(False)
        hb_.addWidget(self.bt_prz); hb_.addWidget(self.bt_prz_stop)
        self.pasek_prz = QProgressBar(); self.pasek_prz.setTextVisible(True)
        self.lb_prz = QLabel('—'); self.lb_prz.setWordWrap(True)
        f.addRow(self.ch_prz_zapis); f.addRow('Zapis', self.cb_prz_dokl); f.addRow('Limit katalogu', self.sp_prz_limit)
        f.addRow('Wszystkie: co', self.sp_prz_co); f.addRow(hb_); f.addRow(self.pasek_prz); f.addRow(self.lb_prz)
        self._prz_watek = None; self._prz_bajty = 0
        self._sek['zapis'] = g

        # --- pułapka optyczna (ogniskowa wiązek bloku 6)
        g = Zwijana('Pułapka optyczna: ogniskowa wiązek wokół koloidu (okienko 11)'); f = QFormLayout(g)
        self.ch_og = QCheckBox('pokaż podgląd ogniskowej wybranej cząsteczki (wczytaj albo policz)'); self.ch_og.setChecked(False)
        self.ch_og.setToolTip('Pole wszystkich 768 wiązek bloku 6 wokół wybranej cząsteczki (sekcja sfery: „Cząsteczka”), w przekroju 3D '
                              'rzutu sfery, w każdym checkpoincie, liczone z wag (app/ogniskowa.py). Pułapka = punkt, w którym pchnięcia '
                              'wszystkich wiązek się znoszą (zero pola). Liczy się w tle i zostaje obok historii w cache. '
                              'Po uruchomieniu programu wyłączone: nic się nie wczytuje ani nie liczy, dopóki tego nie zaznaczysz '
                              '(albo nie klikniesz „Wczytaj / policz”).')
        self.cb_og_czas = WaskiCombo(); self.cb_og_czas.addItems(['oś uczenia: blok 6 w każdym checkpoincie',
                                                                  'podkroki: przejście przez 6 bloków (suwak podkroków)'])
        self.cb_og_czas.setToolTip('Oś uczenia: ognisko 768 wiązek bloku 6 w każdym checkpoincie (skok cząsteczki w bloku 6). Podkroki: '
                                   'w podkrokach MLP ognisko 768 wiązek danego bloku, w podkrokach uwagi 600 wkładów od innych cząsteczek '
                                   '(tam cząsteczkę przesuwają inne cząsteczki, nie lasery); w każdym podkroku suma wkładów = prawdziwy ruch.')
        self.ch_og_pole = QCheckBox('licz też lokalne pole na siatce i szukaj zera (wolniej)'); self.ch_og_pole.setChecked(False)
        self.ch_og_pole.setToolTip('Bez tego liczone są tylko wiązki wybranej cząsteczki. Z tym: pole na siatce G³, zero pola (Newton, jakobian\n'
                                   'analityczny) i kształt pola w koloidzie: około 0,07 s na checkpoint przy siatce 17 (RTX 3070), czyli około\n'
                                   'minuty na 806 checkpointów; każda dodatkowa cząsteczka „Licz razem z” około 0,04 s na checkpoint.')
        self.sp_og_G = _spin(9, 33, 17, 2); self.sp_og_G.setToolTip('Węzłów siatki pola na bok (G³ punktów).')
        self.cb_og_lupa = QComboBox(); self.cb_og_lupa.addItems(['lupa: kula pola = ułamek długości wiązek', 'prawdziwa skala (bez lupy)'])
        self.cb_og_lupa.setToolTip('Podkroki: komórka tokenu ma tylko 1–15 % długości wiązek, więc pole w prawdziwej skali chowa się w jasnym\n'
                                   'środku ogniska. Lupa pokazuje kulę komórki powiększoną wokół cząsteczki do wybranego ułamka długości\n'
                                   'wiązek (powiększenie w opisie okienka); kierunki i proporcje strzałek bez zmian.')
        self.sp_og_lupa = _spin(0.1, 1.0, 0.6, 0.05, 2, ' × wiązek')
        self.sp_og_lupa.setToolTip('Promień pokazanej kuli pola (z lupą) jako ułamek długości wiązek okienka 11.')
        for w_ in (self.cb_og_lupa, self.sp_og_lupa):
            (w_.currentIndexChanged if w_ is self.cb_og_lupa else w_.valueChanged).connect(lambda *_: self._og_opcje())
        self.ed_og_razem = QLineEdit(); self.ed_og_razem.setPlaceholderText('np. 12, 101 (najwyżej 8)')
        self.ed_og_razem.setToolTip('Pułapki tych cząsteczek liczą się RAZEM z wybraną, w jednym przejściu przez checkpointy: wczytanie wag\n'
                                    'i przejście sieci są wspólne, osobno liczy się tylko pole w komórce każdej cząsteczki. Trzy cząsteczki\n'
                                    'zajmują około 1,7 × czasu jednej. Każda zostaje w osobnym pliku obok historii, więc przełączenie\n'
                                    '„Cząsteczka” na którąś z nich (albo „▾ policzone”) pokazuje jej pułapkę od razu, bez liczenia.')
        self.bt_og_razem = QToolButton(); self.bt_og_razem.setText('▾ policzone')
        self.bt_og_razem.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.bt_og_razem.setToolTip('Przejście do cząsteczki, której pułapka jest już policzona (z tej listy i wybrana), z tą samą siatką i zasięgiem.')
        m_ = QMenu(self.bt_og_razem); m_.aboutToShow.connect(lambda m_=m_: self._og_menu_policzone(m_)); self.bt_og_razem.setMenu(m_)
        self.sp_og_zas = _spin(0.05, 1.0, 1.0, 0.05, 2, ' × komórki')
        self.sp_og_zas.setToolTip('Pole liczy się tylko w komórce tokenu: połowa boku sześcianu = ułamek r_kom, gdzie r_kom = połowa odległości '
                                  'do najbliższego innego tokenu w rzucie 3D (w każdym checkpoincie). Przy 1 żaden inny token nie leży w sześcianie. '
                                  'Zero pola (pułapka) jest szukane tylko w środku komórki; siatka nie rośnie.')
        hb = QHBoxLayout(); self.bt_og = QPushButton('Wczytaj / policz'); self.bt_og_stop = QPushButton('Przerwij')
        self.bt_og.clicked.connect(lambda: self._og_zlec(wymus=True) if self.ch_og.isChecked() else self.ch_og.setChecked(True))
        self.bt_og_stop.clicked.connect(self._og_przerwij)
        self.bt_og_stop.setEnabled(False); hb.addWidget(self.bt_og); hb.addWidget(self.bt_og_stop)
        self.pasek_og = QProgressBar(); self.pasek_og.setTextVisible(True)
        self.bt_pom = QPushButton('Policz pułapkę w podkrokach dla całej historii → plik'); self.bt_pom_stop = QPushButton('Przerwij')
        self.bt_pom.setToolTip('Dla wybranej cząsteczki, w każdym checkpoincie (co n) i każdym z 12 podkroków: skok, komórka, pole w komórce\n'
                               'i jego część liniowa J (sztywność κ w 3 osiach, dywergencja, wir, gdzie jest zero pola i czy pułapka trzyma),\n'
                               'kształt ogniska (tensor kierunków, wydłużenie, koherencja wiązek), ile laserów świeci, ilu tokenów słucha.\n'
                               'Zapis: katalog historii / pulapka_podkrokow_t<cząsteczka>_G<siatka>.npz i .csv (pulapka_obliczenia.py).')
        self.sp_pom_co = _spin(1, 200, 1, 1, sufiks=' ckpt'); self.sp_pom_co.setToolTip('Co ile checkpointów liczyć (ostatni zawsze).')
        self.sp_pom_G = _spin(3, 17, 5, 2); self.sp_pom_G.setToolTip('Węzłów siatki pola na bok do dopasowania J (5 wystarcza: pole jest liniowe, R² > 0,99).')
        self.pasek_pom = QProgressBar(); self.pasek_pom.setTextVisible(True); self.lb_pom = QLabel(''); self.lb_pom.setWordWrap(True)
        self.bt_pom_stop.setEnabled(False)
        self.bt_pom.clicked.connect(self._pomiary_start); self.bt_pom_stop.clicked.connect(self._pomiary_stop)
        self.ch_og_wiazki = QCheckBox('768 wiązek zbiegających się w cząsteczce (zbliżenie)'); self.ch_og_wiazki.setChecked(True)
        self.ch_og_wiazki.setToolTip('Każdy reflektor wysyła do wybranej cząsteczki jedną wiązkę z kierunku −d_c (d_c = P v_c w rzucie sfery), '
                                     'w kolorze reflektora; błękit = ujemna jasność. Wszystkie 768 wiązek zbiega się w cząsteczce: to jej ognisko.')
        self.cb_og_jasn = WaskiCombo(); self.cb_og_jasn.addItems(['wkład do skoku: wiązka z kierunku, w który pcha, jasność |z − z̄|·|P v|', 'jak na sferze: kierunek v_c, jasność z, błękit = ujemna dioda'])
        self.cb_og_jasn.setToolTip('Wkład do skoku liczony względem średniej po tokenach, jak położenia na sferze: suma wkładów = rysowany skok (Z20).')
        self.sp_og_gw = _spin(0.05, 20, 0.6, 0.1, 2, ' ×'); self.sp_og_gw.setToolTip('Jasność wiązek (najjaśniejsza = 1).')
        self.sp_og_dlw = _spin(0.1, 10, 1.0, 0.1, 1, ' ×'); self.sp_og_dlw.setToolTip('Długość wiązek w zbliżeniu.')
        self.cb_og_wk = WaskiCombo(); self.cb_og_wk.addItems(['wkłady: brak', 'wkłady: strzałki od cząsteczki', 'wkłady: łańcuch głowa–ogon (koniec = skok)'])
        self.cb_og_wk.setCurrentIndex(1)
        self.cb_og_wk.setToolTip('Wkład wiązki c do skoku = (z_jc − z̄_c)·P v_c. Strzałki: każdy wkład od cząsteczki (długości przeskalowane, '
                                 'kierunki prawdziwe). Łańcuch: wkłady dodawane jeden za drugim, od najdłuższego; koniec łańcucha to dokładnie skok, '
                                 'a długość drogi pokazuje, ile pchnięć się znosi (Z18).')
        self.sp_og_dlwk = _spin(0.1, 10, 1.0, 0.1, 1, ' ×'); self.sp_og_dlwk.setToolTip('Długość strzałek wkładów.')
        self.ch_og_sw = QCheckBox('światło ogniska (wiązki gaussowskie)'); self.ch_og_sw.setChecked(True)
        self.ch_og_sw.setToolTip('Każda wiązka to wiązka gaussowska z przewężeniem w cząsteczce, w kolorze reflektora; światło wszystkich wiązek '
                                 'sumuje się i daje kształt ogniska. Ognisko jest wydłużone wzdłuż kierunku, z którego przychodzi najwięcej światła '
                                 '(osie: białe odcinki; wartości w opisie okienka).')
        self.sp_og_jsw = _spin(0.01, 20, 0.35, 0.05, 2, ' ×'); self.sp_og_jsw.setToolTip('Jasność światła ogniska.')
        self.sp_og_szer = _spin(0.002, 0.3, 0.02, 0.005, 3, ' × L'); self.sp_og_szer.setToolTip('Przewężenie wiązki w ognisku (ułamek długości wiązki).')
        self.sp_og_rozb = _spin(0.02, 1.0, 0.15, 0.01, 2, ' rad'); self.sp_og_rozb.setToolTip('Kąt rozbieżności wiązki: jak szybko rozszerza się poza ogniskiem.')
        self.ch_og_linie = QCheckBox('cienkie linie wiązek'); self.ch_og_linie.setChecked(True)
        self.ch_og_refl = QCheckBox('reflektory: wiązki od reflektorów na sferze, skala jak sfera (film 12)'); self.ch_og_refl.setChecked(False)
        self.ch_og_refl.setToolTip('Okienko 11 w skali sfery reflektorów: 768 reflektorów na zewnętrznej sferze (kolory jak w filmie 12), '
                                   'szklana sfera i wiązka każdego reflektora do wybranej cząsteczki: reflektor → punkt załamania na szkle → '
                                   'cząsteczka (ostatni odcinek wzdłuż kierunku wiązki). Kamera przejmuje odległość i kąt widoku sfery i przestaje '
                                   'śledzić cząsteczkę. W podkrokach uwagi źródłami są inne cząsteczki.')
        self.sp_og_jrefl = _spin(0.05, 20, 1.0, 0.1, 2, ' ×'); self.sp_og_jrefl.setToolTip('Jasność wiązek od reflektorów.')
        self.ch_og_osieo = QCheckBox('osie kształtu ogniska'); self.ch_og_osieo.setChecked(True)
        self.ch_og_obrys = QCheckBox('obrys ogniska 3D (powierzchnia równej jasności światła)'); self.ch_og_obrys.setChecked(False)
        self.ch_og_obrys.setToolTip('Powierzchnia, na której suma światła wszystkich wiązek gaussowskich spada do wybranego ułamka jasności '
                                    'w cząsteczce (jak objętość ogniska w optyce). Kształt pochodzi z danych: z kierunków i jasności wiązek; '
                                    'ognisko wydłuża się wzdłuż kierunku, z którego przychodzi najwięcej światła. Rozmiar zależy od przewężenia '
                                    'i rozbieżności wiązek (parametry rysunku), więc porównywać można kształty, nie rozmiary bezwzględne.')
        self.sp_og_poz = _spin(0.02, 0.98, 0.5, 0.05, 2, ' szczytu')
        self.sp_og_poz.setToolTip('Poziom obrysu: 0,5 = połowa jasności w cząsteczce (FWHM), 0,135 = 1/e².')
        self.sp_og_jobr = _spin(0.02, 1.0, 0.15, 0.05, 2); self.sp_og_jobr.setToolTip('Krycie powierzchni obrysu.')
        self.sp_og_pobr = _spin(1, 200, 20, 1, 0, ' ×')
        self.sp_og_pobr.setToolTip('Lupa: obrys rysowany powiększony wokół cząsteczki (prawdziwy ma około 1–2 przewężeń wiązki, czyli kilka '
                                   'pikseli). Kształt i proporcje bez zmian; 1 = prawdziwy rozmiar. Warstwice: złote = przekroje wzdłuż najdłuższej '
                                   'osi obrysu, białe = wzdłuż drugiej osi.')
        self.ch_og_mgla = QCheckBox('lokalne pole: mgła (kolor = wiązki)'); self.ch_og_mgla.setChecked(False)
        self.ch_og_mgla.setToolTip('Punkt siatki ma kolor wiązek o największym wkładzie |z_c − z̄_c|·|P v_c| (kolory reflektorów jak na sferze, '
                                   'mieszanka trzech największych; nasycenie = udział największej). Jasny tam, gdzie pchnięcia się znoszą: '
                                   'exp(−|f|² / 2σ²), σ = ostrość × mediana |f| na siatce.')
        self.sp_og_jas = _spin(0.005, 5, 0.25, 0.02, 3); self.sp_og_ostr = _spin(0.02, 3, 0.35, 0.05, 2)
        self.sp_og_tlo = _spin(0, 1, 0.05, 0.01, 2); self.sp_og_rozm = _spin(2, 120, 22, 1, 0, ' px')
        self.sp_og_ostr.setToolTip('Ostrość ogniska: σ = ostrość × mediana |f|. Mniej = węższe, jaśniejsze ognisko.')
        self.sp_og_tlo.setToolTip('Jasność pola poza ogniskiem (0 = widać tylko ognisko).')
        self.sp_og_rozm.setToolTip('Rozmycie: średnica plamki każdego punktu siatki na ekranie.')
        self.cb_og_strz = WaskiCombo(); self.cb_og_strz.addItems(['brak', 'pole wiązek (pełne pchnięcie)', 'kształt pułapki (pole względem skoku koloidu)'])
        self.cb_og_strz.setCurrentIndex(1)                                   # widać od razu po policzeniu pola (oś uczenia)
        self.cb_og_strz.setToolTip('Pełne pole: dokąd wiązki pchają cząstkę postawioną w danym punkcie. Kształt pułapki: to samo minus skok '
                                   'samego koloidu, czyli jak wiązki rozpychają albo ściągają i obracają otoczenie koloidu (siodło z wirem).')
        self.sp_og_strz = _spin(0.1, 10, 0.8, 0.1, 1, ' ×'); self.sp_og_pods = _spin(1, 8, 3)
        self.sp_og_strz.setToolTip('Długość strzałek (1 = typowa strzałka długości odstępu między strzałkami).')
        self.ch_og_plyn = QCheckBox('strzałki pola płyną wzdłuż pola (animacja, także na pauzie)'); self.ch_og_plyn.setChecked(True)
        self.ch_og_plyn.setToolTip('Każda strzałka przesuwa się wzdłuż swojego kierunku, pojawia się i gaśnie (faza losowa w każdym węźle), '
                                   'więc widać, dokąd pole niesie. Długość i kierunek bez zmian.')
        self.sp_og_tempo = _spin(0.05, 5, 0.6, 0.05, 2, ' /s'); self.sp_og_tempo.setToolTip('Tempo płynięcia strzałek (przebiegi na sekundę).')
        self._og_anim = QTimer(self); self._og_anim.setInterval(33); self._og_anim.timeout.connect(self._og_anim_tik)
        self.sp_og_pods.setToolTip('Strzałka co tyle węzłów siatki.')
        self.ch_og_osie = QCheckBox('osie siodła i wir (w koloidzie i w pułapce)'); self.ch_og_osie.setChecked(True)
        self.ch_og_osie.setToolTip('Wartości własne Jakobianu 3 × 3 pola: błękit = ściąga (groty do środka), czerwień = wypycha (groty na zewnątrz), '
                                   'fiolet = wir (okrąg z grotami w kierunku obrotu). Długość ∝ |λ|.')
        self.ch_og_skok = QCheckBox('skok koloidu w bloku 6 (pchnięcie MLP)'); self.ch_og_skok.setChecked(True)
        self.ch_og_tlo = QCheckBox('pozostałe cząsteczki w tle (przyciemnione)'); self.ch_og_tlo.setChecked(True)
        self.ch_og_tlo.setToolTip('Pozostałe 599 cząsteczek w kolorach cząsteczek, przyciemnione; w podkrokach płyną razem z przejściem '
                                  'przez sieć, w osi uczenia z checkpointami (te same współrzędne co sfera).')
        self.sp_og_jtlo = _spin(0.02, 1.0, 0.35, 0.05, 2, ' ×'); self.sp_og_jtlo.setToolTip('Jasność cząsteczek w tle.')
        self.ch_og_ruch = QCheckBox('wypadkowy ruch koloidu w nadchodzącym kroku (zielony)'); self.ch_og_ruch.setChecked(True)
        self.ch_og_ruch.setToolTip('Zielona strzałka: dokąd koloid przejdzie z bieżącego checkpointu do następnego (położenie następne − bieżące, '
                                   'wypadkowa całego kroku uczenia). Pomarańczowy skok to pchnięcie wiązek w tym checkpoincie; w opisie kąt '
                                   'między nimi. W kadrze komórki długość jest przycięta do ścianki sześcianu, kierunek prawdziwy.')
        self.cb_og_slad = WaskiCombo(); self.cb_og_slad.addItems(['brak', 'koloid', 'koloid i pułapka']); self.cb_og_slad.setCurrentIndex(1)
        self.sp_og_slad = _spin(1, 2000, 30, 5, sufiks=' ckpt'); self.sp_og_slad.setToolTip('Długość śladu w checkpointach; kropki = skoki między checkpointami.')
        self.cb_og_uklad = WaskiCombo(); self.cb_og_uklad.addItems(['ślad w układzie laboratorium', 'ślad względem pułapki (pułapka stoi)'])
        self.cb_og_sledz = WaskiCombo(); self.cb_og_sledz.addItems(['kamera nie śledzi', 'środek kadru: pułapka', 'środek kadru: koloid'])
        self.cb_og_sledz.setCurrentIndex(2)
        self.cb_og_sledz.setToolTip('Okienko 11: wybrany punkt jest zawsze na środku okienka (obrót myszą zostaje). Gdy zera pola nie ma '
                                    'w komórce, „pułapka” to najsłabsze pole w komórce tokenu.')
        self.cb_og_skala = WaskiCombo(); self.cb_og_skala.addItems(['skala: swobodna (przybliżenie myszą)',
                                                                   'skala: sześcian zawsze tej samej wielkości na ekranie',
                                                                   'skala: stała przez całe uczenie (jedna odległość kamery)'])
        self.cb_og_skala.setCurrentIndex(1)
        self.cb_og_skala.setToolTip('Sześcian tej samej wielkości: odległość kamery dopasowana w każdym checkpoincie tak, że sześcian komórki '
                                    'zawsze tak samo wypełnia okienko (rozmiar komórki w ten sposób znika z obrazu). Stała: jedna odległość '
                                    'na całe uczenie (90. percentyl wielkości komórki), więc widać, jak komórka rośnie i maleje; w nielicznych '
                                    'checkpointach z największą komórką sześcian wychodzi poza okienko. W obu wiązki i wkłady są przycięte '
                                    'do sześcianu, a przybliżenie myszą nie zostaje.')
        self.ch_og_kolej = QCheckBox('odtwarzanie: najpierw lasery, potem koloid'); self.ch_og_kolej.setChecked(True)
        self.ch_og_kolej.setToolTip('Okienko 11: w pierwszej połowie przejścia między checkpointami zmienia się pole wiązek, pułapka i skok, '
                                    'w drugiej koloid przechodzi na nowe miejsce. Obie zmiany pochodzą z tego samego kroku uczenia; kolejność '
                                    'pokazuje łańcuch: zmienione lasery → nowe pole → nowe położenie.')
        self.ch_og_sfera = QCheckBox('pokaż pułapkę także na dużej sferze'); self.ch_og_sfera.setChecked(False)
        self.lb_og = QLabel('—'); self.lb_og.setWordWrap(True)
        f.addRow(self.ch_og); f.addRow('Czas okienka 11', self.cb_og_czas); f.addRow(self.ch_og_pole)
        f.addRow('Siatka (węzłów na bok)', self.sp_og_G); f.addRow('Zasięg siatki', self.sp_og_zas)
        w_ = QWidget(); l_ = QHBoxLayout(w_); l_.setContentsMargins(0, 0, 0, 0); l_.addWidget(self.ed_og_razem, 1); l_.addWidget(self.bt_og_razem)
        f.addRow('Licz razem z', w_)
        w_ = QWidget(); l_ = QHBoxLayout(w_); l_.setContentsMargins(0, 0, 0, 0); l_.addWidget(self.cb_og_lupa, 1); l_.addWidget(self.sp_og_lupa)
        f.addRow('Pole w podkrokach', w_)
        f.addRow(hb); f.addRow(self.pasek_og)
        w_ = QWidget(); l_ = QHBoxLayout(w_); l_.setContentsMargins(0, 0, 0, 0)
        for x_ in (self.bt_pom, QLabel('co'), self.sp_pom_co, QLabel('siatka'), self.sp_pom_G, self.bt_pom_stop):
            l_.addWidget(x_, 1 if x_ is self.bt_pom else 0)
        f.addRow(w_); f.addRow(self.pasek_pom); f.addRow(self.lb_pom)
        f.addRow(self.ch_og_wiazki); f.addRow('Jasność wiązek', self.cb_og_jasn); f.addRow('  wzmocnienie', self.sp_og_gw)
        f.addRow('  długość wiązek', self.sp_og_dlw); f.addRow(self.ch_og_refl); f.addRow('  jasność od reflektorów', self.sp_og_jrefl)
        f.addRow(self.ch_og_linie); f.addRow(self.ch_og_sw)
        f.addRow('  jasność światła', self.sp_og_jsw); f.addRow('  przewężenie', self.sp_og_szer); f.addRow('  rozbieżność', self.sp_og_rozb)
        f.addRow(self.ch_og_osieo); f.addRow(self.ch_og_obrys); f.addRow('  poziom obrysu', self.sp_og_poz)
        f.addRow('  krycie obrysu', self.sp_og_jobr); f.addRow('  powiększenie obrysu', self.sp_og_pobr); f.addRow('Wkłady', self.cb_og_wk); f.addRow('  długość wkładów', self.sp_og_dlwk)
        f.addRow(self.ch_og_mgla); f.addRow('Jasność mgły', self.sp_og_jas); f.addRow('Ostrość ogniska', self.sp_og_ostr)
        f.addRow('Tło pola', self.sp_og_tlo); f.addRow('Rozmycie', self.sp_og_rozm)
        f.addRow('Rodzaj strzałek', self.cb_og_strz); f.addRow('  długość', self.sp_og_strz); f.addRow('  gęstość: co', self.sp_og_pods)
        f.addRow(self.ch_og_plyn); f.addRow('  tempo płynięcia', self.sp_og_tempo)
        f.addRow(self.ch_og_osie); f.addRow(self.ch_og_skok); f.addRow(self.ch_og_ruch)
        f.addRow(self.ch_og_tlo); f.addRow('  jasność tła', self.sp_og_jtlo)
        f.addRow('Ślad', self.cb_og_slad); f.addRow('Długość śladu', self.sp_og_slad); f.addRow('Układ śladu', self.cb_og_uklad)
        f.addRow('Kadr', self.cb_og_sledz); f.addRow('Skala kadru', self.cb_og_skala); f.addRow(self.ch_og_kolej); f.addRow(self.ch_og_sfera); f.addRow(self.lb_og)
        for w in (self.ch_og_wiazki, self.ch_og_mgla, self.ch_og_osie, self.ch_og_skok, self.ch_og_ruch, self.ch_og_plyn, self.ch_og_tlo, self.ch_og_kolej, self.ch_og_sfera,
                  self.ch_og_sw, self.ch_og_linie, self.ch_og_osieo, self.ch_og_obrys):
            w.toggled.connect(self._og_opcje)
        for w in (self.sp_og_jas, self.sp_og_ostr, self.sp_og_tlo, self.sp_og_rozm, self.sp_og_strz, self.sp_og_pods, self.sp_og_slad,
                  self.sp_og_gw, self.sp_og_dlw, self.sp_og_dlwk, self.sp_og_jsw, self.sp_og_szer, self.sp_og_rozb, self.sp_og_jrefl,
                  self.sp_og_poz, self.sp_og_jobr, self.sp_og_pobr, self.sp_og_jtlo):
            w.valueChanged.connect(self._og_opcje)
        for w in (self.cb_og_strz, self.cb_og_slad, self.cb_og_uklad, self.cb_og_sledz, self.cb_og_skala, self.cb_og_jasn, self.cb_og_wk,
                  self.cb_og_czas):
            w.currentIndexChanged.connect(self._og_opcje)
        self.ch_og_refl.toggled.connect(self._og_kamera_refl); self.ch_og_refl.toggled.connect(self._og_opcje)
        self._og_timer = QTimer(self); self._og_timer.setSingleShot(True); self._og_timer.setInterval(500)
        self._og_timer.timeout.connect(self._og_zlec)
        for w in (self.sp_og_G, self.sp_og_zas, self.sp_tok):
            w.valueChanged.connect(lambda *_: self._og_timer.start())
        self.ch_og.toggled.connect(lambda *_: self._og_zlec())
        self.ch_og.toggled.connect(lambda *_: self._og_czas(self.t) if self.h is not None else None)
        self.ch_og_pole.toggled.connect(lambda *_: self._og_zlec())
        self.cb_og_czas.currentIndexChanged.connect(lambda *_: self._zlec_podkroki())
        self.cb_og_czas.currentIndexChanged.connect(self._og_czas_zmiana)
        self._og = None; self._og_watek = None; self._og_klucz = None
        self._sek['pulapka'] = g

        # --- mapy
        g = Zwijana('Okienka 4–5: mapy na zdjęciu'); f = QFormLayout(g)
        self.cb_mapa = WaskiCombo()
        self.cb_mapa.addItems(['gdzie centra odpłynęły najdalej', 'drgania (token od centrum)', 'średnia atencja 6 bloków']
                              + [f'atencja blok {b + 1}' for b in range(6)] + ['łączne natężenie wybranych reflektorów Σ z_jc'])
        self.sp_krycie = _spin(0, 1, 0.6, 0.05, 2); self.ch_pierw = QCheckBox('atencja: pierwiastek (jak film 16)'); self.ch_pierw.setChecked(True)
        self.ch_rm = QCheckBox('ramka modelu (błękitna)'); self.ch_rm.setChecked(True)
        self.ch_rg = QCheckBox('ramka prawdziwa (biała)'); self.ch_rg.setChecked(True)
        f.addRow('Mapa w okienku 5', self.cb_mapa); f.addRow('Krycie mapy', self.sp_krycie)
        f.addRow(self.ch_pierw); f.addRow(self.ch_rm); f.addRow(self.ch_rg)
        self.cb_mapa.currentIndexChanged.connect(self._opcje); self.sp_krycie.valueChanged.connect(self._opcje)
        for w in (self.ch_pierw, self.ch_rm, self.ch_rg):
            w.toggled.connect(self._opcje)
        self._sek['mapy'] = g

        # --- dolina pewności i IoU (okienko 12)
        g = Zwijana('Dolina pewności i IoU wybranej cząsteczki (okienko 12)'); f = QFormLayout(g)
        self.ch_dol = QCheckBox('licz dolinę w bieżącym checkpoincie'); self.ch_dol.setChecked(True)
        self.ch_dol.setToolTip('Gdzie w przestrzeni stanu cząsteczki (192 wymiary, wejście MLP bloku 6) głowa dawałaby wysoką pewność\n'
                               'i dobre IoU ramki. Głowa czyta z położenia tylko 5 kierunków (ściany doliny); w pozostałych 187\n'
                               '(dno doliny) odpowiedź z bliska się nie zmienia. Rysowany jest przekrój 3D przez cząsteczkę.\n'
                               'Liczy się w tle przy każdej zmianie checkpointu (oś uczenia, nie podkroki).')
        self.cb_dol_baza = WaskiCombo()
        import dolina as DL_
        self.cb_dol_baza.addItems(list(DL_.OPIS_BAZ))
        self.cb_dol_baza.setToolTip('Które 3 kierunki z 192 tworzą przekrój:\n'
                                    '• osie sfery: dokładnie te, które widać na sferze reflektorów (przekrój leży w rysowanym rzucie);\n'
                                    '• ściany: 3 najsilniejsze kierunki czytane przez głowę (dolina z bliska jest wąska);\n'
                                    '• w poprzek: 2 kierunki czytane + 1 kierunek dna (rynna: wąsko w poprzek, długo wzdłuż);\n'
                                    '• dno: 3 kierunki, których głowa nie czyta (z bliska nic się nie zmienia, daleko widać zakrzywienie dna).\n'
                                    'Na dużej sferze dla baz innych niż osie sfery rysuje się cień przekroju (jego rzut na osie sfery).')
        self.sp_dol_G = _spin(12, 72, 40, 4); self.sp_dol_G.setToolTip('Punkty siatki na bok sześcianu przekroju (G³ wywołań głowy).')
        self.sp_dol_zas = _spin(0.1, 3.0, 1.0, 0.1, 2, ' × r99')
        self.sp_dol_zas.setToolTip('Połowa boku sześcianu przekroju jako ułamek promienia chmury cząsteczek (99. percentyl odległości od środka).')
        self.sp_dol_p = _spin(0.0, 0.999, 0.5, 0.05, 3); self.sp_dol_p.setToolTip('Próg pewności p = σ(logit) w dolinie.')
        self.sp_dol_iou = _spin(0.0, 0.999, 0.5, 0.05, 3); self.sp_dol_iou.setToolTip('Próg IoU ramki tej cząsteczki z prawdziwą ramką.')
        self.ch_dol_mgla = QCheckBox('mgła: punkty doliny (jaśniejsze = dalej nad progami)'); self.ch_dol_mgla.setChecked(True)
        self.ch_dol_obrys = QCheckBox('obrys: brzeg doliny (powierzchnia progu)'); self.ch_dol_obrys.setChecked(True)
        self.sp_dol_tlo = _spin(0, 1, 0.0, 0.05, 2)
        self.sp_dol_tlo.setToolTip('Jasność miejsc, gdzie nad progiem jest tylko jedno: pomarańczowe = sama pewność, niebieskie = samo IoU (0 = ukryte).')
        self.sp_dol_jas = _spin(0.05, 5, 0.6, 0.05, 2); self.sp_dol_rozm = _spin(1, 30, 6.0, 0.5, 1)
        self.ch_dol_sfera = QCheckBox('dolina także na sferze reflektorów (wokół cząsteczki)'); self.ch_dol_sfera.setChecked(True)
        self.ch_dol_kamera = QCheckBox('kamera okienka 12 obraca się powoli'); self.ch_dol_kamera.setChecked(False)
        f.addRow(self.ch_dol); f.addRow('Przekrój', self.cb_dol_baza); f.addRow('Siatka G', self.sp_dol_G)
        f.addRow('Pół boku', self.sp_dol_zas); f.addRow('Próg pewności', self.sp_dol_p); f.addRow('Próg IoU', self.sp_dol_iou)
        f.addRow(self.ch_dol_mgla); f.addRow(self.ch_dol_obrys); f.addRow('Tylko jedno nad progiem', self.sp_dol_tlo)
        f.addRow('Jasność mgły', self.sp_dol_jas); f.addRow('Wielkość punktów [px]', self.sp_dol_rozm)
        f.addRow(self.ch_dol_sfera); f.addRow(self.ch_dol_kamera)
        self.lb_dol = QLabel('—'); self.lb_dol.setWordWrap(True); f.addRow(self.lb_dol)
        for w in (self.ch_dol, self.ch_dol_sfera):
            w.toggled.connect(lambda *_: self._dol_zlec())
        self.cb_dol_baza.currentIndexChanged.connect(lambda *_: self._dol_zlec())
        for w in (self.sp_dol_G, self.sp_dol_zas):
            w.valueChanged.connect(lambda *_: self._dol_zlec())
        for w in (self.sp_dol_p, self.sp_dol_iou):
            w.valueChanged.connect(lambda *_: self._dol_progi())
        for w in (self.ch_dol_mgla, self.ch_dol_obrys):
            w.toggled.connect(lambda *_: self._dol_wyglad())
        for w in (self.sp_dol_tlo, self.sp_dol_jas, self.sp_dol_rozm):
            w.valueChanged.connect(lambda *_: self._dol_wyglad())
        self.ch_dol_kamera.toggled.connect(lambda on: self._dol_obrot.start() if on else self._dol_obrot.stop())
        self.sp_tok.valueChanged.connect(lambda *_: self._dol_zlec())
        self._dol = None; self._dol_klucz = None
        self._sek['dolina'] = g

        # --- mechanika reflektora (okienko 13)
        g = Zwijana('Reflektor: mechanika — odrzut wiązek i ruch (okienko 13)'); f = QFormLayout(g)
        self.ch_mech = QCheckBox('licz mechanikę wybranego reflektora (blok 6)'); self.ch_mech.setChecked(True)
        self.ch_mech.setToolTip('Reflektor = „Reflektor” z sekcji sfery (neuron c bloku 6), token = „Cząsteczka”.\n'
                                'Siła pola celu na koloid j to −∂L/∂x_j; wiązka przenosi ją na reflektor proporcjonalnie do jasności z_jc.\n'
                                'Wypadkowy odrzut na igłę v_c to dokładnie −∂L/∂v_c = Σ_j z_jc F_j (zasada Z22).\n'
                                'Wykonany ruch = różnica wag do następnego checkpointu (suma 10 albo 100 kroków AdamW).')
        self.cb_mech_zr = WaskiCombo()
        self.cb_mech_zr.addItems(['odrzut z tego zdjęcia (sonda)', 'odrzut z paczki 64 klatek zbioru uczącego'])
        self.cb_mech_zr.setToolTip('Sonda: to jedno zdjęcie (mały kawałek prawdziwego odrzutu). Paczka: 64 stałe klatki zbioru uczącego,\n'
                                   'bez augmentacji, fp32, bez obcinania gradientu — najbliżej tego, co naprawdę przestawiało reflektor.\n'
                                   'Pierwsze liczenie paczki wczytuje 64 klatki z archiwów (kilka sekund), potem ~1 s na checkpoint.\n'
                                   'Linki tokenów są zawsze z oglądanego zdjęcia.')
        self.sp_mech_przes = _spin(0, 500, 0, 1, 0, ' ×'); self.sp_mech_przes.setSpecialValueText('auto')
        self.sp_mech_przes.setToolTip('Przesada obrotów i przesunięć na rysunku (0 = auto: największy obrót ≈ 25°). Kąty w opisie są prawdziwe.')
        self.sp_mech_linki = _spin(0, 80, 20, 2); self.sp_mech_linki.setToolTip('Ile najsilniejszych linek tokenów rysować (wybrany token zawsze).')
        self.ch_mech_cel = QCheckBox('celownik w_c'); self.ch_mech_cel.setChecked(True)
        self.ch_mech_prog = QCheckBox('suwak progu b_c'); self.ch_mech_prog.setChecked(True)
        self.ch_mech_nap = QCheckBox('napisy'); self.ch_mech_nap.setChecked(True)
        f.addRow(self.ch_mech); f.addRow('Odrzut', self.cb_mech_zr); f.addRow('Przesada ruchu', self.sp_mech_przes)
        f.addRow('Linki tokenów', self.sp_mech_linki)
        hb = QHBoxLayout(); hb.addWidget(self.ch_mech_cel); hb.addWidget(self.ch_mech_prog); hb.addWidget(self.ch_mech_nap); f.addRow(hb)
        self.lb_mech = QLabel('—'); self.lb_mech.setWordWrap(True); f.addRow(self.lb_mech)
        self.ch_mech.toggled.connect(lambda *_: self._mech_zlec())
        self.cb_mech_zr.currentIndexChanged.connect(lambda *_: self._mech_zlec())
        for w in (self.sp_tok, self.sp_refl):
            w.valueChanged.connect(lambda *_: self._mech_zlec())
        for w in (self.sp_mech_przes, self.sp_mech_linki):
            w.valueChanged.connect(lambda *_: self._mech_rysuj())
        for w in (self.ch_mech_cel, self.ch_mech_prog, self.ch_mech_nap):
            w.toggled.connect(lambda *_: self._mech_rysuj())
        self._mech = None
        self._sek['mech'] = g

        # --- pasek osi czasu
        os_ = QWidget(); ho = QHBoxLayout(os_); ho.setContentsMargins(4, 2, 4, 2)
        self.bt_okna = QToolButton(); self.bt_okna.setText('Okienka ▾'); self.bt_okna.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.bt_start = QPushButton('⏮'); self.bt_play = QPushButton('▶'); self.bt_play.setCheckable(True)
        self.bt_start.setToolTip('Na początek osi czasu.'); self.bt_play.setToolTip('Odtwarzanie / pauza (spacja).')
        for b_ in (self.bt_start, self.bt_play):
            b_.setMaximumWidth(48)                                           # ikony: miejsce zostaje dla licznika czasu
        self.suwak = SuwakAB(); self.suwak.setRange(0, 0)
        self.cb_tempo = WaskiCombo(); self.cb_tempo.addItems(['tempo jak w filmie 16 (wolno na początku)', 'stałe: checkpointy na sekundę'])
        self.cb_tempo.setToolTip('Film 16: v = min(0,8 + krok/250, 150) kroków na klatkę; stałe: tyle checkpointów na sekundę.')
        self.sp_tempo = _spin(0.05, 200, 1.0, 0.25, 2, ' ×'); self.sp_tempo.setToolTip('Mnożnik tempa (albo checkpointy/s w trybie stałym).')
        self.lb_czas = QLabel('—'); _staly_uklad(self.lb_czas)               # bez sztywnego minimum: dzieli miejsce z suwakiem
        f_ = self.lb_czas.font(); f_.setPointSize(f_.pointSize() + 2); self.lb_czas.setFont(f_)
        self.ch_petla = QCheckBox('pętla'); self.ch_petla.setToolTip('Po dojściu do końca uczenia odtwarzanie wraca do kroku 0 (także z podkrokami).')
        self.ab = [None, None]                                               # punkty A i B (indeks checkpointu, None = brak)
        self.bt_a = QPushButton('A'); self.bt_b = QPushButton('B'); self.bt_ab0 = QPushButton('✕AB')
        self.bt_a.setToolTip('Punkt A = bieżący krok: odtwarzanie zaczyna się od A (z pętlą: wraca do A). Kolor błękitny na suwaku.')
        self.bt_b.setToolTip('Punkt B = bieżący krok: odtwarzanie kończy się w B (z pętlą: skacze do A). Kolor pomarańczowy na suwaku.')
        self.bt_ab0.setToolTip('Usuń punkty A i B (cała oś czasu).')
        for b_ in (self.bt_a, self.bt_b, self.bt_ab0):
            b_.setMaximumWidth(44)
        self.bt_a.clicked.connect(lambda: self._ustaw_ab(0)); self.bt_b.clicked.connect(lambda: self._ustaw_ab(1))
        self.bt_ab0.clicked.connect(lambda: self._ustaw_ab(None))
        self.ch_petla.toggled.connect(lambda *_: self._ustaw_ab(-1))
        for w in (self.bt_okna, self.bt_start, self.bt_play, self.ch_petla, self.bt_a, self.bt_b, self.bt_ab0):
            ho.addWidget(w)
        ho.addWidget(self.suwak, 3); ho.addWidget(self.cb_tempo); ho.addWidget(self.sp_tempo); ho.addWidget(bez_ukladu(self.lb_czas), 2)
        self.suwak.valueChanged.connect(lambda v: None if self._blokada else self.ustaw_czas(v / 100.0))
        self.bt_play.toggled.connect(self._graj); self.bt_start.clicked.connect(lambda: self.ustaw_czas(self._zakres_ab()[0]))

        # --- okienka (doki)
        self.okna = QMainWindow(); self.okna.setDockNestingEnabled(True)
        self.okna.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowNestedDocks
                                 | QMainWindow.DockOption.AllowTabbedDocks)
        pl_.addWidget(os_); pl_.addWidget(self.okna, 1)
        self._pl = pl_
        self.doki = {}
        self.v1, self.e1, self.lb1 = self._okno_2d('1', '1. Ruch globalny (układ inercjalny): cała sieć naraz', 'globalny')
        self.v2, self.e2, self.lb2 = self._okno_2d('2', '2. Odkształcenie lokalne: studnie odpływają od adresów', 'lokalny')
        self.v3, self.e3, self.lb3 = self._okno_2d('3', '3. Drgania: tokeny wokół centrów studni', 'drgania')
        self.vs = WidokGL(); self.vs.setBackgroundColor(TLO); self.vs.setCameraPosition(distance=135, elevation=24, azimuth=-10)
        self.es = SferaReflektorow(LD.R_SFERY); self.vs.addItem(self.es)
        self.vs.klik_px.connect(self._klik_sfera)
        self._dok('s', 'Sfera reflektorów: 768 neuronów, 600 wiązek każdy (film 12)', self._z_napisem(self.vs, 'lbs'))
        self.v4 = Widok2D(); self.e4 = MapyNaZdjeciu(); self.v4.addItem(self.e4)
        self._dok('4', '4. Atencja 6 bloków (ile uwagi dostaje patch)', self._z_napisem(self.v4, 'lb4'))
        self.v5 = Widok2D(); self.e5 = MapyNaZdjeciu(); self.v5.addItem(self.e5)
        self._dok('5', '5. Mapa na zdjęciu', self._z_napisem(self.v5, 'lb5'))
        self.v4.kadruj(47, 21, 98, 48); self.v5.kadruj(15, 10, 32, 22)
        self.e4.kafelki = [dict(mapa=b, x=(b % 3) * 32, y=-(b // 3) * 24, norma=0, gamma=0.5, lut=2, alfa=0.62) for b in range(6)]
        self.p6 = self._wykres('6', '6. Wielkość trzech składowych ruchu', 'średnio [komórki siatki]')
        self.p7 = self._wykres('7', '7. IoU z checkpointu na checkpoint', 'IoU')
        self.p8 = self._wykres('8', '8. Pewność z checkpointu na checkpoint', 'pewność')
        self.p9 = self._wykres('9', '9. Pary przeciwnych reflektorów (cos < −0,5)', 'odsetek kanałów')
        self.p10 = self._wykres('10', '10. Obrót igieł reflektorów (gradient): mediana', '°/100 kroków')
        self.p10.setLogMode(y=True)
        # okienko 11: pułapka optyczna i koloid skaczący w polu jej wiązek (ogniskowa.py); ta sama pułapka może być na dużej sferze
        self.es_og = PoleOgniskowej(); self.es_og.setVisible(False); self.vs.addItem(self.es_og)
        self.vog = WidokGL(); self.vog.setBackgroundColor(TLO); self.vog.setCameraPosition(distance=30, elevation=22, azimuth=-35)
        self.eog = PoleOgniskowej(); self.vog.addItem(self.eog)
        self.eog_pp = PoleOgniskowej(); self.eog_pp.opcje.update(tylko_pole=True, slad=0)   # pole w komórce w podkrokach
        self.eog_pp.setVisible(False); self.eog_pp.setDepthValue(1); self.vog.addItem(self.eog_pp)   # nad wiązkami
        self._pp_pole = None; self._pp_klucz = None; self._pp_cache = {}
        self.watek_pp = PolePodkrokowWatek(); self.watek_pp.gotowe.connect(self._pole_pod_gotowe)
        self._dok('11', '11. Pułapka optyczna: koloid skacze w polu wiązek (blok 6)', self._z_napisem(self.vog, 'lb11'))
        # okienko 12: dolina pewności i IoU wybranej cząsteczki (dolina.py); ta sama dolina wokół cząsteczki na dużej sferze
        self.es_dol = DolinaGPU(); self.es_dol.setVisible(False); self.es_dol.setDepthValue(2); self.vs.addItem(self.es_dol)
        self.vdol = WidokGL(); self.vdol.setBackgroundColor(TLO); self.vdol.setCameraPosition(distance=42, elevation=20, azimuth=-40)
        self.edol = DolinaGPU(); self.vdol.addItem(self.edol)
        self.watek_dol = DolinaWatek(); self.watek_dol.gotowe.connect(self._dol_gotowe)
        self._dol_obrot = QTimer(self); self._dol_obrot.setInterval(33)
        self._dol_obrot.timeout.connect(lambda: self.vdol.orbit(0.25, 0) if self.widac('12') else None)
        self._dok('12', '12. Dolina pewności i IoU: gdzie cząsteczka daje dobrą odpowiedź (192 wymiary, przekrój 3D)',
                  self._z_napisem(self.vdol, 'lb12'))
        self.doki['12'].visibilityChanged.connect(lambda on: self._dol_zlec() if on else None)
        # okienko 13: rysunek techniczny wybranego reflektora (reflektor_mechanika.py, reflektor_rysunek.py)
        self.vmech = WidokGL(); self.vmech.setBackgroundColor(TLO)
        self.vmech.setCameraPosition(distance=50, elevation=36, azimuth=-62)
        from PyQt6.QtGui import QVector3D
        self.vmech.opts['center'] = QVector3D(0.0, 0.0, 0.5)
        self.emech = SchematGPU(); self.vmech.addItem(self.emech)
        self._mech_napisy = []
        self.watek_mech = MechanikaWatek(); self.watek_mech.gotowe.connect(self._mech_gotowe)
        self._dok('13', '13. Reflektor: mechanika — odrzut wiązek, sprężyny i wykonany ruch (blok 6)', self._z_napisem(self.vmech, 'lb13'))
        self.doki['13'].visibilityChanged.connect(lambda on: self._mech_zlec() if on else None)
        self.rzuty.buduj_okienka()                                      # okienka R1–R8: rzuty z 192 wymiarów
        self._buduj_przejscie()
        self._buduj_panel()
        self._uklad_domyslny()
        self.rzuty.uklad_domyslny()
        self.menu_okienek = m = QMenu('Okienka', self)
        m.setToolTipsVisible(True)
        for d in self.doki_ster.values():
            m.addAction(d.toggleViewAction())
        m.addSeparator()
        grupy = (('Ruch i mapy (1–10)', ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')),
                 ('Sfera, pułapka, dolina, mechanika', ('s', '11', '12', '13', 'p')),
                 ('Rzuty z 192 wymiarów (R0–R12)', tuple(k for k in self.doki if k.startswith('R'))))
        for nazwa, klucze in grupy:
            pm = m.addMenu(nazwa)
            for k in klucze:
                if k in self.doki:
                    pm.addAction(self.doki[k].toggleViewAction())
        m.addSeparator()
        mu = m.addMenu('Układy okienek')
        for nazwa, fn in (('Domyślny: sfera duża, rzuty i wykresy w kartach', self._przywroc_uklad),
                          ('Sfera + pułapka optyczna obok siebie', lambda: self._uklad_gotowy('pulapka')),
                          ('Sfera + rzuty R1–R7 obok siebie', lambda: self._uklad_gotowy('rzuty')),
                          ('Sfera na całe okno (reszta w kartach)', lambda: self._uklad_gotowy('sfera'))):
            a = QAction(nazwa, self); a.triggered.connect(fn); mu.addAction(a)
        mu.addSeparator()
        self._sloty = {}
        for i in (1, 2, 3):
            a = QAction(f'Zapamiętaj bieżący układ jako {i}', self); a.setShortcut(f'Ctrl+Shift+{i}')
            a.triggered.connect(lambda _=False, i=i: self._slot(i, zapisz=True)); mu.addAction(a); self.addAction(a)
        for i in (1, 2, 3):
            a = QAction(f'Przywróć układ {i}', self); a.setShortcut(f'Ctrl+{i}')
            a.triggered.connect(lambda _=False, i=i: self._slot(i)); mu.addAction(a); self.addAction(a)
        self.bt_okna.setMenu(m)
        self._stan_domyslny = self.okna.saveState()                     # układ domyślny bez rozmiarów (rozmiary: _rozmiary_domyslne)
        # listy to WaskiCombo: długie pozycje nie wymuszają szerokości panelu ani okienek (wcześniej pasek „Przejścia”
        # trzymał 1700 px i ściskał sferę do 80 px); w rozwiniętej liście tekst jest cały
        for cb in self.findChildren(QComboBox):
            if not cb.toolTip():                                         # bez własnego dymka: dymek = pełny wybrany tekst
                cb.setToolTip(cb.currentText())
                cb.currentTextChanged.connect(cb.setToolTip)
        import dymki
        import grupy_parametrow
        self._dymki_zmienione = dymki.zastosuj(self)   # dymki: co robi każdy parametr (także po najechaniu na nazwę wiersza)
        grupy_parametrow.zastosuj(self)           # nagłówki podgrup w długich sekcjach (pułapka, sfera, dane, …)

    def _z_napisem(self, widok, nazwa):
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(widok, 1)
        lb = QLabel(''); lb.setStyleSheet('color:#ffd166; padding:2px 6px; background:#0b1014'); _staly_uklad(lb)
        v.addWidget(bez_ukladu(lb))                                   # napis zmieniany co klatkę nie przelicza układu okna
        setattr(self, nazwa, lb)
        return w

    def _okno_2d(self, klucz, tytul, tryb):
        v = Widok2D(); e = RuchPlaski(tryb); v.addItem(e)
        v.klik_px.connect(lambda x, y, v=v, e=e: self._klik_2d(v, e, x, y))
        w = self._z_napisem(v, f'lb{klucz}')
        self._dok(klucz, tytul, w)
        return v, e, getattr(self, f'lb{klucz}')

    def _dok(self, klucz, tytul, widget):
        d = QDockWidget(tytul, self.okna); d.setObjectName(f'dok_lasery_{klucz}'); d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        d.toggleViewAction().setText(tytul)
        from okna import pasek_doku
        from panel_ustawien import podlacz_menu
        pasek_doku(d)                                   # ⛶: maksymalizacja na ekranie okienka (też dwuklik, F11)
        podlacz_menu(d, lambda pos, k=klucz: self.menu_okienka(k, pos))     # prawy klik: ustawienia tego okienka
        d.visibilityChanged.connect(lambda on: self._odslonieto() if on else None)
        self.doki[klucz] = d
        return d

    def widac(self, klucz):
        """Czy okienko naprawdę widać: otwarte i (odpięte albo z niepustym widocznym obszarem). QDockWidget.isVisible() jest
        prawdą także dla okienka w nieaktywnej karcie, więc samo isVisible() kazało liczyć i rysować schowane okienka."""
        d = self.doki.get(klucz)
        return d is not None and d.isVisible() and (d.isFloating() or not d.visibleRegion().isEmpty())

    def _odslonieto(self):
        """Okienko odsłonięte (karta, otwarcie, odpięcie): po chwili jedno odświeżenie bieżącej chwili (zbiorcze dla wielu naraz)."""
        if self.h is None:
            return
        if not hasattr(self, '_odsl_timer'):
            self._odsl_timer = QTimer(self); self._odsl_timer.setSingleShot(True); self._odsl_timer.setInterval(0)
            self._odsl_timer.timeout.connect(self._po_odslonieciu)
        self._odsl_timer.start()

    def _po_odslonieciu(self):
        if self.h is None:
            return
        if not self.timer.isActive():
            self.ustaw_czas(self.t)
        self._og_zlec()                                                  # pułapka czekała na widoczne okienko 11

    def _dok_sterowania(self, klucz, tytul, widget):
        """Okienko sterujące (parametry, podkroki): jak okienka widoków można je odpiąć i przesunąć, ale nie należy do self.doki
        (nie jest widokiem osi czasu)."""
        from okna import Przewijane
        d = QDockWidget(tytul, self.okna); d.setObjectName(f'dok_lasery_ster_{klucz}'); d.setWidget(Przewijane(widget))
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        d.toggleViewAction().setText(tytul)
        from okna import pasek_doku
        pasek_doku(d)
        if not hasattr(self, 'doki_ster'):
            self.doki_ster = {}
        self.doki_ster[klucz] = d
        return d

    # ------------------------------------------------------------------ panel parametrów i ustawienia okienek pod prawym klikiem
    # okienko → sekcje panelu z jego ustawieniami (prawy klik w okienko wypożycza je do popupu); R* pyta rzuty_okienka
    SEKCJE_OKIENEK = {'1': ('plaskie',), '2': ('plaskie',), '3': ('plaskie',), 's': ('sfera', 'wybor'), '4': ('mapy',), '5': ('mapy',),
                      '11': ('pulapka',), '12': ('dolina',), '13': ('mech',), 'p': ('przejscie',),
                      '6': ('dane',), '7': ('dane',), '8': ('dane',), '9': ('dane',), '10': ('dane',)}
    # strony panelu: (klucz, nazwa na liście, sekcje, dymek)
    STRONY = (('dane', 'Dane, etykiety', ('dane', 'etykiety'), 'Historia uczenia, zdjęcie, etykieta narzędzia i obszary oka'),
              ('zestaw', 'Kilka zdjęć', ('zestaw',), 'Kilka zdjęć naraz na sferze reflektorów (600 koloidów na zdjęcie)'),
              ('sfera', 'Sfera', ('wybor', 'sfera'), 'Sfera reflektorów, wybór cząsteczek i reflektorów'),
              ('pulapka', 'Pułapka (11)', ('pulapka',), 'Pułapka optyczna: ogniskowa wiązek wokół koloidu, obrys ogniska 3D'),
              ('podkroki', 'Podkroki (P)', ('przejscie', 'zapis'), 'Okienko P (przejście przez sieć) i zapis przejść na dysk'),
              ('rzuty', 'Rzuty R0–R12', ('rzuty',), 'Rzuty z 192 wymiarów do 3D'),
              ('plaskie', 'Ruch 1–3', ('plaskie',), 'Rzut płaski: ruch globalny, studnie, drgania'),
              ('mapy', 'Mapy 4–5', ('mapy',), 'Atencja i mapy na zdjęciu'),
              ('dolina', 'Dolina (12)', ('dolina',), 'Dolina pewności i IoU'),
              ('mech', 'Mechanika (13)', ('mech',), 'Mechanika reflektora'),
              ('widok', 'Widok', ('stereo',), 'Stereo 3D'))

    def _buduj_panel(self):
        from panel_ustawien import PanelParametrow
        self._buduj_etykiety()
        self._buduj_wybor()
        self._buduj_zestaw()
        self.panel = PanelParametrow()
        self._strona_sekcji = {}
        for klucz, nazwa, sekcje, dymek in self.STRONY:
            self.panel.dodaj_strone(klucz, nazwa, [self._sek[s] for s in sekcje], dymek)
            for s in sekcje:
                self._strona_sekcji.setdefault(s, klucz)
        self.panel.setMinimumWidth(330)
        self.dok_panel = self._dok_sterowania('parametry', 'Parametry', self.panel)
        self.dok_panel.setToolTip('F2 chowa i pokazuje, ❐ odpina jako osobne okno. Prawy klik w okienko: jego ustawienia przy kursorze.')

    def _buduj_wybor(self):
        """Wybór cząsteczki i reflektora: osobna sekcja (strona sfery i popup sfery), bo czyta go większość okienek."""
        g = Zwijana('Wybrana cząsteczka i reflektor'); f = QFormLayout(g)
        for etykieta, pole in self._wiersze_wyboru:
            f.addRow(etykieta, pole)
        bt = QToolButton(); bt.setText('Zaznacz obszar ▾'); bt.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        bt.setToolTip('Wpisuje do „+ cząsteczki” wszystkie patche wybranego obszaru (etykieta narzędzia z sekcji „Etykiety”,\n'
                      'źrenica i rogówka z obrysów Cataract-LMM). Działa z „Wiązki: do wybranych cząsteczek” i „Wektory dla: wybranych”.')
        m = QMenu(bt)
        for kod, nazwa in ((3, 'narzędzie'), (2, 'źrenica'), (1, 'rogówka'), (0, 'tło'), (-1, 'wyczyść')):
            a = m.addAction(nazwa); a.triggered.connect(lambda _=False, k=kod: self._zaznacz_obszar(k))
        bt.setMenu(m)
        f.addRow('', bt)
        self._sek['wybor'] = g

    # ------------------------------------------------------------------ kilka zdjęć naraz (sfera reflektorów)
    def _buduj_zestaw(self):
        """Zestaw zdjęć tej samej historii wag: sfera pokazuje 600 koloidów każdego zdjęcia (600·n cząsteczek, 600·n·768 wiązek),
        reszta okienek — aktywne zdjęcie. Domyślny limit 5, najwyżej 32 (jak paczka uczenia): każde zdjęcie to osobna historia
        (pierwsze liczenie czyta wszystkie pliki wag), a sfera rysuje n razy więcej wiązek."""
        from PyQt6.QtWidgets import QListWidget
        g = Zwijana('Kilka zdjęć naraz na sferze reflektorów'); f = QFormLayout(g)
        self._zestaw = []                                       # [(id zdjęcia jak cb_zdjecie.currentData(), opis)]
        self.hz = None; self._zestaw_hs = []; self._zestaw_watek = None
        self.ch_zestaw = QCheckBox('sfera pokazuje wszystkie zdjęcia zestawu'); self.ch_zestaw.setChecked(True)
        self.ch_zestaw.setToolTip('Włączone: na sferze reflektorów są koloidy wszystkich policzonych zdjęć zestawu (każde ze swoim środkiem\n'
                                  'chmury, wspólne reflektory i rzut 3D, bo zależą tylko od wag). Wyłączone: tylko aktywne zdjęcie.')
        self.lw_zestaw = QListWidget(); self.lw_zestaw.setMaximumHeight(130)
        self.lw_zestaw.setToolTip('Zdjęcia zestawu w kolejności dodania; numer = które z kolei 600 cząsteczek na sferze (0: 0–599, 1: 600–1199, …).')
        self.sp_zestaw_limit = _spin(1, 32, 5, 1, sufiks=' zdjęć')
        self.sp_zestaw_limit.setToolTip('Najwięcej zdjęć w zestawie. Domyślnie 5; do 32 (paczka uczenia), ale każde zdjęcie dokłada 460 800\n'
                                        'wiązek na sferze i ~0,7 GB natężeń. Powyżej ok. 3 GB natężenia nie trafiają w całości na kartę\n'
                                        '(sfera wysyła wtedy przy każdej zmianie checkpointu tylko parę checkpointów).')
        b_dod = QPushButton('+ bieżące zdjęcie'); b_dod.clicked.connect(self._zestaw_dodaj)
        b_dod.setToolTip('Dodaje zdjęcie wybrane w „Dane → Zdjęcie” (historia, zakres kroków i okno jak w sekcji Dane).')
        b_us = QPushButton('− zaznaczone'); b_us.clicked.connect(self._zestaw_usun)
        b_us.setToolTip('Usuwa z zestawu zdjęcia zaznaczone na liście (Ctrl/Shift + klik: kilka). Policzone historie zostają w cache.')
        b_cz = QPushButton('wyczyść'); b_cz.clicked.connect(self._zestaw_czysc)
        b_cz.setToolTip('Usuwa wszystkie zdjęcia z zestawu; sfera wraca do jednego, aktywnego zdjęcia.')
        hb = QHBoxLayout(); hb.addWidget(b_dod); hb.addWidget(b_us); hb.addWidget(b_cz)
        self.bt_zestaw = QPushButton('Policz / wczytaj zestaw'); self.bt_zestaw.clicked.connect(self._zestaw_licz)
        self.bt_zestaw.setToolTip('Historia każdego zdjęcia z cache albo liczona na GPU (minuty na zdjęcie przy pierwszym liczeniu).')
        self.bt_zestaw_stop = QPushButton('Przerwij'); self.bt_zestaw_stop.setEnabled(False)
        self.bt_zestaw_stop.setToolTip('Przerywa liczenie zestawu; historie już policzone zostają w cache i wczytają się od razu.')
        self.lw_zestaw.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.bt_zestaw_stop.clicked.connect(lambda: setattr(self._zestaw_watek, 'stop', True) if self._zestaw_watek else None)
        hb2 = QHBoxLayout(); hb2.addWidget(self.bt_zestaw); hb2.addWidget(self.bt_zestaw_stop)
        self.pasek_zestaw = QProgressBar(); self.pasek_zestaw.setTextVisible(True)
        self.cb_zestaw_akt = WaskiCombo()
        self.cb_zestaw_akt.setToolTip('Aktywne zdjęcie: jego historię pokazują okienka 1–13, rzuty R i podkroki; wybór cząsteczki dotyczy jego\n'
                                      '600 koloidów. Klik w koloid innego zdjęcia na sferze też przełącza aktywne zdjęcie.')
        self.cb_zestaw_akt.currentIndexChanged.connect(self._zestaw_aktywne)
        self.lb_zestaw = QLabel('Dodaj zdjęcia (+), potem „Policz / wczytaj zestaw”.'); self.lb_zestaw.setWordWrap(True)
        f.addRow(self.ch_zestaw); f.addRow(self.lw_zestaw); f.addRow(hb); f.addRow('Limit', self.sp_zestaw_limit)
        f.addRow(hb2); f.addRow(self.pasek_zestaw); f.addRow('Aktywne zdjęcie', self.cb_zestaw_akt); f.addRow(self.lb_zestaw)
        self.ch_zestaw.toggled.connect(lambda *_: self._zestaw_na_sferze())
        self._sek['zestaw'] = g

    def _zestaw_odswiez_liste(self):
        self.lw_zestaw.clear()
        for i, (_, opis) in enumerate(self._zestaw):
            self.lw_zestaw.addItem(f'{i}. {opis}')

    def _zestaw_dodaj(self):
        zd = self.cb_zdjecie.currentData()
        if not zd:
            self.lb_zestaw.setText('Najpierw wybierz zdjęcie w „Dane”.'); return
        if any(z == zd for z, _ in self._zestaw):
            self.lb_zestaw.setText('To zdjęcie już jest w zestawie.'); return
        if len(self._zestaw) >= self.sp_zestaw_limit.value():
            self.lb_zestaw.setText(f'Limit {self.sp_zestaw_limit.value()} zdjęć (pole „Limit”, najwyżej 32).'); return
        film = self.cb_film.currentData() or 'plik'
        self._zestaw.append((str(zd), f'{film}  {self.cb_zdjecie.currentText()}'))
        self._zestaw_odswiez_liste()

    def _zestaw_usun(self):
        rzedy = sorted({self.lw_zestaw.row(it) for it in self.lw_zestaw.selectedItems()}, reverse=True)
        for r in rzedy:
            del self._zestaw[r]
        self._zestaw_odswiez_liste()

    def _zestaw_czysc(self):
        self._zestaw = []; self._zestaw_odswiez_liste()
        self.hz = None; self._zestaw_hs = []
        self.cb_zestaw_akt.blockSignals(True); self.cb_zestaw_akt.clear(); self.cb_zestaw_akt.blockSignals(False)
        self._zestaw_na_sferze()

    def _zestaw_licz(self):
        if self._zestaw_watek is not None or not self._zestaw:
            return
        n = len(self._zestaw)
        if n > 5:
            from PyQt6.QtWidgets import QMessageBox
            if QMessageBox.question(self, 'Dużo zdjęć', f'{n} zdjęć: sfera narysuje {n * 460800:,} wiązek, a każde niepoliczone zdjęcie to '
                                    'minuty liczenia na GPU. Kontynuować?'.replace(',', ' ')) != QMessageBox.StandardButton.Yes:
                return
        wagi, kroki, _rgb, _gt, okno = self._wybor()
        if len(kroki) < 2:
            self.lb_zestaw.setText('Za mało checkpointów w zakresie (sekcja Dane).'); return
        w = ZestawLiczenie(wagi, kroki, [z for z, _ in self._zestaw], okno)
        w.postep.connect(lambda a, b, o: (self.pasek_zestaw.setRange(0, b), self.pasek_zestaw.setValue(a), self.lb_zestaw.setText(o)))
        w.gotowe.connect(self._zestaw_gotowe)
        w.blad.connect(lambda s: self.lb_zestaw.setText(f'Zestaw nie policzony: {s}'))
        w.finished.connect(lambda: (setattr(self, '_zestaw_watek', None), self.bt_zestaw.setEnabled(True), self.bt_zestaw_stop.setEnabled(False)))
        self._zestaw_watek = w; self.bt_zestaw.setEnabled(False); self.bt_zestaw_stop.setEnabled(True)
        self.lb_zestaw.setText(f'Wczytuję / liczę {n} historii…')
        w.start()

    def _zestaw_gotowe(self, katalogi):
        try:
            hs = [LD.Historia(k) for k in katalogi]
            for h_ in hs:
                if (self.cb_etyk.currentData() or 'dotkniete') != 'dotkniete':
                    h_.ustaw_etykiete(self.cb_etyk.currentData(), self.sp_etyk_prog.value())
            self.hz = LD.HistoriaZestawu(hs)
        except Exception as e:                                           # noqa: BLE001 (np. różne checkpointy)
            self.lb_zestaw.setText(f'Zestaw nie wczytany: {e}'); return
        self._zestaw_hs = hs
        self.cb_zestaw_akt.blockSignals(True); self.cb_zestaw_akt.clear()
        for i, (_, opis) in enumerate(self._zestaw[:len(hs)]):
            self.cb_zestaw_akt.addItem(f'{i}. {opis}')
        self.cb_zestaw_akt.blockSignals(False)
        self.lb_zestaw.setText(f'Zestaw: {len(hs)} zdjęć, {self.hz.T} koloidów na sferze, {self.hz.T * 768:,} wiązek.'.replace(',', ' '))
        self.cb_zestaw_akt.setCurrentIndex(0)
        self._zestaw_aktywne(0)

    def _zestaw_aktywne(self, i):
        """Aktywne zdjęcie zestawu: wczytuje jego historię jak „Policz / wczytaj” (okienka 1–13, R, podkroki), sfera zostaje zestawem."""
        if self.hz is None or not 0 <= i < len(self._zestaw_hs):
            return
        self._gotowe(str(self._zestaw_hs[i].katalog))

    def _zestaw_idx(self):
        """Numer aktywnego zdjęcia w zestawie (gdy sfera pokazuje zestaw i aktywna historia do niego należy), inaczej None."""
        if self.hz is None or self.h is None or not self.ch_zestaw.isChecked():
            return None
        for i, h_ in enumerate(self.hz.hs):
            if Path(h_.katalog) == Path(self.h.katalog):
                return i
        return None

    def _h_sfery(self):
        """Historia rysowana na sferze: zestaw zdjęć albo aktywna historia."""
        return self.hz if self._zestaw_idx() is not None else self.h

    def _tok_sfery(self, j):
        """Numer cząsteczki aktywnego zdjęcia → numer na sferze (w zestawie: 600·i + j)."""
        i = self._zestaw_idx()
        return int(j) if i is None else i * LD.TOK + int(j)

    def _zestaw_na_sferze(self):
        """Przełącza sferę między zestawem a aktywnym zdjęciem (bez ponownego liczenia)."""
        if self.h is None:
            return
        hs = self._h_sfery()
        self.es.ustaw_historie(hs, self.cb_skala.currentIndex() == 1)
        self._skala = None; self._ks = None
        self._opcje()

    def _buduj_etykiety(self):
        import etykiety_narzedzia as EN
        g = Zwijana('Etykiety: narzędzie (obrys) i obszary oka'); f = QFormLayout(g)
        self.cb_etyk = WaskiCombo()
        for k, opis in EN.TRYBY:
            self.cb_etyk.addItem(opis, k)
        self.cb_etyk.setToolTip('Które z 600 patchy są narzędziem (kolor „narzędzie”, „Wektory dla: narzędzia”, przycisk „narzędzie”,\n'
                                'pierścienie w okienkach 1–3, okienko 2: narzędzie/reszta):\n'
                                '• dotknięta obrysem: każdy patch, w który wchodzi choć jeden piksel obrysu narzędzia — całe narzędzie,\n'
                                '  także patche w większości z tłem (domyślnie);\n'
                                '• pokrycie ≥ próg: tylko patche w co najmniej „próg” wypełnione narzędziem (cienki cystotom: 2–3 patche);\n'
                                '• środek łaty w obrysie: środkowy piksel patcha leży na narzędziu;\n'
                                '• ramka GT: środek patcha w prostokącie ramki (dużo tła).\n'
                                'Rzuty R1–R12 używają etykiety z chwili liczenia stanów 192D (R9 ma własny próg).')
        self.sp_etyk_prog = _spin(0.01, 1.0, 0.5, 0.05, 2); self.sp_etyk_prog.setToolTip('Próg pokrycia patcha dla trybu „pokrycie ≥ próg”.')
        self.sp_oko_prog = _spin(0.05, 1.0, 0.5, 0.05, 2)
        self.sp_oko_prog.setToolTip('Patch należy do źrenicy / rogówki, gdy ich obrys pokrywa co najmniej tyle patcha (jak obszary_oka.py).\n'
                                    'Pierwszeństwo: narzędzie > źrenica > rogówka > tło. ViT nie był uczony na obszarach oka.')
        self.lb_etyk = QLabel('—'); self.lb_etyk.setWordWrap(True)
        f.addRow('Narzędzie', self.cb_etyk); f.addRow('  próg pokrycia', self.sp_etyk_prog); f.addRow('Obszary oka: próg', self.sp_oko_prog)
        f.addRow(self.lb_etyk)
        self.cb_etyk.currentIndexChanged.connect(lambda *_: self._etykieta_zmiana())
        self.sp_etyk_prog.valueChanged.connect(lambda *_: self._etykieta_zmiana() if self.cb_etyk.currentData() == 'pokrycie' else None)
        self.sp_oko_prog.valueChanged.connect(lambda *_: self._etykieta_zmiana(tylko_oko=True))
        self._sek['etykiety'] = g

    def _etykieta_zmiana(self, tylko_oko=False):
        """Nowa etykieta narzędzia (tryb z listy): przelicza h.narz i wszystko, co z niej wynika, i odświeża okienka."""
        h = self.h
        if h is None:
            return
        if not tylko_oko and hasattr(h, 'ustaw_etykiete'):
            h.ustaw_etykiete(self.cb_etyk.currentData() or 'dotkniete', self.sp_etyk_prog.value())
            for e in (self.e1, self.e2, self.e3):
                e.ustaw_historie(h)
            if self.hz is not None:                                    # zestaw: ta sama etykieta we wszystkich zdjęciach
                for h_ in self.hz.hs:
                    h_.ustaw_etykiete(self.cb_etyk.currentData() or 'dotkniete', self.sp_etyk_prog.value())
                self.hz.narz = np.concatenate([np.asarray(h_.narz, bool) for h_ in self.hz.hs])
            self.es.ustaw_skale(self._h_sfery(), self.cb_skala.currentIndex() == 1)
            cz = h.qc[:, h.narz].mean(1) if h.narz.any() else np.zeros((h.K, 2))
            from scipy.ndimage import uniform_filter1d
            self._cz = uniform_filter1d(cz, 9, axis=0, mode='nearest')
            if self.d_pod is not None and '_pp' in self.d_pod:
                self.d_pod['_pp'] = plaski_podkrokow(self.d_pod, h.g, h.narz)
        self._opis_etykiety()
        self._ks = None
        self._kolory_czastek()
        if self.d_pod is not None:
            self.ep.ustaw_podkroki(self.d_pod, self._kol_cz, self._uklad())
            if self.ch_pod.isChecked():
                self.es.ustaw_podkroki(self.d_pod, h.narz)
        self._opcje()

    def _opis_etykiety(self):
        h = self.h
        if h is None:
            self.lb_etyk.setText('—'); return
        import lasery_dane as LD_
        k = h.obszary(self.sp_oko_prog.value()) if hasattr(h, 'obszary') else h.narz.astype(np.uint8) * 3
        ile = ', '.join(f'{LD_.KLASY_OKA[i]} {int((k == i).sum())}' for i in (3, 2, 1, 0)) if hasattr(LD_, 'KLASY_OKA') else ''
        bez_oka = '' if getattr(h, 'oko', None) is not None else ' Brak obrysów oka dla tego zdjęcia.'
        self.lb_etyk.setText(f'{h.etykieta}.  Patche: {ile}.{bez_oka}')

    def _zaznacz_obszar(self, kod):
        if kod < 0 or self.h is None:
            self.ed_tok.setText(''); self.ed_tok.editingFinished.emit(); return
        k = self.h.obszary(self.sp_oko_prog.value()) if hasattr(self.h, 'obszary') else self.h.narz.astype(np.uint8) * 3
        tok = np.flatnonzero(k == kod)
        self.ed_tok.setText(', '.join(str(int(t)) for t in tok)); self.ed_tok.editingFinished.emit()

    def menu_okienka(self, klucz, pos):
        """Prawy klik w okienko: popup z jego ustawieniami (sekcje panelu wypożyczone na czas otwarcia) i akcjami okienka."""
        from okna import przelacz, przelacz_dok
        from panel_ustawien import PopupUstawien
        d = self.doki.get(klucz) or self.doki_ster.get(klucz)
        if klucz.startswith('R') and hasattr(self.rzuty, 'sekcje_okienka'):
            sekcje = self.rzuty.sekcje_okienka(klucz)
            strona = 'rzuty'
        else:
            nazwy = self.SEKCJE_OKIENEK.get(klucz, ())
            sekcje = [self._sek[n] for n in nazwy]
            strona = self._strona_sekcji.get(nazwy[0]) if nazwy else None
        akcje = []
        if d is not None:
            akcje = [('⧉', 'Odepnij okienko jako osobne okno albo przypnij z powrotem.', lambda: d.setFloating(not d.isFloating())),
                     ('⛶', 'Maksymalizuj okienko na jego ekranie (drugi raz wraca).', lambda: przelacz_dok(d)),
                     ('🖵', 'Pełny ekran okienka (Esc wraca), np. stereo na monitorze 3D.', lambda: przelacz(d.widget())),
                     ('🗙', 'Zamknij okienko (wraca z menu „Okienka”).', d.close)]
        p = PopupUstawien.istniejacy(klucz, sekcje)
        if p is not None:                                                # to samo okienko drugi raz: przywołanie, bez przestawiania
            p.show(); p.raise_(); p.activateWindow()
            return
        tytul = d.windowTitle() if d is not None else klucz
        p = PopupUstawien(tytul.split(':')[0], sekcje, akcje,
                          na_panel=(lambda: (self.dok_panel.show(), self.dok_panel.raise_(), self.panel.pokaz(strona))) if strona else None)
        p.klucz = klucz
        from wygoda import ulatw
        ulatw(p)
        p.pokaz_przy(pos)

    def _wykres(self, klucz, tytul, os_y):
        p = pg.PlotWidget(); p.setBackground('#0b1014'); p.showGrid(x=True, y=True, alpha=0.25)
        p.setLabel('left', os_y); p.setMouseEnabled(x=True, y=False)
        kur = pg.InfiniteLine(pos=0, angle=90, movable=True, pen=pg.mkPen('w', width=1.5))
        p.addItem(kur)
        kur.sigPositionChanged.connect(lambda l: None if self._blokada else self.ustaw_czas(float(l.value())))
        p.scene().sigMouseClicked.connect(lambda ev, p=p: self._klik_wykres(p, ev))
        p.kursor = kur
        self._dok(klucz, tytul, p)
        return p

    def _uklad_domyslny(self):
        """Główne miejsce mają widoki: panel parametrów wąską kolumną po lewej (na całą wysokość), podkroki cienkim paskiem
        u góry, pod nim duża sfera (z rzutami R w kartach) i obok pułapka optyczna (z doliną, mechaniką i okienkiem P w kartach),
        na dole niższy wiersz: 1 | 2 | 3 | mapy 4–5 | wykresy 6–10."""
        d, s = self.doki, self.doki_ster
        A, Hz, Vt = Qt.DockWidgetArea, Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        o = self.okna
        o.setCorner(Qt.Corner.TopLeftCorner, A.LeftDockWidgetArea); o.setCorner(Qt.Corner.BottomLeftCorner, A.LeftDockWidgetArea)
        o.addDockWidget(A.LeftDockWidgetArea, s['parametry'])
        o.addDockWidget(A.TopDockWidgetArea, s['podkroki'], Vt)
        o.addDockWidget(A.TopDockWidgetArea, d['s'], Vt)
        o.splitDockWidget(d['s'], d['11'], Hz)
        for a, b in (('11', '12'), ('12', '13'), ('13', 'p')):
            o.tabifyDockWidget(d[a], d[b])
        d['11'].raise_()
        o.addDockWidget(A.BottomDockWidgetArea, d['1'], Hz)
        for a, b in (('1', '2'), ('2', '3'), ('3', '4'), ('4', '6')):
            o.splitDockWidget(d[a], d[b], Hz)
        o.tabifyDockWidget(d['4'], d['5']); d['4'].raise_()
        for a, b in (('6', '7'), ('7', '8'), ('8', '9'), ('9', '10')):
            o.tabifyDockWidget(d[a], d[b])
        d['6'].raise_()

    def showEvent(self, ev):
        """Przy pierwszym pokazaniu: rozmiary doków według prawdziwej wielkości okna (przed pokazaniem Qt je ignoruje
        i sfera, główny widok, zostawała paskiem szerokości kilku centymetrów)."""
        super().showEvent(ev)
        if not getattr(self, '_rozmiary_ok', False):
            self._rozmiary_ok = True
            QTimer.singleShot(0, self._rozmiary_domyslne)

    def _rozmiary_domyslne(self, zastosuj_zapis=True):
        if zastosuj_zapis:
            self._rozmiary_gotowe = True
            if getattr(self, '_uklad_ok', False):                   # zapisany układ był już zastosowany przed pokazaniem okna:
                self._zastosuj_uklad()                              # jeszcze raz, teraz z prawdziwą wielkością okna (rozmiary doków)
                return
        d, s = self.doki, self.doki_ster
        Vt, Hz = Qt.Orientation.Vertical, Qt.Orientation.Horizontal
        W, H = max(self.okna.width() - 440, 700), max(self.okna.height(), 600)
        self.okna.resizeDocks([s['parametry']], [440], Hz)
        self.okna.resizeDocks([s['podkroki'], d['s'], d['1']], [s['podkroki'].sizeHint().height(), int(H * 0.66), int(H * 0.30)], Vt)
        self.okna.resizeDocks([d['s'], d['11']], [int(W * 0.60), int(W * 0.40)], Hz)
        self.okna.resizeDocks([d[k] for k in ('1', '2', '3', '4', '6')], [W // 5] * 5, Hz)

    def _uklad_gotowy(self, nazwa):
        """Gotowe układy z menu „Okienka → Układy okienek” (zaczynają od domyślnego)."""
        self._przywroc_uklad()
        d, o = self.doki, self.okna
        Vt, Hz, A = Qt.Orientation.Vertical, Qt.Orientation.Horizontal, Qt.DockWidgetArea
        W, H = max(o.width() - 440, 700), max(o.height(), 600)
        if nazwa == 'pulapka':
            d['s'].raise_(); d['11'].raise_()
            o.resizeDocks([d['s'], d['11']], [W // 2, W // 2], Hz)
            o.resizeDocks([d['s'], d['1']], [int(H * 0.78), int(H * 0.2)], Vt)
        elif nazwa == 'rzuty':
            r = [k for k in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7') if k in d]
            for k in r:                                          # z kart sfery do osobnej kolumny po prawej
                o.removeDockWidget(d[k])
            o.addDockWidget(A.RightDockWidgetArea, d[r[0]])
            for a, b in zip(r, r[1:]):
                o.tabifyDockWidget(d[a], d[b])
            for k in r:
                d[k].show()
            d[r[0]].raise_(); d['s'].raise_()
            o.resizeDocks([d[r[0]]], [int(W * 0.42)], Hz)
            o.resizeDocks([d['s'], d['1']], [int(H * 0.75), int(H * 0.22)], Vt)
        elif nazwa == 'sfera':
            d['s'].raise_()
            o.resizeDocks([d['s'], d['11']], [int(W * 0.86), int(W * 0.14)], Hz)
            o.resizeDocks([d['s'], d['1']], [int(H * 0.86), int(H * 0.12)], Vt)

    def _slot(self, i, zapisz=False):
        """Ctrl+Shift+1..3 zapamiętuje bieżący układ okienek, Ctrl+1..3 go przywraca (sloty zapisują się w ustawienia.json)."""
        if zapisz:
            self._sloty[i] = bytes(self.okna.saveState().toBase64()).decode('ascii')
            self._komunikat(f'Zapamiętano układ okienek {i} (Ctrl+{i} przywraca).')
        elif i in self._sloty:
            for d in list(self.doki.values()) + list(self.doki_ster.values()):
                d.setFloating(False)
            self.okna.restoreState(QByteArray.fromBase64(self._sloty[i].encode('ascii')))
        else:
            self._komunikat(f'Układ {i} nie jest zapamiętany: Ctrl+Shift+{i} zapisuje bieżący.')

    def _komunikat(self, tekst, ms=6000):
        okno = self.window()
        if hasattr(okno, 'statusBar'):
            okno.statusBar().showMessage(tekst, ms)

    def uklad_okienek(self):
        """Układ okienek (doki: położenie, odpięcie, rozmiary) i zapamiętane sloty, do ustawienia.json (ustawienia.stan)."""
        okno = self.window()
        return dict(okna=bytes(self.okna.saveState().toBase64()).decode('ascii'), wersja_ukladu=2,
                    sloty={str(k): v for k, v in self._sloty.items()}, zestaw=[list(z) for z in getattr(self, '_zestaw', [])],
                    okno=bytes(okno.saveGeometry().toBase64()).decode('ascii') if okno is not self else None)

    def ustaw_uklad_okienek(self, v):
        """Przywraca zapisany układ okienek od razu, także przed pierwszym pokazaniem okna: startują tylko okienka, które były
        otwarte i widoczne przy zapisie (widoki w zamkniętych i schowanych w kartach okienkach nie tworzą kontekstu OpenGL ani
        nie liczą). Po pokazaniu okna układ stosuje się jeszcze raz, żeby rozmiary doków pasowały do prawdziwej wielkości."""
        self._uklad_zapisany = v
        if isinstance(v, dict) and v.get('zestaw') and not self._zestaw:     # lista zdjęć zestawu (liczy się dopiero na żądanie)
            self._zestaw = [(str(a), str(b)) for a, b in v['zestaw']][:32]
            self._zestaw_odswiez_liste()
        self._zastosuj_uklad()

    def _zastosuj_uklad(self):
        v = getattr(self, '_uklad_zapisany', None)
        if not isinstance(v, dict) or v.get('wersja_ukladu') != 2:          # układ z dawnej zakładki (lewa kolumna): domyślny
            return
        try:
            self._sloty.update({int(k): s for k, s in (v.get('sloty') or {}).items()})
            okno = self.window()
            if v.get('okno') and okno is not self and not getattr(self, '_rozmiary_gotowe', False):
                okno.restoreGeometry(QByteArray.fromBase64(v['okno'].encode('ascii')))   # wielkość i ekran głównego okna
            self._uklad_ok = bool(self.okna.restoreState(QByteArray.fromBase64(v['okna'].encode('ascii'))))
        except Exception:                                                # noqa: BLE001 (stary albo uszkodzony zapis: układ domyślny)
            import traceback
            traceback.print_exc()

    def _przywroc_uklad(self):
        for d in list(self.doki.values()) + list(self.doki_ster.values()):
            d.setFloating(False); d.show()
        if self._stan_domyslny is not None:
            self.okna.restoreState(self._stan_domyslny)
            self._rozmiary_domyslne(zastosuj_zapis=False)

    # ------------------------------------------------------------------ okienko „Przejście przez sieć” (podkroki w głębokości)
    def _buduj_przejscie(self):
        pasek = QWidget(); pv = QVBoxLayout(pasek); pv.setContentsMargins(4, 0, 4, 2); pv.setSpacing(2)
        rp = QHBoxLayout()
        self.ch_pod = QCheckBox('Podkroki sieci we wszystkich widokach')
        self.ch_pod.setToolTip('Sfera reflektorów i rzuty 1–3 pokazują jedno przejście przez sieć bieżącego checkpointu: łodyga + PE, '
                               'potem w każdym z 6 bloków uwaga (siły między cząsteczkami) i MLP (768 laserów bloku). Czas cząsteczek = głębokość.')
        self.bt_pod = QPushButton('▶ podkroki'); self.bt_pod.setCheckable(True); self.bt_pod.toggled.connect(self._graj_podkroki)
        self.bt_pod.setToolTip('Animacja przejścia sygnału przez sieć dla bieżącego checkpointu (w pętli).')
        self.suwak_pod = QSlider(Qt.Orientation.Horizontal); self.suwak_pod.setRange(0, 1200)
        self.suwak_pod.valueChanged.connect(lambda x: None if self._blokada else self.ustaw_podkrok(x / 100.0))
        self.sp_pod_tempo = _spin(0.1, 12, 1.0, 0.1, 1, ' podkr./s')
        self.lb_podkrok = QLabel('podkrok 0/12: wejście bloków (łodyga + PE)'); _staly_uklad(self.lb_podkrok)
        self.ch_sync = QCheckBox('synchronizuj kroki z podkrokami')
        self.ch_sync.setToolTip('Odtwarzanie osi czasu (▶ na górze): w każdym checkpoincie przechodzą wszystkie 12 podkroków sieci '
                                '(tempo: podkr./s), dopiero potem następny checkpoint (co n).')
        self.sp_sync_co = _spin(1, 200, 1, 1, sufiks=' ckpt'); self.sp_sync_co.setToolTip('Co ile checkpointów skakać po pełnym przejściu sieci.')
        self.sp_przejscie = _spin(0, 6, 0.5, 0.25, 2, ' podkr.')
        self.sp_przejscie.setToolTip('Długość płynnego powrotu z wyjścia sieci do wejścia następnego checkpointu (w podkrokach); 0 = natychmiast.')
        self.ch_wyp = QCheckBox('wypadkowe wektory'); self.ch_wyp.setChecked(True)
        self.ch_wyp.setToolTip('Wypadkowe przesunięcie każdej cząsteczki w bieżącym podkroku (suma sił uwagi albo laserów): '
                               'niebieskie = uwaga, pomarańczowe = MLP; wybrana cząsteczka grubo z grotem.')
        self.sp_skala_wyp = _spin(0.1, 50, 3.0, 0.5, 1, ' ×'); self.sp_skala_wyp.setToolTip('Długość wypadkowych wektorów na ekranie.')
        self.cb_plynnie = QComboBox(); self.cb_plynnie.addItems(['płynne przejście uwaga ↔ MLP', 'osobne podkroki (skokowo)'])
        self.cb_plynnie.setToolTip('Jak wiązki zmieniają się na granicy podkroków (sfera, okienko 11, „Przejście przez sieć”):\n'
                                   '• płynne: wiązki MLP gasną, a linie i wkłady uwagi następnego podkroku rozjaśniają się (przenikanie);\n'
                                   '  reflektory stoją, w nowym bloku z tych samych miejsc świecą neurony tego bloku pod innymi kątami;\n'
                                   '• osobne podkroki: każdy podkrok osobno, wiązki przełączają się od razu na granicy (jak dawniej).')
        self.sp_plynnie = _spin(0.1, 1.0, 0.6, 0.1, 2, ' podkr.')
        self.sp_plynnie.setToolTip('Szerokość przenikania na granicy podkroków (w podkrokach): 1 = przez cały podkrok, 0,2 = krótko '
                                   'przy granicy. W połowie przenikania obie warstwy wiązek świecą po połowie.')
        # dwa wiersze, żeby nic nie wychodziło za prawą krawędź: [podkroki ▶ suwak tempo opis] i [synchronizacja, powrót, wypadkowe]
        for x in (self.ch_pod, self.bt_pod, self.suwak_pod, self.sp_pod_tempo, self.lb_podkrok):
            rp.addWidget(bez_ukladu(x) if x is self.lb_podkrok else x, 3 if x is self.suwak_pod else 2 if x is self.lb_podkrok else 0)
        rp2 = QHBoxLayout()
        for x in (self.ch_sync, self.sp_sync_co, QLabel('powrót'), self.sp_przejscie, self.ch_wyp, QLabel('długość'), self.sp_skala_wyp,
                  self.cb_plynnie, self.sp_plynnie):
            rp2.addWidget(x)
        rp2.addStretch(1)
        self._rp2 = QWidget(); self._rp2.setLayout(rp2); rp2.setContentsMargins(0, 0, 0, 0)
        for x in (self.ch_wyp,):
            x.toggled.connect(lambda *_: self.ustaw_podkrok(self.s_pod))
        self.sp_skala_wyp.valueChanged.connect(lambda *_: self.ustaw_podkrok(self.s_pod))
        self.cb_plynnie.currentIndexChanged.connect(lambda i: (self.sp_plynnie.setEnabled(i == 0), self.ustaw_podkrok(self.s_pod)))
        self.sp_plynnie.valueChanged.connect(lambda *_: self.ustaw_podkrok(self.s_pod))
        pv.addLayout(rp); pv.addWidget(self._rp2); self._rp2.setVisible(False)
        self.diagram = DiagramSieci(); self.diagram.klik.connect(lambda s: self.ustaw_podkrok(s))
        self.diagram.setToolTip('Schemat MałegoViT (diagram_maly_vit.pdf). Klik w uwagę albo MLP bloku przechodzi do tego podkroku.')
        self.diagram.setVisible(False)
        pv.addWidget(self.diagram)
        # podkroki sieci: osobne okienko (dok „Podkroki sieci”), które można odpiąć i postawić gdziekolwiek, także na drugim ekranie
        pasek.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.dok_podkroki = self._dok_sterowania('podkroki', 'Podkroki sieci: przejście sygnału przez 6 bloków', pasek)
        self.ch_pod.toggled.connect(self._tryb_podkrokow)
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(2)
        g_p = Zwijana('Okienko P: przejście przez sieć (siły uwagi, lasery MLP, tory)'); f_p = QFormLayout(g_p)
        self._sek['przejscie'] = g_p
        r = QHBoxLayout(); r.setContentsMargins(4, 0, 4, 0)
        self.cb_sily = WaskiCombo(); self.cb_sily.addItems(['siły uwagi: brak', 'siły na wybraną cząsteczkę', 'wszystkie pary (linie)'])
        self.cb_sily.setCurrentIndex(1)
        self.cb_sily.setToolTip('Podkrok uwagi: f_ij = Σ_h A^h_ij W_o v^h_j, wkład cząsteczki j w przesunięcie i. Linie = kogo słucha i, '
                                'wektory = siły (kolor = kolor cząsteczki j).')
        self.sp_gain_a = _spin(0.01, 100, 1.0, 0.1, 2); self.sp_gain_a.setToolTip('Jasność linii uwagi (× średnia atencja głów).')
        self.sp_skala_sil = _spin(0.01, 1000, 10.0, 1.0, 2); self.sp_skala_sil.setToolTip('Długość wektorów sił f_ij na ekranie (mnożnik).')
        self.ch_lasery = QCheckBox('lasery MLP'); self.ch_lasery.setChecked(True)
        self.ch_lasery.setToolTip('Podkrok MLP: 768 wektorów z_jc · v_c na wybraną cząsteczkę (kolor = laser, błękit = ujemna jasność z < 0).')
        self.sp_skala_l = _spin(0.01, 1000, 3.0, 0.5, 2); self.sp_skala_l.setToolTip('Długość wektorów laserów na ekranie (mnożnik).')
        self.cb_kolor_cz = WaskiCombo(); self.cb_kolor_cz.addItems(['kolor patcha ze zdjęcia', 'położenie patcha na obrazie (siatka 30 × 20: barwa = kolumna, jasność = wiersz)', 'narzędzie',
                                                                    'obszary oka: narzędzie, źrenica, rogówka, tło'])
        self.cb_tory = WaskiCombo(); self.cb_tory.addItems(['tory: brak', 'tor wybranej cząsteczki', 'tory wszystkich'])
        self.cb_tory.setCurrentIndex(1); self.cb_tory.setToolTip('Droga cząsteczki przez kolejne podkroki (głębokość sieci).')
        self.cb_uklad = WaskiCombo(); self.cb_uklad.addItems(['względem środka, jedna skala na checkpoint (nie LayerNorm)', 'względem środka chmury',
                                                              'surowy strumień rezydualny'])
        self.cb_uklad.setToolTip('Jedna skala: x − średnia tokenów w każdym stanie, podzielone przez JEDNĄ liczbę na cały checkpoint '
                                 '(średnia norma tokenu po 13 stanach), więc ruch bierze się tylko z sił, nie ze zmiany skali. To nie jest '
                                 'LayerNorm (ten normuje każdy token osobno). Względem środka: bez dzielenia (norma rośnie z głębokością). '
                                 'W obu układach względem środka siły są wycentrowane tak samo (A_ij − ā_j, z_jc − z̄_c, biasy znikają), '
                                 'więc suma wektorów = przesunięcie cząsteczki. Surowy: pełny stan z biasami, widać wspólny dryf tokenów. '
                                 'Każdy układ ma własny rzut PCA 3D.')
        # opcje okienka P są w panelu (strona „Podkroki”) i pod prawym klikiem w okienko; samo okienko to widok i opis
        f_p.addRow('Siły uwagi', self.cb_sily); f_p.addRow('  jasność linii uwagi', self.sp_gain_a)
        f_p.addRow('  długość sił', self.sp_skala_sil); f_p.addRow(self.ch_lasery); f_p.addRow('  długość laserów', self.sp_skala_l)
        f_p.addRow('Tory', self.cb_tory); f_p.addRow('Układ', self.cb_uklad); f_p.addRow('Kolor cząsteczek', self.cb_kolor_cz)
        self.vp = WidokGL(); self.vp.setBackgroundColor(TLO); self.vp.setCameraPosition(distance=250, elevation=20, azimuth=-60)
        self.ep = PrzejscieGPU(); self.vp.addItem(self.ep); self.vp.klik_px.connect(self._klik_przejscie)
        v.addWidget(self.vp, 1)
        self.lb_pod = QLabel('Przejście przez sieć liczy się dla bieżącego checkpointu osi czasu (po wczytaniu historii).')
        self.lb_pod.setWordWrap(True); self.lb_pod.setStyleSheet('color:#ffd166; padding:2px 6px; background:#0b1014')
        self.lb_pod.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        v.addWidget(bez_ukladu(self.lb_pod, 3 * self.lb_pod.fontMetrics().height() + 10))   # 3 linie, bez przeliczania układu
        self._dok('p', 'Przejście przez sieć: podkroki (uwaga = siły między cząsteczkami, MLP = lasery)', w)
        self.d_pod = None; self.s_pod = 0.0; self._kamera_pod = False
        self._kol_cz = None
        self.watek_pod = PodkrokiWatek(); self.watek_pod.gotowe.connect(self._podkroki_gotowe)
        self.timer_pod = QTimer(self); self.timer_pod.setInterval(16); self.timer_pod.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer_pod.timeout.connect(self._klatka_podkrokow)
        for x in (self.cb_sily, self.cb_kolor_cz, self.cb_uklad, self.cb_tory):
            x.currentIndexChanged.connect(self._opcje_przejscia)
        for x in (self.sp_gain_a, self.sp_skala_sil, self.sp_skala_l):
            x.valueChanged.connect(self._opcje_przejscia)
        self.ch_lasery.toggled.connect(self._opcje_przejscia)
        self.doki['p'].visibilityChanged.connect(self._p_widoczne)

    def _widac_p(self):
        d = self.doki['p']
        return d.isVisible() and (d.isFloating() or not d.visibleRegion().isEmpty())

    def _p_widoczne(self, wid):
        """Okienko szczegółów podkroków odsłonięte: dosyła dane bieżącego checkpointu, jeśli są nowsze niż w nim."""
        if not wid:
            return
        d = self.d_pod
        if d is not None and getattr(self, '_ep_krok', None) != d['krok']:
            if getattr(self, '_kol_cz', None) is None:
                self._kolory_czastek()
            self.ep.ustaw_podkroki(d, self._kol_cz, self._uklad()); self._ep_krok = d['krok']
            if not self._kamera_pod:
                self._kadruj_przejscie(); self._kamera_pod = True
        self._zlec_podkroki()

    def _kolory_czastek(self):
        if self.h is None:
            return
        self._kol_cz = self._kolory_tryb(self.cb_kolor_cz.currentIndex())

    def _kolory_tryb(self, tryb, h=None):
        """[600, 4] kolory cząstek: 0 kolor patcha ze zdjęcia, 1 położenie na siatce (barwa = kolumna, jasność = wiersz), 2 narzędzie,
        3 obszary oka (narzędzie błękit, źrenica fiolet, rogówka zieleń, tło szare; etykieta i próg z sekcji „Etykiety”)."""
        h = self.h if h is None else h
        K = np.ones((LD.TOK, 4), np.float32)
        if tryb == 0:
            K[:, :3] = h.rgb.reshape(LD.H, 16, LD.W, 16, 3).mean((1, 3)).reshape(-1, 3) / 255.0
            K[:, :3] = np.clip(0.25 + 0.9 * K[:, :3], 0, 1)
        elif tryb == 1:
            import colorsys
            r, c = np.divmod(np.arange(LD.TOK), LD.W)
            K[:, :3] = np.array([colorsys.hsv_to_rgb(cc / LD.W, 0.85, 0.45 + 0.55 * rr / (LD.H - 1)) for rr, cc in zip(r, c)])
        elif tryb == 2 or not hasattr(h, 'obszary'):
            K[:, :3] = np.where(h.narz[:, None], [0.29, 0.94, 1.0], [0.55, 0.6, 0.65])
        else:
            kol = np.array([[0.5, 0.53, 0.58], [0.35, 0.85, 0.45], [0.78, 0.45, 1.0], [0.29, 0.94, 1.0]], np.float32)
            K[:, :3] = kol[h.obszary(self.sp_oko_prog.value())]
        return K

    def _kolory_sfery(self, tryb):
        """Kolory cząsteczek sfery: zestaw = kolory każdego zdjęcia po kolei (600·n), inaczej aktywnego zdjęcia."""
        hs = self._h_sfery()
        return np.concatenate([self._kolory_tryb(tryb, h_) for h_ in hs.hs]) if hs is self.hz else self._kolory_tryb(tryb)

    def _opcje_przejscia(self, *_):
        o = self.ep.opcje
        o.update(sily=self.cb_sily.currentIndex(), gain=self.sp_gain_a.value(), skala_sil=self.sp_skala_sil.value(),
                 lasery=self.ch_lasery.isChecked(), skala_l=self.sp_skala_l.value(), tory=self.cb_tory.currentIndex(),
                 tok=self.sp_tok.value(), rozm=self.sp_rozm3.value())
        if self.sender() in (self.cb_kolor_cz, self.cb_uklad) and self.d_pod is not None:
            self._kolory_czastek(); self.ep.ustaw_podkroki(self.d_pod, self._kol_cz, self._uklad())
            if self.sender() is self.cb_uklad:
                self._kadruj_przejscie()
        self.ep.update()
        self._opis_podkroku()

    def _zlec_podkroki(self):
        h = self.h
        widac = self.doki['p'].isVisible() and (not self.doki['p'].visibleRegion().isEmpty() or self.doki['p'].isFloating())
        if h is None or not (widac or self.ch_pod.isChecked() or self._og_tryb_pod()):
            return
        krok = int(h.kroki[int(np.clip(round(self.t), 0, h.K - 1))])
        if self.d_pod is not None and self.d_pod.get('krok') == krok and self.d_pod.get('_wagi') == h.wagi:
            return
        pam = self._pamiec_pod()
        if krok in pam:                                              # policzony wcześniej (np. z wyprzedzeniem)
            self._podkroki_gotowe(pam[krok]); return
        if getattr(self, '_zlecony', None) == (h.wagi, krok) and (self.watek_pod.isRunning() or self.watek_pod.w_kolejce(krok)):
            return                                                   # ten checkpoint już się liczy
        self._zlecony = (h.wagi, krok)
        self.watek_pod.zlec(h.wagi, krok, h.rgb, getattr(self, '_baza_pod', None), g=h.g, narz=h.narz, **self._prz_arg())

    def _pamiec_pod(self):
        """Policzone podkroki bieżącej historii i zdjęcia: krok → słownik (najwyżej 8, najstarsze wypadają)."""
        klucz = (self.h.wagi, id(self.h)) if self.h is not None else None
        if getattr(self, '_pam_klucz', None) != klucz:
            from collections import OrderedDict
            baza = PZ.wczytaj_baze(PZ.katalog(self.h)) if self.h is not None else None   # rzut zapisanych przejść
            self._pam_klucz, self._pam_pod, self._baza_pod = klucz, OrderedDict(), baza
        return self._pam_pod

    # ------------------------------------------------------------------ zapis przejść na dysk
    def _prz_arg(self):
        if self.h is None:
            return {}
        zapis = self.ch_prz_zapis.isChecked() and self._prz_bajty < self.sp_prz_limit.value() * 1e9
        return dict(kat=PZ.katalog(self.h), zapis=zapis, f16=self.cb_prz_dokl.currentIndex() == 1)

    def _prz_opis(self, dopisek=''):
        if self.h is None:
            return
        kat = PZ.katalog(self.h)
        n, b = PZ.spis(kat); self._prz_bajty = b
        self.lb_prz.setText(f'Zapisane przejścia: {n} z {self.h.K} checkpointów, {b / 1e9:.2f} GB w {kat}. '
                            + ('Wspólny rzut 3D: baza.npz. ' if (kat / 'baza.npz').exists() else '') + dopisek)

    def _prz_wszystkie(self):
        h = self.h
        if h is None or (self._prz_watek is not None and self._prz_watek.isRunning()):
            return
        self._pamiec_pod()
        w = WszystkiePrzejsciaWatek(h, PZ.katalog(h), self.cb_prz_dokl.currentIndex() == 1, self._baza_pod,
                                    self.sp_prz_co.value(), self.sp_prz_limit.value() * 1e9, self._prz_bajty)
        w.postep.connect(lambda i, n, o: (self.pasek_prz.setMaximum(n), self.pasek_prz.setValue(i), self.pasek_prz.setFormat(o)))
        w.baza.connect(self._prz_baza)
        w.koniec.connect(lambda o: (self.bt_prz.setEnabled(True), self.bt_prz_stop.setEnabled(False), self._prz_opis(o)))
        self._prz_watek = w; self.bt_prz.setEnabled(False); self.bt_prz_stop.setEnabled(True); self.pasek_prz.setValue(0)
        w.start()

    def _prz_baza(self, baza):
        if getattr(self, '_baza_pod', None) is None:
            self._baza_pod = baza

    def _prz_przerwij(self):
        if self._prz_watek is not None:
            self._prz_watek.stop = True

    def _z_wyprzedzeniem(self, j):
        """Zlecenie podkroków checkpointu j w tle (bez wyświetlania), żeby następny krok był gotowy bez czekania."""
        h = self.h
        if h is None or j is None or getattr(self, '_baza_pod', None) is None:
            return
        krok = int(h.kroki[j])
        if krok not in self._pamiec_pod() and not self.watek_pod.w_kolejce(krok):
            self.watek_pod.zlec(h.wagi, krok, h.rgb, self._baza_pod, pilne=False, g=h.g, narz=h.narz, **self._prz_arg())

    def _podkroki_gotowe(self, d):
        if 'blad' in d:
            self.lb_pod.setText(f'Przejście przez sieć: błąd {d["blad"]}'); return
        if self.h is None:
            return
        d['_wagi'] = self.h.wagi
        if d.get('_zapisano'):
            self._prz_bajty += int(d['_zapisano'])
            self._prz_opis()
        pam = self._pamiec_pod()
        pam[d['krok']] = d; pam.move_to_end(d['krok'])
        while len(pam) > 8:
            pam.popitem(last=False)
        if self._baza_pod is None:
            self._baza_pod = d.get('baza')                           # wspólny rzut 3D dla kolejnych checkpointów
        krok_teraz = int(self.h.kroki[int(np.clip(round(self.t), 0, self.h.K - 1))])
        if d['krok'] != krok_teraz or getattr(self, '_w_przejsciu', False):
            return                                                   # policzony z wyprzedzeniem: czeka w pamięci
        self.d_pod = d
        if getattr(self, '_kol_cz', None) is None:
            self._kolory_czastek()
        if self._widac_p():
            self.ep.ustaw_podkroki(d, self._kol_cz, self._uklad()); self._ep_krok = d['krok']
            if not self._kamera_pod:
                self._kadruj_przejscie(); self._kamera_pod = True
        self._opcje_przejscia()
        self.diagram.ustaw_lodyge(self.h.rgb, d)
        self._stat_pod = self._statystyki_podkrokow(d)
        if self.ch_pod.isChecked():
            self._zastosuj_podkroki()
        self.ustaw_podkrok(self.s_pod)

    # ------------------------------------------------------------------ tryb „podkroki sieci we wszystkich widokach”
    def _plaski_podkrokow(self, d):
        if '_pp' not in d:
            d['_pp'] = plaski_podkrokow(d, self.h.g, self.h.narz)
        return d['_pp']

    def _zastosuj_podkroki(self):
        d = self.d_pod
        if d is None or self.h is None:
            return
        self._pp = pp = self._plaski_podkrokow(d)
        for e in (self.e1, self.e2, self.e3):
            e.ustaw_historie(pp)
        self.es.ustaw_podkroki(d, self.h.narz)
        zakres = np.abs(np.concatenate([pp.wezly[..., :2], pp.glb], 1)).max((0, 1)) + 1.5
        self.v1.kadruj(0, 0, 2 * zakres[0], 2 * zakres[1])
        self.ustaw_podkrok(self.s_pod)

    def _dopasuj_podkroki(self):
        """Okienko podkroków przypięte u góry albo u dołu dostaje wysokość treści (schemat sieci i drugi wiersz opcji pokazują się
        dopiero w trybie podkroków); szerokości nie rusza, a gdy jej brakuje, treść się przewija zamiast rozpychać okno."""
        d = self.dok_podkroki
        if d.isFloating() or not d.isVisible() or self.okna.dockWidgetArea(d) not in (Qt.DockWidgetArea.TopDockWidgetArea,
                                                                                      Qt.DockWidgetArea.BottomDockWidgetArea):
            return
        wys = d.widget().sizeHint().height() + (d.titleBarWidget().sizeHint().height() if d.titleBarWidget() else 0) + 4
        self.okna.resizeDocks([d], [min(wys, max(120, self.okna.height() // 2))], Qt.Orientation.Vertical)

    def _tryb_podkrokow(self, on):
        self.diagram.setVisible(on); self._rp2.setVisible(on)            # drugi wiersz opcji podkroków tylko w trybie podkroków
        QTimer.singleShot(0, self._dopasuj_podkroki)
        h = self.h
        if h is None:
            return
        if on:
            if self.bt_play.isChecked() and not self.ch_sync.isChecked():
                self.bt_play.setChecked(False)
            if self.d_pod is not None and self.d_pod.get('krok') == int(h.kroki[int(np.clip(round(self.t), 0, h.K - 1))]):
                self._zastosuj_podkroki()
            else:
                self._zlec_podkroki()
        else:
            self._pp = None
            for e in (self.e1, self.e2, self.e3):
                e.ustaw_historie(h)
            self.es.ustaw_historie(self._h_sfery(), self.cb_skala.currentIndex() == 1)
            zakres = np.abs(np.concatenate([h.wezly[..., :2], h.atomy[..., :2], h.glb], 1)).max((0, 1)) + 1.5
            self.v1.kadruj(0, 0, 2 * zakres[0], 2 * zakres[1])
            for e in (self.e1, self.e2, self.e3):
                e.opcje['pod_kol'] = None
            self.ustaw_czas(self.t)

    def _statystyki_podkrokow(self, d):
        if '_stat' not in d:
            d['_stat'] = statystyki_podkrokow(d)
        return d['_stat']

    def _uklad(self):
        return ('_n', '_c', '')[self.cb_uklad.currentIndex()]

    def _kadruj_przejscie(self):
        from PyQt6.QtGui import QVector3D
        P = self.d_pod['pos' + self._uklad()].reshape(-1, 3)
        lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
        c = (lo + hi) / 2
        self.vp.opts['center'] = QVector3D(*[float(x) for x in c])
        self.vp.setCameraPosition(distance=2.4 * float(np.linalg.norm(hi - lo)), elevation=20, azimuth=-60)

    def _plynnie(self):
        """Szerokość przenikania uwaga ↔ MLP w podkrokach (0 = osobne podkroki, skokowo)."""
        return self.sp_plynnie.value() if hasattr(self, 'cb_plynnie') and self.cb_plynnie.currentIndex() == 0 else 0.0

    def ustaw_podkrok(self, s):
        self.s_pod = s = float(np.clip(s, 0, 12))
        self.es.opcje['plynnie'] = self.ep.opcje['plynnie'] = self._plynnie()
        self.ep.ustaw_podkrok(s); self.diagram.ustaw(s)
        self._blokada = True; self.suwak_pod.setValue(int(round(s * 100))); self._blokada = False
        k = int(np.clip(np.floor(s), 0, 11))
        nazwa = ('wyjście sieci → LayerNorm → głowa' if s >= 12 else 'wejście bloków (łodyga + PE)' if s == 0 else
                 f'blok {k // 2 + 1}: ' + ('uwaga — siły między cząsteczkami' if k % 2 == 0 else 'MLP — 768 laserów bloku'))
        st = getattr(self, '_stat_pod', None)
        if st is not None and s < 12:
            sr, pr = float(st['stos'][k]), float(st['prom'][k])
            nazwa += (f';  rozrzut chmury ×{pl(sr, 3)}, promieniowo {pr:+.3f} r ('
                      + ('do środka' if pr < 0 else 'na zewnątrz') + ')')
        self.lb_podkrok.setText(f'podkrok {pl(s, 2)}/12: {nazwa}')
        self.es.opcje.update(wypadkowe=self.ch_wyp.isChecked(), skala_wyp=self.sp_skala_wyp.value())
        kol_pod = None if (s >= 12 or not self.ch_wyp.isChecked()) else ((0.35, 0.8, 1.0) if k % 2 == 0 else (1.0, 0.62, 0.25))
        for e in (self.e1, self.e2, self.e3):
            e.opcje['pod_kol'] = kol_pod if self.ch_pod.isChecked() else None
            e.opcje['pod_skala'] = self.sp_skala_wyp.value()
        pp = getattr(self, '_pp', None)
        if self.ch_pod.isChecked() and pp is not None:
            for e in (self.e1, self.e2, self.e3):
                e.opcje['L'] = int(min(np.floor(s), 3)); e.ustaw_czas(s)
            self.es.ustaw_podkrok(s)
            u = s - np.floor(s); a = int(np.clip(np.floor(s), 0, 12)); b = min(a + 1, 12)
            par = pp.par[a] * (1 - u) + pp.par[b] * u
            self.lb1.setText(f'podkroki sieci: przesunięcie {pl(float(np.hypot(*par[:2])))} kom.   obrót {par[2]:+.1f}°   skala ×{pl(par[3], 3)}'
                             f'   (mapa liniowa stanów na adresy patchy, R² {pl(pp.r2)})')
            self.lb2.setText(f'odkształcenie siatki w głębokości: średnio {pl(float(pp.lok[a].mean() * (1 - u) + pp.lok[b].mean() * u))} kom.;  '
                             f'{nazwa}')
            self.lb3.setText(f'droga tokenów przez ostatnie podkroki (ślad); przesunięcie w podkroku średnio {pl(float(pp.vib[b].mean()))} kom.')
            cz = pp.cz[a] * (1 - u) + pp.cz[b] * u
            if self.ch_kamera.isChecked():
                self.v3.kadruj(cz[0], cz[1], self.sp_zbl.value(), self.sp_zbl.value() * 2 / 3)
            zb = self.sp_zbl.value()
            self.e2.opcje['box'] = (cz[0] - zb / 2, cz[1] - zb / 3, cz[0] + zb / 2, cz[1] + zb / 3) if self.ch_kamera.isChecked() else None
            self.lbs.setText(f'podkroki: {nazwa};  sfera = stany względem środka, jedna skala na checkpoint, reflektory bloku {k // 2 + 1} '
                             f'(błękit = ujemna jasność)')
        self._opis_podkroku()
        if self._og_tryb_pod():
            self._og_podkrok(s)

    def _graj_podkroki(self, on):
        self.bt_pod.setText('⏸ podkroki' if on else '▶ podkroki')
        self._zegar_pod = None
        self._s_anim = self.s_pod
        self.timer_pod.start() if on else self.timer_pod.stop()

    def _klatka_podkrokow(self):
        """Pętla 0 → 12 podkroków i krótka pauza na wyjściu sieci."""
        a = getattr(self, '_s_anim', self.s_pod) + self.sp_pod_tempo.value() * self._dt('pod')
        if a > 12.8:
            a = 0.0
        self._s_anim = a
        self.ustaw_podkrok(min(a, 12.0))

    def _opis_podkroku(self):
        d = self.d_pod
        if d is None:
            return
        s = self.s_pod; k = int(np.clip(np.floor(s), 0, 11)); j = self.sp_tok.value()
        rr, cc = divmod(j, LD.W)
        pos = d['pos']
        if s >= 12:
            self.lb_pod.setText(f'krok {d["krok"]}: koniec przejścia → LayerNorm → głowa: pewność {pl(d["pewnosc"])}, '
                                f'ramka [{", ".join(f"{x:.0f}" for x in d["ramka"])}] px;  patch {j} (wiersz {rr}, kolumna {cc}): '
                                f'mapa środka {pl(float(d["mapa"][j]))}.  Błąd względny „suma sił = ruch”: surowy {d["sprawdzenie"]:.1e}, '
                                f'środek {d.get("sprawdzenie_c", float("nan")):.1e}, jedna skala {d.get("sprawdzenie_n", float("nan")):.1e}')
            return
        b = k // 2
        dx = np.linalg.norm(pos[k + 1] - pos[k], axis=1)
        if k % 2 == 0:
            A = d['ah'][b, :, j].mean(0); A[j] = 0
            top = np.argsort(A)[::-1][:3]
            f = np.einsum('hj,hjd->jd', d['ah'][b, :, j], d['uh'][b])
            mf = np.linalg.norm(f, axis=1); mf[j] = 0
            gr = np.stack(np.divmod(np.arange(LD.TOK), LD.W), 1)
            sas = np.abs(gr - gr[j]).max(1) <= 1
            udz = float(mf[sas].sum() / max(mf.sum(), 1e-12))
            self.lb_pod.setText(f'krok {d["krok"]}, blok {b + 1}, UWAGA: Δx_i = Σ_j f_ij + b_o, f_ij = Σ_h A^h_ij W_o v^h_j.  '
                                f'Średnie |Δx| {pl(float(dx.mean()))};  patch {j} (w{rr}, k{cc}): |Δx| {pl(float(dx[j]))}, słucha najbardziej '
                                + ', '.join(f'{t} (w{t // LD.W}, k{t % LD.W}) {A[t]:.0%}' for t in top)
                                + f';  udział sąsiadów na siatce w sile {udz:.0%} (8 z 599 patchy).')
        else:
            z = d['z'][b, j]; L = z[:, None] * d['pv'][b]
            m = np.linalg.norm(L, axis=1); top = np.argsort(m)[::-1][:3]
            self.lb_pod.setText(f'krok {d["krok"]}, blok {b + 1}, MLP: Δx_j = Σ_c z_jc v_c + b₂ (768 laserów).  Średnie |Δx| {pl(float(dx.mean()))};  '
                                f'patch {j} (w{rr}, k{cc}): |Δx| {pl(float(dx[j]))}, świeci {(z > 0).sum()} laserów, najsilniejsze '
                                + ', '.join(f'c{c} (z {pl(float(z[c]))})' for c in top) + '.')

    def _klik_przejscie(self, x, y):
        if self.d_pod is None:
            return
        s = self.s_pod; a = int(np.clip(np.floor(s), 0, 11)); u = s - a
        X = self.d_pod['pos' + self._uklad()]
        P = X[a] * (1 - u) + X[min(a + 1, 12)] * u
        self._wybierz_tok(self.vp, P, x, y)

    # ------------------------------------------------------------------ wybór danych
    def _lista_historii(self):
        from historie_checkpointow import HISTORIE
        self._historie = [(h.nazwa, h.wagi, h.na_epoke) for h in HISTORIE if h.wagi.is_dir()]
        self._historie.sort(key=lambda x: '35 000' not in x[0])          # bieg 35 000 (filmy) na początku
        self.cb_historia.blockSignals(True)
        self.cb_historia.clear()
        for n, w, _ in self._historie:
            self.cb_historia.addItem(n, str(w))
        self.cb_historia.blockSignals(False)

    def _inny_katalog(self):
        k = QFileDialog.getExistingDirectory(self, 'Katalog wag (k<krok>.pt)', str(LD.ROOT / 'assets'))
        if k:
            self.cb_historia.addItem(f'katalog: {Path(k).name}', k)
            self.cb_historia.setCurrentIndex(self.cb_historia.count() - 1)

    def _lista_zdjec(self):
        """Filmy zbioru uczącego (kolejność manifestu), na końcu grupa plików testowych."""
        self._filmy = {}
        try:
            for r in LD.zbior_uczacy():
                self._filmy.setdefault(r['film'], []).append(r)
        except OSError as e:
            self.lb_gt.setText(f'Brak manifestu zbioru uczącego: {e}')
        self.cb_film.blockSignals(True)
        self.cb_film.clear()
        for film, w in self._filmy.items():
            self.cb_film.addItem(f'{film} ({len(w)} klatek, nr {w[0]["nr"]}–{w[-1]["nr"]})', film)
        self.cb_film.addItem('Pliki testowe (app/zdjeciatestowe, poza zbiorem uczącym)', '')
        self.cb_film.blockSignals(False)
        self._klatki_filmu()

    def _klatki_filmu(self):
        film = self.cb_film.currentData()
        self.cb_zdjecie.blockSignals(True)
        self.cb_zdjecie.clear()
        if film:
            for r in self._filmy[film]:
                self.cb_zdjecie.addItem(f'kl. {r["klatka"]:06d}  (nr {r["nr"]})', f'{LD.PREFIKS_UCZACY}{r["nr"]}')
        else:
            if LD.ZDJECIE_FILMU.exists():
                self.cb_zdjecie.addItem('TRAIN 0 (zdjęcie filmów 2, 12 i 16)', str(LD.ZDJECIE_FILMU))
            for p in sorted((LD.APP / 'zdjeciatestowe').glob('*.jpg')):
                ucz = LD.czy_w_zbiorze_uczacym(p)
                self.cb_zdjecie.addItem(p.name + (f'  (w zbiorze uczącym, nr {ucz["nr"]})' if ucz else ''), str(p))
        self.cb_zdjecie.blockSignals(False)
        self._po_wyborze()

    def _plik_zdjecia(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Zdjęcie', str(LD.APP / 'zdjeciatestowe'), 'Obrazy (*.jpg *.jpeg *.png)')
        if p:
            self.cb_film.setCurrentIndex(self.cb_film.count() - 1)        # grupa plików
            self.cb_zdjecie.addItem(Path(p).name, p)
            self.cb_zdjecie.setCurrentIndex(self.cb_zdjecie.count() - 1)

    def _wybor(self):
        """(katalog wag, lista kroków, rgb, ramka, okno) dla bieżących kontrolek."""
        wagi = Path(self.cb_historia.currentData() or '')
        kroki = sorted(int(p.stem[1:]) for p in wagi.glob('k*.pt') if p.stem[1:].isdigit()) if wagi.is_dir() else []
        kroki = [k for k in kroki if self.sp_od.value() <= k <= self.sp_do.value()][::self.sp_co.value()]
        zd = self.cb_zdjecie.currentData()
        if not zd:
            return wagi, kroki, None, None, self.sp_okno.value()
        rgb, roz = LD.wczytaj_zdjecie(zd)
        gt, opis = LD.ramka_prawdziwa(zd, roz)
        self.lb_gt.setText(opis + ('' if gt is None else f': [{", ".join(f"{v:.0f}" for v in gt)}] px'))
        return wagi, kroki, rgb, gt, self.sp_okno.value()

    def _po_wyborze(self):
        try:
            wagi, kroki, rgb, gt, okno = self._wybor()
        except Exception as e:                                           # noqa: BLE001
            self.lb_stan.setText(f'Błąd wyboru: {e}'); return
        if rgb is None or len(kroki) < 2:
            self.lb_stan.setText('Za mało checkpointów w zakresie albo brak zdjęcia.'); return
        jest = (LD.CACHE / LD.klucz(wagi, kroki, rgb, gt, okno) / 'dane.npz').exists()
        odst = np.median(np.diff(kroki)) if len(kroki) > 1 else 0
        uw = '' if odst <= okno else f'  Uwaga: mediana odstępu checkpointów {odst:.0f} kroków > okno ±{okno}: centra = same tokeny, drgań nie będzie.'
        self.lb_stan.setText(f'{len(kroki)} checkpointów ({kroki[0]}–{kroki[-1]}). '
                             + ('Jest w cache: wczyta się od razu.' if jest else 'Nie liczone: pierwsze liczenie czyta wszystkie pliki wag (minuty).') + uw)

    def _wczytaj_z_cache(self):
        try:
            wagi, kroki, rgb, gt, okno = self._wybor()
            if rgb is not None and len(kroki) >= 2 and (LD.CACHE / LD.klucz(wagi, kroki, rgb, gt, okno) / 'dane.npz').exists():
                self._licz()
        except Exception:                                                # noqa: BLE001
            pass

    def _licz(self):
        if self.watek is not None:
            return
        wagi, kroki, rgb, gt, okno = self._wybor()
        if rgb is None or len(kroki) < 2:
            self.lb_stan.setText('Za mało checkpointów albo brak zdjęcia.'); return
        self.watek = Liczenie(wagi, kroki, rgb, gt, okno)
        self.watek.postep.connect(self._postep); self.watek.gotowe.connect(self._gotowe); self.watek.blad.connect(self._blad)
        self.watek.finished.connect(lambda: setattr(self, 'watek', None))
        self.bt_licz.setEnabled(False); self.bt_stop.setEnabled(True)
        self.pasek.setRange(0, len(kroki)); self.pasek.setValue(0)
        self.lb_stan.setText('Liczenie na GPU (wczytanie modelu i timm może chwilę potrwać)...')
        self.watek.start()

    def _przerwij(self):
        if self.watek is not None:
            self.watek.stop = True

    def _postep(self, i, n, opis):
        self.pasek.setRange(0, n); self.pasek.setValue(i); self.lb_stan.setText(opis)

    def _blad(self, tekst):
        self.bt_licz.setEnabled(True); self.bt_stop.setEnabled(False)
        self.lb_stan.setText(f'Nie policzono: {tekst}')

    def _gotowe(self, katalog):
        self.bt_licz.setEnabled(True); self.bt_stop.setEnabled(False)
        self.h = h = LD.Historia(katalog)
        if hasattr(h, 'ustaw_etykiete') and (self.cb_etyk.currentData() or 'dotkniete') != 'dotkniete':
            h.ustaw_etykiete(self.cb_etyk.currentData(), self.sp_etyk_prog.value())     # etykieta wybrana w sekcji „Etykiety”
        self._opis_etykiety()
        self.lb_stan.setText(f'Wczytano {h.K} checkpointów (kroki {int(h.kroki[0])}–{int(h.kroki[-1])}), {Path(katalog).name}. '
                             f'Odstęp sąsiadów w kroku startu {pl(h.a0, 3)}; 3 kierunki rzutu niosą {h.udzial_D:.0%} energii przesunięć.')
        for e in (self.e1, self.e2, self.e3):
            e.ustaw_historie(h)
        self.es.ustaw_historie(self._h_sfery(), self.cb_skala.currentIndex() == 1)   # zestaw zdjęć albo to zdjęcie
        self.e4.ustaw_historie(h); self.e5.ustaw_historie(h)
        self.sp_tok.setValue(h.fok); self.sp_refl.setValue(h.refl_pokaz)
        self._wykresy()
        self._blokada = True
        self.suwak.setRange(0, (h.K - 1) * 100)
        self._ustaw_ab(None)                                         # nowa historia: punkty A i B od nowa
        self._blokada = False
        self.v2.kadruj(0, 0, LD.W + 3, LD.H + 3)
        zakres = np.abs(np.concatenate([h.wezly[..., :2], h.atomy[..., :2], h.glb], 1)).max((0, 1)) + 1.5
        self.v1.kadruj(0, 0, 2 * zakres[0], 2 * zakres[1])
        cz = h.qc[:, h.narz].mean(1) if h.narz.any() else np.zeros((h.K, 2))
        from scipy.ndimage import uniform_filter1d
        self._cz = uniform_filter1d(cz, 9, axis=0, mode='nearest')
        self._skala = None
        self.d_pod = None; self._kamera_pod = False
        self._kolory_czastek()
        self._og = None; self._og_klucz = None
        self.eog.czysc(); self.es_og.czysc()
        self._dol = None; self._dol_zlecony = None; self.edol.czysc(); self.es_dol.czysc(); self.es_dol.setVisible(False)
        self._mech = None; self._mech_zlecony = None; self.emech.czysc()
        self._prz_opis()
        self._opcje()
        self.rzuty.ustaw_historie(h)
        self.ustaw_czas(0.0)
        self._og_zlec()

    def _wykresy(self):
        h = self.h
        ix = np.arange(h.K)
        tk = [k for k in (0, 300, 1000, EP, EP * 5, EP * 10, EP * 20, EP * 40, EP * 60) if h.kroki[0] <= k <= h.kroki[-1]]
        nazwy = {0: '0', 300: '300', 1000: '1k', EP: 'e1', EP * 5: 'e5', EP * 10: 'e10', EP * 20: 'e20', EP * 40: 'e40', EP * 60: 'e60'}
        ticks = [[(h.indeks(k), nazwy[k]) for k in tk]]
        for p in (self.p6, self.p7, self.p8, self.p9, self.p10):
            for it in [i for i in p.listDataItems()]:
                p.removeItem(it)
            p.getAxis('bottom').setTicks(ticks)
        if getattr(h, 'obr_dryf', None) is not None:                     # obrót igieł: drganie (100 kroków) i dryf netto (2000 kroków)
            if self.p10.plotItem.legend is None:
                self.p10.addLegend(offset=(5, 5))
            else:
                self.p10.plotItem.legend.clear()
            self.p10.plot(ix, np.median(h.obr_drg[..., 3], 1), pen=pg.mkPen('#e8e8e8', width=1.2), name='drganie (100 kroków)')
            self.p10.plot(ix, np.median(h.obr_dryf[..., 3], 1), pen=pg.mkPen('#ffaa50', width=1.5), name='dryf netto (2000 kroków)')
            self.p10.setXRange(0, h.K - 1)
        if self.p6.plotItem.legend is None:
            self.p6.addLegend(offset=(5, 5))
        else:
            self.p6.plotItem.legend.clear()
        self.p6.plot(ix, h.glob, pen=pg.mkPen('#ffd166', width=1.3), name='1. globalny')
        self.p6.plot(ix, h.lok_m, pen=pg.mkPen('#c264ff', width=1.3), name='2. lokalny (centra od adresów)')
        self.p6.plot(ix, h.vib_m, pen=pg.mkPen('#ff7a45', width=1.1), name='3. drgania')
        if np.isfinite(h.iou).any():
            self.p7.plot(ix, np.nan_to_num(h.iou), pen=pg.mkPen('#4af0ff', width=1.3))
        self.p8.plot(ix, h.pew, pen=pg.mkPen('#ffb000', width=1.3))
        self.p9.plot(ix, h.pary, pen=pg.mkPen('#ff9a5c', width=1.3))
        for p in (self.p7, self.p8, self.p9):
            p.setYRange(0, 1)
        for p in (self.p6, self.p7, self.p8, self.p9):
            p.setXRange(0, h.K - 1)

    # ------------------------------------------------------------------ opcje
    def _opcje(self, *_):
        for e, v in ((self.e1, self.v1), (self.e2, self.v2), (self.e3, self.v3)):
            e.opcje.update(rozm=self.sp_rozm.value(), jas=self.sp_jas.value(), sig=self.sp_sig.value(), strz=self.sp_strz.value(),
                           narz=self.ch_narz2.isChecked(), tok=self.sp_tok.value() if self.h else -1,
                           pxc=v.px_na_komorke())
            e.update()
        o = self.es.opcje
        self.es.ustaw_wybor(self._wybrane_refl(), [self._tok_sfery(j) for j in self._wybrane_tok()])   # numery na sferze
        swiatlo = self.ch_swiatlo.isChecked()
        o.update(wiazki=self.cb_wiazki.currentIndex() if swiatlo else 0, gain=self.sp_gain.value(), prog=self.sp_prog.value(), tok=self._tok_sfery(self.sp_tok.value()),
                 refl=self.sp_refl.value(), sfera=self.sp_sfera.value() if self.ch_sfera.isChecked() else 0.0,
                 grub_w=self.sp_grub_w.value(), kierunek=self.ch_kier.isChecked(), ekspozycja=self.sp_eksp.value(),
                 widok=self.cb_widok_s.currentIndex(), wek_kol=self.cb_wek_kol.currentIndex(), wek_pul=self.cb_wek_pul.currentIndex(),
                 filtr=(0, 1, 2)[self.cb_wek_dla.currentIndex()], skala_wek=self.sp_skala_wek.value(),
                 kol_sciezki=self.cb_kol_sciezki.currentIndex(), zanik=float(self.sp_zanik.value()),
                 sily=(self.cb_sily.currentIndex() if hasattr(self, 'cb_sily') else 1) if swiatlo else 0, gain_a=self.sp_gain_a.value() if hasattr(self, 'sp_gain_a') else 1.0, reflektory=self.ch_refl.isChecked(), rozm=self.sp_rozm3.value(),
                 rozm_pul=self.sp_rozm_pul.value(), jas_pul=self.sp_jas_pul.value(), skala_chmury=self.sp_skala_chm.value(),
                 obrot=self.cb_obrot.currentIndex(), dl_obr=self.sp_dl_obr.value())
        if self.h is not None:
            wyg = (self.cb_wyg.currentIndex(), self.sp_wyg_okno.value(), self.sp_wyg_alfa.value())
            if getattr(self.es, '_wyg', None) != wyg:
                self.es.ustaw_wygladzanie(*wyg)
            ks = self.cb_kol_sciezki.currentIndex()
            if getattr(self, '_ks', None) != (ks, id(self.h)) and ks in (1, 2, 3, 5):
                self._ks = (ks, id(self.h)); self.es.ustaw_kolory_sciezek(self._kolory_sfery(ks - 1 if ks < 4 else 3))
        if self.h is not None and getattr(self, '_skala', None) != self.cb_skala.currentIndex():
            self._skala = self.cb_skala.currentIndex(); self.es.ustaw_skale(self._h_sfery(), self._skala == 1)
            if getattr(self, '_og', None) is not None:                     # pułapka w tej samej skali co sfera
                self._og_zastosuj(self._og_klucz, self._og, kamera=False)
        if hasattr(self, 'eog'):
            self.eog.opcje['skala_chmury'] = self.es_og.opcje['skala_chmury'] = self.sp_skala_chm.value()
        if getattr(self, '_dol', None) is not None:
            self._dol_przeksztalcenie()
        self.es.update()
        al = self.sp_krycie.value(); gam = 0.5 if self.ch_pierw.isChecked() else 1.0
        for k in self.e4.kafelki:
            k.update(alfa=max(al, 0.05), gamma=gam)
        mapa = self.cb_mapa.currentIndex()
        if self.h is not None:
            kody = {0: dict(mapa=7, norma=1, vmax=self.h.LOK99, gamma=1.0, lut=1), 1: dict(mapa=8, norma=1, vmax=self.h.VIB97, gamma=1.0, lut=0),
                    2: dict(mapa=6, norma=0, gamma=gam, lut=2),
                    9: dict(mapa=9, norma=1, gamma=1.0, lut=3, vmax=self._vmax_oswietlenia())}
            k = kody.get(mapa, dict(mapa=mapa - 3, norma=0, gamma=gam, lut=2))
            self.e5.kafelki = [dict(x=0, y=0, alfa=al, **k)]
        for e in (self.e4, self.e5):
            e.ramki = dict(model=self.ch_rm.isChecked(), gt=self.ch_rg.isChecked()); e.update()
        if hasattr(self, 'ep'):
            self._opcje_przejscia()
        if self.h is not None:
            self.ustaw_czas(self.t)

    def _wybrane_refl(self):
        return lista_liczb(f'{self.sp_refl.value()} {self.ed_refl.text()}', 768)

    def _wybrane_tok(self):
        return lista_liczb(f'{self.sp_tok.value()} {self.ed_tok.text()}', 600)

    def _vmax_oswietlenia(self):
        """Skala mapy łącznego oświetlenia: 99,5. percentyl sumy z wybranych reflektorów, po wszystkich checkpointach (co ~20.)."""
        c = tuple(self._wybrane_refl())
        if getattr(self, '_vmax_osw', (None,))[0] != (id(self.h), c):
            ks = range(0, self.h.K, max(1, self.h.K // 20))
            v = np.concatenate([MapyNaZdjeciu.oswietlenie(self.h.Z[k], c) for k in ks])
            self._vmax_osw = ((id(self.h), c), max(float(np.percentile(v, 99.5)), 1e-6))
        return self._vmax_osw[1]

    def _tok_narzedzia(self):
        if self.h is not None:
            self.sp_tok.setValue(self.h.fok)

    def _refl_pokaz(self):
        if self.h is not None:
            self.sp_refl.setValue(self.h.refl_pokaz)

    # ------------------------------------------------------------------ pułapka optyczna (ogniskowa wiązek, okienko 11)
    def _og_klucz_teraz(self):
        h = self.h
        G = int(self.sp_og_G.value()) if self.ch_og_pole.isChecked() else 1        # G = 1: tylko wiązki wybranej cząsteczki
        return (str(h.katalog), int(self.sp_tok.value()), G, round(float(self.sp_og_zas.value()), 2))

    def _og_zlec(self, wymus=False):
        """Pułapka dla wybranej cząsteczki: z cache od razu, inaczej liczenie w tle (stare liczenie innej cząsteczki jest przerywane)."""
        h = self.h
        if h is None or not self.ch_og.isChecked():
            return
        klucz = self._og_klucz_teraz()
        if klucz == self._og_klucz and self._og is not None and not wymus:
            return
        if not wymus and not (self.widac('11') or (self.ch_og_sfera.isChecked() and self.widac('s'))):
            self.lb_og.setText('Pułapka policzy się, gdy okienko 11 (albo pułapka na sferze) będzie widoczne.')
            return                                                       # nic nie liczymy dla niewidocznego okienka (_odslonieto)
        w = self._og_watek
        if w is not None and w.isRunning():
            if getattr(w, 'klucz', None) == klucz:
                return
            k_ = getattr(w, 'klucz', None)
            if k_ is not None and (k_[0], k_[2], k_[3]) == (klucz[0], klucz[2], klucz[3]) and klucz[1] in w.toki:
                self.lb_og.setText(f'Pułapka cząsteczki {klucz[1]} liczy się razem z {", ".join(map(str, w.toki))}; pokaże się po skończeniu.')
                return                                                   # ta cząsteczka już się liczy razem z inną: czekamy
            w.stop = True
            self._og_stare = [x for x in getattr(self, '_og_stare', []) if x.isRunning()] + [w]
        d = OGN.wczytaj(h, *klucz[1:])
        if d is not None:
            self._og_zastosuj(klucz, d)
            return
        razem = [t for t in lista_liczb(self.ed_og_razem.text(), self._n_tok()) if t != klucz[1]][:8]
        w = OgniskowaWatek(h, *klucz[1:], razem=razem); w.klucz = klucz
        w.postep.connect(self._og_postep)
        w.gotowe.connect(lambda args, W, k=klucz: self._og_gotowe(k, W))
        w.blad.connect(self._og_blad)
        self._og_watek = w; self.bt_og_stop.setEnabled(True); self.pasek_og.setValue(0)
        self.lb_og.setText(f'Liczę pułapkę cząsteczki {klucz[1]}' + (f' (razem z {", ".join(map(str, razem))})' if razem else '')
                           + f' we wszystkich {h.K} checkpointach (w tle)…')
        w.start()

    def _og_przerwij(self):
        if self._og_watek is not None:
            self._og_watek.stop = True

    def _og_postep(self, i, n, opis):
        self.pasek_og.setMaximum(n); self.pasek_og.setValue(i); self.pasek_og.setFormat(opis)

    def _og_blad(self, tekst):
        self.bt_og_stop.setEnabled(False)
        self.lb_og.setText(f'Pułapka: {tekst}')

    def _og_gotowe(self, klucz, W):
        """W = {cząsteczka: wynik} (wybrana i liczone razem z nią); pokazujemy tę, która jest wybrana TERAZ (mogła się zmienić na
        jedną z liczonych razem w trakcie liczenia)."""
        self.bt_og_stop.setEnabled(False)
        if self.h is None:
            return
        teraz = self._og_klucz_teraz()
        if (teraz[0], teraz[2], teraz[3]) == (klucz[0], klucz[2], klucz[3]) and teraz[1] in W:
            self._og_zastosuj(teraz, W[teraz[1]])
        elif teraz != self._og_klucz:
            self._og_zlec()                                              # wybrano coś innego w trakcie: teraz to

    def _n_tok(self):
        return int(self.sp_tok.maximum()) + 1

    def _og_menu_policzone(self, m):
        """Menu „▾ policzone”: wybrana cząsteczka i cząsteczki z „Licz razem z”, ✓ = pułapka gotowa na dysku (ta sama siatka i zasięg)."""
        m.clear()
        h = self.h
        if h is None:
            m.addAction('najpierw wczytaj historię').setEnabled(False)
            return
        _, tok, G, zas = self._og_klucz_teraz()
        toki = list(dict.fromkeys([tok] + lista_liczb(self.ed_og_razem.text(), self._n_tok())))
        for t in toki:
            gotowa = OGN.plik(h, t, G, zas).exists()
            a = m.addAction(f'{"✓" if gotowa else "…"} cząsteczka {t}' + (' (wybrana)' if t == tok else ''))
            a.triggered.connect(lambda _=False, t=t: self.sp_tok.setValue(t))

    def _og_zastosuj(self, klucz, d, kamera=True):
        h = self.h
        if h is None or len(d['kroki']) != h.K or np.any(np.asarray(d['kroki']) != np.asarray(h.kroki)):
            return
        self._og, self._og_klucz = d, klucz
        r = h.r99[-1] * np.ones(h.K) if self.cb_skala.currentIndex() == 1 else h.r99
        s = 0.55 * self.es.R / np.maximum(np.asarray(r, np.float64), 1e-9)
        for e in (self.eog, self.es_og):
            e.ustaw_dane(d, s); e.ustaw_kolory(h.kol)
        self.eog.ustaw_reflektory(h.rfix)
        Lb = np.maximum(1.5 * np.linalg.norm(d['skok'], axis=1), 0.5 * d['L']) * s
        self._og_dist_zbl = 4.0 * float(np.median(Lb)) * self.sp_skala_chm.value()
        if kamera and not self.ch_og_refl.isChecked():                   # zbliżenie: kadr na wiązki i skok cząsteczki
            self.vog.opts['distance'] = self._og_dist_zbl
            if int(d['G']) > 1 and self.cb_og_sledz.currentIndex() == 0:
                self.cb_og_sledz.setCurrentIndex(2)                          # pole tokenu: koloid na środku kadru
        # kadr o stałej skali: jedna odległość na całe uczenie, z 90. percentyla promienia kuli komórki
        # (wokół koloidu: L; wokół pułapki: L + odległość pułapka–koloid), w jednostkach ekranu
        Ls = np.asarray(d['L'], np.float64) * s
        off = np.linalg.norm(np.asarray(d['ogn'], np.float64) - np.asarray(d['srodek'], np.float64), axis=1) * s
        self._og_R_stale = (float(np.percentile(Ls + off, 90)), float(np.percentile(Ls, 90)))
        zb = np.asarray(d['zb'])
        rk = np.asarray(d['r_kom']) / np.maximum(np.asarray(d['r99']), 1e-12)
        self.lb_og.setText(f'Pułapka cząsteczki {klucz[1]}: komórka tokenu (połowa odległości do sąsiada) {pl(float(np.median(rk)), 3)} r99 '
                           f'(mediana; {pl(float(rk.min()), 3)}–{pl(float(rk.max()), 3)}); zero pola w komórce w {zb.mean():.0%} checkpointów '
                           f'({int(zb.sum())} z {len(zb)}). Zgodność z siecią: jasności {float(np.max(d["blad_z"])):.1e}, '
                           f'położenia {float(np.max(d["blad_pos"])):.1e}.')
        self._og_opcje()

    def _og_kamera_refl(self, on):
        """Reflektory w okienku 11: kamera jak na sferze reflektorów (odległość, pole widzenia, kąty, środek), bez śledzenia;
        po wyłączeniu powrót do zbliżenia i śledzenia koloidu."""
        from PyQt6.QtGui import QVector3D
        if on:
            self._og_sledz_przed = (self.cb_og_sledz.currentIndex(), self.cb_og_skala.currentIndex())
            self.cb_og_sledz.setCurrentIndex(0); self.cb_og_skala.setCurrentIndex(0)
            for k in ('fov', 'elevation', 'azimuth'):
                self.vog.opts[k] = self.vs.opts[k]
            # pole widzenia pyqtgraph jest poziome: ta sama liczba pikseli na jednostkę sceny przy innej szerokości okienka
            self.vog.opts['distance'] = self.vs.opts['distance'] * max(self.vog.width(), 1) / max(self.vs.width(), 1)
            c = self.vs.opts['center']; self.vog.opts['center'] = QVector3D(c.x(), c.y(), c.z())
        else:
            self.vog.opts['distance'] = getattr(self, '_og_dist_zbl', 30.0)
            sl_, sk_ = getattr(self, '_og_sledz_przed', (2, 1))
            self.cb_og_sledz.setCurrentIndex(sl_); self.cb_og_skala.setCurrentIndex(sk_)
        self.vog.update()

    def _og_opcje(self, *_):
        sk = self.sp_skala_chm.value()
        wspolne = dict(mgla=self.ch_og_mgla.isChecked(), jas=self.sp_og_jas.value(), ostr=self.sp_og_ostr.value(),
                       tlo=self.sp_og_tlo.value(), rozm=self.sp_og_rozm.value(), strzalki=self.cb_og_strz.currentIndex(),
                       dl_strz=self.sp_og_strz.value(), pods=self.sp_og_pods.value(), osie=self.ch_og_osie.isChecked(),
                       skok=self.ch_og_skok.isChecked(), skala_chmury=sk)
        self.eog.opcje.update(wspolne, slad=self.cb_og_slad.currentIndex(), n_slad=self.sp_og_slad.value(),
                              wsp=self.cb_og_uklad.currentIndex(), pudlo=self.ch_og_mgla.isChecked() or self.cb_og_strz.currentIndex() > 0,
                              wiazki=self.ch_og_wiazki.isChecked(), jasnosc=self.cb_og_jasn.currentIndex(), gain_w=self.sp_og_gw.value(),
                              dl_wiazki=self.sp_og_dlw.value(), wklady=self.cb_og_wk.currentIndex(), dl_wkladow=self.sp_og_dlwk.value(),
                              swiatlo=self.ch_og_sw.isChecked(), jas_sw=self.sp_og_jsw.value(), szer_sw=self.sp_og_szer.value(),
                              rozb_sw=self.sp_og_rozb.value(), linie_wiazek=self.ch_og_linie.isChecked(), osie_ogniska=self.ch_og_osieo.isChecked(),
                              refl=self.ch_og_refl.isChecked(), jas_refl=self.sp_og_jrefl.value(), R=float(self.es.R),
                              sfera=float(self.es.opcje.get('sfera', 0.10)), obrys=self.ch_og_obrys.isChecked(),
                              poziom_obr=self.sp_og_poz.value(), jas_obr=self.sp_og_jobr.value(), pow_obr=self.sp_og_pobr.value(),
                              przytnij=self.cb_og_skala.currentIndex() > 0, pulapka_zast=self.cb_og_sledz.currentIndex() == 1,
                              ruch_nast=self.ch_og_ruch.isChecked(), plyn=self.ch_og_plyn.isChecked(), tlo_czastki=self.ch_og_tlo.isChecked())
        for k_ in ('mgla', 'jas', 'ostr', 'tlo', 'rozm', 'strzalki', 'dl_strz', 'pods', 'plyn', 'skala_chmury'):
            self.eog_pp.opcje[k_] = self.eog.opcje[k_]                   # pole podkroków: te same opcje mgły i strzałek
        anim = (self.ch_og.isChecked() and self.ch_og_plyn.isChecked() and self.cb_og_strz.currentIndex() > 0
                and ((self._og is not None and int(self._og['G']) > 1) or self._og_tryb_pod()))
        if anim and not self._og_anim.isActive():
            self._og_anim.start()
        elif not anim and self._og_anim.isActive():
            self._og_anim.stop()
        self.es_og.opcje.update(wspolne, slad=0, wsp=0, pudlo=False, wiazki=False, swiatlo=False, obrys=False, wklady=0)  # na sferze bez ogniska
        if self.h is not None:
            self._og_czas(self.t)

    @staticmethod
    def _og_opis_obrysu(iw):
        ob = (iw or {}).get('obrys')
        if not ob:
            return ''
        a = ob['osie']
        return (f';  obrys ogniska ({pl(ob["poziom"])} szczytu, lupa ×{ob["pow"]:.0f}): osie {pl(a[0])}/{pl(a[1])}/{pl(a[2])} w0, '
                f'wydłużenie ×{pl(a[0] / max(a[2], 1e-12))}')

    def _og_anim_tik(self):
        """Płynące strzałki pola: faza rośnie z czasem rzeczywistym; przerysowanie tylko okienka 11, gdy jest widoczne."""
        import time as _t
        if not self.vog.isVisible() or self.vog.visibleRegion().isEmpty():
            return
        self.eog.opcje['faza'] = self.eog_pp.opcje['faza'] = (_t.perf_counter() * self.sp_og_tempo.value()) % 1000.0
        self.vog.update()

    def _og_czas_zmiana(self, i):
        """Podkroki: reflektory w skali sfery włączane same (ta sama skala w uwadze i MLP, reflektory zawsze widoczne);
        powrót do osi uczenia wyłącza je, jeśli włączyło je to przełączenie."""
        if i == 1 and not self.ch_og_refl.isChecked():
            self._og_refl_auto = True; self.ch_og_refl.setChecked(True)
        elif i == 0 and getattr(self, '_og_refl_auto', False):
            self._og_refl_auto = False; self.ch_og_refl.setChecked(False)

    def _og_tytul(self, tekst):
        """Tytuł okienka 11 zależny od trybu (oś uczenia: blok 6; podkroki: bieżący blok i podkrok); zmiana tylko gdy inny."""
        dk = self.doki.get('11')
        if dk is not None and dk.windowTitle() != tekst:
            dk.setWindowTitle(tekst)

    def _og_tryb_pod(self):
        return hasattr(self, 'cb_og_czas') and self.cb_og_czas.currentIndex() == 1 and self.ch_og.isChecked()

    def _og_podkrok(self, s):
        """Okienko 11 w podkrokach: ognisko wybranej cząsteczki w podkroku s (dane przejścia przez sieć, układ '_n', skala sfery).
        MLP bloku b: 768 wiązek bloku, wkład c = (z_jc − z̄_c)·P v_c. Uwaga bloku b: 600 wkładów od innych cząsteczek,
        Σ_h (A^h_jk − ā^h_k)·u^h_k. W obu: suma wkładów porównana z prawdziwym ruchem stanu j w tym podkroku."""
        d = self.d_pod
        self.eog.ustaw_tryb_pod(True)
        if d is None or self.h is None or 'pos_n' not in d:
            self.eog._ognisko_czysc()
            self.lb11.setText('Podkroki: liczę przejście przez sieć dla tego checkpointu…'); self._zlec_podkroki(); return
        from wspolne import kolory_kanalow
        j = int(self.sp_tok.value()); pos = d['pos_n']
        sc = 0.55 * self.es.R / max(float(np.percentile(np.linalg.norm(pos, axis=2), 99)), 1e-9)
        s = float(np.clip(s, 0, 12)); k = int(np.clip(np.floor(s), 0, 11)); f = s - k if s < 12 else 1.0
        if getattr(self, '_og_lb_pod', (None,))[0] != (id(d), j, sc):     # jedna długość wiązek na całe przejście (12 podkroków)
            ruchy = np.linalg.norm(np.diff(pos[:, j], axis=0), axis=1) * sc
            self._og_lb_pod = ((id(d), j, sc), 1.5 * float(ruchy.max()))
        Lst = self._og_lb_pod[1]
        kolej = self.ch_og_kolej.isChecked()
        ff, fc = (min(2 * f, 1.0), max(2 * f - 1, 0.0)) if kolej else (1.0, f)
        p0, p1 = pos[k, j] * sc, pos[k + 1, j] * sc; ruch = p1 - p0
        c = p0 + fc * ruch; b = k // 2
        if self.ch_og_tlo.isChecked():                                    # pozostałe cząsteczki w tle, w tej samej chwili podkroku
            kc = getattr(self, '_kol_cz', None)
            self.eog.ustaw_tlo(((1 - fc) * pos[k] + fc * pos[k + 1]) * sc, None if kc is None else kc[:, :3],
                               self.sp_og_jtlo.value(), bez=j)
        from lasery_gpu import przenikanie, reflektory_przejscia
        fp = self._plynnie()
        kier_r, kol_r_, kl = reflektory_przejscia(d, s, fp)               # reflektory na sferze (jak sfera w podkrokach)
        if getattr(self, '_og_rf_pod', (None, None))[0] != kl:
            self._og_rf_pod = (kl, 50.0 * kier_r)
        kol_r = 0.25 + 0.75 * kol_r_
        wm, bm, ba = przenikanie(s, fp)
        if 1e-3 < wm < 0.999:                                             # płynne przejście: wiązki MLP i wkłady uwagi naraz
            self._og_przenikanie(d, j, sc, c, wm, bm, ba, kol_r, Lst)
            co = (f'przejście: blok {bm + 1} MLP {wm:.0%} ↔ blok {ba + 1} uwaga {1 - wm:.0%} '
                  '(wiązki jednego podkroku gasną, następnego się rozjaśniają)')
        elif k % 2 == 1 or wm >= 0.999:                                   # MLP bloku b: 768 wiązek
            b = bm
            pv = d['pv_n'][b] * sc; zc = d['z_c'][b, j].astype(np.float64)
            kol = kol_r
            wk = zc[:, None] * pv
            if self.cb_og_jasn.currentIndex() == 0:                         # wiązka wchodzi z kierunku, w który pcha
                kier, jas = wk, np.linalg.norm(wk, axis=1)
            else:                                                           # jak na sferze: kierunek v_c, błękit = ujemna dioda
                zz = d['z'][b, j].astype(np.float64); kier, jas = pv, np.abs(zz) * np.linalg.norm(pv, axis=1)
                kol = np.where((zz < 0)[:, None], np.array([0.30, 0.85, 1.0]), kol)
            self.eog.ustaw_ognisko(c, kier, wk, jas, kol, ruch=ruch, zrodla=self._og_rf_pod[1], Lb_staly=Lst,
                                   zapalone=float((d['z'][b, j] > 0.05).mean()), ujemne=float((d['z'][b, j] < -0.05).mean()))
            co = f'blok {b + 1}, MLP: 768 wiązek bloku zbiega się w cząsteczce'
        else:                                                             # uwaga bloku b: 600 wkładów od innych cząsteczek
            b = ba
            wk = np.einsum('hk,hkd->kd', d['ah_c'][b, :, j, :].astype(np.float64), d['uh_n'][b].astype(np.float64)) * sc
            kol = getattr(self, '_kol_cz', None)
            kol = kol[:, :3] if kol is not None else np.tile([0.35, 0.8, 1.0], (wk.shape[0], 1))
            self.eog.ustaw_ognisko(c, wk, wk, np.linalg.norm(wk, axis=1), kol, ruch=ruch, zrodla=pos[k] * sc, prosto=True,
                                   Lb_staly=Lst, refl_tlo=(self._og_rf_pod[1], kol_r))
            co = f'blok {b + 1}, uwaga: 600 wkładów od innych cząsteczek (tu przesuwają ją cząsteczki, nie lasery)'
        co += self._og_pole_podkroku(d, j, sc, s, fp, Lst)
        b = k // 2
        self._og_tytul(f'11. Pułapka optyczna: koloid w polu wiązek, podkroki — blok {b + 1}, {"MLP (768 laserów)" if k % 2 else "uwaga (600 cząsteczek)"}')
        if self.cb_og_sledz.currentIndex():
            from PyQt6.QtGui import QVector3D
            q = c * self.sp_skala_chm.value()
            self.vog.opts['center'] = QVector3D(float(q[0]), float(q[1]), float(q[2]))
            if self.cb_og_skala.currentIndex() and not self.ch_og_refl.isChecked():   # stały kadr: kula o promieniu długości wiązek
                R = Lst * self.sp_og_dlw.value() * self.sp_skala_chm.value() * 1.1
                w_, h_ = max(self.vog.width(), 1), max(self.vog.height(), 1)
                self.vog.opts['distance'] = R / (np.tan(np.radians(self.vog.opts['fov']) / 2) * min(1.0, h_ / w_))
            self.vog.update()
        iw = self.eog.info_wiazek or {}
        lam = iw.get('ksztalt', np.full(3, np.nan))
        blad = iw.get('blad')
        self.lb11.setText(f'krok {int(d["krok"])}, podkrok {pl(s, 2)}/12, cząsteczka {j}: {co};  suma wkładów = ruch w tym podkroku '
                          f'(błąd względny {blad:.1e});  skok = {iw.get("wypadkowa", 0) / max(iw.get("suma", 1), 1e-12):.0%} sumy długości wkładów;  '
                          f'kształt ogniska: rozkład kierunków {pl(lam[0])}/{pl(lam[1])}/{pl(lam[2])} (izotropowo 0,33), '
                          f'wydłużenie ×{pl(iw.get("wydluzenie", 1.0))}{self._og_opis_obrysu(iw)}' if blad is not None
                          else f'podkrok {pl(s, 2)}: brak danych')

    def _pomiary_start(self):
        if self.h is None:
            self.lb_pom.setText('Najpierw wczytaj historię uczenia.'); return
        w = getattr(self, '_pom_watek', None)
        if w is not None and w.isRunning():
            return
        w = PomiaryPulapkiWatek(self.h, self.sp_tok.value(), self.sp_pom_co.value(), self.sp_pom_G.value())
        w.postep.connect(lambda i, n, t: (self.pasek_pom.setMaximum(n), self.pasek_pom.setValue(i), self.lb_pom.setText(t)))
        w.koniec.connect(lambda t: (self.lb_pom.setText(t), self.bt_pom.setEnabled(True), self.bt_pom_stop.setEnabled(False)))
        self._pom_watek = w; self.bt_pom.setEnabled(False); self.bt_pom_stop.setEnabled(True)
        self.lb_pom.setText(f'Liczę pułapkę cząsteczki {self.sp_tok.value()} w podkrokach…'); w.start()

    def _pomiary_stop(self):
        w = getattr(self, '_pom_watek', None)
        if w is not None:
            w.stop = True

    def _og_pole_podkroku(self, d, j, sc, s, fp, Lst):
        """Pole sił w kuli komórki cząsteczki j w podkrokach (mgła i strzałki jak w osi uczenia): zleca policzenie dla tego
        checkpointu (pole_podkrokow, w tle, ~0,3 s), a gotowe pokazuje w chwili s; w płynnym przejściu pole bieżącego i
        następnego podkroku przenikają się razem z wiązkami. Lupa (domyślnie): kula komórki pokazana w promieniu ułamka długości
        wiązek Lst (jedna skala ekranu dla kuli w każdym podkroku). Zwraca dopisek do opisu okienka."""
        if not (self.cb_og_strz.currentIndex() > 0 or self.ch_og_mgla.isChecked()):
            self.eog_pp.setVisible(False); return ''
        if getattr(self, '_baza_pod', None) is None:
            self.eog_pp.setVisible(False); return ''
        klucz = (str(self.h.wagi), int(d['krok']), j, int(self.sp_og_G.value()), round(float(self.sp_og_zas.value()), 2))
        if klucz != self._pp_klucz:
            self._pp_klucz = klucz
            if klucz in self._pp_cache:
                self._pp_pole = self._pp_cache[klucz]
            else:
                self._pp_pole = None
                dd = dict(d); dd['baza'] = self._baza_pod
                kc = getattr(self, '_kol_cz', None)
                self.watek_pp.zlec(klucz, self.h.wagi, int(d['krok']), self.h.rgb, j, dd, G=klucz[3], zasieg=klucz[4],
                                   kol_cz=None if kc is None else kc[:, :3])
        if self._pp_pole is None:
            self.eog_pp.setVisible(False)
            return '  [pole w komórce: liczę…]'
        w = self._pp_pole
        lupa = self.cb_og_lupa.currentIndex() == 0
        pow_ = (self.sp_og_lupa.value() * Lst / np.maximum(w['L'] * sc, 1e-30)) if lupa else np.ones(12)
        kl = (id(w), lupa, round(self.sp_og_lupa.value(), 3), round(float(Lst), 9), round(float(sc), 9))
        if kl != getattr(self, '_pp_pokaz', None):                       # kula pokazana w promieniu L · powiększenie (wokół x_k)
            self.eog_pp.ustaw_dane(dict(w, L=(w['L'] * pow_).astype(np.float32)), np.full(12, sc, np.float32))
            self._pp_pokaz = kl
        n = int(round(s)); t = float(min(np.floor(s), 11))              # pole podkroku k (stan przed podkrokiem), środek w x_k
        if fp > 0 and 1 <= n <= 11 and abs(s - n) < fp / 2:              # przenikanie pól na granicy podkroków (jak wiązki)
            u = (s - n + fp / 2) / fp; t = (n - 1) + u * u * (3 - 2 * u)
        self.eog_pp.ustaw_chwile(t, False); self.eog_pp.setVisible(True)
        k = int(t)
        return (f';  pole w komórce (promień {pl(float(w["L"][k]) / max(float(np.linalg.norm(w["skok"][k])), 1e-30))} skoku'
                + (f', pokazane w lupie ×{pow_[k]:.0f}' if lupa else '') +
                '): strzałki = dokąd wiązki pchnęłyby cząsteczkę postawioną w tym miejscu')

    def _pole_pod_gotowe(self, klucz, w):
        if 'blad_watku' in w:
            self.lb11.setText(f'Pole w podkrokach: błąd {w["blad_watku"]}'); return
        self._pp_cache[klucz] = w
        while len(self._pp_cache) > 8:
            self._pp_cache.pop(next(iter(self._pp_cache)))
        if klucz == self._pp_klucz and self.d_pod is not None:
            self._pp_pole = w
            self._og_opcje()

    def _og_przenikanie(self, d, j, sc, c, wm, bm, ba, kol_r, Lst):
        """Okienko 11, płynne przejście podkroków: 768 wiązek MLP bloku bm (jasność × wm) i 600 wkładów uwagi bloku ba
        (× 1 − wm) naraz; wkłady i skok ważone tak samo, więc suma wkładów = ważony ruch obu podkroków."""
        pos = d['pos_n']
        pv = d['pv_n'][bm] * sc; zc = d['z_c'][bm, j].astype(np.float64); wkM = zc[:, None] * pv
        if self.cb_og_jasn.currentIndex() == 0:
            kierM, jasM, kolM = wkM, np.linalg.norm(wkM, axis=1), kol_r
        else:
            zz = d['z'][bm, j].astype(np.float64); kierM, jasM = pv, np.abs(zz) * np.linalg.norm(pv, axis=1)
            kolM = np.where((zz < 0)[:, None], np.array([0.30, 0.85, 1.0]), kol_r)
        wkA = np.einsum('hk,hkd->kd', d['ah_c'][ba, :, j, :].astype(np.float64), d['uh_n'][ba].astype(np.float64)) * sc
        kolA = getattr(self, '_kol_cz', None)
        kolA = kolA[:, :3] if kolA is not None else np.tile([0.35, 0.8, 1.0], (wkA.shape[0], 1))
        jasA = np.linalg.norm(wkA, axis=1)
        ruchM = (pos[2 * bm + 2, j] - pos[2 * bm + 1, j]) * sc; ruchA = (pos[2 * ba + 1, j] - pos[2 * ba, j]) * sc
        nM, nA = len(wkM), len(wkA)
        self.eog.ustaw_ognisko(
            c, np.concatenate([kierM, wkA]), np.concatenate([wm * wkM, (1 - wm) * wkA]),
            np.concatenate([jasM / max(float(jasM.max()), 1e-12) * wm, jasA / max(float(jasA.max()), 1e-12) * (1 - wm)]),
            np.concatenate([np.asarray(kolM, np.float64)[:, :3], np.asarray(kolA, np.float64)]),
            ruch=wm * ruchM + (1 - wm) * ruchA, zrodla=np.concatenate([self._og_rf_pod[1], pos[2 * ba] * sc]),
            prosto=np.r_[np.zeros(nM, bool), np.ones(nA, bool)], Lb_staly=Lst, jas_wzgl=True)

    def _og_czas(self, t):
        tryb_pod = self.ch_pod.isChecked() and getattr(self, '_pp', None) is not None
        self.eog.setVisible(self.ch_og.isChecked())
        if not self.ch_og.isChecked():                                   # podgląd wyłączony: nic nie wczytujemy ani nie rysujemy
            self.es_og.setVisible(False)
            self.lb11.setText('Podgląd ogniskowej wyłączony. Zaznacz „pokaż podgląd ogniskowej” albo kliknij „Wczytaj / policz” '
                              '(sekcja „Pułapka optyczna”).')
            return
        if not self.widac('11') and not (self.ch_og_sfera.isChecked() and self.widac('s')):
            return                                                        # pułapki nigdzie nie widać: nic nie liczymy (_odslonieto)
        if self._og_tryb_pod():
            self.es_og.setVisible(False)
            self._og_podkrok(self.s_pod)
            return
        self.eog_pp.setVisible(False)
        self.eog.ustaw_tryb_pod(False)
        self._og_tytul('11. Pułapka optyczna: koloid w polu wiązek, oś uczenia — blok 6 (podkroki: „Czas okienka 11”)')
        d = self._og
        self.es_og.setVisible(bool(self.ch_og_sfera.isChecked() and d is not None and not tryb_pod))
        if d is None:
            self.lb11.setText('Pułapka optyczna: włącz „licz pułapkę wybranej cząsteczki” (sekcja „Pułapka optyczna”).')
            return
        self.eog.ustaw_chwile(t, self.ch_og_kolej.isChecked())
        if self.ch_og_tlo.isChecked():                                    # pozostałe cząsteczki w tle (jak koloid: ułamek fc)
            e_, h_ = self.eog, self.h
            P = ((1 - e_.fc) * h_.pos[e_.ka] * float(e_.skala_k[e_.ka]) + e_.fc * h_.pos[e_.kb] * float(e_.skala_k[e_.kb]))
            kc = getattr(self, '_kol_cz', None)
            self.eog.ustaw_tlo(P, None if kc is None else kc[:, :3], self.sp_og_jtlo.value(), bez=int(self.sp_tok.value()))
        else:
            self.eog.ustaw_tlo(None)
        if self.es_og.visible():
            self.es_og.ustaw_chwile(t, False)
        sl = self.cb_og_sledz.currentIndex()
        if sl:
            from PyQt6.QtGui import QVector3D
            sk = self.sp_skala_chm.value()
            p = (self.eog.pulapka, self.eog.koloid)[sl - 1]() * sk
            self.vog.opts['center'] = QVector3D(float(p[0]), float(p[1]), float(p[2]))
            skala = self.cb_og_skala.currentIndex()
            if skala and int(d['G']) > 1:                                   # kula opisana na sześcianie (wokół środka kadru) w okienku
                if skala == 1:                                               # sześcian zawsze tej samej wielkości
                    R = self.eog.pol_boku() + (np.linalg.norm(self.eog.pulapka() - self.eog.koloid()) if sl == 1 else 0.0)
                else:                                                        # jedna odległość kamery na całe uczenie
                    R = self._og_R_stale[0 if sl == 1 else 1]
                R *= sk * 1.05
                w_, h_ = max(self.vog.width(), 1), max(self.vog.height(), 1)
                self.vog.opts['distance'] = R / (np.tan(np.radians(self.vog.opts['fov']) / 2) * min(1.0, h_ / w_))
            self.vog.update()
        k = self.eog.k_teraz(); r99 = float(d['r99'][k]); zb = bool(d['zb'][k])
        top, topu = (d['top_ogn'][k], d['top_ogn_u'][k]) if zb else (d['top_kol'][k], d['top_kol_u'][k])
        wiazki = ', '.join(f'#{int(c)} {u:.1%}'.replace('.', ',') for c, u in zip(top[:3], topu[:3]))
        skok = float(np.linalg.norm(d['skok'][k])) / r99
        rk = float(d['r_kom'][k])
        if zb:
            odl = float(np.linalg.norm(d['ogn'][k] - d['srodek'][k])) / rk
            o_p = f'pułapka w komórce, {pl(odl)} r_kom od koloidu: {OGN.opis_lambda(d["lam"][k])}'
        else:
            lin = float(np.linalg.norm(d['ogn_lin'][k] - d['srodek'][k])) / rk
            o_p = (f'brak zera pola w komórce tokenu (oszacowanie liniowe {pl(lin, 1)} r_kom, szara strzałka do brzegu); '
                   f'„pułapka” = najsłabsze pole w komórce')
        o_p += f';  komórka r_kom = {pl(rk / r99, 3)} r99 (najbliższy sąsiad: token {int(d["sasiad"][k])})'
        iw = getattr(self.eog, 'info_wiazek', None) if self.ch_og_wiazki.isChecked() else None
        o_w = ''
        if iw:
            top_w = ', '.join(f'#{int(c)} {u:.1%}'.replace('.', ',') for c, u in zip(iw['top'], iw['udzial']))
            o_w = (f'768 wiązek: zapalonych {iw["zapalone"]:.0%}, ujemnych {iw["ujemne"]:.0%}; skok = {iw["wypadkowa"] / max(iw["suma"], 1e-12):.0%} '
                   f'sumy długości wkładów (reszta się znosi); najsilniejsze {top_w}{self._og_opis_obrysu(iw)};  ')
        ir = getattr(self.eog, 'info_ruch', None)
        if ir and self.ch_og_ruch.isChecked():
            kat = float(np.degrees(np.arccos(np.clip(ir['cos'], -1, 1))))
            o_w += (f'nadchodzący krok (zielony): ruch {pl(ir["dl"] / r99)} r99, kąt do skoku {kat:.0f}°, '
                    f'{pl(ir["stosunek"])} × długość skoku;  ')
        self.lb11.setText(f'krok {int(d["kroki"][k])}, cząsteczka {int(d["tok"])}:  {o_w}skok {pl(skok)} r99 (pomarańczowy);  '
                          f'kształt pola w koloidzie: {OGN.opis_lambda(d["lam0"][k])};  {o_p};  wiązki {"w pułapce" if zb else "w koloidzie"}: {wiazki}')

    # ------------------------------------------------------------------ dolina pewności i IoU (okienko 12)
    def _dol_klucz_teraz(self):
        h = self.h
        return (str(h.katalog), int(np.clip(round(self.t), 0, h.K - 1)), int(self.sp_tok.value()), self.cb_dol_baza.currentIndex(),
                int(self.sp_dol_G.value()), round(float(self.sp_dol_zas.value()), 3))

    def _dol_potrzebna(self):
        return self.ch_dol.isChecked() and (self.widac('12') or (self.ch_dol_sfera.isChecked() and self.widac('s')))

    def _dol_zlec(self):
        """Zleca przekrój doliny dla bieżącego checkpointu i cząsteczki (tylko przy zmianie klucza; liczy się ostatnie zlecenie)."""
        if self.h is None:
            return
        if not self._dol_potrzebna():
            self.es_dol.setVisible(False)
            return
        tryb_pod = self.ch_pod.isChecked() and getattr(self, '_pp', None) is not None
        self.es_dol.setVisible(self._dol is not None and self.ch_dol_sfera.isChecked() and not tryb_pod)
        k = self._dol_klucz_teraz()
        if k == getattr(self, '_dol_zlecony', None):
            return
        self._dol_zlecony = k
        import dolina as DL_
        self.watek_dol.zlec(k, self.h, k[1], k[2], DL_.BAZY[k[3]], k[4], k[5])

    def _dol_czas(self, tryb_pod):
        if self.h is None:
            return
        if tryb_pod:                                                 # podkroki mają inny rzut: dolina tylko w osi uczenia
            self.es_dol.setVisible(False)
            return
        self._dol_zlec()

    def _dol_gotowe(self, klucz, d):
        if 'blad_watku' in d:
            self.lb_dol.setText(f'Dolina: {d["blad_watku"]}'); return
        if self.h is None or klucz[0] != str(self.h.katalog):
            return
        self._dol = d; self._dol_klucz = klucz
        self.edol.ustaw_dane(d); self.es_dol.ustaw_dane(d)
        self._dol_wyglad(); self._dol_progi(); self._dol_przeksztalcenie()
        tryb_pod = self.ch_pod.isChecked() and getattr(self, '_pp', None) is not None
        self.es_dol.setVisible(self.ch_dol.isChecked() and self.ch_dol_sfera.isChecked() and not tryb_pod)

    def _dol_progi(self):
        for e in (self.edol, self.es_dol):
            e.opcje.update(p_min=self.sp_dol_p.value(), iou_min=self.sp_dol_iou.value())
        d = self._dol
        if d is None:
            return
        import dolina as DL_
        tr = DL_.obrys(d, self.sp_dol_p.value(), self.sp_dol_iou.value())
        self.edol.ustaw_obrys(tr); self.es_dol.ustaw_obrys(tr)
        tekst = DL_.opis(d, self.sp_dol_p.value(), self.sp_dol_iou.value())
        self.lb_dol.setText(tekst)
        cien = '' if d['baza'] == 'sfera' else '  Na sferze: cień przekroju (rzut na osie sfery).'
        self.lb12.setText(tekst + cien)

    def _dol_wyglad(self):
        for e, jas in ((self.edol, 1.0), (self.es_dol, 0.6)):
            e.opcje.update(mgla=self.ch_dol_mgla.isChecked(), obrys=self.ch_dol_obrys.isChecked(), tlo=self.sp_dol_tlo.value(),
                           jas=self.sp_dol_jas.value() * jas, rozm=self.sp_dol_rozm.value() * (1.0 if e is self.edol else 0.6))
            e.update()

    def _dol_przeksztalcenie(self):
        """Okienko 12: sześcian przekroju o połowie boku 10 wokół cząsteczki. Sfera: położenie cząsteczki p_j + M3 a w skali sfery
        (ta sama skala co cząsteczki: 0,55 R / r99 i skala chmury)."""
        d, h = self._dol, self.h
        if d is None or h is None:
            return
        s = 10.0 / max(d['Z'], 1e-9)
        self.edol.ustaw_przeksztalcenie(np.diag([s, s, s, 1.0]))
        i = d['i']
        r = float(h.r99[-1] if self.cb_skala.currentIndex() == 1 else h.r99[i])
        k = 0.55 * self.es.R / max(r, 1e-9) * self.sp_skala_chm.value()
        M = np.eye(4); M[:3, :3] = k * d['M3']; M[:3, 3] = k * d['pos0']
        self.es_dol.ustaw_przeksztalcenie(M)

    # ------------------------------------------------------------------ mechanika reflektora (okienko 13)
    def _mech_zlec(self):
        """Zleca mechanikę reflektora dla bieżącego checkpointu (tylko gdy okienko 13 widać i klucz się zmienił)."""
        h = self.h
        if h is None or not self.ch_mech.isChecked() or not self.widac('13'):
            return
        i = int(np.clip(np.floor(self.t), 0, h.K - 1))
        zr = ('sonda', 'zbior')[self.cb_mech_zr.currentIndex()]
        k = (str(h.katalog), i, int(self.sp_refl.value()), zr, int(self.sp_tok.value()))
        if k == getattr(self, '_mech_zlecony', None):
            return
        self._mech_zlecony = k
        lic = getattr(self.watek_mech, '_liczydlo', None)
        if zr == 'zbior' and (lic is None or lic._zbior is None):
            self.lb_mech.setText('Wczytuję 64 klatki zbioru uczącego z archiwów (raz)…')
        self.watek_mech.zlec(k, h, i, k[2], zr, k[4], 80)

    def _mech_gotowe(self, klucz, d):
        if 'blad_watku' in d:
            self.lb_mech.setText(f'Mechanika: {d["blad_watku"]}'); return
        if self.h is None or klucz[0] != str(self.h.katalog):
            return
        self._mech = d
        import reflektor_mechanika as RM
        tekst = RM.opis(d)
        self.lb_mech.setText(tekst); self.lb13.setText(tekst)
        self._mech_rysuj()

    def _mech_rysuj(self):
        d, h = self._mech, self.h
        if d is None or h is None:
            return
        import reflektor_rysunek as RR
        u = float(np.clip(self.t - d['i'], 0.0, 1.0)) if not d['ostatni'] else 1.0
        przes = self.sp_mech_przes.value() or float(np.clip(25.0 / max(d['igla']['kat'], d['cel']['kat'], 1e-3), 1, 500))
        S = RR.schemat(d, u, przes, n_linek=self.sp_mech_linki.value(), celownik=self.ch_mech_cel.isChecked(),
                       prog=self.ch_mech_prog.isChecked(), linki=self.sp_mech_linki.value() > 0)
        S.napis((-19.0, 0.0, 10.5), f'reflektor {d["c"]}, krok {d["krok"]}: ruch narysowany ×{przes:.0f} (kąty w opisie są prawdziwe)',
                (255, 209, 102, 255))
        self.emech.ustaw_schemat(S)
        import pyqtgraph.opengl as gl_
        nap = S.nap if self.ch_mech_nap.isChecked() else []
        while len(self._mech_napisy) < len(nap):
            it = gl_.GLTextItem(); self.vmech.addItem(it); self._mech_napisy.append(it)
        for k, it in enumerate(self._mech_napisy):
            if k < len(nap):
                p_, t_, kol = nap[k]
                it.setData(pos=tuple(float(x) for x in p_), text=t_, color=kol); it.setVisible(True)
            else:
                it.setVisible(False)

    # ------------------------------------------------------------------ czas
    def ustaw_czas(self, t):
        h = self.h
        if h is None:
            return
        self.t = t = float(np.clip(t, 0, h.K - 1))
        tryb_pod = self.ch_pod.isChecked() and getattr(self, '_pp', None) is not None
        # tylko okienka, które naprawdę widać (okienko w nieaktywnej karcie albo zamknięte nie liczy ani nie wysyła nic na GPU;
        # po odsłonięciu dostaje bieżącą chwilę w _odslonieto)
        para = (('4', self.e4), ('5', self.e5)) if tryb_pod else (('1', self.e1), ('2', self.e2), ('3', self.e3), ('4', self.e4),
                                                                  ('5', self.e5), ('s', self.es))
        for k, e in para:
            if self.widac(k):
                e.ustaw_czas(t)
        if self.es.opcje['wiazki'] and not tryb_pod and self.widac('s'):
            self.es.ustaw_natezenia(self._h_sfery(), t)
        if self.e5.kafelki and self.e5.kafelki[0]['mapa'] == 9 and self.widac('5'):
            self.e5.ustaw_reflektor(h, t, self._wybrane_refl())
        self._blokada = True
        self.suwak.setValue(int(round(t * 100)))
        for k, p in (('6', self.p6), ('7', self.p7), ('8', self.p8), ('9', self.p9), ('10', self.p10)):
            if self.widac(k):
                p.kursor.setValue(t)
        self._blokada = False
        krok = h.krok(t)
        if not self.bt_play.isChecked():
            self.krok_odtw = krok
        iou_ = h.interp(np.nan_to_num(h.iou), t); pew = h.interp(h.pew, t)
        self.lb_czas.setText(f'krok {krok:,.0f}   epoka {pl(krok / EP)}   IoU {pl(iou_)}   pewność {pl(pew)}'.replace(',', ' ', 1)
                             if krok >= 1000 else f'krok {krok:.0f}   epoka {pl(krok / EP)}   IoU {pl(iou_)}   pewność {pl(pew)}')
        par = h.interp(h.par, t)
        self.lb1.setText(f'przesunięcie {pl(float(np.hypot(*par[:2])))} kom.   obrót {par[2]:+.1f}°   skala ×{pl(par[3], 3)}')
        self.lb2.setText(f'średnie przesunięcie centrów {pl(float(h.interp(h.lok_m, t)))} kom.;  narzędzie/reszta ×{pl(float(h.interp(h.narz_lok, t)))}')
        self.lb3.setText(f'średnie drgania {pl(float(h.interp(h.vib_m, t)))} kom.   (centrum = średnia ±{int(h.okno)} kroków)')
        self._zlec_podkroki()
        obr = ''
        if self.cb_obrot.currentIndex() and getattr(h, 'obr_dryf', None) is not None:
            i_ = int(round(t)); c_ = self.sp_refl.value()
            obr = (f';  obrót igieł (mediana): drganie {pl(float(np.median(h.obr_drg[i_, :, 3])), 1)}, dryf {pl(float(np.median(h.obr_dryf[i_, :, 3])), 2)} '
                   f'°/100 kroków, od startu {pl(float(h.obr_kat0[i_]), 0)}°;  reflektor {c_}: dryf {pl(float(h.obr_dryf[i_, c_, 3]), 2)}')
        self.lbs.setText(f'pary przeciwnych reflektorów {h.interp(h.pary, t):.0%};  wiązki: {self.cb_wiazki.currentText()} '
                         f'(kolor = laser, błękit = ujemna jasność z < 0)' + obr)
        self.lb4.setText('bloki 1 2 3 (góra), 4 5 6 (dół); ' + ('pierwiastek z mapy, ' if self.ch_pierw.isChecked() else '') + 'normowana w każdym bloku;  biała = prawdziwa ramka, błękitna = ramka modelu')
        self.lb5.setText(f'{self.cb_mapa.currentText()}')
        self._og_czas(t)
        self._dol_czas(tryb_pod)
        if not tryb_pod and getattr(self, '_mech', None) is not None and self.widac('13'):
            self._mech_zlec(); self._mech_rysuj()                   # ruch w obrębie checkpointu: u = t − i
        self.rzuty.ustaw_czas(t)                                     # okienka R1–R8 (tylko widoczne)
        if tryb_pod:
            return
        if self.ch_kamera.isChecked() and hasattr(self, '_cz'):
            cz = h.interp(self._cz, t)
            self.v3.kadruj(cz[0], cz[1], self.sp_zbl.value(), self.sp_zbl.value() * 2 / 3)
        zb = self.sp_zbl.value()
        if hasattr(self, '_cz'):
            cz = h.interp(self._cz, t)
            self.e2.opcje['box'] = (cz[0] - zb / 2, cz[1] - zb / 3, cz[0] + zb / 2, cz[1] + zb / 3) if self.ch_kamera.isChecked() else None
        L = int(np.floor(t)) - int(np.floor(h.indeks(max(krok - self.sp_slad.value(), float(h.kroki[0])))))
        for e, v in ((self.e1, self.v1), (self.e2, self.v2), (self.e3, self.v3)):
            e.opcje['pxc'] = v.px_na_komorke(); e.opcje['L'] = int(np.clip(L, 0, 400)) if self.sp_slad.value() > 0 else 0

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
                              vib=np.zeros((2, LD.TOK), np.float32), LOK99=nowe.LOK99, VIB97=nowe.VIB97, r2=nowe.r2, cz=st(stare.cz[12], nowe.cz[0]))
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
            i, tok = divmod(j, LD.TOK)
            if i != self._zestaw_idx():
                self.cb_zestaw_akt.setCurrentIndex(i)                    # wczytuje historię tego zdjęcia
            self.sp_tok.setValue(tok)
