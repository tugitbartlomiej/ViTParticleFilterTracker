"""Wątki zakładki: liczenie historii, zestawu zdjęć, pułapki, mapy pułapek, pola w podkrokach, doliny, mechaniki, pomiarów pułapki,
przejść przez sieć; oraz rzuty płaskie podkroków (plaski_podkrokow, statystyki_podkrokow)."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread

from laser_app.compute import focal_trap, history as hist, passage_store, trap_map


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
    par, mapy = hist.prokrustes2d(S, g)
    QC = np.stack([hist.zastosuj(S[t], mapy[t]) for t in range(len(S))]).astype(np.float32)
    GLB = np.stack([(g - mapy[t][3]) @ mapy[t][1].T / mapy[t][2] + mapy[t][0] for t in range(len(S))]).astype(np.float32)
    lok = np.linalg.norm(QC - g, axis=-1).astype(np.float32)
    vib = np.concatenate([np.zeros((1, hist.TOK)), np.linalg.norm(np.diff(QC, axis=0), axis=-1)]).astype(np.float32)
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


class HistoryWorker(QThread):
    postep = pyqtSignal(int, int, str)
    gotowe = pyqtSignal(str)
    blad = pyqtSignal(str)

    def __init__(self, wagi, kroki, rgb, gt, okno):
        super().__init__()
        self.args = (wagi, kroki, rgb, gt, okno)
        self.stop = False

    def run(self):
        try:
            out = hist.policz(*self.args, postep=lambda i, n, o: self.postep.emit(i, n, o), przerwij=lambda: self.stop)
            self.gotowe.emit(str(out))
        except hist.Interrupted:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class ImageSetWorker(QThread):
    """Historie kilku zdjęć tej samej historii wag (te same checkpointy): z cache od razu, inaczej `history.policz` po kolei."""
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
                rgb, roz = hist.wczytaj_zdjecie(zd)
                gt, _ = hist.ramka_prawdziwa(zd, roz)
                out = hist.policz(wagi, kroki, rgb, gt, okno, przerwij=lambda: self.stop,
                                postep=lambda a, b, o, i=i: self.postep.emit(a, b, f'zdjęcie {i + 1}/{len(zdjecia)}: {o}'))
                katalogi.append(str(out))
            self.gotowe.emit(katalogi)
        except hist.Interrupted:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class FocalTrapWorker(QThread):
    """Pułapka optyczna we wszystkich checkpointach historii, w tle: wybrana cząsteczka i (opcjonalnie) cząsteczki „razem”
    w jednym przejściu (focal_trap.policz_wiele: wagi i sieć wspólne). gotowe(args, {cząsteczka: wynik})."""
    postep = pyqtSignal(int, int, str)
    gotowe = pyqtSignal(object, object)
    blad = pyqtSignal(str)

    def __init__(self, h, tok, G, zasieg, razem=(), dokladnosc='float32'):
        super().__init__()
        self.h, self.args = h, (int(tok), int(G), float(zasieg))
        self.toki = list(dict.fromkeys([int(tok)] + [int(t) for t in razem]))
        self.dokladnosc = dokladnosc
        self.stop = False

    def run(self):
        try:
            W = focal_trap.policz_wiele(self.h, self.toki, *self.args[1:], postep=lambda i, n, o: self.postep.emit(i, n, o),
                                 przerwij=lambda: self.stop, dokladnosc=self.dokladnosc)
            self.gotowe.emit(self.args, W)
        except focal_trap.Interrupted:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class TrapMapWorker(QThread):
    """Mapa pułapek wszystkich 600 cząsteczek przez całą historię (trap_map.policz), w tle."""
    postep = pyqtSignal(int, int, str)
    gotowe = pyqtSignal(object)
    blad = pyqtSignal(str)

    def __init__(self, h, zasieg, dokladnosc):
        super().__init__()
        self.h, self.zasieg, self.dokladnosc = h, float(zasieg), dokladnosc
        self.stop = False

    def run(self):
        try:
            d = trap_map.policz(self.h, self.zasieg, postep=lambda i, n, o: self.postep.emit(i, n, o), przerwij=lambda: self.stop,
                          dokladnosc=self.dokladnosc)
            self.gotowe.emit(d)
        except focal_trap.Interrupted:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class SubstepFieldWorker(QThread):
    """Pole sił w kuli komórki wybranej cząsteczki w 12 podkrokach jednego checkpointu (substep_field.policz), w tle.
    Zlecenia nadpisują się: liczy się tylko ostatnie (przy przewijaniu checkpointów stare są pomijane)."""
    gotowe = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._zlecenie = None

    def zlec(self, klucz, *args, plik=None, **kw):
        """plik: pamięć dyskowa wyniku (disk_cache): gdy jest, wynik z dysku; inaczej liczenie i zapis."""
        self._zlecenie = (klucz, args, kw, plik)
        if not self.isRunning():
            self.start()

    def run(self):
        from laser_app.compute import disk_cache, substep_field
        while self._zlecenie is not None:
            klucz, args, kw, plik = self._zlecenie; self._zlecenie = None
            try:
                d = disk_cache.wczytaj(plik) if plik is not None else None
                if d is None:
                    d = substep_field.policz(*args, **kw)
                    if plik is not None:
                        disk_cache.zapisz(plik, d)
                self.gotowe.emit(klucz, d)
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.gotowe.emit(klucz, dict(blad_watku=f'{type(e).__name__}: {e}'))


class ValleyWorker(QThread):
    """Przekrój doliny pewności i IoU (valley.ValleySolver.policz) w tle. Zlecenia nadpisują się: liczy się tylko ostatnie."""
    gotowe = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._zlecenie = None
        self._liczydlo = None

    def zlec(self, klucz, h, *args, plik=None):
        self._zlecenie = (klucz, h, args, plik)
        if not self.isRunning():
            self.start()

    def run(self):
        from laser_app.compute import disk_cache, valley
        while self._zlecenie is not None:
            klucz, h, args, plik = self._zlecenie; self._zlecenie = None
            try:
                d = disk_cache.wczytaj(plik) if plik is not None else None
                if d is None:
                    if self._liczydlo is None or self._liczydlo.h is not h:
                        self._liczydlo = valley.ValleySolver(h)
                    d = self._liczydlo.policz(*args)
                    if plik is not None:
                        disk_cache.zapisz(plik, d)
                self.gotowe.emit(klucz, d)
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.gotowe.emit(klucz, dict(blad_watku=f'{type(e).__name__}: {e}'))


class MechanicsWorker(QThread):
    """Mechanika reflektora (reflector_mechanics.MechanicsSolver.policz) w tle. Zlecenia nadpisują się: liczy się tylko ostatnie."""
    gotowe = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._zlecenie = None
        self._liczydlo = None

    def zlec(self, klucz, h, *args, plik=None):
        self._zlecenie = (klucz, h, args, plik)
        if not self.isRunning():
            self.start()

    def run(self):
        from laser_app.compute import disk_cache, reflector_mechanics
        while self._zlecenie is not None:
            klucz, h, args, plik = self._zlecenie; self._zlecenie = None
            try:
                d = disk_cache.wczytaj(plik) if plik is not None else None
                if d is None:
                    if self._liczydlo is None or self._liczydlo.h is not h:
                        self._liczydlo = reflector_mechanics.MechanicsSolver(h)
                    d = self._liczydlo.policz(*args)
                    if plik is not None:
                        disk_cache.zapisz(plik, d)
                self.gotowe.emit(klucz, d)
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                self.gotowe.emit(klucz, dict(blad_watku=f'{type(e).__name__}: {e}'))


class TrapMeasurementsWorker(QThread):
    """Pomiary pułapki w podkrokach dla całej historii (trap_measurements.policz_historie), w tle; zapis npz + csv."""
    postep = pyqtSignal(int, int, str)
    koniec = pyqtSignal(str)

    def __init__(self, h, tok, co, G):
        super().__init__()
        self.h, self.tok, self.co, self.G = h, int(tok), int(co), int(G)
        self.stop = False

    def run(self):
        from laser_app.compute import trap_measurements
        try:
            p = trap_measurements.policz_historie(self.h, self.tok, self.co, self.G, postep=lambda i, n, t: self.postep.emit(i, n, t),
                                   przerwij=lambda: self.stop)
            self.koniec.emit(f'zapisano: {p} (i .csv obok)' if p else 'przerwano przed pierwszym checkpointem')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.koniec.emit(f'błąd {type(e).__name__}: {e}')


class SubstepsWorker(QThread):
    """Liczy `history.podkroki` po kolei dla zleconych checkpointów (kolejka bez powtórzeń; `pilne` wskakuje na początek)."""
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
                d = passage_store.wczytaj(kat, krok) if kat is not None else None
                if d is not None:                                    # zapisane wcześniej: bez liczenia
                    d['baza'] = passage_store.wczytaj_baze(kat) or baza; d['_z_dysku'] = True
                else:
                    with passage_store.ZAMEK_SIECI:
                        d = hist.podkroki(wagi, krok, rgb, baza=baza)
                    if kat is not None and zapis:
                        passage_store.zapisz_baze(kat, d['baza'])
                        bz = passage_store.wczytaj_baze(kat)
                        if bz is not None and np.array_equal(bz['P_n'], d['baza']['P_n']):   # tylko w rzucie z baza.npz
                            d['_zapisano'] = passage_store.zapisz(kat, d, f16)
                if g is not None:                                    # cięższe rachunki CPU też tutaj, nie w wątku okna
                    d['_pp'] = plaski_podkrokow(d, g, narz)
                    d['_stat'] = statystyki_podkrokow(d)
                self.gotowe.emit(d)
            except Exception as e:                                       # noqa: BLE001
                self.gotowe.emit({'blad': f'{type(e).__name__}: {e}'})


class AllPassagesWorker(QThread):
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
            baza = passage_store.wczytaj_baze(self.kat) or self.baza_
            nowe = pom = 0; bajty = self.juz_b; t0 = time.time()
            for i, j in enumerate(idx):
                if self.stop:
                    self.koniec.emit(f'przerwano: zapisano {nowe} nowych, {pom} było już zapisanych'); return
                krok = int(h.kroki[j])
                if passage_store.plik(self.kat, krok).exists():
                    pom += 1; continue
                if bajty > self.limit_b:
                    self.koniec.emit(f'limit miejsca ({bajty / 1e9:.1f} GB): zapisano {nowe} nowych'); return
                with passage_store.ZAMEK_SIECI:
                    d = hist.podkroki(h.wagi, krok, h.rgb, baza=baza)
                if baza is None:
                    baza = d['baza']; self.baza.emit(baza)
                passage_store.zapisz_baze(self.kat, baza)
                bajty += passage_store.zapisz(self.kat, d, self.f16); nowe += 1
                tt = (time.time() - t0) / nowe * (len(idx) - i - 1)
                self.postep.emit(i + 1, len(idx), f'krok {krok}: {i + 1}/{len(idx)}, {bajty / 1e9:.1f} GB, zostało ~{tt / 60:.0f} min')
            self.koniec.emit(f'gotowe: zapisano {nowe} nowych, {pom} było już zapisanych ({bajty / 1e9:.1f} GB)')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.koniec.emit(f'błąd {type(e).__name__}: {e}')
