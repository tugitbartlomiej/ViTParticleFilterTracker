"""Wątki okienek rzutów: stany 192D i bazy (ProjectionWorker), miary w czasie (MetricsWorker), rama Schura (FrameCWorker)."""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread

from laser_app.compute import projections


class ProjectionWorker(QThread):
    postep = pyqtSignal(int, int, str)
    gotowe = pyqtSignal(object, object)
    blad = pyqtSignal(str)

    def __init__(self, h):
        super().__init__()
        self.h = h
        self.stop = False

    def run(self):
        try:
            d = projections.policz(self.h, postep=lambda i, n, o: self.postep.emit(i, n, o), przerwij=lambda: self.stop)
            self.gotowe.emit(str(self.h.katalog), d)
        except projections.Interrupted:
            self.blad.emit('przerwano')
        except Exception as e:                                           # noqa: BLE001
            traceback.print_exc()
            self.blad.emit(f'{type(e).__name__}: {e}')


class MetricsWorker(QThread):
    """Miary separacji w czasie uczenia (projections.miary_w_czasie_gpu) w tle; wynik zapisywany obok stanów 192D."""
    gotowe = pyqtSignal(object, object)
    postep = pyqtSignal(int, int)

    def __init__(self, dane, rdz, mies, wyklucz, plik, klucz):
        super().__init__()
        self.a = (dane, rdz, mies, wyklucz); self.plik = plik; self.klucz = klucz

    def run(self):
        try:
            w = projections.miary_w_czasie_gpu(*self.a, postep=lambda a, b: self.postep.emit(a, b))
            if w is not None:
                np.savez(self.plik, **w)
            self.gotowe.emit(self.klucz, w)
        except Exception:                                                # noqa: BLE001
            traceback.print_exc()
            self.gotowe.emit(self.klucz, None)


class FrameCWorker(QThread):
    """Rama Schura (projections.policz_rame_C) w tle. Zlecenia nadpisują się: liczy się tylko ostatnie."""
    gotowe = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self._zlecenie = None

    def zlec(self, klucz, kw):
        self._zlecenie = (klucz, kw)
        if not self.isRunning():
            self.start()

    def run(self):
        while self._zlecenie is not None:
            klucz, kw = self._zlecenie
            self._zlecenie = None
            try:
                self.gotowe.emit(klucz, projections.policz_rame_C(**kw))
            except Exception as e:                                       # noqa: BLE001
                traceback.print_exc()
                self.gotowe.emit(klucz, dict(blad_watku=f'{type(e).__name__}: {e}'))
