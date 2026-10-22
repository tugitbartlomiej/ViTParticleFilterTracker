"""Kilka zdjęć naraz na sferze reflektorów (strona „Kilka zdjęć”, limit 1–32)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton

from laser_app.compute import history as hist
from laser_app.lasers_tab.workers import ImageSetWorker
from laser_app.ui.layout_widgets import NarrowCombo
from laser_app.ui.widgets import _spin, CollapsibleSection


class ImageSetMixin:
    """Kilka zdjęć naraz na sferze reflektorów (strona „Kilka zdjęć”, limit 1–32).
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    # ------------------------------------------------------------------ kilka zdjęć naraz (sfera reflektorów)
    def _buduj_zestaw(self):
        """Zestaw zdjęć tej samej historii wag: sfera pokazuje 600 koloidów każdego zdjęcia (600·n cząsteczek, 600·n·768 wiązek),
        reszta okienek — aktywne zdjęcie. Domyślny limit 5, najwyżej 32 (jak paczka uczenia): każde zdjęcie to osobna historia
        (pierwsze liczenie czyta wszystkie pliki wag), a sfera rysuje n razy więcej wiązek."""
        from PyQt6.QtWidgets import QListWidget
        g = CollapsibleSection('Kilka zdjęć naraz na sferze reflektorów'); f = QFormLayout(g)
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
        self.cb_zestaw_akt = NarrowCombo()
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
        w = ImageSetWorker(wagi, kroki, [z for z, _ in self._zestaw], okno)
        w.postep.connect(lambda a, b, o: (self.pasek_zestaw.setRange(0, b), self.pasek_zestaw.setValue(a), self.lb_zestaw.setText(o)))
        w.gotowe.connect(self._zestaw_gotowe)
        w.blad.connect(lambda s: self.lb_zestaw.setText(f'Zestaw nie policzony: {s}'))
        w.finished.connect(lambda: (setattr(self, '_zestaw_watek', None), self.bt_zestaw.setEnabled(True), self.bt_zestaw_stop.setEnabled(False)))
        self._zestaw_watek = w; self.bt_zestaw.setEnabled(False); self.bt_zestaw_stop.setEnabled(True)
        self.lb_zestaw.setText(f'Wczytuję / liczę {n} historii…')
        w.start()

    def _zestaw_gotowe(self, katalogi):
        try:
            hs = [hist.History(k) for k in katalogi]
            for h_ in hs:
                if (self.cb_etyk.currentData() or 'dotkniete') != 'dotkniete':
                    h_.ustaw_etykiete(self.cb_etyk.currentData(), self.sp_etyk_prog.value())
            self.hz = hist.ImageSetHistory(hs)
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
        return int(j) if i is None else i * hist.TOK + int(j)

    def _zestaw_na_sferze(self):
        """Przełącza sferę między zestawem a aktywnym zdjęciem (bez ponownego liczenia)."""
        if self.h is None:
            return
        hs = self._h_sfery()
        self.es.ustaw_historie(hs, self.cb_skala.currentIndex() == 1)
        self._skala = None; self._ks = None
        self._opcje()
