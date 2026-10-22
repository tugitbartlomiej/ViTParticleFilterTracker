"""Okno „Obliczenia: eksport i import” (menu Plik, przyciski w sekcjach pułapki i wyboru cząsteczek).

Lista obliczeń policzonych dla bieżącej historii (compute/exports.lista): historia, rzuty, przejścia, pułapki cząsteczek, mapa
pułapek, pomiary w podkrokach, pamięć pól w podkrokach / doliny / mechaniki; rozmiary i daty. Zaznaczone obliczenia idą do
jednego pliku zip (z manifestem i sumami SHA256); plik wczytany na tym albo innym komputerze trafia do tej samej historii
(albo do historii o tych samych krokach i zdjęciu) i aplikacja potem tylko je wczytuje, bez liczenia. Przebiegi wybranych
cząsteczek na sferze (położenia przez checkpointy) zapisuje „Przebiegi → CSV”."""
from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from laser_app.compute import exports


class _Zadanie(QThread):
    """Eksport albo import w tle (pliki historii mają setki MB)."""
    postep = pyqtSignal(int, int, str)
    koniec = pyqtSignal(object, str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.koniec.emit(self.fn(lambda i, n, o: self.postep.emit(i, n, o)), '')
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.koniec.emit(None, f'{type(e).__name__}: {e}')


def _mb(b):
    return f'{b / 2 ** 20:,.1f} MB'.replace(',', ' ').replace('.', ',')


class ComputationsDialog(QDialog):
    def __init__(self, zakladka, parent=None):
        super().__init__(parent)
        self.z = zakladka
        self.setWindowTitle('Obliczenia: eksport i import')
        self.resize(820, 640)
        v = QVBoxLayout(self)
        self.lb_hist = QLabel(); self.lb_hist.setWordWrap(True); self.lb_hist.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.lb_hist)
        r = QHBoxLayout()
        r.addWidget(QLabel('Cząsteczki:'))
        self.ed_toki = QLineEdit(); self.ed_toki.setPlaceholderText('np. 226, 12, 100-105')
        self.ed_toki.setToolTip('Cząsteczki do przebiegów na sferze i do „Zaznacz obliczenia cząsteczek” (domyślnie wybrana i „+ cząsteczki”).')
        r.addWidget(self.ed_toki, 1)
        b = QPushButton('Zaznacz obliczenia cząsteczek'); b.clicked.connect(self._zaznacz_czastki)
        b.setToolTip('Zaznacza na liście wszystko, co jest policzone dla tych cząsteczek (pułapki, pomiary, pamięć pól i doliny).')
        r.addWidget(b)
        v.addLayout(r)
        self.lb_stan = QLabel(); self.lb_stan.setWordWrap(True)
        v.addWidget(self.lb_stan)
        self.drzewo = QTreeWidget(); self.drzewo.setHeaderLabels(['Obliczenie', 'Cząsteczka', 'Parametry', 'Rozmiar', 'Data'])
        self.drzewo.setColumnWidth(0, 300); self.drzewo.setColumnWidth(1, 80); self.drzewo.setColumnWidth(2, 210)
        self.drzewo.setToolTip('Zaznacz obliczenia do eksportu. Grupa zaznacza wszystkie swoje pozycje.')
        v.addWidget(self.drzewo, 1)
        r = QHBoxLayout()
        self.ch_przebiegi = QCheckBox('dołącz przebiegi cząsteczek na sferze (CSV + NPZ)'); self.ch_przebiegi.setChecked(True)
        self.ch_przebiegi.setToolTip('W pliku eksportu także przebiegi cząsteczek z pola „Cząsteczki” (do analizy poza aplikacją).')
        r.addWidget(self.ch_przebiegi); r.addStretch(1)
        self.ch_nadpisz = QCheckBox('import: nadpisz istniejące pliki')
        self.ch_nadpisz.setToolTip('Bez tego pliki, które już są w historii, zostają, a import dokłada tylko brakujące.')
        r.addWidget(self.ch_nadpisz)
        v.addLayout(r)
        r = QHBoxLayout()
        for tekst, fn, opis in (('Eksportuj zaznaczone…', self._eksportuj, 'Zaznaczone obliczenia do jednego pliku zip (z sumami SHA256).'),
                                ('Wczytaj obliczenia z pliku…', self._importuj, 'Plik zip z eksportu: obliczenia trafiają do historii '
                                                                                  'i aplikacja wczytuje je zamiast liczyć.'),
                                ('Przebiegi → CSV…', self._przebiegi, 'Położenia cząsteczek z pola „Cząsteczki” na sferze przez wszystkie '
                                                                      'checkpointy: CSV (i NPZ obok).'),
                                ('Otwórz katalog historii', self._otworz, 'Katalog z plikami obliczeń tej historii.'),
                                ('Odśwież', self.odswiez, 'Lista od nowa (np. po liczeniu w tle).')):
            b = QPushButton(tekst); b.setToolTip(opis); b.clicked.connect(fn); r.addWidget(b)
        v.addLayout(r)
        self.pasek = QProgressBar(); self.pasek.setTextVisible(True); self.pasek.setVisible(False)
        v.addWidget(self.pasek)
        self._zadanie = None
        self.drzewo.itemChanged.connect(self._zmiana)
        h = getattr(zakladka, 'h', None)
        if h is not None:
            toki = [int(zakladka.sp_tok.value())] + [t for t in zakladka._wybrane_tok() if t != int(zakladka.sp_tok.value())]
            self.ed_toki.setText(', '.join(map(str, toki[:20])))
        self.odswiez()

    # ------------------------------------------------------------------ lista
    def _katalog(self):
        h = getattr(self.z, 'h', None)
        return None if h is None else Path(h.katalog)

    def _toki(self):
        from laser_app.ui.widgets import lista_liczb
        return lista_liczb(self.ed_toki.text(), 10 ** 6)

    def odswiez(self):
        kat = self._katalog()
        self.drzewo.blockSignals(True); self.drzewo.clear()
        if kat is None:
            self.lb_hist.setText('Nie wczytano historii: można tylko wczytać obliczenia z pliku (trafią do cache historii).')
            self.lb_stan.setText('')
            self.drzewo.blockSignals(False)
            return
        self._wpisy = exports.lista(kat)
        grupy = {}
        for i, w in enumerate(self._wpisy):
            g = grupy.get(w['rodzaj'])
            if g is None:
                g = grupy[w['rodzaj']] = QTreeWidgetItem(self.drzewo, [w['opis']])
                g.setFlags(g.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
                g.setCheckState(0, Qt.CheckState.Unchecked); g.setExpanded(w['rodzaj'] not in ('przejscia', 'rzuty', 'historia'))
            it = QTreeWidgetItem(g, ['', '' if w['tok'] is None else str(w['tok']), w['parametry'], _mb(w['bajty']),
                                     time.strftime('%Y-%m-%d %H:%M', time.localtime(w['czas']))])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable); it.setCheckState(0, Qt.CheckState.Unchecked)
            it.setData(0, Qt.ItemDataRole.UserRole, i)
            it.setToolTip(2, '\n'.join(w['pliki'][:12]) + ('\n…' if len(w['pliki']) > 12 else ''))
        for g in grupy.values():
            g.setText(3, _mb(sum(self._wpisy[g.child(j).data(0, Qt.ItemDataRole.UserRole)]['bajty'] for j in range(g.childCount()))))
        self.drzewo.blockSignals(False)
        razem = sum(w['bajty'] for w in self._wpisy)
        self.lb_hist.setText(f'Historia {kat.name} ({self.z.h.K} checkpointów): {len(self._wpisy)} obliczeń, {_mb(razem)}.\n{kat}')
        self._opis_czastki()

    def _opis_czastki(self):
        """Co jest policzone dla cząsteczek z pola „Cząsteczki” (pierwsza i liczba pozostałych)."""
        toki = self._toki()
        if not toki:
            self.lb_stan.setText(''); return
        t = toki[0]
        mo = [w for w in self._wpisy if w['tok'] == t]
        czesci = [f'{exports.OPISY[w["rodzaj"]].split(" (")[0].lower()}: {w["parametry"]}' for w in mo]
        self.lb_stan.setText(f'Cząsteczka {t}: ' + ('; '.join(czesci) if czesci else 'nic jeszcze nie policzono (przebiegi na sferze są w historii)')
                             + (f'.  Pozostałe w polu: {len(toki) - 1}.' if len(toki) > 1 else '.'))

    def _zmiana(self, *_):
        pass

    def _zaznaczone(self):
        out = []
        for gi in range(self.drzewo.topLevelItemCount()):
            g = self.drzewo.topLevelItem(gi)
            for j in range(g.childCount()):
                it = g.child(j)
                if it.checkState(0) == Qt.CheckState.Checked:
                    out.append(self._wpisy[it.data(0, Qt.ItemDataRole.UserRole)])
        return out

    def _zaznacz_czastki(self):
        toki = set(self._toki())
        for gi in range(self.drzewo.topLevelItemCount()):
            g = self.drzewo.topLevelItem(gi)
            for j in range(g.childCount()):
                it = g.child(j)
                w = self._wpisy[it.data(0, Qt.ItemDataRole.UserRole)]
                if w['tok'] is not None and w['tok'] in toki:
                    it.setCheckState(0, Qt.CheckState.Checked); g.setExpanded(True)
        self._opis_czastki()

    # ------------------------------------------------------------------ akcje
    def _start(self, fn, po):
        if self._zadanie is not None and self._zadanie.isRunning():
            return
        self.pasek.setVisible(True); self.pasek.setValue(0)
        z = _Zadanie(fn)
        z.postep.connect(lambda i, n, o: (self.pasek.setMaximum(max(n, 1)), self.pasek.setValue(i), self.pasek.setFormat(o)))
        z.koniec.connect(lambda wynik, blad: (self.pasek.setVisible(False), po(wynik, blad)))
        self._zadanie = z
        z.start()

    def _eksportuj(self):
        kat = self._katalog()
        wpisy = self._zaznaczone()
        if kat is None or not wpisy:
            QMessageBox.information(self, 'Eksport', 'Zaznacz obliczenia na liście (albo „Zaznacz obliczenia cząsteczek”).'); return
        toki = self._toki()
        nazwa = f'obliczenia_{kat.name}' + (f'_t{toki[0]}' if toki and all(w['tok'] in (None, *toki) for w in wpisy) else '') + '.zip'
        cel, _ = QFileDialog.getSaveFileName(self, 'Eksport obliczeń', str(Path.home() / nazwa), 'Obliczenia (*.zip)')
        if not cel:
            return
        dodatki = {}
        if self.ch_przebiegi.isChecked() and toki:
            import tempfile
            tmp = Path(tempfile.mkdtemp()) / f'przebiegi_{kat.name}.csv'
            c, n = exports.przebiegi(self.z.h, toki, tmp, R=float(self.z.es.R), skala_stala=self.z.cb_skala.currentIndex() == 1)
            dodatki = {f'przebiegi/{c.name}': c.read_bytes(), f'przebiegi/{n.name}': n.read_bytes()}
        self._start(lambda postep: exports.eksportuj(kat, wpisy, cel, postep, dodatki),
                    lambda wynik, blad: QMessageBox.information(self, 'Eksport', blad or f'Zapisano {len(wpisy)} obliczeń:\n{wynik}'))

    def _importuj(self):
        plik, _ = QFileDialog.getOpenFileName(self, 'Wczytaj obliczenia', str(Path.home()), 'Obliczenia (*.zip)')
        if not plik:
            return
        from laser_app.compute import history
        try:
            man = exports.czytaj_manifest(plik)
        except Exception as e:                                           # noqa: BLE001
            QMessageBox.warning(self, 'Import', f'To nie jest plik eksportu obliczeń: {e}'); return
        cel = exports.cel_importu(man, history.CACHE)
        istnieje = sum((cel / r).exists() for r in man.get('pliki', {}))
        opis = (f'Plik z {man.get("utworzono", "?")}: {len(man.get("wpisy", []))} obliczeń, {len(man.get("pliki", {}))} plików, '
                f'{_mb(sum(f["bajty"] for f in man.get("pliki", {}).values()))}.\nHistoria: {man.get("klucz")} '
                f'({man.get("opis_historii", {}).get("kroki", "?")} checkpointów) → {cel}\n'
                f'Już istnieje: {istnieje} plików ({"zostaną nadpisane" if self.ch_nadpisz.isChecked() else "zostaną pominięte"}).')
        if QMessageBox.question(self, 'Import obliczeń', opis + '\n\nWczytać?') != QMessageBox.StandardButton.Yes:
            return
        nadpisz = self.ch_nadpisz.isChecked()
        self._start(lambda postep: exports.importuj(plik, history.CACHE, nadpisz, postep), self._po_imporcie)

    def _po_imporcie(self, wynik, blad):
        if blad or wynik is None:
            QMessageBox.warning(self, 'Import', blad or 'nieudany'); return
        kat, zapisane, pominiete, bledy = wynik
        tekst = f'Zapisano {len(zapisane)} plików, pominięto {len(pominiete)} (już były) w\n{kat}'
        if bledy:
            tekst += '\n\nBłędy:\n' + '\n'.join(bledy[:10])
        h = getattr(self.z, 'h', None)
        if h is not None and Path(h.katalog) == Path(kat):
            self.z.po_imporcie_obliczen()
        elif not bledy:
            tekst += '\n\nTo inna historia: wybierz te same checkpointy i zdjęcie w sekcji „Dane”, wczyta się bez liczenia.'
        QMessageBox.information(self, 'Import', tekst)
        self.odswiez()

    def _przebiegi(self):
        h = getattr(self.z, 'h', None)
        toki = self._toki()
        if h is None or not toki:
            QMessageBox.information(self, 'Przebiegi', 'Wczytaj historię i wpisz cząsteczki.'); return
        cel, _ = QFileDialog.getSaveFileName(self, 'Przebiegi cząsteczek na sferze',
                                             str(Path.home() / f'przebiegi_{Path(h.katalog).name}_t{toki[0]}.csv'), 'CSV (*.csv)')
        if not cel:
            return
        c, n = exports.przebiegi(h, toki, cel, R=float(self.z.es.R), skala_stala=self.z.cb_skala.currentIndex() == 1)
        QMessageBox.information(self, 'Przebiegi', f'{len(toki)} cząsteczek × {h.K} checkpointów:\n{c}\n{n}')

    def _otworz(self):
        kat = self._katalog()
        if kat is not None:
            os.startfile(str(kat))
