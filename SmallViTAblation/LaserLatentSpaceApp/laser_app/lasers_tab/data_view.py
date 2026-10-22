"""Dane zakładki: wybór historii checkpointów i zdjęcia, liczenie albo wczytanie historii (history.policz w wątku), wykresy 6–10."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QFileDialog, QFormLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton
import numpy as np
import pyqtgraph as pg

from laser_app.compute import history as hist
from laser_app.lasers_tab.shared import EP, pl
from laser_app.lasers_tab.workers import HistoryWorker
from laser_app.ui.layout_widgets import NarrowCombo
from laser_app.ui.widgets import _spin, CollapsibleSection


class DataMixin:
    """Dane zakładki: wybór historii checkpointów i zdjęcia, liczenie albo wczytanie historii (history.policz w wątku), wykresy 6–10.
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_sek_dane(self):
        """Sekcja „Dane”: historia checkpointów, zakres kroków, film i zdjęcie, liczenie."""
        # --- dane
        g = CollapsibleSection('Dane: historia uczenia i zdjęcie'); f = QFormLayout(g)
        self.cb_historia = NarrowCombo(); self.cb_historia.setToolTip('Katalog checkpointów k<krok>.pt (historia uczenia).')
        b = QPushButton('Inny katalog wag…'); b.clicked.connect(self._inny_katalog)
        f.addRow('Historia', self.cb_historia); f.addRow('', b)
        self.sp_od = _spin(0, 10 ** 7, 0, 100); self.sp_do = _spin(0, 10 ** 7, 10 ** 7, 100); self.sp_co = _spin(1, 100, 1)
        self.sp_od.setToolTip('Pierwszy krok uczenia (checkpointy z mniejszym krokiem są pomijane).')
        self.sp_do.setToolTip('Ostatni krok uczenia.')
        self.sp_co.setToolTip('Co który checkpoint brać (1 = wszystkie). Drgania wymagają gęstych checkpointów (okno ± kroków).')
        f.addRow('Od kroku', self.sp_od); f.addRow('Do kroku', self.sp_do); f.addRow('Co n-ty checkpoint', self.sp_co)
        self.cb_film = NarrowCombo()
        self.cb_film.setToolTip('Film zbioru uczącego biegu 35 000 (MANIFEST_35000: 117 filmów, 35 000 klatek z ramką narzędzia)\n'
                                'albo pliki z app/zdjeciatestowe (większość z nich NIE była w zbiorze uczącym).')
        self.cb_zdjecie = NarrowCombo(); b2 = QPushButton('Zdjęcie z pliku…'); b2.clicked.connect(self._plik_zdjecia)
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

    # ------------------------------------------------------------------ wybór danych
    def _lista_historii(self):
        from laser_app.compute.checkpoint_histories import HISTORIE
        self._historie = [(h.nazwa, h.wagi, h.na_epoke) for h in HISTORIE if h.wagi.is_dir()]
        self._historie.sort(key=lambda x: '35 000' not in x[0])          # bieg 35 000 (filmy) na początku
        self.cb_historia.blockSignals(True)
        self.cb_historia.clear()
        for n, w, _ in self._historie:
            self.cb_historia.addItem(n, str(w))
        self.cb_historia.blockSignals(False)

    def _inny_katalog(self):
        k = QFileDialog.getExistingDirectory(self, 'Katalog wag (k<krok>.pt)', str(hist.ROOT / 'assets'))
        if k:
            self.cb_historia.addItem(f'katalog: {Path(k).name}', k)
            self.cb_historia.setCurrentIndex(self.cb_historia.count() - 1)

    def _lista_zdjec(self):
        """Filmy zbioru uczącego (kolejność manifestu), na końcu grupa plików testowych."""
        self._filmy = {}
        try:
            for r in hist.zbior_uczacy():
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
                self.cb_zdjecie.addItem(f'kl. {r["klatka"]:06d}  (nr {r["nr"]})', f'{hist.PREFIKS_UCZACY}{r["nr"]}')
        else:
            if hist.ZDJECIE_FILMU.exists():
                self.cb_zdjecie.addItem('TRAIN 0 (zdjęcie filmów 2, 12 i 16)', str(hist.ZDJECIE_FILMU))
            for p in sorted((hist.APP / 'zdjeciatestowe').glob('*.jpg')):
                ucz = hist.czy_w_zbiorze_uczacym(p)
                self.cb_zdjecie.addItem(p.name + (f'  (w zbiorze uczącym, nr {ucz["nr"]})' if ucz else ''), str(p))
        self.cb_zdjecie.blockSignals(False)
        self._po_wyborze()

    def _plik_zdjecia(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Zdjęcie', str(hist.APP / 'zdjeciatestowe'), 'Obrazy (*.jpg *.jpeg *.png)')
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
        rgb, roz = hist.wczytaj_zdjecie(zd)
        gt, opis = hist.ramka_prawdziwa(zd, roz)
        self.lb_gt.setText(opis + ('' if gt is None else f': [{", ".join(f"{v:.0f}" for v in gt)}] px'))
        return wagi, kroki, rgb, gt, self.sp_okno.value()

    def _po_wyborze(self):
        try:
            wagi, kroki, rgb, gt, okno = self._wybor()
        except Exception as e:                                           # noqa: BLE001
            self.lb_stan.setText(f'Błąd wyboru: {e}'); return
        if rgb is None or len(kroki) < 2:
            self.lb_stan.setText('Za mało checkpointów w zakresie albo brak zdjęcia.'); return
        jest = (hist.CACHE / hist.klucz(wagi, kroki, rgb, gt, okno) / 'dane.npz').exists()
        odst = np.median(np.diff(kroki)) if len(kroki) > 1 else 0
        uw = '' if odst <= okno else f'  Uwaga: mediana odstępu checkpointów {odst:.0f} kroków > okno ±{okno}: centra = same tokeny, drgań nie będzie.'
        self.lb_stan.setText(f'{len(kroki)} checkpointów ({kroki[0]}–{kroki[-1]}). '
                             + ('Jest w cache: wczyta się od razu.' if jest else 'Nie liczone: pierwsze liczenie czyta wszystkie pliki wag (minuty).') + uw)

    def _wczytaj_z_cache(self):
        try:
            wagi, kroki, rgb, gt, okno = self._wybor()
            if rgb is not None and len(kroki) >= 2 and (hist.CACHE / hist.klucz(wagi, kroki, rgb, gt, okno) / 'dane.npz').exists():
                self._licz()
        except Exception:                                                # noqa: BLE001
            pass

    def _licz(self):
        if self.watek is not None:
            return
        wagi, kroki, rgb, gt, okno = self._wybor()
        if rgb is None or len(kroki) < 2:
            self.lb_stan.setText('Za mało checkpointów albo brak zdjęcia.'); return
        self.watek = HistoryWorker(wagi, kroki, rgb, gt, okno)
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
        self.h = h = hist.History(katalog)
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
        self.v2.kadruj(0, 0, hist.W + 3, hist.H + 3)
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
        self._mp = None; self.gl_mp.widoczna(False); self.gl_mp_s.widoczna(False)
        self._og_lista_gotowych()
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
