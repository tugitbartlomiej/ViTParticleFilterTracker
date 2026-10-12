"""Zakładka „Ruch zbioru”, część fizyki ruchu: strona opcji „Fizyka ruchu” (temperatura, reżim dyfuzji, MSD,
energia, pasma okresów, zakresy właściwości), panel wykresów fizyki i obszary oka z segmentacji.

Klasa bazowa dla `zakladka_ruchu.ZakladkaRuchu` (korzysta z jej `cloud`, `opcje`, `timeline`, `_int`, `_float`)."""
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QLabel, QPushButton, QVBoxLayout, QWidget

from styl import rola
from uklad import dodaj, pole_wyboru, przycisk, staly_napis


class FizykaRuchu:
    def _strona_fizyki(self):
        self.opcje.dodaj_strone('Fizyka ruchu')
        row = self.opcje.wiersz('Fizyka ruchu')
        self.spin_fz_okno = self._int(row, 'okno (zapisy)', 0, 2000, 5,
                                      'Ile zapisów w każdą stronę bieżącego kroku liczyć do właściwości ruchu. Okno ruchome '
                                      'pokazuje, jak reżimy zmieniają się w czasie uczenia: karta liczy okna wokół obu '
                                      'sąsiednich zapisów i układ „Fizyka ruchu” płynnie przechodzi między nimi. 0 = cała oś '
                                      'czasu: właściwości stałe, więc w tym układzie cząsteczki stoją podczas odtwarzania.')
        self.spin_fz_rozg = self._int(row, 'koniec rozgrzewki (krok)', 0, 10000000, 60,
                                      'Krok treningu, do którego liczony jest „ruch w rozgrzewce” (od kroku 0). Dla '
                                      'historii K1000 kolektywny przepływ trwa do około 30–60 kroków; dla historii po '
                                      'epokach pierwszy zapis to koniec epoki 1.')
        self.spin_fz_tau = self._int(row, 'τ dla α (zapisy)', 2, 100, 4,
                                     'Opóźnienie do wykładnika MSD: α = log(MSD(τ) / MSD(1)) / log τ. 1 = dyfuzja, '
                                     'mniej = uwięzienie (subdyfuzja), 2 = ruch balistyczny.')
        row = self.opcje.wiersz('Fizyka ruchu')
        self.combo_fz_osie = []
        for nazwa, dom in (('X', 0), ('Y', 6), ('Z', 1)):
            c = QComboBox()
            for i, t in self.WLASCIWOSCI.items():
                c.addItem(t, i)
            c.setCurrentIndex(c.findData(dom))
            dodaj(row, f'oś {nazwa}', c, f'Która właściwość ruchu leży na osi {nazwa} układu „Fizyka ruchu”. Zakres osi: '
                  'percentyle 1–99 widocznych cząsteczek.')
            self.combo_fz_osie.append(c)
        row = self.opcje.wiersz('Fizyka ruchu')
        self.spin_fz_pa = self._float(row, 'próg α dyfuzji', 0, 2, .7, 2,
                                      'Reżim: dyfuzyjny, gdy α ≥ ten próg; uwięziony, gdy mniejsze (i nie balistyczny).')
        self.spin_fz_pb = self._float(row, 'próg balistyczny', 0, 1, .3, 2,
                                      'Reżim balistyczny: VACF i prostoliniowość większe od tego progu (cząsteczka jedzie '
                                      'w jednym kierunku, jak w rozgrzewce).')
        self.combo_fz_obszar = QComboBox()
        for i, t in enumerate(('wszystkie obszary',) + tuple(f'tylko {o}' for o in self.OBSZARY)):
            self.combo_fz_obszar.addItem(t, i)
        dodaj(row, 'pokaż', self.combo_fz_obszar,
              'Filtr widocznych cząsteczek po obszarze oka z segmentacji Cataract-LMM (annotation.csv: wielokąty '
              'narzędzia, źrenicy i rogówki). Bez segmentacji: narzędzie wg ramki GT, reszta jako tło.')
        self.combo_fz_hist = QComboBox()
        for i, t in self.WLASCIWOSCI.items():
            self.combo_fz_hist.addItem(t, i)
        dodaj(row, 'histogram', self.combo_fz_hist, 'Właściwość na histogramie w panelu bocznym (osobno dla obszarów oka).')
        self.btn_fz = przycisk(row, 'Odśwież statystyki', 'Odczytuje z karty właściwości wszystkich widocznych '
                               'cząsteczek i przelicza histogramy, AUC i mapę rozgrzewki (dzieje się też samo po '
                               'zmianie kroku w pauzie).')
        # --- częstotliwość ruchu i energia
        row = self.opcje.wiersz('Fizyka ruchu')
        self.spin_pas_lo = self._float(row, 'pasmo: okres od (zapisy)', 2, 10000, 2, 1,
                                       'Dolna granica pasma okresów ruchu dla koloru „Udział energii w paśmie” (16). Okres w '
                                       'zapisach: 2 = zawracanie co zapis (najszybsze drgania), duże wartości = wolny dryf. Widmo '
                                       'liczy karta z przyrostów cząsteczki w oknie (najwyżej 128 kolejnych przyrostów ze środka '
                                       'okna), z pominięciem średniego dryfu (f = 0).')
        self.spin_pas_hi = self._float(row, 'do', 2, 10000, 4, 1, 'Górna granica pasma okresów (zapisy, włącznie).')
        self.pas_label = rola(QLabel(''), 'opis'); row.addWidget(self.pas_label)
        self.combo_fz_en = QComboBox()
        for t, v in (('kinetyczna ½⟨|Δx|²⟩', 0), ('potencjalna ½κ⟨|x − x̄|²⟩', 1), ('całkowita', 2)):
            self.combo_fz_en.addItem(t, v)
        dodaj(row, 'energia (kolor 15)', self.combo_fz_en,
              'Którą energię pokazuje kolor „Energia cząsteczki”: kinetyczna = połowa średniego kwadratu przesunięcia na zapis '
              '(z ruchem kolektywnym), potencjalna = energia sprężyny Ornsteina-Uhlenbecka ½κ⟨|x − x̄|²⟩ wokół średniego '
              'położenia w oknie, całkowita = suma. Skala log, zakres z percentyli 2–98.')
        # --- zakresy z kolorami (kolor 17)
        row = self.opcje.wiersz('Fizyka ruchu')
        self.combo_zak_c = QComboBox()
        for i, t in self.WLASCIWOSCI.items():
            self.combo_zak_c.addItem(t, i)
        self.combo_zak_c.setCurrentIndex(self.combo_zak_c.findData(8))
        dodaj(row, 'zakresy (kolor 17)', self.combo_zak_c,
              'Właściwość kolorowana zakresami: cząsteczka dostaje kolor pierwszego zakresu, do którego należy jej wartość '
              '(w jednostkach właściwości, bez logarytmu), reszta jest szara. Np. okres dominujący 2–3 zapisy = szybkie '
              'drgania, 3–6 = średnie, powyżej 6 = wolne; albo temperatura czy energia.')
        self.zak_wiersze = []
        for q, (od, do, kol) in enumerate(((2.0, 3.0, '#ff4026'), (3.0, 6.0, '#ffd933'), (6.0, 1e6, '#4d99ff'))):
            chk = pole_wyboru(row, f'#{q + 1}', f'Włącza zakres {q + 1}.', zaznaczone=True)
            lo = QDoubleSpinBox(); lo.setRange(-1e9, 1e9); lo.setDecimals(4); lo.setValue(od); lo.setMaximumWidth(95)
            hi = QDoubleSpinBox(); hi.setRange(-1e9, 1e9); hi.setDecimals(4); hi.setValue(do); hi.setMaximumWidth(95)
            lo.setToolTip(f'Początek zakresu {q + 1} (włącznie).'); hi.setToolTip(f'Koniec zakresu {q + 1} (włącznie).')
            btn = QPushButton(' '); btn.setMaximumWidth(36); btn.setToolTip(f'Kolor zakresu {q + 1}: kliknij, aby zmienić.')
            btn.setProperty('kolor', kol); btn.setStyleSheet(f'background:{kol};')
            btn.clicked.connect(lambda _, b=btn: self._wybierz_kolor_zakresu(b))
            for w in (lo, hi, btn):
                row.addWidget(w)
            self.zak_wiersze.append((chk, lo, hi, btn))
        self.btn_zak_tercyle = przycisk(row, 'Zakresy z tercyli', 'Ustawia trzy zakresy na tercyle wybranej właściwości '
                                        'widocznych cząsteczek (po jednej trzeciej w każdym kolorze).')
        self.fz_label = rola(QLabel('Fizyka ruchu: wybierz układ „Fizyka ruchu” albo kolor temperatury, reżimu lub α.'), 'opis')
        self.fz_label.setWordWrap(True)
        self.opcje._strony['Fizyka ruchu'].addWidget(self.fz_label)
        for w in (self.spin_fz_okno, self.spin_fz_rozg, self.spin_fz_tau, self.spin_fz_pa, self.spin_fz_pb):
            w.valueChanged.connect(lambda _: self._display_options())
        for c in self.combo_fz_osie + [self.combo_fz_obszar]:
            c.currentIndexChanged.connect(lambda _: self._display_options())
        self.combo_fz_hist.currentIndexChanged.connect(lambda _: self._fizyka_wykresy())
        self.btn_fz.clicked.connect(lambda: (self.cloud.zadaj_odczyt_fizyki(), self._budz()))
        for w in (self.spin_pas_lo, self.spin_pas_hi):
            w.valueChanged.connect(lambda _: self._display_options())
        for c in (self.combo_fz_en, self.combo_zak_c):
            c.currentIndexChanged.connect(lambda _: self._display_options())
        for chk, lo, hi, _ in self.zak_wiersze:
            chk.toggled.connect(lambda _: self._display_options())
            lo.valueChanged.connect(lambda _: self._display_options()); hi.valueChanged.connect(lambda _: self._display_options())
        self.btn_zak_tercyle.clicked.connect(self._zakresy_z_tercyli)
        self._fz_wynik = None
        self._fz_ost = 0.0
        self._fz_klucz = None
        self._obszary_klasy = None

    def _panel_fizyki(self, rl):
        self.fz_box = QWidget(); fl = QVBoxLayout(self.fz_box); fl.setContentsMargins(0, 0, 0, 0)
        self.fz_opis = staly_napis(rola(QLabel('Fizyka ruchu'), 'opis'), 7)
        fl.addWidget(self.fz_opis)
        self.fz_hist = pg.PlotWidget(title='Rozkład właściwości: obszary oka'); self.fz_hist.setMinimumHeight(170)
        self.fz_hist.addLegend(offset=(5, 5))
        self.fz_temp = pg.PlotWidget(title='Temperatura w czasie (energia drgań własnych na zapis)')
        self.fz_temp.setMinimumHeight(170); self.fz_temp.setLogMode(y=True); self.fz_temp.addLegend(offset=(5, 5))
        self.fz_cos = pg.PlotWidget(title='Reżim w czasie: cos kolejnych przesunięć (> próg = balistyczny)')
        self.fz_cos.setMinimumHeight(150); self.fz_cos.setYRange(-1, 1)
        self.fz_mapa = pg.ImageView(); self.fz_mapa.ui.roiBtn.hide(); self.fz_mapa.ui.menuBtn.hide()
        self.fz_mapa.setMinimumHeight(150)
        self.fz_mapa_label = staly_napis(QLabel('Ruch w rozgrzewce na siatce patchy (mediana po obrazach)'), 2)
        self._fz_kursory = []
        for w in (self.fz_temp, self.fz_cos):
            k = pg.InfiniteLine(angle=90, pen='#ffffff'); w.addItem(k); self._fz_kursory.append(k)
            w.setLabel('bottom', 'krok treningu')
        for w in (self.fz_hist, self.fz_temp, self.fz_cos, self.fz_mapa_label, self.fz_mapa):
            fl.addWidget(w)
        self.fz_box.setVisible(False)
        rl.addWidget(self.fz_box)

    def _obszary_oka(self, timeline):
        """W tle: klasy patchy z segmentacji Cataract-LMM (raz na oś czasu, potem z pliku obok osi)."""
        from concurrent.futures import ThreadPoolExecutor
        import obszary_oka as O
        self._obszary_klasy = None
        self.cloud.ustaw_obszary(None)
        item = self.combo_color.model().item(self.combo_color.findData(11))
        if item is not None:
            item.setEnabled(False)
        if not hasattr(self, '_obszary_pula'):
            self._obszary_pula = ThreadPoolExecutor(1, thread_name_prefix='obszary-oka')
        fut = self._obszary_pula.submit(O.dla_osi, timeline)

        def sprawdz():
            if timeline is not self.timeline:
                return
            if not fut.done():
                QTimer.singleShot(200, sprawdz); return
            try:
                klasy, _, brak = fut.result()
            except Exception as e:                                   # brak frame_id, zipów albo csv: bez obszarów
                self._set_text(self.fz_label, f'Obszary oka niedostępne ({e}); narzędzie wg ramki GT, reszta jako tło.')
                return
            self._obszary_klasy = klasy.reshape(-1)
            self.cloud.ustaw_obszary(self._obszary_klasy)
            if item is not None:
                item.setEnabled(True)
            udzialy = ', '.join(f'{o} {100 * (klasy == c).mean():.1f} %' for c, o in enumerate(self.OBSZARY))
            self._set_text(self.fz_label, f'Obszary oka z segmentacji: {udzialy}' +
                           (f'; {int(brak.sum())} zdjęć bez wpisu w annotation.csv (tło)' if brak.any() else '') + '.')
            self._budz()
        QTimer.singleShot(200, sprawdz)

    def _ustaw_fizyke(self):
        """Parametry fizyki do karty; odczyt statystyk w pauzie po zmianie, w ruchu najwyżej co 1 s."""
        self.cloud.ustaw_fizyke(okno=self.spin_fz_okno.value(), rozgrzewka=self.spin_fz_rozg.value(),
                                tau=self.spin_fz_tau.value(), prog_a=self.spin_fz_pa.value(), prog_b=self.spin_fz_pb.value(),
                                osie=[c.currentData() for c in self.combo_fz_osie])
        self.cloud.ustaw_filtr_obszaru(self.combo_fz_obszar.currentData())
        from PyQt6.QtGui import QColor
        zak = [(lo.value(), hi.value(), 1.0 if chk.isChecked() else 0.0, 0.0) for chk, lo, hi, _ in self.zak_wiersze]
        zkol = [QColor(b.property('kolor')).getRgbF()[:3] for *_, b in self.zak_wiersze]
        self.cloud.ustaw_kolory_fizyki(energia=self.combo_fz_en.currentData(), zak_c=self.combo_zak_c.currentData(),
                                       zak=zak, zkol=zkol, pas=(self.spin_pas_lo.value(), max(self.spin_pas_hi.value(),
                                                                                              self.spin_pas_lo.value())))
        if self.timeline is not None and len(self.timeline.steps) > 1:
            dk = float(np.median(np.diff(self.timeline.steps)))
            self._set_text(self.pas_label, f'≈ {self.spin_pas_lo.value() * dk:.0f}–{self.spin_pas_hi.value() * dk:.0f} kroków '
                           f'treningu (mediana odstępu zapisów {dk:.0f})')
        aktywna = self.combo_uklad.currentData() == 4 or self.combo_color.currentData() in (8, 9, 10, 11, 15, 16, 17)
        if self.combo_uklad.currentData() == 4 and self.spin_fz_okno.value() == 0 and self._playing:
            self._set_text(self.status, 'Układ „Fizyka ruchu” z oknem 0: właściwości z całej osi czasu są stałe, więc '
                           'cząsteczki stoją. Ustaw „okno (zapisy)” > 0 na stronie „Fizyka ruchu”, aby ruszały się w czasie.')
        self.fz_box.setVisible(aktywna)
        if not aktywna or not self.timeline:
            return
        first, count = self._scope()
        klucz = (str(self.timeline.root), first, count, self.spin_fz_okno.value(), self.spin_fz_rozg.value(),
                 self.spin_fz_tau.value(), self.spin_fz_pa.value(), self.spin_fz_pb.value(),
                 self._indices()[0] if self.spin_fz_okno.value() else None, self._obszary_klasy is not None,
                 self.spin_pas_lo.value(), self.spin_pas_hi.value(), self.combo_fz_en.currentData())
        teraz = time.perf_counter()
        if klucz != self._fz_klucz and (not self._playing or teraz - self._fz_ost > 1.0):
            self._fz_klucz, self._fz_ost = klucz, teraz
            self.cloud.zadaj_odczyt_fizyki()

    @staticmethod
    def _auc(a, b):
        """P(a > b) z rang (Mann-Whitney); None, gdy któraś grupa pusta."""
        if len(a) == 0 or len(b) == 0:
            return None
        x = np.concatenate([a, b])
        r = np.empty(len(x)); r[np.argsort(x, kind='mergesort')] = np.arange(1, len(x) + 1)
        return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))

    @classmethod
    def _wartosc(cls, wl, c):
        return (wl[:, 10] + wl[:, 11] if c == 12 else wl[:, c]).astype(np.float64)

    @classmethod
    def _przeksztalc(cls, wl, c):
        v = cls._wartosc(wl, c)
        return np.log(np.maximum(v, 1e-12)) if c in cls.WL_LOG else v

    def _wybierz_kolor_zakresu(self, btn):
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog
        k = QColorDialog.getColor(QColor(btn.property('kolor')), self, 'Kolor zakresu')
        if k.isValid():
            btn.setProperty('kolor', k.name()); btn.setStyleSheet(f'background:{k.name()};')
            self._display_options()

    def _zakresy_z_tercyli(self):
        if self._fz_wynik is None:
            self._set_text(self.fz_label, 'Najpierw odczytaj właściwości: układ „Fizyka ruchu” albo kolor 8–17, potem '
                           '„Odśwież statystyki”.'); return
        v = self._wartosc(self._fz_wynik['wl'], self.combo_zak_c.currentData())
        q = np.percentile(v, [0, 100 / 3, 200 / 3, 100])
        for (chk, lo, hi, _), a, b in zip(self.zak_wiersze, q[:-1], q[1:]):
            lo.blockSignals(True); hi.blockSignals(True)
            lo.setValue(float(a)); hi.setValue(float(b)); chk.setChecked(True)
            lo.blockSignals(False); hi.blockSignals(False)
        self._display_options()

    def _fizyka_gotowa(self, w):
        """Odczyt z karty: zakresy osi i kolorów, klasy obszarów, potem wykresy panelu."""
        wl = w['wl']
        if not len(wl):
            return
        osie = [c.currentData() for c in self.combo_fz_osie]
        tr = [self._przeksztalc(wl, c) for c in osie]
        lo = [float(np.percentile(t, 1)) for t in tr]; hi = [float(np.percentile(t, 99)) for t in tr]
        hi = [h if h > l else l + 1e-6 for l, h in zip(lo, hi)]
        lt = np.log(np.maximum(wl[:, 0], 1e-12))
        kol = (float(np.percentile(lt, 2)), float(np.percentile(lt, 98)),
               float(np.percentile(wl[:, 1], 2)), float(np.percentile(wl[:, 1], 98)))
        if (tuple(lo), tuple(hi), kol) != (self.cloud.fiz['lo'], self.cloud.fiz['hi'], self.cloud.fiz['kol']):
            self.cloud.ustaw_zakresy_fizyki(lo, hi, kol)
        le = self._przeksztalc(wl, {0: 10, 1: 11, 2: 12}[self.combo_fz_en.currentData()])
        self.cloud.ustaw_kolory_fizyki(en_kol=(float(np.percentile(le, 2)), float(np.percentile(le, 98)), 0.0, 0.0))
        klasy = w.get('obszary')
        if klasy is None:
            m = getattr(self.cloud, '_maska', None)
            klasy = (np.asarray(m[w['first']:w['first'] + w['count']]) & 1).astype(np.uint8) * 3 if m is not None \
                else np.zeros(len(wl), np.uint8)
        self._fz_wynik = dict(w, klasy=np.asarray(klasy))
        self._fizyka_wykresy()
        self._budz()

    def _fizyka_wykresy(self):
        w = self._fz_wynik
        if w is None:
            return
        wl, klasy = w['wl'], w['klasy']
        c = self.combo_fz_hist.currentData()
        v = self._przeksztalc(wl, c)
        lo, hi = np.percentile(v, [0.5, 99.5])
        bins = np.linspace(lo, hi if hi > lo else lo + 1e-6, 61)
        self.fz_hist.clear(); self.fz_hist.addLegend(offset=(5, 5))
        for k, (nazwa, kol) in enumerate(zip(self.OBSZARY, self.KOLORY_OBSZAROW)):
            m = klasy == k
            if m.sum() < 20:
                continue
            h, e = np.histogram(v[m], bins=bins, density=True)
            self.fz_hist.plot(e, h, stepMode='center', pen=pg.mkPen(kol, width=2), name=f'{nazwa} ({m.sum():,})')
        self.fz_hist.setLabel('bottom', self.WLASCIWOSCI.get(c, ''))
        # AUC względem tła i udziały reżimów
        tlo = klasy == 0
        linie = []
        for k in (3, 2, 1):
            m = klasy == k
            a_t = self._auc(np.log(np.maximum(wl[m, 0], 1e-12)), np.log(np.maximum(wl[tlo, 0], 1e-12)))
            a_z = self._auc(np.log(np.maximum(wl[m, 6], 1e-12)), np.log(np.maximum(wl[tlo, 6], 1e-12)))
            if a_t is None:
                continue
            rez = wl[m, 7]
            linie.append(f'{self.OBSZARY[k]} vs tło: AUC temperatury {a_t:.2f}, zmiany 768D {a_z:.2f}; reżimy bal./dyf./uw. '
                         f'{100 * (rez == 0).mean():.0f}/{100 * (rez == 1).mean():.0f}/{100 * (rez == 2).mean():.0f} %')
        rez = wl[:, 7]
        med_t = ' · '.join(f'{self.OBSZARY[k]} {np.median(wl[klasy == k, 0]):.3g}' for k in range(4) if (klasy == k).sum() >= 20)
        med_t += '.\nMediana okresu dominującego (zapisy) / udział pasma: ' + ' · '.join(
            f'{self.OBSZARY[k]} {np.median(wl[klasy == k, 8]):.1f} / {np.mean(wl[klasy == k, 9]):.2f}'
            for k in range(4) if (klasy == k).sum() >= 20)
        self._set_text(self.fz_opis, f'Okno {w["okno"][0]}–{w["okno"][1]}, rozgrzewka do kroku {w["rozgrzewka"]}, τ = {w["tau"]}. '
                       f'{len(wl):,} cząsteczek; reżimy bal./dyf./uw.: {100 * (rez == 0).mean():.0f}/'
                       f'{100 * (rez == 1).mean():.0f}/{100 * (rez == 2).mean():.0f} %.\nMediana temperatury: {med_t}.\n'
                       + '\n'.join(linie))
        # serie populacji w czasie
        s = w.get('seria')
        if s is not None:
            kr = np.asarray(s['kroki'], np.float64)
            self.fz_temp.clear(); self.fz_temp.addLegend(offset=(5, 5))
            for k, (nazwa, kol) in enumerate(zip(self.OBSZARY, self.KOLORY_OBSZAROW)):
                t = s['temperatura'][:, k]
                ok = np.isfinite(t) & (t > 0)
                if ok.sum() > 1:
                    self.fz_temp.plot(kr[ok], t[ok], pen=pg.mkPen(kol, width=2), name=nazwa)
            ok = np.isfinite(s['pole']) & (s['pole'] > 0)
            if ok.sum() > 1:
                self.fz_temp.plot(kr[ok], s['pole'][ok], pen=pg.mkPen('#ffd166', width=1, style=Qt.PenStyle.DashLine),
                                  name='pole kolektywne |średni ruch|²')
            self.fz_cos.clear()
            prog = self.spin_fz_pb.value()
            self.fz_cos.addItem(pg.InfiniteLine(pos=prog, angle=0, pen=pg.mkPen('#ffd166', style=Qt.PenStyle.DashLine)))
            self.fz_cos.addItem(pg.InfiniteLine(pos=0, angle=0, pen='#555555'))
            for k, kol in enumerate(self.KOLORY_OBSZAROW):
                cs = s['cos'][:, k]
                ok = np.isfinite(cs)
                if ok.sum() > 1:
                    self.fz_cos.plot(kr[ok], cs[ok], pen=pg.mkPen(kol, width=2))
            for k, wdg in zip(self._fz_kursory, (self.fz_temp, self.fz_cos)):
                wdg.addItem(k)
        # mapa rozgrzewki na siatce patchy i udział wariancji tłumaczony przez położenie patcha
        patch = (w['first'] + np.arange(len(wl))) % 600
        r = np.log(np.maximum(wl[:, 5], 1e-12))
        sr = np.bincount(patch, r, minlength=600) / np.maximum(np.bincount(patch, minlength=600), 1)
        expl = 1 - ((r - sr[patch]) ** 2).mean() / max(r.var(), 1e-30)
        med = np.array([np.median(r[patch == t]) if (patch == t).any() else np.nan for t in range(600)])
        self.fz_mapa.setImage(np.nan_to_num(med.reshape(20, 30).T, nan=float(np.nanmin(med)) if np.isfinite(med).any() else 0),
                              autoLevels=True)
        self._set_text(self.fz_mapa_label, f'Ruch w rozgrzewce (log, mediana po obrazach) na siatce patchy. Położenie '
                       f'patcha tłumaczy {100 * expl:.0f} % jego wariancji.')
