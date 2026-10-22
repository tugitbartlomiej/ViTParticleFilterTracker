"""Etykieta narzędzia (tryby, podgląd łat), obszary oka, wybór cząsteczek i kolory cząsteczek."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel, QMenu, QToolButton, QWidget

from laser_app.compute import history as hist
from laser_app.lasers_tab.workers import plaski_podkrokow
from laser_app.ui.widgets import CollapsibleSection, NarrowCombo, _spin


class LabelsMixin:
    """Etykieta narzędzia (tryby, podgląd łat), obszary oka, wybór cząsteczek i kolory cząsteczek.
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_wybor(self):
        """Wybór cząsteczki i reflektora: osobna sekcja (strona sfery i popup sfery), bo czyta go większość okienek."""
        g = CollapsibleSection('Wybrana cząsteczka i reflektor'); f = QFormLayout(g)
        for etykieta, pole in self._wiersze_wyboru:
            f.addRow(etykieta, pole)
        bt = QToolButton(); bt.setText('Zaznacz obszar ▾'); bt.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        bt.setToolTip('Wpisuje do „+ cząsteczki” wszystkie patche wybranego obszaru (etykieta narzędzia z sekcji „Etykiety”,\n'
                      'źrenica i rogówka z obrysów Cataract-LMM). Działa z „Wiązki: do wybranych cząsteczek” i „Wektory dla: wybranych”.')
        m = QMenu(bt)
        for kod, nazwa in ((3, 'narzędzie'), (2, 'źrenica'), (1, 'rogówka'), (0, 'tło'), (-1, 'wyczyść')):
            a = m.addAction(nazwa); a.triggered.connect(lambda _=False, k=kod: self._zaznacz_obszar(k))
        bt.setMenu(m)
        bt_ob = QToolButton(); bt_ob.setText('Obliczenia i przebiegi…')
        bt_ob.setToolTip('Co jest policzone dla tych cząsteczek (pułapki, pomiary, pamięć pól i doliny), eksport do pliku i wczytanie\n'
                         'z pliku zamiast liczenia, przebiegi cząsteczek na sferze do CSV (okno „Obliczenia: eksport i import”).')
        bt_ob.clicked.connect(self.okno_obliczen)
        w_ = QWidget(); l_ = QHBoxLayout(w_); l_.setContentsMargins(0, 0, 0, 0); l_.addWidget(bt); l_.addWidget(bt_ob); l_.addStretch(1)
        f.addRow('', w_)
        self._sek['wybor'] = g

    def okno_obliczen(self):
        """Okno „Obliczenia: eksport i import” dla bieżącej historii i wybranych cząsteczek."""
        from laser_app.ui.computations_dialog import ComputationsDialog
        d = ComputationsDialog(self, self.window())
        d.exec()

    def po_imporcie_obliczen(self):
        """Po imporcie do bieżącej historii: listy gotowych pułapek, mapa i pułapka z nowych plików (bez liczenia)."""
        self._og_lista_gotowych()
        if getattr(self, '_mp', None) is None and self.ch_mp.isChecked():
            self._mp_zlec()
        if self.ch_og.isChecked() and getattr(self, '_og', None) is None:
            self._og_zlec()

    def _buduj_etykiety(self):
        from laser_app.compute import tool_labels
        g = CollapsibleSection('Etykiety: narzędzie (obrys) i obszary oka'); f = QFormLayout(g)
        self.cb_etyk = NarrowCombo()
        for k, opis in tool_labels.TRYBY:
            self.cb_etyk.addItem(opis, k)
        self.cb_etyk.setToolTip('Które z 600 patchy są narzędziem (kolor „narzędzie”, „Wektory dla: narzędzia”, przycisk „narzędzie”,\n'
                                'pierścienie w okienkach 1–3, okienko 2: narzędzie/reszta):\n'
                                '• dotknięta obrysem: każdy patch, w który wchodzi choć jeden piksel obrysu narzędzia — całe narzędzie,\n'
                                '  także patche w większości z tłem (domyślnie);\n'
                                '• pokrycie ≥ 50 % (jak dawniej, do 2026-10-21): patch, w którym co najmniej połowa pikseli to narzędzie;\n'
                                '• pokrycie ≥ własny próg: tylko patche w co najmniej „próg” wypełnione narzędziem (cienki cystotom: 2–3 patche);\n'
                                '• środek łaty w obrysie: środkowy piksel patcha leży na narzędziu;\n'
                                '• ramka GT: środek patcha w prostokącie ramki (dużo tła).\n'
                                'Rzuty R1–R12 używają etykiety z chwili liczenia stanów 192D (R9 ma własny próg).')
        self.sp_etyk_prog = _spin(0.01, 1.0, 0.5, 0.05, 2); self.sp_etyk_prog.setToolTip('Próg pokrycia patcha dla trybu „pokrycie ≥ próg”.')
        self.sp_oko_prog = _spin(0.05, 1.0, 0.5, 0.05, 2)
        self.sp_oko_prog.setToolTip('Patch należy do źrenicy / rogówki, gdy ich obrys pokrywa co najmniej tyle patcha.\n'
                                    'Pierwszeństwo: narzędzie > źrenica > rogówka > tło. ViT nie był uczony na obszarach oka.')
        self.lb_etyk = QLabel('—'); self.lb_etyk.setWordWrap(True)
        from laser_app.ui.label_preview import LabelPreview
        self.ch_etyk_podglad = QCheckBox('podgląd łat narzędzia na zdjęciu'); self.ch_etyk_podglad.setChecked(True)
        self.ch_etyk_podglad.setToolTip('Siatka 30 × 20 łat na zdjęciu: zielone = narzędzie w wybranym trybie (jaśniejsze = większy udział pikseli\n'
                                        'obrysu), żółta obwódka = mieszane (dotknięte obrysem, ale w tym trybie nie narzędzie), biała = wybrana\n'
                                        'cząsteczka. Najechanie pokazuje udział, klik wybiera cząsteczkę.')
        self.podglad_etyk = LabelPreview(); self.podglad_etyk.klik.connect(self.sp_tok.setValue)
        self.podglad_etyk.setToolTip(self.ch_etyk_podglad.toolTip())
        self.ch_etyk_podglad.toggled.connect(lambda on: (self.podglad_etyk.setVisible(on), self._opis_etykiety()))
        self.lb_etyk_tryby = QLabel(''); self.lb_etyk_tryby.setWordWrap(True)
        self.lb_etyk_tryby.setToolTip('Ile łat narzędzia daje każdy tryb etykiety dla tego zdjęcia (porównanie trybów).')
        f.addRow('Narzędzie', self.cb_etyk); f.addRow('  próg pokrycia', self.sp_etyk_prog); f.addRow('Obszary oka: próg', self.sp_oko_prog)
        f.addRow(self.lb_etyk); f.addRow(self.lb_etyk_tryby); f.addRow(self.ch_etyk_podglad); f.addRow(self.podglad_etyk)
        self.sp_etyk_prog.setEnabled(False)
        self.cb_etyk.currentIndexChanged.connect(lambda *_: self.sp_etyk_prog.setEnabled(self.cb_etyk.currentData() == 'pokrycie'))
        self.sp_tok.valueChanged.connect(lambda v: self.podglad_etyk.ustaw(None, self.podglad_etyk.udzial, self.podglad_etyk.narz,
                                                                          self.podglad_etyk.mies, int(v)) if self.podglad_etyk.isVisible() else None)
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
        from laser_app.compute import history as hist
        k = h.obszary(self.sp_oko_prog.value()) if hasattr(h, 'obszary') else h.narz.astype(np.uint8) * 3
        ile = ', '.join(f'{hist.KLASY_OKA[i]} {int((k == i).sum())}' for i in (3, 2, 1, 0)) if hasattr(hist, 'KLASY_OKA') else ''
        bez_oka = '' if getattr(h, 'oko', None) is not None else ' Brak obrysów oka dla tego zdjęcia.'
        self.lb_etyk.setText(f'{h.etykieta}.  Patche: {ile}.{bez_oka}')
        if getattr(h, 'udzial', None) is not None:
            from laser_app.ui.label_preview import opis_trybow
            self.lb_etyk_tryby.setText('Łaty narzędzia w trybach: ' + opis_trybow(h.udzial, self.sp_etyk_prog.value(), getattr(h, 'srodek', None),
                                                                                   getattr(h, 'narz_ramka', None)))
        else:
            self.lb_etyk_tryby.setText('Brak obrysu narzędzia dla tego zdjęcia: etykieta z ramki GT.')
        if self.ch_etyk_podglad.isChecked():
            self.podglad_etyk.ustaw(h.rgb, getattr(h, 'udzial', None), h.narz, getattr(h, 'mieszane', None), int(self.sp_tok.value()))

    def _zaznacz_obszar(self, kod):
        if kod < 0 or self.h is None:
            self.ed_tok.setText(''); self.ed_tok.editingFinished.emit(); return
        k = self.h.obszary(self.sp_oko_prog.value()) if hasattr(self.h, 'obszary') else self.h.narz.astype(np.uint8) * 3
        tok = np.flatnonzero(k == kod)
        self.ed_tok.setText(', '.join(str(int(t)) for t in tok)); self.ed_tok.editingFinished.emit()

    def _kolory_czastek(self):
        if self.h is None:
            return
        self._kol_cz = self._kolory_tryb(self.cb_kolor_cz.currentIndex())

    def _kolory_tryb(self, tryb, h=None):
        """[600, 4] kolory cząstek: 0 kolor patcha ze zdjęcia, 1 położenie na siatce (barwa = kolumna, jasność = wiersz), 2 narzędzie,
        3 obszary oka (narzędzie błękit, źrenica fiolet, rogówka zieleń, tło szare; etykieta i próg z sekcji „Etykiety”)."""
        h = self.h if h is None else h
        K = np.ones((hist.TOK, 4), np.float32)
        if tryb == 0:
            K[:, :3] = h.rgb.reshape(hist.H, 16, hist.W, 16, 3).mean((1, 3)).reshape(-1, 3) / 255.0
            K[:, :3] = np.clip(0.25 + 0.9 * K[:, :3], 0, 1)
        elif tryb == 1:
            import colorsys
            r, c = np.divmod(np.arange(hist.TOK), hist.W)
            K[:, :3] = np.array([colorsys.hsv_to_rgb(cc / hist.W, 0.85, 0.45 + 0.55 * rr / (hist.H - 1)) for rr, cc in zip(r, c)])
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
