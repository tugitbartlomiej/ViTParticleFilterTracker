"""Okienko 12: dolina pewności i IoU wybranej cząsteczki (valley) i ta sama dolina na sferze."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QLabel

from laser_app.ui.widgets import CollapsibleSection, NarrowCombo, _spin


class ValleyMixin:
    """Okienko 12: dolina pewności i IoU wybranej cząsteczki (valley) i ta sama dolina na sferze.
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_sek_dolina(self):
        """Sekcja doliny pewności i IoU (okienko 12)."""
        # --- dolina pewności i IoU (okienko 12)
        g = CollapsibleSection('Dolina pewności i IoU wybranej cząsteczki (okienko 12)'); f = QFormLayout(g)
        self.ch_dol = QCheckBox('licz dolinę w bieżącym checkpoincie'); self.ch_dol.setChecked(True)
        self.ch_dol.setToolTip('Gdzie w przestrzeni stanu cząsteczki (192 wymiary, wejście MLP bloku 6) głowa dawałaby wysoką pewność\n'
                               'i dobre IoU ramki. Głowa czyta z położenia tylko 5 kierunków (ściany doliny); w pozostałych 187\n'
                               '(dno doliny) odpowiedź z bliska się nie zmienia. Rysowany jest przekrój 3D przez cząsteczkę.\n'
                               'Liczy się w tle przy każdej zmianie checkpointu (oś uczenia, nie podkroki).')
        self.cb_dol_baza = NarrowCombo()
        from laser_app.compute import valley
        self.cb_dol_baza.addItems(list(valley.OPIS_BAZ))
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
        from laser_app.compute import valley
        from laser_app.compute import disk_cache
        plik = disk_cache.plik(self.h.katalog, 'dolina', t=k[2], baza=valley.BAZY[k[3]], G=k[4], z=k[5], k=int(self.h.kroki[k[1]]))
        self.watek_dol.zlec(k, self.h, k[1], k[2], valley.BAZY[k[3]], k[4], k[5], plik=plik)

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
        from laser_app.compute import valley
        tr = valley.obrys(d, self.sp_dol_p.value(), self.sp_dol_iou.value())
        self.edol.ustaw_obrys(tr); self.es_dol.ustaw_obrys(tr)
        tekst = valley.opis(d, self.sp_dol_p.value(), self.sp_dol_iou.value())
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
