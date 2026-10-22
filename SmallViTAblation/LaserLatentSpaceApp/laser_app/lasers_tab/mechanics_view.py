"""Okienko 13: mechanika wybranego reflektora (reflector_mechanics, reflector_drawing)."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QLabel

from laser_app.ui.widgets import CollapsibleSection, NarrowCombo, _spin


class MechanicsMixin:
    """Okienko 13: mechanika wybranego reflektora (reflector_mechanics, reflector_drawing).
    Część LasersTab (lasers_tab/tab.py); korzysta z jej atrybutów przez self."""

    def _buduj_sek_mech(self):
        """Sekcja mechaniki reflektora (okienko 13)."""
        # --- mechanika reflektora (okienko 13)
        g = CollapsibleSection('Reflektor: mechanika — odrzut wiązek i ruch (okienko 13)'); f = QFormLayout(g)
        self.ch_mech = QCheckBox('licz mechanikę wybranego reflektora (blok 6)'); self.ch_mech.setChecked(True)
        self.ch_mech.setToolTip('Reflektor = „Reflektor” z sekcji sfery (neuron c bloku 6), token = „Cząsteczka”.\n'
                                'Siła pola celu na koloid j to −∂L/∂x_j; wiązka przenosi ją na reflektor proporcjonalnie do jasności z_jc.\n'
                                'Wypadkowy odrzut na igłę v_c to dokładnie −∂L/∂v_c = Σ_j z_jc F_j (zasada Z22).\n'
                                'Wykonany ruch = różnica wag do następnego checkpointu (suma 10 albo 100 kroków AdamW).')
        self.cb_mech_zr = NarrowCombo()
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
        from laser_app.compute import reflector_mechanics
        tekst = reflector_mechanics.opis(d)
        self.lb_mech.setText(tekst); self.lb13.setText(tekst)
        self._mech_rysuj()

    def _mech_rysuj(self):
        d, h = self._mech, self.h
        if d is None or h is None:
            return
        from laser_app.compute import reflector_drawing
        u = float(np.clip(self.t - d['i'], 0.0, 1.0)) if not d['ostatni'] else 1.0
        przes = self.sp_mech_przes.value() or float(np.clip(25.0 / max(d['igla']['kat'], d['cel']['kat'], 1e-3), 1, 500))
        S = reflector_drawing.schemat(d, u, przes, n_linek=self.sp_mech_linki.value(), celownik=self.ch_mech_cel.isChecked(),
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
