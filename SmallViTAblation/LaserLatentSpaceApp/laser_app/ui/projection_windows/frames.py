"""Okienka R4 (rama siły B) i R5 (rama Schura C, liczona w tle)."""
from __future__ import annotations

import numpy as np

from laser_app.compute import history as hist, projections
from laser_app.ui.projection_windows.scene import LocalSections
from laser_app.ui.projection_windows.shared import KOL_SILA, KOL_WYBR, pl, proc, S_EKR


class FramesMixin:
    """Okienka R4 (rama siły B) i R5 (rama Schura C, liczona w tle).
    Część ProjectionWindows (ui/projections_view.py); korzysta z jej atrybutów przez self."""

    def _rama_B(self, ir, jt):
        """Rama siły i sieć silna cząsteczki jt w checkpoincie ir (raz na checkpoint, cząsteczkę i próg; kilka ostatnich)."""
        klucz = (ir, jt, round(float(self.sp_B_prog.value()), 3))
        w = self._frB.get(klucz)
        if w is None:
            u = self._wklady(ir, jt)
            silne, reszta = projections.siec_silna(u, klucz[2])
            w = self._frB[klucz] = dict(u=u, fr=projections.rama_sily(u), silne=silne, reszta=reszta)
            while len(self._frB) > 6:
                self._frB.pop(next(iter(self._frB)))
        return w

    # ------------------------------------------------------------------ R4 B: rama siły
    def _rysuj_B(self):
        d, sc = self.dane, self._sceny['R4']
        _, _, _, ir = self._indeksy()
        wybrane = self._wybrane()
        if not wybrane:
            return
        j = wybrane[0]
        wB = self._rama_B(ir, j)
        u, fr, silne, reszta = wB['u'], wB['fr'], wB['silne'], wB['reszta']
        B = fr['B']
        u3, r3, F3 = u[silne] @ B, reszta @ B, fr['F'] @ B
        L = max(float(np.linalg.norm(u3, axis=1).max()), float(np.linalg.norm(F3)), float(np.linalg.norm(r3)), 1e-30)
        s = 1.2 * S_EKR * self.sp_B_skala.value() / L

        def budowa():
            el = LocalSections().strzalki(np.zeros(3), u3 * s, self._kolory_wiazek(u[silne], u3, fr['cos'][silne]))
            el.strzalki(np.zeros(3), (F3 * s)[None], KOL_SILA + (1.0,), gruba=True)
            el.strzalki(np.zeros(3), (r3 * s)[None], (0.62, 0.62, 0.62, 0.9), gruba=True)
            return el.kropki(np.zeros((1, 3)), KOL_WYBR + (1.0,))

        sc.g.opcje.update(czastki=False, strzalki=False, rozm=9.0)
        sc.lokalne(('B', ir, j, round(float(self.sp_B_prog.value()), 3), float(self.sp_B_skala.value()), self.ch_rho.isChecked()),
                   budowa)
        sc.odcinki('zero', None, None)
        nazwy = ['F̂ (100 % siły)', 'poprzeczna 1', 'poprzeczna 2'] if self.ch_osie.isChecked() else None
        napisy = [((F3 * s) * 1.05, 'siła netto', (255, 190, 110, 230)), ((r3 * s) * 1.05, f'reszta {hist.NK - len(silne)} wiązek',
                                                                            (200, 200, 200, 220))]
        sc.podpisy(napisy + sc.osie(nazwy))
        sc.kadruj(np.concatenate([u3 * s, (F3 * s)[None], (r3 * s)[None]]), ('R4', 'stala'), kwantyl=100, zapas=2.0)
        if 'pary' not in wB:
            wB['pary'] = tuple(projections.pary_przeciwne(u[silne], B_) if len(silne) <= 400 else np.nan for B_ in (None, B, d.baza_P3))
        p192, p3, pP3 = wB['pary']
        self._B = dict(ir=ir, j=j, u=u, fr=fr)
        self._lb['R4'].setText(
            f'Cząsteczka {j}, checkpoint {ir + 1}/{d.K} (krok {int(d.kroki[ir])}): |F| = {pl(fr["nF"] / float(d.R[ir]), 3)} R; '
            f'oś 1 = F̂ (100 % siły), osie 2–3 niosą {proc(fr["poprz"])} energii składowych poprzecznych (zero: 2/191 = 1,0 %). '
            f'Sieć silna: {len(silne)} wiązek = {proc(self.sp_B_prog.value())} energii, reszta {hist.NK - len(silne)} jako szara strzałka '
            f'(suma = F dokładnie). Pomarańczowe pchają wzdłuż F̂, niebieskie przeciw (cos w 192D). Pary ≥ 150° w sieci silnej: '
            f'192D {proc(p192)}, ta rama {proc(p3)}, P3 {proc(pP3)} (losowe 3D 6,7 %). Rama lokalna: strzałek różnych cząsteczek nie porównywać.')

    # ------------------------------------------------------------------ R5 C: rama Schura
    def _klucz_C(self):
        _, _, _, ir = self._indeksy()
        wybrane = self._wybrane()
        return (str(self._h.katalog), ir, wybrane[0] if wybrane else 0, self.cb_C_para.currentIndex(), self.cb_C_rzecz.currentIndex(),
                int(self.sp_C_G.value()), round(float(self.sp_C_zas.value()), 3))

    def _zlec_C(self):
        if not self.ch_C.isChecked() or self._h is None:
            return
        k = self._klucz_C()
        if k == self._C_zlecony:
            return
        self._C_zlecony = k
        h = self._h
        self._watek_C.zlec(k, dict(wagi=str(projections._katalog_wag(h)), krok=int(h.kroki[k[1]]), rgb=h.rgb, tok=k[2], para=k[3], rzecz=k[4],
                                   G=k[5], zasieg=k[6], krok0=int(h.kroki[0])))

    def _C_gotowe(self, klucz, w):
        if 'blad_watku' in w:
            self._lb['R5'].setText(f'Rama C: {w["blad_watku"]}')
            return
        if self._h is None or klucz[0] != str(self._h.katalog):
            return
        self._C = (klucz, w)
        if self._z.widac('R5'):
            self._rysuj_C(zlec=False)
        if self._z.widac('R8'):
            self._panel_widmo()

    def _rysuj_C(self, zlec=True):
        if zlec:
            self._zlec_C()
        sc = self._sceny['R5']
        sc.g.opcje.update(czastki=False, strzalki=False, rozm=9.0)
        if self._C is None:
            sc.lokalne(None, None)
            self._lb['R5'].setText('Rama C: liczenie (przejście w przód, jakobian 192 × 192, siatka pola)…' if self.ch_C.isChecked()
                                   else 'Rama C: włącz liczenie w sekcji „R5 C: rama Schura”.')
            return
        klucz, w = self._C
        G = int(round(len(w['siatka']) ** (1 / 3)))
        g = w['siatka'] * (S_EKR / max(w['r'], 1e-30))
        lin = self.cb_C_pole.currentIndex() == 1 and w.get('liniowe') is not None
        pole = w['liniowe'] if lin else w['pole']
        nrm = np.linalg.norm(pole, axis=1)
        dl = 0.9 * (2 * S_EKR / max(G - 1, 1)) * self.sp_C_strz.value() / max(float(nrm.max()), 1e-30)
        gn = np.linalg.norm(g, axis=1)
        rad = np.clip(np.where((nrm > 1e-30) & (gn > 1e-30), (pole * g).sum(1) / np.maximum(nrm * gn, 1e-30), 0.0), -1, 1)[:, None]
        styczny, wyplyw, wplyw = np.array([0.78, 0.62, 0.98]), np.array([1.0, 0.38, 0.3]), np.array([0.32, 0.62, 1.0])
        kk = np.c_[np.where(rad > 0, styczny + (wyplyw - styczny) * rad, styczny + (wplyw - styczny) * (-rad)), 0.4 + 0.55 * np.abs(rad)]
        napisy = []
        o_ = None
        if w['os'] is not None:                                             # kierunek własny rzeczywisty
            lr = float(w['lam_r'])
            kr = (1.0, 0.35, 0.3, 0.95) if lr > 0 else (0.35, 0.65, 1.0, 0.95)
            o_ = np.asarray(w['os']) * 1.3 * S_EKR
            napisy.append((o_ * 1.05, f'λ = {pl(lr, 4)} ({"wypycha" if lr > 0 else "ściąga"})', (230, 230, 230, 230)))
        pk = None
        if w['plaszczyzna'] is not None:                                    # płaszczyzna obrotu (para zespolona), jaśniej = dokąd
            th = np.linspace(0, 2 * np.pi, 97)
            th = th if w['kierunek'] >= 0 else th[::-1]
            pk = 0.85 * S_EKR * (np.cos(th)[:, None] * w['plaszczyzna'][:, 0] + np.sin(th)[:, None] * w['plaszczyzna'][:, 1])
            lp = w['lam_p']
            napisy.append((pk[len(pk) // 4] * 1.12, f'λ = {pl(lp.real, 4)} ± {pl(abs(lp.imag), 4)} i (obrót, jaśniej = dokąd)',
                           (215, 170, 255, 230)))
        QF = np.asarray(w['QF'])
        v_ = None
        if np.linalg.norm(QF) > 1e-30:                                      # skok cząsteczki: kierunek w ramie
            v_ = QF / np.linalg.norm(QF) * 1.4 * S_EKR
            napisy.append((v_ * 1.04, f'skok: {proc(w["skok_w_ramie"])} długości w ramie', (255, 190, 110, 230)))

        def budowa():
            el = LocalSections().strzalki(g, pole * dl, kk)
            if o_ is not None:
                el.odcinki(-o_, o_, kr, gruba=True)
            if pk is not None:
                al = np.linspace(0.15, 0.95, len(pk) - 1)
                el.odcinki(pk[:-1], pk[1:], np.c_[np.tile([0.78, 0.45, 1.0], (len(al), 1)), al], gruba=True)
            if v_ is not None:
                el.strzalki(np.zeros(3), v_[None], KOL_SILA + (1.0,), gruba=True)
            return el.kropki(np.zeros((1, 3)), KOL_WYBR + (1.0,))

        sc.lokalne((klucz, lin, round(float(self.sp_C_strz.value()), 3)), budowa)
        sc.odcinki('zero', None, None)
        osie = sc.osie(['oś ramy 1', 'oś ramy 2', 'oś ramy 3'] if self.ch_osie.isChecked() else None, dl=1.1 * S_EKR)
        sc.podpisy(napisy + osie)
        sc.kadruj(g, ('R5', 'stala'), kwantyl=100, zapas=2.0)
        wid, w0 = w['widmo'], w['widmo0']
        teraz = klucz == self._klucz_C()
        porown = '' if w0 is None else f' (w pierwszym checkpoincie {proc(w0["zesp"])}, Re λ > 0: {w0["dodatnie"]})'
        self._lb['R5'].setText(
            f'Cząsteczka {klucz[2]}, checkpoint {klucz[1] + 1} (krok {w["krok"]})' + ('' if teraz else ' — liczę bieżący…') +
            f'. Rama = płaszczyzna pary zespolonej i kierunek rzeczywisty: niezmiennicza dla J (resztka {w["reszta"]:.1e}); '
            f'pole dokładne poza ramą: mediana {proc(float(np.median(w["poza"])))}. Widmo J: {proc(wid["zesp"])} zespolonych '
            f'(losowa macierz ≈ {proc(projections.GINIBRE_ZESPOLONE)}){porown}, Re λ > 0: {wid["dodatnie"]}, Tr J/192 = {pl(wid["div"], 4)}, '
            f'odchylenie od normalności {pl(wid["henrici"], 2)}. Tokeny nie są punktami stałymi (F ≠ 0): to rama lokalnej deformacji, '
            f'pole = zmiana siły względem skoku samej cząsteczki; czerwone wypływa, niebieskie wpływa, fiolet krąży.')
