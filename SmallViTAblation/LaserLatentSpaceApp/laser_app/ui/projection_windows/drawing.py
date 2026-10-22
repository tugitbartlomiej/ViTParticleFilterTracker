"""Rysowanie okienek R1–R7 (cząsteczki, ślady, siły, wiązki, tory zerowe) i ich opisy (wierność, separacja, ruch)."""
from __future__ import annotations

import traceback

import numpy as np

from laser_app.compute import history as hist, projections
from laser_app.ui.projection_windows.scene import LocalSections
from laser_app.ui.projection_windows.shared import KOL_PRZECIW, KOL_ZGODNA, pl, proc


class DrawingMixin:
    """Rysowanie okienek R1–R7 (cząsteczki, ślady, siły, wiązki, tory zerowe) i ich opisy (wierność, separacja, ruch).
    Część ProjectionWindows (ui/projections_view.py); korzysta z jej atrybutów przez self."""

    # ------------------------------------------------------------------ rysowanie
    def _rysuj(self, k):
        if self._h is None or (self.dane is None and k != 'R0'):
            return
        try:
            if k == 'R0':
                self._rysuj_zdjecie()
            elif k == 'R10':
                self._rysuj_lda()
            elif k == 'R11':
                self._rysuj_miary()
            elif k == 'R12':
                self._rysuj_knn()
            elif k == 'R8':
                self._rysuj_panel()
            elif k == 'R4':
                self._rysuj_B()
            elif k == 'R5':
                self._rysuj_C()
            else:
                self._rysuj_globalny(k)
        except Exception as e:                                           # noqa: BLE001 (błąd jednego okienka nie zatrzymuje zakładki)
            traceback.print_exc()
            if k in self._lb:
                self._lb[k].setText(f'Błąd rysowania: {type(e).__name__}: {e}')

    def _rysuj_globalny(self, k):
        """Okienka R1–R3, R6, R7: historia w bazie okienka na karcie (raz na bazę i opcje), w klatce tylko czas, maska i napisy."""
        d, sc = self.dane, self._sceny[k]
        Cx, Cf, nazwy, b, opis_bazy = self._baza_okienka(k)
        E = self._ekran(Cx)
        i, j, u, ir = self._indeksy()
        P = E[i] * (1 - u) + E[j] * u                                    # tylko kamera, klik i kadr: cząsteczki rysuje karta
        pok, wyb, sl = self._maski()
        wybrane = self._wybrane()
        s = self._skala_k()
        kg = (id(Cx), id(Cf), self.cb_srodek.currentIndex(), self.cb_skala.currentIndex(), int(self.sp_wyg.value()))
        if kg != sc._klucz_g:
            rho = np.clip(np.linalg.norm(Cf, axis=2) / np.maximum(d.FN, 1e-30), 0, 1)   # widoczna część długości siły
            sc.wgraj(kg, E, self._wygladz(E), Cf * s[:, None, None], rho)
        o_ = self.cb_strz.currentIndex()
        M = np.zeros((hist.TOK, 4), np.float32)
        M[:, 0] = np.asarray(self._etykieta()[0] if k == 'R9' else d.narz, bool)
        M[~pok & ~wyb, 0] = 2.0                                           # ukryte; wybrane widać zawsze
        M[wyb, 1] = 1.0
        if k == 'R9':
            M[self._etykieta()[1], 1] = 1.0                                  # tokeny mieszane (albo pominięty brzeg ramki) na żółto
            u_ = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
            if self.cb_L_etyk.currentIndex() == 2 and np.isfinite(u_).all():
                M[(u_ >= 0.25) & ~self._etykieta()[0], 1] = 1.0              # narzędzie z obrysu pominięte przez grupę: na żółto
        if wybrane:
            M[wybrane[0], 1] = 2.0
        M[:, 2] = sl
        M[:, 3] = (pok if o_ == 1 else wyb) if o_ else False
        sc.maska(M)
        L = int(self.sp_slad.value())
        sc.g.opcje.update(czastki=True, tlo=True, rozm=float(self.sp_rozm.value()), L=L, od0=L == 0, strzalki=o_ > 0,
                          strz=float(self.sp_strz.value()), rho=self.ch_rho.isChecked())
        if k == 'R3' and self.ch_W.isChecked() and wybrane:
            jt, sk = wybrane[0], float(s[ir] * self.sp_strz.value())
            sc.lokalne(('W', ir, jt, round(float(self.sp_B_prog.value()), 3), sk, self.ch_rho.isChecked()),
                       lambda: self._wiazki_lokalne(d.baza_W, ir, jt, sk), kotwica=jt)
        else:
            sc.lokalne(None, None)
        sc.g.ustaw_czas(self.t)
        self._ostatnie[k] = (P, pok | wyb)
        if k == 'R6':
            self._tory_zerowe(sc, E, i, u)
        else:
            sc.odcinki('zero', None, None)
        sc.podpisy(sc.osie(nazwy if self.ch_osie.isChecked() else None))
        # kadr z całej historii (co ~50. checkpoint): kamera mieści każdy checkpoint i nie skacze w czasie
        sc.kadruj(E[::max(1, d.K // 50)][:, pok | wyb].reshape(-1, 3), (k, id(Cx), self.cb_skala.currentIndex(), self.cb_srodek.currentIndex()))
        if self.ch_kamera.isChecked() and wybrane:
            sc.srodek(P[wybrane[0]])
        self._lb[k].setText(self._opis(k, b, ir, opis_bazy, pok))

    def _kolory_wiazek(self, U, W3, cos_):
        """Kolor wiązki = znak cos(u_c, F̂) w 192D (pomarańcz pcha wzdłuż siły, błękit przeciw), krycie = ρ = |Pu|/|u|."""
        kk = np.empty((len(U), 4), np.float32)
        kk[:, :3] = np.where(np.asarray(cos_)[:, None] > 0, KOL_ZGODNA, KOL_PRZECIW)
        rho = np.linalg.norm(W3, axis=1) / np.maximum(np.linalg.norm(U, axis=1), 1e-30)
        kk[:, 3] = 0.15 + 0.85 * np.clip(rho, 0, 1) if self.ch_rho.isChecked() else 0.85
        return kk

    def _wiazki_lokalne(self, B, ir, jt, s):
        """R3: wiązki cząsteczki jt w bazie B jako elementy lokalne zakotwiczone w cząsteczce: sieć silna i gruba szara reszta."""
        w = self._rama_B(ir, jt)
        u, fr, silne, reszta = w['u'], w['fr'], w['silne'], w['reszta']
        W3 = u[silne] @ B
        el = LocalSections().strzalki(np.zeros(3), W3 * s, self._kolory_wiazek(u[silne], W3, fr['cos'][silne]))
        return el.strzalki(np.zeros(3), (reszta @ B * s)[None], (0.62, 0.62, 0.62, 0.9), gruba=True)

    def _wklady(self, ir, jt):
        """Wkłady 768 wiązek cząsteczki jt w checkpoincie ir (768, 192); pamięć na kilka ostatnich (każdy ma 1,2 MB)."""
        klucz = (ir, jt)
        u = self._u.get(klucz)
        if u is None:
            z = np.asarray(self._h.Z[ir], np.float64)
            u = self._u[klucz] = projections.wklady(z[jt], z.mean(0), self.dane.V[ir])
            while len(self._u) > 6:
                self._u.pop(next(iter(self._u)))
        return u

    def _zerowe(self, j):
        """Tory zerowe cząsteczki j (n, K, 192) względem wybranego środka (raz na wybór)."""
        klucz = ('zero', j, self.sp_D_zero.value(), self.cb_srodek.currentIndex())
        if klucz not in self._pam:
            self._pam[klucz] = projections.tory_zerowe(self._tor_wzgl(j), self.sp_D_zero.value(), ziarno=j)
        return self._pam[klucz]

    def _tory_zerowe(self, sc, E, i, u):
        """R6: szare tory zerowe pierwszej wybranej cząsteczki do bieżącego checkpointu."""
        wybrane = self._wybrane()
        if not self.ch_D_zero.isChecked() or not wybrane:
            sc.odcinki('zero', None, None)
            return
        j = wybrane[0]
        o = self.cb_D.currentIndex()
        B = None if o == 2 else self._baza_D()[0]
        klucz = ('zero3', j, self.sp_D_zero.value(), self.cb_srodek.currentIndex(), o, None if B is None else B.tobytes())
        if klucz not in self._pam:
            Z0 = self._zerowe(j)                                            # (n, K, 192)
            if B is None:                                                   # każdy tor zerowy we własnej ramie (ta sama przewaga)
                Z3 = np.stack([(z_ - z_[0]) @ projections.rama_toru(z_) for z_ in Z0])
            else:
                Z3 = Z0 @ B
            self._pam[klucz] = Z3.astype(np.float32)
        Z3 = self._wygladz(self._pam[klucz], os_=1)                         # to samo wygładzanie co ślady cząsteczek
        s = self._skala_k()
        T3 = (E[0, j] + Z3 * s[None, :, None]) if B is None else Z3 * s[None, :, None]
        T3 = T3[:, :i + 2]
        if T3.shape[1] > i + 1:                                             # ostatni odcinek do chwili t
            T3[:, i + 1] = T3[:, i] * (1 - u) + T3[:, i + 1] * u
        if T3.shape[1] < 2:
            sc.odcinki('zero', None, None)
            return
        seg = np.stack([T3[:, :-1], T3[:, 1:]], 2).reshape(-1, 3)
        sc.odcinki('zero', seg, np.tile([0.62, 0.62, 0.62, 0.45], (len(seg), 1)))

    def _opis(self, k, b, i, opis_bazy, pok):
        d = self.dane
        kr = int(d.kroki[i])
        if b is not None:
            s1 = (f'{opis_bazy}, checkpoint {i + 1}/{d.K} (krok {kr}). Widać: rozrzut chmury {proc(d.WX[i, b])} '
                  f'(najlepsze 3D {proc(d.own3[i])}), energia sił {proc(d.WF[i, b])}, energia wiązek {proc(d.WW[i, b])}, '
                  f'długość Δ {proc(d.WS[i, b])}, ruch od poprzedniego {proc(d.WD[i, b])}.')
        elif k == 'R7':
            tr = projections.FANO[self.cb_F.currentIndex()]
            os_ = '; '.join(f'{d.nazwy_F7[a]}: rozrzut {proc(d.E7[i, 0, a])}, siły {proc(d.E7[i, 1, a])}, wiązki {proc(d.E7[i, 2, a])}'
                            for a in tr)
            s1 = f'{opis_bazy}, checkpoint {i + 1}/{d.K} (krok {kr}). Osie: {os_}.'
        else:
            s1 = f'{opis_bazy}, checkpoint {i + 1}/{d.K} (krok {kr}).'
        if k == 'R6':
            s1 += ' ' + self._opis_ruchu(pok)
        if k == 'R9':
            s1 += ' ' + self._opis_separacji(i)
        sep = ''
        if np.isfinite(d.sep[i]):
            sep = (f' Separacja w 192D: |Δ| = {pl(float(d.sep[i]))} R; losowe etykiety {pl(float(d.sep0[i, 0]))} R '
                   f'(95 %: {pl(float(d.sep0[i, 1]))} R).')
        return s1 + ' Losowe 3D: 1,6 % energii, 12,5 % długości.' + sep

    def _opis_separacji_podst(self, i, B):
        """Podstawowe miary R9 (bez zera z refitem i walidacji blokowanej): szybkie, także podczas odtwarzania."""
        d = self.dane
        rdz, mies = self._etykieta(i); keep = ~mies
        X = np.asarray(d.X[i], np.float64)[keep]; m = rdz[keep]
        kk = projections.k_sasiadow(m.sum())
        m3, z = projections.separacja_gpu(X @ B, m, int(self.sp_L_perm.value()), k=kk)
        m192, _ = projections.separacja_gpu(X, m, 0, k=kk)
        f = lambda n_: f'{pl(m3[n_])}' + ('' if z is None else f' (losowe etykiety {pl(z[n_])})')
        return (f'Etykieta: {self._opis_etyk(rdz, mies)}. Separacja w tym rzucie: Fisher {f("fisher")}, sylwetka {f("sil")}, '
                f'trafienia {kk}NN {f("hit")}, margines {f("marg")} odstępu tła; '
                f'w 192D: Fisher {pl(m192["fisher"])}, sylwetka {pl(m192["sil"])}, kNN {pl(m192["hit"])}.')

    def _opis_separacji(self, i):
        """R9: miary separacji rdzenia narzędzia od tła w tym rzucie i w 192D, z zerem dla losowych etykiet (raz na checkpoint)."""
        d = self.dane
        B, _ = self._baza_L()
        ciezkie = (self.ch_L_refit.isChecked() and self.sp_L_perm.value() > 0) or self.ch_L_cv.isChecked() or self.ch_L_przestrz.isChecked()
        gra = self._z.bt_play.isChecked()
        if ciezkie and gra and ('Lm', i, 'ciezkie') not in self._pam:
            klucz_l = ('Lm', i, B.tobytes(), self._klucz_etyk(), int(self.sp_L_perm.value()), False, False, False, 0, 0, 0, 0)
            if klucz_l not in self._pam:
                self._pam[klucz_l] = self._opis_separacji_podst(i, B)
            return self._pam[klucz_l] + ' Zero przestrzenne, zero z refitem i walidacja blokowana liczą się na pauzie.'
        klucz = ('Lm', i, B.tobytes(), self._klucz_etyk(), int(self.sp_L_perm.value()), self.ch_L_refit.isChecked(),
                 self.ch_L_cv.isChecked(), self.ch_L_przestrz.isChecked(), int(self.sp_L_bl.value()), int(self.sp_L_N.value()), round(float(self.sp_L_lam.value()), 3),
                 self.cb_L_osie.currentIndex())
        if klucz not in self._pam:
            rdz, mies = self._etykieta(i); keep = ~mies
            X = np.asarray(d.X[i], np.float64)[keep]; m = rdz[keep]
            kk = projections.k_sasiadow(m.sum())
            tekst = self._opis_separacji_podst(i, B)
            lam, osie = float(self.sp_L_lam.value()), self.cb_L_osie.currentIndex()
            if self.ch_L_przestrz.isChecked():
                wyk = rdz | mies
                u = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
                if np.isfinite(u).all():
                    wyk |= u > 0                                          # żadnego piksela narzędzia w położeniach tła
                pr, zp, n = projections.zero_przestrzenne_gpu(np.asarray(d.X[i], np.float32), rdz, wyk, keep, lam, osie, kk)
                fp = lambda v: f'{v:.3f}'.replace('.', ',')
                if zp:
                    tekst += (f' Zero przestrzenne ({n} położeń tej maski w tle, LDA od nowa na tym checkpoincie): narzędzie Fisher '
                              f'{pl(pr["fisher"])} (95 % tła {pl(zp["fisher"][0])}, p = {fp(zp["fisher"][1])}), sylwetka {pl(pr["sil"])} '
                              f'({pl(zp["sil"][0])}, p = {fp(zp["sil"][1])}), {kk}NN {pl(pr["hit"])} ({pl(zp["hit"][0])}, p = {fp(zp["hit"][1])}).')
                else:
                    tekst += f' Zero przestrzenne: za mało położeń maski w tle ({n}).'
            if self.ch_L_refit.isChecked() and self.sp_L_perm.value() > 0:
                zr = projections.zero_z_refitem_gpu(X, m, lam, osie, int(self.sp_L_perm.value()), k=kk)
                tekst += (f' Zero z ponownym dopasowaniem osi ({int(self.sp_L_perm.value())} perm.): sylwetka {pl(zr["sil"])}, '
                          f'kNN {pl(zr["hit"])}, margines {pl(zr["marg"])}, Fisher {pl(zr["fisher"])}.')
            if self.ch_L_cv.isChecked():
                ks = np.arange(max(0, i - int(self.sp_L_N.value()) + 1), i + 1)
                if len(ks) >= 4:
                    cv, _ = projections.cv_blokowane_gpu(np.asarray(d.X[ks], np.float32)[:, keep], m, lam, osie, int(self.sp_L_bl.value()), k=kk)
                    tekst += (f' Walidacja blokowana po checkpointach ({int(self.sp_L_bl.value())} bloków z puli {len(ks)}): sylwetka '
                              f'{pl(cv["sil"])}, kNN {pl(cv["hit"])}, margines {pl(cv["marg"])}, Fisher {pl(cv["fisher"])}.')
                else:
                    tekst += ' Walidacja blokowana: pula za krótka (zwiększ „Pulę dopasowania”).'
            self._pam[klucz] = tekst
        return self._pam[klucz]

    def _opis_ruchu(self, pok):
        """R6: ile ruchu pokazanych cząsteczek widać w tym rzucie i porównanie toru wybranej z torami zerowymi (192D)."""
        d = self.dane
        B, _, _, _ = self._baza_D()
        idx = np.flatnonzero(pok)
        klucz = ('ruch', tuple(idx.tolist()), B.tobytes(), self.cb_srodek.currentIndex())
        if klucz not in self._pam:
            if len(idx) > 120:
                idx = idx[np.linspace(0, len(idx) - 1, 120).astype(int)]    # próbka: miara, nie rysunek
            tory = np.asarray(d.X[:, idx], np.float64)                      # (K, n, 192), jeden odczyt
            if self.cb_srodek.currentIndex() == 1:
                tory = tory - d.srodek_tla()[:, None]
            self._pam[klucz] = projections.udzial_ruchu(tory.transpose(1, 0, 2), B)
        en, dl = self._pam[klucz]
        s = f'Ruch pokazanych w tym rzucie: {proc(en)} energii przemieszczeń, {proc(dl)} długości drogi.'
        wybrane = self._wybrane()
        if wybrane:
            j = wybrane[0]
            klucz = ('ruchj', j, self.sp_D_zero.value(), self.cb_srodek.currentIndex(), self.ch_D_zero.isChecked())
            if klucz not in self._pam:
                tor = self._tor_wzgl(j)
                R = float(d.R[-1])
                droga = float(np.linalg.norm(np.diff(tor, axis=0), axis=1).sum()) / R
                netto = float(np.linalg.norm(tor[-1] - tor[0])) / R
                p1, p3 = projections.udzialy_pca(tor)
                t = (f' Cząsteczka {j}: droga {pl(droga)} R, przesunięcie netto {pl(netto)} R; kąt skrętu (mediana cos) '
                     f'{pl(projections.kat_skretu(tor))}; PCA własnego toru PC1 {proc(p1)}, PC1–3 {proc(p3)}')
                if self.ch_D_zero.isChecked():
                    Z0 = self._zerowe(j)
                    u0 = np.array([projections.udzialy_pca(z_) for z_ in Z0])
                    t += (f'; tory zerowe: PC1 {proc(u0[:, 0].mean())}, PC1–3 {proc(u0[:, 1].mean())}, kąt skrętu '
                          f'{pl(float(np.median([projections.kat_skretu(z_) for z_ in Z0])))}')
                self._pam[klucz] = t + '.'
            s += self._pam[klucz]
        return s

    def _ekran(self, C):
        """Współrzędne ekranu (K, 600, 3) dla współrzędnych C (K, 600, 3) w 192D: środek i skala według opcji (raz na opcje)."""
        klucz = ('ekr', id(C), self.cb_srodek.currentIndex(), self.cb_skala.currentIndex())
        if klucz not in self._pam:
            if len(self._pam) > 48:
                self._pam.clear()
            cen = 0.0
            if self.cb_srodek.currentIndex() == 1:
                m = ~np.asarray(self.dane.narz, bool)
                cen = C[:, m].mean(1, keepdims=True) if m.any() else 0.0
            self._pam[klucz] = (C, ((C - cen) * self._skala_k()[:, None, None]).astype(np.float32))
        return self._pam[klucz][1]

    def _wygladz(self, E, os_=0):
        """Średnia krocząca z ostatnich `Wygładzanie śladu` checkpointów (tylko przeszłość) wzdłuż osi czasu os_ (raz na tablicę)."""
        w = int(self.sp_wyg.value())
        if w <= 1:
            return E
        klucz = ('wyg', id(E), w, os_)
        if klucz not in self._pam:
            K = E.shape[os_]
            ksz = [1] * E.ndim; ksz[os_] = -1
            c0 = np.concatenate([np.zeros_like(np.take(E, [0], axis=os_), dtype=np.float64),
                                 np.cumsum(np.asarray(E, np.float64), axis=os_)], axis=os_)
            hi = np.arange(1, K + 1); lo = np.maximum(hi - w, 0)
            S = (np.take(c0, hi, axis=os_) - np.take(c0, lo, axis=os_)) / (hi - lo).reshape(ksz)
            self._pam[klucz] = (E, S.astype(np.float32))
        return self._pam[klucz][1]

    def _tor_wzgl(self, j):
        """Tor cząsteczki j (K, 192) względem wybranego środka (chmury albo tła)."""
        tor = self.dane.tor(j)
        return tor - self.dane.srodek_tla() if self.cb_srodek.currentIndex() == 1 else tor
