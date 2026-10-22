"""Bazy rzutu okienek (A, A_t, S, W, D, L, Fano) i etykiety klas (narzędzie, tło, łaty mieszane)."""
from __future__ import annotations

import numpy as np

from laser_app.compute import history as hist, projections
from laser_app.ui.projection_windows.shared import pl


class BasesMixin:
    """Bazy rzutu okienek (A, A_t, S, W, D, L, Fano) i etykiety klas (narzędzie, tło, łaty mieszane).
    Część ProjectionWindows (ui/projections_view.py); korzysta z jej atrybutów przez self."""

    def _baza_okienka(self, k):
        """(współrzędne położeń (K, 600, 3), sił (K, 600, 3), nazwy osi, indeks bazy w miernikach albo None, opis)."""
        d = self.dane
        if k == 'R1':
            if self.cb_A.currentIndex() == 1:
                return d.wsp(d.A_t, 'X'), d.wsp(d.A_t, 'F'), ['Δ narzędzie–tło (checkpoint)', 'reszta 1', 'reszta 2'], 2, 'A w każdym checkpoincie'
            return d.wsp(d.baza_A, 'X'), d.wsp(d.baza_A, 'F'), ['Δ narzędzie–tło', 'reszta chmury 1', 'reszta chmury 2'], 1, 'A wspólna'
        if k == 'R2':
            return d.wsp(d.baza_S, 'X'), d.wsp(d.baza_S, 'F'), ['siły 1', 'siły 2', 'siły 3'], 3, 'S'
        if k == 'R3':
            return d.wsp(d.baza_W, 'X'), d.wsp(d.baza_W, 'F'), ['wiązki 1', 'wiązki 2', 'wiązki 3'], 4, 'W'
        if k == 'R7':
            tr = projections.FANO[self.cb_F.currentIndex()]
            klucz = ('F7', tr)
            if klucz not in self._pam:
                self._pam[klucz] = (np.ascontiguousarray(d.wsp(d.baza_F7, 'X')[..., list(tr)]),
                                    np.ascontiguousarray(d.wsp(d.baza_F7, 'F')[..., list(tr)]))
            X3, F3 = self._pam[klucz]
            return X3, F3, [d.nazwy_F7[a] for a in tr], None, f'Fano {self.cb_F.currentIndex() + 1}'
        if k == 'R9':
            B, opis = self._baza_L()
            return d.wsp(B, 'X'), d.wsp(B, 'F'), ['LDA narzędzie–tło', 'oś 2', 'oś 3'], None, opis
        B, nazwy, b, opis = self._baza_D()
        return d.wsp(B, 'X'), d.wsp(B, 'F'), nazwy, b, opis

    def _grupa(self, i):
        """Grupa tokenów wykryta bez etykiet w checkpoincie i (TokenCut na h.Z[i], aktywacje MLP bloku 6), raz na checkpoint."""
        klucz = ('grupa', int(i))
        if klucz not in self._pam:
            self._pam[klucz] = projections.wykryj_grupe(np.asarray(self._h.Z[int(i)], np.float32))
        return self._pam[klucz]

    def _etykieta(self, i=None):
        """(narzędzie, mieszane) bool (600,) dla R9: z obrysu (udział pikseli ≥ próg) albo ze starej ramki GT bez `brzeg` skrajnych
        kolumn i wierszy; mieszane nie wchodzą ani do narzędzia, ani do tła."""
        d = self.dane
        if self.cb_L_etyk.currentIndex() == 2:
            G = self._grupa(self._indeksy()[3] if i is None else i)
            if G.sum() >= 2:
                return G, np.zeros(hist.TOK, bool)
        u = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
        if self.cb_L_etyk.currentIndex() == 0 and np.isfinite(u).all():
            narz = u >= float(self.sp_L_prog.value())
            if narz.sum() >= 2:
                return narz, (u > 0) & ~narz
        ramka = np.asarray(getattr(d, 'narz_ramka', d.narz), bool)
        rdz = projections.rdzen_narzedzia(ramka, int(self.sp_L_brzeg.value()))
        return rdz, ramka & ~rdz

    def _klucz_etyk(self):
        return (self.cb_L_etyk.currentIndex(), self.cb_L_os.currentIndex(), round(float(self.sp_L_prog.value()), 3), int(self.sp_L_brzeg.value()))

    def _etykieta_osi(self, kf):
        """Etykieta do dopasowania osi LDA: przy grupie wykrytej bez etykiet i „oś z etykiety” – obrys (udział ≥ próg),
        w pozostałych przypadkach ta sama co etykieta widoku."""
        if self.cb_L_etyk.currentIndex() == 2 and self.cb_L_os.currentIndex() == 0:
            u = np.asarray(getattr(self.dane, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
            if np.isfinite(u).all() and (u >= float(self.sp_L_prog.value())).sum() >= 2:
                narz = u >= float(self.sp_L_prog.value())
                return narz, (u > 0) & ~narz
        return self._etykieta(kf)

    def _opis_etyk(self, rdz, mies):
        d = self.dane
        if self.cb_L_etyk.currentIndex() == 2 and rdz.sum() >= 2 and not mies.any():
            u = np.asarray(getattr(d, 'udzial', np.full(hist.TOK, np.nan)), np.float32)
            s_ = f'grupa wykryta bez etykiet (TokenCut, latent MLP), {int(rdz.sum())} tokenów'
            if np.isfinite(u).all():
                z = projections.zgodnosc_z_obrysem(rdz, u)
                fp = lambda v: f'{v:.3f}'.replace('.', ',')
                s_ += (f'; sprawdzenie z obrysem (nieużytym do wykrycia): Dice {pl(z["dice"])}, precyzja {pl(z["precyzja"])}, '
                       f'czułość {pl(z["czulosc"])}, p przesunięć grupy = {fp(z["p"])}; żółte = narzędzie pominięte')
            return s_
        u = np.asarray(getattr(d, 'udzial', [np.nan]), np.float32)
        obrys = self.cb_L_etyk.currentIndex() == 0 and np.isfinite(u).all()
        zrodlo = str(getattr(d, 'etykieta', 'obrys')) if obrys else 'ramka GT'
        if obrys and (u >= float(self.sp_L_prog.value())).sum() < 2:
            zrodlo = 'ramka GT (przy tym progu < 2 łaty narzędzia)'
        elif self.cb_L_etyk.currentIndex() == 0 and not obrys:
            zrodlo = 'ramka GT (brak obrysu dla tej klatki)'
        return f'{zrodlo}, {int(rdz.sum())} tokenów narzędzia, {int(mies.sum())} mieszanych (żółte, pominięte)'

    def _baza_L(self):
        """Baza R9 (LDA ze skurczem na puli checkpointów), raz na parametry i checkpoint dopasowania."""
        d = self.dane
        _, _, _, ir = self._indeksy()
        kf = ir if self.cb_L_kiedy.currentIndex() == 1 else d.K - 1
        N = int(self.sp_L_N.value())
        klucz = ('L', kf, N, round(float(self.sp_L_lam.value()), 3), self.cb_L_osie.currentIndex(), self._klucz_etyk())
        if klucz not in self._pam:
            ks = np.arange(max(0, kf - N + 1), kf + 1)
            if len(ks) > 200:
                ks = np.unique(np.concatenate([np.linspace(ks[0], ks[-1], 200).astype(int), [kf]]))   # pula: najwyżej 200 checkpointów
            Xs = np.asarray(d.X[ks], np.float64).reshape(-1, projections.D)
            rdz, mies = self._etykieta_osi(kf)
            keep = np.tile(~mies, len(ks))
            B, cosd = projections.baza_L(Xs[keep], np.tile(rdz, len(ks))[keep], float(self.sp_L_lam.value()), self.cb_L_osie.currentIndex())
            self._pam[klucz] = (B.astype(np.float32), f'L: LDA λ={pl(float(self.sp_L_lam.value()))}, pula {len(ks)} ckpt do {kf + 1}, '
                                                       f'cos(w, Δ) = {pl(cosd)}')
        return self._pam[klucz]

    def _baza_D(self):
        d = self.dane
        o = self.cb_D.currentIndex()
        if o == 0:
            return d.baza_D, ['ruch 1', 'ruch 2', 'ruch 3'], 5, 'D: PCA przemieszczeń wszystkich'
        if o == 3:
            return d.baza_A, ['Δ narzędzie–tło', 'reszta chmury 1', 'reszta chmury 2'], 1, 'A wspólna'
        if o == 1:
            pok, _, _ = self._maski()
            idx = np.flatnonzero(pok)
            klucz = ('D1', tuple(idx.tolist()), self.cb_srodek.currentIndex())
            if klucz not in self._pam:
                S = np.zeros((projections.D, projections.D))
                tlo = d.srodek_tla() if self.cb_srodek.currentIndex() == 1 else None
                for a in range(0, d.K, 64):
                    blok = np.asarray(d.X[a:min(a + 65, d.K)], np.float64)[:, idx]
                    if tlo is not None:
                        blok = blok - tlo[a:a + len(blok)][:, None]
                    dd = np.diff(blok, axis=0).reshape(-1, projections.D)
                    S += dd.T @ dd
                self._pam[klucz] = projections.glowne(S, 3)
            return self._pam[klucz], ['ruch pokazanych 1', 'ruch pokazanych 2', 'ruch pokazanych 3'], None, 'D: PCA przemieszczeń pokazanych'
        j = self._wybrane()[0]
        klucz = ('Dr', j, self.cb_srodek.currentIndex())
        if klucz not in self._pam:
            self._pam[klucz] = projections.rama_toru(self._tor_wzgl(j))
        return self._pam[klucz], [f'start → koniec ({j})', 'odchylenie 1', 'odchylenie 2'], None, f'D: rama cząsteczki {j}'
