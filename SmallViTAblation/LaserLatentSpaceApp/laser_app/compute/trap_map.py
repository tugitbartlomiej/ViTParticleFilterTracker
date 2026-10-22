"""Mapa pułapek optycznych wszystkich 600 cząsteczek zdjęcia przez całą historię uczenia (okienko 14 i nakładka na sferze).

Dla każdego checkpointu i KAŻDEJ cząsteczki j to samo co focal_trap.py dla jednej, ale bez siatki pola G³ (ta zostaje dla
cząsteczek oglądanych z bliska w okienku 11):
- skok f(0) w bloku 6 (pchnięcie 768 wiązek, względem środka chmury), komórka r_kom (połowa odległości do sąsiada w rzucie 3D);
- pułapka: zero pola f(u) = 0 w sześcianie komórki (metoda Newtona dla 600 cząsteczek naraz na GPU, focal_trap.newton_wsadowo:
  jakobian w float32, reszta f w float64), zb = czy zero jest w komórce;
- charakter pułapki z wartości własnych λ jakobianu 3 × 3 w zerze (gdy go nie ma: w koloidzie): ściąga (wszystkie Re λ < 0),
  wypycha (wszystkie > 0), siodło (znaki mieszane), wir = para zespolona (obrót wokół osi);
- liniowe oszacowanie zera u = −J₀⁻¹ f(0) i |f| w zerze względem |f(0)|.
Wynik: `<katalog historii>/mapa_pulapek_z<zasięg·100>.npz` (tablice (K, 600, …)); liczy się raz, potem z pliku.
Na RTX 3070: ok. 30–40 ms na checkpoint dla wszystkich 600 cząsteczek (wczytanie wag i przejście sieci są większą częścią).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from laser_app.compute import focal_trap

WERSJA = 1
TYPY = ('brak zera w komórce', 'ściąga', 'siodło', 'wypycha')
KOLORY_TYPOW = np.array([[0.45, 0.48, 0.55, 0.55],                    # brak zera: szary
                         [0.30, 0.95, 0.45, 1.0],                     # ściąga: zielony
                         [1.00, 0.78, 0.25, 1.0],                     # siodło: żółty
                         [1.00, 0.30, 0.30, 1.0]], np.float32)        # wypycha: czerwony
KOLOR_WIRU = np.array([0.35, 0.85, 1.0], np.float32)                   # domieszka błękitu: wir (λ zespolone)


def plik(h, zasieg=1.0):
    return Path(h.katalog) / f'mapa_pulapek_z{int(round(zasieg * 100)):03d}.npz'


def wczytaj(h, zasieg=1.0):
    p = plik(h, zasieg)
    if not p.exists():
        return None
    d = dict(np.load(p, allow_pickle=False))
    if int(d.get('wersja', 0)) != WERSJA or len(d['kroki']) != len(h.kroki) or np.any(d['kroki'] != np.asarray(h.kroki)):
        return None
    return d


def charakter(lam, zb):
    """Typ pułapki (K, N) int8: 0 brak zera w komórce, 1 ściąga, 2 siodło, 3 wypycha; wir (K, N) bool = para zespolona λ."""
    re = np.real(lam); im = np.imag(lam)
    typ = np.where((re < 0).all(-1), 1, np.where((re > 0).all(-1), 3, 2)).astype(np.int8)
    typ[~np.asarray(zb, bool)] = 0
    wir = (np.abs(im) > 1e-9 * np.maximum(np.abs(lam), 1e-30)).any(-1)
    return typ, wir


def policz(h, zasieg=1.0, postep=None, przerwij=None, urzadzenie='cuda', dokladnosc='float32'):
    """Pułapki wszystkich cząsteczek we wszystkich checkpointach (z pliku, jeśli jest). postep(i, n, opis); przerwij() → wyjątek
    focal_trap.Przerwano. dokladnosc: 'float32' (jakobian w float32, reszta f w float64) albo 'float64'."""
    import torch
    d = wczytaj(h, zasieg)
    if d is not None:
        return d
    dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
    dt = torch.float64 if dokladnosc == 'float64' else torch.float32
    kroki = [int(k) for k in h.kroki]; K = len(kroki)
    n = int(np.asarray(h.pos).shape[1])
    T = dict(ogn=np.zeros((K, n, 3), np.float32), zb=np.zeros((K, n), bool), skok=np.zeros((K, n, 3), np.float32),
             lam=np.zeros((K, n, 3), np.complex64), lam0=np.zeros((K, n, 3), np.complex64), ogn_lin=np.zeros((K, n, 3), np.float32),
             r_kom=np.zeros((K, n), np.float32), fogn=np.zeros((K, n), np.float32), sasiad=np.zeros((K, n), np.int16),
             iteracje=np.zeros(K, np.int16), blad_pos=np.zeros(K, np.float32))
    t0 = time.time()
    tk = torch.arange(n, device=dev)
    for i, k, kt in focal_trap.checkpointy(h, dev, przerwij, dt):
        par, xs, zbar, pc = kt['par'], kt['xs'], kt['zbar'], kt['pc']
        P3, PV = par.p64['P3'], par.p64['PV']
        with torch.no_grad():
            rk, sas, Li = focal_trap.komorki(pc, tk, float(zasieg))
            u, zb, f0, _, it = focal_trap.newton_wsadowo(kt, tk, Li)
            X = xs + u @ P3
            fu = (par.jasnosci(X) - zbar) @ PV.T
            J = par.jakobian(X); J0 = par.jakobian(xs)
            ulin = -focal_trap.rozwiaz3(J0, f0)
        Jn, J0n = J.cpu().numpy(), J0.cpu().numpy()
        for nazwa, M in (('lam', Jn), ('lam0', J0n)):
            ev = np.linalg.eigvals(M)                                              # (n, 3) wszystkie cząsteczki naraz
            T[nazwa][i] = np.take_along_axis(ev, np.argsort(ev.real, axis=1), axis=1)
        pcn = pc.cpu().numpy()
        T['ogn'][i] = pcn + u.cpu().numpy(); T['zb'][i] = zb.cpu().numpy(); T['skok'][i] = f0.cpu().numpy()
        T['ogn_lin'][i] = pcn + ulin.cpu().numpy(); T['r_kom'][i] = rk.cpu().numpy(); T['sasiad'][i] = sas.cpu().numpy()
        T['fogn'][i] = (fu.norm(dim=1) / f0.norm(dim=1).clamp_min(1e-30)).cpu().numpy()
        T['iteracje'][i] = it; T['blad_pos'][i] = kt['blad_pos']
        if postep is not None:
            postep(i + 1, K, f'mapa pułapek 600 cząsteczek: checkpoint {i + 1}/{K} (krok {k}), {time.time() - t0:.0f} s')
    wynik = dict(wersja=WERSJA, zasieg=float(zasieg), kroki=np.asarray(kroki, np.int64), dokladnosc=dokladnosc, **T)
    p = plik(h, zasieg)
    tmp = p.with_name(p.stem + '_tmp.npz')
    np.savez(tmp, **wynik)
    tmp.replace(p)
    typ, wir = charakter(T['lam'], T['zb'])
    p.with_suffix('.json').write_text(json.dumps(dict(K=K, n=n, zasieg=zasieg, czas_s=round(time.time() - t0), dokladnosc=dokladnosc,
                                                      zero_w_komorce=float(T['zb'].mean()),
                                                      udzial_typow={TYPY[t]: float((typ == t).mean()) for t in range(4)},
                                                      wir=float(wir.mean()), blad_pos_maks=float(T['blad_pos'].max())),
                                                 ensure_ascii=False, indent=1), encoding='utf-8')
    return wynik


TRYBY_KOLORU = ('kształt pola w cząsteczce (λ₀): ściąga / siodło / wypycha, błękit = wir',
                'pułapka w komórce (λ w zerze pola; szare = brak zera w komórce)',
                'siła skoku |f(0)| w bloku 6',
                'odległość pułapki od cząsteczki / r_kom (szare = brak zera)')


def kolory(d, k, tryb=0, pos=None):
    """Kolory (n, 4) cząsteczek w checkpoincie k wg TRYBY_KOLORU; pos = położenia koloidów (n, 3) dla trybu 3."""
    if tryb in (0, 1) or (tryb == 3 and pos is None):
        lam, zb = (d['lam0'][k], np.ones(d['zb'][k].shape, bool)) if tryb == 0 else (d['lam'][k], d['zb'][k])
        typ, wir = charakter(lam[None], zb[None])
        typ = typ[0]; wir = wir[0] & (typ > 0)
        c = KOLORY_TYPOW[typ].copy()
        c[wir, :3] = 0.55 * c[wir, :3] + 0.45 * KOLOR_WIRU
        return c
    if tryb == 2:
        v = np.linalg.norm(d['skok'][k], axis=1); v = v / max(float(np.percentile(v, 95)), 1e-30)
    else:
        v = np.linalg.norm(d['ogn'][k] - np.asarray(pos, np.float32), axis=1) / np.maximum(d['r_kom'][k], 1e-30)
    import pyqtgraph as pg
    c = pg.colormap.get('plasma').map(np.clip(v, 0, 1), mode='float').astype(np.float32)
    if c.shape[1] == 3:
        c = np.concatenate([c, np.ones((len(c), 1), np.float32)], 1)
    if tryb == 3:
        c[~d['zb'][k]] = KOLORY_TYPOW[0]
    return c


def maska_filtru(d, k, filtr):
    """Które cząsteczki wyróżnić: 0 wszystkie, 1 siodła (λ₀), 2 ściągające (λ₀), 3 wypychające (λ₀), 4 z wirem (λ₀), 5 z zerem w komórce."""
    n = d['zb'].shape[1]
    if filtr == 0:
        return np.ones(n, bool)
    if filtr == 5:
        return np.asarray(d['zb'][k], bool)
    typ, wir = charakter(d['lam0'][k][None], np.ones((1, n), bool))
    return wir[0] if filtr == 4 else typ[0] == {1: 2, 2: 1, 3: 3}[filtr]


def statystyki(d, k):
    """Opis checkpointu k: kształt pola w cząsteczkach (λ₀), zera pola w komórkach, wiry, iteracje Newtona."""
    n = d['zb'].shape[1]
    typ0, wir0 = charakter(d['lam0'][k][None], np.ones((1, n), bool))
    typ0, wir0 = typ0[0], wir0[0]
    zb = int(d['zb'][k].sum())
    return (f'pole w cząsteczce: ściąga {int((typ0 == 1).sum())}, siodło {int((typ0 == 2).sum())}, wypycha {int((typ0 == 3).sum())}, '
            f'z wirem {int(wir0.sum())} (z {n});  zero pola w komórce: {zb} cząsteczek;  Newton {int(d["iteracje"][k])} iteracji')


try:
    import pyqtgraph.opengl as _gl

    class TrapMapGL:
        """Rysunek mapy w jednym widoku GL: kropki cząsteczek (kolor wg trybu), linie cząsteczka → pułapka (lupa) i wektory skoku.
        Dwa egzemplarze: okienko 14 i nakładka na dużej sferze (element GL należy do jednego widoku)."""

        def __init__(self):
            self.kropki = _gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=6, pxMode=True)
            self.pulapki = _gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=1.5, antialias=True)
            self.skoki = _gl.GLLinePlotItem(pos=np.zeros((2, 3), np.float32), mode='lines', width=1, antialias=True)
            self.wybrana = _gl.GLScatterPlotItem(pos=np.zeros((1, 3), np.float32), size=16, color=(1, 1, 1, 0.9), pxMode=True)
            self.elementy = (self.skoki, self.pulapki, self.kropki, self.wybrana)
            for e in self.elementy:
                e.setGLOptions('translucent')
            self.wybrana.setDepthValue(3)

        def dodaj_do(self, widok):
            for e in self.elementy:
                widok.addItem(e)

        def widoczna(self, on):
            for e in self.elementy:
                e.setVisible(bool(on))

        def ustaw(self, P, kol, rozm=6.0, Q=None, kol_q=None, S=None, wybrana=None):
            """P (n, 3) kropki, kol (n, 4); Q (m, 2, 3) odcinki do pułapek z kolorami kol_q (m, 4); S (n, 2, 3) wektory skoku."""
            self.kropki.setData(pos=np.asarray(P, np.float32), color=np.asarray(kol, np.float32), size=float(rozm))
            if Q is not None and len(Q):
                self.pulapki.setData(pos=np.asarray(Q, np.float32).reshape(-1, 3), color=np.repeat(np.asarray(kol_q, np.float32), 2, axis=0))
                self.pulapki.setVisible(True)
            else:
                self.pulapki.setVisible(False)
            if S is not None and len(S):
                self.skoki.setData(pos=np.asarray(S, np.float32).reshape(-1, 3), color=(1.0, 1.0, 1.0, 0.35))
                self.skoki.setVisible(True)
            else:
                self.skoki.setVisible(False)
            if wybrana is not None:
                self.wybrana.setData(pos=np.asarray(wybrana, np.float32).reshape(1, 3), size=float(rozm) * 2.6, color=(1, 1, 1, 0.9))
                self.wybrana.setVisible(True)
            else:
                self.wybrana.setVisible(False)
except ImportError:                                                               # bez Qt (testy obliczeń): tylko liczenie
    pass
