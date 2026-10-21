"""Rzuty z 192 wymiarów do 3D dla zakładki „Lasery i koloidy”, okienka R1–R8 (reports/Rzutowanie wielowymiarowego ruchu do 3D.md).

Zasada: wszystko liczymy w pełnych 192 wymiarach, a w 3D tylko rysujemy. Każdy widok podaje, ile czego pokazuje (miernik
wierności), obok zera dla losowej podprzestrzeni 3D: 3/192 = 1,6 % energii, √(3/192) = 12,5 % długości.

Koloid j w checkpoincie t: stan x̃_tj tuż przed MLP bloku 6, względem środka chmury tego zdjęcia, x_tj = x̃_tj − średnia_j x̃_tj
(jak h.pos w lasery_dane, ale bez rzutu). Siła: przesunięcie przez MLP względem średniej po tokenach,
F_tj = Σ_c (z_tjc − z̄_tc) v_tc, czyli dokładnie suma 768 wiązek ze sfery (bias fc2 i część wspólna wszystkim tokenom znikają).
Rzut liniowy zachowuje sumy: rzut F = suma rzutów wiązek, a rzut toru = tor rzutów (interpolacja między checkpointami też).

Bazy (kolumny ortonormalne, 192 × 3):
  P3 — aplikacji (sfera): 3 główne kierunki znormowanych kolumn fc2 ostatniego checkpointu (lasery_dane.policz);
  A  — mapa klas: oś 1 = Δ = średnia narzędzia − średnia tła (100 % separacji, uśrednione po checkpointach), osie 2–3 = PCA
       reszty chmury; A_t: to samo w każdym checkpoincie osobno, osie 2–3 obrócone (Procrustes 2 × 2) do osi wspólnych;
  S  — siły: 3 główne kierunki Σ_t Σ_j F Fᵀ (największa energia sił netto);
  W  — wiązki ważone aktywacją: 3 główne kierunki Σ_t Σ_c E_j[(z_jc − z̄_c)²] v_c v_cᵀ (energia, którą wiązki naprawdę
       dostarczają; następca P3, który waży każdy kierunek fc2 tak samo);
  D  — wędrówka: 3 główne kierunki przemieszczeń x_{t+1} − x_t (kierunki ruchu, a nie rozrzutu chmury);
  F7 — 7 kierunków: Δ, 2 × reszta chmury, 2 × siły, 2 × wiązki, ortonormalizowane w tej kolejności; 7 trójek płaszczyzny Fano
       (FANO): każda para tych kierunków jest razem w dokładnie jednym widoku.
Ramy lokalne wybranej cząsteczki w jednym checkpoincie:
  B — rama siły: oś 1 = F̂ (100 % siły), osie 2–3 = 2 główne kierunki składowych wiązek prostopadłych do F̂ (ich suma jest
      dokładnie zerowa, więc to płaszczyzna, w której wiązki „przeciągają linę”);
  C — rama Schura: podprzestrzeń niezmiennicza pełnego jakobianu J = ∂F/∂x (192 × 192): płaszczyzna jednej pary zespolonej
      i kierunek jednej rzeczywistej wartości własnej; tylko w niej typ ruchu (wir, ściąganie, wypychanie) jest prawdziwy.
Wynik: katalog historii / rzuty / (X.npy, F.npy: (K, 600, 192) float16, dane.npz) i wspólne dla wag
cache_uczenia/lasery/_fc2_<skrót>/V.npy (K, 192, 768) float16.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

import lasery_dane as LD

WERSJA = 1
D = 192
LOS_ENERGIA = 3 / D                                    # losowa podprzestrzeń 3D: oczekiwany udział energii (1,6 %)
LOS_DLUGOSC = float(np.sqrt(3 / D))                    # i typowy udział długości (12,5 %)
COS_PRZECIWNE = float(np.cos(np.radians(150)))         # para „przeciwna”: kąt ≥ 150°
LOS_PRZECIWNE_3D = (1 + COS_PRZECIWNE) / 2             # losowe kierunki w 3D: P(cos ≤ −0,866) = 6,7 %
GINIBRE_ZESPOLONE = 1 - (0.5 + np.sqrt(2 * D / np.pi)) / D   # losowa macierz 192 × 192: ok. 94 % zespolonych (Edelman i in. 1994)
BAZY = ('P3', 'A', 'A_t', 'S', 'W', 'D')
FANO = ((0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5))
NAZWY_F7 = ('Δ narzędzie–tło', 'reszta chmury 1', 'reszta chmury 2', 'siły 1', 'siły 2', 'wiązki 1', 'wiązki 2')
N_PERM = 200                                           # permutacje etykiet narzędzie/tło: zero dla odległości klas


class Przerwano(Exception):
    pass


# ------------------------------------------------------------------ algebra
def _znak(U):
    """Kolumny ze stałym znakiem (największa co do modułu składowa dodatnia), żeby bazy nie przeskakiwały między przeliczeniami."""
    U = np.array(U, np.float64)
    s = np.sign(U[np.abs(U).argmax(0), np.arange(U.shape[1])])
    s[s == 0] = 1
    return U * s


def glowne(C, k=3):
    """k głównych wektorów własnych symetrycznej macierzy C (kolumny, malejąco), ze stałym znakiem."""
    _, U = np.linalg.eigh((C + C.T) / 2)
    return _znak(U[:, ::-1][:, :k])


def _dopisz(out, kandydaci, ile, tol=1e-6):
    """Gram–Schmidt (dwa przebiegi): dopisuje do `out` najwyżej `ile` kandydatów, pomijając zależne od już wybranych."""
    n0 = len(out)
    for v in kandydaci:
        if len(out) - n0 >= ile:
            break
        v = np.array(v, np.float64)
        for _ in range(2):
            for e in out:
                v -= (e @ v) * e
        nv = np.linalg.norm(v)
        if nv > tol:
            out.append(v / nv)
    return out


def baza_A(delta, Cx):
    """Mapa klas: oś 1 = Δ/|Δ| (narzędzie po stronie dodatniej), osie 2–3 = PCA reszty chmury (Cx na dopełnieniu Δ).
    Bez ramki narzędzia (Δ nieokreślone) zwykła PCA chmury."""
    n = np.nan if delta is None else float(np.linalg.norm(delta))
    if not np.isfinite(n) or n < 1e-12:
        return glowne(Cx)
    e1 = np.asarray(delta, np.float64) / n
    Q = np.eye(len(e1)) - np.outer(e1, e1)
    reszta = _dopisz([e1], list(glowne(Q @ Cx @ Q, 3).T), 2)[1:]
    return np.stack([e1] + [_znak(r[:, None])[:, 0] for r in reszta], 1)


def obroc_do(B2, R2):
    """Obrót/odbicie 2 × 2 kolumn B2 (192 × 2) najbliższe R2 (ortogonalny Procrustes): kamera nie skacze między checkpointami."""
    U, _, Vt = np.linalg.svd(B2.T @ R2)
    return B2 @ (U @ Vt)


def baza_F7(A, S_wszystkie, W_wszystkie):
    """7 kierunków płaszczyzny Fano: Δ, 2 × reszta chmury, 2 × siły, 2 × wiązki, ortonormalizowane w tej kolejności."""
    out = _dopisz([], [A[:, 0]], 1)
    _dopisz(out, [A[:, 1], A[:, 2]], 2)
    _dopisz(out, list(S_wszystkie.T), 2)
    _dopisz(out, list(W_wszystkie.T), 2)
    B = np.stack(out, 1)
    return np.concatenate([B[:, :1], _znak(B[:, 1:])], 1)


def fano_pokrywa_pary():
    """Kontrola: każda z 21 par 7 kierunków występuje w dokładnie jednej trójce FANO."""
    from itertools import combinations
    licz = {p: 0 for p in combinations(range(7), 2)}
    for tr in FANO:
        for p in combinations(sorted(tr), 2):
            licz[p] += 1
    return all(v == 1 for v in licz.values())


def udzial(B, C):
    """Ułamek energii (śladu) macierzy C zachowany w podprzestrzeni o ortonormalnych kolumnach B."""
    tr = float(np.trace(C))
    return float(np.trace(B.T @ C @ B)) / tr if tr > 0 else np.nan


# ------------------------------------------------------------------ pliki
def _katalog_wag(h, wymagany=True):
    """Katalog wag historii; cache z innego dysku (np. F:) wskazuje tę samą ścieżkę w bieżącej bazie projektu (LD.ROOT)."""
    p = Path(str(h.wagi))
    if p.is_dir():
        return p
    czesci = p.parts
    if 'SmallViTAblation' in czesci:
        q = LD.ROOT.joinpath(*czesci[czesci.index('SmallViTAblation') + 1:])
        if q.is_dir():
            return q
    if wymagany:
        raise FileNotFoundError(f'brak katalogu wag {p}')
    return p


def plik_V(h):
    s = json.dumps(dict(wagi=str(_katalog_wag(h, False).resolve()), kroki=[int(k) for k in h.kroki]))
    return LD.CACHE / f'_fc2_{hashlib.sha1(s.encode()).hexdigest()[:16]}' / 'V.npy'


def katalog(h):
    return Path(h.katalog) / 'rzuty'


def _V_gotowe(p, K):
    try:
        return p.exists() and np.load(p, mmap_mode='r').shape == (K, D, LD.NK)
    except Exception:                                                    # noqa: BLE001 (uszkodzony plik: policz od nowa)
        return False


class DaneRzutow:
    """Wczytany wynik `policz`: tablice z dane.npz jako atrybuty, X, F, V jako mmap; współrzędne w bazach liczone raz (cache)."""

    def __init__(self, kat, pv):
        d = np.load(kat / 'dane.npz', allow_pickle=False)
        for k in d.files:
            v = d[k]
            setattr(self, k, v.item() if v.ndim == 0 else v)
        self.katalog = Path(kat)
        self.X = np.load(kat / 'X.npy', mmap_mode='r')
        self.F = np.load(kat / 'F.npy', mmap_mode='r')
        self.V = np.load(pv, mmap_mode='r')
        self.K = len(self.kroki)
        self.nazwy_F7 = [str(s) for s in self.nazwy_F7]
        self._wsp = {}
        self._tlo = {}

    def baza(self, nazwa):
        return getattr(self, 'baza_' + nazwa)

    def gpu(self, co='X'):
        """X albo F całej historii na karcie (torch float16, raz na wczytanie: K · 600 · 192 · 2 B, np. 186 MB dla 806
        checkpointów); None bez CUDA albo przy braku pamięci karty (wtedy liczy numpy z mmap)."""
        k = '_g' + co
        if k not in self.__dict__:
            self.__dict__[k] = None
            try:
                import torch
                if torch.cuda.is_available():
                    src = self.X if co == 'X' else self.F
                    self.__dict__[k] = torch.from_numpy(np.ascontiguousarray(src)).to('cuda', non_blocking=False)
            except Exception:                                            # noqa: BLE001 (brak pamięci karty: numpy)
                import traceback
                traceback.print_exc()
        return self.__dict__[k]

    def zwolnij_gpu(self):
        for k in ('_gX', '_gF'):
            self.__dict__.pop(k, None)

    def wsp(self, B, co='X'):
        """Współrzędne (K, 600, k) w bazie B: (192, k) wspólnej albo (K, 192, k) osobnej dla każdego checkpointu; co = 'X' albo 'F'.
        Na karcie (torch) jedno mnożenie całej historii, bez CUDA numpy w blokach z mmap."""
        B = np.asarray(B, np.float32)
        klucz = (co, B.shape, hashlib.sha1(np.ascontiguousarray(B).tobytes()).hexdigest())
        if klucz not in self._wsp:
            g = self.gpu(co)
            out = np.empty((self.K, LD.TOK, B.shape[-1]), np.float32)
            if g is not None:
                import torch
                Bg = torch.from_numpy(B).to(g.device)
                for a in range(0, self.K, 256):
                    blok = g[a:a + 256].float()
                    r = blok @ Bg if B.ndim == 2 else torch.einsum('ktd,kdc->ktc', blok, Bg[a:a + 256])
                    out[a:a + 256] = r.cpu().numpy()
            else:
                src = self.X if co == 'X' else self.F
                for a in range(0, self.K, 64):
                    blok = np.asarray(src[a:a + 64], np.float32)
                    out[a:a + 64] = blok @ B if B.ndim == 2 else np.einsum('ktd,kdc->ktc', blok, B[a:a + 64])
            if len(self._wsp) > 24:
                self._wsp.clear()
            self._wsp[klucz] = out
        return self._wsp[klucz]

    def tor(self, j):
        """Tor cząsteczki j (K, 192) float64 względem środka chmury."""
        return np.asarray(self.X[:, int(j)], np.float64)

    def srodek_tla(self):
        """Środek tła w każdym checkpoincie (K, 192), względem środka chmury (raz z całego X)."""
        if 'tlo' not in self._tlo:
            m = ~np.asarray(self.narz, bool)
            if not m.any():
                m[:] = True
            g = self.gpu('X')
            if g is not None:
                import torch
                mg = torch.from_numpy(m).to(g.device)
                out = torch.cat([g[a:a + 256][:, mg].float().mean(1) for a in range(0, self.K, 256)]).double().cpu().numpy()
            else:
                out = np.empty((self.K, D), np.float64)
                for a in range(0, self.K, 64):
                    out[a:a + 64] = np.asarray(self.X[a:a + 64], np.float32)[:, m].mean(1)
            self._tlo['tlo'] = out
        return self._tlo['tlo']


def wczytaj(h):
    """DaneRzutow dla historii h albo None (brak, inna wersja, inne checkpointy albo brak V)."""
    kat = katalog(h)
    if not (kat / 'dane.npz').exists() or not (kat / 'X.npy').exists():
        return None
    try:
        d = np.load(kat / 'dane.npz', allow_pickle=False)
        if int(d['wersja']) != WERSJA or len(d['kroki']) != len(h.kroki) or not np.array_equal(d['kroki'], h.kroki):
            return None
        pv = plik_V(h)
        if not _V_gotowe(pv, len(h.kroki)):
            pv = Path(str(d['plik_V']))
            if not _V_gotowe(pv, len(h.kroki)):
                return None
        return DaneRzutow(kat, pv)
    except Exception:                                                    # noqa: BLE001 (uszkodzony zapis: policz od nowa)
        import traceback
        traceback.print_exc()
        return None


# ------------------------------------------------------------------ obliczenie stanów 192D i baz
def policz(h, postep=None, przerwij=None, urzadzenie='cuda'):
    """Stany 192D, siły, kolumny fc2, bazy i mierniki wierności dla historii h (lasery_dane.Historia); zwraca DaneRzutow.
    Raz na historię i zdjęcie, potem z dysku. postep(i, n, opis); przerwij() → True kończy (wyjątek Przerwano, nic nie zostaje)."""
    import torch
    gotowe = wczytaj(h)
    if gotowe is not None:
        return gotowe
    wagi = _katalog_wag(h)
    out = katalog(h)
    tmp = out.with_name('rzuty_tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    pv = plik_V(h)
    pisz_V = not _V_gotowe(pv, len(h.kroki))
    pv_tmp = pv.with_name('V_tmp.npy')
    if pisz_V:
        pv.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
    M = LD.modul_modelu()
    net = M.MalyViT().to(dev).eval().requires_grad_(False)
    pam = {}
    net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))   # stan x̃ przed MLP bloku 6
    net.bloki[-1].mlp.act.register_forward_hook(lambda m, a, o: pam.__setitem__('z', o.detach()))
    rgb = torch.from_numpy(np.ascontiguousarray(h.rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
    kroki = [int(k) for k in h.kroki]
    K, T = len(kroki), LD.TOK
    X = np.lib.format.open_memmap(tmp / 'X.npy', 'w+', np.float16, (K, T, D))
    Fm = np.lib.format.open_memmap(tmp / 'F.npy', 'w+', np.float16, (K, T, D))
    Vm = np.lib.format.open_memmap(pv_tmp, 'w+', np.float16, (K, D, LD.NK)) if pisz_V else None
    narz_np = np.asarray(h.narz, bool)
    narz = torch.as_tensor(narz_np, device=dev)
    nt = int(narz_np.sum())
    klasy = 0 < nt < T
    f64 = dict(dtype=torch.float64, device=dev)
    SX, SF, SW, SD = (torch.zeros(D, D, **f64) for _ in range(4))
    MEAN = np.zeros((K, D), np.float32); DELTA = np.full((K, D), np.nan, np.float32)
    R = np.zeros(K, np.float32); PR = np.zeros(K, np.float32); OWN3 = np.zeros(K, np.float32); FN = np.zeros((K, T), np.float32)
    SEP = np.full(K, np.nan, np.float32); SEP0 = np.full((K, 2), np.nan, np.float32)
    if klasy:
        g = torch.Generator().manual_seed(0)
        perm = torch.argsort(torch.rand(N_PERM, T, generator=g), 1)[:, :nt]
        MP = torch.zeros(N_PERM, T, dtype=torch.float64).scatter_(1, perm, 1.0)
        MP = (MP / nt - (1 - MP) / (T - nt)).to(dev)                  # wiersz · x = „narzędzie” − „tło” dla losowych etykiet
    from concurrent.futures import ThreadPoolExecutor
    pula = ThreadPoolExecutor(4)                                       # odczyt checkpointów z wyprzedzeniem (dysk jest wąskim gardłem)
    odczyt = lambda k: (lambda st: st.get('net', st))(torch.load(wagi / f'k{k:05d}.pt', map_location='cpu', weights_only=True))
    kolejka = [pula.submit(odczyt, k) for k in kroki[:6]]
    t0 = time.time()

    def sprzataj():
        nonlocal X, Fm, Vm
        for f_ in kolejka:
            f_.cancel()
        pula.shutdown(wait=True)
        X = Fm = Vm = None
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)
        if pisz_V and pv_tmp.exists():
            pv_tmp.unlink()

    try:
        prev = None
        V = None
        for i, k in enumerate(kroki):
            if przerwij is not None and przerwij():
                raise Przerwano()
            net.load_state_dict(kolejka.pop(0).result(), strict=True)
            if i + 6 < K:
                kolejka.append(pula.submit(odczyt, kroki[i + 6]))
            with torch.no_grad():
                net(rgb)
                x = pam['x'][0].double(); z = pam['z'][0].double()
                mu = x.mean(0); xc = x - mu
                zc = z - z.mean(0)
                V = net.bloki[-1].mlp.fc2.weight.double()                # (192, 768): kolumny = kierunki wiązek v_c
                f = zc @ V.T                                             # (600, 192): siła netto = suma 768 wiązek (jak sfera)
                C = xc.T @ xc
                SX += C; SF += f.T @ f; SW += (V * zc.pow(2).mean(0)) @ V.T
                if prev is not None:
                    dd = xc - prev
                    SD += dd.T @ dd
                prev = xc
                ev = torch.linalg.eigvalsh(C).clamp_min(0)
                PR[i] = float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-300))
                OWN3[i] = float(ev[-3:].sum() / ev.sum().clamp_min(1e-300))
                R[i] = float(xc.pow(2).sum(1).mean().sqrt())
                FN[i] = f.norm(dim=1).float().cpu().numpy()
                if klasy:
                    dl = xc[narz].mean(0) - xc[~narz].mean(0)
                    DELTA[i] = dl.float().cpu().numpy()
                    SEP[i] = float(dl.norm()) / max(R[i], 1e-30)
                    dn = (MP @ xc).norm(dim=1) / max(R[i], 1e-30)
                    SEP0[i] = [float(dn.mean()), float(torch.quantile(dn, 0.95))]
                X[i] = xc.half().cpu().numpy(); Fm[i] = f.half().cpu().numpy(); MEAN[i] = mu.float().cpu().numpy()
                if Vm is not None:
                    Vm[i] = V.half().cpu().numpy()
            if postep is not None:
                postep(i + 1, 2 * K, f'rzuty: stany 192D, checkpoint {i + 1}/{K} (krok {k}), {time.time() - t0:.0f} s')
        pula.shutdown(wait=False)
        # --- bazy wspólne dla wszystkich checkpointów
        Vn = V / V.norm(dim=0, keepdim=True).clamp_min(1e-12)           # P3 jak lasery_dane.policz (ostatni checkpoint)
        U_, _, _ = torch.linalg.svd(Vn - Vn.mean(1, keepdim=True), full_matrices=False)
        P3 = U_[:, :3].cpu().numpy()
        try:
            wersja_h = int(json.loads((Path(h.katalog) / 'opis.json').read_text(encoding='utf-8')).get('wersja', 1))
        except Exception:                                                # noqa: BLE001
            wersja_h = LD.WERSJA
        ref = np.asarray(X[-1] if wersja_h >= 2 else Fm[-1], np.float64)  # wersja 1 lasery_dane: h.pos = rzut sumy wiązek, nie stanu
        pos = np.asarray(h.pos[-1], np.float64)
        P3 *= np.sign(np.sign(((ref @ P3) * pos).sum(0)) + 0.5)          # znaki osi zgodne ze sferą (h.pos)
        zgodnosc = float(np.abs(ref @ P3 - pos).max() / max(np.abs(pos).max(), 1e-30))
        SXn, SFn, SWn, SDn = (a.cpu().numpy() for a in (SX, SF, SW, SD))
        dbar = np.nanmean(DELTA, 0) if klasy else None
        A = baza_A(dbar, SXn)
        S6, W6 = glowne(SFn, 6), glowne(SWn, 6)
        S, W, Dm = S6[:, :3], W6[:, :3], glowne(SDn, 3)
        F7 = baza_F7(A, S6, W6)
        # --- drugi przebieg: osobna mapa klas w każdym checkpoincie i mierniki wierności wszystkich baz
        Z = h.Z
        Vsrc = Vm if Vm is not None else np.load(pv, mmap_mode='r')
        A_T = np.zeros((K, D, 3), np.float32)
        WX, WF, WW, WS, WD = (np.full((K, len(BAZY)), np.nan, np.float32) for _ in range(5))
        E7 = np.zeros((K, 3, 7), np.float32)
        prevx = None
        for i in range(K):
            if przerwij is not None and przerwij():
                raise Przerwano()
            x = np.asarray(X[i], np.float64); f = np.asarray(Fm[i], np.float64)
            zc = np.asarray(Z[i], np.float64); zc -= zc.mean(0)
            V_ = np.asarray(Vsrc[i], np.float64)
            Cx, Cf, Cw = x.T @ x, f.T @ f, (V_ * (zc ** 2).mean(0)) @ V_.T
            dl = DELTA[i].astype(np.float64) if klasy else None
            Ai = baza_A(dl, Cx)
            if Ai.shape[1] == 3 and klasy and np.isfinite(dl).all():
                Q = np.eye(D) - np.outer(Ai[:, 0], Ai[:, 0])
                ref, _ = np.linalg.qr(Q @ A[:, 1:])
                Ai[:, 1:] = obroc_do(Ai[:, 1:], ref)
            A_T[i] = Ai
            dd = None if prevx is None else x - prevx
            for b, B in enumerate((P3, A, Ai, S, W, Dm)):
                WX[i, b], WF[i, b], WW[i, b] = udzial(B, Cx), udzial(B, Cf), udzial(B, Cw)
                if klasy and np.isfinite(dl).all():
                    WS[i, b] = np.linalg.norm(B.T @ dl) / max(np.linalg.norm(dl), 1e-30)
                if dd is not None:
                    WD[i, b] = float(((dd @ B) ** 2).sum() / max((dd ** 2).sum(), 1e-300))
            for c_, C_ in enumerate((Cx, Cf, Cw)):
                E7[i, c_] = np.einsum('dk,de,ek->k', F7, C_, F7) / max(np.trace(C_), 1e-300)
            prevx = x
            if postep is not None and (i % 8 == 0 or i == K - 1):
                postep(K + i + 1, 2 * K, f'rzuty: bazy i mierniki wierności, checkpoint {i + 1}/{K}')
        np.savez(tmp / 'dane.npz', wersja=WERSJA, kroki=np.asarray(kroki, np.int64), narz=narz_np,
                 baza_P3=P3, baza_A=A, baza_S=S, baza_W=W, baza_D=Dm, baza_F7=F7, A_t=A_T, nazwy_F7=np.array(NAZWY_F7),
                 mean=MEAN, delta=DELTA, R=R, PR=PR, own3=OWN3, FN=FN, sep=SEP, sep0=SEP0,
                 WX=WX, WF=WF, WW=WW, WS=WS, WD=WD, E7=E7, zgodnosc_P3=zgodnosc, plik_V=str(pv),
                 czas_s=float(time.time() - t0), wagi=str(wagi))
    except BaseException:
        Vsrc = Z = None
        sprzataj()
        raise
    Vsrc = Z = None
    X.flush(); Fm.flush()
    if Vm is not None:
        Vm.flush()
    X = Fm = Vm = None
    gc.collect()
    if pisz_V:
        os.replace(pv_tmp, pv)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return wczytaj(h)


# ------------------------------------------------------------------ rama siły (B)
def wklady(z_j, z_sr, V):
    """Wkłady 768 wiązek do siły cząsteczki: u_c = (z_jc − z̄_c) v_c, tablica (768, 192); suma wierszy = F_j."""
    return (np.asarray(z_j, np.float64) - np.asarray(z_sr, np.float64))[:, None] * np.asarray(V, np.float64).T


def rama_sily(u):
    """Rama siły cząsteczki z wkładów u (768, 192): oś 1 = F̂, osie 2–3 = 2 główne kierunki składowych prostopadłych do F̂.
    Σ_c u_c⊥ = (I − F̂F̂ᵀ) F = 0, więc osie 2–3 to płaszczyzna znoszenia się wiązek. Zwraca bazę i miary (192D)."""
    F = u.sum(0)
    nF = float(np.linalg.norm(F))
    e1 = F / nF if nF > 1e-12 else glowne(u.T @ u, 1)[:, 0]
    a = u @ e1
    up = u - a[:, None] * e1
    _, s, Vt = np.linalg.svd(up, full_matrices=False)
    B = np.stack(_dopisz([e1], list(Vt[:4]), 2), 1)
    B = np.concatenate([B[:, :1], _znak(B[:, 1:])], 1)
    en = float((s ** 2).sum())
    nu = np.linalg.norm(u, axis=1)
    return dict(B=B, F=F, nF=nF, a=a, cos=a / np.maximum(nu, 1e-30), norma=nu,
                poprz=float((s[:2] ** 2).sum() / en) if en > 0 else np.nan, energia_poprz=en)


def siec_silna(u, prog=0.8):
    """Najsilniejsze wiązki (malejąco |u_c|), które razem niosą ułamek `prog` energii Σ|u_c|², i suma reszty (jedna strzałka).
    Suma silnych + reszta = F dokładnie (jak sieć silna i słaba sił kontaktowych, Radjai i in. 1998)."""
    e = (u ** 2).sum(1)
    kol = np.argsort(-e)
    c = np.cumsum(e[kol]) / max(float(e.sum()), 1e-300)
    n = int(min(np.searchsorted(c, prog) + 1, len(kol)))
    return kol[:n], u[kol[n:]].sum(0)


def pary_przeciwne(u, B=None):
    """Odsetek par wkładów (wiersze u) o kącie ≥ 150°: w 192D, a z bazą B (192 × k) w jej rzucie. 768 × 768 kosinusów na karcie
    (torch), bez CUDA w numpy."""
    v = np.asarray(u, np.float64) if B is None else np.asarray(u, np.float64) @ B
    n = np.linalg.norm(v, axis=1)
    v = v[n > 1e-30] / n[n > 1e-30, None]
    m = len(v)
    if m < 2:
        return np.nan
    try:
        import torch
        if torch.cuda.is_available():
            g = torch.from_numpy(v).to('cuda', torch.float32)
            c = torch.triu(g @ g.T, 1)
            return float(((c <= COS_PRZECIWNE) & (c != 0)).sum().item() / (m * (m - 1) / 2))
    except Exception:                                                    # noqa: BLE001 (karta zajęta: numpy)
        pass
    c = v @ v.T
    iu = np.triu_indices(m, 1)
    return float((c[iu] <= COS_PRZECIWNE).mean())


# ------------------------------------------------------------------ pole bloku 6 i jakobian (C)
def _gelu(u):
    from scipy.special import erf
    return 0.5 * u * (1 + erf(u / np.sqrt(2)))


def _gelu_p(u):
    """Pochodna dokładnego GELU: Φ(u) + u φ(u)."""
    from scipy.special import erf
    return 0.5 * (1 + erf(u / np.sqrt(2))) + u * np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)


def parametry_bloku6(st, eps):
    """LN₂ (γ, β, eps), fc1 (W₁ 768 × 192, b₁) i fc2 (V 192 × 768) ostatniego bloku z checkpointu (state dict) jako float64."""
    nr = max(int(k.split('.')[1]) for k in st if k.startswith('bloki.'))
    g = lambda n: st[f'bloki.{nr}.{n}'].detach().double().cpu().numpy()
    return dict(g=g('norm2.weight'), b=g('norm2.bias'), W1=g('mlp.fc1.weight'), b1=g('mlp.fc1.bias'), V=g('mlp.fc2.weight'),
                eps=float(eps))


def pole(par, X):
    """Wyjście MLP bloku 6 bez biasu fc2 w punktach X (n, 192): V · GELU(W₁ · LN(x) + b₁). Różnice pola nie zależą od biasu
    ani od średniej z̄ po tokenach, więc to samo pole co w ogniskowa.py."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    mu = X.mean(1, keepdims=True)
    sig = np.sqrt(X.var(1, keepdims=True) + par['eps'])
    u = ((X - mu) / sig * par['g'] + par['b']) @ par['W1'].T + par['b1']
    return _gelu(u) @ par['V'].T


def jakobian(par, x):
    """Dokładny jakobian J = ∂F/∂x (192 × 192) przez LayerNorm: V diag(GELU′(u)) W₁ diag(γ) (I − 11ᵀ/D − x̂x̂ᵀ/D) / σ."""
    x = np.asarray(x, np.float64)
    n = len(x)
    sig = np.sqrt(x.var() + par['eps'])
    xh = (x - x.mean()) / sig
    u = (xh * par['g'] + par['b']) @ par['W1'].T + par['b1']
    P = np.eye(n) - 1.0 / n - np.outer(xh, xh) / n
    return (par['V'] * _gelu_p(u)) @ (par['W1'] * par['g']) @ P / sig


def widmo_stat(J, tol=None):
    """Miary widma J niezależne od rzutu: udział zespolonych, liczba Re λ > 0, dywergencja Tr J / n, odchylenie od normalności
    Henriciego √(‖J‖²_F − Σ|λ|²)/‖J‖_F (0 = normalna)."""
    lam = np.linalg.eigvals(J)
    tol = 1e-9 * max(np.abs(lam).max(), 1e-30) if tol is None else tol
    fro2 = float((J ** 2).sum())
    return dict(lam=lam, zesp=float((np.abs(lam.imag) > tol).mean()), dodatnie=int((lam.real > tol).sum()),
                div=float(np.trace(J) / len(J)), henrici=float(np.sqrt(max(fro2 - float((np.abs(lam) ** 2).sum()), 0.0) / max(fro2, 1e-300))))


def rama_schura(J, para=0, rzecz=0, F=None):
    """Trójwymiarowa podprzestrzeń niezmiennicza J: płaszczyzna jednej pary zespolonej i kierunek jednej rzeczywistej wartości własnej.
    para: 0 = największe |Im λ| (najszybszy obrót), 1 = największe Re λ (najmniej stabilna), 2 = największy rzut F̂ na płaszczyznę;
    rzecz: 0 = największe Re λ (najsilniej wypycha), 1 = najmniejsze Re λ (najsilniej ściąga), 2 = największy rzut F̂.
    Rozpięcie tych wektorów własnych jest dokładnie niezmiennicze (J Q = Q T), tak jak pierwsze wektory Schura po uporządkowaniu,
    więc T = QᵀJQ ma dokładnie wybrane wartości własne."""
    lam, W = np.linalg.eig(J)
    tol = 1e-9 * max(np.abs(lam).max(), 1e-30)
    zesp = np.flatnonzero(lam.imag > tol)
    rz = np.flatnonzero(np.abs(lam.imag) <= tol)
    Fh = None if F is None or np.linalg.norm(F) < 1e-30 else np.asarray(F, np.float64) / np.linalg.norm(F)

    def plaszczyzna(i):
        return np.linalg.qr(np.stack([W[:, i].real, W[:, i].imag], 1))[0]

    ip = ir = None
    if len(zesp):
        if para == 1:
            ip = zesp[np.argmax(lam[zesp].real)]
        elif para == 2 and Fh is not None:
            ip = zesp[np.argmax([np.linalg.norm(plaszczyzna(i).T @ Fh) for i in zesp])]
        else:
            ip = zesp[np.argmax(lam[zesp].imag)]
    if len(rz):
        if rzecz == 1:
            ir = rz[np.argmin(lam[rz].real)]
        elif rzecz == 2 and Fh is not None:
            wr = W[:, rz].real / np.maximum(np.linalg.norm(W[:, rz].real, axis=0), 1e-30)
            ir = rz[np.argmax(np.abs(wr.T @ Fh))]
        else:
            ir = rz[np.argmax(lam[rz].real)]
    kol, wybrane = [], []
    if ip is not None:
        kol += [W[:, ip].real, W[:, ip].imag]; wybrane.append(lam[ip])
    if ir is not None:
        kol.append(W[:, ir].real); wybrane.append(lam[ir])
    zapas = [W[:, i].real for i in np.argsort(-np.abs(lam.real))] + list(np.eye(len(J)))
    Q = np.stack(_dopisz([], kol + zapas, 3), 1)
    T = Q.T @ J @ Q
    JQ = J @ Q
    reszta = float(np.linalg.norm(JQ - Q @ T) / max(np.linalg.norm(JQ), 1e-300))
    lt, wt = np.linalg.eig(T)                                            # struktura w ramie (współrzędne 3D)
    ic = np.flatnonzero(np.abs(lt.imag) > 1e-9 * max(np.abs(lt).max(), 1e-30))
    plaszcz = os_ = None; lam_p = lam_r = None
    if len(ic):
        c = ic[np.argmax(lt[ic].imag)]
        plaszcz = np.linalg.qr(np.stack([wt[:, c].real, wt[:, c].imag], 1))[0]
        lam_p = lt[c]
        r = [i for i in range(3) if i not in ic]
        if r:
            os_ = wt[:, r[0]].real / max(np.linalg.norm(wt[:, r[0]].real), 1e-30); lam_r = lt[r[0]].real
    else:
        k = int(np.argmax(lt.real)); os_ = wt[:, k].real / max(np.linalg.norm(wt[:, k].real), 1e-30); lam_r = lt[k].real
    kier = 0.0
    if plaszcz is not None:                                              # kierunek obrotu w płaszczyźnie: znak składowej stycznej T·c
        th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        c_ = np.cos(th)[:, None] * plaszcz[:, 0] + np.sin(th)[:, None] * plaszcz[:, 1]
        st_ = -np.sin(th)[:, None] * plaszcz[:, 0] + np.cos(th)[:, None] * plaszcz[:, 1]
        kier = float(np.sign(np.mean(((c_ @ T.T) * st_).sum(1))))
    return dict(Q=Q, T=T, wybrane=np.array(wybrane), reszta=reszta, niezmiennicza=ip is not None and ir is not None,
                plaszczyzna=plaszcz, os=os_, lam_p=lam_p, lam_r=lam_r, kierunek=kier)


def pole_w_ramie(par, x, Q, r, G=7, J=None):
    """Pole na siatce G³ w sześcianie [−r, r]³ ramy Q (192 × 3) wokół x: zmiana siły względem skoku samej cząsteczki,
    d(u) = Qᵀ (F(x + Qu) − F(x)) (pole nieliniowe, dokładne), część liniowa (QᵀJQ) u i ułamek zmiany poza ramą."""
    lin = np.linspace(-r, r, G)
    g = np.stack(np.meshgrid(lin, lin, lin, indexing='ij'), -1).reshape(-1, 3)
    F0 = pole(par, x)[0]
    dF = pole(par, np.asarray(x, np.float64)[None] + g @ Q.T) - F0
    d = dF @ Q
    n = np.linalg.norm(dF, axis=1)
    poza = np.where(n > 1e-30, np.linalg.norm(dF - d @ Q.T, axis=1) / np.maximum(n, 1e-300), 0.0)
    lin_ = None if J is None else g @ (Q.T @ J @ Q).T
    return dict(siatka=g, pole=d, liniowe=lin_, poza=poza, F0=F0)


_siec = {}
_widmo0 = {}


def policz_rame_C(wagi, krok, rgb, tok, para=0, rzecz=0, G=7, zasieg=0.5, krok0=None, urzadzenie='cuda'):
    """Rama Schura (C) wybranej cząsteczki w jednym checkpoincie: przejście w przód (stan x̃ dokładnie, nie z float16), pełny
    jakobian, podprzestrzeń niezmiennicza, pole na siatce i widmo; dla porównania widmo tej samej cząsteczki w checkpoincie krok0."""
    import torch
    dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
    M = LD.modul_modelu()
    if dev.type not in _siec:
        net = M.MalyViT().to(dev).eval().requires_grad_(False)
        pam = {}
        net.bloki[-1].norm2.register_forward_hook(lambda m, a, o: pam.__setitem__('x', a[0].detach()))
        _siec[dev.type] = (net, pam)
    net, pam = _siec[dev.type]
    x_in = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)

    def stan(k):
        st = torch.load(Path(wagi) / f'k{int(k):05d}.pt', map_location=dev, weights_only=True)
        st = st.get('net', st)
        net.load_state_dict(st, strict=True)
        with torch.no_grad():
            net(x_in)
        return pam['x'][0].double().cpu().numpy(), parametry_bloku6(st, net.bloki[-1].norm2.eps)

    X, par = stan(krok)
    x = X[int(tok)]
    J = jakobian(par, x)
    F0 = pole(par, x)[0]
    rama = rama_schura(J, para, rzecz, F0)
    dist = np.linalg.norm(X - x, axis=1); dist[int(tok)] = np.inf
    r = float(zasieg * dist.min())
    siatka = pole_w_ramie(par, x, rama['Q'], r, G, J)
    wid = widmo_stat(J)
    w0 = None
    if krok0 is not None and int(krok0) != int(krok):
        k0 = (str(wagi), int(krok0), hashlib.sha1(np.ascontiguousarray(rgb).tobytes()).hexdigest(), int(tok))
        if k0 not in _widmo0:
            X0, par0 = stan(krok0)
            _widmo0[k0] = {k_: v for k_, v in widmo_stat(jakobian(par0, X0[int(tok)])).items() if k_ != 'lam'}
        w0 = _widmo0[k0]
    QF = rama['Q'].T @ F0
    return dict(rama, **siatka, widmo=wid, widmo0=w0, r=r, krok=int(krok), tok=int(tok),
                skok_w_ramie=float(np.linalg.norm(QF) / max(np.linalg.norm(F0), 1e-300)), QF=QF,
                ginibre=dict(srodek=float(np.trace(J) / len(J)),
                             promien=float(np.std(J - np.eye(len(J)) * np.trace(J) / len(J)) * np.sqrt(len(J)))))


# ------------------------------------------------------------------ wędrówka (D)
def rama_toru(tor):
    """Rama wędrówki jednej cząsteczki (tor (K, 192)): oś 1 = start → koniec (przesunięcie netto w 100 %), osie 2–3 = 2 główne
    kierunki odchyleń toru od tej prostej."""
    d = np.asarray(tor, np.float64) - tor[0]
    n = float(np.linalg.norm(d[-1]))
    e1 = d[-1] / n if n > 1e-12 else glowne(d.T @ d, 1)[:, 0]
    od = d - np.outer(d @ e1, e1)
    B = np.stack(_dopisz([e1], list(glowne(od.T @ od, 4).T) + list(np.eye(d.shape[1])), 2), 1)
    return np.concatenate([B[:, :1], _znak(B[:, 1:])], 1)


def udzial_ruchu(tory, B):
    """Ile ruchu (przemieszczeń między checkpointami) widać w bazie B: (ułamek energii, ułamek długości drogi). tory: (n, K, 192)."""
    kr = np.diff(np.asarray(tory, np.float64), axis=-2)
    kb = kr @ B
    en = float((kb ** 2).sum() / max((kr ** 2).sum(), 1e-300))
    dl = float(np.linalg.norm(kb, axis=-1).sum() / max(np.linalg.norm(kr, axis=-1).sum(), 1e-300))
    return en, dl


def tory_zerowe(tor, n=8, ziarno=0):
    """Model zerowy wędrówki: błądzenie losowe o tych samych długościach kroków co tor, w losowych kierunkach 192D.
    Zwraca (n, K, 192), start w tym samym punkcie co tor."""
    rng = np.random.default_rng(ziarno)
    tor = np.asarray(tor, np.float64)
    dl = np.linalg.norm(np.diff(tor, axis=0), axis=1)
    g = rng.standard_normal((n, len(dl), tor.shape[1]))
    g /= np.linalg.norm(g, axis=2, keepdims=True)
    return np.concatenate([np.repeat(tor[None, :1], n, 0), tor[0] + np.cumsum(g * dl[None, :, None], 1)], 1)


def udzialy_pca(tor):
    """Udział wariancji PC1 i PC1–3 w PCA samego toru (położenia minus średnia). Dla błądzenia losowego PC1 ≈ 60 % i krzywe
    Lissajous (Antognini i Sohl-Dickstein 2018), więc sam kształt toru w PCA niczego nie dowodzi bez toru zerowego."""
    c = np.asarray(tor, np.float64)
    c = c - c.mean(0)
    s = np.linalg.svd(c, compute_uv=False) ** 2
    t = max(float(s.sum()), 1e-300)
    return float(s[0] / t), float(s[:3].sum() / t)


def kat_skretu(tor):
    """Mediana cos kąta między kolejnymi krokami toru w 192D: > 0 marsz w jedną stronę, ≈ 0 błądzenie, < 0 drgania w miejscu."""
    k = np.diff(np.asarray(tor, np.float64), axis=0)
    n = np.linalg.norm(k, axis=1)
    ok = (n[1:] > 1e-30) & (n[:-1] > 1e-30)
    if not ok.any():
        return np.nan
    c = (k[1:] * k[:-1]).sum(1)[ok] / (n[1:] * n[:-1])[ok]
    return float(np.median(c))


# ------------------------------------------------------------------ separacja narzędzia od tła (R9, L)
def rdzen_narzedzia(narz, brzeg=0, W=LD.W):
    """Maska tokenów narzędzia bez `brzeg` skrajnych kolumn i wierszy ramki (prostokątna ramka obejmuje piksele tła na brzegach)."""
    narz = np.asarray(narz, bool)
    if brzeg <= 0 or not narz.any():
        return narz
    idx = np.flatnonzero(narz)
    r, c = np.divmod(idx, W)
    ok = np.ones(len(idx), bool)
    for v in (c, r):                                                     # kolumny i wiersze osobno; wymiar za wąski (np. 2 wiersze) zostaje
        if v.max() - v.min() + 1 > 2 * brzeg:
            ok &= (v >= v.min() + brzeg) & (v <= v.max() - brzeg)
    out = np.zeros_like(narz)
    out[idx[ok]] = True
    return out if out.any() else narz


def baza_L(Xs, m, lam=0.1, osie=0):
    """Oś 1 = kierunek LDA ze skurczem: w ∝ (S_w + λ·tr(S_w)/D · I)⁻¹ Δ (Δ = średnia narzędzia − tła, S_w = kowariancja wewnątrz klas;
    Xs (n, 192) może być pulą wielu checkpointów, m (n,) etykiety). Osie 2–3 w dopełnieniu w: 0 = największy rozrzut wewnątrz klas,
    1 = najmniejszy, 2 = PCA tła, 3 = PCA narzędzia. Zwraca (B (192, 3), udział Δ w osi 1 = cos(w, Δ))."""
    Xs = np.asarray(Xs, np.float64); m = np.asarray(m, bool)
    Xt, Xb = Xs[m], Xs[~m]
    delta = Xt.mean(0) - Xb.mean(0)
    ct, cb = Xt - Xt.mean(0), Xb - Xb.mean(0)
    Sw = (ct.T @ ct + cb.T @ cb) / max(len(Xs) - 2, 1)
    w = np.linalg.solve(Sw + lam * np.trace(Sw) / len(Sw) * np.eye(len(Sw)), delta)
    w /= max(np.linalg.norm(w), 1e-30)
    Q = np.eye(len(w)) - np.outer(w, w)
    C = (Sw, Sw, cb.T @ cb / max(len(cb), 1), ct.T @ ct / max(len(ct), 1))[int(osie)]
    e, U = np.linalg.eigh(Q @ C @ Q)
    kol = [U[:, 1], U[:, 2]] if osie == 1 else [U[:, -1], U[:, -2]]   # najmniejsze: pomijamy zerową wartość w kierunku w
    B = np.stack(_dopisz([w], kol + list(np.eye(len(w))), 2), 1)
    B = np.concatenate([B[:, :1], _znak(B[:, 1:])], 1)
    return B, float(w @ delta / max(np.linalg.norm(delta), 1e-30))


def miary_separacji(P, m, k=10):
    """Separacja klasy m w punktach P (n, d): sylwetka narzędzia, trafienia kNN (udział narzędzia wśród k sąsiadów tokenu narzędzia),
    margines = min odległość narzędzie–tło / mediana odstępu najbliższych sąsiadów tła, stosunek Fishera |Δ|²/wariancja wewnątrz."""
    from scipy.spatial.distance import cdist
    P = np.asarray(P, np.float64); m = np.asarray(m, bool)
    t, b = P[m], P[~m]
    if len(t) < 2 or len(b) < 2:
        return dict(sil=np.nan, hit=np.nan, marg=np.nan, fisher=np.nan)
    Dtb = cdist(t, b); Dtt = cdist(t, t); np.fill_diagonal(Dtt, np.nan)
    a = np.nanmean(Dtt, 1); bq = Dtb.mean(1)
    sil = float(np.mean((bq - a) / np.maximum(np.maximum(a, bq), 1e-30)))
    Dbb = cdist(b, b); np.fill_diagonal(Dbb, np.inf)
    nn = float(np.median(Dbb.min(1)))
    Dall = cdist(P, P); np.fill_diagonal(Dall, np.inf)
    nnk = np.argsort(Dall, 1)[:, :k]
    dm = t.mean(0) - b.mean(0)
    within = (t.var(0).sum() * len(t) + b.var(0).sum() * len(b)) / len(P)
    return dict(sil=sil, hit=float(m[nnk][m].mean()), marg=float(Dtb.min() / max(nn, 1e-30)), fisher=float(dm @ dm / max(within, 1e-30)))


def zero_separacji(P, n_narz, n=100, ziarno=0, k=10):
    """95. percentyl miar separacji dla losowych etykiet (n_narz losowych tokenów jako „narzędzie”)."""
    rng = np.random.default_rng(ziarno)
    out = []
    for _ in range(n):
        m = np.zeros(len(P), bool); m[rng.permutation(len(P))[:n_narz]] = True
        out.append(miary_separacji(P, m, k))
    return {kk: float(np.percentile([o[kk] for o in out], 95)) for kk in out[0]}
