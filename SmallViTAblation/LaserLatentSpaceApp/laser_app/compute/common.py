"""Stałe i funkcje wspólne dla widoków aplikacji: siatka patchy małego ViT, kolory i kierunki 768 kanałów.

Latent małego ViT to 600 tokenów = siatka 20 x 30 patchy 16 x 16 pikseli na wejściu 480 x 320.
Kanał c ma w każdym widoku ten sam kolor (koło barw hsv) i ten sam kierunek (spirala Fibonacciego).
"""
from pathlib import Path

import numpy as np

KOMORKA = 16                                   # piksele wejścia ViT na patch
WIERSZE, KOLUMNY = 20, 30                      # siatka patchy, 600 tokenów
SZER, WYS = KOLUMNY * KOMORKA, WIERSZE * KOMORKA   # 480 x 320, wejście ViT
KANALY = 768
from laser_app.paths import PROJECT_ROOT  # noqa: E402



class Interrupted(Exception):
    """Liczenie przerwane na żądanie (przerwij() → True); nic nie zostaje zapisane."""

WYNIKI = PROJECT_ROOT / 'outputs' / 'latent_editor_sd_20261006'

PCA_SKLADOWE = (3, 4, 8, 16, 32, 128)          # wybór liczby wektorów po PCA w widokach 3D

CZASTECZKI_CO = (1, 2, 3, 4, 5, 7, 10)         # wybór liczby cząsteczek: co n-ty patch w wierszach i kolumnach


def izopowierzchnia(V, poziom, krok, kierunek='descent'):
    """Powierzchnia V = poziom na regularnej siatce o boku `krok` (jak skimage.measure.marching_cubes): (wierzchołki (n, 3) od
    rogu siatki, trójkąty (m, 3), normalne (n, 3)). Normalne wskazują spadek V ('descent') albo wzrost ('ascent').
    Bez scikit-image: pyqtgraph.isosurface i normalne z gradientu V w najbliższym węźle (obrys ogniska 3D i doliny działa
    w każdym Pythonie z pyqtgraph)."""
    try:
        from skimage.measure import marching_cubes
        v, f, n, _ = marching_cubes(V, poziom, spacing=(krok,) * 3, gradient_direction=kierunek)
        return v, f, n
    except ImportError:
        pass
    import pyqtgraph as pg
    v, f = pg.isosurface(np.ascontiguousarray(V, np.float64), float(poziom))
    if len(f) == 0:
        raise ValueError('brak powierzchni na tym poziomie')
    g = np.stack(np.gradient(np.asarray(V, np.float64)), -1)
    ix = np.clip(np.rint(v).astype(np.int64), 0, np.array(V.shape) - 1)
    n = g[ix[:, 0], ix[:, 1], ix[:, 2]]
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return v * krok, np.asarray(f, np.int64), (n if kierunek == 'ascent' else -n)


def tokeny_co(n):
    """Tokeny co n-ty patch w wierszach i w kolumnach siatki, od środka pierwszego bloku n x n (równo po obrazie).
    n = 1: wszystkie 600; 2: 150; 3: 70; 4: 35; 5: 24; 7: 12; 10: 6."""
    w = np.arange(n // 2, WIERSZE, n)
    k = np.arange(n // 2, KOLUMNY, n)
    return (w[:, None] * KOLUMNY + k[None, :]).reshape(-1)


def opisy_czasteczek():
    return [f'wszystkie {WIERSZE * KOLUMNY}' if n == 1 else f'co {n}. patch: {len(tokeny_co(n))}' for n in CZASTECZKI_CO]


_kolory = {}


def kolory_kanalow(n=KANALY):
    """[n, 4] RGBA float32: kolor wektora c z koła barw hsv (kopia, można zmieniać)."""
    if n not in _kolory:
        import matplotlib
        _kolory[n] = matplotlib.colormaps['hsv'](np.arange(n) / n).astype(np.float32)
    return _kolory[n].copy()


def kierunki(polsfera, n=KANALY):
    """n kierunków jednostkowych [n, 3]: spirala Fibonacciego, wektor c ma wysokość malejącą z c.
    polsfera=False: cała kula (z od 1 do −1), True: półsfera nad obrazem (z od 1 do 0)."""
    c = np.arange(n) + .5
    z = 1 - c / n if polsfera else 1 - 2 * c / n
    r = np.sqrt(np.clip(1 - z * z, 0, None))
    phi = np.pi * (3 - np.sqrt(5)) * np.arange(n)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1).astype(np.float32)


def pca(X, k=max(PCA_SKLADOWE)):
    """PCA wierszy X [N, C]: średnia mu [C], składowe W [C, k] (kolumny ortonormalne, od największej wariancji)
    i udział wariancji każdej składowej [k]. Wynik składowych: (x − mu) @ W. Składowe są zagnieżdżone, więc
    pierwsze m kolumn W to PCA do m wymiarów. Znak każdej składowej ustalony tak, żeby jej największy co do
    modułu ładunek był dodatni (ta sama baza przy ponownym liczeniu)."""
    X = np.asarray(X, np.float32)
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    k = min(k, len(S))
    W = Vt[:k].T.copy()
    znak = np.sign(W[np.abs(W).argmax(0), np.arange(k)]); znak[znak == 0] = 1
    W *= znak
    wariancja = S ** 2
    return mu, W.astype(np.float32), (wariancja[:k] / max(float(wariancja.sum()), 1e-12)).astype(np.float32)


def baza_pca_rgb(X):
    """PCA 3 składowych latentu X [N, 768] do kolorów RGB: (mu, W [768, 3], lo [3], hi [3]); lo / hi = 1. / 99. centyl
    wyników, więc kolory nie przepalają się na pojedynczych tokenach (jak wizualizacje cech DINOv2 i FeatUp)."""
    X = np.asarray(X, np.float32)
    mu, W, _ = pca(X, 3)
    S = (X - mu) @ W
    return mu, W, np.percentile(S, 1, 0).astype(np.float32), np.percentile(S, 99, 0).astype(np.float32)


def kolor_pca_rgb(Z, baza):
    """Latent [600, 768] -> kolory [600, 3] w 0..1: 3 główne składowe jako R, G, B."""
    if baza is None:
        return np.full((len(Z), 3), .5, np.float32)
    mu, W, lo, hi = baza
    S = (np.asarray(Z, np.float32) - mu) @ W
    return np.clip((S - lo) / np.maximum(hi - lo, 1e-9), 0, 1).astype(np.float32)


def kolor_podobienstwa(Z, tok):
    """Kosinus latentu każdego patcha z patchem tok (po odjęciu średniej klatki) -> kolory [600, 3] (RdBu: czerwony
    = podobny, niebieski = przeciwny); bez wybranego patcha szary."""
    if tok is None:
        return np.full((len(Z), 3), .35, np.float32)
    import matplotlib
    X = np.asarray(Z, np.float32); X = X - X.mean(0)
    n = np.linalg.norm(X, axis=1)
    c = X @ X[tok] / np.maximum(n * n[tok], 1e-12)
    return matplotlib.colormaps['RdBu_r']((np.clip(c, -1, 1) + 1) / 2)[:, :3].astype(np.float32)


def tokeny_w_prostokacie(x0, y0, x1, y1):
    """Patche (siatka 20 x 30 po 16 px), których środek leży w prostokącie pikseli obrazu 480 x 320."""
    xa, xb = sorted((x0, x1)); ya, yb = sorted((y0, y1))
    t = np.arange(600)
    cx, cy = (t % 30) * 16 + 8, (t // 30) * 16 + 8
    return t[(cx >= xa) & (cx <= xb) & (cy >= ya) & (cy <= yb)]


def rysuj_zaznaczenie(ax, tokeny, kolory=None):
    """Zaznaczone patche jako półprzezroczysta nakładka na obrazie (matplotlib, extent 0..480 x 0..320):
    żółta albo, z kolorami [n, 3+] (jak cząstki w przestrzeni makro i krzywe wykresów), każdy patch w swoim kolorze
    z ramką."""
    if tokeny is None or len(tokeny) == 0:
        return
    m = np.zeros((20, 30, 4), np.float32)
    r, c = np.divmod(np.asarray(tokeny, int), 30)
    if kolory is None:
        m[r, c] = (1, .9, .1, .38)
    else:
        from matplotlib.patches import Rectangle
        k = np.asarray(kolory, np.float32)[:, :3]
        m[r, c, :3] = k; m[r, c, 3] = .42
        for rr, cc, kk in zip(r, c, k):
            ax.add_patch(Rectangle((cc * 16, rr * 16), 16, 16, fill=False, ec=tuple(kk), lw=1.1, zorder=6))
    ax.imshow(m, extent=(0, 480, 320, 0), interpolation='nearest', zorder=5)


def ctrl_wcisniety():
    """Czy wciśnięty jest Ctrl (zaznaczanie: dodaj / przełącz zamiast zastąp)."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)


def polacz_zaznaczenie(stare, nowe, dodaj):
    """Nowe zaznaczenie patchy: zastąp (dodaj=False) albo dołącz do starego (Ctrl); kolejność dodawania zostaje,
    więc kolory już zaznaczonych patchy się nie zmieniają."""
    nowe = [int(t) for t in np.atleast_1d(np.asarray(nowe, int))]
    if not dodaj or stare is None:
        return np.asarray(list(dict.fromkeys(nowe)), int)
    st = [int(t) for t in stare]
    return np.asarray(st + [t for t in dict.fromkeys(nowe) if t not in set(st)], int)


def przelacz_w_zaznaczeniu(stare, tok):
    """Ctrl + klik: patch tok dochodzi do zaznaczenia albo z niego wypada."""
    st = [] if stare is None else [int(t) for t in stare]
    return np.asarray([t for t in st if t != tok] if tok in st else st + [int(tok)], int)


def paleta_zaznaczenia(n):
    """Osobne kolory RGBA [n, 4] dla n zaznaczonych patchy: tab10, tab20, a dla większej liczby tęcza."""
    import matplotlib
    if n <= 10:
        k = matplotlib.colormaps['tab10'](np.arange(n))
    elif n <= 20:
        k = matplotlib.colormaps['tab20'](np.r_[0:20:2, 1:20:2][:n])        # najpierw ciemniejsze, potem jasne
    else:
        k = matplotlib.colormaps['hsv'](np.arange(n) / n * .92)
    return np.asarray(k, np.float32).reshape(n, 4)
