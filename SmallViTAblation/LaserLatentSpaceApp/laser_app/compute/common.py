"""Stałe i funkcje wspólne dla widoków aplikacji: siatka patchy małego ViT, kolory i kierunki 768 kanałów.

Latent małego ViT to 600 tokenów = siatka 20 x 30 patchy 16 x 16 pikseli na wejściu 480 x 320.
Kanał c ma w każdym widoku ten sam kolor (koło barw hsv) i ten sam kierunek (spirala Fibonacciego).
"""

import numpy as np

KOMORKA = 16                                   # piksele wejścia ViT na patch
WIERSZE, KOLUMNY = 20, 30                      # siatka patchy, 600 tokenów
SZER, WYS = KOLUMNY * KOMORKA, WIERSZE * KOMORKA   # 480 x 320, wejście ViT
KANALY = 768


class Interrupted(Exception):
    """Liczenie przerwane na żądanie (przerwij() → True); nic nie zostaje zapisane."""


PCA_SKLADOWE = (3, 4, 8, 16, 32, 128)          # wybór liczby wektorów po PCA w widokach 3D


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


