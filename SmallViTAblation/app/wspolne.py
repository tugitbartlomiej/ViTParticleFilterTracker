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
WYNIKI = Path(__file__).resolve().parents[1] / 'outputs' / 'latent_editor_sd_20261006'

_kolory = None


def kolory_kanalow():
    """[768, 4] RGBA float32: kolor kanału c z koła barw hsv (kopia, można zmieniać)."""
    global _kolory
    if _kolory is None:
        import matplotlib
        _kolory = matplotlib.colormaps['hsv'](np.arange(KANALY) / KANALY).astype(np.float32)
    return _kolory.copy()


def kierunki(polsfera):
    """768 kierunków jednostkowych [768, 3]: spirala Fibonacciego, kanał c ma wysokość malejącą z c.
    polsfera=False: cała kula (z od 1 do −1), True: półsfera nad obrazem (z od 1 do 0)."""
    c = np.arange(KANALY) + .5
    z = 1 - c / KANALY if polsfera else 1 - 2 * c / KANALY
    r = np.sqrt(np.clip(1 - z * z, 0, None))
    phi = np.pi * (3 - np.sqrt(5)) * np.arange(KANALY)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1).astype(np.float32)
