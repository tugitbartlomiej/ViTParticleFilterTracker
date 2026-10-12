"""Wczytanie gęstej historii tokenów małego ViT (bieg Eden 1818732), JEDNO miejsce.

Pułapki danych:
- LATENT_GESTY.npy ma 22,1 GB. Czytać tylko przez mmap i po jednym obrazie (1,85 GB na obraz w RAM).
- Oś 0 to krok uczenia: indeks t = krok t (stan 0 przed uczeniem, potem po każdej aktualizacji 1..1000).
- Tokeny w kolejności i = y*30 + x na siatce 20x30.
- Pomiar w trybie eval, ale statystyki BatchNorm zmieniają się w każdym kroku uczenia. Ruch tokenów to zmiana wag
  oraz BN, a nie samych wag (patrz OPIS_DANYCH.md w katalogu źródłowym).
- maski.npz: 'tool' = token przecina maskę narzędzia co najmniej jednym pikselem (218 tokenów łącznie),
  'background' = tło. Token o tym samym indeksie w innym obrazie to inny fragment sceny.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ZRODLO = (Path(__file__).resolve().parents[2] / 'maly_vit_ruch_czastek_gesta_historia' / 'wyniki' / 'pelny'
          / 'dane')
T, N_OBR, N_TOK, D = 1001, 12, 600, 768


def latent():
    """Pamięć mapowana (1001, 12, 600, 768) float32. Tylko do odczytu."""
    a = np.load(ZRODLO / 'LATENT_GESTY.npy', mmap_mode='r', allow_pickle=False)
    assert a.shape == (T, N_OBR, N_TOK, D) and a.dtype == np.float32, a.shape
    return a


def obraz(lat, i):
    """Pełna historia jednego obrazu w RAM: (1001, 600, 768) float32."""
    return np.ascontiguousarray(lat[:, i])


def maski():
    m = np.load(ZRODLO / 'maski.npz')
    return m['tool'].astype(bool), m['background'].astype(bool)
