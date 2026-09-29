"""Wczytanie latentu sond: zapis z Edena co 10 kroków i nagranie lokalne co krok (PLAN.md).

Pułapki:
- Eden (`maly_vit_wektor_tozsamosci/wyniki_eden/petla_s*/`): `wspolny/k01` ma 51 zapisów (krok 0..500), pozostałe
  pliki po 50; ramię `bez` od kroku 1010 do 4500 uczy się na stałej bazie 1500 klatek. Latent w float16.
- Nagranie (`nagranie_s*/sondy_lat.npy`): wiersz t = stan PO t aktualizacjach (wiersz 0 = przed uczeniem); paczka
  z wiersza t paczek (`paczka_*`, t = 0..T−1) prowadzi ze stanu t do stanu t+1.
- Do kroku 300 lr rośnie liniowo (rozgrzewka), więc dopasowanie i test biorą kroki ≥ 1000, jak w pętli.
- Kanał 0 = tokeny narzędzia, kanał 1 = tło; sondy bez narzędzia (12 z 192) mają kanał narzędzia pusty (zera).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EDEN = HERE.parent / 'maly_vit_wektor_tozsamosci' / 'wyniki_eden'
ZIARNA_EDEN = tuple(range(47, 57))


def eden(s: int, ramie: str = 'bez', kanal: int = 0):
    """Zwraca (kroki (T,), Z (T, 192, 768) float32) dla ramienia od kroku 1010 do 4500 (bez rozgrzewki)."""
    d = EDEN / f'petla_s{s}' / ramie
    pl = [d / f'k{k:02d}' / 'LATENT.npz' for k in range(3, 10)]
    K = np.concatenate([np.load(p)['kroki'] for p in pl])
    Z = np.concatenate([np.load(p)['latent'][:, :, kanal] for p in pl]).astype(np.float32)
    return K, Z


def obecnosc(s: int = 47):
    """Maska 192 sond z narzędziem (z META nagrania albo z opisu Edena)."""
    m = HERE / f'nagranie_s{s}' / 'META.json'
    if m.exists():
        return np.array(json.loads(m.read_text(encoding='utf-8'))['obecn'])
    return None


def nagranie(s: int = 47):
    """Słownik tablic nagrania (memmap dla dużych)."""
    d = HERE / f'nagranie_s{s}'
    out = {p.stem: np.load(p, mmap_mode='r') for p in d.glob('*.npy')}
    out['META'] = json.loads((d / 'META.json').read_text(encoding='utf-8'))
    return out


def umiejetnosc(y, p):
    """100 · (1 − SSE / SS) przyrostu względem braku zmiany."""
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(100 * (1 - ((y - p) ** 2).sum() / (y ** 2).sum()))
