"""Odtworzenie paczek uczenia małego ViT z ziarna: które obrazy bazy, odbicie i jasność / kontrast w każdym kroku.

Losowanie jak w trening.Uczen.kroki (experiments/maly_vit_od_zera i kopie): kolejka indeksów uzupełniana
permutacją bazy, gdy ma mniej niż 32 elementy; w każdym kroku kolejno: (permutacja, jeśli potrzebna), 32 losowania
odbicia rng.random(32) < 0,5, potem rng.uniform(0,8, 1,2, (32, 2)) (kontrast, jasność). Generator default_rng(ziarno)
tworzony tuż przed pętlą, nic innego go nie zużywa (bieg.py gęstej historii).

Warianty: reset_co = N zeruje kolejkę co N kroków (nowe wywołanie kroki(), jak w nagranie.py), nowe_ziarno = (krok, z)
podmienia generator od danego kroku (nagranie.py: default_rng(1000 s + 3) od kroku 1000).
"""
import numpy as np

PACZKA = 32


def odtworz_paczki(kroki, n_bazy=1500, ziarno=47, reset_co=None, nowe_ziarno=None):
    """Zwraca (idx [kroki, 32] int, odb [kroki, 32] bool, a [kroki, 32, 2] float32)."""
    rng = np.random.default_rng(ziarno)
    kol = []
    I = np.zeros((kroki, PACZKA), np.int64); O = np.zeros((kroki, PACZKA), bool)
    A = np.zeros((kroki, PACZKA, 2), np.float32)
    for t in range(kroki):
        if reset_co and t % reset_co == 0:
            kol = []
        if nowe_ziarno is not None and t == nowe_ziarno[0]:
            rng = np.random.default_rng(nowe_ziarno[1])
        if len(kol) < PACZKA:
            kol += rng.permutation(n_bazy).tolist()
        I[t] = kol[:PACZKA]; kol = kol[PACZKA:]
        O[t] = rng.random(PACZKA) < .5
        A[t] = rng.uniform(.8, 1.2, (PACZKA, 2)).astype(np.float32)
    return I, O, A
