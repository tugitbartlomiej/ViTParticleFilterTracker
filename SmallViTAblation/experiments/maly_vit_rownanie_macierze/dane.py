"""Wczytanie nagrań uczenia małego ViT dla równania z uczonymi macierzami (PLAN.md).

Nagrania z `../maly_vit_rownanie_latentu/nagranie.py` (katalogi `nagranie_s<ziarno>_sily/`). Pułapki:
- wiersz t w `sondy_lat` to stan PO t aktualizacjach; paczka z wiersza t (`paczka_*`) prowadzi ze stanu t do t+1
  i jest znana w chwili t (jej gradient liczy to samo przejście wstecz, które i tak robi uczenie);
- `paczka_g`: gradient straty względem latentu obrazu (średnia po tokenach), float32, rzędu 1e-5 na kanale
  narzędzia i 1e-7 na całym obrazie, więc każdą z 1536 liczb standaryzujemy osobno statystykami z kroków uczenia;
- kanał 0 = tokeny narzędzia (zera, gdy obraz go nie ma), kanał 1 = tło sondy albo cały obraz paczki;
- sondy: 12 wycinków × 16 wersji w kolejności wycinków; wycinki 0–7 mają rolę 'fit', 8–11 'holdout'; wersja
  empty_black (indeks 8 w każdym wycinku) nie ma narzędzia i nie wchodzi ani do uczenia, ani do oceny;
- `sila_nowa` zaczyna się od kroku META['sily_od'] (s47: 900, s48: 2000); lr jest stałe (5e-4) od kroku 300.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parent
NAGRANIA = HERE.parent / 'maly_vit_rownanie_latentu'
OD = 600        # pierwszy wczytany krok: placebo sięga 250 kroków wstecz, W 15 paczek, historia 34 kroki przed 1000


def wektor_tozsamosci(L, obecn):
    """Kierunek tożsamości jak w `maly_vit_wektor_tozsamosci/wektor_tozsamosci_21ep.py` (LDA ze ściąganiem).

    L (192, 2, 768): średni latent sond. Pozytywy: kanał narzędzia sond z narzędziem. Negatywy: tło wszystkich sond
    i obszar narzędzia w empty_black. Standaryzacja negatywami, wektor o długości 1.
    """
    X, y = [], []
    for i in range(len(L)):
        X.append(L[i, 0]); y.append(int(obecn[i]))
        X.append(L[i, 1]); y.append(0)
    X, y = np.array(X, np.float64), np.array(y)
    neg = X[y == 0]
    mu, sd = neg.mean(0), neg.std(0) + 1e-6
    w = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit((X - mu) / sd, y).coef_[0]
    return dict(mu=mu, sd=sd, w=w / np.linalg.norm(w))


class Nagranie:
    """Jedno nagranie od kroku `od`. Indeks i w tablicach odpowiada krokowi od + i."""

    def __init__(self, s: int, od: int = OD):
        k = NAGRANIA / f'nagranie_s{s}_sily'
        self.meta = json.loads((k / 'META.json').read_text(encoding='utf-8'))
        self.s, self.od = s, od
        L = np.load(k / 'sondy_lat.npy', mmap_mode='r')
        self.z = torch.from_numpy(np.ascontiguousarray(L[od:, :, 0]))        # (T + 1 − od, 192, 768) f16
        self.zt = torch.from_numpy(np.ascontiguousarray(L[od:, :, 1]))
        self.p_lat = torch.from_numpy(np.ascontiguousarray(np.load(k / 'paczka_lat.npy', mmap_mode='r')[od:]))
        self.p_g = torch.from_numpy(np.ascontiguousarray(np.load(k / 'paczka_g.npy', mmap_mode='r')[od:]))
        self.p_ma = torch.from_numpy(np.load(k / 'paczka_ma_narz.npy')[od:])
        self.lr = torch.from_numpy(np.load(k / 'lr.npy')[od:])
        self.sily_od = self.meta['sily_od']
        self._S = np.load(k / 'sila_nowa.npy', mmap_mode='r')
        self.obecn = np.array(self.meta['obecn'])
        self.rola = np.array([d['rola'] for d in self.meta['opis']])
        self.tozs = wektor_tozsamosci(L[960:1001:10].astype(np.float64).mean(0), self.obecn)   # krok 1000, 5 zapisów

    def sondy(self, rola: str):
        return np.where(self.obecn & (self.rola == rola))[0]

    def sila(self, t0: int, t1: int, sondy, rodzaj: str = 'nowa'):
        """Siła w krokach t0..t1−1: (t1 − t0, len(sondy), 768) float32.

        'nowa' = paczka kroku t, czyli klatki, których sieć jeszcze nie widziała: w prognozie NIEDOZWOLONA (reguła
        użytkownika z 2026-09-29), tylko jako odniesienie. 'ped' = pęd AdamW, suma gradientów paczek do t−1: dozwolony
        (mianownik AdamW zawiera nowy gradient z wagą 1 − β2 = 0,001, przeciek pomijalny, ale nazwany)."""
        assert t0 >= self.sily_od, (t0, self.sily_od)
        S = self._S if rodzaj == 'nowa' else np.load(NAGRANIA / f'nagranie_s{self.s}_sily' / f'sila_{rodzaj}.npy', mmap_mode='r')
        return np.asarray(S[t0 - self.sily_od:t1 - self.sily_od][:, sondy], np.float32)


def umiejetnosc(y, p):
    """100 · (1 − SSE / SS) przyrostu względem braku zmiany."""
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(100 * (1 - ((y - p) ** 2).sum() / (y ** 2).sum()))
