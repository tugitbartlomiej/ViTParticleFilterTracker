"""Prognoza latentu tożsamości narzędzia na jeden krok naprzód (wersja 4.2, PLAN.md).

Równanie (`../../maly_vit_tozsamosc_jeden_krok/WYNIKI.md`, commit a5813796; `docs/23`): przyrost latentu sondy między
chwilą t a t+1 to suma dwóch pchnięć, oba ze współczynnikiem 1, bez dopasowania:
    Δz(t+1) = S_pęd(t) + S_paczka(t)
S_pęd: zmiana latentu sond, gdy wagi przesunąć tylko o część kroku AdamW pochodzącą z pamięci optymalizatora (paczki
do t−1); S_paczka: to samo dla części kroku z gradientu paczki kroku t (klatki, które tworzą latent t+1). Zanik wag
(S_zanik) jest pomijalny i tu go nie ma. Siła tożsamości (rzut latentu narzędzia na wektor tożsamości) jest liniowa w
latencie, więc to samo równanie przewiduje jej przyrost. Na nagraniach s47 i s48: 95,2 i 95,6 procent umiejętności dla
siły tożsamości (cały latent 95,3 i 95,5), gdzie umiejętność = 100 · (1 − SSE / SS) przyrostu wobec braku zmiany.

Tylko jeden krok: pchnięcia dalszych kroków zależą od przyszłych wag. W pętli równanie służy do dwóch rzeczy:
  1. prognoza na następny krok bez nowych klatek (pęd + średnia paczka obecnego zbioru) wskazuje kotwice, w których
     tożsamość narzędzia będzie po kroku najsłabsza;
  2. prognoza z paczką złożoną z kandydatów mówi, gdzie będzie latent tożsamości po kroku, jeśli uczyć na tych klatkach;
     pęd jest wspólny dla wszystkich kandydatów, więc o różnicy między nimi decyduje S_paczka.
`Sprawdzian` sprawdza równanie na prawdziwych krokach uczenia każdego ramienia (ta sama paczka, po kroku pomiar).
"""
from __future__ import annotations

import numpy as np

import rownanie as RW
import sila as SI


def lat(u, sondy, d=None):
    """Latent narzędzia 192 sond (fp32, tryb ewaluacji) przy wagach θ (+ d)."""
    return RW.latent_narzedzia_sond(u, sondy, d)


def ds(dz, wek):
    """Przyrost siły tożsamości z przyrostu latentu narzędzia (..., 768); siła jest liniowa w latencie."""
    return (np.asarray(dz, np.float64) / wek['sd']) @ wek['w']


def umiejetnosc(y, p):
    """100 · (1 − SSE / SS) przyrostu wobec braku zmiany (jak `maly_vit_rownanie_macierze/dane.py`)."""
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(100 * (1 - ((y - p) ** 2).sum() / max((y ** 2).sum(), 1e-30)))


def pchniecie_pedu(u, sondy, obecn, wek, z0):
    """S_pęd(t) na siłę tożsamości 180 sond z narzędziem (mianownik AdamW bez bieżącej paczki; jej udział w
    mianowniku to 0,1 procent przy β2 = 0,999)."""
    _, ped = RW.czesci_kroku(u, None)
    return ds((lat(u, sondy, ped) - z0)[obecn], wek)


def pchniecie_paczki(u, zb, idx, sondy, obecn, wek, z0, odbicia=(False, True)):
    """S_paczka(t) grupy klatek zb[idx] na siłę tożsamości 180 sond: średnia po odbiciu (uczenie odbija połowę
    klatek) i szum = połowa różnicy między wersją z odbiciem i bez. Zwraca (średnia (180,), szum (180,))."""
    P = []
    for o in odbicia:
        nowa, _ = RW.czesci_kroku(u, RW.gradient_paczki(u, zb, np.asarray(idx), o))
        P.append(ds((lat(u, sondy, nowa) - z0)[obecn], wek))
    P = np.stack(P)
    return P.mean(0), (np.abs(P[0] - P[-1]) / 2 if len(P) > 1 else np.zeros_like(P[0]))


def prognoza_bez_nowych(u, zb, sondy, obecn, wek, rng, n_paczek, paczka=32):
    """Prognoza na krok t+1 bez nowych klatek: s(t) + S_pęd(t) + średnie S_paczka z n_paczek losowych paczek obecnego
    zbioru (losowe odbicie jak w uczeniu). Zwraca dict(z0, s0, ped, paczka, s1) - wektory (180,)."""
    z0 = lat(u, sondy)
    s0 = SI.sila(z0[obecn], wek).astype(np.float64)
    ped = pchniecie_pedu(u, sondy, obecn, wek, z0)
    acc = np.zeros_like(s0)
    for _ in range(n_paczek):
        idx = rng.choice(len(zb), min(paczka, len(zb)), replace=False)
        acc += pchniecie_paczki(u, zb, idx, sondy, obecn, wek, z0, odbicia=(bool(rng.random() < .5),))[0]
    pacz = acc / max(n_paczek, 1)
    return dict(z0=z0, s0=s0, ped=ped, paczka=pacz, s1=s0 + ped + pacz)


class Sprawdzian:
    """Równanie jednego kroku na prawdziwych krokach uczenia. Przed krokiem: z0, S_pęd i S_paczka z dokładnie tej
    paczki, którą uczenie zaraz użyje (po odbiciu i zmianie jasności), z mianownikiem AdamW liczonym z jej gradientem.
    Po kroku: zmierzony przyrost (z uwzględnieniem wszystkiego, co krok zmienił, także buforów BatchNorm).
    Kroki: pierwsze `na_poczatku` każdego wywołania `kroki` i co `co`-ty. Wynik zbierany osobno dla obrotu."""

    def __init__(self, sondy, obecn, wek, na_poczatku=10, co=50):
        self.sondy, self.obecn, self.wek = sondy, obecn, wek
        self.na_poczatku, self.co = na_poczatku, co
        self.zeruj()

    def zeruj(self):
        self.y, self.p_rown, self.p_ped = [], [], []                  # siła tożsamości (n, 180)
        self.sse = dict(rownanie=0.0, ped=0.0); self.ss = 0.0         # cały latent 180 × 768
        self.kroki = []

    def czy(self, j):
        return j < self.na_poczatku or (j + 1) % self.co == 0

    def przed(self, u, x, heat, ltrb, poz):
        z0 = lat(u, self.sondy)
        nowa, ped = RW.czesci_kroku(u, RW.gradient_tensorow(u, x, heat, ltrb, poz))
        self._z0 = z0
        self._pn = lat(u, self.sondy, nowa) - z0
        self._pp = lat(u, self.sondy, ped) - z0

    def po(self, u):
        o = self.obecn
        dz = (lat(u, self.sondy) - self._z0)[o].astype(np.float64)
        pr = (self._pn + self._pp)[o].astype(np.float64); pp = self._pp[o].astype(np.float64)
        self.ss += float((dz ** 2).sum())
        self.sse['rownanie'] += float(((dz - pr) ** 2).sum()); self.sse['ped'] += float(((dz - pp) ** 2).sum())
        self.y.append(ds(dz, self.wek)); self.p_rown.append(ds(pr, self.wek)); self.p_ped.append(ds(pp, self.wek))
        self.kroki.append(int(u.krok))

    def wynik(self):
        if not self.y:
            return None
        y, pr, pp = (np.stack(v) for v in (self.y, self.p_rown, self.p_ped))
        out = dict(krokow=len(y), kroki=self.kroki,
                   tozsamosc=dict(rownanie=umiejetnosc(y, pr), sam_ped=umiejetnosc(y, pp)),
                   caly_latent=dict(rownanie=float(100 * (1 - self.sse['rownanie'] / self.ss)),
                                    sam_ped=float(100 * (1 - self.sse['ped'] / self.ss))),
                   przyrost_sredni=dict(zmierzony=float(y.mean()), rownanie=float(pr.mean())),
                   znak=float((np.sign(y) == np.sign(pr)).mean()))
        self.zeruj()
        return out
