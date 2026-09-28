"""Latent tożsamości narzędzia w małym ViT (definicje użytkownika, PLAN.md).

Neuron = jedna z 768 aktywacji MLP ostatniego bloku. Na sondzie liczymy jego średnią po tokenach narzędzia i po tokenach
tła. Negatywy: tokeny tła wszystkich 192 sond oraz obszar narzędzia w sondach „empty_black” (narzędzie usunięte).
Neuron „zapala się” na sondzie, gdy aktywacja na narzędziu przekracza średnią negatywów + 2 odchylenia.

  rdzeń (core latent)  neurony, które zapalają się ZAWSZE, gdy narzędzie jest na obrazie: w każdym z 15 wariantów
                       (obroty, długość, skala, przesunięcia, jasność, perspektywa, czarne i żółte tło) na ≥ 75 %
                       źródeł, a na negatywach najwyżej w 10 %;
  pełny latent         neurony wybiórcze dla narzędzia (negatywy ≤ 10 %), zapalone na ≥ 30 % sond z narzędziem —
                       rdzeń plus neurony zależne od wariantu (kąt, tło, rozbłysk).
"""
from __future__ import annotations

import numpy as np

WARIANTY_Z = ('original', 'rotation90', 'rotation180', 'rotation270', 'black', 'yellow', 'scale08', 'length08',
              'shift_left', 'shift_right', 'shift_up', 'shift_down', 'foreground_dim', 'foreground_bright',
              'perspective_mild')
PROG_SD, NEG_MAX, RDZEN_MIN, PELNY_MIN = 2.0, 0.10, 0.75, 0.30


def statystyki(L, obecn, opis):
    """L: (192, 2, 768) latent jednej chwili (średnia kilku zapisów). Zwraca słownik z progami i zbiorami neuronów."""
    L = np.asarray(L, np.float64)
    nar, tlo = L[:, 0], L[:, 1]
    neg = np.concatenate([tlo, nar[~obecn]])
    mu, sd = neg.mean(0), neg.std(0) + 1e-6
    prog = mu + PROG_SD * sd
    zap = nar[obecn] > prog                                         # (180, 768)
    frac_pos, frac_neg = zap.mean(0), (neg > prog).mean(0)
    war = np.array([o['wariant'] for o in opis])[obecn]
    zw = np.stack([zap[war == v].mean(0) for v in WARIANTY_Z])      # (15, 768) udział źródeł z zapalonym neuronem
    selekt = frac_neg <= NEG_MAX
    rdzen = np.where(selekt & (zw.min(0) >= RDZEN_MIN))[0]
    pelny = np.where(selekt & (frac_pos >= PELNY_MIN))[0]
    return dict(mu=mu, sd=sd, prog=prog, frac_pos=frac_pos, frac_neg=frac_neg, zapal_warianty=zw, rdzen=rdzen,
                pelny=pelny, zap=zap, war=war)


def podsumowanie(st):
    """Liczby do raportu obrotu: wielkość rdzenia i pełnego latentu, domknięcie tożsamości per wariant."""
    r, p = st['rdzen'], st['pelny']
    ref = r if len(r) else p
    dom = {}
    for v in WARIANTY_Z:
        m = st['war'] == v
        dom[v] = float((st['zap'][m][:, ref].mean(1) >= .5).mean()) if len(ref) else 0.0
    return dict(rdzen=int(len(r)), pelny=int(len(p)), domkniecie_wariantow=dom,
                domkniecie_srednie=float(np.mean(list(dom.values()))),
                najslabszy_wariant=min(dom, key=dom.get) if dom else None)


def sledzone(st, minimum=32):
    """Neurony śledzone przez Koopmana: pełny latent; gdy za mały — 32 najbardziej wybiórcze."""
    if len(st['pelny']) >= minimum:
        return st['pelny']
    wyn = st['frac_pos'] - st['frac_neg']
    return np.sort(np.argsort(-wyn)[:minimum])


def stan(L_hist, st, neur, obecn):
    """Historia latentu (T,192,2,768) → stan Koopmana (T, P_z_narzędziem, M): aktywacje narzędzia śledzonych neuronów
    wystandaryzowane stałymi mu, sd z chwili identyfikacji."""
    nar = np.asarray(L_hist[:, obecn, 0][..., neur], np.float64)
    return (nar - st['mu'][neur]) / st['sd'][neur]


# ---------------------------------------------------------------- wersja 2 (maly_vit_wektor_tozsamosci)
WSZYSTKIE = np.arange(768)


def wektor(L, obecn):
    """Wektor tożsamości na wszystkich 768 neuronach: średnia aktywacja na narzędziu minus średnia negatywów,
    w jednostkach sd negatywów, znormalizowany do długości 1. L: (192, 2, 768)."""
    L = np.asarray(L, np.float64)
    nar, tlo = L[:, 0], L[:, 1]
    neg = np.concatenate([tlo, nar[~obecn]])
    u = (nar[obecn].mean(0) - neg.mean(0)) / (neg.std(0) + 1e-6)
    return u / np.linalg.norm(u)
