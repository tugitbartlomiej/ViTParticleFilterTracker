"""Prognoza latentu tożsamości narzędzia równaniem ruchu (PLAN.md, wersja 4.1; docs/23_METODA_PRZEWIDYWANIA_LATENTU.md).

Siła tożsamości sondy s (rzut latentu narzędzia na wektor tożsamości) porusza się jak cząstka z bezwładnością:
    Δs(t+1) = a·Δs(t) + b·F(t),   a = 0,83, b = 0,98 (zmierzone na 10 ziarnach),
gdzie F(t) to siła paczki: zmiana siły tożsamości sond po wirtualnym kroku AdamW z gradientu paczki. Prognoza na H
kroków przy stałej sile F (średnia siła paczek losowanych z OBECNEGO zbioru uczącego, czyli klatek, które już w nim są)
i prędkości początkowej v0 (nachylenie historii siły z ostatnich 100 kroków):
    Δs(H) = v0·a(1 − a^H)/(1 − a) + b·F/(1 − a)·(H − a(1 − a^H)/(1 − a)).
Reguła o przyszłych klatkach: używane są tylko stany do chwili t, obecny zbiór i obecny stan optymalizatora; paczki
przyszłych kroków nie są znane i nie wchodzą.

Trafność: po obrocie prognoza (na 100 i 500 kroków) kontra zmierzona siła, obok „braku zmiany” i prognozy Koopmana albo
Langevina z wersji 3 (`sila.prognozy`, wybór testem wstecznym).
"""
from __future__ import annotations

import numpy as np
import torch

import rownanie as RW
import sila as SI

A, B = 0.83, 0.98


def sila_paczek(u, zb, sondy, obecn, wek, n_paczek, rng, paczka=32):
    """Średnia siła paczki F (180,) na siłę tożsamości sond z narzędziem, z n_paczek losowych paczek obecnego zbioru."""
    s0 = SI.sila(RW.latent_narzedzia_sond(u, sondy)[obecn], wek)
    F = np.zeros_like(s0, dtype=np.float64)
    for _ in range(n_paczek):
        idx = rng.choice(len(zb), paczka, replace=False)
        d, _ = RW.czesci_kroku(u, RW.gradient_paczki(u, zb, idx))
        F += SI.sila(RW.latent_narzedzia_sond(u, sondy, d)[obecn], wek) - s0
    return F / n_paczek


def sila_zbioru(u, zb, sondy, obecn, wek, paczka=32):
    """Średnia siła paczki liczona z CAŁEGO obecnego zbioru uczącego: gradienty kolejnych paczek po 32 klatki, każdy
    obcięty do normy 1 jak w uczeniu, uśrednione; z tej średniej część kroku AdamW i zmiana siły tożsamości sond.
    Szum pojedynczych paczek znosi się, zostaje wspólny kierunek ruchu (wartość oczekiwana siły paczki)."""
    acc, n = None, 0
    kolej = np.arange(len(zb))
    for s in range(0, len(kolej) - paczka + 1, paczka):
        g = RW.gradient_paczki(u, zb, kolej[s:s + paczka])
        if acc is None:
            acc = {k: v.clone() for k, v in g.items()}
        else:
            for k, v in g.items():
                acc[k] += v
        n += 1
    g = {k: v / n for k, v in acc.items()}
    d, _ = RW.czesci_kroku(u, g)
    z0 = RW.latent_narzedzia_sond(u, sondy)
    return SI.sila(RW.latent_narzedzia_sond(u, sondy, d)[obecn], wek) - SI.sila(z0[obecn], wek), n


def predkosc(S, co=10, okno=10):
    """Nachylenie (na krok) siły każdej sondy z ostatnich `okno` zapisów (zapis co `co` kroków)."""
    y = np.asarray(S[-okno:], np.float64)
    t = np.arange(len(y), dtype=np.float64) * co
    t = t - t.mean()
    return (t[:, None] * (y - y.mean(0))).sum(0) / (t ** 2).sum()


def zmiana(v0, F, H, a=A, b=B):
    """Przewidziana zmiana siły po H krokach."""
    ger = a * (1 - a ** H) / (1 - a)
    return v0 * ger + b * F / (1 - a) * (H - ger)


def ocena(pred, teraz, real):
    """Błąd prognoz zmiany siły 180 sond: RMSE / RMSE braku zmiany (poniżej 1 = lepiej niż brak zmiany) i zgodność znaku
    zmiany na sondach o zmianie powyżej mediany."""
    d = real - teraz
    e0 = float(np.sqrt((d ** 2).mean())) + 1e-12
    duze = np.abs(d) >= np.median(np.abs(d))
    out = dict(brak_zmiany_rmse=e0, zmiana_srednia=float(d.mean()))
    for k, p in pred.items():
        dp = p - teraz
        out[k] = dict(wzgl_braku_zmiany=float(np.sqrt(((p - real) ** 2).mean()) / e0),
                      znak=float((np.sign(dp[duze]) == np.sign(d[duze])).mean()),
                      zmiana_przewidziana_srednia=float(dp.mean()))
    return out


@torch.no_grad()
def sila_wycinkow(u, obrazy176, maski176, wek):
    """Siła tożsamości narzędzia na wycinkach 176×176 (jak sondy): latent MLP ostatniego bloku uśredniony po tokenach
    narzędzia (udział maski w tokenie ≥ 0,25), rzut na wektor tożsamości. Autocast bf16 jak `Uczen.latent_sond`."""
    net = u.net; net.eval(); net.zapisuj_latent(True)
    X = torch.from_numpy(np.ascontiguousarray(obrazy176)).permute(0, 3, 1, 2).float() / 255.
    g = 176 // 16
    tn = torch.from_numpy(maski176.reshape(len(maski176), g, 16, g, 16).mean((2, 4)).reshape(len(maski176), -1) >= .25)
    out = []
    for s in range(0, len(X), 48):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            net(X[s:s + 48].to(u.dev))
        lat = net._latent.float(); a = tn[s:s + 48].to(u.dev).float()
        out.append(((lat * a[..., None]).sum(1) / a.sum(1, keepdim=True).clamp(min=1)).cpu().numpy())
    net.zapisuj_latent(False)
    return SI.sila(np.concatenate(out), wek)
