"""Kontrole dla testów „jaka to substancja” (dh.py, 2026-10-11) na układach odniesienia: szkło BD (duży bieg stały, osie `_duze`,
φ = 0,58 / 0,61 / 0,62 / 0,64; kontrola dodatnia — prawdziwe oddziaływania między cząstkami) i OU-B (dołki stygnące o rozrzucie
sztywności 10×, niezależny szum; kontrola ujemna — brak oddziaływań, brak wspólnego napędu). Te same funkcje co w pokretla_analiza.py
na Edenie dla tokenów ViT: trwałość ruchliwości (zamrożona czy przemijająca), korelacja krzyżowa położeń par z opóźnieniem
(Meiners–Quake / Moran), rozbieganie par (turbulencja), rozkład energii kroku, dwupunktowa korelacja przesunięć.
Uruchomienie: F:/Python/3.13/python.exe substancja_kontrola.py  → SUBSTANCJA_KONTROLA.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dh  # noqa: E402
import odciski as O  # noqa: E402

OSIE = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui'
Q16 = 65535.0
WYN = {}


def wczytaj_bd(folder):
    m = json.load(open(os.path.join(folder, 'manifest.json'), encoding='utf-8'))
    steps = m['steps']
    X = np.empty((len(steps), m['point_count'], 3), np.float32)
    for i, s in enumerate(steps):
        f = m['files'][str(s)]
        q = np.frombuffer(open(os.path.join(folder, f['name']), 'rb').read(), dtype='<u2').reshape(-1, 4).astype(np.float32) / Q16
        X[i] = np.array(f['quant']['lo'], np.float32) + q[:, :3] * np.array(f['quant']['span'], np.float32)
    return X, np.array(steps), m


def odwin(X, L):
    """Zdejmuje okresowe brzegi pudła: trajektorie ciągłe (skoki > L/2 cofane)."""
    Y = X.copy()
    d = np.diff(Y, axis=0)
    d -= L * np.round(d / L)
    Y[1:] = Y[0] + np.cumsum(d, axis=0)
    return Y


def testy(P, kroki, okno, N6, blok, d_shell, nazwa, taus_kk=(0, 1, 2, 3, 5, 10, 20, 30, 50, 100), W=100, dlug=200):
    od, do = okno
    w = dict(nazwa=nazwa, N=int(P.shape[1]), okno=[int(od), int(do)], d_shell=float(d_shell))
    P1 = dh.klatka_cr(P, kroki, od, N6)
    w['trwalosc_ruchliwosci_CR'] = dh.trwalosc_ruchliwosci(P1, kroki, okno, blok, tau=10, dlug=dlug, pole='blok', nazwa=nazwa + ' CR')
    w['trwalosc_ruchliwosci_lab'] = dh.trwalosc_ruchliwosci(P, kroki, okno, blok, tau=10, dlug=dlug, pole='blok', nazwa=nazwa + ' lab')
    w['korelacja_krzyzowa'] = dh.korelacja_krzyzowa_par(P, kroki, okno, N6, blok, W=W, taus=taus_kk, pole='populacja', nazwa=nazwa)
    lagi = [l for l in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, do - od) if l <= do - od]
    w['rozbieganie_par'] = dh.rozbieganie_par(P, kroki, od, lagi, N6, d_shell, nazwa=nazwa)
    w['rozklad_wariancji'] = dh.rozklad_wariancji(P, kroki, okno, [1, 10, 100], N6, blok, nazwa=nazwa)
    w['korelacja_przesuniec'] = dh.korelacja_przesuniec(P, kroki, okno, [1, 10, 100], blok, d_shell, nazwa=nazwa)
    return w


# ================================================================== 1. szkło BD (duży bieg stały)
bd = json.load(open(os.path.join(HERE, 'eden', 'wyniki', 'MODEL_SZKLA_EDEN_stala.json'), encoding='utf-8'))['biegi']
for phi in ('0.58', '0.61', '0.62', '0.64'):
    folder = os.path.join(OSIE, f'20261010T_szklo_eden_szklo_BD_eden_stala_phi_{phi}_duze')
    if not os.path.isdir(folder):
        dh.log('brak', folder); continue
    X, steps, m = wczytaj_bd(folder)
    L = bd[f'phi_{phi}']['L']
    co = m['uklad']['klatka_co_rekordow']
    N6b, dnn, dsh = O.knn_3d_pudlo(X[0], L, 6)
    d_shell = float(np.median(dsh))
    Y = odwin(X, L)
    kl = np.arange(len(steps))
    nazwa = f'BD φ={phi}'
    w = testy(Y, kl, (0, len(kl) - 1), N6b, None, d_shell, nazwa, W=100, dlug=200)
    w.update(tau_B_w_klatkach=bd[f'phi_{phi}']['tau_B_w_rekordach'] / co, tau_1e_klatki=(bd[f'phi_{phi}']['tau_1e'] or float('inf')) / co,
             a2_szczyt_tau_klatki=bd[f'phi_{phi}']['a2_szczyt_tau'] / co, L=L, klatka_co_rekordow=co)
    WYN[nazwa] = w
    del X, Y

# ================================================================== 2. OU-B (dołki stygnące, niezależny szum) — jak heterogenicznosc.py
u = json.load(open(os.path.join(HERE, 'UKLAD_SZKLA.json'), encoding='utf-8'))
plateau_vit = u['etapy']['S1']['l_klatka'] ** 2
N_OU, D_OU, T_REF, KONIEC = 6000, 32, 300, 2188
r2 = np.random.default_rng(1)
kappa = 0.07 * np.exp(r2.normal(0, 1.0, N_OU))
k_med = float(np.median(kappa))
T_plateau = plateau_vit * k_med / (2 * D_OU)
T_styg = lambda t, T0: T0 * (1 + max(t - T_REF, 0) / 400.0) ** (-0.79)
T0_B = T_plateau / T_styg(1000, 1.0)
x = np.zeros((N_OU, D_OU), np.float32)
P = np.empty((KONIEC + 1, N_OU, D_OU), np.float32)
ek = np.exp(-kappa)[:, None].astype(np.float32)
amp0 = np.sqrt((1 - np.exp(-2 * kappa)) / kappa)[:, None].astype(np.float32)
for t in range(KONIEC + 1):
    x = x * ek + np.sqrt(T_styg(t, T0_B)) * amp0 * r2.standard_normal((N_OU, D_OU)).astype(np.float32)
    P[t] = x
kr = np.arange(KONIEC + 1)
N6o, dnn, dsh = O.knn_w_blokach(P[T_REF], N_OU, 6)
WYN['OU-B'] = testy(P, kr, (T_REF, KONIEC), N6o, None, float(np.median(dsh)), 'OU-B (dołki stygnące, szum niezależny)', W=100, dlug=200)

# ================================================================== 3. OU-B + wspólny szum zdjęcia (Moran): 10 „zdjęć” × 600 dołków
rng3 = np.random.default_rng(5)
blok = 600; S = 10; N3 = S * blok
kap3 = 0.07 * np.exp(rng3.normal(0, 1.0, N3))
ek3 = np.exp(-kap3)[:, None].astype(np.float32); amp3 = np.sqrt((1 - np.exp(-2 * kap3)) / kap3)[:, None].astype(np.float32)
x = np.zeros((N3, D_OU), np.float32); P = np.empty((KONIEC + 1, N3, D_OU), np.float32)
for t in range(KONIEC + 1):
    wsp = np.repeat(rng3.standard_normal((S, D_OU)).astype(np.float32), blok, axis=0)             # wspólny szum zdjęcia (udział 50 % wariancji)
    x = x * ek3 + np.sqrt(T_styg(t, T0_B)) * amp3 * (np.sqrt(0.5) * rng3.standard_normal((N3, D_OU)).astype(np.float32) + np.sqrt(0.5) * wsp)
    P[t] = x
N6m, dnn, dsh = O.knn_w_blokach(P[T_REF], blok, 6)
WYN['OU-Moran'] = testy(P, kr, (T_REF, KONIEC), N6m, blok, float(np.median(dsh)), 'OU-B + wspólny szum zdjęcia (Moran, 50 %)', W=100, dlug=200)

json.dump(dict(opis=__doc__, wyniki=WYN), open(os.path.join(HERE, 'SUBSTANCJA_KONTROLA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
dh.log('zapisano SUBSTANCJA_KONTROLA.json')
