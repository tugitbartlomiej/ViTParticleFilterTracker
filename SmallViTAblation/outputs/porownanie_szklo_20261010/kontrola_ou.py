"""Kontrola negatywna: ile odcisków „szkła” daje zwykła cząstka w dołku harmonicznym z szumem (Ornstein–Uhlenbeck),
bez żadnych oddziaływań między cząstkami. Jeśli OU spełnia dane kryterium, to kryterium nie odróżnia szkła od uwięzienia.

Warianty (N = 6000 cząstek, d = 32 jak PCA32, krok = 1 krok uczenia, okno 300–2188 jak dla tokenów):
  OU-A  stała temperatura, dołki o różnej sztywności (zamrożona heterogeniczność, rozrzut jak u tokenów: 10×);
  OU-B  jak A, ale kąpiel stygnie jak zmierzona energia tokenów: T ∝ (1 + t/t0)^-0,79;
  OU-C  jak B, plus powolny dryf środków dołków (błądzenie losowe): ucieczka bez przeskoków;
  OU-D  jak C, plus pęd (AR(1) na szumie, ρ = 0,59 jak u tokenów).
Kalibracja: sztywność κ i T0 dobrane tak, by plateau MSD równało się plateau tokenów względem klatki (S1), więc
kryterium Lindemanna spełnia się „z definicji”; d_shell wzięte z tokenów. Wynik: KONTROLA_OU.json, kontrola_ou.png.
Uruchomienie: F:/Python/3.13/python.exe kontrola_ou.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odciski as O  # noqa: E402

N, D = 6000, 32
T_REF, KONIEC = 300, 2188
rng = np.random.default_rng(1)
t0 = time.time()
u = json.load(open(os.path.join(HERE, 'UKLAD_SZKLA.json'), encoding='utf-8'))
S1 = u['etapy']['S1']
d_shell, d_nn = u['skale']['d_shell'], u['skale']['d_nn']
plateau_vit = S1['l_klatka'] ** 2                                 # MSD w plateau tokenów względem klatki
ROZRZUT = 1.0                                                   # ln κ: rms ruchliwości tokenów ma rozrzut ≈ 10× (σ² ∝ 1/κ)
kappa_het = 0.07 * np.exp(rng.normal(0, ROZRZUT, N))            # sztywność: czas relaksacji ≈ 14 kroków przy medianie
kappa_hom = np.full(N, 0.07, np.float32)
kappa = kappa_het
# plateau OU (d wymiarów, stała T): MSD_plateau = 2 d T / κ  → T0 z plateau tokenów przy medianie κ (po stygnięciu do kroku ~1000)
def T_styg(t, T0):
    return T0 * (1 + max(t - T_REF, 0) / 400.0) ** (-0.79)


def symuluj(wariant, T0, n=KONIEC + 1, dryf=0.0, rho=0.0, kappa=None):
    kappa = kappa_het if kappa is None else kappa
    x = np.zeros((N, D), np.float32)
    c = np.zeros((N, D), np.float32)
    szum_prev = np.zeros((N, D), np.float32)
    P = np.empty((n, N, D), np.float32)
    ek = np.exp(-kappa)[:, None].astype(np.float32)                 # dokładny krok OU (dt = 1): x' = c + (x−c) e^-κ + sqrt(T/κ (1−e^-2κ)) ξ
    amp0 = np.sqrt((1 - np.exp(-2 * kappa)) / kappa)[:, None].astype(np.float32)
    for t in range(n):
        T = T0 if wariant.startswith('OU-A') else T_styg(t, T0)
        xi = rng.standard_normal((N, D)).astype(np.float32)
        if rho > 0:
            xi = rho * szum_prev + np.sqrt(1 - rho ** 2) * xi      # szum skorelowany w czasie (pęd), ta sama wariancja
            szum_prev = xi
        if dryf > 0:
            c += np.sqrt(dryf) * rng.standard_normal((N, D)).astype(np.float32)
        x = c + (x - c) * ek + np.sqrt(T) * amp0 * xi
        P[t] = x
    return P


# kalibracja T0: plateau przy stałej T = 2 d T/κ_med → T0 tak, by przy kroku ~1000 (po stygnięciu) plateau = plateau tokenów
k_med = float(np.median(kappa))
T_plateau = plateau_vit * k_med / (2 * D)
T0_B = T_plateau / T_styg(1000, 1.0)
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
wyniki = {}
WAR = [('OU-A0', dict(T0=T_plateau, kappa=kappa_hom)), ('OU-A', dict(T0=T_plateau)), ('OU-B', dict(T0=T0_B)),
       ('OU-C', dict(T0=T0_B, dryf=plateau_vit / D * 2e-4)), ('OU-D', dict(T0=T0_B, dryf=plateau_vit / D * 2e-4, rho=0.59))]
OPIS = {'OU-A0': 'stała T, dołki jednakowe', 'OU-A': 'stała T, dołki o rozrzucie 10×', 'OU-B': 'A + stygnięcie jak tokeny',
        'OU-C': 'B + dryf dołków', 'OU-D': 'C + pęd ρ=0,59', 'OU-B czas termiczny': 'B po przeskalowaniu czasem termicznym (jak S3)'}
for nazwa, par in WAR:
    P = symuluj(nazwa, **par)
    w = O.odciski(P, np.arange(len(P)), (T_REF, KONIEC), d_shell, d_nn, odejmij_pole=True, tw_lista=(300, 600, 900, 1200, 1500, 1800),
                  krotkie_tw=(300, 600, 900, 1200, 1500, 2000), nazwa=nazwa)
    w['karta'], w['spelnione'] = O.karta(w, None)
    w['parametry'] = dict({k: v for k, v in par.items() if k != 'kappa'}, kappa_mediana=k_med, opis=OPIS[nazwa])
    wyniki[nazwa] = w
    log(nazwa, 'karta', w['spelnione'], '| Lindemann', round(w['lindemann_shell'], 3), 'c', round(w['c_plateau'], 3), 'α plateau', round(w['alfa_plateau'], 2),
        'a2', round(w['a2_plateau'], 2), 'szczyt', round(w['a2_szczyt'], 2), '@', w['a2_szczyt_tau'], 'ogon', round(w['ogony'].get(10, {}).get('krotnosc_gaussa_3sigma', 0), 2),
        'f_q', round(w['f_q'], 2), 'β', round(w['beta'], 2), 'μ', {k: round(v, 2) for k, v in w['mu'].items()}, 'krótkie', round(w['stosunek_krotkich'], 2),
        'α krótkie', round(w['alfa_krotkie'], 2))
    if nazwa == 'OU-B':                                          # czas termiczny jak S3: s = Σ E(t)/Ē na oknie 300–2188
        E = np.array([((P[t + 1] - P[t]) - (P[t + 1] - P[t]).mean(0)).__pow__(2).sum(1).mean() for t in range(T_REF, KONIEC)])
        s = np.concatenate([[0.0], np.cumsum(E / E.mean())])
        K = int(np.floor(s[-1]))
        P3 = np.empty((K + 1, N, D), np.float32)
        tt = np.arange(T_REF, KONIEC + 1)
        for k in range(K + 1):
            j = min(max(int(np.searchsorted(s, k, side='right') - 1), 0), len(s) - 2)
            f = (k - s[j]) / max(s[j + 1] - s[j], 1e-12)
            P3[k] = P[tt[j]] * (1 - f) + P[tt[j + 1]] * f
        w3 = O.odciski(P3, np.arange(K + 1), (0, K), d_shell, d_nn, odejmij_pole=True, tw_lista=(0, 100, 200, 400, 800, 1200),
                       krotkie_tw=(0, 300, 600, 900, 1200, 1500), nazwa='OU-B czas termiczny')
        w3['karta'], w3['spelnione'] = O.karta(w3, None)
        w3['parametry'] = dict(opis=OPIS['OU-B czas termiczny'])
        wyniki['OU-B czas termiczny'] = w3
        log('OU-B czas termiczny', 'karta', w3['spelnione'], 'μ', {k: round(v, 2) for k, v in w3['mu'].items()}, 'krótkie', round(w3['stosunek_krotkich'], 2))
        del P3
    del P

# zestawienie z tokenami (S1, S4) i modelem BD (lokalny MODEL_SZKLA.json, jeśli jest)
karty = {k: dict(spelnione=v['spelnione'], pozycje=[(o['kryterium'], o['ok']) for o in v['karta']]) for k, v in wyniki.items()}
for k in ('S1', 'S4'):
    kk, n = O.karta(u['etapy'][k], u['trwalosc'].get('1300'))
    karty[f'ViT {k}'] = dict(spelnione=n, pozycje=[(o['kryterium'], o['ok']) for o in kk])
try:
    mb = json.load(open(os.path.join(HERE, 'MODEL_SZKLA.json'), encoding='utf-8'))
    for k, v in mb['biegi'].items():
        if 'karta' in v:
            karty[f'BD {k}'] = dict(spelnione=v['spelnione'], pozycje=[(o['kryterium'], o['ok']) for o in v['karta']])
except Exception:
    pass
# które kryteria spełnia każde OU (= „za darmo”), a których żadne OU nie spełnia (= odróżniające)
kryteria = [o['kryterium'] for o in wyniki['OU-A']['karta']]
za_darmo = [kr for i, kr in enumerate(kryteria) if all(wyniki[w]['karta'][i]['ok'] for w in wyniki)]
odrozniajace = [kr for i, kr in enumerate(kryteria) if not any(wyniki[w]['karta'][i]['ok'] for w in wyniki)]
json.dump(dict(opis=__doc__, plateau_tokenow=plateau_vit, d_shell=d_shell, warianty=wyniki, karty=karty, za_darmo=za_darmo, odrozniajace=odrozniajace),
          open(os.path.join(HERE, 'KONTROLA_OU.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('za darmo (spełnia każde OU):', za_darmo)
log('odróżniające (żadne OU nie spełnia):', odrozniajace)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
KOL = {'OU-A0': '#5a5f69', 'OU-A': '#8a8f99', 'OU-B': '#4cd964', 'OU-C': '#4a8cff', 'OU-D': '#ff4433', 'OU-B czas termiczny': '#c77dff'}
fig, ax = plt.subplots(1, 3, figsize=(19, 6.5), dpi=100)
for k, w in wyniki.items():
    if k == 'OU-B czas termiczny':
        continue
    ax[0].loglog(w['taus'], w['msd'], color=KOL[k], lw=1.6, label=f'{k}: {OPIS[k]}')
    ax[1].semilogx(w['taus'], w['a2'], color=KOL[k], lw=1.6, label=k)
ax[0].loglog(S1['taus'], S1['msd'], '--', color='#ffd166', lw=2.2, label='tokeny ViT S1 (klatka)')
ax[1].semilogx(S1['taus'], S1['a2'], '--', color='#ffd166', lw=2.2, label='tokeny ViT S1')
ax[0].set(xlabel='τ (kroki)', ylabel='MSD', title='MSD: cząstki OU bez oddziaływań a tokeny')
ax[1].set(xlabel='τ', ylabel='α₂', title='parametr niegaussowski')
kol = list(karty)
mat = np.array([[1 if karty[k]['pozycje'][i][1] else 0 for k in kol] for i in range(len(kryteria))])
ax[2].imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax[2].set_xticks(range(len(kol)))
ax[2].set_xticklabels([f"{k}\n{karty[k]['spelnione']}/{len(kryteria)}" for k in kol], fontsize=7, rotation=30)
ax[2].set_yticks(range(len(kryteria)))
ax[2].set_yticklabels([n[:56] for n in kryteria], fontsize=7)
ax[2].set_title('karta: OU (kontrola negatywna), tokeny ViT, model BD')
for a in ax[:2]:
    a.grid(True, which='both', alpha=0.4)
    a.legend(fontsize=8)
fig.suptitle('Kontrola negatywna: które odciski szkła daje sama cząstka w dołku z szumem', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(HERE, 'kontrola_ou.png'))
log('GOTOWE')
