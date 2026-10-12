"""Cztery stopnie przeniesienia ruchu tokenów ViT do „układu szkła” i karta zgodności po każdym stopniu.

S0  względem populacji (minus pole kolektywne) — jak dotąd.
S1  względem klatki: minus średnie przesunięcie 6 sąsiadów w latencie (sąsiedzi z kroku 300, w obrębie zdjęcia);
    usuwa wspólny przepływ sąsiedztw (Vivek i in. 2017).
S2  jednakowe cząstki: trajektoria każdego tokenu przeskalowana przez jego własną ruchliwość (rms przesunięcia
    co 1 krok) do wspólnej średniej; usuwa zamrożoną heterogeniczność (narzędzie gorętsze od tła), zostaje
    heterogeniczność dynamiczna jak w szkle jednoskładnikowym.
S3  czas termiczny: krok przeskalowany energią drgań (s = Σ E(t)/Ē), czyli zegar, w którym kąpiel ma stałą
    temperaturę; w koloidzie termostat jest stały, u nas stygnie sam (E ∝ t^-0,79 przy stałym lr).
S4  bez pędu: innowacja AR(1) przyrostów (ρ = VACF(1)), przeskalowana 1/(1-ρ), żeby długoczasowa dyfuzja została;
    usuwa bezwładność Adama i daje przetłumiony ruch jak w zawiesinie (Hunter i Weeks 2012).

Dane: gęsty zapis biegu 35 000, 10 sond × 600 tokenów, PCA32. Wynik: UKLAD_SZKLA.json, uklad_szkla.png.
Uruchomienie: F:/Python/3.13/python.exe uklad_szkla_analiza.py
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import odciski as O

G = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty'
E35K = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T162654_natural_78cd84bd6edd'
OUT = os.path.dirname(os.path.abspath(__file__))
NS, H, W = 10, 20, 30
N = NS * 600
T_REF, GESTE_DO = 300, 2188
t0 = time.time()


def log(*a):
    print(*a, f'[{time.time() - t0:.0f} s]', flush=True)


klasy = np.load(os.path.join(E35K, 'obszary_oka.npz'))['klasy'][:NS].reshape(-1)
Z, kroki = [], []
for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
    z = np.load(p)
    kr = z['kroki']
    Z.append(np.ascontiguousarray(z['pca32'][:, :NS].astype(np.float32)).reshape(len(kr), N, 32))
    kroki.append(kr)
    del z
Z = np.concatenate(Z)
kroki = np.concatenate(kroki).astype(int)
ind = {int(k): i for i, k in enumerate(kroki)}
K_MAX = int(kroki[-1])
log('wczytane', Z.shape)

# geometria w kroku odniesienia (oryginalny układ): sąsiedzi w latencie, odstępy, pary siatki
N6, dnn, dsh = O.knn_w_blokach(Z[ind[T_REF]], 600, 6)
d_nn, d_shell = float(np.median(dnn)), float(np.median(dsh))
idx = np.arange(600)
rr, cc = idx // W, idx % W
pary = np.concatenate([np.stack([np.concatenate([idx[cc < W - 1], idx[rr < H - 1]]),
                                 np.concatenate([idx[cc < W - 1] + 1, idx[rr < H - 1] + W])], 1) + im * 600 for im in range(NS)])
trw = O.trwalosc_sasiedztwa(Z, kroki, T_REF, (310, 400, 1000, 1300, GESTE_DO, 5000, 10000, K_MAX))
log('geometria: d_nn', round(d_nn, 3), 'd_shell', round(d_shell, 3), 'trwałość', {k: round(v, 2) for k, v in trw.items()})
TW = (300, 1000, 2000, 3000, 5000, 10000, 15000, 20000)
wyn, karty = {}, {}

# S0
w = O.odciski(Z, kroki, (T_REF, GESTE_DO), d_shell, d_nn, pary=pary, tw_lista=TW, odejmij_pole=True, nazwa='S0 względem populacji')
wyn['S0'] = w
karty['S0'] = O.karta(w, trw.get(1300))
log('S0', karty['S0'][1], 'spełnionych')

# S1: względem klatki, sąsiedzi z kroku 300
Z300 = Z[ind[T_REF]].copy()
P = np.empty_like(Z)
for t in range(len(kroki)):
    P[t] = Z[t] - (Z[t][N6] - Z300[N6]).mean(1)
del Z
w = O.odciski(P, kroki, (T_REF, GESTE_DO), d_shell, d_nn, pary=pary, tw_lista=TW, odejmij_pole=True, nazwa='S1 względem klatki')
wyn['S1'] = w
karty['S1'] = O.karta(w, trw.get(1300))
log('S1', karty['S1'][1], 'spełnionych; Lindemann', round(w['lindemann_shell'], 3), 'c', round(w['c_plateau'], 3))

# S2: jednakowe cząstki (ruchliwość własna → średnia); geometria bez zmian
acc = np.zeros(N)
n_acc = 0
for t in range(T_REF, GESTE_DO):
    dd = P[ind[t + 1]] - P[ind[t]]
    dd -= dd.mean(0)
    acc += (dd ** 2).sum(1)
    n_acc += 1
sig = np.sqrt(acc / n_acc)
skala = (sig.mean() / np.maximum(sig, 1e-12)).astype(np.float32)
P0 = P[ind[T_REF]].copy()
for t in range(len(kroki)):
    P[t] = P0 + (P[t] - P0) * skala[:, None]
w = O.odciski(P, kroki, (T_REF, GESTE_DO), d_shell, d_nn, pary=pary, tw_lista=TW, odejmij_pole=True, nazwa='S2 jednakowe cząstki')
wyn['S2'] = w
wyn['S2']['rozrzut_ruchliwosci'] = dict(sigma_min=float(sig.min()), sigma_mediana=float(np.median(sig)), sigma_max=float(sig.max()),
                                        sigma_obszary=[float(np.median(sig[klasy == c])) for c in range(4)])
karty['S2'] = O.karta(w, trw.get(1300))
log('S2', karty['S2'][1], 'spełnionych; a2 szczyt', round(w['a2_szczyt'], 2), 'przy', w['a2_szczyt_tau'])

# S3: czas termiczny na części gęstej (300..2188)
E = np.array([((P[ind[t + 1]] - P[ind[t]]) - (P[ind[t + 1]] - P[ind[t]]).mean(0)).__pow__(2).sum(1).mean() for t in range(T_REF, GESTE_DO)])
s = np.concatenate([[0.0], np.cumsum(E / E.mean())])                       # s(t) dla t = 300..2188, s(2188) = 1888
K = int(np.floor(s[-1]))
P3 = np.empty((K + 1, N, 32), np.float32)
tt = np.arange(T_REF, GESTE_DO + 1)
for k in range(K + 1):
    j = int(np.searchsorted(s, k, side='right') - 1)
    j = min(max(j, 0), len(s) - 2)
    f = (k - s[j]) / max(s[j + 1] - s[j], 1e-12)
    P3[k] = P[ind[tt[j]]] * (1 - f) + P[ind[tt[j + 1]]] * f
kroki3 = np.arange(K + 1)
w = O.odciski(P3, kroki3, (0, K), d_shell, d_nn, pary=pary, tw_lista=(0, 100, 200, 400, 800, 1200), krotkie_tw=(0, 300, 600, 900, 1200, 1500),
              odejmij_pole=True, nazwa='S3 czas termiczny')
w['mapa_czasu'] = dict(krok=[int(x) for x in tt[::50]], s=[float(x) for x in s[::50]])
wyn['S3'] = w
karty['S3'] = O.karta(w, trw.get(1300))
log('S3', karty['S3'][1], 'spełnionych; μ', w['mu'], 'stosunek krótkich', round(w['stosunek_krotkich'], 2))

# S4: bez pędu (AR(1) na przyrostach w czasie termicznym)
D = np.diff(P3, axis=0)
D -= D.mean(1, keepdims=True)
rho = float((D[1:] * D[:-1]).sum(2).mean() / (D * D).sum(2).mean())
eps = D[1:] - rho * D[:-1]
P4 = np.empty_like(P3)
P4[0] = P3[0]
P4[1] = P3[0] + D[0]
P4[2:] = P3[0] + D[0] + np.cumsum(eps, axis=0) / (1 - rho)
del D, eps
w = O.odciski(P4, kroki3, (0, K), d_shell, d_nn, pary=pary, tw_lista=(0, 100, 200, 400, 800, 1200), krotkie_tw=(0, 300, 600, 900, 1200, 1500),
              odejmij_pole=True, nazwa='S4 bez pędu')
w['rho'] = rho
wyn['S4'] = w
karty['S4'] = O.karta(w, trw.get(1300))
log('S4', karty['S4'][1], 'spełnionych; ρ', round(rho, 3), 'α krótkie', round(w['alfa_krotkie'], 2))

json.dump(dict(opis=__doc__, skale=dict(d_nn=d_nn, d_shell=d_shell), trwalosc=trw, etapy=wyn,
               karty={k: dict(spelnione=v[1], z=len(v[0]), pozycje=v[0]) for k, v in karty.items()}, czas_s=round(time.time() - t0)),
          open(os.path.join(OUT, 'UKLAD_SZKLA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- rysunek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
KOL = {'S0': '#8a8f99', 'S1': '#ffd166', 'S2': '#4cd964', 'S3': '#4a8cff', 'S4': '#ff4433'}
fig, ax = plt.subplots(2, 3, figsize=(19, 11), dpi=100)
for k, w in wyn.items():
    ax[0, 0].loglog(w['taus'], w['msd'], color=KOL[k], lw=2 if k == 'S1' else 1.4, label=w['nazwa'])
    ax[0, 1].semilogx(w['taus'], w['a2'], color=KOL[k], lw=1.6, label=w['nazwa'])
    ax[0, 2].semilogx(w['taus'], w['fs'], color=KOL[k], lw=1.6, label=w['nazwa'])
    cx = sorted(int(x) for x in w['c_tau'])
    ax[1, 0].semilogx(cx, [w['c_tau'][str(x)] if str(x) in w['c_tau'] else w['c_tau'][x] for x in cx], 'o-', color=KOL[k], lw=1.4, ms=4, label=w['nazwa'])
    pts = sorted((tw, v[10]) for tw, v in w['starzenie'].items() if 10 in v or '10' in v)
    if pts:
        ax[1, 1].loglog([p[0] + 1 for p in pts], [p[1] for p in pts], 'o-', color=KOL[k], lw=1.4, ms=4, label=w['nazwa'] + (f' (μ={w["mu"][10]:.2f})' if 10 in w['mu'] else ''))
ax[0, 0].axhline(d_shell ** 2, color='w', ls='--', lw=0.8)
ax[0, 0].text(1.1, d_shell ** 2 * 1.15, 'd_shell²', fontsize=9)
ax[0, 0].axhline((0.15 * d_shell) ** 2, color='#ff4433', ls=':', lw=0.8)
ax[0, 0].text(1.1, (0.15 * d_shell) ** 2 * 1.15, '(0,15 d_shell)² szkło', fontsize=9, color='#ff4433')
ax[0, 0].set(xlabel='τ (kroki / jednostki czasu termicznego)', ylabel='MSD', title='MSD po kolejnych stopniach')
ax[0, 1].axhline(0, color='w', lw=0.6)
ax[0, 1].set(xlabel='τ', ylabel='α₂ (1D)', title='parametr niegaussowski')
ax[0, 2].axhline(np.exp(-1), color='w', ls=':', lw=0.8)
ax[0, 2].set(xlabel='τ', ylabel='Fs(q, τ), q = 2π/d_shell·√(d/3)', title='funkcja rozpraszania', ylim=(-0.05, 1.05))
ax[1, 0].axhspan(0.25, 0.55, color='#ffd166', alpha=0.12)
ax[1, 0].axhline(0, color='w', lw=0.6)
ax[1, 0].set(xlabel='τ', ylabel='c(τ) = 1 − MSD(2τ)/(2MSD(τ))', title='antykorelacja kolejnych przesunięć (pas: Weeks 0,25–0,55)')
ax[1, 1].set(xlabel='t_w + 1', ylabel='MSD(τ=10; t_w)', title='starzenie: ruch w klatce a wiek')
for a in ax.flat[:5]:
    a.grid(True, which='both', alpha=0.4)
    a.legend(fontsize=8)
a = ax[1, 2]
nazwy = [karty[k][0][i]['kryterium'] for i in range(len(O.KRYTERIA)) for k in ['S0']][:len(O.KRYTERIA)]
mat = np.array([[1 if karty[k][0][i]['ok'] else 0 for k in wyn] for i in range(len(O.KRYTERIA))])
a.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
a.set_xticks(range(len(wyn)))
a.set_xticklabels([f'{k}\n{karty[k][1]}/{len(O.KRYTERIA)}' for k in wyn], fontsize=9)
a.set_yticks(range(len(O.KRYTERIA)))
a.set_yticklabels([n[:58] for n in nazwy], fontsize=7)
a.set_title('karta zgodności ze szkłem (zielone = spełnione)')
fig.suptitle('Ruch tokenów ViT przeniesiony do układu szkła: od populacji do przetłumionej klatki o stałej temperaturze', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(OUT, 'uklad_szkla.png'))
log('GOTOWE')
