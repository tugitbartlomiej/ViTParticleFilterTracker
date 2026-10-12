"""Kontrola rzutu: czy odciski szkła zależą od tego, że patrzymy przez PCA32? Dla sond z pełnym zapisem 768D (gesty/pelne768_*.npz,
8 sond, co 1 krok do 1094) liczymy te same odciski S0/S1 w trzech przestrzeniach z TYCH SAMYCH latentów i tego samego okna:
  pelne768  — surowy latent (GELU MLP ostatniego bloku), float32;
  pca32     — rzut na bazę PCA32 z kroku 0 (BAZA_PCA32.npz), jak w całym badaniu;
  rp128     — losowy rzut Gaussa na 128 wymiarów (Johnson–Lindenstrauss), bez wiedzy o danych.
Każda przestrzeń ma własnych 6 sąsiadów z kroku 300 i własne d_shell; podajemy też zgodność zbiorów sąsiadów (Jaccard) między
przestrzeniami. Pamięć: NS_P sond × okno (OD, DO) × 768 × 4 B — domyślnie 3 sondy, kroki 300–800 (≈ 2,8 GB).
Wynik: KONTROLA_RZUTU.json.  Uruchomienie: F:/Python/3.13/python.exe kontrola_rzutu.py [NS_P] [OD] [DO]"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odciski as O  # noqa: E402

G = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty'
NS_P = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OD = int(sys.argv[2]) if len(sys.argv) > 2 else 300
DO = int(sys.argv[3]) if len(sys.argv) > 3 else 800
TOK = 600
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

baza = np.load(os.path.join(G, 'BAZA_PCA32.npz'))
mean, comp = baza['mean'].astype(np.float32), baza['comp'].astype(np.float32)          # (768,), (32, 768)
R = (np.random.default_rng(0).standard_normal((768, 128)) / np.sqrt(128)).astype(np.float32)
Z, kroki = [], []
for p in sorted(glob.glob(os.path.join(G, 'pelne768_*.npz'))):
    z = np.load(p)
    k = z['kroki']
    m = (k >= OD) & (k <= DO)
    if m.any():
        Z.append(np.ascontiguousarray(z['latent'][m][:, :NS_P].astype(np.float32)).reshape(int(m.sum()), NS_P * TOK, 768))
        kroki.append(k[m])
    del z
Z = np.concatenate(Z); kroki = np.concatenate(kroki).astype(int)
assert (np.diff(kroki) == 1).all(), 'zapis pełny nie co 1 krok w tym oknie'
log('pełne latenty', Z.shape)
przestrzenie = {'pelne768': Z, 'pca32': (Z - mean) @ comp.T, 'rp128': Z @ R}
ind = {int(k): i for i, k in enumerate(kroki)}
TW = tuple(x for x in (OD, OD + 150, OD + 300, OD + 400) if x < DO - 50)
KT = tuple(x for x in (OD, OD + 150, OD + 300, OD + 450) if x < DO - 5)
wyn, N6s = {}, {}
for nazwa, X in przestrzenie.items():
    N6, dnn, dsh = O.knn_w_blokach(X[ind[OD]], TOK, 6)
    N6s[nazwa] = N6
    d_nn, d_shell = float(np.median(dnn)), float(np.median(dsh))
    w0 = O.odciski(X, kroki, (OD, DO), d_shell, d_nn, odejmij_pole=True, tw_lista=TW, krotkie_tw=KT, nazwa=f'{nazwa} S0')
    w0['karta'], w0['spelnione'] = O.karta(w0, None)
    Zr = X[ind[OD]]
    P1 = np.empty_like(X)
    for t in range(len(X)):
        P1[t] = X[t] - (X[t][N6] - Zr[N6]).mean(1)
    w1 = O.odciski(P1, kroki, (OD, DO), d_shell, d_nn, odejmij_pole=True, tw_lista=TW, krotkie_tw=KT, nazwa=f'{nazwa} S1')
    w1['karta'], w1['spelnione'] = O.karta(w1, None)
    del P1
    wyn[nazwa] = dict(wymiar=int(X.shape[-1]), d_nn=d_nn, d_shell=d_shell, S0=w0, S1=w1)
    log(f"{nazwa:9s} d_shell {d_shell:.3f} | S0 karta {w0['spelnione']} Lind {w0['lindemann_shell']:.3f} c {w0['c_plateau']:.3f} a2 {w0['a2_szczyt']:.2f}@{w0['a2_szczyt_tau']} "
        f"| S1 karta {w1['spelnione']} Lind {w1['lindemann_shell']:.3f} c {w1['c_plateau']:.3f} α {w1['alfa_plateau']:.2f} a2 {w1['a2_szczyt']:.2f}@{w1['a2_szczyt_tau']} "
        f"f_q {w1['f_q']:.2f} β {w1['beta']:.2f} ogon10 {w1['ogony'].get(10, {}).get('krotnosc_gaussa_3sigma', float('nan')):.2f} μ10 {w1.get('mu', {}).get('10')}")
jacc = {}
for a in N6s:
    for b in N6s:
        if a < b:
            jacc[f'{a}|{b}'] = float(np.mean([len(set(N6s[a][i]) & set(N6s[b][i])) / len(set(N6s[a][i]) | set(N6s[b][i])) for i in range(len(N6s[a]))]))
log('zgodność sąsiadów (Jaccard)', {k: round(v, 3) for k, v in jacc.items()})
json.dump(dict(opis=__doc__, NS_P=NS_P, okno=[OD, DO], jaccard_sasiadow=jacc, przestrzenie=wyn),
          open(os.path.join(HERE, 'KONTROLA_RZUTU.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('zapisano KONTROLA_RZUTU.json')
