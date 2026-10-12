"""Siatka patchy w gęstym zapisie RP128 biegu pokręteł (gesty/pca32_*.npz, S sond × 600 tokenów): ile położenia tokenu wyjaśnia numer
patcha (R² szablonu pozycji liczonego na połowie sond, sprawdzanego na drugiej połowie), odsetek 6 najbliższych sąsiadów w latencie
będących sąsiednimi patchami na siatce 20 × 30 (losowo 8/599), Spearman odległość latent–siatka, Jaccard sąsiadów z pierwszym krokiem
okna i z krokiem 0, w wybranych krokach. Kontrola kryształu dla biegów: bez kodowania pozycji, CNN, lr 2e-3, ziarna.
Uruchomienie: python siatka_gesty.py <gesty> <out> <nazwa> [kroki: 0,10,50,100,300,1000,2188,3280]"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

G, OUT, NAZWA = sys.argv[1], sys.argv[2], sys.argv[3]
KROKI = [int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else '0,10,50,100,300,1000,2188,3280').split(',')]
os.makedirs(OUT, exist_ok=True)
H, W, TOK = 20, 30, 600
siatka = np.stack([np.arange(TOK) // W, np.arange(TOK) % W], 1).astype(float)
Dg = cdist(siatka, siatka); sas_siatki = (Dg <= np.sqrt(2) + 1e-9) & (Dg > 0)
rng = np.random.default_rng(0); pary = rng.choice(TOK, (3000, 2)); pary = pary[pary[:, 0] != pary[:, 1]]
dane = {}
for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
    z = np.load(p); k = z['kroki']
    for kk in KROKI:
        i = np.where(k == kk)[0]
        if len(i):
            dane[kk] = z['pca32'][i[0]].astype(np.float32)
    del z
if not dane:
    sys.exit('brak kroków')
S = next(iter(dane.values())).shape[0]
tr = np.arange(S) < S // 2; va = ~tr


def knn6(Xi):
    Dm = cdist(Xi, Xi); np.fill_diagonal(Dm, np.inf)
    return np.argsort(Dm, 1)[:, :6]


wzgl = lambda X: X - X.mean(1, keepdims=True)
X0 = wzgl(dane[min(dane)]); N6_0 = [knn6(X0[s]) for s in range(S)]
punkty = []
for kk in sorted(dane):
    X = wzgl(dane[kk]); P = X[tr].mean(0)
    r2 = float(1 - ((X[va] - P) ** 2).sum() / ((X[va] - X[va].mean((0, 1), keepdims=True)) ** 2).sum())
    sas, sp, jac = [], [], []
    for s in range(S):
        N6 = knn6(X[s]); sas.append(float(sas_siatki[np.arange(TOK)[:, None], N6].mean()))
        sp.append(spearmanr(np.linalg.norm(X[s, pary[:, 0]] - X[s, pary[:, 1]], axis=1), Dg[pary[:, 0], pary[:, 1]]).statistic)
        jac.append(np.mean([len(set(N6[i]) & set(N6_0[s][i])) / len(set(N6[i]) | set(N6_0[s][i])) for i in range(0, TOK, 2)]))
    punkty.append(dict(krok=int(kk), r2_numer_patcha=r2, sasiedzi_to_sasiednie_patche=float(np.mean(sas)), losowo=8 / 599, spearman_latent_siatka=float(np.mean(sp)), jaccard_z_pierwszym=float(np.mean(jac))))
    print(f'{NAZWA} krok {kk:5d}: R² numer patcha {r2:.2f}, 6NN = sąsiednie patche {np.mean(sas):.2f} (losowo 0,013), Spearman latent–siatka {np.mean(sp):.2f}, Jaccard z krokiem {min(dane)} {np.mean(jac):.2f}', flush=True)
json.dump(dict(opis=__doc__.split('\n')[0], bieg=NAZWA, sond=int(S), punkty=punkty), open(os.path.join(OUT, f'SIATKA_{NAZWA}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
