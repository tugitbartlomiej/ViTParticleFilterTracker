"""Czy centrum drgań tokenu wyznacza położenie patcha (kodowanie pozycji z kroku 0)? Eksport ruch60 (rp128.npy: 806 checkpointów ×
100 zdjęć × 600 tokenów × 128; siatka patchy 20 × 30, token j = wiersz j // 30, kolumna j % 30). Współrzędne względne x = z − średnia
tokenów zdjęcia. Dla wybranych checkpointów:
  1. szablon pozycji P_j = średnia x_j po zdjęciach TRAIN (co token o numerze j ma wspólnego niezależnie od treści); udział wariancji
     wyjaśniony przez P na zdjęciach VALID (R², bez przecieku), osobno treść zdjęcia = x − P;
  2. sąsiedzi w latencie a sąsiedzi na siatce: odsetek 6 najbliższych w latencie, które są sąsiednimi patchami (odległość na siatce ≤ √2;
     losowo 8/599), i Spearman odległość w latencie – odległość na siatce;
  3. pamięć startu: Jaccard 6 sąsiadów z krokiem 0 i z końcem 5. epoki; przesunięcie od kroku 0 w odstępach sąsiadów;
  4. skąd centrum: centrum etapu c_i = średnia x_i po etapie (siatka co 100 kroków). R² na VALID trzech przewidywań c_i:
     (a) położenie z kroku 0 (liniowa mapa 128→128 dopasowana na TRAIN), (b) sam szablon pozycji etapu P (numer patcha),
     (c) położenie z końca poprzedniego etapu (liniowa mapa).
Uruchomienie: python pozycja_patchy.py <eksport> <out> → POZYCJA_PATCHY.json, pozycja_patchy.png"""
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

EKS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
META = json.load(open(os.path.join(EKS, 'META.json'), encoding='utf-8'))
kroki = np.array(META['kroki'])
rp = np.load(os.path.join(EKS, 'rp128.npy'), mmap_mode='r')
K, N, TOK, D = rp.shape
H, Wd = 20, 30
EP = 1094
tr = np.array([z['split'] == 'TRAIN' for z in META['zdjecia']]); va = ~tr
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
siatka = np.stack([np.arange(TOK) // Wd, np.arange(TOK) % Wd], 1).astype(float)
Dg = cdist(siatka, siatka)
sas_siatki = (Dg <= np.sqrt(2) + 1e-9) & (Dg > 0)
rng = np.random.default_rng(0)
pary = rng.choice(TOK, (3000, 2))
pary = pary[pary[:, 0] != pary[:, 1]]


def wzgl(t):
    X = np.asarray(rp[t]).astype(np.float32)
    return X - X.mean(1, keepdims=True)


def knn6(Xi):
    Dm = cdist(Xi, Xi); np.fill_diagonal(Dm, np.inf)
    N6 = np.argsort(Dm, 1)[:, :6]
    return N6, float(np.median(np.take_along_axis(Dm, N6, 1).mean(1)))


def r2(pred, cel):
    return float(1 - ((cel - pred) ** 2).sum() / ((cel - cel.mean((0, 1), keepdims=True)) ** 2).sum())


def mapa(Xa, Ya):
    """Liniowa mapa (z wyrazem wolnym) Xa (n,600,D) → Ya na TRAIN; zwraca predyktor."""
    A = np.concatenate([Xa[tr].reshape(-1, D), np.ones((int(tr.sum()) * TOK, 1))], 1)
    M, *_ = np.linalg.lstsq(A, Ya[tr].reshape(-1, D), rcond=None)
    return lambda Z: np.concatenate([Z.reshape(-1, D), np.ones((len(Z) * TOK, 1))], 1).dot(M).reshape(len(Z), TOK, D)


X0 = wzgl(0)
N6_0 = [knn6(X0[n])[0] for n in range(N)]
t5 = int(np.argmin(abs(kroki - 5 * EP)))
N6_5 = [knn6(wzgl(t5)[n])[0] for n in range(N)]
WYBR = [0, 10, 50, 100, 300, 1000, EP, 2 * EP, 5 * EP, 10 * EP, 20 * EP, 30 * EP, 40 * EP, 50 * EP, 60 * EP]
punkty = []
for kk in WYBR:
    t = int(np.argmin(abs(kroki - kk)))
    X = wzgl(t)
    P = X[tr].mean(0)
    war = ((X[va] - X[va].mean((0, 1), keepdims=True)) ** 2).sum()
    r2_poz = float(1 - ((X[va] - P) ** 2).sum() / war)
    sas_sia, sp, jac0, jac5, dsh = [], [], [], [], []
    for n in range(N):
        N6, d = knn6(X[n]); dsh.append(d)
        sas_sia.append(float(sas_siatki[np.arange(TOK)[:, None], N6].mean()))
        dl = np.linalg.norm(X[n, pary[:, 0]] - X[n, pary[:, 1]], axis=1)
        sp.append(spearmanr(dl, Dg[pary[:, 0], pary[:, 1]]).statistic)
        jac0.append(np.mean([len(set(N6[i]) & set(N6_0[n][i])) / len(set(N6[i]) | set(N6_0[n][i])) for i in range(0, TOK, 2)]))
        jac5.append(np.mean([len(set(N6[i]) & set(N6_5[n][i])) / len(set(N6[i]) | set(N6_5[n][i])) for i in range(0, TOK, 2)]))
    d_shell = float(np.median(dsh))
    pkt = dict(krok=int(kroki[t]), epoka=float(kroki[t] / EP), r2_szablon_pozycji_VALID=r2_poz, sasiedzi_to_sasiednie_patche=float(np.mean(sas_sia)),
               losowo=8 / 599, spearman_latent_siatka=float(np.mean(sp)), jaccard_z_krokiem_0=float(np.mean(jac0)), jaccard_z_epoka_5=float(np.mean(jac5)),
               d_shell=d_shell, przesuniecie_od_0_do_dshell=float(np.sqrt(((X - X0) ** 2).sum(-1).mean()) / d_shell),
               norma_wzgl=float(np.sqrt((X ** 2).sum(-1).mean())), norma_szablonu=float(np.sqrt((P ** 2).sum(-1).mean())))
    punkty.append(pkt)
    log(f'krok {pkt["krok"]:6d} (ep {pkt["epoka"]:5.2f}): R² numer patcha {r2_poz:.2f}, 6NN = sąsiednie patche {pkt["sasiedzi_to_sasiednie_patche"]:.2f} (losowo 0,013), '
        f'Spearman latent–siatka {pkt["spearman_latent_siatka"]:.2f}, Jaccard z krokiem 0 {pkt["jaccard_z_krokiem_0"]:.2f}, z ep. 5 {pkt["jaccard_z_epoka_5"]:.2f}, '
        f'przesunięcie od 0 {pkt["przesuniecie_od_0_do_dshell"]:.1f} odstępu')

# skąd centrum etapu
ETAPY = [(1, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
centra = {}
prev_end = None
for e1, e2 in ETAPY:
    idx = [i for i, k in enumerate(kroki) if k >= 1000 and k % 100 == 0 and EP * e1 <= k < EP * e2]
    C = np.zeros((N, TOK, D), np.float64)
    for i in idx:
        C += wzgl(i)
    C = (C / len(idx)).astype(np.float32)
    P = C[tr].mean(0)
    wynik = dict(r2_z_kroku_0=r2(mapa(X0, C)(X0[va]), C[va]), r2_numer_patcha=r2(np.broadcast_to(P, C[va].shape), C[va]))
    Xs = wzgl(idx[0])
    wynik['r2_z_poczatku_etapu'] = r2(mapa(Xs, C)(Xs[va]), C[va])
    if prev_end is not None:
        wynik['r2_z_centrum_poprzedniego_etapu'] = r2(mapa(prev_end, C)(prev_end[va]), C[va])
    prev_end = C
    centra[f'{e1}-{e2}'] = wynik
    log(f'centrum epok {e1}–{e2}: R² z kroku 0 {wynik["r2_z_kroku_0"]:.2f}, z numeru patcha {wynik["r2_numer_patcha"]:.2f}, z początku etapu {wynik["r2_z_poczatku_etapu"]:.2f}'
        + (f', z centrum poprzedniego etapu {wynik["r2_z_centrum_poprzedniego_etapu"]:.2f}' if 'r2_z_centrum_poprzedniego_etapu' in wynik else ''))

json.dump(dict(opis=__doc__.split('\n')[0], punkty=punkty, centra_etapow=centra), open(os.path.join(OUT, 'POZYCJA_PATCHY.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    sys.exit(0)
plt.rcParams.update({'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8',
                     'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100)
x = [max(p['krok'], 1) for p in punkty]
a = ax[0]
for kl, c, et in (('r2_szablon_pozycji_VALID', '#ffffff', 'R² numeru patcha (VALID)'), ('sasiedzi_to_sasiednie_patche', '#ffd166', '6 sąsiadów = sąsiednie patche'),
                  ('spearman_latent_siatka', '#4a8cff', 'Spearman odległość latent–siatka'), ('jaccard_z_krokiem_0', '#ff4433', 'Jaccard sąsiadów z krokiem 0'),
                  ('jaccard_z_epoka_5', '#c77dff', 'Jaccard sąsiadów z epoką 5')):
    a.plot(x, [p[kl] for p in punkty], 'o-', color=c, label=et)
a.axhline(8 / 599, color='#8a8f99', ls=':', lw=1); a.set_xscale('log'); a.set_xlabel('krok'); a.set_ylim(-0.05, 1); a.legend(fontsize=8); a.grid(True, alpha=0.4)
a.set_title('1–3. ile położenia tokenu to numer patcha i pamięć startu', fontsize=10)
a = ax[1]
nm = list(centra); xx = np.arange(len(nm)); w = 0.27
a.bar(xx, [centra[e]['r2_z_kroku_0'] for e in nm], w, color='#ff4433', label='z położenia w kroku 0')
a.bar(xx + w, [centra[e]['r2_numer_patcha'] for e in nm], w, color='#ffffff', label='z samego numeru patcha')
a.bar(xx + 2 * w, [centra[e].get('r2_z_centrum_poprzedniego_etapu', np.nan) for e in nm], w, color='#4a8cff', label='z centrum poprzedniego etapu')
a.set_xticks(xx + w); a.set_xticklabels([f'ep. {e}' for e in nm]); a.set_ylabel('R² centrum na VALID'); a.set_ylim(min(0, a.get_ylim()[0]), 1); a.legend(fontsize=8); a.grid(True, axis='y', alpha=0.4)
a.set_title('4. co wyznacza centrum drgań w danym etapie', fontsize=10)
fig.suptitle('Czy centrum drgań tokenu to położenie patcha z kroku 0?', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pozycja_patchy.png'))
log('zapisano POZYCJA_PATCHY.json, pozycja_patchy.png')
