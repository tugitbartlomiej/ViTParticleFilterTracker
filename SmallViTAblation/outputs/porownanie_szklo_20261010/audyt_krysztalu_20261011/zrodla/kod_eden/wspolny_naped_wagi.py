"""Odjęcie wspólnego napędu w skali checkpointów z PRAWDZIWYCH przyrostów wag (Eden, torch tylko do wczytania wag). Dane: 806 checkpointów
biegu 35 000 (wagi/kNNNNN.pt) i eksport ruch60 (rp128.npy: 100 zdjęć × 600 tokenów × 128 po tych samych checkpointach).
(1) Ruch wag: ‖Δθ‖ między kolejnymi checkpointami, kosinus kolejnych przyrostów (trwałość kierunku), odległość od startu kontra długość
    drogi (ile dryfu to kierunek, ile błądzenie), widmo macierzy Grama przyrostów (jak niskiego rzędu jest napęd).
(2) Regresja przesunięć tokenów ΔZ(t) na czynnikach niskiego rzędu przyrostów wag c(t) (główne składowe Grama Δθ w etapie), K = 1…20,
    z poziomem przypadku K/n i kontrolą z przetasowanym czasem; porównanie z czynnikami z pola przesunięć INNYCH zdjęć (bez wag).
(3) Na resztach: korelacja przesunięć par tokenów między zdjęciami i w zdjęciu, MSD(1 checkpoint) oryginał kontra reszty, udział modu zdjęcia.
Etapy jak w ruch60_analiza.py (siatka co 100 kroków): epoki 1–5, 5–10, 10–20, 20–30, 30–40, 40–50, 50–60 i wczesny co 10 kroków (100–1000).
Uruchomienie: python wspolny_naped_wagi.py <wagi> <eksport> <out> → NAPED_WAGI.json, naped_wagi.png"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

WAGI, EKS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
META = json.load(open(EKS / 'META.json', encoding='utf-8'))
kroki = np.array(META['kroki'])
rp = np.load(EKS / 'rp128.npy', mmap_mode='r')
K_, N, TOK, D = rp.shape
EP = 1094
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- wagi: wektor parametrów na każdym checkpoincie
pliki = {int(m[1]): p for p in WAGI.iterdir() if (m := re.fullmatch(r'k([0-9]+)\.pt', p.name))}
st0 = torch.load(pliki[int(kroki[0])], map_location='cpu', weights_only=True)['net']
klucze = [k for k, v in st0.items() if v.dtype.is_floating_point and not any(x in k for x in ('running_mean', 'running_var', 'num_batches_tracked'))]
P = int(sum(st0[k].numel() for k in klucze))
log(f'parametrów uczonych {P} w {len(klucze)} tensorach; checkpointów {len(kroki)}')
Theta = np.empty((len(kroki), P), np.float32)
for i, k in enumerate(kroki):
    st = torch.load(pliki[int(k)], map_location='cpu', weights_only=True)['net']
    Theta[i] = torch.cat([st[q].reshape(-1).float() for q in klucze]).numpy()
    if i % 100 == 0:
        log(f'wczytano {i + 1}/{len(kroki)}')
dTh = np.diff(Theta, axis=0)                                                                    # (K_−1, P) przyrosty między kolejnymi checkpointami
normy = np.linalg.norm(dTh, axis=1)
cos_kolejne = (dTh[1:] * dTh[:-1]).sum(1) / (normy[1:] * normy[:-1] + 1e-12)
odl_od_startu = np.linalg.norm(Theta - Theta[0], axis=1)
droga = np.concatenate([[0], np.cumsum(normy)])
norma_theta = np.linalg.norm(Theta, axis=1)
ruch_wag = dict(krok=[int(k) for k in kroki[1:]], norma_przyrostu=[float(v) for v in normy], cos_kolejnych=[float(v) for v in cos_kolejne],
                odleglosc_od_startu=[float(v) for v in odl_od_startu], dlugosc_drogi=[float(v) for v in droga], norma_theta=[float(v) for v in norma_theta])
log(f'‖Δθ‖ co 100 kroków: epoka 2 {np.median(normy[(kroki[1:] > 2 * EP) & (kroki[1:] < 3 * EP)]):.3f}, epoka 30 {np.median(normy[(kroki[1:] > 30 * EP) & (kroki[1:] < 31 * EP)]):.3f}, '
    f'epoka 59 {np.median(normy[(kroki[1:] > 59 * EP)]):.3f}; odległość od startu / droga na końcu {odl_od_startu[-1] / droga[-1]:.3f}; ‖θ‖ {norma_theta[0]:.1f} → {norma_theta[-1]:.1f}')

# ---------------------------------------------------------------- etapy
ETAPY = {'wczesny_100_1000_co10': [i for i, k in enumerate(kroki) if 100 <= k <= 1000 and k % 10 == 0]}
for e1, e2 in ((1, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)):
    ETAPY[f'{e1}-{e2}'] = [i for i, k in enumerate(kroki) if k >= 1000 and k % 100 == 0 and EP * e1 <= k < EP * e2]
ETAPY['10-60_razem'] = [i for i, k in enumerate(kroki) if k >= 1000 and k % 100 == 0 and EP * 10 <= k < EP * 60]
KS = (1, 2, 3, 5, 10, 20, 50)


def czynniki(Gm, K):
    w, V = np.linalg.eigh(Gm)
    idx = np.argsort(w)[::-1][:K]
    return V[:, idx], float(w[idx].sum() / max(w.sum(), 1e-12))


def kor_miedzy(dZ, par=4000):
    """Średni kosinus przesunięć par tokenów z RÓŻNYCH zdjęć (ten sam checkpoint) i par losowych w zdjęciu."""
    n = dZ.shape[0]
    a = rng.integers(0, N, par); b = (a + rng.integers(1, N, par)) % N; i = rng.integers(0, TOK, par); j = rng.integers(0, TOK, par)
    A = dZ[:, a, i]; B = dZ[:, b, j]
    miedzy = float(((A * B).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-12)).mean())
    B2 = dZ[:, a, j]
    w_zdj = float(((A * B2).sum(-1) / (np.linalg.norm(A, axis=-1) * np.linalg.norm(B2, axis=-1) + 1e-12)).mean())
    return miedzy, w_zdj


etapy = {}
for nazwa, idx in ETAPY.items():
    n = len(idx) - 1
    if n < 8:
        continue
    Z = np.asarray(rp[idx]).astype(np.float32)                                                    # (n+1, N, 600, D)
    dZ = np.diff(Z, axis=0); del Z                                                                 # (n, N, 600, D)
    E = float((dZ ** 2).sum())
    mod = dZ.mean(2, keepdims=True); udzial_modu = float((mod ** 2).sum() * TOK / E)
    dT = dTh[np.array(idx[:-1])]                                                                    # przyrosty wag między tymi checkpointami (n, P)
    Gw = (dT.astype(np.float64) @ dT.astype(np.float64).T)                                        # Gram przyrostów wag (n × n)
    Dm = dZ.reshape(n, -1)                                                                        # (n, N·600·D)
    # czynniki z pola innych zdjęć: Gram pola bez zdjęcia badanego ≈ Gram całości minus wkład zdjęcia; tu dla prostoty: połowa zdjęć jako „inne”
    pol = rng.permutation(N); A_, B_ = pol[:N // 2], pol[N // 2:]
    Gpole_A = dZ[:, A_].reshape(n, -1).astype(np.float64); Gpole_A = Gpole_A @ Gpole_A.T          # pole zdjęć A
    DB = dZ[:, B_].reshape(n, -1); E_B = float((DB.astype(np.float64) ** 2).sum())
    wyj_w, wyj_wk, wyj_p, en_w = {}, {}, {}, {}
    for K in KS:
        if K >= n:
            continue
        U, en = czynniki(Gw, K); en_w[K] = en
        wyj_w[K] = float(((U @ (U.T @ Dm)).astype(np.float64) ** 2).sum() / E)
        Up = U[rng.permutation(n)]
        wyj_wk[K] = float(((Up @ (Up.T @ Dm)).astype(np.float64) ** 2).sum() / E)
        Ua, _ = czynniki(Gpole_A, K)
        wyj_p[K] = float(((Ua @ (Ua.T @ DB)).astype(np.float64) ** 2).sum() / E_B)
    K_res = max(k for k in wyj_w if k <= max(5, n // 10))
    U, _ = czynniki(Gw, K_res)
    R = (Dm - U @ (U.T @ Dm)).reshape(n, N, TOK, D)
    m_or, z_or = kor_miedzy(dZ); m_re, z_re = kor_miedzy(R); m_mod, z_mod = kor_miedzy(dZ - mod)
    etapy[nazwa] = dict(checkpointow=n + 1, kroki=[int(kroki[idx[0]]), int(kroki[idx[-1]])], udzial_modu_zdjecia=udzial_modu,
                        wyjasnione_przez_czynniki_wag=wyj_w, kontrola_przetasowana=wyj_wk, poziom_przypadku={K: K / n for K in wyj_w},
                        energia_czynnikow_w_przyrostach_wag=en_w, wyjasnione_przez_pole_innych_zdjec=wyj_p, K_reszt=K_res,
                        korelacja_miedzy_zdjeciami=dict(oryginal=m_or, bez_modu=m_mod, reszty=m_re), korelacja_par_w_zdjeciu=dict(oryginal=z_or, bez_modu=z_mod, reszty=z_re),
                        msd1_oryginal=float((dZ ** 2).sum(-1).mean()), msd1_reszty=float((R ** 2).sum(-1).mean()),
                        norma_dtheta_mediana=float(np.median(np.linalg.norm(dT, axis=1))))
    log(f'{nazwa:>22s} n={n:3d}: wyjaśnione przez wagi K={list(wyj_w)[:5]} → {[round(wyj_w[k], 2) for k in list(wyj_w)[:5]]} (przypadek {[round(k / n, 2) for k in list(wyj_w)[:5]]}), '
        f'pole innych zdjęć {[round(wyj_p[k], 2) for k in list(wyj_p)[:5]]}, mod {udzial_modu:.2f}; korelacja między zdjęciami {m_or:.2f} → reszty {m_re:.2f}; MSD1 {etapy[nazwa]["msd1_oryginal"]:.2f} → {etapy[nazwa]["msd1_reszty"]:.2f}')
    del dZ, Dm, R, dT, Gw, DB

wyn = dict(opis=__doc__.split('\n')[0], parametrow=P, checkpointow=len(kroki), ruch_wag=ruch_wag, etapy=etapy)
json.dump(wyn, open(OUT / 'NAPED_WAGI.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:  # noqa: BLE001
    log(f'bez rysunku (matplotlib: {e}); JSON zapisany')
    sys.exit(0)
plt.rcParams.update({'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8',
                     'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(2, 2, figsize=(14, 9), dpi=100)
kk = kroki[1:] / EP; u = kroki[1:] >= 1000
a = ax[0, 0]
a.plot(kk[u], normy[u], color='#ffffff', lw=0.8, label='‖Δθ‖ na 100 kroków')
a2 = a.twinx(); a2.plot((kroki[2:] / EP)[u[1:]], cos_kolejne[u[1:]], color='#ffd166', lw=0.6, alpha=0.8, label='cos kolejnych przyrostów'); a2.set_ylabel('cos', color='#ffd166'); a2.set_ylim(-1, 1)
a.set_xlabel('epoka'); a.set_ylabel('‖Δθ‖'); a.legend(loc='upper left', fontsize=8); a2.legend(loc='upper right', fontsize=8); a.grid(True, alpha=0.4); a.set_title('1. ruch wag: długość kroku i trwałość kierunku', fontsize=10)
a = ax[0, 1]
a.plot(kroki / EP, odl_od_startu, color='#ffffff', label='odległość θ od startu')
a.plot(kroki / EP, droga, color='#4a8cff', label='długość drogi Σ‖Δθ‖')
a.plot(kroki / EP, norma_theta, color='#ff4433', label='‖θ‖')
a.set_xlabel('epoka'); a.legend(fontsize=8); a.grid(True, alpha=0.4); a.set_title(f'1b. dryf kontra błądzenie (koniec: odległość/droga {odl_od_startu[-1] / droga[-1]:.2f})', fontsize=10)
a = ax[1, 0]
kol = plt.cm.plasma(np.linspace(0.1, 0.95, len(etapy)))
for c, (n_, v) in zip(kol, etapy.items()):
    ks = sorted(v['wyjasnione_przez_czynniki_wag']); a.plot(ks, [v['wyjasnione_przez_czynniki_wag'][k] for k in ks], 'o-', color=c, label=f'{n_} (n={v["checkpointow"] - 1})')
    a.plot(ks, [v['poziom_przypadku'][k] for k in ks], ':', color=c, lw=0.8)
a.set_xscale('log'); a.set_xlabel('liczba czynników K z przyrostów wag'); a.set_ylabel('udział energii ΔZ wyjaśniony'); a.set_ylim(0, 1); a.legend(fontsize=7); a.grid(True, alpha=0.4)
a.set_title('2. ile ruchu tokenów tłumaczą główne kierunki przyrostów wag (kropki: przypadek K/n)', fontsize=10)
a = ax[1, 1]
nm = list(etapy); x = np.arange(len(nm)); w = 0.27
a.bar(x, [etapy[n_]['korelacja_miedzy_zdjeciami']['oryginal'] for n_ in nm], w, color='#ffffff', label='oryginał')
a.bar(x + w, [etapy[n_]['korelacja_miedzy_zdjeciami']['bez_modu'] for n_ in nm], w, color='#ffd166', label='bez modu zdjęcia')
a.bar(x + 2 * w, [etapy[n_]['korelacja_miedzy_zdjeciami']['reszty'] for n_ in nm], w, color='#4a8cff', label='reszty po czynnikach wag')
a.set_xticks(x + w); a.set_xticklabels(nm, rotation=30, ha='right', fontsize=8); a.axhline(0, color='#8a8f99', lw=0.8); a.set_ylabel('⟨cos⟩ par tokenów z różnych zdjęć')
a.legend(fontsize=8); a.grid(True, axis='y', alpha=0.4); a.set_title('3. korelacja między zdjęciami przed i po odjęciu napędu', fontsize=10)
fig.suptitle(f'Wspólny napęd z prawdziwych przyrostów wag: {P / 1e6:.2f} M parametrów, {len(kroki)} checkpointów, {N} zdjęć × {TOK} tokenów (rzut 128D)', fontsize=12)
fig.tight_layout(); fig.savefig(OUT / 'naped_wagi.png')
log('zapisano NAPED_WAGI.json, naped_wagi.png')
