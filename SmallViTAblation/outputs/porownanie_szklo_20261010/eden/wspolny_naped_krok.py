"""Odjęcie wspólnego napędu w skali kroku (Eden, bez torch). Dane: gęsty zapis RP128 biegu pokręteł (gesty/pca32_*.npz, klucz 'pca32'
= rzut 128D, S sond × 600 tokenów, co 1 krok). Napęd Δθ(t) nie jest zapisany co krok, więc estymujemy go jako „nieznane wejście”:
czynniki niskiego rzędu c_{−s}(t) ∈ R^K to główne składowe w czasie pola przesunięć tokenów POZOSTAŁYCH sond (macierz Grama T×T,
eigh), bez sondy badanej. Regresja: Δx_i(t) ≈ B_i c_{−s}(t) dla każdego tokenu sondy s; reszta r_i(t) = ruch własny.
Raport: (1) udział energii przesunięć wyjaśniony przez czynniki z innych zdjęć w funkcji K (z poziomem przypadku K/(T−1) i kontrolą
z czasem przetasowanym); (2) korelacje przesunięć sąsiadów (6 NN), par losowych w zdjęciu i par między zdjęciami: oryginał, bez modu
zdjęcia, reszty; (3) trwałość heterogeniczności ruchliwości (Spearman rankingu |r|² między połówkami okna) i α₂ niezmienniczy
dla oryginału i reszt; (4) MSD „położeń resztowych” (suma skumulowana reszt) kontra MSD oryginału: czy klatka zostaje po odjęciu napędu;
(5) czasy implikowane VAMP (deeptime) dla: położeń względnych x̃, położeń resztowych y, stanu rozszerzonego [x̃, c(t)].
Uruchomienie: python wspolny_naped_krok.py <gesty> <od> <do> <n_sond> <out> <nazwa> → NAPED_<nazwa>.json, naped_<nazwa>.png"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

G, OD, DO, N_SOND, OUT, NAZWA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
os.makedirs(OUT, exist_ok=True)
TOK = 600
KS = (1, 3, 10, 30, 100)
K_STAR = 30
K_VAMP = 10
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

# ---------------------------------------------------------------- dane
Xs, kroki = [], []
for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
    z = np.load(p)
    k = z['kroki']
    m = (k >= OD) & (k <= DO)
    if m.any():
        Xs.append(z['pca32'][m][:, :N_SOND].astype(np.float32))
        kroki.append(k[m])
    del z
X = np.concatenate(Xs); kroki = np.concatenate(kroki).astype(int); del Xs
assert (np.diff(kroki) == 1).all()
T, S, _, D = X.shape
F = TOK * D
log(f'{NAZWA}: {T} kroków ({kroki[0]}–{kroki[-1]}), {S} sond × {TOK} tokenów × {D}')
N6 = [np.argsort(cdist(X[0, s], X[0, s]) + np.eye(TOK) * 1e9, 1)[:, :6] for s in range(S)]        # sąsiedzi z początku okna
dX = np.diff(X, axis=0)                                                                           # (T−1, S, 600, D)
Tm = T - 1
E_tot = float((dX ** 2).sum())

# ---------------------------------------------------------------- macierze Grama pola przesunięć każdej sondy
Gs = np.zeros((S, Tm, Tm))
for s in range(S):
    Dm = dX[:, s].reshape(Tm, F).astype(np.float64)
    Gs[s] = Dm @ Dm.T
    del Dm
Gall = Gs.sum(0)
log('macierze Grama gotowe')


def czynniki(Gm, K):
    """Ortonormalne wektory czasowe K głównych składowych pola (kolumny U, Tm × K) i udział ich energii."""
    w, V = np.linalg.eigh(Gm)
    idx = np.argsort(w)[::-1][:K]
    return V[:, idx], float(w[idx].sum() / w.sum())


rng = np.random.default_rng(0)
wyjasnione = {K: [] for K in KS}; wyjasnione_kontrola = {K: [] for K in KS}; energia_czynnikow = {K: [] for K in KS}
Rres = np.empty_like(dX)                                                                          # reszty dla K_STAR
Cfac = []                                                                                         # czynniki c_{−s}(t) K_VAMP dla każdej sondy
for s in range(S):
    Dm = dX[:, s].reshape(Tm, F)
    E_s = float((Dm.astype(np.float64) ** 2).sum())
    Gm = Gall - Gs[s]
    for K in KS:
        U, en = czynniki(Gm, K)
        Fit = U @ (U.T @ Dm)                                                                      # regresja na ortonormalnych czynnikach
        wyjasnione[K].append(float((Fit.astype(np.float64) ** 2).sum() / E_s))
        energia_czynnikow[K].append(en)
        Up = U[rng.permutation(Tm)]                                                               # kontrola: czynniki z przetasowanym czasem
        Fp = Up @ (Up.T @ Dm)
        wyjasnione_kontrola[K].append(float((Fp.astype(np.float64) ** 2).sum() / E_s))
        if K == K_STAR:
            Rres[:, s] = (Dm - Fit).reshape(Tm, TOK, D)
        if K == K_VAMP:
            Cfac.append((U * np.sqrt(np.maximum(np.linalg.eigvalsh(Gm)[::-1][:K], 0)))[:, :K])      # skalowane energią (tylko do stanu rozszerzonego)
        del U, Fit, Up, Fp
    log(f'sonda {s + 1}/{S}: wyjaśnione przez czynniki innych zdjęć {[round(wyjasnione[K][-1], 3) for K in KS]} (kontrola {[round(wyjasnione_kontrola[K][-1], 3) for K in KS]})')
    del Dm
udzial = {K: dict(srednia=float(np.mean(wyjasnione[K])), min=float(np.min(wyjasnione[K])), max=float(np.max(wyjasnione[K])),
                  kontrola_przetasowana=float(np.mean(wyjasnione_kontrola[K])), poziom_przypadku=K / Tm, energia_czynnikow_w_polu_innych=float(np.mean(energia_czynnikow[K])))
          for K in KS}
mod = dX.mean(2, keepdims=True)
udzial_modu = float((mod ** 2).sum() * TOK / E_tot)
dXm = dX - mod
log('udział modu zdjęcia:', round(udzial_modu, 3), '| wyjaśnione:', {K: round(v['srednia'], 3) for K, v in udzial.items()})


# ---------------------------------------------------------------- (2) korelacje przesunięć: sąsiedzi, pary losowe w zdjęciu, pary między zdjęciami
def kor(A, B):
    """Średnia kosinusowa podobieństwa wektorów przesunięć par (po czasie i parach)."""
    num = (A * B).sum(-1); den = np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1) + 1e-12
    return float((num / den).mean())


def korelacje(dZ):
    sas, los, miedzy = [], [], []
    for s in range(S):
        i = np.arange(TOK)
        sas.append(kor(dZ[:, s, i[:, None].repeat(6, 1)], dZ[:, s, N6[s]]))
        j = rng.permutation(TOK)
        los.append(kor(dZ[:, s], dZ[:, s, j]))
        s2 = (s + 1) % S
        miedzy.append(kor(dZ[:, s], dZ[:, s2, rng.permutation(TOK)]))
    return dict(sasiedzi=float(np.mean(sas)), pary_losowe_w_zdjeciu=float(np.mean(los)), pary_miedzy_zdjeciami=float(np.mean(miedzy)))


korel = dict(oryginal=korelacje(dX), bez_modu_zdjecia=korelacje(dXm), reszty_K30=korelacje(Rres))
log('korelacje:', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in korel.items()})


# ---------------------------------------------------------------- (3) heterogeniczność zamrożona i α₂ (niezmienniczy na obroty)
def hetero(dZ):
    h = Tm // 2
    m1 = (dZ[:h] ** 2).sum(-1).mean(0); m2 = (dZ[h:] ** 2).sum(-1).mean(0)                         # (S, 600)
    return float(np.mean([spearmanr(m1[s], m2[s]).statistic for s in range(S)]))


def alfa2(dZ):
    r2 = (dZ ** 2).sum(-1)
    return float(D / (D + 2) * (r2 ** 2).mean() / r2.mean() ** 2 - 1)


hz = dict(oryginal=hetero(dX), bez_modu_zdjecia=hetero(dXm), reszty_K30=hetero(Rres))
a2 = dict(oryginal=alfa2(dX), bez_modu_zdjecia=alfa2(dXm), reszty_K30=alfa2(Rres))
log('heterogeniczność (Spearman połówek):', {k: round(v, 3) for k, v in hz.items()}, '| α₂:', {k: round(v, 3) for k, v in a2.items()})

# ---------------------------------------------------------------- (4) MSD położeń: oryginał względny kontra reszty skumulowane
Xr = X - X.mean(2, keepdims=True)
Y = np.concatenate([np.zeros((1, S, TOK, D), np.float32), np.cumsum(Rres, axis=0, dtype=np.float32)])   # położenia resztowe
Ym = np.concatenate([np.zeros((1, S, TOK, D), np.float32), np.cumsum(dXm, axis=0, dtype=np.float32)])   # bez modu zdjęcia (= Xr − Xr[0])
TAUS = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 400, 800) if t < T // 2]
msd = {}
for nazwa, Z in (('oryginal_wzgledny', Xr), ('bez_modu_zdjecia', Ym), ('reszty_K30', Y)):
    msd[nazwa] = [float(((Z[t:] - Z[:-t]) ** 2).sum(-1).mean()) for t in TAUS]
d_shell = float(np.median([np.median(np.take_along_axis(cdist(X[0, s], X[0, s]), N6[s], 1).mean(1)) for s in range(S)]))
log('MSD/d_shell²:', {k: [round(v / d_shell ** 2, 3) for v in vv] for k, vv in msd.items()})

# ---------------------------------------------------------------- (5) VAMP: x̃, y, [x̃, c(t)]
vamp = {}
try:
    from deeptime.decomposition import VAMP
    Xt = Xr - Xr.mean(0, keepdims=True)
    Yt = Y - Y.mean(0, keepdims=True)
    C_t = np.stack([np.concatenate([np.zeros((1, K_VAMP)), Cfac[s]]) for s in range(S)])          # (S, T, K) czynniki wyrównane do położeń
    C_t = (C_t - C_t.mean(1, keepdims=True)) / (C_t.std((0, 1), keepdims=True) + 1e-9) * float(Xt.std())
    war = {'polozenia_wzgledne': lambda s, i: Xt[:, s, i], 'polozenia_resztowe': lambda s, i: Yt[:, s, i],
           'rozszerzony_x_i_czynniki': lambda s, i: np.concatenate([Xt[:, s, i], C_t[s]], 1)}
    for nazwa, f in war.items():
        traj = [f(s, i).astype(np.float64) for s in range(S) for i in range(0, TOK, 6)]
        vamp[nazwa] = {}
        for tau in (1, 2, 3, 5, 8, 12, 20, 30):
            if tau * 4 >= T:
                continue
            m_ = VAMP(lagtime=tau, dim=8).fit(traj).fetch_model()
            sv = np.clip(m_.singular_values[:8], 1e-9, 1 - 1e-9)
            vamp[nazwa][tau] = [round(float(-tau / math.log(v)), 1) for v in sv]
        log(f'VAMP {nazwa}: τ=1 {vamp[nazwa].get(1, [])[:3]} τ=5 {vamp[nazwa].get(5, [])[:3]} τ=20 {vamp[nazwa].get(20, [])[:3]}')
        del traj
except Exception as e:  # noqa: BLE001
    vamp = {'uwaga': f'deeptime: {type(e).__name__}: {e}'[:300]}
    log(vamp)

wyn = dict(opis=__doc__.split('\n')[0], bieg=NAZWA, okno=[int(kroki[0]), int(kroki[-1])], sond=S, tokenow=TOK, wymiar=D, K_reszt=K_STAR, K_vamp=K_VAMP,
           udzial_modu_zdjecia=udzial_modu, wyjasnione_przez_czynniki_innych_zdjec=udzial, korelacje_przesuniec=korel, heterogenicznosc_spearman_polowek=hz,
           alfa2_inw=a2, d_shell=d_shell, msd_taus=TAUS, msd=msd, vamp_czasy_implikowane=vamp)
json.dump(wyn, open(os.path.join(OUT, f'NAPED_{NAZWA}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8',
                     'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(2, 2, figsize=(13, 9), dpi=100)
a = ax[0, 0]
a.plot(KS, [udzial[K]['srednia'] for K in KS], 'o-', color='#ffffff', label='czynniki z innych zdjęć')
a.plot(KS, [udzial[K]['kontrola_przetasowana'] for K in KS], 's--', color='#ff4433', label='kontrola: czas przetasowany')
a.plot(KS, [udzial[K]['poziom_przypadku'] for K in KS], ':', color='#8a8f99', label='K/(T−1)')
a.axhline(udzial_modu, color='#ffd166', lw=1, ls='--', label=f'mod zdjęcia ({udzial_modu:.2f})')
a.set_xscale('log'); a.set_xlabel('liczba czynników K'); a.set_ylabel('udział energii przesunięć'); a.set_ylim(0, 1); a.legend(fontsize=8); a.grid(True, alpha=0.4)
a.set_title('1. ile ruchu tokenu wyjaśnia napęd widoczny w INNYCH zdjęciach', fontsize=10)
a = ax[0, 1]
nazwy = list(korel); x = np.arange(3); w = 0.25
for j, (n, c) in enumerate(zip(nazwy, ('#ffffff', '#ffd166', '#4a8cff'))):
    a.bar(x + j * w, [korel[n]['sasiedzi'], korel[n]['pary_losowe_w_zdjeciu'], korel[n]['pary_miedzy_zdjeciami']], w, color=c, label=n)
a.set_xticks(x + w); a.set_xticklabels(['sąsiedzi (6 NN)', 'pary losowe w zdjęciu', 'pary między zdjęciami']); a.axhline(0, color='#8a8f99', lw=0.8)
a.set_ylabel('⟨cos(Δx_i, Δx_j)⟩'); a.legend(fontsize=8); a.grid(True, axis='y', alpha=0.4); a.set_title('2. korelacje przesunięć przed i po odjęciu napędu', fontsize=10)
a = ax[1, 0]
for n, c in zip(msd, ('#ffffff', '#ffd166', '#4a8cff')):
    a.plot(TAUS, np.array(msd[n]) / d_shell ** 2, 'o-', color=c, label=n)
a.plot(TAUS, np.array(TAUS) * msd['reszty_K30'][0] / d_shell ** 2, ':', color='#8a8f99', lw=1, label='dyfuzja ∝ τ')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('τ [kroki]'); a.set_ylabel('MSD / d_shell²'); a.legend(fontsize=8); a.grid(True, alpha=0.4)
a.set_title(f'4. czy klatka zostaje po odjęciu napędu (hetero: {hz["oryginal"]:.2f} → {hz["reszty_K30"]:.2f}, α₂: {a2["oryginal"]:.2f} → {a2["reszty_K30"]:.2f})', fontsize=10)
a = ax[1, 1]
if isinstance(vamp, dict) and 'polozenia_wzgledne' in vamp:
    for n, c in zip(vamp, ('#ffffff', '#4a8cff', '#c77dff')):
        taus = sorted(vamp[n]); a.plot(taus, [vamp[n][t][0] for t in taus], 'o-', color=c, label=f'{n}, proces 1'); a.plot(taus, [vamp[n][t][2] for t in taus], 's--', color=c, lw=0.8, ms=3)
    a.plot(taus, taus, ':', color='#8a8f99', lw=1, label='τ')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('opóźnienie τ [kroki]'); a.set_ylabel('czas implikowany [kroki]'); a.legend(fontsize=7)
a.grid(True, alpha=0.4); a.set_title('5. test Markowa: położenia, reszty, stan rozszerzony o napęd', fontsize=10)
fig.suptitle(f'{NAZWA}: odjęcie wspólnego napędu (czynniki z pola innych zdjęć, K={K_STAR}), kroki {kroki[0]}–{kroki[-1]}', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, f'naped_{NAZWA}.png'))
log(f'zapisano NAPED_{NAZWA}.json, naped_{NAZWA}.png')
