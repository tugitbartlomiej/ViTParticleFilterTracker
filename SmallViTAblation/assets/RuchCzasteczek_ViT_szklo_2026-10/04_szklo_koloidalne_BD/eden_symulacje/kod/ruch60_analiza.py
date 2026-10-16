"""Ruch tokenów w skali epok (co 100 kroków, 806 checkpointów do 60. epoki) z eksportu ruch60_rp.py (asset
assets/MalyViT_35000_od_zera_s47/trajektorie_60_epok/ruch60_rp128_50tr_50val): 100 zdjęć × 600 tokenów w rzucie 128D.
Współrzędne względne x = z − średnia tokenów zdjęcia (bez modu zdjęcia). Etapy: epoki 1–5, 5–10, 10–20, 20–30, 30–40, 40–50, 50–60
(siatka co 100 kroków, bez dodatkowych końców epok). Na każdym etapie:
  1. MSD(L) dla opóźnień L = 1…40 checkpointów (100…4000 kroków), względne i bezwzględne, w jednostkach odstępu sąsiadów d_shell;
     wykładnik α z MSD ∝ L^α (0 = uwięzione, 1 = dyfuzja, 2 = dryf kierunkowy);
  2. trwałość sąsiedztwa: Jaccard zbiorów 6 najbliższych sąsiadów między początkiem etapu i L checkpointów później
     (ciało stałe: sąsiedzi ci sami; ciecz: spada do poziomu losowego 6/600);
  3. operator jednokrokowy A (128×128, opóźnienie 100 kroków) na drganiach wokół średniej etapu: najmniejsze kwadraty i rozmaitości
     piDMD (symetryczny, skośnie symetryczny, ortogonalny, diagonalny, skalarny), błąd względem trwania, widmo;
  4. czasy implikowane VAMP (deeptime) dla opóźnień 1, 2, 4, 8, 16 checkpointów: Markow czy pamięć w skali epok;
  5. kontrola rzutu: MSD w pełnych 768D kontra 128D dla 4 zdjęć;
  6. sygnał praktyczny: Spearman między ruchliwością zdjęcia (MSD(1)) a jego pewnością / IoU, osobno TRAIN i VALID;
  7. udział modu zdjęcia w energii przesunięć co 100 kroków.
Uruchomienie: F:/Python/3.13/python.exe ruch60_analiza.py [katalog eksportu] → RUCH60.json, ruch60.png"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
KAT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'assets', 'RuchCzasteczek_ViT_szklo_2026-10', '03_pomiar_naturalny_60_epok_eden', 'ruch60_rp128_50tr_50val')
if len(sys.argv) > 2:                                                                 # katalog wyjściowy (na Edenie: wyniki/ruch60)
    HERE = sys.argv[2]
    os.makedirs(HERE, exist_ok=True)
META = json.load(open(os.path.join(KAT, 'META.json'), encoding='utf-8'))
kroki = np.array(META['kroki'])
rp = np.load(os.path.join(KAT, 'rp128.npy'), mmap_mode='r')
p768 = np.load(os.path.join(KAT, 'pelne768.npy'), mmap_mode='r')
oceny = np.load(os.path.join(KAT, 'oceny.npy'))
K, N, TOK, D = rp.shape
EP = 1094
split = np.array([0 if z['split'] == 'TRAIN' else 1 for z in META['zdjecia']])
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
ETAPY = [(1, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
LAGI_VAMP = (1, 2, 4, 8, 16)


def knn6(Z):
    """Z (600, D) → indeksy 6 najbliższych (600, 6), odstęp pierwszego i średni odstęp powłoki 6 (mediany po tokenach)."""
    Dm = cdist(Z, Z)
    np.fill_diagonal(Dm, np.inf)
    N6 = np.argsort(Dm, 1)[:, :6]
    d6 = np.take_along_axis(Dm, N6, 1)
    return N6, float(np.median(d6[:, 0])), float(np.median(d6.mean(1)))


def operatory(Xt):
    """Xt (n, N, 600, D) drgania wokół średniej etapu → błędy jednego kroku i widmo (jak markow_test.py)."""
    Cxx = np.zeros((D, D)); Cyx = np.zeros((D, D)); Cyy = np.zeros((D, D))
    for s in range(Xt.shape[1]):
        A_ = Xt[:-1, s].reshape(-1, D).astype(np.float64); B_ = Xt[1:, s].reshape(-1, D).astype(np.float64)
        Cxx += A_.T @ A_; Cyx += B_.T @ A_; Cyy += B_.T @ B_
    E_tr = float(np.trace(Cyy - Cyx - Cyx.T + Cxx))
    blad = lambda A: float(np.trace(Cyy - A @ Cyx.T - Cyx @ A.T + A @ Cxx @ A.T) / E_tr)
    A_ls = Cyx @ np.linalg.pinv(Cxx)
    U_, _, Vt_ = np.linalg.svd(Cyx)
    ops = {'najmniejsze_kwadraty': A_ls, 'symetryczny': solve_continuous_lyapunov(Cxx, Cyx + Cyx.T), 'skosnie_symetryczny': solve_continuous_lyapunov(Cxx, Cyx - Cyx.T),
           'ortogonalny': U_ @ Vt_, 'diagonalny': np.diag(np.diag(Cyx) / np.diag(Cxx)), 'skalarny': np.eye(D) * (np.trace(Cyx) / np.trace(Cxx)),
           'zero (centrum)': np.zeros((D, D))}
    lam = np.linalg.eigvals(A_ls)
    mod = np.clip(abs(lam), 1e-9, 1 - 1e-9)
    return dict(bledy_wzgledem_trwania={k: blad(v) for k, v in ops.items()}, stala_skalarna=float(np.trace(Cyx) / np.trace(Cxx)),
                widmo_LS=dict(udzial_zespolonych=float((abs(lam.imag) > 1e-6).mean()), modul_med=float(np.median(mod)), modul_max=float(mod.max()),
                              czas_zaniku_med_kroki=float(-100 / math.log(np.median(mod))), czas_zaniku_max_kroki=float(-100 / math.log(mod.max())),
                              asymetria=float(np.linalg.norm(A_ls - A_ls.T) / np.linalg.norm(A_ls + A_ls.T))))


wyn = dict(opis=__doc__.split('\n')[0], K=int(K), N=int(N), tokenow=int(TOK), wymiar=int(D), etapy={}, wczesny={})
try:
    from deeptime.decomposition import VAMP
except Exception as e:  # noqa: BLE001
    VAMP = None
    wyn['uwaga_vamp'] = f'deeptime niedostępny: {e}'

for e1, e2 in ETAPY:
    idx = [i for i, k in enumerate(kroki) if k >= 1000 and k % 100 == 0 and EP * e1 <= k < EP * e2]
    n = len(idx)
    X = np.asarray(rp[idx]).astype(np.float32)                                            # (n, N, 600, D)
    mod_ = X.mean(2, keepdims=True)
    Xr = X - mod_
    udzial_modu = float((np.diff(mod_, axis=0) ** 2).sum() * TOK / (np.diff(X, axis=0) ** 2).sum())
    # 1. MSD i α
    N6_0, dnn, dsh = zip(*[knn6(Xr[0, s]) for s in range(N)])
    d_shell = float(np.median(dsh)); d_nn = float(np.median(dnn))
    Lmax = min(40, n - 2)
    msd_r = np.array([((Xr[L:] - Xr[:-L]) ** 2).sum(-1).mean() for L in range(1, Lmax + 1)])
    msd_a = np.array([((X[L:] - X[:-L]) ** 2).sum(-1).mean() for L in range(1, Lmax + 1)])
    Ls = np.arange(1, Lmax + 1)
    sel = (Ls >= 2) & (Ls <= 20)
    alfa, _ = np.polyfit(np.log(Ls[sel]), np.log(msd_r[sel]), 1)
    alfa_a, _ = np.polyfit(np.log(Ls[sel]), np.log(msd_a[sel]), 1)
    # 2. trwałość sąsiedztwa
    jacc = {}
    for L in (1, 5, 10, 20, 40):
        if L >= n:
            continue
        v = []
        for s in range(N):
            N6_L, _, _ = knn6(Xr[L, s])
            v.append(np.mean([len(set(N6_0[s][i]) & set(N6_L[i])) / len(set(N6_0[s][i]) | set(N6_L[i])) for i in range(0, TOK, 3)]))
        jacc[L] = float(np.mean(v))
    # 3. operator na drganiach wokół średniej etapu
    Xt = Xr - Xr.mean(0, keepdims=True)
    op = operatory(Xt)
    # 4. VAMP
    vamp = {}
    if VAMP is not None:
        traj = [Xt[:, s, i].astype(np.float64) for s in range(N) for i in range(0, TOK, 10)]
        for L in LAGI_VAMP:
            if L * 3 >= n:
                continue
            m_ = VAMP(lagtime=L, dim=8).fit(traj).fetch_model()
            sv = np.clip(m_.singular_values[:8], 1e-9, 1 - 1e-9)
            vamp[L] = dict(sv=[round(float(v), 4) for v in sv], czasy_kroki=[round(float(-L * 100 / math.log(v)), 0) for v in sv])
        del traj
    # 6. sygnał praktyczny
    ruch_zdj = ((Xr[1:] - Xr[:-1]) ** 2).sum(-1).mean((0, 2))                            # (N,) MSD(1) zdjęcia
    oc = oceny[idx].mean(0)                                                                 # (N, 2) pewność, IoU
    syg = {}
    for nazwa, m in (('TRAIN', split == 0), ('VALID', split == 1), ('wszystkie', np.ones(N, bool))):
        syg[nazwa] = dict(spearman_ruch_pewnosc=float(spearmanr(ruch_zdj[m], oc[m, 0]).statistic), spearman_ruch_iou=float(spearmanr(ruch_zdj[m], oc[m, 1]).statistic),
                          ruch_sredni=float(ruch_zdj[m].mean()), pewnosc=float(oc[m, 0].mean()), iou=float(oc[m, 1].mean()))
    # 5. kontrola rzutu na 4 zdjęciach
    P = np.asarray(p768[idx]).astype(np.float32)
    Pr = P - P.mean(2, keepdims=True)
    pel = META['pelne768_indeksy']
    msd768 = float(((Pr[1:] - Pr[:-1]) ** 2).sum(-1).mean()); msd128 = float(((Xr[1:, pel] - Xr[:-1, pel]) ** 2).sum(-1).mean())
    _, _, dsh768 = zip(*[knn6(Pr[0, j]) for j in range(len(pel))])
    kontrola = dict(msd1_768=msd768, msd1_128=msd128, stosunek=msd768 / msd128, d_shell_768=float(np.median(dsh768)),
                    d_shell_128_tych_zdjec=float(np.median([knn6(Xr[0, s])[2] for s in pel])))
    wyn['etapy'][f'{e1}-{e2}'] = dict(checkpointow=n, kroki=[int(kroki[idx[0]]), int(kroki[idx[-1]])], d_nn=d_nn, d_shell=d_shell,
                                      msd_wzgledne=[float(v) for v in msd_r], msd_bezwzgledne=[float(v) for v in msd_a], L=[int(v) for v in Ls],
                                      msd1_do_dshell2=float(msd_r[0] / d_shell ** 2), przesuniecie_1_do_dshell=float(math.sqrt(msd_r[0]) / d_shell),
                                      alfa_wzgledne=float(alfa), alfa_bezwzgledne=float(alfa_a), jaccard=jacc, jaccard_losowy=6 / TOK, operator=op, vamp=vamp,
                                      sygnal=syg, kontrola_768=kontrola, udzial_modu_zdjecia=udzial_modu)
    log(f'epoki {e1:2d}–{e2:2d} ({n} ckpt): √MSD(100 kroków)/d_shell {math.sqrt(msd_r[0]) / d_shell:.2f}, α_wzgl {alfa:.2f} α_bezwzgl {alfa_a:.2f}, Jaccard {[round(v, 2) for v in jacc.values()]}, '
        f'operator: LS {op["bledy_wzgledem_trwania"]["najmniejsze_kwadraty"]:.2f} sym {op["bledy_wzgledem_trwania"]["symetryczny"]:.2f} orto {op["bledy_wzgledem_trwania"]["ortogonalny"]:.2f} '
        f'skal {op["bledy_wzgledem_trwania"]["skalarny"]:.2f} (stała {op["stala_skalarna"]:.2f}), VAMP czasy τ=1→{vamp.get(1, {}).get("czasy_kroki", ["-"])[:3]} τ=8→{vamp.get(8, {}).get("czasy_kroki", ["-"])[:3]}, '
        f'768/128 {kontrola["stosunek"]:.2f}, Spearman ruch–IoU TRAIN {syg["TRAIN"]["spearman_ruch_iou"]:+.2f} VALID {syg["VALID"]["spearman_ruch_iou"]:+.2f}, mod {udzial_modu:.0%}')
    del X, Xr, Xt, P, Pr

# wczesny etap (co 10 kroków do 1000) tylko MSD i α, dla porównania ze skalą kroku
idx = [i for i, k in enumerate(kroki) if 100 <= k <= 1000 and k % 10 == 0]
X = np.asarray(rp[idx]).astype(np.float32); Xr = X - X.mean(2, keepdims=True)
_, _, dsh = zip(*[knn6(Xr[0, s]) for s in range(N)])
msd_r = np.array([((Xr[L:] - Xr[:-L]) ** 2).sum(-1).mean() for L in range(1, 41)])
Ls = np.arange(1, 41); sel = (Ls >= 2) & (Ls <= 20)
alfa, _ = np.polyfit(np.log(Ls[sel]), np.log(msd_r[sel]), 1)
wyn['wczesny'] = dict(kroki=[int(kroki[idx[0]]), int(kroki[idx[-1]])], co=10, d_shell=float(np.median(dsh)), msd_wzgledne=[float(v) for v in msd_r], alfa_wzgledne=float(alfa),
                      przesuniecie_1_do_dshell=float(math.sqrt(msd_r[0]) / np.median(dsh)))
log(f'wczesny (kroki 100–1000 co 10): √MSD(10 kroków)/d_shell {wyn["wczesny"]["przesuniecie_1_do_dshell"]:.2f}, α {alfa:.2f}')
json.dump(wyn, open(os.path.join(HERE, 'RUCH60.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99',
                     'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a',
                     'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
kol = plt.cm.plasma(np.linspace(0.15, 0.95, len(ETAPY)))
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), dpi=100)
E = wyn['etapy']
a = ax[0, 0]
for c, (k, v) in zip(kol, E.items()):
    a.plot(np.array(v['L']) * 100, np.array(v['msd_wzgledne']) / v['d_shell'] ** 2, color=c, label=f'epoki {k} (α={v["alfa_wzgledne"]:.2f})')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('opóźnienie [kroki]'); a.set_ylabel('MSD / d_shell²'); a.legend(fontsize=7); a.grid(True, alpha=0.4)
a.set_title('1. MSD w skali epok (względne, bez modu zdjęcia)', fontsize=10)
a = ax[0, 1]
for c, (k, v) in zip(kol, E.items()):
    Lj = sorted(v['jaccard']); a.plot(np.array(Lj) * 100, [v['jaccard'][L] for L in Lj], 'o-', color=c, label=f'epoki {k}')
a.axhline(6 / TOK, color='#8a8f99', ls='--', lw=1, label='losowo 6/600'); a.set_xscale('log'); a.set_xlabel('opóźnienie [kroki]'); a.set_ylabel('Jaccard 6 sąsiadów'); a.set_ylim(0, 1)
a.legend(fontsize=7); a.grid(True, alpha=0.4); a.set_title('2. trwałość sąsiedztwa: ciało stałe trzyma sąsiadów', fontsize=10)
a = ax[0, 2]
nazwy = ['najmniejsze_kwadraty', 'symetryczny', 'skosnie_symetryczny', 'ortogonalny', 'diagonalny', 'skalarny', 'zero (centrum)']
w = 0.8 / len(E)
for j, (c, (k, v)) in enumerate(zip(kol, E.items())):
    a.bar(np.arange(len(nazwy)) + j * w, [v['operator']['bledy_wzgledem_trwania'][nm] for nm in nazwy], w, color=c, label=f'epoki {k}')
a.axhline(1, color='#8a8f99', ls='--', lw=1); a.set_xticks(np.arange(len(nazwy)) + 0.4); a.set_xticklabels([nm.replace('_', ' ') for nm in nazwy], rotation=25, ha='right', fontsize=8)
a.set_ylabel('błąd 1 kroku (100 kroków) / trwanie'); a.legend(fontsize=7); a.grid(True, axis='y', alpha=0.4); a.set_title('3. struktura operatora (piDMD) co 100 kroków', fontsize=10)
a = ax[1, 0]
for c, (k, v) in zip(kol, E.items()):
    if v['vamp']:
        Lv = sorted(v['vamp']); a.plot(np.array(Lv) * 100, [v['vamp'][L]['czasy_kroki'][0] for L in Lv], 'o-', color=c, label=f'epoki {k}, proces 1')
        a.plot(np.array(Lv) * 100, [v['vamp'][L]['czasy_kroki'][2] for L in Lv], 's--', color=c, lw=0.8, ms=3)
xs = np.array(LAGI_VAMP) * 100; a.plot(xs, xs, ':', color='#8a8f99', lw=1, label='granica rozdzielczości')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('opóźnienie τ [kroki]'); a.set_ylabel('czas implikowany [kroki]'); a.legend(fontsize=7); a.grid(True, alpha=0.4)
a.set_title('4. test Markowa w skali epok (VAMP; kreskowane: proces 3)', fontsize=10)
a = ax[1, 1]
ep_mid = [(e1 + e2) / 2 for e1, e2 in ETAPY]
a.plot(ep_mid, [v['przesuniecie_1_do_dshell'] for v in E.values()], 'o-', color='#ffffff', label='√MSD(100 kroków) / d_shell')
a.plot(ep_mid, [v['alfa_wzgledne'] for v in E.values()], 's-', color='#ffd166', label='α (MSD ∝ L^α), względne')
a.plot(ep_mid, [v['alfa_bezwzgledne'] for v in E.values()], 's--', color='#ffd166', lw=0.8, label='α bezwzględne')
a.plot(ep_mid, [v['kontrola_768']['stosunek'] for v in E.values()], '^-', color='#4a8cff', label='MSD 768D / 128D (4 zdjęcia)')
a.plot(ep_mid, [v['udzial_modu_zdjecia'] for v in E.values()], 'd-', color='#ff4433', label='udział modu zdjęcia w energii')
a.set_xlabel('epoka'); a.legend(fontsize=7); a.grid(True, alpha=0.4); a.set_title('5. ruch a epoka', fontsize=10)
a = ax[1, 2]
a.plot(ep_mid, [v['sygnal']['TRAIN']['spearman_ruch_iou'] for v in E.values()], 'o-', color='#ffffff', label='Spearman ruch–IoU, TRAIN')
a.plot(ep_mid, [v['sygnal']['VALID']['spearman_ruch_iou'] for v in E.values()], 'o--', color='#ffffff', label='VALID')
a.plot(ep_mid, [v['sygnal']['TRAIN']['spearman_ruch_pewnosc'] for v in E.values()], 's-', color='#c77dff', label='Spearman ruch–pewność, TRAIN')
a.plot(ep_mid, [v['sygnal']['VALID']['spearman_ruch_pewnosc'] for v in E.values()], 's--', color='#c77dff', label='VALID')
a.axhline(0, color='#8a8f99', lw=0.8); a.set_xlabel('epoka'); a.set_ylabel('Spearman po 50 zdjęciach'); a.legend(fontsize=7); a.grid(True, alpha=0.4)
a.set_title('6. ruchliwość zdjęcia a jego błąd', fontsize=10)
fig.suptitle(f'Ruch tokenów przez 60 epok: {N} zdjęć × {TOK} tokenów, rzut 128D, checkpointy co 100 kroków', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'ruch60.png'))
log('zapisano RUCH60.json, ruch60.png')
