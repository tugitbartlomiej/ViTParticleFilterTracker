"""Kryształ czy ciecz: zmodyfikowany parametr Lindemanna Zahna–Mareta (PRL 85, 3656) na parach sąsiadów, bo cała siatka dryfuje.
γ_L(τ) = ⟨|Δu_i(τ) − Δu_j(τ)|²⟩_{j ∈ 6 NN(i)} / (2 a²), gdzie Δu = przesunięcie po czasie τ, a = odstęp sąsiadów na starcie okna.
Kryształ: γ_L nasyca się (2D: < 0,033; tu podajemy wartość i wykładnik wzrostu); ciecz: rośnie liniowo; heksatyk/pośrednie: potęgowo.
Dodatkowo trwałość kierunku wiązań (odpowiednik g6): ⟨cos(b_ij(t), b_ij(t+τ))⟩ dla wiązań sąsiadów, i parametr niegaussowski
względnych przesunięć sąsiadów (Zahn–Maret: różny w trzech fazach). Dwa źródła: (A) eksport ruch60 (checkpointy co 100 kroków, etapy
epok 1–5 … 50–60, τ do 4000 kroków) i (B) gęste okna RP128 biegów pokręteł (co 1 krok, τ do 800). Współrzędne względne (bez modu zdjęcia).
Uruchomienie: python lindemann_wzgledny.py ruch60 <eksport> <out> | python lindemann_wzgledny.py gesty <gesty> <od> <do> <n_sond> <out> <nazwa>"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist

t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
TRYB = sys.argv[1]
TOK = 600


def knn6(Xi):
    Dm = cdist(Xi, Xi); np.fill_diagonal(Dm, np.inf)
    N6 = np.argsort(Dm, 1)[:, :6]
    return N6, float(np.median(np.take_along_axis(Dm, N6, 1).mean(1)))


def lindemann(X, taus):
    """X (T, N, 600, D) współrzędne względne; zwraca γ_L(τ), MSD bezwzgl./a², trwałość wiązań, α₂ względne, a."""
    T, N = X.shape[:2]
    N6, a = zip(*[knn6(X[0, n]) for n in range(N)])
    N6 = np.stack(N6); a2 = np.median(a) ** 2
    ii = np.arange(TOK)[:, None]
    wyn = dict(a=float(np.sqrt(a2)), tau=[], gamma_L=[], msd_do_a2=[], trwalosc_wiazan=[], alfa2_wzgledne=[], udzial_par_dalej_niz_a=[])
    for tau in taus:
        if tau >= T:
            break
        g, m, c, a2w, dal = [], [], [], [], []
        for n in range(N):
            dU = X[tau:, n] - X[:-tau, n]                                          # (T−τ, 600, D)
            rel = dU[:, ii, :] - dU[:, N6[n], :]                                   # (T−τ, 600, 6, D) względne przesunięcia par
            r2 = (rel ** 2).sum(-1)
            g.append(r2.mean() / (2 * a2)); m.append((dU ** 2).sum(-1).mean() / a2)
            D_ = X.shape[-1]
            a2w.append(D_ / (D_ + 2) * (r2 ** 2).mean() / r2.mean() ** 2 - 1)
            dal.append((r2 > a2).mean())
            b0 = X[:-tau, n][:, N6[n], :] - X[:-tau, n][:, ii, :]; b1 = X[tau:, n][:, N6[n], :] - X[tau:, n][:, ii, :]
            c.append(((b0 * b1).sum(-1) / (np.linalg.norm(b0, axis=-1) * np.linalg.norm(b1, axis=-1) + 1e-12)).mean())
        for k, v in (('tau', tau), ('gamma_L', np.mean(g)), ('msd_do_a2', np.mean(m)), ('trwalosc_wiazan', np.mean(c)), ('alfa2_wzgledne', np.mean(a2w)), ('udzial_par_dalej_niz_a', np.mean(dal))):
            wyn[k].append(float(v))
    t_ = np.array(wyn['tau'], float); gl = np.array(wyn['gamma_L'])
    sel = t_ >= t_[len(t_) // 2]
    wyn['wykladnik_gamma_L_pozna_polowa'] = float(np.polyfit(np.log(t_[sel]), np.log(gl[sel]), 1)[0]) if sel.sum() >= 3 else None
    wyn['wykladnik_msd_pozna_polowa'] = float(np.polyfit(np.log(t_[sel]), np.log(np.array(wyn['msd_do_a2'])[sel]), 1)[0]) if sel.sum() >= 3 else None
    return wyn


if TRYB == 'ruch60':
    EKS, OUT = sys.argv[2], sys.argv[3]
    os.makedirs(OUT, exist_ok=True)
    META = json.load(open(os.path.join(EKS, 'META.json'), encoding='utf-8')); kroki = np.array(META['kroki'])
    rp = np.load(os.path.join(EKS, 'rp128.npy'), mmap_mode='r')
    EP = 1094
    wyn = {}
    for e1, e2 in ((1, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)):
        idx = [i for i, k in enumerate(kroki) if k >= 1000 and k % 100 == 0 and EP * e1 <= k < EP * e2]
        X = np.asarray(rp[idx]).astype(np.float32); X -= X.mean(2, keepdims=True)
        w = lindemann(X, [1, 2, 3, 5, 8, 12, 20, 30, 40]); w['krok_co'] = 100
        wyn[f'{e1}-{e2}'] = w
        log(f'epoki {e1}–{e2}: a {w["a"]:.2f}; γ_L(100 kroków) {w["gamma_L"][0]:.3f}, γ_L(4000) {w["gamma_L"][-1]:.3f} (wykładnik {w["wykladnik_gamma_L_pozna_polowa"]:.2f}); '
            f'MSD/a² {w["msd_do_a2"][0]:.2f} → {w["msd_do_a2"][-1]:.2f} (wykł. {w["wykladnik_msd_pozna_polowa"]:.2f}); wiązania cos {w["trwalosc_wiazan"][0]:.2f} → {w["trwalosc_wiazan"][-1]:.2f}; '
            f'α₂ wzgl {w["alfa2_wzgledne"][0]:.2f}; pary > a {w["udzial_par_dalej_niz_a"][-1]:.2f}')
        del X
    idx = [i for i, k in enumerate(kroki) if 100 <= k <= 1000 and k % 10 == 0]
    X = np.asarray(rp[idx]).astype(np.float32); X -= X.mean(2, keepdims=True)
    w = lindemann(X, [1, 2, 3, 5, 8, 12, 20, 30, 40, 60, 80]); w['krok_co'] = 10; wyn['wczesny_100_1000'] = w
    log(f'wczesny co 10: γ_L(10) {w["gamma_L"][0]:.3f}, γ_L(800) {w["gamma_L"][-1]:.3f} (wykł. {w["wykladnik_gamma_L_pozna_polowa"]:.2f}), MSD/a² {w["msd_do_a2"][-1]:.2f}')
    json.dump(dict(opis=__doc__.split('\n')[0], zrodlo='ruch60', etapy=wyn), open(os.path.join(OUT, 'LINDEMANN_ruch60.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    NAZWA = 'ruch60'
else:
    G, OD, DO, N_SOND, OUT, NAZWA = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
    os.makedirs(OUT, exist_ok=True)
    Xs, kr = [], []
    for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
        z = np.load(p); k = z['kroki']; m = (k >= OD) & (k <= DO)
        if m.any():
            Xs.append(z['pca32'][m][:, :N_SOND].astype(np.float32)); kr.append(k[m])
    X = np.concatenate(Xs); kr = np.concatenate(kr); del Xs
    assert (np.diff(kr) == 1).all()
    X -= X.mean(2, keepdims=True)
    w = lindemann(X, [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 800]); w['krok_co'] = 1
    log(f'{NAZWA}: a {w["a"]:.2f}; γ_L: ' + ' '.join(f'{t}:{g:.3f}' for t, g in zip(w['tau'], w['gamma_L'])) + f' (wykł. późny {w["wykladnik_gamma_L_pozna_polowa"]:.2f}); '
        f'MSD/a²: {w["msd_do_a2"][0]:.3f} → {w["msd_do_a2"][-1]:.2f} (wykł. {w["wykladnik_msd_pozna_polowa"]:.2f}); wiązania {w["trwalosc_wiazan"][0]:.2f} → {w["trwalosc_wiazan"][-1]:.2f}; α₂ wzgl {w["alfa2_wzgledne"][0]:.2f} → {w["alfa2_wzgledne"][-1]:.2f}')
    wyn = {NAZWA: w}
    json.dump(dict(opis=__doc__.split('\n')[0], zrodlo=G, okno=[int(kr[0]), int(kr[-1])], etapy=wyn), open(os.path.join(OUT, f'LINDEMANN_{NAZWA}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    sys.exit(0)
plt.rcParams.update({'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8',
                     'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(1, 3, figsize=(16, 5), dpi=100)
kol = plt.cm.plasma(np.linspace(0.1, 0.95, len(wyn)))
for c, (n, w) in zip(kol, wyn.items()):
    t_ = np.array(w['tau']) * w['krok_co']
    ax[0].plot(t_, w['gamma_L'], 'o-', color=c, label=f'{n} (wykł. {w["wykladnik_gamma_L_pozna_polowa"]:.2f})')
    ax[0].plot(t_, w['msd_do_a2'], ':', color=c, lw=1)
    ax[1].plot(t_, w['trwalosc_wiazan'], 'o-', color=c, label=n)
    ax[2].plot(t_, w['alfa2_wzgledne'], 'o-', color=c, label=n)
ax[0].axhline(0.033, color='#8a8f99', ls='--', lw=1, label='próg 2D Zahna–Mareta 0,033')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('τ [kroki]'); ax[0].set_ylabel('γ_L (ciągła), MSD/a² (kropki)'); ax[0].legend(fontsize=7); ax[0].grid(True, alpha=0.4)
ax[0].set_title('zmodyfikowany Lindemann: względem sąsiadów kontra bezwzględny', fontsize=10)
ax[1].set_xscale('log'); ax[1].set_xlabel('τ [kroki]'); ax[1].set_ylabel('⟨cos⟩ wiązania b_ij(t)·b_ij(t+τ)'); ax[1].set_ylim(0, 1); ax[1].legend(fontsize=7); ax[1].grid(True, alpha=0.4); ax[1].set_title('trwałość kierunku wiązań (odpowiednik g₆)', fontsize=10)
ax[2].set_xscale('log'); ax[2].set_xlabel('τ [kroki]'); ax[2].set_ylabel('α₂ względnych przesunięć sąsiadów'); ax[2].legend(fontsize=7); ax[2].grid(True, alpha=0.4); ax[2].set_title('niegaussowość względna', fontsize=10)
fig.suptitle(f'Kryterium Zahna–Mareta dla tokenów: {NAZWA}', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, f'lindemann_{NAZWA}.png'))
log('zapisano')
