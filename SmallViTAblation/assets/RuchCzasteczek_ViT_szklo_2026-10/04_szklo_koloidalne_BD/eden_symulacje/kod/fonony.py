"""Fonony: czy drgania tokenów mają strukturę falową na siatce patchy 20 × 30 (jak w kryształach koloidalnych: Keim 2004, Chen 2012),
czy są białe w przestrzeni (niezależne węzły / szkło). Pole przesunięć tokenów jednego zdjęcia Δx(r, t) ∈ R^D na siatce r = (wiersz, kolumna).
  1. widmo przestrzenne: 2D FFT pola przesunięć (osobno każdy wymiar), moc |Δx̂(k)|² uśredniona po czasie, zdjęciach i wymiarach, w pasmach |k|;
     kontrola: tokeny przetasowane na siatce (białe widmo). Kryształ sprężysty: moc ∝ 1/|k|² przy małych k; podajemy wykładnik.
  2. dyspersja relaksacji: autokorelacja czasowa każdego modu Fouriera, czas 1/e w funkcji |k|; fonony tłumione: długie fale relaksują wolniej.
  3. udział modów długofalowych (|k| ≤ 2 najniższe pasma) w energii drgań, kontra kontrola.
Dane: ruch60 (checkpointy co 100 kroków, etapy) albo gęste okno RP128 (co 1 krok). Współrzędne względne (bez modu zdjęcia).
Uruchomienie: python fonony.py ruch60 <eksport> <out> | python fonony.py gesty <gesty> <od> <do> <n_sond> <out> <nazwa>"""
import glob
import json
import os
import sys
import time

import numpy as np

t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
TRYB = sys.argv[1]
H, W = 20, 30
TOK = H * W
ky = np.fft.fftfreq(H) * H; kx = np.fft.fftfreq(W) * W                       # liczby falowe w jednostkach 2π/L
KK = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)                             # (H, W) |k|
PASMA = np.array([0.5, 1.5, 2.5, 3.5, 5, 7, 10, 15])
rng = np.random.default_rng(0)


def analiza(X, lag, taus):
    """X (T, N, 600, D) względne; lag w indeksach. Zwraca widmo mocy w pasmach, kontrolę, dyspersję czasów relaksacji."""
    T, N, _, D = X.shape
    dX = X[lag:] - X[:-lag]                                                    # (T−lag, N, 600, D)
    P = np.zeros((H, W)); Pk = np.zeros((H, W))
    perm = rng.permutation(TOK)
    F_all = []
    for n in range(N):
        pole = dX[:, n].reshape(-1, H, W, D)                                    # (t, H, W, D)
        F = np.fft.fft2(pole, axes=(1, 2))                                      # (t, H, W, D)
        P += (abs(F) ** 2).sum((0, 3))
        Fk = np.fft.fft2(dX[:, n][:, perm].reshape(-1, H, W, D), axes=(1, 2)); Pk += (abs(Fk) ** 2).sum((0, 3))
        F_all.append(F)
    P[0, 0] = Pk[0, 0] = 0                                                      # k = 0 to mod zdjęcia (odjęty)
    pasma = {}
    for a, b in zip(PASMA[:-1], PASMA[1:]):
        m = (KK >= a) & (KK < b)
        if m.any():
            pasma[f'{a}-{b}'] = dict(k_sr=float(KK[m].mean()), moc=float(P[m].mean() / P[KK > 0].mean()), kontrola=float(Pk[m].mean() / Pk[KK > 0].mean()), n_modow=int(m.sum()))
    ks = np.array([v['k_sr'] for v in pasma.values()]); ms = np.array([v['moc'] for v in pasma.values()])
    wykl = float(np.polyfit(np.log(ks[:5]), np.log(ms[:5]), 1)[0])
    m_dl = (KK > 0) & (KK < 2.5)
    udzial_dl = float(P[m_dl].sum() / P[KK > 0].sum()); udzial_dl_k = float(Pk[m_dl].sum() / Pk[KK > 0].sum())
    # dyspersja: autokorelacja czasowa modów położenia (nie przesunięć) w pasmach
    disp = {}
    Fx = [np.fft.fft2((X[:, n] - X[:, n].mean(0)).reshape(T, H, W, D), axes=(1, 2)) for n in range(N)]      # (T, H, W, D) mody położenia
    for a, b in zip(PASMA[:-1], PASMA[1:]):
        m = (KK >= a) & (KK < b)
        if not m.any():
            continue
        C = []
        for tau in taus:
            if tau >= T:
                break
            num = den = 0.0
            for n in range(N):
                A = Fx[n][:T - tau][:, m]; B = Fx[n][tau:][:, m]
                num += float((A * B.conj()).real.sum()); den += float((abs(Fx[n][:, m]) ** 2).sum()) * (T - tau) / T
            C.append(num / den)
        C = np.array(C); t_ = np.array(taus[:len(C)], float)
        pon = np.where(C < np.exp(-1))[0]
        tau_e = float(t_[pon[0]]) if len(pon) else float('inf')
        disp[f'{a}-{b}'] = dict(k_sr=float(KK[m].mean()), C=[round(float(c), 4) for c in C], tau_1e=tau_e)
    return dict(widmo=pasma, wykladnik_mocy_male_k=wykl, udzial_dlugofalowy=udzial_dl, udzial_dlugofalowy_kontrola=udzial_dl_k, dyspersja=disp, taus=list(taus[:len(C)]))


if TRYB == 'ruch60':
    EKS, OUT = sys.argv[2], sys.argv[3]; os.makedirs(OUT, exist_ok=True)
    META = json.load(open(os.path.join(EKS, 'META.json'), encoding='utf-8')); kroki = np.array(META['kroki'])
    rp = np.load(os.path.join(EKS, 'rp128.npy'), mmap_mode='r'); EP = 1094
    wyn = {}
    for e1, e2 in ((1, 5), (10, 20), (30, 40), (50, 60)):
        idx = [i for i, k in enumerate(kroki) if k >= 1000 and k % 100 == 0 and EP * e1 <= k < EP * e2]
        X = np.asarray(rp[idx]).astype(np.float32); X -= X.mean(2, keepdims=True)
        w = analiza(X, 1, [1, 2, 3, 5, 8, 12, 20, 30]); w['krok_co'] = 100; wyn[f'{e1}-{e2}'] = w
        log(f'epoki {e1}–{e2}: moc w pasmach {[round(v["moc"], 2) for v in w["widmo"].values()]} (kontrola {[round(v["kontrola"], 2) for v in w["widmo"].values()]}), '
            f'wykładnik {w["wykladnik_mocy_male_k"]:.2f}, udział długofalowy {w["udzial_dlugofalowy"]:.2f} vs {w["udzial_dlugofalowy_kontrola"]:.2f}; τ_1e modów: {[(round(v["k_sr"], 1), v["tau_1e"]) for v in w["dyspersja"].values()]}')
        del X
    NAZWA = 'ruch60'
else:
    G, OD, DO, N_SOND, OUT, NAZWA = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]; os.makedirs(OUT, exist_ok=True)
    Xs, kr = [], []
    for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
        z = np.load(p); k = z['kroki']; m = (k >= OD) & (k <= DO)
        if m.any():
            Xs.append(z['pca32'][m][:, :N_SOND].astype(np.float32)); kr.append(k[m])
    X = np.concatenate(Xs); del Xs; X -= X.mean(2, keepdims=True)
    wyn = {}
    for lag in (1, 10, 100):
        w = analiza(X, lag, [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500]); w['krok_co'] = 1; wyn[f'lag_{lag}'] = w
        log(f'{NAZWA} lag {lag}: moc w pasmach {[round(v["moc"], 2) for v in w["widmo"].values()]} (kontrola {[round(v["kontrola"], 2) for v in w["widmo"].values()]}), '
            f'wykładnik {w["wykladnik_mocy_male_k"]:.2f}, udział długofalowy {w["udzial_dlugofalowy"]:.2f} vs {w["udzial_dlugofalowy_kontrola"]:.2f}; τ_1e modów: {[(round(v["k_sr"], 1), v["tau_1e"]) for v in w["dyspersja"].values()]}')
json.dump(dict(opis=__doc__.split('\n')[0], etapy=wyn), open(os.path.join(OUT, f'FONONY_{NAZWA}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    sys.exit(0)
plt.rcParams.update({'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8',
                     'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=100)
kol = plt.cm.plasma(np.linspace(0.1, 0.95, len(wyn)))
for c, (n, w) in zip(kol, wyn.items()):
    ks = [v['k_sr'] for v in w['widmo'].values()]
    ax[0].plot(ks, [v['moc'] for v in w['widmo'].values()], 'o-', color=c, label=f'{n} (wykł. {w["wykladnik_mocy_male_k"]:.2f})')
    ax[0].plot(ks, [v['kontrola'] for v in w['widmo'].values()], ':', color=c, lw=1)
    ax[1].plot([v['k_sr'] for v in w['dyspersja'].values()], [v['tau_1e'] * w['krok_co'] for v in w['dyspersja'].values()], 'o-', color=c, label=n)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('|k| [2π/L]'); ax[0].set_ylabel('moc / średnia (kropki: tokeny przetasowane)'); ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.4)
ax[0].set_title('widmo przestrzenne przesunięć na siatce patchy', fontsize=10)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('|k| [2π/L]'); ax[1].set_ylabel('czas relaksacji modu [kroki]'); ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.4)
ax[1].set_title('dyspersja: fonony tłumione = długie fale relaksują wolniej', fontsize=10)
fig.suptitle(f'Fonony tokenów: {NAZWA}', fontsize=12); fig.tight_layout(); fig.savefig(os.path.join(OUT, f'fonony_{NAZWA}.png'))
log('zapisano')
