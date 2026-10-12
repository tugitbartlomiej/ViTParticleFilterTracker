"""Odciski palca liczone WZGLĘDEM KLATKI (cage-relative), jak w badaniach szkieł 2D (Illing i in. 2017, Vivek i in. 2017):
przesunięcie tokenu minus średnie przesunięcie jego sąsiadów. Usuwa to przepływ kolektywny każdej skali (globalny, zdjęcia,
obszaru) i zostawia samo grzechotanie w klatce. Sąsiedzi: 6 najbliższych w latencie wewnątrz zdjęcia (w chwili startu)
oraz, dla kontroli, 4 sąsiedzi na siatce patchy.
Wynik: ODCISKI_KLATKOWE.json, odciski_klatkowe.png.  Uruchomienie: F:/Python/3.13/python.exe odciski_klatkowe.py
"""
import glob
import json
import os
import time

import numpy as np

G = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty'
E35K = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T162654_natural_78cd84bd6edd'
OUT = os.path.dirname(os.path.abspath(__file__))
NS = 10
H, W = 20, 30
N = NS * 600
OBSZARY = ('tło', 'rogówka', 'źrenica', 'narzędzie')
T_STAC = 300
GESTE_DO = 2188
KK = 6
rng = np.random.default_rng(0)
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

idx = np.arange(600)
rr, cc = idx // W, idx % W
SAS_SIATKA = []            # lista 4 sąsiadów (z powtórzeniem brzegowym, żeby mieć stałą liczbę 4)
for i in range(600):
    r, c = rr[i], cc[i]
    s = [i + dc + dr * W for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)) if 0 <= r + dr < H and 0 <= c + dc < W]
    while len(s) < 4:
        s.append(s[0])
    SAS_SIATKA.append(s)
SAS_SIATKA = np.concatenate([np.asarray(SAS_SIATKA) + im * 600 for im in range(NS)])      # (N, 4)
PARY_SIATKI = np.concatenate([np.stack([np.concatenate([idx[cc < W - 1], idx[rr < H - 1]]),
                                        np.concatenate([idx[cc < W - 1] + 1, idx[rr < H - 1] + W])], 1) + im * 600 for im in range(NS)])


def odleglosci(x):
    n2 = (x ** 2).sum(1)
    D = n2[:, None] + n2[None, :] - 2 * x @ x.T
    np.maximum(D, 0, out=D)
    np.fill_diagonal(D, np.inf)
    return np.sqrt(D)


_knn_cache = {}


def knn(k):
    if k not in _knn_cache:
        x = Z[ind[k]]
        out, dnn = [], []
        for im in range(NS):
            D = odleglosci(x[im * 600:(im + 1) * 600])
            out.append(np.argpartition(D, KK, axis=1)[:, :KK] + im * 600)
            dnn.append(D.min(1))
        _knn_cache[k] = (np.concatenate(out), np.concatenate(dnn))
    return _knn_cache[k]


def przes_cr(a, b, rodzaj='knn'):
    """Przesunięcie a→b względem klatki: minus średnie przesunięcie sąsiadów (sąsiedzi wyznaczeni w chwili a)."""
    d = Z[ind[b]] - Z[ind[a]]
    sas = knn(a)[0] if rodzaj == 'knn' else SAS_SIATKA
    return d - d[sas].mean(1)


def przes_pole(a, b):
    d = Z[ind[b]] - Z[ind[a]]
    return d - d.mean(0)


def starty(tau, n=40, od=T_STAC, do=GESTE_DO):
    return np.unique(np.linspace(od, do - tau, n).astype(int))


d_nn = float(np.median(knn(T_STAC)[1]))
d_nn_ob = [float(np.median(knn(T_STAC)[1][klasy == c])) for c in range(4)]
x = Z[ind[T_STAC]]
R_obraz = float(np.median(np.concatenate([np.linalg.norm(x[im * 600:(im + 1) * 600] - x[im * 600:(im + 1) * 600].mean(0), axis=1) for im in range(NS)])))
d_siatka = float(np.median(np.linalg.norm(x[PARY_SIATKI[:, 0]] - x[PARY_SIATKI[:, 1]], axis=1)))
q = 2 * np.pi / d_nn
U = rng.standard_normal((64, 32))
U /= np.linalg.norm(U, axis=1, keepdims=True)
log('d_nn', round(d_nn, 3), 'd_siatka', round(d_siatka, 3), 'R_obraz', round(R_obraz, 3))

TAUS = np.unique(np.round(np.logspace(0, np.log10(1800), 30)).astype(int))
wyn = {}
for rodzaj in ('knn', 'siatka'):
    msd = np.zeros((len(TAUS), 4))
    msd_all = np.zeros(len(TAUS))
    a2_1d = np.zeros(len(TAUS))
    a2_d = np.zeros(len(TAUS))
    fs1 = np.zeros(len(TAUS))
    fs05 = np.zeros(len(TAUS))
    cos_sas = np.zeros(len(TAUS))
    ucieczki = np.zeros(len(TAUS))
    ogony = {}
    for it, tau in enumerate(TAUS):
        S = starty(tau, 40 if tau < 300 else 24)
        m2 = np.zeros(32)
        m4 = np.zeros(32)
        C = np.zeros((32, 32))
        r2s, f1, f05, cs, uc, ds = [], 0.0, 0.0, 0.0, 0.0, []
        for s in S:
            d = przes_cr(s, s + tau, rodzaj)
            r2 = (d ** 2).sum(1)
            r2s.append(r2)
            m2 += (d ** 2).mean(0)
            m4 += (d ** 4).mean(0)
            C += d.T @ d / N
            pr = d @ U.T
            f1 += np.cos(q * pr).mean()
            f05 += np.cos(0.5 * q * pr).mean()
            n = np.linalg.norm(d, axis=1)
            i, j = PARY_SIATKI[:, 0], PARY_SIATKI[:, 1]
            cs += ((d[i] * d[j]).sum(1) / np.maximum(n[i] * n[j], 1e-20)).mean()
            uc += (np.sqrt(r2) > 0.5 * d_nn).mean()
            if tau in (1, 10, 50, 300, 1000):
                ds.append(d / d.std(0))
        r2 = np.stack(r2s)
        msd[it] = [r2[:, klasy == c].mean() for c in range(4)]
        msd_all[it] = r2.mean()
        m2 /= len(S)
        m4 /= len(S)
        C /= len(S)
        a2_1d[it] = float((m4 / (3 * m2 ** 2) - 1).mean())
        a2_d[it] = float((r2 ** 2).mean() / r2.mean() ** 2 - (1 + 2 * np.trace(C @ C) / np.trace(C) ** 2))
        fs1[it] = f1 / len(S)
        fs05[it] = f05 / len(S)
        cos_sas[it] = cs / len(S)
        ucieczki[it] = uc / len(S)
        if ds:
            u = np.abs(np.concatenate(ds).reshape(-1))
            ogony[int(tau)] = dict(udzial_powyzej_3sigma=float((u > 3).mean()), udzial_powyzej_4sigma=float((u > 4).mean()), gauss_3sigma=0.0027, gauss_4sigma=6.3e-5)
        if it % 5 == 0:
            log(rodzaj, 'tau', int(tau), 'msd', round(float(msd_all[it]), 4), 'a2', round(float(a2_1d[it]), 3), 'Fs', round(float(fs1[it]), 3), 'cos_sas', round(float(cos_sas[it]), 3))
    lt = np.log(TAUS.astype(float))
    lm = np.log(msd_all)
    alfa = np.full(len(TAUS), np.nan)
    alfa[1:-1] = (lm[2:] - lm[:-2]) / (lt[2:] - lt[:-2])
    plateau = (TAUS >= 35) & (TAUS <= 80)
    l_kl = float(np.sqrt(msd_all[plateau].mean()))
    f_q = float(fs1[plateau].mean())
    sel = (TAUS >= 100) & (fs1 > 0.02) & (fs1 < 0.98 * f_q)
    beta = tau_alfa = float('nan')
    if sel.sum() >= 3:
        yy = np.log(-np.log(fs1[sel] / f_q))
        A = np.stack([np.ones(sel.sum()), lt[sel]], 1)
        (b0, b1), *_ = np.linalg.lstsq(A, yy, rcond=None)
        beta, tau_alfa = float(b1), float(np.exp(-b0 / b1))

    def M(xq):
        return np.exp(np.interp(np.log(xq), lt, lm))

    c_tau = {int(xq): float(1 - M(2 * xq) / (2 * M(xq))) for xq in (1, 2, 5, 10, 20, 40, 60, 100, 200, 500, 800)}
    # długie czasy (co 10, t_w ≈ 2200–2700) i starzenie
    TAU_L = (10, 30, 100, 300, 1000, 3000, 10000, 20000)
    dl = {}
    for tau in TAU_L:
        v = [(przes_cr(s, s + tau, rodzaj), ) for s in range(2200, 2700, 100) if s + tau in ind]
        if v:
            dl[tau] = dict(msd=float(np.mean([(d[0] ** 2).sum(1).mean() for d in v])), Fs=float(np.mean([np.cos(q * (d[0] @ U.T)).mean() for d in v])),
                           ucieczki=float(np.mean([(np.linalg.norm(d[0], axis=1) > 0.5 * d_nn).mean() for d in v])))
    TW_S, TAU_S = (300, 1000, 2000, 3000, 5000, 10000, 15000, 20000), (10, 100, 1000)
    msd_tw = {}
    for tw_ in TW_S:
        msd_tw[tw_] = {}
        for tau in TAU_S:
            v = [(przes_cr(s, s + tau, rodzaj) ** 2).sum(1).mean() for s in range(tw_, tw_ + 50, 10) if s in ind and s + tau in ind]
            if v:
                msd_tw[tw_][tau] = float(np.mean(v))
    mu = {}
    for tau in TAU_S:
        pts = [(tw_, msd_tw[tw_][tau]) for tw_ in TW_S if tau in msd_tw[tw_]]
        if len(pts) >= 3:
            xs, ys = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            A = np.stack([np.ones(len(xs)), xs], 1)
            mu[tau] = float(-np.linalg.lstsq(A, ys, rcond=None)[0][1])
    wyn[rodzaj] = dict(taus=[int(t) for t in TAUS], msd=[float(v) for v in msd_all], msd_obszary=msd.tolist(),
                       alfa_lokalna=[None if np.isnan(v) else float(v) for v in alfa],
                       rezimy=dict(tau_1_3=float(np.nanmean(alfa[(TAUS >= 2) & (TAUS <= 3)])), plateau_35_80=float(np.nanmean(alfa[plateau])),
                                   tau_800_1500=float(np.nanmean(alfa[(TAUS >= 800) & (TAUS <= 1500)]))),
                       klatka=dict(l_klatka=l_kl, lindemann_l_do_dnn=l_kl / d_nn, l_do_d_siatka=l_kl / d_siatka, l_do_R_obraz=l_kl / R_obraz,
                                   lindemann_obszary=[float(np.sqrt(msd[plateau, c].mean())) / d_nn_ob[c] for c in range(4)]),
                       a2_1d=[float(v) for v in a2_1d], a2_pelny=[float(v) for v in a2_d],
                       szczyt_a2=dict(wartosc=float(a2_1d[TAUS >= 3].max()), tau=int(TAUS[TAUS >= 3][int(np.argmax(a2_1d[TAUS >= 3]))])),
                       ogony=ogony, Fs_q1=[float(v) for v in fs1], Fs_q05=[float(v) for v in fs05], plateau_f_q=f_q, beta=beta, tau_alfa=tau_alfa,
                       cos_sasiadow_po_usunieciu_klatki=[float(v) for v in cos_sas], udzial_uciekajacych_ponad_pol_dnn=[float(v) for v in ucieczki],
                       antykorelacja_c=c_tau, dlugie=dl, msd_tw=msd_tw, wykladnik_mu=mu)
    log(rodzaj, 'klatka', round(l_kl, 3), 'Lindemann', round(l_kl / d_nn, 3), 'f_q', round(f_q, 3), 'beta', round(beta, 2), 'tau_alfa', round(tau_alfa, 0), 'mu', mu)

# kontrola: te same wielkości dla przesunięcia tylko bez pola globalnego (jak w pierwszym skrypcie), do zestawienia na rysunku
msd_pole = np.array([np.mean([(przes_pole(s, s + tau) ** 2).sum(1).mean() for s in starty(tau, 16)]) for tau in TAUS])
wyn['bez_pola_globalnego'] = dict(taus=[int(t) for t in TAUS], msd=[float(v) for v in msd_pole])
wyn['skale'] = dict(d_nn=d_nn, d_nn_obszary=d_nn_ob, d_siatka=d_siatka, R_obraz=R_obraz, q=float(q))
wyn['opis'] = 'Przesunięcia względem klatki (minus średnie przesunięcie 6 sąsiadów w latencie / 4 sąsiadów na siatce), 6000 tokenów, PCA32, t ≥ 300.'
wyn['czas_s'] = round(time.time() - t0)
json.dump(wyn, open(os.path.join(OUT, 'ODCISKI_KLATKOWE.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(1, 3, figsize=(19, 6.2), dpi=100)
a = ax[0]
a.loglog(TAUS, msd_pole, color='#8a8f99', lw=1.6, label='bez pola globalnego (populacja)')
a.loglog(TAUS, wyn['siatka']['msd'], color='#4a8cff', lw=1.6, label='względem 4 sąsiadów na siatce')
a.loglog(TAUS, wyn['knn']['msd'], color='#ffd166', lw=2.4, label='względem 6 sąsiadów w latencie (klatka)')
a.axhline(d_nn ** 2, color='w', ls='--', lw=0.8)
a.text(TAUS[0], d_nn ** 2 * 1.15, 'd_nn²', fontsize=9)
a.axhline((0.1 * d_nn) ** 2, color='#ff4433', ls=':', lw=0.8)
a.text(TAUS[0], (0.1 * d_nn) ** 2 * 1.15, '(0,1 d_nn)²  Lindemann', fontsize=9, color='#ff4433')
k = wyn['knn']
for tq in (2, 50, 1000):
    i = int(np.argmin(abs(TAUS - tq)))
    if k['alfa_lokalna'][i] is not None:
        a.annotate(f"α={k['alfa_lokalna'][i]:.2f}", (TAUS[i], k['msd'][i]), textcoords='offset points', xytext=(6, -14), fontsize=9, color='#ffd166')
a.set_xlabel('τ (kroki)')
a.set_ylabel('MSD (PCA32)')
a.set_title(f"MSD względem klatki: ℓ/d_nn = {k['klatka']['lindemann_l_do_dnn']:.2f} (populacja: {np.sqrt(msd_pole[(TAUS >= 35) & (TAUS <= 80)].mean()) / d_nn:.2f})")
a.grid(True, which='both', alpha=0.4)
a.legend(fontsize=9)
a = ax[1]
a.semilogx(TAUS, k['a2_1d'], color='#ffd166', lw=2.2, label='α₂ 1D (klatka, latent)')
a.semilogx(TAUS, wyn['siatka']['a2_1d'], color='#4a8cff', lw=1.4, label='α₂ 1D (klatka, siatka)')
a.semilogx(TAUS, k['a2_pelny'], color='#ff4433', lw=1.4, label='α₂ 32D (klatka, latent)')
a.axhline(0, color='w', lw=0.6)
a.set_xlabel('τ (kroki)')
a.set_ylabel('α₂')
a.set_title(f"Parametr niegaussowski: szczyt {k['szczyt_a2']['wartosc']:.2f} przy τ = {k['szczyt_a2']['tau']}")
a.grid(True, which='both', alpha=0.4)
a.legend(fontsize=9)
a = ax[2]
a.semilogx(TAUS, k['Fs_q1'], color='#ffd166', lw=2.2, label='Fs, q = 2π/d_nn (t_w 300–2188)')
a.semilogx(TAUS, k['Fs_q05'], color='#4cd964', lw=1.4, label='Fs, q = π/d_nn')
if k['dlugie']:
    a.semilogx(list(k['dlugie'].keys()), [v['Fs'] for v in k['dlugie'].values()], 'o-', mfc='none', color='#ffd166', lw=1, label='q = 2π/d_nn, t_w ≈ 2200–2700')
a.axhline(np.exp(-1), color='w', ls=':', lw=0.8)
a.set_ylim(-0.05, 1.05)
a.set_xlabel('τ (kroki)')
a.set_ylabel('Fs(q, τ) względem klatki')
a.set_title(f"plateau f_q = {k['plateau_f_q']:.2f}, β = {k['beta']:.2f}, τ_α ≈ {k['tau_alfa']:.0f}")
a.grid(True, which='both', alpha=0.4)
a.legend(fontsize=9)
fig.suptitle('Grzechotanie w klatce po usunięciu ruchu sąsiadów (6000 cząsteczek ViT, bieg 35 000, PCA32)', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, 'odciski_klatkowe.png'))
log('GOTOWE')
