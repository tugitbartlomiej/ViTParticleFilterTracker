"""Odciski palca dynamiki szklistej policzone wprost z gęstego zapisu trajektorii tokenów
(bieg 35 000 klatek, 10 sond × 600 tokenów, PCA32, co 1 krok do 2188, potem co 10 do 22 970).

Wszystko po odjęciu pola kolektywnego (średnie przesunięcie populacji w danym kroku), tak jak w śledzeniu koloidów
odejmuje się dryf całej próbki. Liczone wielkości, z których każda ma odpowiednik w literaturze o szkle koloidalnym:
  MSD(τ) i wykładnik lokalny, rozmiar klatki względem odległości do najbliższego sąsiada (kryterium Lindemanna),
  parametr niegaussowski α2(τ) i ogony rozkładu van Hove'a, funkcja rozpraszania Fs(q,τ) z wykładnikiem β,
  kooperatywność sąsiadów (siatka zdjęcia i sąsiedzi w latencie), skupianie szybkich cząstek,
  trwałość sąsiedztwa (czy klatka zachowuje sąsiadów), starzenie MSD(τ; t_w) i stygnięcie przy stałym lr.
Wynik: ODCISKI_SZKLA.json, odciski_szkla.png.  Uruchomienie: F:/Python/3.13/python.exe odciski_szkla.py
"""
import glob
import json
import os
import time

import numpy as np

G = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty'
E35K = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261009T162654_natural_78cd84bd6edd'
DANE = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/animacje_ruchu_20261010/edit/dane'
OUT = os.path.dirname(os.path.abspath(__file__))
NS = 10                      # sond w podpróbce (10 × 600 = 6000 tokenów)
H, W = 20, 30                # siatka patchy (480 × 320 / 16)
N = NS * 600
OBSZARY = ('tło', 'rogówka', 'źrenica', 'narzędzie')
T_STAC = 300                 # koniec rozgrzewki lr; od tego kroku lr jest stały (5e-4)
GESTE_DO = 2188
KK = 6                       # sąsiedzi w latencie
rng = np.random.default_rng(0)
t0 = time.time()


def log(*a):
    print(*a, f'[{time.time() - t0:.0f} s]', flush=True)


# ---------------------------------------------------------------- dane
klasy = np.load(os.path.join(E35K, 'obszary_oka.npz'))['klasy'][:NS].reshape(-1)
Z, kroki = [], []
for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
    z = np.load(p)
    kr = z['kroki']
    Z.append(np.ascontiguousarray(z['pca32'][:, :NS].astype(np.float32)).reshape(len(kr), N, 32))
    kroki.append(kr)
    log(os.path.basename(p), int(kr[0]), int(kr[-1]))
    del z
Z = np.concatenate(Z)
kroki = np.concatenate(kroki).astype(int)
ind = {int(k): i for i, k in enumerate(kroki)}
assert all(k in ind for k in range(GESTE_DO + 1))
K_MAX = int(kroki[-1])
log('wczytane', Z.shape, 'ostatni krok', K_MAX)


def przes(a, b):
    """Przesunięcie między krokami a i b bez pola kolektywnego."""
    d = Z[ind[b]] - Z[ind[a]]
    return d - d.mean(0)


def udzial_pola(a, b):
    d = Z[ind[b]] - Z[ind[a]]
    f = d.mean(0)
    return float((f ** 2).sum() / (d ** 2).sum(1).mean())


def odleglosci(x):
    n2 = (x ** 2).sum(1)
    D = n2[:, None] + n2[None, :] - 2 * x @ x.T
    np.maximum(D, 0, out=D)
    np.fill_diagonal(D, np.inf)
    return np.sqrt(D)


idx = np.arange(600)
rr, cc = idx // W, idx % W
P_R = idx[cc < W - 1]
P_D = idx[rr < H - 1]
GRID_PAIRS = np.concatenate([np.stack([P_R, P_R + 1], 1), np.stack([P_D, P_D + W], 1)])
PARY_SIATKI = np.concatenate([GRID_PAIRS + im * 600 for im in range(NS)])


def knn_w_obrazie(k, kk=KK):
    """Najbliżsi sąsiedzi w latencie wewnątrz tego samego zdjęcia (N, kk) w kroku k, plus odległości i udział sąsiadów siatki."""
    x = Z[ind[k]]
    knn, dnn, dgrid, R, na_siatce = [], [], [], [], []
    for im in range(NS):
        xi = x[im * 600:(im + 1) * 600]
        D = odleglosci(xi)
        nn = np.argsort(D, axis=1)[:, :kk]
        knn.append(nn + im * 600)
        dnn.append(D.min(1))
        dgrid.append(np.linalg.norm(xi[GRID_PAIRS[:, 0]] - xi[GRID_PAIRS[:, 1]], axis=1))
        R.append(np.linalg.norm(xi - xi.mean(0), axis=1))
        na_siatce.append((np.maximum(abs(rr[nn] - rr[:, None]), abs(cc[nn] - cc[:, None])) <= 1).mean())
    return dict(knn=np.concatenate(knn), dnn=np.concatenate(dnn), dgrid=np.concatenate(dgrid), R=np.concatenate(R),
                udzial_knn_na_siatce=float(np.mean(na_siatce)))


def starty(tau, n=40, od=T_STAC, do=GESTE_DO):
    return np.unique(np.linspace(od, do - tau, n).astype(int))


# ---------------------------------------------------------------- skale odniesienia
skale = {}
for k in (T_STAC, 1000, GESTE_DO, 10000, K_MAX):
    s = knn_w_obrazie(k)
    x = Z[ind[k]]
    dnn_all = odleglosci(x).min(1)
    skale[k] = dict(d_nn_obraz=float(np.median(s['dnn'])), d_nn_obraz_obszary=[float(np.median(s['dnn'][klasy == c])) for c in range(4)],
                    d_nn_wszystkie=float(np.median(dnn_all)), d_siatka=float(np.median(s['dgrid'])),
                    R_obraz=float(np.median(s['R'])), R_wszystkie=float(np.median(np.linalg.norm(x - x.mean(0), axis=1))),
                    udzial_knn_na_siatce=s['udzial_knn_na_siatce'])
    log('skale', k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in skale[k].items()})
d_nn = skale[T_STAC]['d_nn_obraz']
q = 2 * np.pi / d_nn

# ---------------------------------------------------------------- statystyki stacjonarne (t ≥ 300, co 1 krok)
TAUS = np.unique(np.round(np.logspace(0, np.log10(1800), 34)).astype(int))
U = rng.standard_normal((64, 32))
U /= np.linalg.norm(U, axis=1, keepdims=True)
QM = (0.5, 1.0, 2.0)
msd = np.zeros((len(TAUS), 4))
msd_all = np.zeros(len(TAUS))
a2_1d = np.zeros(len(TAUS))
a2_pc1 = np.zeros(len(TAUS))
a2_d = np.zeros(len(TAUS))
fs = {qm: np.zeros(len(TAUS)) for qm in QM}
ogony = {}
for it, tau in enumerate(TAUS):
    S = starty(tau)
    m2 = np.zeros(32)
    m4 = np.zeros(32)
    C = np.zeros((32, 32))
    r2s, cosv, ds = [], {qm: 0.0 for qm in QM}, []
    for s in S:
        d = przes(s, s + tau)
        r2s.append((d ** 2).sum(1))
        m2 += (d ** 2).mean(0)
        m4 += (d ** 4).mean(0)
        C += d.T @ d / N
        pr = d @ U.T
        for qm in QM:
            cosv[qm] += np.cos(qm * q * pr).mean()
        if tau in (1, 10, 50, 300, 1000):
            ds.append(d / d.std(0))
    r2 = np.stack(r2s)
    msd[it] = [r2[:, klasy == c].mean() for c in range(4)]
    msd_all[it] = r2.mean()
    m2 /= len(S)
    m4 /= len(S)
    C /= len(S)
    a2_1d[it] = float((m4 / (3 * m2 ** 2) - 1).mean())
    a2_pc1[it] = float(m4[0] / (3 * m2[0] ** 2) - 1)
    a2_d[it] = float((r2 ** 2).mean() / r2.mean() ** 2 - (1 + 2 * np.trace(C @ C) / np.trace(C) ** 2))
    for qm in QM:
        fs[qm][it] = cosv[qm] / len(S)
    if ds:
        u = np.abs(np.concatenate(ds).reshape(-1))
        h, kr = np.histogram(u, bins=np.linspace(1.5, 5.5, 33), density=True)
        xc = 0.5 * (kr[1:] + kr[:-1])
        ok = h > 0
        y = np.log(h[ok])
        def r2fit(X):
            A = np.stack([np.ones(ok.sum()), X[ok]], 1)
            co, *_ = np.linalg.lstsq(A, y, rcond=None)
            return 1 - ((y - A @ co) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ogony[int(tau)] = dict(udzial_powyzej_3sigma=float((u > 3).mean()), udzial_powyzej_4sigma=float((u > 4).mean()),
                               gauss_3sigma=0.0027, gauss_4sigma=6.3e-5,
                               R2_ogon_wykladniczy=float(r2fit(xc)), R2_ogon_gaussowski=float(r2fit(xc ** 2)))
    if it % 6 == 0:
        log('tau', int(tau), 'msd', round(float(msd_all[it]), 3), 'a2', round(float(a2_1d[it]), 3), 'Fs', round(float(fs[1.0][it]), 3))

lt = np.log(TAUS.astype(float))
lm = np.log(msd_all)
alfa = np.full(len(TAUS), np.nan)
alfa[1:-1] = (lm[2:] - lm[:-2]) / (lt[2:] - lt[:-2])
plateau = (TAUS >= 35) & (TAUS <= 80)
l_klatka = float(np.sqrt(msd_all[plateau].mean()))
l_klatka_ob = [float(np.sqrt(msd[plateau, c].mean())) for c in range(4)]
f_q = float(fs[1.0][plateau].mean())
# β i τ_α z części α-relaksacji (τ ≥ 100), rozciągnięty eksponens Fs = f_q exp(-(τ/τ_α)^β)
sel = (TAUS >= 100) & (fs[1.0] > 0.02) & (fs[1.0] < 0.98 * f_q)
beta = tau_alfa = float('nan')
if sel.sum() >= 3:
    yy = np.log(-np.log(fs[1.0][sel] / f_q))
    A = np.stack([np.ones(sel.sum()), lt[sel]], 1)
    (b0, b1), *_ = np.linalg.lstsq(A, yy, rcond=None)
    beta = float(b1)
    tau_alfa = float(np.exp(-b0 / b1))
log('klatka', round(l_klatka, 3), 'Lindemann', round(l_klatka / d_nn, 3), 'f_q', round(f_q, 3), 'beta', beta, 'tau_alfa', tau_alfa)

# ---------------------------------------------------------------- Fs i MSD w długich czasach (co 10 kroków, t_w ≥ 2200)
TAU_L = (10, 30, 100, 300, 1000, 3000, 10000, 20000)
fs_dlugie, msd_dlugie = {}, {}
for tau in TAU_L:
    v, m = [], []
    for s in range(2200, 2700, 100):
        if s + tau in ind:
            d = przes(s, s + tau)
            v.append(np.cos(q * (d @ U.T)).mean())
            m.append((d ** 2).sum(1).mean())
    if v:
        fs_dlugie[tau] = float(np.mean(v))
        msd_dlugie[tau] = float(np.mean(m))
d = przes(T_STAC, K_MAX)
calk = float(np.sqrt((d ** 2).sum(1).mean()))
log('Fs długie', {k: round(v, 3) for k, v in fs_dlugie.items()})

# ---------------------------------------------------------------- VACF (z wcześniejszego liczenia, ta sama definicja)
tw = np.load(os.path.join(DANE, 'trajektorie_wlasnosci.npz'))
vacf = dict(lags=[int(x) for x in tw['lags']], vacf=[float(x) for x in tw['vacf'].mean(1)])

# ---------------------------------------------------------------- kooperatywność sąsiadów i skupianie szybkich
koop = {}
for tau in (10, 100, 1000):
    res = {k: [] for k in ('cos_siatka', 'cos_knn', 'cos_los_obraz', 'cos_los_inne', 'cos_siatka_szybcy', 'skup_siatka', 'skup_knn',
                           'zgodni_szybcy_siatka', 'udzial_pola')}
    for s in starty(tau, 16):
        d = przes(s, s + tau)
        n = np.linalg.norm(d, axis=1)
        knn = knn_w_obrazie(s)['knn']

        def cosp(i, j):
            return (d[i] * d[j]).sum(1) / np.maximum(n[i] * n[j], 1e-20)

        i, j = PARY_SIATKI[:, 0], PARY_SIATKI[:, 1]
        res['cos_siatka'].append(cosp(i, j).mean())
        res['cos_knn'].append(cosp(np.repeat(np.arange(N), KK), knn.reshape(-1)).mean())
        a = rng.integers(0, N, 40000)
        b = (a // 600) * 600 + rng.integers(0, 600, 40000)
        m = a != b
        res['cos_los_obraz'].append(cosp(a[m], b[m]).mean())
        a = rng.integers(0, N, 40000)
        b = rng.integers(0, N, 40000)
        m = a // 600 != b // 600
        res['cos_los_inne'].append(cosp(a[m], b[m]).mean())
        szybcy = n >= np.quantile(n, 0.95)
        pf1 = szybcy[i] | szybcy[j]
        res['cos_siatka_szybcy'].append(cosp(i[pf1], j[pf1]).mean())
        res['skup_siatka'].append(np.concatenate([szybcy[j][szybcy[i]], szybcy[i][szybcy[j]]]).mean() / 0.05)
        res['skup_knn'].append(szybcy[knn[szybcy]].mean() / 0.05)
        pf = szybcy[i] & szybcy[j]
        if pf.any():
            res['zgodni_szybcy_siatka'].append((cosp(i[pf], j[pf]) > 0).mean())
        res['udzial_pola'].append(udzial_pola(s, s + tau))
    koop[int(tau)] = {k: float(np.mean(v)) for k, v in res.items() if len(v)}
    log('kooperatywność', tau, {k: round(v, 3) for k, v in koop[int(tau)].items()})

# ---------------------------------------------------------------- trwałość sąsiedztwa (czy klatka trzyma tych samych sąsiadów)
knn0 = knn_w_obrazie(T_STAC)['knn']
knn0_12 = knn_w_obrazie(T_STAC, 12)['knn']
trwalosc = {}
for k in (301, 310, 400, 1000, GESTE_DO, 5000, 10000, K_MAX):
    k6 = knn_w_obrazie(k)['knn']
    k12 = knn_w_obrazie(k, 12)['knn']
    z6 = np.mean([len(set(knn0[i]) & set(k6[i])) / KK for i in range(N)])
    z12 = np.mean([len(set(knn0_12[i]) & set(k12[i])) / 12 for i in range(N)])
    trwalosc[int(k)] = dict(zachowani_k6=float(z6), zachowani_k12=float(z12), losowo_k6=KK / 599, losowo_k12=12 / 599)
log('trwałość sąsiedztwa', {k: round(v['zachowani_k6'], 3) for k, v in trwalosc.items()})

# ---------------------------------------------------------------- starzenie przy stałym lr
TW_D, TAU_D = (300, 500, 800, 1200, 1600, 2000), (1, 3, 10, 30, 100, 180)
msd_tw = {}
for tw_ in TW_D:
    msd_tw[tw_] = {}
    for tau in TAU_D:
        if tw_ + tau + 20 > GESTE_DO:
            continue
        msd_tw[tw_][tau] = float(np.mean([(przes(s, s + tau) ** 2).sum(1).mean() for s in range(tw_, tw_ + 20, 4)]))
TW_S, TAU_S = (300, 1000, 2000, 3000, 5000, 10000, 15000, 20000), (10, 30, 100, 300, 1000, 2000)
msd_tw_s = {}
for tw_ in TW_S:
    msd_tw_s[tw_] = {}
    for tau in TAU_S:
        v = [(przes(s, s + tau) ** 2).sum(1).mean() for s in range(tw_, tw_ + 50, 10) if s in ind and s + tau in ind]
        if v:
            msd_tw_s[tw_][tau] = float(np.mean(v))


def wykladnik(xs, ys):
    xs, ys = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    A = np.stack([np.ones(len(xs)), xs], 1)
    (a0, a1), *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(a1)


mu = {}
for tau in (10, 100, 1000):
    pts = [(tw_, msd_tw_s[tw_][tau]) for tw_ in TW_S if tau in msd_tw_s[tw_]]
    if len(pts) >= 3:
        mu[tau] = dict(wykladnik_mu=-wykladnik([p[0] for p in pts], [p[1] for p in pts]), punkty=pts)
# energia kroku (temperatura) w czasie: co 1 krok w części gęstej, co 10 dalej
E1 = np.array([(przes(t, t + 1) ** 2).sum(1).mean() for t in range(0, GESTE_DO)])
t10 = [k for k in range(300, K_MAX - 10 + 1, 10) if k in ind and k + 10 in ind]
E10 = np.array([(przes(t, t + 10) ** 2).sum(1).mean() for t in t10])
t10 = np.array(t10)
tt = np.arange(GESTE_DO)
szczyt = int(tt[np.argmax(E1)])
styg = dict(szczyt_krok=szczyt, E1_szczyt=float(E1.max()), E1_300=float(E1[295:305].mean()), E1_1000=float(E1[995:1005].mean()),
            E1_2000=float(E1[1995:2005].mean()), E1_2188=float(E1[2170:2188].mean()),
            wykladnik_E1_300_2188=wykladnik(tt[300:], E1[300:]),
            E10_300=float(E10[(t10 >= 300) & (t10 < 400)].mean()), E10_2200=float(E10[(t10 >= 2200) & (t10 < 2400)].mean()),
            E10_10000=float(E10[(t10 >= 10000) & (t10 < 10500)].mean()), E10_22000=float(E10[t10 >= 22000].mean()),
            wykladnik_E10_2200_koniec=wykladnik(t10[t10 >= 2200], E10[t10 >= 2200]),
            wykladnik_E10_300_koniec=wykladnik(t10, E10), lr_po_rozgrzewce='stały 5e-4 (bez harmonogramu), więc stygnięcie jest własne')
log('stygnięcie', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in styg.items()})

# ---------------------------------------------------------------- zapis
wyn = dict(
    opis='Odciski palca dynamiki szklistej z gęstego zapisu, 10 sond × 600 tokenów, PCA32, po odjęciu pola kolektywnego; okno stacjonarne t ≥ 300.',
    podprobka=dict(sond=NS, tokenow=N, udzial_wariancji_pca32=0.986),
    skale=skale, q=float(q), d_nn_300=d_nn,
    msd=dict(taus=[int(x) for x in TAUS], msd=[float(x) for x in msd_all], msd_obszary=msd.tolist(), alfa_lokalna=[None if np.isnan(x) else float(x) for x in alfa],
             reżimy=dict(balistyczny_tau_1_3=float(np.nanmean(alfa[(TAUS >= 2) & (TAUS <= 3)])), plateau_tau_35_80=float(np.nanmean(alfa[plateau])),
                         dyfuzyjny_tau_800_1500=float(np.nanmean(alfa[(TAUS >= 800) & (TAUS <= 1500)])))),
    klatka=dict(l_klatka=l_klatka, l_klatka_obszary=l_klatka_ob, lindemann_l_do_dnn=l_klatka / d_nn,
                lindemann_obszary=[l_klatka_ob[c] / skale[T_STAC]['d_nn_obraz_obszary'][c] for c in range(4)],
                l_do_d_siatka=l_klatka / skale[T_STAC]['d_siatka'], l_do_R_obraz=l_klatka / skale[T_STAC]['R_obraz'],
                przesuniecie_1000_krokow_do_dnn=float(np.sqrt(msd_all[np.argmin(abs(TAUS - 1000))])) / d_nn,
                przesuniecie_300_do_konca=calk, przesuniecie_300_do_konca_do_dnn=calk / d_nn, przesuniecie_300_do_konca_do_R=calk / skale[T_STAC]['R_obraz'],
                kryterium='Lindemann: rms drgań w klatce / odległość do sąsiada ≈ 0,1–0,15 na granicy topnienia; szkło i kryształ poniżej'),
    niegaussowski=dict(taus=[int(x) for x in TAUS], a2_1d=[float(x) for x in a2_1d], a2_pc1=[float(x) for x in a2_pc1], a2_pelny=[float(x) for x in a2_d],
                       szczyt_a2_1d=dict(wartosc=float(a2_1d.max()), tau=int(TAUS[int(np.argmax(a2_1d))])),
                       ogony_van_hove=ogony),
    fs=dict(taus=[int(x) for x in TAUS], q_razy_dnn='2π', **{f'Fs_q{qm}': [float(x) for x in fs[qm]] for qm in QM}, plateau_f_q=f_q, beta=beta, tau_alfa=tau_alfa,
            dlugie_t_w_2200=dict(taus=list(fs_dlugie.keys()), Fs=list(fs_dlugie.values()), msd=list(msd_dlugie.values()))),
    vacf=vacf, kooperatywnosc=koop, trwalosc_sasiedztwa=trwalosc,
    starzenie=dict(lr='stały po kroku 300', msd_tw_geste=msd_tw, msd_tw_rzadkie=msd_tw_s, wykladnik_mu=mu, stygniecie=styg,
                   E1=dict(krok=[int(x) for x in tt[::4]], E1=[float(x) for x in E1[::4]]), E10=dict(krok=[int(x) for x in t10], E10=[float(x) for x in E10])),
    czas_s=round(time.time() - t0))
json.dump(wyn, open(os.path.join(OUT, 'ODCISKI_SZKLA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- rysunek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
KOL = ['#8a8f99', '#4cd964', '#4a8cff', '#ff4433']
fig, ax = plt.subplots(2, 2, figsize=(16, 11), dpi=100)
a = ax[0, 0]
for c in range(4):
    a.loglog(TAUS, msd[:, c], color=KOL[c], lw=1.6, label=OBSZARY[c])
a.loglog(TAUS, msd_all, color='#ffd166', lw=2.4, label='wszystkie')
a.axhline(d_nn ** 2, color='w', ls='--', lw=0.8)
a.text(TAUS[0], d_nn ** 2 * 1.1, 'd_nn² (odległość do sąsiada)', fontsize=9)
a.axhspan(msd_all[plateau].min(), msd_all[plateau].max(), color='#ffd166', alpha=0.12)
for tq, lab in ((2, f'α={np.nanmean(alfa[(TAUS >= 2) & (TAUS <= 3)]):.2f}'), (50, f'α={np.nanmean(alfa[plateau]):.2f}'), (1000, f'α={np.nanmean(alfa[(TAUS >= 800) & (TAUS <= 1500)]):.2f}')):
    i = int(np.argmin(abs(TAUS - tq)))
    a.annotate(lab, (TAUS[i], msd_all[i]), textcoords='offset points', xytext=(6, -14), fontsize=9, color='#ffd166')
a.set_xlabel('τ (kroki uczenia)')
a.set_ylabel('MSD (PCA32, bez pola kolektywnego)')
a.set_title(f'MSD: balistyczny → klatka → dyfuzja; ℓ_klatka/d_nn = {l_klatka / d_nn:.2f}')
a.grid(True, which='both', alpha=0.4)
a.legend(fontsize=9)
a = ax[0, 1]
a.semilogx(TAUS, a2_1d, color='#ffd166', lw=2.2, label='α₂ (1D, średnia po składowych)')
a.semilogx(TAUS, a2_pc1, color='#4a8cff', lw=1.4, label='α₂ (PC1)')
a.semilogx(TAUS, a2_d, color='#ff4433', lw=1.4, label='α₂ (32D, baza anizotropowa)')
a.axhline(0, color='w', lw=0.6)
a.set_xlabel('τ (kroki)')
a.set_ylabel('parametr niegaussowski α₂')
a.set_title(f'α₂: szczyt {a2_1d.max():.2f} przy τ = {TAUS[int(np.argmax(a2_1d))]}')
a.grid(True, which='both', alpha=0.4)
a.legend(fontsize=9)
a = ax[1, 0]
for qm, kol in zip(QM, ('#4cd964', '#ffd166', '#ff4433')):
    a.semilogx(TAUS, fs[qm], color=kol, lw=2 if qm == 1 else 1.3, label=f'q = {qm:g}·2π/d_nn (t_w 300–2188)')
if fs_dlugie:
    a.semilogx(list(fs_dlugie.keys()), list(fs_dlugie.values()), 'o-', color='#ffd166', mfc='none', lw=1, label='q = 2π/d_nn, t_w ≈ 2200–2700 (co 10)')
a.axhline(np.exp(-1), color='w', ls=':', lw=0.8)
a.text(TAUS[0], np.exp(-1) + 0.02, '1/e', fontsize=9)
a.set_ylim(-0.05, 1.05)
a.set_xlabel('τ (kroki)')
a.set_ylabel('Fs(q, τ)')
a.set_title(f'Funkcja rozpraszania: plateau f_q = {f_q:.2f}, β = {beta:.2f}, τ_α ≈ {tau_alfa:.0f}')
a.grid(True, which='both', alpha=0.4)
a.legend(fontsize=8)
a = ax[1, 1]
for tau, kol in zip((10, 100, 1000), ('#4cd964', '#ffd166', '#ff4433')):
    pts = [(tw_, msd_tw_s[tw_][tau]) for tw_ in TW_S if tau in msd_tw_s[tw_]]
    if pts:
        a.loglog([p[0] for p in pts], [p[1] for p in pts], 'o-', color=kol, lw=1.6, label=f'MSD(τ={tau}; t_w)' + (f', μ = {mu[tau]["wykladnik_mu"]:.2f}' if tau in mu else ''))
a.loglog(t10, E10 * 10, color='#8a8f99', lw=1, alpha=0.9, label='energia 10 kroków ×10 (temperatura)')
a.set_xlabel('t_w (krok uczenia, lr stały od 300)')
a.set_ylabel('MSD / energia')
a.set_title('Starzenie przy stałym lr: ruch słabnie z wiekiem jak w szkle')
a.grid(True, which='both', alpha=0.4)
a.legend(fontsize=9)
fig.suptitle('Odciski palca dynamiki szklistej w ruchu tokenów ViT (bieg 35 000, 6000 cząsteczek, PCA32)', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT, 'odciski_szkla.png'))
log('GOTOWE')
