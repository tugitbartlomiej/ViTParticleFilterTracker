"""Wspólne odciski palca dynamiki szklistej dla dowolnej trajektorii cząsteczek (ViT w latencie, układ szkła,
symulacja browna). Jedna implementacja, żeby liczby były porównywalne między danymi a modelem.

Wejście: P (T, N, d) float32 położenia, kroki (T,) rosnące int, okno (od, do) kroków, w którym statystyki są
„stacjonarne”; skale geometrii (d_nn, d_shell) policzone przez wywołującego w oryginalnym układzie.
Wszystkie przesunięcia liczone między rekordami o krokach a i b; opcjonalnie minus średnie przesunięcie populacji.

Kryteria zgodności ze szkłem (karta) z literatury: Weeks i Weitz 2002 (c ≈ 0,26–0,5 w plateau, 5–10 % cząstek
w przegrupowaniu, szersze niż Gauss rozkłady), Hunter i Weeks 2012 (koloid: krótkie czasy dyfuzyjne; starzenie
wydłuża plateau nie zmieniając drgań krótkoczasowych), tabela I PRL 2002 (r_msd/2a = 0,07–0,10 w szkle,
0,27 w cieczy przechłodzonej), Chaudhuri, Berthier, Kob 2007 (wykładnicze ogony), Rabani 1998 / Sarkar 2025
(trwałość sąsiedztwa), kryterium Lindemanna (≈ 0,1–0,15 odstępu sąsiadów).
"""
import numpy as np

TAUS_DOMYSLNE = np.unique(np.round(np.logspace(0, np.log10(1800), 30)).astype(int))


def _lstsq(x, y):
    """Prosta y = a0 + a1 x metodą najmniejszych kwadratów; punkty nieskończone pomijane; za mało punktów → NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.ptp(x[m]) == 0:
        return np.array([np.nan, np.nan])
    A = np.stack([np.ones(int(m.sum())), x[m]], 1)
    return np.linalg.lstsq(A, y[m], rcond=None)[0]


def wykladnik(xs, ys):
    """Nachylenie log-log (punkty z x ≤ 0 albo y ≤ 0 pomijane)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(_lstsq(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)))[1])


def knn_w_blokach(x, blok=600, kk=6):
    """Najbliżsi sąsiedzi wewnątrz bloków (zdjęć) po blok cząstek: (N, kk) indeksy, (N,) d_nn, (N,) d_shell (średnia z kk)."""
    N = len(x)
    out, dnn, dsh = [], [], []
    for a in range(0, N, blok):
        xi = x[a:a + blok]
        n2 = (xi ** 2).sum(1)
        D = n2[:, None] + n2[None, :] - 2 * xi @ xi.T
        np.maximum(D, 0, out=D)
        np.fill_diagonal(D, np.inf)
        D = np.sqrt(D)
        nn = np.argpartition(D, kk, axis=1)[:, :kk]
        out.append(nn + a)
        dnn.append(D.min(1))
        dsh.append(np.take_along_axis(D, nn, 1).mean(1))
    return np.concatenate(out), np.concatenate(dnn), np.concatenate(dsh)


def knn_3d_pudlo(x, L, kk=6, porcja=2000):
    """Najbliżsi sąsiedzi w pudle periodycznym 3D (symulacja): (N, kk), d_nn, d_shell."""
    N = len(x)
    out = np.empty((N, kk), int)
    dnn = np.empty(N)
    dsh = np.empty(N)
    for a in range(0, N, porcja):
        dx = x[a:a + porcja, None, :] - x[None, :, :]
        dx -= L * np.round(dx / L)
        D = np.sqrt((dx ** 2).sum(-1))
        for i in range(len(D)):
            D[i, a + i] = np.inf
        nn = np.argpartition(D, kk, axis=1)[:, :kk]
        out[a:a + porcja] = nn
        dnn[a:a + porcja] = D.min(1)
        dsh[a:a + porcja] = np.take_along_axis(D, nn, 1).mean(1)
    return out, dnn, dsh


def przesuniecie(P, ind, a, b, odejmij_pole=True):
    d = P[ind[b]] - P[ind[a]]
    return d - d.mean(0) if odejmij_pole else d


def odciski(P, kroki, okno, d_shell, d_nn=None, taus=None, n_start=40, odejmij_pole=True, pary=None,
            tw_lista=(300, 1000, 2000, 3000, 5000, 10000, 15000, 20000), krotkie_tw=(300, 600, 1000, 1500, 2000),
            rng=None, nazwa=''):
    """Pełny zestaw odcisków dla trajektorii P. Zwraca słownik (wszystko float/list, gotowe do JSON)."""
    rng = rng or np.random.default_rng(0)
    T, N, d = P.shape
    kroki = np.asarray(kroki, int)
    ind = {int(k): i for i, k in enumerate(kroki)}
    od, do = okno
    taus = TAUS_DOMYSLNE if taus is None else np.asarray(taus, int)
    taus = taus[taus < do - od]
    U = rng.standard_normal((64, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    q = 2 * np.pi / d_shell * np.sqrt(d / 3.0)         # ten sam rzut na kierunek co w 3D przy tym samym MSD/d_shell²

    def starty(tau, n):
        s = np.unique(np.linspace(od, do - tau, n).astype(int))
        return [x for x in s if x in ind and x + tau in ind]

    msd = np.zeros(len(taus))
    a2 = np.zeros(len(taus))
    fs = np.zeros(len(taus))
    uciek = np.zeros(len(taus))
    cos_sas = np.full(len(taus), np.nan)
    ogon = {}
    for it, tau in enumerate(taus):
        S = starty(tau, n_start if tau < 300 else max(12, n_start // 2))
        if not S:
            msd[it] = np.nan
            continue
        m2 = np.zeros(d)
        m4 = np.zeros(d)
        r2s, f, uc, cs, ds = [], 0.0, 0.0, [], []
        for s in S:
            dd = przesuniecie(P, ind, s, s + tau, odejmij_pole)
            r2 = (dd ** 2).sum(1)
            r2s.append(r2.mean())
            m2 += (dd ** 2).mean(0)
            m4 += (dd ** 4).mean(0)
            f += np.cos(q * (dd @ U.T)).mean()
            uc += (np.sqrt(r2) > 0.5 * d_shell).mean()
            if pary is not None:
                n = np.linalg.norm(dd, axis=1)
                i, j = pary[:, 0], pary[:, 1]
                cs.append(((dd[i] * dd[j]).sum(1) / np.maximum(n[i] * n[j], 1e-20)).mean())
            if tau in (1, 10, 100):
                ds.append(dd / np.maximum(dd.std(0), 1e-20))
        msd[it] = np.mean(r2s)
        m2 /= len(S)
        m4 /= len(S)
        a2[it] = float((m4 / (3 * m2 ** 2) - 1).mean())
        fs[it] = f / len(S)
        uciek[it] = uc / len(S)
        if cs:
            cos_sas[it] = float(np.mean(cs))
        if ds:
            u = np.abs(np.concatenate(ds).reshape(-1))
            ogon[int(tau)] = dict(powyzej_3sigma=float((u > 3).mean()), powyzej_4sigma=float((u > 4).mean()),
                                  krotnosc_gaussa_3sigma=float((u > 3).mean() / 0.0027))
    ok = np.isfinite(msd) & (msd > 0)
    taus, msd, a2, fs, uciek, cos_sas = taus[ok], msd[ok], a2[ok], fs[ok], uciek[ok], cos_sas[ok]
    lt, lm = np.log(taus.astype(float)), np.log(msd)
    alfa = np.full(len(taus), np.nan)
    alfa[1:-1] = (lm[2:] - lm[:-2]) / (lt[2:] - lt[:-2])
    plateau = (taus >= 35) & (taus <= 80)
    if not plateau.any():
        plateau = (taus >= taus.max() / 40) & (taus <= taus.max() / 15)
    l_kl = float(np.sqrt(msd[plateau].mean()))

    def M(x):
        return np.exp(np.interp(np.log(x), lt, lm))

    c_tau = {int(x): float(1 - M(2 * x) / (2 * M(x))) for x in (1, 2, 5, 10, 20, 40, 60, 100, 200, 500) if 2 * x <= taus.max()}
    c_plateau = float(np.mean([c_tau[x] for x in (20, 40, 60, 100) if x in c_tau])) if any(x in c_tau for x in (20, 40, 60, 100)) else float('nan')
    f_q = float(fs[plateau].mean())
    sel = (taus >= 100) & (fs > 0.02) & (fs < 0.98 * f_q)
    beta = tau_alfa = float('nan')
    if sel.sum() >= 3:
        b0, b1 = _lstsq(lt[sel], np.log(-np.log(fs[sel] / f_q)))
        beta, tau_alfa = float(b1), float(np.exp(-b0 / b1))
    i_e = np.where(fs < np.exp(-1))[0]
    tau_1e = float(taus[i_e[0]]) if len(i_e) else float('inf')
    m_a2 = taus >= 3
    i_pk = int(np.argmax(np.where(m_a2, a2, -np.inf)))
    i_dt = int(np.argmin(abs(taus - 10)))

    # starzenie: MSD(τ; t_w) dla t_w z listy (wymaga rekordów t_w..t_w+40 i t_w+τ)
    starz = {}
    for tw in tw_lista:
        starz[int(tw)] = {}
        for tau in (10, 100, 1000):
            v = [(przesuniecie(P, ind, s, s + tau, odejmij_pole) ** 2).sum(1).mean()
                 for s in range(tw, tw + 50, 10) if s in ind and s + tau in ind]
            if v:
                starz[int(tw)][tau] = float(np.mean(v))
    mu = {}
    for tau in (10, 100, 1000):
        pts = [(tw, starz[tw][tau]) for tw in starz if tau in starz[tw] and tw > 0]
        if len(pts) >= 3:
            mu[tau] = -wykladnik([p[0] for p in pts], [p[1] for p in pts])
    krotkie = {}
    for tw in krotkie_tw:
        v = [(przesuniecie(P, ind, s, s + 1, odejmij_pole) ** 2).sum(1).mean() for s in range(tw, tw + 20) if s in ind and s + 1 in ind]
        if v:
            krotkie[int(tw)] = float(np.mean(v))
    kk = sorted(krotkie)
    stosunek_krotkich = float(krotkie[kk[-1]] / krotkie[kk[0]]) if len(kk) >= 2 else float('nan')
    # energia kroku w oknie i jej wykładnik
    tt = [t for t in range(od, do) if t in ind and t + 1 in ind]
    E1 = np.array([(przesuniecie(P, ind, t, t + 1, odejmij_pole) ** 2).sum(1).mean() for t in tt]) if len(tt) > 10 else np.array([])
    m_E = (E1 > 0) & (np.asarray(tt) > 0) if len(E1) else np.zeros(0, bool)
    wyk_E = wykladnik(np.asarray(tt)[m_E], E1[m_E]) if m_E.sum() > 10 else float('nan')
    # VACF przyrostów co 1 krok (po odjęciu pola) na podpróbce startów
    vac = {}
    if 1 in [int(x) for x in taus]:
        S = starty(1, 200)[:150]
        D0 = np.stack([przesuniecie(P, ind, s, s + 1, odejmij_pole) for s in S if s + 2 in ind and s + 6 in ind and s + 11 in ind])
        den = (D0 * D0).sum(2).mean()
        for lag in (1, 2, 5, 10):
            Dl = np.stack([przesuniecie(P, ind, s + lag, s + lag + 1, odejmij_pole) for s in S if s + 2 in ind and s + 6 in ind and s + 11 in ind])
            vac[lag] = float((D0 * Dl).sum(2).mean() / den)
    return dict(
        nazwa=nazwa, wymiar=int(d), N=int(N), okno=[int(od), int(do)], d_shell=float(d_shell), d_nn=None if d_nn is None else float(d_nn),
        taus=[int(x) for x in taus], msd=[float(x) for x in msd], alfa=[None if np.isnan(x) else float(x) for x in alfa],
        alfa_krotkie=float(np.nanmean(alfa[(taus >= 2) & (taus <= 3)])) if ((taus >= 2) & (taus <= 3)).any() else float('nan'),
        alfa_plateau=float(np.nanmean(alfa[plateau])),
        alfa_dlugie=float(np.nanmean(alfa[taus >= 800])) if (taus >= 800).any() else float('nan'),
        l_klatka=l_kl, lindemann_shell=l_kl / d_shell, lindemann_nn=None if d_nn is None else l_kl / d_nn,
        c_tau=c_tau, c_plateau=c_plateau,
        a2=[float(x) for x in a2], a2_plateau=float(a2[plateau].mean()), a2_szczyt=float(a2[i_pk]), a2_szczyt_tau=int(taus[i_pk]),
        ogony=ogon, fs=[float(x) for x in fs], f_q=f_q, beta=beta, tau_alfa=tau_alfa, tau_1e=tau_1e,
        uciekajacy=[float(x) for x in uciek], uciekajacy_tau10=float(uciek[i_dt]), uciekajacy_szczyt_a2=float(uciek[i_pk]),
        cos_sasiadow=[None if np.isnan(x) else float(x) for x in cos_sas],
        cos_sasiadow_tau10=None if np.isnan(cos_sas[i_dt]) else float(cos_sas[i_dt]),
        starzenie=starz, mu=mu, krotkie_tw=krotkie, stosunek_krotkich=stosunek_krotkich,
        E1=dict(krok=[int(x) for x in tt[::5]], E=[float(x) for x in E1[::5]]), wykladnik_E=wyk_E, vacf=vac)


def trwalosc_sasiedztwa(P_geom, kroki, od, kroki_test, blok=600, kk=6, L=None):
    """Udział sąsiadów (kk najbliższych w kroku `od`) zachowanych w późniejszych krokach; losowo ≈ kk/(blok-1)."""
    ind = {int(k): i for i, k in enumerate(kroki)}
    f = (lambda x: knn_w_blokach(x, blok, kk)[0]) if L is None else (lambda x: knn_3d_pudlo(x, L, kk)[0])
    k0 = f(P_geom[ind[od]])
    out = {}
    for k in kroki_test:
        if k not in ind:
            continue
        kt = f(P_geom[ind[k]])
        out[int(k)] = float(np.mean([len(set(k0[i]) & set(kt[i])) / kk for i in range(len(k0))]))
    return out


KRYTERIA = [
    ('plateau MSD (wykładnik α przy τ 35–80 ≤ 0,35)', 'alfa_plateau', lambda v: v <= 0.35, 'szkło: bliski 0'),
    ('antykorelacja c w plateau 0,25–0,55', 'c_plateau', lambda v: 0.25 <= v <= 0.55, 'Weeks i Weitz 2002: 0,26 … ≈0,5'),
    ('rozmiar klatki ≤ 0,32 odstępu sąsiadów (ciecz przechłodzona)', 'lindemann_shell', lambda v: v <= 0.32, 'PRL 2002 tabela I: 0,07–0,27'),
    ('rozmiar klatki ≤ 0,15 odstępu sąsiadów (gęste szkło)', 'lindemann_shell', lambda v: v <= 0.15, 'Lindemann; φ ≥ 0,52'),
    ('ogony szersze niż Gauss (≥ 2× powyżej 3σ przy τ=10)', 'ogon10', lambda v: v >= 2.0, 'Chaudhuri 2007; Weeks 2002'),
    ('α₂ w plateau 0,5–3', 'a2_plateau', lambda v: 0.5 <= v <= 3.0, 'rozkłady warunkowe 1,0–1,8'),
    ('szczyt α₂ na końcu plateau (τ ≥ 10)', 'a2_szczyt_tau', lambda v: v >= 10, 'α₂ maksimum przy Δt*'),
    ('5–15 % cząstek ucieka z klatki przy τ ≈ 10', 'uciekajacy_tau10', lambda v: 0.05 <= v <= 0.15, '5–10 % przy Δt*'),
    ('krótkie czasy dyfuzyjne jak w koloidzie (α 0,8–1,2 przy τ 1–3)', 'alfa_krotkie', lambda v: 0.8 <= v <= 1.2, 'Hunter i Weeks 2012: brak reżimu balistycznego'),
    ('trwałość sąsiedztwa ≥ 0,5 po 1000 krokach', 'trwalosc_1000', lambda v: v >= 0.5, 'Rabani 1998; Sarkar 2025'),
    ('starzenie: MSD maleje z t_w (μ > 0,05)', 'mu10', lambda v: v > 0.05, 'Hunter i Weeks 2012 rys. 15'),
    ('starzenie nie zmienia drgań krótkoczasowych (MSD(τ=1) w 0,5–2×)', 'stosunek_krotkich', lambda v: 0.5 <= v <= 2.0, 'koloid: plateau się wydłuża, drgania nie'),
    ('sąsiedzi nie płyną razem (cos ≤ 0,3 przy τ=10)', 'cos_sasiadow_tau10', lambda v: v is not None and v <= 0.3, 'w szkle korelacja dodatnia, ale słaba'),
    ('dwustopniowa relaksacja Fs (plateau 0,3–0,97, β 0,4–1,0)', 'fs_dwa', lambda v: v, 'rozciągnięty eksponens'),
]


def karta(w, trwalosc_1000=None):
    """Karta zgodności: lista (kryterium, wartość, czy spełnione, źródło) i liczba spełnionych."""
    wart = dict(w)
    wart['ogon10'] = w['ogony'].get(10, w['ogony'].get(1, {})).get('krotnosc_gaussa_3sigma', float('nan'))
    wart['mu10'] = w['mu'].get(10, float('nan'))
    wart['trwalosc_1000'] = float('nan') if trwalosc_1000 is None else trwalosc_1000
    wart['fs_dwa'] = bool(0.3 <= w['f_q'] <= 0.97 and 0.4 <= (w['beta'] if np.isfinite(w['beta']) else 0) <= 1.0)
    out = []
    for opis, klucz, test, zrodlo in KRYTERIA:
        v = wart.get(klucz)
        try:
            ok = bool(test(v)) if v is not None and not (isinstance(v, float) and np.isnan(v)) else False
        except Exception:
            ok = False
        out.append(dict(kryterium=opis, wartosc=(None if v is None or (isinstance(v, float) and np.isnan(v)) else (float(v) if not isinstance(v, bool) else v)), ok=ok, zrodlo=zrodlo))
    return out, int(sum(o['ok'] for o in out))
