"""Ślepa klasyfikacja klasy dynamiki tokenów (Eden, CPU): te same miary liczone na układach wzorcowych o ZNANEJ klasie (symulacja
Langevina 2D, 600 cząstek z etykietami siatki 20 × 30) i na tokenach; klasę tokenów wyznacza najbliższy sąsiad wśród wzorców, nie hipoteza.
Klasy wzorców (każda w kilku temperaturach, skalach czasu próbkowania i ziarnach):
  krysztal            sprężyny między sąsiadami siatki (pozioma/pionowa k = 1, przekątna k = 0,5), szum biały
  krysztal_pelzajacy  jak wyżej + słaby dołek do kotwic, które powoli dryfują gładkim polem (pełzanie narzucone z zewnątrz)
  krysztal_napedzany  kryształ + przestrzennie gładki (wspólny) szum
  pulapki_naped       BEZ oddziaływań: każda cząstka w swoim dołku, kotwice dryfują gładko, szum gładki (wspólny napęd) — główna alternatywa
  pulapki             BEZ oddziaływań, dołki stałe, szum biały
  ciecz               dwuskładnikowe miękkie dyski, φ = 0,70, okresowe pudło
  szklo               dwuskładnikowe miękkie dyski, φ = 0,88, niska T, start losowy i schłodzenie
Miary (te same funkcje dla wzorców i tokenów; trajektoria (T, 600, D) po odjęciu środka masy w każdej klatce):
  siatka      odsetek 6 najbliższych sąsiadów (położenia średnie w czasie) będących sąsiadami na siatce etykiet
  wyk_gL      wykładnik γ_L (Zahn–Maret, pary sąsiadów siatki) w górnej połowie opóźnień
  wyk_msd     wykładnik MSD w górnej połowie opóźnień
  dlugie      udział mocy przyrostów (opóźnienie T/10) w falach |k| ≤ 2,5 na siatce etykiet
  precyzja    udział |macierzy precyzji| fluktuacji (2000 próbek, ściągnięcie 0,1) na parach sąsiadów siatki — lokalność sztywności
  wiazania    ⟨cos⟩ kierunku wiązań sąsiadów po T/4
  alfa2       parametr niegaussowski przesunięć po T/10
Wynik: SLEPA.json (miary wzorców, trafność kNN z pominięciem jednego, klasy tokenów z głosami i odległością), slepa.png.
Uruchomienie: python slepa_klasyfikacja.py <ruch60 eksport> <katalog pokretla/wyniki> <out> [watki]"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

EKS, POK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
WAT = int(sys.argv[4]) if len(sys.argv) > 4 else 16
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
H, W, TOK = 20, 30, 600
siatka = np.stack([np.arange(TOK) // W, np.arange(TOK) % W], 1)
Dg = np.sqrt(((siatka[:, None] - siatka[None]) ** 2).sum(-1))
SAS = (Dg <= np.sqrt(2) + 1e-9) & (Dg > 0)
PARY = np.array([(i, j) for i in range(TOK) for j in range(i + 1, TOK) if Dg[i, j] == 1])          # pary sąsiadów pozioma/pionowa
kx = np.minimum(np.arange(H), H - np.arange(H))[:, None]; ky = np.minimum(np.arange(W), W - np.arange(W))[None]
KMOD = np.sqrt(kx ** 2 + ky ** 2)
NAZWY = ['siatka', 'wyk_gL', 'wyk_msd', 'dlugie', 'precyzja', 'wiazania', 'alfa2']


# ---------------------------------------------------------------- miary
def miary(X, box=None, rng=None, Xprec=None):
    """X: (T, 600, D) położenia (rozwinięte), etykieta = indeks tokenu = węzeł siatki 20 × 30. Xprec: osobna (gęstsza) trajektoria do precyzji."""
    rng = rng or np.random.default_rng(0)
    X = X - X.mean(1, keepdims=True)
    T, N, D = X.shape
    sr = X.mean(0)
    if box is not None:
        sr = np.mod(sr, box)
        d = sr[:, None] - sr[None]; d -= box * np.round(d / box); Dl = np.sqrt((d ** 2).sum(-1))
    else:
        g = (sr ** 2).sum(1); Dl = np.sqrt(np.maximum(g[:, None] + g[None] - 2 * sr @ sr.T, 0))
    np.fill_diagonal(Dl, np.inf)
    nn = np.argsort(Dl, 1)[:, :6]
    siat = float(SAS[np.arange(N)[:, None], nn].mean())
    lagi = np.unique(np.round(np.geomspace(1, max(2, T // 3), 10)).astype(int))
    i, j = PARY[:, 0], PARY[:, 1]
    a2 = float(((X[0, i] - X[0, j]) ** 2).sum(-1).mean())
    gL, msd = [], []
    for L in lagi:
        dX = X[L:] - X[:-L]
        gL.append(float(((dX[:, i] - dX[:, j]) ** 2).sum(-1).mean() / (2 * a2)))
        msd.append(float((dX ** 2).sum(-1).mean()))
    gor = slice(len(lagi) // 2, None)
    wyk = lambda y: float(np.polyfit(np.log(lagi[gor]), np.log(np.array(y)[gor] + 1e-30), 1)[0])
    Lf = max(1, T // 10)
    dX = (X[Lf:] - X[:-Lf]).reshape(T - Lf, H, W, D)
    P = (np.abs(np.fft.fft2(dX, axes=(1, 2))) ** 2).sum((0, 3))
    dlugie = float(P[(KMOD > 0) & (KMOD <= 2.5)].sum() / P[KMOD > 0].sum())
    Lb = max(1, T // 4)
    b0 = X[:-Lb, j] - X[:-Lb, i]; b1 = X[Lb:, j] - X[Lb:, i]
    wiaz = float(((b0 * b1).sum(-1) / (np.linalg.norm(b0, axis=-1) * np.linalg.norm(b1, axis=-1) + 1e-12)).mean())
    d = X[Lf:] - X[:-Lf]; r2 = (d ** 2).sum(-1)
    alfa2 = float(D * (r2 ** 2).mean() / ((D + 2) * r2.mean() ** 2) - 1)
    Y = X if Xprec is None else Xprec - Xprec.mean(1, keepdims=True)
    u = (Y - Y.mean(0)).transpose(0, 2, 1).reshape(-1, N)                                       # próbki (T·D, 600)
    u = u[rng.choice(len(u), min(2000, len(u)), replace=False)]
    S = np.cov(u, rowvar=False); S = 0.9 * S + 0.1 * np.trace(S) / N * np.eye(N)
    Pm = np.abs(np.linalg.inv(S)); np.fill_diagonal(Pm, 0)
    prec = float(Pm[SAS].sum() / Pm.sum())
    return dict(siatka=siat, wyk_gL=wyk(gL), wyk_msd=wyk(msd), dlugie=dlugie, precyzja=prec, wiazania=wiaz, alfa2=alfa2)


# ---------------------------------------------------------------- układy wzorcowe
def jadro_gladkie(l):
    K = np.exp(-Dg ** 2 / (2 * l ** 2)); w, V = np.linalg.eigh(K + 1e-8 * np.eye(TOK))
    Ls = V * np.sqrt(np.maximum(w, 0)); return Ls / np.sqrt((Ls ** 2).sum(1, keepdims=True))


SPR = [(i, j, 1.0, 1.0) for i in range(TOK) for j in range(i + 1, TOK) if Dg[i, j] == 1] + \
      [(i, j, np.sqrt(2), 0.5) for i in range(TOK) for j in range(i + 1, TOK) if abs(Dg[i, j] - np.sqrt(2)) < 1e-9]
SI, SJ, SL, SK = (np.array(v) for v in zip(*SPR))


def sily_sprezyn(x, L=None):
    L = SL if L is None else L
    r = x[SJ] - x[SI]; d = np.linalg.norm(r, axis=1, keepdims=True)
    f = (SK * (d[:, 0] - L))[:, None] * r / d
    F = np.zeros_like(x); np.add.at(F, SI, f); np.add.at(F, SJ, -f); return F


def sily_dyskow(x, sig, box):
    d = x[:, None] - x[None]; d -= box * np.round(d / box)
    r = np.sqrt((d ** 2).sum(-1)) + np.eye(len(x)); s = (sig[:, None] + sig[None]) / 2
    m = (r < s) & ~np.eye(len(x), dtype=bool)
    f = np.where(m, (1 - r / s) / s / r, 0.0)
    return (f[:, :, None] * d).sum(1)


def symuluj(par):
    klasa, T_, co, ziarno = par['klasa'], par['T'], par['co'], par['ziarno']
    rng = np.random.default_rng(ziarno)
    dt, KLATKI, ROZGRZEWKA = 0.05, 1000, 3000
    s0 = siatka[:, ::-1].astype(float)                                                            # (x, y) = (kolumna, wiersz)
    box, sig = None, None
    if klasa in ('ciecz', 'szklo'):
        phi = 0.70 if klasa == 'ciecz' else 0.88
        sig = np.where(rng.random(TOK) < 0.5, 1.0, 1.4)
        A = (np.pi / 4 * (sig ** 2)).sum() / phi; Lx = np.sqrt(A * 1.5); box = np.array([Lx, Lx / 1.5])
        x = rng.random((TOK, 2)) * box
        for _ in range(3000):                                                                     # schładzanie / usunięcie nakładań
            x = x + dt * sily_dyskow(x, sig, box)
    else:
        s0 = s0 + par.get('sd', 0.0) * rng.standard_normal((TOK, 2))                             # nieporządek statyczny: przesunięte kotwice / kształt spoczynkowy
        x = s0 + 0.01 * rng.standard_normal((TOK, 2))
    Lsp = np.linalg.norm(s0[SJ] - s0[SI], axis=1)                                                 # długości spoczynkowe sprężyn z (nieuporządkowanego) kształtu
    gl = jadro_gladkie(par.get('l', 3.0)) if klasa in ('krysztal_napedzany', 'pulapki_naped', 'krysztal_pelzajacy', 'siatka_zewn') else None
    J = None
    if par.get('g', 0.0) > 0:                                                                     # sprzężenie wszystkich ze wszystkimi (losowe, symetryczne, promień ≈ g)
        J = rng.standard_normal((TOK, TOK)); J = (J + J.T) / 2 * par['g'] / (2 * np.sqrt(TOK) / np.sqrt(2)); np.fill_diagonal(J, 0)
    kot = s0.copy(); vk = par.get('v', 0.0)
    w = par.get('w', 0.0)

    def krok(x, kot):
        if klasa in ('ciecz', 'szklo'):
            F = sily_dyskow(x, sig, box)
        elif klasa.startswith('krysztal'):
            F = sily_sprezyn(x, Lsp)
            if klasa == 'krysztal_pelzajacy':
                F = F + 0.1 * (kot - x)
        else:
            F = 1.0 * (kot - x)
        if J is not None:
            F = F + J @ (x - kot)
        xi = rng.standard_normal((TOK, 2))
        if gl is not None and w > 0:
            xi = np.sqrt(1 - w) * xi + np.sqrt(w) * (gl @ rng.standard_normal((TOK, 2)))
        if vk > 0:
            kot = kot + vk * np.sqrt(dt) * (gl @ rng.standard_normal((TOK, 2)))
        return x + dt * F + np.sqrt(2 * T_ * dt) * xi, kot

    for _ in range(ROZGRZEWKA):
        x, kot = krok(x, kot)
    if box is not None:                                                                           # etykiety siatki: wiersze wg y, w wierszu wg x
        xm = np.mod(x, box); rz = np.argsort(xm[:, 1]); perm = np.concatenate([r[np.argsort(xm[r, 0])] for r in np.array_split(rz, H)])
    else:
        perm = np.arange(TOK)
    zap = np.empty((KLATKI, TOK, 2)); xr = x.copy()
    for t in range(KLATKI):
        for _ in range(co):
            xn, kot = krok(x, kot)
            dx = xn - x
            if box is not None:
                dx -= box * np.round(dx / box); xn = np.mod(xn, box)
            xr = xr + dx; x = xn
        zap[t] = xr[perm]
    m = miary(zap[::10], box=box, rng=rng, Xprec=zap)                                             # box: średnie położenia w pudle okresowym
    return dict(par, **m)


SIATKA_PAR = []
for z in (1, 2):
    for co in (2, 10, 40):
        for T_ in (0.005, 0.03):
            SIATKA_PAR += [dict(klasa='krysztal', T=T_, co=co, ziarno=100 * z + co),
                           dict(klasa='krysztal_pelzajacy', T=T_, co=co, v=0.05, l=4.0, ziarno=200 * z + co),
                           dict(klasa='krysztal_napedzany', T=T_, co=co, w=0.8, l=3.0, ziarno=300 * z + co),
                           dict(klasa='pulapki_naped', T=T_, co=co, w=0.8, l=3.0, v=0.05, ziarno=400 * z + co),
                           dict(klasa='pulapki_naped', T=T_, co=co, w=0.95, l=5.0, v=0.1, ziarno=450 * z + co),
                           dict(klasa='pulapki', T=T_, co=co, ziarno=500 * z + co)]
        SIATKA_PAR += [dict(klasa='ciecz', T=T_, co=co, ziarno=600 * z + co) for T_ in (0.005, 0.02)]
        SIATKA_PAR += [dict(klasa='szklo', T=T_, co=co, ziarno=700 * z + co) for T_ in (2e-4, 1e-3)]

RUNDA2 = os.environ.get('RUNDA2') == '1'
if RUNDA2:                                                                                        # wzorce z nieporządkiem statycznym i sprzężeniem globalnym
    for z in (1, 2):
        for co in (2, 10, 40):
            for T_ in (0.03, 0.1):
                for sd in (0.2, 0.35):
                    SIATKA_PAR += [dict(klasa='siatka_zewn', T=T_, co=co, sd=sd, w=0.8, l=3.0, v=0.01, g=0.5, ziarno=800 * z + co + int(100 * sd)),
                                   dict(klasa='siatka_zewn', T=T_, co=co, sd=sd, w=0.5, l=3.0, v=0.0, g=0.8, ziarno=850 * z + co + int(100 * sd)),
                                   dict(klasa='krysztal_napedzany', T=T_, co=co, sd=sd, w=0.8, l=3.0, ziarno=900 * z + co + int(100 * sd)),
                                   dict(klasa='pulapki_naped', T=T_, co=co, sd=sd, w=0.8, l=3.0, v=0.01, ziarno=950 * z + co + int(100 * sd))]


# ---------------------------------------------------------------- tokeny
def tokeny_ruch60():
    import json as js
    M = js.load(open(f'{EKS}/META.json', encoding='utf-8')); kroki = np.array(M['kroki']); EP = 1094
    rp = np.load(f'{EKS}/rp128.npy', mmap_mode='r')
    zd = list(range(10)) + list(range(50, 60))
    out = []
    for e1, e2 in ((1, 5), (10, 20), (30, 40), (50, 60)):
        idx = [i for i, k in enumerate(kroki) if k % 100 == 0 and EP * e1 <= k < EP * e2][:100]
        for n in zd:
            X = np.asarray(rp[idx, n]).astype(np.float64)
            out.append(dict(zrodlo='ruch60', etap=f'{e1}-{e2}', zdjecie=n, split='TRAIN' if n < 50 else 'VALID', **miary(X)))
    return out


def tokeny_gesty(bieg, od=300, do=2188):
    pl = sorted(glob.glob(f'{POK}/{bieg}/gesty/pca32_*.npz'))
    if not pl:
        return []
    kr, Z = [], []
    for p in pl:
        z = np.load(p); k = z['kroki']; m = (k >= od) & (k <= do)
        if m.any():
            kr.append(k[m]); Z.append(z['pca32'][m].astype(np.float32))
    kr = np.concatenate(kr); Z = np.concatenate(Z); o = np.argsort(kr); kr, Z = kr[o], Z[o]
    sel = np.unique(np.round(np.linspace(0, len(kr) - 1, 100)).astype(int))
    return [dict(zrodlo=bieg, etap=f'{od}-{do}', zdjecie=s, **miary(Z[sel, s].astype(np.float64))) for s in range(Z.shape[1])]


if __name__ == '__main__':
    log(f'wzorców {len(SIATKA_PAR)}, wątków {WAT}')
    with Pool(WAT) as pool:
        wz = []
        for r in pool.imap_unordered(symuluj, SIATKA_PAR):
            wz.append(r); log(f'{r["klasa"]:>20s} T={r["T"]:<6} co={r["co"]:<3}: ' + ', '.join(f'{k} {r[k]:.3f}' for k in NAZWY))
    tok = tokeny_ruch60(); log(f'tokeny ruch60: {len(tok)}')
    for b in ('lr_5e-4_rp', 'lr_5e-4_rp_z48_zostaw', 'lr_5e-4_rp_z49_zostaw', 'lr_2e-3_rp_zostaw', 'bezpoz_lr_5e-4_rp', 'cnn_lr_5e-4_rp_zostaw'):
        t = tokeny_gesty(b); tok += t; log(f'tokeny {b}: {len(t)}')

    F = np.array([[r[k] for k in NAZWY] for r in wz]); F[:, NAZWY.index('alfa2')] = np.arcsinh(F[:, NAZWY.index('alfa2')])
    mu, sd = F.mean(0), F.std(0) + 1e-12; Z = (F - mu) / sd; kl = np.array([r['klasa'] for r in wz])
    Dz = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)); np.fill_diagonal(Dz, np.inf)
    KNN = 3
    loo = np.mean([max(set(v := list(kl[np.argsort(Dz[i])[:KNN]])), key=v.count) == kl[i] for i in range(len(kl))])
    prog = float(np.quantile(Dz.min(1), 0.95))
    klasy = sorted(set(kl)); cen = {c: Z[kl == c].mean(0) for c in klasy}
    macierz = {c: {d: int(np.sum((kl == c) & (np.array([max(set(v := list(kl[np.argsort(Dz[i])[:KNN]])), key=v.count) for i in range(len(kl))]) == d))) for d in klasy} for c in klasy}
    log(f'trafność kNN (k={KNN}) z pominięciem jednego na wzorcach: {loo:.3f}; próg „poza wzorcami” {prog:.2f}')
    for r in tok:
        f = np.array([r[k] for k in NAZWY]); f[NAZWY.index('alfa2')] = np.arcsinh(f[NAZWY.index('alfa2')]); z = (f - mu) / sd
        d = np.sqrt(((Z - z) ** 2).sum(1)); o = np.argsort(d)[:KNN]; v = list(kl[o])
        r.update(klasa_knn=max(set(v), key=v.count), glosy={c: v.count(c) for c in set(v)}, odl_najblizszy=float(d[o[0]]), poza_wzorcami=bool(d[o[0]] > prog),
                 odl_centroidy={c: float(np.sqrt(((cen[c] - z) ** 2).sum())) for c in klasy})
    grupy = {}
    for r in tok:
        g = f'{r["zrodlo"]} {r["etap"]}'; grupy.setdefault(g, []).append(r)
    podsum = {}
    for g, rs in grupy.items():
        c = [r['klasa_knn'] for r in rs]
        podsum[g] = dict(n=len(rs), klasy={k: c.count(k) / len(c) for k in sorted(set(c))}, poza_wzorcami=float(np.mean([r['poza_wzorcami'] for r in rs])),
                         miary={k: float(np.median([r[k] for r in rs])) for k in NAZWY},
                         odl_centroidy={k: float(np.median([r['odl_centroidy'][k] for r in rs])) for k in klasy})
        log(f'{g:>36s}: {podsum[g]["klasy"]}, poza wzorcami {podsum[g]["poza_wzorcami"]:.2f}; ' + ', '.join(f'{k} {podsum[g]["miary"][k]:.3f}' for k in NAZWY))
    wz_tab = {c: {k: float(np.median([r[k] for r in wz if r['klasa'] == c])) for k in NAZWY} for c in klasy}
    for c in klasy:
        log(f'wzorzec {c:>20s}: ' + ', '.join(f'{k} {wz_tab[c][k]:.3f}' for k in NAZWY))
    json.dump(dict(opis=__doc__.split('\n')[0], miary=NAZWY, wzorce=wz, wzorce_mediany=wz_tab, trafnosc_loo=float(loo), k=KNN, macierz_pomylek_loo=macierz,
                   prog_poza_wzorcami=prog, tokeny=tok, podsumowanie=podsum), open(f'{OUT}/SLEPA.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        U, s_, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); P2 = (Z - Z.mean(0)) @ Vt[:2].T
        fig, ax = plt.subplots(1, 1, figsize=(11, 8), dpi=110)
        kol = dict(zip(klasy, plt.cm.tab10(np.arange(len(klasy)))))
        for c in klasy:
            ax.scatter(*P2[kl == c].T, s=30, color=kol[c], label=f'wzorzec: {c}', alpha=0.8)
        for g, rs in grupy.items():
            f = np.array([[r[k] for k in NAZWY] for r in rs]); f[:, NAZWY.index('alfa2')] = np.arcsinh(f[:, NAZWY.index('alfa2')])
            p = (((f - mu) / sd) - Z.mean(0)) @ Vt[:2].T
            ax.scatter(*p.T, s=12, marker='x', color='k' if 'ruch60' in g else 'grey')
            ax.annotate(g, p.mean(0), fontsize=7)
        ax.legend(fontsize=8); ax.set_title(f'Ślepa klasyfikacja: wzorce (kolory) i tokeny (x); trafność LOO {loo:.2f}'); fig.tight_layout(); fig.savefig(f'{OUT}/slepa.png')
    except Exception as e:  # noqa: BLE001
        log('bez rysunku', e)
    log('KONIEC')
