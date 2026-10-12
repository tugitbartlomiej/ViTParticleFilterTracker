"""Heterogeniczność dynamiczna i ruch kooperatywny: test, który odróżnia szkło wielu ciał od niezależnych cząstek w dołkach.

Cząstka w dołku (OU) może dać plateau MSD, antykorelację, rozciągnięty eksponens i starzenie, ale NIE może dać
skorelowanych w przestrzeni fluktuacji ruchliwości ani ruchu kooperatywnego, bo nie wie o sąsiadach. Szkło daje
(Lacevic 2003, Berthier 2005, Donati 1997). Liczymy więc, dla tych samych układów co w odciskach:

  1. C_k(τ): korelacja Pearsona (po cząstkach, w jednej chwili) między fluktuacją ruchliwości cząstki δm_i(t,τ) i średnią
     fluktuacją ruchliwości jej k-tej powłoki sąsiadów (k=1: 6 najbliższych w chwili odniesienia; k=2, 3: sąsiedzi
     sąsiadów). Ruchliwość m_i = |Δ_i(t,τ)| / s_i, gdzie s_i to własna RMS ruchliwość cząstki w oknie (jednakowe cząstki,
     jak etap S2), a δm to odchylenie od średniej bieżącej po początkach (okno ±W) — usuwa statyczną heterogeniczność
     (gorętsze obszary) i trend stygnięcia. Kontrola zerowa: ta sama korelacja z sąsiadami wziętymi z INNEJ, losowej
     chwili (ta sama struktura statyczna, brak współfluktuacji). Zanik C_k z k to długość korelacji w powłokach.
     Ruchliwość jest WZGLĘDNA: m_i dzielone przez średnią ruchliwość zdjęcia (bloku) w tej samej chwili, bo w sieci wielkość
     kroku skacze z paczki na paczkę wspólnie dla całego zdjęcia (to nie heterogeniczność, lecz wspólna temperatura chwili);
     wariant bez tego normowania zapisany jako C_surowe.
  2. Skupianie ruchliwych: P(sąsiad ruchliwy | cząstka ruchliwa) / P(ruchliwy); ruchliwe = górne 10 % δm. Losowo: 1.
  3. Ruch kooperatywny: średni cos(Δ_i, Δ_j) dla par sąsiadów minus dla par losowych z tego samego bloku (zdjęcia),
     w układzie bez pola bloku (ViT: bez średniego ruchu zdjęcia; BD/OU: bez średniego ruchu populacji).
  4. ViT: C_1 dla sąsiadów z INNYCH zdjęć (czy latent jest ośrodkiem: bliskość w latencie między zdjęciami niesie
     skorelowaną ruchliwość) oraz w dwóch układach: lab (bez pola zdjęcia) i CR (względem klatki, jak S1).

Układy: tokeny ViT wczesne (kroki 300–2188, co 1) i późne (2188–22 970, co 10) z gęstego zapisu 100 sond (NS pierwszych);
szkło BD z Edena (bieg duży, stała T) przy φ = 0,58 / 0,61 / 0,62 / 0,64, 4096 kul, 2000 klatek; kontrola OU-B (jak w
kontrola_ou.py: dołki o rozrzucie 10×, kąpiel stygnie jak tokeny).
Wynik: HETEROGENICZNOSC.json, heterogenicznosc.png.  Uruchomienie: F:/Python/3.13/python.exe heterogenicznosc.py [NS]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odciski as O  # noqa: E402

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TOK = 600
G = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty'
OSIE = r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui'
Q16 = 65535.0
rng = np.random.default_rng(7)
t0 = time.time()


def log(*a):
    print(*a, f'[{time.time() - t0:.0f} s]', flush=True)


# ------------------------------------------------------------------ powłoki sąsiadów (macierze rzadkie uśredniające)
def powloki(N6, k_max=3):
    N = len(N6)
    A1 = sp.csr_matrix((np.ones(N6.size), (np.repeat(np.arange(N), N6.shape[1]), N6.ravel())), shape=(N, N))
    A1.data[:] = 1.0
    wynik = [A1]
    zajete = (A1 + sp.identity(N, format='csr')).astype(bool)
    poprz = A1.astype(bool)
    for _ in range(2, k_max + 1):
        nast = (poprz @ A1).astype(bool)
        nast = (nast.astype(np.int8) - nast.multiply(zajete).astype(np.int8)).astype(bool)   # bez już zajętych
        nast.eliminate_zeros()
        wynik.append(nast.astype(np.float64).tocsr())
        zajete = (zajete + nast).astype(bool)
        poprz = nast
    srednie = []
    for A in wynik:
        A = A.tocsr().astype(np.float64)
        deg = np.asarray(A.sum(1)).ravel()
        deg[deg == 0] = np.inf
        srednie.append(sp.diags(1.0 / deg) @ A)
    rozmiar = [float(np.asarray(A.sum(1)).ravel().mean()) for A in wynik]
    return srednie, rozmiar


def knn_poza_blokiem(x, blok, kk=6, porcja=1500):
    """kk najbliższych sąsiadów spoza własnego bloku (zdjęcia): (N, kk)."""
    N = len(x)
    n2 = (x ** 2).sum(1)
    out = np.empty((N, kk), int)
    for a in range(0, N, porcja):
        D = n2[a:a + porcja, None] + n2[None, :] - 2 * x[a:a + porcja] @ x.T
        for i in range(len(D)):
            b = ((a + i) // blok) * blok
            D[i, b:b + blok] = np.inf
        out[a:a + porcja] = np.argpartition(D, kk, axis=1)[:, :kk]
    return out


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float('nan')


# ------------------------------------------------------------------ analiza jednego układu w jednym układzie odniesienia
def analiza(P, kroki, okno, taus, N6, blok, nazwa, pole='blok', N6_cross=None, W=5, max_pocz=80, prog=0.10):
    """P: (T, N, d) trajektorie w docelowym układzie (lab albo CR); kroki: numer kroku każdego rekordu; okno: (od, do) w krokach;
    taus w KROKACH (muszą być wielokrotnościami odstępu rekordów); N6: sąsiedzi (N, 6) z chwili odniesienia; blok: cząstek na blok
    (zdjęcie) albo None (populacja); pole: 'blok' odejmuje średnie przesunięcie bloku, 'populacja' — populacji."""
    ind = {int(k): i for i, k in enumerate(kroki)}
    w_oknie = [int(k) for k in kroki if okno[0] <= k <= okno[1]]
    T, N, d = P.shape
    Ak, rozm = powloki(N6, 3)
    A1 = Ak[0]
    if N6_cross is not None:
        Ac = powloki(N6_cross, 1)[0][0]

    def pole_(D):
        if pole == 'blok' and blok:
            return D - np.repeat(D.reshape(-1, blok, d).mean(1), blok, axis=0)
        return D - D.mean(0)

    # własna ruchliwość s_i: RMS przesunięcia między kolejnymi rekordami w oknie (bez pola)
    acc = np.zeros(N); n_acc = 0
    for a, b in zip(w_oknie[:-1], w_oknie[1:]):
        D = pole_(P[ind[b]] - P[ind[a]])
        acc += (D ** 2).sum(1); n_acc += 1
    s = np.sqrt(acc / max(n_acc, 1)); s[s <= 0] = np.nan
    wyn = dict(nazwa=nazwa, uklad=pole, N=int(N), rekordow_w_oknie=len(w_oknie), rozmiar_powlok=rozm, taus=[], C=[], C_null=[], C_glob=[], C_surowe=[],
               skupianie=[], cos_sasiedzi=[], cos_losowe=[], cos_roznica=[], C_cross=[], C_cross_null=[], n_poczatkow=[])
    for tau in taus:
        pocz = [t for t in w_oknie if t + tau <= okno[1] and (t + tau) in ind]
        if len(pocz) < 6:
            continue
        if len(pocz) > max_pocz:
            pocz = [pocz[i] for i in np.linspace(0, len(pocz) - 1, max_pocz).round().astype(int)]
        M = np.empty((len(pocz), N), np.float32); Msur = np.empty((len(pocz), N), np.float32)
        kos_s, kos_l = [], []
        for o, t in enumerate(pocz):
            D = pole_(P[ind[t + tau]] - P[ind[t]])
            mo = np.linalg.norm(D, axis=1) / s
            if blok:
                mo = mo / np.repeat(np.nanmean(mo.reshape(-1, blok), 1), blok)      # względem średniej ruchliwości zdjęcia w tej chwili
            else:
                mo = mo / np.nanmean(mo)
            M[o] = mo
            Msur[o] = np.linalg.norm(D, axis=1) / s
            # ruch kooperatywny: cos między sąsiadami kontra pary losowe z bloku
            Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
            kos_s.append(float((Dn[:, None, :] * Dn[N6]).sum(2).mean()))
            if blok:
                j = (np.arange(N) // blok) * blok + rng.integers(0, blok, N)
            else:
                j = rng.integers(0, N, N)
            kos_l.append(float((Dn * Dn[j]).sum(1).mean()))
        ok = np.isfinite(M).all(0) & np.isfinite(Msur).all(0)
        M = M[:, ok]; Msur = Msur[:, ok]
        w_ = min(W, (len(pocz) - 1) // 2)
        dM = M - (uniform_filter1d(M, 2 * w_ + 1, axis=0, mode='nearest') if w_ >= 1 else M.mean(0))
        dMg = M - M.mean(0)
        dMs = Msur - (uniform_filter1d(Msur, 2 * w_ + 1, axis=0, mode='nearest') if w_ >= 1 else Msur.mean(0))   # surowe: bez normowania chwilowego
        A1o = A1[ok][:, ok]; Ako = [A[ok][:, ok] for A in Ak]
        Cs, Cn, Cg, sk, Cc, Ccn, Csur = [], [], [], [], [], [], []
        for o in range(len(pocz)):
            Cs.append([corr(dM[o], A @ dM[o]) for A in Ako])
            o2 = (o + rng.integers(1, len(pocz))) % len(pocz)
            Cn.append(corr(dM[o], A1o @ dM[o2]))
            Cg.append(corr(dMg[o], A1o @ dMg[o]))
            Csur.append(corr(dMs[o], A1o @ dMs[o]))
            ruchliwe = dM[o] >= np.quantile(dM[o], 1 - prog)
            sk.append(float((A1o @ ruchliwe.astype(float))[ruchliwe].mean() / prog))
            if N6_cross is not None:
                Aco = Ac[ok][:, ok]
                Cc.append(corr(dM[o], Aco @ dM[o])); Ccn.append(corr(dM[o], Aco @ dM[o2]))
        Cs = np.array(Cs)
        wyn['taus'].append(int(tau)); wyn['n_poczatkow'].append(len(pocz))
        wyn['C'].append([float(x) for x in np.nanmean(Cs, 0)]); wyn['C_null'].append(float(np.nanmean(Cn))); wyn['C_glob'].append(float(np.nanmean(Cg))); wyn['C_surowe'].append(float(np.nanmean(Csur)))
        wyn['skupianie'].append(float(np.mean(sk)))
        wyn['cos_sasiedzi'].append(float(np.mean(kos_s))); wyn['cos_losowe'].append(float(np.mean(kos_l))); wyn['cos_roznica'].append(float(np.mean(kos_s) - np.mean(kos_l)))
        if N6_cross is not None:
            wyn['C_cross'].append(float(np.nanmean(Cc))); wyn['C_cross_null'].append(float(np.nanmean(Ccn)))
    C1 = np.array([c[0] for c in wyn['C']]) if wyn['C'] else np.array([np.nan])
    i = int(np.nanargmax(C1)) if np.isfinite(C1).any() else 0
    wyn['szczyt'] = dict(tau=wyn['taus'][i] if wyn['taus'] else None, C1=float(C1[i]), C_powloki=wyn['C'][i] if wyn['C'] else None,
                         null=wyn['C_null'][i] if wyn['C_null'] else None, skupianie=wyn['skupianie'][i] if wyn['skupianie'] else None,
                         cos_roznica=wyn['cos_roznica'][i] if wyn['cos_roznica'] else None)
    log(f"{nazwa} [{pole}]: C1 max {wyn['szczyt']['C1']:.3f} @τ={wyn['szczyt']['tau']} (null {wyn['szczyt']['null']:.3f}; surowe {max(wyn['C_surowe']):.3f}), "
        f"powłoki {['%.3f' % c for c in (wyn['szczyt']['C_powloki'] or [])]}, skupianie {wyn['szczyt']['skupianie']:.2f}, "
        f"cos sąsiedzi−losowe {wyn['szczyt']['cos_roznica']:.3f}"
        + (f", cross C1 {max(wyn['C_cross']):.3f} (null {np.mean(wyn['C_cross_null']):.3f})" if wyn['C_cross'] else ''))
    return wyn


def klatka_cr(P, kroki, ref_krok, N6):
    ind = {int(k): i for i, k in enumerate(kroki)}
    Zr = P[ind[ref_krok]]
    Q = np.empty_like(P)
    for t in range(len(P)):
        Q[t] = P[t] - (P[t][N6] - Zr[N6]).mean(1)
    return Q


WYNIKI = {}

# ================================================================== 1. tokeny ViT: gęsty zapis sond
def wczytaj_gesty(od, do):
    Z, kr = [], []
    for p in sorted(glob.glob(os.path.join(G, 'pca32_*.npz'))):
        z = np.load(p)
        k = z['kroki']
        m = (k >= od) & (k <= do)
        if m.any():
            Z.append(np.ascontiguousarray(z['pca32'][m][:, :NS].astype(np.float32)).reshape(int(m.sum()), NS * TOK, 32))
            kr.append(k[m])
        del z
    return np.concatenate(Z), np.concatenate(kr).astype(int)


log('ViT wczesne: wczytywanie', NS, 'sond')
Z, kroki = wczytaj_gesty(300, 2188)
log('wczytane', Z.shape)
ind = {int(k): i for i, k in enumerate(kroki)}
N6, dnn, dsh = O.knn_w_blokach(Z[ind[300]], TOK, 6)
N6x = knn_poza_blokiem(Z[ind[300]], TOK, 6)
taus_w = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
WYNIKI['ViT wczesne lab'] = analiza(Z, kroki, (300, 2188), taus_w, N6, TOK, 'ViT wczesne (300–2188)', pole='blok', N6_cross=N6x)
P1 = klatka_cr(Z, kroki, 300, N6)
del Z
WYNIKI['ViT wczesne CR'] = analiza(P1, kroki, (300, 2188), taus_w, N6, TOK, 'ViT wczesne (300–2188)', pole='blok', N6_cross=N6x)
del P1

log('ViT późne: wczytywanie')
Z, kroki = wczytaj_gesty(2188, 22970)
Z, kroki = Z[::1], kroki[::1]
log('wczytane', Z.shape, 'odstęp rekordów', int(np.median(np.diff(kroki))))
ind = {int(k): i for i, k in enumerate(kroki)}
ref = int(kroki[0])
N6l, _, _ = O.knn_w_blokach(Z[ind[ref]], TOK, 6)
N6xl = knn_poza_blokiem(Z[ind[ref]], TOK, 6)
taus_p = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000]
WYNIKI['ViT późne lab'] = analiza(Z, kroki, (5000, 22970), taus_p, N6l, TOK, 'ViT późne (5000–22970)', pole='blok', N6_cross=N6xl)
P1 = klatka_cr(Z, kroki, ref, N6l)
del Z
WYNIKI['ViT późne CR'] = analiza(P1, kroki, (5000, 22970), taus_p, N6l, TOK, 'ViT późne (5000–22970)', pole='blok', N6_cross=N6xl)
del P1

# ================================================================== 2. szkło BD z Edena (bieg duży, stała T)
def wczytaj_bd(folder):
    m = json.load(open(os.path.join(folder, 'manifest.json'), encoding='utf-8'))
    steps = m['steps']
    X = np.empty((len(steps), m['point_count'], 3), np.float32)
    for i, s in enumerate(steps):
        f = m['files'][str(s)]
        q = np.frombuffer(open(os.path.join(folder, f['name']), 'rb').read(), dtype='<u2').reshape(-1, 4).astype(np.float32) / Q16
        X[i] = np.array(f['quant']['lo'], np.float32) + q[:, :3] * np.array(f['quant']['span'], np.float32)
    return X, np.array(steps), m


bd = json.load(open(os.path.join(HERE, 'eden', 'wyniki', 'MODEL_SZKLA_EDEN_stala.json'), encoding='utf-8'))['biegi']
for phi in ('0.58', '0.61', '0.62', '0.64'):
    folder = os.path.join(OSIE, f'20261010T_szklo_eden_szklo_BD_eden_stala_phi_{phi}_duze')
    if not os.path.isdir(folder):
        log('brak', folder); continue
    X, steps, m = wczytaj_bd(folder)
    L = bd[f'phi_{phi}']['L']
    tauB = bd[f'phi_{phi}']['tau_B_w_rekordach'] / m['uklad']['klatka_co_rekordow']        # τ_B w klatkach osi
    N6b, _, _ = O.knn_3d_pudlo(X[0], L, 6)
    taus_b = [1, 2, 3, 5, 10, 20, 45, 100, 200, 400, 800]
    nazwa = f'BD φ={phi}'
    w = analiza(X, steps, (0, int(steps[-1])), taus_b, N6b, None, nazwa, pole='populacja')
    w['tau_B_w_klatkach'] = tauB; w['a2_szczyt_tau_klatki'] = bd[f'phi_{phi}']['a2_szczyt_tau'] / m['uklad']['klatka_co_rekordow']
    w['tau_1e_klatki'] = (bd[f'phi_{phi}']['tau_1e'] or float('inf')) / m['uklad']['klatka_co_rekordow']
    WYNIKI[f'{nazwa} lab'] = w
    Xc = klatka_cr(X, steps, int(steps[0]), N6b)
    del X
    w = analiza(Xc, steps, (0, int(steps[-1])), taus_b, N6b, None, nazwa, pole='populacja')
    w['tau_B_w_klatkach'] = tauB
    WYNIKI[f'{nazwa} CR'] = w
    del Xc

# ================================================================== 3. kontrola OU-B (jak kontrola_ou.py)
u = json.load(open(os.path.join(HERE, 'UKLAD_SZKLA.json'), encoding='utf-8'))
plateau_vit = u['etapy']['S1']['l_klatka'] ** 2
N_OU, D_OU, T_REF, KONIEC = 6000, 32, 300, 2188
r2 = np.random.default_rng(1)
kappa = 0.07 * np.exp(r2.normal(0, 1.0, N_OU))
k_med = float(np.median(kappa))
T_plateau = plateau_vit * k_med / (2 * D_OU)


def T_styg(t, T0):
    return T0 * (1 + max(t - T_REF, 0) / 400.0) ** (-0.79)


T0_B = T_plateau / T_styg(1000, 1.0)
x = np.zeros((N_OU, D_OU), np.float32)
P = np.empty((KONIEC + 1, N_OU, D_OU), np.float32)
ek = np.exp(-kappa)[:, None].astype(np.float32)
amp0 = np.sqrt((1 - np.exp(-2 * kappa)) / kappa)[:, None].astype(np.float32)
for t in range(KONIEC + 1):
    x = x * ek + np.sqrt(T_styg(t, T0_B)) * amp0 * r2.standard_normal((N_OU, D_OU)).astype(np.float32)
    P[t] = x
kr_ou = np.arange(KONIEC + 1)
N6o, _, _ = O.knn_w_blokach(P[T_REF], N_OU, 6)
WYNIKI['OU-B lab'] = analiza(P, kr_ou, (T_REF, KONIEC), taus_w, N6o, None, 'OU-B (dołki stygnące)', pole='populacja')
Pc = klatka_cr(P, kr_ou, T_REF, N6o)
del P
WYNIKI['OU-B CR'] = analiza(Pc, kr_ou, (T_REF, KONIEC), taus_w, N6o, None, 'OU-B (dołki stygnące)', pole='populacja')
del Pc

json.dump(dict(opis=__doc__, NS=NS, wyniki=WYNIKI), open(os.path.join(HERE, 'HETEROGENICZNOSC.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('zapisano HETEROGENICZNOSC.json')

# ================================================================== rysunek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(2, 2, figsize=(16, 11), dpi=100)
kol = {'ViT wczesne': '#ffd166', 'ViT późne': '#ff8c42', 'BD φ=0.58': '#6fa8dc', 'BD φ=0.61': '#4a8cff', 'BD φ=0.62': '#b36bff', 'BD φ=0.64': '#ff4433', 'OU-B': '#9aa3ad'}
for klucz, w in WYNIKI.items():
    if not w['taus']:
        continue
    baza = klucz.rsplit(' ', 1)[0]
    c = kol.get(baza, 'w'); ls = '-' if klucz.endswith('CR') else ':'
    t = np.array(w['taus'], float)
    if baza.startswith('BD'):
        t = t / w['tau_B_w_klatkach']; et = f'{baza} (τ/τ_B)'
    elif baza.startswith('ViT'):
        et = f'{baza} (kroki)'
    else:
        et = f'{baza} (kroki)'
    C1 = [c_[0] for c_ in w['C']]
    ax[0, 0].semilogx(t, C1, ls, color=c, lw=2 if ls == '-' else 1.4, marker='o', ms=3, label=f"{et} {'CR' if ls == '-' else 'lab'}")
    if klucz.endswith('CR'):
        ax[0, 0].semilogx(t, w['C_null'], '--', color=c, lw=0.8, alpha=0.7)
    ax[0, 1].semilogx(t, w['skupianie'], ls, color=c, lw=2 if ls == '-' else 1.4, marker='o', ms=3, label=f"{et} {'CR' if ls == '-' else 'lab'}")
    if klucz.endswith('lab'):
        ax[1, 0].semilogx(t, w['cos_roznica'], '-', color=c, lw=2, marker='o', ms=3, label=et)
    if klucz.endswith('CR'):
        ax[1, 1].plot([1, 2, 3], w['szczyt']['C_powloki'], '-o', color=c, lw=2, label=f"{baza} @ τ={w['szczyt']['tau']}")
        if w['C_cross']:
            ax[1, 1].plot([1], [max(w['C_cross'])], 'D', color=c, ms=9, mfc='none', label=f'{baza}: sąsiedzi z innych zdjęć')
ax[0, 0].axhline(0, color='w', lw=0.6, ls=':')
ax[0, 0].set(xlabel='τ (ViT, OU: kroki; BD: τ/τ_B)', ylabel='C₁(τ)', title='korelacja fluktuacji ruchliwości z 6 sąsiadami (kreskowane: kontrola z innej chwili)')
ax[0, 1].axhline(1, color='w', lw=0.6, ls=':')
ax[0, 1].set(xlabel='τ', ylabel='P(sąsiad ruchliwy | ruchliwy) / P(ruchliwy)', title='skupianie 10 % najruchliwszych (1 = losowo)')
ax[1, 0].axhline(0, color='w', lw=0.6, ls=':')
ax[1, 0].set(xlabel='τ', ylabel='⟨cos⟩ sąsiedzi − ⟨cos⟩ pary losowe', title='ruch kooperatywny (układ lab bez pola zdjęcia / populacji)')
ax[1, 1].axhline(0, color='w', lw=0.6, ls=':')
ax[1, 1].set(xlabel='powłoka sąsiadów k', ylabel='C_k w szczycie', title='zanik korelacji z odległością (długość korelacji w powłokach)', xticks=[1, 2, 3])
for a in ax.flat:
    a.grid(True, which='both', alpha=0.4); a.legend(fontsize=7)
fig.suptitle(f'Heterogeniczność dynamiczna i ruch kooperatywny: tokeny ViT ({NS} sond), szkło BD (Eden, N=4096), kontrola OU', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(HERE, 'heterogenicznosc.png'))
log('zapisano heterogenicznosc.png')
