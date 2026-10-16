"""Funkcje heterogeniczności dynamicznej (wyjęte z heterogenicznosc.py, żeby dało się je importować, np. na Edenie):
powloki, knn_poza_blokiem, corr, analiza(P, kroki, okno, taus, N6, blok, nazwa, pole, N6_cross, W, max_pocz, prog), klatka_cr.
Testy „jaka to substancja” (2026-10-11): trwalosc_ruchliwosci (heterogeniczność zamrożona czy przemijająca), korelacja_przesuniec
(dwupunktowa korelacja wektorów przesunięć w funkcji odległości, jak mikroreologia dwupunktowa), rozbieganie_par (test turbulencji:
Richardson/Batchelor kontra nasycenie), rozklad_wariancji (energia kroku: wspólny dryf / pole zdjęcia / pole sąsiedztwa / ruch własny)."""
import time

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import uniform_filter1d

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


def _pole(D, pole, blok):
    """Odejmuje pole: średnie przesunięcie bloku (zdjęcia) albo populacji."""
    if pole == 'blok' and blok:
        return D - np.repeat(D.reshape(-1, blok, D.shape[-1]).mean(1), blok, axis=0)
    return D - D.mean(0)


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
        return _pole(D, pole, blok)

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
        if ok.sum() < 10:
            log(f'{nazwa}: τ={tau} pominięte (tylko {int(ok.sum())} cząstek ze skończoną ruchliwością)')
            continue
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
    if not wyn['taus']:
        wyn['szczyt'] = dict(tau=None, C1=float('nan'), C_powloki=None, null=None, skupianie=None, cos_roznica=None)
        log(f'{nazwa} [{pole}]: brak τ z wystarczającą liczbą cząstek (ruch zerowy?)')
        return wyn
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


# ------------------------------------------------------------------ testy „jaka to substancja” (2026-10-11)
def trwalosc_ruchliwosci(P, kroki, okno, blok, tau=10, dlug=200, pole='blok', nazwa=''):
    """Heterogeniczność zamrożona (quenched, narzucona przez krajobraz) czy przemijająca (jak w szkle)?
    Ruchliwość cząstki w podoknie w (dlug kroków): m_i(w) = RMS |Δz_i(τ)| po wszystkich początkach w podoknie (po odjęciu pola).
    Korelacja Spearmana rankingów m_i między podoknami w funkcji odstępu (w podoknach): surowa (zawiera różnice między zdjęciami),
    względna (m_i / średnia zdjęcia w podoknie: ranking WEWNĄTRZ zdjęć) i ranking samych zdjęć. Szkło: korelacja zanika do ~0 poza τ_α;
    krajobraz: pozostaje wysoka (ruchliwe miejsca są stałe)."""
    from scipy.stats import spearmanr
    ind = {int(k): i for i, k in enumerate(kroki)}
    od, do = okno
    uzyte, M = [], []
    for g in range(od, do - tau, dlug):
        pocz = [t for t in range(g, min(g + dlug, do - tau + 1)) if t in ind and t + tau in ind]
        if len(pocz) < 10:
            continue
        acc = np.zeros(P.shape[1])
        for t in pocz:
            D = _pole(P[ind[t + tau]] - P[ind[t]], pole, blok)
            acc += (D ** 2).sum(1)
        M.append(np.sqrt(acc / len(pocz))); uzyte.append(int(g))
    M = np.array(M)
    W = len(M)
    if W < 2:
        return dict(nazwa=nazwa, uklad=pole, tau=tau, dlug=dlug, podokna=uzyte, uwaga='za mało podokien')
    if blok:
        Mz = M.reshape(W, -1, blok).mean(2)
        Mrel = M / np.repeat(Mz, blok, axis=1)
    else:
        Mz = None
        Mrel = M / M.mean(1, keepdims=True)

    def mac(X):
        R = np.eye(len(X))
        for a in range(len(X)):
            for b in range(a + 1, len(X)):
                R[a, b] = R[b, a] = float(spearmanr(X[a], X[b]).correlation)
        return R

    lagi = list(range(1, W))
    po_lagu = lambda R: [float(np.nanmean([R[a, a + l] for a in range(W - l)])) for l in lagi]
    R, Rr = mac(M), mac(Mrel)
    Rz = mac(Mz) if Mz is not None and Mz.shape[1] >= 3 else None
    wyn = dict(nazwa=nazwa, uklad=pole, tau=tau, dlug=dlug, podokna=uzyte, lagi=lagi, spearman_po_lagu=po_lagu(R),
               spearman_wzgledny_po_lagu=po_lagu(Rr), spearman_zdjec_po_lagu=(po_lagu(Rz) if Rz is not None else None),
               pierwsze_ostatnie=float(R[0, -1]), pierwsze_ostatnie_wzgledne=float(Rr[0, -1]),
               pierwsze_ostatnie_zdjec=(float(Rz[0, -1]) if Rz is not None else None),
               cv_ruchliwosci=float(np.mean(Mrel.std(1) / Mrel.mean(1))))
    log(f"{nazwa} [{pole}] trwałość ruchliwości (τ={tau}, podokna {dlug}): Spearman lag1 {wyn['spearman_po_lagu'][0]:.3f} → ostatni {wyn['spearman_po_lagu'][-1]:.3f}; "
        f"względny {wyn['spearman_wzgledny_po_lagu'][0]:.3f} → {wyn['spearman_wzgledny_po_lagu'][-1]:.3f}"
        + (f"; zdjęcia {wyn['spearman_zdjec_po_lagu'][0]:.3f} → {wyn['spearman_zdjec_po_lagu'][-1]:.3f}" if Rz is not None else ''))
    return wyn


def korelacja_przesuniec(P, kroki, okno, taus, blok, d_shell, n_pocz=24, n_par=240000, rng_=None, nazwa=''):
    """Dwupunktowa korelacja wektorów przesunięć (jak w mikroreologii dwupunktowej, Crocker 2000): C(r, τ) = <Δz_i·Δz_j> / <|Δz|²>
    dla losowych par w odległości r (odstęp w chwili startu, w jednostkach d_shell). Pary z tego samego zdjęcia: po odjęciu średniej
    populacji (zostaje pole zdjęcia) i po odjęciu średniej zdjęcia (pole wewnątrz zdjęcia); pary z różnych zdjęć: po odjęciu średniej
    populacji. Medium sprężyste/hydrodynamika: zanik potęgowy z r; pole gładkie od wspólnych wag: szeroki, płaski; szkło: krótki zasięg."""
    rng_ = rng_ or np.random.default_rng(11)
    ind = {int(k): i for i, k in enumerate(kroki)}
    od, do = okno
    T, N, d = P.shape
    jeden_blok = not blok or blok >= N                                 # bez podziału na zdjęcia (szkło BD, OU): tylko pary „wewnątrz”
    blok = N if jeden_blok else blok
    kraw = np.array([0, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 1e9]) * d_shell
    klucze = ['wewnatrz_populacja', 'wewnatrz_blok'] + ([] if jeden_blok else ['miedzy_populacja'])
    wyn = dict(nazwa=nazwa, d_shell=float(d_shell), krawedzie_d_shell=[float(k / d_shell) for k in kraw[:-1]] + ['inf'], taus=[], C={k: [] for k in klucze},
               n_par={k: [] for k in klucze}, r_sr={k: [] for k in klucze})
    for tau in taus:
        pocz = [int(t) for t in np.linspace(od, do - tau, n_pocz).astype(int) if int(t) in ind and int(t) + tau in ind]
        if not pocz:
            continue
        acc = {k: np.zeros(len(kraw) - 1) for k in klucze}; cnt = {k: np.zeros(len(kraw) - 1) for k in klucze}; rs = {k: np.zeros(len(kraw) - 1) for k in klucze}
        m = max(n_par // len(pocz), 1000)
        for t in pocz:
            X = P[ind[t]]
            D = P[ind[t + tau]] - X
            Dp = D - D.mean(0); Db = _pole(D, 'blok', blok)
            i = rng_.integers(0, N, m)
            jw = (i // blok) * blok + rng_.integers(0, blok, m); jw = np.where(jw == i, (i // blok) * blok + (i + 1) % blok, jw)
            jm = rng_.integers(0, N, m); zle = (jm // blok) == (i // blok); jm[zle] = (jm[zle] + blok) % N
            for k, A, j in [('wewnatrz_populacja', Dp, jw), ('wewnatrz_blok', Db, jw)] + ([] if jeden_blok else [('miedzy_populacja', Dp, jm)]):
                r = np.linalg.norm(X[i] - X[j], axis=1)
                dot = (A[i] * A[j]).sum(1) / (A ** 2).sum(1).mean()
                b = np.clip(np.digitize(r, kraw) - 1, 0, len(kraw) - 2)
                np.add.at(acc[k], b, dot); np.add.at(cnt[k], b, 1); np.add.at(rs[k], b, r)
        wyn['taus'].append(int(tau))
        for k in klucze:
            with np.errstate(invalid='ignore', divide='ignore'):
                wyn['C'][k].append([float(v) if c > 50 else None for v, c in zip(acc[k] / cnt[k], cnt[k])])
                wyn['r_sr'][k].append([float(v / d_shell) if c > 50 else None for v, c in zip(rs[k] / cnt[k], cnt[k])])
            wyn['n_par'][k].append([int(c) for c in cnt[k]])
    if wyn['taus']:
        c = wyn['C']['wewnatrz_blok'][0]; cm = wyn['C']['miedzy_populacja'][0] if not jeden_blok else []
        log(f"{nazwa} korelacja przesunięć τ={wyn['taus'][0]}: wewnątrz zdjęcia (bez pola zdjęcia) {[None if v is None else round(v, 3) for v in c[:10]]}; "
            f"między zdjęciami {[None if v is None else round(v, 3) for v in cm[:10]]}")
    return wyn


def rozbieganie_par(P, kroki, od, lagi, N6, d_shell, nazwa=''):
    """Test turbulencji: dla par (cząstka, jej sąsiad z chwili od) <|r_ij(od+lag)|²> / <|r_ij(od)|²> i udział par wciąż bliżej niż d_shell.
    Richardson: ∝ t³, Batchelor (gładki przepływ chaotyczny): wykładniczo, ciało stałe/szkło: nasycenie na ~2·klatka²."""
    ind = {int(k): i for i, k in enumerate(kroki)}
    i = np.repeat(np.arange(len(N6)), N6.shape[1]); j = N6.ravel()
    X0 = P[ind[od]]
    r0 = np.linalg.norm(X0[i] - X0[j], axis=1); r0m = float((r0 ** 2).mean())
    wyn = dict(nazwa=nazwa, od=int(od), d_shell=float(d_shell), r0_sr=float(r0.mean()), lagi=[], r2_wzgledne=[], udzial_blisko=[], r_sr=[])
    for l in lagi:
        if od + l not in ind:
            continue
        X = P[ind[od + l]]
        r = np.linalg.norm(X[i] - X[j], axis=1)
        wyn['lagi'].append(int(l)); wyn['r2_wzgledne'].append(float((r ** 2).mean() / r0m)); wyn['udzial_blisko'].append(float((r < d_shell).mean()))
        wyn['r_sr'].append(float(r.mean()))
    L = np.array(wyn['lagi'], float); R = np.array(wyn['r2_wzgledne'])
    m = (L >= 10) & (L <= 1000) & (R > 1)
    wyn['wykladnik_10_1000'] = float(np.polyfit(np.log(L[m]), np.log(R[m] - 1), 1)[0]) if m.sum() >= 3 else None
    m2 = (L >= 100) & (R > 1)
    wyn['wykladnik_od_100'] = float(np.polyfit(np.log(L[m2]), np.log(R[m2] - 1), 1)[0]) if m2.sum() >= 3 else None
    log(f"{nazwa} rozbieganie par: r²/r0² {[round(v, 2) for v in wyn['r2_wzgledne']]} przy lagach {wyn['lagi']}; wykładnik (r²−r0²) ∝ lag^{wyn['wykladnik_10_1000']}")
    return wyn


def rozklad_wariancji(P, kroki, okno, taus, N6, blok, n_pocz=30, nazwa=''):
    """Energia przesunięcia Δz(τ) rozłożona na: wspólny dryf populacji, pole zdjęcia (średnia zdjęcia bez dryfu), pole sąsiedztwa
    (średnia 6 sąsiadów z chwili odniesienia po odjęciu pola zdjęcia) i ruch własny (reszta = układ CR). Udziały względem <|Δz|²>."""
    ind = {int(k): i for i, k in enumerate(kroki)}
    od, do = okno
    A1 = powloki(N6, 1)[0][0]
    wyn = dict(nazwa=nazwa, taus=[], udzial=dict(wspolny=[], zdjecie=[], sasiedztwo=[], wlasny=[]), E_calk=[])
    for tau in taus:
        pocz = [int(t) for t in np.linspace(od, do - tau, n_pocz).astype(int) if int(t) in ind and int(t) + tau in ind]
        if not pocz:
            continue
        u = dict(wspolny=[], zdjecie=[], sasiedztwo=[], wlasny=[]); Et = []
        for t in pocz:
            D = P[ind[t + tau]] - P[ind[t]]
            tot = float((D ** 2).sum(1).mean())
            g = D.mean(0); D1 = D - g
            b = np.repeat(D1.reshape(-1, blok, D.shape[-1]).mean(1), blok, axis=0) if blok else np.zeros_like(D1)
            D2 = D1 - b
            l = A1 @ D2; D3 = D2 - l
            u['wspolny'].append(float((g ** 2).sum()) / tot); u['zdjecie'].append(float((b ** 2).sum(1).mean()) / tot)
            u['sasiedztwo'].append(float((l ** 2).sum(1).mean()) / tot); u['wlasny'].append(float((D3 ** 2).sum(1).mean()) / tot); Et.append(tot)
        wyn['taus'].append(int(tau)); wyn['E_calk'].append(float(np.mean(Et)))
        for k in u:
            wyn['udzial'][k].append(float(np.mean(u[k])))
    log(f"{nazwa} rozkład energii kroku: " + '; '.join(f"τ={t}: wspólny {wyn['udzial']['wspolny'][i]:.2f} zdjęcie {wyn['udzial']['zdjecie'][i]:.2f} "
                                                      f"sąsiedztwo {wyn['udzial']['sasiedztwo'][i]:.2f} własny {wyn['udzial']['wlasny'][i]:.2f}" for i, t in enumerate(wyn['taus'])))
    return wyn


def korelacja_krzyzowa_par(P, kroki, okno, N6, blok, W=100, taus=(0, 1, 2, 3, 5, 10, 20, 30, 50, 100), n_pocz=60, pole='blok', rng_=None, nazwa=''):
    """Test Meinersa–Quake'a (PRL 1999) / Morana: korelacja krzyżowa POŁOŻEŃ względem środka własnej klatki z opóźnieniem,
    C_ij(τ) = <u_i(t+τ)·u_j(t)> / sqrt(<|u_i|²><|u_j|²>), u_i = z_i − średnia krocząca (±W kroków) z_i, po odjęciu pola (zdjęcia albo populacji).
    Pary: 6 sąsiadów z chwili odniesienia, losowe z tego samego zdjęcia, losowe z innych zdjęć; dla porównania autokorelacja C_ii(τ).
      sprzężenie przez ruchliwość (hydrodynamika, niezależny szum): C_ij(0) ≈ 0, ujemny płat przy τ ~ czas klatki;
      wspólny szum w osobnych studniach (Moran): C_ij(0) > 0, ten sam znak dla wszystkich τ, zanik jak autokorelacja;
      sprężyny/klatka zbiorowa (szkło): C_ij(0) > 0, a korelacja PRZYROSTÓW rośnie z τ (dh.analiza C1 narasta do τ_α)."""
    rng_ = rng_ or np.random.default_rng(13)
    ind = {int(k): i for i, k in enumerate(kroki)}
    od, do = okno
    rek = [i for i, k in enumerate(kroki) if od <= k <= do]
    assert (np.diff(kroki[rek]) == 1).all(), 'potrzebny zapis co 1 krok'
    T, N, d = P.shape
    jeden_blok = not blok or blok >= N
    blok = N if jeden_blok else blok
    X = P[rek]                                                         # (Tw, N, d)
    Xp = np.stack([_pole(X[t], pole, blok) for t in range(len(X))])
    U = Xp - uniform_filter1d(Xp, 2 * W + 1, axis=0, mode='nearest')    # położenie względem środka klatki (średnia krocząca)
    U = U[W:-W] if len(U) > 2 * W + 20 else U                           # brzegi filtra odrzucone
    Tw = len(U)
    war = (U ** 2).sum(2).mean(0)                                       # <|u_i|²>
    war[war <= 0] = np.nan
    i6 = np.repeat(np.arange(N), N6.shape[1]); j6 = N6.ravel()
    m = 6 * N
    iw = rng_.integers(0, N, m); jw = (iw // blok) * blok + rng_.integers(0, blok, m); jw = np.where(jw == iw, (iw // blok) * blok + (iw + 1) % blok, jw)
    im_ = rng_.integers(0, N, m); jm = rng_.integers(0, N, m); zle = (jm // blok) == (im_ // blok); jm[zle] = (jm[zle] + blok) % N
    pary = dict(sasiedzi=(i6, j6), losowe_zdjecie=(iw, jw))
    if not jeden_blok:
        pary['losowe_inne_zdjecia'] = (im_, jm)
    wyn = dict(nazwa=nazwa, uklad=pole, W=W, rekordow=int(Tw), taus=[], auto=[], C={k: [] for k in pary})
    for tau in taus:
        if tau >= Tw - 10:
            continue
        pocz = np.unique(np.linspace(0, Tw - 1 - tau, n_pocz).astype(int))
        auto, cc = [], {k: [] for k in pary}
        for t in pocz:
            A, Bm = U[t + tau], U[t]
            auto.append(float(np.nanmean((A * Bm).sum(1) / war)))
            for k, (i, j) in pary.items():
                cc[k].append(float(np.nanmean((A[i] * Bm[j]).sum(1) / np.sqrt(war[i] * war[j]))))
        wyn['taus'].append(int(tau)); wyn['auto'].append(float(np.mean(auto)))
        for k in pary:
            wyn['C'][k].append(float(np.mean(cc[k])))
    if wyn['taus']:
        c6 = wyn['C']['sasiedzi']
        inne = wyn['C']['losowe_inne_zdjecia'][0] if 'losowe_inne_zdjecia' in wyn['C'] else None
        wyn['werdykt'] = dict(C_sasiedzi_0=c6[0], C_sasiedzi_min=float(min(c6)), tau_min=int(wyn['taus'][int(np.argmin(c6))]),
                              C_losowe_zdjecie_0=wyn['C']['losowe_zdjecie'][0], C_inne_zdjecia_0=inne)
        log(f"{nazwa} [{pole}] korelacja krzyżowa położeń: sąsiedzi C(0) {c6[0]:.3f}, min {min(c6):.3f} @τ={wyn['werdykt']['tau_min']}, "
            f"losowe w zdjęciu C(0) {wyn['C']['losowe_zdjecie'][0]:.3f}, inne zdjęcia C(0) {inne}; auto {[round(a, 2) for a in wyn['auto']]}")
    return wyn
