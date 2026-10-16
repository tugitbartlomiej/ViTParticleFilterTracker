"""MORI–ZWANZIG, STUDNIE I WIRY NA PUŁAPKACH KOLOIDÓW (2026-10-13).

Pułapka tokenu j: c_j(t) = średnia krocząca jego położenia z ±w checkpointów (jak w aplikacji), położenia względne
(bez średniej tokenów zdjęcia). Przestrzenie: x192 (strumień rezydualny po ostatnim bloku), z768 (latent 768 laserów),
rp128 (losowy rzut z eksportu ruch60). Bieg 35 000 (s47), 806 checkpointów; siatka G2 co 100 kroków (1000–65 600).

1. Uogólnione równanie Langevina (Mori–Zwanzig) dla chmury pułapek zdjęcia, we współrzędnych POD a(t) (r = 10, baza z części
   uczącej 70 %):  Δa(t) = M a(t) + b  [dryf]  + B u(t)  [wymuszenie = sterowanie: 10 składowych przyrostu wag]
                          + Σ_{s=1..L} K_s Δa(t − s)  [jądro pamięci]  + ε(t)  [szum = reszta].
   Hierarchia modeli M0 (nic), M1 dryf, M2 dryf + pamięć, M3 dryf + sterowanie, M4 dryf + sterowanie + pamięć; R² przyrostów
   na części testowej. Reszta ε modelu M4 (i M1): czy jest biała (autokorelacja, udział mocy w paśmie rytmu epoki 1094 kroków),
   czy jest wspólna dla różnych zdjęć (korelacja energii reszty |ε_i(t)|² między zdjęciami, pierwsza składowa po zdjęciach =
   „zgubiony sygnał”), i z czym koreluje: |przyrost wag|, część przyrostu wag poza 10 składowymi, strata uczenia (LOG.json),
   faza epoki, zmiana pewności i IoU zdjęcia (oceny eksportu ruch60).
2. Studnie (operator transferu na pułapkach, 600 tokenów jako niezależni wędrowcy): w oknach przesuwnych po G2 pułapki zdjęcia
   w jego PCA-5, k-średnich 20 stanów, macierz przejść przy opóźnieniu τ, czasy implikowane, liczba wolnych modów (stany
   metastabilne = studnie), trwałość przynależności; surogat: przyrosty przetasowane w czasie (ta sama permutacja dla tokenów).
3. Pole dryfu pułapek: w każdej chwili lokalne dopasowanie afiniczne v ≈ M (c − c_j) + v_j na 20 najbliższych sąsiadach w PCA-3
   (jak rzut aplikacji) i PCA-10; dywergencja tr M (ujścia < 0 = studnie, źródła > 0), wirowość ‖(M − Mᵀ)/2‖ / ‖(M + Mᵀ)/2‖,
   mapy na siatce 20 × 30, globalny obrót chmury (część antysymetryczna dopasowania całej chmury, stopnie na 100 kroków);
   wszystko także dla surogatu.
Uruchomienie jak koopman_pulapki.py (te same argumenty) plus --log LOG.json biegu.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

import koopman_pulapki as KP

t0 = time.time()


def log(*s):
    print(*s, f'[{time.time() - t0:.0f} s]', flush=True)


# ===================================================================================== 1. Mori–Zwanzig
def dopasuj(cechy, y, Ttr, ridge=1e-6):
    X = np.concatenate(cechy, 1)
    Xt, yt = X[:Ttr], y[:Ttr]
    G = Xt.T @ Xt
    G += ridge * np.trace(G) / len(G) * np.eye(len(G))
    W = np.linalg.solve(G, Xt.T @ yt)
    return X @ W


def mori_zwanzig(a, u, dt, L=4, u2=None):
    """a (T, r) współrzędne POD pułapek, u (T − 1, k) wejście dla kroku t → t + 1 (u2: drugie wejście, np. ostatni blok)."""
    T, r = a.shape
    da = np.diff(a, axis=0)                                               # (T − 1, r): przyrost t → t + 1
    t_ = np.arange(L, T - 1)                                              # chwile z pełną pamięcią
    y = da[t_]
    jeden = np.ones((len(t_), 1))
    dryf = [a[t_], jeden]
    pam = [da[t_ - s] for s in range(1, L + 1)]
    ster = [u[t_]]
    Ttr = int(len(t_) * 0.7)
    modele = {'M1_dryf': dryf, 'M2_dryf_pamiec': dryf + pam, 'M3_dryf_sterowanie': dryf + ster, 'M4_dryf_sterowanie_pamiec': dryf + ster + pam}
    if u2 is not None:
        modele['M3b_dryf_sterowanie_blok6'] = dryf + [u2[t_]]
        modele['M4b_dryf_sterowanie_blok6_pamiec'] = dryf + [u2[t_]] + pam
    sst = float(((y[Ttr:] - y[:Ttr].mean(0)) ** 2).sum())
    out = dict(R2_test={'M0_nic': float(1 - (y[Ttr:] ** 2).sum() / max(sst, 1e-30))})
    reszty = {}
    for n_, c_ in modele.items():
        yh = dopasuj(c_, y, Ttr)
        out['R2_test'][n_] = float(1 - ((y[Ttr:] - yh[Ttr:]) ** 2).sum() / max(sst, 1e-30))
        reszty[n_] = y - yh
    for n_ in ('M1_dryf', 'M4_dryf_sterowanie_pamiec'):
        e = reszty[n_]
        en = (e ** 2).sum(1)
        ac = [float((e[:-k] * e[k:]).sum() / (e ** 2).sum()) for k in range(1, 11)]
        P = np.abs(np.fft.rfft(e - e.mean(0), axis=0)) ** 2
        P = P.sum(1); f = np.fft.rfftfreq(len(e), d=dt)                    # cykle na krok uczenia
        fe = 1.0 / 1094
        pas = (np.abs(f - fe) < 0.15 * fe)
        sasiad = (np.abs(f - fe) < 0.6 * fe) & ~pas
        out[f'reszta_{n_}'] = dict(autokorelacja_1_10=ac, prog_bialego=float(2 / np.sqrt(len(e))),
                                  moc_epoki_do_sasiedztwa=(float(P[pas].mean() / max(P[sasiad].mean(), 1e-30)) if pas.any() and sasiad.any() else None),
                                  nachylenie_widma=float(np.polyfit(np.log(f[1:]), np.log(P[1:] + 1e-30), 1)[0]))
        reszty[n_ + '_energia'] = en
    return out, reszty, t_


# ===================================================================================== 2. studnie (operator transferu)
def msm(dtraj, n, lag):
    C = np.zeros((n, n))
    for d_ in dtraj:
        np.add.at(C, (d_[:-lag], d_[lag:]), 1)
    C = C + C.T + 1e-3                                                    # odwracalna estymacja (symetryzacja liczników)
    P = C / C.sum(1, keepdims=True)
    lam = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    return lam


def studnie(C, dt, n_st=20, lagi=(1, 2, 5), rng=None):
    """C (T, 600, d) pułapki w oknie → czasy implikowane, liczba wolnych modów, trwałość klastrów."""
    T, n, d = C.shape
    X = C.reshape(-1, d).astype(np.float64)
    mu = X.mean(0); U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Y = ((X - mu) @ Vt[:5].T).reshape(T, n, 5)
    cent, lab = kmeans2(Y.reshape(-1, 5), n_st, minit='++', seed=int(rng.integers(1 << 30)))
    lab = lab.reshape(T, n)
    dtraj = [lab[:, j] for j in range(n)]
    out = {}
    for L in lagi:
        if T <= L + 2:
            continue
        lam = msm(dtraj, n_st, L)[1:9]
        lam = np.clip(lam, 1e-9, 1 - 1e-9)
        ts = -L * dt / np.log(lam)
        # liczba studni = 1 + liczba wolnych modów przed największą przerwą widma (test syntetyczny: 2 studnie → przerwa 28×,
        # błądzenie i wir → ~1,1); przerwa < 3 oznacza brak wyraźnych studni
        ilorazy = ts[:-1] / np.maximum(ts[1:], 1e-9)
        g = int(np.argmax(ilorazy))
        out[L] = dict(czasy=[float(v) for v in ts], przerwa_1_2=float(ilorazy[0]), najwieksza_przerwa=float(ilorazy[g]),
                      liczba_studni=(g + 2 if ilorazy[g] >= 3 else 1))
    out['trwalosc_klastra'] = float((lab[1:] == lab[:-1]).mean())
    out['zmiany_na_token'] = float((lab[1:] != lab[:-1]).sum(0).mean())
    return out


# ===================================================================================== 3. pole dryfu: ujścia, źródła, wiry
def pole_dryfu(C, dims=(3, 10), k=20):
    T, n, d = C.shape
    X = C.reshape(-1, d).astype(np.float64)
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    out = {}
    for D in dims:
        Y = ((X - mu) @ Vt[:D].T).reshape(T, n, D)
        V = np.diff(Y, axis=0)                                            # prędkości pułapek
        div = np.zeros((T - 1, n)); rot = np.zeros((T - 1, n)); sym = np.zeros((T - 1, n))
        gl_rot, gl_div = [], []
        for t in range(T - 1):
            P_, V_ = Y[t], V[t]
            _, nb = cKDTree(P_).query(P_, k)
            dX = P_[nb] - P_[:, None, :]                                  # (n, k, D)
            dV = V_[nb] - V_[:, None, :]
            G = np.einsum('nki,nkj->nij', dX, dX) + 1e-9 * np.eye(D)
            M = np.linalg.solve(G, np.einsum('nki,nkj->nij', dX, dV)).transpose(0, 2, 1)   # dV ≈ M dX
            div[t] = np.trace(M, axis1=1, axis2=2)
            A_ = (M - M.transpose(0, 2, 1)) / 2; S_ = (M + M.transpose(0, 2, 1)) / 2
            rot[t] = np.linalg.norm(A_, axis=(1, 2)); sym[t] = np.linalg.norm(S_, axis=(1, 2))
            Pc = P_ - P_.mean(0); Vc = V_ - V_.mean(0)                     # globalne dopasowanie całej chmury
            Mg = np.linalg.lstsq(Pc, Vc, rcond=None)[0].T
            Ag = (Mg - Mg.T) / 2
            gl_rot.append(float(np.degrees(np.linalg.norm(Ag, 2)))); gl_div.append(float(np.trace(Mg)))
        h = (T - 1) // 2
        div_j = div.mean(0)
        trw = float(np.corrcoef(div[:h].mean(0), div[h:].mean(0))[0, 1]) if h >= 2 else None
        out[D] = dict(dywergencja_sr=float(div.mean()), dywergencja_ujscia_udzial=float((div_j < 0).mean()),
                      dywergencja_trwalosc_mapy=trw, wirowosc_do_scinania=float(np.median(rot / np.maximum(sym, 1e-30))),
                      obrot_globalny_stopnie_na_krok_siatki=float(np.median(gl_rot)), dywergencja_globalna=float(np.median(gl_div)),
                      mapa_dywergencji=div_j.astype(np.float32), mapa_wirowosci=rot.mean(0).astype(np.float32))
    return out


def surogat(X, rng):
    dX = np.diff(X, axis=0); dX = dX[rng.permutation(len(dX))]
    return np.concatenate([X[:1], X[0] + np.cumsum(dX, 0)]).astype(np.float32)


# ===================================================================================== główny przebieg
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wagi', required=True); ap.add_argument('--manifest', required=True); ap.add_argument('--zipy', required=True)
    ap.add_argument('--rp128', required=True); ap.add_argument('--out', required=True); ap.add_argument('--log', default=None)
    ap.add_argument('--n_192', type=int, default=25); ap.add_argument('--n_768', type=int, default=10); ap.add_argument('--n_128', type=int, default=25)
    ap.add_argument('--w', type=int, default=3)
    ap.add_argument('--okno', type=int, default=60, help='okno przesuwne studni/pola (punkty siatki G2 = ×100 kroków)')
    ap.add_argument('--krok_okna', type=int, default=30)
    ap.add_argument('--proba', action='store_true')
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    if a.proba:
        a.n_192 = a.n_768 = a.n_128 = 2
    rng = np.random.default_rng(0)
    TOK = 600
    meta = json.load(open(Path(a.rp128) / 'META.json', encoding='utf-8'))
    kroki = np.array(meta['kroki']); zdj = meta['zdjecia']; n = len(zdj) // 2

    # ---------------------------------------------------------------- forward (jak koopman_pulapki.py)
    import torch
    import dane as D
    import model as M
    from trening35000 import wczytaj_z_zipow, wiersze_35000

    def wybierz(m):
        return list(range(min(m, n))) + list(range(n, n + min(m, n)))
    idx192, idx768, idx128 = wybierz(a.n_192), wybierz(a.n_768), wybierz(a.n_128)
    potrzebne = sorted(set(idx192) | set(idx768))
    wt = wiersze_35000(a.manifest)
    wal_w, wal_X = D.walidacja()
    tr_rows = [wt[zdj[j]['pool_row']] for j in potrzebne if zdj[j]['split'] == 'TRAIN']
    Xtr = wczytaj_z_zipow(tr_rows, a.zipy)
    rows, X, it = [], [], 0
    for j in potrzebne:
        if zdj[j]['split'] == 'TRAIN':
            rows.append(tr_rows[it]); X.append(Xtr[it]); it += 1
        else:
            rows.append(wal_w[zdj[j]['val_row']]); X.append(wal_X[zdj[j]['val_row']])
    X = np.stack(X)
    pozycja = {j: i for i, j in enumerate(potrzebne)}
    kroki_pliki = {int(re.fullmatch(r'k([0-9]+)\.pt', p.name)[1]): p for p in Path(a.wagi).iterdir() if re.fullmatch(r'k([0-9]+)\.pt', p.name)}
    G2 = [i for i, k_ in enumerate(kroki) if 1000 <= k_ <= 65600 and k_ % 100 == 0]
    K = len(G2)
    net = M.MalyViT().to('cuda').eval(); net.zapisuj_latent(True)
    wyj = {}
    net.bloki[-1].register_forward_hook(lambda m, i, o: wyj.__setitem__('x', o))
    n_par = sum(p_.numel() for p_ in net.parameters())
    nazwy_par = [(nm_, p_.numel()) for nm_, p_ in net.named_parameters()]
    TH = np.zeros((K, n_par), np.float32)
    x192 = np.zeros((K, len(idx192), TOK, 192), np.float16); z768 = np.zeros((K, len(idx768), TOK, 768), np.float16)
    p192 = [pozycja[j] for j in idx192]; p768 = [pozycja[j] for j in idx768]
    Xg = torch.from_numpy(X).cuda()
    log(f'forward: {K} checkpointów siatki G2, {len(potrzebne)} zdjęć')
    for ik, k_ in enumerate(G2):
        st = torch.load(kroki_pliki[int(kroki[k_])], map_location='cpu', weights_only=False)
        net.load_state_dict(st['net'], strict=True)
        TH[ik] = torch.cat([p_.detach().float().reshape(-1) for p_ in net.parameters()]).cpu().numpy()
        xs, zs = [], []
        with torch.no_grad():
            for s in range(0, len(X), 50):
                x = Xg[s:s + 50].permute(0, 3, 1, 2).float() / 255.
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, lat = net(x)
                xs.append(wyj['x'].float().half().cpu()); zs.append(lat.float().half().cpu())
        xs = torch.cat(xs).numpy(); zs = torch.cat(zs).numpy()
        x192[ik] = xs[p192]; z768[ik] = zs[p768]
    del Xg, net
    kroki_g = kroki[G2]
    dt = 100.0
    rp = np.load(Path(a.rp128) / 'rp128.npy', mmap_mode='r')
    oceny = np.load(Path(a.rp128) / 'oceny.npy')[G2]                    # (K, 100, 2) pewność, IoU

    # ---------------------------------------------------------------- sygnały globalne na krokach pułapki (t → t + 1, okno ±w)
    w = a.w
    Tc = K - 2 * w
    dTH = (TH[2 * w + 1:] - TH[:Tc - 1]) / (2 * w + 1)                  # (Tc − 1, P): przyrost wag dla kroku pułapki
    Ttr_u = int(Tc * 0.7)
    Dt = dTH[:Ttr_u].astype(np.float64)
    ev, V = np.linalg.eigh(Dt @ Dt.T); ev, V = ev[::-1], V[:, ::-1]
    baza = (Dt.T @ V[:, :10]) / np.sqrt(ev[:10])
    U10 = dTH.astype(np.float64) @ baza                                 # (Tc − 1, 10)
    # drugie wejście: 10 składowych przyrostu wag samego ostatniego bloku (bloki.5.*)
    sl6 = np.zeros(TH.shape[1], bool); o_ = 0
    for nm_, n_el in nazwy_par:
        if nm_.startswith('bloki.5.'):
            sl6[o_:o_ + n_el] = True
        o_ += n_el
    D6 = dTH[:, sl6].astype(np.float64)
    ev6, V6 = np.linalg.eigh(D6[:Ttr_u] @ D6[:Ttr_u].T); ev6, V6 = ev6[::-1], V6[:, ::-1]
    baza6 = (D6[:Ttr_u].T @ V6[:, :10]) / np.sqrt(ev6[:10])
    U10_b6 = D6 @ baza6
    del D6
    dth_norm = np.linalg.norm(dTH, axis=1)
    dth_poza = np.linalg.norm(dTH - (U10 @ baza.T), axis=1)              # część przyrostu wag poza 10 składowymi
    del Dt
    kroki_c = kroki_g[w:K - w]                                           # chwile pułapek
    sygnaly = dict(norma_przyrostu_wag=dth_norm, przyrost_wag_poza_10=dth_poza,
                   faza_epoki_sin=np.sin(2 * np.pi * kroki_c[:-1] / 1094), faza_epoki_cos=np.cos(2 * np.pi * kroki_c[:-1] / 1094))
    if a.log and Path(a.log).exists():
        L_ = json.load(open(a.log, encoding='utf-8'))
        lk = np.array([r_['krok'] for r_ in L_]); ls_ = np.array([r_['strata'] for r_ in L_])
        sl = np.interp(kroki_g, lk, ls_)
        sc = KP.srednia_kroczaca(sl[:, None], w)[:, 0]
        sygnaly['strata'] = np.diff(sc); sygnaly['strata_poziom'] = sc[:-1]
    WYN = dict(opis=__doc__, w=w, okno=a.okno, kroki_pulapek=[int(v) for v in kroki_c], udzial_wariancji_U10=float(ev[:10].sum() / ev.sum()),
               udzial_wariancji_U10_blok6=float(ev6[:10].sum() / ev6.sum()),
               przestrzenie={})
    MAPY = {}

    przestrzenie = {
        'x192': (idx192, lambda j: x192[:, idx192.index(j)].astype(np.float32)),
        'z768': (idx768, lambda j: z768[:, idx768.index(j)].astype(np.float32)),
        'rp128': (idx128, lambda j: np.asarray(rp[np.asarray(G2), j], np.float32)),
    }
    okna = list(range(0, Tc - a.okno + 1, a.krok_okna)) or [0]
    for pn, (idx, pobierz) in przestrzenie.items():
        per = {'TRAIN': [], 'VALID': []}
        energie, nazwy = [], []
        for j in idx:
            Xr = pobierz(j); Xr = Xr - Xr.mean(1, keepdims=True)
            C = KP.srednia_kroczaca(Xr, w)                                # (Tc, 600, d) pułapki
            Cs = KP.srednia_kroczaca(surogat(Xr, rng), w)
            wyn = {}
            # 1. Mori–Zwanzig na całej siatce i w reżimach
            Z = C.reshape(Tc, -1).astype(np.float64); Ttr = int(Tc * 0.7)
            Z = Z - Z[:Ttr].mean(0)
            _, aa, _, _ = KP.pod(Z, Ttr, 10)
            mz, reszty, t_ = mori_zwanzig(aa, U10, dt, u2=U10_b6)
            # surogat: te same przyrosty położeń w innej kolejności, ta sama średnia krocząca → ile „pamięci” daje sama konstrukcja pułapki
            Zs = Cs.reshape(Tc, -1).astype(np.float64); Zs = Zs - Zs[:Ttr].mean(0)
            _, aas, _, _ = KP.pod(Zs, Ttr, 10)
            mzs, _, _ = mori_zwanzig(aas, U10, dt, u2=U10_b6)
            mz['surogat_R2_test'] = mzs['R2_test']
            mz['surogat_reszta_M4'] = mzs.get('reszta_M4_dryf_sterowanie_pamiec')
            # korelacje energii reszty z sygnałami i ze zmianą pewności/IoU zdjęcia
            e4 = reszty['M4_dryf_sterowanie_pamiec_energia']
            ocj = KP.srednia_kroczaca(oceny[:, j, :].astype(np.float64), w)
            dpew, diou = np.diff(ocj[:, 0]), np.diff(ocj[:, 1])
            kor = {}
            for sn, sv in list(sygnaly.items()) + [('zmiana_pewnosci', dpew), ('zmiana_IoU', diou), ('abs_zmiana_IoU', np.abs(diou))]:
                v_ = np.asarray(sv)[t_]
                if np.std(v_) > 0:
                    kor[sn] = float(np.corrcoef(e4, v_)[0, 1])
            mz['korelacje_energii_reszty_M4'] = kor
            wyn['MZ'] = mz
            energie.append(e4 / e4.mean()); nazwy.append(j)
            # 2–3. okna przesuwne: studnie i pole dryfu (dane i surogat)
            ok, ok_s = [], []
            for o in okna:
                sl_ = slice(o, o + a.okno)
                st_ = studnie(C[sl_], dt, rng=rng); pd_ = pole_dryfu(C[sl_])
                ss_ = studnie(Cs[sl_], dt, rng=rng); ps_ = pole_dryfu(Cs[sl_])
                if j in (idx[0], idx[len(idx) // 2]):
                    for D_ in pd_:
                        MAPY[f'{pn}_{j}_okno{o}_D{D_}_dyw'] = pd_[D_]['mapa_dywergencji']; MAPY[f'{pn}_{j}_okno{o}_D{D_}_wir'] = pd_[D_]['mapa_wirowosci']
                for d_ in (pd_, ps_):
                    for D_ in d_:
                        d_[D_].pop('mapa_dywergencji'); d_[D_].pop('mapa_wirowosci')
                ok.append(dict(krok_od=int(kroki_c[o]), krok_do=int(kroki_c[min(o + a.okno, Tc) - 1]), studnie=st_, pole=pd_))
                ok_s.append(dict(studnie=ss_, pole=ps_))
            wyn['okna'] = ok; wyn['okna_surogat'] = ok_s
            per[zdj[j]['split']].append(wyn)
            log(f'{pn}: zdjęcie {j} ({zdj[j]["split"]}) gotowe')
        E = np.stack(energie)                                             # (zdjęć, czas): energia reszty M4
        Ec = E - E.mean(1, keepdims=True)
        Cm = np.corrcoef(E); iu = np.triu_indices(len(E), 1)
        u_, s_, vt_ = np.linalg.svd(Ec, full_matrices=False)
        wspolna = vt_[0] * np.sign(u_[:, 0].mean())
        kor_w = {sn: float(np.corrcoef(wspolna, np.asarray(sv)[t_])[0, 1]) for sn, sv in sygnaly.items() if np.std(np.asarray(sv)[t_]) > 0}
        WYN['przestrzenie'][pn] = dict(
            TRAIN=dict(zdjec=len(per['TRAIN']), mediany=KP.mediana_drzewa(per['TRAIN'])),
            VALID=dict(zdjec=len(per['VALID']), mediany=KP.mediana_drzewa(per['VALID'])),
            reszta_wspolna=dict(korelacja_energii_miedzy_zdjeciami_med=float(np.median(Cm[iu])),
                                udzial_pierwszej_skladowej=float(s_[0] ** 2 / (s_ ** 2).sum()),
                                pierwsza_skladowa_korelacje=kor_w, pierwsza_skladowa=wspolna.astype(float).tolist(),
                                kroki=[int(v) for v in kroki_c[t_]]))
        json.dump(WYN, open(OUT / 'MZ_STUDNIE.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    np.savez_compressed(OUT / 'MAPY_POLA.npz', **MAPY)
    log('zapisano', OUT)


if __name__ == '__main__':
    main()
