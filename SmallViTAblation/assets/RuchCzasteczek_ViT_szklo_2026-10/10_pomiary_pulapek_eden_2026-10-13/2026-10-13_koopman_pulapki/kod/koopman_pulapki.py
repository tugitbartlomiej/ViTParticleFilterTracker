"""KOOPMAN NA PUŁAPKACH KOLOIDÓW (2026-10-13): czy ruch pułapek tokenów w czasie uczenia jest prostą dynamiką liniową
(kilka modów Koopmana o określonych czasach), czy błądzeniem, którego nic nie przewiduje lepiej niż „stoi w miejscu”?

Pułapka tokenu (jak w aplikacji, zakładka „Lasery i koloidy”): średnia krocząca jego położenia z ±w checkpointów,
c_j(t) = ⟨x_j⟩_[t−w, t+w]; drgania v_j = x_j − c_j. Położenia względne: x_j − średnia tokenów zdjęcia (bez modu zdjęcia).
Trzy przestrzenie tego samego biegu 35 000 (s47, 806 checkpointów, 60 epok):
  rp128 — losowy rzut 768 → 128 (plik eksportu ruch60, 100 zdjęć: 50 TRAIN + 50 VALID),
  z768  — pełny latent (GELU MLP ostatniego bloku, 768 „laserów”), 20 zdjęć (10 TRAIN + 10 VALID), liczony tu,
  x192  — stan tokenu w strumieniu rezydualnym po ostatnim bloku (192D, wejście końcowej LayerNorm), 100 zdjęć, liczony tu.
Siatki czasu: G1 = kroki 0–1000 co 10 (w = 3 → ±30 kroków), G2 = kroki 1000–65 600 co 100 (w = 3 → ±300 kroków, jak aplikacja);
reżimy G2: R1 epoki ~1–5 (kroki 1000–5470), R2 ~5–20 (5470–21 880), R3 ~20–60 (21 880–65 600).

Dla pułapek (i dla porównania drgań) w każdym reżimie i zdjęciu:
  A. wspólny operator jednej cząstki (d × d, ten sam dla 600 tokenów): c_j(t+1) − μ_j ≈ A (c_j(t) − μ_j), dopasowany na pierwszych
     70 % reżimu, prognoza h = 1, 5, 20 kroków siatki na ostatnich 30 %; błąd względem prognozy „stoi” (c(t+h) = c(t)) i „dryf”
     (c(t) + h·(c(t) − c(t−1))); widmo A: czasy zaniku (w krokach uczenia), udział par zespolonych, kąt obrotu na 100 kroków;
  B. DMD całej chmury pułapek zdjęcia (stan 600·d, baza POD r = 20 z części uczącej): to samo porównanie prognoz, widmo,
     mapy modów na siatce 20 × 30 (gładkość: korelacja energii modu sąsiednich patchy; udział energii na tokenach narzędzia
     wobec udziału tych tokenów), surogat: przyrosty współrzędnych POD przetasowane w czasie (błądzenie o tych samych krokach);
  C. test Markowa VAMP na współrzędnych POD: czasy implikowane w funkcji opóźnienia (płaskie = Markow w tych współrzędnych);
  D. DMD ze sterowaniem (DMDc): a(t+1) = A a(t) + B u(t), gdzie u to k = 3 albo 10 głównych składowych przyrostu WAG sieci
     (ten sam napęd dla wszystkich tokenów i zdjęć; przyrost uśredniony tym samym oknem co pułapka, więc u(t) odpowiada
     dokładnie krokowi c(t) → c(t+1)); baza składowych z części uczącej, prognoza na teście ze znanym wejściem;
     kontrole: to samo u przetasowane w czasie i losowe wejście Gaussa. Pełny przyrost wag dawałby tautologię δz ≈ J δθ,
     więc pytanie brzmi, czy KILKA wspólnych kierunków w ~2,9 mln wag steruje całą chmurą pułapek;
  E. surogat błądzenia: przyrosty położeń przetasowane w czasie (ta sama permutacja dla wszystkich tokenów), z nich pułapki
     i drgania i te same prognozy A i B (r = 5) — poziom „przewidywalności” wynikający z samej konstrukcji (średnia krocząca).
Wynik: KOOPMAN_PULAPKI.json (mediany po zdjęciach, osobno TRAIN i VALID) i MODY.npz (mapy modów kilku zdjęć).
Uruchomienie (Eden, GPU, katalog kodu biegu 35 000 na PYTHONPATH):
  python koopman_pulapki.py --wagi .../pelny/wagi --manifest .../MANIFEST_35000.jsonl --zipy ... --rp128 .../ruch60.../ --out ...
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

t0 = time.time()


def log(*s):
    print(*s, f'[{time.time() - t0:.0f} s]', flush=True)


# ===================================================================================== narzędzia Koopmana
def srednia_kroczaca(Y, w):
    """Y (T, ...) → (T − 2w, ...) średnia z okna 2w + 1 (tylko pełne okna)."""
    c = np.cumsum(Y, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros((1,) + Y.shape[1:]), c])
    return ((c[2 * w + 1:] - c[:-2 * w - 1]) / (2 * w + 1)).astype(np.float32)


def widmo(A, dt):
    lam = np.linalg.eigvals(A.astype(np.float64))
    mod = np.abs(lam)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.where(mod < 1, -dt / np.log(np.clip(mod, 1e-12, 1 - 1e-12)), np.inf)
    kat = np.degrees(np.abs(np.angle(lam))) * (100.0 / dt)                 # stopnie na 100 kroków uczenia
    return dict(mod_med=float(np.median(mod)), mod_max=float(mod.max()), ponad_1=float((mod > 1.0001).mean()),
                zespolone=float((np.abs(lam.imag) > 1e-9).mean()), tau_med=float(np.median(tau[np.isfinite(tau)])) if np.isfinite(tau).any() else None,
                tau_10_najdluzszych=[float(v) for v in np.sort(tau[np.isfinite(tau)])[::-1][:10]], kat_med_na_100=float(np.median(kat)),
                kat_max_na_100=float(kat.max())), lam


def bledy_prognoz(seq, prognoza, hs, start):
    """seq (T, ...) całość; prognoza(t, h) → przewidywany seq[t + h] dla t ≥ start; zwraca stosunki do „stoi” i „dryf”."""
    out = {}
    T = len(seq)
    for h in hs:
        ts = [t for t in range(max(start, 1), T - h)]
        if len(ts) < 3:
            continue
        e_m = e_s = e_d = 0.0
        for t in ts:
            y = seq[t + h]
            e_m += float(((y - prognoza(t, h)) ** 2).sum())
            e_s += float(((y - seq[t]) ** 2).sum())
            e_d += float(((y - (seq[t] + h * (seq[t] - seq[t - 1]))) ** 2).sum())
        out[h] = dict(do_stoi=e_m / max(e_s, 1e-30), do_dryfu=e_m / max(e_d, 1e-30), dryf_do_stoi=e_d / max(e_s, 1e-30))
    return out


def operator_wspolny(C, dt, hs=(1, 5, 20), frac=0.7, ridge=1e-6):
    """C (T, n, d): trajektorie n cząstek; jeden operator d × d dla wszystkich."""
    T, n, d = C.shape
    Ttr = int(T * frac)
    if Ttr < 4:
        return None
    mu = C[:Ttr].mean(0, keepdims=True)
    X = (C - mu).astype(np.float64)
    a, b = X[:Ttr - 1].reshape(-1, d), X[1:Ttr].reshape(-1, d)
    G = a.T @ a
    G += ridge * np.trace(G) / d * np.eye(d)
    A = np.linalg.solve(G, a.T @ b).T                                     # b ≈ a Aᵀ
    potegi = {}

    def prog(t, h):
        if h not in potegi:
            potegi[h] = np.linalg.matrix_power(A, h)
        return X[t] @ potegi[h].T

    wsp, _ = widmo(A, dt)
    return dict(prognozy=bledy_prognoz(X, prog, hs, Ttr), widmo=wsp, T=T, uczace=Ttr)


def pod(Z, Ttr, r):
    """Z (T, D) wyśrodkowane częścią uczącą → baza POD (D, r) z Grama części uczącej, współrzędne (T, r), udział energii."""
    Zt = Z[:Ttr]
    G = Zt @ Zt.T
    ev, V = np.linalg.eigh(G)
    ev, V = ev[::-1], V[:, ::-1]
    r = int(min(r, (ev > ev[0] * 1e-10).sum()))
    s = np.sqrt(np.maximum(ev[:r], 1e-30))
    Phi = (Zt.T @ V[:, :r]) / s                                           # (D, r) ortonormalna
    a = Z @ Phi
    energia_test = float((a[Ttr:] ** 2).sum() / max((Z[Ttr:] ** 2).sum(), 1e-30)) if len(Z) > Ttr else None
    return Phi, a, float(ev[:r].sum() / ev.sum()), energia_test


def dmd(a, Ttr, ridge=1e-8):
    x, y = a[:Ttr - 1].astype(np.float64), a[1:Ttr].astype(np.float64)
    G = x.T @ x
    G += ridge * np.trace(G) / len(G) * np.eye(len(G))
    return np.linalg.solve(G, x.T @ y).T


def vamp_czasy(a, lagi, dt, k=8):
    out = {}
    for L in lagi:
        if len(a) <= L + 5:
            continue
        x, y = a[:-L] - a[:-L].mean(0), a[L:] - a[L:].mean(0)
        n = len(x)
        C00, C11, C01 = x.T @ x / n, y.T @ y / n, x.T @ y / n

        def isq(C):
            e, V = np.linalg.eigh(C)
            e = np.maximum(e, e.max() * 1e-10)
            return V @ np.diag(e ** -0.5) @ V.T
        sv = np.linalg.svd(isq(C00) @ C01 @ isq(C11), compute_uv=False)[:k]
        sv = np.clip(sv, 1e-9, 1 - 1e-9)
        out[L] = dict(czasy=[float(-L * dt / np.log(s)) for s in sv], sv=[float(s) for s in sv])
    return out


def mapy_modow(Phi, A, n_tok, d, maska, sasiedzi, ile=3):
    """Mody DMD w pełnej przestrzeni: energia na token (600), gładkość na siatce i wzbogacenie na narzędziu dla najwolniejszych."""
    lam, W = np.linalg.eig(A)
    kol = np.argsort(-np.abs(lam))[:ile]
    wyn, mapy = [], []
    for k in kol:
        m = (Phi @ W[:, k]).reshape(n_tok, d)
        e = (np.abs(m) ** 2).sum(1); e = e / e.sum()
        i, j = sasiedzi
        gl = float(np.corrcoef(e[i], e[j])[0, 1])
        wz = float(e[maska].sum() / max(maska.mean(), 1e-9)) if maska is not None and maska.any() else None
        wyn.append(dict(mod_lambda=float(np.abs(lam[k])), kat=float(np.degrees(abs(np.angle(lam[k])))), gladkosc_sasiedzi=gl, wzbogacenie_narzedzie=wz,
                        udzial_10proc_tokenow=float(np.sort(e)[::-1][:n_tok // 10].sum())))
        mapy.append(e.astype(np.float32))
    return wyn, np.stack(mapy)


def sasiedzi_siatki(h=20, w=30):
    idx = np.arange(h * w).reshape(h, w)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    return a, b


def dmdc(a, u, Ttr, ridge=1e-8):
    x, y, uu = a[:Ttr - 1].astype(np.float64), a[1:Ttr].astype(np.float64), u[:Ttr - 1].astype(np.float64)
    O = np.concatenate([x, uu], 1)
    G = O.T @ O
    G += ridge * np.trace(G) / len(G) * np.eye(len(G))
    M_ = np.linalg.solve(G, O.T @ y)
    r = a.shape[1]
    return M_[:r].T, M_[r:].T                                             # A (r × r), B (r × k)


def prognozy_dmdc(Z, Ttr, wejscia, hs, r=10):
    Phi, a, _, _ = pod(Z, Ttr, min(r, Ttr - 2))
    out = {}
    for nazwa, u in wejscia.items():
        A, B = dmdc(a, u, Ttr)

        def prog(t, h, A=A, B=B, a=a, u=u):
            x = a[t].copy()
            for s_ in range(h):
                x = A @ x + B @ u[t + s_]
            return Z[t] + (x - a[t]) @ Phi.T
        out[nazwa] = dict(prognozy=bledy_prognoz(Z, prog, hs, Ttr), widmo_A=widmo(A, 1.0)[0]['mod_max'])
    return out


def analiza_trajektorii(Xrel, dt, w, maska, sas, r=20, hs=(1, 5, 20), n_sur=3, rng=None, z_drganiami=True, wejscia=None, surogat=True):
    """Xrel (T, 600, d) położenia względne w jednym reżimie → wyniki A, B, C (i D z wejściami) dla pułapek, A, B dla drgań, E surogat."""
    out = {}
    if surogat:                                                           # E: błądzenie z tymi samymi przyrostami w innej kolejności
        dX = np.diff(Xrel, axis=0); dX = dX[rng.permutation(len(dX))]
        Xs = np.concatenate([Xrel[:1], Xrel[0] + np.cumsum(dX, 0)]).astype(np.float32)
        Cs = srednia_kroczaca(Xs, w); Vs = Xs[w:len(Xs) - w] - Cs
        out['surogat'] = {}
        for nazwa, S in (('pulapki', Cs), ('drgania', Vs)):
            T_ = len(S); Ttr_ = int(T_ * 0.7)
            e = dict(A_operator=(operator_wspolny(S, dt, hs) or {}).get('prognozy') if T_ >= 8 else None)
            if Ttr_ >= 6:
                Z_ = S.reshape(T_, -1).astype(np.float64); Z_ = Z_ - Z_[:Ttr_].mean(0)
                Phi_, a_, _, _ = pod(Z_, Ttr_, min(5, Ttr_ - 2)); A_ = dmd(a_, Ttr_); pot_ = {}

                def prog_s(t, h, A=A_, a=a_, Phi=Phi_, Z=Z_, pot=pot_):
                    if h not in pot:
                        pot[h] = np.linalg.matrix_power(A, h)
                    return Z[t] + (a[t] @ pot[h].T - a[t]) @ Phi.T
                e['B5'] = bledy_prognoz(Z_, prog_s, hs, Ttr_)
            out['surogat'][nazwa] = e
    C = srednia_kroczaca(Xrel, w)
    V = Xrel[w:len(Xrel) - w] - C
    zestawy = [('pulapki', C)] + ([('drgania', V)] if z_drganiami else [])
    mapy = None
    for nazwa, S in zestawy:
        T, n, d = S.shape
        res = dict(T=T)
        res['A_operator'] = operator_wspolny(S, dt, hs) if T >= 8 else None
        Ttr = int(T * 0.7)
        if Ttr >= 6:
            Z = S.reshape(T, -1).astype(np.float64)
            Z = Z - Z[:Ttr].mean(0)
            # rozmiar bazy POD: kontrola dodatnia (3 wspólne oscylatory) — r = 20 łapie własny szum tokenów, więc zapisujemy
            # prognozy dla kilku r bez wybierania najlepszego; widmo, surogat, VAMP i mapy dla r = 10
            res['B_dmd_r'] = {}
            for rr in (5, 20):
                Phi_, a_, e1_, e2_ = pod(Z, Ttr, min(rr, Ttr - 2))
                A_ = dmd(a_, Ttr); pot_ = {}

                def prog_(t, h, A=A_, a=a_, Phi=Phi_, Z=Z, pot=pot_):
                    if h not in pot:
                        pot[h] = np.linalg.matrix_power(A, h)
                    return Z[t] + (a[t] @ pot[h].T - a[t]) @ Phi.T
                res['B_dmd_r'][rr] = dict(energia_bazy_ucz=e1_, energia_bazy_test=e2_, widmo=widmo(A_, dt)[0], prognozy=bledy_prognoz(Z, prog_, hs, Ttr))
            Phi, a, en_tr, en_te = pod(Z, Ttr, min(10, Ttr - 2))
            A = dmd(a, Ttr)
            pot = {}

            def prog(t, h, A=A, a=a, Phi=Phi, Z=Z):
                # prognoza w PEŁNEJ przestrzeni: model zmienia tylko część w bazie POD, reszta stoi (próba 2026-10-13:
                # porównanie w samej bazie faworyzowało „stoi”, bo dane testowe wychodzą poza bazę uczącą)
                if h not in pot:
                    pot[h] = np.linalg.matrix_power(A, h)
                return Z[t] + (a[t] @ pot[h].T - a[t]) @ Phi.T
            wsp, _ = widmo(A, dt)
            res['B_dmd'] = dict(r=int(a.shape[1]), energia_bazy_ucz=en_tr, energia_bazy_test=en_te, widmo=wsp, prognozy=bledy_prognoz(Z, prog, hs, Ttr))
            sur = []
            for _ in range(n_sur):
                da = np.diff(a[:Ttr], axis=0); da = da[rng.permutation(len(da))]
                a_s = np.concatenate([a[:1], a[0] + np.cumsum(da, 0)])
                A_s = dmd(a_s, len(a_s))
                ws_, _ = widmo(A_s, dt)
                sur.append(ws_)
            res['B_surogat'] = dict(mod_med=float(np.median([s_['mod_med'] for s_ in sur])), zespolone=float(np.median([s_['zespolone'] for s_ in sur])),
                                    tau_med=float(np.median([s_['tau_med'] for s_ in sur if s_['tau_med']])) if any(s_['tau_med'] for s_ in sur) else None,
                                    kat_med_na_100=float(np.median([s_['kat_med_na_100'] for s_ in sur])))
            res['C_vamp'] = vamp_czasy(a, [1, 2, 4, 8, 16], dt)
            if nazwa == 'pulapki':
                res['B_mody'], mapy = mapy_modow(Phi, A, n, d, maska, sas)
                if wejscia is not None:
                    res['D_dmdc'] = {f'r{rr}': prognozy_dmdc(Z, Ttr, wejscia, hs, r=rr) for rr in (5, 10)}
        out[nazwa] = res
    return out, mapy


# ===================================================================================== agregacja
def mediana_drzewa(lista):
    """Mediany po zdjęciach dla zagnieżdżonych słowników liczb (listy liczb: mediana elementowa)."""
    lista = [x for x in lista if x is not None]
    if not lista:
        return None
    p = lista[0]
    if isinstance(p, dict):
        klucze = set().union(*[x.keys() for x in lista if isinstance(x, dict)])
        return {k: mediana_drzewa([x.get(k) for x in lista if isinstance(x, dict)]) for k in sorted(klucze, key=str)}
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        v = [float(x) for x in lista if x is not None and np.isfinite(x)]
        return float(np.median(v)) if v else None
    if isinstance(p, list) and p and isinstance(p[0], (int, float)):
        n = min(len(x) for x in lista)
        return [float(np.median([x[i] for x in lista])) for i in range(n)] if n else []
    if isinstance(p, list) and p and isinstance(p[0], dict):
        n = min(len(x) for x in lista)
        return [mediana_drzewa([x[i] for x in lista]) for i in range(n)]
    return None


# ===================================================================================== główny przebieg
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wagi', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--zipy', required=True)
    ap.add_argument('--rp128', required=True, help='katalog eksportu ruch60_rp128_50tr_50val (rp128.npy, META.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_192', type=int, default=50, help='zdjęć 192D na część (TRAIN, VALID)')
    ap.add_argument('--n_768', type=int, default=10, help='zdjęć 768D na część')
    ap.add_argument('--n_128', type=int, default=50)
    ap.add_argument('--w', type=int, default=3)
    ap.add_argument('--proba', action='store_true', help='2+2 zdjęcia')
    a = ap.parse_args()
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    TOK = 600
    meta = json.load(open(Path(a.rp128) / 'META.json', encoding='utf-8'))
    kroki = np.array(meta['kroki'])
    zdj = meta['zdjecia']
    n = len(zdj) // 2
    if a.proba:
        a.n_192 = a.n_768 = a.n_128 = 2

    # ---------------------------------------------------------------- forward po 806 checkpointach: x192 i z768 (do RAM)
    import torch
    import dane as D
    import model as M
    from trening35000 import wczytaj_z_zipow, wiersze_35000

    def wybierz(m):
        return list(range(min(m, n))) + list(range(n, n + min(m, n)))
    idx192, idx768 = wybierz(a.n_192), wybierz(a.n_768)
    potrzebne = sorted(set(idx192) | set(idx768))
    wt = wiersze_35000(a.manifest)
    wal_w, wal_X = D.walidacja()
    rows, X = [], []
    tr_rows = [wt[zdj[j]['pool_row']] for j in potrzebne if zdj[j]['split'] == 'TRAIN']
    Xtr = wczytaj_z_zipow(tr_rows, a.zipy) if tr_rows else np.zeros((0, 320, 480, 3), np.uint8)
    it = 0
    for j in potrzebne:
        if zdj[j]['split'] == 'TRAIN':
            rows.append(tr_rows[it]); X.append(Xtr[it]); it += 1
        else:
            rows.append(wal_w[zdj[j]['val_row']]); X.append(wal_X[zdj[j]['val_row']])
    X = np.stack(X)
    for j, r_ in zip(potrzebne, rows):
        assert r_['frame_id'] == zdj[j]['frame_id'], (j, r_['frame_id'], zdj[j]['frame_id'])
    pozycja = {j: i for i, j in enumerate(potrzebne)}
    # maski narzędzia (środek patcha w ramce), 600 tokenów
    yy, xx = np.mgrid[0:20, 0:30]
    cx, cy = (xx.ravel() + .5) * 16, (yy.ravel() + .5) * 16
    maski = {}
    for j, r_ in zip(potrzebne, rows):
        m_ = np.zeros(TOK, bool)
        for b in r_['ramki']:
            m_ |= (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
        maski[j] = m_
    kroki_pliki = {int(re.fullmatch(r'k([0-9]+)\.pt', p.name)[1]): p for p in Path(a.wagi).iterdir() if re.fullmatch(r'k([0-9]+)\.pt', p.name)}
    kk = list(range(len(kroki)))                                        # próba: wszystkie checkpointy, mniej zdjęć
    K = len(kk)
    dev = 'cuda'
    net = M.MalyViT().to(dev).eval()
    net.zapisuj_latent(True)
    wyj = {}
    net.bloki[-1].register_forward_hook(lambda m, i, o: wyj.__setitem__('x', o))
    Xg = torch.from_numpy(X).to(dev)
    n_par = sum(p_.numel() for p_ in net.parameters())
    TH = np.zeros((K, n_par), np.float32)                               # wektory wag każdego checkpointu (napęd do DMDc)
    x192 = np.zeros((K, len(idx192), TOK, 192), np.float16)
    z768 = np.zeros((K, len(idx768), TOK, 768), np.float16)
    p192 = [pozycja[j] for j in idx192]; p768 = [pozycja[j] for j in idx768]
    log(f'forward: {K} checkpointów, {len(potrzebne)} zdjęć (192D: {len(idx192)}, 768D: {len(idx768)})')
    for ik, k_ in enumerate(kk):
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
        if ik % 100 == 0:
            log(f'  checkpoint {ik + 1}/{K} (krok {int(kroki[k_])})')
    del Xg, net
    kroki_k = kroki[kk]

    # ---------------------------------------------------------------- siatki czasu i reżimy
    def siatka(od, do, co):
        return [i for i, k_ in enumerate(kroki_k) if od <= k_ <= do and k_ % co == 0]
    G1 = siatka(0, 1000, 10)
    G2 = siatka(1000, 65600, 100)
    dt1 = float(np.median(np.diff(kroki_k[G1]))); dt2 = float(np.median(np.diff(kroki_k[G2])))
    assert np.all(np.diff(kroki_k[G1]) == dt1) and np.all(np.diff(kroki_k[G2]) == dt2), 'siatka nierówna'
    rezimy = {'G1_0_1000': (G1, dt1)}
    for nazwa, od, do in (('R1_1000_5470', 1000, 5470), ('R2_5470_21880', 5470, 21880), ('R3_21880_65600', 21880, 65600)):
        rezimy[nazwa] = ([i for i in G2 if od <= kroki_k[i] <= do], dt2)
    log('reżimy: ' + ', '.join(f'{k_}: {len(v[0])} punktów co {v[1]:.0f} kroków' for k_, v in rezimy.items()))

    rp = np.load(Path(a.rp128) / 'rp128.npy', mmap_mode='r')
    idx128 = wybierz(a.n_128)
    przestrzenie = {
        'rp128': (idx128, lambda j, ii: np.asarray(rp[np.asarray(kk)[ii], j], np.float32)),   # ii: indeksy w kk
        'z768': (idx768, lambda j, ii: z768[ii, idx768.index(j)].astype(np.float32)),
        'x192': (idx192, lambda j, ii: x192[ii, idx192.index(j)].astype(np.float32)),
    }
    def wejscia_rezimu(ii, w, k_lista=(3, 10)):
        """u(t) dla kroku pułapki c(t) → c(t+1): (θ[t + 2w + 1] − θ[t]) / (2w + 1) w indeksach reżimu; składowe główne z części uczącej."""
        ii = np.asarray(ii)
        Tc = len(ii) - 2 * w
        Ttr = int(Tc * 0.7)
        D_ = (TH[ii[2 * w + 1:]] - TH[ii[:Tc - 1]]) / (2 * w + 1)        # (Tc − 1, P)
        Dt = D_[:Ttr - 1].astype(np.float64)
        G = Dt @ Dt.T
        ev, V = np.linalg.eigh(G); ev, V = ev[::-1], V[:, ::-1]
        out, wyj = {}, {}
        for k in k_lista:
            k_ = int(min(k, (ev > ev[0] * 1e-10).sum()))
            baza = (Dt.T @ V[:, :k_]) / np.sqrt(ev[:k_])                   # (P, k)
            u = (D_.astype(np.float64) @ baza)
            out[f'wagi_k{k}'] = u
            wyj[f'udzial_wariancji_k{k}'] = float(ev[:k_].sum() / ev.sum())
        u10 = out[f'wagi_k{k_lista[-1]}']
        out['tasowane_k10'] = u10[np.random.default_rng(1).permutation(len(u10))]
        out['losowe_k10'] = np.random.default_rng(2).normal(size=u10.shape) * u10.std(0)
        return out, wyj

    sas = sasiedzi_siatki()
    WYN = dict(opis=__doc__, w=a.w, kroki_siatek={k_: [int(kroki_k[i]) for i in v[0]] for k_, v in rezimy.items()}, przestrzenie={})
    MODY = {}
    for pn, (idx, pobierz) in przestrzenie.items():
        per = {rz: {'TRAIN': [], 'VALID': []} for rz in rezimy}
        if pn == list(przestrzenie)[0]:
            WEJ = {}
            for rz, (ii, dt) in rezimy.items():
                if len(ii) >= 2 * a.w + 8:
                    WEJ[rz], WYN.setdefault('wejscia_udzial_wariancji', {})[rz] = wejscia_rezimu(ii, a.w)
            log('wejścia DMDc: udział wariancji przyrostu wag w składowych ' + str(WYN.get('wejscia_udzial_wariancji')))
        for j in idx:
            split = zdj[j]['split']
            for rz, (ii, dt) in rezimy.items():
                if len(ii) < 2 * a.w + 8:
                    continue
                Xr = pobierz(j, np.array(ii))
                Xr = Xr - Xr.mean(1, keepdims=True)                     # względem średniej tokenów zdjęcia
                wynik, mapy = analiza_trajektorii(Xr, dt, a.w, maski.get(j), sas, rng=rng, wejscia=WEJ.get(rz))
                per[rz][split].append(wynik)
                if mapy is not None and j in (idx[0], idx[len(idx) // 2]):
                    MODY[f'{pn}_{rz}_{zdj[j]["split"]}_{j}'] = mapy
            log(f'{pn}: zdjęcie {j} ({split}) gotowe')
        WYN['przestrzenie'][pn] = {rz: {s_: dict(zdjec=len(v), mediany=mediana_drzewa(v)) for s_, v in d_.items()} for rz, d_ in per.items()}
        json.dump(WYN, open(OUT / 'KOOPMAN_PULAPKI.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    np.savez_compressed(OUT / 'MODY.npz', **MODY)
    log('zapisano', OUT)


if __name__ == '__main__':
    main()
