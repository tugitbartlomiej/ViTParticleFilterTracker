"""Sufit przewidywalności latentu małego ViT na h kroków uczenia (główny horyzont h = 100).

Rozstrzyga: czy nawet wyrocznia znająca dokładną gładką ścieżkę przyszłości zmniejsza błąd prognozy „bez zmian”
o 85 procent. Obala cel 85 procent: sufit wyroczni poniżej 0,85 dla danej wielkości i okresu uczenia.
Podłoga: ta sama wyrocznia na samych drganiach (dryf usunięty), czyli zysk z samego odszumienia startu.
Osiągalne bez uczenia: średnia z ostatnich w stanów (C0) i jej przedłużenie o ostatni dryf (C1, C2).

Uruchomienie (Python 3.11):  py -3.11 sufit.py --wyniki ../wyniki
Próba:                       py -3.11 sufit.py --wyniki ../wyniki/proba --obrazy 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

import dane

H = (10, 50, 100)
W = (5, 11, 21, 51)
GRANICE = (0, 100, 300, 600, dane.T)
OKRESY = ('0-100', '100-300', '300-600', '600-koniec')
METRYKI = ('pers_w', 'wyr', 'jit_p', 'jit_w', 'D2', 's2', 'n_w',
           'pers_c', 'c0', 'c1', 'c2', 'nf', 'cont', 'cos', 'n_c')
IM = {m: i for i, m in enumerate(METRYKI)}
CH = 32                  # jednostek naraz (ogranicza pamięć tymczasową)
N_TLO = 100              # losowe tokeny tła na obraz
# Prognoza liniowa z własnej historii: Δ = x_{t+h} - x_t ≈ Σ β_j f_j, β wspólne dla jednostek i współrzędnych.
# Cechy: odchylenie średniej z ostatnich k stanów od x_t oraz przesunięcia z ostatnich l kroków. Tylko przeszłość.
K_SR = (5, 21, 51, 101, 201)
LAGI = (10, 50, 100)
CECHY = tuple(f'sr{k}' for k in K_SR) + tuple(f'lag{l}' for l in LAGI)
T0_LIN = max(max(K_SR) - 1, max(LAGI))     # pierwszy start z pełną historią


def sq(a):
    return np.einsum('tud,tud->tu', a, a, dtype=np.float64)


def srednia_centralna(Y, w):
    """Średnia ruchoma szerokości w wyśrodkowana na t; NaN tam, gdzie okno wychodzi poza 0..T-1."""
    T, r = Y.shape[0], w // 2
    c = np.zeros((T + 1,) + Y.shape[1:], np.float64)
    np.cumsum(Y, axis=0, dtype=np.float64, out=c[1:])
    M = np.full(Y.shape, np.nan, np.float32)
    M[r:T - r] = (c[w:] - c[:-w]) / w
    return M


def do_okresow(t, X):
    """Suma wierszy X (po czasie startu t) w czterech okresach -> (4, U)."""
    out = np.zeros((len(OKRESY), X.shape[1]))
    for k in range(len(OKRESY)):
        sel = (t >= GRANICE[k]) & (t < GRANICE[k + 1])
        if sel.any():
            out[k] = X[sel].sum(0)
    return out


def statystyki_lin(y, lin):
    """Dodaje do lin[h] sumy FᵀF, FᵀΔ, ΔᵀΔ po okresach (dla dopasowania krzyżowego między obrazami)."""
    T = y.shape[0]
    c = np.zeros((T + 1,) + y.shape[1:], np.float64)
    np.cumsum(y, axis=0, dtype=np.float64, out=c[1:])
    for h in H:
        t = np.arange(T0_LIN, T - h)
        for k in range(len(OKRESY)):
            tk = t[(t >= GRANICE[k]) & (t < GRANICE[k + 1])]
            if len(tk) == 0:
                continue
            F = np.stack([((c[tk + 1] - c[tk + 1 - kk]) / kk).astype(np.float32) - y[tk] for kk in K_SR]
                         + [y[tk] - y[tk - l] for l in LAGI])
            dlt = y[tk + h] - y[tk]
            L = lin[h]
            L['G'][k] += np.einsum('fnud,gnud->fg', F, F, dtype=np.float64)
            L['b'][k] += np.einsum('fnud,nud->f', F, dlt, dtype=np.float64)
            L['c'][k] += np.einsum('nud,nud->', dlt, dlt, dtype=np.float64)
            L['n'][k] += dlt.shape[0] * dlt.shape[1]
            del F, dlt


def analizuj(Y):
    """Y (T, U, d) -> {(w, h): (len(METRYKI), 4, U)}, autokorelacja przyrostów (2, 4, U), statystyki liniowe."""
    T, U = Y.shape[:2]
    out = {(w, h): np.zeros((len(METRYKI), len(OKRESY), U)) for w in W for h in H}
    ac = np.zeros((2, len(OKRESY), U))
    nf = len(CECHY)
    lin = {h: dict(G=np.zeros((len(OKRESY), nf, nf)), b=np.zeros((len(OKRESY), nf)), c=np.zeros(len(OKRESY)),
                   n=np.zeros(len(OKRESY))) for h in H}
    for u0 in range(0, U, CH):
        y = np.ascontiguousarray(Y[:, u0:u0 + CH], dtype=np.float32)
        s = slice(u0, u0 + y.shape[1])
        statystyki_lin(y, lin)
        dy = y[1:] - y[:-1]
        tt = np.arange(T - 2)
        ac[0, :, s] += do_okresow(tt, np.einsum('tud,tud->tu', dy[:-1], dy[1:], dtype=np.float64))
        ac[1, :, s] += do_okresow(tt, sq(dy[:-1]))
        del dy
        for w in W:
            r = w // 2
            M = srednia_centralna(y, w)
            J = y - M
            for h in H:
                a = out[(w, h)]
                t = np.arange(r, T - h - r)
                A, Mh, Mt = y[t + h], M[t + h], M[t]
                jed = np.ones((len(t), y.shape[1]))
                for nazwa, X in (('pers_w', sq(A - y[t])), ('wyr', sq(A - Mh)), ('jit_p', sq(J[t + h] - J[t])),
                                 ('jit_w', sq(J[t + h])), ('D2', sq(Mh - Mt)), ('s2', sq(J[t])), ('n_w', jed)):
                    a[IM[nazwa], :, s] += do_okresow(t, X)
                del A, Mh, Mt
                tc = np.arange(h + 2 * r, T - h - r)
                if len(tc) == 0:
                    continue
                Ac, Tm = y[tc + h], M[tc - r]            # Tm = średnia z ostatnich w stanów do kroku tc włącznie
                trend = Tm - M[tc - r - h]               # przesunięcie tej średniej przez ostatnie h kroków
                vf, vp = M[tc + h] - M[tc], M[tc] - M[tc - h]
                nf, npp = sq(vf), sq(vp)
                cos = np.einsum('tud,tud->tu', vf, vp, dtype=np.float64) / np.sqrt(nf * npp + 1e-30)
                for nazwa, X in (('pers_c', sq(Ac - y[tc])), ('c0', sq(Ac - Tm)), ('c1', sq(Ac - Tm - trend)),
                                 ('c2', sq(Ac - Tm - 0.5 * trend)), ('nf', nf), ('cont', sq(vf - vp)),
                                 ('cos', cos), ('n_c', np.ones_like(cos))):
                    a[IM[nazwa], :, s] += do_okresow(tc, X)
            del M, J
    return out, ac, lin


def podsumuj_lin(stat_obrazow, obrazy):
    """Dopasowanie krzyżowe: β z jednej połowy obrazów, błąd na drugiej, i odwrotnie. Plus β z całości (w próbie)."""
    pol = len(obrazy) // 2
    A, B = list(range(pol)), list(range(pol, len(obrazy)))
    wyn = {}
    for h in H:
        dh = {}
        for k, ok in enumerate(OKRESY):
            S = lambda idx, key: sum(stat_obrazow[i][h][key][k] for i in idx)
            if S(A + B, 'n') == 0:
                dh[ok] = None
                continue

            def beta(idx):
                G, b = S(idx, 'G'), S(idx, 'b')
                return np.linalg.solve(G + 1e-9 * np.trace(G) * np.eye(len(b)), b)

            def sse(idx, bt):
                return S(idx, 'c') - 2 * bt @ S(idx, 'b') + bt @ S(idx, 'G') @ bt

            c_all = S(A + B, 'c')
            d = {}
            if A and B:
                d['liniowy_krzyzowo_mse'] = float(1 - (sse(B, beta(A)) + sse(A, beta(B))) / c_all)
                d['liniowy_krzyzowo_rmse'] = float(1 - np.sqrt((sse(B, beta(A)) + sse(A, beta(B))) / c_all))
            b_all = beta(A + B)
            d['liniowy_w_probie_mse'] = float(1 - sse(A + B, b_all) / c_all)
            d['beta_w_probie'] = {n: float(v) for n, v in zip(CECHY, b_all)}
            G, b = S(A + B, 'G'), S(A + B, 'b')
            for j, kk in enumerate(K_SR):            # sama średnia z ostatnich kk stanów jako prognoza
                d[f'srednia_ost{kk}_mse'] = float(1 - (c_all - 2 * b[j] + G[j, j]) / c_all)
            d['n_jedn_x_startow'] = int(S(A + B, 'n'))
            dh[ok] = d
        wyn[f'h{h}'] = dh
    return wyn


def kierunek_lda(lat, tool, bg, krok=1000):
    Z = np.asarray(lat[krok], dtype=np.float64)              # (12, 600, 768)
    xt, xb = Z[tool], Z[bg]
    mt, mb = xt.mean(0), xb.mean(0)
    lw = LedoitWolf().fit(np.concatenate([xt - mt, xb - mb]))
    w = np.linalg.solve(lw.covariance_, mt - mb)
    w /= np.linalg.norm(w)
    pt, pb = xt @ w, xb @ w
    dprim = (pt.mean() - pb.mean()) / np.sqrt(0.5 * (pt.var() + pb.var()))
    return w.astype(np.float32), dict(krok=krok, shrinkage=float(lw.shrinkage_), d_prim=float(dprim),
                                      n_narz=int(len(xt)), n_tlo=int(len(xb)))


def wielkosci(X, tool, bg_wyb, bg, w_lda, w_los):
    """Wszystkie badane wielkości jednego obrazu, każda jako (T, U, d)."""
    Xn = X[:, tool]
    sr_n = Xn.mean(1, keepdims=True)
    sr_t = X[:, bg].mean(1, keepdims=True)
    rz = (Xn @ w_lda)[..., None]
    rl = (Xn @ w_los)[..., None]
    return {
        'tok_narz': Xn, 'tok_tlo': X[:, bg_wyb], 'sr_narz': sr_n, 'sr_obraz': X.mean(1, keepdims=True),
        'wekt_tozs': sr_n - sr_t, 'rzut_tok': rz, 'sila_obrazu': rz.mean(1, keepdims=True),
        'rzut_tok_los': rl, 'sila_obrazu_los': rl.mean(1, keepdims=True),
    }


def podsumuj(a, ac):
    """Sumy (len(METRYKI), 4, U) -> liczby dla każdego okresu."""
    g = lambda m, k: a[IM[m], k]
    wyn = {}
    for k, ok in enumerate(OKRESY):
        n_w, n_c = g('n_w', k).sum(), g('n_c', k).sum()
        d = dict(n_startow_wyr=int(round(g('n_w', k)[0])) if n_w else 0,
                 n_startow_przycz=int(round(g('n_c', k)[0])) if n_c else 0, n_jedn=int(a.shape[2]))
        if n_w:
            P, O = g('pers_w', k).sum(), g('wyr', k).sum()
            d['sufit_mse'] = 1 - O / P
            d['sufit_rmse'] = 1 - np.sqrt(O / P)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = g('wyr', k) / g('pers_w', k)
            d['sufit_mse_mediana_jedn'] = float(np.nanmedian(1 - ratio))
            D2, s2 = g('D2', k).sum() / n_w, g('s2', k).sum() / n_w
            d['D2_do_sigma2'] = D2 / s2
            d['sufit_wzor_mse'] = 1 - s2 / (D2 + 2 * s2)
            d['podloga_drgania_mse'] = 1 - g('jit_w', k).sum() / g('jit_p', k).sum()
        if n_c:
            Pc = g('pers_c', k).sum()
            for c in ('c0', 'c1', 'c2'):
                d[f'{c}_mse'] = 1 - g(c, k).sum() / Pc
                d[f'{c}_rmse'] = 1 - np.sqrt(g(c, k).sum() / Pc)
            d['trwalosc_dryfu_cos'] = g('cos', k).sum() / n_c
            d['przedluzenie_dryfu_mse'] = 1 - g('cont', k).sum() / g('nf', k).sum()
        if ac[1, k].sum():
            d['autokorelacja_przyrostow_lag1'] = ac[0, k].sum() / ac[1, k].sum()
        wyn[ok] = {kk: (float(v) if isinstance(v, (float, np.floating)) else v) for kk, v in d.items()}
    return wyn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wyniki', required=True)
    p.add_argument('--obrazy', default='all')
    arg = p.parse_args()
    wyn_dir = Path(arg.wyniki)
    wyn_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lat = dane.latent()
    tool, bg = dane.maski()
    w_lda, info_lda = kierunek_lda(lat, tool, bg)
    rng = np.random.default_rng(20261003)
    w_los = rng.standard_normal(dane.D).astype(np.float32)
    w_los /= np.linalg.norm(w_los)
    print('LDA', info_lda, 'cos(LDA, los)', float(w_lda @ w_los), flush=True)
    obrazy = range(dane.N_OBR) if arg.obrazy == 'all' else [int(x) for x in arg.obrazy.split(',')]
    sumy, sumy_ac, sumy_lin = {}, {}, {}
    for i in obrazy:
        X = dane.obraz(lat, i)
        bg_idx = np.flatnonzero(bg[i])
        bg_wyb = np.sort(rng.choice(bg_idx, min(N_TLO, len(bg_idx)), replace=False))
        for q, Y in wielkosci(X, tool[i], bg_wyb, bg[i], w_lda, w_los).items():
            a, ac, lin = analizuj(Y)
            for key, v in a.items():
                sumy.setdefault(q, {}).setdefault(key, []).append(v)
            sumy_ac.setdefault(q, []).append(ac)
            sumy_lin.setdefault(q, []).append(lin)
        del X
        print(f'obraz {i} gotowy, {time.time() - t0:.0f} s', flush=True)
    wynik = dict(
        meta=dict(zrodlo=str(dane.ZRODLO / 'LATENT_GESTY.npy'), bieg='Eden 1818732', ziarno=47, obrazy=list(obrazy),
                  horyzonty=H, szerokosci_w=W, okresy=OKRESY, granice=GRANICE, n_tlo_na_obraz=N_TLO, lda=info_lda,
                  cos_lda_los=float(w_lda @ w_los), czas_s=time.time() - t0,
                  definicje=dict(sufit='1 - błąd wyroczni m_{t+h} / błąd prognozy bez zmian; łącznie po jednostkach',
                                 c0='średnia z ostatnich w stanów', c1='c0 + przesunięcie tej średniej przez h kroków',
                                 c2='c0 + połowa tego przesunięcia',
                                 podloga='wyrocznia na samych drganiach e = x - m (bez dryfu)')),
        wielkosci={})
    for q, d in sumy.items():
        ac = np.concatenate(sumy_ac[q], axis=2)
        wynik['wielkosci'][q] = {f'w{w}_h{h}': podsumuj(np.concatenate(d[(w, h)], axis=2), ac) for (w, h) in d}
        wynik['wielkosci'][q]['liniowy'] = podsumuj_lin(sumy_lin[q], list(obrazy))
    (wyn_dir / 'METRYKI.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
    print('zapisano', wyn_dir / 'METRYKI.json', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
