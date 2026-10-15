"""Langevin i Koopman na PUŁAPKACH z wejściami z sieci (sygnaly_sieci.py → sygnaly_*.npz).

Pułapka tokenu: średnia krocząca ±w albo średnia z bloków B (bez nakładania) latentu ostatniego bloku (rzut 128D, względem
średniej tokenów zdjęcia). Przejście pułapki t → t+1 i wejścia uśrednione tymi samymi wagami (koopman_lokalny.u_pulapki).
Wejścia (wszystkie liczone z sieci w danym kroku):
  F  — „własna siła”: przesunięcie latentu tokenu od kroku AdamW tylko na stracie tego zdjęcia (wektor 128D na token);
  G  — „z góry sieci”: przyrost stanu tokenu wchodzącego do ostatniego bloku (192D na token), mapowany wspólną macierzą 128 × 192;
  s  — skalary: zdjęcie w paczce, strata paczki, normy kroku wag (łodyga, 6 bloków, głowa).
Model Langevina dla przyrostu pułapki Δc_j (wspólne współczynniki dla 600 tokenów, dopasowanie na 70 % czasu, R² na 30 %):
  L1 dryf powrotny: −γ (c_j − c̄_j);  L2 + α F_j;  L3 + M G_j;  L4 + skalary (B s);  L5 = L1 + F + G + s.
Kontrole: F przetasowane w czasie, F innego tokenu (permutacja przestrzenna), G przetasowane w czasie.
Reszta L5 (szum Moriego–Zwanziga): autokorelacja i wspólna składowa między sondami.
Uruchomienie: python sygnaly_analiza.py <katalog sygnaly_*> [...]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import koopman_lokalny as KL  # noqa: E402


def wczytaj(kat):
    pl = sorted(glob.glob(os.path.join(kat, 'sygnaly_*.npz')))
    d = {}
    for p in pl:
        z = np.load(p)
        for k in z.files:
            d.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v) for k, v in d.items()}, json.load(open(os.path.join(kat, 'meta.json'), encoding='utf-8'))


def pulapki(Y, nazwa):
    return KL.srednia_kroczaca(Y, int(nazwa.split('w')[-1])) if nazwa.startswith('krocz') else KL.bloki(Y, int(nazwa.split('B')[-1]))


def wej_pulapki(u_raw, nazwa, T):
    """u_raw (K − 1, ...) dowolnego kształtu → (T − 1, ...) tymi samymi wagami co przejście pułapki."""
    sh = u_raw.shape
    return KL.u_pulapki(u_raw.reshape(sh[0], -1).astype(np.float64), nazwa)[:T - 1].reshape((-1,) + sh[1:])


def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / max((y ** 2).sum(), 1e-30))     # względem modelu zerowego (Δc = 0)


def langevin(C, F, G, S_, frac=0.7, ridge=1e-6):
    """C (T, n, d) pułapki; F (T−1, n, d) własna siła; G (T−1, n, g) wejście z góry; S_ (T−1, k) skalary → R² testowe modeli."""
    T, n, d = C.shape
    Ttr = int((T - 1) * frac)
    dC = np.diff(C, axis=0).astype(np.float64)
    mu = C[:Ttr].mean(0)
    cech = {'dryf': (C[:-1] - mu).astype(np.float64)}

    def fit(lista, y, tr):
        # wspólne współczynniki: macierze d × p dla każdej cechy wektorowej, B dla skalarów (te same dla tokenów)
        X = np.concatenate([x.reshape(len(x), n, -1) for x in lista], 2)          # (T−1, n, p)
        Xt, yt = X[:tr].reshape(-1, X.shape[2]), y[:tr].reshape(-1, d)
        Gm = Xt.T @ Xt; Gm += ridge * np.diag(np.diag(Gm)) + 1e-10 * np.diag(Gm).mean() * np.eye(len(Gm))
        W = np.linalg.solve(Gm, Xt.T @ yt)
        return (X.reshape(-1, X.shape[2]) @ W).reshape(y.shape)

    Fs = F.astype(np.float64); Gs = G.astype(np.float64)
    Ss = np.repeat(S_[:, None, :], n, 1).astype(np.float64)
    rng = np.random.default_rng(0)
    perm_t = rng.permutation(T - 1); perm_n = rng.permutation(n)
    modele = {'L1_dryf': [cech['dryf']], 'L2_dryf_F': [cech['dryf'], Fs], 'L3_dryf_G': [cech['dryf'], Gs],
              'L4_dryf_skalary': [cech['dryf'], Ss], 'L5_wszystko': [cech['dryf'], Fs, Gs, Ss],
              'K_F_czas_tasowane': [cech['dryf'], Fs[perm_t]], 'K_F_inny_token': [cech['dryf'], Fs[:, perm_n]],
              'K_G_czas_tasowane': [cech['dryf'], Gs[perm_t]], 'F_sama': [Fs], 'G_sama': [Gs]}
    out = {}
    for nm, lst in modele.items():
        yh = fit(lst, dC, Ttr)
        out[nm] = r2(dC[Ttr:], yh[Ttr:])
        if nm == 'L5_wszystko':
            e = dC - yh
            en = (e ** 2).sum((1, 2))
            out['reszta_L5_autokorelacja'] = [float((e[:-k] * e[k:]).sum() / (e ** 2).sum()) for k in (1, 2, 3, 5, 10)]
            out['_energia_reszty'] = en
    # kierunek: czy pułapka idzie wzdłuż własnej siły
    cosF = (dC * Fs).sum(-1) / (np.linalg.norm(dC, axis=-1) * np.linalg.norm(Fs, axis=-1) + 1e-12)
    out['cos_dC_F'] = float(cosF.mean())
    return out


def main():
    for kat in sys.argv[1:]:
        d, meta = wczytaj(kat)
        z6, xin, zwl = d['z6'].astype(np.float32), d['xin'].astype(np.float32), d['zwl'].astype(np.float32)
        K, S = z6.shape[:2]
        sk = np.concatenate([d['w_paczce'].astype(np.float64), d['strata_paczki'][:, None], d['normy_krokow']], 1)[:-1]   # wejścia kroku k: stan k → k+1
        sk_std = (sk - sk.mean(0)) / (sk.std(0) + 1e-12)
        print(f'== {os.path.basename(kat)}: {K} kroków, {S} sond, start {meta["krok_startu"]}')
        WYN = {}
        for nazwa in ('krocz_w10', 'krocz_w25', 'bloki_B10', 'bloki_B25'):
            per, energie = [], []
            for s in range(S):
                x = z6[:, s] - z6[:, s].mean(1, keepdims=True)
                C = pulapki(x, nazwa); T = len(C)
                F = wej_pulapki(zwl[:-1, s], nazwa, T)                                  # siła w stanie k działa w kroku k
                G = wej_pulapki(np.diff(xin[:, s], axis=0), nazwa, T)
                Sx = wej_pulapki(np.concatenate([sk_std[:, :], sk[:, s:s + 1]], 1), nazwa, T)
                o = langevin(C, F, G, Sx)
                energie.append(o.pop('_energia_reszty')); per.append(o)
            E = np.stack(energie); E = E / E.mean(1, keepdims=True)
            iu = np.triu_indices(S, 1)
            med = {k: (float(np.median([p[k] for p in per])) if not isinstance(per[0][k], list) else np.median([p[k] for p in per], 0).round(3).tolist()) for k in per[0]}
            med['reszta_wspolna_korelacja_sond'] = float(np.median(np.corrcoef(E)[iu])) if S > 1 else None
            WYN[nazwa] = med
            print(f'  {nazwa}: ' + ', '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in med.items()))
        json.dump(WYN, open(os.path.join(kat, 'SYGNALY_ANALIZA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
