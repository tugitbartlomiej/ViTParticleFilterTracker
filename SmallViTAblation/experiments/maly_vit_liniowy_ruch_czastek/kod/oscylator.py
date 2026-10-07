"""Etap 1c: liniowy model ruchu każdej cząstki z bezwładnością (dyskretny Langevin z tłumieniem, AR(2)).

Stan cząstki i: (z_t, v_t = z_t − z_(t−1)) w R^(2*768). Modele kroku (bez przyszłości: tylko stany do t):
  P  persystencja:   z_(t+1) = z_t
  V  stała prędkość: v_(t+1) = v_t
  B  pęd:            v_(t+1) = beta_i v_t                                   (beta_i na token, skalar)
  O  oscylator:      v_(t+1) = beta_i v_t − kappa_i (z_t − m_t)            (m_t = średnia ostatnich W kroków, przyczynowa)
  K  Koopman PCA:    s_(t+1) = A s_t + b lr_t + c, s = [p_t, p_t − p_(t−1)], p = rzut z_t na wspólne PCA-k
                     (jedna macierz A dla wszystkich cząstek klatki; odtworzenie z = mu + p U)
Dopasowanie na krokach 300..T_UCZ, test T_UCZ+1..1000 (stały lr). Miara: błąd z_(t+h) z jazdy swobodnej
modelu (h kroków bez nowych danych) względem persystencji, h = 1, 5, 20, 50; osobno narzędzie i tło.
Uruchomienie z katalogu app: python ../experiments/maly_vit_liniowy_ruch_czastek/kod/oscylator.py 3
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

TU = Path(__file__).resolve().parent
sys.path.insert(0, str(TU.parents[2] / 'app'))
import analiza_uczenia as AU  # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T_UCZ, OD, W, HS, KPCA = 700, 300, 50, (1, 5, 20, 50), 32


def srednia_kroczaca(x, w):
    """m_t = średnia z_(t−w+1..t) (przyczynowa), x [L, n, 768]."""
    c = torch.cumsum(torch.cat([torch.zeros_like(x[:1]), x]), 0)
    t = torch.arange(len(x), device=x.device)
    lo = (t - w + 1).clamp_min(0)
    return (c[t + 1] - c[lo]) / (t + 1 - lo).float()[:, None, None]


def jazda(z0, v0, m0, beta, kappa, h):
    """h kroków oscylatora od (z0, v0) przy stałym środku m0; beta, kappa [n] albo skalary."""
    z, v = z0.clone(), v0.clone()
    for _ in range(h):
        v = beta[:, None] * v - kappa[:, None] * (z - m0)
        z = z + v
    return z


def main():
    b = AU.znajdz_biegi()[0]; o = AU.opis_biegu(b)
    wynik = {}
    for i in [int(x) for x in sys.argv[1:]] or [3]:
        Z = AU.wczytaj_klatke(b, i); narz = torch.from_numpy(o['narzedzie'][i]).to(dev)
        X = torch.from_numpy(np.asarray(Z)).to(dev)                                  # float16 [T, 600, 768]
        T = len(X)
        bl = {m: {h: torch.zeros(600, device=dev) for h in HS} for m in 'PVBOK'}
        par = dict(beta=torch.zeros(600, device=dev), kappa=torch.zeros(600, device=dev))
        for t0 in range(0, 600, 100):
            x = X[:, t0:t0 + 100].float()                                           # [T, n, 768]
            v = torch.cat([torch.zeros_like(x[:1]), x[1:] - x[:-1]])
            m = srednia_kroczaca(x, W)
            # dopasowanie beta, kappa na OD..T_UCZ-1: v_(t+1) = beta v_t − kappa (z_t − m_t), MNK na token
            sl = slice(OD, T_UCZ)
            a1 = v[sl]; a2 = -(x[sl] - m[sl]); y = v[OD + 1:T_UCZ + 1]
            g11 = (a1 * a1).sum((0, 2)); g12 = (a1 * a2).sum((0, 2)); g22 = (a2 * a2).sum((0, 2))
            r1 = (a1 * y).sum((0, 2)); r2 = (a2 * y).sum((0, 2))
            det = (g11 * g22 - g12 ** 2).clamp_min(1e-12)
            beta = (g22 * r1 - g12 * r2) / det; kappa = (g11 * r2 - g12 * r1) / det
            beta_b = r1 / g11.clamp_min(1e-12)                                       # model B: sam pęd
            par['beta'][t0:t0 + 100] = beta; par['kappa'][t0:t0 + 100] = kappa
            zero = torch.zeros_like(beta)
            for h in HS:
                ts = torch.arange(T_UCZ, T - h, device=dev)
                z0, v0, m0, cel = x[ts], v[ts], m[ts], x[ts + h]
                f = lambda zz: ((zz - cel) ** 2).sum(-1).sum(0)                     # [n]
                bl['P'][h][t0:t0 + 100] = f(z0)
                bl['V'][h][t0:t0 + 100] = f(z0 + h * v0)
                zb = z0.clone(); vb = v0.clone()
                for _ in range(h):
                    vb = beta_b[None, :, None] * vb; zb = zb + vb
                bl['B'][h][t0:t0 + 100] = f(zb)
                zo, vo = z0.clone(), v0.clone()
                for _ in range(h):
                    vo = beta[None, :, None] * vo - kappa[None, :, None] * (zo - m0); zo = zo + vo
                bl['O'][h][t0:t0 + 100] = f(zo)
            del x, v, m
        # K: wspólna PCA-k cząstek klatki (z kroków 300..T_UCZ), stan [p, dp], jedna macierz A + wejście lr
        pula = X[OD:T_UCZ + 1:5].float().reshape(-1, 768)
        mu = pula.mean(0)
        U = torch.linalg.svd(pula[::7] - mu, full_matrices=False)[2][:KPCA].T                      # [768, k]
        P = torch.stack([(X[t].float() - mu) @ U for t in range(T)])                                 # [T, 600, k]
        dP = torch.cat([torch.zeros_like(P[:1]), P[1:] - P[:-1]])
        S = torch.cat([P, dP], -1)                                                                  # [T, 600, 2k]
        lr = torch.from_numpy(AU and np.minimum(1.0, (np.arange(T) + 1) / 300)).float().to(dev)
        Xs = torch.cat([S[OD:T_UCZ], lr[OD:T_UCZ, None, None].expand(-1, 600, 1),
                        torch.ones(T_UCZ - OD, 600, 1, device=dev)], -1).reshape(-1, 2 * KPCA + 2)
        Ys = S[OD + 1:T_UCZ + 1].reshape(-1, 2 * KPCA)
        Wk = torch.linalg.lstsq(Xs.double().cpu(), Ys.double().cpu()).solution.float().to(dev)        # [2k+2, 2k]
        rk = float(torch.linalg.eigvals(Wk[:2 * KPCA].T.cpu()).abs().max())
        for h in HS:
            ts = torch.arange(T_UCZ, T - h, device=dev)
            s = S[ts]
            for j in range(h):
                s = torch.cat([s, lr[ts + j, None, None].expand(-1, 600, 1), torch.ones(len(ts), 600, 1, device=dev)], -1) @ Wk
            # błąd w pełnym latencie: z_hat = z_t + (p_hat − p_t) U  (zmiana tylko w podprzestrzeni k)
            for t0 in range(0, 600, 100):
                zt = X[ts, t0:t0 + 100].float(); cel = X[ts + h, t0:t0 + 100].float()
                zh = zt + (s[:, t0:t0 + 100, :KPCA] - P[ts, t0:t0 + 100]) @ U.T
                bl['K'][h][t0:t0 + 100] = ((zh - cel) ** 2).sum(-1).sum(0)
        del X; torch.cuda.empty_cache()
        w = {}
        for mdl in 'VBOK':
            w[mdl] = {}
            for h in HS:
                r = (bl[mdl][h] / bl['P'][h].clamp_min(1e-12)).cpu().numpy()
                nn = o['narzedzie'][i]
                w[mdl][h] = dict(narz=round(float(np.median(r[nn])), 4), tlo=round(float(np.median(r[~nn])), 4),
                                 calosc=round(float(bl[mdl][h].sum() / bl['P'][h].sum()), 4))
        nn = o['narzedzie'][i]
        w['parametry'] = dict(beta_mediana=round(float(par['beta'].median()), 4), kappa_mediana=round(float(par['kappa'].median()), 4),
                              beta_narz=round(float(par['beta'][nn].median()), 4), kappa_narz=round(float(par['kappa'][nn].median()), 4),
                              promien_widmowy_K=round(rk, 4))
        wynik[str(i)] = w
        print(i, json.dumps(w, ensure_ascii=False), flush=True)
    (TU.parent / 'wyniki' / 'OSCYLATOR.json').write_text(json.dumps(dict(
        opis='błąd jazdy swobodnej h kroków / błąd persystencji (mniej = lepiej), mediana po tokenach i całość; '
             f'dopasowanie kroki {OD}..{T_UCZ}, test {T_UCZ}..1000',
        wyniki=wynik), ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
