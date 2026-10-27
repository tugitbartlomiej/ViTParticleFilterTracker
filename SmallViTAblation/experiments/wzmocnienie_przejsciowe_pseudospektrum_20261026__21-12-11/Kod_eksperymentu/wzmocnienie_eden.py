"""Co wynika z nienormalności pola MLP: przejściowe wzmocnienie, stała Kreissa, abscysa numeryczna i realna wrażliwość na zaburzenia,
na trzech układach (detekcja MałyViT z kampanii kompensacji; MałyViT i DeiT-Tiny na CIFAR-10). Bez treningu; modele, dane i jakobiany
jak w rodzicach (import pomiar_mosty_eden.py i pomiar_mosty_cifar.py). Na Edenie nie ma scipy: wszystko numpy/torch.

Operator zlinearyzowanego półkroku MLP dla tokenu: A = I + J, J = jakobian pola MLP (192 × 192, jak w rodzicu). Dla macierzy normalnej
‖Aᵏ‖₂ = ρ(A)ᵏ, gdzie ρ = max|1 + λ|; dla nienormalnej ‖Aᵏ‖₂ może być dużo większe (wzrost przejściowy), choć widmo tego nie zdradza.

W (widmo i normy, na tokenach wybranych jak w rodzicach: detekcja = wszystkie narzędzia + 20 tła na klatkę diagnostyczną; CIFAR = 16
   losowych tokenów obrazu), dla każdego bloku:
   G_k = ‖Aᵏ‖₂ / ρᵏ (k = 1…6): wzmocnienie przejściowe względem normalnego odniesienia o tym samym widmie;
   ω = λ_max((A + Aᵀ)/2) − 1 (abscysa numeryczna J) wobec max Re λ(J): ω > max Re λ to warunek konieczny wzrostu przejściowego;
   K = sup_{|z| > ρ} (|z| − ρ) · ‖(zI − A)⁻¹‖₂ na siatce |z| = ρ(1 + ε), ε ∈ {0,02; 0,05; 0,1; 0,2; 0,4}, 12 kątów (dolne oszacowanie
   stałej Kreissa względem promienia widmowego; ‖(zI − A)⁻¹‖₂ = 1/σ_min(zI − A));
   ‖N‖/‖J‖ (Henrici, jak w rodzicu) dla odniesienia.
   Złożenie po blokach (ten sam token): P = A_NB ⋯ A_1; ‖P‖₂ wobec Π_b ρ_b.
F (zaburzenia, na GPU, nieliniowo): dla klatki i bloku b zaburzamy wybrane tokeny przed półkrokiem MLP (po uwadze) o δ · ‖x_i‖
   (δ ∈ {0,01; 0,03; 0,1}) w kierunku losowym albo najgorszym (prawy wektor osobliwy A_i dla σ_max) i mierzymy:
   wzmocnienie a_i = ‖Δx_out‖/‖Δx_in‖ w samym półkroku MLP wobec przewidywań liniowych: losowy kierunek → ‖A‖_F/√192 (nienormalne;
   ‖A‖_F² = Σ|1+λ|² + ‖N‖²) i √(Σ|1+λ|²/192) (normalne odniesienie); najgorszy kierunek → σ_max(A) = ‖A‖₂ wobec ρ;
   skutek dla wyjścia po przepuszczeniu reszty sieci: detekcja: ΔIoU ramki szczytu i korelacja mapy ciepła z niezaburzoną;
   CIFAR: Δmargines logitów i zmiana poprawności.

Wyjście: <out>/wyniki.npz (W, P, F per token/klatka) i <out>/podsumowanie.json (mediany per checkpoint i blok).
Poprawka 2026-10-26 22:05 (po pierwszym biegu): zaburzenie w F było przykładane do stanu 2b (wejście bloku) zamiast 2b+1 (po uwadze),
przez co pomijało uwagę bloku b i Δ wyjścia nie zależało od δ; tryb --bez_W liczy ponownie samą część F do wyniki_F.npz.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pomiar_mosty_eden as PM  # noqa: E402
import pomiar_mosty_cifar as PC  # noqa: E402

C = 192
KMAX = 6
EPS = (0.02, 0.05, 0.1, 0.2, 0.4)
N_KAT = 12
DELTY = (0.01, 0.03, 0.1)
N_TLA = 20                       # detekcja: tokenów tła na klatkę (plus wszystkie narzędzia)
N_TOK_CIFAR = 16
KOL_W = ['nN_nJ', 'normA', 'rho', 'normA_F', 'rho_F'] + [f'G{k}' for k in range(1, KMAX + 1)] + ['omega', 'maxRe', 'kreiss']
KOL_P = ['normP', 'prod_rho', 'G_P']
KOL_F = ['amp_med', 'amp_q90', 'pred_nn_med', 'pred_norm_med', 'wyj_delta', 'wyj_kor']   # wyj_delta: ΔIoU (detekcja) / Δmargines (CIFAR)


# ---------------------------------------------------------------- W: widmo i normy jednej macierzy (proces puli)
def widmo_i_wzmocnienie(J):
    """J (192, 192) float64 → wektor KOL_W (bez nN_nJ liczonego osobno) oraz λ."""
    A = np.eye(C) + J
    lam = np.linalg.eigvals(J)
    rho = float(np.abs(1 + lam).max())
    nJ = float(np.sqrt((J * J).sum())); nN = float(np.sqrt(max(nJ * nJ - float((np.abs(lam) ** 2).sum()), 0.0)))
    normA_F = float(np.sqrt((A * A).sum())); rho_F = float(np.sqrt((np.abs(1 + lam) ** 2).sum()))
    G = []; Ak = np.eye(C)
    for k in range(1, KMAX + 1):
        Ak = Ak @ A
        G.append(float(np.linalg.norm(Ak, 2)) / rho ** k)
    normA = G[0] * rho
    S = 0.5 * (J + J.T); omega = float(np.linalg.eigvalsh(S).max()); maxRe = float(lam.real.max())
    kre = 0.0
    for e in EPS:
        r = rho * (1 + e)
        for th in np.linspace(0, 2 * np.pi, N_KAT, endpoint=False):
            z = r * np.exp(1j * th)
            smin = np.linalg.svd(z * np.eye(C) - A, compute_uv=False).min()
            kre = max(kre, (r - rho) / max(smin, 1e-300))
    return np.array([nN / max(nJ, 1e-30), normA, rho, normA_F, rho_F] + G + [omega, maxRe, kre]), lam


def norma2(M):
    return float(np.linalg.norm(M, 2))


# ---------------------------------------------------------------- modele: wspólny interfejs
class Uklad:
    def __init__(self, rodzaj, wagi, dane, dev):
        self.rodzaj, self.wagi, self.dev = rodzaj, Path(wagi), dev
        if rodzaj == 'detekcja':
            idd, Xd, gtd = PM.wczytaj_klatki(Path(dane), 'evaluation', tylko_diag=True)
            narz, tlo, najw = PM.maski(gtd)
            self.X, self.ident, self.narz, self.tlo, self.najw, self.y = Xd, idd, narz, tlo, najw, None
        else:
            idx, X, y, idx_d, Xd, yd = PC.wczytaj_cifar(dane, PC.N_NA_KLASE, 5)          # 50 obrazów diagnostycznych (5 na klasę)
            self.X, self.ident, self.y = Xd, idx_d, yd; self.narz = self.tlo = self.najw = None

    def wczytaj(self, k):
        if self.rodzaj == 'detekcja':
            net, krok = PM.wczytaj_siec(self.wagi / f'k{k:05d}.pt', self.dev)
        else:
            net, krok = PC.wczytaj_siec(self.wagi / f'k{k:05d}.pt', 'deit' if self.rodzaj == 'cifar_deit' else 'malyvit', self.dev)
        assert krok in (k, -1), (krok, k)
        return net

    def przejscie(self, net, x):
        if self.rodzaj == 'detekcja':
            return PM.przejscie(net, x, 'base')
        return PC.przejscie(net, x)

    def reszta(self, net, t, od):
        return PM.reszta_sieci(net, t, od, 'base') if self.rodzaj == 'detekcja' else PC.reszta_sieci(net, t, od)

    def wyjscie(self, o, sl):
        """(wartość per klatka: IoU albo margines, mapa do korelacji albo None)."""
        if self.rodzaj == 'detekcja':
            bx, sc, mapa, _, _ = PM.ramki_szczytu(o)
            return PM.iou_xyxy(bx, self.najw[sl]), mapa
        ok, marg = PC.ok_marg(o, torch.as_tensor(self.y[sl], device=o.device))
        return marg, None

    def tokeny(self, i, T, rng):
        if self.rodzaj == 'detekcja':
            tl = np.flatnonzero(self.tlo[i]); nr = np.flatnonzero(self.narz[i])
            los = rng.choice(tl, min(N_TLA, len(tl)), replace=False)
            return np.concatenate([nr, los]), np.r_[np.ones(len(nr), bool), np.zeros(len(los), bool)]
        start = 1 if self.rodzaj == 'cifar_deit' else 0
        tok = np.sort(rng.choice(np.arange(start, T), N_TOK_CIFAR, replace=False))
        return tok, np.zeros(len(tok), bool)


def mlp_polkrok(blk, xa):
    return xa + blk.ls2(blk.mlp.fc2(blk.mlp.act(blk.mlp.fc1(blk.norm2(xa)))))


# ---------------------------------------------------------------- główna pętla
def pomiar(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    U = Uklad(args.uklad, args.wagi, args.dane, dev)
    if args.proba:
        U.X = U.X[:args.proba]; U.ident = U.ident[:args.proba]
        if U.y is not None: U.y = U.y[:args.proba]
        if U.narz is not None: U.narz, U.tlo, U.najw = U.narz[:args.proba], U.tlo[:args.proba], U.najw[:args.proba]
    N = len(U.X); rng = np.random.default_rng(2026); torch.manual_seed(2026)
    W_tab, P_tab, F_tab, czasy = [], [], [], {}
    pula = Pool(max(1, args.cpus))
    print(f'układ {args.uklad}, wagi {args.wagi}, kroki {args.kroki}, klatki/obrazy {N}, {dev}', flush=True)
    try:
        for k in args.kroki:
            t0 = time.time(); net = U.wczytaj(k); NB = len(net.bloki)
            zad, meta = [], []
            with torch.no_grad():
                for s in range(0, N, args.paczka):
                    x = torch.from_numpy(U.X[s:s + args.paczka]).to(dev).permute(0, 3, 1, 2).float() / 255
                    sl = slice(s, s + len(x))
                    stany, XM, AM, o = U.przejscie(net, x)
                    T = stany.shape[2]
                    baza, mapa0 = U.wyjscie(o, sl)
                    for i in range(len(x)):
                        tok, jest_narz = U.tokeny(s + i, T, rng)
                        tok_t = torch.as_tensor(tok, device=dev)
                        # --- W: jakobiany do puli, F: zaburzenia na GPU
                        J_bloki = []
                        for b, blk in enumerate(net.bloki):
                            J = PM.jakobiany(blk, XM[b, i], AM[b, i], tok_t)                                   # (n, 192, 192) float64
                            J_bloki.append(J)
                            if not args.bez_W:
                                Jn = J.cpu().numpy()
                                for j in range(len(tok)):
                                    zad.append(Jn[j]); meta.append((s + i, b, int(tok[j]), float(jest_narz[j])))
                            A = torch.eye(C, device=dev, dtype=torch.float64)[None] + J
                            normA_F = A.flatten(1).norm(dim=1)                                                 # (n,)
                            Uu, Ss, Vh = torch.linalg.svd(A)
                            v_worst = Vh[:, 0, :].float(); sig = Ss[:, 0].float()                              # (n, 192), (n,)
                            xa = stany[2 * b + 1, i]                                                           # (T, 192) po uwadze bloku b (stan 2b+1)
                            xm0 = mlp_polkrok(blk, xa[None])[0]
                            for d in DELTY:
                                for rodzaj, kier in ((0, torch.randn(len(tok), C, device=dev)), (1, v_worst)):
                                    kier = kier / kier.norm(dim=1, keepdim=True)
                                    dx = torch.zeros_like(xa); dx[tok_t] = d * xa[tok_t].norm(dim=1, keepdim=True) * kier
                                    xm1 = mlp_polkrok(blk, (xa + dx)[None])[0]
                                    amp = (xm1 - xm0)[tok_t].norm(dim=1) / dx[tok_t].norm(dim=1).clamp_min(1e-12)
                                    pred_nn = (normA_F.float() / math.sqrt(C)) if rodzaj == 0 else sig
                                    o1 = U.reszta(net, xm1[None], b + 1)
                                    w1, mapa1 = U.wyjscie(o1, slice(s + i, s + i + 1))
                                    kor = float(PM.korelacja_wierszy(mapa1, mapa0[i:i + 1])[0]) if mapa0 is not None else np.nan
                                    F_tab.append([k, s + i, b, d, rodzaj, float(amp.median()), float(amp.quantile(0.9)), float(pred_nn.median()),
                                                  np.nan, float(w1[0] - baza[i]), kor])
                        # --- złożenie po blokach (ten sam token): P = A_NB ⋯ A_1
                        Pm = torch.eye(C, device=dev, dtype=torch.float64)[None].repeat(len(tok), 1, 1)
                        for J in J_bloki:
                            Pm = (torch.eye(C, device=dev, dtype=torch.float64)[None] + J) @ Pm
                        normP = torch.linalg.matrix_norm(Pm, 2).cpu().numpy()
                        for j in range(len(tok)):
                            P_tab.append([k, s + i, tok[j], float(jest_narz[j]), float(normP[j]), np.nan, np.nan])
            tF = time.time() - t0
            wyn = pula.map(widmo_i_wzmocnienie, zad, chunksize=2) if not args.bez_W else []
            for (i, b, tok_j, narz_j), (m, lam) in zip(meta, wyn):
                W_tab.append([k, i, b, tok_j, narz_j] + list(m))
            czasy[str(k)] = dict(F_s=round(tF, 1), W_s=round(time.time() - t0 - tF, 1), n_mac=len(zad))
            print(f'k{k}: {len(zad)} bloków×klatek, F {tF:.0f} s, W {time.time() - t0 - tF:.0f} s', flush=True)
    finally:
        pula.close(); pula.join()
    W = np.array(W_tab, np.float64); P = np.array(P_tab, np.float64); F = np.array(F_tab, np.float64)
    if args.bez_W:                                                     # tryb „tylko F”: W i P z wcześniejszego wyniki.npz tego biegu
        z0 = np.load(out / 'wyniki.npz'); W, P = z0['W'], z0['P']
    # uzupełnienia: prod_rho dla P i pred_norm dla F z tabeli W
    if len(W):
        for r in P:
            w = W[(W[:, 0] == r[0]) & (W[:, 1] == r[1]) & (W[:, 3] == r[2])]
            r[5] = float(np.prod(w[:, 5 + KOL_W.index('rho')])); r[6] = r[4] / max(r[5], 1e-300)
        for r in F:
            w = W[(W[:, 0] == r[0]) & (W[:, 1] == r[1]) & (W[:, 2] == r[2])]
            r[8] = float(np.median(w[:, 5 + KOL_W.index('rho_F')] / math.sqrt(C))) if r[4] == 0 else float(np.median(w[:, 5 + KOL_W.index('rho')]))
    np.savez_compressed(out / ('wyniki_F.npz' if args.bez_W else 'wyniki.npz'), W=W, P=P, F=F, kroki=np.array(args.kroki), ident=np.array(U.ident),
                        kol_W=np.array(['krok', 'klatka', 'blok', 'token', 'narz'] + KOL_W), kol_P=np.array(['krok', 'klatka', 'token', 'narz'] + KOL_P),
                        kol_F=np.array(['krok', 'klatka', 'blok', 'delta', 'rodzaj'] + KOL_F))
    (out / ('podsumowanie_F.json' if args.bez_W else 'podsumowanie.json')).write_text(json.dumps(podsumuj(W, P, F, args, czasy), indent=1, ensure_ascii=False), encoding='utf-8')
    print('zapisano', out, flush=True)


def podsumuj(W, P, F, args, czasy):
    q = PM._q
    s = dict(uklad=args.uklad, wagi=str(args.wagi), kroki=[int(k) for k in args.kroki], czasy=czasy, W={}, P={}, F={})
    for k in args.kroki:
        w = W[W[:, 0] == k]; NB = int(w[:, 2].max()) + 1 if len(w) else 0
        s['W'][str(k)] = {m: [q(w[w[:, 2] == b, 5 + j]) for b in range(NB)] for j, m in enumerate(KOL_W)}
        p = P[P[:, 0] == k]; s['P'][str(k)] = {m: q(p[:, 4 + j]) for j, m in enumerate(KOL_P)}
        f = F[F[:, 0] == k]
        s['F'][str(k)] = {f'{m}_d{d}_{"los" if r == 0 else "najg"}': [q(f[(f[:, 2] == b) & (f[:, 3] == d) & (f[:, 4] == r), 5 + j]) for b in range(NB)]
                          for j, m in enumerate(KOL_F) for d in DELTY for r in (0, 1)}
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uklad', required=True, choices=['detekcja', 'cifar_maly', 'cifar_deit'])
    p.add_argument('--wagi', required=True); p.add_argument('--dane', required=True); p.add_argument('--out', required=True)
    p.add_argument('--kroki', type=int, nargs='+', required=True); p.add_argument('--paczka', type=int, default=10)
    p.add_argument('--cpus', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    p.add_argument('--proba', type=int, default=0)
    p.add_argument('--bez_W', action='store_true', help='tylko część F (GPU); W i P przepisane z istniejącego wyniki.npz; zapis do wyniki_F.npz')
    pomiar(p.parse_args())


if __name__ == '__main__':
    main()
