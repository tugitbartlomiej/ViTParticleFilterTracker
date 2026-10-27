"""Czy nienormalność pola MLP ma strukturę „balanced amplification” (Murphy & Miller 2009) i nienormalnego wzmacniacza fotonicznego
(Makris 2014, Midya 2022): ukryty łańcuch feedforward o niskiej randze w formie Schura, rozdzielność kierunku wejściowego i wyjściowego
wzmacnianego modu (zawór jednokierunkowy), wyrównanie wzmacnianego kierunku z odczytem głowy, oraz skąd bierze się nienormalność
(wagi W₂W₁ czy bramkowanie GELU) i jak zależy od liczby „żywych reflektorów” (neuronów w zakresie liniowym). Trzy układy jak w rodzicu
wzmocnienie_przejsciowe_pseudospektrum (detekcja 3 seedy, MałyViT i DeiT na CIFAR-10), checkpointy k0, wczesny, koniec. Bez scipy.

Oznaczenia: J = jakobian pola MLP dla tokenu (192 × 192), A = I + J, λ = widmo J. Forma Schura bez scipy: J = VΛV⁻¹ (eig), V = QR,
T = Q*JQ = RΛR⁻¹ (górnotrójkątna), N = T − diag(T) (część nilpotentna; ‖N‖_F = odchylenie Henriciego). Para osobliwa pola J (nie A = I+J, bo przy małym J wektory osobliwe A są
wyznaczone przez I i u₁ ≈ v₁ niezależnie od struktury): σ₁(J), u₁ (wyjście, „pisz”), v₁ (wejście, „czytaj”). G₁ = ‖A‖₂/ρ jak w rodzicu.

S (Schur, per token, pula procesów):
  c5 = Σ₅ σᵢ(N)² / ‖N‖_F²  (koncentracja łańcucha feedforward; dla macierzy Ginibre'a rozmyta), PR_N = (Σσᵢ²)²/Σσᵢ⁴ (ranga efektywna N),
  cos_vu = |⟨v₁, u₁⟩| dla pary osobliwej J (rozdzielność czytaj/pisz: 1 dla macierzy normalnej, ≈ 0 dla czystego feedforward),
  zawor = ‖J u₁‖ / σ₁(J) (drugie przejście przez pole: czy to, co pole napisało, jest znów czytane; feedforward → ≈ 0, normalne → 1),
  sigma1 = σ₁(J), rho = max|1+λ|, nN_nJ, G1 = ‖I+J‖₂/ρ.
O (odczyt, GPU, autograd, 8 tokenów na klatkę; uzupełnienie BA3c/BA3d 2026-10-27 01:05: energia u₁/v₁ w podprzestrzeni odczytu wszystkich
  wyjść głowy (5 w detekcji, w tym osobno 4 ramki; 10 logitów w CIFAR; QR gradientów) oraz łańcuch między blokami cos²(u₁ᵇ, v₁ᵇ⁺¹)): r = ∂(wyjście tokenu)/∂x_b,token przez resztę sieci (detekcja: logit ciepła tego tokenu;
  CIFAR: margines logitów klasy prawdziwej; DeiT: przez uwagę do CLS), unormowany; cos²(u₁, r), cos²(v₁, r) (losowy kierunek: 1/192).
R (reflektory, per token, GPU): liczba neuronów z GELU′(a) ∈ (0,1; 0,9) („żywe”), średnia GELU′; per blok (per checkpoint):
  nN_nJ macierzy samych wag W₂W₁ i W₂·diag(średnia GELU′)·W₁, oraz rozkład |cos(w₁ᵢ, w₂ᵢ)| po neuronach (wzajemność reflektora: czy
  reflektor emituje w kierunku, z którego czyta), średnia i średnia ważona |g| gdzie g = średnia GELU′ᵢ.

Wyjście: <out>/wyniki.npz (S, O, R per token; BLK per blok) i <out>/podsumowanie.json (mediany per checkpoint i blok).
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
import wzmocnienie_eden as WZ  # noqa: E402

C = 192
N_ODCZYT = 8
KOL_S = ['nN_nJ', 'sigma1', 'rho', 'c5', 'PR_N', 'cos_vu', 'zawor', 'G1']
KOL_O = ['cos2_u_r', 'cos2_v_r', 'cos2_los_r', 'norma_r', 'E_u_glowa', 'E_v_glowa', 'E_los_glowa', 'E_u_ramka', 'E_v_ramka', 'k_glowa']
KOL_O2 = ['cos2_u_nextv', 'cos2_v_nextv', 'cos2_u_nextu']
KOL_R = ['zywe', 'gelu_sr', 'gelu_zero', 'gelu_jeden']
KOL_BLK = ['nN_nJ_W2W1', 'nN_nJ_W2gW1', 'cos_w1w2_sr', 'cos_w1w2_waz', 'cos_w1w2_q90', 'n_neuronow_g_gt_0_5']


def schur_i_para(J):
    """J (192, 192) float64 → wektor KOL_S."""
    lam, V = np.linalg.eig(J)
    Q, R = np.linalg.qr(V)
    T = R @ np.diag(lam) @ np.linalg.inv(R)
    N = T - np.diag(np.diag(T))
    sN = np.linalg.svd(N, compute_uv=False); s2 = sN ** 2; tot = s2.sum()
    c5 = float(np.sort(s2)[::-1][:5].sum() / max(tot, 1e-300)); prN = float(tot ** 2 / max((s2 ** 2).sum(), 1e-300))
    nJ = float(np.sqrt((J * J).sum())); nN = float(np.sqrt(tot).real)
    A = np.eye(C) + J
    U, S, Vh = np.linalg.svd(J)                                        # para osobliwa POLA J (nie I+J: przy małym J I+J ≈ I i u₁ ≈ v₁ zawsze)
    u1, v1, s1 = U[:, 0], Vh[0], float(S[0])
    rho = float(np.abs(1 + lam).max())
    zawor = float(np.linalg.norm(J @ u1) / s1)                         # drugie przejście przez pole: feedforward → ≈ 0, normalne → 1
    sA = float(np.linalg.norm(A, 2))
    return np.array([nN / max(nJ, 1e-30), s1, rho, c5, prN, abs(float(u1 @ v1)), zawor, sA / rho]), u1, v1


def pomiar(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    U = WZ.Uklad(args.uklad, args.wagi, args.dane, dev)
    if args.proba:
        U.X = U.X[:args.proba]; U.ident = U.ident[:args.proba]
        if U.y is not None: U.y = U.y[:args.proba]
        if U.narz is not None: U.narz, U.tlo, U.najw = U.narz[:args.proba], U.tlo[:args.proba], U.najw[:args.proba]
    N = len(U.X); rng = np.random.default_rng(2026); torch.manual_seed(2026)
    S_tab, O_tab, O2_tab, R_tab, BLK_tab, czasy = [], [], [], [], [], {}
    pula = Pool(max(1, args.cpus))
    print(f'układ {args.uklad}, wagi {args.wagi}, kroki {args.kroki}, klatki/obrazy {N}, {dev}', flush=True)
    try:
        for k in args.kroki:
            t0 = time.time(); net = U.wczytaj(k); NB = len(net.bloki)
            # --- per blok: wagi i wzajemność reflektorów
            gsum = [torch.zeros(768, device=dev, dtype=torch.float64) for _ in range(NB)]; gcnt = 0
            zad, meta, odczyt = [], [], []
            for s in range(0, N, args.paczka):
                x = torch.from_numpy(U.X[s:s + args.paczka]).to(dev).permute(0, 3, 1, 2).float() / 255
                with torch.no_grad():
                    stany, XM, AM, o = U.przejscie(net, x)
                T = stany.shape[2]
                for i in range(len(x)):
                    tok, jest_narz = U.tokeny(s + i, T, rng); tok_t = torch.as_tensor(tok, device=dev)
                    sel = rng.choice(len(tok), min(N_ODCZYT, len(tok)), replace=False)
                    for b, blk in enumerate(net.bloki):
                        with torch.no_grad():
                            J = PM.jakobiany(blk, XM[b, i], AM[b, i], tok_t).cpu().numpy()
                            a = AM[b, i].double(); gp = 0.5 * (1 + torch.erf(a / math.sqrt(2.0))) + a * torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
                            gsum[b] += gp.mean(0);
                            gpt = gp[tok_t]                                                     # (n, 768)
                            zywe = ((gpt > 0.1) & (gpt < 0.9)).sum(1); zero = (gpt <= 0.1).sum(1); jeden = (gpt >= 0.9).sum(1)
                            for j in range(len(tok)):
                                R_tab.append([k, s + i, b, int(tok[j]), float(jest_narz[j]), float(zywe[j]), float(gpt[j].mean()), float(zero[j]), float(jeden[j])])
                                zad.append(J[j]); meta.append((s + i, b, int(tok[j]), float(jest_narz[j])))
                        # --- odczyt: gradient wyjścia tokenu po stanie po bloku b (przez resztę sieci), 8 tokenów
                        xm = stany[2 * b + 2, i].detach()[None].clone().requires_grad_(True)             # (1, T, 192) po MLP bloku b
                        o1 = U.reszta(net, xm, b + 1)[0]                                                 # (T, 5) albo (10,)
                        for j in sel:
                            tj = int(tok[j])
                            if U.rodzaj == 'detekcja':
                                cele = [o1[tj, q] for q in range(5)]                                  # logit ciepła + 4 odległości ramki
                                y = None
                            else:
                                y = int(U.y[s + i]); l = o1.clone(); l[y] = -1e9
                                cele = [o1[y] - l.max()] + [o1[q] for q in range(o1.shape[0])]      # margines + 10 logitów
                            grads = []
                            for cel in cele:
                                g, = torch.autograd.grad(cel, xm, retain_graph=True)
                                grads.append(g[0, tj].detach().double().cpu().numpy())
                            r = grads[0]; nr = float(np.linalg.norm(r))
                            Rm = np.stack(grads[1:] if y is not None else grads)                     # (k, 192): podprzestrzeń odczytu głowy
                            Qg, _ = np.linalg.qr(Rm.T); Qg = Qg[:, :np.linalg.matrix_rank(Rm)]
                            if U.rodzaj == 'detekcja':
                                Rr = np.stack(grads[1:5]); Qr, _ = np.linalg.qr(Rr.T); Qr = Qr[:, :np.linalg.matrix_rank(Rr)]
                            else:
                                Qr = None
                            odczyt.append((s + i, b, tj, r / max(nr, 1e-30), nr, Qg, Qr))
                gcnt += len(x)
            for b, blk in enumerate(net.bloki):
                W1 = blk.mlp.fc1.weight.detach().double(); W2 = blk.mlp.fc2.weight.detach().double(); g = gsum[b] / max(gcnt, 1)
                def nn_nj(M):
                    Mn = M.cpu().numpy(); lam = np.linalg.eigvals(Mn); nJ = np.sqrt((Mn * Mn).sum()); return float(np.sqrt(max(nJ * nJ - (np.abs(lam) ** 2).sum(), 0)) / max(nJ, 1e-30))
                cs = torch.nn.functional.cosine_similarity(W1, W2.T, dim=1).abs()                       # (768,) |cos(w1_i, w2_i)|
                BLK_tab.append([k, b, nn_nj(W2 @ W1), nn_nj(W2 @ torch.diag(g) @ W1), float(cs.mean()), float((cs * g.abs()).sum() / g.abs().sum().clamp_min(1e-30)),
                                float(cs.quantile(0.9)), float((g > 0.5).sum())])
            tG = time.time() - t0
            wyn = pula.map(schur_i_para, zad, chunksize=4)
            klucz = {}
            for (i, b, tj, narz), (m, u1, v1) in zip(meta, wyn):
                S_tab.append([k, i, b, tj, narz] + list(m)); klucz[(i, b, tj)] = (u1, v1)
            for (i, b, tj, r, nr, Qg, Qr) in odczyt:
                u1, v1 = klucz[(i, b, tj)]; los = rng.normal(size=C); los /= np.linalg.norm(los)
                E = lambda Q, x: float(((Q.T @ x) ** 2).sum()) if Q is not None else np.nan
                O_tab.append([k, i, b, tj, float(u1 @ r) ** 2, float(v1 @ r) ** 2, float(los @ r) ** 2, nr,
                              E(Qg, u1), E(Qg, v1), E(Qg, los), E(Qr, u1), E(Qr, v1), float(Qg.shape[1])])
            for (i, b, tj), (u1, v1) in klucz.items():                                                # łańcuch między blokami
                if (i, b + 1, tj) in klucz:
                    u2, v2 = klucz[(i, b + 1, tj)]
                    O2_tab.append([k, i, b, tj, float(u1 @ v2) ** 2, float(v1 @ v2) ** 2, float(u1 @ u2) ** 2])
            czasy[str(k)] = dict(GPU_s=round(tG, 1), Schur_s=round(time.time() - t0 - tG, 1), n_mac=len(zad))
            print(f'k{k}: {len(zad)} macierzy, GPU {tG:.0f} s, Schur {time.time() - t0 - tG:.0f} s', flush=True)
    finally:
        pula.close(); pula.join()
    S = np.array(S_tab); O = np.array(O_tab); O2 = np.array(O2_tab); R = np.array(R_tab); BLK = np.array(BLK_tab)
    np.savez_compressed(out / 'wyniki.npz', S=S, O=O, O2=O2, R=R, BLK=BLK, kroki=np.array(args.kroki), ident=np.array(U.ident),
                        kol_S=np.array(['krok', 'klatka', 'blok', 'token', 'narz'] + KOL_S), kol_O=np.array(['krok', 'klatka', 'blok', 'token'] + KOL_O), kol_O2=np.array(['krok', 'klatka', 'blok', 'token'] + KOL_O2),
                        kol_R=np.array(['krok', 'klatka', 'blok', 'token', 'narz'] + KOL_R), kol_BLK=np.array(['krok', 'blok'] + KOL_BLK))
    (out / 'podsumowanie.json').write_text(json.dumps(podsumuj(S, O, R, BLK, args, czasy, O2), indent=1, ensure_ascii=False), encoding='utf-8')
    print('zapisano', out, flush=True)


def podsumuj(S, O, R, BLK, args, czasy, O2=None):
    q = PM._q
    s = dict(uklad=args.uklad, wagi=str(args.wagi), kroki=[int(k) for k in args.kroki], czasy=czasy, S={}, O={}, O2={}, R={}, BLK={})
    for k in args.kroki:
        w = S[S[:, 0] == k]; NB = int(w[:, 2].max()) + 1 if len(w) else 0
        s['S'][str(k)] = {m: [q(w[w[:, 2] == b, 5 + j]) for b in range(NB)] for j, m in enumerate(KOL_S)}
        o = O[O[:, 0] == k]; s['O'][str(k)] = {m: [q(o[o[:, 2] == b, 4 + j]) for b in range(NB)] for j, m in enumerate(KOL_O)}
        if O2 is not None and len(O2):
            o2 = O2[O2[:, 0] == k]; s['O2'][str(k)] = {m: [q(o2[o2[:, 2] == b, 4 + j]) for b in range(NB - 1)] for j, m in enumerate(KOL_O2)}
        r = R[R[:, 0] == k]; s['R'][str(k)] = {m: [q(r[r[:, 2] == b, 5 + j]) for b in range(NB)] for j, m in enumerate(KOL_R)}
        bl = BLK[BLK[:, 0] == k]; s['BLK'][str(k)] = {m: [float(bl[bl[:, 1] == b, 2 + j][0]) for b in range(NB)] for j, m in enumerate(KOL_BLK)}
        # korelacja (per token) żywych reflektorów z G1 w bloku środkowym
        b = NB // 2
        ws = w[w[:, 2] == b]; rs = r[r[:, 2] == b]
        if len(ws) and len(ws) == len(rs):
            s['S'][str(k)]['rho_zywe_G1_bstar'] = float(np.corrcoef(rs[:, 5], ws[:, 5 + KOL_S.index('G1')])[0, 1])
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uklad', required=True, choices=['detekcja', 'cifar_maly', 'cifar_deit'])
    p.add_argument('--wagi', required=True); p.add_argument('--dane', required=True); p.add_argument('--out', required=True)
    p.add_argument('--kroki', type=int, nargs='+', required=True); p.add_argument('--paczka', type=int, default=10)
    p.add_argument('--cpus', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    p.add_argument('--proba', type=int, default=0)
    pomiar(p.parse_args())


if __name__ == '__main__':
    main()
