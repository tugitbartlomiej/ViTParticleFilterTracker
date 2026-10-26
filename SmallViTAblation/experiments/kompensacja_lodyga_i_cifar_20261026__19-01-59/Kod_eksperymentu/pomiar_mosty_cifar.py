"""Pomiar „mostów do fizyki” na checkpointach biegów klasyfikacji CIFAR-10 (DeiT-Tiny z łatką 4: 64 tokeny + CLS, 12 bloków;
MałyViT na obrazie 32 px ×4 = 128 px: 64 tokeny, 6 bloków, głowa na średniej tokenów). Drugi zbiór i druga architektura dla tez
A (nienormalność pola MLP), B (uwaga ściska, MLP rozdziela, trening przenosi ściskanie) i C (kolejność uwaga → MLP) z eksperymentu
mosty_fizyki_24seedy_ablacje_20261026__13-04-31. Miary A i B liczy ten sam kod co tam (import z pomiar_mosty_eden.py); zmienia się
tylko model, dane i odczyt wyjścia (dokładność top-1 zamiast IoU/AP).

Stany przejścia: 0 = po łodydze (patch_embed + CLS + pos_embed w DeiT; 4 konwolucje + sin-cos w MałymViT), 2b+1 = po uwadze bloku b,
2b+2 = po MLP bloku b; 2·NB półkroków. Tokeny: DeiT 65 (CLS = indeks 0, oznaczony flagą 'cls'), MałyViT 64. Maska „narz” z rodzica
to tu CLS (DeiT) albo pusta (MałyViT): miary klasowe per obraz z rodzica (sep_u, D_u_nn, cos_nt) są NaN, bo klasa jest cechą obrazu,
nie tokenu. Zamiast nich separacja klas na poziomie obrazów (sekcja „sep” niżej).

A: pełne jakobiany 192 × 192 pola MLP każdego bloku dla 16 losowych tokenów obrazu (+ CLS w DeiT) na obrazach diagnostycznych
   (10 na klasę): te same 12 miar co w rodzicu (‖J‖, ‖N‖, ‖N‖/‖J‖, ‖Ω‖/‖S‖, tr J, znaki widma, |1+λ|>1, max Re λ, Q, R).
B: per obraz i stan: D_u, D_x (średnia odległość par tokenów na sferze / surowa), r (porządek), PR_u (ranga efektywna); per półkrok:
   mediana ‖Δx‖, udział styczny, zachowanie 5 sąsiadów (kod rodzica; kolumny klasowe NaN).
sep (per checkpoint i stan, na całym zbiorze N obrazów): osadzenie obrazu = średnia tokenów obrazu (bez CLS) albo CLS (DeiT),
   na sferze u (LayerNorm bez afinicznych) i surowe X; separacja = średnia po parach klas ‖μ_c − μ_c'‖ / √(średnia wariancja
   wewnątrzklasowa); dokładność najbliższego centroidu (ten sam zbiór, miara opisowa).
C: per blok: ‖S−R‖/‖S−X‖, ‖S−P‖/‖S−X‖, cos(S−X, P−X) (mediany po tokenach obrazu) oraz skutek dla wyjścia po przepuszczeniu reszty
   sieci: poprawność top-1 i margines (logit klasy prawdziwej − maks. pozostałych) dla R, P, A_only, M_only.
Dokładność top-1 na N obrazach testowych (200 na klasę, ziarno 2026) per checkpoint.

Wyjście: <out>/wyniki.npz i <out>/podsumowanie.json (mediany per checkpoint, blok, stan; stosunki D_u, iloczyny, Δsep).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402  (MałyViT: ten sam plik co w kampanii i w biegu CIFAR)
import pomiar_mosty_eden as PM  # noqa: E402
from model_deit_cifar import DeitCifar, MEAN as DMEAN, STD as DSTD  # noqa: E402

SIATKA = [0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65000, 80000, 93780]
N_NA_KLASE, N_DIAG_NA_KLASE, N_TOK_JAK = 200, 10, 16
MIARY_A = PM.MIARY_A
MIARY_B_STAN = PM.MIARY_B_STAN
MIARY_B_POL = PM.MIARY_B_POL
MIARY_C = ['SR', 'SP', 'cosSP', 'ok_R', 'ok_P', 'ok_A', 'ok_M', 'marg_R', 'marg_P', 'marg_A', 'marg_M']
MIARY_SEP = ['sep_u_mean', 'sep_x_mean', 'sep_u_cls', 'acc_centroid_u_mean', 'acc_centroid_u_cls']


# ---------------------------------------------------------------- modele
class MalyViTCifar(nn.Module):
    """Klasyfikator('vit') z trening_klasyfikacja.py: szkielet MałyViT (głowa gęsta = Identity) + głowa liniowa na średniej tokenów;
    obraz 32 px powiększony dwuliniowo ×4 (jak w treningu). Klucze state_dict: szkielet.*, glowa.*."""

    def __init__(self, n_klas=10, skala=4):
        super().__init__()
        self.szkielet = M.MalyViT()
        self.szkielet.glowa = nn.Identity()
        self.glowa = nn.Linear(192, n_klas)
        self.skala = skala
        self.cls = False

    @property
    def bloki(self):
        return self.szkielet.bloki

    def stan0(self, x):
        x = F.interpolate(x, scale_factor=self.skala, mode='bilinear', align_corners=False)
        x = (x - M.MEAN.to(x.device, x.dtype)) / M.STD.to(x.device, x.dtype)
        f = self.szkielet.lodyga(x)
        B, C, h, w = f.shape
        return f.flatten(2).transpose(1, 2) + M.sincos_2d(h, w, C, x.device).to(f.dtype)

    def wyjscie(self, t):
        return self.glowa(self.szkielet.norm(t).mean(1))


class DeitCifarPomiar(DeitCifar):
    """DeiT-Tiny (timm VisionTransformer) z rozbiciem na stan 0, bloki i głowę na CLS."""

    def __init__(self):
        super().__init__()
        self.cls = True

    def stan0(self, x):
        x = (x - DMEAN.to(x.device, x.dtype)) / DSTD.to(x.device, x.dtype)
        v = self.vit
        t = v.patch_embed(x)
        t = v._pos_embed(t)                  # CLS + pos_embed (+ pos_drop = Identity w eval)
        t = v.patch_drop(t)
        return v.norm_pre(t)

    def wyjscie(self, t):
        return self.vit.forward_head(self.vit.norm(t))


def wczytaj_siec(sciezka, rodzaj, dev):
    net = (DeitCifarPomiar() if rodzaj == 'deit' else MalyViTCifar()).to(dev).eval().requires_grad_(False)
    st = torch.load(sciezka, map_location='cpu', weights_only=False)
    net.load_state_dict(st.get('net', st), strict=True)
    return net, int(st.get('krok', -1))


def przejscie(net, x):
    """Stany (2NB+1, B, T, 192), wejścia MLP (NB, B, T, 192), przed-aktywacje (NB, B, T, 768), logity (B, 10)."""
    t = net.stan0(x)
    stany, XM, AM = [t], [], []
    for blk in net.bloki:
        xa, xm, xin, a = PM.polkroki(blk, t, 'base')
        stany += [xa, xm]; XM.append(xin); AM.append(a); t = xm
    return torch.stack(stany), torch.stack(XM), torch.stack(AM), net.wyjscie(t)


def reszta_sieci(net, t, od_bloku):
    for blk in list(net.bloki)[od_bloku:]:
        t = PM.polkroki(blk, t, 'base')[1]
    return net.wyjscie(t)


# ---------------------------------------------------------------- dane
def wczytaj_cifar(sciezka, n_na_klase, n_diag_na_klase, seed=2026):
    d = np.load(sciezka)
    X, y = d['X_test'], d['y_test'].astype(np.int64)
    rng = np.random.default_rng(seed)
    idx, idx_d = [], []
    for c in range(10):
        w = rng.permutation(np.flatnonzero(y == c))
        idx.append(w[:n_na_klase]); idx_d.append(w[:n_diag_na_klase])
    idx = np.sort(np.concatenate(idx)); idx_d = np.sort(np.concatenate(idx_d))
    return idx, X[idx], y[idx], idx_d, X[idx_d], y[idx_d]


# ---------------------------------------------------------------- sep: separacja klas na poziomie obrazów
def separacja_klas(E, y):
    """E (N, 192) osadzenia obrazów, y (N,) → (separacja, dokładność najbliższego centroidu)."""
    kl = [c for c in range(10) if (y == c).sum() >= 2]                               # klasy z ≥ 2 obrazami (próba dymna ma ich mniej)
    if len(kl) < 2:
        return float('nan'), float('nan')
    mu = torch.stack([E[y == c].mean(0) for c in kl]); ind = torch.full((10,), -1, dtype=torch.long); ind[kl] = torch.arange(len(kl))
    m = ind[y] >= 0; y = ind[y[m]]; E = E[m]
    war = ((E - mu[y]) ** 2).sum(1).mean()
    D = torch.cdist(mu, mu); iu = torch.triu_indices(len(kl), len(kl), 1)
    sep = D[iu[0], iu[1]].mean() / war.clamp_min(1e-12).sqrt()
    acc = (torch.cdist(E, mu).argmin(1) == y).float().mean()
    return float(sep), float(acc)


def miary_sep(stany_all, y, cls):
    """stany_all (S, N, T, 192) na CPU (float32), y (N,) → (S, len(MIARY_SEP))."""
    S = stany_all.shape[0]; out = np.full((S, len(MIARY_SEP)), np.nan, np.float32)
    y = torch.as_tensor(y)
    for s in range(S):
        X = stany_all[s]
        pat = X[:, 1:] if cls else X
        em_x = pat.mean(1); em_u = PM.sfera(em_x)
        out[s, 0], out[s, 3] = separacja_klas(em_u, y)
        out[s, 1], _ = separacja_klas(em_x, y)
        if cls:
            out[s, 2], out[s, 4] = separacja_klas(PM.sfera(X[:, 0]), y)
    return out


# ---------------------------------------------------------------- C: komutator i skutek dla wyjścia
def ok_marg(logity, y):
    l = logity.float(); ok = (l.argmax(1) == y).float()
    lt = l.gather(1, y[:, None])[:, 0]; l2 = l.clone(); l2.scatter_(1, y[:, None], -1e9)
    return ok.cpu().numpy(), (lt - l2.max(1).values).cpu().numpy()


def miary_C(net, stany, y):
    B = stany.shape[1]; NB = len(net.bloki)
    out = np.full((B, NB, len(MIARY_C)), np.nan, np.float32)
    for b, blk in enumerate(net.bloki):
        t = stany[2 * b]
        S, R, P, A_only, M_only = PM.warianty_bloku(blk, t, 'base')
        dS = S - t; nS = dS.norm(dim=-1).clamp_min(1e-12)
        sr = ((S - R).norm(dim=-1) / nS).median(1).values; sp = ((S - P).norm(dim=-1) / nS).median(1).values
        cs = F.cosine_similarity(dS, P - t, dim=-1).median(1).values
        kol = [sr.cpu().numpy(), sp.cpu().numpy(), cs.cpu().numpy()]
        oks, margs = [], []
        for alt in (R, P, A_only, M_only):
            o, m = ok_marg(reszta_sieci(net, alt, b + 1), y); oks.append(o); margs.append(m)
        out[:, b] = np.stack(kol + oks + margs, 1)
    return out


# ---------------------------------------------------------------- główna pętla
def pomiar(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wagi = Path(args.wagi); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    kroki = [k for k in SIATKA if (wagi / f'k{k:05d}.pt').exists()] if args.kroki is None else args.kroki
    idx, X, y, idx_d, Xd, yd = wczytaj_cifar(args.dane, N_NA_KLASE, N_DIAG_NA_KLASE)
    if args.proba:
        idx, X, y = idx[:args.proba], X[:args.proba], y[:args.proba]; idx_d, Xd, yd = idx_d[:4], Xd[:4], yd[:4]
    N, Nd = len(X), len(Xd)
    net0, _ = wczytaj_siec(wagi / f'k{kroki[0]:05d}.pt', args.model, dev)
    NB = len(net0.bloki); cls = net0.cls; T = 65 if cls else 64; S = 2 * NB + 1
    del net0
    print(f'wagi {wagi} model {args.model} bloki {NB} tokeny {T}; kroki {kroki}; obrazy {N}, diagnostyczne {Nd}; {dev}', flush=True)
    rng = np.random.default_rng(2026)
    wyb = []
    for i in range(Nd):
        los = np.sort(rng.choice(np.arange(1 if cls else 0, T), N_TOK_JAK, replace=False))
        wyb.append((np.r_[[0], los] if cls else los, np.r_[np.ones(1, bool), np.zeros(len(los), bool)] if cls else np.zeros(len(los), bool)))
    narz = np.zeros((N, T), bool); tlo = np.ones((N, T), bool)
    narz[:, 0] = True                       # DeiT: CLS; MałyViT: token 0 jako zapychacz (miary_B rodzica wymagają ≥ 1 tokena „narz”)
    if cls:
        tlo[:, 0] = False
    narz_t = torch.as_tensor(narz, device=dev); tlo_t = torch.as_tensor(tlo, device=dev)
    B_stan = np.full((len(kroki), N, S, len(MIARY_B_STAN)), np.nan, np.float32)
    B_pol = np.full((len(kroki), N, S - 1, len(MIARY_B_POL)), np.nan, np.float32)
    Cm = np.full((len(kroki), N, NB, len(MIARY_C)), np.nan, np.float32)
    SEP = np.full((len(kroki), S, len(MIARY_SEP)), np.nan, np.float32)
    ok = np.full((len(kroki), N), np.nan, np.float32); marg = np.full((len(kroki), N), np.nan, np.float32); acc = np.full(len(kroki), np.nan)
    A_tab = []; czasy = {}
    y_t = torch.as_tensor(y, device=dev)
    pula = Pool(max(1, args.cpus))
    try:
        for ik, k in enumerate(kroki):
            t0 = time.time()
            net, krok = wczytaj_siec(wagi / f'k{k:05d}.pt', args.model, dev)
            assert krok in (k, -1), (krok, k)
            stany_cpu = torch.empty((S, N, T, 192), dtype=torch.float32)
            with torch.no_grad():
                for s in range(0, N, args.paczka):
                    x = torch.from_numpy(X[s:s + args.paczka]).to(dev).permute(0, 3, 1, 2).float() / 255
                    sl = slice(s, s + len(x))
                    stany, XM, AM, o = przejscie(net, x)
                    ok[ik, sl], marg[ik, sl] = ok_marg(o, y_t[sl])
                    bs, bp = PM.miary_B(stany, narz_t[sl], tlo_t[sl])
                    B_stan[ik, sl] = bs; B_pol[ik, sl] = bp
                    Cm[ik, sl] = miary_C(net, stany, y_t[sl])
                    stany_cpu[:, sl] = stany.float().cpu()
                acc[ik] = float(np.nanmean(ok[ik]))
                SEP[ik] = miary_sep(stany_cpu, y, cls)
                del stany_cpu
                tB = time.time() - t0
                zad = []
                for s in range(0, Nd, args.paczka):
                    x = torch.from_numpy(Xd[s:s + args.paczka]).to(dev).permute(0, 3, 1, 2).float() / 255
                    _, XM, AM, _ = przejscie(net, x)
                    for i in range(len(x)):
                        tok, jest_cls = wyb[s + i]
                        for b, blk in enumerate(net.bloki):
                            J = PM.jakobiany(blk, XM[b, i], AM[b, i], torch.as_tensor(tok, device=dev)).cpu().numpy()
                            zad.append((s + i, b, tok, jest_cls, J))
                widma = pula.map(PM._widmo, [z[4] for z in zad], chunksize=4)
                for (i, b, tok, jest_cls, J), lam in zip(zad, widma):
                    m = PM.zlozyc_A(J, lam)
                    A_tab.append(np.column_stack([np.full(len(tok), k), np.full(len(tok), i), np.full(len(tok), b), tok, jest_cls.astype(float), m]))
            czasy[str(k)] = dict(B_C_s=round(tB, 1), A_s=round(time.time() - t0 - tB, 1), n_jak=sum(len(z[2]) for z in zad))
            print(f'k{k}: acc {acc[ik]:.3f}, sep_u_mean końcowa {SEP[ik, -1, 0]:.2f}, B+C {tB:.0f} s, A {time.time() - t0 - tB:.0f} s', flush=True)
    finally:
        pula.close(); pula.join()
    A_arr = np.concatenate(A_tab) if A_tab else np.zeros((0, 5 + len(MIARY_A)))
    np.savez_compressed(out / 'wyniki.npz', kroki=np.array(kroki), idx_obrazow=idx, y=y, idx_diag=idx_d, y_diag=yd, cls=np.array(cls), n_blokow=np.array(NB),
                        B_stan=B_stan, B_pol=B_pol, C=Cm, sep=SEP, ok=ok, marg=marg, acc=acc, A=A_arr,
                        miary_A=np.array(['krok', 'obraz', 'blok', 'token', 'cls'] + MIARY_A), miary_B_stan=np.array(MIARY_B_STAN),
                        miary_B_pol=np.array(MIARY_B_POL), miary_C=np.array(MIARY_C), miary_sep=np.array(MIARY_SEP))
    (out / 'podsumowanie.json').write_text(json.dumps(podsumuj(kroki, B_stan, B_pol, Cm, SEP, acc, A_arr, args, czasy, NB, cls), indent=1, ensure_ascii=False),
                                           encoding='utf-8')
    print('zapisano', out, flush=True)


def podsumuj(kroki, B_stan, B_pol, Cm, SEP, acc, A_arr, args, czasy, NB, cls):
    q = PM._q; S = B_stan.shape[2]
    s = dict(wagi=str(args.wagi), model=args.model, bloki=NB, cls=bool(cls), kroki=[int(k) for k in kroki], n_obrazow=int(B_stan.shape[1]), czasy=czasy,
             acc={str(k): float(acc[i]) for i, k in enumerate(kroki)}, B_stan={}, B_pol={}, C={}, A={}, sep={})
    for i, k in enumerate(kroki):
        s['B_stan'][str(k)] = {m: [q(B_stan[i, :, st, j]) for st in range(S)] for j, m in enumerate(MIARY_B_STAN)}
        s['B_pol'][str(k)] = {m: [q(B_pol[i, :, p, j]) for p in range(S - 1)] for j, m in enumerate(MIARY_B_POL)}
        s['C'][str(k)] = {m: [q(Cm[i, :, b, j]) for b in range(NB)] for j, m in enumerate(MIARY_C)}
        s['sep'][str(k)] = {m: SEP[i, :, j].tolist() for j, m in enumerate(MIARY_SEP)}
        if len(A_arr):
            w = A_arr[A_arr[:, 0] == k]
            s['A'][str(k)] = {m: {kl: [q(w[(w[:, 2] == b) & (w[:, 4] == (1.0 if kl == 'cls' else 0.0)), 5 + j]) for b in range(NB)]
                                  for kl in ('cls', 'tok')} for j, m in enumerate(MIARY_A)}
        Du = np.nanmedian(B_stan[i, :, :, MIARY_B_STAN.index('D_u')], 0); st = Du[1:] / Du[:-1]
        sep = SEP[i, :, 0]; ds = sep[1:] - sep[:-1]
        s['B_pol'][str(k)]['stosunek_D_u_mediany'] = st.tolist()
        s['B_pol'][str(k)]['iloczyn_uwaga'] = float(np.prod(st[0::2])); s['B_pol'][str(k)]['iloczyn_mlp'] = float(np.prod(st[1::2]))
        s['B_pol'][str(k)]['dsep_uwaga'] = float(np.nansum(ds[0::2])); s['B_pol'][str(k)]['dsep_mlp'] = float(np.nansum(ds[1::2]))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wagi', required=True); p.add_argument('--dane', required=True, help='cifar10_uint8.npz'); p.add_argument('--out', required=True)
    p.add_argument('--model', required=True, choices=['deit', 'malyvit'])
    p.add_argument('--kroki', type=int, nargs='*', default=None); p.add_argument('--paczka', type=int, default=100)
    p.add_argument('--cpus', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    p.add_argument('--proba', type=int, default=0, help='tylko tyle obrazów (i 4 diagnostyczne): sprawdzenie kodu')
    pomiar(p.parse_args())


if __name__ == '__main__':
    main()
