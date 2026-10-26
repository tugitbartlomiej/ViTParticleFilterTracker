"""Pomiar „mostów do fizyki” na checkpointach jednego biegu MałegoViT (siatka 12 checkpointów kampanii kompensacji), bez zależności
od aplikacji: tylko numpy i torch (na Edenie nie ma scipy).

Wejście: katalog wag `k<krok>.pt` ({'net', 'krok', ...}), katalog danych kampanii (`dataset.json` + `images/*.npz`, klucz 'rgb',
320 × 480 × 3 uint8, ramki GT xyxy w pikselach 480 × 320), wariant architektury (jak w trening_ablacje.py: 'base', 'parallel', ...;
wariant wpływa tylko na to, skąd MLP bierze wejście: 'parallel' = z wejścia bloku, reszta = po uwadze).

Stany przejścia: 0 = po łodydze i kodowaniu pozycji, 2b+1 = po uwadze bloku b, 2b+2 = po MLP bloku b (b = 0..5); 12 półkroków.
Tokeny narzędzia: komórki wewnątrz ramki GT (poz > 0 w model.cele, tak jak w stracie), tło: heat < 0,05; środek pominięty.

A (klatki diagnostyczne, pełne jakobiany 192 × 192 pola MLP każdego bloku, tokeny: wszystkie narzędzia + 60 losowych tła na klatkę):
  ‖J‖_F, odchylenie od normalności Henriciego ‖N‖_F = √(‖J‖_F² − Σ|λ|²) (bez rozkładu Schura), ‖N‖/‖J‖, ‖Ω‖_F/‖S‖_F, tr J,
  liczby λ: Re > 0, Re < 0, |Im| > 0, |1 + λ| > 1 (kierunki rozciągane w jednym podkroku), max Re λ, Q = −½ tr A², R = −⅓ tr A³
  (A = J − tr J/192 · I).
B (wszystkie klatki, 13 stanów, sfera u = LayerNorm bez afinicznych): średnia odległość par (wszystkie, narzędzie–narzędzie, tło–tło)
  na u i na X, parametr porządku r = ‖mean u‖ (wszystkie / narzędzie / tło), cos średnich klas, separacja klas na u i na X, ranga
  efektywna (participation ratio) na u; na półkrok: mediana ‖Δx‖ narzędzia i tła, udział styczny ruchu, zachowanie 5 sąsiadów.
C (wszystkie klatki, każdy blok): S = M(A(X)), R = A(M(X)), P = X + attn + mlp, A_only, M_only; ‖S−R‖/‖S−X‖, ‖S−P‖/‖S−X‖,
  cos(S−X, P−X) per token (mediany narzędzie / tło), oraz skutek dla wyjścia po przepuszczeniu reszty sieci: IoU ramki szczytu z GT,
  maks. ciepło, korelacja mapy ciepła z mapą po S.
IoU/pewność ramki szczytu i AP@0,5 (101 punktów, jak w treningu) na wszystkich klatkach z GT.

Wyjście: <out>/wyniki.npz (tablice per klatka / token) i <out>/podsumowanie.json (mediany per checkpoint, blok, klasa).
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
import model as M  # noqa: E402

SIATKA = [0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65640]
H, W, T, C, NB = 20, 30, 600, 192, 6
WARIANTY = ('base', 'parallel', 'frozen_mlp', 'frozen_attn', 'no_wd', 'shuffled_labels', 'frozen_both')
KNN = 5
N_TLA_JAK = 60                      # losowych tokenów tła na klatkę do jakobianów (plus wszystkie narzędzia)
MIARY_A = ['nJ', 'nN', 'nN_nJ', 'om_s', 'trJ', 'n_plus', 'n_minus', 'n_cplx', 'n_exp', 'maxRe', 'Q', 'R']
MIARY_B_STAN = ['D_u', 'D_u_nn', 'D_u_tt', 'D_x', 'D_x_nn', 'D_x_tt', 'r', 'r_n', 'r_t', 'cos_nt', 'sep_u', 'sep_x', 'PR_u']
MIARY_B_POL = ['dx_n', 'dx_t', 'styczny', 'sasiedzi']
MIARY_C = ['SR', 'SP', 'cosSP', 'SR_n', 'SP_n', 'cosSP_n', 'iou_R', 'iou_P', 'iou_A', 'iou_M', 'cieplo_R', 'cieplo_P', 'cieplo_A',
           'cieplo_M', 'kor_R', 'kor_P', 'kor_A', 'kor_M']


# ---------------------------------------------------------------- dane
def wczytaj_klatki(dane, rola, tylko_diag):
    rows = json.loads((dane / 'dataset.json').read_text(encoding='utf-8'))['rows']
    rows = [r for r in rows if r['role'] == rola and (r['diagnostic'] or not tylko_diag) and len(r['gt']) > 0]
    X = np.stack([np.load(dane / r['image'])['rgb'] for r in rows])                       # (N, 320, 480, 3) uint8
    gt = [np.asarray(r['gt'], np.float32).reshape(-1, 4) for r in rows]
    ident = [r['id'] for r in rows]
    return ident, X, gt


def maski(gt):
    """narz (N, 600): komórki wewnątrz ramki GT (poz > 0 jak w stracie); tlo: heat < 0,05; największa ramka GT (N, 4)."""
    heat, _, poz = M.cele([np.asarray(g, np.float32) for g in gt], H, W)
    narz = (poz > 0).reshape(len(gt), -1).numpy(); tlo = (heat < 0.05).reshape(len(gt), -1).numpy()
    najw = np.stack([g[np.argmax((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))] for g in gt])
    return narz, tlo, najw


def iou_xyxy(b, g):
    x0, y0 = np.maximum(b[..., 0], g[..., 0]), np.maximum(b[..., 1], g[..., 1])
    x1, y1 = np.minimum(b[..., 2], g[..., 2]), np.minimum(b[..., 3], g[..., 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) + (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1]) - inter + 1e-9)


def ap50(detekcje, ramki_gt):
    """AP@0,5 jednej klasy, interpolacja 101-punktowa (kopia trening.ap50 z biegu 35 000)."""
    wszystkie, n_gt = [], 0
    for i, ((bx, sc), gt) in enumerate(zip(detekcje, ramki_gt)):
        n_gt += len(gt)
        for b, s in zip(bx, sc):
            wszystkie.append((float(s), i, b))
    wszystkie.sort(key=lambda x: -x[0])
    uzyte = [np.zeros(len(g), bool) for g in ramki_gt]
    tp = np.zeros(len(wszystkie))
    for j, (s, i, b) in enumerate(wszystkie):
        g = ramki_gt[i]
        if len(g):
            v = np.array([iou_xyxy(b, gg) for gg in g]); v[uzyte[i]] = -1
            k = int(v.argmax())
            if v[k] >= .5:
                tp[j] = 1; uzyte[i][k] = True
    ctp = np.cumsum(tp); rec = ctp / max(n_gt, 1); prec = ctp / np.arange(1, len(tp) + 1)
    for j in range(len(prec) - 2, -1, -1):
        prec[j] = max(prec[j], prec[j + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side='left')
        ap += prec[k] if k < len(prec) else 0.0
    return ap / 101


# ---------------------------------------------------------------- sieć: półkroki i warianty
def wczytaj_siec(sciezka, dev):
    net = M.MalyViT().to(dev).eval().requires_grad_(False)
    st = torch.load(sciezka, map_location='cpu', weights_only=False)
    net.load_state_dict(st.get('net', st), strict=True)
    return net, int(st.get('krok', -1))


def wejscie_blokow(net, x):
    """Tokeny po łodydze i kodowaniu pozycji (stan 0)."""
    x = (x - M.MEAN.to(x.device, x.dtype)) / M.STD.to(x.device, x.dtype)
    f = net.lodyga(x)
    B, Cc, h, w = f.shape
    return f.flatten(2).transpose(1, 2) + M.sincos_2d(h, w, Cc, x.device).to(f.dtype)


def polkroki(blk, t, wariant):
    """(po uwadze, po MLP, wejście MLP, przed-aktywacje fc1) dla jednego bloku."""
    xa = t + blk.ls1(blk.attn(blk.norm1(t)))
    xin = t if wariant == 'parallel' else xa
    a = blk.mlp.fc1(blk.norm2(xin))
    xm = xa + blk.ls2(blk.mlp.drop2(blk.mlp.fc2(blk.mlp.drop1(blk.mlp.act(a)))))
    return xa, xm, xin, a


def przejscie(net, x, wariant):
    """Stany (13, B, 600, 192), wejścia MLP (6, B, 600, 192), przed-aktywacje (6, B, 600, 768), wyjście głowy (B, 600, 5)."""
    t = wejscie_blokow(net, x)
    stany, XM, AM = [t], [], []
    for blk in net.bloki:
        xa, xm, xin, a = polkroki(blk, t, wariant)
        stany += [xa, xm]; XM.append(xin); AM.append(a); t = xm
    o = net.glowa(net.norm(t))
    return torch.stack(stany), torch.stack(XM), torch.stack(AM), o


def reszta_sieci(net, t, od_bloku, wariant):
    for blk in list(net.bloki)[od_bloku:]:
        t = polkroki(blk, t, wariant)[1]
    return net.glowa(net.norm(t))


def warianty_bloku(blk, t, wariant):
    """S, R, P, A_only, M_only dla bloku z wejściem t (B, 600, 192)."""
    S = polkroki(blk, t, wariant)[1]
    y = t + blk.ls2(blk.mlp(blk.norm2(t)))                                    # najpierw MLP
    R = y + blk.ls1(blk.attn(blk.norm1(y)))
    A_only = t + blk.ls1(blk.attn(blk.norm1(t)))
    M_only = y
    P = A_only + (y - t)
    return S, R, P, A_only, M_only


# ---------------------------------------------------------------- głowa: ramka szczytu, IoU, mapy
def ramki_szczytu(o):
    """o (B, 600, 5) → ramka top-1 (B, 4), pewność (B,), mapa ciepła (B, 600), 10 ramek do AP."""
    bx, sc = M.dekoduj(o.reshape(-1, H, W, 5).float(), k=10)
    return bx[:, 0].cpu().numpy(), sc[:, 0].cpu().numpy(), o[..., 0].float().sigmoid().cpu().numpy(), bx.cpu().numpy(), sc.cpu().numpy()


def korelacja_wierszy(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


# ---------------------------------------------------------------- B: sfera, odległości, porządek
def sfera(X):
    c = X - X.mean(-1, keepdim=True)
    return c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def srednia_odleglosc(D, m1, m2):
    """Średnia odległość par (i ∈ m1, j ∈ m2, i ≠ j) per klatka; NaN gdy brak par."""
    w = (m1[:, :, None] & m2[:, None, :]).float()
    w = w * (1 - torch.eye(D.shape[1], device=D.device))[None]
    n = w.sum((1, 2))
    return torch.where(n > 0, (D * w).sum((1, 2)) / n.clamp_min(1), torch.full_like(n, float('nan')))


def miary_B(stany, narz, tlo):
    """stany (13, B, 600, 192) float32 na GPU; narz, tlo (B, 600) bool → stan (B, 13, len(MIARY_B_STAN)), polkrok (B, 12, len(MIARY_B_POL))."""
    S, B = stany.shape[0], stany.shape[1]
    out_s = torch.full((B, S, len(MIARY_B_STAN)), float('nan'), device=stany.device)
    out_p = torch.full((B, S - 1, len(MIARY_B_POL)), float('nan'), device=stany.device)
    wsz = torch.ones_like(narz)
    prev_u, prev_knn = None, None
    for s in range(S):
        X = stany[s]; U = sfera(X)
        Du = torch.cdist(U, U); Dx = torch.cdist(X, X)
        mu = U.mean(1)
        m_n = narz.float()[..., None]; m_t = tlo.float()[..., None]
        nn_ = narz.sum(1).clamp_min(1)[:, None]; nt_ = tlo.sum(1).clamp_min(1)[:, None]
        mu_n = (U * m_n).sum(1) / nn_; mu_t = (U * m_t).sum(1) / nt_
        mx_n = (X * m_n).sum(1) / nn_; mx_t = (X * m_t).sum(1) / nt_
        var_n = (((U - mu_n[:, None]) ** 2).sum(-1) * narz).sum(1) / nn_[:, 0]; var_t = (((U - mu_t[:, None]) ** 2).sum(-1) * tlo).sum(1) / nt_[:, 0]
        vx_n = (((X - mx_n[:, None]) ** 2).sum(-1) * narz).sum(1) / nn_[:, 0]; vx_t = (((X - mx_t[:, None]) ** 2).sum(-1) * tlo).sum(1) / nt_[:, 0]
        ma = narz.sum(1) >= 2
        cov = torch.einsum('bti,btj->bij', U - mu[:, None], U - mu[:, None]) / T
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        kol = [srednia_odleglosc(Du, wsz, wsz), srednia_odleglosc(Du, narz, narz), srednia_odleglosc(Du, tlo, tlo),
               srednia_odleglosc(Dx, wsz, wsz), srednia_odleglosc(Dx, narz, narz), srednia_odleglosc(Dx, tlo, tlo),
               mu.norm(dim=1), torch.where(ma, mu_n.norm(dim=1), torch.nan), mu_t.norm(dim=1),
               torch.where(ma, torch.nn.functional.cosine_similarity(mu_n, mu_t, dim=1), torch.nan),
               torch.where(ma, (mu_n - mu_t).norm(dim=1) / (0.5 * (var_n + var_t)).clamp_min(1e-12).sqrt(), torch.nan),
               torch.where(ma, (mx_n - mx_t).norm(dim=1) / (0.5 * (vx_n + vx_t)).clamp_min(1e-12).sqrt(), torch.nan),
               ev.sum(1) ** 2 / (ev ** 2).sum(1).clamp_min(1e-30)]
        out_s[:, s] = torch.stack(kol, 1)
        Dk = Du.clone(); Dk.diagonal(dim1=1, dim2=2).fill_(float('inf'))
        knn = Dk.topk(KNN, dim=2, largest=False).indices                                   # (B, 600, k)
        if s > 0:
            dx = X - stany[s - 1]
            u_prev = prev_u
            rad = (dx * u_prev).sum(-1, keepdim=True) * u_prev                            # składowa wzdłuż kierunku tokenu na sferze
            tang = ((dx - rad) ** 2).sum(-1).sum(1) / (dx ** 2).sum(-1).sum(1).clamp_min(1e-30)
            ndx = dx.norm(dim=-1)
            med_n = torch.stack([torch.where(narz[b].any(), ndx[b][narz[b]].median(), torch.nan) for b in range(B)])
            med_t = torch.stack([ndx[b][tlo[b]].median() for b in range(B)])
            wspolne = (prev_knn[:, :, :, None] == knn[:, :, None, :]).any(-1).float().mean(-1).mean(1)
            out_p[:, s - 1] = torch.stack([med_n, med_t, tang, wspolne], 1)
        prev_u, prev_knn = U, knn
    return out_s.cpu().numpy(), out_p.cpu().numpy()


# ---------------------------------------------------------------- C: komutator
def miary_C(net, stany, narz, tlo, najw, mapa_S, wariant):
    """(B, 6, len(MIARY_C)); mapa_S: mapa ciepła po prawdziwym przejściu (B, 600)."""
    B = stany.shape[1]
    out = np.full((B, NB, len(MIARY_C)), np.nan, np.float32)
    for b, blk in enumerate(net.bloki):
        t = stany[2 * b]
        S, R, P, A_only, M_only = warianty_bloku(blk, t, wariant)
        dS = S - t; nS = dS.norm(dim=-1).clamp_min(1e-12)
        sr = (S - R).norm(dim=-1) / nS; sp = (S - P).norm(dim=-1) / nS
        cs = torch.nn.functional.cosine_similarity(dS, P - t, dim=-1)
        def med(v, m):
            return np.array([float(v[i][m[i]].median()) if m[i].any() else np.nan for i in range(B)])
        kol = [med(sr, tlo | narz), med(sp, tlo | narz), med(cs, tlo | narz), med(sr, narz), med(sp, narz), med(cs, narz)]
        for alt in (R, P, A_only, M_only):
            o = reszta_sieci(net, alt, b + 1, wariant)
            bx, sc, mapa, _, _ = ramki_szczytu(o)
            kol += [iou_xyxy(bx, najw), sc, korelacja_wierszy(mapa, mapa_S)]
        # kolejność w MIARY_C: SR SP cosSP SR_n SP_n cosSP_n, potem iou_R iou_P iou_A iou_M, cieplo_*, kor_*
        kol = kol[:6] + [kol[6], kol[9], kol[12], kol[15], kol[7], kol[10], kol[13], kol[16], kol[8], kol[11], kol[14], kol[17]]
        out[:, b] = np.stack(kol, 1)
    return out


# ---------------------------------------------------------------- A: jakobiany
def jakobiany(blk, xm, a, idx):
    """J (n, 192, 192) float64 dla tokenów idx (n,) klatki: xm (600, 192) wejście MLP, a (600, 768) przed-aktywacje."""
    ln = blk.norm2
    W1 = blk.mlp.fc1.weight.double(); W2 = blk.mlp.fc2.weight.double(); g = ln.weight.double()
    x = xm[idx].double(); aa = a[idx].double()
    mu = x.mean(-1, keepdim=True); sig = (x.var(-1, unbiased=False, keepdim=True) + ln.eps).sqrt(); xh = (x - mu) / sig
    I = torch.eye(C, device=x.device, dtype=torch.float64)
    JLN = ((I - 1.0 / C)[None] - xh[:, :, None] * xh[:, None, :] / C) * (g / sig)[:, :, None]
    gp = 0.5 * (1 + torch.erf(aa / math.sqrt(2.0))) + aa * torch.exp(-0.5 * aa * aa) / math.sqrt(2 * math.pi)
    Hh = torch.einsum('cd,jde->jce', W1, JLN) * gp[:, :, None]
    return torch.einsum('fc,jce->jfe', W2, Hh)


def _widmo(J):
    lam = np.linalg.eigvals(J)
    return lam


def miary_A(J):
    """J (n, 192, 192) float64 numpy → (n, len(MIARY_A)); wartości własne liczone w puli procesów przez wywołującego."""
    n = len(J)
    nJ = np.sqrt((J ** 2).sum((1, 2)))
    S = 0.5 * (J + J.transpose(0, 2, 1)); Om = 0.5 * (J - J.transpose(0, 2, 1))
    om_s = np.sqrt((Om ** 2).sum((1, 2))) / np.maximum(np.sqrt((S ** 2).sum((1, 2))), 1e-30)
    trJ = np.trace(J, axis1=1, axis2=2)
    A = J - (trJ / C)[:, None, None] * np.eye(C)[None]
    trA2 = (A * A.transpose(0, 2, 1)).sum((1, 2)); trA3 = ((A @ A) * A.transpose(0, 2, 1)).sum((1, 2))
    return nJ, om_s, trJ, -0.5 * trA2, -trA3 / 3.0


def zlozyc_A(J, lam):
    nJ, om_s, trJ, Q, R = miary_A(J)
    tol = 1e-6 * nJ[:, None]
    nN = np.sqrt(np.maximum(nJ ** 2 - (np.abs(lam) ** 2).sum(1), 0.0))
    re, im = lam.real, lam.imag
    return np.stack([nJ, nN, nN / np.maximum(nJ, 1e-30), om_s, trJ, (re > tol).sum(1), (re < -tol).sum(1), (np.abs(im) > tol).sum(1),
                     (np.abs(1 + lam) > 1).sum(1), re.max(1), Q, R], 1).astype(np.float64)


# ---------------------------------------------------------------- główna pętla
def pomiar(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wagi = Path(args.wagi); dane = Path(args.dane); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    kroki = [k for k in SIATKA if (wagi / f'k{k:05d}.pt').exists()] if args.kroki is None else args.kroki
    ident, X, gt = wczytaj_klatki(dane, args.rola, tylko_diag=False)
    narz_np, tlo_np, najw = maski(gt)
    idd, Xd, gtd = wczytaj_klatki(dane, args.rola, tylko_diag=True)
    narz_d, tlo_d, _ = maski(gtd)
    if args.proba:
        ident, X, gt, narz_np, tlo_np, najw = ident[:args.proba], X[:args.proba], gt[:args.proba], narz_np[:args.proba], tlo_np[:args.proba], najw[:args.proba]
        idd, Xd, gtd, narz_d, tlo_d = idd[:4], Xd[:4], gtd[:4], narz_d[:4], tlo_d[:4]
    N, Nd = len(X), len(Xd)
    print(f'wagi {wagi} kroki {kroki}; klatki {args.rola}: {N} z GT, diagnostyczne {Nd}; wariant {args.wariant}; {dev}', flush=True)
    rng = np.random.default_rng(2026)
    wyb = []                                                                       # tokeny do jakobianów na klatkę diagnostyczną
    for i in range(Nd):
        tl = np.flatnonzero(tlo_d[i]); los = rng.choice(tl, min(N_TLA_JAK, len(tl)), replace=False)
        nr = np.flatnonzero(narz_d[i]); wyb.append((np.concatenate([nr, los]), np.r_[np.ones(len(nr), bool), np.zeros(len(los), bool)]))
    B_stan = np.full((len(kroki), N, 13, len(MIARY_B_STAN)), np.nan, np.float32)
    B_pol = np.full((len(kroki), N, 12, len(MIARY_B_POL)), np.nan, np.float32)
    Cm = np.full((len(kroki), N, NB, len(MIARY_C)), np.nan, np.float32)
    iou = np.full((len(kroki), N), np.nan, np.float32); pewn = np.full((len(kroki), N), np.nan, np.float32); ap = np.full(len(kroki), np.nan)
    A_tab = []                                                                     # wiersze: krok, klatka, blok, token, narzędzie, miary
    czasy = {}
    narz_t = torch.as_tensor(narz_np, device=dev); tlo_t = torch.as_tensor(tlo_np, device=dev)
    pula = Pool(max(1, args.cpus))
    try:
        for ik, k in enumerate(kroki):
            t0 = time.time()
            net, krok = wczytaj_siec(wagi / f'k{k:05d}.pt', dev)
            assert krok in (k, -1), (krok, k)
            det = []
            with torch.no_grad():
                for s in range(0, N, args.paczka):
                    x = torch.from_numpy(X[s:s + args.paczka]).to(dev).permute(0, 3, 1, 2).float() / 255
                    stany, XM, AM, o = przejscie(net, x, args.wariant)
                    bx, sc, mapa, bx10, sc10 = ramki_szczytu(o)
                    sl = slice(s, s + len(x))
                    iou[ik, sl] = iou_xyxy(bx, najw[sl]); pewn[ik, sl] = sc
                    det += [(bx10[i], sc10[i]) for i in range(len(x))]
                    bs, bp = miary_B(stany, narz_t[sl], tlo_t[sl])
                    B_stan[ik, sl] = bs; B_pol[ik, sl] = bp
                    Cm[ik, sl] = miary_C(net, stany, narz_t[sl], tlo_t[sl], najw[sl], mapa, args.wariant)
                ap[ik] = ap50(det, [g for g in gt])
                tB = time.time() - t0
                # A: jakobiany na klatkach diagnostycznych
                zad = []
                for s in range(0, Nd, args.paczka):
                    x = torch.from_numpy(Xd[s:s + args.paczka]).to(dev).permute(0, 3, 1, 2).float() / 255
                    _, XM, AM, _ = przejscie(net, x, args.wariant)
                    for i in range(len(x)):
                        idx, jest_narz = wyb[s + i]
                        for b, blk in enumerate(net.bloki):
                            J = jakobiany(blk, XM[b, i], AM[b, i], torch.as_tensor(idx, device=dev)).cpu().numpy()
                            zad.append((s + i, b, idx, jest_narz, J))
                widma = pula.map(_widmo, [z[4] for z in zad], chunksize=4)
                for (i, b, idx, jest_narz, J), lam in zip(zad, widma):
                    m = zlozyc_A(J, lam)
                    A_tab.append(np.column_stack([np.full(len(idx), k), np.full(len(idx), i), np.full(len(idx), b), idx, jest_narz.astype(float), m]))
            czasy[str(k)] = dict(B_C_s=round(tB, 1), A_s=round(time.time() - t0 - tB, 1), n_jak=sum(len(z[2]) for z in zad))
            print(f'k{k}: AP50 {ap[ik]:.3f}, IoU med {np.nanmedian(iou[ik]):.3f}, B+C {tB:.0f} s, A {time.time() - t0 - tB:.0f} s', flush=True)
    finally:
        pula.close(); pula.join()
    A_arr = np.concatenate(A_tab) if A_tab else np.zeros((0, 5 + len(MIARY_A)))
    np.savez_compressed(out / 'wyniki.npz', kroki=np.array(kroki), ident=np.array(ident), ident_diag=np.array(idd), narz=narz_np, tlo=tlo_np,
                        B_stan=B_stan, B_pol=B_pol, C=Cm, iou=iou, pewnosc=pewn, ap50=ap, A=A_arr,
                        miary_A=np.array(['krok', 'klatka', 'blok', 'token', 'narz'] + MIARY_A), miary_B_stan=np.array(MIARY_B_STAN),
                        miary_B_pol=np.array(MIARY_B_POL), miary_C=np.array(MIARY_C))
    (out / 'podsumowanie.json').write_text(json.dumps(podsumuj(kroki, B_stan, B_pol, Cm, iou, pewn, ap, A_arr, args, czasy), indent=1, ensure_ascii=False),
                                           encoding='utf-8')
    print('zapisano', out, flush=True)


def _q(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return [float(np.percentile(v, q)) for q in (10, 50, 90)] if len(v) else None


def podsumuj(kroki, B_stan, B_pol, Cm, iou, pewn, ap, A_arr, args, czasy):
    """Mediany per checkpoint: B po klatkach i stanach, C po klatkach i blokach, A po tokenach (narzędzie / tło) i blokach."""
    s = dict(wagi=str(args.wagi), wariant=args.wariant, rola=args.rola, kroki=[int(k) for k in kroki], n_klatek=int(iou.shape[1]), czasy=czasy,
             ap50={str(k): float(ap[i]) for i, k in enumerate(kroki)},
             iou={str(k): _q(iou[i]) for i, k in enumerate(kroki)}, pewnosc={str(k): _q(pewn[i]) for i, k in enumerate(kroki)},
             B_stan={}, B_pol={}, C={}, A={})
    for i, k in enumerate(kroki):
        s['B_stan'][str(k)] = {m: [_q(B_stan[i, :, st, j]) for st in range(13)] for j, m in enumerate(MIARY_B_STAN)}
        s['B_pol'][str(k)] = {m: [_q(B_pol[i, :, p, j]) for p in range(12)] for j, m in enumerate(MIARY_B_POL)}
        s['C'][str(k)] = {m: [_q(Cm[i, :, b, j]) for b in range(NB)] for j, m in enumerate(MIARY_C)}
        if len(A_arr):
            w = A_arr[A_arr[:, 0] == k]
            s['A'][str(k)] = {m: {kl: [_q(w[(w[:, 2] == b) & (w[:, 4] == (1.0 if kl == 'narz' else 0.0)), 5 + j]) for b in range(NB)]
                                  for kl in ('narz', 'tlo')} for j, m in enumerate(MIARY_A)}
    # pochodne do predykcji: iloczyny stosunków D_u po półkrokach uwagi i MLP, suma Δsep_u
    for i, k in enumerate(kroki):
        Du = np.nanmedian(B_stan[i, :, :, MIARY_B_STAN.index('D_u')], 0)                # (13,)
        sep = np.nanmedian(B_stan[i, :, :, MIARY_B_STAN.index('sep_u')], 0)
        st = Du[1:] / Du[:-1]; ds = sep[1:] - sep[:-1]
        s['B_pol'][str(k)]['stosunek_D_u_mediany'] = st.tolist()
        s['B_pol'][str(k)]['iloczyn_uwaga'] = float(np.prod(st[0::2])); s['B_pol'][str(k)]['iloczyn_mlp'] = float(np.prod(st[1::2]))
        s['B_pol'][str(k)]['dsep_uwaga'] = float(np.nansum(ds[0::2])); s['B_pol'][str(k)]['dsep_mlp'] = float(np.nansum(ds[1::2]))
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wagi', required=True); p.add_argument('--dane', required=True); p.add_argument('--out', required=True)
    p.add_argument('--wariant', default='base', choices=WARIANTY); p.add_argument('--rola', default='evaluation', choices=['evaluation', 'discovery'])
    p.add_argument('--kroki', type=int, nargs='*', default=None); p.add_argument('--paczka', type=int, default=25)
    p.add_argument('--cpus', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    p.add_argument('--proba', type=int, default=0, help='tylko tyle klatek (i 4 diagnostyczne): sprawdzenie kodu')
    pomiar(p.parse_args())


if __name__ == '__main__':
    main()
