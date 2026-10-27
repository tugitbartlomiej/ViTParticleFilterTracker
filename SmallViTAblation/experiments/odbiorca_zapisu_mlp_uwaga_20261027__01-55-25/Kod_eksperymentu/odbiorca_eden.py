"""Kto czyta to, co pisze wzmacniany mod MLP? Dziecko balanced_amplification_schur_odczyt_20261027__00-31-41 (wersja 2: u₁ nie leży
w odczycie głowy ani w wejściu v₁ MLP następnego bloku). Trzy układy jak w rodzicu (detekcja 3 seedy, MałyViT i DeiT na CIFAR-10),
checkpointy k0, wczesny, koniec. Bez treningu, bez scipy. Cała sieć liczona w float64 (kopia wag .double()), uwaga liczona ręcznie
(te same wagi timm, bez SDPA) — zgodność z blokiem timm sprawdzana przy każdym checkpoincie (`zgodnosc_reczna` w podsumowaniu).

Oznaczenia: dla bloku b i tokenu J_b = W₂·diag(GELU′)·W₁·J_LN2 (jakobian pola MLP, jak w rodzicu, PM.jakobiany), para osobliwa:
u₁ᵇ (lewy, „pisz”), v₁ᵇ (prawy, „czytaj”), σ₁ (przez eigh JᵀJ:
v₁ = wiodący wektor własny, u₁ = J v₁ / σ₁; równoważne SVD rodzica z dokładnością do znaku). Kierunki d: u1 = u₁ᵇ, v1 = v₁ᵇ, los = losowy jednostkowy (nowy na token i blok),
v1n = v₁ᵇ⁺¹ (ten sam token). Bloki b = 0…NB−2.

A (liniowo, wszystkie wybrane tokeny): dla odbiorcy c = b+off, off ∈ {1, 2} (c < NB), stan x_c = wejście bloku c (stan 2c), x_c′ = po uwadze
  bloku c (stan 2c+1):
  R_q, R_k, R_v = W_q/k/v · J_LN1(x_c) (192 × 192; W_qkv = attn.qkv.weight podzielone po 192 wiersze), R_att = [R_q; R_k; R_v],
  R_m = W₁ · J_LN2(x_c′) (768 × 192), R_mg = diag(GELU′(a_c)) · R_m (opisowo: odczyt z bramkowaniem).
  g_X(d) = ‖R_X d‖² / mediana ‖R_X r‖² po 64 losowych r (ten sam token); gmax_X = σ₁(R_X)² / ta sama mediana (sufit);
  E_X(d) = energia d w 16 wiodących prawych wektorach osobliwych R_X (losowe 16/192); Enull(d) = energia d w jądrze J_LN1(x_c),
  span{1, x̂_c} (losowe 2/192); cos²(u₁ᵇ, v₁ᵇ⁺¹), cos²(v₁ᵇ, v₁ᵇ⁺¹), cos²(u₁ᵇ, u₁ᵇ⁺¹) (kontrola zgodności z rodzicem), σ₁(J_b).
O (odczyt głowy, 8 tokenów na klatkę, jak w rodzicu): gradienty wszystkich wyjść tokenu (detekcja: 5 = ciepło + 4 ramki tego tokenu;
  CIFAR: 10 logitów) po stanie tokenu po bloku b; E_glowa(d) = energia d w ich rozpięciu, g_glowa(d) jak g_X.
B (nieliniowo, 8 tokenów): stan po MLP bloku b (wejście bloku b+1), jedna kopia na token i kierunek d ∈ {u1, los, v1n, v1}: x_tok += δ·s·d,
  δ ∈ {0,03; 0,1}, s = mediana normy tokenu w tej klatce i stanie. Przez blok b+1 (ręcznie) i resztę sieci:
  dattn_wl / dattn_inne = ‖Δ wyjścia podwarstwy uwagi b+1‖ tego tokenu / pozostałych tokenów (Frobenius),
  dA_wiersz = ‖ΔA[:, tok, :]‖, dA_kol = ‖ΔA[:, inne, tok]‖, dA_inne_wiersze = ‖ΔA[:, inne, :]‖ (A = mapa uwagi, wszystkie głowy),
  dmlp_wl / dmlp_inne = ‖Δ wyjścia MLP b+1‖ (po zmianie uwagi), dmlp_bezp_wl = Δ MLP b+1 tego tokenu, gdy zmienia się tylko rezydual
  (uwaga niezaburzona), dattn_q / dattn_k / dattn_v = ‖Δ wyjścia uwagi‖ (wszystkie tokeny), gdy zaburzenie wchodzi tylko przez q / k / v
  (dattn_k_inne, dattn_v_inne: pozostałe tokeny), dblok = ‖Δ wyjścia bloku b+1‖, dwyj = ‖Δ wyjścia sieci‖ (detekcja: mapa 600 × 5;
  CIFAR: 10 logitów), dwyj_wl (detekcja: wyjścia tego tokenu), dwyj_cieplo (detekcja: kanał ciepła wszystkich tokenów).
C (wrażliwość reszty sieci, 8 tokenów, JVP torch.func): ‖J_reszty d‖ dla d ∈ {u1, v1, v1n, 4 × losowy}, J_reszty = ∂(wyjście sieci) /
  ∂(stan tokenu po bloku b); jvp (całe wyjście), jvp_cieplo i jvp_wl (detekcja: kanał ciepła, wyjścia tego tokenu).

Tokeny: reguła rodzica (WZ.Uklad.tokeny), generator tokenów ponownie ziarnem 2026 dla każdego checkpointu (te same tokeny we wszystkich
checkpointach; przy k0 identyczne z rodzicem); kierunki losowe z osobnego generatora (ziarno 2027).
Wyjście: <out>/wyniki.npz (A, O, B, C) i <out>/podsumowanie.json (mediany per checkpoint i blok).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KAT = Path(__file__).resolve().parent
EXP = KAT.parents[1]                                                     # experiments/
RODZIC = 'balanced_amplification_schur_odczyt_20261027__00-31-41'
DZIADEK = 'wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11'


def _laduj(nazwa, sciezka):
    spec = importlib.util.spec_from_file_location(nazwa, sciezka)
    mod = importlib.util.module_from_spec(spec); sys.modules[nazwa] = mod; spec.loader.exec_module(mod)
    return mod


# Uklad z dziadka (pod nazwą 'wzmocnienie_eden', żeby balanced_eden rodzica użył tego samego modułu, a nie swojej identycznej kopii);
# pomiar_mosty_eden / pomiar_mosty_cifar / model ładuje dziadek ze swojego folderu (kopie w rodzicu mają identyczne SHA256).
WZ = _laduj('wzmocnienie_eden', EXP / DZIADEK / 'Kod_eksperymentu' / 'wzmocnienie_eden.py')
BE = _laduj('balanced_eden_rodzic', EXP / RODZIC / 'Kod_eksperymentu' / 'balanced_eden.py')
PM = WZ.PM
assert BE.WZ is WZ and BE.PM is PM

C = BE.C
N_ODCZYT = BE.N_ODCZYT
DELTY = (0.03, 0.1)
K_TOP = 16
N_REF = 64
N_LOS_C = 4
OFFY = (1, 2)
KOL_X = ['q', 'k', 'v', 'att', 'm', 'mg']
KIER_A = ['u1', 'v1', 'los', 'v1n']
KOL_A = ([f'g_{x}_{d}' for x in KOL_X for d in KIER_A] + [f'E_{x}_{d}' for x in KOL_X for d in KIER_A] + [f'gmax_{x}' for x in KOL_X]
         + [f'Enull_{d}' for d in KIER_A] + ['cos2_u1_v1n', 'cos2_v1_v1n', 'cos2_u1_u1n', 'sigma1'])
KOL_O = [f'E_glowa_{d}' for d in KIER_A] + [f'g_glowa_{d}' for d in KIER_A] + ['k_glowa']
KIER_B = ['u1', 'los', 'v1n', 'v1']
KOL_B = ['dattn_wl', 'dattn_inne', 'dA_wiersz', 'dA_kol', 'dA_inne_wiersze', 'dmlp_wl', 'dmlp_inne', 'dmlp_bezp_wl', 'dattn_q', 'dattn_k',
         'dattn_v', 'dattn_k_inne', 'dattn_v_inne', 'dblok', 'dwyj', 'dwyj_wl', 'dwyj_cieplo']
KIER_C = ['u1', 'v1', 'v1n'] + [f'los{j}' for j in range(N_LOS_C)]
KOL_C = ['jvp', 'jvp_cieplo', 'jvp_wl']


# ---------------------------------------------------------------- sieć liczona ręcznie (float64, bez SDPA)
def uwaga_reczna(attn, hq, hk, hv):
    """Wyjście podwarstwy uwagi timm (przed rezydualem, po proj) i mapa A (B, H, T, T); q z hq, k z hk, v z hv (wejścia po LN1)."""
    assert getattr(attn, 'gate', None) is None
    W, bb = attn.qkv.weight, attn.qkv.bias
    H = attn.num_heads; B, T, Cc = hq.shape; hd = getattr(attn, 'head_dim', Cc // H); D = H * hd

    def rzut(h, s):
        return F.linear(h, W[s * D:(s + 1) * D], None if bb is None else bb[s * D:(s + 1) * D]).reshape(B, T, H, hd).transpose(1, 2)
    q, k, v = attn.q_norm(rzut(hq, 0)), attn.k_norm(rzut(hk, 1)), rzut(hv, 2)
    A = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
    o = (A @ v).transpose(1, 2).reshape(B, T, D)
    if hasattr(attn, 'norm'):
        o = attn.norm(o)
    return attn.proj(o), A


def blok_reczny(blk, t):
    h = blk.norm1(t); ao, A = uwaga_reczna(blk.attn, h, h, h)
    xa = t + blk.ls1(ao)
    mo = blk.ls2(blk.mlp(blk.norm2(xa)))
    return xa + mo, xa, ao, mo, A, h


def wyjscie(U, net, t):
    return net.glowa(net.norm(t)) if U.rodzaj == 'detekcja' else net.wyjscie(t)


def reszta_reczna(U, net, t, od):
    for blk in list(net.bloki)[od:]:
        t = blok_reczny(blk, t)[0]
    return wyjscie(U, net, t)


# ---------------------------------------------------------------- pomocnicze
def gelu_p(a):
    return 0.5 * (1 + torch.erf(a / math.sqrt(2.0))) + a * torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)


def jln(ln, x):
    """Jakobian LayerNorm (n, C, C) w stanach x (n, C) oraz x̂ (n, C) — jak w PM.jakobiany."""
    g = ln.weight.double(); mu = x.mean(-1, keepdim=True); sig = (x.var(-1, unbiased=False, keepdim=True) + ln.eps).sqrt(); xh = (x - mu) / sig
    I = torch.eye(C, device=x.device, dtype=torch.float64)
    return ((I - 1.0 / C)[None] - xh[:, :, None] * xh[:, None, :] / C) * (g / sig)[:, :, None], xh


def miary_grama(Gs, D, Rr):
    """Gs (nx, n, C, C) Gramy R^T R, D (n, nd, C) kierunki, Rr (m, C) losowe → g (nx, n, nd), E (nx, n, nd), gmax (nx, n)."""
    nx, n = Gs.shape[:2]
    quad = torch.einsum('njc,xncd,njd->xnj', D, Gs, D)
    ref = torch.einsum('mc,xncd,md->xnm', Rr, Gs, Rr).median(2).values.clamp_min(1e-300)
    ev, V = torch.linalg.eigh(Gs.reshape(nx * n, C, C))
    Vt = V[:, :, -K_TOP:].reshape(nx, n, C, K_TOP); ev = ev[:, -1].reshape(nx, n)
    E = (torch.einsum('xnck,njc->xnjk', Vt, D) ** 2).sum(-1)
    return quad / ref[:, :, None], E, ev / ref


def jedn(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-300)


def normy_wyj(U, dO, tj):
    """dO (B, T, 5) albo (B, 10) → (całość, własny token, kanał ciepła) (B,) każde."""
    if U.rodzaj == 'detekcja':
        ar = torch.arange(len(dO), device=dO.device)
        return dO.flatten(1).norm(dim=1), dO[ar, tj].norm(dim=1), dO[..., 0].norm(dim=1)
    nan = torch.full((len(dO),), float('nan'), device=dO.device, dtype=dO.dtype)
    return dO.norm(dim=1), nan, nan


def sprawdz_reczny(U, net, x):
    """Względna różnica: blok ręczny vs blok timm oraz całe wyjście ręczne vs U.przejscie."""
    stany, XM, AM, o = U.przejscie(net, x)
    blk = net.bloki[0]
    d1 = float((blok_reczny(blk, stany[0])[0] - blk(stany[0])).norm() / blk(stany[0]).norm())
    o2 = reszta_reczna(U, net, stany[0], 0)
    d2 = float((o2 - o).norm() / o.norm())
    return max(d1, d2)


# ---------------------------------------------------------------- główna pętla
def pomiar(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    U = WZ.Uklad(args.uklad, args.wagi, args.dane, dev)
    if args.proba:
        U.X = U.X[:args.proba]; U.ident = U.ident[:args.proba]
        if U.y is not None: U.y = U.y[:args.proba]
        if U.narz is not None: U.narz, U.tlo, U.najw = U.narz[:args.proba], U.tlo[:args.proba], U.najw[:args.proba]
    N = len(U.X); torch.manual_seed(2026)
    rng_k = np.random.default_rng(2027)
    tn = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float64)
    A_tab, O_tab, B_tab, C_tab, czasy = [], [], [], [], {}
    print(f'układ {args.uklad}, wagi {args.wagi}, kroki {args.kroki}, klatki/obrazy {N}, {dev}', flush=True)
    for k in args.kroki:
        t0 = time.time(); net = U.wczytaj(k).double(); NB = len(net.bloki)
        rng_t = np.random.default_rng(2026)
        tA = tO = tB = tC = 0.0
        with torch.no_grad():
            zg = sprawdz_reczny(U, net, torch.from_numpy(U.X[:2]).to(dev).permute(0, 3, 1, 2).double() / 255)
        assert zg < 1e-9, f'blok ręczny niezgodny z timm: {zg}'
        # stałe per blok: W_qkv^T W_qkv
        WTW = []
        for blk in net.bloki:
            Wq = blk.attn.qkv.weight.detach()
            WTW.append([Wq[s * C:(s + 1) * C].T @ Wq[s * C:(s + 1) * C] for s in range(3)])
        for s in range(0, N, args.paczka):
            x = torch.from_numpy(U.X[s:s + args.paczka]).to(dev).permute(0, 3, 1, 2).double() / 255
            with torch.no_grad():
                stany, XM, AM, o = U.przejscie(net, x)
            T = stany.shape[2]
            for i in range(len(x)):
                fr = s + i
                tok, jest_narz = U.tokeny(fr, T, rng_t); tok_t = torch.as_tensor(tok, device=dev); n = len(tok)
                sel = np.sort(rng_t.choice(n, min(N_ODCZYT, n), replace=False)); sel_t = torch.as_tensor(sel, device=dev)
                ta = time.time()
                with torch.no_grad():
                    u1, v1, s1 = [], [], []
                    for b, blk in enumerate(net.bloki):
                        J = PM.jakobiany(blk, XM[b, i], AM[b, i], tok_t)
                        ev, V = torch.linalg.eigh(J.transpose(1, 2) @ J)                       # para osobliwa przez JᵀJ (SVD float64 na GPU jest bardzo wolne)
                        v_ = V[:, :, -1]; sg = ev[:, -1].clamp_min(0).sqrt()
                        u1.append(jedn((J @ v_[:, :, None])[..., 0])); v1.append(v_); s1.append(sg)
                    los = [jedn(tn(rng_k.normal(size=(n, C)))) for _ in range(NB)]
                    Rr = jedn(tn(rng_k.normal(size=(N_REF, C))))
                    # ---------------- A: odczyt liniowy odbiorców c = b+1, b+2
                    for b in range(NB - 1):
                        D = torch.stack([u1[b], v1[b], los[b], v1[b + 1]], 1)                       # (n, 4, C)
                        kos = [((u1[b] * v1[b + 1]).sum(1) ** 2), ((v1[b] * v1[b + 1]).sum(1) ** 2), ((u1[b] * u1[b + 1]).sum(1) ** 2), s1[b]]
                        for off in OFFY:
                            c = b + off
                            if c >= NB:
                                continue
                            blk = net.bloki[c]
                            JL1, xh = jln(blk.norm1, stany[2 * c, i][tok_t])
                            JL2, _ = jln(blk.norm2, stany[2 * c + 1, i][tok_t])
                            Gs = [JL1.transpose(1, 2) @ WTW[c][j] @ JL1 for j in range(3)]
                            Rm = torch.einsum('fc,nce->nfe', blk.mlp.fc1.weight, JL2)                    # (n, 768, C)
                            Gm = Rm.transpose(1, 2) @ Rm
                            gp = gelu_p(AM[c, i][tok_t])
                            Rmg = Rm * gp[:, :, None]; Gmg = Rmg.transpose(1, 2) @ Rmg
                            gg, EE, gm = miary_grama(torch.stack([Gs[0], Gs[1], Gs[2], Gs[0] + Gs[1] + Gs[2], Gm, Gmg]), D, Rr)
                            jed = torch.full((n, C), 1.0 / math.sqrt(C), device=dev, dtype=torch.float64); xhn = jedn(xh)
                            En = (D @ jed[:, :, None])[..., 0] ** 2 + (D @ xhn[:, :, None])[..., 0] ** 2       # (n, 4)
                            M = torch.cat([gg.permute(1, 0, 2).reshape(n, -1), EE.permute(1, 0, 2).reshape(n, -1), gm.T, En, torch.stack(kos, 1)], 1).cpu().numpy()
                            for j in range(n):
                                A_tab.append([k, fr, b, off, int(tok[j]), float(jest_narz[j])] + list(M[j]))
                tA += time.time() - ta
                # ---------------- O: odczyt głowy (gradienty wyjść tokenu po stanie po bloku b)
                to_ = time.time()
                for b in range(NB - 1):
                    xm = stany[2 * b + 2, i].detach()[None].clone().requires_grad_(True)
                    o1 = reszta_reczna(U, net, xm, b + 1)[0]
                    nout = o1.shape[-1]
                    Rg = []
                    for j in sel:
                        tj = int(tok[j]); gr = []
                        for q in range(nout):
                            cel = o1[tj, q] if U.rodzaj == 'detekcja' else o1[q]
                            g, = torch.autograd.grad(cel, xm, retain_graph=True)
                            gr.append(g[0, tj].detach())
                        Rg.append(torch.stack(gr))
                    Rg = torch.stack(Rg)                                                               # (m, nout, C)
                    del o1, xm
                    with torch.no_grad():
                        D = torch.stack([u1[b][sel_t], v1[b][sel_t], los[b][sel_t], v1[b + 1][sel_t]], 1)
                        Qg, _ = torch.linalg.qr(Rg.transpose(1, 2))                                    # (m, C, nout)
                        Eg = (torch.einsum('mck,mjc->mjk', Qg, D) ** 2).sum(-1)
                        Gg = Rg.transpose(1, 2) @ Rg
                        quad = torch.einsum('mjc,mcd,mjd->mj', D, Gg, D)
                        ref = torch.einsum('rc,mcd,rd->mr', Rr, Gg, Rr).median(1).values.clamp_min(1e-300)
                        M = torch.cat([Eg, quad / ref[:, None], torch.full((len(sel), 1), float(nout), device=dev, dtype=torch.float64)], 1).cpu().numpy()
                    for jj, j in enumerate(sel):
                        O_tab.append([k, fr, b, int(tok[j]), float(jest_narz[j])] + list(M[jj]))
                tO += time.time() - to_
                # ---------------- B: zaburzenia nieliniowe przez blok b+1 i resztę sieci
                tb = time.time()
                m = len(sel); tsel = tok_t[sel_t]
                with torch.no_grad():
                    for b in range(NB - 1):
                        blk = net.bloki[b + 1]
                        x0 = stany[2 * b + 2, i]                                                      # (T, C)
                        skala = float(x0.norm(dim=1).median())
                        y0, xa0, ao0, mo0, A0, h0 = blok_reczny(blk, x0[None])
                        out0 = reszta_reczna(U, net, y0, b + 2)
                        Dk = torch.stack([u1[b][sel_t], los[b][sel_t], v1[b + 1][sel_t], v1[b][sel_t]], 1)   # (m, 4, C)
                        nk = len(KIER_B); Bn = m * nk
                        tj = tsel.repeat_interleave(nk); kier = Dk.reshape(Bn, C)
                        ar = torch.arange(Bn, device=dev)
                        Mi = torch.ones(Bn, T, device=dev, dtype=torch.float64); Mi[ar, tj] = 0
                        for d in DELTY:
                            X = x0[None].repeat(Bn, 1, 1); X[ar, tj] += d * skala * kier
                            y, xa, ao, mo, A, h = blok_reczny(blk, X)
                            dao = blk.ls1(ao) - blk.ls1(ao0); dmo = mo - mo0; dA = A - A0
                            mob = blk.ls2(blk.mlp(blk.norm2(xa0 + (X - x0[None]))))
                            wl = lambda Z: Z[ar, tj].norm(dim=-1)
                            inne = lambda Z: ((Z ** 2).sum(-1) * Mi).sum(1).sqrt()
                            calosc = lambda Z: Z.flatten(1).norm(dim=1)
                            h0b = h0.expand(Bn, -1, -1)
                            aq, _ = uwaga_reczna(blk.attn, h, h0b, h0b); ak, _ = uwaga_reczna(blk.attn, h0b, h, h0b); av, _ = uwaga_reczna(blk.attn, h0b, h0b, h)
                            aq, ak, av = blk.ls1(aq) - ao0, blk.ls1(ak) - ao0, blk.ls1(av) - ao0
                            sA = (dA ** 2).sum(-1).sum(1)                                              # (Bn, T) suma po głowach i kolumnach
                            dA_w = dA[ar, :, tj, :].flatten(1).norm(dim=1)
                            dA_k = ((dA[ar, :, :, tj] ** 2).sum(1) * Mi).sum(1).sqrt()
                            dA_iw = (sA * Mi).sum(1).sqrt()
                            out1 = reszta_reczna(U, net, y, b + 2)
                            dw, dw_wl, dw_c = normy_wyj(U, out1 - out0, tj)
                            kol = [wl(dao), inne(dao), dA_w, dA_k, dA_iw, wl(dmo), inne(dmo), wl(mob - mo0), calosc(aq), calosc(ak), calosc(av),
                                   inne(ak), inne(av), calosc(y - y0), dw, dw_wl, dw_c]
                            Mb = torch.stack(kol, 1).cpu().numpy()
                            for r in range(Bn):
                                jj = r // nk
                                B_tab.append([k, fr, b, int(tok[sel[jj]]), float(jest_narz[sel[jj]]), d, r % nk, d * skala] + list(Mb[r]))
                tB += time.time() - tb
                # ---------------- C: wrażliwość reszty sieci (JVP) od stanu po bloku b
                tc = time.time()
                for b in range(NB - 1):
                    x0 = stany[2 * b + 2, i].detach()
                    Dl = jedn(tn(rng_k.normal(size=(m, N_LOS_C, C))))
                    Dk = torch.cat([torch.stack([u1[b][sel_t], v1[b][sel_t], v1[b + 1][sel_t]], 1), Dl], 1)   # (m, 7, C)
                    nk = len(KIER_C); Bn = m * nk; tj = tsel.repeat_interleave(nk)
                    wyn = []
                    for p0 in range(0, Bn, args.paczka_jvp):
                        sl = slice(p0, min(Bn, p0 + args.paczka_jvp)); nb_ = sl.stop - sl.start
                        Xp = x0[None].repeat(nb_, 1, 1)
                        Tg = torch.zeros_like(Xp); Tg[torch.arange(nb_, device=dev), tj[sl]] = Dk.reshape(Bn, C)[sl]
                        _, jv = torch.func.jvp(lambda Z: reszta_reczna(U, net, Z, b + 1), (Xp,), (Tg,))
                        wyn.append(torch.stack(normy_wyj(U, jv.detach(), tj[sl]), 1))
                    Mc = torch.cat(wyn, 0).cpu().numpy()
                    for r in range(Bn):
                        jj = r // nk
                        C_tab.append([k, fr, b, int(tok[sel[jj]]), float(jest_narz[sel[jj]]), r % nk] + list(Mc[r]))
                tC += time.time() - tc
        czasy[str(k)] = dict(A_s=round(tA, 1), O_s=round(tO, 1), B_s=round(tB, 1), C_s=round(tC, 1), razem_s=round(time.time() - t0, 1),
                             zgodnosc_reczna=zg)
        print(f'k{k}: A {tA:.0f} s, O {tO:.0f} s, B {tB:.0f} s, C {tC:.0f} s, razem {time.time() - t0:.0f} s, zgodność ręczna {zg:.1e}', flush=True)
        del net
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    A_ = np.array(A_tab, np.float64); O_ = np.array(O_tab, np.float64); B_ = np.array(B_tab, np.float64); C_ = np.array(C_tab, np.float64)
    np.savez_compressed(out / 'wyniki.npz', A=A_, O=O_, B=B_, C=C_, kroki=np.array(args.kroki), ident=np.array(U.ident),
                        kol_A=np.array(['krok', 'klatka', 'blok', 'off', 'token', 'narz'] + KOL_A),
                        kol_O=np.array(['krok', 'klatka', 'blok', 'token', 'narz'] + KOL_O),
                        kol_B=np.array(['krok', 'klatka', 'blok', 'token', 'narz', 'delta', 'kier', 'dx'] + KOL_B),
                        kol_C=np.array(['krok', 'klatka', 'blok', 'token', 'narz', 'kier'] + KOL_C),
                        kier_B=np.array(KIER_B), kier_C=np.array(KIER_C), delty=np.array(DELTY))
    (out / 'podsumowanie.json').write_text(json.dumps(podsumuj(A_, O_, B_, C_, args, czasy), indent=1, ensure_ascii=False), encoding='utf-8')
    print('zapisano', out, flush=True)


def podsumuj(A, O, B, Cc, args, czasy):
    q = PM._q
    s = dict(uklad=args.uklad, wagi=str(args.wagi), kroki=[int(k) for k in args.kroki], czasy=czasy, A={}, O={}, B={}, C={})
    for k in args.kroki:
        a = A[A[:, 0] == k]; NB = int(a[:, 2].max()) + 2 if len(a) else 0
        s['A'][str(k)] = {f'off{off}': {m: [q(a[(a[:, 2] == b) & (a[:, 3] == off), 6 + j]) for b in range(NB - 1)] for j, m in enumerate(KOL_A)}
                          for off in OFFY}
        o = O[O[:, 0] == k]; s['O'][str(k)] = {m: [q(o[o[:, 2] == b, 5 + j]) for b in range(NB - 1)] for j, m in enumerate(KOL_O)}
        bb = B[B[:, 0] == k]
        s['B'][str(k)] = {f'd{d}_{kn}': {m: [q(bb[(bb[:, 2] == b) & (bb[:, 5] == d) & (bb[:, 6] == ki), 8 + j]) for b in range(NB - 1)]
                                          for j, m in enumerate(KOL_B)} for d in DELTY for ki, kn in enumerate(KIER_B)}
        c = Cc[Cc[:, 0] == k]
        s['C'][str(k)] = {kn: {m: [q(c[(c[:, 2] == b) & (c[:, 5] == ki), 6 + j]) for b in range(NB - 1)] for j, m in enumerate(KOL_C)}
                          for ki, kn in enumerate(KIER_C)}
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uklad', required=True, choices=['detekcja', 'cifar_maly', 'cifar_deit'])
    p.add_argument('--wagi', required=True); p.add_argument('--dane', required=True); p.add_argument('--out', required=True)
    p.add_argument('--kroki', type=int, nargs='+', required=True); p.add_argument('--paczka', type=int, default=10)
    p.add_argument('--paczka_jvp', type=int, default=28)
    p.add_argument('--cpus', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    p.add_argument('--proba', type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(max(1, a.cpus))
    pomiar(a)


if __name__ == '__main__':
    main()
