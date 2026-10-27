"""Znak kanału normy: czy MLP, odpowiadając na pchnięcie tokenu w stronę odczytu, zwiększa czy zmniejsza normę tokenu, i czy ta zmiana
normy wzmacnia (zatrzask) czy osłabia (stabilizator) skutek pchnięcia dla wyjścia sieci. Dziecko odbiorca_zapisu_mlp_uwaga_20261027__01-55-25
(20–26 % energii zapisu u₁ w jądrze LN = span{1̂, ê}). Trzy układy jak w rodzicach, checkpointy k0 / wczesny / koniec, bez treningu, bez scipy.
Sieć w float64, uwaga liczona ręcznie (funkcje rodzica `blok_reczny`, `reszta_reczna`; zgodność z timm sprawdzana przy każdym checkpoincie).

Oznaczenia (blok b = 0…NB−2, jak w rodzicu): x_a = wejście MLP bloku b (stan 2b+1), x_c = x_a + MLP(x_a) (stan 2b+2), P = centrowanie,
ρ(x) = ‖P x‖, ê = P x_c / ρ(x_c) (kierunek radialny, „norma”), 1̂ = 1/√C („średnia”). J = jakobian pola MLP w x_a (PM.jakobiany), para
osobliwa (u₁, v₁, σ₁) przez eigh JᵀJ jak w rodzicu.

L (wszystkie wybrane tokeny): E_rad / E_mean dla u₁, v₁, losowego; opisowo rzeczywista aktualizacja MLP m₀: cos(m₀, ê(x_a)), udział
  radialny m₀, wzrost normy przez MLP ρ(x_c)/ρ(x_a) − 1 i przez uwagę ρ(x_a)/ρ(x_in) − 1, ρ(x_in) / mediana klatki.
S (8 tokenów na klatkę; detekcja: do 4 narzędzia, reszta tło): odczyt y tokenu (detekcja: logit ciepła tego tokenu; CIFAR: margines
  logit[y] − max logit[inne]), r_a = ∂y/∂x_a, r_c = ∂y/∂x_c (autograd). Kierunki pchnięcia d ∈ {v1, r (= r̂_a), los}, znak s = sign⟨d, r_a⟩.
  c_rad = s⟨ê, Jd⟩/‖Jd‖, c_mean = s⟨1̂, Jd⟩/‖Jd‖, η = s⟨ê, Jd⟩·ρ(x_a)/ρ(x_c), dy_lin = s⟨r_a, d⟩, rozkład liniowy dy_lin = ⟨r_c, d⟩
  + ⟨r_c, ê⟩⟨ê, Jd⟩ + ⟨r_c, 1̂⟩⟨1̂, Jd⟩ + ⟨r_c, P_⊥ Jd⟩ → F_id, F_rad, F_mean, F_perp (każde / dy_lin; suma = 1 kontroluje kod),
  rc_mean = |⟨r_c, 1̂⟩| / ‖r_c‖ (musi być ≈ 0: LN każdego odbiorcy usuwa średnią).
B (te same tokeny i kierunki, δ ∈ {0,03; 0,1} × mediana normy x_a w klatce): x_a⁺ = x_a + δ·skala·s·d jednego tokenu, Δm = MLP(x_a⁺) −
  MLP(x_a) = a·ê + μ·1̂ + Δm_⊥; warianty stanu tokenu po bloku b: pełny, bez_rad (− a·ê), bez_mean (− μ·1̂), bez_perp (− Δm_⊥),
  bez_mlp (− Δm); każdy przez resztę sieci (bloki b+1…), Δy = y − y₀.
Wyjście: <out>/wyniki.npz (L, S, B) i <out>/podsumowanie.json.
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

KAT = Path(__file__).resolve().parent
EXP = KAT.parents[1]                                                                  # experiments/
RODZIC = 'odbiorca_zapisu_mlp_uwaga_20261027__01-55-25'


def _laduj(nazwa, sciezka):
    spec = importlib.util.spec_from_file_location(nazwa, sciezka)
    mod = importlib.util.module_from_spec(spec); sys.modules[nazwa] = mod; spec.loader.exec_module(mod)
    return mod


OD = _laduj('odbiorca_eden_rodzic', EXP / RODZIC / 'Kod_eksperymentu' / 'odbiorca_eden.py')
WZ, PM, C = OD.WZ, OD.PM, OD.C
N_SEL = 8
N_NARZ = 4
DELTY = (0.03, 0.1)
KIER = ['v1', 'r', 'los']
WAR = ['pelny', 'bez_rad', 'bez_mean', 'bez_perp', 'bez_mlp']
KOL_L = ['E_rad_u1', 'E_mean_u1', 'E_rad_v1', 'E_mean_v1', 'E_rad_los', 'E_mean_los', 'cos_m0_ea', 'Erad_m0', 'wzrost_mlp', 'wzrost_att',
         'rho_wzgl', 'sigma1']
KOL_S = ['s', 'c_rad', 'c_mean', 'eta', 'dy_lin', 'F_id', 'F_rad', 'F_mean', 'F_perp', 'rc_mean', 'n_ra', 'n_rc']
KOL_B = [f'dy_{w}' for w in WAR] + ['a', 'mu', 'n_perp', 'n_dm', 'rho_c', 'dx']


def mlp_czesc(blk, xa):
    return blk.ls2(blk.mlp(blk.norm2(xa)))


def cel(U, out, tj, lab):
    """y dla wierszy out: detekcja out (B, T, 5) → logit ciepła tokenu tj (B,); CIFAR out (B, 10) → margines (B,)."""
    if U.rodzaj == 'detekcja':
        ar = torch.arange(len(out), device=out.device)
        return out[ar, tj, 0]
    o = out.clone(); y = o[:, lab].clone(); o[:, lab] = -float('inf')
    return y - o.max(1).values


def centr(x):
    return x - x.mean(-1, keepdim=True)


def wybierz(U, jest, rng):
    n = len(jest)
    if U.rodzaj != 'detekcja':
        return np.sort(rng.choice(n, min(N_SEL, n), replace=False))
    nr, tl = np.flatnonzero(jest), np.flatnonzero(~jest)
    a = rng.choice(nr, min(N_NARZ, len(nr)), replace=False)
    b = rng.choice(tl, min(N_SEL - len(a), len(tl)), replace=False)
    return np.sort(np.concatenate([a, b]).astype(int))


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
    rng_k = np.random.default_rng(2029)
    tn = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float64)
    jed = torch.full((C,), 1.0 / math.sqrt(C), device=dev, dtype=torch.float64)
    L_tab, S_tab, B_tab, czasy = [], [], [], {}
    print(f'układ {args.uklad}, wagi {args.wagi}, kroki {args.kroki}, klatki/obrazy {N}, {dev}', flush=True)
    for k in args.kroki:
        t0 = time.time(); net = U.wczytaj(k).double(); NB = len(net.bloki)
        rng_t = np.random.default_rng(2026); rng_s = np.random.default_rng(2028)
        tL = tS = tB = 0.0; kontrola = dict(stan_xa=0.0, stan_xc=0.0)
        with torch.no_grad():
            zg = OD.sprawdz_reczny(U, net, torch.from_numpy(U.X[:2]).to(dev).permute(0, 3, 1, 2).double() / 255)
        assert zg < 1e-9, f'blok ręczny niezgodny z timm: {zg}'
        for s0 in range(0, N, args.paczka):
            x = torch.from_numpy(U.X[s0:s0 + args.paczka]).to(dev).permute(0, 3, 1, 2).double() / 255
            with torch.no_grad():
                stany, XM, AM, o = U.przejscie(net, x)
            T = stany.shape[2]
            for i in range(len(x)):
                fr = s0 + i; lab = int(U.y[fr]) if U.y is not None else -1
                tok, jest = U.tokeny(fr, T, rng_t); tok_t = torch.as_tensor(tok, device=dev); n = len(tok)
                sel = wybierz(U, jest, rng_s); sel_t = torch.as_tensor(sel, device=dev); m = len(sel); tsel = tok_t[sel_t]
                for b in range(NB - 1):
                    blk = net.bloki[b]
                    xin, xa, xc = stany[2 * b, i], stany[2 * b + 1, i], stany[2 * b + 2, i]
                    # ---------------- L
                    ta = time.time()
                    with torch.no_grad():
                        kontrola['stan_xa'] = max(kontrola['stan_xa'], float((xa - XM[b, i]).abs().max()))
                        kontrola['stan_xc'] = max(kontrola['stan_xc'], float((xa + mlp_czesc(blk, xa[None])[0] - xc).abs().max()))
                        J = PM.jakobiany(blk, XM[b, i], AM[b, i], tok_t)
                        ev, V = torch.linalg.eigh(J.transpose(1, 2) @ J)
                        v1 = V[:, :, -1]; s1 = ev[:, -1].clamp_min(0).sqrt(); u1 = OD.jedn((J @ v1[:, :, None])[..., 0])
                        los = OD.jedn(tn(rng_k.normal(size=(n, C))))
                        Pa, Pc, Pin = centr(xa[tok_t]), centr(xc[tok_t]), centr(xin[tok_t])
                        ra, rc, rin = Pa.norm(dim=1), Pc.norm(dim=1), Pin.norm(dim=1)
                        ec, ea = Pc / rc[:, None], Pa / ra[:, None]
                        m0 = xc[tok_t] - xa[tok_t]
                        rho_fr = centr(xin).norm(dim=1).median()
                        er = lambda d: (d * ec).sum(1) ** 2
                        em = lambda d: (d @ jed) ** 2
                        cm = (m0 * ea).sum(1) / m0.norm(dim=1).clamp_min(1e-300)
                        Mi = torch.stack([er(u1), em(u1), er(v1), em(v1), er(los), em(los), cm, cm ** 2, rc / ra - 1, ra / rin - 1, rin / rho_fr, s1], 1).cpu().numpy()
                    for j in range(n):
                        L_tab.append([k, fr, b, int(tok[j]), float(jest[j])] + list(Mi[j]))
                    tL += time.time() - ta
                    # ---------------- S: odczyt (autograd) i rozkład liniowy
                    ts = time.time()
                    xa_l = xa.detach()[None].clone().requires_grad_(True)
                    xc_l = xa_l + mlp_czesc(blk, xa_l)
                    o1 = OD.reszta_reczna(U, net, xc_l, b + 1)
                    Ra, Rc = [], []
                    for jj in range(m):
                        tj = int(tsel[jj])
                        y = cel(U, o1, torch.tensor([tj], device=dev), lab)[0]
                        ga, gc = torch.autograd.grad(y, (xa_l, xc_l), retain_graph=True)
                        Ra.append(ga[0, tj].detach()); Rc.append(gc[0, tj].detach())
                    del o1, xa_l, xc_l
                    with torch.no_grad():
                        Ra, Rc = torch.stack(Ra), torch.stack(Rc)
                        D = torch.stack([v1[sel_t], OD.jedn(Ra), los[sel_t]], 1)                       # (m, 3, C)
                        s = torch.sign((D * Ra[:, None]).sum(-1)); s[s == 0] = 1
                        Jd = torch.einsum('mce,mje->mjc', J[sel_t], D)
                        e = ec[sel_t][:, None]; nJd = Jd.norm(dim=-1).clamp_min(1e-300)
                        jr, jm = (Jd * e).sum(-1), Jd @ jed
                        Jperp = Jd - jr[..., None] * e - jm[..., None] * jed
                        dyl = s * (D * Ra[:, None]).sum(-1)
                        rcr, rcm = (Rc[:, None] * e).sum(-1), (Rc @ jed)[:, None]
                        Fid = s * (D * Rc[:, None]).sum(-1) / dyl
                        Frad = s * rcr * jr / dyl; Fmean = s * rcm * jm / dyl; Fperp = s * (Jperp * Rc[:, None]).sum(-1) / dyl
                        rcmean = (rcm[:, 0].abs() / Rc.norm(dim=1).clamp_min(1e-300))[:, None].expand(-1, 3)
                        eta = s * jr * (ra[sel_t] / rc[sel_t])[:, None]
                        Ms = torch.stack([s, s * jr / nJd, s * jm / nJd, eta, dyl, Fid, Frad, Fmean, Fperp, rcmean,
                                          Ra.norm(dim=1)[:, None].expand(-1, 3), Rc.norm(dim=1)[:, None].expand(-1, 3)], -1).cpu().numpy()
                    for jj in range(m):
                        for di in range(len(KIER)):
                            S_tab.append([k, fr, b, int(tsel[jj]), float(jest[sel[jj]]), di] + list(Ms[jj, di]))
                    tS += time.time() - ts
                    # ---------------- B: kontrfakty nieliniowe przez resztę sieci
                    tb = time.time()
                    with torch.no_grad():
                        skala = float(xa.norm(dim=1).median())
                        o0 = OD.reszta_reczna(U, net, xc[None], b + 1)
                        y0 = cel(U, o0.repeat(m, *([1] * (o0.dim() - 1))), tsel, lab)
                        wiersze, stany_tok, tj_all, meta = [], [], [], []
                        for jj in range(m):
                            tj = int(tsel[jj]); e1 = ec[sel_t][jj]
                            for di in range(len(KIER)):
                                for d in DELTY:
                                    dx = d * skala * s[jj, di] * D[jj, di]
                                    xp = xa[tj] + dx
                                    dm = mlp_czesc(blk, xp[None, None])[0, 0] - m0[sel_t][jj]
                                    a = (dm * e1).sum(); mu = dm @ jed; perp = dm - a * e1 - mu * jed
                                    pel = xc[tj] + dx + dm
                                    for w in (pel, pel - a * e1, pel - mu * jed, pel - perp, pel - dm):
                                        stany_tok.append(w); tj_all.append(tj)
                                    meta.append((jj, di, d, float(a), float(mu), float(perp.norm()), float(dm.norm()), float(rc[sel_t][jj]),
                                                 float(dx.norm())))
                        Wt = torch.stack(stany_tok); tj_t = torch.as_tensor(tj_all, device=dev); Bn = len(Wt)
                        dy = torch.empty(Bn, device=dev, dtype=torch.float64)
                        for p0 in range(0, Bn, args.paczka_kopii):
                            sl = slice(p0, min(Bn, p0 + args.paczka_kopii)); nb_ = sl.stop - sl.start
                            X = xc[None].repeat(nb_, 1, 1); X[torch.arange(nb_, device=dev), tj_t[sl]] = Wt[sl]
                            dy[sl] = cel(U, OD.reszta_reczna(U, net, X, b + 1), tj_t[sl], lab)
                        y0r = y0.repeat_interleave(len(KIER) * len(DELTY) * len(WAR))
                        dy = (dy - y0r).reshape(-1, len(WAR)).cpu().numpy()
                    for r, (jj, di, d, a, mu, npp, ndm, rho_c, ndx) in enumerate(meta):
                        B_tab.append([k, fr, b, int(tsel[jj]), float(jest[sel[jj]]), di, d] + list(dy[r]) + [a, mu, npp, ndm, rho_c, ndx])
                    tB += time.time() - tb
        czasy[str(k)] = dict(L_s=round(tL, 1), S_s=round(tS, 1), B_s=round(tB, 1), razem_s=round(time.time() - t0, 1), zgodnosc_reczna=zg, **kontrola)
        print(f'k{k}: L {tL:.0f} s, S {tS:.0f} s, B {tB:.0f} s, razem {time.time() - t0:.0f} s, zgodność ręczna {zg:.1e}, '
              f'stan_xa {kontrola["stan_xa"]:.1e}, stan_xc {kontrola["stan_xc"]:.1e}', flush=True)
        del net
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    L_ = np.array(L_tab, np.float64); S_ = np.array(S_tab, np.float64); B_ = np.array(B_tab, np.float64)
    np.savez_compressed(out / 'wyniki.npz', L=L_, S=S_, B=B_, kroki=np.array(args.kroki), ident=np.array(U.ident),
                        kol_L=np.array(['krok', 'klatka', 'blok', 'token', 'narz'] + KOL_L),
                        kol_S=np.array(['krok', 'klatka', 'blok', 'token', 'narz', 'kier'] + KOL_S),
                        kol_B=np.array(['krok', 'klatka', 'blok', 'token', 'narz', 'kier', 'delta'] + KOL_B),
                        kier=np.array(KIER), war=np.array(WAR), delty=np.array(DELTY))
    (out / 'podsumowanie.json').write_text(json.dumps(podsumuj(L_, S_, B_, args, czasy), indent=1, ensure_ascii=False), encoding='utf-8')
    print('zapisano', out, flush=True)


def podsumuj(L, S, B, args, czasy):
    q = lambda v: float(np.median(v)) if len(v) else float('nan')
    s = dict(uklad=args.uklad, wagi=str(args.wagi), kroki=[int(k) for k in args.kroki], czasy=czasy, L={}, S={}, B={})
    for k in args.kroki:
        for grupa, maska in (('wszystkie', None), ('narz', 1.0), ('tlo', 0.0)):
            l = L[L[:, 0] == k]; ss = S[S[:, 0] == k]; bb = B[B[:, 0] == k]
            if maska is not None:
                l, ss, bb = l[l[:, 4] == maska], ss[ss[:, 4] == maska], bb[bb[:, 4] == maska]
            NB1 = int(L[:, 2].max()) + 1
            s['L'].setdefault(str(k), {})[grupa] = {c: [q(l[l[:, 2] == b, 5 + j]) for b in range(NB1)] for j, c in enumerate(KOL_L)}
            s['S'].setdefault(str(k), {})[grupa] = {kn: {c: [q(ss[(ss[:, 2] == b) & (ss[:, 5] == ki), 6 + j]) for b in range(NB1)]
                                                         for j, c in enumerate(KOL_S)} for ki, kn in enumerate(KIER)}
            s['B'].setdefault(str(k), {})[grupa] = {f'{kn}_d{d}': {c: [q(bb[(bb[:, 2] == b) & (bb[:, 5] == ki) & np.isclose(bb[:, 6], d), 7 + j])
                                                                       for b in range(NB1)] for j, c in enumerate(KOL_B)}
                                                    for ki, kn in enumerate(KIER) for d in DELTY}
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--uklad', required=True, choices=['detekcja', 'cifar_maly', 'cifar_deit'])
    p.add_argument('--wagi', required=True); p.add_argument('--dane', required=True); p.add_argument('--out', required=True)
    p.add_argument('--kroki', type=int, nargs='+', required=True); p.add_argument('--paczka', type=int, default=10)
    p.add_argument('--paczka_kopii', type=int, default=60)
    p.add_argument('--cpus', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    p.add_argument('--proba', type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(max(1, a.cpus))
    pomiar(a)


if __name__ == '__main__':
    main()
