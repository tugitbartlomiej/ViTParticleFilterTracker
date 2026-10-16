"""FDT wzdłuż kierunku szumu (2026-10-13): czy token ma pułapkę tam, gdzie pcha go szum paczek?

Wyniki fal 1–2: odpowiedź χ na stałe pole w losowym kierunku (i osobno w widocznej i niewidocznej podprzestrzeni fc2) rośnie
liniowo bez nasycenia, a przemieszczenie z szumu się nasyca; T_eff = Δ/χ spada jak ~1/t. W układzie liniowym ta sama siła zwrotna
działa na przesunięcie z szumu i od pola, więc szum musi omijać kierunki płaskie (kowariancja szumu ≈ hesjan/Fisher: temperatura
kierunku ∝ jego sztywność). Test: pole wzdłuż głównego kierunku szumu każdego tokenu (S) wobec losowego (L).
Przewidywanie hipotezy: χ_S nasyca się (pułapka), T_S prawie stałe; χ_L rośnie liniowo, T_L → 0.

Faza wstępna (--pilot kroków, strumienie paczek C i D — inne niż A i B pomiaru, żeby kierunek nie był dopasowany do szumu pomiaru):
dwie repliki bez pola, różnica z_C − z_D co krok → SVD na token (kroki × 768) → ε_S,j = pierwszy prawy wektor osobliwy
(znak losowy), udział pierwszej składowej w wariancji szumu zapisany. Potem wszystkie repliki wracają do stanu t_w.
Repliki pomiaru: a0, b0, ±S, ±3S, ±L (L: losowy Gauss na token, jak fala 1).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import model as M
import trening as T
import dane as D

t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stan', required=True)
    ap.add_argument('--bieg', required=True, help='katalog biegu z sondami (dane/metadata.json, gesty/SONDY.json)')
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kroki', type=int, default=2000)
    ap.add_argument('--gesto', type=int, default=300)
    ap.add_argument('--co', type=int, default=5)
    ap.add_argument('--frac_s', type=float, default=3e-4, help='|∇L_pola S| / |∇L|')
    ap.add_argument('--frac_l', type=float, default=3e-4, help='|∇L_pola L| / |∇L|')
    ap.add_argument('--pilot', type=int, default=150, help='kroki fazy wstępnej (kierunek szumu)')
    ap.add_argument('--beta1', type=float, default=None)
    ap.add_argument('--wd', type=float, default=None, help='nadpisanie weight decay (grupa macierzy)')
    ap.add_argument('--n_sond', type=int, default=12)
    ap.add_argument('--ziarno', type=int, default=1013)
    ap.add_argument('--zipy', default='/home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/annotations')
    ap.add_argument('--proba', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from trening_pokretla import wiersze_35000, wczytaj_z_zipow

    meta_b = json.load(open(os.path.join(a.bieg, 'dane', 'metadata.json'), encoding='utf-8'))
    sondy = json.load(open(os.path.join(a.bieg, 'gesty', 'SONDY.json'), encoding='utf-8'))['sondy'][:a.n_sond]
    sondy_pool = list(sondy)
    st = torch.load(a.stan, map_location='cuda', weights_only=False)
    krok0 = int(st['krok'])
    log(f'stan {a.stan} (krok {krok0}), lr {a.lr}, β1 {a.beta1 or 0.9}, wd {a.wd}, sondy {sondy}')

    wiersze = wiersze_35000(meta_b['manifest'])
    if a.proba:
        wyb = sorted(set(np.random.default_rng(0).choice(len(wiersze), 512, replace=False).tolist()) | set(sondy))
        mapa = {w: i for i, w in enumerate(wyb)}
        wiersze = [wiersze[i] for i in wyb]; sondy = [mapa[s] for s in sondy]; a.kroki = 40; a.gesto = 20; a.co = 5
    X = wczytaj_z_zipow(wiersze, a.zipy)
    zb = D.Zbior(wiersze, X)
    S = len(sondy)
    Xp = torch.from_numpy(np.ascontiguousarray(X[sondy])).cuda().permute(0, 3, 1, 2).float() / 255.
    log(f'{len(zb)} klatek, {S} sond')

    T.LR = a.lr

    def replika():
        u = T.Uczen(47)
        u.wczytaj(dict(net=st['net'], opt=copy.deepcopy(st['opt']), krok=st['krok']))   # głęboka kopia stanu AdamW
        for b in u.net.bloki:
            b.attn.fused_attn = False
        for i, g in enumerate(u.opt.param_groups):
            g['lr'] = a.lr
            if a.beta1 is not None:
                g['betas'] = (a.beta1, g['betas'][1])
            if a.wd is not None and g['weight_decay'] > 0:
                g['weight_decay'] = a.wd
        u.net.train()
        return u

    #       nazwa  pole  mnożnik  strumień
    PLAN = [('a0', None, 0., 'A'), ('b0', None, 0., 'B'), ('pS', 'S', 1., 'A'), ('mS', 'S', -1., 'A'),
            ('pS3', 'S', 3., 'A'), ('mS3', 'S', -3., 'A'), ('pL', 'L', 1., 'A'), ('mL', 'L', -1., 'A')]
    R = {n: replika() for n, _, _, _ in PLAN}
    TOK = 600
    N = S * TOK

    # podprzestrzenie z W₂ ostatniego bloku w t_w (tylko do rozkładu przemieszczeń)
    W2 = R['a0'].net.bloki[-1].mlp.fc2.weight.detach().float()           # (192, 768)
    Q, _ = torch.linalg.qr(W2.T)
    PV = Q @ Q.T
    PN = torch.eye(768, device='cuda') - PV
    gen = torch.Generator(device='cuda').manual_seed(a.ziarno)
    EL = torch.randn(S, TOK, 768, generator=gen, device='cuda'); EL = EL / EL.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def latent(net):
        net.eval(); net.zapisuj_latent(True)
        out = []
        for s in range(0, S, 6):
            _, lat = net(Xp[s:s + 6])
            out.append(lat.float())
        net.zapisuj_latent(False); net.train()
        return torch.cat(out)

    def strata_pola(net, E):
        net.eval(); net.zapisuj_latent(True)
        tot = 0.0
        for s in range(0, S, 6):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, lat = net(Xp[s:s + 6])
            tot = tot + (lat.float() * E[s:s + 6]).sum()
        net.zapisuj_latent(False); net.train()
        return -tot / N

    cel0, cel1 = zb.cele

    class Strumien:
        def __init__(self, ziarno):
            self.rng = np.random.default_rng(ziarno); self.kol = []

        def paczka(self):
            P = T.PACZKA
            if len(self.kol) < P:
                self.kol += self.rng.permutation(len(zb)).tolist()
            idx = np.array(self.kol[:P]); self.kol = self.kol[P:]
            odb = self.rng.random(P) < .5
            x = torch.from_numpy(zb.X[idx]).cuda().permute(0, 3, 1, 2).float() / 255.
            ob = torch.from_numpy(odb).cuda()
            x = torch.where(ob[:, None, None, None], x.flip(3), x)
            aa = torch.from_numpy(self.rng.uniform(.8, 1.2, (P, 2)).astype(np.float32)).cuda()
            m = x.mean((1, 2, 3), keepdim=True)
            x = ((x - m) * aa[:, 0, None, None, None] + m * aa[:, 1, None, None, None]).clamp(0, 1)
            it = torch.from_numpy(idx); o_ = torch.from_numpy(odb)
            heat = torch.where(o_[:, None, None], cel1[0][it], cel0[0][it]).cuda()
            ltrb = torch.where(o_[:, None, None, None], cel1[1][it], cel0[1][it]).cuda()
            poz = torch.where(o_[:, None, None], cel1[2][it], cel0[2][it]).cuda()
            return x, heat, ltrb, poz

    strumienie = {'A': Strumien(a.ziarno), 'B': Strumien(a.ziarno + 1)}

    # ---------------------------------------------------------------- faza wstępna: kierunek szumu każdego tokenu
    def krok_bez_pola(u, paczka):
        x, heat, ltrb, poz = paczka
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = u.net(x)
        L, _ = M.strata(o, heat, ltrb, poz)
        u.opt.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(u.net.parameters(), 1.0); u.opt.step()

    uc, ud = R['a0'], R['b0']
    sc, sd = Strumien(a.ziarno + 7), Strumien(a.ziarno + 8)
    pil = a.pilot if not a.proba else 20
    Dz = torch.empty(pil, S, TOK, 768, device='cuda', dtype=torch.float16)
    for k in range(pil):
        krok_bez_pola(uc, sc.paczka()); krok_bez_pola(ud, sd.paczka())
        Dz[k] = (latent(uc.net) - latent(ud.net)).half()
    ES = torch.empty(S, TOK, 768, device='cuda'); udzial1 = torch.empty(S, TOK, device='cuda')
    Dz = Dz.permute(1, 2, 0, 3).reshape(S * TOK, pil, 768)
    for c in range(0, S * TOK, 300):
        X_ = Dz[c:c + 300].float()
        X_ = X_ - X_.mean(1, keepdim=True)
        _, sv, Vh = torch.linalg.svd(X_, full_matrices=False)
        ES.view(-1, 768)[c:c + 300] = Vh[:, 0, :]
        udzial1.view(-1)[c:c + 300] = sv[:, 0] ** 2 / (sv ** 2).sum(1)
    del Dz
    znak = torch.where(torch.rand(S, TOK, generator=gen, device='cuda') < .5, -1.0, 1.0)
    ES = ES * znak[..., None]
    cos_SL = float((ES * EL).sum(-1).abs().mean())
    udz_V = float(((ES @ PV) ** 2).sum(-1).mean())
    log(f'faza wstępna {pil} kroków: udział 1. składowej szumu {float(udzial1.mean()):.3f} (mediana {float(udzial1.median()):.3f}), '
        f'|cos(ε_S, ε_L)| {cos_SL:.3f}, udział ε_S w podprzestrzeni widocznej {udz_V:.3f} (losowy 0,25)')
    for n in R:                                                  # powrót wszystkich replik do stanu t_w
        R[n].net.load_state_dict(st['net']); R[n].opt.load_state_dict(copy.deepcopy(st['opt']))
        for g in R[n].opt.param_groups:
            g['lr'] = a.lr
            if a.beta1 is not None:
                g['betas'] = (a.beta1, g['betas'][1])
            if a.wd is not None and g['weight_decay'] > 0:
                g['weight_decay'] = a.wd
    POLA = {'S': ES, 'L': EL}

    # kalibracja h0 dla obu pól
    u = R['a0']; params = list(u.net.parameters())
    sA = Strumien(a.ziarno + 99)
    norma = lambda: float(torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)))
    gl, gf = [], {'S': [], 'L': []}
    for _ in range(5):
        x, heat, ltrb, poz = sA.paczka()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = u.net(x)
        L, _ = M.strata(o, heat, ltrb, poz)
        u.opt.zero_grad(set_to_none=True); L.backward(); gl.append(norma())
        for k, E in POLA.items():
            u.opt.zero_grad(set_to_none=True); strata_pola(u.net, E).backward(); gf[k].append(norma())
    u.opt.zero_grad(set_to_none=True)
    u.net.load_state_dict(st['net'])
    frac = {'S': a.frac_s, 'L': a.frac_l}
    h0 = {k: frac[k] * float(np.median(gl)) / float(np.median(gf[k])) for k in POLA}
    log(f'kalibracja: |∇L| {np.median(gl):.3g}, |∇L_S(h=1)| {np.median(gf["S"]):.3g}, |∇L_L(h=1)| {np.median(gf["L"]):.3g} → h0 {h0}')

    z0 = latent(R['a0'].net)
    m0 = {k: (z0 * E).sum(-1) for k, E in POLA.items()}
    zap = {'kroki': [], 'strata_a': []}

    def dodaj(k, v):
        zap.setdefault(k, []).append(v.cpu().numpy() if torch.is_tensor(v) else v)

    def zapisz(k, Z, info):
        zap['kroki'].append(k); zap['strata_a'].append(info.get('L', np.nan))
        for K, E in POLA.items():
            mm = {n: (Z[n] * E).sum(-1) for n in Z}
            dodaj(f'm_a_{K}', mm['a0'] - m0[K]); dodaj(f'm_b_{K}', mm['b0'] - m0[K])
            dodaj(f'd1_{K}', (mm[f'p{K}'] - mm[f'm{K}']) / (2 * h0[K]))
            dodaj(f's1_{K}', (mm[f'p{K}'] + mm[f'm{K}']) / 2 - mm['a0'])
            if f'p{K}3' in mm:
                dodaj(f'd3_{K}', (mm[f'p{K}3'] - mm[f'm{K}3']) / (6 * h0[K]))
        da = Z['a0'] - z0; dab = Z['a0'] - Z['b0']
        dodaj('msdV_a', ((da @ PV) ** 2).sum(-1)); dodaj('msdN_a', ((da @ PN) ** 2).sum(-1))
        dodaj('abV', ((dab @ PV) ** 2).sum(-1)); dodaj('abN', ((dab @ PN) ** 2).sum(-1))
        dodaj('respS', ((Z['pS'] - Z['mS']) / (2 * h0['S'])).norm(dim=-1))
        dodaj('respL', ((Z['pL'] - Z['mL']) / (2 * h0['L'])).norm(dim=-1))

    zapisz(0, {n: z0 for n, _, _, _ in PLAN}, {})
    for n, _, _, _ in PLAN:
        R[n].net.load_state_dict(st['net'])

    for k in range(1, a.kroki + 1):
        paczki = {s: strumienie[s].paczka() for s in strumienie}
        info = {}
        for n, pole, mn, s in PLAN:
            u = R[n]; x, heat, ltrb, poz = paczki[s]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, _ = u.net(x)
            L, _ = M.strata(o, heat, ltrb, poz)
            if pole is not None:
                L = L + mn * h0[pole] * strata_pola(u.net, POLA[pole])
            u.opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(u.net.parameters(), 1.0)
            u.opt.step()
            if n == 'a0':
                info = dict(L=float(L))
        if k <= a.gesto or k % a.co == 0 or k == a.kroki:
            zapisz(k, {n: latent(R[n].net) for n, _, _, _ in PLAN}, info)
        if k % 100 == 0:
            c = {K: float(np.mean(zap[f'd1_{K}'][-1])) for K in POLA}
            lin = float(np.mean(zap['d3_S'][-1])) / max(c['S'], 1e-12)
            DS = 0.5 * float(np.mean(zap['m_a_S'][-1] ** 2)); DL = 0.5 * float(np.mean(zap['m_a_L'][-1] ** 2))
            log(f'krok {k}: strata {info["L"]:.4f} | χ_S {c["S"]:.4g} Δ_S {DS:.4g} T_S {DS / max(c["S"], 1e-12):.4g} χ3/χ {lin:.3f} | '
                f'χ_L {c["L"]:.4g} Δ_L {DL:.4g} T_L {DL / max(c["L"], 1e-12):.4g}')

    wyn = {k: np.asarray(v, np.float32) for k, v in zap.items()}
    wyn['kroki'] = np.asarray(zap['kroki'], np.int64)
    wyn['udzial1_szumu'] = udzial1.cpu().numpy()
    np.savez(os.path.join(a.out, 'FDT3.npz'), **wyn)
    json.dump(dict(opis=__doc__, stan=a.stan, bieg=a.bieg, krok_tw=krok0, lr=a.lr, beta1=a.beta1 or 0.9, wd=a.wd, h0=h0, frac=frac,
                   grad_strata=float(np.median(gl)), grad_pol_h1={k: float(np.median(v)) for k, v in gf.items()}, sondy=sondy_pool,
                   pilot=pil, udzial1_szumu_sr=float(udzial1.mean()), cos_S_L=cos_SL, udzial_S_widoczna=udz_V,
                   kroki=a.kroki, gesto=a.gesto, co=a.co, ziarno=a.ziarno, plan=PLAN, proba=a.proba, gpu=torch.cuda.get_device_name(0),
                   czas_s=round(time.time() - t0)),
              open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('zapisano', a.out)


if __name__ == '__main__':
    main()
