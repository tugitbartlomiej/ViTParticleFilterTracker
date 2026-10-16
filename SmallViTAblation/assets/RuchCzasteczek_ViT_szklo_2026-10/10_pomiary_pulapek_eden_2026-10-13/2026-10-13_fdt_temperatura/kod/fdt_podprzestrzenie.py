"""FDT w dwóch podprzestrzeniach latentu (2026-10-13): czy token ma pułapkę w kierunkach, których strata nie widzi?

Latent z_j (768 = 768 laserów ostatniego bloku) działa na resztę sieci tylko przez fc2: Δx_j = W₂ z_j + b₂ (W₂: 192 × 768).
  * podprzestrzeń WIDOCZNA = przestrzeń wierszy W₂ (192 kierunki): zmiana z tu zmienia wyjście sieci, więc strata stawia opór;
  * podprzestrzeń NIEWIDOCZNA = jądro W₂ (576 kierunków): kombinacje laserów, których siły się znoszą (np. para przeciwlaserów
    rozjaśniona po równo). Przesunięcie z tutaj nie zmienia nic w wyjściu tego zdjęcia; opór może dać tylko sprzężenie przez wagi
    (zmiana W₁ potrzebna do przesunięcia sondy zmienia też inne zdjęcia) i weight decay.
Pierwsza fala (fdt_odpowiedz.py, pole w losowym kierunku, w 75 % niewidocznym): odpowiedź χ rośnie liniowo bez nasycenia,
a przemieszczenie z szumu się nasyca (klatka). Hipoteza: pułapka istnieje tylko w podprzestrzeni widocznej.

Protokół jak w fdt_odpowiedz.py (repliki, te same paczki, pole −h·(1/N) Σ ε_j·z_j na sondach włączone w t_w), ale dwa pola:
  V: ε_j = P_V g_j / |P_V g_j|,  N: ε_j = P_N g_j / |P_N g_j|  (g_j losowy Gauss; P_V, P_N z W₂ repliki w t_w);
repliki: a0 (h = 0), b0 (h = 0, inny strumień paczek), ±V (h0_V), ±N (h0_N), ±3N (liniowość w N, gdzie odpowiedź rośnie najdłużej).
Zapis na token: m wzdłuż ε_V i ε_N dla a0 i b0, odpowiedzi d (±h)/2h, nieliniowości, oraz przemieszczenia w obu podprzestrzeniach
|P_V Δz|², |P_N Δz|² (a0 od t_w i a0 − b0).
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
    ap.add_argument('--frac_v', type=float, default=1e-3, help='|∇L_pola V| / |∇L|')
    ap.add_argument('--frac_n', type=float, default=3e-4, help='|∇L_pola N| / |∇L|')
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
    PLAN = [('a0', None, 0., 'A'), ('b0', None, 0., 'B'), ('pV', 'V', 1., 'A'), ('mV', 'V', -1., 'A'),
            ('pN', 'N', 1., 'A'), ('mN', 'N', -1., 'A'), ('pN3', 'N', 3., 'A'), ('mN3', 'N', -3., 'A')]
    R = {n: replika() for n, _, _, _ in PLAN}
    TOK = 600
    N = S * TOK

    # podprzestrzenie z W₂ ostatniego bloku w t_w
    W2 = R['a0'].net.bloki[-1].mlp.fc2.weight.detach().float()           # (192, 768)
    Q, _ = torch.linalg.qr(W2.T)                                         # (768, 192): baza przestrzeni wierszy
    PV = Q @ Q.T
    PN = torch.eye(768, device='cuda') - PV
    gen = torch.Generator(device='cuda').manual_seed(a.ziarno)
    G = torch.randn(S, TOK, 768, generator=gen, device='cuda')
    EV = G @ PV; EV = EV / EV.norm(dim=-1, keepdim=True)
    EN = G @ PN; EN = EN / EN.norm(dim=-1, keepdim=True)
    POLA = {'V': EV, 'N': EN}

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

    # kalibracja h0 dla obu pól
    u = R['a0']; params = list(u.net.parameters())
    sA = Strumien(a.ziarno + 99)
    norma = lambda: float(torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None)))
    gl, gf = [], {'V': [], 'N': []}
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
    frac = {'V': a.frac_v, 'N': a.frac_n}
    h0 = {k: frac[k] * float(np.median(gl)) / float(np.median(gf[k])) for k in POLA}
    log(f'kalibracja: |∇L| {np.median(gl):.3g}, |∇L_V(h=1)| {np.median(gf["V"]):.3g}, |∇L_N(h=1)| {np.median(gf["N"]):.3g} → h0 {h0}')

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
        # odpowiedź pola N widziana w podprzestrzeni V i odwrotnie (przeciek przez sprzężenie wag), pełne normy
        dodaj('respN_wV', (((Z['pN'] - Z['mN']) / (2 * h0['N'])) @ PV).norm(dim=-1))
        dodaj('respN_wN', (((Z['pN'] - Z['mN']) / (2 * h0['N'])) @ PN).norm(dim=-1))
        dodaj('respV_wV', (((Z['pV'] - Z['mV']) / (2 * h0['V'])) @ PV).norm(dim=-1))
        dodaj('respV_wN', (((Z['pV'] - Z['mV']) / (2 * h0['V'])) @ PN).norm(dim=-1))

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
            lin = float(np.mean(zap['d3_N'][-1])) / max(c['N'], 1e-12)
            DV = 0.5 * float(np.mean(zap['m_a_V'][-1] ** 2)); DN = 0.5 * float(np.mean(zap['m_a_N'][-1] ** 2))
            log(f'krok {k}: strata {info["L"]:.4f} | χ_V {c["V"]:.4g} Δ_V {DV:.4g} T_V {DV / max(c["V"], 1e-12):.4g} | '
                f'χ_N {c["N"]:.4g} Δ_N {DN:.4g} T_N {DN / max(c["N"], 1e-12):.4g} χ3/χ {lin:.3f} | '
                f'MSD V {np.mean(zap["msdV_a"][-1]):.4g} N {np.mean(zap["msdN_a"][-1]):.4g}')

    wyn = {k: np.asarray(v, np.float32) for k, v in zap.items()}
    wyn['kroki'] = np.asarray(zap['kroki'], np.int64)
    np.savez(os.path.join(a.out, 'FDT2.npz'), **wyn)
    json.dump(dict(opis=__doc__, stan=a.stan, bieg=a.bieg, krok_tw=krok0, lr=a.lr, beta1=a.beta1 or 0.9, wd=a.wd, h0=h0, frac=frac,
                   grad_strata=float(np.median(gl)), grad_pol_h1={k: float(np.median(v)) for k, v in gf.items()}, sondy=sondy_pool,
                   kroki=a.kroki, gesto=a.gesto, co=a.co, ziarno=a.ziarno, plan=PLAN, proba=a.proba, gpu=torch.cuda.get_device_name(0),
                   czas_s=round(time.time() - t0)),
              open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('zapisano', a.out)


if __name__ == '__main__':
    main()
