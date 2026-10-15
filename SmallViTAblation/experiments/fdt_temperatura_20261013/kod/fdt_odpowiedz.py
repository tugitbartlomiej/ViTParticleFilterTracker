"""FDT: temperatura efektywna tokenów w czasie uczenia (2026-10-13).

Pytanie: czy ruch tokenów wokół pułapek w czasie uczenia da się opisać równaniem Langevina z jedną temperaturą?
Same drgania mówią tylko T/K (temperatura przez sztywność). Odpowiedź na małą siłę mówi 1/K. Razem dają T osobno.
Test fluktuacyjno-dyssypacyjny (FDT) w postaci przemieszczeń (Cugliandolo–Kurchan–Parisi):
    równowaga:            Δ(t, t_w) = ½⟨(m(t) − m(t_w))²⟩ = T · χ(t, t_w)       (prosta w wykresie χ od Δ, nachylenie 1/T)
    dwie temperatury:     prosta łamie się: szybkie drgania przy T_szybka, wolny dryf przy T_wolna > T_szybka.

Protokół (od zapisanego stanu biegu, w chwili t_w = krok stanu):
  * kilka replik tej samej sieci (wagi + AdamW), te same paczki i augmentacje w każdym kroku (wspólny generator);
  * repliki z polem: do straty dodane L_h = −h · (1/N) Σ_j ε_j · z_j, gdzie z_j to latent 768 (GELU MLP ostatniego bloku)
    tokenu j sond (forward sond w trybie eval, z gradientem), ε_j losowy wektor jednostkowy, niezależny dla każdego tokenu;
    pole działa od t_w do końca (protokół „pole włączone w t_w”);
  * h = ±h0 i ±3·h0 (h0 tak, by gradient pola był ułamkiem `--frac` gradientu straty przed obcięciem);
  * odpowiedź własna: χ_j(t) = [ε_j·z_j(+h) − ε_j·z_j(−h)] / (2h)  (składniki krzyżowe od pól innych tokenów znoszą się w średniej,
    bo ε są niezależne); nieliniowość: (m(+h) + m(−h))/2 − m(0) oraz χ z 3·h0 wobec h0;
  * bliźniak h = 0 z tymi samymi paczkami: poziom niedeterminizmu GPU;
  * replika z innym strumieniem paczek (h = 0): rozkład przemieszczenia na szum (różnica replik) i dryf wspólny.
Zapis co krok do `--gesto`, potem co `--co`: liczby na token (float32), bez pełnych tensorów (mała kwota na Edenie).
Uruchomienie: python fdt_odpowiedz.py --stan <pelny/stan_eNN.pt> --bieg <katalog biegu> --lr <lr> --out <katalog> [--kroki 3000]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
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
torch.use_deterministic_algorithms(True)                         # bliźniak h = 0 musi dać zero (próba 1: niedeterminizm ≈ odpowiedź)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stan', required=True)
    ap.add_argument('--bieg', required=True, help='katalog biegu (dane/metadata.json, gesty/SONDY.json)')
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kroki', type=int, default=3000)
    ap.add_argument('--gesto', type=int, default=300, help='zapis co krok do tego kroku')
    ap.add_argument('--co', type=int, default=5, help='potem zapis co tyle kroków')
    ap.add_argument('--frac', type=float, default=0.02, help='|∇L_h0| / |∇L| przed obcięciem')
    ap.add_argument('--beta1', type=float, default=None, help='nadpisanie β1 AdamW (pęd = „masa”)')
    ap.add_argument('--n_sond', type=int, default=12)
    ap.add_argument('--ziarno', type=int, default=1013)
    ap.add_argument('--zipy', default='/home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/annotations')
    ap.add_argument('--proba', action='store_true', help='512 klatek uczących, 40 kroków: sprawdzenie kodu')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from trening_pokretla import wiersze_35000, wczytaj_z_zipow

    meta_b = json.load(open(os.path.join(a.bieg, 'dane', 'metadata.json'), encoding='utf-8'))
    sondy = json.load(open(os.path.join(a.bieg, 'gesty', 'SONDY.json'), encoding='utf-8'))['sondy'][:a.n_sond]
    st = torch.load(a.stan, map_location='cuda', weights_only=False)
    krok0 = int(st['krok'])
    log(f'stan {a.stan} (krok {krok0}), lr {a.lr}, β1 {a.beta1 or 0.9}, sondy {sondy}')

    # ---------------------------------------------------------------- dane: cały zbiór uczący + sondy
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

    # ---------------------------------------------------------------- repliki
    T.LR = a.lr

    def replika():
        u = T.Uczen(47)
        # głęboka kopia: Optimizer.load_state_dict nie kopiuje tensorów na tym samym urządzeniu, więc bez niej
        # wszystkie repliki dzieliłyby jeden pęd i jedno v AdamW (próba 2: bliźniak h = 0 ≠ 0)
        u.wczytaj(dict(net=st['net'], opt=copy.deepcopy(st['opt']), krok=st['krok']))
        for b in u.net.bloki:                                    # łączona atencja (SDPA) ma niedeterministyczny backward
            b.attn.fused_attn = False
        for g in u.opt.param_groups:
            g['lr'] = a.lr
            if a.beta1 is not None:
                g['betas'] = (a.beta1, g['betas'][1])
        u.net.train()
        return u

    #            nazwa      h (w jednostkach h0)  strumień paczek
    PLAN = [('a0', 0.0, 'A'), ('a0b', 0.0, 'A'), ('p1', 1.0, 'A'), ('m1', -1.0, 'A'), ('p3', 3.0, 'A'), ('m3', -3.0, 'A'), ('b0', 0.0, 'B')]
    R = {n: replika() for n, _, _ in PLAN}
    TOK = 600
    gen = torch.Generator(device='cuda').manual_seed(a.ziarno)
    E = torch.randn(S, TOK, 768, generator=gen, device='cuda')
    E = E / E.norm(dim=-1, keepdim=True)                         # ε_j: jednostkowe, niezależne dla tokenów
    N = S * TOK

    @torch.no_grad()
    def latent(net):
        net.eval(); net.zapisuj_latent(True)
        out = []
        for s in range(0, S, 6):
            _, lat = net(Xp[s:s + 6])                            # fp32, bez autocast: dokładne różnice
            out.append(lat.float())
        net.zapisuj_latent(False); net.train()
        return torch.cat(out)                                    # (S, 600, 768)

    def strata_pola(net):
        """−(1/N) Σ ε·z na sondach (eval, z gradientem): gradient przesuwa z_j wzdłuż +ε_j."""
        net.eval(); net.zapisuj_latent(True)
        tot = 0.0
        for s in range(0, S, 6):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, lat = net(Xp[s:s + 6])
            tot = tot + (lat.float() * E[s:s + 6]).sum()
        net.zapisuj_latent(False); net.train()
        return -tot / N

    # ---------------------------------------------------------------- paczki (kopia logiki Uczen.kroki, wspólna dla replik strumienia)
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

    # ---------------------------------------------------------------- kalibracja h0 na kopii (bez kroków)
    u = R['a0']; params = list(u.net.parameters())
    sA = Strumien(a.ziarno + 99)
    gl, gf = [], []
    for _ in range(5):
        x, heat, ltrb, poz = sA.paczka()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = u.net(x)
        L, _ = M.strata(o, heat, ltrb, poz)
        u.opt.zero_grad(set_to_none=True); L.backward()
        gl.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))))
        u.opt.zero_grad(set_to_none=True); strata_pola(u.net).backward()
        gf.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))))
    u.opt.zero_grad(set_to_none=True)
    u.net.load_state_dict(st['net'])                             # bufory BN zmienione przez forward w trybie train
    h0 = a.frac * float(np.median(gl)) / float(np.median(gf))
    log(f'kalibracja: |∇L| {np.median(gl):.3g}, |∇L_pola(h=1)| {np.median(gf):.3g} → h0 {h0:.4g}')

    # ---------------------------------------------------------------- krok zero: położenia startowe
    z0 = latent(R['a0'].net)
    m0 = (z0 * E).sum(-1)                                        # (S, 600)
    zap = dict(kroki=[], m_a=[], m_b=[], d1=[], d3=[], s1=[], s3=[], blizniak=[], msd_a=[], msd_b=[], ab=[], strata_a=[], gn_a=[])

    def zapisz(k, Z, info):
        mm = {n: (Z[n] * E).sum(-1) for n in Z}
        zap['kroki'].append(k)
        zap['m_a'].append((mm['a0'] - m0).cpu().numpy())
        zap['m_b'].append((mm['b0'] - m0).cpu().numpy())
        zap['d1'].append(((mm['p1'] - mm['m1']) / (2 * h0)).cpu().numpy())
        zap['d3'].append(((mm['p3'] - mm['m3']) / (6 * h0)).cpu().numpy())
        zap['s1'].append(((mm['p1'] + mm['m1']) / 2 - mm['a0']).cpu().numpy())
        zap['s3'].append(((mm['p3'] + mm['m3']) / 2 - mm['a0']).cpu().numpy())
        zap['blizniak'].append((mm['a0b'] - mm['a0']).cpu().numpy())
        zap['msd_a'].append(((Z['a0'] - z0) ** 2).sum(-1).cpu().numpy())
        zap['msd_b'].append(((Z['b0'] - z0) ** 2).sum(-1).cpu().numpy())
        zap['ab'].append(((Z['a0'] - Z['b0']) ** 2).sum(-1).cpu().numpy())
        zap['strata_a'].append(info.get('L', np.nan)); zap['gn_a'].append(info.get('gn', np.nan))

    zapisz(0, {n: z0 for n, _, _ in PLAN}, {})
    for n, _, _ in PLAN:                                         # wszystkie repliki startują z identycznym stanem
        R[n].net.load_state_dict(st['net'])

    for k in range(1, a.kroki + 1):
        paczki = {s: strumienie[s].paczka() for s in strumienie}
        info = {}
        for n, hh, s in PLAN:
            u = R[n]; x, heat, ltrb, poz = paczki[s]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o, _ = u.net(x)
            L, _ = M.strata(o, heat, ltrb, poz)
            if hh != 0.0:
                L = L + hh * h0 * strata_pola(u.net)
            u.opt.zero_grad(set_to_none=True)
            L.backward()
            gn = torch.nn.utils.clip_grad_norm_(u.net.parameters(), 1.0)
            u.opt.step()
            if n == 'a0':
                info = dict(L=float(L), gn=float(gn))
        if k <= a.gesto or k % a.co == 0 or k == a.kroki:
            zapisz(k, {n: latent(R[n].net) for n, _, _ in PLAN}, info)
        if k % 100 == 0:
            d1 = np.mean(zap['d1'][-1]); bl = np.abs(zap['blizniak'][-1]).mean(); da = np.abs(zap['d1'][-1] * 2 * h0).mean()
            log(f'krok {k}: strata {info["L"]:.4f}, χ śr {d1:.4g}, χ(3h)/χ(h) {np.mean(zap["d3"][-1]) / max(d1, 1e-12):.3f}, '
                f'|bliźniak| {bl:.3g} vs |odpowiedź| {da:.3g}, MSD_a {np.mean(zap["msd_a"][-1]):.4g}, |a−b|² {np.mean(zap["ab"][-1]):.4g}')

    wyn = {k: np.asarray(v, np.float32) for k, v in zap.items()}
    wyn['kroki'] = np.asarray(zap['kroki'], np.int64)
    np.savez(os.path.join(a.out, 'FDT.npz'), **wyn)
    json.dump(dict(opis=__doc__, stan=a.stan, bieg=a.bieg, krok_tw=krok0, lr=a.lr, beta1=a.beta1 or 0.9, h0=h0, frac=a.frac,
                   grad_strata=float(np.median(gl)), grad_pola_h1=float(np.median(gf)), sondy=sondy, kroki=a.kroki, gesto=a.gesto, co=a.co,
                   ziarno=a.ziarno, plan=PLAN, proba=a.proba, gpu=torch.cuda.get_device_name(0), czas_s=round(time.time() - t0)),
              open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('zapisano', a.out)


if __name__ == '__main__':
    main()
