"""SYGNAŁY SIECI JAKO WEJŚCIA DLA KOOPMANA I LANGEVINA NA PUŁAPKACH (2026-10-13).

Powtórka uczenia od zapisanego stanu (wagi + AdamW, przepis biegu 35 000: paczka 32, augmentacja, AdamW, obcięcie gradientu 1,
bf16) przez --kroki kroków; w każdym kroku dla --n_sond zdjęć-sond zapis:
  z6     — latent ostatniego bloku (GELU MLP, 768 laserów) w losowym rzucie 128D (ta sama macierz co obserwator_rp, ziarno 0),
           z niego liczy się pułapki (średnia krocząca lub bloki);
  xin    — stan tokenu WCHODZĄCY do ostatniego bloku (wyjście bloku 5, 192D): wejście „z góry sieci” (wiązka wejściowa);
  zwl    — „własna siła” zdjęcia: przesunięcie z6 po kroku wag tylko na stracie tego zdjęcia,
           Δθ_wł = −lr · (1 − β1) · g_wł / (32 · max(|g_paczki|, 1)) / (√v̂ + ε)  (próba: bez 1/32 i przycięcia |z_wł| ≈ 7, za dużo)  (wkład tego gradientu do kroku AdamW przy bieżącym stanie v),
           liczone różnicą skończoną w fp32 (eval, bez augmentacji) — da się policzyć w KAŻDYM kroku, choć zdjęcie bywa w paczce
           raz na epokę;
  skalary — czy sonda jest w paczce, strata paczki przed krokiem, strata sondy (eval), normy kroku wag w łodydze, 6 blokach i głowie.
Zapis porcjami do npz (float16 dla tensorów), mała kwota na Edenie: 4 sondy × 1100 kroków ≈ 2,4 GB.
Uruchomienie: python sygnaly_sieci.py --stan .../stan_eNN.pt --bieg <katalog biegu z sondami> --lr 5e-4 --out DIR [--kroki 1100]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import dane as D
import model as M
import trening as T

t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stan', required=True); ap.add_argument('--bieg', required=True); ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--kroki', type=int, default=1100); ap.add_argument('--n_sond', type=int, default=4)
    ap.add_argument('--porcja', type=int, default=100); ap.add_argument('--ziarno', type=int, default=2026)
    ap.add_argument('--zipy', default='/home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/annotations')
    ap.add_argument('--proba', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from trening_pokretla import wiersze_35000, wczytaj_z_zipow
    meta_b = json.load(open(os.path.join(a.bieg, 'dane', 'metadata.json'), encoding='utf-8'))
    sondy = json.load(open(os.path.join(a.bieg, 'gesty', 'SONDY.json'), encoding='utf-8'))['sondy'][:a.n_sond]
    st = torch.load(a.stan, map_location='cuda', weights_only=False)
    wiersze = wiersze_35000(meta_b['manifest'])
    if a.proba:
        wyb = sorted(set(np.random.default_rng(0).choice(len(wiersze), 512, replace=False).tolist()) | set(sondy))
        mapa = {w: i for i, w in enumerate(wyb)}; wiersze = [wiersze[i] for i in wyb]; sondy = [mapa[s] for s in sondy]; a.kroki = 30; a.porcja = 10
    X = wczytaj_z_zipow(wiersze, a.zipy); zb = D.Zbior(wiersze, X); S = len(sondy)
    Xp = torch.from_numpy(np.ascontiguousarray(X[sondy])).cuda().permute(0, 3, 1, 2).float() / 255.
    cel0, cel1 = zb.cele
    it_s = torch.tensor(sondy)
    heat_s, ltrb_s, poz_s = cel0[0][it_s].cuda(), cel0[1][it_s].cuda(), cel0[2][it_s].cuda()
    T.LR = a.lr
    u = T.Uczen(47); u.wczytaj(dict(net=st['net'], opt=copy.deepcopy(st['opt']), krok=st['krok']))
    for g in u.opt.param_groups:
        g['lr'] = a.lr
    net = u.net; params = list(net.parameters()); nazwy = [n for n, _ in net.named_parameters()]
    grupy = {'lodyga': [i for i, n in enumerate(nazwy) if n.startswith('lodyga')]}
    for b in range(6):
        grupy[f'blok{b}'] = [i for i, n in enumerate(nazwy) if n.startswith(f'bloki.{b}.')]
    grupy['glowa'] = [i for i, n in enumerate(nazwy) if not (n.startswith('lodyga') or n.startswith('bloki.'))]
    gR = torch.Generator().manual_seed(0)
    R = (torch.randn(768, 128, generator=gR) / 128 ** 0.5).cuda()
    wyj = {}
    net.bloki[-2].register_forward_hook(lambda m, i, o: wyj.__setitem__('xin', o))
    log(f'stan krok {st["krok"]}, lr {a.lr}, sondy {sondy}, {len(zb)} klatek')

    @torch.no_grad()
    def odczyt():
        net.eval(); net.zapisuj_latent(True)
        _, lat = net(Xp)                                                   # fp32, bez autocast
        z = (lat.float() @ R); xin = wyj['xin'].float()
        net.zapisuj_latent(False); net.train()
        return z, xin

    skala = {'c': 1.0 / (T.PACZKA * 3.7)}                              # wkład zdjęcia w paczce: 1/32 straty × przycięcie (aktualizowane co krok)

    def sila_wlasna():
        """Przesunięcie z6 (rzut 128D) od kroku tylko na stracie każdej sondy, z bieżącym prekondycjonerem AdamW."""
        b1, b2 = u.opt.param_groups[0]['betas']; eps = u.opt.param_groups[0]['eps']
        stany = [u.opt.state.get(p, {}) for p in params]
        z0, _ = odczyt()
        wyn, straty = [], []
        theta0 = [p.detach().clone() for p in params]
        for s in range(S):
            net.eval(); net.zero_grad(set_to_none=True)
            o, _ = net(Xp[s:s + 1])
            L, _ = M.strata(o, heat_s[s:s + 1], ltrb_s[s:s + 1], poz_s[s:s + 1])
            L.backward(); straty.append(float(L))
            with torch.no_grad():
                for p, q in zip(params, stany):
                    if p.grad is None or 'exp_avg_sq' not in q:
                        continue
                    kr = float(q['step']) if 'step' in q else 1.0
                    vhat = q['exp_avg_sq'] / (1 - b2 ** kr)
                    p.add_(-a.lr * (1 - b1) / (1 - b1 ** kr) * skala['c'] * p.grad / (vhat.sqrt() + eps))
            z1, _ = odczyt()
            wyn.append((z1[s] - z0[s]))
            with torch.no_grad():
                for p, t_ in zip(params, theta0):
                    p.copy_(t_)
        net.zero_grad(set_to_none=True); net.train()
        return torch.stack(wyn), straty

    rng = np.random.default_rng(a.ziarno); kol = []
    buf = {k: [] for k in ('krok', 'z6', 'xin', 'zwl', 'w_paczce', 'strata_paczki', 'strata_sond', 'normy_krokow')}
    n_por = 0

    def zrzuc():
        nonlocal n_por
        if not buf['krok']:
            return
        np.savez(os.path.join(a.out, f'sygnaly_{n_por:03d}.npz'), **{k: np.stack(v) if k in ('z6', 'xin', 'zwl', 'normy_krokow', 'w_paczce', 'strata_sond') else np.array(v)
                                                                        for k, v in buf.items()})
        n_por += 1
        for v in buf.values():
            v.clear()

    for k in range(a.kroki + 1):
        z, xin = odczyt()
        zwl, st_s = sila_wlasna()
        # paczka (logika Uczen.kroki)
        P_ = T.PACZKA
        if len(kol) < P_:
            kol += rng.permutation(len(zb)).tolist()
        idx = np.array(kol[:P_]); kol = kol[P_:]
        odb = rng.random(P_) < .5
        x = torch.from_numpy(zb.X[idx]).cuda().permute(0, 3, 1, 2).float() / 255.
        x = torch.where(torch.from_numpy(odb).cuda()[:, None, None, None], x.flip(3), x)
        aa = torch.from_numpy(rng.uniform(.8, 1.2, (P_, 2)).astype(np.float32)).cuda()
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * aa[:, 0, None, None, None] + m * aa[:, 1, None, None, None]).clamp(0, 1)
        it = torch.from_numpy(idx); o_ = torch.from_numpy(odb)
        heat = torch.where(o_[:, None, None], cel1[0][it], cel0[0][it]).cuda()
        ltrb = torch.where(o_[:, None, None, None], cel1[1][it], cel0[1][it]).cuda()
        poz = torch.where(o_[:, None, None], cel1[2][it], cel0[2][it]).cuda()
        przed = [p.detach().clone() for p in params]
        net.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x)
        L, _ = M.strata(o, heat, ltrb, poz)
        u.opt.zero_grad(set_to_none=True); L.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); u.opt.step()
        skala['c'] = 1.0 / (T.PACZKA * max(gn, 1.0))                      # jak w prawdziwym kroku: strata/32, potem przycięcie do normy 1
        with torch.no_grad():
            normy = [float(torch.sqrt(sum(((params[i] - przed[i]) ** 2).sum() for i in ii))) if ii else 0.0 for ii in grupy.values()]
        # zapis: stan PRZED krokiem k (z, xin, siła własna w tym stanie) i wejścia kroku k (paczka, normy kroku)
        buf['krok'].append(k); buf['z6'].append(z.half().cpu().numpy()); buf['xin'].append(xin.half().cpu().numpy())
        buf['zwl'].append(zwl.half().cpu().numpy()); buf['w_paczce'].append(np.isin(np.array(sondy), idx))
        buf['strata_paczki'].append(float(L)); buf['strata_sond'].append(np.array(st_s, np.float32)); buf['normy_krokow'].append(np.array(normy, np.float32))
        if len(buf['krok']) >= a.porcja:
            zrzuc()
        if k % 100 == 0:
            log(f'krok {k}: strata {float(L):.4f}, |z_wł| {float(zwl.norm(dim=-1).mean()):.3g}, |Δz6 (wejście)| —, sondy w paczce {int(np.isin(np.array(sondy), idx).sum())}')
    zrzuc()
    json.dump(dict(opis=__doc__, stan=a.stan, krok_startu=int(st['krok']), lr=a.lr, sondy=sondy, kroki=a.kroki, grupy_norm=list(grupy),
                   porcji=n_por, proba=a.proba, czas_s=round(time.time() - t0)), open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('zapisano', a.out)


if __name__ == '__main__':
    main()
