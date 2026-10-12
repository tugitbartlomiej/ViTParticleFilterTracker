"""TEST JĄDRA (2026-10-11): czy pole przesunięć tokenów (korelacje ruchliwości sąsiadów, między zdjęciami, rozkład energii kroku)
jest tym, co przewiduje sama struktura sieci — wspólne wagi θ i Jakobian J_i = ∂z_i/∂θ — a nie oddziaływanie między cząstkami?

Z zapisanego stanu (wagi + AdamW) biegu z pokrętłami liczymy przesunięcia tokenów sond Δz_i = z_i(θ + Δθ) − z_i(θ) dla kilku rodzin Δθ:
  los_prek   — losowe kierunki gradientu przepuszczone przez prekondycjoner AdamW (1/(√v̂+ε)), norma jak prawdziwy krok;
  los_izo    — losowe kierunki izotropowe w przestrzeni wag o tej samej normie;
  paczka     — prawdziwy krok AdamW (ze zapisanego stanu pędu) na losowej paczce 32 klatek uczących (bez augmentacji);
  srednia    — krok AdamW na średnim gradiencie wszystkich paczek (część „siły”);
  szum       — paczka − srednia (część „szumu” minibatcha, w przybliżeniu).
Każdą rodzinę traktujemy jak K kroków pseudo-trajektorii (P_k = P_{k−1} + Δz_k w rzucie 128D) i liczymy TE SAME miary co dla
zmierzonych przyrostów (dh.analiza τ=1: C1 i powłoki, między zdjęciami, cos; dh.rozklad_wariancji τ=1), a obok — zmierzone przyrosty
z gesty w oknie 200 kroków przed stanem. Dodatkowo mapa ruchliwości: Spearman między |Δz_i| przewidzianym (los_prek) a zmierzonym.
Uruchomienie (Eden, GPU): python jadro_pola.py <folder_biegu> <pelny/stan_eNN.pt> <out.json> [--n_kier 100] [--n_paczek 40] [--okno od do]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dane as D  # noqa: E402
import trening as T  # noqa: E402
import model as M  # noqa: E402
import dh  # noqa: E402
import odciski as O  # noqa: E402
from trening_pokretla import wiersze_35000, wczytaj_z_zipow  # noqa: E402

t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
torch.backends.cudnn.allow_tf32 = False                      # pełne fp32 w różnicach skończonych (TF32 w konwolucjach łodygi dawało szum ~40 % energii kroku)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('stan'); ap.add_argument('out')
    ap.add_argument('--n_kier', type=int, default=100)
    ap.add_argument('--n_paczek', type=int, default=40)
    ap.add_argument('--okno', type=int, nargs=2, default=None, help='okno zmierzonych przyrostów (domyślnie 200 kroków przed krokiem stanu)')
    ap.add_argument('--zipy', default='/home2/faculty/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking/annotations')
    a = ap.parse_args()
    RUN = a.run
    meta = json.load(open(os.path.join(RUN, 'dane', 'metadata.json'), encoding='utf-8'))
    pk = meta.get('pokretla', {})
    sondy = json.load(open(os.path.join(RUN, 'gesty', 'SONDY.json'), encoding='utf-8'))
    TOK = int(sondy.get('tokenow', 600))
    baza = np.load(os.path.join(RUN, 'gesty', 'BAZA_PCA32.npz'))
    R = torch.from_numpy(baza['comp'].astype(np.float32)).cuda()          # (128, 768); mean = 0 dla rzutu losowego
    srednia_b = torch.from_numpy(baza['mean'].astype(np.float32)).cuda()
    st = torch.load(a.stan, map_location='cuda', weights_only=False)
    krok = int(st['krok'])
    log(f'bieg {os.path.basename(RUN)}, stan krok {krok}, model {pk.get("model", "vit")}, opt {pk.get("opt", "adamw")}, lr {meta.get("lr")}, tokenów {TOK}')

    # --- model i optymalizator ze stanu
    T.LR = float(meta.get('lr', T.LR)); T.PACZKA = int(pk.get('paczka', 32))
    u = T.Uczen(int(pk.get('ziarno', 47)))
    if pk.get('model') == 'cnn':
        import model_cnn as MC
        u.net = MC.MalyCNN(jadro=int(pk.get('jadro', 5))).to(u.dev)
    dec = [p for n, p in u.net.named_parameters() if p.ndim >= 2]; nodec = [p for n, p in u.net.named_parameters() if p.ndim < 2]
    if pk.get('opt', 'adamw') == 'adamw':
        u.opt = torch.optim.AdamW([dict(params=dec, weight_decay=float(pk.get('wd', T.WD))), dict(params=nodec, weight_decay=0)], lr=T.LR, betas=(0.9, 0.999))
    else:
        u.opt = torch.optim.SGD([dict(params=dec, weight_decay=float(pk.get('wd', 0.0))), dict(params=nodec, weight_decay=0)], lr=T.LR, momentum=float(pk.get('momentum', 0.9)))
    u.wczytaj(st)
    for g in u.opt.param_groups:
        g['lr'] = T.LR
    net = u.net
    params = [p for p in net.parameters()]
    n_param = sum(p.numel() for p in params)

    # --- klatki: sondy + próbka do paczek
    wiersze = wiersze_35000(meta['manifest'])
    rng = np.random.default_rng(123)
    probki = rng.choice(len(wiersze), a.n_paczek * T.PACZKA, replace=False)
    idx_all = list(sondy['sondy']) + [int(i) for i in probki]
    X_all = wczytaj_z_zipow([wiersze[i] for i in idx_all], a.zipy)
    zb = D.Zbior([wiersze[i] for i in idx_all], X_all)
    S = len(sondy['sondy'])
    Xs = torch.from_numpy(np.ascontiguousarray(X_all[:S])).cuda().permute(0, 3, 1, 2).float() / 255.
    cel0 = zb.cele[0]
    log(f'{S} sond, {a.n_paczek} paczek × {T.PACZKA}; parametrów {n_param}')

    @torch.no_grad()
    def latent():
        net.eval(); net.zapisuj_latent(True)
        out = []
        for s in range(0, S, 50):
            _, lat = net(Xs[s:s + 50])                                   # fp32, bez autocast: dokładne różnice skończone
            out.append(lat.float())
        net.zapisuj_latent(False)
        return torch.cat(out).reshape(-1, lat.shape[-1])                 # (S·TOK, 768)

    z0 = latent()
    theta0 = [p.detach().clone() for p in params]
    stan0 = copy.deepcopy(net.state_dict())                 # parametry I bufory (statystyki BatchNorm łodygi zmieniają się przy każdym forward w trybie train)
    nazwy_param = {n for n, _ in net.named_parameters()}
    bufory0 = {k: v.detach().clone() for k, v in net.state_dict().items() if k not in nazwy_param}   # running_mean/var/num_batches_tracked
    log(f'buforów BN: {len(bufory0)}')
    opt0 = copy.deepcopy(u.opt.state_dict())

    def przywroc():
        net.load_state_dict(stan0)
        u.opt.load_state_dict(copy.deepcopy(opt0))

    def przywroc_bufory():
        with torch.no_grad():
            sd = net.state_dict()
            for k, v in bufory0.items():
                sd[k].copy_(v)

    def dz_po_przesunieciu(delta):
        with torch.no_grad():
            for p, d in zip(params, delta):
                p.add_(d)
        z = latent() - z0
        przywroc()
        return z

    # --- prawdziwe kroki AdamW na paczkach (i średni gradient)
    pacz, pacz_wagi, bn, grad_sum = [], [], [], None
    norma_kroku = []
    for b in range(a.n_paczek):
        it = torch.arange(S + b * T.PACZKA, S + (b + 1) * T.PACZKA)
        x = torch.from_numpy(X_all[it.numpy()]).cuda().permute(0, 3, 1, 2).float() / 255.
        heat, ltrb, poz = cel0[0][it].cuda(), cel0[1][it].cuda(), cel0[2][it].cuda()
        net.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = net(x)
        L, _ = M.strata(o, heat, ltrb, poz)
        u.opt.zero_grad(set_to_none=True); L.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        g = [p.grad.detach().clone() for p in params]
        grad_sum = g if grad_sum is None else [gs + gi for gs, gi in zip(grad_sum, g)]
        bn.append(latent() - z0)                          # (a) tylko statystyki BN zmienione przez forward w trybie train (bez kroku wag)
        u.opt.step()
        with torch.no_grad():
            delta = [p.detach() - q for p, q in zip(params, theta0)]
            norma_kroku.append(float(torch.sqrt(sum((d ** 2).sum() for d in delta))))
        pacz.append(latent() - z0)                        # (b) wagi + BN: dokładnie to, co widzi obserwator w treningu
        przywroc_bufory()
        pacz_wagi.append(latent() - z0)                   # (c) same wagi (bufory BN przywrócone)
        przywroc()
        del x, o, L
    norma = float(np.mean(norma_kroku))
    log(f'paczki: {len(pacz)}, norma kroku AdamW {norma:.4f}, <|Δz|²> {float(np.mean([(z ** 2).sum(1).mean().item() for z in pacz])):.4g}')
    # krok na średnim gradiencie
    for p, gs in zip(params, grad_sum):
        p.grad = gs / a.n_paczek
    u.opt.step()
    with torch.no_grad():
        delta_sr = [p.detach() - q for p, q in zip(params, theta0)]
        norma_sr = float(torch.sqrt(sum((d ** 2).sum() for d in delta_sr)))
    przywroc_bufory()
    z_sr = latent() - z0                                  # same wagi (średni gradient), bez BN
    przywroc()
    szum = [z - z_sr for z in pacz_wagi]                  # część szumu minibatcha w samych wagach
    # kontrola numeryczna: ten sam forward bez żadnej zmiany wag (niedeterminizm GPU) — poziom szumu pomiaru różnic skończonych
    zero = [latent() - z0 for _ in range(12)]
    log(f'kontrola numeryczna: <|Δz|²> {float(np.mean([(z ** 2).sum(1).mean().item() for z in zero])):.3g} (paczka {float((pacz[0] ** 2).sum(1).mean()):.3g})')

    # --- kierunki losowe: prekondycjonowane AdamW i izotropowe, norma = norma kroku
    prek = []
    for p in params:
        s_ = u.opt.state.get(p, {})
        v = s_.get('exp_avg_sq')
        prek.append(1.0 / (v.sqrt() + 1e-8) if v is not None else torch.ones_like(p))
    gen = torch.Generator(device='cuda').manual_seed(0)
    los_prek, los_izo = [], []
    for k in range(a.n_kier):
        g = [torch.randn(p.shape, generator=gen, device='cuda') for p in params]
        d = [gi * pi for gi, pi in zip(g, prek)]
        n = torch.sqrt(sum((x ** 2).sum() for x in d)); d = [x * (norma / n) for x in d]
        los_prek.append(dz_po_przesunieciu(d))
        n = torch.sqrt(sum((x ** 2).sum() for x in g)); d = [x * (norma / n) for x in g]
        los_izo.append(dz_po_przesunieciu(d))
    log(f'kierunki losowe: {a.n_kier} prekondycjonowanych i izotropowych')

    # --- zmierzone przyrosty z gesty (okno przed krokiem stanu)
    od, do = a.okno if a.okno else (krok - 200, krok)
    Z, kr = [], []
    for p in sorted(glob.glob(os.path.join(RUN, 'gesty', 'pca32_*.npz'))):
        z = np.load(p); k = z['kroki']; m = (k >= od) & (k <= do)
        if m.any():
            Z.append(np.ascontiguousarray(z['pca32'][m].astype(np.float32)).reshape(int(m.sum()), -1, z['pca32'].shape[-1])); kr.append(k[m])
    Zm = np.concatenate(Z) if Z else None; km = np.concatenate(kr).astype(int) if Z else None
    if Zm is not None:
        assert (np.diff(km) == 1).all(), 'okno zmierzone nie co 1 krok'
        log(f'zmierzone: {len(km)} rekordów {km[0]}–{km[-1]}')

    # --- pseudo-trajektorie i te same miary
    def rzut(z):
        return ((z - srednia_b) @ R.T).cpu().numpy()                      # (N, 128)

    P0 = rzut(z0)
    N6, dnn, dsh = O.knn_w_blokach(P0, TOK, 6)
    d_shell = float(np.median(dsh))
    N6x = dh.knn_poza_blokiem(P0, TOK, 6)
    wyn = dict(bieg=os.path.basename(RUN.rstrip('/')), stan=os.path.basename(a.stan), krok=krok, parametrow=int(n_param), sond=S, tokenow=TOK, d_shell=d_shell,
               norma_kroku_adamw=norma, norma_kroku_sredni_gradient=norma_sr, n_paczek=a.n_paczek, n_kier=a.n_kier, okno_zmierzone=[int(od), int(do)], rodziny={})
    rodziny = dict(los_prek=los_prek, los_izo=los_izo, paczka=pacz, paczka_wagi=pacz_wagi, bn=bn, srednia=[z_sr], szum=szum, numeryczna=zero)
    for nazwa, lista in rodziny.items():
        if len(lista) < 3:
            E = float((lista[0] ** 2).sum(1).mean()); wyn['rodziny'][nazwa] = dict(K=1, E_768=E, E_768_na_token=[float(v) for v in (lista[0] ** 2).sum(1).cpu().numpy()[::60]])
            continue
        P = np.empty((len(lista) + 1, len(P0), P0.shape[1]), np.float32); P[0] = P0
        E768 = []
        for k, z in enumerate(lista):
            P[k + 1] = P[k] + rzut(z); E768.append(float((z ** 2).sum(1).mean()))
        kroki = np.arange(len(lista) + 1)
        ruchl = np.sqrt(np.mean([(z ** 2).sum(1).cpu().numpy() for z in lista], 0))
        wyn['rodziny'][nazwa] = dict(K=len(lista), E_768=float(np.mean(E768)), ruchliwosc=ruchl)
        try:
            h = dh.analiza(P, kroki, (0, len(lista)), [1], N6, TOK, f'{nazwa}', pole='blok', N6_cross=N6x, W=0, max_pocz=len(lista))
            hp = dh.analiza(P, kroki, (0, len(lista)), [1], N6, TOK, f'{nazwa} populacja', pole='populacja', N6_cross=N6x, W=0, max_pocz=len(lista))
            rv = dh.rozklad_wariancji(P, kroki, (0, len(lista)), [1], N6, TOK, n_pocz=len(lista), nazwa=nazwa)
            wyn['rodziny'][nazwa].update(het_blok=h['szczyt'], het_blok_cross=(h['C_cross'][0] if h['C_cross'] else None), het_populacja=hp['szczyt'],
                                         het_populacja_cross=(hp['C_cross'][0] if hp['C_cross'] else None), cos_sasiedzi=(h['cos_sasiedzi'][0] if h['cos_sasiedzi'] else None),
                                         cos_losowe=(h['cos_losowe'][0] if h['cos_losowe'] else None), rozklad=dict(zip(rv['udzial'].keys(), [v[0] for v in rv['udzial'].values()])))
        except Exception as e:                            # np. rodzina o zerowej wariancji
            log(f'{nazwa}: analiza nieudana: {e!r}'); wyn['rodziny'][nazwa]['blad'] = repr(e)
    if Zm is not None:
        h = dh.analiza(Zm, km, (int(km[0]), int(km[-1])), [1], N6, TOK, 'zmierzone', pole='blok', N6_cross=N6x, W=5, max_pocz=200)
        hp = dh.analiza(Zm, km, (int(km[0]), int(km[-1])), [1], N6, TOK, 'zmierzone populacja', pole='populacja', N6_cross=N6x, W=5, max_pocz=200)
        rv = dh.rozklad_wariancji(Zm, km, (int(km[0]), int(km[-1])), [1], N6, TOK, n_pocz=60, nazwa='zmierzone')
        Dm = np.diff(Zm, axis=0); ruchl_m = np.sqrt((Dm ** 2).sum(2).mean(0))
        wyn['zmierzone'] = dict(K=len(km) - 1, E_128=float((Dm ** 2).sum(2).mean()), het_blok=h['szczyt'], het_blok_cross=(h['C_cross'][0] if h['C_cross'] else None),
                                het_populacja=hp['szczyt'], het_populacja_cross=(hp['C_cross'][0] if hp['C_cross'] else None), cos_sasiedzi=h['cos_sasiedzi'][0], cos_losowe=h['cos_losowe'][0],
                                rozklad=dict(zip(rv['udzial'].keys(), [v[0] for v in rv['udzial'].values()])))
        from scipy.stats import spearmanr
        wyn['mapa_ruchliwosci_spearman'] = {}
        for nazwa in ('los_prek', 'los_izo', 'paczka', 'paczka_wagi', 'bn', 'szum', 'numeryczna'):
            r_ = wyn['rodziny'][nazwa].pop('ruchliwosc')
            # ruchliwość przewidywana w 768D vs zmierzona w 128D (rzut zachowuje normy w średniej); także wewnątrz zdjęć (względna)
            rel = lambda v: v / np.repeat(v.reshape(-1, TOK).mean(1), TOK)
            wyn['mapa_ruchliwosci_spearman'][nazwa] = dict(surowa=float(spearmanr(r_, ruchl_m).correlation), wzgledna=float(spearmanr(rel(r_), rel(ruchl_m)).correlation),
                                                         zdjecia=float(spearmanr(r_.reshape(-1, TOK).mean(1), ruchl_m.reshape(-1, TOK).mean(1)).correlation) if S >= 4 else None)
        log('mapa ruchliwości (Spearman przewidziana vs zmierzona):', wyn['mapa_ruchliwosci_spearman'])
    else:
        for nazwa in list(wyn['rodziny']):
            wyn['rodziny'][nazwa].pop('ruchliwosc', None)
    json.dump(wyn, open(a.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('zapisano', a.out)


if __name__ == '__main__':
    main()
