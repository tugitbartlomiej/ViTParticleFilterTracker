#!/usr/bin/env python3
"""Etap 3: liniowy model ruchu KAŻDEJ cząstki z sił zapisanych przez bieg_sily.py (PLAN.md).

Dla tokenu i, kroku t: dz_t = Z[t+1] − Z[t] (R^768), siły S_k(t) (k = nowa, ped, zan, buf).
1. Liniowość w wagach: R^2 dz_t ≈ S_nowa + S_ped + S_zan + S_buf (bez parametrów).
2. Udziały przyczyn: <S_k, dz> / ||dz||^2 (średnio po krokach), osobno narzędzie / tło.
3. Model wyjaśniający (wolno użyć sił bieżącego kroku, docs/23), skalary na token, MNK na krokach 300..T_UCZ,
   test T_UCZ..koniec:
     E1  dz_t = a dz_(t−1) + b S_nowa(t)                                   (równanie docs/23 na tokenie)
     E2  dz_t = a dz_(t−1) + b S_nowa(t) + c S_buf(t) + d S_zan(t)
     E3  dz_t = a dz_(t−1) − k (z_t − m_t) + b S_nowa + c S_buf + d S_zan   (Langevin z bezwładnością, sprężyną, wymuszeniem)
     W   jak E3, ale jedne parametry dla wszystkich cząstek klatki (wspólne prawo ruchu)
4. Prognoza (bez sił przyszłych kroków): P1 dz_t = a dz_(t−1) − k (z_t − m_t) jednym krokiem.
Miara: umiejętność = 1 − SSE / SS(dz) na teście (100 = idealnie, 0 = jak „brak ruchu”), mediana po tokenach.
Wynik: wyniki/DOPASOWANIE.json (i krótkie podsumowanie na stdout).

    python -u dopasuj_sily.py --dane <wyniki/pelny/dane>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

T_UCZ, OD, W = 700, 300, 50


def mnk(cechy, y):
    """Skalarne współczynniki na token: cechy [L, n, 768] lista, y [L, n, 768] -> [n, p] (MNK po krokach i kanałach)."""
    p = len(cechy)
    G = torch.stack([torch.stack([(cechy[a] * cechy[b]).sum((0, 2)) for b in range(p)], -1) for a in range(p)], -2)
    r = torch.stack([(c * y).sum((0, 2)) for c in cechy], -1)
    G = G + 1e-9 * torch.eye(p, device=y.device) * G.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
    return torch.linalg.solve(G, r)                                          # [n, p]


def mnk_wspolne(cechy, y):
    p = len(cechy)
    G = torch.stack([torch.stack([(cechy[a] * cechy[b]).sum() for b in range(p)]) for a in range(p)])
    r = torch.stack([(c * y).sum() for c in cechy])
    return torch.linalg.solve(G, r)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dane', required=True); ap.add_argument('--wyj', default=None)
    a = ap.parse_args()
    d = Path(a.dane); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta = json.loads((d / 'metadata.json').read_text(encoding='utf-8'))
    Z = np.load(d / 'Z.npy', mmap_mode='r'); S = {k: np.load(d / f'S_{k}.npy', mmap_mode='r') for k in ('nowa', 'ped', 'zan', 'buf')}
    T = S['nowa'].shape[0]; t_ucz = min(T_UCZ, int(T * .7)); od = min(OD, max(2, int(T * .3)))
    narz_all = np.array(meta['narzedzie'], bool)
    wynik = dict(kroki=T, t_ucz=t_ucz, od=od, klatki=meta['klatki'], klatki_wyniki={})
    for j in range(Z.shape[1]):
        narz = narz_all[j]
        acc = {}
        for t0 in range(0, 600, 150):
            sl = slice(t0, t0 + 150)
            z = torch.from_numpy(np.asarray(Z[:, j, sl], np.float32)).to(dev)              # [T+1, n, 768]
            dz = z[1:] - z[:-1]                                                             # [T, n, 768]
            s = {k: torch.from_numpy(np.asarray(v[:, j, sl], np.float32)).to(dev) for k, v in S.items()}
            c = torch.cumsum(torch.cat([torch.zeros_like(z[:1]), z]), 0)
            tt = torch.arange(len(z), device=dev); lo = (tt - W + 1).clamp_min(0)
            m = (c[tt + 1] - c[lo]) / (tt + 1 - lo).float()[:, None, None]
            ssum = s['nowa'] + s['ped'] + s['zan'] + s['buf']
            # 1. liniowość (wszystkie kroki)
            r2_lin = 1 - ((dz - ssum) ** 2).sum((0, 2)) / (dz ** 2).sum((0, 2)).clamp_min(1e-30)
            # 2. udziały
            nn = (dz ** 2).sum(-1).clamp_min(1e-30)
            udz = {k: ((v * dz).sum(-1) / nn).mean(0) for k, v in s.items()}
            # 3-4. modele na krokach t = od..t_ucz-1 (dz_(t−1) wymaga t >= 1)
            U = slice(od, t_ucz); Te = slice(t_ucz, T)
            prev = torch.cat([torch.zeros_like(dz[:1]), dz[:-1]])
            spr = -(z[:-1] - m[:-1])                                                        # −(z_t − m_t)
            modele = {'E1': [prev, s['nowa']], 'E2': [prev, s['nowa'], s['buf'], s['zan']],
                      'E3': [prev, spr, s['nowa'], s['buf'], s['zan']], 'P1': [prev, spr]}
            for nazwa, cech in modele.items():
                wsp = mnk([x[U] for x in cech], dz[U])                                       # [n, p]
                pred = sum(wsp[None, :, q, None] * cech[q][Te] for q in range(len(cech)))
                sk = 1 - ((dz[Te] - pred) ** 2).sum((0, 2)) / (dz[Te] ** 2).sum((0, 2)).clamp_min(1e-30)
                acc.setdefault(nazwa, []).append(sk); acc.setdefault(nazwa + '_wsp', []).append(wsp)
            acc.setdefault('r2_lin', []).append(r2_lin)
            for k, v in udz.items():
                acc.setdefault('udz_' + k, []).append(v)
            acc.setdefault('_z', []).append((z, dz, s, prev, spr))
            del c, m, ssum
        cat = {k: torch.cat(v).cpu().numpy() for k, v in acc.items() if k != '_z'}
        # W: wspólne parametry E3 dla całej klatki (dopasowane na wszystkich tokenach naraz)
        G = None; r = None
        for (z, dz, s, prev, spr) in acc['_z']:
            cech = [prev, spr, s['nowa'], s['buf'], s['zan']]
            Gi = torch.stack([torch.stack([(cech[p_][slice(od, t_ucz)] * cech[q][slice(od, t_ucz)]).sum() for q in range(5)]) for p_ in range(5)])
            ri = torch.stack([(cc[slice(od, t_ucz)] * dz[slice(od, t_ucz)]).sum() for cc in cech])
            G = Gi if G is None else G + Gi; r = ri if r is None else r + ri
        wsp_w = torch.linalg.solve(G, r)
        skw = []
        for (z, dz, s, prev, spr) in acc['_z']:
            cech = [prev, spr, s['nowa'], s['buf'], s['zan']]
            pred = sum(wsp_w[q] * cech[q][t_ucz:] for q in range(5))
            skw.append(1 - ((dz[t_ucz:] - pred) ** 2).sum((0, 2)) / (dz[t_ucz:] ** 2).sum((0, 2)).clamp_min(1e-30))
        cat['W'] = torch.cat(skw).cpu().numpy()
        del acc['_z']
        f = lambda v: dict(narz=round(float(np.median(v[narz])) * 100, 1), tlo=round(float(np.median(v[~narz])) * 100, 1))
        w = {k: f(cat[k]) for k in ('r2_lin', 'E1', 'E2', 'E3', 'W', 'P1')}
        w['udzialy'] = {k[4:]: f(cat[k]) for k in cat if k.startswith('udz_')}
        w['wspolczynniki_E3_mediana'] = dict(zip(['a_ped', 'k_sprezyna', 'b_paczka', 'c_batchnorm', 'd_zanik'],
                                                 np.median(cat['E3_wsp'], 0).round(4).tolist()))
        w['wspolczynniki_W'] = dict(zip(['a_ped', 'k_sprezyna', 'b_paczka', 'c_batchnorm', 'd_zanik'], wsp_w.cpu().numpy().round(4).tolist()))
        wynik['klatki_wyniki'][str(meta['klatki'][j])] = w
        np.savez(Path(a.wyj or d.parent / 'dopasowanie').with_suffix('') .as_posix() + f'_klatka{meta["klatki"][j]}.npz',
                 **{k: v for k, v in cat.items()})
        print(meta['klatki'][j], json.dumps(w, ensure_ascii=False), flush=True)
    Path(a.wyj or d.parent / 'DOPASOWANIE.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
