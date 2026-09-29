#!/usr/bin/env python3
"""Ile kierunków przestrzeni latentu koduje tożsamość narzędzia — w obrazie cząstkowym (pytanie użytkownika
2026-09-29): każdy token to cząstka o 768 współrzędnych; porównujemy cząstki narzędzia z cząstkami w TYM SAMYM
miejscu na zdjęciu bez narzędzia (`tla.py`: zamalowane oko i czarne tło).

Sieci: stan po rozgrzewce (krok 1000), ziarna 47–56. Cząstki: każdy token narzędzia sondy osobno (bez uśredniania).
Dwie miary liczby kierunków:
  energia — kierunki główne różnic „cząstka z narzędziem − ta sama cząstka bez narzędzia” (rozkład własny macierzy
            momentów różnic): ile kierunków niesie 50 % i 90 % całej różnicy;
  kolejne kierunki rozdzielające — znajdź kierunek najlepiej oddzielający cząstki z narzędziem od cząstek bez (LDA),
            usuń go z przestrzeni i szukaj następnego w tym, co zostało (iteracyjne rzutowanie na dopełnienie,
            Ravfogel i in. 2020, INLP); liczymy kierunki, póki kolejny sam rozdziela cząstki na wycinkach wyłączonych
            z uczenia z AUC ≥ 0,9 (i ≥ 0,75 dla porównania; 4 części po 3 wycinki). AUC 0,5 = rzut monetą, 1 = bez pomyłek.

    <venv>/python kierunki_tozsamosci.py  ->  wyniki/KIERUNKI_TOZSAMOSCI.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

import wektor_z_tlami as WZ

MAKS, FOLDY = 40, 4                     # 4 części po 3 wycinki: kierunek sprawdzany na wycinkach spoza uczenia


@torch.no_grad()
def czastki(net, obrazy, wagi):
    """Latent każdego zaznaczonego tokenu osobno: lista tablic (k_i, 768)."""
    net.eval(); net.zapisuj_latent(True)
    X = torch.from_numpy(np.stack(obrazy)).permute(0, 3, 1, 2).float().div(255)
    out = []
    for s in range(0, len(X), 96):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, lat = net(X[s:s + 96].cuda())
        lat = lat.float().cpu().numpy()
        for j in range(lat.shape[0]):
            out.append(lat[j][wagi[s + j]].astype(np.float64))
    net.zapisuj_latent(False)
    return out


def auc(a, b):
    a = np.sort(a); b = np.sort(b)
    return float(np.searchsorted(b, a, side='left').sum() + .5 * (np.searchsorted(b, a, side='right') - np.searchsorted(b, a, side='left')).sum()) / (len(a) * len(b))


def main() -> int:
    man = json.loads((WZ.PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    zrodlo = np.array([m['source_file'] for m in man])
    tn = [WZ.tokeny(WZ.PROBE / m['mask'])[0] for m in man]
    obr_s = [WZ.wczytaj(WZ.PROBE / m['image']) for m in man]
    tla = json.loads((WZ.HERE / 'tla' / 'manifest.json').read_text(encoding='utf-8'))
    obr_t = {(t['sonda'], t['typ']): WZ.wczytaj(WZ.HERE / 'tla' / t['image']) for t in tla}
    poz = np.where(obecn)[0]
    net = WZ.M.MalyViT().cuda()
    wyn = {}
    for d in sorted((WZ.V3 / 'wyniki_eden').glob('petla_s*')):
        s = d.name.split('_s')[1]
        net.load_state_dict(torch.load(d / 'wspolny' / 'k02.pt', map_location='cpu', weights_only=False)['net'])
        wag = [tn[i] for i in poz]
        P = czastki(net, [obr_s[i] for i in poz], wag)
        O = czastki(net, [obr_t[(i, 'oko')] for i in poz], wag)
        C = czastki(net, [obr_t[(i, 'czarne')] for i in poz], wag)
        print(s, 'cząstki policzone', flush=True)
        zr = np.concatenate([[zrodlo[i]] * len(p) for i, p in zip(poz, P)])
        P, O, C = np.concatenate(P), np.concatenate(O), np.concatenate(C)
        # energia różnic w parach (cząstka z narzędziem − ta sama bez narzędzia)
        D = np.concatenate([P - O, P - C])
        ev = np.linalg.eigvalsh(D.T @ D / len(D))[::-1]
        c = np.cumsum(ev) / ev.sum()
        energia = dict(n50=int(np.searchsorted(c, .5) + 1), n90=int(np.searchsorted(c, .9) + 1),
                       udzial_1=float(c[0]), udzial_5=float(c[4]))
        # kolejne kierunki rozdzielające (dopełnienie po każdym znalezionym kierunku)
        X = np.concatenate([P, O, C]); y = np.r_[np.ones(len(P)), np.zeros(len(O) + len(C))].astype(int)
        g = np.concatenate([zr, zr, zr])
        neg = X[y == 0]; mu, sd = neg.mean(0), neg.std(0) + 1e-6
        Z = (X - mu) / sd
        zrodla = np.unique(zr)
        Q = np.eye(Z.shape[1]); aucs = []; t0 = time.time()
        for k in range(MAKS):
            Zk = Z @ Q
            w = WZ.SI.lda(Zk, y); w /= np.linalg.norm(w) + 1e-12
            a_h = []
            for f in range(FOLDY):                                            # wycinki wyłączone z uczenia kierunku
                u = ~np.isin(g, zrodla[f::FOLDY])
                wz = WZ.SI.lda(Zk[u], y[u]); wz /= np.linalg.norm(wz) + 1e-12
                r = Zk[~u] @ wz
                a_h.append(auc(r[y[~u] == 1], r[y[~u] == 0]))
            aucs.append(float(np.mean(a_h)))
            v = Q @ w; v /= np.linalg.norm(v)                                  # kierunek w pełnej przestrzeni
            Q = Q - np.outer(v, v)                                             # usuń kierunek (v leży w obrazie Q)
            if k % 5 == 4:
                print(f'  {s} kierunek {k + 1}: AUC {aucs[-1]:.2f} ({time.time() - t0:.0f} s)', flush=True)
            if aucs[-1] < .6:
                break
        n90 = next((i for i, a in enumerate(aucs) if a < .9), len(aucs))
        n75 = next((i for i, a in enumerate(aucs) if a < .75), len(aucs))
        wyn[s] = dict(czastek_z_narzedziem=int(len(P)), energia=energia, auc_kolejnych=aucs,
                      kierunkow_auc_090=n90, kierunkow_auc_075=n75)
        print(s, f"cząstek {len(P)} | energia różnic: 50 % w {energia['n50']}, 90 % w {energia['n90']} kierunkach "
              f"(pierwszy {energia['udzial_1']:.0%}) | kolejne kierunki rozdzielające: AUC≥0,9 {n90}, ≥0,75 {n75} | "
              f"AUC pierwszych: {[round(a, 2) for a in aucs[:8]]}", flush=True)
    Zs = list(wyn.values())
    pod = dict(energia_n50=float(np.mean([z['energia']['n50'] for z in Zs])), energia_n90=float(np.mean([z['energia']['n90'] for z in Zs])),
               energia_udzial_pierwszego=float(np.mean([z['energia']['udzial_1'] for z in Zs])),
               kierunkow_auc_090=float(np.mean([z['kierunkow_auc_090'] for z in Zs])),
               kierunkow_auc_075=float(np.mean([z['kierunkow_auc_075'] for z in Zs])),
               kierunkow_auc_090_zakres=[int(min(z['kierunkow_auc_090'] for z in Zs)), int(max(z['kierunkow_auc_090'] for z in Zs))])
    (WZ.HERE / 'wyniki' / 'KIERUNKI_TOZSAMOSCI.json').write_text(json.dumps(dict(podsumowanie=pod, ziarna=wyn), indent=1,
                                                                        ensure_ascii=False), encoding='utf-8')
    print('srednio', pod)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
