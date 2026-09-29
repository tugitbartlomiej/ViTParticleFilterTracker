#!/usr/bin/env python3
"""Zapis kierunków tożsamości z obrazu cząstkowego (`kierunki_tozsamosci.py` tylko je liczył, nie zapisywał).

Dla każdej sieci (krok 1000, ziarna 47–56): 6 kolejnych kierunków rozdzielających cząstki narzędzia od cząstek
na tych samych tokenach bez narzędzia (LDA, usunięcie kierunku, następny w reszcie przestrzeni), wyznaczonych na
wszystkich wycinkach. Porównanie pierwszego kierunku z wektorem `nowy` z `WEKTORY.npz` (ten liczony na średniej
z tokenów sondy): cosinus w surowym latencie (kierunek w/sd) i AUC obu na cząstkach.

    <venv>/python zapisz_kierunki_czastek.py  ->  wyniki/KIERUNKI_CZASTEK.npz, wyniki/KIERUNKI_CZASTEK.json
"""
from __future__ import annotations

import json

import numpy as np
import torch

import kierunki_tozsamosci as KT
import wektor_z_tlami as WZ

ILE = 6


def surowy(w, sd):
    """Kierunek w surowym latencie: rzut ((x − mu)/sd)·w = x·(w/sd) + stała."""
    v = w / sd
    return v / np.linalg.norm(v)


def main() -> int:
    man = json.loads((WZ.PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    tn = [WZ.tokeny(WZ.PROBE / m['mask'])[0] for m in man]
    obr_s = [WZ.wczytaj(WZ.PROBE / m['image']) for m in man]
    tla = json.loads((WZ.HERE / 'tla' / 'manifest.json').read_text(encoding='utf-8'))
    obr_t = {(t['sonda'], t['typ']): WZ.wczytaj(WZ.HERE / 'tla' / t['image']) for t in tla}
    poz = np.where(obecn)[0]
    stare = np.load(WZ.HERE / 'wyniki' / 'WEKTORY.npz')
    net = WZ.M.MalyViT().cuda()
    zapis, wyn = {}, {}
    for d in sorted((WZ.V3 / 'wyniki_eden').glob('petla_s*')):
        s = d.name.split('_s')[1]
        net.load_state_dict(torch.load(d / 'wspolny' / 'k02.pt', map_location='cpu', weights_only=False)['net'])
        wag = [tn[i] for i in poz]
        P = np.concatenate(KT.czastki(net, [obr_s[i] for i in poz], wag))
        O = np.concatenate(KT.czastki(net, [obr_t[(i, 'oko')] for i in poz], wag))
        C = np.concatenate(KT.czastki(net, [obr_t[(i, 'czarne')] for i in poz], wag))
        X = np.concatenate([P, O, C]); y = np.r_[np.ones(len(P)), np.zeros(len(O) + len(C))].astype(int)
        neg = X[y == 0]; mu, sd = neg.mean(0), neg.std(0) + 1e-6
        Z = (X - mu) / sd
        Q = np.eye(Z.shape[1]); W = []
        for _ in range(ILE):
            w = WZ.SI.lda(Z @ Q, y)
            v = Q @ w; v /= np.linalg.norm(v)
            W.append(v); Q = Q - np.outer(v, v)
        W = np.array(W)                                                       # (6, 768), w przestrzeni (x − mu)/sd
        zapis[f's{s}_w'] = W; zapis[f's{s}_mu'] = mu; zapis[f's{s}_sd'] = sd
        r1 = Z @ W[0]
        nw, nmu, nsd = stare[f's{s}_nowy_w'], stare[f's{s}_nowy_mu'], stare[f's{s}_nowy_sd']
        rn = ((X - nmu) / nsd) @ nw
        wyn[s] = dict(cos_pierwszy_z_nowym=float(abs(surowy(W[0], sd) @ surowy(nw, nsd))),
                      auc_pierwszy_na_czastkach=KT.auc(r1[y == 1], r1[y == 0]),
                      auc_nowy_na_czastkach=KT.auc(rn[y == 1], rn[y == 0]))
        print(s, {k: round(v, 3) for k, v in wyn[s].items()}, flush=True)
    np.savez(WZ.HERE / 'wyniki' / 'KIERUNKI_CZASTEK.npz', **zapis)
    sr = {k: float(np.mean([w[k] for w in wyn.values()])) for k in next(iter(wyn.values()))}
    (WZ.HERE / 'wyniki' / 'KIERUNKI_CZASTEK.json').write_text(json.dumps(dict(srednio=sr, ziarna=wyn), indent=1,
                                                                          ensure_ascii=False), encoding='utf-8')
    print('srednio', {k: round(v, 3) for k, v in sr.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
