#!/usr/bin/env python3
"""Zapis wektorów tożsamości (stary, nowy, razem) każdej sieci po rozgrzewce i ich skupienie na neuronach:
ile z 768 liczb latentu niesie 50 % i 90 % wagi wektora (suma kwadratów współrzędnych).

    <venv>/python zapisz_wektory.py  ->  wyniki/WEKTORY.npz, wyniki/WEKTORY_SKUPIENIE.json
"""
import json
import numpy as np
import torch
import wektor_z_tlami as WZ

man = json.loads((WZ.PROBE / 'manifest.json').read_text(encoding='utf-8'))
obecn = np.array([bool(m['semantic_presence']) for m in man])
tok = [WZ.tokeny(WZ.PROBE / m['mask']) for m in man]; tn = [t[0] for t in tok]; tt = [t[1] for t in tok]
obr_s = [WZ.wczytaj(WZ.PROBE / m['image']) for m in man]
tla = json.loads((WZ.HERE / 'tla' / 'manifest.json').read_text(encoding='utf-8'))
obr_t = {(t['sonda'], t['typ']): WZ.wczytaj(WZ.HERE / 'tla' / t['image']) for t in tla}
poz = np.where(obecn)[0]; pust = np.where(~obecn)[0]
net = WZ.M.MalyViT().cuda()
zapis, sk = {}, {}


def skupienie(w):
    q = np.sort(w ** 2)[::-1]; c = np.cumsum(q) / q.sum()
    return dict(n50=int(np.searchsorted(c, .5) + 1), n90=int(np.searchsorted(c, .9) + 1))


for d in sorted((WZ.V3 / 'wyniki_eden').glob('petla_s*')):
    s = d.name.split('_s')[1]
    net.load_state_dict(torch.load(d / 'wspolny' / 'k02.pt', map_location='cpu', weights_only=False)['net'])
    nar = WZ.latent(net, [obr_s[i] for i in poz], [tn[i] for i in poz])
    tlo = WZ.latent(net, obr_s, tt); pusta = WZ.latent(net, [obr_s[i] for i in pust], [tn[i] for i in pust])
    oko = WZ.latent(net, [obr_t[(i, 'oko')] for i in poz], [tn[i] for i in poz])
    cz = WZ.latent(net, [obr_t[(i, 'czarne')] for i in poz], [tn[i] for i in poz])
    Lk = np.load(WZ.V2 / 'wyniki_eden' / d.name / 'wspolny' / 'k02' / 'LATENT.npz')['latent'][-5:].astype(np.float32).mean(0)
    wek = dict(stary=WZ.SI.wektor(Lk, obecn), nowy=WZ.lda_wektor(nar, np.concatenate([oko, cz])),
               razem=WZ.lda_wektor(nar, np.concatenate([oko, cz, tlo, pusta])))
    sk[s] = {}
    for n, m in wek.items():
        for k in ('mu', 'sd', 'w'):
            zapis[f's{s}_{n}_{k}'] = m[k]
        sk[s][n] = skupienie(m['w'])
    print(s, sk[s], flush=True)
np.savez(WZ.HERE / 'wyniki' / 'WEKTORY.npz', **zapis)
sr = {n: {k: float(np.mean([sk[s][n][k] for s in sk])) for k in ('n50', 'n90')} for n in ('stary', 'nowy', 'razem')}
(WZ.HERE / 'wyniki' / 'WEKTORY_SKUPIENIE.json').write_text(json.dumps(dict(srednio=sr, ziarna=sk), indent=1), encoding='utf-8')
print('srednio', sr)
