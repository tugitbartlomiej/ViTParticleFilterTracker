"""Etap 6: korekta latentu tylko w kierunkach naturalnych. z' = z0 + (u · s) U^T dla każdego tokenu, U = PCA-r tokenów
prawdziwych obrazów (400 obrazów manifestu przy k1000), s = ich odchylenia; cel detekcja (jak etap 1) plus
λ · średnia u^2 (Mahalanobis) i kara za wyjście poza zakresy z checkpointu. Czy detekcję da się uzyskać latentem
bliskim rozkładowi prawdziwych tokenów? Miary naturalności jak w ETAP5_NORMALIZACJA (poza, maha).
Uruchomienie z SmallViTAblation: F:/Python/python.exe experiments/latent_cel_do_obrazow/kod/korekta_naturalna.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow'
os.environ['KOREKTA_WYJSCIE'] = str(OUT / 'korekty')
sys.path.insert(0, str(ROOT / 'app'))
import numpy as np
import torch

import korekta_latentu as KL

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


def main(n_cel=8, rr=(64, 256), lamy=(0.0, 0.01, 0.1), kroki=400):
    st = KL.Stanowisko(dict(KL.checkpointy())['gęsta historia k1000'])
    rows = KL.manifest(); rng = np.random.default_rng(0)
    T = []
    for i in range(len(rows)):
        z = st.przebieg(KL.obraz_manifestu(rows[i])[0])['z']
        T.append(z[torch.from_numpy(rng.choice(600, 100, replace=False)).to(st.dev)])
    T = torch.cat(T); mu = T.mean(0)
    _, S_, Vt = torch.linalg.svd(T - mu, full_matrices=False)
    cele = json.loads((OUT / 'ETAP1_CELE.json').read_text(encoding='utf-8'))['cele'][:n_cel]
    W2 = st.blok.mlp.fc2.weight.detach()
    wyn = []
    for r in rr:
        U = Vt[:r].T; s = S_[:r] / (len(T) - 1) ** .5; U64 = Vt[:64].T; s64 = S_[:64] / (len(T) - 1) ** .5

        def nienat(z):
            y = z - mu; p = y @ U64
            return float(((y - p @ U64.T) ** 2).sum(1).sqrt().mean()), float(((p / s64) ** 2).sum(1).sqrt().mean())
        for lam in lamy:
            for c in cele:
                rgb, ram = KL.obraz_manifestu(rows[c['i']])
                b = st.przebieg(rgb); cel_t = KL.cel_z_ramek(ram, st.dev)
                z0 = b['z']
                u = torch.zeros(600, r, device=st.dev, requires_grad=True)
                opt = torch.optim.Adam([u], lr=0.05)
                najl = None
                for k in range(kroki + 1):
                    z = z0 + (u * s) @ U.T
                    o = st.ogon(z[None], b['r'][None])
                    L = KL.strata_detekcji(o, ram)[0] + lam * (u ** 2).mean() \
                        + 10 * (((st.h_min - z).clamp(min=0) ** 2) + ((z - st.h_max).clamp(min=0) ** 2)).sum(1).mean()
                    if najl is None or float(L) < najl[0]:
                        najl = (float(L), u.detach().clone())
                    if k == kroki:
                        break
                    opt.zero_grad(); L.backward(); opt.step()
                with torch.no_grad():
                    z = torch.minimum(torch.maximum(z0 + (najl[1] * s) @ U.T, st.h_min), st.h_max)
                    m = st.ocena(st.ogon(z[None], b['r'][None]), ram, cel_t); m0 = st.ocena(b['o'][None], ram, cel_t)
                    po, ma = nienat(z); po0, ma0 = nienat(z0)
                    rek = dict(cel=c['i'], r=r, lam=lam, iou=(m0['iou_top'], m['iou_top']), pewnosc=(m0['pewnosc_top'], m['pewnosc_top']),
                               fp=m['fp'], poza=(po0, po), maha=(ma0, ma), norma_dz=float((z - z0).norm()), norma_z=float(z0.norm()),
                               W2_zmiana_wzgl=float(((z - z0) @ W2.T).norm() / (z0 @ W2.T).norm()),
                               sztuczny_iou=c['najlepszy']['iou_top'])
                    np.save(OUT / f'zn_m{c["i"]:04d}_r{r}_l{lam}.npy', z.cpu().numpy().astype(np.float32))
                wyn.append(rek)
                log(r, lam, c['i'], 'IoU', [round(x, 3) for x in rek['iou']], 'pewn', [round(x, 3) for x in rek['pewnosc']],
                    'poza', [round(x, 2) for x in rek['poza']], '||dz||', round(rek['norma_dz'], 1), 'W2', round(rek['W2_zmiana_wzgl'], 3))
            (OUT / 'ETAP6_KOREKTA_NATURALNA.json').write_text(json.dumps(dict(wyniki=wyn), indent=1, ensure_ascii=False), encoding='utf-8')
    pods = {}
    for r in rr:
        for lam in lamy:
            w = [x for x in wyn if x['r'] == r and x['lam'] == lam]
            pods[f'r{r}_lam{lam}'] = dict(iou=float(np.mean([x['iou'][1] for x in w])), pewnosc=float(np.mean([x['pewnosc'][1] for x in w])),
                                         poza=float(np.mean([x['poza'][1] for x in w])), maha=float(np.mean([x['maha'][1] for x in w])),
                                         norma_dz=float(np.mean([x['norma_dz'] for x in w])), W2=float(np.mean([x['W2_zmiana_wzgl'] for x in w])),
                                         iou_powyzej_05=int(sum(x['iou'][1] >= .5 for x in w)), n=len(w))
    out = dict(wyniki=wyn, podsumowanie=pods, opis=__doc__)
    (OUT / 'ETAP6_KOREKTA_NATURALNA.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    for k, v in pods.items():
        log(k, {a: round(b, 3) for a, b in v.items()})
    log('KONIEC')


if __name__ == '__main__':
    main()
