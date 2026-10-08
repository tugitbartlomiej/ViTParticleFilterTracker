"""Etap 12 (zlecenie użytkownika): pchnięcie zdjęć znalezionych przez DINO dla obrazów SD (etap 11) z kontrolą losowych.

Pchnięcie jednego kroku AdamW (990 -> 991) obrazu dołożonego jako 33. do paczki kroku na latent obrazu celu (sonda,
pchniecie.py). Kandydaci na cel: najbliższe zdjęcie z puli dla każdego z 11 obrazów SD (bez powtórzeń), tyle samo
losowych zdjęć tej samej puli (kontrola; 3 losowania) i sam obraz celu z etykietą. Etykiety z manifestu puli (cystotom).
Miary: cos(S, δ) i cos(S, δ_vis) z przyrostem δ = z_sztuczny (etap 8, minmax) − z_oryginał i jego częścią
widoczną dla fc2, cos(W2 S, W2 δ), zmiana straty celu w pierwszym rzędzie <dL/dz, S> (ujemna = pomaga).
"""
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow'
sys.path.insert(0, str(ROOT / 'app'))
import cv2
import numpy as np
import torch
from PIL import Image
from scipy import stats

import korekta_latentu as KL
import galezie_uczenia as GU
from pchniecie import Pchniecie
from stan_uczenia import Odtworzenie

DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
CELE = [135, 102, 369, 11, 354, 224, 236, 270]


def main():
    man = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    sz = json.loads((OUT / 'ETAP11_SZUKANIE_DINO.json').read_text(encoding='utf-8'))['wyniki']
    rows = KL.manifest()

    def klatka(j):
        m = man[j]
        with zipfile.ZipFile(m['archive']) as zf:
            im = np.asarray(Image.open(io.BytesIO(zf.read(m['member']))).convert('RGB'))
        h, w = im.shape[:2]
        r = [[g['bbox_xywh'][0] * 480 / w, g['bbox_xywh'][1] * 320 / h, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * 480 / w,
              (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * 320 / h] for g in m.get('gt', [])]
        return cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA), np.asarray(r, np.float32).reshape(-1, 4)

    st = KL.Stanowisko(dict(KL.checkpointy())['gęsta historia k1000'])
    Vr = st.Vr; W2 = st.blok.mlp.fc2.weight.detach()
    b = GU.bieg_gestej_historii(); od = Odtworzenie(b); od.przygotuj()
    X, R, D = [], [], []
    for i in CELE:
        rgb, ram = KL.obraz_manifestu(rows[i]); X.append(rgb); R.append(ram)
        z0 = st.przebieg(rgb)['z'].cpu().numpy()
        D.append(np.load(OUT / f'zb_m{i:04d}_minmax.npy') - z0)
    P = Pchniecie(od, np.stack(X), np.stack([KL.maski_gt(r)['obiekt'] for r in R]), R)
    P.przygotuj(990, 0)
    z_gt = {i for i in range(len(man)) if man[i].get('gt')}
    dozw = np.array(sorted(z_gt))
    rng = np.random.default_rng(12)
    cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    wyn = []
    for t, i in enumerate(CELE):
        d = torch.from_numpy(D[t]).to(P.dev); dv = (d @ Vr) @ Vr.T
        g = P._grad_straty_sondy(t)
        zn = []
        for x in sz:
            if x['cel'] == i and x['k'] != 'zdjecie':
                j = x['top'][0]['j']
                if j not in [a for a, _ in zn] and man[j].get('gt'):
                    zn.append((j, x['k']))
        kand = [('cel (sam obraz z etykietą)', None, X[t], R[t])]
        for j, k in zn:
            im, ram = klatka(j); kand.append(('znalezione z SD', k, im, ram))
        for los in range(3):
            for j in rng.choice(dozw, len(zn), replace=False):
                im, ram = klatka(int(j)); kand.append((f'losowe (losowanie {los})', None, im, ram))
        for nazwa, k, im, ram in kand:
            S = P.pchniecie_kandydata(im, ram, t)
            wyn.append(dict(cel=i, zrodlo=nazwa.split(' (losowanie')[0], nazwa=nazwa, k=k, cos_przyrost=cos(S, d),
                            cos_widoczna=cos(S, dv), cos_wyjscie_fc2=cos(S @ W2.T, d @ W2.T),
                            zmiana_straty=float((S * g).sum()), dlugosc=float(S.norm())))
        sr = lambda zr, m: np.mean([r[m] for r in wyn if r['cel'] == i and r['zrodlo'] == zr])
        print(i, 'znalezionych', len(zn), {zr: (round(sr(zr, 'cos_przyrost'), 4), round(sr(zr, 'zmiana_straty'), 5))
                                           for zr in ('cel', 'znalezione z SD', 'losowe')}, flush=True)
    zr_l = ['cel', 'znalezione z SD', 'losowe']
    def zrodlo(r):
        return 'cel' if r['zrodlo'].startswith('cel') else r['zrodlo']
    pods = {z: {m: float(np.mean([r[m] for r in wyn if zrodlo(r) == z])) for m in
                ('cos_przyrost', 'cos_widoczna', 'cos_wyjscie_fc2', 'zmiana_straty', 'dlugosc')} for z in zr_l}
    para = {}
    for m in ('cos_przyrost', 'cos_widoczna', 'cos_wyjscie_fc2', 'zmiana_straty'):
        a = np.array([np.mean([r[m] for r in wyn if r['cel'] == i and zrodlo(r) == 'znalezione z SD']) for i in CELE])
        bb = np.array([np.mean([r[m] for r in wyn if r['cel'] == i and zrodlo(r) == 'losowe']) for i in CELE])
        lepsze = int((a < bb).sum()) if m == 'zmiana_straty' else int((a > bb).sum())
        para[m] = dict(znalezione=a.tolist(), losowe=bb.tolist(), lepsze_na_celach=lepsze, n=len(CELE),
                       p_wilcoxon=float(stats.wilcoxon(a, bb).pvalue))
    # zależność od kroku przyrostu (k = 0: 100 %, k = 10: oryginał)
    po_k = {}
    for k in range(11):
        v = [r for r in wyn if r['k'] == k and zrodlo(r) == 'znalezione z SD']
        if v:
            po_k[f'{100 - 10 * k}%'] = dict(n=len(v), cos_przyrost=float(np.mean([r['cos_przyrost'] for r in v])),
                                           zmiana_straty=float(np.mean([r['zmiana_straty'] for r in v])))
    out = dict(wyniki=wyn, podsumowanie=pods, pary=para, po_przyroscie=po_k, opis=__doc__)
    (OUT / 'ETAP12_PCHNIECIE_ZNALEZIONYCH.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    for z, v in pods.items():
        print(z, {a: round(b, 5) for a, b in v.items()})
    for m, v in para.items():
        print('pary', m, 'znalezione lepsze na', v['lepsze_na_celach'], '/', v['n'], 'p', round(v['p_wilcoxon'], 4))
    for k, v in po_k.items():
        print('przyrost', k, v)
    print('KONIEC')


if __name__ == '__main__':
    main()
