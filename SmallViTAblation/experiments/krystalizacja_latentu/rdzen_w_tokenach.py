#!/usr/bin/env python3
"""Czy w latencie sieci (tokeny backbone) jest kierunek, ktory zapala sie zawsze tam, gdzie jest narzedzie?

Pytanie autora (2026-09-23): nie wyjscie dekodera, tylko sam latent. Latent to tokeny ostatniej
warstwy backbone: 24 x 24 kawalki obrazu, kazdy opisany profilem zredukowanym do 32 liczb
(CZASTKI_pool_b0500, detektor uczony na 500 klatkach, 9683 klatki puli, bez etykiet przy zapisie).

Etykiety sluza tylko do wskazania, ktore kawalki leza NA narzedziu:
  narzedzie   kawalki, ktorych srodek lezy blisko odcinka od czubka (tip_point) do przeciwleglego
              rogu ramki, czyli wzdluz trzonka
  tlo         kawalki poza ramka powiekszona o polowe

Kierunek rdzenia: roznica srednich narzedzie - tlo, wybielona kowariancja wewnatrz klas (LDA),
liczony na polowie filmow, sprawdzany na drugiej. Pytania:
  1. czy jeden kierunek oddziela kawalki narzedzia od tla (AUC na obcych filmach)
  2. czy jest ZAWSZE aktywny: w ilu klatkach najsilniejszy kawalek narzedzia przewyzsza
     95. centyl tla tej samej klatki
  3. czy to jeden wektor: AUC drugiego kierunku po usunieciu pierwszego, i AUC pelnej regresji
     logistycznej na 32 liczbach
  4. czy aktywnosc zalezy od kata i wydluzenia narzedzia oraz od filmu

    py -3.11 experiments/krystalizacja_latentu/rdzen_w_tokenach.py \\
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \\
        --manifest feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl \\
        --zipy ../Dataset/Cataract-LMM/3_Object_Tracking/annotations \\
        --out experiments/krystalizacja_latentu/wyniki/dowody
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np


def auc(pos, neg):
    x = np.concatenate([pos, neg]); r = np.argsort(np.argsort(x)) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--zipy', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--klatek', type=int, default=4000)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    d = np.load(args.pula / 'czesc000.npz')
    fid = d['frame_ids']
    rng = np.random.default_rng(0)
    wyb = np.sort(rng.choice(len(fid), min(args.klatek, len(fid)), replace=False))
    P = d['profil_red'][wyb].astype(np.float64)                 # klatki x 576 x 32
    fid = [str(f) for f in fid[wyb]]
    man = {}
    for line in args.manifest.open(encoding='utf-8'):
        r = json.loads(line); man[r['frame_id']] = r
    H = W = 24
    ys, xs = (np.arange(H) + .5) / H * 480, (np.arange(W) + .5) / W * 720
    gy, gx = np.meshgrid(ys, xs, indexing='ij'); gy, gx = gy.ravel(), gx.ravel()

    narz, tlo, kat, wydl, film = [], [], [], [], []
    for i, f in enumerate(fid):
        r = man[f]; b = r['annotations'][0]['bbox']; tip = r.get('tip_point')
        x0, y0, bw, bh = b
        if tip is None or bw < 8 or bh < 8:
            narz.append(None); tlo.append(None); kat.append(np.nan); wydl.append(np.nan); film.append(f.split('/')[-2]); continue
        rogi = np.array([[x0, y0], [x0 + bw, y0], [x0, y0 + bh], [x0 + bw, y0 + bh]])
        koniec = rogi[np.argmax(((rogi - tip) ** 2).sum(1))]
        a_ = np.array(tip, float); v = koniec - a_; L2 = max((v ** 2).sum(), 1e-9)
        t = np.clip(((np.stack([gx, gy], 1) - a_) @ v) / L2, 0, 1)
        odl = np.linalg.norm(np.stack([gx, gy], 1) - (a_ + t[:, None] * v), axis=1)
        wew = (gx >= x0) & (gx <= x0 + bw) & (gy >= y0) & (gy <= y0 + bh)
        n_ = np.flatnonzero(wew & (odl < 12))                       # ok. pol kawalka od osi narzedzia
        m = .5
        poza = ~((gx >= x0 - m * bw) & (gx <= x0 + bw + m * bw) & (gy >= y0 - m * bh) & (gy <= y0 + bh + m * bh))
        narz.append(n_ if len(n_) else None); tlo.append(np.flatnonzero(poza))
        kat.append(np.degrees(np.arctan2(abs(v[1]), abs(v[0])))); wydl.append(np.log(max(bw, 1) / max(bh, 1)))
        film.append(f.split('/')[-2])
    ok = np.array([n is not None for n in narz])
    film = np.array(film); kat = np.array(kat); wydl = np.array(wydl)
    filmy = np.unique(film[ok]); rng.shuffle(filmy)
    uczace = np.isin(film, filmy[:len(filmy) // 2]) & ok
    test = np.isin(film, filmy[len(filmy) // 2:]) & ok

    def zbierz(maska):
        tp = np.concatenate([P[i, narz[i]] for i in np.flatnonzero(maska)])
        tn = np.concatenate([P[i, rng.choice(tlo[i], min(20, len(tlo[i])), replace=False)] for i in np.flatnonzero(maska)])
        return tp, tn
    tp_a, tn_a = zbierz(uczace); tp_b, tn_b = zbierz(test)
    Sw = np.cov(np.concatenate([tp_a - tp_a.mean(0), tn_a - tn_a.mean(0)]).T) + 1e-6 * np.eye(32)
    w1 = np.linalg.solve(Sw, tp_a.mean(0) - tn_a.mean(0)); w1 /= np.linalg.norm(w1)
    auc1 = auc(tp_b @ w1, tn_b @ w1)
    # drugi kierunek po usunieciu pierwszego
    Pr = np.eye(32) - np.outer(w1, w1)
    Sw2 = Pr @ Sw @ Pr + 1e-6 * np.eye(32)
    w2 = np.linalg.lstsq(Sw2, Pr @ (tp_a.mean(0) - tn_a.mean(0)), rcond=None)[0]; w2 = Pr @ w2; w2 /= np.linalg.norm(w2)
    auc2 = auc(tp_b @ w2, tn_b @ w2)
    # pelna regresja logistyczna (gradient prosty)
    X = np.concatenate([tp_a, tn_a]); y = np.r_[np.ones(len(tp_a)), np.zeros(len(tn_a))]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd; beta = np.zeros(33)
    Xa = np.c_[Xs, np.ones(len(Xs))]
    for _ in range(300):
        pz = 1 / (1 + np.exp(-Xa @ beta)); beta -= 0.5 * Xa.T @ (pz - y) / len(y)
    sc = lambda Z: np.c_[(Z - mu) / sd, np.ones(len(Z))] @ beta
    auc_pelne = auc(sc(tp_b), sc(tn_b))
    cos_log_lda = float(abs(np.dot(beta[:32] / sd, w1)) / np.linalg.norm(beta[:32] / sd))

    # "zawsze aktywny": w kazdej klatce testowej
    zapalony, kontrast, idx_t = [], [], np.flatnonzero(test)
    for i in idx_t:
        a_n = P[i, narz[i]] @ w1; a_t = P[i, tlo[i]] @ w1
        zapalony.append(a_n.max() > np.quantile(a_t, .95)); kontrast.append(np.median(a_n) - np.median(a_t))
    zapalony = np.array(zapalony); kontrast = np.array(kontrast)
    rk = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
    fm = film[idx_t]
    po_filmie = {f: float(zapalony[fm == f].mean()) for f in np.unique(fm)}
    raport = dict(klatek_uczacych=int(uczace.sum()), klatek_testowych=int(test.sum()),
                  filmow_testowych=int(len(np.unique(fm))),
                  auc_jeden_kierunek=auc1, auc_drugi_kierunek_po_usunieciu=auc2, auc_pelna_regresja_32=auc_pelne,
                  cos_regresja_vs_kierunek=cos_log_lda,
                  udzial_klatek_z_zapalonym_rdzeniem=float(zapalony.mean()),
                  najgorszy_film=[min(po_filmie, key=po_filmie.get), min(po_filmie.values())],
                  korelacja_kontrastu_z_katem=rk(kontrast, kat[idx_t]),
                  korelacja_kontrastu_z_wydluzeniem=rk(kontrast, wydl[idx_t]),
                  udzial_zapalonych_wg_kata={f'{a}-{a + 30}': float(zapalony[(kat[idx_t] >= a) & (kat[idx_t] < a + 30)].mean())
                                            for a in (0, 30, 60)})
    print(json.dumps(raport, indent=1, ensure_ascii=False))

    # rysunek: mapy aktywacji rdzenia na kilku klatkach testowych
    args.out.mkdir(parents=True, exist_ok=True)
    przyklady = list(idx_t[np.argsort(kontrast)[::-1][:3]]) + list(idx_t[np.argsort(kontrast)[:2]]) + \
                list(rng.choice(idx_t, 3, replace=False))
    fig, ax = plt.subplots(2, len(przyklady), figsize=(3.2 * len(przyklady), 5))
    for c, i in enumerate(przyklady):
        f = fid[i]; film_, nr = f.split('/')[-2], int(f.split('/')[-1])
        with zipfile.ZipFile(args.zipy / f'{film_}.zip') as z:
            im = Image.open(io.BytesIO(z.read(f'{film_}/{film_}_{nr:06d}.jpg'))).convert('RGB').resize((720, 480))
        mapa = (P[i] @ w1).reshape(H, W)
        ax[0, c].imshow(im); ax[0, c].axis('off')
        x0, y0, bw, bh = man[f]['annotations'][0]['bbox']
        ax[0, c].add_patch(plt.Rectangle((x0, y0), bw, bh, fill=False, ec='lime', lw=1))
        ax[0, c].set_title(('silny' if c < 3 else 'slaby' if c < 5 else 'losowy') + f'\n{film_}', fontsize=7)
        ax[1, c].imshow(im); ax[1, c].imshow(mapa, cmap='inferno', alpha=.6, extent=(0, 720, 480, 0)); ax[1, c].axis('off')
    fig.suptitle('Aktywacja jednego kierunku latentu (tokeny backbone) na klatkach z filmow, ktorych nie uzyto do jego wyznaczenia')
    fig.tight_layout(); fig.savefig(args.out / 'RDZEN_tokeny.png', dpi=130); plt.close(fig)
    np.save(args.out / 'RDZEN_kierunek_32.npy', w1)
    (args.out / 'REPORT_rdzen_tokeny.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
