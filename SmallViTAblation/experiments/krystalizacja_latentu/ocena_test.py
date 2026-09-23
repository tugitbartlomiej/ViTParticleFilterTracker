#!/usr/bin/env python3
"""Ocena detektorow na zbiorze TESTOWYM, ktorego zaden bieg nie widzial (lokalnie, RTX, 2026-09-23).

Zbior testowy: 25 operacji z split_manifest.json (split = test). W przygotowaniu danych na Edenie
test nie byl eksportowany (test_images_exported = 0), wiec zaden detektor ani wybor wag go nie dotknal.
Bierzemy co --krok klatke kazdego filmu, RAZEM z klatkami bez narzedzia (fałszywe alarmy).

Mierzymy dla kazdego detektora:
  mAP 0,5:0,95 i mAP 0,5 (pycocotools) na klatkach z narzedziem, jak w walidacji na Edenie
  trafienie klatki: najlepsza ramka z pewnoscia >= prog i IoU >= 0,5
  falszywy alarm: pewnosc >= prog na klatce, w ktorej adnotacja nie ma cystotomu
  wynik dla kazdego filmu i serie kolejnych chybien (jak dlugo detektor gubi narzedzie)
Dla detektora --detektor-latentu dodatkowo wektor narzedzia w tokenach backbone (bez etykiet):
  kontrast pod obrysem narzedzia i "obecnosc" z samej mapy (najsilniejszy kawalek wobec reszty klatki),
  zeby sprawdzic, czy slaby wektor przewiduje klatki, na ktorych detektor sie myli.

    <venv rfdetr>/python experiments/krystalizacja_latentu/ocena_test.py \\
        --zipy ../Dataset/Cataract-LMM/3_Object_Tracking/annotations \\
        --split feature_cache/eden_static36/data/tracking_static36/split_manifest.json \\
        --detektor b1500_best=results/eden_static36_2026-09-15/tracking_random_b1500_s42/best.pth ... \\
        --baza results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500/baza.npz \\
        --wektor experiments/krystalizacja_latentu/wyniki/dowody/RDZEN_bez_etykiet_32.npy \\
        --out experiments/krystalizacja_latentu/wyniki/test
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np

from test_ksztaltu import maska_z_seg, wczytaj_film


def iou_xywh(a, b):
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2]); y2 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum(a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter, 1e-9)


def serie(maska):
    """Dlugosci kolejnych ciagow True."""
    d, n = [], 0
    for v in maska:
        if v:
            n += 1
        elif n:
            d.append(n); n = 0
    if n:
        d.append(n)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zipy', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--detektor', action='append', required=True)
    ap.add_argument('--baza', type=Path, required=True)
    ap.add_argument('--wektor', type=Path, required=True)
    ap.add_argument('--detektor-latentu', default='ut_b0500')
    ap.add_argument('--krok', type=int, default=10)
    ap.add_argument('--prog', type=float, default=0.5)
    ap.add_argument('--resolution', type=int, default=384)
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    import rfdetr
    import torch
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    split = json.loads(args.split.read_text(encoding='utf-8'))['groups']
    filmy = sorted(f for f, s in split.items() if s == 'test')
    assert all((args.zipy / f'{f}.zip').exists() for f in filmy), 'brak archiwum filmu testowego'
    klatki, zipy = [], {}
    for f in filmy:
        z, film, wiersze = wczytaj_film(args.zipy / f'{f}.zip')
        jest = set(z.namelist()); zipy[f] = z
        wiersze = [r for r in wiersze if f'{f}/{r["filename"]}' in jest]
        wiersze.sort(key=lambda r: int(r['frame_number']))
        for r in wiersze[::args.krok]:
            ma = r['cystotome_bbox'] not in ('', '[]')
            klatki.append(dict(film=f, nr=int(r['frame_number']), plik=r['filename'], narzedzie=ma,
                               bbox=json.loads(r['cystotome_bbox']) if ma else None,
                               seg=r['cystotome_seg'] if r['cystotome_seg'] not in ('', '[]') else None))
    n = len(klatki)
    ma = np.array([k['narzedzie'] for k in klatki])
    gt = np.array([k['bbox'] if k['narzedzie'] else [np.nan] * 4 for k in klatki], np.float32)
    print(f'[test] {n} klatek z {len(filmy)} filmow, z narzedziem {ma.sum()}, bez {(~ma).sum()}', flush=True)

    def obraz(k):
        return Image.open(io.BytesIO(zipy[k['film']].read(f'{k["film"]}/{k["plik"]}'))).convert('RGB')

    baza = dict(np.load(args.baza)); wek = np.load(args.wektor)
    args.out.mkdir(parents=True, exist_ok=True)
    # COCO: tylko klatki z narzedziem, jak walidacja na Edenie (n_images = n_annotations)
    idx_ma = np.flatnonzero(ma)
    coco_gt = dict(images=[dict(id=int(i) + 1, width=720, height=480) for i in idx_ma],
                   annotations=[dict(id=j + 1, image_id=int(i) + 1, category_id=1, bbox=[float(x) for x in gt[i]],
                                     area=float(gt[i, 2] * gt[i, 3]), iscrowd=0) for j, i in enumerate(idx_ma)],
                   categories=[dict(id=1, name='instrument')])
    G = COCO(); G.dataset = coco_gt; G.createIndex()

    raport = dict(filmy=filmy, krok=args.krok, prog=args.prog, klatek=n, z_narzedziem=int(ma.sum()), wyniki={})
    for spec in args.detektor:
        nazwa, sciezka = spec.split('=', 1)
        t0 = time.time()
        det = rfdetr.RFDETRNano(resolution=args.resolution, pretrain_weights=sciezka)
        net = det.model.model.eval()
        cap = {}
        uchwyty = []
        latent = nazwa == args.detektor_latentu
        if latent:
            backbone = next(m for _, m in net.named_modules() if type(m).__name__ == 'WindowedDinov2WithRegistersBackbone')
            gamma = backbone.layernorm.weight.detach().float().cpu().numpy(); beta = backbone.layernorm.bias.detach().float().cpu().numpy()
            uchwyty.append(backbone.register_forward_hook(
                lambda mod, i, o: cap.__setitem__('fm', (o.feature_maps if hasattr(o, 'feature_maps') else o[0])[-1])))
        ramki = np.full((n, args.top, 4), np.nan, np.float32); pew = np.zeros((n, args.top), np.float32)
        wektor = np.full((n, 4), np.nan, np.float32)        # kontrast pod obrysem, zapalony, obecnosc z mapy, maks z mapy
        for s in range(0, n, 32):
            part = [obraz(k) for k in klatki[s:s + 32]]
            with torch.inference_mode():
                dets = det.predict(part, threshold=0.0)
            if not isinstance(dets, list):
                dets = [dets]
            for j, d in enumerate(dets):
                o = np.argsort(-d.confidence)[:args.top]
                xy = d.xyxy[o]
                ramki[s + j, :len(o)] = np.c_[xy[:, :2], xy[:, 2:] - xy[:, :2]]; pew[s + j, :len(o)] = d.confidence[o]
            if latent:
                fm = cap['fm'].float(); b, kk, h, w = fm.shape
                z = fm.permute(0, 2, 3, 1).reshape(b, h * w, kk).cpu().numpy()
                a_ = (((z - beta) / np.abs(gamma) - baza['srodek']) @ baza['kier']) @ wek
                for j in range(b):
                    k = klatki[s + j]; a = a_[j]
                    med, sd = np.median(a), a.std() + 1e-9
                    wektor[s + j, 2] = (a.max() - med) / sd; wektor[s + j, 3] = a.max()
                    if k['seg']:
                        m = maska_z_seg(k['seg'], (part[j].height, part[j].width))
                        ul = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).ravel()
                        dal = cv2.resize(cv2.dilate(m, np.ones((61, 61), np.uint8)).astype(np.float32), (w, h),
                                         interpolation=cv2.INTER_AREA).ravel() == 0
                        if ul.max() > .15 and dal.sum() > 20:
                            tl = a[dal]; nn = a[ul > .15]
                            wektor[s + j, 0] = (nn.mean() - np.median(tl)) / (tl.std() + 1e-9)
                            wektor[s + j, 1] = float(nn.max() > np.quantile(tl, .95))
            if s % 3200 == 0:
                print(f'[{nazwa}] {s}/{n}  {time.time() - t0:.0f}s', flush=True)
        for u in uchwyty:
            u.remove()

        # mAP (pycocotools) na klatkach z narzedziem
        dt = [dict(image_id=int(i) + 1, category_id=1, bbox=[float(x) for x in ramki[i, t]], score=float(pew[i, t]))
              for i in idx_ma for t in range(args.top) if pew[i, t] > 0.001 and np.isfinite(ramki[i, t, 0])]
        D = G.loadRes(dt); E = COCOeval(G, D, 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
        p1 = pew[:, 0]; iou1 = iou_xywh(ramki[:, 0], gt)
        traf = ma & (p1 >= args.prog) & (iou1 >= .5)
        chyb = ma & ~traf
        fa = ~ma & (p1 >= args.prog)
        r = dict(mAP_50_95=float(E.stats[0]), mAP_50=float(E.stats[1]), mAP_75=float(E.stats[2]),
                 trafione_klatki=float(traf[ma].mean()), brak_detekcji=float((ma & (p1 < args.prog))[ma].mean()),
                 zla_ramka=float((ma & (p1 >= args.prog) & (iou1 < .5))[ma].mean()),
                 falszywe_alarmy_bez_narzedzia=float(fa[~ma].mean()) if (~ma).any() else None,
                 iou_mediana_trafionych=float(np.median(iou1[traf])), pewnosc_mediana_z=float(np.median(p1[ma])),
                 pewnosc_mediana_bez=float(np.median(p1[~ma])) if (~ma).any() else None, sekundy=time.time() - t0)
        po_filmie = {}
        for f in filmy:
            mf = np.array([k['film'] == f for k in klatki])
            dl = serie(chyb[mf & ma])
            po_filmie[f] = dict(klatek_z=int((mf & ma).sum()), trafione=float(traf[mf & ma].mean()) if (mf & ma).any() else None,
                                fa=float(fa[mf & ~ma].mean()) if (mf & ~ma).any() else None,
                                najdluzsza_seria_chybien=max(dl) if dl else 0)
        r['po_filmie'] = po_filmie
        if latent:
            wz = ma & np.isfinite(wektor[:, 0])
            r['wektor_zapalony_udzial'] = float(wektor[wz, 1].mean())
            r['wektor_kontrast_mediana'] = float(np.median(wektor[wz, 0]))
            def auc(p, q):
                x = np.r_[p, q]; rk = np.argsort(np.argsort(x)) + 1
                return float((rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))
            # czy slaby wektor przewiduje chybienie detektora (na klatkach z narzedziem)
            r['auc_slaby_kontrast_przewiduje_chybienie'] = auc(-wektor[wz & chyb, 0], -wektor[wz & traf, 0])
            r['auc_obecnosc_z_mapy_narzedzie_vs_brak'] = auc(wektor[ma, 2], wektor[~ma, 2])
            r['auc_pewnosc_detektora_narzedzie_vs_brak'] = auc(p1[ma], p1[~ma])
        raport['wyniki'][nazwa] = r
        np.savez_compressed(args.out / f'TEST_{nazwa}.npz', ramki=ramki, pewnosc=pew, ramka_gt=gt, narzedzie=ma,
                            wektor=wektor, film=np.array([k['film'] for k in klatki]), nr=np.array([k['nr'] for k in klatki]))
        print(f'\n=== {nazwa}', json.dumps({k: v for k, v in r.items() if k != 'po_filmie'}, indent=1), flush=True)
        (args.out / 'REPORT_test.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
        del det, net; torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
