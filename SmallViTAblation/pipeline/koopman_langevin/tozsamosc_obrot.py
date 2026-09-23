#!/usr/bin/env python3
"""Test obrotu, wydluzenia i tla: gdzie detektor zapisuje tozsamosc narzedzia (2026-09-23).

Wedlug Lenca i Vedaldiego (IJCV 2019): reprezentacja phi jest rownowazna (equivariant) wzgledem
przeksztalcenia g, gdy phi(g x) ~ M_g phi(x). Kierunki, na ktorych M_g jest tozsamoscia, sa
NIEZMIENNICZE: tam siec zapisuje, CO to jest. Kierunki, ktore M_g obraca, zapisuja, JAK narzedzie lezy.

Dla kazdej klatki walidacyjnej (filmy, ktorych zaden detektor nie widzial) robimy wersje:
  orig                 bez zmian
  rot_+-15/30/45       obrot calego obrazu wokol srodka ramki narzedzia (zmienia kat narzedzia)
  dl_0.7, dl_1.4       sciskanie / rozciaganie wzdluz dluzszego boku ramki (skrocenie / wydluzenie)
  tlo_a, tlo_b         wycinek z narzedziem wklejony (z miekka krawedzia) na klatke innej operacji

i zapisujemy z kazdego detektora, bez etykiet w samym pomiarze:
  zapytanie            wyjscie dekodera najlepszego zapytania (wejscie glowicy klasy): tu sa
                       "to jest narzedzie" oraz ramka
  tokeny_ramki         srednia tokenow ostatniej warstwy backbone wewnatrz PRZEWIDZIANEJ ramki
  opis_koopmana        tylko dla detektora, z ktorego liczono Koopmana (ut_b0500): opis 3 x 3 x 32
                       dokladnie jak w CZASTKI_pool_b0500, zeby rzutowac na mody tozsamosci
  ramka, pewnosc       przewidziana ramka i pewnosc; ramka z etykiety po przeksztalceniu tylko do oceny

Uruchamiane na klastrze (wymaga rfdetr):

    python -P tozsamosc_obrot.py --data <katalog z images/ i annotations_val.json> \
        --detektor ut_b0500=<pth> --detektor B1m_250=<pth> ... --baza-koopmana <baza.npz> \
        --out <katalog>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

WARIANTY = ['orig', 'rot_-45', 'rot_-30', 'rot_-15', 'rot_+15', 'rot_+30', 'rot_+45',
            'dl_0.7', 'dl_1.4', 'tlo_a', 'tlo_b']


def crop_descriptor(prof, box, grid, cells=3):
    """Kopia z experiments/krystalizacja_latentu/pokrycie_a_bledy.py (obraz 720 x 480)."""
    h, w = grid
    ys = (np.arange(h) + .5) / h * 480
    xs = (np.arange(w) + .5) / w * 720
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    gy, gx = gy.ravel(), gx.ravel()
    k = prof.shape[-1]
    out = np.full((len(prof), cells * cells * k), np.nan, dtype=np.float32)
    for t in range(len(prof)):
        x0, y0, bw, bh = box[t]
        if not np.isfinite(box[t]).all() or bw < 8 or bh < 8:
            continue
        u, v = (gx - x0) / bw, (gy - y0) / bh
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        if inside.sum() < 4:
            continue
        cell = (np.clip((v[inside] * cells).astype(int), 0, cells - 1) * cells
                + np.clip((u[inside] * cells).astype(int), 0, cells - 1))
        acc = np.zeros((cells * cells, k))
        cnt = np.zeros(cells * cells)
        np.add.at(acc, cell, prof[t][inside])
        np.add.at(cnt, cell, 1)
        empty = cnt == 0
        acc[~empty] /= cnt[~empty, None]
        if empty.any():
            acc[empty] = acc[~empty].mean(0)
        out[t] = acc.ravel()
    return out


def przeksztalc(img, box, wariant, dawcy, rng):
    """Zwraca obraz po przeksztalceniu i ramke z etykiety po tym samym przeksztalceniu."""
    from PIL import Image, ImageFilter
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    rogi = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], float)
    if wariant == 'orig':
        return img, box
    if wariant.startswith('rot_'):
        kat = float(wariant[4:])
        out = img.rotate(kat, resample=Image.BILINEAR, center=(cx, cy), fillcolor=(0, 0, 0))
        r = np.deg2rad(kat)                                   # PIL obraca przeciwnie do wskazowek zegara
        c, s = np.cos(r), np.sin(r)
        d = rogi - [cx, cy]
        p = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1) + [cx, cy]
        return out, [p[:, 0].min(), p[:, 1].min(), np.ptp(p[:, 0]), np.ptp(p[:, 1])]
    if wariant.startswith('dl_'):
        sk = float(wariant[3:])
        if w >= h:                                            # wzdluz x: x_in = (x - cx)/sk + cx
            out = img.transform(img.size, Image.AFFINE, (1 / sk, 0, cx - cx / sk, 0, 1, 0), resample=Image.BILINEAR)
            return out, [cx - w * sk / 2, y, w * sk, h]
        out = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1 / sk, cy - cy / sk), resample=Image.BILINEAR)
        return out, [x, cy - h * sk / 2, w, h * sk]
    if wariant.startswith('tlo_'):
        dawca = dawcy[0 if wariant == 'tlo_a' else 1]
        m = 0.12
        x0, y0 = int(max(0, x - m * w)), int(max(0, y - m * h))
        x1, y1 = int(min(img.width, x + w + m * w)), int(min(img.height, y + h + m * h))
        wycinek = img.crop((x0, y0, x1, y1))
        maska = Image.new('L', wycinek.size, 0)
        brzeg = max(2, int(0.08 * min(wycinek.size)))
        maska.paste(255, (brzeg, brzeg, wycinek.width - brzeg, wycinek.height - brzeg))
        maska = maska.filter(ImageFilter.GaussianBlur(brzeg / 2))
        out = dawca.copy()
        out.paste(wycinek, (x0, y0), maska)
        return out, box
    raise ValueError(wariant)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--detektor', action='append', required=True, help='nazwa=sciezka.pth albo nazwa=coco')
    ap.add_argument('--baza-koopmana', type=Path, help='baza.npz z CZASTKI_pool_b0500 (srodek, kier)')
    ap.add_argument('--detektor-koopmana', default='ut_b0500')
    ap.add_argument('--na-film', type=int, default=12)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--resolution', type=int, default=384)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)

    import rfdetr
    import torch
    from PIL import Image, ImageFilter

    coco =json.loads((args.data / 'annotations_val.json').read_text(encoding='utf-8'))
    box_of = {}
    for a in coco['annotations']:
        box_of.setdefault(a['image_id'], a['bbox'])
    po_filmie: dict[str, list] = {}
    for im in coco['images']:
        if im['id'] in box_of and box_of[im['id']][2] >= 16 and box_of[im['id']][3] >= 8:
            po_filmie.setdefault(im['video_id'], []).append(im)
    rng = np.random.default_rng(args.seed)
    klatki = []
    for film in sorted(po_filmie):
        lista = sorted(po_filmie[film], key=lambda r: r['frame_id'])
        for i in np.linspace(0, len(lista) - 1, min(args.na_film, len(lista))).round().astype(int):
            klatki.append(lista[i])
    filmy = sorted(po_filmie)
    dawcy = []
    for im in klatki:                                        # dwie klatki z INNYCH filmow jako tlo
        inne = [f for f in filmy if f != im['video_id']]
        wyb = rng.choice(len(inne), 2, replace=False)
        dawcy.append([po_filmie[inne[j]][int(rng.integers(len(po_filmie[inne[j]])))] for j in wyb])
    print(f'[obrot] klatek {len(klatki)} z {len(filmy)} filmow, wariantow {len(WARIANTY)}', flush=True)

    obrazy, ramki_gt = [], []
    for im, dw in zip(klatki, dawcy):
        src = Image.open(args.data / im['file_name']).convert('RGB')
        dimg = []
        for r in dw:                                         # wlasne narzedzie dawcy zamazujemy, zeby nie bylo dwoch
            d_ = Image.open(args.data / r['file_name']).convert('RGB').resize(src.size)
            bx, by, bw, bh = box_of[r['id']]
            ob = (int(max(0, bx - .3 * bw)), int(max(0, by - .3 * bh)),
                  int(min(d_.width, bx + 1.3 * bw)), int(min(d_.height, by + 1.3 * bh)))
            d_.paste(d_.crop(ob).filter(ImageFilter.GaussianBlur(25)), ob[:2])
            dimg.append(d_)
        rz, rb = [], []
        for w in WARIANTY:
            o, b = przeksztalc(src, box_of[im['id']], w, dimg, rng)
            rz.append(o); rb.append(b)
        obrazy.append(rz); ramki_gt.append(rb)
    ramki_gt = np.array(ramki_gt, np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(0, min(4, len(obrazy))):                  # podglad przeksztalcen
        w_, h_ = 240, 160
        pod = Image.new('RGB', (w_ * len(WARIANTY), h_))
        for j, o in enumerate(obrazy[i]):
            pod.paste(o.resize((w_, h_)), (j * w_, 0))
        pod.save(args.out / f'PODGLAD_{i}.jpg', quality=85)

    baza = dict(np.load(args.baza_koopmana)) if args.baza_koopmana else None
    wynik_meta = dict(frame_ids=[im['frame_id'] for im in klatki], warianty=WARIANTY, detektory={})
    for spec in args.detektor:
        nazwa, sciezka = spec.split('=', 1)
        t0 = time.perf_counter()
        kw = dict(resolution=args.resolution)
        if sciezka != 'coco':
            kw['pretrain_weights'] = sciezka
        det = rfdetr.RFDETRNano(**kw)
        net = det.model.model.eval()
        backbone = next(m for _, m in net.named_modules() if type(m).__name__ == 'WindowedDinov2WithRegistersBackbone')
        gamma = backbone.layernorm.weight.detach().float().cpu().numpy()
        beta = backbone.layernorm.bias.detach().float().cpu().numpy()
        cap = {}
        def hook_fm(module, inputs, output):
            maps = output.feature_maps if hasattr(output, 'feature_maps') else output[0]
            cap['fm'] = maps[-1]
        hs_ = [backbone.register_forward_hook(hook_fm),
               net.register_forward_hook(lambda m, i, o: cap.__setitem__('out', o)),
               net.class_embed.register_forward_pre_hook(lambda m, i: cap.__setitem__('hs', i[0]))]
        plaskie = [o for rz in obrazy for o in rz]
        zap, tok, opis, rp, pw = [], [], [], [], []
        for s in range(0, len(plaskie), args.batch):
            part = plaskie[s:s + args.batch]
            sizes = np.array([[o.width, o.height] for o in part], np.float32)
            with torch.inference_mode():
                det.predict(part, threshold=0.0)
            out = cap['out']
            if isinstance(out, dict):
                logits, coords = out['pred_logits'], out['pred_boxes']
            else:
                coords, logits = (out[0], out[1]) if out[0].shape[-1] == 4 else (out[1], out[0])
            hs = cap['hs']
            if hs.dim() == 4:
                hs = hs[-1]
            b = logits.shape[0]
            nq = min(hs.shape[1], logits.shape[1])
            score, best = logits[:, :nq].float().sigmoid().max(-1).values.max(-1)
            ar = torch.arange(b, device=logits.device)
            zap.append(hs[ar, best].float().cpu().numpy())
            c = coords[ar, best].float().clamp(0, 1).cpu().numpy()
            box = np.stack([(c[:, 0] - c[:, 2] / 2) * sizes[:, 0], (c[:, 1] - c[:, 3] / 2) * sizes[:, 1],
                            c[:, 2] * sizes[:, 0], c[:, 3] * sizes[:, 1]], 1)
            rp.append(box); pw.append(score.cpu().numpy())
            fm = cap['fm'].float()
            bb, k, h, w = fm.shape
            z = fm.permute(0, 2, 3, 1).reshape(bb, h * w, k).cpu().numpy()
            ys, xs = (np.arange(h) + .5) / h, (np.arange(w) + .5) / w
            ins = ((xs[None, None, :] - c[:, 0, None, None]) ** 2 <= (c[:, 2, None, None] / 2) ** 2) & \
                  ((ys[None, :, None] - c[:, 1, None, None]) ** 2 <= (c[:, 3, None, None] / 2) ** 2)
            ins = ins.reshape(bb, h * w)
            tok.append(np.stack([z[j][ins[j]].mean(0) if ins[j].any() else z[j].mean(0) for j in range(bb)]))
            if baza is not None and nazwa == args.detektor_koopmana:
                prof = (z - beta) / np.abs(gamma)
                red = (prof - baza['srodek']) @ baza['kier']
                opis.append(crop_descriptor(red, box, (h, w)))
        for hh in hs_:
            hh.remove()
        n, v = len(obrazy), len(WARIANTY)
        zapis = dict(zapytanie=np.concatenate(zap).reshape(n, v, -1).astype(np.float32),
                     tokeny_ramki=np.concatenate(tok).reshape(n, v, -1).astype(np.float32),
                     ramka=np.concatenate(rp).reshape(n, v, 4).astype(np.float32),
                     pewnosc=np.concatenate(pw).reshape(n, v).astype(np.float32),
                     ramka_gt=ramki_gt)
        if opis:
            zapis['opis_koopmana'] = np.concatenate(opis).reshape(n, v, -1).astype(np.float32)
        np.savez_compressed(args.out / f'OBROT_{nazwa}.npz', **zapis)
        wynik_meta['detektory'][nazwa] = dict(sciezka=sciezka, sekundy=round(time.perf_counter() - t0, 1),
                                              wymiary={k_: list(v_.shape) for k_, v_ in zapis.items()})
        print(f'[obrot] {nazwa}: {time.perf_counter() - t0:.0f}s, pewnosc orig {np.median(zapis["pewnosc"][:, 0]):.3f}', flush=True)
        del det, net
        torch.cuda.empty_cache()
    (args.out / 'META.json').write_text(json.dumps(wynik_meta, indent=1, ensure_ascii=False), encoding='utf-8')
    print('[obrot] gotowe', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
