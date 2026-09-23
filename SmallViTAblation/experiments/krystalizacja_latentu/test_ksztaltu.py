#!/usr/bin/env python3
"""Test domkniecia tozsamosci narzedzia z narzedziem wycietym PO KSZTALCIE (lokalnie, RTX, 2026-09-23).

Poprzedni test wklejal prostokatny wycinek (narzedzie razem z kawalkiem wlasnego tla) i nie dalo sie
oddzielic wplywu otoczenia od wplywu sztucznej naklejki. Tu narzedzie wycinamy dokladnie po obrysie
z adnotacji Cataract-LMM (cystotome_seg), a wlasne narzedzie klatki usuwamy przez wypelnienie
otoczeniem (cv2.inpaint). Wersje kazdej klatki walidacyjnej (filmy, ktorych detektor nie widzial):

  orig        klatka bez zmian
  usuniete    narzedzie usuniete, nic nie wklejone (czy detektor widzi narzedzie tam, gdzie go nie ma)
  swoje       narzedzie usuniete i wklejone z powrotem po obrysie (kontrola sklejania: to samo otoczenie)
  obce        narzedzie wklejone po obrysie na klatke INNEJ operacji z usunietym jej narzedziem
  rot_-30/+30 samo narzedzie obrocone wokol czubka, wklejone na wlasne tlo (tlo sie nie rusza)
  dl_0.7/1.3  samo narzedzie skrocone / wydluzone wzdluz osi od czubka, wklejone na wlasne tlo

Mierzymy dla detektora ut_b0500 (w jego latencie wyznaczono wektor narzedzia):
  pewnosc detekcji i IoU z ramka narzedzia po przeksztalceniu
  aktywacje wektora narzedzia (bez etykiet, RDZEN_bez_etykiet_32.npy) na kawalkach pod narzedziem
  wobec tla tej samej klatki: kontrast (w odchyleniach tla) i czy "zapalony" (maks. > 95. centyl tla)
oraz pewnosc drugiego detektora (1500 klatek) jako porownanie.

    <venv rfdetr>/python experiments/krystalizacja_latentu/test_ksztaltu.py \\
        --zipy ../Dataset/Cataract-LMM/3_Object_Tracking/annotations \\
        --split feature_cache/eden_static36/data/tracking_static36/split_manifest.json \\
        --detektor ut_b0500=results/koopman_langevin_epochs/checkpoints/ut_b0500_best.pth \\
        --detektor b1500=results/eden_static36_2026-09-15/tracking_dino_coverage_b1500_s42/best.pth \\
        --baza results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500/baza.npz \\
        --wektor experiments/krystalizacja_latentu/wyniki/dowody/RDZEN_bez_etykiet_32.npy \\
        --out experiments/krystalizacja_latentu/wyniki/dowody/ksztalt
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

WARIANTY = ['orig', 'usuniete', 'swoje', 'obce', 'rot_-30', 'rot_+30', 'dl_0.7', 'dl_1.3']


def wczytaj_film(zip_path: Path):
    z = zipfile.ZipFile(zip_path)
    film = zip_path.stem
    wiersze = list(csv.DictReader(io.StringIO(z.read(f'{film}/annotation.csv').decode('utf-8'))))
    return z, film, wiersze


def maska_z_seg(seg_txt: str, shape):
    seg = json.loads(seg_txt)
    m = np.zeros(shape[:2], np.uint8)
    for poly in seg:
        pts = np.array(poly, np.float32).reshape(-1, 2).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m


def usun(img, m):
    return cv2.inpaint(img, cv2.dilate(m, np.ones((9, 9), np.uint8)) * 255, 7, cv2.INPAINT_TELEA)


def wklej(tlo, img, m):
    a = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8)).astype(np.float32), (5, 5), 0)[..., None]
    return (tlo * (1 - a) + img * a).astype(np.uint8)


def przeksztalc_narzedzie(img, m, tip, wariant):
    """Obraca albo skaluje wzdluz osi samo narzedzie (piksele i maske) wokol czubka."""
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if wariant.startswith('rot_'):
        A = cv2.getRotationMatrix2D((float(tip[0]), float(tip[1])), float(wariant[4:]), 1.0)
    else:
        s = float(wariant[3:])
        pts = np.c_[xs, ys].astype(float)
        far = pts[np.argmax(((pts - tip) ** 2).sum(1))]
        u = (far - tip) / max(np.linalg.norm(far - tip), 1e-6)
        R = np.eye(2) + (s - 1) * np.outer(u, u)                      # rozciaganie wzdluz osi u
        A = np.c_[R, np.asarray(tip) - R @ np.asarray(tip)]
    img2 = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m2 = cv2.warpAffine(m, A, (w, h), flags=cv2.INTER_NEAREST)
    return img2, m2


def ramka(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return [np.nan] * 4
    return [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zipy', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--detektor', action='append', required=True)
    ap.add_argument('--baza', type=Path, required=True)
    ap.add_argument('--wektor', type=Path, required=True)
    ap.add_argument('--detektor-latentu', default='ut_b0500')
    ap.add_argument('--na-film', type=int, default=12)
    ap.add_argument('--resolution', type=int, default=384)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    import rfdetr
    import torch
    from PIL import Image

    split = json.loads(args.split.read_text(encoding='utf-8'))['groups']
    filmy = sorted(f for f, s in split.items() if s == 'val' and (args.zipy / f'{f}.zip').exists())
    rng = np.random.default_rng(0)
    klatki = []                                                          # (film, wiersz)
    dane = {}
    for f in filmy:
        z, film, wiersze = wczytaj_film(args.zipy / f'{f}.zip')
        jest = set(z.namelist())                                         # adnotacje bywaja dla klatek spoza archiwum
        dobre = [r for r in wiersze if r['cystotome_seg'] not in ('', '[]') and r['cystotome_tip'] not in ('', '[]')
                 and f'{f}/{r["filename"]}' in jest]
        dane[f] = (z, dobre)
        for i in np.linspace(0, len(dobre) - 1, min(args.na_film, len(dobre))).round().astype(int):
            klatki.append((f, dobre[i]))
    print(f'[ksztalt] klatek {len(klatki)} z {len(filmy)} filmow walidacyjnych', flush=True)

    def obraz(f, r):
        z = dane[f][0]
        im = np.array(Image.open(io.BytesIO(z.read(f'{f}/{r["filename"]}'))).convert('RGB'))
        m = maska_z_seg(r['cystotome_seg'], im.shape)
        tip = np.array(json.loads(r['cystotome_tip']), float)
        return im, m, tip

    obrazy, ramki_gt, maski = [], [], []
    for f, r in klatki:
        im, m, tip = obraz(f, r)
        bez = usun(im, m)
        inne = [g for g in filmy if g != f]
        g = inne[int(rng.integers(len(inne)))]
        rd = dane[g][1][int(rng.integers(len(dane[g][1])))]
        im2, m2, _ = obraz(g, rd)
        im2 = cv2.resize(im2, (im.shape[1], im.shape[0])); m2 = cv2.resize(m2, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        bez_obce = usun(im2, m2)
        war = {'orig': (im, m), 'usuniete': (bez, np.zeros_like(m)), 'swoje': (wklej(bez, im, m), m),
               'obce': (wklej(bez_obce, im, m), m)}
        for w in ('rot_-30', 'rot_+30', 'dl_0.7', 'dl_1.3'):
            i2, mm = przeksztalc_narzedzie(im, m, tip, w)
            war[w] = (wklej(bez, i2, mm), mm)
        for w in WARIANTY:
            obrazy.append(Image.fromarray(war[w][0])); maski.append(war[w][1]); ramki_gt.append(ramka(war[w][1]))
    ramki_gt = np.array(ramki_gt, np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    for k in range(3):                                                    # podglad wejsc modelu
        pod = Image.new('RGB', (240 * len(WARIANTY), 160))
        for j in range(len(WARIANTY)):
            pod.paste(obrazy[k * len(WARIANTY) + j].resize((240, 160)), (j * 240, 0))
        pod.save(args.out / f'WEJSCIA_{k}.jpg', quality=88)

    baza = dict(np.load(args.baza)); wek = np.load(args.wektor)
    wyniki = {}
    for spec in args.detektor:
        nazwa, sciezka = spec.split('=', 1)
        det = rfdetr.RFDETRNano(resolution=args.resolution, pretrain_weights=sciezka)
        net = det.model.model.eval()
        backbone = next(m for _, m in net.named_modules() if type(m).__name__ == 'WindowedDinov2WithRegistersBackbone')
        gamma = backbone.layernorm.weight.detach().float().cpu().numpy(); beta = backbone.layernorm.bias.detach().float().cpu().numpy()
        cap = {}
        backbone.register_forward_hook(lambda mod, i, o: cap.__setitem__('fm', (o.feature_maps if hasattr(o, 'feature_maps') else o[0])[-1]))
        net.register_forward_hook(lambda mod, i, o: cap.__setitem__('out', o))
        pew, rp, akt_n, akt_t, zap = [], [], [], [], []
        for s in range(0, len(obrazy), 32):
            part = obrazy[s:s + 32]
            sizes = np.array([[o.width, o.height] for o in part], np.float32)
            with torch.inference_mode():
                det.predict(part, threshold=0.0)
            out = cap['out']
            logits, coords = (out['pred_logits'], out['pred_boxes']) if isinstance(out, dict) else \
                ((out[1], out[0]) if out[0].shape[-1] == 4 else (out[0], out[1]))
            sc, best = logits.float().sigmoid().max(-1).values.max(-1)
            ar = torch.arange(len(part), device=logits.device)
            c = coords[ar, best].float().clamp(0, 1).cpu().numpy()
            rp.append(np.stack([(c[:, 0] - c[:, 2] / 2) * sizes[:, 0], (c[:, 1] - c[:, 3] / 2) * sizes[:, 1],
                                c[:, 2] * sizes[:, 0], c[:, 3] * sizes[:, 1]], 1)); pew.append(sc.cpu().numpy())
            if nazwa == args.detektor_latentu:
                fm = cap['fm'].float(); b, k, h, w = fm.shape
                z = fm.permute(0, 2, 3, 1).reshape(b, h * w, k).cpu().numpy()
                red = ((z - beta) / np.abs(gamma) - baza['srodek']) @ baza['kier']
                a_ = red @ wek                                            # aktywacja wektora narzedzia na kazdym kawalku
                for j in range(b):
                    m = maski[s + j]
                    ul = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).ravel()
                    daleko = cv2.resize(cv2.dilate(m, np.ones((61, 61), np.uint8)).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).ravel() == 0
                    tl = a_[j][daleko]
                    zap.append(a_[j].reshape(h, w))
                    if ul.max() > 0.15:
                        n_ = a_[j][ul > 0.15]
                        akt_n.append([(np.mean(n_) - np.median(tl)) / (np.std(tl) + 1e-9), float(n_.max() > np.quantile(tl, .95))])
                    else:                                                  # brak narzedzia (wersja usuniete): miejsce, gdzie bylo
                        m0 = maski[(s + j) - ((s + j) % len(WARIANTY))]
                        ul0 = cv2.resize(m0.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).ravel()
                        n_ = a_[j][ul0 > 0.15] if (ul0 > 0.15).any() else a_[j][:1]
                        akt_n.append([(np.mean(n_) - np.median(tl)) / (np.std(tl) + 1e-9), float(n_.max() > np.quantile(tl, .95))])
        n, V = len(klatki), len(WARIANTY)
        pew = np.concatenate(pew).reshape(n, V); rp = np.concatenate(rp).reshape(n, V, 4)
        gt = ramki_gt.reshape(n, V, 4)
        x1 = np.maximum(rp[..., 0], gt[..., 0]); y1 = np.maximum(rp[..., 1], gt[..., 1])
        x2 = np.minimum(rp[..., 0] + rp[..., 2], gt[..., 0] + gt[..., 2]); y2 = np.minimum(rp[..., 1] + rp[..., 3], gt[..., 1] + gt[..., 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / np.maximum(rp[..., 2] * rp[..., 3] + gt[..., 2] * gt[..., 3] - inter, 1e-9)
        r = {'pewnosc_mediana': {w: float(np.median(pew[:, i])) for i, w in enumerate(WARIANTY)},
             'iou_mediana': {w: float(np.nanmedian(iou[:, i])) for i, w in enumerate(WARIANTY)},
             'udzial_wykrytych_iou_0_5': {w: float(np.nanmean((iou[:, i] > .5) & (pew[:, i] > .5))) for i, w in enumerate(WARIANTY)}}
        zapis = dict(pewnosc=pew, ramka=rp, ramka_gt=gt)
        if akt_n:
            A = np.array(akt_n).reshape(n, V, 2)
            r['wektor_kontrast_mediana'] = {w: float(np.median(A[:, i, 0])) for i, w in enumerate(WARIANTY)}
            r['wektor_zapalony_udzial'] = {w: float(A[:, i, 1].mean()) for i, w in enumerate(WARIANTY)}
            zapis['wektor'] = A; zapis['mapy'] = np.stack(zap).reshape(n, V, 24, 24).astype(np.float16)
        np.savez_compressed(args.out / f'KSZTALT_{nazwa}.npz', **zapis)
        wyniki[nazwa] = r
        print(f'\n=== {nazwa}', flush=True)
        for k_, v_ in r.items():
            print(f'  {k_:28s}', {w: round(x, 2) for w, x in v_.items()}, flush=True)
        del det, net; torch.cuda.empty_cache()
    (args.out / 'REPORT_ksztalt.json').write_text(json.dumps(dict(klatek=len(klatki), filmow=len(filmy), warianty=WARIANTY,
                                                             klatki=[f'{f}/{r["frame_number"]}' for f, r in klatki],
                                                             wyniki=wyniki), indent=2, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
