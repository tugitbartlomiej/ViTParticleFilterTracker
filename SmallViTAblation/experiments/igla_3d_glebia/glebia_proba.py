#!/usr/bin/env python3
"""Etap 1 (PLAN.md): czy Depth Anything V2 widzi igłę cystotomu jako sztywny prosty przedmiot.

Dla 300 kolejnych klatek każdego z trzech filmów: głębia względna (większa = bliżej) na pikselach igły w funkcji
odległości od końcówki wzdłuż osi, dopasowanie prostej (R², nachylenie), kontrast igły wobec pierścienia tła wokół niej,
głębia igły względem tęczówki w jednostkach rozrzutu głębi rogówki, spójność nachylenia między kolejnymi klatkami
i kontrola: nachylenie głębi tła wzdłuż tej samej osi (czy model nie powtarza po prostu kształtu oka).

    python glebia_proba.py [--filmy TR_0002_S1_P03 ...] [--dl 300] [--cpu]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'serwowizja_zawias_narzedzia'))
import zawias as ZW  # noqa: E402

MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
FILMY = ['TR_0002_S1_P03', 'TR_0006_S1_P03', 'TR_0020_S1_P03']
WYN = HERE / 'wyniki'


def odcinek(v, dl):
    """Początek odcinka dl klatek z największą liczbą okien testu zawiasu (indeks gęsty jak w ZAWIAS.json)."""
    zaw = json.loads((HERE.parent / 'serwowizja_zawias_narzedzia' / 'wyniki' / 'ZAWIAS.json').read_text(encoding='utf-8'))
    st = np.array([o['start'] for o in zaw['okna'][v]['oko']])
    if len(st) == 0:
        return 0
    return int(max(st, key=lambda s: int(((st >= s) & (st + 30 <= s + dl)).sum())))


def maska(polys, h, w):
    m = np.zeros((h, w), np.uint8)
    for p in polys or []:
        p = np.rint(np.asarray(p, np.float64).reshape(-1, 2)).astype(np.int32)
        if len(p) >= 3:
            cv2.fillPoly(m, [p], 1)
    return m.astype(bool)


def prosta(s, d):
    A = np.stack([np.ones_like(s), s], 1)
    wsp, *_ = np.linalg.lstsq(A, d, rcond=None)
    r = d - A @ wsp
    return wsp, float(1 - (r ** 2).sum() / max(((d - d.mean()) ** 2).sum(), 1e-12))


def klatka(D, r, h, w):
    tip, seg = ZW.js(r['cystotome_tip']), ZW.js(r['cystotome_seg'])
    if tip is None or seg is None:
        return None
    mi = maska(seg, h, w)
    if mi.sum() < 30:
        return None
    ys, xs = np.nonzero(mi)
    P = np.stack([xs, ys], 1).astype(np.float64)
    tip = np.asarray(tip, np.float64)
    C = np.cov((P - P.mean(0)).T)
    w_, V = np.linalg.eigh(C)
    u = V[:, 1]
    if (P.mean(0) - tip) @ u < 0:
        u = -u
    s = (P - tip) @ u
    dn = D[ys, xs]
    (a, b), r2 = prosta(s, dn)
    pier = cv2.dilate(mi.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool) & ~mi
    yr, xr = np.nonzero(pier)
    dr = D[yr, xr]
    (ar, br), r2r = prosta((np.stack([xr, yr], 1) - tip) @ u, dr)
    mad = float(np.median(np.abs(dr - np.median(dr)))) * 1.4826 + 1e-9
    rog = maska(ZW.js(r['cornea_seg']), h, w)
    zr = maska(ZW.js(r['pupil_seg']), h, w)
    igla_szer = cv2.dilate(mi.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    teczowka = rog & ~zr & ~igla_szer
    tlo_rog = rog & ~igla_szer
    if teczowka.sum() < 200 or tlo_rog.sum() < 200:
        sig, d_tecz = np.nan, np.nan
    else:
        q1, q3 = np.percentile(D[tlo_rog], [25, 75]); sig = float(q3 - q1) + 1e-9
        d_tecz = float(np.median(D[teczowka]))
    return dict(r2=r2, nachylenie=float(b), nachylenie_tla=float(br), r2_tla=r2r,
                kontrast=float((np.median(dn) - np.median(dr)) / mad),
                konc_wzgl_teczowki=float((a - d_tecz) / sig) if np.isfinite(sig) else np.nan,
                nachylenie_na_100px_wzgl=float(100 * b / sig) if np.isfinite(sig) else np.nan,
                dlugosc_px=float(s.max()), pikseli=int(mi.sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--filmy', nargs='*', default=FILMY)
    ap.add_argument('--dl', type=int, default=300)
    ap.add_argument('--paczka', type=int, default=8)
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dev = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
    proc = AutoImageProcessor.from_pretrained(MODEL)
    net = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev).eval()
    WYN.mkdir(exist_ok=True)
    t0 = time.time(); wyn, podglad = {}, []
    for v in a.filmy:
        with zipfile.ZipFile(ZW.ARCH / f'{v}.zip') as z:
            rows = list(csv.DictReader(io.StringIO(z.read(f'{v}/annotation.csv').decode('utf-8-sig'))))
            rows.sort(key=lambda r: int(r['frame_number']))
            st = odcinek(v, a.dl)
            rows = rows[st:st + a.dl]
            rek = []
            for i in range(0, len(rows), a.paczka):
                ims = [cv2.imdecode(np.frombuffer(z.read(f'{v}/{r["filename"]}'), np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
                       for r in rows[i:i + a.paczka]]
                h, w = ims[0].shape[:2]
                x = proc(images=ims, return_tensors='pt').to(dev)
                with torch.no_grad():
                    d = net(**x).predicted_depth[:, None]
                    d = torch.nn.functional.interpolate(d, size=(h, w), mode='bilinear', align_corners=False)[:, 0]
                d = d.float().cpu().numpy()
                for im, D, r in zip(ims, d, rows[i:i + a.paczka]):
                    k = klatka(D, r, h, w)
                    rek.append(dict(klatka=int(r['frame_number']), **(k or {})))
                    if k and len(podglad) < 4 * len(a.filmy) and int(r['frame_number']) % 75 == 0:
                        podglad.append((v, int(r['frame_number']), im, D, r))
        ok = [x for x in rek if 'r2' in x]
        kol = lambda n: np.array([x[n] for x in ok], float)
        b, bt = kol('nachylenie_na_100px_wzgl'), kol('nachylenie')
        kolejne = [(ok[j]['nachylenie'], ok[j + 1]['nachylenie']) for j in range(len(ok) - 1)
                   if ok[j + 1]['klatka'] == ok[j]['klatka'] + 1]
        kolejne = np.array(kolejne) if kolejne else np.zeros((0, 2))
        wyn[v] = dict(start=st, klatek=len(rek), z_igla=len(ok),
                      r2_mediana=round(float(np.median(kol('r2'))), 3),
                      r2_tla_mediana=round(float(np.median(kol('r2_tla'))), 3),
                      kontrast_mediana=round(float(np.median(kol('kontrast'))), 2),
                      igla_blizej_tla_odsetek=round(float(np.mean(kol('kontrast') > 1)), 3),
                      konc_wzgl_teczowki_mediana=round(float(np.nanmedian(kol('konc_wzgl_teczowki'))), 3),
                      nachylenie_na_100px_kwartyle=[round(float(q), 3) for q in np.nanpercentile(b, [25, 50, 75])],
                      korelacja_nachylen_kolejnych=round(float(np.corrcoef(kolejne.T)[0, 1]), 3) if len(kolejne) > 10 else None,
                      korelacja_nachylenia_igly_i_tla=round(float(np.corrcoef(bt, kol('nachylenie_tla'))[0, 1]), 3),
                      klatki=rek)
        print(v, {k_: v_ for k_, v_ in wyn[v].items() if k_ != 'klatki'}, f'{time.time() - t0:.0f} s', flush=True)
    (WYN / 'GLEBIA_PROBA.json').write_text(json.dumps(dict(model=MODEL, urzadzenie=dev, dl=a.dl, filmy=wyn),
                                                      ensure_ascii=False, indent=1), encoding='utf-8')
    rysuj(podglad, WYN / 'GLEBIA_PODGLAD.png')
    print(f'gotowe {time.time() - t0:.0f} s', flush=True)
    return 0


def rysuj(podglad, sciezka):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not podglad:
        return
    wyb = podglad[::max(1, len(podglad) // 4)][:4]
    fig, ax = plt.subplots(len(wyb), 3, figsize=(15, 3.6 * len(wyb)))
    ax = np.atleast_2d(ax)
    for (v, f, im, D, r), (a1, a2, a3) in zip(wyb, ax):
        h, w = D.shape
        mi = maska(ZW.js(r['cystotome_seg']), h, w)
        a1.imshow(im); a1.contour(mi, levels=[0.5], colors='yellow', linewidths=0.8)
        a1.set_title(f'{v}, klatka {f}', fontsize=9)
        g = a2.imshow(D, cmap='magma'); a2.contour(mi, levels=[0.5], colors='cyan', linewidths=0.6)
        a2.set_title('głębia względna (jaśniej = bliżej)', fontsize=9); fig.colorbar(g, ax=a2, fraction=0.03)
        tip = np.asarray(ZW.js(r['cystotome_tip']), float)
        ys, xs = np.nonzero(mi); P = np.stack([xs, ys], 1).astype(float)
        u = np.linalg.eigh(np.cov((P - P.mean(0)).T))[1][:, 1]
        if (P.mean(0) - tip) @ u < 0:
            u = -u
        a3.plot((P - tip) @ u, D[ys, xs], ',', alpha=0.4, label='igła')
        pier = cv2.dilate(mi.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool) & ~mi
        yr, xr = np.nonzero(pier)
        a3.plot((np.stack([xr, yr], 1) - tip) @ u, D[yr, xr], ',', alpha=0.4, color='gray', label='tło wokół igły')
        a3.set_xlabel('odległość od końcówki wzdłuż osi [px]'); a3.set_ylabel('głębia względna')
        a3.legend(markerscale=20, fontsize=8)
        for a_ in (a1, a2):
            a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle('Depth Anything V2 Small na klatkach kapsuloreksji: czy igła ma własną, liniową głębię?', fontsize=11)
    fig.tight_layout(); fig.savefig(sciezka, dpi=100); plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())
