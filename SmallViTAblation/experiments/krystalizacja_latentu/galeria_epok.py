#!/usr/bin/env python3
"""Galeria dowodow na wagach z kolejnych epok: te same klatki walidacyjne, trzy otoczenia narzedzia.

Odtwarza lokalnie dokladnie te klatki i przeksztalcenia, ktore widzial detektor w
pipeline/koopman_langevin/tozsamosc_obrot.py (ten sam wybor klatek i dawcow tla, to samo ziarno),
i rysuje detekcje z zapisu OBROT_<detektor>.npz.

Wiersze dla kazdej klatki: wlasne otoczenie (orig), kontrola klejenia (tlo_swoje), obce otoczenie (tlo_a).
Kolumny: epoki. Zolta ramka: detekcja z pewnoscia >= 0,5, czerwona: ponizej; zielona cienka: etykieta.

    py -3.11 experiments/krystalizacja_latentu/galeria_epok.py --dane <results_obrot_epoki> \\
        --adnotacje experiments/krystalizacja_latentu/dane/annotations_val.json \\
        --zipy ../Dataset/Cataract-LMM/3_Object_Tracking/annotations --ziarno 42 --out <katalog dowodow>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from galeria_sondy import LeniwyKatalog, czcionka

KORZEN = Path(__file__).resolve().parents[2]


def odtworz(katalog, na_film=12, seed=0):
    """Kopia wyboru klatek i dawcow z tozsamosc_obrot.main (kolejnosc losowan bez zmian)."""
    coco = json.loads((katalog / 'annotations_val.json').read_text(encoding='utf-8'))
    box_of = {}
    for a in coco['annotations']:
        box_of.setdefault(a['image_id'], a['bbox'])
    po_filmie: dict[str, list] = {}
    for im in coco['images']:
        if im['id'] in box_of and box_of[im['id']][2] >= 16 and box_of[im['id']][3] >= 8:
            po_filmie.setdefault(im['video_id'], []).append(im)
    rng = np.random.default_rng(seed)
    klatki = []
    for film in sorted(po_filmie):
        lista = sorted(po_filmie[film], key=lambda r: r['frame_id'])
        for i in np.linspace(0, len(lista) - 1, min(na_film, len(lista))).round().astype(int):
            klatki.append(lista[i])
    filmy = sorted(po_filmie)
    dawcy = []
    for im in klatki:
        inne = [f for f in filmy if f != im['video_id']]
        wyb = rng.choice(len(inne), 2, replace=False)
        wlasny = sorted(po_filmie[im['video_id']], key=lambda r: r['frame_id'])
        nr_ = int(im['frame_id'].split('/')[-1])
        daleko = [r for r in wlasny if abs(int(r['frame_id'].split('/')[-1]) - nr_) >= 250] or wlasny
        dawcy.append([po_filmie[inne[j]][int(rng.integers(len(po_filmie[inne[j]])))] for j in wyb]
                     + [im, daleko[int(rng.integers(len(daleko)))]])
    return klatki, dawcy, box_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dane', type=Path, required=True)
    ap.add_argument('--adnotacje', type=Path, required=True)
    ap.add_argument('--zipy', type=Path, required=True)
    ap.add_argument('--ziarno', default='42')
    ap.add_argument('--epoki', type=int, nargs='+', default=[1, 5, 7, 15, 29])
    ap.add_argument('--ile-klatek', type=int, default=6)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)

    spec = importlib.util.spec_from_file_location('tozsamosc_obrot', KORZEN / 'pipeline/koopman_langevin/tozsamosc_obrot.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    meta = json.loads((args.dane / 'META.json').read_text(encoding='utf-8'))
    war = meta['warianty']
    katalog = LeniwyKatalog(args.adnotacje, args.zipy, args.out / '_obrazy')
    klatki, dawcy, box_of = odtworz(katalog)
    assert [im['frame_id'] for im in klatki] == meta['frame_ids'], 'odtworzone klatki nie zgadzaja sie z zapisem'

    zapisy = {e: np.load(args.dane / f'OBROT_b1500_s{args.ziarno}_epoch_{e:03d}.npz') for e in args.epoki}
    io_, isw, iob = war.index('orig'), war.index('tlo_swoje'), war.index('tlo_a')
    assert np.allclose(zapisy[args.epoki[0]]['ramka_gt'][:, io_], [box_of[im['id']] for im in klatki], atol=1e-2)
    # klatki: najwiekszy spadek pewnosci na obcym tle miedzy najlepsza epoka a ostatnia, i dwie kontrprzykladowe
    p_b = zapisy[args.epoki[2]]['pewnosc'].astype(float); p_e = zapisy[args.epoki[-1]]['pewnosc'].astype(float)
    spadek = (p_b[:, iob] - p_e[:, iob]) * (p_e[:, io_] > .5)
    wybor = list(np.argsort(-spadek)[:args.ile_klatek - 2]) + list(np.argsort(spadek)[:2])

    W, H = 300, 200
    f_ = czcionka(15)
    wiersze = [(io_, 'wlasne tlo'), (isw, 'kontrola klejenia\n(wlasne tlo)'), (iob, 'obce tlo')]
    plotno = Image.new('RGB', (190 + len(args.epoki) * W, 36 + len(wybor) * len(wiersze) * H), 'white')
    dr = ImageDraw.Draw(plotno)
    for c, e in enumerate(args.epoki):
        dr.text((190 + c * W + 6, 8), f'epoka {e}', fill='black', font=f_)
    for r, j in enumerate(wybor):
        im = klatki[j]
        src = Image.open(katalog / im['file_name']).convert('RGB')
        dimg = []
        for rr in dawcy[j]:
            d_ = Image.open(katalog / rr['file_name']).convert('RGB').resize(src.size)
            bx, by, bw, bh = box_of[rr['id']]
            ob = (int(max(0, bx - .3 * bw)), int(max(0, by - .3 * bh)), int(min(d_.width, bx + 1.3 * bw)),
                  int(min(d_.height, by + 1.3 * bh)))
            d_.paste(d_.crop(ob).filter(ImageFilter.GaussianBlur(25)), ob[:2])
            dimg.append(d_)
        for w_i, (vi, etyk) in enumerate(wiersze):
            obraz, gt = mod.przeksztalc(src, box_of[im['id']], war[vi], dimg, None)
            y0 = 36 + (r * len(wiersze) + w_i) * H
            dr.text((6, y0 + H // 2 - 20), f'klatka {r + 1}\n{etyk}', fill='black', font=f_)
            for c, e in enumerate(args.epoki):
                z = zapisy[e]
                p = float(z['pewnosc'][j, vi]); px, py, pw, ph = z['ramka'][j, vi]
                kad = obraz.copy(); g = ImageDraw.Draw(kad)
                g.rectangle([gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]], outline=(0, 255, 0), width=2)
                kolor = (255, 220, 0) if p >= .5 else (255, 60, 60)
                g.rectangle([px, py, px + pw, py + ph], outline=kolor, width=5)
                kad = kad.resize((W - 4, H - 4)); g2 = ImageDraw.Draw(kad)
                g2.rectangle([0, 0, 118, 24], fill=(0, 0, 0)); g2.text((4, 3), f'pewnosc {p:.2f}', fill=kolor, font=f_)
                plotno.paste(kad, (190 + c * W + 2, y0 + 2))
    args.out.mkdir(parents=True, exist_ok=True)
    plotno.save(args.out / f'GALERIA_epoki_s{args.ziarno}.jpg', quality=90)
    print('zapisane', args.out / f'GALERIA_epoki_s{args.ziarno}.jpg', [klatki[j]['frame_id'] for j in wybor])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
