#!/usr/bin/env python3
"""Rysunki do oceny na zbiorze testowym (ocena_test.py): przebieg w czasie i galeria chybien.

  TEST_przebieg.png   dla kilku filmow testowych: pewnosc detektora, trafienie (IoU >= 0,5) i sila
                      wektora narzedzia w latencie, klatka po klatce
  TEST_chybienia.jpg  klatki, na ktorych najlepszy detektor chybia: zielona etykieta, zolta/czerwona
                      ramka detektora z pewnoscia
  TEST_modele.png     mAP i udzial trafionych klatek dla kazdych wag

    py -3.11 experiments/krystalizacja_latentu/wykresy_test.py --dane experiments/krystalizacja_latentu/wyniki/test \\
        --zipy ../Dataset/Cataract-LMM/3_Object_Tracking/annotations
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from galeria_sondy import czcionka
from ocena_test import iou_xywh


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dane', type=Path, required=True)
    ap.add_argument('--zipy', type=Path, required=True)
    ap.add_argument('--latent', default='ut_b0500')
    ap.add_argument('--najlepszy', default='b1500_best')
    ap.add_argument('--filmow', type=int, default=4)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rap = json.loads((args.dane / 'REPORT_test.json').read_text(encoding='utf-8'))
    prog = rap['prog']
    modele = list(rap['wyniki'])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(modele))
    ax[0].bar(x - .2, [rap['wyniki'][m]['mAP_50_95'] for m in modele], .4, label='mAP 0,5:0,95')
    ax[0].bar(x + .2, [rap['wyniki'][m]['mAP_50'] for m in modele], .4, label='mAP 0,5')
    ax[1].bar(x - .2, [rap['wyniki'][m]['trafione_klatki'] for m in modele], .4, label='trafione klatki z narzędziem')
    ax[1].bar(x + .2, [rap['wyniki'][m]['falszywe_alarmy_bez_narzedzia'] or 0 for m in modele], .4, label='fałszywe alarmy bez narzędzia')
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(modele, rotation=20); a.set_ylim(0, 1); a.legend(fontsize=8); a.grid(axis='y', alpha=.3)
    fig.suptitle(f'Zbiór testowy: {rap["klatek"]} klatek z {len(rap["filmy"])} operacji, których żaden detektor nie widział')
    fig.tight_layout(); fig.savefig(args.dane / 'TEST_modele.png', dpi=140); plt.close(fig)

    L = np.load(args.dane / f'TEST_{args.latent}.npz'); B = np.load(args.dane / f'TEST_{args.najlepszy}.npz')
    film = L['film']; ma = L['narzedzie']; gt = L['ramka_gt']
    def traf(Z):
        return ma & (Z['pewnosc'][:, 0] >= prog) & (iou_xywh(Z['ramki'][:, 0], gt) >= .5)
    tL, tB = traf(L), traf(B)
    po = rap['wyniki'][args.najlepszy]['po_filmie']
    wyb = sorted(po, key=lambda f: po[f]['trafione'] if po[f]['trafione'] is not None else 1)
    wyb = wyb[:args.filmow // 2] + wyb[-(args.filmow - args.filmow // 2):]
    fig, ax = plt.subplots(len(wyb), 1, figsize=(13, 2.6 * len(wyb)))
    for a, f in zip(np.atleast_1d(ax), wyb):
        m = film == f; nr = L['nr'][m]
        a.fill_between(nr, 0, 1, where=~ma[m], color='0.85', step='mid', label='brak narzędzia w adnotacji')
        a.plot(nr, B['pewnosc'][m, 0], color='C0', lw=.8, label=f'pewność {args.najlepszy}')
        a.plot(nr, L['pewnosc'][m, 0], color='C2', lw=.8, alpha=.7, label=f'pewność {args.latent}')
        ch = m & ma & ~tB
        a.plot(L['nr'][ch], np.full(ch.sum(), 1.03), '|', color='r', ms=8, label=f'chybienie {args.najlepszy}')
        w = L['wektor'][m, 0]
        b = a.twinx(); b.plot(nr, w, color='C1', lw=.6, alpha=.8); b.set_ylabel('kontrast wektora', color='C1', fontsize=8)
        a.set_ylim(0, 1.08); a.set_title(f'{f}: trafione {po[f]["trafione"]:.2f}, najdłuższa seria chybień {po[f]["najdluzsza_seria_chybien"]} próbek', fontsize=9)
        a.set_ylabel('pewność')
    np.atleast_1d(ax)[0].legend(fontsize=7, loc='lower left', ncol=4)
    np.atleast_1d(ax)[-1].set_xlabel('numer klatki w filmie')
    fig.tight_layout(); fig.savefig(args.dane / 'TEST_przebieg.png', dpi=130); plt.close(fig)

    # galeria chybien najlepszego detektora: po jednej z roznych filmow
    ch = np.flatnonzero(ma & ~tB)
    rng = np.random.default_rng(0); rng.shuffle(ch)
    uzyte, pokaz = set(), []
    for i in ch:
        if film[i] not in uzyte:
            uzyte.add(film[i]); pokaz.append(i)
        if len(pokaz) == 12:
            break
    W, H = 360, 240; f_ = czcionka(14)
    plotno = Image.new('RGB', (4 * W, 3 * H), 'white')
    for k, i in enumerate(pokaz):
        with zipfile.ZipFile(args.zipy / f'{film[i]}.zip') as z:
            im = Image.open(io.BytesIO(z.read(f'{film[i]}/{film[i]}_{int(L["nr"][i]):06d}.jpg'))).convert('RGB')
        g = ImageDraw.Draw(im)
        x0, y0, w_, h_ = gt[i]; g.rectangle([x0, y0, x0 + w_, y0 + h_], outline=(0, 255, 0), width=2)
        p = float(B['pewnosc'][i, 0]); px, py, pw, ph = B['ramki'][i, 0]
        kol = (255, 220, 0) if p >= prog else (255, 60, 60)
        g.rectangle([px, py, px + pw, py + ph], outline=kol, width=4)
        im = im.resize((W - 4, H - 4)); g = ImageDraw.Draw(im)
        g.rectangle([0, 0, W, 22], fill=(0, 0, 0))
        g.text((4, 3), f'{film[i][:7]} kl.{int(L["nr"][i])}  pewność {p:.2f}  IoU {float(iou_xywh(B["ramki"][i, 0], gt[i])):.2f}', fill=kol, font=f_)
        plotno.paste(im, ((k % 4) * W + 2, (k // 4) * H + 2))
    plotno.save(args.dane / 'TEST_chybienia.jpg', quality=90)
    print('zapisane rysunki w', args.dane)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
