#!/usr/bin/env python3
"""Podglad 24 losowych klatek z selekcji „trudne”: ramka prawdziwa (zielona) i najlepsza predykcja detektora
z kroku 245 (czerwona, z pewnoscia). Sluzy do oceny, czy trudnosc to realny przypadek, czy blad oznaczenia.
Zapis: wyniki/dobor_klatek/trening5ep/TRUDNE_PODGLAD.png.

    py -3.11 podglad_trudnych.py
"""
from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image, ImageDraw

import efekt_lokalny as E

OUT = E.HERE / 'wyniki' / 'dobor_klatek' / 'trening5ep'


def main():
    sys.path.insert(0, str(E.PETLA))
    from wyszukiwanie import read_manifest
    cfg = json.loads(E.INDEX_CFG.read_text())
    byid = {r['frame_id']: r for r in read_manifest(cfg['manifest_path'], cfg['data_root'])}
    sel = json.loads((OUT / 'selekcja_trudne.json').read_text())['dodane_szczegoly']
    Z = np.load(OUT / 'TRUDNE_PREDYKCJE.npz')
    pos = {f: i for i, f in enumerate(Z['frame_ids'])}
    off, B, S = Z['offsets'], Z['boxes'], Z['scores']
    pick = np.random.default_rng(0).choice(len(sel), 24, replace=False)
    W, H = 360, 240
    sheet = Image.new('RGB', (6 * W, 4 * H), 'white')
    for k, j in enumerate(pick):
        fid = sel[j]['frame_id']; r = byid[fid]; i = pos[fid]
        im = Image.open(r['path']).convert('RGB'); sx, sy = W / im.width, H / im.height
        im = im.resize((W, H)); d = ImageDraw.Draw(im)
        for a in r['annotations']:
            x, y, w, h = a['bbox']
            d.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], outline=(0, 255, 0), width=2)
        b, s = B[off[i]:off[i + 1]], S[off[i]:off[i + 1]]
        if len(s):
            m = int(s.argmax()); x0, y0, x1, y1 = b[m]
            d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 0, 0), width=2)
            d.text((x0 * sx + 2, y0 * sy + 2), f'{s[m]:.2f}', fill=(255, 0, 0))
        d.text((4, 4), f'{sel[j]["video"][:7]}/{fid.split("/")[-1]} t={sel[j]["trudnosc"]:.2f}', fill=(255, 255, 0))
        sheet.paste(im, ((k % 6) * W, (k // 6) * H))
    sheet.save(OUT / 'TRUDNE_PODGLAD.png')
    print('zapisano', OUT / 'TRUDNE_PODGLAD.png')


if __name__ == '__main__':
    main()
