#!/usr/bin/env python3
"""Czy dodane klatki treningowe byly trudne dla detektora i czy jedna epoka je nauczyla (sekcja 7e).

Zestawy klatek train (wszystkie z puli, adnotacje dostarczone):
  dobrane   134 klatki ramienia koopman_fastgan (wyszukiwarka dla obrazow FastGAN przewidzianego latentu),
  losowe    134 klatki ramienia losowe (nieuzyte klatki train, ziarno 43),
  widziane  134 klatki losowane (ziarno 7) z 1500 klatek pretrainingu, ktore detektor juz widzial.
Wagi: start model_0000245.pth i 9 wag FINAL.pth kampanii 134 klatek. Miary jak w efekt_lokalny.py: AP@0,5 i
AP@0,5:0,95 (COCO, 101 punktow) i F1 przy pewnosci 0,5. Zapis: wyniki/dobor_klatek/efekt_lokalny/TRUDNOSC_DODANYCH.json.

    <venv rfdetr> python trudnosc_dodanych.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import efekt_lokalny as E

SEL = E.HERE / 'wyniki' / 'dobor_klatek' / 'trening150'


def main():
    import torch
    from PIL import Image
    from rfdetr import RFDETRNano
    sys.path.insert(0, str(E.PETLA))
    from wyszukiwanie import read_manifest
    cfg = json.loads(E.INDEX_CFG.read_text())
    byid = {r['frame_id']: r for r in read_manifest(cfg['manifest_path'], cfg['data_root'])}
    dobrane = [d['frame_id'] for d in json.loads((SEL / 'selekcja_koopman_fastgan.json').read_text())['dodane_szczegoly']]
    losowe = [d['frame_id'] for d in json.loads((SEL / 'selekcja_losowe.json').read_text())['dodane_szczegoly']]
    baza = json.loads((SEL / 'selekcja_bez_dodania.json').read_text())['frame_ids']
    widziane = list(np.random.default_rng(7).choice(baza, 134, replace=False))
    zestawy = {'dobrane': dobrane, 'losowe': losowe, 'widziane': widziane}
    assert not set(dobrane) & set(baza) and not set(losowe) & set(baza) and all(byid[f]['split'] == 'train' for f in dobrane + losowe)
    ids = sorted(set(dobrane + losowe + widziane))
    rows = [byid[f] for f in ids]
    pos = {f: i for i, f in enumerate(ids)}
    gt = [np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                    for a in r['annotations']], float).reshape(-1, 4) for r in rows]
    progi = [round(.5 + .05 * i, 2) for i in range(10)]
    wyniki = {}
    for nazwa, sciezka in E.zestawy_wag().items():
        t0 = time.time()
        model = RFDETRNano(pretrain_weights=str(sciezka), num_classes=1, resolution=384, amp=False,
                           device='cuda', trust_checkpoint=True)
        pred = []
        for s in range(0, len(rows), 32):
            ims = []
            for r in rows[s:s + 32]:
                with Image.open(r['path']) as im:
                    ims.append(im.convert('RGB'))
            with torch.inference_mode():
                dets = model.predict(ims, threshold=.05)
            for d in (dets if isinstance(dets, list) else [dets]):
                pred.append((np.asarray(d.xyxy, np.float32).reshape(-1, 4), np.asarray(d.confidence, np.float32).reshape(-1)))
        wyniki[nazwa] = {}
        for zn, fr in zestawy.items():
            w = E.ap_coco(pred, gt, [pos[f] for f in fr], progi)
            tp, fp, fn = w['tp_fp_fn']
            wyniki[nazwa][zn] = dict(ap50=w[.5], ap50_95=float(np.mean([w[t] for t in progi])), tp=tp, fp=fp, fn=fn,
                                     f1=2 * tp / max(2 * tp + fp + fn, 1))
        print(f'{nazwa:24s} ' + '  '.join(f'{zn} AP50 {wyniki[nazwa][zn]["ap50"]:.3f} F1 {wyniki[nazwa][zn]["f1"]:.3f}'
                                         for zn in zestawy) + f'  ({time.time() - t0:.0f} s)', flush=True)
        del model
        torch.cuda.empty_cache()
    (E.OUT / 'TRUDNOSC_DODANYCH.json').write_text(json.dumps(dict(zestawy={k: v for k, v in zestawy.items()}, wyniki=wyniki),
                                                            indent=1, ensure_ascii=False))
    print('zapisano', E.OUT / 'TRUDNOSC_DODANYCH.json')


if __name__ == '__main__':
    main()
