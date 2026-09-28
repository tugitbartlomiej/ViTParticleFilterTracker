#!/usr/bin/env python3
"""Arkusze do obejrzenia: 72 sondy z odpowiedzią detektora i próbka zbioru walidacyjnego.

  SONDY_ARKUSZ.png   12 obrazów źródłowych × 6 wariantów (256×256, wycinek wokół narzędzia z filmów TRENINGOWYCH);
                     zielona ramka = prawdziwe położenie (z maski), czerwona = najpewniejsze wykrycie detektora,
                     napis: q = max(pewność × IoU) po ostatniej warstwie dekodera (to, co liczą bramki)
  WAL_PROBKA.png     12 losowych klatek walidacyjnych (całe kadry 720×480 z innych filmów) z ramkami narzędzia
Wagi: RF-DETR Nano, pilot pętli, ramię `bez`, ostatni obrót k08 (mAP@0,5 val 0,907).

    <venv rfdetr>/python arkusz_sond.py  ->  wyniki/SONDY_ARKUSZ.png, wyniki/WAL_PROBKA.png
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import odczyt_dekodera as OD

WARIANTY = ('original', 'rotation90', 'black', 'yellow', 'length08', 'foreground_bright')
NAZWY = {'original': 'oryginał', 'rotation90': 'obrót 90°', 'black': 'czarne tło', 'yellow': 'żółte tło',
         'length08': 'krótsze narzędzie', 'foreground_bright': 'jaśniejsze narzędzie'}
WAGI = OD.HERE.parent / 'petla_uczenia_koopman_fastgan/wyniki/lancuchy/bez/k08'
WAL = OD.O.PETLA / 'wyniki/local_1810448/run/datasets/pretrain/valid/_annotations.coco.json'


def czcionka(r):
    try:
        return ImageFont.truetype('arial.ttf', r)
    except OSError:
        return ImageFont.load_default()


def main() -> int:
    import torch
    import torch.nn.functional as Fn
    from rfdetr import RFDETRNano
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    rows, _, _ = OD.O.sondy()
    man = json.loads((OD.PROBE / 'manifest.json').read_text(encoding='utf-8'))
    wagi = sorted(WAGI.glob('model_*.pth'))[-1]
    wr = RFDETRNano(pretrain_weights=str(wagi), num_classes=1, resolution=384, amp=False, device='cuda')
    net = wr.model.model.to(device='cuda', dtype=torch.float32).eval()
    res = wr.model_config.resolution
    mean = torch.tensor(wr.means, device='cuda').view(3, 1, 1); std = torch.tensor(wr.stds, device='cuda').view(3, 1, 1)
    kafle = {}
    for r in rows:
        m = man[[i for i, x in enumerate(man) if x['image'] == r['image']][0]]
        img = Image.open(OD.PROBE / m['image']).convert('RGB')
        g = OD.ramka(np.asarray(Image.open(OD.PROBE / m['mask']).convert('L')) > 127)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1).cuda()[None]
        t = nested_tensor_from_tensor_list([(Fn.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)[0] - mean) / std])
        with torch.no_grad():
            out = net(t)
        sc = out['pred_logits'][0, :, 0].sigmoid().cpu().numpy()
        bx = OD.cxcywh_xyxy(out['pred_boxes'][0].cpu().numpy())
        q = float((sc * OD.iou(bx, g)).max()); j = int(sc.argmax())
        d = ImageDraw.Draw(img); W, H = img.size
        d.rectangle([g[0] * W, g[1] * H, g[2] * W, g[3] * H], outline=(0, 230, 0), width=2)
        d.rectangle([bx[j, 0] * W, bx[j, 1] * H, bx[j, 2] * W, bx[j, 3] * H], outline=(255, 40, 40), width=2)
        d.rectangle([0, H - 22, W, H], fill=(0, 0, 0))
        d.text((4, H - 20), f'q {q:.2f}   pewność {sc[j]:.2f}', fill=(255, 255, 255), font=czcionka(15))
        kafle[(r['source_file'], r['variant'])] = img
    zrodla = sorted({k[0] for k in kafle})
    T, NAG, BOK = 256, 34, 150
    ark = Image.new('RGB', (BOK + T * len(WARIANTY), NAG + T * len(zrodla)), (255, 255, 255))
    d = ImageDraw.Draw(ark)
    for c, v in enumerate(WARIANTY):
        d.text((BOK + c * T + 8, 8), NAZWY[v], fill=(0, 0, 0), font=czcionka(18))
    for r_, s in enumerate(zrodla):
        d.text((6, NAG + r_ * T + T // 2 - 20), s.replace('_S1_P03_', '\nklatka ').replace('.png', ''), fill=(0, 0, 0), font=czcionka(14))
        for c, v in enumerate(WARIANTY):
            ark.paste(kafle[(s, v)], (BOK + c * T, NAG + r_ * T))
    ark.save(OD.HERE / 'wyniki/SONDY_ARKUSZ.png')
    # próbka walidacji
    a = json.loads(WAL.read_text(encoding='utf-8'))
    ann = {}
    for x in a['annotations']:
        ann.setdefault(x['image_id'], []).append(x['bbox'])
    random.seed(0); wyb = random.sample(a['images'], 12)
    K = 360; kaf = []
    for im in wyb:
        img = Image.open(im['file_name']).convert('RGB'); d = ImageDraw.Draw(img)
        for b in ann.get(im['id'], []):
            d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=(0, 230, 0), width=3)
        img = img.resize((K, K * img.height // img.width))
        ImageDraw.Draw(img).text((4, 4), Path(im['file_name']).stem, fill=(255, 255, 0), font=czcionka(14))
        kaf.append(img)
    h = kaf[0].height
    wal = Image.new('RGB', (4 * K, 3 * h), (255, 255, 255))
    for i, k in enumerate(kaf):
        wal.paste(k, ((i % 4) * K, (i // 4) * h))
    wal.save(OD.HERE / 'wyniki/WAL_PROBKA.png')
    print('wagi', wagi, '| walidacja: obrazów', len(a['images']), 'filmów', len({Path(i['file_name']).stem[:14] for i in a['images']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
