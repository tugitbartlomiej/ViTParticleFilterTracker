#!/usr/bin/env python3
"""Odczyt stanu dekodera na 72 sondach dla LOKALNYCH wag RF-DETR Nano (bramki G1 i G2 z
../sterowanie_langevin_kalman/PLAN.md bez Edenu). Te same miary co odczyt_dekodera.py (Large).

Zestawy:
  nano5ep    kampania 5 epok (koopman_tozsamosc_kierunki/.../trening5ep): 4 ramiona × 3 ziarna × 5 epok, mAP@0,5
             całego val po każdej epoce z metrics.csv  → G1: czy jakość sond wiąże się z mAP MIĘDZY ramionami
  nanopilot  pilot petla_uczenia_koopman_fastgan: 3 pętle × 8 obrotów, ziarno 42, mAP z REPORT.json
  nanogesty  bieg 1810662 (koopman_gesta_historia): wagi co 5 aktualizacji, kroki 5..805 → G2: trajektoria stanu

    <venv rfdetr>/python odczyt_dekodera_nano.py --zestaw nano5ep|nanopilot|nanogesty  ->  wyniki/NANO_<zestaw>.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import odczyt_dekodera as OD

EXP = OD.HERE.parent
T5 = EXP / 'koopman_tozsamosc_kierunki/wyniki/dobor_klatek/trening5ep'
PILOT = EXP / 'petla_uczenia_koopman_fastgan/wyniki/lancuchy'
GESTY = EXP / 'koopman_gesta_historia/wyniki/run_1810662/pretrain'


def lista(zestaw):
    out = []
    if zestaw == 'nano5ep':
        for s in (42, 43, 44):
            for ram in ('bez_dodania', 'losowe', 'trudne', 'koopman_fastgan'):
                r = list(csv.DictReader(open(T5 / f'raporty/seed{s}_{ram}/trening/metrics.csv', encoding='utf-8')))
                mapy = {int(x['epoch']): float(x['val/mAP_50']) for x in r if x.get('val/mAP_50')}
                for e in range(1, 6):
                    out.append(dict(klucz=f's{s}_{ram}_ep{e}', sciezka=T5 / f'wagi/seed{s}_{ram}_ep{e}.pth', mAP50=mapy.get(e - 1),
                                    ziarno=s, ramie=ram, czas=e))
    elif zestaw == 'nanopilot':
        for p in ('koopman', 'losowe', 'bez'):
            for d in sorted((PILOT / p).glob('k*')):
                w = sorted(d.glob('model_*.pth'))
                if w:
                    rep = json.loads((d / 'REPORT.json').read_text(encoding='utf-8'))
                    out.append(dict(klucz=f'{p}_{d.name}', sciezka=w[-1], mAP50=rep['validation'][0]['val/mAP_50'],
                                    ziarno=42, ramie=p, czas=int(d.name[1:])))
    else:
        for w in sorted(GESTY.glob('model_*.pth')):
            out.append(dict(klucz=w.stem, sciezka=w, mAP50=None, ziarno=0, ramie='gesty', czas=int(w.stem.split('_')[1])))
    return out


def main(argv=None) -> int:
    import torch
    import torch.nn.functional as Fn
    from PIL import Image
    from rfdetr import RFDETRNano
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zestaw', required=True, choices=['nano5ep', 'nanopilot', 'nanogesty'])
    a = ap.parse_args(argv)
    rows, _, _ = OD.O.sondy()
    man = json.loads((OD.PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obrazy, gt = [], []
    for r in rows:
        m = man[[i for i, x in enumerate(man) if x['image'] == r['image']][0]]
        obrazy.append(np.asarray(Image.open(OD.PROBE / m['image']).convert('RGB'), dtype=np.float32) / 255.)
        gt.append(OD.ramka(np.asarray(Image.open(OD.PROBE / m['mask']).convert('L')) > 127))
    wagi = lista(a.zestaw)
    wr = RFDETRNano(pretrain_weights=str(wagi[0]['sciezka']), num_classes=1, resolution=384, amp=False, device='cuda')
    net = wr.model.model.to(device='cuda', dtype=torch.float32).eval()
    res = wr.model_config.resolution
    mean = torch.tensor(wr.means, device='cuda').view(3, 1, 1); std = torch.tensor(wr.stds, device='cuda').view(3, 1, 1)
    wej = [nested_tensor_from_tensor_list([(Fn.interpolate(torch.from_numpy(x).permute(2, 0, 1).cuda()[None], size=(res, res),
                                                            mode='bilinear', align_corners=False)[0] - mean) / std]) for x in obrazy]
    wyn = []
    for i, w in enumerate(wagi):
        sd = torch.load(w['sciezka'], map_location='cpu', weights_only=False)
        net.load_state_dict(sd['model'] if 'model' in sd else sd, strict=True)
        q, prop = [], []
        with torch.no_grad():
            for t, g in zip(wej, gt):
                out = net(t)
                warstwy = list(out.get('aux_outputs', [])) + [dict(pred_logits=out['pred_logits'], pred_boxes=out['pred_boxes'])]
                ql = []
                for L in warstwy:
                    sc = L['pred_logits'][0, :, 0].sigmoid().cpu().numpy()
                    ql.append(float((sc * OD.iou(OD.cxcywh_xyxy(L['pred_boxes'][0].cpu().numpy()), g)).max()))
                q.append(ql)
                prop.append(float(OD.iou(OD.cxcywh_xyxy(out['enc_outputs']['pred_boxes'][0].cpu().numpy()), g).max()))
        wyn.append(dict({k: v for k, v in w.items() if k != 'sciezka'}, sciezka=str(w['sciezka']), q=q, propozycje_iou=prop))
        if i % 10 == 0 or i == len(wagi) - 1:
            print(f'{i + 1}/{len(wagi)} {w["klucz"]}: jakosc {np.mean(np.array(q)[:, -1]):.3f} mAP {w["mAP50"]}', flush=True)
    (OD.HERE / 'wyniki' / f'NANO_{a.zestaw}.json').write_text(json.dumps(dict(opis=__doc__, wynik=wyn), ensure_ascii=False),
                                                              encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
