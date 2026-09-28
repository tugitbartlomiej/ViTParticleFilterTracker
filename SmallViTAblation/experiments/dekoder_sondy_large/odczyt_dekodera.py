#!/usr/bin/env python3
"""Odczyt stanu DEKODERA RF-DETR LargeDeprecated na 72 sondach z zapisanych wag pętli Large (bez uczenia).

Po co: latent kręgosłupa nie wiąże się z mAP (../petla_od_poczatku_large/DIAGNOZA_PORAZKI.md). Sprawdzamy, czy stan
dekodera jest lepszym kandydatem na „latent tożsamości” dla równania Langevina: czy niesie jakość detekcji sondy
i czy jest niezmienniczy względem obrotu, tła i jasności narzędzia.

Dla każdej sondy i każdej wagi (ziarna 42/43/44; k00 i koopman k01..k06):
  - propozycje przed dekoderem (two-stage): max IoU z ramką prawdziwą wśród 300 propozycji;
  - po każdej warstwie dekodera l = 1..3: zapytanie o najwyższej pewności, jego pewność i IoU z ramką prawdziwą,
    oraz „jakość” q_l = max_zapytania(pewność × IoU) (jak trudność = 1 − q w wyborze klatek);
  - stan ukryty (384 liczby) zapytania wybranego po ostatniej warstwie.
Ramka prawdziwa = prostokąt otaczający maskę narzędzia sondy.

    <venv rfdetr>/python odczyt_dekodera.py     ->  wyniki/DEKODER.npz, wyniki/DEKODER.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PETLA_L = HERE.parent / 'petla_od_poczatku_large'
sys.path.insert(0, str(PETLA_L))
import obrot_large as O  # noqa: E402

PROBE = O.PETLA / 'wyniki/local_1810448/run/probe'


def ramka(maska):
    ys, xs = np.nonzero(maska)
    h, w = maska.shape
    return np.array([xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h])


def iou(b, g):
    x0, y0 = np.maximum(b[..., 0], g[0]), np.maximum(b[..., 1], g[1])
    x1, y1 = np.minimum(b[..., 2], g[2]), np.minimum(b[..., 3], g[3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ab = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]); ag = (g[2] - g[0]) * (g[3] - g[1])
    return inter / (ab + ag - inter + 1e-9)


def cxcywh_xyxy(b):
    return np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2,
                     b[..., 1] + b[..., 3] / 2], -1)


def main() -> int:
    import torch
    import torch.nn.functional as Fn
    from PIL import Image
    from rfdetr import RFDETRLargeDeprecated
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    rows, mapa, fit = O.sondy()
    man = json.loads((PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obrazy, gt = [], []
    for r in rows:
        m = man[[i for i, x in enumerate(man) if x['image'] == r['image']][0]]
        obrazy.append(np.asarray(Image.open(PROBE / m['image']).convert('RGB'), dtype=np.float32) / 255.)
        gt.append(ramka(np.asarray(Image.open(PROBE / m['mask']).convert('L')) > 127))
    wagi = [(s, 'k00', O.W / f's{s}/wspolny/k00/model_0000160.pth') for s in (42, 43, 44)]
    wagi += [(s, f'k{k:02d}', O.W / f's{s}/koopman/k{k:02d}/model_0000160.pth') for s in (42, 43, 44) for k in range(1, 7)]
    wyn = {}; stany = {}
    for s, k, p in wagi:
        wr = RFDETRLargeDeprecated(pretrain_weights=str(p), num_classes=1, amp=False, device='cuda')
        net = wr.model.model.to(device='cuda', dtype=torch.float32).eval()
        res = wr.model_config.resolution
        mean = torch.tensor(wr.means, device='cuda').view(3, 1, 1); std = torch.tensor(wr.stds, device='cuda').view(3, 1, 1)
        zlap = {}
        h = net.transformer.register_forward_hook(lambda mod, inp, out: zlap.__setitem__('hs', out[0]))
        q, pew, iu, prop, hid = [], [], [], [], []
        with torch.no_grad():
            for x, g in zip(obrazy, gt):
                t = torch.from_numpy(x).permute(2, 0, 1).cuda()
                t = Fn.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False)[0]
                out = net(nested_tensor_from_tensor_list([(t - mean) / std]))
                warstwy = [a for a in out.get('aux_outputs', [])] + [dict(pred_logits=out['pred_logits'], pred_boxes=out['pred_boxes'])]
                ql, pl, il = [], [], []
                for w in warstwy:
                    sc = w['pred_logits'][0, :, 0].sigmoid().cpu().numpy(); bx = cxcywh_xyxy(w['pred_boxes'][0].cpu().numpy())
                    io = iou(bx, g); j = int(sc.argmax())
                    ql.append(float((sc * io).max())); pl.append(float(sc[j])); il.append(float(io[j]))
                enc = out['enc_outputs']; eb = cxcywh_xyxy(enc['pred_boxes'][0].cpu().numpy())
                prop.append(float(iou(eb, g).max()))
                hs = zlap['hs']
                hs = hs[-1] if hs.dim() == 4 else hs
                j = int(out['pred_logits'][0, :, 0].argmax())
                hid.append(hs[0, j].float().cpu().numpy())
                q.append(ql); pew.append(pl); iu.append(il)
        h.remove()
        key = f's{s}_{k}'
        wyn[key] = dict(q=q, pewnosc=pew, iou=iu, propozycje_iou=prop)
        stany[key] = np.stack(hid)
        print(f'{key}: jakosc po warstwach {np.mean(q, 0).round(3).tolist()}, IoU propozycji {np.mean(prop):.3f}', flush=True)
        del wr, net; torch.cuda.empty_cache()
    (HERE / 'wyniki').mkdir(exist_ok=True)
    np.savez_compressed(HERE / 'wyniki/DEKODER.npz', **stany)
    (HERE / 'wyniki/DEKODER.json').write_text(json.dumps(dict(opis=__doc__, sondy=[r['image'] for r in rows], wynik=wyn),
                                                          ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
