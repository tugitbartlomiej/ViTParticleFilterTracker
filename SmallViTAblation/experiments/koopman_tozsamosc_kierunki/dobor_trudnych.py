#!/usr/bin/env python3
"""Selekcja kontrolna „trudne”: 134 nieuzyte klatki train, na ktorych detektor z kroku 245 radzi sobie najgorzej.

Po co: kontrola hipotezy „trudne przypadki wzmacniaja detektor” bez generatora. Pula ta sama co dla wyszukiwarki
(klatki train spoza 1500 pretrainingu i spoza kotwic, 9 136). Trudnosc klatki = 1 - max_k (pewnosc_k * IoU_k z
ramka prawdziwa), liczona z predykcji model_0000245.pth (prog 0,05). Wybor: klatki od najtrudniejszej, najwyzej
2 na film (daje 83 filmy, porownywalnie z 84 filmami ramienia koopman_fastgan). Rundy po filmach jak w wyszukiwarce
odrzucono: brały najtrudniejsza klatke z kazdego filmu nawet gdy byla latwa (min. trudnosc 0,49 przy medianie puli
0,48). Zapis: wyniki/dobor_klatek/trening5ep/selekcja_trudne.json (+ TRUDNE_PREDYKCJE.npz; gdy istnieje, predykcje
nie sa liczone ponownie).

    <venv rfdetr> python dobor_trudnych.py
"""
from __future__ import annotations

import collections
import json
import sys
import time

import numpy as np

import efekt_lokalny as E

SEL150 = E.HERE / 'wyniki' / 'dobor_klatek' / 'trening150'
OUT = E.HERE / 'wyniki' / 'dobor_klatek' / 'trening5ep'
LIMIT_NA_FILM = 2


def main():
    import torch
    from PIL import Image
    from rfdetr import RFDETRNano
    sys.path.insert(0, str(E.PETLA))
    from wyszukiwanie import read_manifest
    cfg = json.loads(E.INDEX_CFG.read_text())
    rows_all = read_manifest(cfg['manifest_path'], cfg['data_root'])
    wyk = set(json.loads((E.HERE / 'wyniki' / 'dobor_klatek' / 'wykluczone_pretrain_1500_plus_kotwice.json').read_text()))
    pula = sorted([r for r in rows_all if r['split'] == 'train' and r['annotations'] and r['frame_id'] not in wyk],
                  key=lambda r: r['frame_id'])
    print('pula', len(pula), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    zapis = OUT / 'TRUDNE_PREDYKCJE.npz'
    if zapis.exists():
        Z = np.load(zapis)
        assert list(Z['frame_ids']) == [r['frame_id'] for r in pula]
        trud = Z['trudnosc']
        print('predykcje wczytane z', zapis.name, flush=True)
    else:
        trud = policz_trudnosc(pula, zapis, torch, Image, RFDETRNano)
    wybor, ile = [], collections.Counter()
    for i in np.argsort(-trud, kind='stable'):
        if ile[pula[i]['group_id']] < LIMIT_NA_FILM:
            wybor.append(int(i)); ile[pula[i]['group_id']] += 1
        if len(wybor) == 134:
            break
    zapisz(pula, trud, wybor)


def policz_trudnosc(pula, zapis, torch, Image, RFDETRNano):
    model = RFDETRNano(pretrain_weights=str(E.START), num_classes=1, resolution=384, amp=False, device='cuda',
                       trust_checkpoint=True)
    trud, t0 = np.zeros(len(pula)), time.time()
    boxes, scores, offs = [], [], [0]
    for s in range(0, len(pula), 32):
        ims = []
        for r in pula[s:s + 32]:
            with Image.open(r['path']) as im:
                ims.append(im.convert('RGB'))
        with torch.inference_mode():
            dets = model.predict(ims, threshold=.05)
        for j, d in enumerate(dets if isinstance(dets, list) else [dets]):
            b = np.asarray(d.xyxy, np.float32).reshape(-1, 4); c = np.asarray(d.confidence, np.float32).reshape(-1)
            r = pula[s + j]
            gt = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                           for a in r['annotations']], float)
            best = float((c[:, None] * E.iou_macierz(b, gt)).max()) if len(c) else 0.0
            trud[s + j] = 1 - best
            boxes.append(b); scores.append(c); offs.append(offs[-1] + len(c))
    print(f'predykcje {time.time() - t0:.0f} s', flush=True)
    np.savez_compressed(zapis, frame_ids=np.array([r['frame_id'] for r in pula]), trudnosc=trud,
                        boxes=np.concatenate(boxes), scores=np.concatenate(scores), offsets=np.array(offs))
    return trud


def zapisz(pula, trud, wybor):
    baza = json.loads((SEL150 / 'selekcja_bez_dodania.json').read_text())
    dobrane = {d['frame_id'] for d in json.loads((SEL150 / 'selekcja_koopman_fastgan.json').read_text())['dodane_szczegoly']}
    losowe = {d['frame_id'] for d in json.loads((SEL150 / 'selekcja_losowe.json').read_text())['dodane_szczegoly']}
    ids = [pula[i]['frame_id'] for i in wybor]
    sel = dict(run_id='kontynuacja5ep_od_245_trudne', baza=baza['baza'], dodane=len(ids),
               frame_ids=list(baza['frame_ids']) + ids,
               dodane_szczegoly=[dict(frame_id=pula[i]['frame_id'], video=pula[i]['group_id'], trudnosc=float(trud[i]),
                                      zrodlo=f'najgorsze dla model_0000245.pth, najwyzej {LIMIT_NA_FILM} na film')
                                 for i in wybor])
    (OUT / 'selekcja_trudne.json').write_text(json.dumps(sel, indent=1, ensure_ascii=False))
    t = trud[wybor]
    print(f'trudne: {len(ids)} klatek z {len({pula[i]["group_id"] for i in wybor})} filmow, trudnosc {t.min():.3f}..{t.max():.3f} '
          f'(mediana puli {np.median(trud):.3f}, dobrane {np.median([trud[k] for k, r in enumerate(pula) if r["frame_id"] in dobrane]):.3f}, '
          f'losowe {np.median([trud[k] for k, r in enumerate(pula) if r["frame_id"] in losowe]):.3f}); '
          f'wspolne z dobranymi {len(set(ids) & dobrane)}, z losowymi {len(set(ids) & losowe)}; '
          f'klatek bez zadnej detekcji >= 0,5 IoU*pewnosc: {(trud > .5).sum()}', flush=True)


if __name__ == '__main__':
    main()
