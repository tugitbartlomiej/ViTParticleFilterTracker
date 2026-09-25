#!/usr/bin/env python3
"""Kampania 5 epok (sekcja 7f): krzywe uczenia, efekt globalny i lokalny czterech ramion, ocena kryterium z PLAN.md.

Wejscia:
  wyniki/dobor_klatek/trening5ep/wagi/seed{S}_{ramie}_ep{k}.pth  (k = 1..5, model_0000{103k}.pth z Edena)
  wyniki/dobor_klatek/trening5ep/raporty/seed{S}_{ramie}/{REPORT.json, trening/metrics.csv}
  wyniki/dobor_klatek/efekt_lokalny/PODOBIENSTWO_VAL.npz i PREDYKCJE_start_0245.npz (sekcja 7d)
Etapy:
  predykcje  ep5: cale val (12 043 klatki); ep1..4: tylko 10 procent klatek val najblizszych kotwicom (1 204).
  analiza    AP@0,5 i AP@0,5:0,95 (COCO 101 pkt, jak efekt_lokalny.py) na podzbiorach, roznice sparowane wzgledem
             bez_dodania i losowe, krzywe po epokach (globalna z metrics.csv Edena, lokalna z predykcji), kryterium
             -> wyniki/dobor_klatek/trening5ep/efekt_lokalny/EFEKT_5EP.json, EFEKT_5EP.png.

    <venv rfdetr> python efekt_lokalny_5ep.py --etap wszystko
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

import efekt_lokalny as E

KAMP = E.HERE / 'wyniki' / 'dobor_klatek' / 'trening5ep'
WAGI = KAMP / 'wagi'
RAP = KAMP / 'raporty'
OUT = KAMP / 'efekt_lokalny'
RAMIONA = ['bez_dodania', 'koopman_fastgan', 'losowe', 'trudne']
ZIARNA = [42, 43, 44]
EPOKI = [1, 2, 3, 4, 5]
PROGI = [round(.5 + .05 * i, 2) for i in range(10)]


def podzbiory(rows):
    P = np.load(E.OUT / 'PODOBIENSTWO_VAL.npz')
    assert list(P['frame_ids']) == [r['frame_id'] for r in rows]
    cos = P['cos']; m = cos.max(1); N = len(rows); order = np.argsort(-m)
    pz = {'wszystkie': np.arange(N), 'dolna_polowa': order[N // 2:], 'top10proc': order[:N // 10],
          'cos_ge_0.7': np.where(m >= .7)[0], 'top134': order[:134]}
    for j, k in enumerate(E.KOTWICE):
        pz[f'kotwica_{k}_top50'] = np.argsort(-cos[:, j])[:50]
    pz['kotwice_top50_suma'] = np.unique(np.concatenate([pz[f'kotwica_{k}_top50'] for k in E.KOTWICE]))
    return pz


def przewiduj(model, rows, idx, batch, torch, Image):
    boxes, scores, offs = [], [], [0]
    for s in range(0, len(idx), batch):
        ims = []
        for i in idx[s:s + batch]:
            with Image.open(rows[i]['path']) as im:
                ims.append(im.convert('RGB'))
        with torch.inference_mode():
            dets = model.predict(ims, threshold=.05)
        for d in (dets if isinstance(dets, list) else [dets]):
            boxes.append(np.asarray(d.xyxy, np.float32).reshape(-1, 4))
            scores.append(np.asarray(d.confidence, np.float32).reshape(-1))
            offs.append(offs[-1] + len(scores[-1]))
    return np.concatenate(boxes), np.concatenate(scores), np.array(offs)


def etap_predykcje(args):
    import torch
    from PIL import Image
    from rfdetr import RFDETRNano
    _, rows = E.wczytaj_val()
    pz = podzbiory(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ZIARNA:
        for a in RAMIONA:
            for k in EPOKI:
                w = WAGI / f'seed{s}_{a}_ep{k}.pth'
                cel = OUT / f'PRED_seed{s}_{a}_ep{k}.npz'
                if cel.exists() or not w.exists():
                    continue
                idx = pz['wszystkie'] if k == EPOKI[-1] else pz['top10proc']
                t0 = time.time()
                model = RFDETRNano(pretrain_weights=str(w), num_classes=1, resolution=384, amp=False, device='cuda',
                                   trust_checkpoint=True)
                B, S, off = przewiduj(model, rows, idx, args.batch, torch, Image)
                np.savez_compressed(cel, idx=idx, boxes=B, scores=S, offsets=off)
                print(f'seed{s}_{a}_ep{k}: {len(idx)} klatek, {time.time() - t0:.0f} s', flush=True)
                del model
                torch.cuda.empty_cache()


def wczytaj_pred(sciezka, N):
    D = np.load(sciezka)
    idx = D['idx'] if 'idx' in D.files else np.arange(N)
    off, B, S = D['offsets'], D['boxes'], D['scores']
    pred = [(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))] * N
    for j, i in enumerate(idx):
        pred[i] = (B[off[j]:off[j + 1]], S[off[j]:off[j + 1]])
    return pred, set(int(i) for i in idx)


def miary(pred, gt, idx):
    w = E.ap_coco(pred, gt, list(idx), PROGI)
    tp, fp, fn = w['tp_fp_fn']
    return dict(n=int(len(idx)), ap50=w[.5], ap50_95=float(np.mean([w[t] for t in PROGI])), f1=2 * tp / max(2 * tp + fp + fn, 1))


def krzywa_globalna(s, a):
    p = RAP / f'seed{s}_{a}' / 'trening' / 'metrics.csv'
    if not p.exists():
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        if r.get('val/mAP_50'):
            out[int(r['epoch'])] = float(r['val/mAP_50'])
    return [out.get(e) for e in range(len(EPOKI))]


def etap_analiza(args):
    _, rows = E.wczytaj_val()
    N = len(rows)
    gt = [np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                    for a in r['annotations']], float).reshape(-1, 4) for r in rows]
    pz = podzbiory(rows)
    start, _ = wczytaj_pred(E.OUT / 'PREDYKCJE_start_0245.npz', N)
    wyniki, krzywe, globalne = {}, {}, {}
    wyniki['start_0245'] = {pn: miary(start, gt, idx) for pn, idx in pz.items()}
    for s in ZIARNA:
        for a in RAMIONA:
            n = f'seed{s}_{a}'
            rp = RAP / n / 'REPORT.json'
            if rp.exists():
                v = json.loads(rp.read_text())['validation'][0]
                globalne[n] = dict(mAP50=v['val/mAP_50'], mAP50_95=v['val/mAP_50_95'], F1=v['val/F1'])
            f5 = OUT / f'PRED_{n}_ep{EPOKI[-1]}.npz'
            if not f5.exists():
                continue
            pred, _ = wczytaj_pred(f5, N)
            wyniki[n] = {pn: miary(pred, gt, idx) for pn, idx in pz.items()}
            kr = {'top134': [wyniki['start_0245']['top134']['ap50']], 'top10proc': [wyniki['start_0245']['top10proc']['ap50']]}
            for k in EPOKI:
                fk = OUT / f'PRED_{n}_ep{k}.npz'
                if not fk.exists():
                    kr['top134'].append(None); kr['top10proc'].append(None); continue
                pk, _ = wczytaj_pred(fk, N)
                kr['top134'].append(E.ap_coco(pk, gt, list(pz['top134']), [.5])[.5])
                kr['top10proc'].append(E.ap_coco(pk, gt, list(pz['top10proc']), [.5])[.5])
            kr['globalna_eden'] = krzywa_globalna(s, a)
            krzywe[n] = kr
            print(f'{n:26s} val {globalne.get(n, {}).get("mAP50", float("nan")):.4f}  top134 {wyniki[n]["top134"]["ap50"]:.4f}  '
                  f'top10% {wyniki[n]["top10proc"]["ap50"]:.4f}  cos>=0,7 {wyniki[n]["cos_ge_0.7"]["ap50"]:.4f}', flush=True)
    roznice = {}
    for ref in ('bez_dodania', 'losowe'):
        for a in RAMIONA:
            if a == ref:
                continue
            key = f'{a}-{ref}'
            roznice[key] = {}
            pary = [s for s in ZIARNA if f'seed{s}_{a}' in wyniki and f'seed{s}_{ref}' in wyniki]
            for pn in pz:
                d = [wyniki[f'seed{s}_{a}'][pn]['ap50'] - wyniki[f'seed{s}_{ref}'][pn]['ap50'] for s in pary]
                roznice[key][pn] = dict(per_ziarno=d, srednia=float(np.mean(d)) if d else None,
                                        sd=float(np.std(d, ddof=1)) if len(d) > 1 else None)
            dg = [globalne[f'seed{s}_{a}']['mAP50'] - globalne[f'seed{s}_{ref}']['mAP50'] for s in ZIARNA
                  if f'seed{s}_{a}' in globalne and f'seed{s}_{ref}' in globalne]
            roznice[key]['val_eden_mAP50'] = dict(per_ziarno=dg, srednia=float(np.mean(dg)) if dg else None,
                                                  sd=float(np.std(dg, ddof=1)) if len(dg) > 1 else None)
    kryterium = {}
    for a in ('koopman_fastgan', 'trudne'):
        lok = roznice.get(f'{a}-losowe', {}).get('top134', {}).get('per_ziarno', [])
        glob = roznice.get(f'{a}-bez_dodania', {}).get('val_eden_mAP50', {}).get('srednia')
        kryterium[a] = dict(lokalnie_lepsze_od_losowych_we_wszystkich_ziarnach=bool(len(lok) == 3 and all(x > 0 for x in lok)),
                            roznice_top134_vs_losowe=lok, globalnie_vs_bez_dodania=glob,
                            globalnie_nie_gorsze=bool(glob is not None and glob >= -.002))
        kryterium[a]['spelnione'] = kryterium[a]['lokalnie_lepsze_od_losowych_we_wszystkich_ziarnach'] and kryterium[a]['globalnie_nie_gorsze']
    for key in roznice:
        r = roznice[key]
        print(f'd({key}): val {r["val_eden_mAP50"]["srednia"] if r["val_eden_mAP50"]["srednia"] is None else round(r["val_eden_mAP50"]["srednia"], 4)}  '
              + '  '.join(f'{pn} {r[pn]["srednia"]:+.4f}' for pn in ('top134', 'cos_ge_0.7', 'top10proc', 'dolna_polowa') if r[pn]['srednia'] is not None), flush=True)
    print('kryterium:', json.dumps(kryterium, ensure_ascii=False), flush=True)
    raport = dict(podzbiory={pn: int(len(i)) for pn, i in pz.items()}, globalne_eden=globalne, wyniki=wyniki, krzywe=krzywe,
                  roznice=roznice, kryterium=kryterium,
                  uwaga='AP lokalne z predykcji progu 0,05 (efekt_lokalny.ap_coco); val_eden = VALIDATION.json z Edena. '
                        'Krzywe: epoka 0 = start model_0000245.pth; epoki 1-4 lokalnie tylko na 10 procentach val najblizszych kotwicom.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'EFEKT_5EP.json').write_text(json.dumps(raport, indent=1, ensure_ascii=False))
    rysuj(krzywe, roznice, pz)
    print('zapisano', OUT / 'EFEKT_5EP.json', flush=True)


def rysuj(krzywe, roznice, pz):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kol = dict(bez_dodania='k', koopman_fastgan='C0', losowe='C1', trudne='C3')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (klucz, tyt, x0) in zip(axes[0], [('globalna_eden', 'mAP@0,5 całego val (Eden) po epokach', 1),
                                               ('top134', 'AP@0,5 na 134 klatkach val najbliższych kotwicom', 0)]):
        for a in RAMIONA:
            ys = [krzywe[f'seed{s}_{a}'][klucz] for s in ZIARNA if f'seed{s}_{a}' in krzywe and krzywe[f'seed{s}_{a}'][klucz]]
            if not ys:
                continue
            Y = np.array([[np.nan if v is None else v for v in y] for y in ys], float)
            if klucz == 'globalna_eden':
                Y = Y[:, :len(EPOKI)]
            x = np.arange(x0, x0 + Y.shape[1])
            ax.errorbar(x, np.nanmean(Y, 0), yerr=np.nanstd(Y, 0), fmt='o-', color=kol[a], capsize=3, label=a)
        ax.set_xlabel('epoka kontynuacji (0 = krok 245)'); ax.set_title(tyt); ax.grid(alpha=.3); ax.legend(fontsize=8)
    nazwy = ['wszystkie', 'dolna_polowa', 'top10proc', 'cos_ge_0.7', 'top134', 'kotwice_top50_suma']
    x = np.arange(len(nazwy))
    for ax, ref in zip(axes[1], ('bez_dodania', 'losowe')):
        arms = [a for a in RAMIONA if a != ref]
        for j, a in enumerate(arms):
            r = roznice.get(f'{a}-{ref}')
            if not r:
                continue
            sr = [r[pn]['srednia'] if r[pn]['srednia'] is not None else np.nan for pn in nazwy]
            sd = [r[pn]['sd'] or 0 for pn in nazwy]
            ax.errorbar(x + (j - 1) * .15, sr, yerr=sd, fmt='o', color=kol[a], capsize=3, label=f'{a} − {ref}')
        ax.axhline(0, color='k', lw=.8)
        ax.set_xticks(x); ax.set_xticklabels([f'{pn}\n(n={len(pz[pn])})' for pn in nazwy], fontsize=7, rotation=20)
        ax.set_ylabel('Δ AP@0,5 po 5 epokach (średnia ± SD, 3 ziarna)'); ax.set_title(f'Różnice względem: {ref}')
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'EFEKT_5EP.png', dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--etap', choices=['predykcje', 'analiza', 'wszystko'], default='wszystko')
    ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()
    if a.etap in ('predykcje', 'wszystko'):
        etap_predykcje(a)
    if a.etap in ('analiza', 'wszystko'):
        etap_analiza(a)


if __name__ == '__main__':
    main()
