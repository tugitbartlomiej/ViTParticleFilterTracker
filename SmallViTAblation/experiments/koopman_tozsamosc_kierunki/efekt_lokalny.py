#!/usr/bin/env python3
"""Efekt lokalny kontynuacji (sekcja 7d): jakosc detekcji na klatkach val podobnych do kotwic.

Etapy (kazdy zapisuje do wyniki/dobor_klatek/efekt_lokalny/):
  podobienstwo  DINOv2 ViT-S/14 rownoleglej sesji (ten sam model, wycinek i preprocessing co indeks train) liczy
                cechy wycinkow wszystkich adnotacji val (12 043 klatki) i cosinus do zapytan target.png trzech
                kotwic z fastgan_K73_krok0245_gn -> PODOBIENSTWO_VAL.npz (frame_ids, cos [N x 3]).
  predykcje     kazdy zestaw wag (9 x FINAL.pth kampanii 134 klatek + start model_0000245.pth) przewiduje ramki na
                wszystkich klatkach val (prog 0,05, batch) -> PREDYKCJE_{nazwa}.npz.
  analiza       AP@0,5 i AP@0,5:0,95 (COCO, 101 punktow, klasa jedna) oraz TP/FP/FN przy pewnosci 0,5 na podzbiorach
                val zdefiniowanych podobienstwem: top-134 (tyle, ile dolozono do treningu), cos >= 0,7, 10 procent
                najblizszych, top-50 na kotwice, dolna polowa (kontrola), wszystkie; roznice sparowane ramie - bez_dodania
                po ziarnach; decyle podobienstwa -> EFEKT_LOKALNY.json, EFEKT_LOKALNY.png.

    <venv rfdetr> python efekt_lokalny.py --etap wszystko   (KMP_DUPLICATE_LIB_OK=TRUE przez faiss/torch nie jest tu
    potrzebne: faiss nieuzywany; cosinusy liczone w numpy)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PETLA = EXP / 'koopman_fastgan_petla'                      # rownolegla sesja: tylko odczyt
INDEX_CFG = PETLA / 'wyniki' / 'local_1810448' / 'run' / 'index' / 'config.json'
GN = HERE / 'wyniki' / 'fastgan_K73_krok0245_gn'
OUT = HERE / 'wyniki' / 'dobor_klatek' / 'efekt_lokalny'
WAGI = HERE / 'wyniki' / 'dobor_klatek' / 'trening150' / 'wagi'
START = EXP / 'koopman_gesta_historia' / 'wyniki' / 'run_1810662' / 'pretrain' / 'model_0000245.pth'
KOTWICE = ['00', '24', '48']
RAMIONA = ['bez_dodania', 'koopman_fastgan', 'losowe']
ZIARNA = [42, 43, 44]


def wczytaj_val():
    sys.path.insert(0, str(PETLA))
    from wyszukiwanie import read_manifest
    cfg = json.loads(INDEX_CFG.read_text())
    rows = [r for r in read_manifest(cfg['manifest_path'], cfg['data_root']) if r['split'] == 'val']
    rows.sort(key=lambda r: r['frame_id'])
    return cfg, rows


def etap_podobienstwo(args):
    from PIL import Image
    sys.path.insert(0, str(PETLA))
    from wyszukiwanie import DinoEncoder, prepare_query_image, resolve_model_source, square_crop_box
    cfg, rows = wczytaj_val()
    src = cfg['model_source']
    source = resolve_model_source(torch_hub_dir=src['torch_hub_dir'], weights=src['weights'])
    assert source['model_sha256'] == src['model_sha256'], 'inny model DINOv2 niz w indeksie train'
    enc = DinoEncoder(source, device=args.device, pooling=cfg['pooling'])
    assert enc.preprocessing == cfg['preprocessing']
    zapytania = {}
    for k in KOTWICE:
        for rodzaj in ('target', 'current'):
            with Image.open(GN / f'kotwica_{k}' / f'{rodzaj}.png') as im:
                crop, _ = prepare_query_image(im)
            zapytania[f'{k}_{rodzaj}'] = enc.encode([crop])[0]
    Q = np.stack([zapytania[f'{k}_target'] for k in KOTWICE])          # 3 x d
    Qc = np.stack([zapytania[f'{k}_current'] for k in KOTWICE])
    print('cos target vs current tej samej kotwicy:', [round(float(Q[i] @ Qc[i]), 4) for i in range(3)], flush=True)
    margin = cfg['margin']
    cos = np.full((len(rows), 3), -1.0); cos_c = np.full((len(rows), 3), -1.0); n_ann = np.zeros(len(rows), int)
    batch, idx, t0 = [], [], time.time()

    def flush():
        F = enc.encode(batch)                                           # b x d, znormalizowane
        S, Sc = F @ Q.T, F @ Qc.T
        for j, i in enumerate(idx):
            cos[i] = np.maximum(cos[i], S[j]); cos_c[i] = np.maximum(cos_c[i], Sc[j])
        batch.clear(); idx.clear()

    for i, r in enumerate(rows):
        with Image.open(r['path']) as source_im:
            image = source_im.convert('RGB')
        for a in r['annotations']:
            box = square_crop_box(a['bbox'], margin)
            batch.append(image.crop(box)); idx.append(i); n_ann[i] += 1
            if len(batch) >= args.batch:
                flush()
        if i % 1000 == 0:
            print(f'  {i}/{len(rows)} klatek, {time.time() - t0:.0f} s', flush=True)
    if batch:
        flush()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'PODOBIENSTWO_VAL.npz', frame_ids=np.array([r['frame_id'] for r in rows]),
                        cos=cos, cos_current=cos_c, n_ann=n_ann, kotwice=np.array(KOTWICE), Q=Q, Qc=Qc)
    m = cos.max(1)
    print(f'val {len(rows)} klatek, adnotacji {int(n_ann.sum())}; max-cos: mediana {np.median(m):.3f}, '
          f'>=0,7: {(m >= .7).sum()}, >=0,75: {(m >= .75).sum()}, top-134 prog {np.sort(m)[-134]:.3f}; '
          f'{time.time() - t0:.0f} s', flush=True)


def zestawy_wag():
    z = {'start_0245': START}
    for p in sorted(WAGI.glob('seed*_*.pth')):
        z[p.stem] = p
    return z


def etap_predykcje(args):
    import torch
    from PIL import Image
    from rfdetr import RFDETRNano
    torch.backends.cudnn.benchmark = False
    _, rows = wczytaj_val()
    OUT.mkdir(parents=True, exist_ok=True)
    for nazwa, sciezka in zestawy_wag().items():
        cel = OUT / f'PREDYKCJE_{nazwa}.npz'
        if cel.exists() and not args.nadpisz:
            print('pomijam gotowe', cel.name, flush=True); continue
        t0 = time.time()
        model = RFDETRNano(pretrain_weights=str(sciezka), num_classes=1, resolution=384, amp=False,
                           device=args.device, trust_checkpoint=True)
        boxes, scores, offs = [], [], [0]
        for s in range(0, len(rows), args.batch):
            ims = []
            for r in rows[s:s + args.batch]:
                with Image.open(r['path']) as im:
                    ims.append(im.convert('RGB'))
            with torch.inference_mode():
                dets = model.predict(ims, threshold=args.prog)
            if not isinstance(dets, list):
                dets = [dets]
            for d in dets:
                boxes.append(np.asarray(d.xyxy, dtype=np.float32).reshape(-1, 4))
                scores.append(np.asarray(d.confidence, dtype=np.float32).reshape(-1))
                offs.append(offs[-1] + len(scores[-1]))
            if (s // args.batch) % 100 == 0:
                print(f'  {nazwa}: {s}/{len(rows)}, {time.time() - t0:.0f} s', flush=True)
        np.savez_compressed(cel, frame_ids=np.array([r['frame_id'] for r in rows]), boxes=np.concatenate(boxes),
                            scores=np.concatenate(scores), offsets=np.array(offs), prog=args.prog, wagi=str(sciezka))
        print(f'{nazwa}: {offs[-1]} ramek na {len(rows)} klatkach, {time.time() - t0:.0f} s', flush=True)
        del model
        torch.cuda.empty_cache()


def iou_macierz(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: np.clip(x[:, 2:] - x[:, :2], 0, None).prod(-1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-12)


def ap_coco(pred, gt, idx, progi):
    """pred: lista (boxes, scores) per klatka; gt: lista xyxy per klatka; idx: indeksy klatek podzbioru.
    Zwraca AP (101 punktow) dla kazdego progu IoU oraz TP/FP/FN przy pewnosci 0,5 i IoU 0,5."""
    recs = [(float(s), i, k) for i in idx for k, s in enumerate(pred[i][1])]
    recs.sort(key=lambda t: -t[0])
    npos = sum(len(gt[i]) for i in idx)
    ious = {}
    for i in idx:
        if len(pred[i][1]) and len(gt[i]):
            ious[i] = iou_macierz(pred[i][0], gt[i])
    wyn = {}
    for thr in progi:
        matched = {i: np.zeros(len(gt[i]), bool) for i in idx}
        tp = np.zeros(len(recs)); fp = np.zeros(len(recs))
        for n, (s, i, k) in enumerate(recs):
            best, best_iou = -1, thr
            if i in ious:
                row = ious[i][k].copy(); row[matched[i]] = -1
                j = int(row.argmax())
                if row[j] >= best_iou:
                    best = j
            if best >= 0:
                matched[i][best] = True; tp[n] = 1
            else:
                fp[n] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / max(npos, 1); prec = ctp / np.maximum(ctp + cfp, 1e-12)
        # COCO: precyzja monotoniczna od konca, 101 punktow recall
        for n in range(len(prec) - 2, -1, -1):
            prec[n] = max(prec[n], prec[n + 1])
        rs = np.linspace(0, 1, 101); ap = 0.0
        for r in rs:
            m = np.searchsorted(rec, r, side='left')
            ap += prec[m] if m < len(prec) else 0.0
        wyn[thr] = ap / 101
        if abs(thr - .5) < 1e-9:
            sel = np.array([s >= .5 for s, _, _ in recs], bool)
            TP = int(tp[sel].sum()); FP = int(fp[sel].sum()); FN = npos - TP
            wyn['tp_fp_fn'] = (TP, FP, FN)
    return wyn


def etap_analiza(args):
    _, rows = wczytaj_val()
    P = np.load(OUT / 'PODOBIENSTWO_VAL.npz')
    assert list(P['frame_ids']) == [r['frame_id'] for r in rows]
    cos = P['cos']; m = cos.max(1); N = len(rows)
    gt = [np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]]
                    for a in r['annotations']], float).reshape(-1, 4) for r in rows]
    order = np.argsort(-m)
    podzbiory = {'wszystkie': np.arange(N), 'top134': order[:134], 'cos_ge_0.7': np.where(m >= .7)[0],
                 'top10proc': order[:N // 10], 'dolna_polowa': order[N // 2:]}
    for j, k in enumerate(KOTWICE):
        podzbiory[f'kotwica_{k}_top50'] = np.argsort(-cos[:, j])[:50]
    podzbiory['kotwice_top50_suma'] = np.unique(np.concatenate([podzbiory[f'kotwica_{k}_top50'] for k in KOTWICE]))
    decyle = np.array_split(order, 10)                                 # decyl 0 = najbardziej podobne
    progi = [round(.5 + .05 * i, 2) for i in range(10)]
    pred_all = {}
    for f in sorted(OUT.glob('PREDYKCJE_*.npz')):
        D = np.load(f); off = D['offsets']; B = D['boxes']; S = D['scores']   # npz czyta tablice przy kazdym dostepie
        pred_all[f.stem[len('PREDYKCJE_'):]] = [(B[off[i]:off[i + 1]], S[off[i]:off[i + 1]]) for i in range(N)]
    print('zestawy wag:', list(pred_all), flush=True)
    wyniki = {}
    for nazwa, pred in pred_all.items():
        wyniki[nazwa] = {}
        for pn, idx in podzbiory.items():
            w = ap_coco(pred, gt, list(idx), progi)
            tp, fp, fn = w['tp_fp_fn']
            wyniki[nazwa][pn] = dict(n=int(len(idx)), ap50=w[.5], ap50_95=float(np.mean([w[t] for t in progi])),
                                     tp=tp, fp=fp, fn=fn, f1=2 * tp / max(2 * tp + fp + fn, 1))
        wyniki[nazwa]['decyle_ap50'] = [ap_coco(pred, gt, list(d), [.5])[.5] for d in decyle]
        print(f'{nazwa:24s} wszystkie AP50 {wyniki[nazwa]["wszystkie"]["ap50"]:.4f}  top134 {wyniki[nazwa]["top134"]["ap50"]:.4f}  '
              f'cos>=0.7 {wyniki[nazwa]["cos_ge_0.7"]["ap50"]:.4f}  top10% {wyniki[nazwa]["top10proc"]["ap50"]:.4f}  '
              f'dol50% {wyniki[nazwa]["dolna_polowa"]["ap50"]:.4f}', flush=True)
    # roznice sparowane po ziarnach
    roznice = {}
    for ramie in RAMIONA[1:]:
        roznice[ramie] = {}
        for pn in podzbiory:
            d = {met: [wyniki[f'seed{s}_{ramie}'][pn][met] - wyniki[f'seed{s}_bez_dodania'][pn][met]
                       for s in ZIARNA if f'seed{s}_{ramie}' in wyniki and f'seed{s}_bez_dodania' in wyniki]
                 for met in ('ap50', 'ap50_95', 'f1')}
            roznice[ramie][pn] = {met: dict(per_ziarno=v, srednia=float(np.mean(v)) if v else None,
                                            sd=float(np.std(v, ddof=1)) if len(v) > 1 else None) for met, v in d.items()}
        dd = np.array([[wyniki[f'seed{s}_{ramie}']['decyle_ap50'][q] - wyniki[f'seed{s}_bez_dodania']['decyle_ap50'][q]
                        for q in range(10)] for s in ZIARNA if f'seed{s}_{ramie}' in wyniki])
        roznice[ramie]['decyle_ap50'] = dict(srednia=dd.mean(0).tolist(), sd=dd.std(0, ddof=1).tolist() if len(dd) > 1 else None)
    for ramie in RAMIONA[1:]:
        print(f'd({ramie} - bez_dodania): ' + '  '.join(f'{pn} {roznice[ramie][pn]["ap50"]["srednia"]:+.4f}'
                                                       for pn in ('wszystkie', 'top134', 'cos_ge_0.7', 'top10proc', 'dolna_polowa')), flush=True)
    opis = {pn: dict(n=int(len(idx)), prog_maxcos=float(m[idx].min()) if len(idx) else None) for pn, idx in podzbiory.items()}
    raport = dict(podzbiory=opis, decyle_prog=[float(m[d].min()) for d in decyle], progi_iou=progi, prog_pewnosci_f1=.5,
                  wyniki=wyniki, roznice_vs_bez_dodania=roznice, uwaga='AP COCO 101 punktow liczone lokalnie z predykcji '
                  'prog 0,05; globalne wartosci moga sie roznic od VALIDATION.json z Edena (inna implementacja).')
    (OUT / 'EFEKT_LOKALNY.json').write_text(json.dumps(raport, indent=1, ensure_ascii=False))
    rysuj(wyniki, roznice, podzbiory, m, decyle)
    print('zapisano', OUT / 'EFEKT_LOKALNY.json', flush=True)


def rysuj(wyniki, roznice, podzbiory, m, decyle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.hist(m, bins=50, color='gray', alpha=.7)
    for pn, c in (('top134', 'C3'), ('cos_ge_0.7', 'C1'), ('top10proc', 'C2')):
        idx = podzbiory[pn]
        if len(idx):
            ax.axvline(m[idx].min(), color=c, label=f'{pn}: n={len(idx)}, cos ≥ {m[idx].min():.3f}')
    ax.set_xlabel('max cos DINOv2 do trzech kotwic (klatka val)'); ax.set_ylabel('liczba klatek'); ax.legend(fontsize=8)
    ax.set_title('Podobieństwo klatek val do kotwic')
    ax = axes[1]
    nazwy = ['wszystkie', 'dolna_polowa', 'top10proc', 'cos_ge_0.7', 'top134', 'kotwice_top50_suma']
    x = np.arange(len(nazwy))
    for k, (ramie, c) in enumerate((('koopman_fastgan', 'C0'), ('losowe', 'C1'))):
        sr = [roznice[ramie][pn]['ap50']['srednia'] for pn in nazwy]
        sd = [roznice[ramie][pn]['ap50']['sd'] or 0 for pn in nazwy]
        ax.errorbar(x + (k - .5) * .15, sr, yerr=sd, fmt='o', color=c, capsize=3,
                    label='dobrane (Koopman+FastGAN) − bez dodania' if ramie == 'koopman_fastgan' else 'losowe − bez dodania')
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([f'{pn}\n(n={len(podzbiory[pn])})' for pn in nazwy], fontsize=7, rotation=20)
    ax.set_ylabel('Δ AP@0,5 (średnia ± SD po 3 ziarnach)'); ax.set_title('Efekt dodania klatek na podzbiorach val'); ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    ax = axes[2]
    q = np.arange(10)
    for ramie, c in (('koopman_fastgan', 'C0'), ('losowe', 'C1')):
        sr = np.array(roznice[ramie]['decyle_ap50']['srednia']); sd = roznice[ramie]['decyle_ap50']['sd']
        ax.errorbar(q, sr, yerr=np.array(sd) if sd else None, fmt='o-', color=c, capsize=3, label=ramie)
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(q); ax.set_xticklabels([f'{i}\n≥{m[d].min():.2f}' for i, d in enumerate(decyle)], fontsize=7)
    ax.set_xlabel('decyl podobieństwa (0 = najbardziej podobne do kotwic)'); ax.set_ylabel('Δ AP@0,5 vs bez dodania')
    ax.set_title('Efekt wzdłuż podobieństwa'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'EFEKT_LOKALNY.png', dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--etap', choices=['podobienstwo', 'predykcje', 'analiza', 'wszystko'], default='wszystko')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--prog', type=float, default=.05); ap.add_argument('--nadpisz', action='store_true')
    a = ap.parse_args()
    if a.etap in ('podobienstwo', 'wszystko'):
        etap_podobienstwo(a)
    if a.etap in ('predykcje', 'wszystko'):
        etap_predykcje(a)
    if a.etap in ('analiza', 'wszystko'):
        etap_analiza(a)


if __name__ == '__main__':
    main()
