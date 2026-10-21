"""Podsumowanie oceny na 25 filmach TEST niewidzianych w treningu: grupa bez etykiet, głowica, 192D i LDA wobec zera
przestrzennego; tryb 'glowny' (seed 47 i 3047) oraz 'seedy' (24 seedy, rozrzut między seedami). Bootstrap po filmach."""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parents[1] / 'Dane'
rng = np.random.default_rng(0)


def boot(y, f, n=2000):
    y = np.asarray(y, float); u = np.unique(f); idx = {a: np.flatnonzero(f == a) for a in u}
    b = [y[np.concatenate([idx[a] for a in rng.choice(u, len(u))])].mean() for _ in range(n)]
    return [round(float(y.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def p05(r, widok):
    return [w['sep'][widok + '_p']['fisher'] <= 0.05 for w in r]


out = {}
if (D / 'niewidziane_glowny.jsonl').exists():
    W = [json.loads(l) for l in (D / 'niewidziane_glowny.jsonl').read_text(encoding='utf-8').splitlines()]
    for c in dict.fromkeys(w['ckpt'] for w in W):
        r = [w for w in W if w['ckpt'] == c]; f = np.array([w['film'] for w in r])
        rs = [w for w in r if w['sep'] and '192D_p' in w['sep']]; fs = np.array([w['film'] for w in rs])
        s = dict(n=len(r), filmy=len(set(f)), iou=boot([w['iou'] for w in r], f),
                 grupa_dice=boot([w['grupa']['dice'] for w in r], f), grupa_precyzja=boot([w['grupa']['precyzja'] for w in r], f),
                 grupa_prec_sas1=boot([w['grupa']['precyzja_sas1'] for w in r], f), grupa_czulosc=boot([w['grupa']['czulosc'] for w in r], f),
                 grupa_n_med=float(np.median([w['grupa']['n'] for w in r])), glowica_dice=boot([w['glowica']['dice'] for w in r], f),
                 n_sep=len(rs), p05_192D=boot(p05(rs, '192D'), fs), p05_LDA=boot(p05(rs, 'LDA'), fs),
                 fisher192_med=round(float(np.median([w['sep']['192D']['fisher'] for w in rs])), 2),
                 fisher192_tlo_med=round(float(np.median([w['sep']['192D_tlo_med']['fisher'] for w in rs])), 2))
        out['glowny_' + c] = s
        print(f"{c:13s} IoU {s['iou'][0]:.2f} | grupa Dice {s['grupa_dice']} prec {s['grupa_precyzja'][0]:.2f} ±1 {s['grupa_prec_sas1'][0]:.2f} "
              f"n {s['grupa_n_med']:.0f} | głowica {s['glowica_dice'][0]:.2f} | p≤0,05 192D {s['p05_192D']} LDA {s['p05_LDA']} | "
              f"Fisher 192D {s['fisher192_med']} vs tło {s['fisher192_tlo_med']}")
if (D / 'niewidziane_seedy.jsonl').exists():
    W = [json.loads(l) for l in (D / 'niewidziane_seedy.jsonl').read_text(encoding='utf-8').splitlines()]
    kroki = sorted({w['ckpt'].split('_')[1] for w in W})
    seedy = sorted({w['ckpt'].split('_')[0] for w in W}, key=lambda s: int(s[1:]))
    print('\nseedy:', len(seedy))
    for k in kroki:
        d = []; p = []
        for sd in seedy:
            r = [w for w in W if w['ckpt'] == f'{sd}_{k}']
            if not r:
                continue
            d.append(np.mean([w['grupa']['dice'] for w in r]))
            rs = [w for w in r if w['sep'] and '192D_p' in w['sep']]
            if rs:
                p.append(np.mean(p05(rs, '192D')))
        s = dict(n_seedow=len(d), dice_srednia=round(float(np.mean(d)), 3), dice_min=round(float(np.min(d)), 3), dice_max=round(float(np.max(d)), 3))
        if p:
            s.update(p05_192D_srednia=round(float(np.mean(p)), 3), p05_192D_min=round(float(np.min(p)), 3), p05_192D_max=round(float(np.max(p)), 3))
        out['seedy_' + k] = s
        print(k, s)
(D / 'niewidziane_podsumowanie.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
