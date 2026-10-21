"""LDA z obrysem na wczesnych checkpointach seedu 47: odsetek klatek z p ≤ 0,05 wobec zera przestrzennego, bootstrap po filmach."""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parents[1] / 'Dane'
W = [json.loads(l) for l in (D / 'lda_przebieg_uczenia_s47.jsonl').read_text(encoding='utf-8').splitlines()]
rng = np.random.default_rng(0); out = {}
for krok in sorted({w['krok'] for w in W}):
    r = [w for w in W if w['krok'] == krok and 'p_przestrz' in w['prog_0.25']]
    f = np.array([w['frame_id'].split('/')[-2] for w in r]); u = np.unique(f); idx = {a: np.flatnonzero(f == a) for a in u}
    s = dict(n=len(r), iou=round(float(np.mean([w['iou'] for w in r])), 3))
    for widok, q in (('LDA', 'fisher'), ('192D', 'fisher'), ('PCA3', 'hit'), ('PCA3', 'sil')):
        y = np.array([w['prog_0.25']['p_przestrz'][widok][q] <= 0.05 for w in r], float)
        b = [y[np.concatenate([idx[a] for a in rng.choice(u, len(u))])].mean() for _ in range(2000)]
        s[f'{widok}_{q}_p05'] = [round(float(y.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    s['fisher_LDA_med'] = round(float(np.median([w['prog_0.25']['LDA']['fisher'] for w in r])), 2)
    s['fisher_LDA_med_maska_w_tle'] = round(float(np.median([w['prog_0.25']['mediana_przestrz']['LDA']['fisher'] for w in r])), 2)
    out[krok] = s; print(krok, s)
(D / 'lda_przebieg_uczenia_s47_podsumowanie.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
