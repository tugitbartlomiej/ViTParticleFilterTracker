"""Wyniki TEST wybranej konfiguracji z przedziałami bootstrap po filmach (2000 losowań)."""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parents[1] / 'Dane'
d = np.load(D / 'test_miary.npz'); f = d['film']; u = np.unique(f); idx = {a: np.flatnonzero(f == a) for a in u}
rng = np.random.default_rng(0); los = [np.concatenate([idx[a] for a in rng.choice(u, len(u))]) for _ in range(2000)]
out = {}
for k in d.files:
    if not k.endswith('__wybrana') and not k.endswith('__glowica'):
        continue
    v = d[k]; s = {}
    for j, nazwa in enumerate(('dice', 'czulosc', 'precyzja', 'precyzja_sas1')):
        b = [v[l, j].mean() for l in los]
        s[nazwa] = [round(float(v[:, j].mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    s['n_med'] = float(np.median(v[:, 4])); out[k] = s
    print(k, s)
out['zero_losowa_grupa_precyzja'] = float(d['czestosc_narz'].mean()); print('losowa grupa: precyzja', round(out['zero_losowa_grupa_precyzja'], 3))
(D / 'test_podsumowanie.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
