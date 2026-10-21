"""Podsumowanie odkrycia bez etykiet: kandydat 1 (bez wyboru), dowolny z 3 (wybór po etykiecie, oznaczony), zera."""
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D = Path(__file__).resolve().parents[1] / 'Dane'
W = [json.loads(l) for l in (D / 'odkrycie_bez_etykiet.jsonl').read_text(encoding='utf-8').splitlines()]
rng = np.random.default_rng(0)
def bf(y, f):
    y = np.asarray(y, float); u = np.unique(f); idx = {a: np.flatnonzero(f == a) for a in u}
    b = [y[np.concatenate([idx[a] for a in rng.choice(u, len(u))])].mean() for _ in range(2000)]
    return f'{y.mean():.2f} [{np.percentile(b, 2.5):.2f}–{np.percentile(b, 97.5):.2f}]'
out = {}
for model in ('epoka400_s3047', 'nieuczony_s47_k0'):
    r = [w for w in W if w['model'] == model]
    f = np.array([w['frame_id'].split('/')[-2] for w in r])
    k1 = [w['kandydaci'][0] for w in r]
    jest = np.array([k is not None for k in k1])
    prec = np.array([k['precyzja'] if k else 0.0 for k in k1])
    dice = np.array([(k['dice'] or 0.0) if k else 0.0 for k in k1])
    czul = np.array([(k['czulosc'] or 0.0) if k else 0.0 for k in k1])
    trafia = prec >= 0.5
    best_dice = np.array([max([(k['dice'] or 0) for k in w['kandydaci'] if k] or [0]) for w in r])
    s = dict(klatek=len(r), z_odkryciem=bf(jest, f), kand1_trafia_narzedzie_prec_ge_05=bf(trafia, f),
             kand1_precyzja_sr=bf(prec, f), kand1_dice_sr=bf(dice, f), kand1_czulosc_sr=bf(czul, f),
             kand1_mediana_n=float(np.median([k['n'] for k in k1 if k])) if jest.any() else None,
             najlepszy_z_3_dice_sr_UWAGA_wybor_po_etykiecie=bf(best_dice, f),
             zero_losowa_grupa_precyzja=f"{np.mean([w['czestosc_narz'] for w in r]):.3f}")
    out[model] = s
    print(model, json.dumps(s, ensure_ascii=False, indent=1))
(D / 'odkrycie_bez_etykiet_podsumowanie.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8', newline='\n')
