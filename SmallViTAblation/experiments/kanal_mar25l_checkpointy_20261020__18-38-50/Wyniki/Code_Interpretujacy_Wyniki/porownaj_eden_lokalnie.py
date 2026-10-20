"""Porównanie redukcji pilota: Eden (H100, torch 2.10) vs lokalnie (torch 2.14). Wynik: Wyniki/Dane/eden/POROWNANIE_EDEN_LOKALNIE.json.

Miary wag liczone są w float64 z tych samych checkpointów (oczekiwana zgodność ~1e-12); miary tokenowe zależą od
forwardu FP32 na innym GPU i innej wersji biblioteki (oczekiwane drobne różnice). Decyzje hipotez muszą być identyczne.
"""
import csv
import json
import math
import sys
from pathlib import Path

exp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
dane = exp / 'Wyniki' / 'Dane'
loc, ede = dane / 'redukcja_pilot_lokalny', dane / 'eden' / 'redukcja_pilot'
KEYS = {'SETS_WEIGHTS.csv': ('seed', 'step', 'block', 'set'), 'SETS_TOKENS.csv': ('seed', 'step', 'block', 'set', 'panel'),
        'POPULATION.csv': ('seed', 'step', 'block'), 'POPULATION_DATA.csv': ('seed', 'step', 'block', 'panel'),
        'DETECTION.csv': ('seed', 'step', 'panel'), 'HEAD_GLU.csv': ('seed', 'step', 'panel', 'anchor', 'set')}


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


out = {}
for name, key in KEYS.items():
    a = {tuple(r[k] for k in key): r for r in csv.DictReader(open(loc / name, encoding='utf-8'))}
    b = {tuple(r[k] for k in key): r for r in csv.DictReader(open(ede / name, encoding='utf-8'))}
    rec = dict(rows_local=len(a), rows_eden=len(b), missing=len(set(a) ^ set(b)), columns={})
    for col in next(iter(a.values())):
        if col in key:
            continue
        worst, nonnum_diff, nan_mismatch = 0.0, 0, 0
        for k in set(a) & set(b):
            x, y = a[k].get(col), b[k].get(col)
            fx, fy = num(x), num(y)
            if fx is None or fy is None:
                if (fx is None) != (fy is None) or (fx is None and x != y):
                    nan_mismatch += 1
                continue
            worst = max(worst, abs(fx - fy))
        rec['columns'][col] = dict(max_abs_diff=worst, missing_or_text_mismatch=nan_mismatch)
    rec['worst_column'] = max(rec['columns'].items(), key=lambda kv: kv[1]['max_abs_diff'])[0] if rec['columns'] else None
    rec['worst_value'] = rec['columns'][rec['worst_column']]['max_abs_diff'] if rec['worst_column'] else None
    out[name] = rec
hl = json.loads((loc / 'HYPOTHESES_AND_SUMMARY.json').read_text(encoding='utf-8'))
he = json.loads((ede / 'HYPOTHESES_AND_SUMMARY.json').read_text(encoding='utf-8'))
out['decisions'] = {p: dict(local=hl['decisions'][p]['verdict'], eden=he['decisions'][p]['verdict'],
                            local_counts=hl['decisions'][p]['counts'], eden_counts=he['decisions'][p]['counts'])
                    for p in hl['decisions']}
out['timing_equal'] = hl['timing_counts'] == he['timing_counts']
(dane / 'eden' / 'POROWNANIE_EDEN_LOKALNIE.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
for name, rec in out.items():
    if name.endswith('.csv'):
        print(f'{name:22s} rows {rec["rows_local"]}/{rec["rows_eden"]} missing {rec["missing"]} worst {rec["worst_column"]} = {rec["worst_value"]:.3g}',
              'nan-mismatch cols:', [c for c, v in rec['columns'].items() if v['missing_or_text_mismatch']])
print('decisions', out['decisions'], 'timing equal', out['timing_equal'])
