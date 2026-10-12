"""Tabela główna z METRYKI.json: horyzont h, wszystkie wielkości i okresy (sufit wyroczni, podłoga, osiągalne)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

plik = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'wyniki' / 'pelny' / 'METRYKI.json'
h = int(sys.argv[2]) if len(sys.argv) > 2 else 100
m = json.loads(plik.read_text(encoding='utf-8'))


def f(v, k):
    x = None if v is None else v.get(k)
    return '-' if x is None else f'{x:.2f}'


print(f'h = {h}; MSE: 1 - błąd/błąd prognozy bez zmian\n')
print('| wielkość | okres | sufit w=5 | sufit w=21 | sufit w=51 | podłoga w=21 | D²/σ² w=21 | trwałość dryfu w=21 '
      '| autokor. przyrostów | średnia ost. 51 | liniowy krzyżowo | liniowy krzyżowo RMSE |')
print('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for q, d in m['wielkosci'].items():
    for ok in m['meta']['okresy']:
        v5, v21, v51 = d[f'w5_h{h}'][ok], d[f'w21_h{h}'][ok], d[f'w51_h{h}'][ok]
        L = d['liniowy'][f'h{h}'][ok]
        print(f"| {q} | {ok} | {f(v5, 'sufit_mse')} | {f(v21, 'sufit_mse')} | {f(v51, 'sufit_mse')} "
              f"| {f(v21, 'podloga_drgania_mse')} | {f(v21, 'D2_do_sigma2')} | {f(v21, 'trwalosc_dryfu_cos')} "
              f"| {f(v21, 'autokorelacja_przyrostow_lag1')} | {f(L, 'srednia_ost51_mse')} "
              f"| {f(L, 'liniowy_krzyzowo_mse')} | {f(L, 'liniowy_krzyzowo_rmse')} |")
