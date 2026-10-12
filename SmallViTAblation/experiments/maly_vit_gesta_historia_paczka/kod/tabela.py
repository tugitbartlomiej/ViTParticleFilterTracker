"""Tabela z POROWNANIE.json (jednego albo kilku plików): paczka x miara, horyzonty 1, 5, 10, 100."""
from __future__ import annotations

import json
import sys
from pathlib import Path

pliki = [Path(p) for p in sys.argv[1:]] or [Path(__file__).resolve().parents[1] / 'wyniki' / 'porownanie' / 'POROWNANIE.json']
paczki = {}
for p in pliki:
    if p.exists():
        paczki.update(json.loads(p.read_text(encoding='utf-8'))['paczki'])
for grupa in ('wszystkie', 'narzedzie'):
    print(f'\n== {grupa} (1 - SSE / SSE bez zmian)')
    print('| paczka | równ. h1 | h5 | h10 | h100 | OU h10 | sufit h10 | sufit h100 | podłoga h10 | D²/σ² h10 | D²/σ² h100 '
          '| acf 1 | acf 2 | acf 3 | acf 5 | acf 10 | krok wzgl. | mAP koniec |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for pk in sorted(paczki, key=int):
        w = paczki[pk][grupa]; r = w['rownanie_J5']; a = w['autokorelacja_przyrostow']
        print(f"| {pk} | {r[0]:.3f} | {r[4]:.3f} | {r[9]:.3f} | {r[10]:.3f} | {w['ou'][9]:.3f} | {w['sufit'][9]:.3f} | {w['sufit'][10]:.3f} "
              f"| {w['podloga'][9]:.3f} | {w['D2_do_sigma2'][9]:.2f} | {w['D2_do_sigma2'][10]:.2f} "
              f"| {a['1']:.2f} | {a['2']:.2f} | {a['3']:.2f} | {a['5']:.2f} | {a['10']:.2f} | {w['sredni_krok_wzgledny']:.4f} "
              f"| {paczki[pk].get('mAP50_koniec', '-')} |")
