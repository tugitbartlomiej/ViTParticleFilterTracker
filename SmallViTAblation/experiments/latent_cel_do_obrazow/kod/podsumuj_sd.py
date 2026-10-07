"""Tabele etapów 5-9 (SD prowadzone, korekta naturalna, zakresy naturalne z marginesami) z plików ETAP*.json."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[3] / 'outputs' / 'latent_cel_do_obrazow'


def rd(n):
    p = OUT / n
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def sd(nazwa):
    d = rd(nazwa)
    if not d:
        return
    print(f'\n## {nazwa}')
    print('| wariant | runda | cos do celu | ułamek celu α | zmiana latentu / ||z|| | IoU (etykieta celu) | pewność | strata | n |')
    print('|---|---|---|---|---|---|---|---|---|')
    w = d['wyniki']
    for war in dict.fromkeys(x['wariant'] for x in w):
        for r in sorted({x['runda'] for x in w if x['wariant'] == war}):
            v = [x for x in w if x['wariant'] == war and x['runda'] == r]
            m = lambda k: np.nanmean([np.nan if x[k] is None else x[k] for x in v])
            print(f'| {war} | {r} | {m("cos_do_korekty"):+.3f} | {m("alfa"):.4f} | {m("zmiana_wzgl"):.3f} | {m("iou_top"):.3f} | '
                  f'{m("pewnosc_top"):.3f} | {m("strata"):.3f} | {len(v)} |')
    o = [x['obraz_wyjsciowy'] for x in w if x['runda'] == 1]
    print(f'| obraz wyjściowy | - | - | - | - | {np.mean([x["iou_top"] or 0 for x in o]):.3f} | {np.mean([x["pewnosc_top"] for x in o]):.3f} | '
          f'{np.mean([x["strata"] for x in o]):.3f} | |')


def zakresy(nazwa):
    d = rd(nazwa)
    if not d or 'podsumowanie' not in d:
        return
    print(f'\n## {nazwa}  (szerokość kanału: ckpt {d["szerokosc_ckpt"]:.2f}, naturalne {d["szerokosc_przedzialu"]})')
    print('prawdziwe tokeny:', {k: round(v, 2) for k, v in d['prawdziwe']['nienat'].items()})
    print('| przedział | IoU | IoU >= 0,5 | pewność | poza | maha | ||Δz|| | zmiana W2 |')
    print('|---|---|---|---|---|---|---|---|')
    for k, v in d['podsumowanie'].items():
        print(f'| {k} | {v["iou"]:.3f} | {v["iou_powyzej_05"]}/{v["n"]} | {v["pewnosc"]:.3f} | {v["poza"]:.2f} | {v["maha"]:.1f} | '
              f'{v["norma_dz"]:.0f} | {v["W2"]:.2f} |')


sd('ETAP5_SD_PROWADZONE.json')
n = rd('ETAP6_KOREKTA_NATURALNA.json')
if n and 'podsumowanie' in n:
    print('\n## ETAP6_KOREKTA_NATURALNA.json')
    print('| r, λ | IoU | IoU >= 0,5 | pewność | poza | maha | ||Δz|| | zmiana W2 |')
    print('|---|---|---|---|---|---|---|---|')
    for k, v in n['podsumowanie'].items():
        print(f'| {k} | {v["iou"]:.3f} | {v["iou_powyzej_05"]}/{v["n"]} | {v["pewnosc"]:.3f} | {v["poza"]:.2f} | {v["maha"]:.1f} | '
              f'{v["norma_dz"]:.0f} | {v["W2"]:.2f} |')
sd('ETAP7_SD_NATURALNY.json')
zakresy('ETAP8_ZAKRESY_NATURALNE.json')
zakresy('ETAP8_ZAKRESY_NATURALNE_marginesy.json')
for p in sorted(OUT.glob('ETAP9_SD_*.json')):
    sd(p.name)
r = rd('ETAP5_NORMALIZACJA.json')
if r:
    print('\n## normalizacja w jądrze fc2 (etap 5)')
    for k, v in r.items():
        if isinstance(v, dict) and 'sztuczny' in v:
            print(k, 'poza', round(v['sztuczny']['poza'], 1), '->', round(v['znormalizowany']['poza'], 1), 'IoU', round(v['sztuczny']['iou'], 3),
                  '->', round(v['znormalizowany']['iou'], 3), 'W2', round(v['W2_zmiana_wzgl'], 2))
