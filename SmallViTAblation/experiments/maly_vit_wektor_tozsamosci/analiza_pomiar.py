#!/usr/bin/env python3
"""Pomiar wersji 2 (PLAN.md, „Kryteria”): czy prognoza wektora tożsamości z poprzedniego obrotu bije „brak zmiany”,
stabilność wektora, porównanie z wersją 1 na tych samych ziarnach.

    py -3.11 analiza_pomiar.py  ->  wyniki_eden/POMIAR.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W2 = HERE / 'wyniki_eden'
W1 = HERE.parent / 'maly_vit_od_zera' / 'wyniki_eden'
RAMIONA = ('koopman', 'losowe', 'bez')


def jz(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main() -> int:
    ziarna = sorted(d for d in W2.glob('petla_s*') if all((d / r / 'KONIEC.json').exists() for r in RAMIONA))
    wyn = dict(ziaren=len(ziarna), ramiona={})
    for r in RAMIONA:
        spr = {}                                                      # obrót -> model -> lista po ziarnach
        cosy = {}
        for d in ziarna:
            for k in range(3, 10):
                p = jz(d / r / f'k{k:02d}' / 'OBROT.json')['pomiar']
                if p.get('wektor_cos_poprz_obrot') is not None:
                    cosy.setdefault(k, []).append(p['wektor_cos_poprz_obrot'])
                s = p.get('sprawdzenie_poprz_obrotu')
                if s:
                    for m in ('koopman', 'langevin'):
                        spr.setdefault(str(k), {}).setdefault(m, []).append(s[m])
            s = jz(d / r / 'KONIEC.json')['pomiar_koncowy'].get('sprawdzenie_poprz_obrotu')
            if s:
                for m in ('koopman', 'langevin'):
                    spr.setdefault('koniec', {}).setdefault(m, []).append(s[m])
        tab = {}
        for k, mm in spr.items():
            tab[k] = {m: dict(wektor_mediana=float(np.median([x['wektor'] for x in v])),
                              wektor_ponizej_1=int(sum(x['wektor'] < 1 for x in v)),
                              sondy_mediana=float(np.median([x['sondy'] for x in v])),
                              znak_mediana=float(np.median([x['znak'] for x in v])), n=len(v)) for m, v in mm.items()}
        kryt = {m: sum(1 for k in map(str, range(4, 10)) if k in tab and tab[k][m]['wektor_mediana'] < 1
                       and tab[k][m]['wektor_ponizej_1'] >= 8) for m in ('koopman', 'langevin')}
        wyn['ramiona'][r] = dict(sprawdzenie=tab, cos_poprz_obrot={k: float(np.median(v)) for k, v in cosy.items()},
                                 obrotow_z_trafna_prognoza=kryt,
                                 prognoza_dziala={m: bool(v >= 4) for m, v in kryt.items()})
    # porównanie z wersją 1: średnia mAP@0,5 w oknie 1600–4500 kroków, ramię koopman (i pozostałe), pary ziaren
    por = {}
    for r in RAMIONA:
        pary = []
        for d in ziarna:
            f1 = W1 / d.name / r / 'KONIEC.json'
            if not f1.exists():
                continue
            m = lambda f: float(np.mean([o['mAP50'] for o in jz(f)['oceny'] if o['ramie'] == r and 1600 <= o['krok'] <= 4500]))  # noqa: E731
            pary.append((d.name, m(f1), m(d / r / 'KONIEC.json')))
        if pary:
            dd = np.array([b - a for _, a, b in pary])
            por[r] = dict(par=len(pary), v1=float(np.mean([a for _, a, _ in pary])), v2=float(np.mean([b for _, _, b in pary])),
                          roznica=float(dd.mean()), v2_lepsza=int((dd > 0).sum()), pary=pary)
    wyn['porownanie_z_wersja_1'] = por
    (W2 / 'POMIAR.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    for r, x in wyn['ramiona'].items():
        print(r, 'obroty z trafną prognozą:', x['obrotow_z_trafna_prognoza'])
        for k, mm in x['sprawdzenie'].items():
            print('  ', k, {m: (round(v['wektor_mediana'], 3), v['wektor_ponizej_1'], round(v['znak_mediana'], 2)) for m, v in mm.items()})
        print('   cos z poprz. obrotem:', {k: round(v, 3) for k, v in x['cos_poprz_obrot'].items()})
    for r, x in por.items():
        print('v1 -> v2', r, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items() if k != 'pary'})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
