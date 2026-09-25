#!/usr/bin/env python3
"""Podsumowanie kontynuacji treningu od model_0000245.pth z dodanymi klatkami (czesc 7c WYNIKI.md).

Czyta REPORT.json pobrane z Edena (wyniki/dobor_klatek/{trening,trening150}/raporty/seed{S}_{ramie}/REPORT.json),
liczy per ramie srednia, rozrzut (SD, n-1) i zakres po ziarnach oraz roznice sparowane (to samo ziarno) wzgledem
ramienia bez_dodania. Zapisuje wyniki/dobor_klatek/PODSUMOWANIE_KONTYNUACJA.json i rysunek KONTYNUACJA.png.

Kampanie: trening = pilot z 17 dodanymi klatkami (top-6 na kotwice, 100 aktualizacji; przerwany na zyczenie
uzytkownika po 4 ramionach), trening150 = 134 dodane klatki (top-50 na kotwice, 103 aktualizacje, 3 ziarna).

    py -3.11 podsumuj_kontynuacje.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAZA = HERE / 'wyniki' / 'dobor_klatek'
KAMPANIE = {'trening': 'pilot 17 klatek (top-6 na kotwice, 100 aktualizacji)',
            'trening150': '134 klatki (top-50 na kotwice, 103 aktualizacje)'}
RAMIONA = ['bez_dodania', 'koopman_fastgan', 'losowe']
METRYKI = {'mAP50': 'val/mAP_50', 'mAP50_95': 'val/mAP_50_95', 'F1': 'val/F1', 'mAR': 'val/mAR',
           'precision': 'val/precision', 'recall': 'val/recall'}


def wczytaj(kamp: str) -> dict:
    out = {}
    for p in sorted((BAZA / kamp / 'raporty').glob('seed*_*/REPORT.json')):
        r = json.loads(p.read_text())
        v = r['validation'][0]
        out[(int(r['seed']), r['ramie'])] = dict({k: float(v[m]) for k, m in METRYKI.items()},
                                                 klatek_train=r['klatek_train'], dodane=r.get('dodane'),
                                                 global_step=r['global_step'], sekundy=r['sekundy'])
    return out


def statystyki(wyn: dict) -> dict:
    ziarna = sorted({s for s, _ in wyn})
    res = {'ziarna': ziarna, 'ramiona': {}, 'roznice_vs_bez_dodania': {}}
    for ramie in RAMIONA:
        rows = {s: wyn[(s, ramie)] for s in ziarna if (s, ramie) in wyn}
        if not rows:
            continue
        d = {'n_ziaren': len(rows), 'ziarna': sorted(rows), 'klatek_train': rows[min(rows)]['klatek_train'],
             'dodane': rows[min(rows)]['dodane'], 'per_ziarno': {str(s): {k: r[k] for k in METRYKI} for s, r in rows.items()}}
        for k in METRYKI:
            x = np.array([r[k] for r in rows.values()])
            d[k] = dict(srednia=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None,
                        min=float(x.min()), max=float(x.max()))
        res['ramiona'][ramie] = d
    for ramie in RAMIONA[1:]:
        pary = [s for s in ziarna if (s, ramie) in wyn and (s, 'bez_dodania') in wyn]
        if not pary:
            continue
        d = {'n_par': len(pary), 'ziarna': pary}
        for k in METRYKI:
            dx = np.array([wyn[(s, ramie)][k] - wyn[(s, 'bez_dodania')][k] for s in pary])
            d[k] = dict(srednia=float(dx.mean()), sd=float(dx.std(ddof=1)) if len(dx) > 1 else None,
                        per_ziarno={str(s): float(v) for s, v in zip(pary, dx)})
        res['roznice_vs_bez_dodania'][ramie] = d
    return res


def rysuj(wszystko: dict, sciezka: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kamp = [k for k in KAMPANIE if wszystko[k]['wyniki']]
    fig, axes = plt.subplots(len(kamp), 3, figsize=(13, 3.6 * len(kamp)), squeeze=False)
    for i, k in enumerate(kamp):
        wyn = wszystko[k]['wyniki']
        ziarna = sorted({s for s, _ in wyn})
        for j, (m, tyt) in enumerate([('mAP50', 'mAP@0.5 (walidacja)'), ('mAP50_95', 'mAP@0.5:0.95'), ('F1', 'F1')]):
            ax = axes[i, j]
            for s in ziarna:
                xs = [a for a, r in enumerate(RAMIONA) if (s, r) in wyn]
                ys = [wyn[(s, RAMIONA[a])][m] for a in xs]
                ax.plot(xs, ys, 'o-', label=f'ziarno {s}', alpha=.85)
            for a, r in enumerate(RAMIONA):
                ys = [wyn[(s, r)][m] for s in ziarna if (s, r) in wyn]
                if ys:
                    ax.hlines(np.mean(ys), a - .25, a + .25, color='k', lw=2)
            ax.set_xticks(range(len(RAMIONA)))
            ax.set_xticklabels(['bez dodania', 'dobrane\n(Koopman+FastGAN)', 'losowe'])
            ax.set_title(f'{KAMPANIE[k].split(" (")[0]}: {tyt}', fontsize=10)
            ax.grid(alpha=.3)
            if j == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Kontynuacja od kroku 245 (1 epoka): wpływ dodanych klatek; kreska = średnia po ziarnach', fontsize=11)
    fig.tight_layout()
    fig.savefig(sciezka, dpi=120)
    plt.close(fig)


def main() -> None:
    wszystko = {}
    for k, opis in KAMPANIE.items():
        wyn = wczytaj(k)
        wszystko[k] = {'opis': opis, 'wyniki': wyn, 'statystyki': statystyki(wyn) if wyn else None}
        print(f'== {k}: {opis}; ramion policzonych: {len(wyn)}')
        st = wszystko[k]['statystyki']
        if not st:
            continue
        for ramie, d in st['ramiona'].items():
            per = ' '.join(f'{s}:{v["mAP50"]:.4f}' for s, v in d['per_ziarno'].items())
            sd = d['mAP50']['sd']
            print(f'  {ramie:16s} n={d["n_ziaren"]} mAP50 {d["mAP50"]["srednia"]:.4f} +- {sd if sd is None else round(sd, 4)}'
                  f'  mAP50-95 {d["mAP50_95"]["srednia"]:.4f}  F1 {d["F1"]["srednia"]:.4f}   [{per}]')
        for ramie, d in st['roznice_vs_bez_dodania'].items():
            print(f'  d({ramie} - bez_dodania) n={d["n_par"]}: mAP50 {d["mAP50"]["srednia"]:+.4f} '
                  f'(per ziarno {d["mAP50"]["per_ziarno"]}), mAP50-95 {d["mAP50_95"]["srednia"]:+.4f}, F1 {d["F1"]["srednia"]:+.4f}')
    out = {k: {'opis': v['opis'], 'statystyki': v['statystyki'],
               'wyniki': {f'seed{s}_{r}': w for (s, r), w in v['wyniki'].items()}} for k, v in wszystko.items()}
    (BAZA / 'PODSUMOWANIE_KONTYNUACJA.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    rysuj(wszystko, BAZA / 'KONTYNUACJA.png')
    print('zapisano', BAZA / 'PODSUMOWANIE_KONTYNUACJA.json', 'i', BAZA / 'KONTYNUACJA.png')


if __name__ == '__main__':
    main()
