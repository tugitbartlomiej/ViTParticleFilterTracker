#!/usr/bin/env python3
"""Analiza kampanii pętli małego ViT, wersja 2 (PLAN.md).

Kryterium (ustalone przed wynikiem): pole pod krzywą mAP@0,5 w obrotach 3–9 = średnia ocen co 100 kroków po
rozgrzewce (2 obroty); `koopman` > `losowe` i > `bez` w ≥ 8 z 10 ziaren. Wtórnie: domknięcie tożsamości po obrotach, jakość
prognozy (test wsteczny względem braku zmiany), reszty FastGAN, podobieństwo wybranych klatek.

    py -3.11 analiza.py [katalog ...]  ->  <pierwszy katalog>/ANALIZA.json, ANALIZA.png
    (domyślnie wyniki/; kilka katalogów = ziarna z nich razem, np. wyniki wyniki_eden)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE / 'wyniki'
RAMIONA = ('koopman', 'losowe', 'bez')


def jz(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main() -> int:
    import sys
    global W
    katalogi = [HERE / k for k in sys.argv[1:]] or [W]
    W = katalogi[0]
    wyn, krzywe = dict(ziarna={}, katalogi=[str(k) for k in katalogi]), {}
    for d in sorted(x for k in katalogi for x in k.glob('petla_s*')):
        if not d.is_dir():
            continue
        s = d.name.split('_s')[1]
        z = {}
        for r in RAMIONA:
            f = d / r / 'KONIEC.json'
            if not f.exists():
                continue
            oc = [o for o in jz(f)['oceny'] if o['ramie'] == r]
            obr = []
            for k in range(3, 10):
                o = d / r / f'k{k:02d}' / 'OBROT.json'
                if o.exists():
                    x = jz(o)
                    obr.append(dict(k=k, mAP50=x['ocena']['mAP50'], domkniecie=x['tozsamosc_po']['domkniecie_srednie'],
                                    rdzen=x['tozsamosc_po']['rdzen'], pelny=x['tozsamosc_po']['pelny'],
                                    prognoza=x.get('prognoza'), fastgan=x.get('fastgan'),
                                    podobienstwo=float(np.mean([w['podobienstwo'] for w in x['wybrane']]))
                                    if x.get('wybrane') and 'podobienstwo' in x['wybrane'][0] else None))
            z[r] = dict(pole=float(np.mean([o['mAP50'] for o in oc])), koncowy=oc[-1]['mAP50'],
                        krzywa=[(o['krok'], o['mAP50']) for o in oc], obroty=obr)
            krzywe.setdefault(r, []).append(np.array([o['mAP50'] for o in oc]))
        if len(z) == 3:
            z['koopman_wygrywa'] = bool(z['koopman']['pole'] > z['losowe']['pole'] and z['koopman']['pole'] > z['bez']['pole'])
            z['roznica_koopman_losowe'] = z['koopman']['pole'] - z['losowe']['pole']
            z['roznica_koopman_bez'] = z['koopman']['pole'] - z['bez']['pole']
            z['roznica_losowe_bez'] = z['losowe']['pole'] - z['bez']['pole']
        wyn['ziarna'][s] = z
    pelne = [z for z in wyn['ziarna'].values() if 'koopman_wygrywa' in z]
    if pelne:
        wyn['podsumowanie'] = dict(
            ziaren=len(pelne), koopman_wygrywa=int(sum(z['koopman_wygrywa'] for z in pelne)),
            kryterium_spelnione=bool(len(pelne) == 10 and sum(z['koopman_wygrywa'] for z in pelne) >= 8),
            pole_srednie={r: float(np.mean([z[r]['pole'] for z in pelne])) for r in RAMIONA},
            roznica_koopman_losowe=float(np.mean([z['roznica_koopman_losowe'] for z in pelne])),
            roznica_koopman_bez=float(np.mean([z['roznica_koopman_bez'] for z in pelne])),
            roznica_losowe_bez=float(np.mean([z['roznica_losowe_bez'] for z in pelne])),
            koopman_lepszy_od_losowego=int(sum(z['roznica_koopman_losowe'] > 0 for z in pelne)),
            koopman_lepszy_od_bez=int(sum(z['roznica_koopman_bez'] > 0 for z in pelne)))
        if len(pelne) >= 5:
            from scipy.stats import wilcoxon, binomtest
            for k in ('roznica_koopman_losowe', 'roznica_koopman_bez', 'roznica_losowe_bez'):
                x = [z[k] for z in pelne]
                wyn['podsumowanie'][k + '_wilcoxon_p'] = float(wilcoxon(x).pvalue)
                wyn['podsumowanie'][k + '_znak_p'] = float(binomtest(sum(v > 0 for v in x), len(x)).pvalue)
    (W / 'ANALIZA.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r, c in zip(RAMIONA, ('tab:red', 'tab:blue', 'tab:gray')):
            if r in krzywe:
                n = min(len(x) for x in krzywe[r]); A = np.stack([x[:n] for x in krzywe[r]])
                kroki = 1000 + 100 * np.arange(1, n + 1)
                ax.plot(kroki, A.mean(0), color=c, label=f'{r} (ziaren {len(A)})')
                ax.fill_between(kroki, A.min(0), A.max(0), color=c, alpha=.12)
        ax.set_xlabel('krok uczenia'); ax.set_ylabel('mAP@0,5 (próbka val 3000)')
        ax.set_title('Mały ViT, wersja 2 (wektor tożsamości): dobór klatek po rozgrzewce')
        ax.legend(); fig.tight_layout(); fig.savefig(W / 'ANALIZA.png', dpi=130)
    except Exception as e:  # wykres pomocniczy
        print('wykres:', e)
    print(json.dumps(wyn.get('podsumowanie', {}), indent=1, ensure_ascii=False))
    for s, z in wyn['ziarna'].items():
        print(s, {r: round(z[r]['pole'], 4) for r in RAMIONA if r in z})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
