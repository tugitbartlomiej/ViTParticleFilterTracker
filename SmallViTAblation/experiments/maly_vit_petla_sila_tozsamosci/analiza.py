#!/usr/bin/env python3
"""Analiza kampanii pętli małego ViT, wersja 3 (PLAN.md, kryteria ustalone przed wynikiem).

- Główne: pole pod krzywą mAP@0,5 w obrotach 3–9 (średnia ocen co 100 kroków po rozgrzewce): `tozsamosc` > `losowe`,
  > `kolejne` i > `bez` w ≥ 8 z 10 ziaren.
- Rola prognozy: `tozsamosc` > `current` w ≥ 8 z 10 ziaren.
- Prognoza w pętli (ramię `tozsamosc`): błąd siły / brak zmiany < 1 w ≥ 8 z 10 ziaren w ≥ 4 z 6 obrotów 4–9
  (sprawdzenie prognozy z poprzedniego obrotu; model wybrany testem wstecznym).
- FastGAN w pętli: w każdym obrocie mediana odpornej części zmiany siły ≥ 0,5 i zgodność znaku ≥ 0,8 (6 kotwic × ziarna).

    <python> analiza.py [katalog ...]  ->  <pierwszy katalog>/ANALIZA.json, ANALIZA.png   (domyślnie wyniki_eden/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAMIONA = ('tozsamosc', 'current', 'losowe', 'kolejne', 'bez')
KONTROLE = ('current', 'losowe', 'kolejne', 'bez')


def jz(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main() -> int:
    katalogi = [HERE / k for k in sys.argv[1:]] or [HERE / 'wyniki_eden']
    W = katalogi[0]
    wyn, krzywe = dict(ziarna={}, katalogi=[str(k) for k in katalogi]), {}
    prog_obr = {r: {k: [] for k in range(4, 10)} for r in RAMIONA}           # błąd / brak zmiany, model wybrany
    fg_obr = {k: dict(czesc=[], odporna=[], znak=[], wspolne90=[]) for k in range(3, 10)}
    modele = {}
    for d in sorted(x for k in katalogi for x in k.glob('petla_s*')):
        if not d.is_dir():
            continue
        s = d.name.split('_s')[1]
        z = {}
        for r in RAMIONA:
            f = d / r / 'KONIEC.json'
            if not f.exists():
                continue
            kon = jz(f)
            oc = [o for o in kon['oceny'] if o['ramie'] == r]
            obr = []
            for k in range(3, 10):
                o = d / r / f'k{k:02d}' / 'OBROT.json'
                if not o.exists():
                    continue
                x = jz(o)
                sp = x.get('sprawdzenie_poprz_obrotu')
                if sp is not None:
                    prog_obr[r][k].append(sp['wybrany']['sondy'])
                modele.setdefault(r, []).append(x['prognoza']['model'])
                if r == 'tozsamosc':
                    for a in x.get('fastgan', []):
                        if 'czesc_zmiany' in a:
                            fg_obr[k]['czesc'].append(a['czesc_zmiany']); fg_obr[k]['odporna'].append(a['czesc_zmiany_odporna'])
                            fg_obr[k]['znak'].append(float(np.sign(a['zmiana_osiagnieta_odporna']) == np.sign(a['przewidziana_zmiana'])))
                            fg_obr[k]['wspolne90'].append(a['wspolne_top90_ze_startem'])
                obr.append(dict(k=k, mAP50=x['ocena']['mAP50'], model=x['prognoza']['model'],
                                sprawdzenie=None if sp is None else {m: sp[m]['sondy'] for m in ('koopman', 'langevin', 'wybrany')}))
            z[r] = dict(pole=float(np.mean([o['mAP50'] for o in oc])), koncowy=oc[-1]['mAP50'], dodane=len(kon['dodane']),
                        dodane_w_obrocie={k: sum(1 for x in kon['dodane'] if x['obrot'] == k) for k in range(3, 10)},
                        krzywa=[(o['krok'], o['mAP50']) for o in oc], obroty=obr,
                        sprawdzenie_koncowe=kon.get('sprawdzenie_koncowe', {}).get('wybrany'))
            krzywe.setdefault(r, []).append(np.array([o['mAP50'] for o in oc]))
        if all(r in z for r in RAMIONA):
            for c in KONTROLE:
                z[f'roznica_tozsamosc_{c}'] = z['tozsamosc']['pole'] - z[c]['pole']
            z['wygrywa_glowne'] = bool(all(z[f'roznica_tozsamosc_{c}'] > 0 for c in ('losowe', 'kolejne', 'bez')))
        wyn['ziarna'][s] = z
    pelne = [z for z in wyn['ziarna'].values() if 'wygrywa_glowne' in z]
    pod = dict(ziaren=len(pelne))
    if pelne:
        pod['pole_srednie'] = {r: float(np.mean([z[r]['pole'] for z in pelne])) for r in RAMIONA}
        pod['dodane_srednio'] = {r: float(np.mean([z[r]['dodane'] for z in pelne])) for r in RAMIONA}   # limit list wyszukiwania
        pod['glowne_wygrane'] = int(sum(z['wygrywa_glowne'] for z in pelne))
        pod['kryterium_glowne'] = bool(len(pelne) == 10 and pod['glowne_wygrane'] >= 8)
        for c in KONTROLE:
            x = [z[f'roznica_tozsamosc_{c}'] for z in pelne]
            pod[f'tozsamosc_minus_{c}'] = dict(srednia=float(np.mean(x)), lepsza_w=int(sum(v > 0 for v in x)))
            if len(x) >= 5:
                from scipy.stats import binomtest, wilcoxon
                pod[f'tozsamosc_minus_{c}'].update(wilcoxon_p=float(wilcoxon(x).pvalue),
                                                   znak_p=float(binomtest(sum(v > 0 for v in x), len(x)).pvalue))
        pod['kryterium_rola_prognozy'] = bool(len(pelne) == 10 and pod['tozsamosc_minus_current']['lepsza_w'] >= 8)
    pr = {}
    for r in RAMIONA:
        pr[r] = {k: dict(mediana=float(np.median(v)), ponizej_1=int(sum(e < 1 for e in v)), ziaren=len(v))
                 for k, v in prog_obr[r].items() if v}
    pod['prognoza_w_petli'] = pr
    ok_obr = sum(1 for k, v in pr['tozsamosc'].items() if v['ziaren'] == 10 and v['ponizej_1'] >= 8)
    pod['kryterium_prognozy'] = dict(obrotow_spelnionych=ok_obr, spelnione=bool(ok_obr >= 4))
    pod['model_wybierany'] = {r: {m: v.count(m) for m in set(v)} for r, v in modele.items()}
    fg = {k: dict(czesc_mediana=float(np.median(v['czesc'])), odporna_mediana=float(np.median(v['odporna'])),
                  znak=float(np.mean(v['znak'])), wspolne_top90_ze_startem=float(np.mean(v['wspolne90'])), kotwic=len(v['czesc']))
          for k, v in fg_obr.items() if v['czesc']}
    pod['fastgan'] = fg
    pod['kryterium_fastgan'] = bool(fg and all(v['odporna_mediana'] >= .5 and v['znak'] >= .8 for v in fg.values()))
    wyn['podsumowanie'] = pod
    (W / 'ANALIZA.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r, c in zip(RAMIONA, ('tab:red', 'tab:orange', 'tab:blue', 'tab:green', 'tab:gray')):
            if r in krzywe:
                n = min(len(x) for x in krzywe[r]); A = np.stack([x[:n] for x in krzywe[r]])
                kroki = 1000 + 100 * np.arange(1, n + 1)
                ax.plot(kroki, A.mean(0), color=c, label=f'{r} (ziaren {len(A)})')
                ax.fill_between(kroki, A.min(0), A.max(0), color=c, alpha=.10)
        ax.set_xlabel('krok uczenia'); ax.set_ylabel('mAP@0,5 (próbka val 3000)')
        ax.set_title('Mały ViT, wersja 3: dobór klatek według prognozy siły tożsamości')
        ax.legend(); fig.tight_layout(); fig.savefig(W / 'ANALIZA.png', dpi=130)
    except Exception as e:  # wykres pomocniczy
        print('wykres:', e)
    print(json.dumps(pod, indent=1, ensure_ascii=False))
    for s, z in wyn['ziarna'].items():
        print(s, {r: round(z[r]['pole'], 4) for r in RAMIONA if r in z})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
