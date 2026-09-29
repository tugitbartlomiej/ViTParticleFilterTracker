"""Zbiorczo T2 z Edena (ziarna 47–56): czy równanie z siłą paczki przekracza 80 na każdym ziarnie.

Obala: którekolwiek ziarno z `t1+paczka` albo z równaniem prostym ≤ 80, placebo wyraźnie powyżej `t1`
(przeciek), albo ruch nagrania co 10 kroków niezgodny z zapisem Edena v2 tego samego ziarna (inne uczenie niż
w pętli). Wejście: `wyniki_eden/T2_s*.json` (z `rownanie_eden.slurm`), wyjście: `wyniki_eden/ZBIORCZO.json`.
"""
from __future__ import annotations

import json

import numpy as np

import dane

WAR = ('t1', 't1+paczka', 't1+paczka_placebo', 't1+jadro', 't1+paczka+ped', 'same_sily')
PROSTE = ('rownanie_proste', 'a', 'b', 'historia_av', 'a_samo', 'mediana_po_krokach', 'p10_po_krokach',
          'mediana_po_sondach', 'najgorsza_sonda')


def ruch_eden(s: int):
    """Norma i korelacja kolejnych przyrostów co 10 kroków w zapisie Edena v2, kroki 1010–2990, sondy z narzędziem."""
    K, Z = dane.eden(s)
    Z = Z[(K >= 1010) & (K <= 2990)]
    Z = Z[:, np.abs(Z).sum((0, 2)) > 0]
    D = np.diff(Z, axis=0)
    return dict(norma=round(float(np.linalg.norm(D, axis=-1).mean()), 4),
                korelacja=round(float((D[1:] * D[:-1]).sum() / np.sqrt((D[1:] ** 2).sum() * (D[:-1] ** 2).sum())), 4))


def main():
    pliki = sorted((dane.HERE / 'wyniki_eden').glob('T2_s*.json'))
    W = [json.loads(p.read_text(encoding='utf-8')) for p in pliki]
    wiersze = []
    for w in W:
        r = dict(ziarno=w['ziarno'], **{k: w['umiejetnosc_test'][k] for k in WAR}, **{k: w[k] for k in PROSTE},
                 ruch_co_10=w['ruch_co_10'], ruch_co_10_eden=ruch_eden(w['ziarno']), sekundy=w['sekundy'])
        wiersze.append(r)
    def opis(k, zrodlo=None):
        v = np.array([(r[zrodlo][k] if zrodlo else r[k]) for r in wiersze], float)
        return dict(mediana=round(float(np.median(v)), 3), min=round(float(v.min()), 3), max=round(float(v.max()), 3))
    zb = dict(ziarna=[r['ziarno'] for r in wiersze],
              **{k: opis(k) for k in WAR + PROSTE},
              placebo_minus_t1_max=round(max(abs(r['t1+paczka_placebo'] - r['t1']) for r in wiersze), 3),
              ziaren_paczka_powyzej_80=sum(r['t1+paczka'] > 80 for r in wiersze),
              ziaren_proste_powyzej_80=sum(r['rownanie_proste'] > 80 for r in wiersze),
              ruch_co_10_norma=opis('norma', 'ruch_co_10'), ruch_co_10_norma_eden=opis('norma', 'ruch_co_10_eden'),
              ruch_co_10_korelacja=opis('korelacja', 'ruch_co_10'),
              ruch_co_10_korelacja_eden=opis('korelacja', 'ruch_co_10_eden'))
    (dane.HERE / 'wyniki_eden' / 'ZBIORCZO.json').write_text(
        json.dumps(dict(zbiorczo=zb, ziarna=wiersze), ensure_ascii=False, indent=1), encoding='utf-8')
    print('| ziarno | t1 | +paczka | placebo | +pęd | same siły | proste | a | b | p10 kroków | najgorsza sonda '
          '| ruch 10 (nagr./Eden) | korel. 10 (nagr./Eden) |')
    print('|' + '---|' * 13)
    for r in wiersze:
        print(f"| {r['ziarno']} | {r['t1']:.1f} | {r['t1+paczka']:.1f} | {r['t1+paczka_placebo']:.1f} "
              f"| {r['t1+paczka+ped']:.1f} | {r['same_sily']:.1f} | {r['rownanie_proste']:.1f} | {r['a']:.3f} "
              f"| {r['b']:.3f} | {r['p10_po_krokach']:.1f} | {r['najgorsza_sonda']:.1f} "
              f"| {r['ruch_co_10']['norma']:.3f} / {r['ruch_co_10_eden']['norma']:.3f} "
              f"| {r['ruch_co_10']['korelacja']:.3f} / {r['ruch_co_10_eden']['korelacja']:.3f} |")
    print(json.dumps(zb, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
