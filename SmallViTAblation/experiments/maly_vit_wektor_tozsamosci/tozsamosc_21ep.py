#!/usr/bin/env python3
"""Tożsamość narzędzia po 21 epokach (krok 1000, koniec rozgrzewki wersji 2), metodą użytkownika na sondach.

Sondy: 12 wycinków narzędzia z ręczną maską × 16 wersji (oryginał, 3 obroty, czarne i żółte tło, skala, długość,
4 przesunięcia, przyciemnione i rozjaśnione narzędzie, perspektywa; plus `empty_black` bez narzędzia jako negatyw).
Definicje z `kod/tozsamosc.py`: neuron zapala się na sondzie, gdy aktywacja na narzędziu > średnia negatywów + 2 sd;
pełny latent = wybiórczy (negatywy ≤ 10 %) i zapalony na ≥ 30 % sond z narzędziem; rdzeń = zapalony w KAŻDEJ z 15
wersji na ≥ 75 % wycinków. Dodatkowo: te same liczby dla sieci nieuczonej (krok 0) jako odniesienie.

    py -3.11 tozsamosc_21ep.py  ->  wyniki_eden/TOZSAMOSC_21EP.json, wyniki_eden/TOZSAMOSC_21EP.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE / 'wyniki_eden'
sys.path.insert(0, str(HERE / 'kod'))
import tozsamosc as TZ  # noqa: E402

PROBE = HERE.parent / 'maly_vit_od_zera' / 'eden' / 'wejscia' / 'probe'
KROK = 1000


def main() -> int:
    man = json.loads((PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    opis = [dict(wariant=m['variant']) for m in man]
    wyn, rys = {}, {}
    for d in sorted(W.glob('petla_s*')):
        z1, z2 = np.load(d / 'wspolny' / 'k01' / 'LATENT.npz'), np.load(d / 'wspolny' / 'k02' / 'LATENT.npz')
        assert z1['kroki'][0] == 0 and z2['kroki'][-1] == KROK
        L = z2['latent'][-5:].astype(np.float32).mean(0)               # kroki 960–1000
        st, st0 = TZ.statystyki(L, obecn, opis), TZ.statystyki(z1['latent'][0].astype(np.float32), obecn, opis)
        pod = TZ.podsumowanie(st)
        sel = st['frac_neg'] <= TZ.NEG_MAX
        zw = st['zapal_warianty']                                       # (15, 768) udział wycinków z zapalonym neuronem
        mn = zw.min(0)
        pel = st['pelny']
        kolejn = pel[np.argsort(-zw[:, pel].mean(0))]
        wyn[d.name] = dict(
            krok=KROK, pelny=int(len(pel)), rdzen=int(len(st['rdzen'])),
            prawie_rdzen_50=int((sel & (mn >= .50)).sum()), prawie_rdzen_25=int((sel & (mn >= .25)).sum()),
            najlepszy_min_po_wersjach=float(mn[sel].max()) if sel.any() else 0.0,
            domkniecie_wersji=pod['domkniecie_wariantow'], domkniecie_srednie=pod['domkniecie_srednie'],
            pelny_siec_nieuczona=int(len(st0['pelny'])),
            wspolne_z_nieuczona=int(len(set(pel.tolist()) & set(st0['pelny'].tolist()))),
            neurony_pelnego=[dict(neuron=int(n), wersje={v: round(float(zw[i, n]), 3) for i, v in enumerate(TZ.WARIANTY_Z)},
                                  negatywy=round(float(st['frac_neg'][n]), 3)) for n in kolejn])
        rys[d.name] = zw[:, kolejn]
        print(f"{d.name}: pełny {len(pel)} (sieć nieuczona {len(st0['pelny'])}, wspólnych {wyn[d.name]['wspolne_z_nieuczona']}), "
              f"rdzeń {len(st['rdzen'])}, min. po wersjach ≥0,5: {wyn[d.name]['prawie_rdzen_50']}, ≥0,25: "
              f"{wyn[d.name]['prawie_rdzen_25']}, najlepszy {wyn[d.name]['najlepszy_min_po_wersjach']:.2f}", flush=True)
    dom = {v: float(np.mean([x['domkniecie_wersji'][v] for x in wyn.values()])) for v in TZ.WARIANTY_Z}
    wyn['_srednio'] = dict(domkniecie_wersji=dom, pelny=float(np.mean([x['pelny'] for x in wyn.values()])))
    (W / 'TOZSAMOSC_21EP.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    print('domknięcie wersji (średnio 10 ziaren):', {v: round(x, 2) for v, x in sorted(dom.items(), key=lambda t: -t[1])})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)
    for a, (s, M) in zip(ax.ravel(), rys.items()):
        im = a.imshow(M.T, aspect='auto', cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        a.set_xticks(range(len(TZ.WARIANTY_Z))); a.set_xticklabels(TZ.WARIANTY_Z, rotation=90, fontsize=7)
        a.set_ylabel('neuron pełnego latentu (od najsilniejszego)', fontsize=7)
        a.set_title(f'{s}: {M.shape[1]} neuronów, rdzeń {wyn[s]["rdzen"]}', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.6, label='udział wycinków, na których neuron się zapala')
    fig.suptitle('Tożsamość narzędzia po 21 epokach (krok 1000): które neurony zapalają się w której wersji sondy')
    fig.savefig(W / 'TOZSAMOSC_21EP.png', dpi=110)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
