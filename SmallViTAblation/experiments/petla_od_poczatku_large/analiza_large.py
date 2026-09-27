#!/usr/bin/env python3
"""Analiza pętli od początku dostrajania na RF-DETR LargeDeprecated (PLAN.md): mAP@0,5 całego val po obrotach dla
pętli koopman, losowe i bez w każdym ziarnie, pole pod krzywą (obroty 1–6), tempo dojścia do progów, różnice
sparowane (to samo ziarno), kryterium z PLAN.md; prognoza Koopmana (przewidziana vs zmierzona zmiana skalara
tożsamości w obrocie), pokrycie wyboru z obrazem obecnym, wynik „do nauczenia” wybranych klatek.

Wejścia: wyniki/s{S}/STAN.json, wyniki/s{S}/{petla}/kXX/REPORT.json, wyniki/s{S}/fastgan/kXX/{PROGNOZA,WYBOR}.json.
Zapis: wyniki/ANALIZA.json, wyniki/ANALIZA.png.

    py -3.11 analiza_large.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE / 'wyniki'
PETLE = ('koopman', 'losowe', 'bez')
ZIARNA = (42, 43, 44)
PROGI = (0.80, 0.85, 0.88, 0.90)


def jz(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def main() -> int:
    wyn = {}
    for s in ZIARNA:
        d0 = W / f's{s}' / 'wspolny' / 'k00' / 'REPORT.json'
        if not d0.exists():
            continue
        k0 = jz(d0)['validation'][0]['val/mAP_50']
        wyn[s] = {'k00': k0}
        for p in PETLE:
            ks = sorted(int(x.name[1:]) for x in (W / f's{s}' / p).glob('k*') if (x / 'REPORT.json').exists()) if (W / f's{s}' / p).exists() else []
            wyn[s][p] = {k: jz(W / f's{s}' / p / f'k{k:02d}' / 'REPORT.json')['validation'][0]['val/mAP_50'] for k in ks}
    raport = dict(ziarna={}, roznice={}, kryterium=None, prognozy={}, wybor={})
    for s, d in wyn.items():
        r = dict(k00=d['k00'])
        for p in PETLE:
            ks = sorted(d[p]); m = [d[p][k] for k in ks]
            r[p] = dict(obroty=ks, mAP50=m, pole=float(np.mean(m)) if m else None,
                        tempo={str(t): next((k for k, v in zip(ks, m) if v >= t), None) for t in PROGI})
        raport['ziarna'][str(s)] = r
        for a, b in (('koopman', 'losowe'), ('koopman', 'bez'), ('losowe', 'bez')):
            wsp = sorted(set(d[a]) & set(d[b]))
            raport['roznice'].setdefault(f'{a}-{b}', {})[str(s)] = dict(
                obroty=wsp, mAP50=[d[a][k] - d[b][k] for k in wsp],
                pole=float(np.mean([d[a][k] - d[b][k] for k in wsp])) if wsp else None,
                obroty_1_3=float(np.mean([d[a][k] - d[b][k] for k in wsp if k <= 3])) if any(k <= 3 for k in wsp) else None)
        for k in range(1, 7):
            f = W / f's{s}' / 'fastgan' / f'k{k:02d}'
            if (f / 'PROGNOZA.json').exists():
                raport['prognozy'].setdefault(str(s), {})[str(k)] = jz(f / 'PROGNOZA.json')
            if (f / 'WYBOR.json').exists():
                raport['wybor'].setdefault(str(s), {})[str(k)] = jz(f / 'WYBOR.json')
    pelne = [s for s in wyn if all(len(wyn[s][p]) == 6 for p in PETLE)]
    if len(pelne) == 3:
        kl = [raport['roznice']['koopman-losowe'][str(s)]['pole'] for s in pelne]
        kb = [raport['roznice']['koopman-bez'][str(s)]['pole'] for s in pelne]
        kb13 = [raport['roznice']['koopman-bez'][str(s)]['obroty_1_3'] for s in pelne]
        raport['kryterium'] = dict(pole_koopman_minus_losowe=kl, pole_koopman_minus_bez=kb, obroty_1_3_koopman_minus_bez=kb13,
                                   spelnione=bool(all(x > 0 for x in kl) and all(x > 0 for x in kb) and np.mean(kb13) >= 0.005))
    (W / 'ANALIZA.json').write_text(json.dumps(raport, indent=1, ensure_ascii=False), encoding='utf-8')
    rysuj(raport)
    for s, r in raport['ziarna'].items():
        print(f's{s} k00 {r["k00"]:.4f} | ' + ' | '.join(f'{p}: ' + ' '.join(f'{v:.4f}' for v in r[p]['mAP50']) for p in PETLE))
    print('kryterium:', raport['kryterium'])
    return 0


def rysuj(r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kol = dict(koopman='C0', losowe='C1', bez='k')
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for s, d in r['ziarna'].items():
        for p in PETLE:
            ax[0].plot([0] + d[p]['obroty'], [d['k00']] + d[p]['mAP50'], 'o-', color=kol[p], alpha=.6,
                       label=p if s == str(ZIARNA[0]) else None)
    ax[0].set_title('mAP@0,5 całego val po obrotach (RF-DETR LargeDeprecated)'); ax[0].set_xlabel('obrót (160 aktualizacji)')
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    for key, c in (('koopman-losowe', 'C0'), ('koopman-bez', 'C2')):
        for s, d in r['roznice'].get(key, {}).items():
            ax[1].plot(d['obroty'], d['mAP50'], 'o-', color=c, alpha=.6, label=key if s == str(ZIARNA[0]) else None)
    ax[1].axhline(0, color='k', lw=.8); ax[1].set_title('Różnice sparowane mAP@0,5 (to samo ziarno)')
    ax[1].set_xlabel('obrót'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(W / 'ANALIZA.png', dpi=120); plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())
