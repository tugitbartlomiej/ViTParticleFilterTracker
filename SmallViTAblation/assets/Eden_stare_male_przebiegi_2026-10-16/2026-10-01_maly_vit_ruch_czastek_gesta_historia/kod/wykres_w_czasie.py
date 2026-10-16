"""AUC narzędzie/tło w czasie uczenia z gotowego REPORT.json (bez 22 GB tokenów), z fazami uczenia.

    python wykres_w_czasie.py <katalog analizy z REPORT.json>  ->  AUC_W_CZASIE.png, FAZY.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FAZY = [(0, 20), (20, 100), (100, 300), (300, 1000)]
SERIE = [('norma_po_korekcie_tla', 'Norma ruchu po odjęciu dryfu tła', '-', 'tab:gray'),
         ('kierunek_sredni', 'Średni kierunek ruchu (ze znakiem)', '-', 'tab:red'),
         ('svd_8', '8 kierunków SVD z filmów uczących', '-', 'tab:blue'),
         ('losowe_8', '8 losowych kierunków', '--', 'tab:orange')]


def main(katalog):
    d = Path(katalog)
    r = json.loads((d / 'REPORT.json').read_text(encoding='utf-8'))
    st = np.asarray(r['steps'], float)
    mid = (st[1:] + st[:-1]) / 2
    fazy = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, klucz, tytul in zip(axes, ['mask_1px', 'mask_25pct_same_directions', 'mask_1px_directions_fitted_25pct'],
                                ['Maska od 1 piksela', 'Maska 25 % (te same kierunki)', 'Maska 1 px, kierunki z maski 25 %']):
        w = r['results'][klucz]['auc_w_czasie_srednia_obrazow']
        okno = max(1, len(mid) // 50)
        xs = np.convolve(mid, np.ones(okno) / okno, mode='valid')
        for n, lab, styl, kol in SERIE:
            ax.plot(xs, np.convolve(w[n], np.ones(okno) / okno, mode='valid'), styl, color=kol, lw=1.1, label=lab)
        for a, _ in FAZY[1:]:
            ax.axvline(a, color='black', lw=.5, alpha=.4)
        ax.axhline(.5, ls=':', color='black')
        ax.set(xlabel='Krok uczenia', title=tytul + (f' (średnia ruchoma {okno})' if okno > 1 else ''))
        fazy[klucz] = {f'{a}-{b}': {n: float(np.mean(np.asarray(w[n])[(mid > a) & (mid <= b)])) for n in w} for a, b in FAZY}
    axes[0].set_ylabel('ROC AUC, średnia 4 odłożonych obrazów')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f'Ruch tokenów narzędzia a tła w czasie uczenia: {len(st)} stanów, kroki {int(st[0])}..{int(st[-1])}')
    fig.tight_layout()
    fig.savefig(d / 'AUC_W_CZASIE.png', dpi=150)
    plt.close(fig)
    (d / 'FAZY.json').write_text(json.dumps(fazy, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(fazy['mask_1px'], indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
