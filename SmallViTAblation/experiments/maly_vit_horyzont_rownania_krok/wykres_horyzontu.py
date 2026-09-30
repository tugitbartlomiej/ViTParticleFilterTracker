#!/usr/bin/env python3
"""Wykres: jak spada trafność prognozy latentu tożsamości z liczbą kroków naprzód (wyniki `horyzont.py`).

    python wykres_horyzontu.py 47   ->  wyniki/HORYZONT_s47.png
Lewy panel: umiejętność (100 · (1 − SSE / SS) wobec braku zmiany), obcięta od dołu na −100 %, linia 90 %.
Prawy panel: błąd prognozy względem braku zmiany (RMSE / RMSE braku zmiany = √(1 − umiejętność/100)), skala logarytmiczna;
1 = tyle co „nic się nie zmieni”, 0,32 = umiejętność 90 %. Pasmo: rozrzut między czterema oknami startu.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ETYK = dict(A='A: równanie co krok na prawdziwych wagach\n    (górna granica, nie prognoza)',
            B='B: znane przyszłe klatki, wagi z chwili startu',
            B2='B2: jak B, suma osobnych pchnięć',
            C1='C1: bez przyszłych klatek, sam pęd',
            C2='C2: bez przyszłych klatek, pęd + średnia paczka')
KOLOR = dict(A='#2b7a3d', B='#1f5fb4', B2='#7fa6e0', C1='#c0392b', C2='#e39a2d')
DOL = -100.0


def main() -> int:
    s = int(sys.argv[1]) if len(sys.argv) > 1 else 47
    d = json.loads((HERE / 'wyniki' / f'HORYZONT_s{s}.json').read_text(encoding='utf-8'))
    H = np.array(d['H'])
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for w, r in d['warianty'].items():
        um = np.array([r['tozsamosc'][str(h)] for h in H])
        okna = np.array([r['tozsamosc_okna'][str(h)] for h in H])            # (H, okna)
        ax[0].plot(H, np.clip(um, DOL, None), '-o', ms=4, color=KOLOR[w], label=ETYK[w], lw=2 if w in ('A', 'B', 'C2') else 1.3)
        ax[0].fill_between(H, np.clip(okna.min(1), DOL, None), np.clip(okna.max(1), DOL, None), color=KOLOR[w], alpha=.12)
        bl = np.sqrt(np.clip(1 - um / 100, 1e-6, None))
        blo = np.sqrt(np.clip(1 - okna / 100, 1e-6, None))
        ax[1].plot(H, bl, '-o', ms=4, color=KOLOR[w], lw=2 if w in ('A', 'B', 'C2') else 1.3)
        ax[1].fill_between(H, blo.min(1), blo.max(1), color=KOLOR[w], alpha=.12)
        if r['horyzont_90']:
            ax[0].annotate(f'{r["horyzont_90"]}', (r['horyzont_90'], 90), textcoords='offset points', xytext=(0, 8),
                           color=KOLOR[w], ha='center', fontsize=9, fontweight='bold')
    hz = [f'{w}: {r["horyzont_90"]} kr.' if r['horyzont_90'] else f'{w}: nigdy' for w, r in d['warianty'].items()]
    ax[0].text(0.66, 0.86, 'ostatnie H z umiejętnością ≥ 90 %\n' + '\n'.join(hz), transform=ax[0].transAxes, ha='right',
               va='top', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.6'))
    ax[0].axhline(90, ls='--', color='k', lw=1); ax[0].text(H[-1], 91.5, '90 %', ha='right', fontsize=9)
    ax[0].axhline(0, ls=':', color='gray', lw=1); ax[0].text(H[-1], 2, 'tyle co „brak zmiany”', ha='right', fontsize=8, color='gray')
    ax[0].set_ylim(DOL - 3, 103); ax[0].set_xscale('log')
    ax[0].set_xlabel('ile kroków uczenia naprzód (H)'); ax[0].set_ylabel('umiejętność prognozy siły tożsamości [%]')
    ax[0].set_title('Trafność prognozy (wartości poniżej −100 % ucięte)', fontsize=11)
    ax[0].legend(fontsize=8.5, loc='lower left')
    ax[1].axhline(np.sqrt(.1), ls='--', color='k', lw=1); ax[1].text(H[0], np.sqrt(.1) * 1.08, 'umiejętność 90 %', fontsize=8)
    ax[1].axhline(1, ls=':', color='gray', lw=1); ax[1].text(H[0], 1.08, 'brak zmiany', fontsize=8, color='gray')
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel('ile kroków uczenia naprzód (H)'); ax[1].set_ylabel('błąd prognozy / błąd „braku zmiany”')
    ax[1].set_title('Ile razy prognoza myli się bardziej niż „nic się nie zmieni”', fontsize=11)
    for a in ax:
        a.grid(alpha=.3, which='both')
    fig.suptitle(f'Mały ViT, ziarno {s}: prognoza przyrostu siły tożsamości narzędzia (180 sond, 4 okna startu po 500 kroków)',
                 fontsize=12)
    fig.tight_layout()
    p = HERE / 'wyniki' / f'HORYZONT_s{s}.png'
    fig.savefig(p, dpi=130)
    print(p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
