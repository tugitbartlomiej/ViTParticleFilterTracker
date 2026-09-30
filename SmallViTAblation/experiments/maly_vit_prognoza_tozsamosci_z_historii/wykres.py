#!/usr/bin/env python3
"""Wykres błędu prognozy siły tożsamości wobec liczby kroków naprzód (wyniki `prognoza_z_historii.py`).

    python wykres.py   ->  wyniki/PROGNOZA_Z_HISTORII.png
Oś pionowa: błąd prognozy przyrostu / błąd „braku zmiany” (1 = tyle co „nic się nie zmieni”, 0,32 = umiejętność 90 %).
Wartości powyżej 1,5 ucięte. Cztery panele: Koopman, Langevin, pomieszane z pędem i przyspieszeniem, najlepsze razem.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
GORA = 1.5
PANELE = [
    ('Koopman', ['Koopman: opóźnienia 20, każda sonda', 'Koopman DMD: 180 sond razem, 20 modów', 'Koopman DMD + pęd']),
    ('Langevin', ['bezwładność', 'Langevin OU (tarcie + sprężyna)', 'Langevin + jądro pamięci 20',
                  'Langevin + jądro pamięci 100']),
    ('Pomieszane z pędem i przyspieszeniem', ['Langevin OU + pęd', 'Langevin OU + pęd + przyspieszenie',
                                             'Langevin + jądro 20 + pęd', 'Langevin + jądro 20 + pęd + przyspieszenie',
                                             'Langevin + jądro 20 + pęd, wprost', 'Langevin + jądro 20, wprost (bez pędu)']),
]
ODN = 'równanie co krok na prawdziwych wagach (nie prognoza)'
ZNANE = 'znane klatki kroku t, dalej jądro 20 + pęd'


def main() -> int:
    d = json.loads((HERE / 'wyniki' / 'PROGNOZA_Z_HISTORII.json').read_text(encoding='utf-8'))
    H = np.array(d['H']); M = d['modele']
    bl = {n: np.array([r['blad'][str(h)] for h in H]) for n, r in M.items()}
    fig, ax = plt.subplots(2, 2, figsize=(15, 10.5), sharex=True, sharey=True)
    ax = ax.ravel()
    kol = plt.get_cmap('tab10')

    def rysuj(a, nazwy, tytul):
        for i, n in enumerate(nazwy):
            y = bl[n]
            a.plot(H, np.minimum(y, GORA), '-o', ms=3.5, lw=1.8, color=kol(i), label=n)
            if (y > GORA).any():
                a.plot(H[y > GORA], np.full((y > GORA).sum(), GORA), 'v', color=kol(i), ms=6)
        a.plot(H, np.minimum(bl[ODN], GORA), '--', color='0.45', lw=1.3, label=ODN)
        a.set_title(tytul, fontsize=11)
        a.legend(fontsize=8, loc='lower right')

    for a, (t, n) in zip(ax[:3], PANELE):
        rysuj(a, n, t)
    # najlepsze z każdej rodziny (średni błąd po H) i znane klatki
    najl = []
    for _, nazwy in PANELE:
        najl.append(min(nazwy, key=lambda n: np.mean(np.minimum(bl[n], GORA))))
    rysuj(ax[3], najl + [ZNANE], 'Najlepsze z każdej grupy i wariant ze znanymi klatkami kroku t')
    for a in ax:
        a.axhline(1, ls=':', color='k', lw=1); a.axhline(np.sqrt(.1), ls='--', color='k', lw=.8)
        a.text(H[0], 1.02, 'brak zmiany', fontsize=8); a.text(H[0], np.sqrt(.1) + .02, 'umiejętność 90 %', fontsize=8)
        a.set_xscale('log'); a.set_ylim(0, GORA + .05); a.grid(alpha=.3, which='both')
    for a in ax[2:]:
        a.set_xlabel('ile kroków uczenia naprzód (t+1, t+2, ...)')
    for a in ax[::2]:
        a.set_ylabel('błąd prognozy / błąd „braku zmiany”')
    fig.suptitle('Prognoza siły tożsamości narzędzia (180 sond) ze stanu i historii do chwili t; dopasowanie s47, '
                 'test s47 + s48\n(trójkąt = błąd powyżej 1,5; linia przerywana szara: równanie na prawdziwych wagach, '
                 'nie prognoza)', fontsize=11)
    fig.tight_layout()
    p = HERE / 'wyniki' / 'PROGNOZA_Z_HISTORII.png'
    fig.savefig(p, dpi=125)
    print(p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
