"""Wykres prognozy od kroku 900 (checkpoint 90) do 1000 (checkpoint 100) z PROGNOZA_900.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KAT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'wyniki' / 'prognoza_900'
m = json.loads((KAT / 'PROGNOZA_900.json').read_text(encoding='utf-8'))
sila = np.load(KAT / 'SILA_TOZSAMOSCI.npy')                  # (1001, 12)

# Paleta referencyjna (tryb jasny): kolejność slotów stała.
NIEB, POM, AQUA, FIOL = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
INK, INK2, MUTED, GRID, TLO = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': TLO, 'figure.facecolor': TLO})

h = np.arange(1, 101)
kroki = 900 + h
fig = plt.figure(figsize=(15, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.42, wspace=0.18)

# --- A: przykładowa trajektoria siły tożsamości jednego obrazu
par = m['wielkosci']['sila_tozsamosci']['900']['parametry']
bl = np.abs(sila[901:] - sila[900]).mean(0)
i = int(np.argsort(bl)[len(bl) // 2])                       # obraz o środkowym błędzie prognozy bez zmian
x = sila[:, i]
mu = x[900 - par['k'] + 1:901].mean()
rown = mu + (x[900] - mu) * np.exp(-par['kappa'] * h)
przed = x[900] + h * (x[900] - x[890]) / 10
ax = fig.add_subplot(gs[0, :])
ax.plot(np.arange(600, 1001), x[600:], color=INK, lw=1.2, label='prawdziwy przebieg')
ax.plot(kroki, np.full_like(h, x[900], dtype=float), color=MUTED, lw=2, ls='--', label='bez zmian')
ax.plot(kroki, rown, color=NIEB, lw=2, label=f"równanie (ściąganie do średniej z {par['k']} kroków)")
ax.plot(kroki, przed, color=AQUA, lw=2, label='przedłużenie ruchu z ostatnich 10 kroków')
ax.plot([900 - par['k'] + 1, 1000], [mu, mu], color=NIEB, lw=0.8, ls=':', alpha=0.8)
ax.text(900 - par['k'] + 1, mu, f"średnia z {par['k']} kroków przed startem ", color=NIEB, fontsize=8.5,
        ha='right', va='center')
ax.axvline(900, color=INK2, lw=0.8)
lo, hi = np.percentile(x[600:], [0, 100])
pad = 0.25 * (hi - lo)
ax.set_ylim(lo - pad, hi + pad)
ax.text(902, hi + 0.15 * pad, 'start prognozy: krok 900 (checkpoint 90)', color=INK2, fontsize=9, va='center')
ax.set_xlim(600, 1000)
ax.set_xlabel('krok uczenia')
ax.set_ylabel('siła tożsamości narzędzia')
ax.set_title(f'A. Przykład: siła tożsamości narzędzia w obrazie {i} i trzy prognozy od kroku 900',
             loc='left', color=INK, fontsize=11)
ax.grid(axis='y', color=GRID, lw=0.6)
ax.legend(loc='lower left', frameon=False, ncol=4, fontsize=9)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# --- B, C, D: wynik na każdym kroku 901..1000
panele = [('tokeny_losowe', 'B. Tokeny losowe (768 liczb na token)'),
          ('tokeny_narzedzia', 'C. Tokeny narzędzia (768 liczb na token)'),
          ('sila_tozsamosci', 'D. Siła tożsamości (1 liczba na obraz)')]
for j, (q, tyt) in enumerate(panele):
    d = m['wielkosci'][q]
    ax = fig.add_subplot(gs[1, j])
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(0.85, color=INK2, lw=1, ls=':')
    ax.text(3, 0.87, 'cel 85 %', color=INK2, fontsize=8.5, va='bottom')
    ax.text(99, -0.03, 'bez zmian', color=INK2, fontsize=8.5, va='top', ha='right')
    ax.plot(h, d['900']['sufit'], color=MUTED, lw=1.6, ls='--', label='sufit (wyrocznia)')
    ax.plot(h, d['srednia_po_startach']['rownanie'], color=NIEB, lw=1, alpha=0.45,
            label='równanie, średnio ze startów 700–900')
    ax.plot(h, d['900']['rownanie'], color=NIEB, lw=2, label='równanie, start 900')
    ax.plot(h, d['900']['srednia101'], color=POM, lw=2, label='średnia z ostatnich 101 kroków')
    ax.plot(h, np.clip(d['900']['przedluzenie'], -1.5, None), color=AQUA, lw=2, label='przedłużenie ruchu')
    if q == 'tokeny_losowe' and 'nndmd' in m:
        ax.plot(m['nndmd']['h'], m['nndmd']['skill_mse'], 'o', ms=8, color=FIOL, mec=TLO, mew=2,
                label='NNDMD, checkpointy 91–100')
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, 101)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel('kroki od startu (901..1000)')
    if j == 0:
        ax.set_ylabel('o ile mniejszy błąd niż bez zmian')
    ax.set_title(tyt, loc='left', color=INK, fontsize=11)
    ax.grid(axis='y', color=GRID, lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    p = d['900']['parametry']
    ax.text(99, -0.95, f"k = {p['k']}, κ = {p['kappa']:.3f} na krok", color=INK2, fontsize=8.5, ha='right')
    if q == 'tokeny_losowe':
        ax.text(9, -0.98, 'przedłużenie ruchu spada do −31', color=AQUA, fontsize=8.5, va='bottom')
uchwyty, etyk = [], []
for a_ in fig.axes[1:]:
    for hh, ll in zip(*a_.get_legend_handles_labels()):
        if ll not in etyk:
            uchwyty.append(hh); etyk.append(ll)
fig.legend(uchwyty, etyk, loc='lower center', ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
fig.suptitle('Prognoza latentu małego ViT od checkpointu 90 do 100: wynik na każdym kroku '
             '(1 − błąd² / błąd² prognozy bez zmian)', x=0.06, ha='left', fontsize=12.5, color=INK)
fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12)
fig.savefig(KAT / 'PROGNOZA_900.png', dpi=150)
print('zapisano', KAT / 'PROGNOZA_900.png')
