"""Rysunek testu jądra (JADRO_TABELA.json z jadro_tabela.py): dla każdego stanu (lr 5e-4 krok 2188, lr 1,25e-4 krok 2188, lr 5e-4 epoka 12)
rodziny przesunięć wag kontra zmierzone przyrosty: (1) heterogeniczność C₁ po powłokach i z innych zdjęć, (2) rozkład energii kroku,
(3) Spearman mapy ruchliwości. Uruchomienie: F:/Python/3.13/python.exe jadro_rysunek.py → jadro.png"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, 'JADRO_TABELA.json'), encoding='utf-8'))
stany = [k for k in J if not k.endswith('_tf32') and not k.endswith('_bezBN')]
rodz = ['zmierzone', 'paczka', 'paczka_wagi', 'bn', 'szum', 'los_izo', 'los_prek']
ety = {'zmierzone': 'zmierzone przyrosty', 'paczka': 'krok AdamW (wagi+BN)', 'paczka_wagi': 'same wagi', 'bn': 'same statystyki BN', 'szum': 'szum minibatcha',
       'los_izo': 'losowy kierunek izotropowy', 'los_prek': 'losowy kierunek prekond.'}
kol = {'zmierzone': '#ffffff', 'paczka': '#ffd166', 'paczka_wagi': '#ffb347', 'bn': '#9aa3ad', 'szum': '#ff4433', 'los_izo': '#4a8cff', 'los_prek': '#c77dff'}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(3, len(stany), figsize=(7 * len(stany), 14), dpi=100, squeeze=False)
for j, st in enumerate(stany):
    R = J[st]['rodziny']
    tyt = f"{st} (krok {J[st]['krok']})"
    a = ax[0, j]
    for r in rodz:
        w = R.get(r)
        if not w or w.get('C1') is None:
            continue
        y = [w['C1'], w['pow2'], w['pow3'], w['cross']]
        a.plot([1, 2, 3, 4.5], y, 'o-' if r == 'zmierzone' else 's--', color=kol[r], lw=2.5 if r == 'zmierzone' else 1.5, ms=7 if r == 'zmierzone' else 5, label=ety[r])
    a.set_xticks([1, 2, 3, 4.5]); a.set_xticklabels(['powłoka 1', 'powłoka 2', 'powłoka 3', 'sąsiedzi z innych zdjęć'])
    a.set_ylabel('korelacja ruchliwości (CR, τ=1)'); a.set_title(tyt + ': struktura pola', fontsize=10); a.grid(True, alpha=0.4); a.set_ylim(0, 1)
    if j == 0:
        a.legend(fontsize=8)
    a = ax[1, j]
    nazwy = [r for r in rodz if R.get(r) and R[r].get('wsp') is not None]
    dane = np.array([[R[r]['wsp'], R[r]['zdj'], R[r]['sas'], R[r]['wl']] for r in nazwy])
    x = np.arange(len(nazwy)); dol = np.zeros(len(nazwy))
    for i, (et, c) in enumerate(zip(('dryf populacji', 'pole zdjęcia', 'pole sąsiedztwa', 'ruch własny (CR)'), ('#ff4433', '#ff8c42', '#ffd166', '#4a8cff'))):
        a.bar(x, dane[:, i], bottom=dol, color=c, label=et); dol += dane[:, i]
    a.set_xticks(x); a.set_xticklabels([ety[r] for r in nazwy], rotation=30, ha='right', fontsize=8)
    a.set_ylabel('udział w <|Δz|²>'); a.set_title(tyt + ': rozkład energii kroku', fontsize=10); a.grid(True, axis='y', alpha=0.4)
    if j == 0:
        a.legend(fontsize=8)
    a = ax[2, j]
    nazwy = [r for r in rodz if r != 'zmierzone' and R.get(r) and R[r].get('sp_wzgl') is not None and np.isfinite(R[r]['sp_wzgl'])]
    x = np.arange(len(nazwy))
    a.bar(x - 0.2, [R[r]['sp_wzgl'] for r in nazwy], 0.4, color='#ffd166', label='w zdjęciu (względna)')
    a.bar(x + 0.2, [R[r]['sp_zdj'] if R[r]['sp_zdj'] is not None else 0 for r in nazwy], 0.4, color='#4a8cff', label='między zdjęciami (średnia zdjęcia)')
    a.axhline(0, color='#8a8f99', lw=0.8)
    a.set_xticks(x); a.set_xticklabels([ety[r] for r in nazwy], rotation=30, ha='right', fontsize=8)
    a.set_ylabel('Spearman: przewidziana vs zmierzona ruchliwość'); a.set_title(tyt + ': mapa ruchliwości', fontsize=10); a.grid(True, axis='y', alpha=0.4); a.set_ylim(-0.3, 1)
    if j == 0:
        a.legend(fontsize=8)
fig.suptitle('Test jądra: pole przesunięć tokenów z Jakobianu wspólnych wag (różnice skończone ze stanu biegu) kontra zmierzone przyrosty', fontsize=12)
fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.08, hspace=0.55, wspace=0.25)
fig.savefig(os.path.join(HERE, 'jadro.png'))
print('zapisano jadro.png')
