"""Tabele z Wyniki/Dane/plaszczyzna_3047.npz (średnie po 10 klatkach; w nawiasach rozstęp min–maks po klatkach tam, gdzie podano).
Uruchomienie: python odczyt_3047.py <katalog Wyniki/Dane>"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1]); d = np.load(D / 'plaszczyzna_3047.npz'); kr = d['krok']; ep = d['epoka']
PK = ['u1', 'm1', 'u2', 'm2', 'u3', 'm3', 'u4', 'm4', 'u5', 'm5', 'u6', 'm6']; ST = ['wej'] + PK
WYB = [0, 300, 1000, 3000, 10000, 24000, 65640, 109375, 218750, 328125, 437500]
iw = [int(np.where(kr == k)[0][0]) for k in WYB]
np.set_printoptions(linewidth=220, precision=3, suppress=True)


def tab(nazwa, a, kol, fmt='{:7.3f}'):
    print(f'\n== {nazwa} ==')
    print(' ' * 11 + ''.join(f'{c:>8s}' for c in kol))
    for i in iw:
        print(f'k{kr[i]:>6d} {ep[i]:4.0f}e' + ''.join(' ' + fmt.format(v) for v in a[i]))


print('klatek', d['fisher'].shape[1], 'checkpointów', len(kr), 'narzędzie ≥', float(d['prog_narz']), 'łat narzędzia na klatkę:', (d['udzial'] >= d['prog_narz']).sum(1))
print('IoU średnie:', [f'k{kr[i]}: {d["iou"][i].mean():.3f}' for i in iw])
print('pewność średnia:', [f'k{kr[i]}: {d["pewnosc"][i].mean():.3f}' for i in iw])
tab('r_po / r_przed', d['stos'].mean(1), PK)
tab('mediana cos(Δ, x)', d['rad'].mean(1), PK)
tab('CV-R² afiniczne (192D)', d['r2cv'].mean(1), PK)
tab('CV-R² zero', d['r2cv_null'].mean(1), PK)
tab('udział translacji w 192D', d['trans'].mean(1), PK)
tab('wskaźnik balistyczny: 12 / uwagi / MLP', d['bal'].mean(1), ['12', 'uwagi', 'MLP'])
tab('cos(uwaga_b, MLP_b)', d['cos_am'].mean(1), [f'b{b}' for b in range(1, 7)])
tab('cos(MLP_b, uwaga_b+1)', d['cos_ma'].mean(1), [f'b{b}' for b in range(1, 6)])
tab('cos(MLP_b, MLP_b+1)', d['cos_mm'].mean(1), [f'b{b}' for b in range(1, 6)])
tab('efektywne źródła (mediana po klatkach, średnia po głowach)', np.median(d['att_neff'].mean(3), 1), [f'b{b}' for b in range(1, 7)], '{:7.1f}')
tab('uwaga na 8 sąsiadach (maks po klatkach)', d['att_sas8'].mean(3).max(1), [f'b{b}' for b in range(1, 7)], '{:7.4f}')
tab('średnia odległość uwagi [łaty]', d['att_dist'].mean(3).mean(1), [f'b{b}' for b in range(1, 7)], '{:7.2f}')
tab('wzbogacenie tło→narzędzie (mediana po klatkach)', np.median(d['att_bg2tool'].mean(3), 1), [f'b{b}' for b in range(1, 7)], '{:7.1f}')
tab('wzbogacenie narzędzie→narzędzie (mediana)', np.median(d['att_tool2tool'].mean(3), 1), [f'b{b}' for b in range(1, 7)], '{:7.1f}')
tab('udział 3 największych odbiorców w obrysie ∪ halo', d['hub_top3'].mean(1), [f'b{b}' for b in range(1, 7)])
tab('udział 3 największych odbiorców w narzędziu (≥ 0,25)', d['hub_top3_narz'].mean(1), [f'b{b}' for b in range(1, 7)])
tab('lasery: odsetek z > 0,05', d['mlp_lit'].mean(1), [f'b{b}' for b in range(1, 7)])
tab('Fisher narzędzie/tło po stanach (średnia po klatkach)', d['fisher'].mean(1), ST, '{:7.2f}')
tab('Fisher: min po klatkach', d['fisher'].min(1), ST, '{:7.2f}')
tab('Dirichlet po stanach', d['dir'].mean(1), ST)
print('\n== kiedy (mediana po klatkach) ==')
med = np.median(d['att_neff'].mean(3), 1)
for b in range(6):
    i = np.where(med[:, b] < 50)[0]; print(f'blok {b + 1}: źródła < 50 od kroku {int(kr[i[0]]) if len(i) else None}; min {med[:, b].min():.0f} w k{kr[med[:, b].argmin()]}')
fi = d['fisher'].mean(1)
print('Fisher m6 > 1 od', int(kr[np.argmax(fi[:, 12] > 1)]), '; > 2 od', int(kr[np.argmax(fi[:, 12] > 2)]) if (fi[:, 12] > 2).any() else None, '; > 3 od', int(kr[np.argmax(fi[:, 12] > 3)]) if (fi[:, 12] > 3).any() else None, '; maks', round(float(fi[:, 12].max()), 2), 'w k', int(kr[fi[:, 12].argmax()]))
skok = np.argmax(np.diff(d['fisher'], axis=2), axis=2)
print('podkrok największego skoku Fishera (liczba par klatka × checkpoint ≥ 24000):', {PK[s]: int((skok[kr >= 24000] == s).sum()) for s in range(12)})
bal = d['bal'][:, :, 0].mean(1); print('balistyczny po krokach:', [(int(k), round(float(v), 2)) for k, v in zip(kr, bal)][:14])
