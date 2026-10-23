"""Odczyt Wyniki/Dane/lasery_klasy_tabele.json i lasery_klasy.npz: tabele (mediany po 10 klatkach TEST) do Wyniki/Dane/tabele_odczyt.txt i na stdout.
Użycie: python odczyt_lasery.py <katalog Wyniki/Dane>"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1]); J = json.loads((D / 'lasery_klasy_tabele.json').read_text(encoding='utf-8'))['tabele']
Z = np.load(D / 'lasery_klasy.npz')
KROKI = [0, 1000, 10000, 65640, 218750, 437500]
out = []
p = lambda s='': out.append(s)
row = lambda xs, f='{:7.2f}': ' '.join(f.format(x) for x in xs)
hdr = '  blok:  ' + ' '.join(f'{b:7d}' for b in range(1, 7))

p('Mediany po 10 klatkach TEST (seed 3047). Lasery = wyjścia GELU, zapalony: z > 0,05. Narzędzie: udział >= 0,25; tło: udział == 0. Bloki 1..6.')
p('Klatki: 20 tokenów narzędzia, 560-578 tła na klatkę; 600 tokenów, 768 laserów na blok.\n')
p('=== 1. Odsetek zapalonych laserów (mediana po tokenach klasy): narzędzie / tło / różnica ===')
for k in KROKI:
    t = J[str(k)]; p(f'krok {k}'); p(hdr)
    p('  narzędzie ' + row(t['lit_narz'], '{:7.3f}')); p('  tło       ' + row(t['lit_tlo'], '{:7.3f}')); p('  różnica   ' + row(np.array(t['lit_narz']) - np.array(t['lit_tlo']), '{:7.3f}'))
p('\n=== 2. Liczba laserów (z 768) w klasach p_T/p_B (obie: >=0,5 i >=0,5; tylko_narz: p_T>=0,5, p_B<=0,1; tylko_tło odwrotnie; ciemne: oba <=0,1; reszta mieszane) ===')
for k in KROKI:
    t = J[str(k)]; p(f'krok {k}'); p(hdr)
    for n in ['obie', 'tylko_narzedzie', 'tylko_tlo', 'ciemne', 'mieszane']:
        p(f'  {n:16s}' + row(t['klasy'][n], '{:7.1f}') + '   | zero (etykiety przetasowane): ' + row(t['klasy_zero'][n], '{:6.1f}'))
    p('  |pT-pB| mediana  ' + row(t['absd_mediana'], '{:7.3f}') + '   | zero: ' + row(t['absd_mediana_zero'], '{:6.3f}'))
    p('  odsetek >0,5     ' + row(t['absd_odsetek_gt_05'], '{:7.3f}') + '   | zero: ' + row(t['absd_odsetek_gt_05_zero'], '{:6.3f}'))
    p('  liczba wybiórcz. ' + row(t['n_wybiorczych'], '{:7.1f}') + '   | zero: ' + row(t['n_wybiorczych_zero'], '{:6.1f}'))
p('\n=== 3. Fisher (narzędzie vs tło) w MLP b: przed, po, przed+tylko wkład laserów wybiórczych (|pT-pB|>0,5), przed+tylko wkład niewybiórczych (z biasem) ===')
p('  udział_sel = (F_sel - F_przed)/(F_po - F_przed); Fisher nie jest addytywny, udziały nie muszą sumować się do 1; "zero" = wybiórcze wg przetasowanych etykiet (liczba laserów jak w pkt 2, zero).')
for k in KROKI:
    t = J[str(k)]['fisher']; z = J[str(k)]['fisher_zero']; p(f'krok {k}'); p(hdr)
    for n in ['przed', 'po', 'przed+tylko_wybiorcze', 'przed+tylko_niewybiorcze']:
        p(f'  {n:26s}' + row(t[n], '{:7.2f}'))
    pr, po, se, ni = (np.array(t[n]) for n in ['przed', 'po', 'przed+tylko_wybiorcze', 'przed+tylko_niewybiorcze'])
    with np.errstate(all='ignore'):
        p('  udział_sel                ' + row((se - pr) / (po - pr), '{:7.2f}')); p('  udział_niesel             ' + row((ni - pr) / (po - pr), '{:7.2f}'))
    p('  przed+losowy podzbiór (tyle laserów co wybiórcze) ' + row(J[str(k)]['fisher_los_losowy_podzbior'], '{:7.2f}'))
    p('  zero: przed+wybiórcze(zero)' + row(z['przed+tylko_wybiorcze_zero'], '{:7.2f}') + ' ; przed+niewybiórcze(zero)' + row(z['przed+tylko_niewybiorcze_zero'], '{:7.2f}'))
p('\n=== 4. Mediana cos(v_c, μ_T − μ_B po MLP b) po laserach klasy (v_c = kolumna fc2.weight; NaN: brak laserów w klasie) ===')
for k in KROKI:
    t = J[str(k)]['cos_v']; p(f'krok {k}'); p(hdr)
    for n in ['tylko_narzedzie', 'tylko_tlo', 'niewybiorcze']:
        p(f'  {n:16s}' + row(t[n], '{:7.3f}'))
p('\n=== 5. Błąd odtworzenia Δ MLP: ‖(X_po − X_przed) − (Σ z v + bias)‖/‖X_po − X_przed‖ ===')
b = Z['blad_rekonstrukcji']; p(f'  max = {np.nanmax(b):.3e}, mediana = {np.nanmedian(b):.3e} (46 checkpointów × 10 klatek × 6 bloków)')
p('\n=== 6. Rozrzut między klatkami (blok 4): odsetek zapalonych narzędzie/tło i liczba laserów "tylko narzędzie"/"tylko tło" na każdej klatce, 65640 i 437500 ===')
kr = list(Z['kroki'])
for k in (65640, 437500):
    i = kr.index(k); p(f'krok {k}')
    p('  lit narz ' + row(Z['lit_narz'][i, :, 3], '{:6.3f}')); p('  lit tło  ' + row(Z['lit_tlo'][i, :, 3], '{:6.3f}'))
    p('  tylko_narz ' + row(Z['klasy'][i, :, 3, 1], '{:6.0f}')); p('  tylko_tło  ' + row(Z['klasy'][i, :, 3, 2], '{:6.0f}'))
txt = '\n'.join(out); print(txt)
(D / 'tabele_odczyt.txt').write_text(txt + '\n', encoding='utf-8')
