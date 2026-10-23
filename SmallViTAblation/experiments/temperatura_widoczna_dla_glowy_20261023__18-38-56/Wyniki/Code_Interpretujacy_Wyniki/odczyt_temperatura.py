"""Tabele z Wyniki/Dane/temperatura_glowy_tabele.json i temperatura_glowy.npz; wydruk także do tabele_odczyt.txt.
Uruchomienie: PYTHONIOENCODING=utf-8 W:/Python/3.11/python.exe odczyt_temperatura.py <katalog Wyniki/Dane>"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1]); tab = json.loads((D / 'temperatura_glowy_tabele.json').read_text(encoding='utf-8')); d = np.load(D / 'temperatura_glowy.npz')
GR = ['narz', 'tlo', 'wszystkie']
out = []


def p(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)


p('zero 1 (izotropia) 5/192 =', round(5 / 192, 4), '| kontrola:', json.dumps(tab['kontrola'], ensure_ascii=False))
p('\n== Okna epok, pierwsza różnica (mediana po parach × klatkach; energie na 1000 kroków, [nr] = / R² chmury w k); u_vis, u_rand = udziały ==')
for g in GR:
    p(f'\n-- {g} --')
    p(f'{"okno":>8s} {"n":>3s} {"IoU":>6s} {"E_tot/1000":>11s} {"E_vis/1000":>11s} {"E_inv/1000":>11s} {"E_vis nr":>10s} {"E_inv nr":>10s} {"u_vis":>7s} {"u_rand":>7s} {"lin":>6s}')
    for w in tab['okna']:
        p(f'{w["okno"]:>8s} {w["n_par"]:>3d} {w["iou"]:6.3f} {w["E_tot_1000"][g]:11.4g} {w["E_vis_1000"][g]:11.4g} {w["E_inv_1000"][g]:11.4g} '
          f'{w["E_vis_1000_nr"][g]:10.3e} {w["E_inv_1000_nr"][g]:10.3e} {w["u_vis"][g]:7.4f} {w["u_rand"][g]:7.4f} {w["lin_stosunek"][g]:6.2f}')
p('\n== Okna epok, druga różnica (drganie bez dryfu, tylko trójki o równych odstępach; J = ‖j‖²; [nr] = / R²_k) ==')
for g in GR:
    p(f'\n-- {g} --')
    p(f'{"okno":>8s} {"n":>3s} {"IoU":>6s} {"J_tot":>10s} {"J_vis":>10s} {"J_inv":>10s} {"J_vis nr":>10s} {"J_inv nr":>10s} {"u_vis":>7s} {"u_rand":>7s}')
    for w in tab['jitter']:
        p(f'{w["okno"]:>8s} {w["n_trojek"]:>3d} {w["iou"]:6.3f} {w["J_tot"][g]:10.4g} {w["J_vis"][g]:10.4g} {w["J_inv"][g]:10.4g} '
          f'{w["J_vis_nr"][g]:10.3e} {w["J_inv_nr"][g]:10.3e} {w["u_vis"][g]:7.4f} {w["u_rand"][g]:7.4f}')
p('\n== Werdykt predykcji ==')
for k, v in tab['werdykt'].items():
    p(k, json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}, ensure_ascii=False))
p('\n== Pełna tabela 45 par (mediana po 10 klatkach; narzędzie | tło | wszystkie): E_vis, E_inv na 1000 kroków, u_vis, zero losowe, IoU (lewy koniec) ==')
kr, ep, MET = d['krok'], d['epoka'], d['met_pary']; E1000 = d['met_pary_na1000']; iou = d['iou']
for g, ng in enumerate(GR):
    p(f'\n-- {ng} --')
    p(f'{"krok_a":>7s} {"krok_b":>7s} {"ep_a":>7s} {"ep_b":>7s} {"E_vis/1000":>11s} {"E_inv/1000":>11s} {"u_vis":>7s} {"zero":>7s} {"IoU":>6s}')
    for i in range(len(kr) - 1):
        p(f'{kr[i]:7d} {kr[i + 1]:7d} {ep[i]:7.2f} {ep[i + 1]:7.2f} {np.median(E1000[i, :, g, 1]):11.4g} {np.median(E1000[i, :, g, 2]):11.4g} '
          f'{np.median(MET[i, :, g, 4]):7.4f} {np.median(MET[i, :, g, 5]):7.4f} {iou[i].mean():6.3f}')
p('\nwymiar efektywny: min', int(d['wymiar_efektywny'].min()), 'średnia', float(d['wymiar_efektywny'].mean()), '| wartości osobliwe jakobianu (mediana po tokenach) k0 / k65640 / k437500:',
  [np.round(np.median(d['wart_osobliwe'][list(kr).index(k)].reshape(-1, 5), 0), 3).tolist() for k in (0, 65640, 437500)])
(D / 'tabele_odczyt.txt').write_text('\n'.join(out), encoding='utf-8')
