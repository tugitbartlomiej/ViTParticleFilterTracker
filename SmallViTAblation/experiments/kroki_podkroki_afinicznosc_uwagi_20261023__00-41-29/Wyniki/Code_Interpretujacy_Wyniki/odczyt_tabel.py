"""Tabele z Wyniki/Dane/kroki_podkroki.npz i jakobiany_podkrokow.json (odczyt do odpowiedzi, bez wykresów).
Uruchomienie: python odczyt_tabel.py <katalog Wyniki/Dane>"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
d = np.load(D / 'kroki_podkroki.npz')
kr = d['krok']; K = len(kr)
EP = [0, 1094, 5470, 10940, 21880, 43760, 65640]
wyb = [int(np.argmin(np.abs(kr - e))) for e in EP]
PK = ['u1', 'm1', 'u2', 'm2', 'u3', 'm3', 'u4', 'm4', 'u5', 'm5', 'u6', 'm6']
np.set_printoptions(linewidth=200, precision=3, suppress=True)


def tab(nazwa, a, kol, fmt='{:7.3f}', wiersze=None):
    wiersze = wiersze if wiersze is not None else [f'k{kr[i]:>6d}' for i in wyb]
    print(f'\n== {nazwa} ==')
    print(' ' * 8 + ''.join(f'{c:>8s}' for c in kol))
    for w, i in zip(wiersze, wyb):
        print(f'{w:8s}' + ''.join(' ' + fmt.format(v) for v in a[i]))


print('checkpointów:', K, 'sprawdzenie rozkładu maks:', float(d['sprawdzenie'].max()))
print('pewność w wybranych:', np.round(d['pewnosc'][wyb], 3))
tab('r_po / r_przed (rms)', d['stos'], PK)
tab('mediana |Δ| / r_przed', d['dl'], PK)
tab('mediana cos(Δ, x) (ujemna = do środka)', d['rad'], PK)
tab('CV-R² afiniczne Δ ~ M x + c (192D)', d['r2cv'], PK)
tab('CV-R² zero (przetasowane Δ)', d['r2cv_null'], PK)
tab('R² afiniczne w rzucie 3D aplikacji', d['r2_3d'], PK)
tab('udział translacji |śr Δpos|² / śr |Δpos|² (3D surowe)', d['trans'], PK)
tab('wskaźnik balistyczny |ΣΔ|²/Σ|Δ|²: wszystkie 12 / uwagi / MLP', d['bal'], ['12', 'uwagi', 'MLP'])
tab('cos(uwaga_b, MLP_b)', d['cos_am'], [f'b{b}' for b in range(1, 7)])
tab('cos(MLP_b, uwaga_b+1)', d['cos_ma'], [f'b{b}' for b in range(1, 6)])
tab('cos(MLP_b, MLP_b+1)', d['cos_mm'], [f'b{b}' for b in range(1, 6)])
tab('cos(uwaga_b, uwaga_b+1)', d['cos_aa'], [f'b{b}' for b in range(1, 6)])
for k, nazwa in (('att_neff', 'efektywne źródła uwagi 1/ΣA² (mediana), głowy uśrednione'), ('att_self', 'uwaga na sobie'),
                 ('att_sas8', 'uwaga na 8 sąsiadach'), ('att_dist', 'średnia odległość uwagi [łaty]'), ('att_ent', 'entropia uwagi [nat]'),
                 ('att_bg2tool', 'wzbogacenie tło→narzędzie'), ('att_tool2tool', 'wzbogacenie narzędzie→narzędzie')):
    tab(nazwa, d[k].mean(2), [f'b{b}' for b in range(1, 7)], '{:7.2f}')
tab('efektywne źródła, min po głowach', d['att_neff'].min(2), [f'b{b}' for b in range(1, 7)], '{:7.1f}')
tab('efektywne źródła, maks po głowach', d['att_neff'].max(2), [f'b{b}' for b in range(1, 7)], '{:7.1f}')
tab('lasery: odsetek z > 0,05 (mediana)', d['mlp_lit'], [f'b{b}' for b in range(1, 7)])
tab('lasery: efektywna liczba (Σ|z|)²/Σz²', d['mlp_neff'], [f'b{b}' for b in range(1, 7)], '{:7.1f}')
tab('lasery: udział 10 najjaśniejszych w Σ|z|', d['mlp_top10'], [f'b{b}' for b in range(1, 7)])
ST = ['s0'] + [p for p in PK]
tab('Fisher narzędzie/tło w 13 stanach (Xc)', d['fisher'], ST, '{:7.2f}')
tab('Fisher na kierunkach Xc/|Xc|', d['fisher_u'], ST, '{:7.2f}')
tab(f'czystość kNN narzędzia (zero {float(d["narz"].sum() - 1) / 599:.4f})', d['knn'], ST)
tab('czystość kNN na kierunkach', d['knn_u'], ST)
tab('Dirichlet na siatce (1 = brak gładkości)', d['dir'], ST)
tab('Dirichlet na kierunkach', d['dir_u'], ST)

# kiedy: pierwszy krok, w którym obserwabla przekracza próg
def pierwszy(a, warunek):
    i = np.where(warunek(a))[0]
    return int(kr[i[0]]) if len(i) else None
print('\n== kiedy ==')
for b in range(6):
    ne = d['att_neff'][:, b].mean(1)
    print(f'blok {b + 1}: źródła < 300 od kroku {pierwszy(ne, lambda a: a < 300)}, < 100 od {pierwszy(ne, lambda a: a < 100)}, '
          f'< 50 od {pierwszy(ne, lambda a: a < 50)}; min {ne.min():.0f} w kroku {kr[ne.argmin()]}')
for s in (0, 6, 8, 12):
    f = d['fisher'][:, s]; print(f'stan {ST[s]}: Fisher > 1 od kroku {pierwszy(f, lambda a: a > 1)}, > 3 od {pierwszy(f, lambda a: a > 3)}, maks {f.max():.2f} w {kr[f.argmax()]}')
for s in (1, 7, 9):
    st = d['stos'][:, s]; print(f'podkrok {PK[s]}: r_po/r_przed maks {st.max():.3f} w kroku {kr[st.argmax()]}, w k65640 {st[-1]:.3f}')
# korelacje w czasie: czy skurcz uwagi i rozdział idą razem
from scipy.stats import spearmanr
print('\nSpearman w czasie (806 ckpt): Fisher(s12) vs źródła b4:', round(spearmanr(d['fisher'][:, 12], d['att_neff'][:, 3].mean(1))[0], 3),
      '| Fisher(s12) vs pewność:', round(spearmanr(d['fisher'][:, 12], d['pewnosc'])[0], 3),
      '| bal(12) vs krok:', round(spearmanr(d['bal'][:, 0], kr)[0], 3))

J = D / 'jakobiany_podkrokow.json'
if J.exists():
    j = json.load(open(J, encoding='utf-8'))
    tok = j['tokeny']; rod = j['rodzaj']; kj = j['krok']
    print('\n==== JAKOBIANY ====', 'tokeny', tok, rod, 'błąd maks', j['blad_maks'])
    narz_i = [i for i, r in enumerate(rod) if r.startswith('narz')]; tlo_i = [i for i, r in enumerate(rod) if r == 'tło']
    def sr(klucz, idx):
        a = np.array(j[klucz]); return a[:, idx].mean(1)
    for nazwa, idx in (('narzędzie (3 tokeny)', narz_i), ('tło (3 tokeny)', tlo_i)):
        print(f'\n--- {nazwa} ---')
        print('udział kierunków kurczących (eig sym < 1) map własnych, podkroki', PK)
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('kurczy_wl', idx)[a_], 2))
        print('udział σ < 1 iloczynu po głębokości Π_{s≤t} J_s (po podkroku t)')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('kurczy_prod', idx)[a_], 2))
        print('log σ_max iloczynu po podkroku t')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('log_svmax_prod', idx)[a_], 2))
        print('log σ_min iloczynu po podkroku t')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('log_svmin_prod', idx)[a_], 2))
        print('log σ_max map własnych')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('log_svmax_wl', idx)[a_], 2))
        print('log|det| map własnych')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('logdet', idx)[a_], 3))
        ld = np.array(j['logdet'])[:, idx].sum(2).mean(1)
        print('log|det| całego przejścia Σ_s log|det J_s| (na kierunek: /192):', [f'k{k}: {v:.2f} ({v / 192:+.4f})' for k, v in zip(kj, ld)])
        print('wykładniki Lapunowa po głębokości (log σ_max, log σ_min iloczynu)/12 podkroków:',
              [f'k{k}: {a:+.3f} / {b:+.3f}' for k, a, b in zip(kj, sr('log_svmax_prod', idx)[:, -1] / 12, sr('log_svmin_prod', idx)[:, -1] / 12)])
        print('‖A‖/‖S‖ części J − I')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('asym', idx)[a_], 2))
        print('‖J − I‖_F map własnych')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('odch_wl', idx)[a_], 3))
        print('udział kurczących w mapach bloków M_b = J_MLP J_uwaga, bloki 1..6')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('kurczy_blok', idx)[a_], 2))
        print('nakładanie podprzestrzeni kurczących bloków b,b+1: tr(P_b P_b+1)/min(k) | losowe max(k)/192')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('nakl_blok', idx)[a_], 2), '|', np.round(sr('nakl_blok_losowe', idx)[a_], 2))
        print('to samo dla samych MLP')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('nakl_mlp', idx)[a_], 2), '|', np.round(sr('nakl_mlp_losowe', idx)[a_], 2))
        print('UWAGA pełny jakobian: ‖J_ii − I‖² / Σ_j≠i ‖J_ij‖² (własny / inni), bloki 1..6')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('uw_self', idx)[a_] / sr('uw_inni', idx)[a_], 3))
        print('UWAGA: efektywna liczba cząsteczek wpływających (wagi ‖J_ij‖²)')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('uw_neff', idx)[a_], 1))
        print('UWAGA: udział części barycentrycznej (zamrożone A) ‖J_zamr‖²/‖J_pełny‖² (bez I), i cos')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('uw_zamr', idx)[a_], 3), '|', np.round(sr('uw_cos', idx)[a_], 3))
        print('UWAGA: udział barycentryczny tylko dla innych cząsteczek')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('uw_zamr_inni', idx)[a_], 3))
        print('UWAGA: udział wpływu 8 sąsiadów siatki | tokenów narzędzia (wagi ‖J_ij‖²)')
        for a_, k in enumerate(kj): print(f'k{k:>6d}', np.round(sr('uw_wagi_sas', idx)[a_], 3), '|', np.round(sr('uw_wagi_narz', idx)[a_], 3))
