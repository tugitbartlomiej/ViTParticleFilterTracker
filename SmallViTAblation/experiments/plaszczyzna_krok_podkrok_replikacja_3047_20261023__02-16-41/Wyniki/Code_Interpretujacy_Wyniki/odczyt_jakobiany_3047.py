"""Tabele z Wyniki/Dane/jakobiany_3047.json (średnie po klatkach i tokenach narzędzia / tła). Uruchomienie: python odczyt_jakobiany_3047.py <Wyniki/Dane>"""
import json, sys
from pathlib import Path
import numpy as np
j = json.load(open(Path(sys.argv[1]) / 'jakobiany_3047.json', encoding='utf-8')); kj = j['krok']
np.set_printoptions(linewidth=200, precision=3, suppress=True)
PK = ['u1', 'm1', 'u2', 'm2', 'u3', 'm3', 'u4', 'm4', 'u5', 'm5', 'u6', 'm6']
A = lambda k: np.array(j[k])
print('blad maks', j['blad_maks'])
for nazwa, sl in (('narzedzie', slice(0, 3)), ('tlo', slice(3, 6))):
    print(f'\n--- {nazwa} (srednia po 10 klatkach x 3 tokeny) ---')
    for klucz, tyt in (('kurczy_wl', 'udzial kurczacych map wlasnych ' + str(PK)), ('kurczy_prod', 'udzial sigma<1 iloczynu po podkroku t'),
                       ('logdet', 'log|det| podkrokow'), ('asym', '|A|/|S|')):
        print(tyt)
        for a_, k in enumerate(kj):
            print(f'k{k:>6d}', np.round(A(klucz)[a_, sl].mean(0), 2))
    print('log sigma_max / log sigma_min iloczynu po 12 podkrokach, suma log|det|')
    for a_, k in enumerate(kj):
        print(f'k{k:>6d}', round(float(A('log_svmax_prod')[a_, sl, -1].mean()), 2), round(float(A('log_svmin_prod')[a_, sl, -1].mean()), 2), round(float(A('logdet_suma')[a_, sl].mean()), 2))
    print('nakladanie blokow b,b+1 | losowe')
    for a_, k in enumerate(kj):
        print(f'k{k:>6d}', np.round(A('nakl_blok')[a_, sl].mean(0), 3), '|', np.round(A('nakl_blok_losowe')[a_, sl].mean(0), 3))
print('\n--- pelny jakobian uwagi (token 0 narzedzie / token 3 tlo), bloki 1..6 ---')
for kl, nazwa in (('uw_neff', 'efektywna liczba wplywajacych'), ('uw_zamr_inni', 'udzial barycentryczny (inni)'), ('uw_narz', 'udzial wplywu tokenow narzedzia'), ('uw_sas', 'udzial wplywu 8 sasiadow')):
    print(nazwa)
    for a_, k in enumerate(kj):
        v = A(kl)[a_]; print(f'k{k:>6d} narz', np.round(v[0], 3), ' tlo', np.round(v[1], 3))
print('wlasny/inni')
for a_, k in enumerate(kj):
    v = A('uw_self')[a_] / A('uw_inni')[a_]; print(f'k{k:>6d} narz', np.round(v[0], 2), ' tlo', np.round(v[1], 2))
