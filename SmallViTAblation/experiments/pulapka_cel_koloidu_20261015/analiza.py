"""Analiza outputs/pulapka_cel_koloidu_20261015/pomiar.npz → outputs/pulapka_cel_koloidu_20261015/WYNIKI.md: dokąd uczenie pcha koloid i jaki to ruch fizycznie.

1. Cel: czy pchnięcie pułapki (MLP bloku 6) i ruch w uczeniu idą w stronę spadku błędu; rozkład spadku błędu na ruch koloidu,
   ruch celu (głowy) i współdziałanie.
2. Zapadanie na podprzestrzeń głowy (jak neural regression collapse): udział wariancji w 5 składowych i w podprzestrzeni odczytu.
3. Relaksacja do położenia końcowego: prawo potęgowe d ~ t^−α czy wykładnicze (porównanie R² dopasowań).
4. Odpowiedź z opóźnieniem: C(τ) = średni cos(ΔX_{t+τ}, −g_t) i autokorelacja ruchu A(τ) = cos(ΔX_t, ΔX_{t+τ}) (checkpointy co 100 kroków).
5. Dyfuzja w późnym uczeniu: MSD(τ) ~ τ^β (β = 1 dyfuzja, < 1 uwięzienie / klatka, > 1 dryf balistyczny).
"""
from pathlib import Path

import numpy as np

KAT = Path(__file__).resolve().parents[2] / 'outputs' / 'pulapka_cel_koloidu_20261015'
f = np.load(KAT / 'pomiar.npz')
kr = f['kroki']; K = len(kr); heat, poz = f['heat'], f['poz']; wyb = f['wybrane']
linie = []


def pisz(s=''):
    print(s); linie.append(s)


grupy = {'cząsteczka 226 (środek ramki)': np.array([226]), f'tokeny w ramce ({int((poz > 0).sum())})': np.flatnonzero(poz > 0),
         'tło (heat < 0,05)': np.flatnonzero(heat < 0.05)}
okresy = {'początek': slice(0, K // 3), 'środek': slice(K // 3, 2 * K // 3), 'koniec': slice(2 * K // 3, K - 1)}
pisz('# Cel koloidu w uczeniu i fizyka ruchu (bieg s47, zdjęcie z historii eafb5a973be3d490)')
pisz(f'\n{K} checkpointów, kroki {kr[0]}–{kr[-1]}. Token 226: heat {heat[226]:.2f} (środek ramki).')

pisz('\n## 1. Cel: pułapka, ruch w uczeniu, kto zmniejsza błąd')
for g, idx in grupy.items():
    pisz(f'\n### {g}')
    L = f['L_tok'][:, idx].sum(1)
    pisz(f'- błąd: krok {kr[0]} {L[0]:.3g} → krok {kr[K // 3]} {L[K // 3]:.3g} → krok {kr[-1]} {L[-1]:.3g}')
    for n, s in okresy.items():
        cp = f['cos_push'][s][:, idx]; dlp = f['dL_push'][s][:, idx]; cu = f['cos_ucz'][s][:, idx]
        dk, dc, dr = f['dL_koloid'][s][:, idx].sum(), f['dL_cel'][s][:, idx].sum(), f['dL_razem'][s][:, idx].sum()
        udz = np.median(-dlp / np.maximum(f['L_tok'][s][:, idx], 1e-30))
        pisz(f'- {n}: pułapka w stronę celu cos {np.median(cp):+.2f} (dodatni {np.mean(cp > 0):.0%}), pchnięcie obniża błąd '
             f'w {np.mean(dlp < 0):.0%} (mediana {udz:.1%} błędu); ruch w uczeniu wzdłuż −∇L cos {np.median(cu):+.2f} '
             f'(dodatni {np.mean(cu > 0):.0%}); zmiana błędu: koloid {dk:+.3g}, cel {dc:+.3g}, razem {dr:+.3g}, '
             f'współdziałanie {dr - dk - dc:+.3g}')
dk, dc, dr = f['dL_koloid'].sum(), f['dL_cel'].sum(), f['dL_razem'].sum()
pisz(f'\nCały obraz, całe uczenie: koloidy {dk:+.3g}, cel (głowa) {dc:+.3g}, razem {dr:+.3g}, współdziałanie {dr - dk - dc:+.3g}.')

pisz('\n## 2. Zapadanie na podprzestrzeń głowy (losowo 5/192 = 2,6 %)')
for n, s in okresy.items():
    pisz(f'- {n}: 5 głównych składowych {np.median(f["nrc_pca5"][s]):.1%}; podprzestrzeń odczytu głowy {np.median(f["nrc_glowa"][s]):.1%}; '
         f'token 226 w podprzestrzeni odczytu {np.median(f["nrc_226"][s]):.1%}')

pisz('\n## 3. Relaksacja do położenia końcowego')
t_ok = (kr >= 100) & (kr <= 0.5 * kr[-1])                                   # przed zbliżeniem do końca (tam d → 0 z definicji)
for g, idx in grupy.items():
    d = f['odl_kon'][:, idx].mean(1)
    x, y = np.log(kr[t_ok]), np.log(d[t_ok])
    a_p, b_p = np.polyfit(x, y, 1); r2_p = 1 - np.var(y - (a_p * x + b_p)) / np.var(y)
    a_e, b_e = np.polyfit(kr[t_ok], y, 1); r2_e = 1 - np.var(y - (a_e * kr[t_ok] + b_e)) / np.var(y)
    pisz(f'- {g}: potęgowe d ~ t^{a_p:.2f} (R² {r2_p:.3f}); wykładnicze τ = {-1 / a_e:.0f} kroków (R² {r2_e:.3f})')

pisz('\n## 4. Odpowiedź na siłę błędu z opóźnieniem (checkpointy co 100 kroków od 1100)')
reg = np.flatnonzero((kr[:-1] >= 1100) & (np.diff(kr) == 100))
X, G = f['X_wyb'], f['G_wyb']
dX = X[1:] - X[:-1]


def cosr(a, b):
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-30)


for nazwa, kol in (('226', [list(wyb).index(226)]), ('w ramce', list(range(len(wyb))))):
    C = [np.mean(cosr(dX[reg[reg + t < K - 1] + t][:, kol], -G[reg[reg + t < K - 1]][:, kol])) for t in range(0, 31, 5)]
    A = [np.mean(cosr(dX[reg[reg + t < K - 1] + t][:, kol], dX[reg[reg + t < K - 1]][:, kol])) for t in range(1, 31, 5)]
    pisz(f'- {nazwa}: C(τ) dla τ = 0, 5, …, 30 checkpointów: {np.round(C, 3).tolist()}; A(τ) dla τ = 1, 6, …, 26: {np.round(A, 3).tolist()}')

pisz('\n## 5. Dyfuzja w późnym uczeniu (kroki ≥ 20 000, co 100)')
late = reg[kr[reg] >= 20000]
for nazwa, kol in (('226', [list(wyb).index(226)]), ('w ramce', list(range(len(wyb))))):
    taus = np.array([1, 2, 4, 8, 16, 32, 64])
    msd = [np.mean(np.sum((X[late[late + t < K] + t][:, kol] - X[late[late + t < K]][:, kol]) ** 2, -1)) for t in taus]
    beta = np.polyfit(np.log(taus), np.log(msd), 1)[0]
    pisz(f'- {nazwa}: MSD ~ τ^{beta:.2f} (τ = 1–64 checkpointów po 100 kroków)')
(KAT / 'WYNIKI.md').write_text('\n'.join(linie) + '\n', encoding='utf-8')
print('\nzapisano', KAT / 'WYNIKI.md')
