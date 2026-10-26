"""Tabele liczb (markdown) z ../Dane/podsumowanie.json → ../Dane/tabele.md (wklejane do Opis/README.md). Bez wykresów i wniosków."""
import json
from pathlib import Path

DANE = Path(__file__).resolve().parent.parent / 'Dane'
S = json.loads((DANE / 'podsumowanie.json').read_text(encoding='utf-8'))
kroki = S['kroki']; ost = f'k{kroki[-1]}'
C = S['checkpointy'][ost]

GLOWNE = ['nJ', 'nLam', 'nN', 'N_do_J', 'kappa', 'nS', 'nOm', 'Om_do_S', 'trJ', 'Q', 'R', 'Q_S', 'Q_Om', 'Q_minus_QLam',
          'Q_S_Lam', 'Q_S_N', 'Q_Om_Lam', 'Q_Om_N', 'udzN_w_S2', 'udzN_w_Om2', 'n_re_dod', 'n_re_uj', 'n_zero', 'n_im',
          'max_re', 'rho', 'max_im', 'n_mod_gt1', 'n_1pl_gt1', 'max_mod_1pl', 'norm2_J', 'norm2_IpJ']
TREND = ['N_do_J', 'kappa', 'Om_do_S', 'Q', 'R', 'Q_S', 'Q_Om', 'trJ', 'n_re_dod', 'n_im', 'n_1pl_gt1', 'rho', 'nJ']


def f(v):
    if v is None:
        return '–'
    a = abs(v)
    if v == int(v) and a < 1e5:
        return f'{int(v)}'
    if a != 0 and (a < 1e-3 or a >= 1e4):
        return f'{v:.2e}'
    return f'{v:.3g}' if a < 10 else f'{v:.1f}'


L = []
L.append(f'## Checkpoint {ost}: mediany po 600 tokenach, blok × miara (pełne J 192 × 192)\n')
for zr in ('J192', 'J3'):
    if zr == 'J3':
        L.append(f'\n## Checkpoint {ost}: mediany po 600 tokenach, blok × miara (J3 = P J Pᵀ, 3 × 3)\n')
    L.append('| miara | ' + ' | '.join(f'blok {b}' for b in range(6)) + ' |')
    L.append('|---|' + '---|' * 6)
    for k in GLOWNE:
        L.append(f'| {k} | ' + ' | '.join(f(C[f"blok_{b}"][zr]["wszystkie"][k][1]) for b in range(6)) + ' |')

for zr, tyt in (('J192', 'pełne J'), ('J3', 'J3')):
    L.append(f'\n## Checkpoint {ost}: narzędzie vs tło, percentyle 10 / 50 / 90 ({tyt})\n')
    for b in range(6):
        r = C[f'blok_{b}'][zr]
        L.append(f'\n**Blok {b}** (narzędzie n = {S["n_narz"]}, tło n = {S["n_tlo"]})\n')
        L.append('| miara | wszystkie p10 | p50 | p90 | narzędzie p10 | p50 | p90 | tło p10 | p50 | p90 |')
        L.append('|---|' + '---|' * 9)
        for k in GLOWNE:
            w = []
            for g in ('wszystkie', 'narzedzie', 'tlo'):
                w += [f(x) for x in r[g][k]]
            L.append(f'| {k} | ' + ' | '.join(w) + ' |')

L.append(f'\n## Checkpoint {ost}: Spearman ρ miar z ciepłem głowy (sigmoid glowa[12,:,0]) i z ‖F‖ (surowy skok MLP), 600 tokenów\n')
for zr in ('J192', 'J3'):
    L.append(f'\n**{zr}** (komórka: ρ z ciepłem / ρ z ‖F‖ surowym / ρ z ‖F‖ środkowanym)\n')
    L.append('| miara | ' + ' | '.join(f'blok {b}' for b in range(6)) + ' |')
    L.append('|---|' + '---|' * 6)
    for k in GLOWNE:
        cells = []
        for b in range(6):
            s = C[f'blok_{b}'][zr]['spearman'][k]
            cells.append('–' if s is None else f'{s["cieplo"]:+.2f} / {s["normF_surowe"]:+.2f} / {s["normF_srodkowane"]:+.2f}')
        L.append(f'| {k} | ' + ' | '.join(cells) + ' |')

L.append(f'\n## Checkpoint {ost}: klasy diagramu Q–R dla J3 (bezśladowa część 3 × 3), liczba tokenów (wszystkie / narzędzie / tło)\n')
nk = list(C['blok_0']['klasy_QR3']['wszystkie'].keys())
L.append('| blok | ' + ' | '.join(nk) + ' | Δ>0 (zespolone J3) | J192 z zespolonymi λ |')
L.append('|---|' + '---|' * (len(nk) + 2))
for b in range(6):
    r = C[f'blok_{b}']
    L.append(f'| {b} | ' + ' | '.join(f"{r['klasy_QR3']['wszystkie'][k]} / {r['klasy_QR3']['narzedzie'][k]} / {r['klasy_QR3']['tlo'][k]}"
                                      for k in nk)
             + f" | {r['zespolone']['J3_ognisko_Delta_dodatnia']} (zgodne z widmem J3: {r['zespolone']['zgodnosc_Delta_z_widmem_J3']}/600)"
             + f" | {r['zespolone']['J192']} |")

L.append(f'\n## Checkpoint {ost}: typ znakowy Re λ (niezerowe λ), J3 vs J192, liczba tokenów (wszystkie / narzędzie / tło)\n')
L.append('| blok | J3 stabilny | J3 siodło | J3 niestabilny | J192 stabilny | J192 siodło | J192 niestabilny |')
L.append('|---|' + '---|' * 6)
for b in range(6):
    t = C[f'blok_{b}']['typ_znakow']
    cel = []
    for zr in ('J3', 'J192'):
        for k in ('stabilny', 'siodlo', 'niestabilny'):
            cel.append(f"{t[zr]['wszystkie'][k]} / {t[zr]['narzedzie'][k]} / {t[zr]['tlo'][k]}")
    L.append(f'| {b} | ' + ' | '.join(cel) + ' |')

L.append(f'\n## Checkpoint {ost}: tabela krzyżowa klasa Q–R (J3) × typ znakowy J192 (kolumny: stabilny / siodło / niestabilny)\n')
L.append('| blok | ' + ' | '.join(nk) + ' |')
L.append('|---|' + '---|' * len(nk))
for b in range(6):
    tab = C[f'blok_{b}']['tabela_klasaQR3_x_typ192']
    L.append(f'| {b} | ' + ' | '.join('/'.join(str(x) for x in row) for row in tab) + ' |')

if len(kroki) > 1:
    L.append('\n## Trend z treningiem: mediana po tokenach (wszystkie), blok × checkpoint\n')
    for zr in ('J192', 'J3'):
        for k in TREND:
            L.append(f'\n**{zr} {k}**\n')
            L.append('| blok | ' + ' | '.join(f'k{kr}' for kr in kroki) + ' |')
            L.append('|---|' + '---|' * len(kroki))
            for b in range(6):
                L.append(f'| {b} | ' + ' | '.join(f(v) for v in S['trend_mediany']['wszystkie'][zr][k][b]) + ' |')
    L.append('\n## Trend: mediany narzędzie / tło (pełne J), blok × checkpoint\n')
    for k in ('N_do_J', 'Om_do_S', 'Q', 'R', 'n_1pl_gt1'):
        L.append(f'\n**J192 {k}** (narzędzie / tło)\n')
        L.append('| blok | ' + ' | '.join(f'k{kr}' for kr in kroki) + ' |')
        L.append('|---|' + '---|' * len(kroki))
        for b in range(6):
            n_ = S['trend_mediany']['narzedzie']['J192'][k][b]; t_ = S['trend_mediany']['tlo']['J192'][k][b]
            L.append(f'| {b} | ' + ' | '.join(f'{f(a)} / {f(c)}' for a, c in zip(n_, t_)) + ' |')

# różnice narzędzie − tło: mediana różnicy i Mann–Whitney U (dwustronny), ostatni checkpoint
import sys
import numpy as np
from scipy.stats import mannwhitneyu
Z = np.load(DANE / f'miary_{ost}.npz')
nazwy = [str(x) for x in Z['nazwy']]; nz = Z['narz'].astype(bool)
for zr, m in (('J192', Z['m192']), ('J3', Z['m3'])):
    L.append(f'\n## Checkpoint {ost}: narzędzie − tło ({zr}): mediana(narz) − mediana(tło) [p Manna–Whitneya, dwustronny]\n')
    L.append('| miara | ' + ' | '.join(f'blok {b}' for b in range(6)) + ' |')
    L.append('|---|' + '---|' * 6)
    for k in GLOWNE:
        i = nazwy.index(k); cel = []
        for b in range(6):
            if np.ptp(m[b, :, i]) == 0:
                cel.append('–'); continue
            a_, c_ = m[b, nz, i], m[b, ~nz, i]
            pv = mannwhitneyu(a_, c_, alternative='two-sided').pvalue
            cel.append(f'{f(float(np.median(a_) - np.median(c_)))} [{pv:.1e}]')
        L.append(f'| {k} | ' + ' | '.join(cel) + ' |')

# odniesienie losowe: macierze Ginibre'a (iid N(0,1)) n × n, te same miary (seed 0)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Kod_eksperymentu'))
import schur_miary as sm                                                     # noqa: E402
rng = np.random.default_rng(0)
L.append('\n## Odniesienie: macierze losowe Ginibre’a (iid N(0,1)), mediany miar niezależnych od skali (seed 0)\n')
L.append('| n | liczba macierzy | N_do_J | kappa | Om_do_S | udzN_w_S2 | udzN_w_Om2 | n_im | odsetek Δ>0 (tylko n = 3) |')
L.append('|---|---|---|---|---|---|---|---|---|')
for n, ile in ((192, 200), (3, 20000)):
    mm, _ = sm.miary_wsad(rng.standard_normal((ile, n, n)))
    med = {k: float(np.median(mm[:, sm.NAZWY.index(k)])) for k in ('N_do_J', 'kappa', 'Om_do_S', 'udzN_w_S2', 'udzN_w_Om2', 'n_im')}
    dq = '–'
    if n == 3:
        D, _ = sm.klasa_qr3(mm[:, sm.NAZWY.index('Q')], mm[:, sm.NAZWY.index('R')]); dq = f'{float((D > 0).mean()):.3f}'
    L.append(f'| {n} | {ile} | ' + ' | '.join(f(med[k]) for k in med) + f' | {dq} |')

L.append('\n## Kontrole numeryczne\n')
L.append('| checkpoint | maks. błąd wzgl. J3 vs jak_n aplikacji (6 bloków) | F surowe vs moduł timm |')
L.append('|---|---|---|')
for kr in kroki:
    c = S['checkpointy'][f'k{kr}']
    L.append(f"| k{kr} | {max(c[f'blok_{b}']['zgodnosc_J3_z_jak_n'] for b in range(6)):.1e} | {c['sprawdzenie_F_z_timm']:.1e} |")
L.append('\n| checkpoint | czas przejścia + J [s] | Schur/eig/SVD 192 [s] | 3 × 3 [s] |')
L.append('|---|---|---|---|')
for kr in kroki:
    t = S['czasy_s'][f'k{kr}']
    L.append(f"| k{kr} | {t['przejscie_i_jakobiany_s']:.1f} | {t['schur_eig_svd_192_s']:.1f} | {t['miary_3x3_s']:.1f} |")
(DANE / 'tabele.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('zapisano', DANE / 'tabele.md', len(L), 'wierszy')
