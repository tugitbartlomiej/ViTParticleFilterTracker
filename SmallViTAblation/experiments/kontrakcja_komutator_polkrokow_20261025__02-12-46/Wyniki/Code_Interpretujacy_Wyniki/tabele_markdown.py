"""Tabele liczb (markdown) z ../Dane/mosty_6_checkpointow.json i ../Dane/wszystkie_checkpointy.npz → ../Dane/tabele.md.
Bez interpretacji. Uruchomienie: W:/Python/3.11/python.exe tabele_markdown.py"""
import json
from pathlib import Path

import numpy as np

DANE = Path(__file__).resolve().parent.parent / 'Dane'
d = json.loads((DANE / 'mosty_6_checkpointow.json').read_text(encoding='utf-8'))
A = np.load(DANE / 'wszystkie_checkpointy.npz')
OST = f'k{d["kroki"][-1]}'
L = []
f3 = lambda v: f'{v:.3f}'


def tab(nagl, wiersze):
    L.append('| ' + ' | '.join(nagl) + ' |')
    L.append('|' + '---|' * len(nagl))
    for w in wiersze:
        L.append('| ' + ' | '.join(str(x) for x in w) + ' |')
    L.append('')


def polkroki_tab(kk):
    c = d['checkpointy'][kk]
    st = c['stany']
    L.append(f'### {kk}: 12 półkroków (stosunki D_(s+1)/D_s na u; r, separacja, PR w stanie s+1)\n')
    w = []
    for p in c['polkroki']:
        s1 = st[p['do_stanu']]
        w.append([p['polkrok'], f'{p["z_stanu"]}→{p["do_stanu"]}', f'{p["rodzaj"]} b{p["blok"]}',
                  f3(p['iloraz_D_u_wszystkie']), f3(p['iloraz_D_u_narzedzie']), f3(p['iloraz_D_u_tlo']), f3(p['iloraz_D_u_narz_tlo']),
                  f3(p['iloraz_D_X_wszystkie']),
                  f3(s1['r_wszystkie']), f3(s1['r_narzedzie']), f3(s1['r_tlo']), f3(s1['cos_srednich_narz_tlo']),
                  f3(s1['sep_u']), f'{p["zmiana_sep_u"]:+.3f}', f3(s1['sep_X']), f'{s1["PR_u_wszystkie"]:.2f}', f'{s1["PR_X_wszystkie"]:.2f}',
                  f'{p["norma_dx"]["narzedzie"][1]:.2f}', f'{p["norma_dx"]["tlo"][1]:.2f}',
                  f3(p['udz_styczna']['narzedzie'][1]), f3(p['udz_styczna']['tlo'][1]), f3(p['udz_styczna_suma']),
                  f3(p['udz_radialna_suma']), f3(p['udz_srednia_suma']),
                  f'{p["radialna_wzgl"]["wszystkie"][1]:+.3f}', f'{p["kat_u_deg"]["narzedzie"][1]:.1f}', f'{p["kat_u_deg"]["tlo"][1]:.1f}',
                  f3(float(np.mean(knn_srednie(kk, p['polkrok'] - 1, 'narzedzie')))),   # średnia (mediana ma krok 0,2)
                  f3(float(np.mean(knn_srednie(kk, p['polkrok'] - 1, 'tlo'))))])
    tab(['#', 's→s+1', 'półkrok', 'D_u wsz.', 'D_u narz.', 'D_u tło', 'D_u n–t', 'D_X wsz.', 'r', 'r narz.', 'r tło', 'cos(μn,μt)',
         'sep_u', 'Δsep_u', 'sep_X', 'PR_u', 'PR_X', 'med‖Δx‖ narz.', 'med‖Δx‖ tło', 'udz. styczny narz. (med)',
         'udz. styczny tło (med)', 'udz. styczny Σ', 'udz. radialny Σ', 'udz. wzdłuż 1 Σ', 'med Δ‖c‖/‖c‖', 'kąt u narz. [°]',
         'kąt u tło [°]', 'kNN zach. narz.', 'kNN zach. tło'], w)


_npz = {}


def knn_srednie(kk, s, g):
    if kk not in _npz:
        _npz[kk] = np.load(DANE / f'mosty_{kk}.npz')
    z = _npz[kk]
    m = z['narz'] if g == 'narzedzie' else ~z['narz']
    return z['tok_knn_zachowane_u'][s][m]


def stany_tab(kk):
    st = d['checkpointy'][kk]['stany']
    L.append(f'### {kk}: stany (0–12; „glowa” = net.norm(X_12))\n')
    w = []
    for s, e in enumerate(st):
        w.append(['glowa' if s == 13 else s, f3(e['D_u_wszystkie']), f3(e['D_u_narzedzie']), f3(e['D_u_tlo']), f3(e['D_u_narz_tlo']),
                  f'{e["D_X_wszystkie"]:.2f}', f'{e["norma_srednia_X"]:.2f}', f3(e['r_wszystkie']), f3(e['r_narzedzie']), f3(e['r_tlo']),
                  f3(e['cos_srednich_narz_tlo']), f3(e['sep_u']), f3(e['sep_os_u']), f3(e['sep_X']), f3(e['sep_os_X']),
                  f'{e["PR_u_wszystkie"]:.2f}', f'{e["PR_u_narzedzie"]:.2f}', f'{e["PR_u_tlo"]:.2f}', f'{e["PR_X_wszystkie"]:.2f}'])
    tab(['stan', 'D_u', 'D_u narz.', 'D_u tło', 'D_u n–t', 'D_X', 'śr. ‖x‖', 'r', 'r narz.', 'r tło', 'cos(μn,μt)', 'sep_u', 'sep_os_u',
         'sep_X', 'sep_os_X', 'PR_u', 'PR_u narz.', 'PR_u tło', 'PR_X'], w)


def agregat_tab():
    L.append('### Iloczyny stosunków D_u po 6 półkrokach uwagi i 6 MLP (= wkład do D_u,12/D_u,0), 6 checkpointów\n')
    w = []
    for kk, c in d['checkpointy'].items():
        pk = c['polkroki']; st = c['stany']
        row = [kk]
        for g in ('wszystkie', 'narzedzie', 'tlo'):
            pa = np.prod([p[f'iloraz_D_u_{g}'] for p in pk if p['rodzaj'] == 'uwaga'])
            pm = np.prod([p[f'iloraz_D_u_{g}'] for p in pk if p['rodzaj'] == 'MLP'])
            row += [f3(pa), f3(pm)]
        na = sum(p['iloraz_D_u_wszystkie'] < 1 for p in pk if p['rodzaj'] == 'uwaga')
        nm = sum(p['iloraz_D_u_wszystkie'] < 1 for p in pk if p['rodzaj'] == 'MLP')
        sa = sum(p['zmiana_sep_u'] for p in pk if p['rodzaj'] == 'uwaga'); sm = sum(p['zmiana_sep_u'] for p in pk if p['rodzaj'] == 'MLP')
        row += [f'{na}/6', f'{nm}/6', f'{sa:+.3f}', f'{sm:+.3f}', f3(st[0]['sep_u']), f3(st[12]['sep_u']), f3(st[0]['r_wszystkie']),
                f3(st[12]['r_wszystkie']), f3(st[12]['cos_srednich_narz_tlo']), f3(c['wyjscie_S']['iou']), f3(c['wyjscie_S']['max_cieplo'])]
        w.append(row)
    tab(['checkpoint', 'Π uwaga wsz.', 'Π MLP wsz.', 'Π uwaga narz.', 'Π MLP narz.', 'Π uwaga tło', 'Π MLP tło', 'uwaga ściska',
         'MLP ściska', 'Σ Δsep_u uwaga', 'Σ Δsep_u MLP', 'sep_u 0', 'sep_u 12', 'r_0', 'r_12', 'cos_12', 'IoU', 'maks. ciepło'], w)


def trend_wszystkie():
    kr, ep = A['kroki'], A['epoki']
    L.append(f'### Wszystkie checkpointy (K = {len(kr)}): wybrane epoki (najbliższy checkpoint)\n')
    w = []
    for e in (0, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 6, 10, 15, 20, 30, 40, 50, 60):
        i = int(np.argmin(np.abs(ep - e)))
        Du = A['D_u_wszystkie'][i]
        ra = np.prod(Du[1::2] / Du[0:-1:2]); rm = np.prod(Du[2::2] / Du[1::2])
        w.append([int(kr[i]), f'{ep[i]:.2f}', f3(A['r_wszystkie'][i, 12]), f3(A['r_narzedzie'][i, 12]), f3(A['r_tlo'][i, 12]),
                  f3(A['cos_srednich_narz_tlo'][i, 12]), f3(A['sep_u'][i, 12]), f3(A['sep_X'][i, 12]), f3(A['sep_u_glowa'][i]),
                  f3(Du[12] / Du[0]), f3(ra), f3(rm), f3(A['D_u_narzedzie'][i, 12] / A['D_u_narzedzie'][i, 0]),
                  f'{A["PR_u_wszystkie"][i, 12]:.2f}', f3(A['iou'][i]), f3(A['max_cieplo'][i])])
    tab(['krok', 'epoka', 'r_12', 'r_12 narz.', 'r_12 tło', 'cos_12', 'sep_u,12', 'sep_X,12', 'sep_u glowa', 'D_u,12/D_u,0',
         'Π uwaga', 'Π MLP', 'D_u narz. 12/0', 'PR_u,12', 'IoU', 'maks. ciepło'], w)
    Du = A['D_u_wszystkie']
    lr = np.log(Du[:, 1:] / Du[:, :-1])
    L.append('Udział checkpointów (wszystkie 806), w których półkrok ściska D_u (stosunek < 1), i mediana stosunku:\n')
    tab(['półkrok'] + [f'{s}→{s + 1} ({"A" if s % 2 == 0 else "M"}{s // 2})' for s in range(12)],
        [['odsetek < 1'] + [f'{(lr[:, s] < 0).mean():.2f}' for s in range(12)],
         ['mediana stosunku'] + [f3(float(np.exp(np.median(lr[:, s])))) for s in range(12)],
         ['mediana (epoka ≥ 6)'] + [f3(float(np.exp(np.median(lr[A['epoki'] >= 6, s])))) for s in range(12)]])


def komutator_tab(kk, krotko=False):
    c = d['checkpointy'][kk]
    w = []
    for b in c['komutator']:
        v = b['warianty']
        if krotko:
            w.append([kk, b['blok'], f3(b['SR_do_SX']['wszystkie'][1]), f3(b['SP_do_SX']['wszystkie'][1]), f3(b['cos_SX_PX']['wszystkie'][1]),
                      f3(v['S_prawdziwa']['iou']), f3(v['R_odwrocona']['iou']), f3(v['P_rownolegla']['iou']), f3(v['tylko_uwaga']['iou']),
                      f3(v['tylko_MLP']['iou']), f3(v['bez_bloku']['iou'])])
        else:
            w.append([b['blok'], f3(b['SR_do_SX']['wszystkie'][1]), f3(b['SR_do_SX']['narzedzie'][1]), f3(b['SR_do_SX']['tlo'][1]),
                      f3(b['SP_do_SX']['wszystkie'][1]), f3(b['SP_do_SX']['narzedzie'][1]), f3(b['SP_do_SX']['tlo'][1]),
                      f3(b['cos_SX_PX']['wszystkie'][1]), f3(b['cos_SX_PX']['narzedzie'][1]), f3(b['cos_SX_PX']['tlo'][1]),
                      f3(b['globalnie']['SR_do_SX']), f3(b['globalnie']['SP_do_SX'])]
                     + [f'{f3(v[n]["iou"])} / {v[n]["max_cieplo"]:.3f} / {v[n]["korelacja_mapy_z_S"]:.3f}'
                        for n in ('R_odwrocona', 'P_rownolegla', 'tylko_uwaga', 'tylko_MLP', 'bez_bloku')])
    return w


def main():
    L.append(f'# Tabele liczb (generowane przez Wyniki/Code_Interpretujacy_Wyniki/tabele_markdown.py)\n')
    L.append(f'Narzędzie: {d["n_narz"]} tokenów, tło: {d["n_tlo"]}; {d["etykieta_narzedzia"]}. Mediany po tokenach, jeśli nie zaznaczono.\n')
    L.append('## Most 2\n')
    polkroki_tab(OST)
    stany_tab(OST)
    agregat_tab()
    for kk in d['checkpointy']:
        if kk != OST:
            polkroki_tab(kk)
    trend_wszystkie()
    L.append('## Most 3\n')
    S = d['checkpointy'][OST]['wyjscie_S']
    L.append(f'### {OST}: komutator na blok (S: IoU {S["iou"]:.3f}, maks. ciepło {S["max_cieplo"]:.3f}); '
             'warianty: IoU / maks. ciepło / korelacja mapy z S\n')
    tab(['blok', '‖S−R‖/‖S−X‖', 'narz.', 'tło', '‖S−P‖/‖S−X‖', 'narz.', 'tło', 'cos(S−X,P−X)', 'narz.', 'tło',
         '‖S−R‖_F/‖S−X‖_F', '‖S−P‖_F/‖S−X‖_F', 'R odwrócona', 'P równoległa', 'tylko uwaga', 'tylko MLP', 'bez bloku'],
        komutator_tab(OST))
    L.append('### Pozostałe checkpointy (mediany; IoU ramki szczytu z GT)\n')
    tab(['checkpoint', 'blok', '‖S−R‖/‖S−X‖', '‖S−P‖/‖S−X‖', 'cos(S−X,P−X)', 'IoU S', 'IoU R', 'IoU P', 'IoU tylko uwaga',
         'IoU tylko MLP', 'IoU bez bloku'],
        sum((komutator_tab(kk, True) for kk in d['checkpointy']), []))
    (DANE / 'tabele.md').write_text('\n'.join(L), encoding='utf-8')
    print('zapisano', DANE / 'tabele.md')


if __name__ == '__main__':
    main()
