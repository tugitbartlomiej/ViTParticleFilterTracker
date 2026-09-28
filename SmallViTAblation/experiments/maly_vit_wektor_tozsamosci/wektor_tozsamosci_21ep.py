#!/usr/bin/env python3
"""Wektor tożsamości narzędzia po 21 epokach (krok 1000) na sondach — kierunek w latencie zamiast pojedynczych neuronów.

Dane: dla każdej z 192 sond średni latent (768 liczb) na tokenach narzędzia i na tokenach tła.
Pozytywy: narzędzie w 15 wersjach × 12 wycinków (180). Negatywy: tło wszystkich sond + obszar narzędzia w `empty_black`.
Wektor: liniowy rozdzielacz narzędzie/tło (LDA z automatycznym ściąganiem kowariancji, bo próbek jest mniej niż
wymiarów) na latentach wystandaryzowanych negatywami. Wektor „zapala się” na sondzie, gdy rzut przekracza
średnią + 2 sd rzutów negatywów zbioru uczącego (ta sama reguła co dla neuronów).

Sprawdzenia (bez podglądania):
  - wersja wyłączona: wektor uczony bez jednej wersji (bez jej narzędzia i jej tła), sprawdzany na tej wersji;
  - wycinek wyłączony: uczony bez jednego z 12 wycinków (wszystkie jego wersje), sprawdzany na nim;
  - rdzeń wektorowy (jak definicja użytkownika): wektor zapala się na ≥ 75 % wycinków w KAŻDEJ wersji wyłączonej;
  - odniesienia: ten sam test na sieci nieuczonej (krok 0) i prosty wektor „średnia narzędzia − średnia tła”.

    py -3.11 wektor_tozsamosci_21ep.py  ->  wyniki_eden/WEKTOR_21EP.json, WEKTOR_21EP.npz, WEKTOR_21EP.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
W = HERE / 'wyniki_eden'
sys.path.insert(0, str(HERE / 'kod'))
import tozsamosc as TZ  # noqa: E402

PROBE = HERE.parent / 'maly_vit_od_zera' / 'eden' / 'wejscia' / 'probe'
KROK = 1000


def zbior(L, obecn, war, zrodlo, maska_sond):
    """Próbki (X, y, wersja, źródło, czy_tlo) z sond wybranych maską. L: (192, 2, 768)."""
    X, y, wv, zr = [], [], [], []
    for i in np.where(maska_sond)[0]:
        if obecn[i]:
            X.append(L[i, 0]); y.append(1); wv.append(war[i]); zr.append(zrodlo[i])
        else:                                                         # empty_black: obszar narzędzia bez narzędzia
            X.append(L[i, 0]); y.append(0); wv.append(war[i]); zr.append(zrodlo[i])
        X.append(L[i, 1]); y.append(0); wv.append(war[i]); zr.append(zrodlo[i])  # tło sondy
    return np.array(X, np.float64), np.array(y), np.array(wv), np.array(zr)


def naucz(X, y, prosty=False):
    neg = X[y == 0]
    mu, sd = neg.mean(0), neg.std(0) + 1e-6
    Z = (X - mu) / sd
    if prosty:
        w = Z[y == 1].mean(0) - Z[y == 0].mean(0)
    else:
        w = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Z, y).coef_[0]
    w = w / np.linalg.norm(w)
    r = Z[y == 0] @ w
    return dict(mu=mu, sd=sd, w=w, prog=float(r.mean() + TZ.PROG_SD * r.std()))


def rzut(m, X):
    return ((X - m['mu']) / m['sd']) @ m['w']


def ocen(L, obecn, war, zrodlo, prosty=False):
    wersje, zrodla = list(TZ.WARIANTY_Z), sorted(set(zrodlo))
    wszystkie = np.ones(len(war), bool)
    out = dict(wersja_wylaczona={}, wycinek_wylaczony={})
    for v in wersje:                                                  # wersja wyłączona
        m = naucz(*zbior(L, obecn, war, zrodlo, war != v)[:2], prosty=prosty)
        X, y, _, _ = zbior(L, obecn, war, zrodlo, war == v)
        p = rzut(m, X)
        Xn, yn, _, _ = zbior(L, obecn, war, zrodlo, war == 'empty_black')
        auc = roc_auc_score(np.r_[y, yn[yn == 0]], np.r_[p, rzut(m, Xn[yn == 0])])
        out['wersja_wylaczona'][v] = dict(zapala_sie=float((p[y == 1] > m['prog']).mean()),
                                          tlo_zapala=float((p[y == 0] > m['prog']).mean()), auc=float(auc))
    for s in zrodla:                                                  # wycinek wyłączony
        m = naucz(*zbior(L, obecn, war, zrodlo, zrodlo != s)[:2], prosty=prosty)
        X, y, wv, _ = zbior(L, obecn, war, zrodlo, zrodlo == s)
        p = rzut(m, X)
        out['wycinek_wylaczony'][s] = dict(zapala_sie=float((p[y == 1] > m['prog']).mean()),
                                           tlo_zapala=float((p[y == 0] > m['prog']).mean()),
                                           auc=float(roc_auc_score(y, p)))
    ww = out['wersja_wylaczona']
    out['rdzen_wektorowy'] = bool(min(x['zapala_sie'] for x in ww.values()) >= TZ.RDZEN_MIN)
    out['najslabsza_wersja'] = min(ww, key=lambda k: ww[k]['zapala_sie'])
    out['min_zapalenie_po_wersjach'] = float(min(x['zapala_sie'] for x in ww.values()))
    out['wycinek_srednio_zapala'] = float(np.mean([x['zapala_sie'] for x in out['wycinek_wylaczony'].values()]))
    out['wycinek_srednio_auc'] = float(np.mean([x['auc'] for x in out['wycinek_wylaczony'].values()]))
    m_all = naucz(*zbior(L, obecn, war, zrodlo, wszystkie)[:2], prosty=prosty)
    return out, m_all


def main() -> int:
    man = json.loads((PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    war = np.array([m['variant'] for m in man]); zrodlo = np.array([m['source_file'] for m in man])
    wyn, wektory, rys = {}, {}, {}
    for d in sorted(W.glob('petla_s*')):
        z1, z2 = np.load(d / 'wspolny' / 'k01' / 'LATENT.npz'), np.load(d / 'wspolny' / 'k02' / 'LATENT.npz')
        assert z2['kroki'][-1] == KROK and z1['kroki'][0] == 0
        L21 = z2['latent'][-5:].astype(np.float64).mean(0)
        L0 = z1['latent'][0].astype(np.float64)
        r = {}
        for nazwa, L, prosty in (('lda_21ep', L21, False), ('prosty_21ep', L21, True), ('lda_siec_nieuczona', L0, False)):
            o, m = ocen(L, obecn, war, zrodlo, prosty)
            r[nazwa] = o
            if nazwa == 'lda_21ep':
                wektory[d.name] = m['w']
                X, y, wv, _ = zbior(L, obecn, war, zrodlo, np.ones(len(war), bool))
                rys[d.name] = (rzut(m, X), y, wv, m['prog'])
        wyn[d.name] = r
        a, b, c = r['lda_21ep'], r['prosty_21ep'], r['lda_siec_nieuczona']
        print(f"{d.name}: wektor LDA — najsłabsza wersja wyłączona {a['najslabsza_wersja']} {a['min_zapalenie_po_wersjach']:.2f}, "
              f"rdzeń {'TAK' if a['rdzen_wektorowy'] else 'nie'}, nowy wycinek zapala {a['wycinek_srednio_zapala']:.2f} "
              f"(AUC {a['wycinek_srednio_auc']:.2f}) | prosty: min {b['min_zapalenie_po_wersjach']:.2f} | "
              f"nieuczona: min {c['min_zapalenie_po_wersjach']:.2f}, nowy wycinek {c['wycinek_srednio_zapala']:.2f}", flush=True)
    sr = {}
    for nazwa in ('lda_21ep', 'prosty_21ep', 'lda_siec_nieuczona'):
        sr[nazwa] = {v: dict(zapala_sie=float(np.mean([w_[nazwa]['wersja_wylaczona'][v]['zapala_sie'] for w_ in wyn.values()])),
                             tlo_zapala=float(np.mean([w_[nazwa]['wersja_wylaczona'][v]['tlo_zapala'] for w_ in wyn.values()])),
                             auc=float(np.mean([w_[nazwa]['wersja_wylaczona'][v]['auc'] for w_ in wyn.values()])))
                     for v in TZ.WARIANTY_Z}
        print(nazwa, 'wersja wyłączona, średnio 10 ziaren (zapala się / tło zapala / AUC):')
        print('   ', {v: (round(x['zapala_sie'], 2), round(x['tlo_zapala'], 2), round(x['auc'], 2)) for v, x in sr[nazwa].items()})
    wyn['_srednio'] = sr
    (W / 'WEKTOR_21EP.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    np.savez(W / 'WEKTOR_21EP.npz', **wektory)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5, figsize=(24, 9), constrained_layout=True)
    kol = list(TZ.WARIANTY_Z) + ['empty_black']
    for a, (s, (p, y, wv, prog)) in zip(ax.ravel(), rys.items()):
        for j, v in enumerate(kol):
            k = wv == v
            a.scatter(np.full((k & (y == 0)).sum(), j) + .15, p[k & (y == 0)], s=8, c='tab:gray', alpha=.6)
            a.scatter(np.full((k & (y == 1)).sum(), j) - .15, p[k & (y == 1)], s=8, c='tab:red', alpha=.8)
        a.axhline(prog, color='k', ls=':')
        a.set_xticks(range(len(kol))); a.set_xticklabels(kol, rotation=90, fontsize=7)
        a.set_title(f'{s}: rzut na wektor tożsamości (czerwone narzędzie, szare tło)', fontsize=8)
    fig.suptitle('Wektor tożsamości po 21 epokach (uczony na wszystkich wersjach): rzut każdej sondy; linia = próg zapalenia')
    fig.savefig(W / 'WEKTOR_21EP.png', dpi=105)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
