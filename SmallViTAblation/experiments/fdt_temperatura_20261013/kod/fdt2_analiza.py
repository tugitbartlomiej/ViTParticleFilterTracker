"""Analiza FDT w podprzestrzeniach (fdt_podprzestrzenie.py → FDT2.npz): widoczna V (192, przestrzeń wierszy fc2) i niewidoczna N (576).

Dla każdej podprzestrzeni K ∈ {V, N} (średnie po tokenach; m_K = ε_K·(z(t) − z(t_w))):
  χ_K(t)      odpowiedź własna na pole jednostkowe w K;   χ3/χ (tylko N) — liniowość;
  Δ_K(t)      = ½⟨m_K²⟩ — przemieszczenie z szumu i dryfu wzdłuż jednego kierunku w K;  Δ_K,szum = ¼⟨(m_a − m_b)²⟩;
  T_K(t)      = Δ_K/χ_K — temperatura efektywna (stała w równowadze);
  wykładniki potęgowe χ ∝ t^a, Δ ∝ t^b w oknach;  „sprężystość”: czy χ się nasyca (pułapka) czy rośnie (brak siły zwrotnej).
Przeciek: |odpowiedź na pole K| w podprzestrzeni V i N (normy pełnych wektorów odpowiedzi).
Uruchomienie: python fdt2_analiza.py <katalog> [<katalog> ...] → FDT2_ANALIZA.json w każdym i fdt2.png w pierwszym
"""
import json
import os
import sys

import numpy as np


def wykladnik(t, y, od, do):
    m = (t >= od) & (t <= do) & (y > 0)
    if m.sum() < 3:
        return None
    return float(np.polyfit(np.log(t[m]), np.log(y[m]), 1)[0])


def analiza(kat):
    d = np.load(os.path.join(kat, 'FDT2.npz'))
    meta = json.load(open(os.path.join(kat, 'meta.json'), encoding='utf-8'))
    t = d['kroki']; n = len(t)
    sr = lambda x: x.reshape(n, -1).mean(1)
    wyn = dict(nazwa=os.path.basename(kat.rstrip('/\\')), lr=meta['lr'], krok_tw=meta['krok_tw'], beta1=meta.get('beta1'), wd=meta.get('wd'),
               h0=meta['h0'], podprzestrzenie={})
    krzywe = dict(t=t)
    for K in ('V', 'N'):
        chi = sr(d[f'd1_{K}']); D = 0.5 * sr(d[f'm_a_{K}'] ** 2); Ds = 0.25 * sr((d[f'm_a_{K}'] - d[f'm_b_{K}']) ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            T = D / chi; Ts = Ds / chi
        lin = sr(d[f'd3_{K}']) / chi if f'd3_{K}' in d else None
        nielin = np.sqrt(sr(d[f's1_{K}'] ** 2)) / np.maximum(np.abs(chi * meta['h0'][K]), 1e-30)
        tab = {}
        for k in (1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000):
            if k > t[-1]:
                continue
            i = int(np.searchsorted(t, k))
            tab[k] = dict(chi=float(chi[i]), Delta=float(D[i]), Delta_szum=float(Ds[i]), T=float(T[i]), T_szum=float(Ts[i]),
                          chi3_do_chi=(float(lin[i]) if lin is not None else None), nieliniowosc=float(nielin[i]))
        wyn['podprzestrzenie'][K] = dict(tabela=tab, wykladnik_chi=dict(krotkie=wykladnik(t, chi, 2, 20), srednie=wykladnik(t, chi, 20, 200), dlugie=wykladnik(t, chi, 300, t[-1])),
                                         wykladnik_Delta=dict(krotkie=wykladnik(t, D, 2, 20), srednie=wykladnik(t, D, 20, 200), dlugie=wykladnik(t, D, 300, t[-1])))
        krzywe.update({f'chi_{K}': chi, f'D_{K}': D, f'Ds_{K}': Ds, f'T_{K}': T})
    # przemieszczenia w pełnych podprzestrzeniach na jeden wymiar (izotropia wewnątrz podprzestrzeni)
    msdV = sr(d['msdV_a']) / 192 / 2; msdN = sr(d['msdN_a']) / 576 / 2
    abV = sr(d['abV']) / 192 / 4; abN = sr(d['abN']) / 576 / 4
    wyn['przemieszczenie_na_wymiar'] = {int(k): dict(V=float(msdV[int(np.searchsorted(t, k))]), N=float(msdN[int(np.searchsorted(t, k))]),
                                                     V_szum=float(abV[int(np.searchsorted(t, k))]), N_szum=float(abN[int(np.searchsorted(t, k))]))
                                        for k in (10, 100, 300, 1000, 2000) if k <= t[-1]}
    wyn['przeciek'] = {k: float(sr(d[k])[-1]) for k in ('respV_wV', 'respV_wN', 'respN_wV', 'respN_wN')}
    krzywe.update(msdV=msdV, msdN=msdN)
    return wyn, krzywe


def main():
    kats = sys.argv[1:]
    W, C = [], []
    for k in kats:
        w_, c_ = analiza(k); W.append(w_); C.append(c_)
        json.dump(w_, open(os.path.join(k, 'FDT2_ANALIZA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f"== {w_['nazwa']} (lr {w_['lr']}, t_w {w_['krok_tw']}, β1 {w_['beta1']}, wd {w_['wd']})")
        for K in ('V', 'N'):
            p = w_['podprzestrzenie'][K]
            print(f'  {K}: wykładnik χ {p["wykladnik_chi"]}, Δ {p["wykladnik_Delta"]}')
            for kk, r in p['tabela'].items():
                if kk in (1, 10, 100, 300, 1000, 2000, 3000):
                    print(f'     t={kk:5d}  χ {r["chi"]:.4g}  Δ {r["Delta"]:.4g} (szum {r["Delta_szum"]:.4g})  T {r["T"]:.4g}  T_szum {r["T_szum"]:.4g}'
                          + (f'  χ3/χ {r["chi3_do_chi"]:.3f}' if r['chi3_do_chi'] is not None else ''))
        print('  przemieszczenie na wymiar:', w_['przemieszczenie_na_wymiar'])
        print('  przeciek (normy odpowiedzi):', w_['przeciek'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for w_, c in zip(W, C):
        lab = f"{w_['nazwa']}"
        for j, K in enumerate(('V', 'N')):
            ls = '-' if K == 'V' else '--'
            ax[0, 0].loglog(c['t'][1:], c[f'chi_{K}'][1:], ls, label=f'{lab} {K}')
            ax[0, 1].loglog(c['t'][1:], c[f'D_{K}'][1:], ls, label=f'{lab} {K}')
            ax[0, 2].loglog(c['t'][1:], c[f'T_{K}'][1:], ls, label=f'{lab} {K}')
            ax[1, j].plot(c[f'D_{K}'], c[f'chi_{K}'], '.-', ms=2, label=lab)
        ax[1, 2].loglog(c['t'][1:], c['msdV'][1:], '-', label=f'{lab} V'); ax[1, 2].loglog(c['t'][1:], c['msdN'][1:], '--', label=f'{lab} N')
    ax[0, 0].set_title('χ(t): odpowiedź na stałe pole (V ciągła, N przerywana)'); ax[0, 1].set_title('Δ(t): przemieszczenie wzdłuż kierunku pola')
    ax[0, 2].set_title('T_eff = Δ/χ'); ax[1, 0].set_title('wykres FDT, V (widoczna)'); ax[1, 1].set_title('wykres FDT, N (niewidoczna)')
    ax[1, 2].set_title('przemieszczenie na wymiar w podprzestrzeni')
    for a_ in ax.ravel():
        a_.grid(alpha=.3); a_.legend(fontsize=6); a_.set_xlabel('krok od t_w')
    ax[1, 0].set_xlabel('Δ'); ax[1, 1].set_xlabel('Δ'); ax[1, 0].set_ylabel('χ'); ax[1, 1].set_ylabel('χ')
    plt.tight_layout(); plt.savefig(os.path.join(kats[0], 'fdt2.png'), dpi=100)


if __name__ == '__main__':
    main()
