"""Analiza FDT (fdt_odpowiedz.py → FDT.npz): wykres χ od Δ, temperatura efektywna, rozkład na szum i dryf, kontrole.

Oznaczenia (średnie po wszystkich tokenach sond; m = ε·(z(t) − z(t_w)), jedna składowa wzdłuż losowego kierunku pola):
  χ(t)       = ⟨d1⟩, odpowiedź własna na jednostkowe pole (h0); χ3 = ⟨d3⟩ z pola 3·h0 (liniowość: χ3/χ ≈ 1);
  Δ(t)       = ½⟨m_a²⟩, przemieszczenie wzdłuż ε (pełne: szum + dryf wspólny dla strumieni paczek);
  Δ_szum(t)  = ¼⟨(m_a − m_b)²⟩, część od szumu paczek (wariancja między strumieniami / 2 · ½);
  Δ_dryf     = Δ − Δ_szum, część powtarzalna przy innej kolejności paczek (dryf pułapki);
  T_eff(t)   = Δ/χ (FDT w równowadze: stała, równa temperaturze kąpieli); T_szum = Δ_szum/χ.
Kontrole: bliźniak (te same paczki, h = 0) — poziom niedeterminizmu wobec odpowiedzi; izotropia: ½⟨m²⟩ wobec ⟨|Δz|²⟩/(2·768).
Uruchomienie: python fdt_analiza.py <katalog z FDT.npz> [kolejne katalogi ...] → FDT_ANALIZA.json i fdt.png w pierwszym katalogu
"""
import json
import os
import sys

import numpy as np


def analiza(kat):
    d = np.load(os.path.join(kat, 'FDT.npz'))
    meta = json.load(open(os.path.join(kat, 'meta.json'), encoding='utf-8'))
    t = d['kroki']
    sr = lambda x: x.reshape(len(t), -1).mean(1)
    chi = sr(d['d1']); chi3 = sr(d['d3'])
    Dl = 0.5 * sr(d['m_a'] ** 2)
    Ds = 0.25 * sr((d['m_a'] - d['m_b']) ** 2)
    Dd = Dl - Ds
    msd768 = sr(d['msd_a']) / (2 * 768)
    h0 = meta['h0']
    blizniak = np.sqrt(sr(d['blizniak'] ** 2)); odp = np.sqrt(sr((d['d1'] * h0) ** 2))
    nielin = np.sqrt(sr(d['s1'] ** 2)) / np.maximum(np.abs(chi * h0), 1e-30)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = Dl / chi; Ts = Ds / chi
    # odpowiedź pojedynczych tokenów: mapa χ_j na końcu (średnia z ostatnich 10 % zapisów) i jej szum
    n = len(t); ost = slice(int(n * 0.9), n)
    chi_j = d['d1'][ost].mean(0).ravel(); chi3_j = d['d3'][ost].mean(0).ravel()
    zgodnosc_j = float(np.corrcoef(chi_j, chi3_j)[0, 1])
    m2_j = 0.5 * (d['m_a'][ost] ** 2).mean(0).ravel()

    def w(x, k):
        i = int(np.searchsorted(t, k)); i = min(i, n - 1)
        return float(x[i])

    punkty = [k for k in (1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 3000) if k <= t[-1]]
    tab = {int(k): dict(chi=w(chi, k), chi3_do_chi=w(chi3, k) / max(w(chi, k), 1e-30), Delta=w(Dl, k), Delta_szum=w(Ds, k), Delta_dryf=w(Dd, k),
                        T_eff=w(T, k), T_szum=w(Ts, k), izotropia=w(Dl, k) / max(w(msd768, k), 1e-30), blizniak_do_odp=w(blizniak, k) / max(w(odp, k), 1e-30),
                        nieliniowosc=w(nielin, k)) for k in punkty}
    # nachylenia wykresu χ(Δ) w dwóch zakresach: krótkie czasy (do 20 kroków) i długie (od 300)
    def nachylenie(m):
        if m.sum() < 3:
            return None
        return float(np.polyfit(Dl[m], chi[m], 1)[0])
    s_kr = nachylenie((t >= 1) & (t <= 20)); s_dl = nachylenie(t >= 300)
    wyn = dict(nazwa=os.path.basename(kat.rstrip('/\\')), lr=meta['lr'], krok_tw=meta['krok_tw'], beta1=meta.get('beta1'), h0=h0, tabela=tab,
               T_krotkie=(1 / s_kr if s_kr else None), T_dlugie=(1 / s_dl if s_dl else None),
               chi_j_zgodnosc_h_3h=zgodnosc_j, chi_j_ujemne=float((chi_j < 0).mean()),
               spearman_chi_j_ruchliwosc=None)
    try:
        from scipy.stats import spearmanr
        wyn['spearman_chi_j_ruchliwosc'] = float(spearmanr(chi_j, m2_j).correlation)
    except Exception:
        pass
    # gorące tokeny czy miękkie pułapki? Grupy tokenów (narzędzie/tło, decyle ruchliwości) i w każdej χ, Δ, T = Δ/χ
    # na końcu okna (ostatnie 10 % zapisów). Ruchliwość = Δ_j z całego okna; T i 1/K rozdzielone dzięki χ.
    S_ = d['d1'].shape[1]
    narz = maska_narzedzia(meta['sondy'], S_)
    grupy = {}
    if narz is not None:
        grupy['narzedzie'] = narz.ravel(); grupy['tlo'] = ~narz.ravel()
    ruch = (0.5 * d['m_a'] ** 2).mean(0).ravel()
    ruch_768 = d['msd_a'].mean(0).ravel()
    dec = np.digitize(ruch_768, np.quantile(ruch_768, np.linspace(0, 1, 11)[1:-1]))
    for q in range(10):
        grupy[f'decyl_ruchliwosci_{q}'] = dec == q
    Ds_j = (0.25 * (d['m_a'][ost] - d['m_b'][ost]) ** 2).mean(0).ravel()
    wyn['grupy'] = {}
    for g, m in grupy.items():
        c_ = float(chi_j[m].mean()); D_ = float(m2_j[m].mean()); Dsz = float(Ds_j[m].mean())
        wyn['grupy'][g] = dict(tokenow=int(m.sum()), chi=c_, chi_blad=float(chi_j[m].std() / np.sqrt(m.sum())), Delta=D_, Delta_szum=Dsz,
                              T=D_ / c_ if c_ > 0 else None, T_szum=Dsz / c_ if c_ > 0 else None, ruchliwosc_768=float(ruch_768[m].mean()))
    krzywe = dict(t=t, chi=chi, chi3=chi3, Delta=Dl, Delta_szum=Ds, Delta_dryf=Dd, T=T, T_szum=Ts)
    return wyn, krzywe


MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'maly_vit_od_zera_35000_21epok_20261009', 'wejscia', 'MANIFEST_35000.jsonl')


def maska_narzedzia(sondy, S_, h=20, w=30, krok=16):
    """(S, 600) bool: środek patcha wewnątrz ramki narzędzia (gt) sondy; ramki skalowane do 480 × 320 jak w wiersze_35000."""
    if not os.path.exists(MANIFEST):
        return None
    wiersze = [json.loads(l) for l in open(MANIFEST, encoding='utf-8')]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (xx.ravel() + .5) * krok, (yy.ravel() + .5) * krok
    out = np.zeros((S_, h * w), bool)
    for i, s in enumerate(sondy[:S_]):
        r = wiersze[s]; sx, sy = 480 / r['width'], 320 / r['height']
        for g in r['gt']:
            x, y, bw, bh = g['bbox_xywh']
            out[i] |= (cx >= x * sx) & (cx <= (x + bw) * sx) & (cy >= y * sy) & (cy <= (y + bh) * sy)
    return out


def main():
    kats = sys.argv[1:]
    wyniki, krzywe = [], []
    for k in kats:
        w_, c_ = analiza(k); wyniki.append(w_); krzywe.append(c_)
        print(json.dumps(w_, ensure_ascii=False, indent=1)[:4000])
    json.dump(wyniki if len(wyniki) > 1 else wyniki[0], open(os.path.join(kats[0], 'FDT_ANALIZA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 3, figsize=(17, 5))
    for w_, c in zip(wyniki, krzywe):
        lab = f"{w_['nazwa']} (lr {w_['lr']:g})"
        ax[0].plot(c['Delta'], c['chi'], '.-', ms=2, label=lab)
        ax[1].loglog(c['t'][1:], c['T'][1:], label=lab + ' całość')
        ax[1].loglog(c['t'][1:], c['T_szum'][1:], '--', label=lab + ' szum')
        ax[2].loglog(c['t'][1:], c['chi'][1:], label=lab + ' χ')
        ax[2].loglog(c['t'][1:], c['Delta'][1:], ':', label=lab + ' Δ')
    ax[0].set_xlabel('Δ = ½⟨(ε·Δz)²⟩'); ax[0].set_ylabel('χ (odpowiedź na pole)'); ax[0].set_title('wykres FDT: prosta = jedna temperatura')
    ax[1].set_xlabel('krok od t_w'); ax[1].set_ylabel('T_eff = Δ/χ'); ax[1].set_title('temperatura efektywna')
    ax[2].set_xlabel('krok od t_w'); ax[2].set_title('odpowiedź i przemieszczenie')
    for a_ in ax:
        a_.legend(fontsize=7); a_.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(kats[0], 'fdt.png'), dpi=110)


if __name__ == '__main__':
    main()
