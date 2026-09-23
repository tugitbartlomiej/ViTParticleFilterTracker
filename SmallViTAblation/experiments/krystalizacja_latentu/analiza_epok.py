#!/usr/bin/env python3
"""Dowod na wielu klatkach: czy w miare uczenia detektor coraz bardziej opiera narzedzie na otoczeniu.

Dane: pipeline/koopman_langevin/tozsamosc_obrot.py uruchomiony na wagach z epok 1, 3, ..., 29 trzech
biegow B2 (1500 klatek, ziarna 42, 1337, 2024) i na wagach biegu z sonda (250 klatek, best i last).
300 klatek walidacyjnych z 25 filmow, ktorych zaden detektor nie widzial.

Warianty i co mierza:
  orig          klatka bez zmian
  rot_*         obrot calego obrazu (inny kat narzedzia)
  dl_*          wydluzenie / skrocenie
  tlo_a, tlo_b  wycinek z narzedziem wklejony na klatke INNEJ operacji (obce otoczenie)
  tlo_film      ten sam wycinek na klatke TEJ SAMEJ operacji, 250+ klatek dalej (podobne otoczenie)
  tlo_swoje     ten sam wycinek wklejony z powrotem na te sama klatke z zamazanym narzedziem
                (szew i zamazana plama sa, otoczenie wlasne) - kontrola artefaktu klejenia

Jesli spadek pewnosci na obcym tle bierze sie z samego klejenia, tlo_swoje spadnie tak samo.
Jesli bierze sie z otoczenia, tlo_swoje zostanie wysoko, tlo_film posrodku, tlo_a/b nisko.

    py -3.11 experiments/krystalizacja_latentu/analiza_epok.py --dane <results_obrot_epoki> \\
        --telemetria-dir experiments/krystalizacja_latentu/dane/biegi --out <katalog dowodow>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

GRUPY = {'wlasne': ['orig'], 'obrot': ['rot_-45', 'rot_-30', 'rot_-15', 'rot_+15', 'rot_+30', 'rot_+45'],
         'wydluzenie': ['dl_0.7', 'dl_1.4'], 'tlo_swoje': ['tlo_swoje'], 'tlo_film': ['tlo_film'],
         'tlo_obce': ['tlo_a', 'tlo_b']}
KOLORY = {'wlasne': 'C0', 'obrot': 'C1', 'wydluzenie': 'C2', 'tlo_swoje': 'C7', 'tlo_film': 'C4', 'tlo_obce': 'C3'}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def zmiana(L, war, grupa):
    o = L[:, war.index('orig')]
    spread = ((o - o.mean(0)) ** 2).sum(1).mean()
    return float(np.mean([((L[:, war.index(g)] - o) ** 2).sum(1).mean() for g in grupa]) / spread)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dane', type=Path, required=True)
    ap.add_argument('--telemetria-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    meta = json.loads((args.dane / 'META.json').read_text(encoding='utf-8'))
    war = meta['warianty']
    wyniki = {}
    for nazwa in meta['detektory']:
        d = np.load(args.dane / f'OBROT_{nazwa}.npz')
        p = d['pewnosc'].astype(float)
        L = d['zapytanie'].astype(float)
        wyniki[nazwa] = dict(
            pew={g: np.median(p[:, [war.index(w) for w in ws]], 0).mean() for g, ws in GRUPY.items()},
            pew_klatki={g: p[:, [war.index(w) for w in ws]].mean(1) for g, ws in GRUPY.items()},
            lg={g: logit(p[:, [war.index(w) for w in ws]]).mean(1) for g, ws in GRUPY.items()},
            R={g: zmiana(L, war, ws) for g, ws in GRUPY.items() if g != 'wlasne'})

    ziarna = sorted({re.search(r'_s(\d+)_', n).group(1) for n in wyniki if n.startswith('b1500')})
    epoki = sorted({int(re.search(r'epoch_(\d+)', n).group(1)) for n in wyniki if n.startswith('b1500')})
    mapy = {}
    for s in ziarna:
        t = json.loads((args.telemetria_dir / f'telemetria_uniform_time_b1500_s{s}' / 'TELEMETRIA.json').read_text())
        mapy[s] = np.array([v for _, v in t['val/mAP_50_95']])
    def seria(klucz, g):
        return np.array([[wyniki[f'b1500_s{s}_epoch_{e:03d}'][klucz][g] for e in epoki] for s in ziarna])

    raport = {'ziarna': ziarna, 'epoki': epoki, 'warianty': war, 'mediana_pewnosci': {}, 'zmiana_latentu': {}}
    for g in GRUPY:
        raport['mediana_pewnosci'][g] = seria('pew', g).mean(0).tolist()
        if g != 'wlasne':
            raport['zmiana_latentu'][g] = seria('R', g).mean(0).tolist()
    najl = {s: int(np.argmax(mapy[s])) + 1 for s in ziarna}                  # epoka (od 1) najlepszego mAP
    raport['epoka_najlepszego_mAP'] = najl

    # test sparowany: roznica logitow (wlasne - obce) w epoce najblizszej najlepszej wobec ostatniej
    testy = {}
    for kontrola in ('tlo_obce', 'tlo_film', 'tlo_swoje', 'obrot'):
        a_, b_ = [], []
        for s in ziarna:
            e_naj = min(epoki, key=lambda e: abs(e - najl[s]))
            w1 = wyniki[f'b1500_s{s}_epoch_{e_naj:03d}']; w2 = wyniki[f'b1500_s{s}_epoch_{epoki[-1]:03d}']
            a_.append(w1['lg']['wlasne'] - w1['lg'][kontrola]); b_.append(w2['lg']['wlasne'] - w2['lg'][kontrola])
        a_, b_ = np.concatenate(a_), np.concatenate(b_)
        w = stats.wilcoxon(b_, a_)
        testy[kontrola] = dict(luka_logitu_najlepsza=float(np.median(a_)), luka_logitu_ostatnia=float(np.median(b_)),
                               klatek_z_wieksza_luka=float((b_ > a_).mean()), p_wilcoxon=float(w.pvalue), n=int(len(a_)))
    raport['luka_najlepsza_vs_ostatnia'] = testy
    for nm in ('sonda250_best', 'sonda250_last'):
        if nm in wyniki:
            raport[nm] = {g: float(v) for g, v in wyniki[nm]['pew'].items()}

    # ---- rysunek
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for g in GRUPY:
        m = seria('pew', g)
        ax[0].plot(epoki, m.mean(0), '-o', ms=3, color=KOLORY[g], label=g)
        ax[0].fill_between(epoki, m.min(0), m.max(0), color=KOLORY[g], alpha=.12)
    ax[0].set_ylabel('mediana pewnosci (300 klatek, 3 ziarna)'); ax[0].legend(fontsize=8, ncol=3)
    for g in ('tlo_obce', 'tlo_film', 'tlo_swoje', 'obrot'):
        luka = np.array([[np.median(wyniki[f'b1500_s{s}_epoch_{e:03d}']['lg']['wlasne'] - wyniki[f'b1500_s{s}_epoch_{e:03d}']['lg'][g])
                          for e in epoki] for s in ziarna])
        ax[1].plot(epoki, luka.mean(0), '-o', ms=3, color=KOLORY[g], label=f'wlasne minus {g}')
        ax[1].fill_between(epoki, luka.min(0), luka.max(0), color=KOLORY[g], alpha=.12)
    ax[1].axhline(0, color='k', lw=.5)
    ax[1].set_ylabel('luka pewnosci (roznica logitow, mediana)'); ax[1].legend(fontsize=8)
    bx = ax[1].twinx()
    for s in ziarna:
        bx.plot(np.arange(1, len(mapy[s]) + 1), mapy[s], 'k-', lw=.8, alpha=.6)
    bx.set_ylabel('mAP walidacji (czarne, 3 ziarna)')
    for s in ziarna:
        for a_ in ax:
            a_.axvline(najl[s], color='k', ls=':', lw=.8)
    for g in ('tlo_obce', 'tlo_film', 'tlo_swoje', 'obrot', 'wydluzenie'):
        m = seria('R', g)
        ax[2].plot(epoki, m.mean(0), '-o', ms=3, color=KOLORY[g], label=g)
    ax[2].set_ylabel('zmiana latentu zapytania (1 = jak inna klatka)'); ax[2].set_xlabel('epoka')
    ax[2].legend(fontsize=8)
    fig.suptitle('Detektor 1500 klatek: pewnosc i latent przy zmianie kata, dlugosci i otoczenia narzedzia')
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'EPOKI_otoczenie.png', dpi=140); plt.close(fig)
    (args.out / 'REPORT_epoki.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: raport[k] for k in ('epoka_najlepszego_mAP', 'luka_najlepsza_vs_ostatnia')}, indent=1, ensure_ascii=False))
    for g in GRUPY:
        print(f'{g:11s}', np.round(raport['mediana_pewnosci'][g], 2).tolist())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
