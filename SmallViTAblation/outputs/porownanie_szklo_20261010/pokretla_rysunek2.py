"""Rysunek 2 pokręteł: wszystkie analizy z eden/wyniki/pokretla/ANALIZA_<bieg>[_od_do].json (fale 1–3): pokrętło lr i gęstości
w oknie wczesnym (300–2188) i końcowym (gęste okno na końcu długich biegów, 12 epok), ziarna jako rozrzut, okno po 60. epoce
(okno60) i biegi z losowym rzutem 128D (_rp) jako osobne znaczniki. Uruchomienie: F:/Python/3.13/python.exe pokretla_rysunek2.py
Wynik: POKRETLA2.json, pokretla2.png"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = 2_926_832
rows = []
for p in sorted(glob.glob(os.path.join(HERE, 'eden', 'wyniki', 'pokretla', 'ANALIZA_*.json'))):
    j = json.load(open(p, encoding='utf-8'))
    nazwa = os.path.basename(p)[len('ANALIZA_'):-len('.json')]
    m = re.match(r'(.+?)(?:_(\d+)_(\d+))?(?:_w\d)?$', nazwa)
    bieg = m.group(1)
    okno = j.get('okno', [int(m.group(2)), int(m.group(3))] if m.group(2) else [300, 2188])
    meta = j.get('metadata', {}); pk = meta.get('pokretla', {})
    lr = meta.get('lr', pk.get('lr')); n_img = meta.get('klatki'); ziarno = pk.get('ziarno', 47)
    mz = re.search(r'_z(\d+)', bieg)
    if mz:
        ziarno = int(mz.group(1)); bieg = bieg[:mz.start()] + bieg[mz.end():]      # lr_5e-4_rp_z48 → bieg lr_5e-4_rp, ziarno 48 (słupki rozrzutu)
    rzut = j.get('rzut', 'pca32')
    S1, S0 = j['S1'], j['S0']; h = j['heterogenicznosc_CR']; hl = j['heterogenicznosc_lab']
    oc = j.get('oceny', [])
    if pk.get('opt', 'adamw') != 'adamw' or pk.get('model', 'vit') != 'vit' or meta.get('zadanie', 'detekcja') != 'detekcja' or re.search(r'_w\d$', nazwa) or bieg.startswith(('sgd', 'cnn', 'klas', 'warstwy')):
        continue                                    # fala 4 (SGD, CNN, Imagenette, warstwy) ma osobny rysunek: substancja_rysunek.py
    rodzaj = ('okno60' if bieg.startswith('okno60') else 'gestosc' if 'gestosc' in bieg else 'ziarno' if bieg.startswith('ziarno') else 'lr')
    faza = 'koniec' if okno[0] > 3000 else 'wczesne'
    rows.append(dict(bieg=bieg, plik=nazwa, rodzaj=rodzaj, faza=faza, okno=okno, lr=lr, zdjec=n_img, ziarno=ziarno, rzut=rzut, wymiar=j.get('wymiar', 32),
                     ogr_na_param=(n_img * 600 * 5 / P) if n_img else None, mAP_koniec=oc[-1]['mAP50'] if oc else None,
                     lindemann=S1['lindemann_shell'], l_klatka=S1['l_klatka'], d_shell=j['d_shell_300'], c=S1['c_plateau'], tau_1e=S1.get('tau_1e'), beta=S1.get('beta'),
                     f_q=S1['f_q'], alfa_plateau=S1['alfa_plateau'], a2=S1['a2_szczyt'], a2_tau=S1['a2_szczyt_tau'], a2_inw=S1.get('a2_inw_szczyt'), a2_inw_tau=S1.get('a2_inw_szczyt_tau'),
                     ucieczki10=S1['uciekajacy_tau10'], mu10=S1.get('mu', {}).get('10'), karta=S1['spelnione'], karta_S0=S0['spelnione'],
                     E768=j['energia_768']['E_300_400'], E768_koniec=j['energia_768']['E_2000_2188'], wykladnik=j['energia_768']['wykladnik_stygniecia_t0_400'],
                     C1=h['szczyt']['C1'], C1_tau=h['szczyt']['tau'], C_powloki=h['szczyt']['C_powloki'], C1_surowe=max(h['C_surowe']) if h.get('C_surowe') else None,
                     cross=max(h['C_cross']) if h['C_cross'] else None, skupianie=h['szczyt']['skupianie'], cos_lab=hl['szczyt']['cos_roznica'],
                     C1_tau_krzywa=dict(zip(h['taus'], [c[0] for c in h['C']])), trwalosc=j['trwalosc']))
json.dump(dict(tabela=rows), open(os.path.join(HERE, 'POKRETLA2.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"{'bieg':<22}{'faza':<8}{'rzut':<7}{'lr':>9}{'zdjęć':>7} {'Lind':>6}{'c':>6}{'τ1e':>7}{'β':>6}{'f_q':>6}{'a2':>6}{'a2inw':>7}{'uc10':>6}{'μ10':>6}{'E768':>7}{'wykł':>6}{'C1':>6}{'@τ':>5}{'cross':>7}{'mAP':>6}")
for r in sorted(rows, key=lambda r: (r['rodzaj'], r['faza'], r['rzut'], r['lr'] or 0, r['zdjec'] or 0)):
    f = lambda v, w=6, d=2: (f'{v:{w}.{d}f}' if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else f'{"-":>{w}}')
    print(f"{r['bieg']:<22}{r['faza']:<8}{r['rzut']:<7}{r['lr']:>9}{r['zdjec']:>7} {f(r['lindemann'],6,3)}{f(r['c'],6,3)}{f(r['tau_1e'],7,0)}{f(r['beta'])}{f(r['f_q'])}{f(r['a2'])}{f(r['a2_inw'],7)}{f(r['ucieczki10'],6,3)}{f(r['mu10'])}{f(r['E768'],7,3)}{f(r['wykladnik'])}{f(r['C1'],6,3)}{r['C1_tau']:>5}{f(r['cross'],7,3)}{f(r['mAP_koniec'],6,3)}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
pola = [('lindemann', 'rozmiar klatki / odstęp sąsiadów (S1)'), ('c', 'antykorelacja c (S1)'), ('tau_1e', 'τ_1e Fs (kroki)'), ('f_q', 'plateau Fs f_q'),
        ('a2_inw', 'szczyt α₂ niezmienniczy (S1)'), ('E768', 'energia kroku 768D (pierwsze 100 kroków okna)'), ('C1', 'heterogeniczność C₁ (względna, CR)'), ('mAP_koniec', 'mAP@0,5 na końcu biegu')]
fig, ax = plt.subplots(4, 4, figsize=(21, 17), dpi=100)
style = {('wczesne', 'pca32'): ('#ffd166', 'o-', 'okno 300–2188, PCA32'), ('koniec', 'pca32'): ('#ff8c42', 's-', 'okno końcowe (12 epok), PCA32'),
         ('wczesne', 'rp128'): ('#8ef0c0', 'o--', 'okno 300–2188, rzut 128D'), ('koniec', 'rp128'): ('#4cd964', 's--', 'okno końcowe, rzut 128D')}
def seria(a, xs, ys, kol, ls, et):
    xs, ys = np.array(xs, float), np.array(ys, float)
    ux = np.unique(xs); my = [np.nanmean(ys[xs == u]) for u in ux]; sy = [np.nanstd(ys[xs == u]) if (xs == u).sum() > 1 else 0 for u in ux]
    a.errorbar(ux, my, yerr=sy, fmt=ls, color=kol, lw=1.8, ms=5, capsize=3, label=et)
for i, (k, tyt) in enumerate(pola):
    a = ax[i // 4, i % 4]            # wiersze 0–1: pokrętło lr
    for (faza, rzut), (kol, ls, et) in style.items():
        sel = [r for r in rows if r['rodzaj'] in ('lr', 'ziarno') and r['faza'] == faza and r['rzut'] == rzut and r[k] is not None and r['lr']]
        if sel:
            seria(a, [r['lr'] for r in sel], [r[k] for r in sel], kol, ls, et)
    for r in rows:
        if r['rodzaj'] == 'okno60' and r[k] is not None:
            a.plot([5e-4], [r[k]], 'D', color='#ff4433' if r['rzut'] == 'pca32' else '#ffb3a7', ms=10, mfc='none', mew=2, label=f"okno po 60. epoce ({r['rzut']})")
    a.set_xscale('log'); a.set_xlabel('lr (35 000 zdjęć)'); a.set_title(tyt, fontsize=10); a.grid(True, which='both', alpha=0.4)
    b = ax[2 + i // 4, i % 4]        # wiersze 2–3: pokrętło gęstości
    for (faza, rzut), (kol, ls, et) in style.items():
        sel = [r for r in rows if r['faza'] == faza and r['rzut'] == rzut and r[k] is not None and r['ogr_na_param'] and (r['rodzaj'] == 'gestosc' or (r['rodzaj'] in ('lr', 'ziarno') and r['lr'] == 5e-4))]
        if sel:
            seria(b, [r['ogr_na_param'] for r in sel], [r[k] for r in sel], kol, ls, et)
    b.set_xscale('log'); b.set_xlabel('ograniczeń na parametr = zdjęć·600·5 / P (lr 5e-4)'); b.set_title(tyt, fontsize=10); b.grid(True, which='both', alpha=0.4)
    if k == 'tau_1e':
        a.set_yscale('log'); b.set_yscale('log')
uch, et = [], []
for a_ in fig.axes:
    for h, l in zip(*a_.get_legend_handles_labels()):
        if l not in et:
            uch.append(h); et.append(l)
fig.legend(uch, et, loc='lower left', fontsize=9, ncol=6)
fig.suptitle('Pokrętła szkła, fale 1–3. Wiersze 1–2: temperatura (lr). Wiersze 3–4: gęstość (zdjęć na parametr). Słupki: rozrzut ziaren 47/48/49.', fontsize=12)
fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.07, hspace=0.5, wspace=0.22)
fig.savefig(os.path.join(HERE, 'pokretla2.png'))
print('zapisano pokretla2.png, POKRETLA2.json')
