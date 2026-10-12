"""Rysunek i tabela dla biegów z pokrętłami (eden/wyniki/pokretla/ANALIZA_<bieg>.json z experiments/maly_vit_szklo_pokretla_20261010):
pokrętło temperatury (lr przy 35 000 zdjęć) i gęstości (zdjęć na parametr przy lr 5e-4). Dla każdego biegu: rozmiar klatki S1,
antykorelacja c, czas relaksacji τ_1e (Fs), szczyt α₂, heterogeniczność C1 (względna, CR) i jej zanik w powłokach, energia kroku
768D w oknie 300–400 i wykładnik stygnięcia, mAP na końcu. Uruchomienie: F:/Python/3.13/python.exe pokretla_rysunek.py
Wynik: POKRETLA.json, pokretla.png"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = 2_926_832                                   # parametrów MalyViT
biegi = {}
for p in sorted(glob.glob(os.path.join(HERE, 'eden', 'wyniki', 'pokretla', 'ANALIZA_*.json'))):
    j = json.load(open(p, encoding='utf-8'))
    biegi[j['bieg']] = j
assert biegi, 'brak eden/wyniki/pokretla/ANALIZA_*.json'


def wiersz(b, j):
    meta = j.get('metadata', {})
    lr = meta.get('lr', meta.get('pokretla', {}).get('lr'))
    n_img = meta.get('klatki')
    oc = j.get('oceny', [])
    S1, S0 = j['S1'], j['S0']
    h = j['heterogenicznosc_CR']; hl = j['heterogenicznosc_lab']
    return dict(bieg=b, lr=lr, zdjec=n_img, ograniczen_na_parametr=(n_img * 600 * 5 / P) if n_img else None,
                mAP_koniec=oc[-1]['mAP50'] if oc else None, mAP_krok=oc[-1]['krok'] if oc else None,
                lindemann_S1=S1['lindemann_shell'], c_S1=S1['c_plateau'], tau_1e_S1=S1.get('tau_1e'), beta_S1=S1.get('beta'),
                a2_szczyt_S1=S1['a2_szczyt'], a2_tau_S1=S1['a2_szczyt_tau'], f_q_S1=S1['f_q'], alfa_plateau_S1=S1['alfa_plateau'],
                ucieczki10_S1=S1['uciekajacy_tau10'], mu10_S1=S1.get('mu', {}).get('10'), karta_S1=S1['spelnione'], karta_S0=S0['spelnione'],
                lindemann_S0=S0['lindemann_shell'], c_S0=S0['c_plateau'],
                E768_300_400=j['energia_768']['E_300_400'], E768_2000_2188=j['energia_768']['E_2000_2188'], wykladnik_stygniecia=j['energia_768']['wykladnik_stygniecia_t0_400'],
                E1_pca32_S0=S0.get('E1'), trwalosc_1300=j['trwalosc'].get('1300', j['trwalosc'].get(1300)),
                C1_CR=h['szczyt']['C1'], C1_CR_tau=h['szczyt']['tau'], C_powloki_CR=h['szczyt']['C_powloki'], C1_CR_surowe=max(h['C_surowe']) if h.get('C_surowe') else None,
                C1_null=h['szczyt']['null'], skupianie_CR=h['szczyt']['skupianie'], cross_CR=max(h['C_cross']) if h['C_cross'] else None,
                C1_lab=hl['szczyt']['C1'], cos_lab=hl['szczyt']['cos_roznica'])


tab = [wiersz(b, j) for b, j in biegi.items()]
json.dump(dict(tabela=tab), open(os.path.join(HERE, 'POKRETLA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for r in tab:
    print(f"{r['bieg']:<14} lr {r['lr']} zdjęć {r['zdjec']} mAP {r['mAP_koniec']} | Lind {r['lindemann_S1']:.3f} c {r['c_S1']:.3f} τ1e {r['tau_1e_S1']} β {r['beta_S1']} "
          f"a2 {r['a2_szczyt_S1']:.2f}@{r['a2_tau_S1']} f_q {r['f_q_S1']:.2f} μ10 {r['mu10_S1']} karta {r['karta_S1']} | E768 {r['E768_300_400']:.4g} wykł {r['wykladnik_stygniecia']} "
          f"| C1 CR {r['C1_CR']:.3f} (surowe {r['C1_CR_surowe']}) powłoki {r['C_powloki_CR']} cross {r['cross_CR']} cos lab {r['cos_lab']:.3f}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
temp = sorted([r for r in tab if r['bieg'].startswith('lr_')], key=lambda r: r['lr'])
gest = sorted([r for r in tab if r['bieg'].startswith('gestosc_') or r['bieg'] == 'lr_5e-4'], key=lambda r: r['zdjec'])
fig, ax = plt.subplots(2, 4, figsize=(20, 9.5), dpi=100)
pola = [('lindemann_S1', 'rozmiar klatki / odstęp sąsiadów (S1)'), ('c_S1', 'antykorelacja c (S1)'), ('tau_1e_S1', 'τ_1e Fs (kroki)'),
        ('a2_szczyt_S1', 'szczyt α₂ (S1)'), ('E768_300_400', 'energia kroku 768D, kroki 300–400'), ('wykladnik_stygniecia', 'wykładnik stygnięcia E ∝ (1+t/400)^−b'),
        ('C1_CR', 'heterogeniczność C₁ (względna, CR)'), ('mAP_koniec', 'mAP@0,5 na końcu')]
for a, (k, tyt) in zip(ax.flat, pola):
    x = [r['lr'] for r in temp]; y = [r[k] if r[k] is not None else np.nan for r in temp]
    a.plot(x, y, 'o-', color='#ffd166', lw=2, label='pokrętło lr (35 000 zdjęć)')
    a.set_xscale('log'); a.set_xlabel('lr'); a.set_title(tyt, fontsize=10); a.grid(True, which='both', alpha=0.4)
    a2 = a.twiny()
    xg = [r['ograniczen_na_parametr'] for r in gest]; yg = [r[k] if r[k] is not None else np.nan for r in gest]
    a2.plot(xg, yg, 's--', color='#4a8cff', lw=1.6, label='pokrętło gęstości (lr 5e-4)')
    a2.set_xscale('log'); a2.set_xlabel('ograniczeń na parametr (zdjęcia·600·5 / P)', color='#4a8cff', fontsize=8)
    if k in ('tau_1e_S1',):
        a.set_yscale('log')
    if k == 'lindemann_S1':
        a.legend(loc='upper left', fontsize=8); a2.legend(loc='lower right', fontsize=8)
fig.suptitle('Pokrętła szkła: temperatura (lr) i gęstość (zdjęć na parametr) a odciski ruchu tokenów w oknie 300–2188', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(HERE, 'pokretla.png'))
print('zapisano pokretla.png, POKRETLA.json')
