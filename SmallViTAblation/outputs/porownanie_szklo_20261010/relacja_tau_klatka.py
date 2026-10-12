"""Relacja czasu relaksacji z rozmiarem klatki na wszystkich oknach RP128 z Edena (ANALIZA_*.json): test, czy biegi różnych
rodzin (ViT AdamW, SGD, CNN, warstwy, Imagenette) leżą na jednej krzywej, jak w szkłach (Larini 2008: ln τ_α liniowe w 1/<u²>),
na prawie OU (τ ∝ 1/u²), czy na osobnych krzywych. Uruchomienie: F:/Python/3.13/python.exe relacja_tau_klatka.py
→ RELACJA_TAU_KLATKA.json, relacja_tau_klatka.png"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = []
for p in sorted(glob.glob(os.path.join(HERE, 'eden', 'wyniki', 'pokretla', 'ANALIZA_*.json'))):
    j = json.load(open(p, encoding='utf-8'))
    if j.get('rzut') != 'rp' and j.get('wymiar') != 128:
        continue
    S1 = j.get('S1') or {}
    if not S1 or S1.get('tau_1e') is None or not math.isfinite(S1['tau_1e']) or S1['tau_1e'] >= 1800:
        continue
    pk = j['metadata'].get('pokretla', {}); b = j['bieg']
    rodz = ('klas' if b.startswith('klas') else 'sgd' if pk.get('opt') == 'sgd' else 'cnn' if pk.get('model') == 'cnn'
            else 'warstwa' if j.get('warstwa') is not None else 'vit')
    W.append(dict(okno_plik=os.path.basename(p)[8:-5], rodzina=rodz, lr=pk.get('lr'), okno=S1['okno'], lindemann=S1['lindemann_shell'],
                  tau_1e=S1['tau_1e'], beta=S1.get('beta'), f_q=S1['f_q'], c=S1['c_plateau'], E768=j['energia_768']['E_300_400'], warstwa=j.get('warstwa')))

u2 = np.array([w['lindemann'] ** 2 for w in W]); lt = np.log([w['tau_1e'] for w in W]); rodz = np.array([w['rodzina'] for w in W])


def fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - (k * x + b)
    return float(k), float(b), float(1 - r.var() / y.var())


wyn = dict(opis=__doc__.split('\n')[0], okien=len(W), wiersze=W, dopasowania={})
for g in ('wszystkie', 'vit', 'sgd', 'cnn', 'warstwa', 'klas'):
    m = np.ones(len(W), bool) if g == 'wszystkie' else rodz == g
    if m.sum() < 3:
        continue
    k1, b1, r1 = fit(np.log(u2[m]), lt[m]); k2, b2, r2 = fit(1 / u2[m], lt[m])
    wyn['dopasowania'][g] = dict(n=int(m.sum()), potegowa=dict(wykladnik_u2=k1, stala=b1, R2=r1), larini=dict(nachylenie=k2, stala=b2, R2=r2))
mv = rodz == 'vit'; k, b, _ = fit(np.log(u2[mv]), lt[mv])
wyn['odchylenie_od_prawa_vit'] = {g: dict(ln=float(np.mean(lt[rodz == g] - (k * np.log(u2[rodz == g]) + b))),
                                          czynnik=float(math.exp(np.mean(lt[rodz == g] - (k * np.log(u2[rodz == g]) + b)))))
                                  for g in ('sgd', 'cnn', 'warstwa', 'klas') if (rodz == g).any()}
json.dump(wyn, open(os.path.join(HERE, 'RELACJA_TAU_KLATKA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for g, d in wyn['dopasowania'].items():
    print(f"{g:<10} n={d['n']:2d} potęgowa: wykładnik {d['potegowa']['wykladnik_u2']:.2f} R² {d['potegowa']['R2']:.2f} | Larini: R² {d['larini']['R2']:.2f}")
print('odchylenia od prawa ViT:', {g: round(d['czynnik'], 2) for g, d in wyn['odchylenie_od_prawa_vit'].items()})

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99',
                     'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a',
                     'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
kol = {'vit': '#ffffff', 'sgd': '#ffd166', 'cnn': '#ff4433', 'warstwa': '#4a8cff', 'klas': '#c77dff'}
ety = {'vit': 'ViT AdamW, detekcja', 'sgd': 'ViT SGD', 'cnn': 'CNN AdamW', 'warstwa': 'ViT warstwy 0–5', 'klas': 'Imagenette (klasyfikacja)'}
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), dpi=100)
for a, x, xl, klucz in ((ax[0], np.log(u2), 'ln (klatka / odstęp sąsiadów)²', 'potegowa'), (ax[1], 1 / u2, '1 / (klatka / odstęp sąsiadów)²  (oś Lariniego)', 'larini')):
    for g in kol:
        m = rodz == g
        a.scatter(x[m], lt[m], s=36, color=kol[g], edgecolor='#10171c', label=ety[g], zorder=3)
    xs = np.linspace(x.min(), x.max(), 100); d = wyn['dopasowania']['vit'][klucz]
    kk = d['wykladnik_u2'] if klucz == 'potegowa' else d['nachylenie']
    a.plot(xs, kk * xs + d['stala'], '--', color='#ffffff', lw=1, alpha=0.7, label=f"prawo ViT AdamW (R² {d['R2']:.2f})")
    a.set_xlabel(xl); a.set_ylabel('ln τ_1e [kroki]'); a.grid(True, alpha=0.4); a.legend(fontsize=8)
ax[0].set_title('potęgowo: τ ∝ u^(−3,6) dla ViT AdamW; rodziny przesunięte', fontsize=10)
ax[1].set_title('test Lariniego (szkła: prosta): brak wspólnej prostej', fontsize=10)
fig.suptitle(f'Czas relaksacji a rozmiar klatki, {len(W)} okien RP128 z Edena', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'relacja_tau_klatka.png'))
print('zapisano RELACJA_TAU_KLATKA.json, relacja_tau_klatka.png')
