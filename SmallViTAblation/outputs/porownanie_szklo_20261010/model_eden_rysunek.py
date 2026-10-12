"""Rysunek i karta zgodności dla modelu szkła z Edena (eden/wyniki/MODEL_SZKLA_EDEN_*.json) obok tokenów ViT (UKLAD_SZKLA.json)
i kontroli negatywnej OU (KONTROLA_OU.json). Wybiera gęstość najbliższą tokenom w układzie klatki (rozmiar klatki i c).
Uruchomienie: F:/Python/3.13/python.exe model_eden_rysunek.py [szybko|duze]  → model_eden_<tryb>.png, MODEL_EDEN_<tryb>.json"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odciski as O  # noqa: E402

TRYB = sys.argv[1] if len(sys.argv) > 1 else 'szybko'
pref = 'szybko_' if TRYB == 'szybko' else ''
biegi = {}
for t in ('stala', 'stygnace'):
    p = os.path.join(HERE, 'eden', 'wyniki', f'MODEL_SZKLA_EDEN_{pref}{t}.json')
    if os.path.isfile(p):
        j = json.load(open(p, encoding='utf-8'))
        for k, v in j['biegi'].items():
            if 'karta' in v:
                v['karta'], v['spelnione'] = O.karta(v, v.get('trwalosc', {}).get('1000'))   # karta wg bieżących kryteriów
                biegi[f'{t} {k}'] = v
assert biegi, 'brak eden/wyniki/MODEL_SZKLA_EDEN_*.json (uruchom eden/pobierz_z_edena.sh szklo)'
vit = json.load(open(os.path.join(HERE, 'UKLAD_SZKLA.json'), encoding='utf-8'))
S1, S4 = vit['etapy']['S1'], vit['etapy']['S4']
ou = json.load(open(os.path.join(HERE, 'KONTROLA_OU.json'), encoding='utf-8'))['warianty'] if os.path.isfile(os.path.join(HERE, 'KONTROLA_OU.json')) else {}
stale = {k: v for k, v in biegi.items() if k.startswith('stala')}
styg = {k: v for k, v in biegi.items() if k.startswith('stygnace')}
cel = dict(lindemann_shell=S1['lindemann_shell'], c_plateau=S1['c_plateau'], f_q=S1['f_q'], alfa_plateau=S1['alfa_plateau'])
odl = {k: abs(np.log(max(v['lindemann_shell'], 1e-6) / cel['lindemann_shell'])) + abs(v['c_plateau'] - cel['c_plateau'])
       + abs(v['f_q'] - cel['f_q']) + abs(v['alfa_plateau'] - cel['alfa_plateau']) for k, v in stale.items()}     # cztery odciski, równe wagi
najblizszy = min(odl, key=odl.get) if odl else None
tab = []
for k, v in biegi.items():
    tab.append(dict(bieg=k, phi=v['phi'], lindemann=v['lindemann_shell'], c=v['c_plateau'], alfa_plateau=v['alfa_plateau'], a2_szczyt=v['a2_szczyt'],
                    a2_szczyt_tau=v['a2_szczyt_tau'], f_q=v['f_q'], beta=v['beta'], tau_1e=v['tau_1e'], ucieczki_10=v['uciekajacy_tau10'],
                    trwalosc_1000=v.get('trwalosc', {}).get('1000'), mu=v.get('mu', {}), krotkie=v.get('stosunek_krotkich'), karta=v['spelnione']))
json.dump(dict(tryb=TRYB, cel=cel, najblizszy=najblizszy, odleglosci=odl, tabela=tab,
               karty={k: [(o['kryterium'], o['ok']) for o in v['karta']] for k, v in biegi.items()}),
          open(os.path.join(HERE, f'MODEL_EDEN_{TRYB}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for r in tab:
    print(f"{r['bieg']:<22} Lind {r['lindemann']:.3f} c {r['c']:.3f} α {r['alfa_plateau']:.2f} a2 {r['a2_szczyt']:.2f}@{r['a2_szczyt_tau']} f_q {r['f_q']:.2f} "
          f"β {r['beta']:.2f} τ1e {r['tau_1e']} uc10 {r['ucieczki_10']:.3f} trw {r['trwalosc_1000']} μ {({a: round(b, 2) for a, b in r['mu'].items()})} karta {r['karta']}")
print('najbliższy tokenom (S1):', najblizszy)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(2, 3, figsize=(19, 11), dpi=100)
kol = matplotlib.colormaps['plasma'](np.linspace(0.15, 0.85, max(len(stale), 2)))
for (k, w), c in zip(stale.items(), kol):
    tb, ds = w['tau_B_w_rekordach'], w['d_shell']
    lw = 2.4 if k == najblizszy else 1.3
    ax[0, 0].loglog(np.array(w['taus']) / tb, np.array(w['msd']) / ds ** 2, color=c, lw=lw, label=f"BD φ={w['phi']}" + (' (najbliższy)' if k == najblizszy else ''))
    ax[0, 1].semilogx(np.array(w['taus']) / tb, w['a2'], color=c, lw=lw, label=f"BD φ={w['phi']}")
    ax[0, 2].semilogx(np.array(w['taus']) / tb, w['fs'], color=c, lw=lw, label=f"BD φ={w['phi']}")
    cx = sorted(int(x) for x in w['c_tau'])
    ax[1, 0].semilogx(np.array(cx) / tb, [w['c_tau'][str(x)] for x in cx], 'o-', color=c, lw=1.2, ms=4, label=f"BD φ={w['phi']}")
sk = 1 / 100.0
for w, c, nazwa in ((S1, '#ffd166', 'ViT S1 klatka'), (S4, '#ff4433', 'ViT S4 bez pędu')):
    ds = w['d_shell']
    ax[0, 0].loglog(np.array(w['taus']) * sk, np.array(w['msd']) / ds ** 2, '--', color=c, lw=2, label=f'{nazwa} (τ/100)')
    ax[0, 1].semilogx(np.array(w['taus']) * sk, w['a2'], '--', color=c, lw=2, label=nazwa)
    ax[0, 2].semilogx(np.array(w['taus']) * sk, w['fs'], '--', color=c, lw=2, label=nazwa)
    cx = sorted(int(x) for x in w['c_tau'])
    ax[1, 0].semilogx(np.array(cx) * sk, [w['c_tau'][str(x)] for x in cx], 's--', color=c, lw=1.4, ms=4, label=nazwa)
ax[0, 0].axhline(0.15 ** 2, color='#ff4433', ls=':', lw=0.8)
ax[0, 0].set(xlabel='τ / τ_B (BD) albo τ/100 kroków (ViT)', ylabel='MSD / d_shell²', title='MSD w jednostkach odstępu sąsiadów')
ax[0, 1].set(xlabel='τ / τ_B', ylabel='α₂', title='parametr niegaussowski')
ax[0, 2].axhline(np.exp(-1), color='w', ls=':', lw=0.8)
ax[0, 2].set(xlabel='τ / τ_B', ylabel='Fs(q, τ)', title='funkcja rozpraszania', ylim=(-0.05, 1.05))
ax[1, 0].axhspan(0.25, 0.55, color='#ffd166', alpha=0.12)
ax[1, 0].set(xlabel='τ / τ_B', ylabel='c(τ)', title='antykorelacja kolejnych przesunięć')
a = ax[1, 1]
wst = styg.get(najblizszy.replace('stala', 'stygnace')) if najblizszy else None
if wst is None and styg:
    wst = list(styg.values())[0]
if wst:
    for tau, c in ((10, '#4cd964'), (100, '#ffd166'), (1000, '#ff4433')):
        pts = sorted((int(tw), v[str(tau)]) for tw, v in wst['starzenie'].items() if str(tau) in v)
        if pts:
            a.loglog([p[0] for p in pts], [p[1] / wst['d_shell'] ** 2 for p in pts], 'o-', color=c, lw=1.6,
                     label=f"BD stygnące φ={wst['phi']} MSD(τ={tau}; t_w)" + (f", μ={wst['mu'][str(tau)]:.2f}" if str(tau) in wst['mu'] else ''))
    if 'T_rec' in wst:
        a.plot(wst['T_rec']['rekord'], np.array(wst['T_rec']['T']) / wst['T_rec']['T'][0] * 0.3, color='#8a8f99', lw=1, label='T(t)/T0 × 0,3')
pts = sorted((int(tw), v['10']) for tw, v in S1['starzenie'].items() if '10' in v)
a.loglog([p[0] * sk for p in pts], [p[1] / S1['d_shell'] ** 2 for p in pts], 's--', color='#4a8cff', lw=1.4, label='ViT S1 MSD(τ=10; t_w/100)')
a.set(xlabel='t_w (rekordy BD) / (kroki ViT / 100)', ylabel='MSD / d_shell²', title='starzenie: szkło stygnące a tokeny')
for q in ax.flat[:5]:
    q.grid(True, which='both', alpha=0.4)
    q.legend(fontsize=8)
a = ax[1, 2]
kryteria = [o['kryterium'] for o in next(iter(biegi.values()))['karta']]
kolumny = {k: [o['ok'] for o in v['karta']] for k, v in biegi.items()}
kolumny['ViT S1'] = [o['ok'] for o in O.karta(S1, vit['trwalosc'].get('1300'))[0]]
kolumny['ViT S4'] = [o['ok'] for o in O.karta(S4, vit['trwalosc'].get('1300'))[0]]
for k in ('OU-A', 'OU-B', 'OU-C'):
    if k in ou:
        kolumny[k] = [o['ok'] for o in ou[k]['karta']]
mat = np.array([[1 if kolumny[k][i] else 0 for k in kolumny] for i in range(len(kryteria))])
a.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
a.set_xticks(range(len(kolumny)))
a.set_xticklabels([f"{k.replace('stygnace', 'styg').replace('stala ', '')}\n{int(mat[:, i].sum())}/{len(kryteria)}" for i, k in enumerate(kolumny)], fontsize=7, rotation=35)
a.set_yticks(range(len(kryteria)))
a.set_yticklabels([n[:58] for n in kryteria], fontsize=7)
a.set_title('karta: model BD (Eden), tokeny ViT, kontrola OU')
fig.suptitle(f'Model szkła koloidalnego z Edena ({TRYB}: N = {next(iter(biegi.values())).get("N", "")}) obok tokenów ViT i kontroli OU', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(HERE, f'model_eden_{TRYB}.png'))
print('zapisano', f'model_eden_{TRYB}.png')
