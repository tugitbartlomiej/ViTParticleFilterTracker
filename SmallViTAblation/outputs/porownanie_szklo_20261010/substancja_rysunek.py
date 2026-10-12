"""Rysunek „jaka to substancja” (2026-10-11): nowe testy z dh.py dla tokenów ViT (eden/wyniki/pokretla/ANALIZA_*.json, pola
trwalosc_ruchliwosci_*, korelacja_krzyzowa_*, korelacja_przesuniec, rozbieganie_par, rozklad_wariancji) obok kontroli
(SUBSTANCJA_KONTROLA.json: szkło BD, OU-B, OU-B + wspólny szum zdjęcia) oraz pokrętła fali 4: SGD (lr/paczka), drugi model (CNN),
skan po warstwach, drugi zbiór (Imagenette), f32 kontra f16. Uruchomienie: F:/Python/3.13/python.exe substancja_rysunek.py
Wynik: substancja.png, SUBSTANCJA.json (tabela)."""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P_VIT = 2_926_493


def wczytaj_analizy():
    rows = {}
    for p in sorted(glob.glob(os.path.join(HERE, 'eden', 'wyniki', 'pokretla', 'ANALIZA_*.json'))):
        j = json.load(open(p, encoding='utf-8'))
        if 'trwalosc_ruchliwosci_CR' not in j:
            continue
        nazwa = os.path.basename(p)[len('ANALIZA_'):-len('.json')]
        m = re.match(r'(.+?)(?:_(\d+)_(\d+))?(?:_w(\d))?$', nazwa)
        bieg, od, do, w = m.group(1), m.group(2), m.group(3), m.group(4)
        meta = j.get('metadata', {}); pk = meta.get('pokretla', {})
        okno = j.get('okno', [int(od), int(do)] if od else [300, 2188])
        S1 = j['S1']; h = j['heterogenicznosc_CR']
        rows[nazwa] = dict(bieg=bieg, warstwa=(int(w) if w is not None else j.get('warstwa')), okno=okno, faza=('koniec' if okno[0] > 3000 else 'wczesne'),
                           lr=meta.get('lr', pk.get('lr')), opt=pk.get('opt', 'adamw'), paczka=pk.get('paczka', 32), model=pk.get('model', 'vit'),
                           zadanie=meta.get('zadanie', 'detekcja'), ziarno=pk.get('ziarno', 47), zapis=j.get('zapis', 'f16'), rzut=j.get('rzut'),
                           lindemann=S1['lindemann_shell'], tau_1e=S1.get('tau_1e'), beta=S1.get('beta'), f_q=S1['f_q'], c=S1['c_plateau'],
                           a2_inw=S1.get('a2_inw_szczyt'), ucieczki10=S1['uciekajacy_tau10'], karta=S1['spelnione'],
                           E768=j['energia_768']['E_300_400'], C1=h['szczyt']['C1'], C1_tau=h['szczyt']['tau'], C_powloki=h['szczyt']['C_powloki'],
                           cross=max(h['C_cross']) if h['C_cross'] else None, mAP=(j.get('oceny') or [{}])[-1].get('mAP50'),
                           trw_CR=j['trwalosc_ruchliwosci_CR'], trw_lab=j['trwalosc_ruchliwosci_lab'], kk_blok=j['korelacja_krzyzowa_blok'],
                           kk_pop=j['korelacja_krzyzowa_populacja'], kp=j['korelacja_przesuniec'], rp=j['rozbieganie_par'], rpc=j.get('rozbieganie_par_calosc'),
                           rv=j['rozklad_wariancji'])
    return rows


rows = wczytaj_analizy()
kontrola = json.load(open(os.path.join(HERE, 'SUBSTANCJA_KONTROLA.json'), encoding='utf-8'))['wyniki'] if os.path.isfile(os.path.join(HERE, 'SUBSTANCJA_KONTROLA.json')) else {}
print(f'{len(rows)} analiz z nowymi testami, {len(kontrola)} kontroli')
tab = []
for n, r in rows.items():
    tab.append(dict(analiza=n, bieg=r['bieg'], warstwa=r['warstwa'], okno=r['okno'], lr=r['lr'], opt=r['opt'], paczka=r['paczka'], model=r['model'], zadanie=r['zadanie'],
                    ziarno=r['ziarno'], zapis=r['zapis'], lindemann=r['lindemann'], tau_1e=r['tau_1e'], beta=r['beta'], f_q=r['f_q'], a2_inw=r['a2_inw'], karta=r['karta'], E768=r['E768'],
                    C1=r['C1'], C1_tau=r['C1_tau'], cross=r['cross'], mAP=r['mAP'],
                    trw_CR_lag1=r['trw_CR'].get('spearman_wzgledny_po_lagu', [None])[0], trw_CR_ost=r['trw_CR'].get('pierwsze_ostatnie_wzgledne'),
                    trw_zdjec_ost=r['trw_CR'].get('pierwsze_ostatnie_zdjec'),
                    kk_sasiedzi_0=r['kk_blok'].get('werdykt', {}).get('C_sasiedzi_0'), kk_sasiedzi_min=r['kk_blok'].get('werdykt', {}).get('C_sasiedzi_min'),
                    kk_losowe_0=r['kk_blok'].get('werdykt', {}).get('C_losowe_zdjecie_0'), kk_pop_losowe_0=r['kk_pop'].get('werdykt', {}).get('C_losowe_zdjecie_0'),
                    kk_pop_inne_0=r['kk_pop'].get('werdykt', {}).get('C_inne_zdjecia_0'),
                    rozbieganie_wykl=r['rp'].get('wykladnik_10_1000'), r2_koniec=r['rp']['r2_wzgledne'][-1] if r['rp']['r2_wzgledne'] else None,
                    rozklad_tau1={k: v[0] for k, v in r['rv']['udzial'].items()} if r['rv']['taus'] else None,
                    rozklad_tau100={k: v[r['rv']['taus'].index(100)] for k, v in r['rv']['udzial'].items()} if 100 in r['rv']['taus'] else None))
for k, w in kontrola.items():
    tab.append(dict(analiza=k, bieg=k, kontrola=True, lindemann=None, trw_CR_lag1=w['trwalosc_ruchliwosci_CR'].get('spearman_wzgledny_po_lagu', [None])[0],
                    trw_CR_ost=w['trwalosc_ruchliwosci_CR'].get('pierwsze_ostatnie_wzgledne'), kk_sasiedzi_0=w['korelacja_krzyzowa'].get('werdykt', {}).get('C_sasiedzi_0'),
                    kk_sasiedzi_min=w['korelacja_krzyzowa'].get('werdykt', {}).get('C_sasiedzi_min'), kk_losowe_0=w['korelacja_krzyzowa'].get('werdykt', {}).get('C_losowe_zdjecie_0'),
                    rozbieganie_wykl=w['rozbieganie_par'].get('wykladnik_10_1000'), r2_koniec=w['rozbieganie_par']['r2_wzgledne'][-1] if w['rozbieganie_par']['r2_wzgledne'] else None,
                    rozklad_tau1={k2: v[0] for k2, v in w['rozklad_wariancji']['udzial'].items()} if w['rozklad_wariancji']['taus'] else None))
json.dump(dict(tabela=tab), open(os.path.join(HERE, 'SUBSTANCJA.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
f = lambda v, d=3: ('-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.{d}f}' if isinstance(v, (int, float)) else str(v))
print(f"{'analiza':<34}{'Lind':>6}{'τ1e':>7}{'C1':>6}{'cross':>7}{'trwR1':>7}{'trwRk':>7}{'trwZ':>6}{'kk0':>7}{'kkmin':>7}{'kklos':>7}{'kkpop':>7}{'kkinne':>7}{'rozb':>6}{'r2k':>6}")
for r in tab:
    print(f"{r['analiza']:<34}{f(r.get('lindemann')):>6}{f(r.get('tau_1e'), 0):>7}{f(r.get('C1')):>6}{f(r.get('cross')):>7}{f(r.get('trw_CR_lag1')):>7}{f(r.get('trw_CR_ost')):>7}"
          f"{f(r.get('trw_zdjec_ost')):>6}{f(r.get('kk_sasiedzi_0')):>7}{f(r.get('kk_sasiedzi_min')):>7}{f(r.get('kk_losowe_0')):>7}{f(r.get('kk_pop_losowe_0')):>7}{f(r.get('kk_pop_inne_0')):>7}"
          f"{f(r.get('rozbieganie_wykl'), 2):>6}{f(r.get('r2_koniec'), 2):>6}")

# ================================================================== rysunek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 9, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig, ax = plt.subplots(3, 4, figsize=(22, 15), dpi=100)
glowne = {k: r for k, r in rows.items() if r['warstwa'] is None and r['zadanie'] == 'detekcja' and r['model'] == 'vit' and r['opt'] == 'adamw'}
wybor = [k for k in ('lr_5e-4_rp', 'lr_1.25e-4_rp', 'lr_2e-3_rp', 'dlugi_lr_5e-4_rp_12034_13128', 'dlugi_lr_1.25e-4_rp_12034_13128', 'okno60_rp_65640_66734') if k in rows]
kol_vit = dict(zip(wybor, ['#ffd166', '#8ef0c0', '#ff8c42', '#ffea00', '#4cd964', '#ff4433']))
kol_k = {'BD φ=0.58': '#6fa8dc', 'BD φ=0.61': '#4a8cff', 'BD φ=0.62': '#b36bff', 'BD φ=0.64': '#c77dff', 'OU-B': '#9aa3ad', 'OU-Moran': '#ffffff'}
# (0,0) trwałość ruchliwości (względna, CR) wg odstępu podokien
a = ax[0, 0]
for k in wybor:
    t = rows[k]['trw_CR']
    if 'lagi' in t:
        a.plot(t['lagi'], t['spearman_wzgledny_po_lagu'], 'o-', color=kol_vit[k], label=f'ViT {k}')
for k, w in kontrola.items():
    t = w['trwalosc_ruchliwosci_CR']
    if 'lagi' in t:
        a.plot(t['lagi'], t['spearman_wzgledny_po_lagu'], 's--', color=kol_k.get(k, 'w'), label=k)
a.axhline(0, color='#8a8f99', lw=0.8); a.set_xlabel('odstęp podokien (×200 kroków/klatek)'); a.set_ylabel('Spearman rankingu ruchliwości (CR, względna)')
a.set_title('Heterogeniczność zamrożona czy przemijająca?', fontsize=10); a.grid(True, alpha=0.4); a.legend(fontsize=7)
# (0,1) korelacja krzyżowa położeń: sąsiedzi (blok) ViT i kontrole
a = ax[0, 1]
for k in wybor:
    kk = rows[k]['kk_blok']
    if kk.get('taus'):
        a.plot(kk['taus'], kk['C']['sasiedzi'], 'o-', color=kol_vit[k], label=f'ViT {k} sąsiedzi')
        a.plot(kk['taus'], kk['C']['losowe_zdjecie'], ':', color=kol_vit[k], lw=1)
for k, w in kontrola.items():
    kk = w['korelacja_krzyzowa']
    if kk.get('taus'):
        a.plot(kk['taus'], kk['C']['sasiedzi'], 's--', color=kol_k.get(k, 'w'), label=f'{k} sąsiedzi')
a.axhline(0, color='#8a8f99', lw=0.8); a.set_xscale('symlog', linthresh=1); a.set_xlabel('opóźnienie τ (kroki/klatki)'); a.set_ylabel('C_ij(τ) położeń względem klatki')
a.set_title('Meiners–Quake / Moran: korelacja krzyżowa par (bez pola zdjęcia; kropki: pary losowe)', fontsize=10); a.grid(True, alpha=0.4); a.legend(fontsize=7)
# (0,2) korelacja krzyżowa w układzie populacji (z polem zdjęcia): sąsiedzi, losowe w zdjęciu, inne zdjęcia
a = ax[0, 2]
for k in wybor:
    kk = rows[k]['kk_pop']
    if kk.get('taus'):
        a.plot(kk['taus'], kk['C']['sasiedzi'], 'o-', color=kol_vit[k], label=f'{k} sąsiedzi')
        a.plot(kk['taus'], kk['C']['losowe_zdjecie'], 's:', color=kol_vit[k], ms=4, label=f'{k} losowe w zdjęciu')
        if 'losowe_inne_zdjecia' in kk['C']:
            a.plot(kk['taus'], kk['C']['losowe_inne_zdjecia'], 'x--', color=kol_vit[k], ms=4, label=f'{k} inne zdjęcia')
a.axhline(0, color='#8a8f99', lw=0.8); a.set_xscale('symlog', linthresh=1); a.set_xlabel('opóźnienie τ (kroki)'); a.set_ylabel('C_ij(τ) położeń (tylko dryf populacji odjęty)')
a.set_title('Wspólny napęd zdjęcia: korelacja położeń par w zdjęciu i między zdjęciami', fontsize=10); a.grid(True, alpha=0.4); a.legend(fontsize=6, ncol=2)
# (0,3) rozkład energii kroku
a = ax[0, 3]
nazwy, dane = [], []
for k in wybor:
    rv = rows[k]['rv']
    if rv['taus']:
        nazwy.append(f'ViT {k}'); dane.append([rv['udzial'][c][0] for c in ('wspolny', 'zdjecie', 'sasiedztwo', 'wlasny')])
for k, w in kontrola.items():
    rv = w['rozklad_wariancji']
    if rv['taus']:
        nazwy.append(k); dane.append([rv['udzial'][c][0] for c in ('wspolny', 'zdjecie', 'sasiedztwo', 'wlasny')])
if dane:
    dane = np.array(dane); x = np.arange(len(nazwy)); dol = np.zeros(len(nazwy))
    for i, (et, kol) in enumerate(zip(('wspólny dryf', 'pole zdjęcia', 'pole sąsiedztwa', 'ruch własny (CR)'), ('#ff4433', '#ff8c42', '#ffd166', '#4a8cff'))):
        a.bar(x, dane[:, i], bottom=dol, color=kol, label=et); dol += dane[:, i]
    a.set_xticks(x); a.set_xticklabels(nazwy, rotation=35, ha='right', fontsize=7)
a.set_ylabel('udział w <|Δz(τ=1)|²>'); a.set_title('Rozkład energii kroku τ=1', fontsize=10); a.legend(fontsize=7); a.grid(True, axis='y', alpha=0.4)
# (1,0) korelacja przesunięć w funkcji odległości
a = ax[1, 0]
for k in wybor[:3]:
    kp = rows[k]['kp']
    if kp['taus']:
        for it, tau in enumerate(kp['taus'][:2]):
            r_ = kp['r_sr']['wewnatrz_blok'][it]; c_ = kp['C']['wewnatrz_blok'][it]
            ok = [i for i in range(len(c_)) if c_[i] is not None and r_[i] is not None]
            a.plot([r_[i] for i in ok], [c_[i] for i in ok], 'o-' if it == 0 else 's--', color=kol_vit[k], ms=4, label=f'ViT {k} τ={tau} w zdjęciu')
            if 'miedzy_populacja' in kp['C'] and it == 0:
                r2_ = kp['r_sr']['miedzy_populacja'][it]; c2 = kp['C']['miedzy_populacja'][it]
                ok = [i for i in range(len(c2)) if c2[i] is not None and r2_[i] is not None]
                a.plot([r2_[i] for i in ok], [c2[i] for i in ok], 'x:', color=kol_vit[k], ms=5, label=f'ViT {k} τ={tau} między zdjęciami')
for k, w in kontrola.items():
    kp = w['korelacja_przesuniec']
    if kp['taus']:
        r_ = kp['r_sr']['wewnatrz_blok'][0]; c_ = kp['C']['wewnatrz_blok'][0]
        ok = [i for i in range(len(c_)) if c_[i] is not None and r_[i] is not None]
        a.plot([r_[i] for i in ok], [c_[i] for i in ok], 's--', color=kol_k.get(k, 'w'), ms=4, label=f'{k} τ=1')
a.axhline(0, color='#8a8f99', lw=0.8); a.set_xscale('log'); a.set_xlabel('odległość pary r / d_shell'); a.set_ylabel('<Δz_i·Δz_j> / <|Δz|²>')
a.set_title('Dwupunktowa korelacja przesunięć (mikroreologia dwupunktowa)', fontsize=10); a.grid(True, which='both', alpha=0.4); a.legend(fontsize=6)
# (1,1) rozbieganie par
a = ax[1, 1]
for k in wybor:
    rp = rows[k]['rpc'] or rows[k]['rp']
    a.loglog(rp['lagi'], rp['r2_wzgledne'], 'o-', color=kol_vit[k], ms=3, label=f'ViT {k}')
for k, w in kontrola.items():
    rp = w['rozbieganie_par']
    a.loglog(rp['lagi'], rp['r2_wzgledne'], 's--', color=kol_k.get(k, 'w'), ms=3, label=k)
xx = np.array([10, 1000]); a.loglog(xx, 1 + 0.02 * (xx / 10) ** 3, '-', color='#555', lw=1, label='Richardson t³'); a.loglog(xx, 1 + 0.02 * (xx / 10), '-', color='#888', lw=1, label='dyfuzja t¹')
a.set_xlabel('lag (kroki/klatki)'); a.set_ylabel('<r_ij²>(lag) / <r_ij²>(0), pary sąsiadów'); a.set_title('Rozbieganie par sąsiadów (test turbulencji)', fontsize=10); a.grid(True, which='both', alpha=0.4); a.legend(fontsize=7)
# (1,2) SGD: klatka vs lr/paczka
a = ax[1, 2]
sgd = [r for r in rows.values() if r['opt'] == 'sgd' and r['warstwa'] is None]
if sgd:
    x = [r['lr'] / r['paczka'] for r in sgd]; y = [r['lindemann'] for r in sgd]
    for r, xi, yi in zip(sgd, x, y):
        a.plot(xi, yi, 'o', color='#ffd166' if r['paczka'] == 32 else '#4a8cff', ms=8); a.annotate(f"lr {r['lr']}, B {r['paczka']}", (xi, yi), fontsize=7, textcoords='offset points', xytext=(4, 4))
    xs = np.array(sorted(x));
    if len(xs) >= 3:
        p = np.polyfit(np.log(x), np.log(y), 1); a.plot(xs, np.exp(p[1]) * xs ** p[0], '--', color='#8a8f99', label=f'klatka ∝ (lr/B)^{p[0]:.2f}')
adam = [r for k, r in glowne.items() if r['faza'] == 'wczesne' and r['ziarno'] == 47 and r['bieg'].startswith('lr_')]
if adam:
    a.plot([r['lr'] / 32 for r in adam], [r['lindemann'] for r in adam], 's-', color='#ff8c42', label='AdamW (lr/32)')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('lr / paczka'); a.set_ylabel('klatka / odstęp sąsiadów (S1)'); a.set_title('SGD z pędem: temperatura lr/B', fontsize=10); a.grid(True, which='both', alpha=0.4); a.legend(fontsize=7)
# (1,3) warstwy
a = ax[1, 3]
war = sorted([r for r in rows.values() if r['warstwa'] is not None], key=lambda r: r['warstwa'])
if war:
    L = [r['warstwa'] for r in war]
    a.plot(L, [r['lindemann'] for r in war], 'o-', color='#ffd166', label='klatka / odstęp (S1)')
    a.plot(L, [r['C1'] for r in war], 's-', color='#4a8cff', label='C₁ heterogeniczność (CR)')
    a.plot(L, [r['cross'] or np.nan for r in war], 'd-', color='#ff4433', label='C między zdjęciami')
    a.plot(L, [r['f_q'] for r in war], '^-', color='#4cd964', label='plateau Fs f_q')
    a2 = a.twinx(); a2.plot(L, [r['tau_1e'] if r['tau_1e'] and np.isfinite(r['tau_1e']) else np.nan for r in war], 'x--', color='#c77dff', label='τ_1e'); a2.set_yscale('log'); a2.set_ylabel('τ_1e (kroki)', color='#c77dff')
    a.set_xticks(L)
a.set_xlabel('blok (warstwa) 0–5'); a.set_title('Skan po warstwach (lr 5e-4, rzut 128D)', fontsize=10); a.grid(True, alpha=0.4); a.legend(fontsize=7, loc='upper left')
# (2,0..3) drugi model i drugi zbiór: słupki klatka, τ_1e, C1, cross dla vit/cnn × detekcja/Imagenette przy trzech lr
grupy = [('ViT detekcja', lambda r: r['model'] == 'vit' and r['zadanie'] == 'detekcja' and r['opt'] == 'adamw' and r['faza'] == 'wczesne' and r['ziarno'] == 47 and r['bieg'].startswith('lr_')),
         ('CNN detekcja', lambda r: r['model'] == 'cnn' and r['zadanie'] == 'detekcja'),
         ('ViT Imagenette', lambda r: r['model'] == 'vit' and r['zadanie'] == 'klasyfikacja' and r['faza'] == 'wczesne'),
         ('CNN Imagenette', lambda r: r['model'] == 'cnn' and r['zadanie'] == 'klasyfikacja')]
kolg = ['#ffd166', '#4a8cff', '#4cd964', '#ff4433']
for j, (pole, tyt, logy) in enumerate([('lindemann', 'klatka / odstęp sąsiadów (S1)', False), ('tau_1e', 'τ_1e Fs (kroki)', True), ('C1', 'heterogeniczność C₁ (CR)', False), ('cross', 'C ruchliwości z sąsiadami z innych zdjęć', False)]):
    a = ax[2, j]
    for g, (et, sel) in enumerate(grupy):
        rr = sorted([r for r in rows.values() if r['warstwa'] is None and sel(r)], key=lambda r: r['lr'] or 0)
        if rr:
            a.plot([r['lr'] for r in rr], [r[pole] if r[pole] is not None else np.nan for r in rr], 'o-', color=kolg[g], label=et)
    a.set_xscale('log'); a.set_xlabel('lr (AdamW)'); a.set_title(tyt, fontsize=10); a.grid(True, which='both', alpha=0.4)
    if logy:
        a.set_yscale('log')
    if j == 0:
        a.legend(fontsize=8)
fig.suptitle('Jaka to substancja? Testy heterogeniczności zamrożonej, korelacji krzyżowej par, rozbiegania, rozkładu energii; pokrętła SGD, warstwy, drugi model, drugi zbiór', fontsize=12)
fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.07, hspace=0.45, wspace=0.28)
fig.savefig(os.path.join(HERE, 'substancja.png'))
print('zapisano substancja.png, SUBSTANCJA.json')
