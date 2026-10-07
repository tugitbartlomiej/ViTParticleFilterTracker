"""Rysunek i tabela wyników eksperymentu latent_cel_do_obrazow (z plików ETAP*.json w outputs/latent_cel_do_obrazow)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'latent_cel_do_obrazow'
rd = lambda n: json.loads((OUT / n).read_text(encoding='utf-8')) if (OUT / n).exists() else None
C = rd('ETAP1_CELE.json'); P = rd('ETAP2C_PCHNIECIE.json'); U = rd('ETAP3_UCZENIE.json')
F = rd('ETAP2D_FC2.json'); M = rd('ETAP4_MALA.json'); KL = rd('KONTROLA_LOSOWA.json'); K = rd('ETAP2A_KONWERTER.json')
NAZWY = dict(A='A oryginał (powtórka)', E='E same cele', B='B konwerter', C='C losowe', D='D SD -> konwerter')
KOL = dict(A='#555555', E='#2ca02c', B='#1f77b4', C='#ff7f0e', D='#9467bd')

f, ax = plt.subplots(2, 3, figsize=(17, 9.5))
a = ax[0, 0]
pk = ('100', '200', '300')
a.plot([0, 100, 200, 300], [0] + [np.mean([c['punkty'][p]['iou_top'] for c in C['cele']]) for p in pk], 'o-', label='IoU najpewniejszej')
a.plot([0, 100, 200, 300], [np.mean([c['przed']['pewnosc_top'] for c in C['cele']])] +
       [np.mean([c['punkty'][p]['pewnosc_top'] for c in C['cele']]) for p in pk], 's-', label='pewność')
a2 = a.twinx(); a2.plot([0, 100, 200, 300], [0] + [np.mean([c['punkty'][p]['norma_dz'] for c in C['cele']]) for p in pk], 'k:', label='||Δz||')
a2.set_ylabel('||Δz|| (||z|| ≈ 240)')
a.set_title('1. Korekta oracle 24 celów (cel detekcja, 768 kanałów)'); a.set_xlabel('krok korekty'); a.legend(loc='center right')
a = ax[0, 1]
zr = ['cel (sam obraz z etykietą)', 'konwerter', 'SD -> konwerter', 'obraz SD', 'losowa z puli']
pod = {}
for r in P['wyniki']:
    z = next(x for x in zr if r['nazwa'].startswith(x))
    pod.setdefault(z, []).append(r)
x = np.arange(len(zr))
a.bar(x - .2, [np.mean([r['cos_do_korekty'] for r in pod[z]]) for z in zr], .4, label='cos(pchnięcie, korekta)')
a.bar(x + .2, [np.mean([r['zmiana_straty'] for r in pod[z]]) * 10 for z in zr], .4, label='zmiana straty celu × 10')
a.axhline(0, color='k', lw=.6); a.set_xticks(x); a.set_xticklabels(['cel', 'konwerter', 'SD->konw.', 'obraz SD', 'losowa'])
a.set_title('2. Pchnięcie jednego kroku (990) na latent celu'); a.legend(fontsize=8)
a = ax[0, 2]
if F:
    zf = list(F['podsumowanie'])
    for i, m in enumerate(('cos_calosc', 'cos_widoczna', 'cos_niewidoczna', 'cos_wyjscia_fc2')):
        a.bar(np.arange(len(zf)) + (i - 1.5) * .2, [F['podsumowanie'][z][m] for z in zf], .2, label=m)
    a.set_xticks(range(len(zf))); a.set_xticklabels([z[:12] for z in zf], fontsize=8); a.axhline(0, color='k', lw=.6)
    a.legend(fontsize=7); a.set_title('2d. Pchnięcie a części korekty względem fc2')
else:
    a.axis('off')
a = ax[1, 0]
for r, v in U['ramiona'].items():
    h = v['historia']
    a.plot([x['krok'] for x in h], [x['iou_celow'] for x in h], 'o-', ms=3, color=KOL[r], label=NAZWY[r])
a.set_xlabel('krok uczenia'); a.set_ylabel('średnie IoU najpewniejszej ramki 24 celów'); a.legend(fontsize=8)
a.set_title('3. Cele w trakcie uczenia (ramiona, deterministycznie)')
a = ax[1, 1]
ram = list(U['ramiona'])
for j, k in enumerate((500, 1000)):
    a.bar(np.arange(len(ram)) + (j - .5) * .38, [next(x for x in U['ramiona'][r]['historia'] if x['krok'] == k)['map50_wal3000'] for r in ram],
          .38, label=f'krok {k}', color=[KOL[r] for r in ram], alpha=.5 + .5 * j)
a.set_xticks(range(len(ram))); a.set_xticklabels([NAZWY[r] for r in ram], rotation=15, fontsize=8)
a.set_ylim(.15, .4); a.set_title('3. mAP@0,5 wal3000 (jasne: 500, ciemne: 1000)')
a = ax[1, 2]
rr = [r for r in ram if r != 'A']
a.bar(range(len(rr)), [U['ramiona'][r]['cos_srednio'] for r in rr], color=[KOL[r] for r in rr])
for i, r in enumerate(rr):
    c = np.array(U['ramiona'][r]['cos_ruchu_celow_do_korekty'])
    a.scatter(np.full(len(c), i) + np.random.default_rng(i).uniform(-.2, .2, len(c)), c, s=8, c='k')
a.axhline(0, color='k', lw=.6); a.set_xticks(range(len(rr))); a.set_xticklabels([NAZWY[r] for r in rr], rotation=15, fontsize=8)
a.set_title('3. cos(z_ramię − z_A, korekta oracle) celów przy k1000')
f.suptitle('Od sztucznego latentu do zdjęć uczących: korekta oracle, zdjęcia z konwertera i SD, uczenie z podmianą 96 zdjęć bazy',
           fontsize=12)
f.tight_layout()
f.savefig(OUT / 'RYSUNEK.png', dpi=110)

# tabela
print('| ramię | IoU celów k500 / k1000 | średnie IoU 600-1000 | AP50 celów k1000 | mAP wal3000 k500 / k1000 | cos do korekty | filmy podmienione |')
print('|---|---|---|---|---|---|---|')
for r in ram:
    v = U['ramiona'][r]; h = {x['krok']: x for x in v['historia']}
    print(f"| {NAZWY[r]} | {h[500]['iou_celow']:.3f} / {h[1000]['iou_celow']:.3f} | "
          f"{np.mean([h[k]['iou_celow'] for k in range(600, 1001, 100)]):.3f} | {h[1000]['ap50_celow']:.3f} | "
          f"{h[500]['map50_wal3000']:.3f} / {h[1000]['map50_wal3000']:.3f} | "
          f"{v.get('cos_srednio', float('nan')):+.3f} | {len({p.split('/')[0] for p in v['podmienione']}) if v['podmienione'] else '-'} |")
print()
for z in zr:
    print(z, round(np.mean([r['cos_do_korekty'] for r in pod[z]]), 4), round(np.mean([r['zmiana_straty'] for r in pod[z]]), 5), len(pod[z]))
if KL:
    print('kontrola losowa cos(z_klatka − z_cel, δ):', round(KL['sredni'], 4), 'max z 20:', round(KL['max_sredni'], 4),
          'konwerter najlepszy:', round(np.mean([w['kandydaci'][0]['cos_do_celu'] for w in K['wyniki']]), 4))
if F:
    print(json.dumps(F['podsumowanie'], indent=1, ensure_ascii=False))
if M:
    for r in M['wyniki']:
        print('mała', r['cel'], r['ulamek'], [round(x, 3) for x in r['iou']], [round(x, 3) for x in r['pewnosc']],
              'SD cos', round(r.get('sd_cos_do_korekty', np.nan), 3), 'SD iou', r.get('sd_iou_z_etykieta_celu'),
              'konw cos', round(r.get('konwerter_cos_najlepszy', np.nan), 3))
