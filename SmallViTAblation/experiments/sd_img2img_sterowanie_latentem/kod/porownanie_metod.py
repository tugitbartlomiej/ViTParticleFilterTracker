"""Porównanie sposobów doboru warunków początkowych SD na jednym obrazie (0055/4019): losowe szukanie szumu, gradient
wstecz przez SD (mieszanka szumów; mieszanka + warunek), regresja bez gradientu (mieszanka szumów), dla odniesienia
gradient po samym warunku (stały szum) i gradient wprost po szumie (przerwany, z logu).

Oś x: liczba wygenerowanych obrazów oraz czas z logów (iteracja gradientu kosztuje więcej niż zwykła generacja).
Oś y: najlepszy dotąd postęp latentu obrazu wzdłuż δ = z* − z0 (cały i widoczny dla fc2). Pod spodem najlepsze obrazy
każdej metody z ostrością, wynikiem w 50 krokach DDIM i najbliższą klatką TRAIN (DINO).
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/sd_img2img_sterowanie_latentem/kod/porownanie_metod.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
W = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem'
K = ROOT / 'experiments' / 'sd_img2img_sterowanie_latentem' / 'kod'
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def czasy(log, wzor):
    t = [float(m.group(1)) for l in log.read_text(encoding='utf-8', errors='ignore').splitlines() if (m := re.match(r'\[\s*([\d.]+) s\] ' + wzor, l))]
    return np.array(t)


def ostr(p):
    g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    serie = []
    p = W / 'optymalizacja_mieszanka' / 'WYNIK.json'
    mz = json.loads((p if p.exists() else W / 'optymalizacja_mieszanka' / 'LOSOWE_SZUKANIE.json').read_text(encoding='utf-8'))
    los = np.array([x['ulamek'] for x in mz['losowe']]); los_w = np.array([x['ulamek_widoczny'] for x in mz['losowe']])
    t_los = czasy(K / 'optymalizacja_mieszanka.log', 'losowy szum')
    dt_los = float(np.median(np.diff(t_los))) if len(t_los) > 2 else 2.0
    b = int(np.argmax(los))
    serie.append(dict(nazwa='losowe szukanie szumu', gen=np.arange(1, len(los) + 1), czas=np.arange(1, len(los) + 1) * dt_los,
                      u=np.maximum.accumulate(los), uw=np.array([los_w[:i + 1][np.argmax(los[:i + 1])] for i in range(len(los))]),
                      obraz=W / 'optymalizacja_mieszanka' / 'losowe_szukanie' / f's{b:02d}.png', spr50=None, szuk=None, kolor='gray'))
    for tryb, nazwa, kol in [('mieszanka', 'gradient: mieszanka szumów', 'tab:blue'), ('mieszanka_warunek', 'gradient: mieszanka + warunek', 'tab:purple')]:
        p = W / f'optymalizacja_{tryb}' / 'WYNIK.json'
        lg = K / f'optymalizacja_{tryb}.log'
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8')); w = d['wyniki']
        elif lg.exists():                # bieg przerwany przed zapisem WYNIK.json: iteracje z logu
            w = [dict(eval(m.group(2)), generacja=int(m.group(1))) for l in lg.read_text(encoding='utf-8', errors='ignore').splitlines()
                 if (m := re.search(r'it +(\d+) (\{.*?\})', l))]
            if not w:
                continue
            d = dict(najlepsza=max(w, key=lambda x: x['ulamek']), sprawdzenie=[])
            nazwa += f' (przerwany po it {w[-1]["generacja"]})'
        else:
            continue
        t = czasy(K / f'optymalizacja_{tryb}.log', r'it +\d+'); dt = float(np.median(np.diff(t))) if len(t) > 2 else 7.0
        u = np.array([x['ulamek'] for x in w]); uw = np.array([x['ulamek_widoczny'] for x in w])
        nb = d['najlepsza']['generacja']
        serie.append(dict(nazwa=nazwa, gen=len(los) + np.arange(1, len(w) + 1), czas=len(los) * dt_los + np.arange(1, len(w) + 1) * dt,
                          u=np.maximum.accumulate(u), uw=np.array([uw[:i + 1][np.argmax(u[:i + 1])] for i in range(len(u))]),
                          obraz=W / f'optymalizacja_{tryb}' / 'obrazy' / f'g{nb:02d}.png',
                          spr50=next((s for s in d['sprawdzenie'] if s['kroki'] == 50), None), szuk=d.get('szukanie_train'), kolor=kol, dt=dt))
    p = W / 'optymalizacja_regresja' / 'WYNIK.json'
    if p.exists():
        d = json.loads(p.read_text(encoding='utf-8')); w = d['wyniki']
        t = czasy(K / 'optymalizacja_regresja.log', r'krok +\d+'); dt_k = float(np.median(np.diff(t))) if len(t) > 2 else 70.0
        gen = np.array([x['generacji'] for x in w]); u = np.array([x['ulamek'] for x in w]); uw = np.array([x['ulamek_widoczny'] for x in w])
        dt_gen = dt_k / max(float(np.median(np.diff(gen))) if len(gen) > 2 else 33, 1)
        serie.append(dict(nazwa='regresja bez gradientu: mieszanka szumów', gen=len(los) + gen, czas=len(los) * dt_los + gen * dt_gen,
                          u=np.maximum.accumulate(u), uw=uw, obraz=W / 'optymalizacja_regresja' / 'obrazy' / f'g{w[-1]["krok"]:02d}.png',
                          spr50=next((s for s in d.get('sprawdzenie', []) if s['kroki'] == 50), None), szuk=d.get('szukanie_train'), kolor='tab:green'))
    odn = []
    p = W / 'optymalizacja_warunku' / 'WYNIK.json'
    if p.exists():
        d = json.loads(p.read_text(encoding='utf-8')); u = np.array([x['ulamek'] for x in d['wyniki']])
        odn.append(('gradient po samym warunku, stały szum ziarno 0 (start 2,8 %)', np.arange(1, len(u) + 1), np.maximum.accumulate(u), 'tab:orange'))
    lg = K / 'optymalizacja_szum.log'
    if lg.exists():
        u = np.array([float(m.group(1)) for l in lg.read_text(encoding='utf-8', errors='ignore').splitlines() if (m := re.search(r"it +\d+ \{.*?'ulamek': ([-\d.]+)", l))])
        if len(u):
            odn.append(('gradient wprost po szumie (rozmywał obraz, przerwany)', len(los) + np.arange(1, len(u) + 1), np.maximum.accumulate(u), 'tab:red'))

    fig = plt.figure(figsize=(18, 13), layout='constrained')
    gs = fig.add_gridspec(2, max(len(serie), 2), height_ratios=[1.3, 1])
    a = fig.add_subplot(gs[0, :max(len(serie), 2) // 2])
    for s in serie:
        a.plot(s['gen'], 100 * s['u'], '-', c=s['kolor'], lw=2, label=s['nazwa'])
        a.plot(s['gen'], 100 * s['uw'], ':', c=s['kolor'], lw=1.5)
    for nazwa, x, u, kol in odn:
        a.plot(x, 100 * u, '--', c=kol, lw=1, alpha=.7, label='odniesienie: ' + nazwa)
    a.axvline(len(los), c='k', lw=.6); a.text(len(los), a.get_ylim()[0], ' koniec losowego szukania', fontsize=8, va='bottom')
    a.set_xlabel('liczba wygenerowanych obrazów (losowe szukanie + metoda)'); a.set_ylabel('najlepszy dotąd postęp wzdłuż δ [%]')
    a.set_title('postęp vs liczba obrazów (linia ciągła: cały przyrost, kropki: część widoczna dla fc2)'); a.legend(fontsize=8); a.grid(alpha=.3)
    a = fig.add_subplot(gs[0, max(len(serie), 2) // 2:])
    for s in serie:
        a.plot(np.array(s['czas']) / 60, 100 * s['u'], '-', c=s['kolor'], lw=2, label=s['nazwa'])
    a.set_xlabel('czas obliczeń z logów [min] (RTX 3070)'); a.set_ylabel('najlepszy dotąd postęp wzdłuż δ [%]')
    a.set_title('postęp vs czas'); a.legend(fontsize=8); a.grid(alpha=.3)
    for i, s in enumerate(serie):
        a = fig.add_subplot(gs[1, i]); a.imshow(cv2.cvtColor(cv2.imread(str(s['obraz'])), cv2.COLOR_BGR2RGB)); a.set_axis_off()
        t = f'{s["nazwa"]}\npostęp {100 * s["u"][-1]:.1f} % (widoczny {100 * s["uw"][-1]:.1f} %), ostrość {ostr(s["obraz"]):.0f}'
        if s['spr50']:
            t += f'\nw 50 krokach DDIM: {100 * s["spr50"]["ulamek"]:.1f} %'
        if s['szuk']:
            t += f'\nTRAIN top-1: {s["szuk"][0]["film"][3:7]}/{s["szuk"][0]["klatka"]} (postęp {100 * s["szuk"][0]["ulamek"]:.1f} %)'
        a.set_title(t, fontsize=9, color=s['kolor'])
    fig.suptitle('Dobór warunków początkowych SD pod sztuczny latent: losowe szukanie, gradient wstecz, regresja', fontsize=14)
    fig.savefig(W / 'POROWNANIE_METOD.png', dpi=110); plt.close(fig)
    pods = [dict(metoda=s['nazwa'], obrazow=int(s['gen'][-1]), czas_min=float(s['czas'][-1]) / 60, postep=float(s['u'][-1]), postep_widoczny=float(s['uw'][-1]),
                 ostrosc=ostr(s['obraz']), postep_50_krokow=s['spr50']['ulamek'] if s['spr50'] else None,
                 train_top1=s['szuk'][0] if s['szuk'] else None) for s in serie]
    (W / 'POROWNANIE_METOD.json').write_text(json.dumps(pods, indent=1, ensure_ascii=False), encoding='utf-8')
    for p in pods:
        print({a: (round(v, 4) if isinstance(v, float) else v) for a, v in p.items() if a != 'train_top1'})


if __name__ == '__main__':
    main()
