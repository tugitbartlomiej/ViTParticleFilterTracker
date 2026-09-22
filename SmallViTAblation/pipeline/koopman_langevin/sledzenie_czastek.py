#!/usr/bin/env python3
"""Czy kwadraty siedza tam, gdzie mysle, i czy "czastki trzymaja sie narzedzia" (styl DAVIS).

Dwa sprawdzenia i dwa rysunki na prawdziwych klatkach:

A. Poprawnosc przypisania kwadratow do miejsc obrazu. Test: czy kwadraty podobne do wzorca
   narzedzia (srednia tokenow z prawdziwej ramki) lezą w prawdziwej ramce. Jesli siatka byla by
   obrocona albo przestawiona, trafienia spadlyby do poziomu losu (udzial pola ramki w kadrze).

B. Dopasowanie kwadratow miedzy klatkami po samych liczbach (jak sledzenie punktow na niedzwiedziu
   z DAVIS): biore kwadrat na narzedziu w klatce odniesienia i w kazdej nastepnej klatce szukam
   kwadratu o najbardziej podobnym opisie. Mierze, jak czesto trafia w prawdziwa ramke narzedzia
   i porownuje z losem.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker')
sys.path.insert(0, str(ROOT))
from SmallViTAblation.pipeline.koopman_langevin.patch_analysis import grid_masks, load_recording

REC = ROOT / 'SmallViTAblation/results/koopman_langevin_epochs/czastki/CZASTKI_b0500'
PKG = ROOT / 'SmallViTAblation/feature_cache/detector_probe_clips'
OUT = ROOT / 'SmallViTAblation/results/koopman_langevin_epochs/rysunki_k5'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'figure.dpi': 130})


def obraz(fid: str) -> Image.Image:
    return Image.open(PKG / 'frames' / fid.split('/')[-2] / f'{fid.split("/")[-1]}.jpg').convert('RGB')


def main() -> int:
    rec = load_recording(REC)
    h, w = rec['meta']['grid']
    fids = rec['frame_ids']
    clips = np.array([f.split('/')[-2] for f in fids])
    tracks = json.loads((PKG / 'tracks.json').read_text(encoding='utf-8'))
    truth = grid_masks(tracks, fids, (h, w))                       # 1 w ramce, -1 pas brzegowy, 0 tlo, -2 brak
    prof = rec['profil_red'].astype(np.float64)
    unit = prof / np.maximum(np.linalg.norm(prof, axis=2, keepdims=True), 1e-9)
    score = rec['score'].astype(float)
    have = truth.min(1) > -2

    # ---------------------------------------------------------------- A. czy siatka lezy tam, gdzie mysle
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(have)
    wzorzec = np.zeros(prof.shape[2])
    for t in idx[rng.choice(len(idx), 400, replace=False)]:
        m = truth[t] == 1
        if m.sum():
            wzorzec += unit[t, m].mean(0)
    wzorzec /= np.linalg.norm(wzorzec)
    trafienia, losowo, trafienia_obrot = [], [], []
    for t in idx:
        sim = unit[t] @ wzorzec
        best = np.argsort(-sim)[:10]
        trafienia.append(float((truth[t][best] == 1).mean()))
        losowo.append(float((truth[t] == 1).mean()))
        obrot = sim.reshape(h, w).T.ravel()                        # gdyby siatka byla transponowana
        trafienia_obrot.append(float((truth[t][np.argsort(-obrot)[:10]] == 1).mean()))
    print(f'A. 10 kwadratow najpodobniejszych do wzorca narzedzia lezy w prawdziwej ramce w {100*np.mean(trafienia):.0f}% przypadkow')
    print(f'   los (udzial pola ramki w kadrze): {100*np.mean(losowo):.0f}%;  przy siatce transponowanej: {100*np.mean(trafienia_obrot):.0f}%')

    # ---------------------------------------------------------------- B. dopasowanie miedzy klatkami
    dobry = [c for c in np.unique(clips) if (have & (clips == c) & (score > .8)).sum() > 60]
    clip = dobry[1]
    sel = np.flatnonzero((clips == clip) & have & (score > .8))
    t0 = int(sel[len(sel) // 4])
    kolejne = [int(s) for s in sel[sel > t0][:7]]
    m0 = np.flatnonzero(truth[t0] == 1)
    sim0 = unit[t0, m0] @ wzorzec
    start = m0[np.argsort(-sim0)[:5]]                              # piec kwadratow najbardziej "narzedziowych"
    fig, ax = plt.subplots(2, 4, figsize=(14, 6.6))
    ax = ax.ravel()
    kolory = plt.get_cmap('tab10')(np.linspace(0, 1, 10))
    W, H = obraz(fids[t0]).size
    wewnatrz = []
    for a, t, tytul in zip(ax, [t0] + kolejne, ['klatka odniesienia'] + [f'+{3*(k+1)} klatek' for k in range(7)]):
        a.imshow(obraz(fids[t])); a.set_xticks([]); a.set_yticks([])
        for j, g0 in enumerate(start):
            g = g0 if t == t0 else int(np.argmax(unit[t] @ unit[t0, g0]))
            ry, rx = divmod(int(g), w)
            a.add_patch(Rectangle((rx * W / w, ry * H / h), W / w, H / h, fill=False, ec=kolory[j], lw=1.8))
            if t != t0:
                wewnatrz.append(truth[t][g] == 1)
        b = tracks[clip]['box'][tracks[clip]['frame'].index(int(fids[t].split('/')[-1]))]
        if b:
            a.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec='yellow', lw=1.2, ls='--'))
        a.set_title(tytul)
    udzial = float(np.mean(wewnatrz))
    los = float(np.mean([(truth[t] == 1).mean() for t in kolejne]))
    fig.suptitle('Czy kwadrat znaleziony po samych liczbach trzyma sie narzedzia?\n'
                 f'Kolorowe kwadraty: najpodobniejszy kwadrat w kazdej klatce. Zolta przerywana: prawdziwa ramka narzedzia. '
                 f'Trafienie w ramke {100*udzial:.0f}% wobec {100*los:.0f}% dla losu.', y=1.03)
    fig.tight_layout(); fig.subplots_adjust(hspace=.18, top=.84); fig.savefig(OUT / '05_czy_czastki_trzymaja_sie_narzedzia.png', bbox_inches='tight'); plt.close(fig)
    print(f'B. dopasowanie po liczbach trafia w prawdziwa ramke w {100*udzial:.0f}% (los {100*los:.0f}%)')

    # ---------------------------------------------------------------- B2. ten sam test na wielu klatkach
    ok, baza = [], []
    for c in dobry[:8]:
        s = np.flatnonzero((clips == c) & have & (score > .8))
        if len(s) < 40:
            continue
        r = int(s[len(s) // 4])
        mm = np.flatnonzero(truth[r] == 1)
        if len(mm) < 5:
            continue
        pocz = mm[np.argsort(-(unit[r, mm] @ wzorzec))[:5]]
        for t in s[(s > r)][:10]:
            for g0 in pocz:
                ok.append(truth[t][int(np.argmax(unit[t] @ unit[r, g0]))] == 1)
            baza.append((truth[t] == 1).mean())
    print(f'B2. na {len(dobry[:8])} klipach: trafienie {100*np.mean(ok):.0f}%, los {100*np.mean(baza):.0f}%')

    # ---------------------------------------------------------------- C. opis Eulera: siatka stoi
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.1))
    for a, t in zip(ax, [t0] + kolejne[:2]):
        a.imshow(obraz(fids[t])); a.set_xticks([]); a.set_yticks([])
        for k in range(1, w):
            a.axvline(k * W / w, color='w', lw=.25, alpha=.5)
        for k in range(1, h):
            a.axhline(k * H / h, color='w', lw=.25, alpha=.5)
        g = int(start[0]); ry, rx = divmod(g, w)
        a.add_patch(Rectangle((rx * W / w, ry * H / h), W / w, H / h, fill=False, ec='cyan', lw=2.2))
        b = tracks[clip]['box'][tracks[clip]['frame'].index(int(fids[t].split('/')[-1]))]
        if b:
            a.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, ec='yellow', lw=1.2, ls='--'))
    ax[0].set_title('Czastka to STALE miejsce w kadrze (niebieski kwadrat)')
    ax[1].set_title('Narzedzie plynie przez nia...')
    ax[2].set_title('...a kwadrat zostaje w tym samym miejscu')
    fig.savefig(OUT / '06_czastka_stoi_narzedzie_plynie.png', bbox_inches='tight'); plt.close(fig)
    print('zapisane do', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
