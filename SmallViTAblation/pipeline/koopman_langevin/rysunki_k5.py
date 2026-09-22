#!/usr/bin/env python3
"""Rysunki objasniajace kroki ramienia K5 na prawdziwych klatkach zapisanych lokalnie.

Zrodlo: nagranie czastek detektora L0 (CZASTKI_b0500) na 16 gestych klipach i te same klatki JPEG.
Ten sam detektor i ta sama procedura, ktora policzyla kod tozsamosci calej puli na Edenie.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
ROOT = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker')
sys.path.insert(0, str(ROOT))
from SmallViTAblation.pipeline.koopman_langevin.patch_analysis import (generalised_koopman, load_recording,
                                                                       translational_pairs)

REC = ROOT / 'SmallViTAblation/results/koopman_langevin_epochs/czastki/CZASTKI_b0500'
FRAMES = ROOT / 'SmallViTAblation/feature_cache/detector_probe_clips/frames'
OUT = ROOT / 'SmallViTAblation/results/koopman_langevin_epochs/rysunki_k5'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 130})


def obraz(fid: str) -> Image.Image:
    clip, n = fid.split('/')[-2], fid.split('/')[-1]
    return Image.open(FRAMES / clip / f'{n}.jpg').convert('RGB')


def main() -> int:
    rec = load_recording(REC)
    h, w = rec['meta']['grid']
    fids = rec['frame_ids']
    clips = np.array([f.split('/')[-2] for f in fids])
    box, score, mask = rec['pred_box'].astype(float), rec['score'].astype(float), rec['maska_pred']
    print(f'nagranie: {len(fids)} klatek, siatka {h}x{w}, klipow {len(set(clips))}', flush=True)

    # ---------------------------------------------------------------- rysunek 1: klatka, kwadraty, ramka
    good = np.flatnonzero((score > .9) & (box[:, 2] > 40) & (box[:, 2] < 260))
    i = int(good[len(good) // 3])
    im = obraz(fids[i]); W, H = im.size
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.1))
    for a in ax:
        a.imshow(im); a.set_xticks([]); a.set_yticks([])
    ax[0].set_title('1. Klatka z filmu, jak widzi ja czlowiek')
    for k in range(1, w):
        ax[1].axvline(k * W / w, color='w', lw=.3, alpha=.6)
    for k in range(1, h):
        ax[1].axhline(k * H / h, color='w', lw=.3, alpha=.6)
    ax[1].set_title(f'2. Siec dzieli klatke na {h*w} kwadratow\ni kazdy opisuje {rec["meta"]["K"]} liczbami')
    m = mask[i].reshape(h, w)
    ax[2].imshow(np.where(m, 1., np.nan), extent=(0, W, H, 0), cmap='autumn', alpha=.45, vmin=0, vmax=1, interpolation='nearest')
    ax[2].add_patch(Rectangle((box[i, 0], box[i, 1]), box[i, 2], box[i, 3], fill=False, ec='yellow', lw=1.6))
    ax[2].set_title(f'3. Ramka detektora (zolta) i {int(m.sum())} kwadratow\nnarzedzia, ktore biore dalej')
    fig.tight_layout(); fig.savefig(OUT / '01_klatka_i_kwadraty.png', bbox_inches='tight'); plt.close(fig)

    # ---------------------------------------------------------------- rysunek 2: pary przy przesunieciu
    clip = clips[i]
    idx = np.flatnonzero((clips == clip) & (score > .8) & (box[:, 2] > 40))
    cx = box[idx, 0] + box[idx, 2] / 2
    a_i, b_i = int(idx[np.argmin(cx)]), int(idx[np.argmax(cx)])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    kolory = plt.cm.tab10(np.linspace(0, 1, 10))
    for a, j in zip(ax, (a_i, b_i)):
        a.imshow(obraz(fids[j])); a.set_xticks([]); a.set_yticks([])
        a.add_patch(Rectangle((box[j, 0], box[j, 1]), box[j, 2], box[j, 3], fill=False, ec='yellow', lw=1.5))
    para = 0
    for gi in np.flatnonzero(mask[a_i]):
        ry, rx = divmod(int(gi), w)
        u = ((rx + .5) / w * 720 - box[a_i, 0]) / max(box[a_i, 2], 1)
        v = ((ry + .5) / h * 480 - box[a_i, 1]) / max(box[a_i, 3], 1)
        px = int((box[b_i, 0] + u * box[b_i, 2]) / 720 * w)
        py = int((box[b_i, 1] + v * box[b_i, 3]) / 480 * h)
        if not (0 <= px < w and 0 <= py < h and mask[b_i, py * w + px]) or para >= 8:
            continue
        if (gi % 3) or para >= 8:
            continue
        c = kolory[para % 10]
        ax[0].add_patch(Rectangle((rx * W / w, ry * H / h), W / w, H / h, fill=False, ec=c, lw=1.8))
        ax[1].add_patch(Rectangle((px * W / w, py * H / h), W / w, H / h, fill=False, ec=c, lw=1.8))
        para += 1
    ax[0].set_title('4. Narzedzie po lewej stronie kadru')
    ax[1].set_title('...i to samo narzedzie gdzie indziej.\nKwadraty tego samego koloru to ta sama czesc narzedzia')
    fig.suptitle('Parujemy kwadraty narzedzia z dwoch klatek tego samego filmu, gdy narzedzie sie przesunelo', y=1.06)
    fig.tight_layout(); fig.savefig(OUT / '02_pary_przy_przesunieciu.png', bbox_inches='tight'); plt.close(fig)

    # ---------------------------------------------------------------- kod tozsamosci
    x0, x1 = translational_pairs(rec, (h, w), per_clip=400)
    lam, modes, mu = generalised_koopman(x0, x1)
    print(f'pary {len(x0)}, stabilnosc osmiu kierunkow {np.round(lam[:8], 2).tolist()}', flush=True)
    slow, fast = modes[:, :8], modes[:, -8:]
    prof = rec['profil_red'].astype(np.float64)
    kod = np.full((len(fids), 8), np.nan)
    szybkie = np.full(len(fids), np.nan)
    for t in range(len(fids)):
        if score[t] > .5 and mask[t].sum() >= 4:
            c = (prof[t, mask[t]] - mu)
            kod[t] = (c @ slow).mean(0)
            szybkie[t] = (c @ fast).mean(0)[0]
    ok = np.isfinite(kod).all(1)

    # rysunek 3: stabilny kierunek przezywa przesuniecie narzedzia, niestabilny nie
    a0, a1 = (x0 - mu) @ modes, (x1 - mu) @ modes
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for a, k, tit, col in ((ax[0], 0, 'kierunek NAJSTABILNIEJSZY', 'tab:blue'), (ax[1], -1, 'kierunek NAJMNIEJ stabilny', 'tab:red')):
        u0, u1 = a0[:, k] / a0[:, k].std(), a1[:, k] / a1[:, k].std()
        a.scatter(u0, u1, s=3, alpha=.25, c=col)
        a.plot([-3, 3], [-3, 3], 'k--', lw=.8)
        a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect('equal')
        a.set_xlabel('wartosc w pierwszej klatce'); a.set_ylabel('wartosc w drugiej klatce')
        a.set_title(f'{tit}\nzgodnosc {np.corrcoef(u0, u1)[0, 1]:.2f}')
    fig.suptitle('5. Kazda kropka to ta sama czesc narzedzia w dwoch klatkach, miedzy ktorymi narzedzie sie przesunelo.\n'
                 'Na lewym wykresie liczba przezywa przesuniecie (kropki na przekatnej), na prawym gubi sie.', y=1.11)
    fig.tight_layout(); fig.savefig(OUT / '03_co_przezywa_przesuniecie.png', bbox_inches='tight'); plt.close(fig)
    print('zgodnosc par: najstabilniejszy %.2f, najmniej stabilny %.2f' % (np.corrcoef(a0[:, 0], a1[:, 0])[0, 1], np.corrcoef(a0[:, -1], a1[:, -1])[0, 1]), flush=True)

    # rysunek 4: mapa stanow narzedzia z miniaturami
    z = (kod[ok] - kod[ok].mean(0)) / kod[ok].std(0)
    rows = np.flatnonzero(ok)
    u, s, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    p = (z - z.mean(0)) @ vt[:2].T
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.scatter(p[:, 0], p[:, 1], s=6, c='0.75', label='klatki puli')
    znane = np.flatnonzero(np.isin(clips[rows], sorted(set(clips))[:4]))
    ax.scatter(p[znane, 0], p[znane, 1], s=14, c='tab:blue', label='klatki juz oznaczone przez czlowieka')
    d = np.linalg.norm(p[:, None, :] - p[None, znane, :], axis=2).min(1)
    pick = [int(np.argmax(d))]
    for _ in range(5):
        dd = np.minimum(d, np.linalg.norm(p[:, None, :] - p[None, pick, :], axis=2).min(1))
        pick.append(int(np.argmax(dd))); d = dd
    ax.scatter(p[pick, 0], p[pick, 1], s=90, marker='*', c='tab:red', label='klatki, ktore wybiera K5 (najdalsze od oznaczonych)')
    for rank, k in enumerate(pick[:6]):
        t = rows[k]
        b = box[t]
        crop = obraz(fids[t]).crop((max(b[0] - 20, 0), max(b[1] - 20, 0), min(b[0] + b[2] + 20, 720), min(b[1] + b[3] + 20, 480))).resize((84, 64))
        ab = matplotlib.offsetbox.AnnotationBbox(matplotlib.offsetbox.OffsetImage(np.asarray(crop), zoom=.75),
                                                 (p[k, 0], p[k, 1]), frameon=True, pad=.15,
                                                 bboxprops=dict(edgecolor='tab:red', lw=1.2),
                                                 xybox=(60 * np.cos(rank / 6 * 2 * np.pi), 60 * np.sin(rank / 6 * 2 * np.pi)),
                                                 boxcoords='offset points', arrowprops=dict(arrowstyle='-', color='tab:red', lw=.8))
        ax.add_artist(ab)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('6. Mapa stanow narzedzia: kazdy punkt to jedna klatka.\nBlisko siebie leza klatki, w ktorych siec widzi narzedzie w podobnej sytuacji.')
    ax.legend(loc='upper left', fontsize=8, framealpha=.9)
    fig.tight_layout(); fig.savefig(OUT / '04_mapa_stanow_i_wybor.png', bbox_inches='tight'); plt.close(fig)
    print('zapisane do', OUT, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
