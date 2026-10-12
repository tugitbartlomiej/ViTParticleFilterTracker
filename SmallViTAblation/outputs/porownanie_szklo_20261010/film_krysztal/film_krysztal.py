"""Film „kryształ tokenów”: jedno zdjęcie uczące (TR_0001_S1_P03 klatka 274, zdjęcie 0 eksportu ruch60), 806 checkpointów biegu 35 000 (60 epok).
Panele: (1) zdjęcie z siatką patchy 20 × 30 pokolorowaną amplitudą drgań tokenu wokół węzła i ramką narzędzia; (2) siatka tokenów w latencie
(rzut 128D → PCA2 wyrównana Prokrustesem do poprzedniej klatki), linie = sąsiednie patche na obrazie, kolor = amplituda drgań;
(3) mapa: ile z 6 sąsiadów tokenu w latencie to sąsiednie patche; (4) przebiegi w czasie: IoU i pewność zdjęcia, średnia amplituda,
odsetek sąsiadów siatkowych, wraz z kursorem. Amplituda = |x(t) − średnia x po ±5 checkpointach|, x = token − średnia tokenów zdjęcia.
Uruchomienie: F:/Python/3.13/python.exe film_krysztal.py [indeks_zdjęcia=0] → krysztal_tokenow.mp4"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).parent
EKS = HERE.parents[2] / 'assets' / 'RuchCzasteczek_ViT_szklo_2026-10' / '03_pomiar_naturalny_60_epok_eden' / 'ruch60_rp128_50tr_50val'
POM = HERE.parent / 'eden' / 'wyniki' / 'pomiar60' / 'manifest.json'
J = int(sys.argv[1]) if len(sys.argv) > 1 else 0
H, W, TOK, EP, PL = 20, 30, 600, 1094, 16
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

META = json.load(open(EKS / 'META.json', encoding='utf-8'))
kroki = np.array(META['kroki'])
zdj = META['zdjecia'][J]
mp = json.load(open(POM, encoding='utf-8'))
wiersz = J if zdj['split'] == 'TRAIN' else 3000 + J - 50
im_meta = mp['images'][wiersz]
assert im_meta['frame_id'] == zdj['frame_id']
X = np.asarray(np.load(EKS / 'rp128.npy', mmap_mode='r')[:, J]).astype(np.float32)          # (K, 600, 128)
X -= X.mean(1, keepdims=True)
oceny = np.load(EKS / 'oceny.npy')[:, J]                                                       # (K, 2)
K = len(X)
log(f'zdjęcie {J} ({zdj["split"]}, {zdj["frame_id"]}), {K} checkpointów')

# amplituda drgań wokół węzła
c = np.stack([X[max(0, t - 5):t + 6].mean(0) for t in range(K)])
amp = np.linalg.norm(X - c, axis=-1)                                                           # (K, 600)
vmax = float(np.percentile(amp, 99))

# siatka i sąsiedzi
siatka = np.stack([np.arange(TOK) // W, np.arange(TOK) % W], 1)
Dg = cdist(siatka, siatka); sas_s = (Dg <= np.sqrt(2) + 1e-9) & (Dg > 0)
wiaz = [(i, i + 1) for i in range(TOK) if i % W < W - 1] + [(i, i + W) for i in range(TOK - W)]
frac_sas = np.empty((K, TOK))
for t in range(K):
    Dm = cdist(X[t], X[t]); np.fill_diagonal(Dm, np.inf)
    frac_sas[t] = sas_s[np.arange(TOK)[:, None], np.argsort(Dm, 1)[:, :6]].mean(1)

# PCA2 + Prokrustes
Y = np.empty((K, TOK, 2), np.float32); prev = None
for t in range(K):
    Z = X[t] - X[t].mean(0)
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    y = Z @ Vt[:2].T
    if prev is None:
        if y[siatka[:, 1] == W - 1, 0].mean() < y[siatka[:, 1] == 0, 0].mean():
            y[:, 0] *= -1
        if y[siatka[:, 0] == H - 1, 1].mean() > y[siatka[:, 0] == 0, 1].mean():
            y[:, 1] *= -1
    else:
        U_, _, Vt2 = np.linalg.svd(y.T @ prev); y = y @ (U_ @ Vt2)
    y /= np.sqrt((y ** 2).sum(1).mean())
    Y[t] = y; prev = y
log('przygotowane: amplituda, sąsiedzi, PCA2')

b = im_meta['bbox_xyxy_480']
cx = (siatka[:, 1] + 0.5) * PL; cy = (siatka[:, 0] + 0.5) * PL
w_ramce = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
import cv2  # noqa: E402
foto = cv2.cvtColor(cv2.imread(str(HERE / 'zdjecie_train0.jpg')), cv2.COLOR_BGR2RGB) if J == 0 else None
foto = cv2.resize(foto, (480, 320), interpolation=cv2.INTER_AREA) if foto is not None else np.zeros((320, 480, 3), np.uint8)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99',
                     'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a',
                     'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], left=0.04, right=0.98, top=0.9, bottom=0.07, wspace=0.12, hspace=0.25)
a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1]); a3 = fig.add_subplot(gs[0, 2]); a4 = fig.add_subplot(gs[1, :])
cmap = plt.cm.inferno
# panel 1
a1.imshow(foto, extent=(0, 480, 320, 0))
pole = a1.imshow(amp[0].reshape(H, W), extent=(0, 480, 320, 0), cmap=cmap, vmin=0, vmax=vmax, alpha=0.55, interpolation='nearest')
a1.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#4af0ff', lw=2))
for g in range(0, 481, PL):
    a1.axvline(g, color='#ffffff', lw=0.2, alpha=0.25)
for g in range(0, 321, PL):
    a1.axhline(g, color='#ffffff', lw=0.2, alpha=0.25)
a1.set_xlim(0, 480); a1.set_ylim(320, 0); a1.set_xticks([]); a1.set_yticks([])
a1.set_title('zdjęcie: patche 16 × 16 pokolorowane amplitudą drgań tokenu', fontsize=12)
# panel 2
lim = float(np.abs(Y).max()) * 1.05
lc = LineCollection([], colors='#8a8f99', linewidths=0.5, alpha=0.6); a2.add_collection(lc)
sc = a2.scatter(Y[0, :, 0], Y[0, :, 1], s=14, c=amp[0], cmap=cmap, vmin=0, vmax=vmax, zorder=3)
sc_r = a2.scatter(Y[0, w_ramce, 0], Y[0, w_ramce, 1], s=40, facecolors='none', edgecolors='#4af0ff', lw=1.0, zorder=4)
a2.set_xlim(-lim, lim); a2.set_ylim(-lim, lim); a2.set_aspect('equal'); a2.set_xticks([]); a2.set_yticks([])
a2.set_title('kryształ w latencie: linie = sąsiednie patche na obrazie', fontsize=12)
cb = fig.colorbar(sc, ax=a2, fraction=0.04, pad=0.01); cb.set_label('amplituda drgań wokół węzła')
# panel 3
mapa = a3.imshow(frac_sas[0].reshape(H, W), cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
a3.add_patch(Rectangle((b[0] / PL - 0.5, b[1] / PL - 0.5), (b[2] - b[0]) / PL, (b[3] - b[1]) / PL, fill=False, ec='#ff4433', lw=2))
a3.set_xticks([]); a3.set_yticks([]); fig.colorbar(mapa, ax=a3, fraction=0.03, pad=0.01).set_label('udział sąsiadów siatkowych')
a3.set_title('ile z 6 sąsiadów w latencie to sąsiednie patche', fontsize=12)
# panel 4
xs = np.arange(K)
a4.plot(xs, oceny[:, 1], color='#4af0ff', lw=1.2, label='IoU zdjęcia')
a4.plot(xs, oceny[:, 0], color='#c77dff', lw=1.0, alpha=0.8, label='pewność')
a4.plot(xs, frac_sas.mean(1), color='#ffd166', lw=1.4, label='sąsiedzi w latencie = sąsiednie patche')
a4.plot(xs, amp.mean(1) / vmax, color='#ff7a45', lw=1.0, alpha=0.9, label='średnia amplituda drgań (skala)')
a4.axhline(8 / 599, color='#8a8f99', ls=':', lw=1)
kur = a4.axvline(0, color='#ffffff', lw=1.5)
tick_k = [0, 100, 500, 1000, EP * 2, EP * 5, EP * 10, EP * 20, EP * 40, EP * 60]
a4.set_xticks([int(np.argmin(abs(kroki - k))) for k in tick_k])
a4.set_xticklabels(['0', '100', '500', '1000', 'ep 2', 'ep 5', 'ep 10', 'ep 20', 'ep 40', 'ep 60'])
a4.set_xlim(0, K - 1); a4.set_ylim(0, 1.02); a4.grid(True, alpha=0.35); a4.legend(loc='lower right', fontsize=10, ncol=4)
a4.set_xlabel('checkpoint (co 10 kroków do 1000, potem co 100)')
tytul = fig.suptitle('', fontsize=17, y=0.975)
fig.text(0.04, 0.925, 'Mały ViT, 600 tokenów jednego zdjęcia, rzut 128D latentu GELU ostatniego MLP. Węzły = pozycje patchy, drgania = szum kroku AdamW, '
         'odkształcenie = treść obrazu.', fontsize=11, color='#cfd3d8')

out = HERE / ('krysztal_tokenow.mp4' if J == 0 else f'krysztal_tokenow_zdj{J}.mp4')
fig.canvas.draw()
wid, hei = fig.canvas.get_width_height()
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{wid}x{hei}', '-r', '30', '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
klatki = list(range(K)) + [K - 1] * 60                                         # 2 s stopklatki na końcu
for n, t in enumerate(klatki):
    pole.set_data(amp[t].reshape(H, W))
    lc.set_segments([[Y[t, i], Y[t, k]] for i, k in wiaz])
    sc.set_offsets(Y[t]); sc.set_array(amp[t]); sc_r.set_offsets(Y[t, w_ramce])
    mapa.set_data(frac_sas[t].reshape(H, W)); kur.set_xdata([t, t])
    tytul.set_text(f'Kryształ tokenów — krok {kroki[t]}  (epoka {kroki[t] / EP:.2f})   IoU {oceny[t, 1]:.2f}   sąsiedzi siatkowi {frac_sas[t].mean():.0%}')
    fig.canvas.draw()
    ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    if n % 100 == 0:
        log(f'klatka {n}/{len(klatki)}')
ff.stdin.close(); ff.wait()
log('zapisano', out, f'{out.stat().st_size / 1e6:.1f} MB')
fig.savefig(HERE / 'krysztal_ostatnia_klatka.png')
