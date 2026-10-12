"""Film 3 z 3: „jak to się uczy” — 60 epok w RÓWNYM tempie (stała liczba kroków uczenia na sekundę filmu, bez spowolnień), scena 3D
z obracającą się kamerą. 600 tokenów obrazu TRAIN 0 (szescian.npz: x, y = płaszczyzna ruchu, głębia = 1. kierunek ruchu), studnie światła
w węzłach (średnie położenie ±300 kroków), cztery lasery na brzegach w płaszczyźnie siatki, cień siatki idealnej (krok 0) na podłodze.
Układ inercjalny: widać, jak uczenie przesuwa, obraca i odkształca całą siatkę. Kolor tokenu = odległość od studni.
Uruchomienie: F:/Python/3.13/python.exe film_lasery_3_uczenie.py [proba]"""
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from lasery_wspolne import EP, H, HERE, TOK, W, ZIELONY, dane, ffmpeg

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
D = dane(0)
kroki, iou, narz = D['kroki'], D['iou'], D['narz']; K = len(kroki)
g, A, Q = D['g3'], D['A3'], D['Q3']
SEK, FPS, KONIEC = 70, 30, 4
NF = SEK * FPS
krok_f = np.concatenate([np.linspace(0, kroki[-1], NF), np.full(KONIEC * FPS, kroki[-1])])
if PROBA:
    krok_f = krok_f[[0, 30, 300, 1200, len(krok_f) - 1]]
sA, sQ = PchipInterpolator(kroki, A, axis=0), PchipInterpolator(kroki, Q, axis=0)
iou_f = np.interp(krok_f, kroki, iou)
ruch = np.linalg.norm(A - Q, axis=-1); vmax = float(np.percentile(ruch, 97))
zl = float(np.percentile(np.abs(A[..., 2]), 99.5)) * 1.15
xl, yl = float(np.abs(A[..., 0]).max()) + 1.0, float(np.abs(A[..., 1]).max()) + 1.0
log(f'klatek {len(krok_f)} ({len(krok_f) / FPS:.0f} s), {kroki[-1] / NF:.0f} kroków na klatkę, głębia ±{zl:.1f}')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'figure.facecolor': '#03060a', 'text.color': '#e8e8e8'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ax = fig.add_axes((-0.05, -0.04, 1.1, 1.06), projection='3d', computed_zorder=False)
ax.set_facecolor('#03060a'); ax.set_axis_off()
ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl); ax.set_zlim(-zl, zl); ax.set_box_aspect((2 * xl, 2 * yl, 2 * zl * 1.3), zoom=1.25)
sas = np.array([(r * W + c, r * W + c + 1) for r in range(H) for c in range(W - 1)] + [(r * W + c, (r + 1) * W + c) for r in range(H - 1) for c in range(W)])
# podłoga: cień siatki idealnej
pod = g.copy(); pod[:, 2] = -zl
ax.add_collection3d(Line3DCollection([pod[[i, j]] for i, j in sas], colors='#2a3a33', linewidths=0.5, alpha=0.8), autolim=False)
# lasery: źródła w płaszczyźnie z = 0 na czterech bokach, wachlarze wiązek do krawędzi siatki
ZR = [(-xl, 0, 0), (xl, 0, 0), (0, -yl, 0), (0, yl, 0)]
kraw = {0: g[g[:, 0] == g[:, 0].min()], 1: g[g[:, 0] == g[:, 0].max()], 2: g[g[:, 1] == g[:, 1].min()], 3: g[g[:, 1] == g[:, 1].max()]}
wiazki = []
for n, z in enumerate(ZR):
    pts = kraw[n][:: max(1, len(kraw[n]) // 10)]
    lc = Line3DCollection([[z, p] for p in pts], colors=[ZIELONY], linewidths=1.6, alpha=0.10); ax.add_collection3d(lc, autolim=False); wiazki.append(lc)
    ax.scatter(*np.array(z)[:, None], s=420, c=[ZIELONY], alpha=0.95, depthshade=False, edgecolors='#ffffff', linewidths=0.6)
    ax.scatter(*np.array(z)[:, None], s=2400, c=[ZIELONY], alpha=0.12, depthshade=False, edgecolors='none')
nit = Line3DCollection([], colors='#ffd27a', linewidths=0.45, alpha=0.25); ax.add_collection3d(nit, autolim=False)
stud_h = ax.scatter([], [], [], s=150, c=[ZIELONY], alpha=0.10, depthshade=False, edgecolors='none')
stud = ax.scatter([], [], [], s=10, c=[ZIELONY], alpha=0.7, depthshade=False, edgecolors='none')
kul = ax.scatter([], [], [], s=22, c=[], cmap='plasma', vmin=0, vmax=vmax, depthshade=False, edgecolors='#ffffff', linewidths=0.3)
nar = ax.scatter([], [], [], s=70, facecolors='none', edgecolors='#4af0ff', linewidths=1.2, depthshade=False)
tytul = fig.text(0.5, 0.955, '', ha='center', fontsize=20)
fig.text(0.5, 0.92, 'Film 3: uczenie w równym tempie (≈ 940 kroków na sekundę) — kamera okrąża siatkę światła', ha='center', fontsize=13, color='#ff9a5c')
fig.text(0.012, 0.02, 'Zielone punkty = studnie światła (centra tokenów, średnia ±300 kroków); kule = 600 tokenów, kolor = odległość od studni (fiolet blisko, żółty daleko); '
         'niebieskie obwódki = narzędzie; szara siatka na dole = adresy w kroku 0; lasery = kodowanie pozycji.', fontsize=10.5, color='#cfd3d8')
ab = fig.add_axes((0.79, 0.07, 0.19, 0.17)); ab.set_facecolor('#0b1014')
ab.plot(kroki / EP, iou, color='#4af0ff', lw=1.0, label='IoU')
ab.plot(kroki / EP, ruch.mean(1) / ruch.mean(1).max(), color='#ffb347', lw=1.0, label='drgania (wzgl.)')
kur = ab.axvline(0, color='#ffffff', lw=1.2); ab.set_xlim(0, kroki[-1] / EP); ab.set_ylim(0, 1.05)
ab.tick_params(colors='#cfd3d8', labelsize=8); ab.grid(True, color='#22292f'); ab.set_xlabel('epoka', color='#cfd3d8', fontsize=9)
ab.legend(fontsize=8, loc='lower right', facecolor='#161e24', edgecolor='#2a333a', labelcolor='#e8e8e8')

ff = None if PROBA else ffmpeg(HERE / 'lasery_3_uczenie.mp4', FPS)
N = len(krok_f)
for f, kr in enumerate(krok_f):
    a, q = sA(kr), sQ(kr)
    u = f / max(1, N - 1)
    ax.view_init(elev=28 + 14 * np.sin(2 * np.pi * u * 1.5), azim=-70 + 360 * u)
    nit.set_segments(a[sas])
    stud._offsets3d = tuple(q.T); stud_h._offsets3d = tuple(q.T)
    kul._offsets3d = tuple(a.T); kul.set_array(np.linalg.norm(a - q, axis=1))
    nar._offsets3d = tuple(a[narz].T)
    mig = 0.10 + 0.03 * np.sin(f * 0.8)
    for lc in wiazki:
        lc.set_alpha(mig)
    kur.set_xdata([kr / EP, kr / EP])
    tytul.set_text(f'epoka {kr / EP:5.2f}   krok {kr:,.0f}   IoU {iou_f[f]:.2f}'.replace(',', ' '))
    fig.canvas.draw()
    if PROBA:
        fig.savefig(HERE / f'lasery_3_proba_{f}.png', dpi=60)
    else:
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if f % 200 == 0:
            log(f'klatka {f}/{N}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
