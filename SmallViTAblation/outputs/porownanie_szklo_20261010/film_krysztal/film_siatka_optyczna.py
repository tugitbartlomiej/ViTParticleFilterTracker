"""Film „koloidy w siatce optycznej”: obraz fizyczny z testów przyczynowych (2026-10-11). Cztery wiązki laserowe (zielone) z czterech
stron interferują i tworzą siatkę studni światła w węzłach = adresy patchy z kodowania pozycji (30 × 20). 600 koloidów = prawdziwe tokeny
obrazu TRAIN 0 z 806 checkpointów (szescian.npz: atomy i węzły z Edena, płaszczyzna x, y), drgają w studniach. Studnie idą za węzłami sieci
(średnia ±300 kroków): siatkę światła powoli przesuwa i odkształca uczenie. Układ współporuszający się (Prokrustes 2D węzłów do siatki
idealnej), więc widać lokalne odkształcenie. Nitki między sąsiadami = wspólne piksele lodygi (sprzężenie z architektury); błyski
z jednego koloidu do wielu = słabe globalne sprzężenie przez uwagę (z testu odpowiedzi: ~2% ruchu, rośnie z uczeniem).
Kolor koloidu = ruchliwość (odległość od studni), niebieska obwódka = narzędzie. Spowolnienie: kroki 0–300 ×8, 300–1100 ×4.
Uruchomienie: F:/Python/3.13/python.exe film_siatka_optyczna.py [zdjęcie=0] [proba] → siatka_optyczna_<zdjęcie>.mp4"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).parent
J = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PROBA = len(sys.argv) > 2 and sys.argv[2] == 'proba'
EP, H, W, TOK = 1094, 20, 30, 600
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
d = np.load(HERE / 'szescian.npz')
kroki = d['kroki']; K = len(kroki); g = d['g'][:, :2]
A, Q = d[f'atomy_{J}'][..., :2].astype(np.float64), d[f'wezly_{J}'][..., :2].astype(np.float64)
iou = d[f'iou_{J}']; narz = d[f'narzedzie_{J}']; split = str(d[f'split_{J}'])

# układ współporuszający się: przesunięcie, obrót, skala węzłów → siatka idealna (Prokrustes 2D)
for t in range(K):
    mq, mg = Q[t].mean(0), g.mean(0); qc, gc = Q[t] - mq, g - mg
    U_, S_, Vt = np.linalg.svd(qc.T @ gc); dd = np.sign(np.linalg.det(U_ @ Vt)); Dg = np.diag([1, dd]); R = U_ @ Dg @ Vt
    sk = float((S_ * np.diag(Dg)).sum() / (qc ** 2).sum())
    Q[t] = (Q[t] - mq) @ R * sk + mg; A[t] = (A[t] - mq) @ R * sk + mg
gest = np.where(kroki < 300, 8, np.where(kroki < 1100, 4, 1))
tt = np.concatenate([np.linspace(t, t + 1, n, endpoint=False) for t, n in zip(range(K - 1), gest[:-1])] + [[K - 1.0]] * 90)
if PROBA:
    tt = tt[[0, 120, 400, 900, len(tt) - 1]]
sA, sQ = CubicSpline(np.arange(K), A, axis=0), CubicSpline(np.arange(K), Q, axis=0)
krok_t = np.interp(tt, np.arange(K), kroki); iou_t = np.interp(tt, np.arange(K), iou)
amp_ref = float(np.percentile(np.linalg.norm(A - Q, axis=-1), 95))
log(f'zdjęcie {J} ({split}), klatek {len(tt)} ({len(tt) / 30:.0f} s), amplituda 95. pc. {amp_ref:.2f} komórki')

# ---------------------------------------------------------------- światło: wiązki + studnie w węzłach (obraz RGB liczony ręcznie)
XL, YL, PX = W / 2 + 2.2, H / 2 + 2.2, 22                                  # zasięg w komórkach, pikseli na komórkę
NX, NY = int(2 * XL * PX), int(2 * YL * PX)
xs = (np.arange(NX) + 0.5) / PX - XL; ys = (np.arange(NY) + 0.5) / PX - YL
SIG = 0.26; R_K = int(3 * SIG * PX) + 1
ky, kx = np.mgrid[-R_K:R_K + 1, -R_K:R_K + 1] / PX
ZIELONY = np.array([0.22, 1.0, 0.48]); BIALOZIEL = np.array([0.75, 1.0, 0.85])
# fala stojąca wiązek: cos² w kierunku propagacji, pasmo o gaussowskim profilu poprzecznym (szerokość sieci)
fx = np.cos(np.pi * (xs[None, :] - 0.5)) ** 2; fy = np.cos(np.pi * (ys[:, None] - 0.5)) ** 2
pas_x = np.exp(-(ys[:, None] / (H / 2 + 0.8)) ** 8); pas_y = np.exp(-(xs[None, :] / (W / 2 + 0.8)) ** 8)
WIAZKI = 0.10 * (fx * pas_x + fy * pas_y)                                  # słaba poświata wiązek i ich interferencji


def swiatlo(q, migotanie):
    I = np.zeros((NY, NX))
    for x_, y_ in q:
        cx, cy = int((x_ + XL) * PX), int((y_ + YL) * PX)
        x0, x1, y0, y1 = cx - R_K, cx + R_K + 1, cy - R_K, cy + R_K + 1
        if x0 < 0 or y0 < 0 or x1 > NX or y1 > NY:
            continue
        ox, oy = (x_ + XL) * PX - cx - 0.5, (y_ + YL) * PX - cy - 0.5
        I[y0:y1, x0:x1] += np.exp(-((kx - ox / PX) ** 2 + (ky - oy / PX) ** 2) / (2 * SIG ** 2))
    I = np.clip(I, 0, 1.4)
    L = (WIAZKI * migotanie)[..., None] * ZIELONY + (0.55 * I)[..., None] * ZIELONY + (0.25 * I ** 3)[..., None] * BIALOZIEL
    return np.clip(L, 0, 1)


import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'figure.facecolor': '#03060a', 'text.color': '#e8e8e8'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
cm_ruch = plt.cm.plasma
sas = [(r * W + c, r * W + c + 1) for r in range(H) for c in range(W - 1)] + [(r * W + c, (r + 1) * W + c) for r in range(H - 1) for c in range(W)]
sas = np.array(sas)


def panel(rect, xl, yl):
    ax = fig.add_axes(rect); ax.set_facecolor('#03060a'); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color('#1d2a24')
    im = ax.imshow(np.zeros((NY, NX, 3)), extent=(-XL, XL, YL, -YL), interpolation='bilinear', zorder=0)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal')
    nit = LineCollection([], colors='#ffd27a', linewidths=0.5, alpha=0.22, zorder=2); ax.add_collection(nit)
    glob = LineCollection([], colors='#b48cff', linewidths=0.6, alpha=0.0, zorder=2); ax.add_collection(glob)
    slad = LineCollection([], linewidths=1.0, zorder=3); ax.add_collection(slad)
    halo = ax.scatter([], [], s=150, c=[], cmap=cm_ruch, vmin=0, vmax=1, alpha=0.18, edgecolors='none', zorder=4)
    kul = ax.scatter([], [], s=26, c=[], cmap=cm_ruch, vmin=0, vmax=1, edgecolors='#ffffff', linewidths=0.35, zorder=5)
    nar = ax.scatter([], [], s=80, facecolors='none', edgecolors='#4af0ff', linewidths=1.3, zorder=6)
    return ax, im, nit, glob, slad, halo, kul, nar


# ---- główny panel z laserami na brzegach
xl_m = (-XL - 2.6, XL + 2.6); yl_m = (YL + 2.6, -YL - 2.6)
ax_m, im_m, nit_m, glob_m, slad_m, halo_m, kul_m, nar_m = panel((0.005, 0.12, 0.63, 0.765), xl_m, yl_m)
lasery = []
for (x_, y_, w_, h_, kier) in ((-XL - 2.4, -1.2, 1.6, 2.4, 'p'), (XL + 0.8, -1.2, 1.6, 2.4, 'l'), (-1.6, -YL - 2.4, 3.2, 1.4, 'd'), (-1.6, YL + 1.0, 3.2, 1.4, 'g')):
    ax_m.add_patch(FancyBboxPatch((x_, y_), w_, h_, boxstyle='round,pad=0.05,rounding_size=0.25', fc='#1b2420', ec='#5b6b63', lw=1.2, zorder=7))
    ap = {'p': (x_ + w_, y_ + h_ / 2), 'l': (x_, y_ + h_ / 2), 'd': (x_ + w_ / 2, y_ + h_), 'g': (x_ + w_ / 2, y_)}[kier]
    lasery.append(ax_m.scatter([ap[0]], [ap[1]], s=260, c=[ZIELONY], alpha=0.9, edgecolors='none', zorder=8))
    ax_m.scatter([ap[0]], [ap[1]], s=900, c=[ZIELONY], alpha=0.12, edgecolors='none', zorder=8)
# stożki wiązek od aperatur do brzegów sieci (kilka warstw = poświata)
from matplotlib.patches import Polygon  # noqa: E402
stozki = []
for ap, (p1, p2) in (((-XL - 0.8, 0.0), ((-W / 2 - 0.5, -H / 2 - 0.5), (-W / 2 - 0.5, H / 2 + 0.5))), ((XL + 0.8, 0.0), ((W / 2 + 0.5, -H / 2 - 0.5), (W / 2 + 0.5, H / 2 + 0.5))),
                     ((0.0, -YL - 1.0), ((-W / 2 - 0.5, -H / 2 - 0.5), (W / 2 + 0.5, -H / 2 - 0.5))), ((0.0, YL + 1.0), ((-W / 2 - 0.5, H / 2 + 0.5), (W / 2 + 0.5, H / 2 + 0.5)))):
    for sz, al in ((1.0, 0.05), (0.6, 0.06), (0.25, 0.08)):
        q1 = np.array(ap) + sz * (np.array(p1) - np.array(ap)); q2 = np.array(ap) + sz * (np.array(p2) - np.array(ap))
        pg = Polygon([ap, tuple(q1 if sz < 1 else p1), tuple(q2 if sz < 1 else p2)], closed=True, fc=ZIELONY, ec='none', alpha=al, zorder=1)
        ax_m.add_patch(pg); stozki.append((pg, al))
ax_m.set_title('Koloidy w siatce optycznej: 600 tokenów ViT w studniach światła (układ współporuszający się z siecią)', fontsize=14, color='#e8e8e8')

# ---- zbliżenie wokół narzędzia
gz = g[narz].mean(0) if narz.any() else np.zeros(2)
cx, cy = float(np.clip(gz[0], -W / 2 + 4, W / 2 - 4)), float(np.clip(gz[1], -H / 2 + 3, H / 2 - 3))
xl_z, yl_z = (cx - 4.2, cx + 4.2), (cy + 3.0, cy - 3.0)
ax_z, im_z, nit_z, glob_z, slad_z, halo_z, kul_z, nar_z = panel((0.645, 0.43, 0.345, 0.455), xl_z, yl_z)
ax_m.add_patch(Rectangle((xl_z[0], yl_z[1]), xl_z[1] - xl_z[0], yl_z[0] - yl_z[1], fill=False, ec='#ffffff', lw=1.0, ls='--', zorder=9))
ax_z.set_title('Zbliżenie (ramka): drgania w studniach i ślady', fontsize=12.5, color='#e8e8e8')

# ---- obraz oka i oś czasu
try:
    oko = plt.imread(HERE / 'zdjecie_train0.jpg')
    ao = fig.add_axes((0.645, 0.135, 0.15, 0.25)); ao.imshow(oko); ao.set_xticks([]); ao.set_yticks([])
    hh, ww = oko.shape[:2]
    for r in range(H + 1):
        ao.axhline(r * hh / H, color='#39ff7a', lw=0.3, alpha=0.5)
    for c in range(W + 1):
        ao.axvline(c * ww / W, color='#39ff7a', lw=0.3, alpha=0.5)
    ao.set_title('obraz → 600 patchy = 600 studni', fontsize=10.5, color='#cfd3d8')
except Exception as e:  # noqa: BLE001
    log('bez zdjęcia', e)
ab = fig.add_axes((0.82, 0.16, 0.17, 0.21)); ab.set_facecolor('#0b1014')
ab.plot(np.arange(K), iou, color='#4af0ff', lw=1.0, label='IoU tego zdjęcia')
amp_k = np.linalg.norm(A - Q, axis=-1).mean(1); ab.plot(np.arange(K), amp_k / amp_k.max(), color='#ffb347', lw=1.0, label='drgania (wzgl.)')
kur = ab.axvline(0, color='#ffffff', lw=1.2)
tk = [0, 300, 1000, EP * 5, EP * 20, EP * 60]
ab.set_xticks([int(np.argmin(abs(kroki - k))) for k in tk]); ab.set_xticklabels(['0', '300', '1k', 'e5', 'e20', 'e60'])
ab.tick_params(colors='#cfd3d8', labelsize=8); ab.grid(True, color='#22292f'); ab.set_ylim(0, 1.05)
ab.legend(fontsize=8, loc='lower right', facecolor='#161e24', edgecolor='#2a333a', labelcolor='#e8e8e8')
tytul = fig.text(0.5, 0.968, '', ha='center', fontsize=17)
faza = fig.text(0.5, 0.937, '', ha='center', fontsize=12.5, color='#ff9a5c')
fig.text(0.008, 0.02, 'Zielone studnie = interferencja 4 wiązek, czyli adresy patchy z kodowania pozycji (bez niego porządek znika). '
         'Studnie przesuwa i odkształca uczenie. Kule = tokeny, kolor = odległość od studni (fiolet blisko, żółty daleko), niebieska obwódka = narzędzie.\n'
         'Żółte nitki między sąsiadami = wspólne piksele lodygi (sprzężenie z architektury, jest od inicjalizacji). '
         'Fioletowe błyski z jednej kuli do wielu = słabe globalne sprzężenie przez uwagę (~2% ruchu, rośnie z uczeniem).',
         fontsize=10.5, color='#cfd3d8')
FAZY = [(0, 'narodziny: koloidy na idealnej siatce patchy; studnie (średnie położenie ±300 kroków) już się przesuwają'), (30, 'pierwsze kroki: koloidy wybijają się ze studni, światło jeszcze stoi'),
        (300, 'formowanie: siatka światła zaczyna płynąć za uczeniem'), (1100, 'epoki 1–5: odkształcenie rośnie, drgania słabną'),
        (EP * 5, 'dojrzałość: studnie wolno pełzną, koloidy drgają w nich')]

ffmpeg = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', '1920x1080', '-r', '30', '-i', '-',
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-maxrate', '12M', '-bufsize', '24M', '-preset', 'medium',
                           str(HERE / f'siatka_optyczna_{J}.mp4')], stdin=subprocess.PIPE)
rng = np.random.default_rng(0)
hist = []; blysk = None; blysk_t = 0
for f, t in enumerate(tt):
    a, q = sA(t), sQ(t)
    mig = 1.0 + 0.08 * np.sin(f * 0.9) + 0.04 * rng.standard_normal()
    L = swiatlo(q, mig)
    im_m.set_data(L); im_z.set_data(L)
    for pg, al in stozki:
        pg.set_alpha(al * mig)
    for lz in lasery:
        lz.set_alpha(float(np.clip(0.75 + 0.25 * np.sin(f * 0.9 + rng.random()), 0.4, 1)))
    ruch = np.clip(np.linalg.norm(a - q, axis=1) / amp_ref, 0, 1)
    seg = a[sas]
    hist.append(a.copy()); hist = hist[-14:]
    if blysk_t <= 0 and rng.random() < 0.08:                                # nowy błysk sprzężenia globalnego
        blysk = (int(rng.integers(TOK)), rng.choice(TOK, 60, replace=False)); blysk_t = 18
    for (nit, glob, slad, halo, kul, nar, zoom) in ((nit_m, glob_m, slad_m, halo_m, kul_m, nar_m, False), (nit_z, glob_z, slad_z, halo_z, kul_z, nar_z, True)):
        nit.set_segments(seg); nit.set_alpha(0.10 if zoom else 0.22)
        halo.set_offsets(a); halo.set_array(ruch); kul.set_offsets(a); kul.set_array(ruch)
        kul.set_sizes(np.full(TOK, 95 if zoom else 22)); halo.set_sizes(np.full(TOK, 520 if zoom else 130))
        nar.set_offsets(a[narz]); nar.set_sizes(np.full(int(narz.sum()), 260 if zoom else 75))
        if blysk is not None and blysk_t > 0:
            i0, cel = blysk; glob.set_segments([[a[i0], a[j]] for j in cel]); glob.set_alpha(0.5 * blysk_t / 18)
        else:
            glob.set_segments([])
        if zoom and len(hist) > 1:
            H_ = np.array(hist); s_ = [H_[k:k + 2, i] for i in range(TOK) for k in range(len(H_) - 1)]
            slad.set_segments(s_); slad.set_color(np.tile(plt.cm.plasma(ruch)[:, None], (1, len(H_) - 1, 1)).reshape(-1, 4) * [1, 1, 1, 0.6])
    blysk_t -= 1
    kur.set_xdata([t, t])
    kr = krok_t[f]
    tytul.set_text(f'krok {kr:,.0f}   epoka {kr / EP:5.2f}   IoU {iou_t[f]:.2f}'.replace(',', ' '))
    faza.set_text([s for k, s in FAZY if kr >= k][-1])
    fig.canvas.draw()
    ffmpeg.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    if PROBA or f in (60, 400, 1000):
        fig.savefig(HERE / f'siatka_optyczna_{J}_{f}.png', dpi=60)
    if f % 150 == 0:
        log(f'klatka {f}/{len(tt)}')
ffmpeg.stdin.close(); ffmpeg.wait()
log('zapisano', HERE / f'siatka_optyczna_{J}.mp4')
