"""Film „kryształ, atomy i siły”: jedno zdjęcie uczące, 806 checkpointów (60 epok), rzut 128D.
  węzeł c_i(t) = średnia położenia tokenu i w oknie ±300 kroków (sieć krystaliczna, odkształca się powoli),
  atom x_i(t) = położenie tokenu (drga wokół węzła), wychylenie u_i = x_i − c_i,
  dwa procesy: (1) DRGANIA — atom wychylony od węzła, siła powrotna −u (sprężynka atom–węzeł),
              (2) PŁYNIĘCIE SIECI — prędkość węzła v_i = dc_i/dt (strzałki), sieć odkształca się i daje atomom nową drogę.
Pomiar procesu („pełzanie”): czy wychylenie atomu teraz wskazuje, dokąd przesunie się jego węzeł? cos(u_i(t), c_i(t+Δ) − c_i(t)) i udział
energii dryfu sieci w ruchu atomów, w etapach → SILY.json.
Obraz: wspólny 2D układ z PCA węzłów, wyrównywany Prokrustesem; płynna interpolacja (splajn kubiczny) między checkpointami,
spowolnienie: kroki 0–300 (narodziny kryształu) 8 klatek na checkpoint, 300–1100 4, potem 1.
Uruchomienie: F:/Python/3.13/python.exe film_sily.py → krysztal_sily.mp4"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).parent
EKS = HERE.parents[2] / 'assets' / 'RuchCzasteczek_ViT_szklo_2026-10' / '03_pomiar_naturalny_60_epok_eden' / 'ruch60_rp128_50tr_50val'
POM = HERE.parent / 'eden' / 'wyniki' / 'pomiar60' / 'manifest.json'
J, H, W, TOK, EP, PL, OKNO = 0, 20, 30, 600, 1094, 16, 300
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
META = json.load(open(EKS / 'META.json', encoding='utf-8'))
kroki = np.array(META['kroki']); K = len(kroki)
X = np.asarray(np.load(EKS / 'rp128.npy', mmap_mode='r')[:, J]).astype(np.float32); X -= X.mean(1, keepdims=True)
iou = np.load(EKS / 'oceny.npy')[:, J, 1]
b = json.load(open(POM, encoding='utf-8'))['images'][J]['bbox_xyxy_480']
siatka = np.stack([np.arange(TOK) // W, np.arange(TOK) % W], 1)
cx = (siatka[:, 1] + 0.5) * PL; cy = (siatka[:, 0] + 0.5) * PL
narz = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
wiaz = np.array([(i, i + 1) for i in range(TOK) if i % W < W - 1] + [(i, i + W) for i in range(TOK - W)])

# ---------------------------------------------------------------- węzły, wychylenia, prędkości sieci (128D)
C = np.stack([X[[u for u in range(K) if abs(kroki[u] - kroki[t]) <= OKNO]].mean(0) for t in range(K)])
U = X - C
dt = np.gradient(kroki.astype(float))
V = np.gradient(C, axis=0) / dt[:, None, None] * 100                            # prędkość węzła na 100 kroków
# pomiar procesu: czy wychylenie zapowiada ruch węzła (pełzanie / aktywacja), w etapach
etapy = {'0-300': (0, 300), '300-1100': (300, 1100), 'ep 1-5': (EP, 5 * EP), 'ep 5-20': (5 * EP, 20 * EP), 'ep 20-60': (20 * EP, 60 * EP + 1)}
sily = {}
for n, (a_, b_) in etapy.items():
    idx = [t for t in range(K) if a_ <= kroki[t] < b_]
    w = {}
    for lag in (300, 1000):
        cs, cs_los = [], []
        for t in idx:
            t2 = int(np.searchsorted(kroki, kroki[t] + lag))
            if t2 >= K:
                continue
            dc = C[t2] - C[t]
            cs.append(((U[t] * dc).sum(1) / (np.linalg.norm(U[t], axis=1) * np.linalg.norm(dc, axis=1) + 1e-9)).mean())
            p = np.random.default_rng(t).permutation(TOK)
            cs_los.append(((U[t] * dc[p]).sum(1) / (np.linalg.norm(U[t], axis=1) * np.linalg.norm(dc[p], axis=1) + 1e-9)).mean())
        w[lag] = dict(cos_wychylenie_ruch_wezla=float(np.mean(cs)) if cs else None, kontrola_losowa=float(np.mean(cs_los)) if cs else None)
    amp = float(np.linalg.norm(U[idx], axis=-1).mean()); vel = float(np.linalg.norm(V[idx], axis=-1).mean())
    sily[n] = dict(amplituda_drgan=amp, predkosc_sieci_na_100_krokow=vel, stosunek=vel / amp, narzedzie_amplituda=float(np.linalg.norm(U[idx][:, narz], axis=-1).mean()),
                   narzedzie_predkosc=float(np.linalg.norm(V[idx][:, narz], axis=-1).mean()), wyprzedzenie=w)
    log(f'{n}: drgania {amp:.2f}, płynięcie sieci {vel:.2f}/100 kroków (narzędzie {sily[n]["narzedzie_amplituda"]:.2f} / {sily[n]["narzedzie_predkosc"]:.2f}); '
        f'cos(wychylenie, przyszły ruch węzła) Δ300 {w[300]["cos_wychylenie_ruch_wezla"]}, Δ1000 {w[1000]["cos_wychylenie_ruch_wezla"]} (los {w[300]["kontrola_losowa"]})')
json.dump(sily, open(HERE / 'SILY.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- wspólny układ 2D: PCA węzłów + Prokrustes, skala globalna
R2 = np.empty((K, 128, 2)); prev = None; sk = np.empty(K)
for t in range(K):
    _, s, Vt = np.linalg.svd(C[t] - C[t].mean(0), full_matrices=False)
    B = Vt[:2].T
    y = (C[t] - C[t].mean(0)) @ B
    if prev is not None:
        Uq, _, Vq = np.linalg.svd(y.T @ prev); Q = Uq @ Vq; B = B @ Q; y = y @ Q
    R2[t] = B; prev = y; sk[t] = np.sqrt((y ** 2).sum(1).mean())
proj = lambda t, Z: ((Z - C[t].mean(0)) @ R2[t]) / sk[t]
Cn = np.stack([proj(t, C[t]) for t in range(K)]); Xn = np.stack([proj(t, X[t]) for t in range(K)])
Vn = np.stack([(V[t] @ R2[t]) / sk[t] for t in range(K)])
vmax = float(np.percentile(np.linalg.norm(V, axis=-1), 99))
# czas filmu: spowolnienie w fazach krytycznych
gest = np.where(kroki < 300, 8, np.where(kroki < 1100, 4, 1))
tt = np.concatenate([np.linspace(t, t + 1, g, endpoint=False) for t, g in zip(range(K - 1), gest[:-1])] + [[K - 1.0]] * 75)
sC, sX, sV = CubicSpline(np.arange(K), Cn, axis=0), CubicSpline(np.arange(K), Xn, axis=0), CubicSpline(np.arange(K), Vn, axis=0)
krok_t = np.interp(tt, np.arange(K), kroki); iou_t = np.interp(tt, np.arange(K), iou)
spd = np.interp(tt, np.arange(K), np.linalg.norm(V, axis=-1).mean(1)); ampl = np.interp(tt, np.arange(K), np.linalg.norm(U, axis=-1).mean(1))
log(f'klatek filmu {len(tt)} ({len(tt) / 30:.0f} s)')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'axes.facecolor': '#0b1014', 'figure.facecolor': '#0b1014', 'axes.edgecolor': '#3a444c',
                     'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#22292f',
                     'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1], left=0.03, right=0.98, top=0.88, bottom=0.07, wspace=0.08, hspace=0.3)
a = fig.add_subplot(gs[:, 0]); az = fig.add_subplot(gs[0, 1]); at = fig.add_subplot(gs[1, 1])
lim = float(np.percentile(np.abs(Xn), 99.7)) * 1.08
for ax_ in (a, az):
    ax_.set_aspect('equal'); ax_.set_xticks([]); ax_.set_yticks([])
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
siec = LineCollection([], colors='#5b8fb0', linewidths=0.7, alpha=0.75); a.add_collection(siec)
spr = LineCollection([], colors='#ffb347', linewidths=0.6, alpha=0.7); a.add_collection(spr)
wez = a.scatter([], [], s=9, c='#5b8fb0', zorder=3)
atomy = a.scatter([], [], s=16, c='#ffffff', edgecolors='none', zorder=4)
at_n = a.scatter([], [], s=55, facecolors='none', edgecolors='#4af0ff', lw=1.3, zorder=5)
co = 2
kw = a.quiver(np.zeros(TOK // co), np.zeros(TOK // co), np.zeros(TOK // co), np.zeros(TOK // co), np.zeros(TOK // co), cmap='plasma', clim=(0, vmax),
              angles='xy', scale_units='xy', scale=1.0, width=0.0022, headwidth=3.5, alpha=0.9, zorder=6)
a.set_title('SIEĆ (niebieska) = węzły; ATOMY (białe) drgają na sprężynkach (pomarańczowe) wokół węzłów; STRZAŁKI = płynięcie sieci (kolor = prędkość)', fontsize=12)
# zbliżenie: okolica narzędzia
zs = az.add_collection(LineCollection([], colors='#5b8fb0', linewidths=1.0, alpha=0.8))
zp = az.add_collection(LineCollection([], colors='#ffb347', linewidths=1.2, alpha=0.85))
zw = az.scatter([], [], s=22, c='#5b8fb0', zorder=3); za = az.scatter([], [], s=40, c='#ffffff', zorder=4)
zn = az.scatter([], [], s=110, facecolors='none', edgecolors='#4af0ff', lw=1.6, zorder=5)
az.set_title('zbliżenie: narzędzie i jego sąsiedzi', fontsize=12)
xs = krok_t
at.plot(np.arange(K), np.linalg.norm(U, axis=-1).mean(1), color='#ffffff', lw=1.0, label='drgania atomów (amplituda)')
at.plot(np.arange(K), np.linalg.norm(V, axis=-1).mean(1), color='#ff7a45', lw=1.0, label='płynięcie sieci (na 100 kroków)')
at.plot(np.arange(K), iou * np.linalg.norm(U, axis=-1).mean(1).max(), color='#4af0ff', lw=0.8, alpha=0.7, label='IoU (skala)')
kur = at.axvline(0, color='#ffffff', lw=1.5)
tick_k = [0, 100, 300, 1000, EP * 5, EP * 20, EP * 60]
at.set_xticks([int(np.argmin(abs(kroki - k))) for k in tick_k]); at.set_xticklabels(['0', '100', '300', '1000', 'ep 5', 'ep 20', 'ep 60'])
at.set_xlim(0, K - 1); at.grid(True, alpha=0.4); at.legend(fontsize=9, loc='upper right'); at.set_title('dwa procesy w czasie', fontsize=12)
s5 = sily['ep 20-60']['wyprzedzenie'][300]
fig.text(0.03, 0.93, 'Proces 1: szum kroku AdamW wychyla atomy z węzłów (drgania).  Proces 2: uczenie powoli przesuwa węzły — sieć się odkształca i atomy dostają nową drogę.',
         fontsize=12, color='#cfd3d8')
fig.text(0.03, 0.905, f'Czy wychylenie atomu zapowiada, dokąd pójdzie węzeł? cos(wychylenie, ruch węzła za 300 kroków), epoki 20–60: {s5["cos_wychylenie_ruch_wezla"]:+.2f} '
         f'(losowo {s5["kontrola_losowa"]:+.2f})', fontsize=12, color='#ffd166')
tytul = fig.suptitle('', fontsize=17, y=0.985)
faza = fig.text(0.98, 0.93, '', ha='right', fontsize=13, color='#ff7a45')
out = HERE / 'krysztal_sily.mp4'
fig.canvas.draw(); wid, hei = fig.canvas.get_width_height()
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{wid}x{hei}', '-r', '30', '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
sel = np.arange(0, TOK, co)
for n, tau in enumerate(tt):
    c, x, v = sC(tau), sX(tau), sV(tau)
    siec.set_segments(c[wiaz]); spr.set_segments(np.stack([c, x], 1)); wez.set_offsets(c); atomy.set_offsets(x); at_n.set_offsets(x[narz])
    kw.set_offsets(c[sel]); kw.set_UVC(v[sel, 0] * 3, v[sel, 1] * 3, np.linalg.norm(v[sel], axis=1) * sk[int(tau)])
    m = np.where(narz)[0]; cen = c[m].mean(0); r = max(float(np.abs(c[m] - cen).max()) * 2.2, 0.25)
    blisko = np.where(np.abs(c - cen).max(1) < r)[0]; wb = wiaz[np.isin(wiaz[:, 0], blisko) & np.isin(wiaz[:, 1], blisko)]
    zs.set_segments(c[wb]); zp.set_segments(np.stack([c[blisko], x[blisko]], 1)); zw.set_offsets(c[blisko]); za.set_offsets(x[blisko]); zn.set_offsets(x[narz])
    az.set_xlim(cen[0] - r, cen[0] + r); az.set_ylim(cen[1] - r, cen[1] + r)
    kur.set_xdata([tau, tau])
    k_ = krok_t[n]
    tytul.set_text(f'Kryształ tokenów: atomy, sieć i siły — krok {k_:.0f} (epoka {k_ / EP:.2f}), IoU {iou_t[n]:.2f}')
    faza.set_text('» narodziny kryształu (8× zwolnione)' if k_ < 300 else '» formowanie (4× zwolnione)' if k_ < 1100 else '')
    fig.canvas.draw(); ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    if n % 200 == 0:
        log(f'klatka {n}/{len(tt)}')
ff.stdin.close(); ff.wait()
fig.savefig(HERE / 'krysztal_sily_ostatnia_klatka.png')
log('zapisano', out, f'{out.stat().st_size / 1e6:.1f} MB')
