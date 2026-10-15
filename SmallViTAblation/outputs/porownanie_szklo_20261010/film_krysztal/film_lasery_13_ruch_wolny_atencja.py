"""Film 13: kopia filmu 2 („lasery_2_ruch”: ruch 600 tokenów TRAIN 0 przez 60 epok rozłożony na trzy składowe), ale WOLNIEJSZA i płynniejsza,
a zamiast panelu objaśnień cztery okna: (4) mapa atencji na zdjęciu oka z narzędziem, (5) gdzie centra odpłynęły najdalej na tym samym
zdjęciu, (6) wielkość trzech składowych ruchu, (7) IoU i jego zmiana z checkpointu na checkpoint.
Czas: interpolacja w KROKACH uczenia (nie w numerach checkpointów), prędkość rośnie płynnie: v = min(0,5 + krok/1100, 20) kroków na klatkę,
więc nie ma skoków przy zmianie gęstości checkpointów (co 10 → co 100 → końce epok). Położenia: PCHIP po krokach (bez przestrzeleń splajnu).
Atencja i ramka: atencja_train0.npz (atencja_train0_prep.py, blok 6 jak w aplikacji). Film 2 i jego skrypt zostają nietknięte.
Uruchomienie: F:/Python/3.13/python.exe -X utf8 film_lasery_13_ruch_wolny_atencja.py [proba | czesc i n]"""
import subprocess
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d

from lasery_wspolne import EP, H, HERE, TOK, W, Swiatlo, dane, ffmpeg, oko, prokrustes2d, zastosuj

ARG = sys.argv[1:]
PROBA = ARG[:1] == ['proba']
CZESC = (int(ARG[1]), int(ARG[2])) if ARG[:1] == ['czesc'] else None
NPROC = 6
WYJ = HERE / 'lasery_13_ruch_wolny_atencja.mp4'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

# ---- tryb główny: równoległe części + sklejenie
if not PROBA and CZESC is None:
    pr = [subprocess.Popen([sys.executable, '-X', 'utf8', __file__, 'czesc', str(i), str(NPROC)]) for i in range(NPROC)]
    kody = [p.wait() for p in pr]
    if any(kody):
        sys.exit(f'błąd części: {kody}')
    lista = HERE / 'lasery_13_czesci.txt'
    lista.write_text(''.join(f"file 'lasery_13_czesc_{i}.mp4'\n" for i in range(NPROC)), encoding='utf-8')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(lista), '-c', 'copy', str(WYJ)], check=True)
    for i in range(NPROC):
        (HERE / f'lasery_13_czesc_{i}.mp4').unlink()
    lista.unlink()
    log('gotowe', WYJ)
    sys.exit()

D = dane(0)
kroki, narz = D['kroki'].astype(float), D['narz']; K = len(kroki)
g = D['g3'][:, :2]; Ql, Al = D['Q3'][..., :2], D['A3'][..., :2]
par, mapy = prokrustes2d(Ql, g)
Qc = np.stack([zastosuj(Ql[t], mapy[t]) for t in range(K)]); Ac = np.stack([zastosuj(Al[t], mapy[t]) for t in range(K)])
Gl = np.stack([(g - mapy[t][3]) @ mapy[t][1].T / mapy[t][2] + mapy[t][0] for t in range(K)])
glob = np.linalg.norm(Gl - g, axis=-1).mean(1); lok = np.linalg.norm(Qc - g, axis=-1); vib = np.linalg.norm(Ac - Qc, axis=-1)
lok_m, vib_m = lok.mean(1), vib.mean(1)
narz_lok = lok[:, narz].mean(1) / np.maximum(lok[:, ~narz].mean(1), 1e-9)
AT = np.load(HERE / 'atencja_train0.npz')
atn, ramka, iou, GT = AT['atencja'], AT['ramka'], AT['iou'].astype(float), AT['gt']
d_iou = np.diff(iou, prepend=iou[0])

# ---- oś czasu: krok uczenia w każdej klatce, płynnie przyspieszający
ss = [0.0] * 45
s = 0.0
while s < kroki[-1]:
    s = min(s + min(0.5 + s / 1100, 20.0), kroki[-1]); ss.append(s)
ss = np.array(ss + [kroki[-1]] * 120)
fr = np.arange(len(ss))
if PROBA:
    fr = fr[[0, 400, 1200, 2500, len(ss) - 1]]
elif CZESC:
    fr = np.array_split(fr, CZESC[1])[CZESC[0]]
log(f'klatek {len(ss)} ({len(ss) / 30:.0f} s), ta część {len(fr)}')
P = lambda X: PchipInterpolator(kroki, X, axis=0)
sAc, sQc, sGl, sQl, sAl, sPar = (P(X) for X in (Ac, Qc, Gl, Ql, Al, par))
sAt, sRam = P(atn), P(ramka)
sIx = lambda s_: float(np.interp(s_, kroki, np.arange(K)))
cz = Qc[:, narz].mean(1)
sCz = P(uniform_filter1d(cz, 9, axis=0, mode='nearest'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'figure.facecolor': '#03060a', 'text.color': '#e8e8e8'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ramka_idx = np.array([0, W - 1, TOK - 1, TOK - W, 0])


def osie(rect, tytul, xl, yl):
    ax = fig.add_axes(rect); ax.set_facecolor('#03060a'); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for s_ in ax.spines.values():
        s_.set_color('#1f2b25')
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_title(tytul, fontsize=12.5, color='#e8e8e8'); return ax


# ---- 1. globalny (inercjalny)
ext = float(np.abs(Ql).max()) + 1.5
a1 = osie((0.01, 0.47, 0.32, 0.42), '1. RUCH GLOBALNY (układ inercjalny): cała sieć naraz', (-ext, ext), (ext * H / W + 1, -ext * H / W - 1))
a1.plot(*g[ramka_idx].T, color='#55606a', lw=1.2, ls='--')
r1, = a1.plot([], [], color='#39ff7a', lw=2.0)
w1 = a1.scatter([], [], s=4, c='#39ff7a', alpha=0.45, edgecolors='none')
at1 = a1.scatter([], [], s=4, c='#ffffff', alpha=0.8, edgecolors='none')
str1 = a1.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#ffd166', lw=2.2))
txt1 = a1.text(0.02, 0.03, '', transform=a1.transAxes, fontsize=10.5, color='#ffd166')
# ---- 2. lokalny
XL, YL = W / 2 + 1.5, H / 2 + 1.5
sw = Swiatlo(XL, YL, PX=16)
a2 = osie((0.34, 0.47, 0.32, 0.42), '2. ODKSZTAŁCENIE LOKALNE: studnie odpływają od adresów', (-XL, XL), (YL, -YL))
im2 = a2.imshow(np.zeros((sw.NY, sw.NX, 3)), extent=(-XL, XL, YL, -YL), interpolation='bilinear')
a2.scatter(g[:, 0], g[:, 1], s=3, c='#55606a', edgecolors='none', zorder=2)
q2 = a2.quiver(g[:, 0], g[:, 1], np.zeros(TOK), np.zeros(TOK), np.zeros(TOK), cmap='plasma', clim=(0, float(np.percentile(lok, 99))),
               angles='xy', scale_units='xy', scale=1, width=0.0022, headwidth=3, zorder=3)
n2 = a2.scatter([], [], s=30, facecolors='none', edgecolors='#4af0ff', linewidths=1.0, zorder=4)
box2, = a2.plot([], [], color='#ffffff', lw=0.9, ls='--', zorder=5)
# ---- 3. drgania
zx0, zy0 = sCz(0.0)
a3 = osie((0.67, 0.47, 0.32, 0.42), '3. DRGANIA: tokeny wokół centrów studni (kamera idzie za narzędziem)', (zx0 - 3.6, zx0 + 3.6), (zy0 + 2.5, zy0 - 2.5))
im3 = a3.imshow(np.zeros((sw.NY, sw.NX, 3)), extent=(-XL, XL, YL, -YL), interpolation='bilinear')
sl3 = LineCollection([], linewidths=1.3, zorder=3); a3.add_collection(sl3)
sp3 = LineCollection([], colors='#ffffff', linewidths=0.8, alpha=0.45, zorder=4); a3.add_collection(sp3)
at3 = a3.scatter([], [], s=110, c=[], cmap='plasma', vmin=0, vmax=float(np.percentile(vib, 97)), edgecolors='#ffffff', linewidths=0.5, zorder=5)
n3 = a3.scatter([], [], s=300, facecolors='none', edgecolors='#4af0ff', linewidths=1.4, zorder=6)
# ---- 4. atencja na zdjęciu
img = oko()


def ramka_xy(b):
    return [b[0], b[2], b[2], b[0], b[0]], [b[1], b[1], b[3], b[3], b[1]]


def okno_oka(rect):
    ax = fig.add_axes(rect); ax.set_xticks([]); ax.set_yticks([]); ax.imshow(img, extent=(0, W * 16, H * 16, 0))
    ax.plot(*ramka_xy(GT), color='#ffffff', lw=1.2, ls='--', zorder=5)
    ax.set_xlim(0, W * 16); ax.set_ylim(H * 16, 0); return ax


a4 = okno_oka((0.01, 0.07, 0.24, 0.34))
mapa4 = a4.imshow(np.zeros((H, W)), extent=(0, W * 16, H * 16, 0), cmap='inferno', vmin=0, vmax=1, alpha=0.62, interpolation='bilinear')
pr4, = a4.plot([], [], color='#4af0ff', lw=2.0, zorder=6)
a4.set_title('4. Atencja bloku 6 (√): ile uwagi dostaje patch', fontsize=11.5, color='#e8e8e8')
a4.text(0.01, 0.02, 'biała przerywana = prawdziwa ramka,  błękitna = ramka modelu', transform=a4.transAxes, fontsize=9, color='#e8e8e8')
# ---- 5. centra na zdjęciu
a5 = okno_oka((0.26, 0.07, 0.24, 0.34))
mapa5 = a5.imshow(np.zeros((H, W)), extent=(0, W * 16, H * 16, 0), cmap='magma', vmin=0, vmax=float(np.percentile(lok, 99)), alpha=0.6,
                  interpolation='bilinear')
t5 = a5.set_title('', fontsize=11.5, color='#e8e8e8')
# ---- 6. trzy składowe
tk = [0, 300, 1000, EP * 5, EP * 20, EP * 60]
tki = [int(np.argmin(abs(kroki - k))) for k in tk]; tkl = ['0', '300', '1k', 'e5', 'e20', 'e60']
a6 = fig.add_axes((0.545, 0.09, 0.20, 0.30)); a6.set_facecolor('#0b1014')
a6.plot(glob, color='#ffd166', lw=1.2, label='1. globalny')
a6.plot(lok_m, color='#c264ff', lw=1.2, label='2. lokalny')
a6.plot(vib_m, color='#ff7a45', lw=1.0, label='3. drgania')
kur6 = a6.axvline(0, color='#ffffff', lw=1.2)
a6.set_xticks(tki); a6.set_xticklabels(tkl)
a6.tick_params(colors='#cfd3d8', labelsize=9); a6.grid(True, color='#22292f'); a6.set_ylabel('średnio [komórki siatki]', color='#cfd3d8')
a6.legend(fontsize=9, loc='upper left', facecolor='#161e24', edgecolor='#2a333a', labelcolor='#e8e8e8')
a6.set_title('6. Wielkość trzech składowych ruchu', fontsize=12, color='#e8e8e8')
# ---- 7. IoU i jego zmiana
a7 = fig.add_axes((0.785, 0.09, 0.18, 0.30)); a7.set_facecolor('#0b1014')
a7b = a7.twinx()
kol = np.where(d_iou >= 0, '#39ff7a', '#ff4d6d')
a7b.bar(np.arange(K), d_iou, width=1.0, color=kol, alpha=0.55)
lim = float(np.percentile(np.abs(d_iou), 99.5)) * 1.1
a7b.set_ylim(-lim, lim); a7b.tick_params(colors='#7f8a93', labelsize=8); a7b.set_ylabel('zmiana IoU od poprzedniego checkpointu', color='#7f8a93', fontsize=9)
a7.plot(iou, color='#4af0ff', lw=1.3, zorder=3)
a7.set_zorder(a7b.get_zorder() + 1); a7.patch.set_visible(False); a7b.set_facecolor('#0b1014')
kur7 = a7.axvline(0, color='#ffffff', lw=1.2); pkt7, = a7.plot([], [], 'o', color='#ffffff', ms=6, zorder=5)
a7.set_xticks(tki); a7.set_xticklabels(tkl); a7.set_ylim(0, 1)
a7.tick_params(colors='#cfd3d8', labelsize=9); a7.grid(True, color='#22292f'); a7.set_ylabel('IoU', color='#4af0ff')
a7.set_title('7. IoU z checkpointu na checkpoint', fontsize=12, color='#e8e8e8')
txt7 = a7.text(0.98, 0.04, '', transform=a7.transAxes, ha='right', fontsize=11, color='#ffffff', zorder=6)
tytul = fig.text(0.5, 0.955, '', ha='center', fontsize=18)
fig.text(0.5, 0.92, 'Film 13: czym są przesunięcia tokenów — wolniej, z atencją, centrami na zdjęciu i IoU', ha='center', fontsize=12.5, color='#ff9a5c')

ff = None if PROBA else ffmpeg(HERE / f'lasery_13_czesc_{CZESC[0]}.mp4')
TRAIL = 36
hist = [(s_, sAc(s_)) for s_ in ss[max(fr[0] - TRAIL, 0):fr[0]]]
for n, f in enumerate(fr):
    kr = ss[f]; ti = sIx(kr); tj = int(round(ti))
    ac, qc, gl, ql, al, p = sAc(kr), sQc(kr), sGl(kr), sQl(kr), sAl(kr), sPar(kr)
    r1.set_data(*gl[ramka_idx].T); w1.set_offsets(ql); at1.set_offsets(al)
    str1.xy = tuple(gl.mean(0)); str1.set_position(tuple(g.mean(0)))
    txt1.set_text(f'przesunięcie {np.hypot(*p[:2]):.2f} kom.   obrót {p[2]:+.1f}°   skala ×{p[3]:.3f}')
    L = sw.rgb(sw.studnie(qc)); im2.set_data(L); im3.set_data(L)
    d = qc - g; q2.set_UVC(d[:, 0], d[:, 1], np.linalg.norm(d, axis=1)); n2.set_offsets(qc[narz])
    hist.append((kr, ac.copy())); hist = [h for h in hist[-TRAIL:] if kr - h[0] <= 200]
    Hh = np.array([h[1] for h in hist]); vv = np.linalg.norm(ac - qc, axis=1)
    if len(Hh) > 1:
        seg = np.stack([Hh[:-1], Hh[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        kk = np.zeros((len(seg), 4)); kk[:] = [1.0, 0.82, 0.4, 1.0]; kk[:, 3] = np.tile(np.linspace(0.04, 0.7, len(Hh) - 1), TOK)
        sl3.set_segments(seg); sl3.set_color(kk)
    sp3.set_segments(np.stack([ac, qc], 1))
    at3.set_offsets(ac); at3.set_array(vv); n3.set_offsets(ac[narz])
    zx, zy = sCz(kr); a3.set_xlim(zx - 3.6, zx + 3.6); a3.set_ylim(zy + 2.5, zy - 2.5)
    box2.set_data([zx - 3.6, zx + 3.6, zx + 3.6, zx - 3.6, zx - 3.6], [zy - 2.5, zy - 2.5, zy + 2.5, zy + 2.5, zy - 2.5])
    m4 = np.maximum(sAt(kr), 0); mapa4.set_data(np.sqrt(m4 / max(m4.max(), 1e-12)).reshape(H, W))
    pr4.set_data(*ramka_xy(sRam(kr)))
    mapa5.set_data(np.linalg.norm(qc - g, axis=1).reshape(H, W))
    t5.set_text(f'5. Gdzie centra odpłynęły najdalej; narzędzie/reszta ×{np.interp(ti, np.arange(K), narz_lok):.2f}')
    kur6.set_xdata([ti, ti]); kur7.set_xdata([ti, ti])
    iou_t = float(np.interp(ti, np.arange(K), iou)); pkt7.set_data([ti], [iou_t])
    txt7.set_text(f'IoU {iou_t:.2f}   zmiana {d_iou[tj]:+.2f}')
    tytul.set_text(f'krok {kr:,.0f}   epoka {kr / EP:5.2f}   IoU {iou_t:.2f}'.replace(',', ' '))
    fig.canvas.draw()
    if PROBA:
        fig.savefig(HERE / f'lasery_13_proba_{f}.png', dpi=60)
    else:
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if n % 200 == 0:
            log(f'część {CZESC[0]}: klatka {n}/{len(fr)}')
if ff:
    ff.stdin.close(); ff.wait()
log('część gotowa')
