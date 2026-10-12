"""Film 2 z 3: „czym są przesunięcia” — pełna interpolacja ruchu 600 tokenów obrazu TRAIN 0 przez 60 epok (szescian.npz, płaszczyzna x, y),
rozłożona na trzy składowe: (1) ruch GLOBALNY całej sieci (przesunięcie, obrót, skala; Prokrustes węzłów do siatki idealnej) — wspólna
zmiana wag działa na wszystkie tokeny naraz; (2) ODKSZTAŁCENIE LOKALNE — przesunięcie centrów studni (węzeł = średnie położenie ±300
kroków) względem adresu z kodowania pozycji, w układzie współporuszającym się; (3) DRGANIA — token wokół swojego centrum, zbliżenie przy
narzędziu ze śladami. Pod spodem: objaśnienie fazy, przebieg wielkości trzech składowych, obraz oka z mapą przesunięcia centrów.
Spowolnienie: kroki 0–300 ×12, 300–1100 ×6, dalej ×1,5. Uruchomienie: F:/Python/3.13/python.exe film_lasery_2_ruch.py [proba]"""
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

from lasery_wspolne import EP, H, HERE, TOK, W, Swiatlo, dane, ffmpeg, oko, prokrustes2d, zastosuj

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
D = dane(0)
kroki, iou, narz = D['kroki'], D['iou'], D['narz']; K = len(kroki)
g = D['g3'][:, :2]; Ql, Al = D['Q3'][..., :2], D['A3'][..., :2]
par, mapy = prokrustes2d(Ql, g)
Qc = np.stack([zastosuj(Ql[t], mapy[t]) for t in range(K)]); Ac = np.stack([zastosuj(Al[t], mapy[t]) for t in range(K)])
# siatka idealna przeniesiona do układu inercjalnego (odwrotność Prokrustesa) = sam ruch globalny
Gl = np.stack([(g - mapy[t][3]) @ mapy[t][1].T / mapy[t][2] + mapy[t][0] for t in range(K)])
glob = np.linalg.norm(Gl - g, axis=-1).mean(1); lok = np.linalg.norm(Qc - g, axis=-1); vib = np.linalg.norm(Ac - Qc, axis=-1)
lok_m, vib_m = lok.mean(1), vib.mean(1)
narz_lok = lok[:, narz].mean(1) / np.maximum(lok[:, ~narz].mean(1), 1e-9) if narz.any() else np.ones(K)
log(f'globalny {glob[-1]:.2f}, lokalny {lok_m[-1]:.2f}, drgania {vib_m[-1]:.2f} komórki (koniec); narzędzie/reszta przesunięcia centrów {narz_lok[-1]:.2f}')
gest = np.where(kroki < 300, 12, np.where(kroki < 1100, 6, 1.5))
tt = np.concatenate([np.linspace(t, t + 1, int(round(n)) if n >= 1 else 1, endpoint=False) if n != 1.5 else np.linspace(t, t + 1, 1 + (t % 2), endpoint=False)
                     for t, n in zip(range(K - 1), gest[:-1])] + [[K - 1.0]] * 90)
if PROBA:
    tt = tt[[0, 200, 600, 1100, len(tt) - 1]]
ix = np.arange(K)
sAc, sQc, sGl, sQl, sAl = (CubicSpline(ix, X, axis=0) for X in (Ac, Qc, Gl, Ql, Al))
sPar = CubicSpline(ix, par, axis=0)
krok_t = np.interp(tt, ix, kroki); iou_t = np.interp(tt, ix, iou)
log(f'klatek {len(tt)} ({len(tt) / 30:.0f} s)')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'figure.facecolor': '#03060a', 'text.color': '#e8e8e8'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
sas = np.array([(r * W + c, r * W + c + 1) for r in range(H) for c in range(W - 1)] + [(r * W + c, (r + 1) * W + c) for r in range(H - 1) for c in range(W)])
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
# ---- 2. lokalny (współporuszający się): studnie + strzałki od adresu do centrum
XL, YL = W / 2 + 1.5, H / 2 + 1.5
sw = Swiatlo(XL, YL, PX=16)
a2 = osie((0.34, 0.47, 0.32, 0.42), '2. ODKSZTAŁCENIE LOKALNE: studnie odpływają od adresów', (-XL, XL), (YL, -YL))
im2 = a2.imshow(np.zeros((sw.NY, sw.NX, 3)), extent=(-XL, XL, YL, -YL), interpolation='bilinear')
a2.scatter(g[:, 0], g[:, 1], s=3, c='#55606a', edgecolors='none', zorder=2)
q2 = a2.quiver(g[:, 0], g[:, 1], np.zeros(TOK), np.zeros(TOK), np.zeros(TOK), cmap='plasma', clim=(0, float(np.percentile(lok, 99))),
               angles='xy', scale_units='xy', scale=1, width=0.0022, headwidth=3, zorder=3)
n2 = a2.scatter([], [], s=30, facecolors='none', edgecolors='#4af0ff', linewidths=1.0, zorder=4)
# ---- 3. drgania (zbliżenie przy narzędziu)
gz = g[narz].mean(0) if narz.any() else np.zeros(2)
cx, cy = float(np.clip(gz[0], -W / 2 + 3.5, W / 2 - 3.5)), float(np.clip(gz[1], -H / 2 + 2.6, H / 2 - 2.6))
a3 = osie((0.67, 0.47, 0.32, 0.42), '3. DRGANIA: tokeny wokół centrów studni (kamera idzie za narzędziem)', (cx - 3.6, cx + 3.6), (cy + 2.5, cy - 2.5))
im3 = a3.imshow(np.zeros((sw.NY, sw.NX, 3)), extent=(-XL, XL, YL, -YL), interpolation='bilinear')
box2, = a2.plot([], [], color='#ffffff', lw=0.9, ls='--', zorder=5)
cz = Qc[:, narz].mean(1) if narz.any() else np.zeros((K, 2))
from scipy.ndimage import uniform_filter1d  # noqa: E402
sCz = CubicSpline(ix, uniform_filter1d(cz, 9, axis=0, mode='nearest'), axis=0)
sl3 = LineCollection([], linewidths=1.3, zorder=3); a3.add_collection(sl3)
sp3 = LineCollection([], colors='#ffffff', linewidths=0.8, alpha=0.45, zorder=4); a3.add_collection(sp3)
at3 = a3.scatter([], [], s=110, c=[], cmap='plasma', vmin=0, vmax=float(np.percentile(vib, 97)), edgecolors='#ffffff', linewidths=0.5, zorder=5)
n3 = a3.scatter([], [], s=300, facecolors='none', edgecolors='#4af0ff', linewidths=1.4, zorder=6)
# ---- 4. objaśnienie
a4 = fig.add_axes((0.01, 0.04, 0.40, 0.37)); a4.axis('off')
a4.add_patch(plt.Rectangle((0, 0), 1, 1, transform=a4.transAxes, fc='#0b1014', ec='#2a333a'))
txt4 = a4.text(0.03, 0.95, '', va='top', fontsize=13, color='#e8e8e8', transform=a4.transAxes, linespacing=1.45)
# ---- 5. przebieg składowych
a5 = fig.add_axes((0.45, 0.07, 0.25, 0.32)); a5.set_facecolor('#0b1014')
a5.plot(ix, glob, color='#ffd166', lw=1.2, label='1. globalny (przesunięcie siatki)')
a5.plot(ix, lok_m, color='#c264ff', lw=1.2, label='2. lokalny (centra od adresów)')
a5.plot(ix, vib_m, color='#ff7a45', lw=1.0, label='3. drgania (token od centrum)')
kur = a5.axvline(0, color='#ffffff', lw=1.2)
tk = [0, 300, 1000, EP * 5, EP * 20, EP * 60]
a5.set_xticks([int(np.argmin(abs(kroki - k))) for k in tk]); a5.set_xticklabels(['0', '300', '1k', 'e5', 'e20', 'e60'])
a5.tick_params(colors='#cfd3d8', labelsize=9); a5.grid(True, color='#22292f'); a5.set_ylabel('średnio [komórki siatki]', color='#cfd3d8')
a5.legend(fontsize=9, loc='upper left', facecolor='#161e24', edgecolor='#2a333a', labelcolor='#e8e8e8')
a5.set_title('Wielkość trzech składowych ruchu', fontsize=12, color='#e8e8e8')
# ---- 6. oko i mapa przesunięcia centrów
a6 = fig.add_axes((0.73, 0.07, 0.26, 0.32)); a6.set_xticks([]); a6.set_yticks([])
img = oko(); a6.imshow(img)
mapa6 = a6.imshow(np.zeros((H, W)), extent=(0, W * 16, H * 16, 0), cmap='magma', vmin=0, vmax=float(np.percentile(lok, 99)), alpha=0.55, interpolation='nearest')
t6 = a6.set_title('', fontsize=11, color='#e8e8e8')
tytul = fig.text(0.5, 0.955, '', ha='center', fontsize=18)
pod = fig.text(0.5, 0.92, 'Film 2: czym są przesunięcia tokenów — ruch rozłożony na trzy składowe', ha='center', fontsize=12.5, color='#ff9a5c')

OPISY = [
    (0, 'KROK 0 — co się w ogóle porusza?',
     'Obraz jest CAŁY CZAS TEN SAM. Token = wektor 768 liczb (tu rzut na płaszczyznę)\n'
     'policzony przez sieć dla jednego patcha. Między krokami uczenia zmieniają się\n'
     'tylko WAGI sieci, więc ten sam patch daje trochę inny wektor:\n'
     'to jest „przesunięcie cząstki”. Na starcie każdy token stoi w swoim adresie.'),
    (30, 'PIERWSZE KROKI — burza',
     'Wagi zmieniają się najszybciej. Każda paczka danych pcha je w innym kierunku,\n'
     'więc tokeny drżą (3.). Część zmiany działa na wszystkie tokeny naraz:\n'
     'cała sieć przesuwa się i obraca (1.) — to wspólny napęd wag.'),
    (300, 'FORMOWANIE — dlaczego centra studni odpływają?',
     'Centrum = średnie położenie tokenu z ±300 kroków (wygładzone drgania).\n'
     'Na starcie = adres z kodowania pozycji. Uczenie dopisuje do tokenu TREŚĆ:\n'
     'patch z narzędziem, źrenicą i tłem kodowany jest inaczej, więc centra\n'
     'rozjeżdżają się od adresów (2.). Siatka odkształca się lokalnie.'),
    (1100, 'EPOKI 1–5 — porządek przetrwał, ale się odkształcił',
     'Adres wyjaśnia już tylko ok. 70% położenia (było 99,8%), reszta to treść.\n'
     '70% najbliższych sąsiadów to nadal sąsiednie patche: siatka trzyma.\n'
     'Drgania maleją razem z uczeniem.'),
    (EP * 5, 'DOJRZAŁOŚĆ — wolne pełzanie i drgania',
     'Studnie pełzną wolno (ok. 1/4 amplitudy drgań na 100 kroków),\n'
     'tokeny drgają w nich; drgania i pełzanie są niezależne (kosinus ≈ 0).\n'
     'W 60. epoce adres wyjaśnia 44% położenia. Im mniej token drga,\n'
     'tym lepsze IoU (Spearman −0,4 do −0,7, także na zdjęciach walidacyjnych).')]

ff = None if PROBA else ffmpeg(HERE / 'lasery_2_ruch.mp4')
hist = []
for f, t in enumerate(tt):
    kr = krok_t[f]
    ac, qc, gl, ql, al = sAc(t), sQc(t), sGl(t), sQl(t), sAl(t); p = sPar(t)
    r1.set_data(*gl[ramka_idx].T); w1.set_offsets(ql); at1.set_offsets(al)
    c0 = g.mean(0); c1 = gl.mean(0); str1.xy = tuple(c1); str1.set_position(tuple(c0))
    txt1.set_text(f'przesunięcie {np.hypot(*p[:2]):.2f} kom.   obrót {p[2]:+.1f}°   skala ×{p[3]:.3f}')
    L = sw.rgb(sw.studnie(qc)); im2.set_data(L); im3.set_data(L)
    d = qc - g; q2.set_UVC(d[:, 0], d[:, 1], np.linalg.norm(d, axis=1)); n2.set_offsets(qc[narz])
    hist.append(ac.copy()); hist = hist[-24:]
    Hh = np.array(hist); vv = np.linalg.norm(ac - qc, axis=1)
    if len(Hh) > 1:
        seg = np.stack([Hh[:-1], Hh[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        al_ = np.tile(np.linspace(0.05, 0.7, len(Hh) - 1), TOK)
        kol = np.zeros((len(seg), 4)); kol[:] = [1.0, 0.82, 0.4, 1.0]; kol[:, 3] = al_
        sl3.set_segments(seg); sl3.set_color(kol)
    sp3.set_segments(np.stack([ac, qc], 1))
    at3.set_offsets(ac); at3.set_array(vv); n3.set_offsets(ac[narz])
    zx, zy = sCz(t); a3.set_xlim(zx - 3.6, zx + 3.6); a3.set_ylim(zy + 2.5, zy - 2.5)
    box2.set_data([zx - 3.6, zx + 3.6, zx + 3.6, zx - 3.6, zx - 3.6], [zy - 2.5, zy - 2.5, zy + 2.5, zy + 2.5, zy - 2.5])
    kur.set_xdata([t, t])
    ti = int(round(t)); mapa6.set_data(np.linalg.norm(qc - g, axis=1).reshape(H, W))
    t6.set_text(f'gdzie centra odpłynęły najdalej (jasno); narzędzie/reszta ×{narz_lok[min(ti, K - 1)]:.2f}')
    nag, tresc = [(a, b) for k, a, b in OPISY if kr >= k][-1]
    txt4.set_text(nag + '\n\n' + tresc)
    tytul.set_text(f'krok {kr:,.0f}   epoka {kr / EP:5.2f}   IoU {iou_t[f]:.2f}'.replace(',', ' '))
    fig.canvas.draw()
    if PROBA:
        fig.savefig(HERE / f'lasery_2_proba_{f}.png', dpi=60)
    else:
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if f % 200 == 0:
            log(f'klatka {f}/{len(tt)}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe')
