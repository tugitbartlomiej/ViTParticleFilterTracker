"""Film „sześcian”: idealny kryształ w chwili 0 (siatka 30 × 20 komórek = patche obrazu, w środku sześcianu), 600 atomów = tokeny,
prawdziwe przesunięcia tokenów z 60 epok rzutowane stałą mapą (szescian_prep.py na Edenie): x, y = 2. i 3. kierunek ruchu, głębia = 1. kierunek
(największy ruch). 1 komórka = odstęp sąsiadów w kroku 0. Komórki (kwadraty między sąsiednimi węzłami) idą za wolnymi węzłami sieci
(średnia ±300 kroków), kolor = zmiana pola: czerwone rosną, niebieskie się kurczą. Atomy (małe kule) drgają wokół węzłów.
Dwa widoki: LEWY = układ inercjalny (nieruchoma rama, cała sieć: jak się przesuwa, obraca i rośnie jako całość),
PRAWY = układ nieinercjalny współporuszający się z siecią (zdjęte przesunięcie, obrót i skala całej sieci, Prokrustes do sieci idealnej),
w zbliżeniu: widać tylko lokalne zwężanie i rozszerzanie komórek. Płynna interpolacja (splajn) i spowolnienie: kroki 0–300 ×8, 300–1100 ×4.
Uruchomienie: F:/Python/3.13/python.exe film_szescian.py [zdjęcie=0] [proba] → szescian_<zdjęcie>.mp4"""
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
kroki = d['kroki']; K = len(kroki); g = d['g']
A, Q = d[f'atomy_{J}'], d[f'wezly_{J}']; iou = d[f'iou_{J}']; narz = d[f'narzedzie_{J}']; udz = float(d[f'udzial_{J}']); split = str(d[f'split_{J}'])
kom = np.array([(r * W + c, r * W + c + 1, (r + 1) * W + c + 1, (r + 1) * W + c) for r in range(H - 1) for c in range(W - 1)])
wiaz = [(r * W + c, r * W + c + 1) for r in range(H) for c in range(W - 1)] + [(r * W + c, (r + 1) * W + c) for r in range(H - 1) for c in range(W)]


def pola(P):
    x, y = P[..., kom, 0], P[..., kom, 1]
    return 0.5 * np.abs((x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y).sum(-1))


def prokrustes(q):
    """Przesunięcie, obrót i skala sieci q (600,3) najlepiej dopasowujące ją do sieci idealnej g: zwraca funkcję przekształcenia i parametry."""
    mq, mg = q.mean(0), g.mean(0)
    qc, gc = q - mq, g - mg
    U_, S_, Vt = np.linalg.svd(qc.T @ gc)
    dd = np.sign(np.linalg.det(U_ @ Vt)); Dg = np.diag([1, 1, dd])
    R = U_ @ Dg @ Vt
    sk = float((S_ * np.diag(Dg)).sum() / (qc ** 2).sum())
    kat = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return (lambda P: (P - mq) @ R * sk + mg), dict(przesuniecie=float(np.linalg.norm(mq - mg)), obrot=kat, skala=1 / sk)


gest = np.where(kroki < 300, 8, np.where(kroki < 1100, 4, 1))
tt = np.concatenate([np.linspace(t, t + 1, n, endpoint=False) for t, n in zip(range(K - 1), gest[:-1])] + [[K - 1.0]] * 90)
if PROBA:
    tt = tt[[0, 300, 900, len(tt) - 1]]
sA, sQ = CubicSpline(np.arange(K), A, axis=0), CubicSpline(np.arange(K), Q, axis=0)
krok_t = np.interp(tt, np.arange(K), kroki); iou_t = np.interp(tt, np.arange(K), iou)
zlim = float(np.percentile(np.abs(A[..., 2]), 99.5)) * 1.1
ext = float(np.percentile(np.abs(Q[..., :2]), 99.9)) + 1.5                       # zasięg całej sieci w układzie inercjalnym
xl_i, yl_i = max(W / 2 + 1.5, ext), max(H / 2 + 1.5, ext * H / W)
log(f'zdjęcie {J} ({split}), klatek {len(tt)} ({len(tt) / 30:.0f} s), głębia ±{zlim:.1f}, zasięg sieci ±{ext:.1f} komórki')

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'figure.facecolor': '#070a0d', 'text.color': '#e8e8e8'})
cmap_pole = plt.cm.RdBu_r
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
P0 = pola(g[None])[0]


def scena(rect, xl, yl, zl, tytul_panelu):
    ax = fig.add_axes(rect, projection='3d', computed_zorder=False)
    ax.set_facecolor('#070a0d'); ax.set_axis_off()
    ax.set_xlim(-xl, xl); ax.set_ylim(-yl, yl); ax.set_zlim(-zl, zl); ax.set_box_aspect((2 * xl, 2 * yl, 2 * zl))
    X0, X1, Y0, Y1, Z0, Z1 = -W / 2, W / 2, -H / 2, H / 2, -zlim, zlim
    nar = np.array([[x, y, z] for x in (X0, X1) for y in (Y0, Y1) for z in (Z0, Z1)])
    kr = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    ax.add_collection3d(Line3DCollection([nar[list(k)] for k in kr], colors='#6b7c88', linewidths=1.1, alpha=0.75))
    ref = Line3DCollection([g[[i, k]] for i, k in wiaz], colors='#2c3943', linewidths=0.5, alpha=0.6); ax.add_collection3d(ref, autolim=False)
    komorki = Poly3DCollection([], linewidths=0.6, edgecolors='#9cc3de', alpha=0.55); ax.add_collection3d(komorki, autolim=False)
    spr = Line3DCollection([], colors='#ffb347', linewidths=0.35, alpha=0.5); ax.add_collection3d(spr, autolim=False)
    at = ax.scatter([], [], [], s=5, c='#f4f4f4', depthshade=False, edgecolors='none', alpha=0.9)
    atn = ax.scatter([], [], [], s=34, facecolors='none', edgecolors='#4af0ff', linewidths=1.2, depthshade=False)
    ax.set_title(tytul_panelu, color='#e8e8e8', fontsize=13, y=0.98)
    return ax, komorki, spr, at, atn


ax_i, kom_i, spr_i, at_i, atn_i = scena((-0.02, 0.2, 0.37, 0.7), xl_i, yl_i, zlim, 'INERCJALNY: cała sieć, nieruchoma rama')
ax_n, kom_n, spr_n, at_n, atn_n = scena((0.29, 0.2, 0.37, 0.7), W / 2 * 0.62, H / 2 * 0.62, zlim * 0.62, 'NIEINERCJALNY, zbliżenie: tylko odkształcenie')
# widok z góry: sieć krystaliczna i lecące atomy (układ współporuszający się, rzut ortogonalny)
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
ag = fig.add_axes((0.665, 0.2, 0.33, 0.68)); ag.set_facecolor('#070a0d'); ag.set_aspect('equal'); ag.set_xticks([]); ag.set_yticks([])
for sp_ in ag.spines.values():
    sp_.set_color('#3a444c')
ag.set_xlim(-W / 2 - 2, W / 2 + 2); ag.set_ylim(H / 2 + 2, -H / 2 - 2)
ag.add_collection(LineCollection([g[[i, k], :2] for i, k in wiaz], colors='#2c3943', linewidths=0.6))
kom_g = PolyCollection([], edgecolors='#9cc3de', linewidths=0.7, alpha=0.6); ag.add_collection(kom_g)
sl_g = LineCollection([], linewidths=0.9); ag.add_collection(sl_g)
at_g = ag.scatter([], [], s=9, c='#ffffff', zorder=4, edgecolors='none')
atn_g = ag.scatter([], [], s=50, facecolors='none', edgecolors='#4af0ff', linewidths=1.3, zorder=5)
ag.set_title('Z GÓRY: sieć krystaliczna i lecące w niej atomy (ślady)', color='#e8e8e8', fontsize=13)
hist_g = []
tytul = fig.text(0.5, 0.955, '', ha='center', fontsize=17)
faza = fig.text(0.5, 0.925, '', ha='center', fontsize=13, color='#ff7a45')
ruch = fig.text(0.30, 0.17, '', fontsize=11.5, color='#ffd166')
fig.text(0.015, 0.025, 'Szara siatka = idealny kryształ (chwila 0, 30 × 20 patchy). Kwadraty = komórki za węzłami sieci: czerwone rosną, niebieskie się kurczą. '
         'Białe kropki = 600 atomów (tokenów), pomarańczowe kreski = sprężynki atom–węzeł, niebieskie obwódki = narzędzie.\n'
         f'1 komórka = odstęp sąsiadów w kroku 0. Osie: x, y = 2. i 3. kierunek ruchu tokenów, głębia = 1. kierunek; rzut niesie {udz:.0%} energii przesunięć.',
         fontsize=10.5, color='#cfd3d8')
ab = fig.add_axes((0.04, 0.12, 0.2, 0.11)); ab.set_facecolor('#0b1014')
rozsz = np.log(pola(Q) / P0)
ab.plot(np.arange(K), np.percentile(rozsz, 90, axis=1), color='#ff6b5b', lw=1, label='rosnące komórki (90. pc.)')
ab.plot(np.arange(K), np.percentile(rozsz, 10, axis=1), color='#5b9bff', lw=1, label='kurczące się (10. pc.)')
ab.plot(np.arange(K), iou, color='#4af0ff', lw=0.8, alpha=0.7, label='IoU')
kur = ab.axvline(0, color='#ffffff', lw=1.2)
tk = [0, 300, 1000, EP * 5, EP * 20, EP * 60]
ab.set_xticks([int(np.argmin(abs(kroki - k))) for k in tk]); ab.set_xticklabels(['0', '300', '1k', 'e5', 'e20', 'e60'])
ab.tick_params(colors='#cfd3d8', labelsize=8); ab.grid(True, color='#22292f')
ab.legend(fontsize=7.5, loc='lower right', facecolor='#161e24', edgecolor='#2a333a', labelcolor='#e8e8e8', ncol=3)
ab.set_title('ln(pole komórki / pole w chwili 0)', fontsize=9.5, color='#e8e8e8')

# obraz oka: zdjęcie z siatką patchy pokolorowaną rozciąganiem komórek wokół każdego węzła (ta sama skala co kwadraty)
import json as _json, zipfile as _zip  # noqa: E402
import cv2 as _cv2  # noqa: E402
_fid = str(d[f'frame_{J}']).split('/')
_film, _nr = _fid[-2], int(_fid[-1])
_raw = _zip.ZipFile(rf'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Dataset/Cataract-LMM/3_Object_Tracking/annotations/{_film}.zip').read(f'{_film}/{_film}_{_nr:06d}.jpg')
_foto = _cv2.resize(_cv2.cvtColor(_cv2.imdecode(np.frombuffer(_raw, np.uint8), _cv2.IMREAD_COLOR), _cv2.COLOR_BGR2RGB), (480, 320), interpolation=_cv2.INTER_AREA)
_mp = _json.load(open(HERE.parent / 'eden' / 'wyniki' / 'pomiar60' / 'manifest.json', encoding='utf-8'))
_bb = _mp['images'][J if split == 'TRAIN' else 3000 + J - 50]['bbox_xyxy_480']
_wezel_kom = [[] for _ in range(TOK)]
for _ci, _k in enumerate(kom):
    for _v in _k:
        _wezel_kom[_v].append(_ci)
def mapa_wezlow(pr):
    return np.array([pr[c].mean() for c in _wezel_kom]).reshape(H, W)
ae = fig.add_axes((0.715, 0.07, 0.25, 0.25))
ae.imshow(_foto, extent=(0, 480, 320, 0))
im_e = ae.imshow(np.zeros((H, W)), extent=(0, 480, 320, 0), cmap=cmap_pole, vmin=-0.8, vmax=0.8, alpha=0.5, interpolation='nearest')
from matplotlib.patches import Rectangle as _Rect  # noqa: E402
ae.add_patch(_Rect((_bb[0], _bb[1]), _bb[2] - _bb[0], _bb[3] - _bb[1], fill=False, ec='#4af0ff', lw=1.8))
for _gx in range(0, 481, 16):
    ae.axvline(_gx, color='#ffffff', lw=0.15, alpha=0.3)
for _gy in range(0, 321, 16):
    ae.axhline(_gy, color='#ffffff', lw=0.15, alpha=0.3)
ae.set_xticks([]); ae.set_yticks([]); ae.set_xlim(0, 480); ae.set_ylim(320, 0)
ae.set_title('oko: każdy patch = węzeł; kolor = rozciąganie komórek wokół niego', fontsize=10, color='#e8e8e8')

out = HERE / (f'proba_{J}.mp4' if PROBA else f'szescian_{J}.mp4')
fig.canvas.draw(); wid, hei = fig.canvas.get_width_height()
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{wid}x{hei}', '-r', '30', '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-maxrate', '12M', '-bufsize', '24M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)],
                      stdin=subprocess.PIPE)


def rysuj(komorki, spr, at, atn, q, a, pr):
    komorki.set_verts([q[k] for k in kom]); komorki.set_facecolor(cmap_pole(np.clip(0.5 + pr / 1.6, 0, 1)))
    spr.set_segments(np.stack([q, a], 1))
    at._offsets3d = (a[:, 0], a[:, 1], a[:, 2]); atn._offsets3d = (a[narz, 0], a[narz, 1], a[narz, 2])


for n, tau in enumerate(tt):
    a, q = sA(tau), sQ(tau)
    pr = np.log(pola(q[None])[0] / P0)
    rysuj(kom_i, spr_i, at_i, atn_i, q, a, pr)
    f, par = prokrustes(q)
    qn, an = f(q), f(a)
    prn = np.log(pola(qn[None])[0] / P0)
    rysuj(kom_n, spr_n, at_n, atn_n, qn, an, prn)
    im_e.set_data(mapa_wezlow(prn))
    kom_g.set_verts([qn[k, :2] for k in kom]); kom_g.set_facecolor(cmap_pole(np.clip(0.5 + prn / 1.6, 0, 1)))
    hist_g.append(an[:, :2]); hist_g = hist_g[-14:]
    if len(hist_g) > 1:
        Hh = np.stack(hist_g); seg = np.stack([Hh[:-1], Hh[1:]], 2).reshape(-1, 2, 2)
        alf = np.repeat(np.linspace(0.03, 0.75, len(Hh) - 1), TOK)
        kol = np.where(np.tile(narz, len(Hh) - 1)[:, None], [[0.29, 0.94, 1.0]], [[1.0, 0.85, 0.5]])
        sl_g.set_segments(seg); sl_g.set_color(np.concatenate([kol, alf[:, None]], 1))
    at_g.set_offsets(an[:, :2]); atn_g.set_offsets(an[narz, :2])
    az = -60 + 70 * n / max(len(tt) - 1, 1); el = 30 - 8 * np.sin(n / max(len(tt) - 1, 1) * np.pi)
    ax_i.view_init(elev=el, azim=az); ax_n.view_init(elev=el + 8, azim=az)
    kur.set_xdata([tau, tau])
    ruch.set_text(f'ruch całej sieci zdjęty w środkowym i prawym widoku: przesunięcie {par["przesuniecie"]:.2f} komórki, obrót {par["obrot"]:.1f}°, skala ×{par["skala"]:.2f}')
    k_ = krok_t[n]
    tytul.set_text(f'Kryształ tokenów w sześcianie — krok {k_:.0f} (epoka {k_ / EP:.2f}), IoU {iou_t[n]:.2f}   [{split}]')
    faza.set_text('» narodziny kryształu (8× zwolnione)' if k_ < 300 else '» formowanie (4× zwolnione)' if k_ < 1100 else '')
    fig.canvas.draw(); ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    if n % 200 == 0:
        log(f'klatka {n}/{len(tt)}')
ff.stdin.close(); ff.wait()
fig.savefig(HERE / (f'proba_{J}.png' if PROBA else f'szescian_{J}_koniec.png'))
log('zapisano', out, f'{out.stat().st_size / 1e6:.1f} MB')
