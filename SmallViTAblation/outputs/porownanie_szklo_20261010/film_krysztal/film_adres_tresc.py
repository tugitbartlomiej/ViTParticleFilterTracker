"""Film „adres i treść”: rozkład przestrzeni tokenów na podprzestrzeń adresu (kierunki szablonu pozycji: średnia tokenu j po 100 zdjęciach,
90 % jego energii, wyznaczana w każdym checkpoincie) i podprzestrzeń treści (reszta). Jedno zdjęcie uczące, 806 checkpointów (60 epok).
Panele: (1) tokeny zdjęcia w 2 głównych osiach ADRESU (wiązania = sąsiednie patche; tu powinna być siatka), (2) te same tokeny w 2 głównych
osiach TREŚCI (tu narzędzie odróżnia się od tła), (3) dwa słupki: ile wariancji położeń i ile drgań przypada na adres, (4) przebiegi w czasie:
obrót osi adresu (cos kątów głównych z osiami z epoki 1), udział wariancji i drgań w adresie, wymiar adresu k.
Drgania = x − średnia x w oknie ±300 kroków (stałe okno czasowe). Dodatkowo: Spearman ruchliwość zdjęcia (pełna / tylko treść / tylko adres)
kontra IoU, epoki 5–60, 100 zdjęć → ADRES_TRESC.json.
Uruchomienie: F:/Python/3.13/python.exe film_adres_tresc.py → adres_tresc.mp4"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).parent
EKS = HERE.parents[2] / 'assets' / 'RuchCzasteczek_ViT_szklo_2026-10' / '03_pomiar_naturalny_60_epok_eden' / 'ruch60_rp128_50tr_50val'
POM = HERE.parent / 'eden' / 'wyniki' / 'pomiar60' / 'manifest.json'
J, H, W, TOK, EP, PL, OKNO = 0, 20, 30, 600, 1094, 16, 300
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
META = json.load(open(EKS / 'META.json', encoding='utf-8'))
kroki = np.array(META['kroki']); K = len(kroki)
rp = np.load(EKS / 'rp128.npy', mmap_mode='r')
oceny = np.load(EKS / 'oceny.npy')
NI = rp.shape[1]
b = json.load(open(POM, encoding='utf-8'))['images'][J]['bbox_xyxy_480']
siatka = np.stack([np.arange(TOK) // W, np.arange(TOK) % W], 1)
cx = (siatka[:, 1] + 0.5) * PL; cy = (siatka[:, 0] + 0.5) * PL
w_ramce = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
wiaz = [(i, i + 1) for i in range(TOK) if i % W < W - 1] + [(i, i + W) for i in range(TOK - W)]

# ---------------------------------------------------------------- przebieg: bazy adresu, udziały, rzuty zdjęcia J
bufor = {}
wz = lambda t: (lambda X: X - X.mean(1, keepdims=True))(np.asarray(rp[t]).astype(np.float32))
B1 = None
A2 = np.empty((K, TOK, 2), np.float32); T2 = np.empty((K, TOK, 2), np.float32)
kk_, cos1, var_adr, drg_adr = np.empty(K, int), np.empty(K), np.empty(K), np.empty(K)
mob = np.zeros((K, NI, 3))                                                      # ruchliwość zdjęcia: pełna, treść, adres
prevA = prevT = None


def wyrownaj(y, prev):
    if prev is not None:
        U_, _, Vt = np.linalg.svd(y.T @ prev); y = y @ (U_ @ Vt)
    return y / np.sqrt((y ** 2).sum(1).mean())


for t in range(K):
    okno = [u for u in range(K) if abs(kroki[u] - kroki[t]) <= OKNO]
    for u in okno:
        if u not in bufor:
            bufor[u] = wz(u).astype(np.float16)
    for u in [u for u in bufor if u not in okno]:
        del bufor[u]
    X = bufor[t].astype(np.float32)
    c = np.mean([bufor[u].astype(np.float32) for u in okno], 0)
    P = X.mean(0); _, s, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    k = int(np.searchsorted(np.cumsum(s ** 2) / (s ** 2).sum(), 0.9) + 1)
    B = Vt[:k].T                                                                # (128, k) adres
    if kroki[t] >= EP and B1 is None:
        B1 = B.copy()
    cos1[t] = float(np.linalg.svd(B.T @ B1, compute_uv=False).mean()) if B1 is not None else np.nan
    Xc = X - X.mean((0, 1), keepdims=True)
    var_adr[t] = float(((Xc.reshape(-1, 128) @ B) ** 2).sum() / (Xc ** 2).sum())
    d = (X - c).reshape(-1, 128)
    da = (d @ B) ** 2
    drg_adr[t] = float(da.sum() / (d ** 2).sum()) if len(okno) > 1 else np.nan
    e_full = (d ** 2).sum(1).reshape(NI, TOK).mean(1); e_adr = da.sum(1).reshape(NI, TOK).mean(1)
    mob[t] = np.stack([e_full, e_full - e_adr, e_adr], 1)
    kk_[t] = k
    xj = X[J] - X[J].mean(0)
    A2[t] = prevA = wyrownaj(xj @ B[:, :2], prevA)
    reszta = xj - (xj @ B) @ B.T
    _, _, Vr = np.linalg.svd(reszta, full_matrices=False)
    T2[t] = prevT = wyrownaj(reszta @ Vr[:2].T, prevT)
    if t % 100 == 0:
        log(f'krok {kroki[t]}: k={k}, cos z ep.1 {cos1[t]:.2f}, wariancja w adresie {var_adr[t]:.2f}, drgania w adresie {drg_adr[t]:.2f}')

# sygnał praktyczny
sel = kroki >= 5 * EP
iou = oceny[sel].mean(0)[:, 1]
syg = {}
for i, n in enumerate(('pelna', 'tylko_tresc', 'tylko_adres')):
    m = mob[sel, :, i].mean(0)
    syg[n] = dict(wszystkie=float(spearmanr(m, iou).statistic), TRAIN=float(spearmanr(m[:50], iou[:50]).statistic), VALID=float(spearmanr(m[50:], iou[50:]).statistic))
log('Spearman ruchliwość–IoU (epoki 5–60):', {k: {kk: round(v, 2) for kk, v in d.items()} for k, d in syg.items()})
json.dump(dict(opis=__doc__.split('\n')[0], okno_krokow=OKNO, spearman_ruch_iou=syg, k=kk_.tolist(), cos_z_epoka1=[None if np.isnan(v) else float(v) for v in cos1],
               wariancja_w_adresie=var_adr.tolist(), drgania_w_adresie=[None if np.isnan(v) else float(v) for v in drg_adr], kroki=kroki.tolist()),
          open(HERE / 'ADRES_TRESC.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- film
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c', 'axes.edgecolor': '#8a8f99',
                     'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#2a333a',
                     'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
kol = plt.cm.twilight((siatka[:, 1] / (W - 1)) * 0.85); kol[:, :3] *= (0.5 + 0.5 * siatka[:, 0:1] / (H - 1))
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55], height_ratios=[1.35, 1], left=0.04, right=0.98, top=0.86, bottom=0.07, wspace=0.15, hspace=0.28)
a1, a2, a3, a4 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :])
panele = []
for a, Y, tyt in ((a1, A2, 'ADRES: 2 główne osie podprzestrzeni pozycji\n(kilka wymiarów ze 128) — tu żyje siatka patchy'),
                  (a2, T2, 'TREŚĆ: 2 główne osie reszty\n(ponad 100 wymiarów) — tu żyje obraz')):
    lim = float(np.percentile(np.abs(Y), 99.5)) * 1.1
    lc = LineCollection([], colors='#8a8f99', linewidths=0.45, alpha=0.55); a.add_collection(lc)
    sc = a.scatter(Y[0, :, 0], Y[0, :, 1], s=13, c=kol, zorder=3)
    scr = a.scatter(Y[0, w_ramce, 0], Y[0, w_ramce, 1], s=46, facecolors='none', edgecolors='#4af0ff', lw=1.2, zorder=4)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_title(tyt, fontsize=12)
    panele.append((lc, sc, scr, Y))
a1.text(0.02, 0.02, 'kolor = kolumna i wiersz patcha na obrazie\nniebieskie obwódki = narzędzie', transform=a1.transAxes, fontsize=9, color='#cfd3d8')
bary = a3.bar(['wariancja\npołożeń', 'drgania'], [var_adr[0], 0], color=['#ffd166', '#ff4433'])
a3.set_ylim(0, 1); a3.axhline(0.5, color='#8a8f99', ls=':', lw=1); a3.grid(True, axis='y', alpha=0.35)
a3.set_title('jaka część przypada\nna ADRES', fontsize=12); tk = a3.text(0.5, 0.93, '', transform=a3.transAxes, ha='center', fontsize=11)
xs = np.arange(K)
a4.plot(xs, cos1, color='#4af0ff', lw=1.6, label='osie adresu: podobieństwo do osi z epoki 1 (1 = stoją)')
a4.plot(xs, var_adr, color='#ffd166', lw=1.4, label='udział wariancji położeń w adresie')
a4.plot(xs, drg_adr, color='#ff4433', lw=1.0, alpha=0.9, label='udział drgań w adresie')
a4.plot(xs, kk_ / 128, color='#c77dff', lw=1.2, label='wymiar adresu k / 128')
kur = a4.axvline(0, color='#ffffff', lw=1.5)
tick_k = [0, 100, 500, 1000, EP * 2, EP * 5, EP * 10, EP * 20, EP * 40, EP * 60]
a4.set_xticks([int(np.argmin(abs(kroki - k))) for k in tick_k]); a4.set_xticklabels(['0', '100', '500', '1000', 'ep 2', 'ep 5', 'ep 10', 'ep 20', 'ep 40', 'ep 60'])
a4.set_xlim(0, K - 1); a4.set_ylim(0, 1.02); a4.grid(True, alpha=0.35); a4.legend(loc='center right', fontsize=10)
a4.set_xlabel('checkpoint (co 10 kroków do 1000, potem co 100)')
tytul = fig.suptitle('', fontsize=17, y=0.975)
s = syg
fig.text(0.04, 0.905, 'Hipoteza: tokeny mają stałą część „adresową” (pozycja patcha) i aktywną „treściową”. Wynik: adres jest zwarty i sztywniejszy, ale jego osie się obracają.',
         fontsize=12, color='#cfd3d8')
fig.text(0.04, 0.88, f'Ruchliwość zdjęcia a IoU (Spearman, epoki 5–60, 100 zdjęć): pełna {s["pelna"]["wszystkie"]:+.2f}, tylko treść {s["tylko_tresc"]["wszystkie"]:+.2f}, '
         f'tylko adres {s["tylko_adres"]["wszystkie"]:+.2f}', fontsize=12, color='#ffd166')
out = HERE / 'adres_tresc.mp4'
fig.canvas.draw(); wid, hei = fig.canvas.get_width_height()
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{wid}x{hei}', '-r', '30', '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
for n, t in enumerate(list(range(K)) + [K - 1] * 60):
    for lc, sc, scr, Y in panele:
        lc.set_segments([[Y[t, i], Y[t, k]] for i, k in wiaz]); sc.set_offsets(Y[t]); scr.set_offsets(Y[t, w_ramce])
    bary[0].set_height(var_adr[t]); bary[1].set_height(0 if np.isnan(drg_adr[t]) else drg_adr[t])
    tk.set_text(f'adres = {kk_[t]} z 128 wymiarów')
    kur.set_xdata([t, t])
    tytul.set_text(f'Adres i treść tokenów — krok {kroki[t]} (epoka {kroki[t] / EP:.2f}), IoU zdjęcia {oceny[t, J, 1]:.2f}')
    fig.canvas.draw(); ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    if n % 200 == 0:
        log(f'klatka {n}')
ff.stdin.close(); ff.wait()
fig.savefig(HERE / 'adres_tresc_ostatnia_klatka.png')
log('zapisano', out, f'{out.stat().st_size / 1e6:.1f} MB')
