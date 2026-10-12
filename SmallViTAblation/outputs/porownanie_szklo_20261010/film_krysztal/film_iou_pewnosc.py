"""Film „IoU a pewność”: 100 zdjęć (50 TRAIN, 50 VALID) przez 806 checkpointów (60 epok). Każdy punkt = zdjęcie: x = pewność szczytu mapy,
y = IoU jego ramki z GT, kolor = ruchliwość kryształu zdjęcia (średnia amplituda drgań tokenów wokół węzłów, ruchliwosc_zdjec.py na Edenie),
ślady = ostatnie chwile. Obok: korelacja Spearmana pewność–IoU po zdjęciach w czasie (TRAIN, VALID), oraz korelacje ruchliwości kryształu
z IoU i z pewnością (wygładzone ±5 checkpointów). Wynik podatności (podatnosc.py) w opisie. Interpolacja i spowolnienie jak w filmach kryształu.
Uruchomienie: F:/Python/3.13/python.exe film_iou_pewnosc.py → iou_pewnosc.mp4"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr

HERE = Path(__file__).parent
EP = 1094
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
d = np.load(HERE / 'ruchliwosc.npz')
kroki, amp, oc, split = d['kroki'], d['amp'], d['oceny'], d['split']
K, N = amp.shape
pew, iou = oc[..., 0], oc[..., 1]
tr = split == 'TRAIN'
sp = lambda a, b: float(spearmanr(a, b).statistic)
kor = {k: np.zeros(K) for k in ('pi_tr', 'pi_va', 'ri_tr', 'ri_va', 'rp_tr', 'rp_va')}
for t in range(K):
    w = slice(max(0, t - 5), t + 6)
    a_, i_, p_ = amp[w].mean(0), iou[w].mean(0), pew[w].mean(0)
    for m, s in ((tr, 'tr'), (~tr, 'va')):
        kor['pi_' + s][t] = sp(pew[t, m], iou[t, m])
        kor['ri_' + s][t] = sp(a_[m], i_[m]); kor['rp_' + s][t] = sp(a_[m], p_[m])
both = ((pew >= 0.9) & (iou >= 0.9)).mean(1)
gest = np.where(kroki < 300, 6, np.where(kroki < 1100, 3, 1))
tt = np.concatenate([np.linspace(t, t + 1, n, endpoint=False) for t, n in zip(range(K - 1), gest[:-1])] + [[K - 1.0]] * 90)
si = lambda Y: PchipInterpolator(np.arange(K), Y, axis=0)
sP, sI, sA = si(pew), si(iou), si(amp)
krok_t = np.interp(tt, np.arange(K), kroki)
vmax = float(np.percentile(amp[kroki >= 1000], 98))
koniec = kroki >= 5 * EP
log(f'klatek {len(tt)}; Spearman pewność–IoU (ep 5–60, średnio) TRAIN {kor["pi_tr"][koniec].mean():+.2f} VALID {kor["pi_va"][koniec].mean():+.2f}; '
    f'ruchliwość–IoU TRAIN {kor["ri_tr"][koniec].mean():+.2f} VALID {kor["ri_va"][koniec].mean():+.2f}; ruchliwość–pewność TRAIN {kor["rp_tr"][koniec].mean():+.2f} VALID {kor["rp_va"][koniec].mean():+.2f}')
json.dump({k: float(v[koniec].mean()) for k, v in kor.items()}, open(HERE / 'IOU_PEWNOSC.json', 'w', encoding='utf-8'), indent=1)
pod = None
try:
    pod = json.load(open(HERE.parent / 'eden' / 'wyniki' / 'podatnosc' / 'PODATNOSC.json', encoding='utf-8'))
except OSError:
    pass

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'axes.facecolor': '#0b1014', 'figure.facecolor': '#070a0d', 'axes.edgecolor': '#3a444c',
                     'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8', 'text.color': '#e8e8e8', 'grid.color': '#22292f',
                     'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], left=0.05, right=0.98, top=0.86, bottom=0.16, wspace=0.16, hspace=0.32)
a = fig.add_subplot(gs[:, 0]); b1 = fig.add_subplot(gs[0, 1]); b2 = fig.add_subplot(gs[1, 1])
a.set_xlim(0, 1.02); a.set_ylim(0, 1.02); a.set_xlabel('pewność wykrycia narzędzia (szczyt mapy)'); a.set_ylabel('IoU ramki z prawdziwą ramką')
a.axvline(0.9, color='#3a444c', ls='--', lw=1); a.axhline(0.9, color='#3a444c', ls='--', lw=1); a.grid(True, alpha=0.5)
for (x, y, t_) in ((0.95, 0.97, 'pewny i trafny'), (0.95, 0.03, 'pewny, ale ramka zła'), (0.03, 0.97, 'trafny, ale niepewny'), (0.03, 0.03, 'ani to, ani to')):
    a.text(x, y, t_, ha='right' if x > 0.5 else 'left', va='top' if y > 0.5 else 'bottom', fontsize=10, color='#8a8f99')
sl = LineCollection([], linewidths=1.0); a.add_collection(sl)
p_tr = a.scatter([], [], s=70, c=[], cmap='inferno', vmin=0, vmax=vmax, edgecolors='#ffffff', linewidths=0.8, marker='o', zorder=4, label='TRAIN (białe obwódki)')
p_va = a.scatter([], [], s=80, c=[], cmap='inferno', vmin=0, vmax=vmax, edgecolors='#4af0ff', linewidths=1.2, marker='D', zorder=4, label='VALID (niebieskie romby)')
cb = fig.colorbar(p_tr, ax=a, fraction=0.035, pad=0.01); cb.set_label('ruchliwość kryształu zdjęcia (amplituda drgań tokenów)')
a.legend(loc='upper left', bbox_to_anchor=(0.0, 0.93), fontsize=10)
a.set_title('każdy punkt = zdjęcie; kolor = jak mocno drga jego kryształ', fontsize=12)
xs = np.arange(K)
b1.plot(xs, kor['pi_tr'], color='#ffffff', lw=1, label='Spearman pewność–IoU, TRAIN')
b1.plot(xs, kor['pi_va'], color='#4af0ff', lw=1, label='VALID')
b1.plot(xs, iou.mean(1), color='#ffd166', lw=1.2, alpha=0.8, label='średnie IoU')
b1.plot(xs, pew.mean(1), color='#c77dff', lw=1.2, alpha=0.8, label='średnia pewność')
b1.plot(xs, both, color='#7bd88f', lw=1.2, alpha=0.8, label='udział zdjęć pewność ≥ 0,9 i IoU ≥ 0,9')
b1.axhline(0, color='#8a8f99', lw=0.8); b1.set_ylim(-0.4, 1.02); b1.grid(True, alpha=0.5); b1.legend(fontsize=8.5, loc='lower right', ncol=2)
b1.set_title('czy pewna sieć to trafna sieć? korelacja po zdjęciach w czasie', fontsize=12)
b2.plot(xs, kor['ri_tr'], color='#ffffff', lw=1.1, label='ruchliwość–IoU, TRAIN'); b2.plot(xs, kor['ri_va'], color='#4af0ff', lw=1.1, label='ruchliwość–IoU, VALID')
b2.plot(xs, kor['rp_tr'], color='#ff9f6b', lw=1.0, ls='--', label='ruchliwość–pewność, TRAIN'); b2.plot(xs, kor['rp_va'], color='#ffd166', lw=1.0, ls='--', label='ruchliwość–pewność, VALID')
b2.axhline(0, color='#8a8f99', lw=0.8); b2.set_ylim(-1, 1); b2.grid(True, alpha=0.5); b2.legend(fontsize=8.5, loc='lower right', ncol=2)
b2.set_title('kryształ a wynik: drgający kryształ = gorsza detekcja?', fontsize=12)
tk = [0, 100, 300, 1000, EP * 5, EP * 20, EP * 60]
kursory = []
for bx in (b1, b2):
    bx.set_xticks([int(np.argmin(abs(kroki - k))) for k in tk]); bx.set_xticklabels(['0', '100', '300', '1k', 'e5', 'e20', 'e60']); bx.set_xlim(0, K - 1)
    kursory.append(bx.axvline(0, color='#ffffff', lw=1.3))
tytul = fig.text(0.5, 0.95, '', ha='center', fontsize=17)
liczby = fig.text(0.5, 0.905, '', ha='center', fontsize=13, color='#ffd166')
if pod is not None:
    ost = pod['wyniki'][max(pod['wyniki'], key=int)]
    gl, lo, no = ost['gradient']['glowa'], ost['gradient']['lodyga'], ost['podatnosc']['norm']['wszystkie']
    opis = (f'Skąd te dwa wyniki (test podatności, epoka 60): IoU ustala głowa — {gl["udzial_grad_ramki"]:.0%} gradientu straty ramki; '
            f'pewność ustala wejście — łodyga niesie {lo["udzial_grad_mapy"]:.0%} gradientu straty mapy, a norma końcowa działa jak pokrętło pewności '
            f'(szum zawsze ją obniża, {no["d_pewnosc"]:+.4f}, IoU bez zmian).\nKryształ (bloki 0–5) służy obu: gradienty mapy i ramki ciągną w nim w tę samą stronę.')
    fig.text(0.05, 0.03, opis, fontsize=11, color='#cfd3d8')
out = HERE / 'iou_pewnosc.mp4'
fig.canvas.draw(); wid, hei = fig.canvas.get_width_height()
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{wid}x{hei}', '-r', '30', '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-maxrate', '10M', '-bufsize', '20M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)],
                      stdin=subprocess.PIPE)
hist = []
for n, tau in enumerate(tt):
    P, I, A = np.clip(sP(tau), 0, 1), np.clip(sI(tau), 0, 1), sA(tau)
    hist.append(np.stack([P, I], 1)); hist = hist[-12:]
    if len(hist) > 1:
        Hh = np.stack(hist); seg = np.stack([Hh[:-1], Hh[1:]], 2).reshape(-1, 2, 2)
        alf = np.repeat(np.linspace(0.03, 0.6, len(Hh) - 1), N)
        kol = np.where(np.tile(tr, len(Hh) - 1)[:, None], [[1, 1, 1]], [[0.29, 0.94, 1.0]])
        sl.set_segments(seg); sl.set_color(np.concatenate([kol, alf[:, None]], 1))
    p_tr.set_offsets(np.stack([P[tr], I[tr]], 1)); p_tr.set_array(A[tr])
    p_va.set_offsets(np.stack([P[~tr], I[~tr]], 1)); p_va.set_array(A[~tr])
    for k_ in kursory:
        k_.set_xdata([tau, tau])
    ti = int(round(tau))
    k_ = krok_t[n]
    tytul.set_text(f'IoU a pewność dla 100 zdjęć — krok {k_:.0f} (epoka {k_ / EP:.2f})' + ('   » zwolnione' if k_ < 1100 else ''))
    liczby.set_text(f'Spearman pewność–IoU: TRAIN {kor["pi_tr"][ti]:+.2f}, VALID {kor["pi_va"][ti]:+.2f}    |    '
                    f'ruchliwość–IoU: TRAIN {kor["ri_tr"][ti]:+.2f}, VALID {kor["ri_va"][ti]:+.2f}    |    pewne i trafne: {both[ti]:.0%}')
    fig.canvas.draw(); ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    if n % 300 == 0:
        log(f'klatka {n}/{len(tt)}')
ff.stdin.close(); ff.wait()
fig.savefig(HERE / 'iou_pewnosc_ostatnia_klatka.png')
log('zapisano', out, f'{out.stat().st_size / 1e6:.1f} MB')
