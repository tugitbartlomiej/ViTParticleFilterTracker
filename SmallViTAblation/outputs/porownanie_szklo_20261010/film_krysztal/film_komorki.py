"""Film „komórki”: zbliżenie na fragment kryształu wokół narzędzia jako małe sześciany 3D. Wszystko z danych (szescian.npz z Edena),
w układzie współporuszającym się z siecią (zdjęty obrót i skala całej sieci):
  • narożniki sześcianu = węzły sieci (wolne położenia tokenów, średnia ±300 kroków), wysokość komórki = 1 odstęp sąsiadów;
  • atomy = tokeny (szybkie położenia); atom należy do komórki, w której rzucie się znajduje; przejście atomu do innej komórki = błysk;
  • ściana jest wypukła (czerwona) albo wklęsła (niebieska) zgodnie z „ciśnieniem” atomów: średnie wychylenie atomów w komórce wzdłuż
    normalnej ściany (atomy przy ścianie ją wypychają, atomy oddalone — wciągają). To odczyt z danych, nie symulacja.
Sprawdzamy też, czy ciśnienie na ścianę zapowiada jej późniejszy ruch: korelacja wypukłości teraz z przesunięciem środka ściany wzdłuż
normalnej po 300 krokach (w tytule). Spowolnienie: kroki 0–300 ×6, 300–1100 ×3.
Uruchomienie: F:/Python/3.13/python.exe film_komorki.py [zdjęcie=0] [proba] → komorki_<zdjęcie>.mp4"""
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
A, Q = d[f'atomy_{J}'], d[f'wezly_{J}']; iou = d[f'iou_{J}']; narz = d[f'narzedzie_{J}']; split = str(d[f'split_{J}'])


def prokrustes(q):
    mq, mg = q.mean(0), g.mean(0); qc, gc = q - mq, g - mg
    U_, S_, Vt = np.linalg.svd(qc.T @ gc); dd = np.sign(np.linalg.det(U_ @ Vt)); Dg = np.diag([1, 1, dd]); R = U_ @ Dg @ Vt
    sk = float((S_ * np.diag(Dg)).sum() / (qc ** 2).sum())
    return lambda P: (P - mq) @ R * sk + mg


# układ współporuszający się dla wszystkich checkpointów
Qn = np.empty_like(Q); An = np.empty_like(A)
for t in range(K):
    f = prokrustes(Q[t]); Qn[t] = f(Q[t]); An[t] = f(A[t])
# wycinek wokół narzędzia: 10 × 7 komórek
rr, cc = np.where(narz.reshape(H, W))
r0 = int(np.clip(rr.mean() - 3.5, 0, H - 8)); c0 = int(np.clip(cc.mean() - 5, 0, W - 11))
R_, C_ = range(r0, r0 + 7), range(c0, c0 + 10)
kom = [(r, c) for r in R_ for c in C_]
nod = lambda r, c: r * W + c
naroza = np.array([[nod(r, c), nod(r, c + 1), nod(r + 1, c + 1), nod(r + 1, c)] for r, c in kom])     # (n_kom, 4)
wezly_w = np.unique(naroza)
atomy_w = np.array([nod(r, c) for r in range(r0, r0 + 8) for c in range(c0, c0 + 11)])               # atomy przy węzłach wycinka


def w_komorce(P2, quad):
    """Czy punkty P2 (n,2) leżą w czworokącie quad (4,2) (wypukły, kolejność obwodowa)."""
    s = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        s.append((b[0] - a[0]) * (P2[:, 1] - a[1]) - (b[1] - a[1]) * (P2[:, 0] - a[0]))
    s = np.stack(s, 1)
    return (s >= 0).all(1) | (s <= 0).all(1)


def stan(q, a):
    """Przynależność atomów wycinka do komórek i wypukłość ścian (n_kom, 4)."""
    nal = np.full(len(atomy_w), -1)
    for k, nar in enumerate(naroza):
        m = (nal < 0) & w_komorce(a[atomy_w, :2], q[nar, :2])
        nal[m] = k
    wyp = np.zeros((len(kom), 4))
    for k, nar in enumerate(naroza):
        sr = q[nar].mean(0); m = nal == k
        if not m.any():
            continue
        u = a[atomy_w[m], :2] - sr[:2]
        for i in range(4):
            p1, p2 = q[nar[i], :2], q[nar[(i + 1) % 4], :2]
            sc = 0.5 * (p1 + p2) - sr[:2]; nrm = sc / (np.linalg.norm(sc) + 1e-9)
            wyp[k, i] = float((u @ nrm).mean())
    return nal, wyp


# pomiar: czy ciśnienie zapowiada ruch ściany po 300 krokach (na checkpointach od kroku 1000)
xs_, ys_ = [], []
for t in range(K):
    if kroki[t] < 1000 or t % 3:
        continue
    t2 = int(np.searchsorted(kroki, kroki[t] + 300))
    if t2 >= K:
        break
    _, wyp = stan(Qn[t], An[t])
    for k, nar in enumerate(naroza):
        sr = Qn[t][nar].mean(0)
        for i in range(4):
            p1, p2 = Qn[t][nar[i], :2], Qn[t][nar[(i + 1) % 4], :2]
            sc = 0.5 * (p1 + p2) - sr[:2]; nrm = sc / (np.linalg.norm(sc) + 1e-9)
            m1 = 0.5 * (Qn[t][nar[i], :2] + Qn[t][nar[(i + 1) % 4], :2]); m2 = 0.5 * (Qn[t2][nar[i], :2] + Qn[t2][nar[(i + 1) % 4], :2])
            xs_.append(wyp[k, i]); ys_.append(float((m2 - m1) @ nrm))
kor_cis = float(np.corrcoef(xs_, ys_)[0, 1])
log(f'wycinek wiersze {r0}–{r0 + 7}, kolumny {c0}–{c0 + 10}; korelacja ciśnienie ściany → jej ruch po 300 krokach: {kor_cis:+.3f} (n={len(xs_)})')

gest = np.where(kroki < 300, 6, np.where(kroki < 1100, 3, 1))
tt = np.concatenate([np.linspace(t, t + 1, n, endpoint=False) for t, n in zip(range(K - 1), gest[:-1])] + [[K - 1.0]] * 75)
if PROBA:
    tt = tt[[0, 200, 700, len(tt) - 1]]
sQ, sA = CubicSpline(np.arange(K), Qn, axis=0), CubicSpline(np.arange(K), An, axis=0)
krok_t = np.interp(tt, np.arange(K), kroki); iou_t = np.interp(tt, np.arange(K), iou)

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'figure.facecolor': '#070a0d', 'text.color': '#e8e8e8'})
cm = plt.cm.RdBu_r
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ax = fig.add_axes((0.0, 0.06, 0.74, 0.86), projection='3d', computed_zorder=False); ax.set_facecolor('#070a0d'); ax.set_axis_off()
gq = g[wezly_w]
cx, cy = gq[:, 0].mean(), gq[:, 1].mean()
cen_t = Qn[:, wezly_w].mean(1)                                                     # środek wycinka w czasie
rozp = Qn[:, wezly_w] - cen_t[:, None]
zak = np.abs(rozp).max(1) + 0.8                                                     # (K, 3) zasięg wycinka w czasie
zak = np.stack([np.convolve(np.pad(zak[:, i], 10, mode='edge'), np.ones(21) / 21, 'valid') for i in range(3)], 1); zak[:, 2] = np.maximum(zak[:, 2], 1.5)
sCen = CubicSpline(np.arange(K), cen_t, axis=0); sZak = CubicSpline(np.arange(K), zak, axis=0)
sciany = Poly3DCollection([], linewidths=0.3, edgecolors='#9cc3de', alpha=0.5); ax.add_collection3d(sciany, autolim=False)
dachy = Poly3DCollection([], linewidths=0.6, edgecolors='#cfe4f2', alpha=0.18, facecolors='#5b8fb0'); ax.add_collection3d(dachy, autolim=False)
at = ax.scatter([], [], [], s=34, c='#ffffff', depthshade=False, edgecolors='none')
atn = ax.scatter([], [], [], s=90, facecolors='none', edgecolors='#4af0ff', linewidths=1.6, depthshade=False)
blysk = ax.scatter([], [], [], s=260, c='#ffd166', alpha=0.85, depthshade=False, edgecolors='none')
tytul = fig.text(0.37, 0.955, '', ha='center', fontsize=17)
faza = fig.text(0.37, 0.92, '', ha='center', fontsize=13, color='#ff7a45')
licz = fig.text(0.76, 0.6, '', fontsize=13, color='#ffd166', va='top')
fig.text(0.76, 0.88, 'Jak czytać', fontsize=15, color='#ffd166')
fig.text(0.76, 0.66,
         '• każdy mały sześcian = komórka między\n  4 sąsiednimi patchami (węzły = narożniki)\n'
         '• białe kule = atomy (tokeny), niebieskie\n  obwódki = narzędzie\n'
         '• ściana CZERWONA = wypchnięta (wypukła),\n  NIEBIESKA = wciągnięta (wklęsła): to średnie\n  wychylenie atomów komórki w stronę ściany\n'
         '• żółty błysk = atom przeszedł do innej komórki',
         fontsize=11.5, color='#cfd3d8', linespacing=1.4)
fig.text(0.76, 0.12, f'Czy ciśnienie atomów przesuwa ściany?\nkorelacja wypukłości teraz z ruchem ściany\npo 300 krokach: {kor_cis:+.2f}\n'
         + ('→ ściany ustępują atomom' if kor_cis > 0.1 else '→ ściany cofają się przeciw atomom' if kor_cis < -0.1 else '→ ściany nie idą za atomami: węzły przesuwa uczenie'),
         fontsize=12, color='#ff9f6b', linespacing=1.4)
out = HERE / (f'proba_k{J}.mp4' if PROBA else f'komorki_{J}.mp4')
fig.canvas.draw(); wid, hei = fig.canvas.get_width_height()
ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{wid}x{hei}', '-r', '30', '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-maxrate', '12M', '-bufsize', '24M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)],
                      stdin=subprocess.PIPE)
poprz, przejscia, rozblysk = None, 0, {}
for n, tau in enumerate(tt):
    q, a = sQ(tau), sA(tau)
    nal, wyp = stan(q, a)
    polys, kol, dach = [], [], []
    for k, nar in enumerate(naroza):
        dol = q[nar].copy(); dol[:, 2] -= 0.5; gor = q[nar].copy(); gor[:, 2] += 0.5
        sr = q[nar].mean(0)
        for i in range(4):
            j = (i + 1) % 4
            p1, p2 = q[nar[i]], q[nar[j]]
            sc = 0.5 * (p1 + p2) - sr; sc[2] = 0; nrm = sc / (np.linalg.norm(sc) + 1e-9)
            srodek = 0.5 * (p1 + p2) + nrm * np.clip(wyp[k, i], -0.6, 0.6) * 0.8
            # ściana z wybrzuszeniem: 4 trójkąty do środka wybrzuszenia
            v = [dol[i], dol[j], gor[j], gor[i]]
            for m_ in range(4):
                polys.append([v[m_], v[(m_ + 1) % 4], srodek]); kol.append(cm(np.clip(0.5 + wyp[k, i] / 1.2, 0, 1)))
        dach.append(gor)
    sciany.set_verts(polys); sciany.set_facecolor(kol); dachy.set_verts(dach)
    aw = a[atomy_w]
    at._offsets3d = (aw[:, 0], aw[:, 1], aw[:, 2]); m_n = narz[atomy_w]
    atn._offsets3d = (aw[m_n, 0], aw[m_n, 1], aw[m_n, 2])
    if poprz is not None:
        zm = (nal != poprz) & (nal >= 0) & (poprz >= 0)
        przejscia += int(zm.sum())
        for i in np.where(zm)[0]:
            rozblysk[i] = 8
    poprz = nal
    rozblysk = {i: v - 1 for i, v in rozblysk.items() if v > 1}
    b_ = np.array(list(rozblysk), int)
    blysk._offsets3d = (aw[b_, 0], aw[b_, 1], aw[b_, 2]) if len(b_) else ([], [], [])
    c_ = sCen(tau); hx, hy, hz = sZak(tau); hh = max(hx, hy * 1.3)
    ax.set_xlim(c_[0] - hh, c_[0] + hh); ax.set_ylim(c_[1] - hh / 1.3, c_[1] + hh / 1.3); ax.set_zlim(c_[2] - hz, c_[2] + hz); ax.set_box_aspect((2 * hh, 2 * hh / 1.3, 2 * hz), zoom=1.75)
    ax.view_init(elev=38 - 8 * np.sin(n / max(len(tt) - 1, 1) * np.pi), azim=-70 + 40 * n / max(len(tt) - 1, 1))
    k_ = krok_t[n]
    tytul.set_text(f'Komórki kryształu wokół narzędzia — krok {k_:.0f} (epoka {k_ / EP:.2f}), IoU {iou_t[n]:.2f}   [{split}]')
    faza.set_text('» narodziny kryształu (6× zwolnione)' if k_ < 300 else '» formowanie (3× zwolnione)' if k_ < 1100 else '')
    licz.set_text(f'atomów w wycinku: {len(atomy_w)}\nw swoich komórkach: {(nal >= 0).mean():.0%}\nprzejść do innej komórki (łącznie): {przejscia}')
    fig.canvas.draw(); ff.stdin.write(fig.canvas.buffer_rgba().tobytes())
    if n % 200 == 0:
        log(f'klatka {n}/{len(tt)}')
ff.stdin.close(); ff.wait()
fig.savefig(HERE / (f'proba_k{J}.png' if PROBA else f'komorki_{J}_koniec.png'))
log('zapisano', out, f'{out.stat().st_size / 1e6:.1f} MB')
