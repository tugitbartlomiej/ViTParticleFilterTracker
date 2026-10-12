"""Film 1 z 3: „lasery się zapalają” — jak kodowanie pozycji MalyViT (sincos_2d, model.py) nadaje adresy 600 patchom.
Analogia: każda częstotliwość ω_k = 10000^(−k/48) kodowania to para laserów o długości fali λ_k = 2π/ω_k patchy, świecąca wzdłuż osi x
(lewy i prawy) albo y (górny i dolny); w patchu (wiersz, kolumna) daje dwie liczby adresu: sin(ω_k · kolumna) i cos(ω_k · kolumna)
(analogicznie dla wiersza). 48 par × 2 osie × sin/cos = 192 liczby = wektor dodawany do tokenu przed blokami.
Pokazujemy 4 pary na oś (k = 0, 4, 8, 16), wartości liczone dokładnie wzorem z model.py; potem wszystkie 48 par naraz jako podobieństwo
adresów (kosinus kodów) względem wędrującego patcha; na końcu tokeny (kolor = średni kolor patcha oka) wpadają do studni — krok 0.
Uruchomienie: F:/Python/3.13/python.exe film_lasery_1_zapalanie.py [proba] → lasery_1_zapalanie.mp4"""
import sys
import time

import numpy as np

from lasery_wspolne import EP, H, HERE, TOK, W, ZIELONY, Swiatlo, ffmpeg, kolor_fali, oko, sincos_2d  # noqa: F401

PROBA = len(sys.argv) > 1 and sys.argv[1] == 'proba'
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)
PE, om = sincos_2d()
d48 = len(om)
g = np.stack([np.arange(TOK) % W - (W - 1) / 2, np.arange(TOK) // W - (H - 1) / 2], 1).astype(float)   # (x, y) środka patcha w komórkach
KS = [0, 4, 8, 16]
PARY = [('x', k) for k in KS] + [('y', k) for k in KS]
XL, YL = W / 2 + 0.5, H / 2 + 0.5
sw = Swiatlo(XL, YL, PX=24)
img = oko()
# średni kolor patcha
kol_patch = img.reshape(H, 16, W, 16, 3).mean((1, 3)).reshape(TOK, 3)
# wybrane patche do kodów: P1, sąsiad P2, daleki P3
WYB = [(10, 8), (10, 9), (3, 25)]
iw = [r * W + c for r, c in WYB]
KOL_WYB = ['#ffffff', '#ffd166', '#ff5ca8']

# podobieństwo pełnych kodów 192D
PEn = PE / np.linalg.norm(PE, axis=1, keepdims=True)
SIM = PEn @ PEn.T

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402
plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 11, 'figure.facecolor': '#03060a', 'text.color': '#e8e8e8'})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
ax = fig.add_axes((0.005, 0.1, 0.64, 0.8)); ax.set_facecolor('#03060a'); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.set_xlim(-XL - 3.2, XL + 3.2); ax.set_ylim(YL + 3.2, -YL - 3.2); ax.set_aspect('equal')
im_oko = ax.imshow(img, extent=(-W / 2, W / 2, H / 2, -H / 2), alpha=0.0, zorder=0)
im_sw = ax.imshow(np.zeros((sw.NY, sw.NX, 4)), extent=(-XL, XL, YL, -YL), interpolation='bilinear', zorder=1)
siatka = [ax.axvline(c - W / 2, ymin=0, ymax=0, color='#39ff7a', lw=0.4, alpha=0.0) for c in range(W + 1)]
linie_x = [ax.plot([c - W / 2, c - W / 2], [-H / 2, H / 2], color='#9fffc0', lw=0.4, alpha=0.0, zorder=2)[0] for c in range(W + 1)]
linie_y = [ax.plot([-W / 2, W / 2], [r - H / 2, r - H / 2], color='#9fffc0', lw=0.4, alpha=0.0, zorder=2)[0] for r in range(H + 1)]
kul = ax.scatter(g[:, 0], g[:, 1], s=34, c=kol_patch, edgecolors='#ffffff', linewidths=0.4, zorder=6, alpha=0.0)
halo = ax.scatter(g[:, 0], g[:, 1], s=120, c=kol_patch, edgecolors='none', zorder=5, alpha=0.0)
wyb = ax.scatter(g[iw, 0], g[iw, 1], s=260, facecolors='none', edgecolors=KOL_WYB, linewidths=2.0, zorder=7, alpha=0.0)
# lasery: 4 źródła (lewy, prawy dla osi x; górny, dolny dla osi y)
ZR = {'l': (-XL - 2.2, 0.0), 'p': (XL + 2.2, 0.0), 'g': (0.0, -YL - 2.2), 'd': (0.0, YL + 2.2)}
obudowy, apertury, stozki = {}, {}, {}
for n, (x_, y_) in ZR.items():
    w_, h_ = (1.6, 2.6) if n in 'lp' else (3.0, 1.4)
    obudowy[n] = ax.add_patch(FancyBboxPatch((x_ - w_ / 2, y_ - h_ / 2), w_, h_, boxstyle='round,pad=0.05,rounding_size=0.25', fc='#141c18', ec='#55645c', lw=1.2, zorder=8))
    apertury[n] = (ax.scatter([x_], [y_], s=300, c=['#333333'], edgecolors='none', zorder=9), ax.scatter([x_], [y_], s=1600, c=['#333333'], alpha=0.0, edgecolors='none', zorder=9))
    if n == 'l':
        pts = [(x_ + 0.8, y_), (-W / 2, -H / 2), (-W / 2, H / 2)]
    elif n == 'p':
        pts = [(x_ - 0.8, y_), (W / 2, -H / 2), (W / 2, H / 2)]
    elif n == 'g':
        pts = [(x_, y_ + 0.7), (-W / 2, -H / 2), (W / 2, -H / 2)]
    else:
        pts = [(x_, y_ - 0.7), (-W / 2, H / 2), (W / 2, H / 2)]
    stozki[n] = ax.add_patch(Polygon(pts, closed=True, fc='#39ff7a', ec='none', alpha=0.0, zorder=1))

# prawy panel: kody trzech patchy
ak = fig.add_axes((0.68, 0.42, 0.3, 0.42)); ak.set_facecolor('#0b1014')
NB = 2 * len(PARY)
xb = np.arange(NB)
bary = [ak.bar(xb + (j - 1) * 0.27, np.zeros(NB), 0.27, color=KOL_WYB[j], label=f'patch wiersz {r}, kol. {c}' + (' (sąsiad)' if j == 1 else ' (daleko)' if j == 2 else ''))
        for j, (r, c) in enumerate(WYB)]
ak.set_ylim(-1.1, 1.1); ak.set_xlim(-0.7, NB - 0.3); ak.axhline(0, color='#55606a', lw=0.8)
ak.set_xticks(xb); ak.set_xticklabels([f'{"sin" if i % 2 == 0 else "cos"}\n{p[0]}{p[1]}' for p in PARY for i in range(2)], fontsize=7.5)
ak.tick_params(colors='#cfd3d8'); ak.grid(True, axis='y', color='#22292f')
ak.legend(fontsize=8.5, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, facecolor='#161e24', edgecolor='#2a333a', labelcolor='#e8e8e8')
ak.set_title('adres patcha = liczby zapalonych laserów (16 ze 192)', fontsize=12, color='#e8e8e8', pad=40)
ap_ = fig.add_axes((0.68, 0.1, 0.3, 0.25)); ap_.set_facecolor('#0b1014'); ap_.axis('off')
opis = ap_.text(0.0, 1.0, '', va='top', fontsize=12.5, color='#e8e8e8', wrap=True, transform=ap_.transAxes)
tytul = fig.text(0.5, 0.955, '', ha='center', fontsize=19)
pod = fig.text(0.5, 0.92, '', ha='center', fontsize=13, color='#ff9a5c')

# ---------------------------------------------------------------- oś czasu (klatki)
F_CIEM, F_OKO, F_PARA, F_ADRES, F_WPAD, F_KON = 75, 135, 105, 330, 270, 120
granice = np.cumsum([0, F_CIEM, F_OKO, F_PARA * len(PARY), F_ADRES, F_WPAD, F_KON])
N_F = granice[-1]
klatki = range(N_F) if not PROBA else [40, 150, 300, 600, 950, 1150, 1450, N_F - 5]
ease = lambda u: 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))
rng = np.random.default_rng(3)
start_los = rng.uniform([-W / 2, -H / 2], [W / 2, H / 2], (TOK, 2))          # tokeny bez adresu: rozrzucone
xs, ys = sw.xs, sw.ys
cx_pix = xs + (W - 1) / 2; ry_pix = ys + (H - 1) / 2                         # współrzędna kolumny / wiersza w pikselach obrazu światła
maska_siatki = ((np.abs(xs)[None, :] <= W / 2) & (np.abs(ys)[:, None] <= H / 2)).astype(float)


def zapal(n, kol, a):
    obudowy[n].set_edgecolor(kol if a > 0.05 else '#55645c')
    apertury[n][0].set_color([kol * (0.3 + 0.7 * a)]); apertury[n][1].set_color([kol]); apertury[n][1].set_alpha(0.25 * a)
    stozki[n].set_facecolor(kol); stozki[n].set_alpha(0.07 * a)


pole = np.zeros((sw.NY, sw.NX, 3))
fale = []                                                                    # (os, k, kolor) zapalone
ff = ffmpeg(HERE / 'lasery_1_zapalanie.mp4') if not PROBA else None
for f in klatki:
    etap = int(np.searchsorted(granice, f, side='right') - 1); u = (f - granice[etap]) / max(1, granice[etap + 1] - granice[etap])
    L = np.zeros((sw.NY, sw.NX, 3)); a_oko = 0.0; a_kul = 0.0; a_siat = 0.0; pos = g
    if etap == 0:
        tytul.set_text('Kodowanie pozycji jako światło laserów'); pod.set_text('MalyViT: obraz 480 × 320 → 600 patchy 16 × 16 px'); opis.set_text('')
    elif etap == 1:
        a_oko = ease(u * 2); a_siat = ease(u * 2 - 0.6)
        tytul.set_text('Obraz dzielimy na 600 patchy (30 × 20)'); pod.set_text('lodyga konwolucyjna robi z każdego patcha token 192D — ale token nie wie, GDZIE leży')
        opis.set_text('Treść patcha to jeszcze nie położenie.\nSieć potrzebuje adresu: tę rolę gra\nkodowanie pozycji (sincos_2d).')
    elif etap == 2:
        a_oko = 0.18; a_siat = 0.35
        i_p = min(int(u * len(PARY)), len(PARY) - 1); up = u * len(PARY) - i_p
        os_, k = PARY[i_p]; lam = 2 * np.pi / om[k]; kol = kolor_fali(lam)
        fale_teraz = PARY[:i_p + 1]
        for j, (o2, k2) in enumerate(fale_teraz):
            jas = 0.9 * ease(up * 2.5) if j == i_p else 0.22
            kk = kolor_fali(2 * np.pi / om[k2])
            wz = (np.sin(om[k2] * cx_pix)[None, :] if o2 == 'x' else np.sin(om[k2] * ry_pix)[:, None]) * np.ones((sw.NY, sw.NX))
            L += ((0.5 + 0.5 * wz) ** 2)[..., None] * kk * jas * maska_siatki[..., None]
        for n in 'lpgd':
            zapal(n, np.array([0.3, 0.3, 0.3]), 0)
        for (o2, k2) in fale_teraz:
            kk = kolor_fali(2 * np.pi / om[k2]); a = ease(up * 3) if (o2, k2) == (os_, k) else 0.55
            for n in ('lp' if o2 == 'x' else 'gd'):
                zapal(n, kk, a)
        tytul.set_text(f'Para laserów {i_p + 1}/8: oś {os_}, częstotliwość k = {k}, długość fali λ = {lam:.1f} patchy')
        pod.set_text('każda para tworzy falę stojącą; w patchu daje 2 liczby adresu: sin i cos fazy')
        opis.set_text(f'ω_k = 10000^(−k/48) = {om[k]:.4f}\nsin(ω_k · {"kolumna" if os_ == "x" else "wiersz"}),  cos(ω_k · {"kolumna" if os_ == "x" else "wiersz"})\n\n'
                      'Krótkie fale (fiolet) odróżniają sąsiadów,\ndługie (czerwień) mówią, w której części\nobrazu leży patch. Razem 48 par na oś.')
        for j, b in enumerate(bary):
            v = np.zeros(NB)
            for q, (o2, k2) in enumerate(PARY[:i_p + 1]):
                kol_ = (WYB[j][1] if o2 == 'x' else WYB[j][0]) * om[k2]
                v[2 * q], v[2 * q + 1] = np.sin(kol_), np.cos(kol_)
                if (o2, k2) == (os_, k):
                    v[2 * q:2 * q + 2] *= ease(up * 2.5)
            for rect, h_ in zip(b, v):
                rect.set_height(h_)
        wyb.set_alpha(1.0)
    elif etap == 3:
        a_oko = 0.1; a_siat = 0.2
        for n, (kol) in zip('lpgd', [kolor_fali(2 * np.pi / om[k]) for k in (0, 8, 4, 16)]):
            zapal(n, kol, 0.8 + 0.2 * np.sin(f * 0.5))
        # wędrujący patch: ścieżka po siatce
        sc = (W - 1) / 2 + 11 * np.cos(2 * np.pi * u * 1.3); sr = (H - 1) / 2 + 7 * np.sin(2 * np.pi * u * 2.1)
        i0 = int(round(sr)) * W + int(round(sc))
        s = SIM[i0]; jas = np.clip((s - np.percentile(s, 50)) / (1 - np.percentile(s, 50)), 0, 1) ** 3
        I = sw.studnie(g, jasnosc=0.25 + 0.75 * jas)
        L = sw.rgb(I) * 1.0
        wyb.set_offsets(g[[i0]]); wyb.set_edgecolors(['#ffffff']); wyb.set_alpha(1.0)
        tytul.set_text('Wszystkie 48 par naraz: każdy patch ma unikalny kod 192 liczb')
        pod.set_text('jasność studni = podobieństwo jej kodu do kodu patcha w białym kółku: sąsiedzi świecą najjaśniej')
        opis.set_text('Interferencja wszystkich wiązek\ntworzy 600 studni światła — po jednej\nna patch. Podobne kody sąsiadów\nsprawiają, że sieć „widzi” ich obok siebie.')
    elif etap == 4:
        a_kul = 1.0; v = ease(u * 1.4)
        pos = start_los * (1 - v)[None] + g * v
        I = sw.studnie(g); L = sw.rgb(I) * (0.4 + 0.6 * v)
        for n, k in zip('lpgd', (0, 8, 4, 16)):
            zapal(n, kolor_fali(2 * np.pi / om[k]), 1.0)
        wyb.set_alpha(0.0)
        tytul.set_text('Tokeny wpadają do studni: treść patcha + adres ze światła')
        pod.set_text('kolor kuli = średni kolor patcha oka; bez kodowania pozycji tokeny nie miałyby studni (kontrola: 34% sąsiadów zgodnych z siatką)')
        opis.set_text('Token = wynik lodygi (treść) +\nkod pozycji (adres).\nNa starcie adres dominuje:\nkażdy token siedzi w swojej studni.')
    else:
        a_kul = 1.0; I = sw.studnie(g); L = sw.rgb(I)
        for n, k in zip('lpgd', (0, 8, 4, 16)):
            zapal(n, kolor_fali(2 * np.pi / om[k]), 0.9 + 0.1 * np.sin(f * 0.7))
        tytul.set_text('Krok 0: idealna siatka 600 tokenów')
        pod.set_text('pomiar: adres wyjaśnia 99,8% położenia tokenów, 97% najbliższych sąsiadów to sąsiednie patche')
        opis.set_text('Dalej (film 2): uczenie zmienia wagi,\nstudnie zaczynają się przesuwać,\na tokeny drgać.')
    im_oko.set_alpha(a_oko)
    for ln in linie_x + linie_y:
        ln.set_alpha(a_siat * 0.6)
    rgba = np.concatenate([np.clip(L, 0, 1), np.clip(L.max(-1, keepdims=True) * 1.6, 0, 1)], -1)
    im_sw.set_data(rgba)
    kul.set_offsets(pos); halo.set_offsets(pos); kul.set_alpha(a_kul); halo.set_alpha(0.25 * a_kul)
    fig.canvas.draw()
    if PROBA:
        fig.savefig(HERE / f'lasery_1_proba_{f}.png', dpi=60)
    else:
        ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
        if f % 200 == 0:
            log(f'klatka {f}/{N_F}')
if ff:
    ff.stdin.close(); ff.wait()
log('gotowe', N_F, 'klatek')
