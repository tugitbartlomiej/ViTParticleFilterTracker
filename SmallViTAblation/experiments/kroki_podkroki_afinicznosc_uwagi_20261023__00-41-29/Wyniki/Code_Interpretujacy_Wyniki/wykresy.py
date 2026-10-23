"""Wykresy eksperymentu kroki × podkroki (PNG 160 dpi + SVG) z Wyniki/Dane/kroki_podkroki.npz i jakobiany_podkrokow.json.
Uruchomienie: python wykresy.py <Wyniki/Dane> <Wyniki/Wykresy>
Kolory: paleta odniesienia skill dataviz (kategorie w stałej kolejności: blok 1 niebieski, 2 pomarańczowy, 3 morski, 4 żółty, 5 różowy,
6 zielony; sprawdzona validate_palette.js: PASS, ostrzeżenie kontrastu dla morskiego/żółtego/różowego → etykiety bezpośrednie);
wielkości: jeden odcień niebieskiego jasny→ciemny; wielkości ze znakiem: niebieski ↔ czerwony z szarym środkiem."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from scipy.ndimage import median_filter

D, W = Path(sys.argv[1]), Path(sys.argv[2]); W.mkdir(parents=True, exist_ok=True)
d = np.load(D / 'kroki_podkroki.npz'); j = json.load(open(D / 'jakobiany_podkrokow.json', encoding='utf-8'))
kr = d['krok']; ep = kr / 1094.0; K = len(kr)
PK = ['u1', 'm1', 'u2', 'm2', 'u3', 'm3', 'u4', 'm4', 'u5', 'm5', 'u6', 'm6']
PK_OPIS = [f'{"uwaga" if s % 2 == 0 else "MLP"} {s // 2 + 1}' for s in range(12)]
ST = ['wejście'] + PK_OPIS
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
SEQ_KROKI = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('seq', SEQ_KROKI)
DIV = LinearSegmentedColormap.from_list('div', ['#0d366b', '#2a78d6', '#86b6ef', '#cde2fb', '#f0efec', '#f6c9c8', '#ee8c8b', '#e34948', '#8f1f1f'])
ORD7 = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#184f95', '#104281', '#0d366b']        # 7 epok: 0, 1, 5, 10, 20, 40, 60
EP7 = [0, 1094, 5470, 10940, 21880, 43760, 65640]; EP7_N = ['0 ep', '1 ep', '5 ep', '10 ep', '20 ep', '40 ep', '60 ep']
I7 = [int(np.argmin(np.abs(kr - e))) for e in EP7]
plt.rcParams.update({'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'semibold',
                     'axes.labelsize': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False, 'legend.fontsize': 8, 'lines.linewidth': 1.6,
                     'text.color': INK, 'axes.titlecolor': INK, 'figure.dpi': 100, 'svg.fonttype': 'none'})
TICKS = [0, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60]


def pl(v):
    return (f'{v:g}').replace('.', ',')


def krawedzie(x):
    m = 0.5 * (x[1:] + x[:-1]); return np.r_[max(x[0] - (m[0] - x[0]), 0), m, x[-1] + (x[-1] - m[-1])]


XE = krawedzie(ep)


def os_epok(ax, etykieta=True):
    ax.set_xscale('symlog', linthresh=1.0, linscale=1.0)
    ax.set_xlim(0, 61)
    ax.set_xticks(TICKS); ax.set_xticklabels([pl(t) for t in TICKS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if etykieta:
        ax.set_xlabel('epoka uczenia (do 1 liniowo, dalej logarytmicznie)')


def wygladz(y, w=9):
    return median_filter(np.asarray(y, float), size=w, mode='nearest')


def heat(fig, ax, Z, cmap, norm, wiersze, tytul, cbar_label):
    pm = ax.pcolormesh(XE, np.arange(len(wiersze) + 1), Z, cmap=cmap, norm=norm, shading='flat', rasterized=True)
    ax.set_yticks(np.arange(len(wiersze)) + 0.5); ax.set_yticklabels(wiersze); ax.invert_yaxis()
    ax.grid(False); os_epok(ax); ax.set_title(tytul, loc='left')
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.03); cb.set_label(cbar_label); cb.outline.set_visible(False)
    return pm


def etykiety_koncowe(ax, ys, nazwy, kolory, x=None, min_px=11):
    """Etykiety na końcu linii, rozsunięte w pikselach, żeby się nie nakładały."""
    x = 61 if x is None else x
    disp = np.array([ax.transData.transform((x, y))[1] for y in ys], float)
    kol = np.argsort(disp); pos = disp.copy()
    for a, b in zip(kol[:-1], kol[1:]):
        if pos[b] - pos[a] < min_px:
            pos[b] = pos[a] + min_px
    sr_przes = (pos - disp).mean(); pos -= sr_przes if abs(sr_przes) < min_px else 0
    inv = ax.transData.inverted()
    for i, (n, c) in enumerate(zip(nazwy, kolory)):
        yd = inv.transform((ax.transData.transform((x, 1))[0], pos[i]))[1]
        ax.annotate(n, (x, yd), xytext=(4, 0), textcoords='offset points', color=c, fontsize=8, va='center', ha='left', annotation_clip=False)


def zapisz(fig, nazwa):
    fig.savefig(W / f'{nazwa}.png', dpi=160, bbox_inches='tight'); fig.savefig(W / f'{nazwa}.svg', bbox_inches='tight'); plt.close(fig)
    print('zapisano', nazwa, flush=True)


# ------------------------------------------------------------------ F1 rozrzut chmury i składowa promieniowa
fig, axs = plt.subplots(2, 1, figsize=(11, 7.2), constrained_layout=True)
Z = np.log2(d['stos']).T
heat(fig, axs[0], Z, DIV, TwoSlopeNorm(vcenter=0, vmin=-1.3, vmax=1.3), PK_OPIS,
     'A. Rozrzut chmury po podkroku / przed podkrokiem (log₂ r_po/r_przed): uwaga ≈ 1, MLP 1 ×2, MLP 4 rośnie do końca uczenia',
     'log₂(r_po / r_przed)   (0 = bez zmiany, +1 = ×2)')
heat(fig, axs[1], d['rad'].T, DIV, TwoSlopeNorm(vcenter=0, vmin=-0.55, vmax=0.55), PK_OPIS,
     'B. Składowa promieniowa przesunięcia, mediana cos(Δ, x): + na zewnątrz chmury (inflacja), − do środka',
     'mediana cos(Δ, x)')
zapisz(fig, 'F1_rozrzut_i_promien')

# ------------------------------------------------------------------ F2 afiniczność
fig, axs = plt.subplots(2, 1, figsize=(11, 7.2), constrained_layout=True)
heat(fig, axs[0], d['r2cv'].T, SEQ, Normalize(0.6, 1.0), PK_OPIS,
     'A. Jak dobrze przesunięcie cząsteczki jest funkcją afiniczną jej własnego położenia (CV-R², 192D; zero z przetasowania −2…−5)',
     'CV-R² dopasowania Δ ~ M·x + c')
heat(fig, axs[1], d['trans'].T, SEQ, Normalize(0, 1), PK_OPIS,
     'B. Udział wspólnego przesunięcia wszystkich cząsteczek (translacja chmury) w ruchu podkroku, rzut 3D aplikacji',
     '|średnia Δ|² / średnia |Δ|²')
zapisz(fig, 'F2_afinicznosc')

# ------------------------------------------------------------------ F3 uwaga
fig, axs = plt.subplots(3, 1, figsize=(11, 9.6), sharex=True, constrained_layout=True)
neff = d['att_neff'].mean(2); wzb = d['att_bg2tool'].mean(2); dist = d['att_dist'].mean(2)
for ax, Y, tyt, ylab, log in ((axs[0], neff, 'A. Efektywna liczba cząsteczek, których słucha cząsteczka (1/ΣA², mediana po cząsteczkach, średnia po głowach); 600 = uwaga jednostajna = czysta translacja',
                               'efektywne źródła', True),
                              (axs[1], wzb, 'B. Ile razy więcej uwagi niż „po równo” tło kieruje na 3 tokeny narzędzia (wzbogacenie; 1 = brak preferencji)',
                               'wzbogacenie tło → narzędzie', True),
                              (axs[2], dist, 'C. Średnia odległość na siatce, na którą patrzy cząsteczka (łaty); jednostajna 12,9; sąsiedzi ≈ 0 % uwagi przez całe uczenie',
                               'odległość uwagi [łaty]', False)):
    for b in range(6):
        ax.plot(ep, wygladz(Y[:, b]), color=CAT[b], label=f'blok {b + 1}')
    if log:
        ax.set_yscale('log')
    ax.set_title(tyt, loc='left'); ax.set_ylabel(ylab); os_epok(ax, etykieta=ax is axs[2])
    fig.canvas.draw()
    etykiety_koncowe(ax, [wygladz(Y[:, b])[-1] for b in range(6)], [f'blok {b + 1}' for b in range(6)], CAT)
axs[0].axhline(600, color=MUTED, lw=0.8); axs[0].text(0.02, 600, 'jednostajna (600)', color=INK2, fontsize=8, va='bottom')
axs[1].axhline(1, color=MUTED, lw=0.8)
axs[2].axhline(12.9, color=MUTED, lw=0.8); axs[2].text(0.02, 12.95, 'jednostajna (12,9)', color=INK2, fontsize=8, va='bottom')
axs[0].legend(ncol=6, loc='lower left', bbox_to_anchor=(0, 0.0))
fig.text(0.01, -0.01, 'Linie: mediana krocząca z 9 checkpointów. Kolory bloków stałe we wszystkich rysunkach.', color=INK2, fontsize=8)
zapisz(fig, 'F3_uwaga')

# ------------------------------------------------------------------ F4 front rozdziału narzędzie–tło
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={'width_ratios': [1.55, 1]}, constrained_layout=True)
heat(fig, axs[0], d['fisher'].T, SEQ, Normalize(0.5, 3.5), ST,
     'A. Rozdział narzędzie/tło (Fisher, 192D) po 13 stanach strumienia: skok przesuwa się z bloku 3 (1 ep) do bloku 4', 'Fisher: |μ_n − μ_t|² / wariancja wewnątrz klas')
ax = axs[1]
for i, (idx, c, n) in enumerate(zip(I7, ORD7, EP7_N)):
    ax.plot(range(13), d['fisher'][idx], color=c, marker='o', ms=3.5, label=n)
ax.set_xticks(range(13)); ax.set_xticklabels(['wej.'] + PK, rotation=0); ax.set_ylabel('Fisher'); ax.set_xlabel('stan strumienia rezydualnego (podkroki)')
ax.set_title('B. Profil po głębokości w wybranych epokach', loc='left'); ax.legend(ncol=2, loc='upper left', title='epoka', title_fontsize=8)
fig.canvas.draw(); etykiety_koncowe(ax, [d['fisher'][idx][-1] for idx in I7], EP7_N, ORD7, x=12)
zapisz(fig, 'F4_front_rozdzialu')

# ------------------------------------------------------------------ F5 spójność po głębokości i korekta sąsiednich podkroków
fig, axs = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=True, constrained_layout=True)
ax = axs[0, 0]
for k_, (n, c) in enumerate(zip(['wszystkie 12 podkroków', 'same podkroki uwagi', 'same podkroki MLP'], CAT[:3])):
    ax.plot(ep, wygladz(d['bal'][:, k_]), color=c, label=n)
ax.axhline(1, color=MUTED, lw=0.8); ax.text(0.02, 1.02, 'błądzenie losowe (1)', color=INK2, fontsize=8, va='bottom')
ax.set_title('A. Wskaźnik balistyczny po głębokości |ΣΔ|² / Σ|Δ|²: 1 ep ruch prosty, 60 ep znów zygzak', loc='left'); ax.set_ylabel('|ΣΔ|² / Σ|Δ|²'); ax.legend(loc='upper right')
for ax, Y, tyt, nazwy in ((axs[0, 1], d['cos_mm'], 'B. cos(MLP b, MLP b+1): wspólny kierunek kolejnych MLP znika z uczeniem', [f'b{b + 1}→{b + 2}' for b in range(5)]),
                          (axs[1, 0], d['cos_ma'], 'C. cos(MLP b, uwaga b+1): późno ujemny = uwaga następnego bloku cofa MLP', [f'b{b + 1}→{b + 2}' for b in range(5)]),
                          (axs[1, 1], d['cos_am'], 'D. cos(uwaga b, MLP b): blok 4 wzmacnia, bloki 5–6 cofają', [f'blok {b + 1}' for b in range(6)])):
    for b in range(Y.shape[1]):
        ax.plot(ep, wygladz(Y[:, b]), color=CAT[b], label=nazwy[b])
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_title(tyt, loc='left'); ax.set_ylabel('mediana cos po cząsteczkach')
    ax.legend(ncol=3, loc='lower left' if ax is axs[1, 1] else 'upper right')
for ax in axs[1]:
    os_epok(ax)
for ax in axs[0]:
    os_epok(ax, etykieta=False)
fig.text(0.01, -0.01, 'Mediany po 600 cząsteczkach, wygładzone medianą kroczącą z 9 checkpointów.', color=INK2, fontsize=8)
zapisz(fig, 'F5_spojnosc_glebokosci')

# ------------------------------------------------------------------ F6 gładkość po siatce i lasery
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
ax = axs[0]
for idx, c, n in zip(I7, ORD7, EP7_N):
    ax.plot(range(13), d['dir'][idx], color=c, marker='o', ms=3.5, label=n)
ax.set_xticks(range(13)); ax.set_xticklabels(['wej.'] + PK); ax.set_ylabel('energia Dirichleta / (2 · wariancja chmury)')
ax.set_title('A. Gładkość cech po siatce zdjęcia: spadek = wygładzanie; spada w podkrokach MLP, nie uwagi', loc='left'); ax.legend(ncol=2, title='epoka', title_fontsize=8)
ax = axs[1]
for b in range(6):
    ax.plot(ep, wygladz(d['mlp_lit'][:, b]), color=CAT[b], label=f'blok {b + 1}')
ax.set_title('B. Odsetek zapalonych laserów (z > 0,05, mediana po cząsteczkach): kod rzadnie, blok 4 zostaje najgęstszy', loc='left')
ax.set_ylabel('odsetek laserów z z > 0,05'); os_epok(ax); ax.legend(ncol=3, loc='upper right')
fig.canvas.draw(); etykiety_koncowe(ax, [wygladz(d['mlp_lit'][:, b])[-1] for b in range(6)], [f'blok {b + 1}' for b in range(6)], CAT)
zapisz(fig, 'F6_gladkosc_lasery')

# ------------------------------------------------------------------ F7 jakobiany: iloczyn po głębokości
rod = j['rodzaj']; narz_i = [i for i, r in enumerate(rod) if r.startswith('narz')]; tlo_i = [i for i, r in enumerate(rod) if r == 'tło']
A = lambda k_: np.array(j[k_])
fig, axs = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, constrained_layout=True)
for wiersz, (idx, nazwa) in enumerate(((narz_i, 'cząsteczki narzędzia (3)'), (tlo_i, 'cząsteczki tła (3)'))):
    for kol, (klucz, tyt, ylab) in enumerate((('kurczy_prod', 'udział kierunków o σ < 1 iloczynu J_t⋯J_1', 'udział σ < 1'),
                                              ('log_svmax_prod', 'log σ_max iloczynu (największe rozciągnięcie)', 'log σ_max'),
                                              ('log_svmin_prod', 'log σ_min iloczynu (najsilniejsze ściśnięcie)', 'log σ_min'))):
        ax = axs[wiersz, kol]; Y = A(klucz)[:, idx].mean(1)
        for a_, (c, n) in enumerate(zip(ORD7, EP7_N)):
            ax.plot(range(1, 13), Y[a_], color=c, marker='o', ms=3, label=n)
        if klucz == 'kurczy_prod':
            ax.axhline(0.5, color=MUTED, lw=0.8); ax.set_ylim(0.4, 0.6)
        else:
            ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f'{"ABC"[kol] if wiersz == 0 else "DEF"[kol]}. {tyt}\n{nazwa}', loc='left'); ax.set_ylabel(ylab)
        if wiersz == 1:
            ax.set_xticks(range(1, 13)); ax.set_xticklabels(PK); ax.set_xlabel('po podkroku t')
axs[0, 0].legend(ncol=2, title='checkpoint', title_fontsize=8, loc='upper left')
fig.text(0.01, -0.01, 'Jakobiany własne cząsteczki (192 × 192, autograd float64, inne cząsteczki ustalone); iloczyn po kolejnych podkrokach. Siodło 50/50 zostaje, '
         'ale rozciągnięcie i ściśnięcie rosną z uczeniem: mapa ściskająca, nie pułapka.', color=INK2, fontsize=8)
zapisz(fig, 'F7_jakobiany_iloczyn')

# ------------------------------------------------------------------ F8 jakobiany: objętość, nakładanie, barycentryczność
fig, axs = plt.subplots(2, 2, figsize=(12.5, 7.8), constrained_layout=True)
ld = A('logdet')
for ax, idx, nazwa in ((axs[0, 0], narz_i, 'narzędzie'), (axs[0, 1], tlo_i, 'tło')):
    Z = ld[:, idx].mean(1)                                                       # (7 ckpt, 12)
    pm = ax.pcolormesh(np.arange(13), np.arange(8), Z, cmap=DIV, norm=TwoSlopeNorm(vcenter=0, vmin=-10.6, vmax=10.6), shading='flat')
    ax.set_xticks(np.arange(12) + 0.5); ax.set_xticklabels(PK); ax.set_yticks(np.arange(7) + 0.5); ax.set_yticklabels(EP7_N); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    suma = Z.sum(1)
    for a_ in range(7):
        ax.text(12.15, a_ + 0.5, f'Σ {suma[a_]:+.1f}', va='center', fontsize=8, color=INK2)
    ax.set_title(f'{"A" if nazwa == "narzędzie" else "B"}. log|det J| podkroku, {nazwa}: MLP 1 rozdyma objętość, bloki 4–6 ją ściskają; Σ = całe przejście', loc='left')
    cb = fig.colorbar(pm, ax=ax, pad=0.08, fraction=0.03); cb.set_label('log|det J_s|'); cb.outline.set_visible(False)
ax = axs[1, 0]
nk, nkl = A('nakl_blok'), A('nakl_blok_losowe')
for a_, (c, n) in enumerate(zip(ORD7, EP7_N)):
    ax.plot(range(1, 6), nk[a_][narz_i + tlo_i].mean(0), color=c, marker='o', ms=3.5, label=n)
ax.plot(range(1, 6), nkl[:, narz_i + tlo_i].mean((0, 1)), color=MUTED, ls='--', lw=1.2, label='losowe podprzestrzenie')
ax.set_xticks(range(1, 6)); ax.set_xticklabels([f'{b}↔{b + 1}' for b in range(1, 6)]); ax.set_xlabel('para kolejnych bloków'); ax.set_ylim(0.45, 0.6)
ax.set_ylabel('tr(P_b P_b+1) / min(k_b, k_b+1)'); ax.set_title('C. Nakładanie podprzestrzeni kurczących kolejnych bloków:\njak losowe (brak ogniskowania naprzemiennego)', loc='left')
ax.legend(ncol=2, fontsize=7.5, loc='upper left')
ax = axs[1, 1]
zb = A('uw_zamr_inni')
for a_, (c, n) in enumerate(zip(ORD7, EP7_N)):
    ax.plot(range(1, 7), zb[a_][narz_i].mean(0), color=c, marker='o', ms=3.5)
    ax.plot(range(1, 7), zb[a_][tlo_i].mean(0), color=c, marker='s', ms=3.5, ls='--')
ax.set_xticks(range(1, 7)); ax.set_xticklabels([f'blok {b}' for b in range(1, 7)]); ax.set_ylim(0, 1.02)
ax.set_ylabel('‖J przy zamrożonych A‖² / ‖J pełny‖²'); ax.set_title('D. Część barycentryczna (afiniczna w chmurze) jakobianu uwagi wobec innych cząsteczek:\n95–98 % na starcie, 20–70 % po uczeniu (linia ciągła: narzędzie, przerywana: tło)', loc='left')
from matplotlib.lines import Line2D
uchw = [Line2D([], [], color=c, marker='o', ms=3.5, label=n) for c, n in zip(ORD7, EP7_N)]
uchw += [Line2D([], [], color=INK2, ls='-', label='narzędzie'), Line2D([], [], color=INK2, ls='--', label='tło')]
ax.legend(handles=uchw, ncol=3, fontsize=7.5, loc='lower left')
zapisz(fig, 'F8_jakobiany_objetosc_nakladanie')

# ------------------------------------------------------------------ F9 kiedy: fazy w jednym rysunku
fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
norm = lambda y: (y - y.min()) / (y.max() - y.min() + 1e-12)
serie = [('źródła uwagi, blok 5 (log; maleje)', 1 - norm(np.log(wygladz(neff[:, 4]))), CAT[4]),
         ('rozrzut MLP 1 (r_po/r_przed; rośnie)', norm(wygladz(d['stos'][:, 1])), CAT[0]),
         ('Fisher na wyjściu (rośnie)', norm(wygladz(d['fisher'][:, 12])), CAT[2]),
         ('pewność detektora', norm(wygladz(d['pewnosc'])), CAT[1]),
         ('rozrzut MLP 4 (rośnie)', norm(wygladz(d['stos'][:, 7])), CAT[3]),
         ('wskaźnik balistyczny (rośnie i wraca)', norm(wygladz(d['bal'][:, 0])), CAT[5])]
for n, y, c in serie:
    ax.plot(ep, y, color=c, label=n)
os_epok(ax); ax.set_ylabel('wartość przeskalowana do [0, 1]'); ax.set_ylim(-0.03, 1.08)
ax.set_title('Kolejność zdarzeń w uczeniu (każda krzywa przeskalowana do swojego zakresu): uwaga zawęża się w ułamku epoki, MLP 1 rozpycha do 10 ep, blok 4 i rozdział rosną do końca', loc='left')
ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=8)
zapisz(fig, 'F9_kolejnosc_zdarzen')
print('gotowe')
