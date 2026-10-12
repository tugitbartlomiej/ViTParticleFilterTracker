"""Rysunki do raportu 27: architektura, potok pomiaru, ruch latentu, autokorelacja, wyniki równań, paczki, sufit.

Wszystkie liczby czytane z plików wyników (ścieżki niżej), nic nie jest przepisywane ręcznie.
Uruchomienie: py -3.11 wykresy.py   ->   rys/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
RYS = HERE / 'rys'; RYS.mkdir(exist_ok=True)
EXP = HERE.parents[1] / 'experiments'
SUF = EXP / 'maly_vit_sufit_przewidywalnosci_100' / 'wyniki'
PAC = EXP / 'maly_vit_gesta_historia_paczka' / 'wyniki'

# Paleta referencyjna (tryb jasny), stała kolejność slotów
NIEB, POM, AQUA, ZOLT, MAG, ZIEL, FIOL, CZER = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'
INK, INK2, MUTED, GRID, TLO, OS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#ffffff', '#c3c2b7'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': OS, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': TLO, 'figure.facecolor': TLO,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200})


def zapisz(fig, nazwa):
    fig.savefig(RYS / nazwa, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('zapisano', RYS / nazwa)


def blok(ax, x, y, w, h, tekst, kolor='#eef4fc', ramka=NIEB, rozmiar=9, gruby=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=kolor, ec=ramka, lw=1.4))
    ax.text(x + w / 2, y + h / 2, tekst, ha='center', va='center', fontsize=rozmiar, color=INK,
            fontweight='bold' if gruby else 'normal', linespacing=1.3)


def strzalka(ax, x0, y0, x1, y1, kolor=INK2, styl='-|>', lw=1.4, ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=styl, mutation_scale=12, color=kolor, lw=lw,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- 1. architektura małego ViT
def architektura():
    fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis('off')
    y0 = 4.3; h = 1.15
    kol = [(0.1, 1.45, 'Obraz\n480 × 320 px\n3 kanały', '#f4f4f2', MUTED),
           (1.85, 1.75, 'Łodyga\n4 konwolucje 3×3,\nkrok 2, BatchNorm,\nGELU', '#eef4fc', NIEB),
           (3.9, 1.6, 'Siatka 30 × 20\n= 600 tokenów\npo 192 liczby\n+ pozycje sin-cos', '#eef4fc', NIEB),
           (5.8, 1.75, '6 bloków\ntransformera\n(uwaga: 3 głowy,\nMLP 192→768→192)', '#eef4fc', NIEB),
           (7.85, 1.3, 'LayerNorm', '#eef4fc', NIEB),
           (9.45, 1.45, 'Głowa\nna każdy token:\nśrodek narzędzia\n+ 4 odległości\ndo krawędzi ramki', '#fdf0ea', POM)]
    for x, w, t, c, r in kol:
        blok(ax, x, y0, w, h + (0.25 if 'Głowa' in t else 0), t, c, r, 8.6)
    for (x, w, *_), (x2, *_r) in zip(kol[:-1], kol[1:]):
        strzalka(ax, x + w, y0 + h / 2, x2, y0 + h / 2)
    # rozwinięcie ostatniego bloku
    ax.add_patch(FancyBboxPatch((1.0, 0.25), 9.0, 2.95, boxstyle='round,pad=0.02,rounding_size=0.1', fc='#fafaf8', ec=OS, lw=1, ls='--'))
    ax.text(1.15, 3.0, 'Wnętrze ostatniego (szóstego) bloku, dla każdego z 600 tokenów osobno', fontsize=9, color=INK2, va='top')
    strzalka(ax, 6.7, y0, 6.0, 3.2, kolor=MUTED, ls='--')
    yb = 1.2; hb = 0.9
    wew = [(1.25, 0.95, 'LayerNorm'), (2.4, 1.25, 'uwaga\n(tokeny\nwymieniają\ninformację)'), (3.85, 0.55, '+'),
           (4.6, 0.95, 'LayerNorm'), (5.75, 1.15, 'Liniowa\n192 → 768'), (7.1, 0.95, 'GELU'), (8.25, 1.15, 'Liniowa\n768 → 192')]
    for x, w, t in wew:
        c, r = ('#e8f7f0', AQUA) if t == 'GELU' else ('#ffffff', OS)
        blok(ax, x, yb, w, hb, t, c, r, 8.2, gruby=(t == 'GELU'))
    for (x, w, _), (x2, *_r) in zip(wew[:-1], wew[1:]):
        strzalka(ax, x + w, yb + hb / 2, x2, yb + hb / 2)
    ax.annotate('LATENT: 768 liczb na token\n(które neurony się „zapalają”)\n12 obrazów × 600 tokenów × 768',
                xy=(7.575, yb), xytext=(7.575, 0.35), ha='center', va='bottom', fontsize=8.6, color=ZIEL,
                fontweight='bold', arrowprops=dict(arrowstyle='-|>', color=AQUA, lw=1.4))
    ax.text(5.5, 6.05, 'Mały ViT (2,93 mln parametrów) uczony od zera do wykrywania narzędzia chirurgicznego',
            ha='center', fontsize=10.5, color=INK, fontweight='bold')
    zapisz(fig, '01_architektura.png')


# ---------------------------------------------------------------- 2. potok pomiaru
def potok():
    fig, ax = plt.subplots(figsize=(11, 6.0)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.0); ax.axis('off')
    y = 3.4; h = 1.4
    b = [(0.1, 1.9, 'Paczka 32 obrazów\nlosowana z 1500\n(odbicie, jasność)', '#f4f4f2', MUTED),
         (2.3, 2.15, 'Krok AdamW\ngradient paczki g\npęd m (pamięta\nok. 10 kroków), skala v', '#fdf0ea', POM),
         (4.75, 1.75, 'Nowe wagi\n(i statystyki\nBatchNorm)', '#eef4fc', NIEB),
         (6.8, 1.95, 'Pomiar: 12 stałych\nobrazów w trybie eval\n(bez uczenia)', '#eef4fc', NIEB),
         (9.05, 1.85, 'Latent po kroku t\n12 × 600 × 768\nzapis co 1 krok', '#e8f7f0', AQUA)]
    for x, w, t, c, r in b:
        blok(ax, x, y, w, h, t, c, r, 8.6)
    for (x, w, *_), (x2, *_r) in zip(b[:-1], b[1:]):
        strzalka(ax, x + w, y + h / 2, x2, y + h / 2)
    ax.add_patch(FancyArrowPatch((9.95, y + h), (3.35, y + h), connectionstyle='arc3,rad=0.18', arrowstyle='-|>',
                                 mutation_scale=12, color=MUTED, lw=1.2, ls='--'))
    ax.text(6.6, 5.85, 'powtórz 1000 razy: kroki uczenia 0 … 1000', ha='center', va='top', fontsize=9, color=INK2)
    # siły
    blok(ax, 2.35, 0.55, 3.7, 1.7, 'Siły kroku (nagranie z docs/23)\nprzesunięcie latentu, gdy wagi przesunąć\no jedną część kroku AdamW:\nS_pęd (z pamięci m), S_nowa (z paczki g),\nS_zanik (zanik wag)', '#fff7e6', ZOLT, 8.3)
    strzalka(ax, 3.35, y, 3.6, 2.25, kolor=ZOLT)
    blok(ax, 6.8, 0.55, 4.05, 1.7, 'Analiza (bez treningu)\n• rozkład ruchu: powolny dryf + drgania\n• sufit przewidywalności\n• równania Langevina: OU, jądro pamięci,\n  pęd optymalizatora; test na 1…100 kroków', '#f4f4f2', MUTED, 8.3)
    strzalka(ax, 9.95, y, 9.4, 2.25, kolor=AQUA)
    strzalka(ax, 6.05, 1.4, 6.8, 1.4, kolor=ZOLT)
    ax.text(5.5, 5.6 - 0.02, '', ha='center')
    zapisz(fig, '02_potok_pomiaru.png')


# ---------------------------------------------------------------- 3. ruch latentu: dryf i drgania
def ruch():
    sila = np.load(SUF / 'prognoza_900' / 'SILA_TOZSAMOSCI.npy')       # (1001, 12)
    x = sila[:, 7]
    k = 21; M = np.convolve(x, np.ones(k) / k, mode='same'); M[:k // 2] = np.nan; M[-(k // 2):] = np.nan
    fig, axs = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.12))
    ax = axs[0]
    ax.plot(np.arange(1001), x, color=INK, lw=0.9, label='prawdziwy przebieg (co 1 krok)')
    ax.plot(np.arange(1001), M, color=NIEB, lw=2.2, label='gładka ścieżka (średnia z 21 kroków)')
    ax.axvspan(0, 300, color='#f4f4f2', zorder=0); ax.text(150, np.nanmin(x) * 0.95, 'rozgrzewka lr\n(kroki 0–300)', ha='center', va='bottom', fontsize=8.5, color=INK2)
    ax.set_ylabel('siła tożsamości narzędzia'); ax.legend(loc='lower right', frameon=False, fontsize=8.8)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_title('Ruch latentu w uczeniu: powolny dryf plus drgania (obraz 7, rzut tokenów narzędzia na kierunek LDA)',
                 loc='left', fontsize=10.5, color=INK)
    ax = axs[1]
    ax.plot(np.arange(1001), x - M, color=POM, lw=0.8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel('drgania'); ax.set_xlabel('krok uczenia'); ax.grid(axis='y', color=GRID, lw=0.6)
    zapisz(fig, '03_ruch_dryf_drgania.png')


# ---------------------------------------------------------------- 4. autokorelacja przyrostów
def autokorelacja():
    a = json.loads((SUF / 'autokorelacja' / 'ACF.json').read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    L = 31
    for (k, v), c, lab in zip(a.items(), (POM, NIEB), ('kroki 0–300 (rozgrzewka)', 'kroki 300–1000')):
        ax.plot(np.arange(L), v['acf'][:L], 'o-', color=c, ms=4, lw=1.8, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.annotate('po 1 kroku: ruch podobny\ndo poprzedniego (pęd AdamW)', xy=(1, a['300-1000']['acf'][1]), xytext=(5, 0.62),
                fontsize=8.6, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED))
    ax.annotate('po 3–5 krokach: odbicie\nw przeciwną stronę', xy=(4, a['300-1000']['acf'][4]), xytext=(9, -0.42),
                fontsize=8.6, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED))
    ax.text(17, 0.15, 'od około 10 kroków: brak związku z przeszłością', fontsize=8.6, color=INK2)
    ax.set_xlabel('opóźnienie (kroki uczenia)'); ax.set_ylabel('podobieństwo przyrostów\n(autokorelacja)')
    ax.set_ylim(-0.55, 1.05); ax.grid(axis='y', color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_title('Jak długo ruch latentu „pamięta” sam siebie (tokeny narzędzia, 12 obrazów, paczka 32)', loc='left', fontsize=10.5, color=INK)
    zapisz(fig, '04_autokorelacja.png')


# ---------------------------------------------------------------- 5. równania na 1..10 kroków
def rownania():
    m = json.loads((SUF / 'rownanie_z_pedem' / 'ROWNANIE_Z_PEDEM.json').read_text(encoding='utf-8'))
    h = np.arange(1, 11)
    serie = [('pelna_sila_znana_iter', 'granica: znana pełna siła każdego przyszłego kroku', ZIEL, '--'),
             ('paczki_znane_iter', 'granica: znane przyszłe paczki, pęd liczony liniowo', ZOLT, '--'),
             ('hist_ped_wprost', 'PROGNOZA: historia latentu + pęd optymalizatora', NIEB, '-'),
             ('hist_wprost', 'PROGNOZA: sama historia latentu (jądro pamięci + OU)', POM, '-'),
             ('hist_ped_iter', 'to samo równanie iterowane krok po kroku', MUTED, ':')]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for key, lab, c, ls in serie:
        y47, y48 = np.array(m['s47_test'][key]), np.array(m['s48_przeniesienie'][key])
        ax.plot(h, y47, ls=ls, color=c, lw=2.2, marker='o', ms=4, label=lab)
        ax.plot(h, y48, ls=ls, color=c, lw=1, alpha=0.5)
    ax.axhline(0.85, color=CZER, lw=1, ls=':'); ax.text(10.25, 0.86, 'cel 85 %', color=CZER, fontsize=8.6, va='bottom')
    ax.axhline(0, color=INK2, lw=0.8); ax.text(10.25, 0.01, 'bez zmian', color=INK2, fontsize=8.6, va='bottom')
    ax.set_xticks(h); ax.set_xlim(0.7, 11.3); ax.set_ylim(-0.3, 1.04)
    ax.set_xlabel('ile kroków uczenia naprzód'); ax.set_ylabel('o ile mniejszy błąd niż „bez zmian”')
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.set_title('Równanie Langevina dla latentu: prognoza i granice (ziarno 47; cienka linia: ziarno 48 bez ponownego dopasowania)',
                 loc='left', fontsize=10, color=INK)
    zapisz(fig, '05_rownania_1_10.png')


# ---------------------------------------------------------------- 6. paczki
def paczki():
    p = {}
    for f in (PAC / 'porownanie' / 'POROWNANIE.json', PAC / 'porownanie_p128' / 'POROWNANIE.json'):
        p.update(json.loads(f.read_text(encoding='utf-8'))['paczki'])
    pk = sorted(p, key=int)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.28))
    ax = axs[0]
    grupy = [('h = 1', 0), ('h = 10', 9), ('h = 100', 10)]
    szer = 0.25
    for i, (k, c) in enumerate(zip(pk, (NIEB, POM, AQUA))):
        w = p[k]['wszystkie']['rownanie_J5']
        xs = np.arange(len(grupy)) + (i - 1) * szer
        ys = [w[j] for _, j in grupy]
        ax.bar(xs, ys, szer * 0.92, color=c, label=f'paczka {k}')
        for xx, yy in zip(xs, ys):
            ax.text(xx, yy + 0.015, f'{yy:.2f}'.replace('.', ','), ha='center', fontsize=8, color=INK2)
    ax.axhline(0.85, color=CZER, lw=1, ls=':'); ax.text(-0.4, 0.86, 'cel 85 %', color=CZER, fontsize=8.5, ha='left')
    ax.set_xticks(range(len(grupy))); ax.set_xticklabels([g for g, _ in grupy]); ax.set_ylim(0, 0.95)
    ax.set_ylabel('o ile mniejszy błąd niż „bez zmian”'); ax.legend(frameon=False, fontsize=8.8, ncol=1, loc='upper right', bbox_to_anchor=(1.0, 0.82))
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_title('Prognoza z historii przy trzech paczkach', loc='left', fontsize=10.5, color=INK)
    ax = axs[1]
    lagi = ['1', '2', '3', '5', '10']
    for k, c in zip(pk, (NIEB, POM, AQUA)):
        a = p[k]['wszystkie']['autokorelacja_przyrostow']
        ax.plot([int(l) for l in lagi], [a[l] for l in lagi], 'o-', color=c, lw=2, ms=4, label=f'paczka {k}')
    ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel('opóźnienie (kroki)'); ax.set_ylabel('autokorelacja przyrostów')
    ax.grid(axis='y', color=GRID, lw=0.6); ax.legend(frameon=False, fontsize=8.5)
    ax.set_title('Większa paczka: ruch mniej gładki', loc='left', fontsize=10.5, color=INK)
    zapisz(fig, '06_paczki.png')


# ---------------------------------------------------------------- 7. sufit i osiągalne, 100 kroków
def sufit():
    m = json.loads((SUF / 'pelny' / 'METRYKI.json').read_text(encoding='utf-8'))
    q = m['wielkosci']['tok_narz']; ok = m['meta']['okresy']
    etyk = ['0–100', '100–300', '300–600', '600–1000']
    serie = [('osiągnięte: model liniowy z historii', [q['liniowy']['h100'][o]['liniowy_krzyzowo_mse'] if q['liniowy']['h100'][o] else np.nan for o in ok], NIEB),
             ('podłoga: samo odszumienie startu', [q['w21_h100'][o]['podloga_drgania_mse'] for o in ok], MUTED),
             ('sufit (wygładzanie 51)', [q['w51_h100'][o]['sufit_mse'] for o in ok], POM),
             ('sufit (wygładzanie 21)', [q['w21_h100'][o]['sufit_mse'] for o in ok], AQUA)]
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    szer = 0.2
    for i, (lab, ys, c) in enumerate(serie):
        xs = np.arange(len(ok)) + (i - 1.5) * szer
        ax.bar(xs, np.nan_to_num(ys), szer * 0.92, color=c, label=lab)
        for xx, yy in zip(xs, ys):
            ax.text(xx, (0 if np.isnan(yy) else yy) + 0.015, 'brak' if np.isnan(yy) else f'{yy:.2f}'.replace('.', ','),
                    ha='center', fontsize=7.6, color=INK2)
    ax.axhline(0.85, color=CZER, lw=1, ls=':')
    ax.set_xticks(range(len(ok))); ax.set_xticklabels([f'start w krokach {e}' for e in etyk])
    ax.set_ylim(0, 1.08); ax.set_ylabel('o ile mniejszy błąd niż „bez zmian”'); ax.grid(axis='y', color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8.3, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.set_title('Prognoza na 100 kroków: co osiągnięto, a co pozwala wyrocznia (tokeny narzędzia, 768 liczb)',
                 loc='left', fontsize=10.5, color=INK)
    zapisz(fig, '07_sufit_100.png')


# ---------------------------------------------------------------- 8. wzory
def wzor(nazwa, tex, szer=7.5, wys=0.62, rozmiar=15):
    fig = plt.figure(figsize=(szer, wys)); fig.text(0.01, 0.5, tex, fontsize=rozmiar, va='center', color=INK)
    zapisz(fig, nazwa)


def wzory():
    wzor('w_umiejetnosc.png', r'$\mathrm{wynik}(h) = 1 - \dfrac{\sum\ \|\hat x_{t+h} - x_{t+h}\|^2}{\sum\ \|x_t - x_{t+h}\|^2}$', 7.5, 0.95)
    wzor('w_sufit.png', r'$S_{\max} \approx 1 - \dfrac{\sigma^2}{D^2 + 2\sigma^2}$', 5, 0.95)
    wzor('w_ou.png', r'$\hat x_{t+h} = \mu_t + (x_t - \mu_t)\, e^{-\kappa h}, \qquad \mu_t = \frac{1}{k}\sum_{j=0}^{k-1} x_{t-j}$', 8.5, 0.95)
    wzor('w_jadro.png', r'$\Delta x_{t+1} = -\kappa\,(x_t - \mu_t) + \sum_{j=0}^{J-1} K_j\, \Delta x_{t-j} + a\, S^{\mathrm{pęd}}_t + b\, S^{\mathrm{zanik}}_t + \mathrm{szum}$', 10, 0.95)
    wzor('w_pelna.png', r'$\Delta x_{t+1} = (\ldots\ \mathrm{jak\ wyżej}) + c\, S^{\mathrm{krok}}_{t}, \qquad S^{\mathrm{krok}}_t = z(\theta_t + d_t) - z(\theta_t)$', 10, 0.8)
    wzor('w_ped.png', r'$m_t = \beta_1\, m_{t-1} + (1-\beta_1)\, g_t, \qquad \beta_1 = 0{,}9\ \Rightarrow\ \mathrm{pamięć} \approx 10\ \mathrm{kroków}$', 9, 0.8)


if __name__ == '__main__':
    architektura(); potok(); ruch(); autokorelacja(); rownania(); paczki(); sufit(); wzory()
