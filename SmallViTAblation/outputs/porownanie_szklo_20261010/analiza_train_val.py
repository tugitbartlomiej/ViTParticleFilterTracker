"""Zdjęcia uczące kontra walidacyjne: czy cząsteczki (tokeny) zdjęć, których sieć nie widziała, poruszają się inaczej?

Dane: pomiar z Edena (pomiar_eden.py): oś czasu aplikacji z 3000 zdjęć TRAIN + 1000 VALID (PCA3 + norma 768D, kroki co 10
do 1000, co 100 do 22 974, końce epok) oraz PCA32 dla 100 + 100 zdjęć (pca32_podzbior.npz) w stałej bazie z kroku 0.

Liczone osobno dla obu grup:
  1. odciski szkła z PCA32 we wczesnej fazie (kroki 300–1000, rekord = 10 kroków): względem populacji (S0) i względem
     klatki (S1): MSD, α, rozmiar klatki, antykorelacja c, α₂, Fs, ucieczki;
  2. energia drgań (temperatura) i udział pola kolektywnego zdjęcia w czasie (co 10, potem co 100 kroków);
  3. trwałość sąsiedztwa (6 sąsiadów w latencie z kroku 300) w czasie;
  4. późna faza (kroki ≥ 1000, co 100): MSD(τ) i starzenie MSD(τ; t_w) względem klatki;
  5. z pełnych klatek PCA3 (4000 zdjęć): ruchliwość zdjęcia (średnie |Δxyz| między zapisami), promień chmury zdjęcia,
     norma 768D od kroku 0; rozkłady TRAIN/VALID w czasie i związek ruchliwości późnej fazy z IoU zdjęcia (Spearman).
Wynik: TRAIN_VAL<sufiks>.json, train_val<sufiks>.png (domyślnie obok skryptu).
Uruchomienie: F:/Python/3.13/python.exe analiza_train_val.py [folder osi] [sufiks wyników] [katalog wyników]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odciski as O  # noqa: E402

OS = Path(sys.argv[1] if len(sys.argv) > 1 else
          r'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/ruch_czasteczek_gui/20261010T_natural_eden_pelny_3000tr_1000val')
SUF = sys.argv[2] if len(sys.argv) > 2 else ''
WY = sys.argv[3] if len(sys.argv) > 3 else HERE
TOK = 600
Q16 = 65535.0
t0 = time.time()


def log(*a):
    print(*a, f'[{time.time() - t0:.0f} s]', flush=True)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


m = json.load(open(OS / 'manifest.json', encoding='utf-8'))
steps = np.array(m['steps'])
images = m['images']
split = np.array([1 if im['split'] == 'VALID' else 0 for im in images])          # 0 TRAIN, 1 VALID
NI = len(images)
metr = np.load(OS / 'metrics.npy')                                                   # (NI, T, 5): conf, iou, delta_l2, krok, both90
GRUPY = ('TRAIN', 'VALID')
wyn = dict(zrodlo=str(OS), kroki=[int(s) for s in steps], n_train=int((split == 0).sum()), n_val=int((split == 1).sum()), grupy={})
log('oś', OS.name, 'zdjęć', NI, 'kroków', len(steps))

# ---------------------------------------------------------------- 1–4: PCA32 podzbiór
z = np.load(OS / 'pca32_podzbior.npz')
P32 = z['pca32'].astype(np.float32)                                                  # (T, n, 600, 32)
kroki32 = z['kroki']
split32 = np.asarray(z['split'])
T32, n32 = P32.shape[:2]
assert (kroki32 == steps).all()
P = P32.reshape(T32, n32 * TOK, 32)
kl = np.repeat(split32, TOK)
del P32
dense = steps <= 1000
rek_d = (steps[dense] // 10).astype(int)                                             # rekord = 10 kroków
late = steps >= 1000
rek_l = (steps[late] // 100).astype(int)                                             # rekord = 100 kroków
for g, nazwa in enumerate(GRUPY):
    sel = kl == g
    Pg = P[:, sel]
    n_img = int(sel.sum() // TOK)
    ref = int(np.argmin(abs(steps - 300)))
    N6, dnn, dsh = O.knn_w_blokach(Pg[ref], TOK, 6)
    d_nn, d_shell = float(np.median(dnn)), float(np.median(dsh))
    out = dict(n_zdjec=n_img, d_nn_300=d_nn, d_shell_300=d_shell)
    # S0 i S1 we wczesnej fazie (rekordy co 10 kroków, okno 30–100 = kroki 300–1000)
    Pd = Pg[dense]
    P1 = np.empty_like(Pd)
    Z300 = Pd[list(rek_d).index(30)]
    for t in range(len(Pd)):
        P1[t] = Pd[t] - (Pd[t][N6] - Z300[N6]).mean(1)
    taus = np.arange(1, 60)
    for etap, Px in (('S0', Pd), ('S1', P1)):
        w = O.odciski(Px, rek_d, (30, 100), d_shell, d_nn, taus=taus, n_start=30, odejmij_pole=True, tw_lista=(30, 45, 60, 75),
                      krotkie_tw=(30, 50, 70, 90), nazwa=f'{nazwa} {etap} (rekord = 10 kroków)')
        w['karta'], w['spelnione'] = O.karta(w, None)
        out[etap] = w
    # energia drgań (temperatura) i udział pola kolektywnego zdjęcia w czasie; CR i względem populacji
    E_pop, E_cr, udz_pole, udz_pole_obraz, kr_E = [], [], [], [], []
    for t in range(1, T32):
        d = Pg[t] - Pg[t - 1]
        dk = (steps[t] - steps[t - 1])
        f = d.mean(0)
        e = (d ** 2).sum(1)
        dw = d - f
        dcr = d - d[N6].mean(1)
        fo = d.reshape(n_img, TOK, 32).mean(1, keepdims=True)
        E_pop.append(float((dw ** 2).sum(1).mean()) / dk ** 2)
        E_cr.append(float((dcr ** 2).sum(1).mean()) / dk ** 2)
        udz_pole.append(float((f ** 2).sum() / e.mean()))
        udz_pole_obraz.append(float((fo ** 2).sum(2).mean() / e.mean()))
        kr_E.append(int(steps[t]))
    out['energia'] = dict(krok=kr_E, E_pop_na_krok2=E_pop, E_klatka_na_krok2=E_cr, udzial_pola_globalnego=udz_pole, udzial_pola_zdjecia=udz_pole_obraz)
    # trwałość sąsiedztwa
    out['trwalosc'] = {}
    for t in range(T32):
        if steps[t] in (310, 400, 600, 1000, 2000, 5000, 10000, 15000, 22974, 32820, 43760, 54700, 65640) or t == T32 - 1:
            N6t = O.knn_w_blokach(Pg[t], TOK, 6)[0]
            out['trwalosc'][int(steps[t])] = float(np.mean([len(set(N6[i]) & set(N6t[i])) / 6 for i in range(len(N6))]))
    # późna faza: MSD(τ) i starzenie względem klatki (rekord = 100 kroków); sąsiedzi z pierwszego rekordu fazy
    Pl = Pg[late]
    N6l = O.knn_w_blokach(Pl[0], TOK, 6)[0]
    P1l = np.empty_like(Pl)
    for t in range(len(Pl)):
        P1l[t] = Pl[t] - (Pl[t][N6l] - Pl[0][N6l]).mean(1)
    tl = np.unique(np.round(np.logspace(0, np.log10(len(rek_l) * 0.6), 18)).astype(int))
    w = O.odciski(P1l, rek_l, (int(rek_l[0]), int(rek_l[-1])), d_shell, d_nn, taus=tl, n_start=24, odejmij_pole=True,
                  tw_lista=tuple(int(x) for x in (10, 30, 50, 100, 150, 200, 300, 400, 500, 600) if x < rek_l[-1] - 10),
                  krotkie_tw=tuple(int(x) for x in (10, 60, 110, 160, 210, 310, 410, 510, 610) if x < rek_l[-1] - 2),
                  nazwa=f'{nazwa} późna faza S1 (rekord = 100 kroków)')
    out['pozna_S1'] = w
    wyn['grupy'][nazwa] = out
    log(nazwa, 'S0', out['S0']['spelnione'], 'S1', out['S1']['spelnione'], 'Lindemann S1', round(out['S1']['lindemann_shell'], 3), 'c', round(out['S1']['c_plateau'], 3),
        'E_klatka(300→1000) śr', round(float(np.mean([e for k, e in zip(kr_E, E_cr) if 300 <= k <= 1000])), 5), 'trwałość', {k: round(v, 2) for k, v in out['trwalosc'].items()})

# ---------------------------------------------------------------- 5: pełne klatki PCA3 (wszystkie zdjęcia)
def klatka(step):
    f = m['files'][str(int(step))]
    q = np.frombuffer((OS / f['name']).read_bytes(), dtype='<u2').reshape(-1, 4).astype(np.float32) / Q16
    lo, span, nm = np.array(f['quant']['lo'], np.float32), np.array(f['quant']['span'], np.float32), np.float32(f['quant']['norm_max'])
    return np.concatenate([lo + q[:, :3] * span, (q[:, 3:] * nm)], 1)


pred = None
ruch, promien, norma_z, kr_r = [], [], [], []
KLATKI_SA = all((OS / m['files'][str(int(s))]['name']).is_file() for s in steps)
if not KLATKI_SA:
    log('brak kompletu klatek q16: sekcja 5 (pełne klatki) pominięta, wynik częściowy')
    json.dump(wyn, open(os.path.join(WY, f'TRAIN_VAL{SUF}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    sys.exit(0)
for i, s in enumerate(steps):
    fr = klatka(s).reshape(NI, TOK, 4)
    promien.append(np.linalg.norm(fr[:, :, :3] - fr[:, :, :3].mean(1, keepdims=True), axis=2).mean(1))
    norma_z.append(fr[:, :, 3].mean(1))
    if pred is not None:
        d = fr[:, :, :3] - pred
        dk = steps[i] - steps[i - 1]
        ruch.append(np.linalg.norm(d - d.mean(1, keepdims=True), axis=2).mean(1) / dk)     # ruch tokenów względem zdjęcia na krok
        kr_r.append(int(s))
    pred = fr[:, :, :3]
    if i % 40 == 0:
        log('klatka', int(s), f'{i + 1}/{len(steps)}')
ruch, promien, norma_z = np.array(ruch), np.array(promien), np.array(norma_z)      # (T-1, NI), (T, NI), (T, NI)
kr_r = np.array(kr_r)
pozne = kr_r >= 10000
iou_konc = metr[:, -1, 1]
conf_konc = metr[:, -1, 0]
ruch_pozny = ruch[pozne].mean(0)
pelne = {}
for g, nazwa in enumerate(GRUPY):
    s = split == g
    pelne[nazwa] = dict(
        ruch_mediana=[float(np.median(ruch[t, s])) for t in range(len(kr_r))], ruch_q25=[float(np.quantile(ruch[t, s], .25)) for t in range(len(kr_r))],
        ruch_q75=[float(np.quantile(ruch[t, s], .75)) for t in range(len(kr_r))],
        promien_mediana=[float(np.median(promien[t, s])) for t in range(len(steps))], norma_mediana=[float(np.median(norma_z[t, s])) for t in range(len(steps))],
        iou_srednie=[float(np.nanmean(metr[s, t, 1])) for t in range(len(steps))], conf_srednie=[float(np.nanmean(metr[s, t, 0])) for t in range(len(steps))],
        ruch_pozny_mediana=float(np.median(ruch_pozny[s])), ruch_pozny_q25=float(np.quantile(ruch_pozny[s], .25)), ruch_pozny_q75=float(np.quantile(ruch_pozny[s], .75)),
        spearman_ruch_pozny_vs_iou=spearman(ruch_pozny[s], iou_konc[s]), spearman_ruch_pozny_vs_conf=spearman(ruch_pozny[s], conf_konc[s]),
        spearman_norma_vs_iou=spearman(norma_z[-1, s], iou_konc[s]), iou_koncowe_mediana=float(np.median(iou_konc[s])))
wyn['pelne_klatki'] = dict(krok_ruchu=[int(k) for k in kr_r], grupy=pelne,
                           stosunek_val_do_train=dict(ruch_pozny=float(np.median(ruch_pozny[split == 1]) / np.median(ruch_pozny[split == 0])),
                                                      promien_koncowy=float(np.median(promien[-1, split == 1]) / np.median(promien[-1, split == 0])),
                                                      norma_koncowa=float(np.median(norma_z[-1, split == 1]) / np.median(norma_z[-1, split == 0]))))
log('pełne klatki: ruch późny VAL/TRAIN', round(wyn['pelne_klatki']['stosunek_val_do_train']['ruch_pozny'], 3),
    'Spearman ruch↔IoU', {g: round(pelne[g]['spearman_ruch_pozny_vs_iou'], 3) for g in GRUPY})
json.dump(wyn, open(os.path.join(WY, f'TRAIN_VAL{SUF}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- rysunek
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Segoe UI', 'font.size': 10, 'axes.facecolor': '#10171c', 'figure.facecolor': '#10171c',
                     'axes.edgecolor': '#8a8f99', 'axes.labelcolor': '#e8e8e8', 'xtick.color': '#cfd3d8', 'ytick.color': '#cfd3d8',
                     'text.color': '#e8e8e8', 'grid.color': '#2a333a', 'legend.facecolor': '#161e24', 'legend.edgecolor': '#2a333a'})
KOL = {'TRAIN': '#ffd166', 'VALID': '#4a8cff'}
fig, ax = plt.subplots(2, 3, figsize=(19, 11), dpi=100)
for nazwa in GRUPY:
    g = wyn['grupy'][nazwa]
    for etap, ls in (('S0', ':'), ('S1', '-')):
        w = g[etap]
        ax[0, 0].loglog(np.array(w['taus']) * 10, w['msd'], ls, color=KOL[nazwa], lw=2 if etap == 'S1' else 1.2, label=f"{nazwa} {etap}")
    e = g['energia']
    ax[0, 1].loglog(e['krok'], e['E_klatka_na_krok2'], color=KOL[nazwa], lw=1.6, label=f'{nazwa} energia w klatce / krok²')
    ax[0, 1].loglog(e['krok'], np.array(e['udzial_pola_zdjecia']) * 1e-3, '--', color=KOL[nazwa], lw=1.2, label=f'{nazwa} udział pola zdjęcia ×1e-3')
    tr = g['trwalosc']
    ax[0, 2].semilogx([int(k) for k in tr], list(tr.values()), 'o-', color=KOL[nazwa], lw=1.6, label=f'{nazwa} zachowani sąsiedzi (z kroku 300)')
    w = g['pozna_S1']
    pts = sorted((tw, v[10]) for tw, v in w['starzenie'].items() if 10 in v)
    if pts:
        ax[1, 0].loglog([p[0] * 100 for p in pts], [p[1] for p in pts], 'o-', color=KOL[nazwa], lw=1.6, label=f"{nazwa} MSD(τ=1000 kroków; t_w) względem klatki" + (f", μ={w['mu'][10]:.2f}" if 10 in w['mu'] else ''))
    pk = wyn['pelne_klatki']['grupy'][nazwa]
    kr = wyn['pelne_klatki']['krok_ruchu']
    ax[1, 1].semilogx(kr, pk['ruch_mediana'], color=KOL[nazwa], lw=2, label=f'{nazwa} ruch tokenów względem zdjęcia / krok (mediana)')
    ax[1, 1].fill_between(kr, pk['ruch_q25'], pk['ruch_q75'], color=KOL[nazwa], alpha=0.15)
    ax[1, 2].semilogx(wyn['kroki'], pk['iou_srednie'], color=KOL[nazwa], lw=2, label=f'{nazwa} IoU')
    ax[1, 2].semilogx(wyn['kroki'], pk['promien_mediana'], '--', color=KOL[nazwa], lw=1.2, label=f'{nazwa} promień chmury zdjęcia (PCA3)')
ax[0, 0].set(xlabel='τ (kroki uczenia)', ylabel='MSD (PCA32), okno 300–1000', title='Wczesna faza: względem populacji (S0) i klatki (S1)')
ax[0, 1].set(xlabel='krok', ylabel='energia na krok²', title='Temperatura klatki i udział pola kolektywnego zdjęcia')
ax[0, 2].set(xlabel='krok', ylabel='udział zachowanych sąsiadów', title='Trwałość sąsiedztwa', ylim=(0, 1))
ax[1, 0].set(xlabel='t_w (krok)', ylabel='MSD', title='Późna faza: starzenie względem klatki')
ax[1, 1].set(xlabel='krok', ylabel='|Δxyz| / krok', title='Ruchliwość zdjęć (4000 zdjęć, PCA3): mediana i kwartyle')
st = wyn['pelne_klatki']['stosunek_val_do_train']
ax[1, 2].set(xlabel='krok', ylabel='IoU / promień', title=f"Detekcja i kondensacja; późny ruch VAL/TRAIN = {st['ruch_pozny']:.2f}, "
                                                        f"Spearman ruch↔IoU: TRAIN {pelne['TRAIN']['spearman_ruch_pozny_vs_iou']:.2f}, VALID {pelne['VALID']['spearman_ruch_pozny_vs_iou']:.2f}")
for a in ax.flat:
    a.grid(True, which='both', alpha=0.4)
    a.legend(fontsize=8)
fig.suptitle(f"Zdjęcia uczące ({wyn['n_train']}) kontra walidacyjne ({wyn['n_val']}): ruch tokenów po checkpointach biegu 35 000 (pomiar na Edenie)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(WY, f'train_val{SUF}.png'))
log('GOTOWE')
