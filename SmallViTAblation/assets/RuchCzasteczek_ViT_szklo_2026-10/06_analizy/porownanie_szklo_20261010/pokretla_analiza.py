"""Analiza jednego biegu z pokrętłami (Eden, 2026-10-10_szklo_pokretla/wyniki/<nazwa>): odciski szkła S0/S1 w oknie 300–2188
(jak UKLAD_SZKLA.json), trwałość sąsiedztwa, heterogeniczność dynamiczna (dh.analiza) w układzie CR, energia kroku w pełnych
768 wymiarach (normy dp z obserwatora) i jej prawo stygnięcia, mAP i strata z biegu. Od 2026-10-11 także testy „jaka to substancja”:
trwałość ruchliwości (zamrożona czy przemijająca), dwupunktowa korelacja przesunięć w funkcji odległości, rozbieganie par
sąsiadów (test turbulencji), rozkład energii kroku (wspólny / zdjęcie / sąsiedztwo / własny). Wynik: jeden JSON.
Uruchomienie: python pokretla_analiza.py <folder_biegu> <out.json> [od do] [podkatalog gesty, np. gesty/w3]
Liczba tokenów na zdjęcie z SONDY.json (tokenow, domyślnie 600)."""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odciski as O  # noqa: E402
import dh  # noqa: E402

RUN, OUT = sys.argv[1], sys.argv[2]
OD = int(sys.argv[3]) if len(sys.argv) > 3 else 300
DO_ = int(sys.argv[4]) if len(sys.argv) > 4 else 2188
GS = sys.argv[5] if len(sys.argv) > 5 else 'gesty'
SONDY = json.load(open(os.path.join(RUN, GS, 'SONDY.json'), encoding='utf-8'))
TOK = int(SONDY.get('tokenow', 600))
t0 = time.time()
log = lambda *a: print(*a, f'[{time.time() - t0:.0f} s]', flush=True)

Z, kr, dp2 = [], [], []
for p in sorted(glob.glob(os.path.join(RUN, GS, 'pca32_*.npz'))):
    z = np.load(p)
    k = z['kroki']
    Z.append(np.ascontiguousarray(z['pca32'].astype(np.float32)).reshape(len(k), -1, z['pca32'].shape[-1]))
    dp2.append((z['normy'][..., 1].astype(np.float32) ** 2).reshape(len(k), -1).mean(1))     # średnia |z(t)−z(t−rekord)|² w 768D
    kr.append(k)
Z = np.concatenate(Z); kroki = np.concatenate(kr).astype(int); dp2 = np.concatenate(dp2)
N = Z.shape[1]
log(RUN, GS, 'rekordów', len(kroki), 'tokenów', N, f'({TOK} na zdjęcie)', 'kroki', kroki[0], kroki[-1], 'zapis', SONDY.get('zapis', 'f16'))
ind = {int(k): i for i, k in enumerate(kroki)}
T_REF, GESTE_DO = OD, min(DO_, int(kroki[kroki <= DO_][-1]))
gesty = (kroki >= T_REF) & (kroki <= GESTE_DO)
assert (np.diff(kroki[gesty]) == 1).all(), 'zapis gęsty nie co 1 krok'
N6, dnn, dsh = O.knn_w_blokach(Z[ind[T_REF]], TOK, 6)
d_nn, d_shell = float(np.median(dnn)), float(np.median(dsh))
trw = O.trwalosc_sasiedztwa(Z, kroki, T_REF, tuple(int(x) for x in (T_REF + 10, T_REF + 100, T_REF + 700, T_REF + 1000, GESTE_DO, int(kroki[-1])) if x in ind), blok=TOK)
wyn = dict(bieg=os.path.basename(RUN.rstrip('/')), N=int(N), wymiar=int(Z.shape[-1]), rzut=SONDY.get('rzut', 'pca32'), zapis=SONDY.get('zapis', 'f16'),
           gesty=GS, tokenow=TOK, warstwa=SONDY.get('warstwa'), sond=int(N // TOK), d_nn_300=d_nn, d_shell_300=d_shell, trwalosc=trw,
           kroki_zapisu=dict(od=int(kroki[0]), do=int(kroki[-1])), okno=[int(T_REF), int(GESTE_DO)])
for f, k in (('metadata.json', 'metadata'), ('OCENY.json', 'oceny'), ('LOG.json', 'log')):
    p = os.path.join(RUN, 'dane', f)
    if os.path.isfile(p):
        wyn[k] = json.load(open(p, encoding='utf-8'))
if 'log' in wyn and isinstance(wyn['log'], list) and wyn['log']:
    wyn['strata'] = wyn['log'][-200:]; del wyn['log']
# energia kroku w 768D i prawo stygnięcia E ∝ (1 + (t−300)/t0)^−b na oknie gęstym
kg = kroki[gesty]; E = dp2[gesty]
m = kg >= T_REF
x = np.log1p((kg[m] - T_REF) / 400.0); y = np.log(np.maximum(E[m], 1e-20))
b = float(-np.polyfit(x, y, 1)[0]) if m.sum() > 10 else None
wyn['energia_768'] = dict(krok=[int(v) for v in kg[::10]], E=[float(v) for v in E[::10]], wykladnik_stygniecia_t0_400=b,
                         E_300_400=float(E[(kg >= T_REF) & (kg < T_REF + 100)].mean()), E_2000_2188=float(E[(kg >= GESTE_DO - 188)].mean()),
                         E_cale_768=dict(krok=[int(v) for v in kroki[::5]], E=[float(v) for v in dp2[::5]]))
TW = tuple(int(x) for x in (T_REF, T_REF + 300, T_REF + 600, T_REF + 900, T_REF + 1200, T_REF + 1500) if x < GESTE_DO - 100)
KT = tuple(int(x) for x in (T_REF, T_REF + 300, T_REF + 600, T_REF + 900, T_REF + 1200, T_REF + 1700) if x < GESTE_DO - 5)
Zg = Z[gesty]; kg = kroki[gesty]
w0 = O.odciski(Zg, kg, (T_REF, GESTE_DO), d_shell, d_nn, odejmij_pole=True, tw_lista=TW, krotkie_tw=KT, nazwa='S0')
w0['karta'], w0['spelnione'] = O.karta(w0, trw.get(T_REF + 1000)); wyn['S0'] = w0
P1 = dh.klatka_cr(Zg, kg, T_REF, N6)
w1 = O.odciski(P1, kg, (T_REF, GESTE_DO), d_shell, d_nn, odejmij_pole=True, tw_lista=TW, krotkie_tw=KT, nazwa='S1')
w1['karta'], w1['spelnione'] = O.karta(w1, trw.get(T_REF + 1000)); wyn['S1'] = w1
log('S0', w0['spelnione'], 'S1', w1['spelnione'], 'Lindemann', round(w1['lindemann_shell'], 3), 'c', round(w1['c_plateau'], 3), 'τ_1e', w1.get('tau_1e'),
    'β', round(w1['beta'], 2) if w1.get('beta') is not None else None, 'E wykładnik', b)
N6x = dh.knn_poza_blokiem(Z[ind[T_REF]], TOK, 6)                 # sąsiedzi z innych zdjęć w chwili odniesienia (pełna oś rekordów)
taus = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000]
wyn['heterogenicznosc_CR'] = dh.analiza(P1, kg, (T_REF, GESTE_DO), taus, N6, TOK, wyn['bieg'] + ' S1', pole='blok', N6_cross=N6x)
wyn['heterogenicznosc_lab'] = dh.analiza(Zg, kg, (T_REF, GESTE_DO), taus, N6, TOK, wyn['bieg'] + ' S0', pole='blok', N6_cross=N6x)
# --- testy „jaka to substancja”
wyn['trwalosc_ruchliwosci_CR'] = dh.trwalosc_ruchliwosci(P1, kg, (T_REF, GESTE_DO), TOK, tau=10, dlug=200, pole='blok', nazwa=wyn['bieg'] + ' S1')
wyn['trwalosc_ruchliwosci_lab'] = dh.trwalosc_ruchliwosci(Zg, kg, (T_REF, GESTE_DO), TOK, tau=10, dlug=200, pole='blok', nazwa=wyn['bieg'] + ' S0')
wyn['korelacja_przesuniec'] = dh.korelacja_przesuniec(Zg, kg, (T_REF, GESTE_DO), [1, 10, 100], TOK, d_shell, nazwa=wyn['bieg'])
lagi = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, GESTE_DO - T_REF]
wyn['rozbieganie_par'] = dh.rozbieganie_par(Zg, kg, T_REF, lagi, N6, d_shell, nazwa=wyn['bieg'])
dostepne = {int(k) - T_REF for k in kroki if k >= T_REF}
lagi_c = sorted(l for l in (set(lagi) | {int(k) - T_REF for k in kroki[kroki > GESTE_DO]}) if l in dostepne)
wyn['rozbieganie_par_calosc'] = dh.rozbieganie_par(Z, kroki, T_REF, lagi_c[::max(1, len(lagi_c) // 40)], N6, d_shell, nazwa=wyn['bieg'] + ' cała oś')
wyn['rozklad_wariancji'] = dh.rozklad_wariancji(Zg, kg, (T_REF, GESTE_DO), [1, 10, 100, 1000], N6, TOK, nazwa=wyn['bieg'])
wyn['korelacja_krzyzowa_blok'] = dh.korelacja_krzyzowa_par(Zg, kg, (T_REF, GESTE_DO), N6, TOK, W=100, pole='blok', nazwa=wyn['bieg'])
wyn['korelacja_krzyzowa_populacja'] = dh.korelacja_krzyzowa_par(Zg, kg, (T_REF, GESTE_DO), N6, TOK, W=100, pole='populacja', nazwa=wyn['bieg'])
json.dump(wyn, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
log('zapisano', OUT)
