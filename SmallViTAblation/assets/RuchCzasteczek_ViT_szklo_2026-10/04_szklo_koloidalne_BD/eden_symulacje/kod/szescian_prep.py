"""Przygotowanie filmu „sześcian”: kryształ idealny w chwili 0 + prawdziwe przesunięcia tokenów (Eden, CPU).
Położenie wyświetlane atomu i: p_i(t) = g_i + s · D (x_i(t) − x_i(0)), gdzie g_i = idealny węzeł (kolumna, wiersz patcha, warstwa 0),
x = token w rzucie 128D względem średniej zdjęcia, D = stała mapa 128 → 3 (3 główne kierunki przesunięć w całym biegu, ta sama dla wszystkich
chwil i atomów, więc trajektorie są wiernym rzutem liniowym), s = 1 komórka / odstęp sąsiadów w kroku 0 (przesunięcie o jeden odstęp
sąsiadów = jedna komórka). Węzeł sieci (wolny): q_i(t) = g_i + s · D (c_i(t) − x_i(0)), c = średnia położenia w oknie ±300 kroków.
Zapis dla każdego zdjęcia: kroki, atomy (K,600,3), wezly (K,600,3), iou, pewnosc, narzedzie (600,), udzial wariancji przesunięć w D.
Uruchomienie: python szescian_prep.py <eksport> <pomiar60> <out.npz> [zdjęcia: 0,50]"""
import json
import sys

import numpy as np
from scipy.spatial.distance import cdist

EKS, POM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ZDJ = [int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else '0,50').split(',')]
H, W, TOK, PL, OKNO = 20, 30, 600, 16, 300
META = json.load(open(f'{EKS}/META.json', encoding='utf-8'))
kroki = np.array(META['kroki']); K = len(kroki)
rp = np.load(f'{EKS}/rp128.npy', mmap_mode='r')
oceny = np.load(f'{EKS}/oceny.npy')
mp = json.load(open(f'{POM}/manifest.json', encoding='utf-8'))
siatka = np.stack([np.arange(TOK) % W, np.arange(TOK) // W], 1).astype(float)      # (kolumna, wiersz)
g = np.concatenate([siatka - siatka.mean(0), np.zeros((TOK, 1))], 1)              # idealny węzeł, jednostka = 1 komórka
wyj = dict(kroki=kroki, g=g)
for J in ZDJ:
    X = np.asarray(rp[:, J]).astype(np.float32); X -= X.mean(1, keepdims=True)
    x0 = X[0]
    Dm = cdist(x0, x0); np.fill_diagonal(Dm, np.inf); a0 = float(np.median(np.sort(Dm, 1)[:, :4].mean(1)))
    C = np.stack([X[[u for u in range(K) if abs(kroki[u] - kroki[t]) <= OKNO]].mean(0) for t in range(K)])
    dX = (X - x0).reshape(-1, 128)
    _, s_, Vt = np.linalg.svd(dX[::7] - 0, full_matrices=False)
    D = Vt[:3]                                                                       # (3, 128)
    udzial = float((s_[:3] ** 2).sum() / (s_ ** 2).sum())
    s = 1.0 / a0
    atomy = g[None] + s * np.einsum('tid,kd->tik', X - x0, D)[..., [1, 2, 0]]       # PC2, PC3 w płaszczyźnie, PC1 (największy ruch) w głąb
    wezly = g[None] + s * np.einsum('tid,kd->tik', C - x0, D)[..., [1, 2, 0]]
    zd = META['zdjecia'][J]; wiersz = J if zd['split'] == 'TRAIN' else 3000 + J - 50
    b = mp['images'][wiersz]['bbox_xyxy_480']
    cx = (siatka[:, 0] + 0.5) * PL; cy = (siatka[:, 1] + 0.5) * PL
    narz = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
    wyj.update({f'atomy_{J}': atomy.astype(np.float32), f'wezly_{J}': wezly.astype(np.float32), f'iou_{J}': oceny[:, J, 1], f'pewnosc_{J}': oceny[:, J, 0],
                f'narzedzie_{J}': narz, f'udzial_{J}': udzial, f'odstep0_{J}': a0, f'split_{J}': zd['split'], f'frame_{J}': zd['frame_id']})
    print(f'zdjęcie {J} ({zd["split"]}): odstęp sąsiadów w kroku 0 {a0:.3f}, 3 kierunki D niosą {udzial:.0%} energii przesunięć; '
          f'przesunięcie atomów na końcu {np.linalg.norm(atomy[-1] - g, axis=1).mean():.2f} komórki, węzłów {np.linalg.norm(wezly[-1] - g, axis=1).mean():.2f}', flush=True)
np.savez_compressed(OUT, **wyj)
print('zapisano', OUT)
