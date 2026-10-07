"""Etap 1b: struktura ruchu cząstek z kroku na krok.

Dla każdego tokenu, w oknach kroków 1-300 (rozgrzewka lr) i 300-1000 (stały lr):
- korelacja kolejnych przyrostów cos(dz_t, dz_(t+1)): −0,5 = szum wokół gładkiego dryfu (x = s + v, v białe),
  0 = błądzenie losowe (Langevin bez sprężyny), > 0 = pęd (ruch ciągły);
- MSD(h) = E||z_(t+h) − z_t||^2: stały w h = szum wokół stanu, liniowy = dyfuzja, kwadratowy = dryf;
- OU na odchyleniu od gładkiego dryfu (średnia krocząca 51 kroków): r_(t+1) = rho r_t + e,
  rho ≈ 0 = szum biały (obserwator: x = s + v), rho ≈ 1 = wolny powrót.
Uruchomienie z katalogu app: python ../experiments/maly_vit_liniowy_ruch_czastek/kod/struktura_szumu.py 3 0 7
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

TU = Path(__file__).resolve().parent
sys.path.insert(0, str(TU.parents[2] / 'app'))
import analiza_uczenia as AU  # noqa: E402

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
b = AU.znajdz_biegi()[0]; o = AU.opis_biegu(b)
H = [1, 2, 5, 10, 20, 50, 100, 200]
OKNA = {'rozgrzewka_1_300': (1, 300), 'staly_lr_300_1000': (300, 1000)}
wynik = {}
for i in [int(x) for x in sys.argv[1:]] or [3]:
    Z = AU.wczytaj_klatke(b, i); n = o['narzedzie'][i]
    X = torch.from_numpy(np.asarray(Z)).to(dev)                                  # float16 na GPU (0,9 GB)
    w = {}
    for nazwa, (a, e) in OKNA.items():
        cos1, msd, rho = np.zeros(600), {h: np.zeros(600) for h in H}, np.zeros(600)
        for t0 in range(0, 600, 100):
            x = X[a:e + 1, t0:t0 + 100].float()                                  # [L, 100, 768]
            d = x[1:] - x[:-1]
            c = torch.nn.functional.cosine_similarity(d[:-1], d[1:], dim=-1)    # [L-2, 100]
            cos1[t0:t0 + 100] = c.mean(0).cpu().numpy()
            for h in H:
                if h < len(x):
                    msd[h][t0:t0 + 100] = ((x[h:] - x[:-h]) ** 2).sum(-1).mean(0).cpu().numpy()
            # gładki dryf: średnia krocząca 51 kroków (splot wzdłuż czasu), reszta r
            k = 51
            xs = torch.nn.functional.avg_pool1d(x.permute(1, 2, 0).reshape(-1, 1, len(x)), k, 1, k // 2,
                                                count_include_pad=False).reshape(100, 768, -1).permute(2, 0, 1)
            r = (x - xs)[k:-k]
            rho[t0:t0 + 100] = ((r[1:] * r[:-1]).sum((0, 2)) / (r[:-1] ** 2).sum((0, 2)).clamp_min(1e-12)).cpu().numpy()
            udz = None
        # udział wariancji reszty (szum) w całym ruchu w oknie: ||x - dryf||^2 / ||x - średnia||^2
        x = X[a:e + 1].float()
        xs = torch.nn.functional.avg_pool1d(x.permute(1, 2, 0).reshape(-1, 1, len(x)), 51, 1, 25,
                                            count_include_pad=False).reshape(600, 768, -1).permute(2, 0, 1)
        szum = float(((x - xs) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
        del x, xs
        f = lambda v: dict(narz=round(float(np.median(v[n])), 4), tlo=round(float(np.median(v[~n])), 4))
        w[nazwa] = dict(cos_kolejnych_przyrostow=f(cos1), rho_reszty_od_dryfu=f(rho),
                        udzial_szumu_w_wariancji=round(szum, 4),
                        msd={h: f(msd[h]) for h in H})
    wynik[str(i)] = w
    print(i, json.dumps(w, ensure_ascii=False), flush=True)
    del X; torch.cuda.empty_cache()
(TU.parent / 'wyniki' / 'STRUKTURA_SZUMU.json').write_text(json.dumps(wynik, ensure_ascii=False, indent=1), encoding='utf-8')
