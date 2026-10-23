"""Obserwable na płaszczyźnie krok uczenia × podkrok przejścia (806 checkpointów × 12 podkroków) z zapisanych przejść przez sieć
(`LaserLatentSpaceApp/cache_uczenia/lasery/7992f2317c64a2e3/przejscia/k*.npz`, bieg s47, klatka TRAIN 0, etykieta z obrysu z historii
140854f7a6dbcbe5 tej samej klatki; sprawdzane: Z[0] obu historii identyczne).

Stany: xn (13, 600, 192) to stan względem środka chmury podzielony przez rms[s]; Xc[s] = xn[s] · rms[s] odtwarza dokładne wycentrowane
stany, więc Δ_s = Xc[s] − Xc[s−1] to dokładne przesunięcie w podkroku bez wyrazów wolnych (ZASADY Z20). Podkrok s = 1..12: nieparzysty =
uwaga bloku (s+1)//2, parzysty = MLP.

Mierzone na checkpoint:
  stos[12]          r_po / r_przed = rms[s] / rms[s−1] (KURCZENIE_UWAGA_MLP na wszystkich 806 checkpointach);
  dl[12]            mediana |Δ_s| / rms[s−1];
  rad[12]           mediana cos(Δ_s, Xc[s−1]) (ujemna = do środka chmury);
  r2cv[12]          CV-R² (2 fałdy po 300 tokenów) afinicznego dopasowania Δ_s ~ M·Xc[s−1] + c w 192D: ile przesunięcia jest
                    funkcją afiniczną WŁASNEGO położenia cząsteczki (MLP: dokładnie funkcja własnego położenia, pytanie o liniowość;
                    uwaga: zależy od innych cząsteczek, pytanie czy to widać); r2cv_null = to samo z przetasowanymi Δ;
  r2_3d[12]         R² (w próbie) tego dopasowania w rzucie 3D aplikacji (pos_c), 4 parametry na współrzędną;
  trans[12]         udział wspólnego przesunięcia: |średnia_i Δpos_s|² / średnia_i |Δpos_s|² w surowym rzucie 3D (pos; z wyrazami wolnymi);
  bal[3]            wskaźnik balistyczny po głębokości: |Σ_s Δ_s|² / Σ_s |Δ_s|² (wszystkie 12, same uwagi, same MLP), mediana po tokenach
                    (błądzenie losowe ≈ 1, ruch prosty = liczba podkroków);
  cos_am[6], cos_ma[5], cos_mm[5], cos_aa[5]   mediany cos między przesunięciami: uwaga_b vs MLP_b, MLP_b vs uwaga_{b+1}, MLP_b vs MLP_{b+1},
                    uwaga_b vs uwaga_{b+1};
  att_neff[6,3]     1/Σ_j A_ij² (mediana po i): efektywna liczba cząsteczek, których słucha cząsteczka (600 = uwaga jednostajna = czyste
                    przesunięcie całej chmury, czyli afiniczna translacja);
  att_self, att_sas8, att_dist[6,3]   masa uwagi na sobie, na 8 sąsiadach siatki, średnia odległość na siatce (w łatach) ważona uwagą;
  att_bg2tool, att_tool2tool[6,3]     wzbogacenie: Σ_{j∈narzędzie} A_ij uśrednione po i z tła / po i z narzędzia, podzielone przez udział
                    narzędzia (1 = brak preferencji);
  mlp_lit[6], mlp_neff[6], mlp_top10[6]   mediana po tokenach: odsetek laserów z z > 0,05, (Σ|z|)²/Σz² (efektywna liczba laserów),
                    udział 10 najjaśniejszych w Σ|z|;
  fisher[13], fisher_u[13]   |μ_narz − μ_tło|² / średnia wariancja wewnątrz klas (ślad) w 192D, na Xc i na kierunkach Xc/|Xc| (łaty mieszane
                    pominięte);
  knn[13], knn_u[13]         czystość 10 najbliższych sąsiadów tokenów narzędzia (udział narzędzia; zero = udział narzędzia w chmurze);
  dir[13], dir_u[13]         energia Dirichleta na siatce 4-sąsiedztwa / (2 · wariancja chmury): 1 = cechy niezależne od sąsiedztwa,
                    < 1 = gładkie po siatce (uwaga jako filtr dolnoprzepustowy), > 1 = sąsiedzi bardziej różni niż losowe pary;
  pewnosc, krok.
Wynik: Wyniki/Dane/kroki_podkroki.npz oraz Wyniki/Dane/kroki_podkroki_tabele.json (wybrane epoki).
Uruchomienie: python pomiar_kroki_podkroki.py <katalog Wyniki/Dane> [co_ktory_checkpoint]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

BAZA = Path(r'W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation')
HIST = BAZA / 'LaserLatentSpaceApp/cache_uczenia/lasery/7992f2317c64a2e3'
HIST_ETY = BAZA / 'LaserLatentSpaceApp/cache_uczenia/lasery/140854f7a6dbcbe5'
WY = Path(sys.argv[1])
CO = int(sys.argv[2]) if len(sys.argv) > 2 else 1
H, W, T, D = 20, 30, 600, 192
PROG_Z = 0.05
K_NN = 10
EPOKI = [0, 1094, 5470, 10940, 21880, 32820, 43760, 54700, 65640]

Za = np.load(HIST / 'Z.npy', mmap_mode='r'); Zb = np.load(HIST_ETY / 'Z.npy', mmap_mode='r')
assert np.array_equal(np.asarray(Za[0]), np.asarray(Zb[0])), 'historie dotyczą różnych klatek'
e = np.load(HIST_ETY / 'etykieta_obrys.npz'); udzial = e['udzial']
narz = udzial >= 0.5; tlo = udzial == 0; miesz = ~narz & ~tlo
print(f'narzędzie {narz.sum()}, mieszane {miesz.sum()}, tło {tlo.sum()}', flush=True)

idx = np.arange(T).reshape(H, W)
kraw = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1), np.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)])
r, c = np.divmod(np.arange(T), W)
dist_siatki = np.sqrt((r[:, None] - r[None]) ** 2 + (c[:, None] - c[None]) ** 2)
sas8 = (np.abs(r[:, None] - r[None]) <= 1) & (np.abs(c[:, None] - c[None]) <= 1); np.fill_diagonal(sas8, False)
rng = np.random.default_rng(0)
perm = rng.permutation(T); FALDY = ((perm[:300], perm[300:]), (perm[300:], perm[:300]))
PERM_NULL = rng.permutation(T)


def r2_cv(Xp, Dl):
    out = []
    for tr, te in FALDY:
        Z = np.c_[Xp[tr], np.ones(len(tr))]
        coef, *_ = np.linalg.lstsq(Z, Dl[tr], rcond=None)
        pred = np.c_[Xp[te], np.ones(len(te))] @ coef
        out.append(1 - ((Dl[te] - pred) ** 2).sum() / ((Dl[te] - Dl[te].mean(0)) ** 2).sum())
    return float(np.mean(out))


def r2_w_probie(Xp, Dl):
    Z = np.c_[Xp, np.ones(len(Xp))]
    coef, *_ = np.linalg.lstsq(Z, Dl, rcond=None)
    return float(1 - ((Dl - Z @ coef) ** 2).sum() / ((Dl - Dl.mean(0)) ** 2).sum())


def fisher(X):
    a, b = X[narz], X[tlo]
    d = a.mean(0) - b.mean(0)
    sw = (a.var(0).sum() * len(a) + b.var(0).sum() * len(b)) / (len(a) + len(b))
    return float(d @ d / sw)


def knn(X):
    Dm = cdist(X[narz], X); Dm[np.arange(narz.sum()), np.where(narz)[0]] = np.inf
    nn = np.argsort(Dm, 1)[:, :K_NN]
    return float(narz[nn].mean())


def dirichlet(X):
    en = ((X[kraw[:, 0]] - X[kraw[:, 1]]) ** 2).sum(1).mean()
    return float(en / (2 * ((X - X.mean(0)) ** 2).sum(1).mean()))


def jedn(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def cosy(A, B):
    return np.median((A * B).sum(1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12))


def pomiar(plik):
    f = np.load(plik)
    rms = f['rms'].astype(np.float64)
    Xc = f['xn'].astype(np.float32) * rms[:, None, None].astype(np.float32)   # (13, 600, 192)
    pos, pos_c = f['pos'], f['pos_c']
    dX = Xc[1:] - Xc[:-1]                                                        # (12, 600, 192)
    w = dict(krok=int(f['s_krok']), pewnosc=float(f['s_pewnosc']), rms=rms, stos=rms[1:] / rms[:-1])
    w['dl'] = np.median(np.linalg.norm(dX, axis=2), 1) / rms[:-1]
    w['rad'] = np.array([cosy(dX[s], Xc[s]) for s in range(12)])
    w['r2cv'] = np.array([r2_cv(Xc[s], dX[s]) for s in range(12)])
    w['r2cv_null'] = np.array([r2_cv(Xc[s], dX[s][PERM_NULL]) for s in range(12)])
    w['r2_3d'] = np.array([r2_w_probie(pos_c[s], pos_c[s + 1] - pos_c[s]) for s in range(12)])
    dp = pos[1:] - pos[:-1]
    w['trans'] = (dp.mean(1) ** 2).sum(1) / (dp ** 2).sum(2).mean(1)
    en = (dX ** 2).sum(2)                                                        # (12, 600)
    bal = lambda sel: float(np.median((dX[sel].sum(0) ** 2).sum(1) / np.maximum(en[sel].sum(0), 1e-12)))
    w['bal'] = np.array([bal(slice(None)), bal(slice(0, 12, 2)), bal(slice(1, 12, 2))])
    w['cos_am'] = np.array([cosy(dX[2 * b], dX[2 * b + 1]) for b in range(6)])
    w['cos_ma'] = np.array([cosy(dX[2 * b + 1], dX[2 * b + 2]) for b in range(5)])
    w['cos_mm'] = np.array([cosy(dX[2 * b + 1], dX[2 * b + 3]) for b in range(5)])
    w['cos_aa'] = np.array([cosy(dX[2 * b], dX[2 * b + 2]) for b in range(5)])
    A = f['ah'].astype(np.float32)                                               # (6, 3, 600, 600)
    w['att_neff'] = np.median(1 / np.maximum((A ** 2).sum(3), 1e-12), 2)
    w['att_self'] = np.median(np.diagonal(A, axis1=2, axis2=3), 2)
    w['att_sas8'] = np.median((A * sas8).sum(3), 2)
    w['att_dist'] = np.median((A * dist_siatki).sum(3), 2)
    w['att_ent'] = np.median(-(A * np.log(np.maximum(A, 1e-12))).sum(3), 2)
    pn = narz.mean()
    w['att_bg2tool'] = A[:, :, tlo][:, :, :, narz].sum(3).mean(2) / pn
    w['att_tool2tool'] = A[:, :, narz][:, :, :, narz].sum(3).mean(2) / pn
    z = f['z'].astype(np.float32)                                                # (6, 600, 768)
    az = np.abs(z)
    w['mlp_lit'] = np.median((z > PROG_Z).mean(2), 1)
    w['mlp_neff'] = np.median(az.sum(2) ** 2 / np.maximum((z ** 2).sum(2), 1e-12), 1)
    top = -np.partition(-az, 10, axis=2)[:, :, :10].sum(2)
    w['mlp_top10'] = np.median(top / np.maximum(az.sum(2), 1e-12), 1)
    U = [jedn(Xc[s]) for s in range(13)]
    w['fisher'] = np.array([fisher(Xc[s]) for s in range(13)]); w['fisher_u'] = np.array([fisher(U[s]) for s in range(13)])
    w['knn'] = np.array([knn(Xc[s]) for s in range(13)]); w['knn_u'] = np.array([knn(U[s]) for s in range(13)])
    w['dir'] = np.array([dirichlet(Xc[s]) for s in range(13)]); w['dir_u'] = np.array([dirichlet(U[s]) for s in range(13)])
    w['sprawdzenie'] = float(f['s_sprawdzenie_c'])
    return w


pliki = sorted((HIST / 'przejscia').glob('k*.npz'))[::CO]
if pliki[-1].name != 'k65640.npz':
    pliki.append(HIST / 'przejscia/k65640.npz')
print(f'{len(pliki)} checkpointów', flush=True)
wyniki = []; t0 = time.time()
for i, p in enumerate(pliki):
    wyniki.append(pomiar(p))
    if i % 50 == 0 or i == len(pliki) - 1:
        print(f'{i + 1}/{len(pliki)} krok {wyniki[-1]["krok"]} {time.time() - t0:.0f} s', flush=True)
klucze = [k for k in wyniki[0] if k != 'krok']
dane = {k: np.stack([w[k] for w in wyniki]) for k in klucze}
dane['krok'] = np.array([w['krok'] for w in wyniki])
dane['narz'] = narz; dane['udzial'] = udzial
WY.mkdir(parents=True, exist_ok=True)
np.savez_compressed(WY / 'kroki_podkroki.npz', **dane)

kr = dane['krok']; wyb = [int(np.argmin(np.abs(kr - e_))) for e_ in EPOKI]
tab = {'kroki': [int(kr[i]) for i in wyb], 'narzedzie': int(narz.sum()), 'tlo': int(tlo.sum()), 'mieszane': int(miesz.sum()),
       'zero_knn': float((narz.sum() - 1) / (T - 1))}
for k in klucze:
    a = dane[k]
    if a.ndim == 1:
        tab[k] = [float(a[i]) for i in wyb]
    elif a.ndim == 2:
        tab[k] = [[round(float(v), 4) for v in a[i]] for i in wyb]
    else:
        tab[k] = [[[round(float(v), 4) for v in row] for row in a[i]] for i in wyb]
json.dump(tab, open(WY / 'kroki_podkroki_tabele.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('zapisano', WY / 'kroki_podkroki.npz', f'{time.time() - t0:.0f} s', flush=True)
