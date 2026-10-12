"""Równanie Langevina na 1..10 kroków z siłą pędu optymalizatora (dozwolone: stany do t, paczki do t-1, stan AdamW).

Dane: nagranie `maly_vit_rownanie_latentu/nagranie_s47_sily` (docs/23): latent 192 sond co krok (kanał 0 =
średnia po tokenach narzędzia), siły w stanie t przepuszczone przez sieć:
  S_ped_t   = z(θ_t + d_ped) - z(θ_t), d_ped z pędu m_{t-1} (pamięć poprzednich paczek, znana w chwili t),
  S_zanik_t = zanik wag (deterministyczny), S_nowa_t = gradient paczki t (NIEZNANY w prognozie; tylko do granicy).
Rozstrzyga: czy z pędem optymalizatora równanie przewiduje 10 kroków naprzód lepiej niż z samej historii latentu
(0,22 na h=10) i czy dochodzi do 85 procent. Obala "pęd wystarczy": wynik na h=10 poniżej 0,5.
Granica górna "paczki znane": to samo równanie z prawdziwą siłą każdej przyszłej paczki (łamie regułę prognozy,
ale mówi, ile w ogóle da się wyciągnąć z liniowego równania przy znanych paczkach).

Modele (współczynniki wspólne dla 768 współrzędnych i 180 sond; dopasowanie starty 1000..1999, cele <= 2000;
test starty 2000..2990 na s47 oraz przeniesienie na s48 bez ponownego dopasowania):
  hist_wprost      Δ_h = f(x_t - μ_t, Δx_t..Δx_{t-4})                                 [osobne wsp. na każde h]
  ped_wprost       Δ_h = f(S_ped_t, S_zanik_t)                                        [osobne wsp. na każde h]
  hist_ped_wprost  Δ_h = f(historia, S_ped_t, S_zanik_t)                              [osobne wsp. na każde h]
  hist_ped_iter    równanie jednego kroku iterowane 10 razy, S_ped gaśnie jak 0,9^k, zanik stały, paczki = 0
  paczki_znane_iter  jak wyżej, ale z prawdziwą S_nowa każdego kroku (granica, nie prognoza)
Uruchomienie: py -3.11 rownanie_z_pedem_10.py --wyniki ../wyniki/rownanie_z_pedem
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

NAG = Path(__file__).resolve().parents[2] / 'maly_vit_rownanie_latentu'
HMAX, J, K_MU, BETA1 = 10, 5, 101, 0.9
FIT = range(1000, 2000)
TEST = range(2000, 2991)
LAM = 1e-4


def wczytaj(s, t_lat0, t_lat1, t_sil0, t_sil1):
    d = NAG / f'nagranie_s{s}_sily'
    meta = json.loads((d / 'META.json').read_text(encoding='utf-8'))
    ob = np.array(meta['obecn']); od = meta['sily_od']
    lat = np.load(d / 'sondy_lat.npy', mmap_mode='r')
    Z = np.asarray(lat[t_lat0:t_lat1 + 1, :, 0, :], dtype=np.float32)[:, ob]          # (T, 180, 768)
    sily = {}
    for k in ('ped', 'zanik', 'nowa', 'suma'):
        a = np.load(d / f'sila_{k}.npy', mmap_mode='r')
        sily[k] = np.ascontiguousarray(a[t_sil0 - od:t_sil1 - od][:, ob])             # float16; indeks i -> krok t_sil0 + i
    return Z, sily, t_lat0, t_sil0


class Dane:
    """Siły w float16 (oszczędność RAM), średnia μ_t liczona na żądanie z ostatnich K_MU stanów."""

    def __init__(self, Z, sily, t0_lat, t0_sil):
        self.Z, self.sily, self.t0, self.ts = Z, sily, t0_lat, t0_sil

    def x(self, t): return self.Z[t - self.t0]
    def sila(self, k, t): return self.sily[k][t - self.ts].astype(np.float32)
    def mu(self, t): i = t - self.t0; return self.Z[i + 1 - K_MU:i + 1].mean(0, dtype=np.float64).astype(np.float32)
    def hist(self, t):
        return [self.x(t) - self.mu(t)] + [self.x(t - j) - self.x(t - j - 1) for j in range(J)]


def cechy(D, t, zestaw):
    f = []
    if 'hist' in zestaw: f += D.hist(t)
    if 'ped' in zestaw: f += [D.sila('ped', t), D.sila('zanik', t)]
    if 'nowa' in zestaw: f += [D.sila('nowa', t)]
    if 'suma' in zestaw: f += [D.sila('suma', t)]
    return np.stack(f, 1)                                                              # (U, F, D)


def dopasuj(D, zestaw, nh):
    F = len(cechy(D, FIT.start, zestaw)[0]); G = np.zeros((F, F)); b = np.zeros((F, nh))
    for t in FIT:
        X = cechy(D, t, zestaw); G += np.einsum('ufd,ugd->fg', X, X)
        for h in range(1, nh + 1):
            if t + h > FIT.stop: continue
            b[:, h - 1] += np.einsum('ufd,ud->f', X, D.x(t + h) - D.x(t))
    return np.linalg.solve(G + LAM * np.trace(G) * np.eye(F), b)


def wprost(D, t, zestaw, beta):
    X = cechy(D, t, zestaw)
    return [D.x(t) + np.einsum('ufd,f->ud', X, beta[:, h]) for h in range(HMAX)]


def iter_(D, t, beta1, tryb):
    """Równanie jednego kroku iterowane HMAX razy.
    tryb 'prognoza': paczki nieznane (ich siła = 0), pęd gaśnie jak 0,9^k, zanik stały.
    tryb 'paczki':   prawdziwa siła każdej przyszłej paczki S_nowa, pęd aktualizowany jak w AdamW:
                     S_ped_{k+1} ≈ β1 (S_ped_k + S_nowa_k)  (m_t = β1 m_{t-1} + (1-β1) g_t, d_ped ∝ β1 m, d_nowa ∝ (1-β1) g).
    tryb 'suma':     prawdziwa pełna siła kroku S_suma (pęd + paczka + zanik, policzona w prawdziwym stanie) - granica."""
    mu = D.mu(t); x = D.x(t).copy(); lagi = [D.x(t - j) - D.x(t - j - 1) for j in range(J)]
    ped, zanik = D.sila('ped', t), D.sila('zanik', t)
    out = []
    for k in range(HMAX):
        if tryb == 'suma':
            f = [x - mu] + lagi + [D.sila('suma', t + k)]
        else:
            f = [x - mu] + lagi + [ped, zanik] + ([D.sila('nowa', t + k)] if tryb == 'paczki' else [])
        d = np.einsum('ufd,f->ud', np.stack(f, 1), beta1[:, 0])
        x = x + d; lagi = [d] + lagi[:-1]; out.append(x.copy())
        ped = BETA1 * (ped + D.sila('nowa', t + k)) if tryb == 'paczki' else BETA1 * ped
    return out


def ocen(D, modele):
    sse = {m: np.zeros(HMAX) for m in modele}; ssp = np.zeros(HMAX)
    for t in TEST:
        prawda = [D.x(t + h) for h in range(1, HMAX + 1)]
        for h in range(HMAX):
            e = D.x(t) - prawda[h]; ssp[h] += np.einsum('ud,ud->', e, e)
        for m, fn in modele.items():
            pred = fn(D, t)
            for h in range(HMAX):
                e = pred[h] - prawda[h]; sse[m][h] += np.einsum('ud,ud->', e, e)
    return {m: (1 - sse[m] / ssp).tolist() for m in modele}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    D47 = Dane(*wczytaj(47, 800, 3000, 900, 3000))
    print('s47 wczytane', D47.Z.shape, f'{time.time() - t0:.0f} s', flush=True)
    b_hist = dopasuj(D47, ('hist',), HMAX)
    b_ped = dopasuj(D47, ('ped',), HMAX)
    b_hp = dopasuj(D47, ('hist', 'ped'), HMAX)
    b_hp1 = dopasuj(D47, ('hist', 'ped'), 1)
    b_hpn1 = dopasuj(D47, ('hist', 'ped', 'nowa'), 1)
    b_hs1 = dopasuj(D47, ('hist', 'suma'), 1)
    print('dopasowane', f'{time.time() - t0:.0f} s', flush=True)
    modele = {
        'hist_wprost': lambda D, t: wprost(D, t, ('hist',), b_hist),
        'ped_wprost': lambda D, t: wprost(D, t, ('ped',), b_ped),
        'hist_ped_wprost': lambda D, t: wprost(D, t, ('hist', 'ped'), b_hp),
        'hist_ped_iter': lambda D, t: iter_(D, t, b_hp1, 'prognoza'),
        'paczki_znane_iter': lambda D, t: iter_(D, t, b_hpn1, 'paczki'),
        'pelna_sila_znana_iter': lambda D, t: iter_(D, t, b_hs1, 'suma'),
    }
    wyn = dict(meta=dict(zrodlo=str(NAG), fit=[FIT.start, FIT.stop - 1], test=[TEST.start, TEST.stop - 1],
                         horyzonty=list(range(1, HMAX + 1)), J=J, k_mu=K_MU, beta1=BETA1, lam=LAM,
                         skill='1 - SSE / SSE prognozy bez zmian, 180 sond x 768 wspolrzednych',
                         wspolczynniki_1_krok=dict(hist_ped=b_hp1[:, 0].round(4).tolist(),
                                                   hist_ped_nowa=b_hpn1[:, 0].round(4).tolist(),
                                                   hist_suma=b_hs1[:, 0].round(4).tolist(),
                                                   nazwy=['x-mu', 'dx_t', 'dx_t-1', 'dx_t-2', 'dx_t-3', 'dx_t-4',
                                                          'S_ped', 'S_zanik', '(S_nowa)'])))
    wyn['s47_test'] = ocen(D47, modele)
    print('s47', {m: [round(v[0], 3), round(v[4], 3), round(v[9], 3)] for m, v in wyn['s47_test'].items()}, flush=True)
    del D47
    D48 = Dane(*wczytaj(48, 1800, 3000, 2000, 3000))
    wyn['s48_przeniesienie'] = ocen(D48, modele)
    print('s48', {m: [round(v[0], 3), round(v[4], 3), round(v[9], 3)] for m, v in wyn['s48_przeniesienie'].items()}, flush=True)
    wyn['meta']['czas_s'] = time.time() - t0
    (out / 'ROWNANIE_Z_PEDEM.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print('zapisano', f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
