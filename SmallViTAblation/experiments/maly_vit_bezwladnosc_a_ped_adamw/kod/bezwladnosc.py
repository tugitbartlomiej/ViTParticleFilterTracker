"""Czy bezwładność ruchu cząstek (v(t+1) ≈ 0,87 v(t) + S) to pęd AdamW? Analiza nagrania sił, bez uczenia.

Dane: ../maly_vit_rownanie_latentu/nagranie_s47_sily i nagranie_s48_sily (nagranie.py): latent sond (średnia po tokenach
narzędzia, 192 sondy, 768 kanałów) po każdym kroku oraz od kroku 900 siły z(θ_t + d) − z(θ_t) części aktualizacji AdamW:
  nowa  d = −lr/b1 (1 − β1) g_t / mianownik_t      (gradient bieżącej paczki)
  ped   d = −lr/b1 β1 m_(t−1) / mianownik_t         (pęd: pamięć poprzednich paczek)
  zanik d = −lr wd θ_t,  suma: wszystkie trzy naraz.
W przestrzeni wag ped_t = β1 (mianownik_(t−1) / mianownik_t)(b1_(t−1)/b1_t) u_(t−1), czyli prawie dokładnie 0,9 razy
poprzednia aktualizacja. Pytanie: ile z tego przenosi się na ruch latentu.

Mierzone (jeden współczynnik wspólny dla kanałów i sond, najmniejsze kwadraty, sondy z narzędziem):
  A  prawo ruchu:      Δz_t = a Δz_(t−1) + S_nowa_t + S_zanik_t + reszta      (a = bezwładność obserwowana)
  B  pęd z prędkości:  S_ped_t = b Δz_(t−1) + reszta                         (ile pędu to poprzednia prędkość)
  C  rekurencja pędu:  S_ped_t = c (S_ped + S_nowa)_(t−1) + reszta            (oczekiwane ≈ 0,9, gdy jakobian i mianownik stoją)
  D  udział:  ile ruchu Δz wyjaśnia suma sił (reszta = bufory BatchNorm i nieliniowość)
Wyrównanie kroków (siła i ↔ przyrost Δz_(900+i+p)) wybrane przez największą zgodność sumy sił z przyrostem.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]
NAG = EXP / 'maly_vit_rownanie_latentu'


def wczytaj(nazwa):
    d = NAG / nazwa
    meta = json.loads((d / 'META.json').read_text(encoding='utf-8'))
    m = lambda n: np.load(d / f'{n}.npy', mmap_mode='r')
    return meta, m('sondy_lat'), {k: m(f'sila_{k}') for k in ('nowa', 'ped', 'zanik', 'suma')}


NAZWY = ('y', 'yp', 'nz', 'ped', 'prev', 'suma', 'nowa', 'zanik')


def _gram(Z, S, ob, od, i, p, porcja=12):
    """Macierz Grama [8, 8] wektorów NAZWY po krokach i sondach (sumowana porcjami sond, bez całych tablic w pamięci)."""
    G = np.zeros((len(NAZWY), len(NAZWY)))
    t = od + i + p
    for k in range(0, len(ob), porcja):
        sl = ob[k:k + porcja]
        zz = np.asarray(Z[:, sl, 0, :], np.float32)
        y = zz[t + 1] - zz[t]; yp = zz[t] - zz[t - 1]
        sn, sz, sp, ss = (np.asarray(S[n][:, sl], np.float32) for n in ('nowa', 'zanik', 'ped', 'suma'))
        w = dict(y=y, yp=yp, nz=sn[i] + sz[i], ped=sp[i], prev=sp[i - 1] + sn[i - 1], suma=ss[i], nowa=sn[i], zanik=sz[i])
        X = np.stack([w[n].reshape(-1) for n in NAZWY]).astype(np.float64)
        G += X @ X.T
    return G


def _skill(G, cel, komb):
    """1 − ||cel − sum c_k x_k||^2 / ||cel||^2 z macierzy Grama; komb = {nazwa: współczynnik}."""
    J = {n: j for j, n in enumerate(NAZWY)}
    v = np.zeros(len(NAZWY)); v[J[cel]] = 1
    for n, c in komb.items():
        v[J[n]] -= c
    return float(1 - v @ G @ v / G[J[cel], J[cel]])


def analiza(nazwa, okna=((900, 1500), (1500, 3000), (900, 3000))):
    meta, Z, S = wczytaj(nazwa)
    od = meta['sily_od']; n = S['nowa'].shape[0]
    ob = np.flatnonzero(np.asarray(meta['obecn'], bool))
    J = {k: j for j, k in enumerate(NAZWY)}
    i_all = np.arange(1, n - 2)
    zgod = {}
    for p in (-1, 0, 1):
        G = _gram(Z, S, ob, od, i_all[::7], p)
        zgod[p] = _skill(G, 'y', {'suma': 1.0})
    p = max(zgod, key=lambda q: zgod[q])
    wynik = dict(bieg=nazwa, sond=int(len(ob)), wyrownanie=int(p), zgodnosc_sumy_sil_z_przyrostem={str(k): v for k, v in zgod.items()})
    for a0, a1 in okna:
        i = np.arange(max(a0 - od, 1), min(a1 - od, n - 2))
        G = _gram(Z, S, ob, od, i, p)
        g = lambda u, v: G[J[u], J[v]]
        a = (g('y', 'yp') - g('nz', 'yp')) / g('yp', 'yp')                 # A
        b = g('ped', 'yp') / g('yp', 'yp')                                  # B
        c = g('ped', 'prev') / g('prev', 'prev')                            # C
        wynik[f'kroki_{a0}_{a1}'] = dict(
            a_bezwladnosc_obserwowana=float(a), skill_prawa_A=_skill(G, 'y', {'yp': a, 'nz': 1.0}),
            b_ped_z_poprzedniej_predkosci=float(b), skill_B=_skill(G, 'ped', {'yp': b}),
            c_rekurencja_pedu=float(c), skill_C=_skill(G, 'ped', {'prev': c}),
            udzial_sumy_sil_w_ruchu=_skill(G, 'y', {'suma': 1.0}),
            skill_bez_pedu_tylko_nowa_zanik=_skill(G, 'y', {'nz': 1.0}),
            skill_nowa_ped_zanik_osobno=_skill(G, 'y', {'nowa': 1.0, 'ped': 1.0, 'zanik': 1.0}),
            udzial_aktualizacji_w_poprzednim_ruchu=float(g('prev', 'yp') / g('yp', 'yp')),
            norma_wzgledna=dict(ped=float(np.sqrt(g('ped', 'ped') / g('y', 'y'))),
                                nowa=float(np.sqrt(g('nowa', 'nowa') / g('y', 'y'))),
                                zanik=float(np.sqrt(g('zanik', 'zanik') / g('y', 'y')))))
        print(nazwa, a0, a1, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in wynik[f'kroki_{a0}_{a1}'].items()}, flush=True)
    return wynik


def main():
    biegi = sys.argv[1:] or ['nagranie_s47_sily', 'nagranie_s48_sily']
    wyniki = [analiza(b) for b in biegi]
    out = Path(__file__).resolve().parents[1] / 'WYNIKI_bezwladnosc.json'
    out.write_text(json.dumps(wyniki, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(wyniki, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
