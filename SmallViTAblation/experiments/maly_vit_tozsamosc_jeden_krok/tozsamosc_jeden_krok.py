"""Umiejętność jednego kroku dla siły tożsamości, gdy równanie widzi paczkę kroku t i pęd AdamW (bez uczenia sieci).

Nagrania: maly_vit_rownanie_latentu/nagranie_s47_sily (siły od 900), nagranie_s48_sily (siły od 2000).
Współczynniki skalarne, wspólne dla 768 liczb i 180 sond, dopasowane na s47 1000-1999; test s47 2000-2998
i s48 2000-2998 tymi samymi współczynnikami. Tożsamość: kierunek LDA z kroku 1000 każdego ziarna (dane.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP / 'maly_vit_rownanie_macierze'))
import dane as DN  # noqa: E402

WARIANTY = {
    'pęd (prognoza)': ['ped'],
    'bezwładność + paczka (równanie proste)': ['v', 'nowa'],
    'paczka + pęd': ['nowa', 'ped'],
    'paczka + pęd + zanik': ['nowa', 'ped', 'zanik'],
    'bezwładność + paczka + pęd': ['v', 'nowa', 'ped'],
}


def wczytaj(s):
    k = DN.NAGRANIA / f'nagranie_s{s}_sily'
    m = json.loads((k / 'META.json').read_text(encoding='utf-8'))
    L = np.load(k / 'sondy_lat.npy', mmap_mode='r')
    ob = np.array(m['obecn'])
    tz = DN.wektor_tozsamosci(L[960:1001:10].astype(np.float64).mean(0), ob)
    F = {n: np.load(k / f'sila_{n}.npy', mmap_mode='r') for n in ('nowa', 'ped', 'zanik')}
    return m['sily_od'], L, ob, tz, F


def bloki(L, F, od, ob, t0, t1, krok=200):
    for a in range(t0, t1, krok):
        t = np.arange(a, min(a + krok, t1))
        Z = L[t[0] - 1:t[-1] + 2][:, ob, 0].astype(np.float32)        # stany t0-1 .. t1
        c = dict(y=Z[2:] - Z[1:-1], v=Z[1:-1] - Z[:-2])
        for n in F:
            c[n] = F[n][t[0] - od:t[-1] + 1 - od][:, ob].astype(np.float32)
        yield c


def main():
    od, L, ob, tz, F = wczytaj(47)
    wyn = {}
    XtX = {k: 0 for k in WARIANTY}; Xty = {k: 0 for k in WARIANTY}
    for c in bloki(L, F, od, ob, 1000, 2000):
        y = c['y'].reshape(-1).astype(np.float64)
        for k, cz in WARIANTY.items():
            X = np.stack([c[n].reshape(-1) for n in cz], 1).astype(np.float64)
            XtX[k] = XtX[k] + X.T @ X; Xty[k] = Xty[k] + X.T @ y
    W = {k: np.linalg.solve(XtX[k], Xty[k]) for k in WARIANTY}
    wyn['wspolczynniki_s47'] = {k: dict(zip(WARIANTY[k], np.round(W[k], 4).tolist())) for k in WARIANTY}
    print('współczynniki', wyn['wspolczynniki_s47'], flush=True)
    W['paczka + pęd + zanik, wszystkie = 1 (bez dopasowania)'] = np.ones(3)
    WAR = dict(WARIANTY, **{'paczka + pęd + zanik, wszystkie = 1 (bez dopasowania)': ['nowa', 'ped', 'zanik']})
    for s in (47, 48):
        if s == 48:
            od, L, ob, tz, F = wczytaj(48)
        t0 = max(2000, od + 1)
        sse = {k: 0.0 for k in WAR}; ss = 0.0
        ide = {k: [] for k in WAR}; idy = []
        for c in bloki(L, F, od, ob, t0, 2999):
            y = c['y']; ss += float((y.astype(np.float64) ** 2).sum())
            idy.append((y / tz['sd']).astype(np.float64) @ tz['w'])
            for k, cz in WAR.items():
                p = sum(W[k][i] * c[n] for i, n in enumerate(cz))
                sse[k] += float(((y - p).astype(np.float64) ** 2).sum())
                ide[k].append((p / tz['sd']).astype(np.float64) @ tz['w'])
        idy = np.concatenate(idy)
        wyn[f's{s}'] = {k: dict(latent=round(100 * (1 - sse[k] / ss), 2),
                                tozsamosc=round(DN.umiejetnosc(idy, np.concatenate(ide[k])), 2)) for k in WAR}
        print(f's{s} test {t0}-2998', json.dumps(wyn[f's{s}'], ensure_ascii=False, indent=1), flush=True)
    out = Path(__file__).resolve().parent / 'wyniki' / 'TOZSAMOSC_JEDEN_KROK.json'
    out.write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    print('zapisano', out)


if __name__ == '__main__':
    main()
