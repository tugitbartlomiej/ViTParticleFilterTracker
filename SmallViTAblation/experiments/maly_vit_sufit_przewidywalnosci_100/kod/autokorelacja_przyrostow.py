"""Autokorelacja przyrostów latentu do opóźnienia 150 kroków: czy ruch to regularne drgania (ujemny płat, okres),
czy pamięć pędu, która po prostu wygasa.

Rozstrzyga, czy równanie z bezwładnością (drugiego rzędu, z okresem) ma co przewidywać na 100 kroków:
koherentne drgania dają wyraźny ujemny płat i powtarzalny okres; gasnąca pamięć pędu daje zanik do zera bez płatu.
Liczone na tokenach narzędzia 12 obrazów, osobno dla kroków 0-300 (rozgrzewka lr) i 300-1000.
Uruchomienie: py -3.11 autokorelacja_przyrostow.py --wyniki ../wyniki/autokorelacja
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dane

LMAX = 150
OKRESY = {'0-300': (0, 300), '300-1000': (300, 1000)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki); out.mkdir(parents=True, exist_ok=True)
    lat = dane.latent(); tool, _ = dane.maski()
    acf = {k: np.zeros(LMAX + 1) for k in OKRESY}
    n = {k: np.zeros(LMAX + 1) for k in OKRESY}
    srednie_cos = {k: [] for k in OKRESY}
    for i in range(dane.N_OBR):
        X = np.asarray(lat[:, i, np.flatnonzero(tool[i]), :], dtype=np.float64)   # (1001, U, 768)
        dX = X[1:] - X[:-1]
        for k, (a, b) in OKRESY.items():
            d = dX[a:b]                                                            # przyrosty kroków a..b-1
            for l in range(LMAX + 1):
                acf[k][l] += np.einsum('tud,tud->', d[:len(d) - l], d[l:])
                n[k][l] += (len(d) - l)
        print('obraz', i, flush=True)
    wyn = {}
    for k in OKRESY:
        r = acf[k] / n[k]
        r = r / r[0]
        ujemne = np.flatnonzero(r < 0)
        pierwszy_ujemny = int(ujemne[0]) if len(ujemne) else None
        minimum = int(np.argmin(r))
        wyn[k] = dict(acf=r.tolist(), pierwsze_ujemne_opoznienie=pierwszy_ujemny, opoznienie_minimum=minimum,
                      wartosc_minimum=float(r[minimum]), suma_acf_do_100=float(r[:101].sum()))
    (out / 'ACF.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, v in wyn.items():
        print(k, 'acf lag 1,5,10,20,30,50,75,100,150:', [round(v['acf'][l], 3) for l in (1, 5, 10, 20, 30, 50, 75, 100, 150)],
              '| pierwsze ujemne:', v['pierwsze_ujemne_opoznienie'], '| min', v['opoznienie_minimum'], round(v['wartosc_minimum'], 3),
              '| suma do 100:', round(v['suma_acf_do_100'], 2))


if __name__ == '__main__':
    main()
