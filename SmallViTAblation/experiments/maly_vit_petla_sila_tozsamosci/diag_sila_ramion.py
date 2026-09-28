#!/usr/bin/env python3
"""Diagnoza po kampanii (nie kryterium): czy dobór klatek przez prognozę zmienił siłę tożsamości sieci inaczej niż
kontrola `current`, i w którą stronę prognozowała metoda. Wejście: wyniki_eden/petla_s*/<ramię>/SILA.npy (zapis co 10
kroków, 180 sond) i OBROT.json.  ->  wyniki_eden/DIAG_SILA.json"""
import json
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parent / 'wyniki_eden'
RAM = ('tozsamosc', 'current', 'losowe', 'kolejne', 'bez')
wyn = {r: [] for r in RAM}
znak, ds_sr = [], []
for d in sorted(W.glob('petla_s*')):
    for r in RAM:
        S = np.load(d / r / 'SILA.npy')                     # (101 + 7*50, 180)
        konce = [S[101 + 50 * i - 5: 101 + 50 * i].mean() for i in range(1, 8)]   # średnia siła na końcu obrotów 3..9
        wyn[r].append(konce)
    for k in range(3, 10):
        o = json.loads((d / 'tozsamosc' / f'k{k:02d}' / 'OBROT.json').read_text(encoding='utf-8'))
        dz = [a['przewidziana_zmiana'] for a in o['fastgan']]
        znak.append(np.mean(np.sign(dz))); ds_sr.append(np.mean(dz))
out = {r: np.mean(v, 0).round(3).tolist() for r, v in wyn.items()}
out['tozsamosc_minus_current_koniec'] = float(np.mean(np.array(wyn['tozsamosc'])[:, -1] - np.array(wyn['current'])[:, -1]))
out['tozsamosc_minus_current_ziarna_wyzej'] = int(np.sum(np.array(wyn['tozsamosc'])[:, -1] > np.array(wyn['current'])[:, -1]))
out['przewidziana_zmiana_kotwic_srednia'] = float(np.mean(ds_sr))
out['udzial_kotwic_ze_spadkiem'] = float(np.mean([(1 - z) / 2 for z in znak]))
(W / 'DIAG_SILA.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps(out, indent=1))
