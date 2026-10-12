"""Analiza dodatkowa po wyniku: czy pod dyfuzją (wagi) i drganiami (latent) jest trwały kierunek dryfu.

Miara bez modelu: cosinus przesunięć w dwóch oknach, które NIE sąsiadują (odstęp co najmniej długości okna). Dla
błądzenia losowego i dla drgań wokół stałego środka wynosi zero; dodatni oznacza wspólny kierunek dryfu, a jego
wartość to w przybliżeniu udział dryfu w energii przesunięcia jednego okna. Okna sąsiednie liczone obok, bo na ich
styku widać krótką pamięć pędu (wagi) albo powrót do środka (latent).

Wagi: z `../wyniki/GRAMY_WAG.npz`. Latent sond: z `LATENT_SOND.npz` biegu gęstej historii.

    py -3.11 dryf_okna.py --wyniki ../wyniki
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kregoslup import BIEG, KROKI, gram

I0 = 30                                                              # krok 300


def il(K, p, q, r, s):
    return K[p, r] - K[p, s] - K[q, r] + K[q, s]


def cos(K, p, q, r, s):
    return float(il(K, p, q, r, s) / np.sqrt(il(K, p, q, p, q) * il(K, r, s, r, s)))


def okna(K):
    T = K.shape[0]
    out = {}
    for w in (5, 10, 20, 35):                                         # 50, 100, 200, 350 kroków
        sas = [cos(K, i + w, i, i + 2 * w, i + w) for i in range(I0, T - 2 * w)]
        odl = [cos(K, i + w, i, j + w, j) for i in range(I0, T - w) for j in range(i + 2 * w, T - w)]
        out[str(10 * w)] = dict(sasiednie=float(np.mean(sas)), n_sasiednich=len(sas),
                                odlegle=(float(np.mean(odl)) if odl else None), n_odleglych=len(odl))
    out['0_300_wobec_300_1000'] = cos(K, I0, 0, T - 1, I0)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki)
    z = np.load(out / 'GRAMY_WAG.npz')
    wyn = dict(meta=dict(opis='cosinus przesunięć w oknach sąsiednich i odległych (odstęp >= długość okna), od kroku 300', kroki=[KROKI[0], KROKI[-1], 10]))
    wyn['wagi'] = {(g.replace('_', '.') if g.startswith('bloki') else g): okna(z[g]) for g in z.files}
    L = np.load(BIEG / 'dane' / 'LATENT_SOND.npz')['latent']
    ob = np.array([i % 16 != 8 for i in range(192)])
    for nazwa, kan in (('latent_narzedzie', 0), ('latent_tlo', 1)):
        X = L[:, ob, kan, :].astype(np.float32).reshape(len(KROKI), -1)
        wyn[nazwa] = okna(gram(X - X[0]))
        # to samo po wygładzeniu: średnia z 5 kolejnych zapisów (50 kroków), co tłumi drgania
        Xs = np.stack([X[max(0, i - 4):i + 1].mean(0) for i in range(len(KROKI))])
        wyn[nazwa + '_wygladzony_50'] = okna(gram(Xs - Xs[0]))
    (out / 'DRYF_OKNA.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    for g in ('cala_siec', 'lodyga', 'bloki.0', 'bloki.4', 'bloki.5', 'norm', 'glowa'):
        d = wyn['wagi'][g]
        print(f'{g:10s}', ' '.join(f"{k}: sas {v['sasiednie']:.3f} odl {v['odlegle'] if v['odlegle'] is None else round(v['odlegle'], 3)}" for k, v in d.items() if k[0] != '0'), '| 0-300/300-1000', round(d['0_300_wobec_300_1000'], 3))
    for g in ('latent_narzedzie', 'latent_narzedzie_wygladzony_50', 'latent_tlo', 'latent_tlo_wygladzony_50'):
        d = wyn[g]
        print(f'{g:30s}', ' '.join(f"{k}: sas {v['sasiednie']:.3f} odl {v['odlegle'] if v['odlegle'] is None else round(v['odlegle'], 3)}" for k, v in d.items() if k[0] != '0'), '| 0-300/300-1000', round(d['0_300_wobec_300_1000'], 3))


if __name__ == '__main__':
    main()
