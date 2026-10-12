"""Analiza dodatkowa po wyniku: te same miary dla błądzenia losowego i podział przesunięcia wag na dryf i dyfuzję.

Po co: próg K1 z PLAN.md (pierwszy kierunek ≥ 0,70) leży poniżej wartości, jaką daje zwykłe błądzenie losowe w wielu
wymiarach (8/π² ≈ 0,81 dla przesunięć względem startu bez centrowania; por. Antognini i Sohl-Dickstein 2018 dla
wersji centrowanej), więc sam nie odróżnia kręgosłupa od dyfuzji. Tu liczę wartości odniesienia i dopasowuję prosty
model „stały dryf + błądzenie z pamięcią jednego zapisu”.

Wszystko z `../wyniki/GRAMY_WAG.npz` (macierze iloczynów przesunięć), bez czytania wag.

    py -3.11 porownanie_z_bladzeniem.py --wyniki ../wyniki
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kregoslup import KROKI, miary

I0 = 30                                                              # krok 300


def gram_bladzenia(n, ro):
    """Oczekiwana macierz iloczynów położeń błądzenia o n przyrostach z korelacją ro sąsiednich przyrostów (wysoki wymiar)."""
    C = np.eye(n) + ro * (np.eye(n, k=1) + np.eye(n, k=-1))
    L = np.tril(np.ones((n, n)))
    K = np.zeros((n + 1, n + 1))
    K[1:, 1:] = L @ C @ L.T
    return K


def wzgledem(K, i0):
    return (K - K[:, [i0]] - K[[i0], :] + K[i0, i0])[i0:, i0:]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--wyniki', required=True); arg = p.parse_args()
    out = Path(arg.wyniki)
    z = np.load(out / 'GRAMY_WAG.npz')
    stare = json.loads((out / 'KREGOSLUP.json').read_text(encoding='utf-8'))
    n = len(KROKI) - 1 - I0
    wyn = dict(meta=dict(opis='porównanie z błądzeniem losowym; od kroku 300 do 1000, zapisy co 10 kroków', przyrostow=n))
    for ro in (0.0, 0.3):
        m = miary(gram_bladzenia(n, ro), 0); m.pop('postep')
        wyn[f'bladzenie_ro_{ro}'] = m
    a = np.arange(n + 1)[:, None] * np.arange(n + 1)[None, :]       # iloczyn numerów zapisów: wkład stałego dryfu
    for g in z.files:
        S = wzgledem(z[g], I0)
        nazwa = g.replace('_', '.') if g.startswith('bloki') else g
        ro = stare['wagi'][nazwa]['300']['autokorelacja_przyrostow']['10']
        Kb = gram_bladzenia(n, ro)
        iu = np.triu_indices(n + 1, 1)
        X = np.stack([a[iu], Kb[iu]], 1).astype(np.float64)
        (d2, s2), *_ = np.linalg.lstsq(X, S[iu], rcond=None)
        dop = X @ np.array([d2, s2])
        r2 = 1 - ((S[iu] - dop) ** 2).sum() / ((S[iu] - S[iu].mean()) ** 2).sum()
        wyn[nazwa] = dict(ro_przyrostow=ro, dryf2_na_przyrost=float(d2), dyfuzja2_na_przyrost=float(s2), stosunek=float(d2 / s2),
                          udzial_dryfu_w_przesunieciu_netto=float(d2 * n * n / (d2 * n * n + s2 * Kb[n, n])), r2_dopasowania=float(r2),
                          pc1=stare['wagi'][nazwa]['300']['udzial_pc1'], netto_do_drogi=stare['wagi'][nazwa]['300']['netto_do_drogi'],
                          okno100=stare['wagi'][nazwa]['300']['cosinus_sasiednich_okien']['100']['mediana'])
    (out / 'POROWNANIE_Z_BLADZENIEM.json').write_text(json.dumps(wyn, ensure_ascii=False, indent=1), encoding='utf-8')
    for k in ('bladzenie_ro_0.0', 'bladzenie_ro_0.3'):
        m = wyn[k]
        print(k, 'PC1', round(m['udzial_pc1'], 3), 'PC2', round(m['udzial_pc2'], 3), 'netto/droga', round(m['netto_do_drogi'], 3),
              'okno50', round(m['cosinus_sasiednich_okien']['50']['mediana'], 3), 'okno100', round(m['cosinus_sasiednich_okien']['100']['mediana'], 3),
              'postep+', round(m['postep_udzial_dodatnich'], 2), 'do90', m['kierunkow_do_90'])
    for g in ('cala_siec', 'lodyga', 'bloki.0', 'bloki.1', 'bloki.2', 'bloki.3', 'bloki.4', 'bloki.5', 'norm', 'glowa'):
        m = wyn[g]
        print(f'{g:10s} ro {m["ro_przyrostow"]:.2f} dryf2/dyf2 {m["stosunek"]:.4f} udzial dryfu {m["udzial_dryfu_w_przesunieciu_netto"]:.3f} R2 {m["r2_dopasowania"]:.3f} '
              f'PC1 {m["pc1"]:.3f} netto/droga {m["netto_do_drogi"]:.3f} okno100 {m["okno100"]:.3f}')


if __name__ == '__main__':
    main()
