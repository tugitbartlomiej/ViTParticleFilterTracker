#!/usr/bin/env python3
"""Koopman i Langevin na wektorze tożsamości wyznaczonym po 21 epokach (metoda użytkownika).

Wektor w (768) i standaryzacja (średnia, sd negatywów) są STAŁE z kroku 1000 (`wektor_tozsamosci_21ep.py`, LDA na
wszystkich wersjach sond). Stan w chwili t: rzut latentu narzędzia każdej ze 180 sond na w, czyli jedna liczba na
sondę — „jak mocno sonda zapala tożsamość”. Bloki po 5 zapisów (punkt co 50 kroków), kroki 50–4500: rozgrzewka
(do 1000) i dalej każde z ramion `koopman`, `losowe`, `bez`.

Prognoza (funkcje z `kod/prognoza.py`, wspólne współczynniki dla sond):
  koopman   Δs_{t+1} = Σ_j a_j Δs_{t−j}                (przyrosty z 4 opóźnieniami)
  langevin  Δs_{t+1} = k·s_t + Σ_j b_j Δs_{t−j} + c_p   (hamowanie k do punktu stałego sondy c_p/−k, plus pamięć)
Chwile startu od kroku 1000 co 250, horyzonty 1, 5, 10 bloków (50, 250, 500 kroków). Porównanie z „brakiem zmiany”
dla sond, dla średnich 15 wersji i dla siły tożsamości (średnia po sondach).

    py -3.11 koopman_na_wektorze.py  ->  wyniki_eden/KOOPMAN_WEKTOR.json, KOOPMAN_WEKTOR.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE / 'wyniki_eden'
sys.path.insert(0, str(HERE / 'kod'))
sys.path.insert(0, str(HERE))
import prognoza as PR  # noqa: E402
import tozsamosc as TZ  # noqa: E402
from wektor_tozsamosci_21ep import PROBE, naucz, zbior  # noqa: E402

RAMIONA = ('bez', 'losowe', 'koopman')
POCZATKI = list(range(1000, 4500, 250))
HORYZONTY = (1, 5, 10)


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def bloki(d, ramie):
    pl = [d / 'wspolny' / 'k01' / 'LATENT.npz', d / 'wspolny' / 'k02' / 'LATENT.npz'] + \
         [d / ramie / f'k{k:02d}' / 'LATENT.npz' for k in range(3, 10)]
    zz = [np.load(p) for p in pl]
    kroki = np.concatenate([z['kroki'] for z in zz]); L = np.concatenate([z['latent'] for z in zz])
    assert kroki[0] == 0 and np.all(np.diff(kroki) == 10) and kroki[-1] == 4500
    L = L[1:]
    nb = len(L) // PR.B_BLOK
    return L[:nb * PR.B_BLOK].astype(np.float32).reshape(nb, PR.B_BLOK, *L.shape[1:]).mean(1)   # (90,192,2,768)


def main() -> int:
    man = json.loads((PROBE / 'manifest.json').read_text(encoding='utf-8'))
    obecn = np.array([bool(m['semantic_presence']) for m in man])
    war = np.array([m['variant'] for m in man]); zrodlo = np.array([m['source_file'] for m in man])
    wz = war[obecn]
    wyn, krzywe = {}, {}
    for d in sorted(W.glob('petla_s*')):
        if not all((d / r / f'k{k:02d}' / 'LATENT.npz').exists() for r in RAMIONA for k in range(3, 10)):
            print(d.name, 'pominięte: brak latentów ramion'); continue
        L21 = np.load(d / 'wspolny' / 'k02' / 'LATENT.npz')['latent'][-5:].astype(np.float64).mean(0)
        m = naucz(*zbior(L21, obecn, war, zrodlo, np.ones(len(war), bool))[:2])          # wektor z kroku 1000
        wyn[d.name] = {}
        for ramie in RAMIONA:
            Lb = bloki(d, ramie)
            S = ((Lb[:, obecn, 0] - m['mu']) / m['sd']) @ m['w']                        # (90, 180) rzuty narzędzia
            T_ = ((Lb[:, :, 1] - m['mu']) / m['sd']) @ m['w']                           # (90, 192) rzuty tła
            krzywe.setdefault(ramie, {})[d.name] = dict(
                wersje=np.stack([S[:, wz == v].mean(1) for v in TZ.WARIANTY_Z], 1), tlo=T_.mean(1), prog=m['prog'])
            Y = S[:, :, None]                                                          # (T, P=180, M=1)
            rr = []
            for t in POCZATKI:
                b0 = t // 50 - 1
                hist = Y[:b0 + 1]
                Wm = {mod: PR.dopasuj(hist, mod) for mod in ('koopman', 'langevin')}
                for H in HORYZONTY:
                    if b0 + H >= len(Y):
                        continue
                    real, teraz = Y[b0 + H, :, 0], hist[-1, :, 0]
                    r = dict(krok=t, H=H, hamowanie_k=float(Wm['langevin'][0, 0]))
                    for mod, Wt in Wm.items():
                        p = PR.rollout(hist, Wt, H, mod)[-1, :, 0]
                        pv = np.array([p[wz == v].mean() for v in TZ.WARIANTY_Z]); rv = np.array([real[wz == v].mean() for v in TZ.WARIANTY_Z])
                        tv = np.array([teraz[wz == v].mean() for v in TZ.WARIANTY_Z])
                        r[mod] = dict(sondy=rmse(p - real) / max(rmse(teraz - real), 1e-12),
                                      wersje=rmse(pv - rv) / max(rmse(tv - rv), 1e-12),
                                      sila=abs(p.mean() - real.mean()) / max(abs(teraz.mean() - real.mean()), 1e-12),
                                      znak_sily=bool(np.sign(p.mean() - teraz.mean()) == np.sign(real.mean() - teraz.mean())))
                    rr.append(r)
            wyn[d.name][ramie] = dict(prognozy=rr, sila_1000=float(S[19].mean()), sila_4500=float(S[-1].mean()),
                                      prog=m['prog'], tlo_4500=float(T_[-1].mean()))
        print(d.name, {r: round(wyn[d.name][r]['sila_4500'] - wyn[d.name][r]['sila_1000'], 2) for r in RAMIONA}, flush=True)
    # podsumowanie
    sr = {}
    for ramie in RAMIONA:
        sr[ramie] = {}
        for H in HORYZONTY:
            for mod in ('koopman', 'langevin'):
                for co in ('sondy', 'wersje', 'sila'):
                    v = [[r[mod][co] for r in wyn[s][ramie]['prognozy'] if r['H'] == H] for s in wyn]
                    A = np.array(v)                                                    # (ziarna, chwile)
                    sr[ramie][f'H{H * 50}_{mod}_{co}'] = dict(mediana=float(np.median(A)), ponizej_1=float((A < 1).mean()),
                                                              mediana_po_chwilach=np.median(A, 0).round(3).tolist())
                znaki = [r[mod]['znak_sily'] for s in wyn for r in wyn[s][ramie]['prognozy'] if r['H'] == H]
                sr[ramie][f'H{H * 50}_{mod}_znak_sily'] = float(np.mean(znaki))
        sr[ramie]['hamowanie_k_mediana'] = float(np.median([r['hamowanie_k'] for s in wyn for r in wyn[s][ramie]['prognozy']]))
        sr[ramie]['sila_1000'] = float(np.mean([wyn[s][ramie]['sila_1000'] for s in wyn]))
        sr[ramie]['sila_4500'] = float(np.mean([wyn[s][ramie]['sila_4500'] for s in wyn]))
    wyn['_srednio'] = sr
    (W / 'KOOPMAN_WEKTOR.json').write_text(json.dumps(wyn, indent=1, ensure_ascii=False), encoding='utf-8')
    for ramie in RAMIONA:
        x = sr[ramie]
        print(f"\n{ramie}: siła tożsamości {x['sila_1000']:.2f} (krok 1000) -> {x['sila_4500']:.2f} (krok 4500); "
              f"hamowanie k mediana {x['hamowanie_k_mediana']:.4f}")
        for H in HORYZONTY:
            print(f"  H={H * 50:3d} kroków: " + '  '.join(
                f"{mod} {co} {x[f'H{H * 50}_{mod}_{co}']['mediana']:.3f} ({x[f'H{H * 50}_{mod}_{co}']['ponizej_1']:.0%}<1)"
                for mod in ('koopman', 'langevin') for co in ('sondy', 'wersje', 'sila'))
                + f"  | znak siły K {x[f'H{H * 50}_koopman_znak_sily']:.2f} L {x[f'H{H * 50}_langevin_znak_sily']:.2f}")
        print('  H=500, Koopman, sondy, mediana po chwilach 1000..4000:', x['H500_koopman_sondy']['mediana_po_chwilach'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.5), constrained_layout=True)
    kr = 50 * np.arange(1, 91) / (1500 / 32)
    cm = plt.get_cmap('tab20')
    for a, ramie in zip(ax, RAMIONA):
        V = np.stack([c['wersje'] for c in krzywe[ramie].values()])                  # (ziarna, 90, 15)
        for j, v in enumerate(TZ.WARIANTY_Z):
            a.plot(kr, np.median(V[:, :, j], 0), color=cm(j), lw=2 if v in ('black', 'yellow', 'original') else 1, label=v)
        a.plot(kr, np.median(np.stack([c['tlo'] for c in krzywe[ramie].values()]), 0), 'k--', label='tło')
        a.axvline(1000 / (1500 / 32), color='gray', ls=':'); a.axhline(0, color='gray', lw=.5)
        a.set_title(f'ramię {ramie}: siła wektora tożsamości (z 21 epok) na sondach, mediana 10 ziaren', fontsize=9)
        a.set_xlabel('epoka bazy'); a.set_ylabel('rzut na wektor tożsamości')
    ax[0].legend(fontsize=6, ncol=2)
    fig.savefig(W / 'KOOPMAN_WEKTOR.png', dpi=110)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
