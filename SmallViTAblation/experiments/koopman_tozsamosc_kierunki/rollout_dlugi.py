#!/usr/bin/env python3
"""Dlugi rollout skalara tozsamosci a_i(t) wspolnym operatorem predkosci (4 opoznienia) od stanu s_fit do konca historii.

Operator z OPERATORY_K{K}_S{S}.npz (dopasowany do stanu s_fit na obrazach fit), trajektorie a z KIERUNKI_K{K}_S{S}.npz.
Rollout bez prawdziwych stanow posrednich, jak w najlepszy.py (historia 4 ostatnich przyrostow). Miary na holdout:
RMSE wzgl. braku zmian dla horyzontow h, blad prognozy SREDNIEJ po obrazach (przyrost ogolny) w jednostkach rozrzutu
miedzy obrazami w stanie s_fit, oraz to samo dla 'ostatni przyrost' i 'alpha'.

    py -3.11 rollout_dlugi.py --K 160 --s-fit 48
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--K', type=int, default=160)
    ap.add_argument('--s-fit', type=int, default=48)
    ap.add_argument('--out', type=Path, default=HERE / 'wyniki')
    args = ap.parse_args(argv)
    K, S = args.K, args.s_fit
    kier = np.load(args.out / f'KIERUNKI_K{K}_S{S}.npz'); op = np.load(args.out / f'OPERATORY_K{K}_S{S}.npz')
    a, steps = kier['a'], kier['steps']                                                  # 72 x K
    cfg_dir = HERE / 'wyniki' / 'tokeny_pelne' if K == 160 else HERE.parent / 'koopman_gesta_historia/wyniki/macierze_wiele_obrazow'
    rows = json.loads((cfg_dir / 'CONFIG.json').read_text(encoding='utf-8'))['rows']
    hold = np.array([r['role'] != 'fit' for r in rows])
    A, mx, my = op['skalar_lda__wspolny_vel_L4__A'], op['skalar_lda__wspolny_vel_L4__mx'], op['skalar_lda__wspolny_vel_L4__my']
    alpha = op['skalar_lda__alpha']
    V = np.diff(a, axis=1); H = K - 1 - S
    rozrzut = float(a[:, S].std())
    traj = {}
    for m in ('koopman_L4', 'alpha', 'ostatni_przyrost', 'brak_zmian'):
        hist = [V[:, S - 1 - l] for l in range(4)][::-1]
        zz = a[:, S].copy(); vv = V[:, S - 1].copy(); out = [zz.copy()]
        for h in range(H):
            if m == 'koopman_L4':
                Xq = np.stack([hist[-1 - l] for l in range(4)], 1)                     # 72 x 4, najnowszy pierwszy
                vn = (my + (Xq - mx) @ A)[:, 0]
            elif m == 'alpha': vn = alpha * vv
            elif m == 'ostatni_przyrost': vn = vv
            else: vn = np.zeros_like(vv)
            zz = zz + vn; hist.append(vn); vv = vn; out.append(zz.copy())
        traj[m] = np.stack(out, 1)                                                      # 72 x (H+1)
    prawda = a[:, S:S + H + 1]
    horyzonty = sorted(set([h for h in (10, 20, 40, 60, 80, 100) if h <= H] + [H]))
    raport = dict(K=K, s_fit=S, krok_s_fit=int(steps[S]), horyzont_max=H, krok_koncowy=int(steps[S + H]),
                  rozrzut_miedzy_obrazami_w_s_fit=rozrzut, przyrost_sredni_rzeczywisty_do_konca=float((prawda[:, -1] - prawda[:, 0]).mean()),
                  przyrost_sredni_rzeczywisty_do_konca_w_rozrzutach=float((prawda[:, -1] - prawda[:, 0]).mean() / rozrzut), wyniki={})
    for m, T in traj.items():
        r = {}
        for h in horyzonty:
            e = T[hold, h] - prawda[hold, h]; e0 = prawda[hold, 0] - prawda[hold, h]
            r[f'h{h}'] = dict(krok=int(steps[S + h]), rmse_wzgl_braku_zmian=float(np.sqrt(np.mean(e ** 2) / np.mean(e0 ** 2))),
                              blad_sredniej_w_rozrzutach=float((T[:, h] - prawda[:, h]).mean() / rozrzut),
                              przyrost_sredni_prognoza=float((T[:, h] - T[:, 0]).mean()), przyrost_sredni_rzeczywisty=float((prawda[:, h] - prawda[:, 0]).mean()))
        raport['wyniki'][m] = r
    (args.out / f'ROLLOUT_DLUGI_K{K}_S{S}.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'K={K} S={S} (krok {steps[S]}), horyzont {H} krokow do kroku {steps[S + H]}; rozrzut w s_fit {rozrzut:.4f}; '
          f'rzeczywisty przyrost sredni do konca {raport["przyrost_sredni_rzeczywisty_do_konca"]:.4f} '
          f'({raport["przyrost_sredni_rzeczywisty_do_konca_w_rozrzutach"]:.3f} rozrzutu)')
    for m, r in raport['wyniki'].items():
        print(f'{m:17s}', ' '.join(f"h{h}(krok {v['krok']}): {v['rmse_wzgl_braku_zmian']:.3f} | sr {v['blad_sredniej_w_rozrzutach']:+.3f}" for h, v in ((int(k[1:]), v) for k, v in r.items())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
