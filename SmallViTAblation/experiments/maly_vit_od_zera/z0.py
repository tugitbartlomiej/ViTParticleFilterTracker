#!/usr/bin/env python3
"""Bramka Z0 (PLAN.md): czy mały ViT uczy się od zera na 1500 klatkach i w jakim tempie.

Ramię bez dodawania, jedno ziarno, `--kroki` aktualizacji; co `--co` kroków mAP@0,5 na próbce val (3000 klatek)
i jakość sond, co 10 kroków latent sond. Wynik: wyniki/z0_s{S}/Z0.json, LATENT.npz, wagi co 1000 kroków.
Z tego wynika długość obrotu pętli (kroki na obrót).

    <venv rfdetr>/python -u z0.py --ziarno 42 --kroki 6000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import dane as D
import trening as T

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ziarno', type=int, default=42)
    ap.add_argument('--kroki', type=int, default=6000)
    ap.add_argument('--co', type=int, default=250)
    a = ap.parse_args(argv)
    out = HERE / 'wyniki' / f'z0_s{a.ziarno}'; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    zb = D.baza(); wal_w, wal_X = D.walidacja(); sondy = D.sondy()
    print(f'dane: baza {len(zb)}, val {len(wal_w)}, sondy {len(sondy[0])} ({time.time() - t0:.0f} s)', flush=True)
    u = T.Uczen(a.ziarno)
    print('parametry', sum(p.numel() for p in u.net.parameters()), flush=True)
    oceny, lat = [], []

    def ocena(u_):
        t1 = time.time()
        ap_ = T.ap50(u_.wykryj(wal_X), [r['ramki'] for r in wal_w])
        q = u_.jakosc_sond(sondy)
        ob = sondy[4]
        oceny.append(dict(krok=u_.krok, mAP50=ap_, q_sond=float(np.nanmean(q[ob])), q_puste=float(np.nanmean(q[~ob])),
                          sekundy=time.time() - t0))
        print(f'krok {u_.krok}: mAP50 {ap_:.4f}  q sond {oceny[-1]["q_sond"]:.3f}  '
              f'strata {u_.log[-1]["strata"] if u_.log else float("nan"):.3f}  ({time.time() - t1:.1f} s oceny, '
              f'{time.time() - t0:.0f} s razem)', flush=True)
        (out / 'Z0.json').write_text(json.dumps(dict(ziarno=a.ziarno, oceny=oceny, log=u_.log), indent=1), encoding='utf-8')
        if u_.krok % 1000 == 0:
            torch.save(u_.net.state_dict(), out / f'wagi_{u_.krok:06d}.pth')

    rng = np.random.default_rng(a.ziarno)
    lat.append((0, u.latent_sond(sondy)))
    ocena(u)
    u.kroki(zb, a.kroki, rng, sondy=sondy, latent_out=lat, co_ocena=a.co, ocena=ocena)
    np.savez_compressed(out / 'LATENT.npz', kroki=np.array([k for k, _ in lat]), latent=np.stack([l for _, l in lat]))
    print(f'gotowe {time.time() - t0:.0f} s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
