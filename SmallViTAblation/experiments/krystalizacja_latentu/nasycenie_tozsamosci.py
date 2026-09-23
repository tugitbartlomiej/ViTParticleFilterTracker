#!/usr/bin/env python3
"""Ile klatek wystarczy, zeby domknac tozsamosc narzedzia (pytanie autora, 2026-09-23).

Nie pytamy, czy dwa zbiory sie pokrywaja. Pytamy, ile klatek trzeba wziac z puli, zeby
przestrzen tozsamosci byla juz obstawiona, i czy da sie to zrobic mniejsza liczba klatek
niz dobor rownomierny w czasie.

  wspolrzedne tozsamosci   mody operatora kroku, ktore trwaja w czasie, ale ZMIENIAJA sie
                           w filmie (mod staly przez caly film to oko i kadr, nie narzedzie)
  kratka                   te wspolrzedne dzielimy na przedzialy, kazda klatka wpada w jeden kwadrat
  pokrycie                 jaki udzial ZAJETYCH kwadratow ma choc jedna klatke wybrana
  masa                     jaki udzial klatek puli lezy w kwadratach juz obstawionych

Porownywane sposoby wyboru przy tym samym budzecie:
  rownomiernie w czasie    co n-ta klatka w kazdym filmie (tak zrobiono ramie B2)
  losowo                   podloga
  po kwadratach            bierzemy po jednej klatce z kwadratu, zaczynajac od najliczniejszych
  najdalsza klatka         zachlanne k-centrum w tych samych wspolrzednych (kontrola)

Werdykt to budzet, przy ktorym kazdy sposob osiaga 95 procent pokrycia.

    py -3.11 experiments/krystalizacja_latentu/nasycenie_tozsamosci.py \\
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \\
        --out experiments/krystalizacja_latentu/wyniki/REPORT_nasycenie.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from krystalizacja import edmd, pairs_within
from pokrycie_a_bledy import crop_descriptor, wczytaj


def wybor_rownomierny(vid: np.ndarray, nr: np.ndarray, budzet: int) -> np.ndarray:
    """Co n-ta klatka w kazdym filmie, z kwota proporcjonalna do dlugosci filmu."""
    out = []
    filmy = np.unique(vid)
    kwota = {v: max(1, round(budzet * (vid == v).sum() / len(vid))) for v in filmy}
    for v in filmy:
        i = np.flatnonzero(vid == v)
        i = i[np.argsort(nr[i])]
        k = min(kwota[v], len(i))
        out.extend(i[np.linspace(0, len(i) - 1, k).round().astype(int)])
    out = list(np.unique(out))
    if len(out) < budzet:                       # zaokraglenia kwot zostawiaja niedobor; dobieramy reszte
        brak = np.setdiff1d(np.arange(len(vid)), np.array(out, dtype=int))
        kolejnosc = np.lexsort((nr[brak], vid[brak]))
        dobor = brak[kolejnosc][np.linspace(0, len(brak) - 1, budzet - len(out)).round().astype(int)]
        out = list(np.unique(np.concatenate([np.array(out, dtype=int), dobor])))
    return np.array(out[:budzet], dtype=int)


def wybor_po_kwadratach(klucz: np.ndarray, budzet: int, seed: int = 0) -> np.ndarray:
    """Po jednej klatce z kwadratu, od najliczniejszych; potem druga runda i tak dalej."""
    rng = np.random.default_rng(seed)
    gniazda: dict[int, list[int]] = {}
    for i, k in enumerate(klucz):
        gniazda.setdefault(int(k), []).append(i)
    kolejnosc = sorted(gniazda, key=lambda k: -len(gniazda[k]))
    for k in kolejnosc:
        rng.shuffle(gniazda[k])
    out, runda = [], 0
    while len(out) < budzet:
        dodano = False
        for k in kolejnosc:
            if runda < len(gniazda[k]):
                out.append(gniazda[k][runda])
                dodano = True
                if len(out) >= budzet:
                    break
        if not dodano:
            break
        runda += 1
    return np.array(out)


def kcentrum(c: np.ndarray, budzet: int, seed: int = 0) -> np.ndarray:
    """Zachlannie bierzemy klatke najdalsza od wszystkiego, co juz wybrane."""
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(c)))
    wybrane = [start]
    d = np.linalg.norm(c - c[start], axis=1)
    while len(wybrane) < budzet:
        j = int(np.argmax(d))
        wybrane.append(j)
        d = np.minimum(d, np.linalg.norm(c - c[j], axis=1))
    return np.array(wybrane)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--prog-re', type=float, default=0.85)
    ap.add_argument('--prog-w-filmie', type=float, default=0.35)
    ap.add_argument('--osie', type=int, default=3)
    ap.add_argument('--kratka', type=int, default=16)
    ap.add_argument('--max-gap', type=int, default=6)
    ap.add_argument('--budzety', type=int, nargs='+', default=[100, 250, 500, 1000, 1500, 2000, 3000, 5000])
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pula = wczytaj(args.pula)
    grid = tuple(pula['meta']['grid'])
    prof = pula['profil_red'].astype(np.float64)
    x = crop_descriptor(prof, pula['pred_box'].astype(np.float64), grid, args.cells)
    ok = np.flatnonzero(np.isfinite(x).all(1) & (pula['score'] > .5))
    fid = [pula['frame_ids'][i] for i in ok]
    vid = np.array([f.split('/')[-2] for f in fid])
    nr = np.array([int(f.split('/')[-1]) for f in fid], float)
    a = x[ok] - x[ok].mean(0)
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    z = a @ (vt[:args.dim].T / np.maximum(sv[:args.dim] / np.sqrt(len(a)), 1e-9))
    par = pairs_within(vid, nr, args.max_gap)
    lam, u, mu = edmd(z[par[:, 0]], z[par[:, 1]])
    wolne = np.flatnonzero(lam.real > args.prog_re)
    kod = ((z - mu) @ u[:, wolne]).real
    mid = {v: kod[vid == v].mean(0) for v in set(vid.tolist())}
    dyn = kod - np.array([mid[v] for v in vid])
    w_filmie = (dyn ** 2).sum(0) / np.maximum(((kod - kod.mean(0)) ** 2).sum(0), 1e-12)
    kand = [k for k in range(len(wolne)) if w_filmie[k] >= args.prog_w_filmie]
    if len(kand) < args.osie:
        kand = list(np.argsort(-w_filmie)[:args.osie])
    osie = kand[:args.osie]
    c = kod[:, osie]
    print(f'klatek {len(fid)}, filmow {len(set(vid.tolist()))}, osie tozsamosci {osie}, '
          f'Re {np.round(lam.real[wolne[osie]], 3).tolist()}', flush=True)

    krawedzie = [np.quantile(c[:, j], np.linspace(0, 1, args.kratka + 1)) for j in range(c.shape[1])]
    klucz = np.zeros(len(c), dtype=np.int64)
    for j in range(c.shape[1]):
        klucz = klucz * args.kratka + np.clip(np.searchsorted(krawedzie[j][1:-1], c[:, j]), 0, args.kratka - 1)
    zajete, licznosc = np.unique(klucz, return_counts=True)
    masa = {int(k): int(n) for k, n in zip(zajete, licznosc)}
    print(f'kwadratow zajetych {len(zajete)} z {args.kratka ** c.shape[1]}, '
          f'najliczniejszy ma {licznosc.max()} klatek', flush=True)

    rng = np.random.default_rng(0)
    sposoby = {
        'rownomiernie w czasie': lambda b: wybor_rownomierny(vid, nr, b),
        'losowo': lambda b: rng.choice(len(c), min(b, len(c)), replace=False),
        'po kwadratach': lambda b: wybor_po_kwadratach(klucz, b),
        'najdalsza klatka': lambda b: kcentrum(c, min(b, len(c))),
    }
    raport = {'klatek': len(fid), 'osie': [int(v) for v in osie], 'kratka': args.kratka,
              'kwadratow_zajetych': int(len(zajete)), 'krzywe': {}}
    print(f'\n{"budzet":>7s} ' + ' '.join(f'{k:>24s}' for k in sposoby))
    krzywe = {k: [] for k in sposoby}
    for b in args.budzety:
        linia = []
        for nazwa, fn in sposoby.items():
            sel = fn(b)
            trafione = set(klucz[sel].tolist())
            pokrycie = len(trafione) / len(zajete)
            m = sum(masa[k] for k in trafione) / len(c)
            krzywe[nazwa].append((b, pokrycie, m, len(sel)))
            linia.append(f'{100 * pokrycie:9.1f}% / {100 * m:6.1f}% masy')
        print(f'{b:7d} ' + ' '.join(f'{v:>24s}' for v in linia), flush=True)
    for nazwa, v in krzywe.items():
        raport['krzywe'][nazwa] = [dict(budzet=b, pokrycie=p, masa=m, wybranych=n) for b, p, m, n in v]

    print(f'\nbudzet potrzebny do 95 procent pokrytych kwadratow:')
    raport['budzet_95'] = {}
    for nazwa, v in krzywe.items():
        trafi = [b for b, p, _, _ in v if p >= .95]
        raport['budzet_95'][nazwa] = int(trafi[0]) if trafi else None
        print(f'   {nazwa:24s} {trafi[0] if trafi else "nie osiagniete"}')

    out_png = args.out.with_suffix('.png')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for nazwa, v in krzywe.items():
        b = [q[0] for q in v]
        ax[0].plot(b, [100 * q[1] for q in v], '-o', ms=4, label=nazwa)
        ax[1].plot(b, [100 * q[2] for q in v], '-o', ms=4, label=nazwa)
    for a_, tyt in zip(ax, ('udzial zajetych kwadratow, ktore sa obstawione',
                            'udzial klatek puli w kwadratach obstawionych')):
        a_.set_xlabel('budzet klatek'); a_.set_ylabel('procent'); a_.set_title(tyt, fontsize=10)
        a_.axhline(95, color='0.6', lw=.8, ls='--'); a_.grid(alpha=.3); a_.legend(fontsize=8)
    fig.suptitle(f'Nasycenie przestrzeni tozsamosci, {len(fid)} klatek puli, kratka {args.kratka}', y=1.0)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches='tight'); plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raport['rysunek'] = str(out_png)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, 'oraz', out_png, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
