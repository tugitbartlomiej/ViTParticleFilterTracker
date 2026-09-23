#!/usr/bin/env python3
"""Puste kwadraty w przestrzeni tozsamosci narzedzia (pomysl autora, 2026-09-23).

Caly rachunek idzie BEZ etykiet: wycinek bierzemy z ramki PRZEWIDZIANEJ przez detektor.

  1. opis klatki     tokeny z wnetrza przewidzianej ramki na siatce `cells` na `cells`
  2. operator kroku  na parach kolejnych klatek tego samego filmu
  3. mody wolne      te, ktore po klatce wracaja prawie takie same (czesc rzeczywista blisko 1)
  4. odsiew sceny    mod stały przez CALY film to oko, swiatlo i kadr, a nie narzedzie;
                     tozsamosc narzedzia musi sie w filmie ZMIENIAC, bo narzedzia sie wymienia
  5. zajetosc        kazda klatka to punkt w kilku wspolrzednych tozsamosci; dzielimy te
                     przestrzen na kratke i szukamy kwadratow, w ktorych sa klatki z puli,
                     a nie ma ani jednej klatki OZNACZONEJ

Etykiety wchodza dopiero na koncu, wylacznie po to, zeby opisac, czym rozni sie zawartosc
pustych kwadratow od reszty puli.

    py -3.11 experiments/krystalizacja_latentu/zajetosc_tozsamosci.py \\
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \\
        --wybor curated/selection_matrix_stage1/matrix_B2_uniform_time_b1500_sel42.json \\
        --manifest feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl \\
        --out experiments/krystalizacja_latentu/wyniki/REPORT_zajetosc.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from krystalizacja import edmd, pairs_within
from pokrycie_a_bledy import crop_descriptor, wczytaj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--wybor', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--prog-re', type=float, default=0.85, help='prog czesci rzeczywistej dla modu wolnego')
    ap.add_argument('--prog-w-filmie', type=float, default=0.35,
                    help='minimalny udzial zmiennosci WEWNATRZ filmu, zeby mod uznac za tozsamosc')
    ap.add_argument('--osie', type=int, default=3, help='ile wspolrzednych tozsamosci bierzemy')
    ap.add_argument('--kratka', type=int, default=6, help='ile przedzialow na os')
    ap.add_argument('--max-gap', type=int, default=6)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pula = wczytaj(args.pula)
    grid = tuple(pula['meta']['grid'])
    prof = pula['profil_red'].astype(np.float64)
    pudlo = pula['pred_box'].astype(np.float64)
    x = crop_descriptor(prof, pudlo, grid, args.cells)
    ok = np.flatnonzero(np.isfinite(x).all(1) & (pula['score'] > .5))
    fid = [pula['frame_ids'][i] for i in ok]
    vid = np.array([f.split('/')[-2] for f in fid])
    nr = np.array([int(f.split('/')[-1]) for f in fid], float)
    a = x[ok] - x[ok].mean(0)
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    z = a @ (vt[:args.dim].T / np.maximum(sv[:args.dim] / np.sqrt(len(a)), 1e-9))
    par = pairs_within(vid, nr, args.max_gap)
    print(f'klatek z opisem {len(ok)} z {len(x)}, filmow {len(set(vid.tolist()))}, par {len(par)}', flush=True)

    lam, u, mu = edmd(z[par[:, 0]], z[par[:, 1]])
    wolne = np.flatnonzero(lam.real > args.prog_re)
    print(f'modow wolnych (Re powyzej {args.prog_re}): {len(wolne)}', flush=True)

    # ---- odsiew sceny: ile zmiennosci modu dzieje sie WEWNATRZ filmu
    kod = ((z - mu) @ u[:, wolne]).real
    mid = {v: kod[vid == v].mean(0) for v in set(vid.tolist())}
    stat = np.array([mid[v] for v in vid])
    dyn = kod - stat
    udzial_w_filmie = (dyn ** 2).sum(0) / np.maximum((kod - kod.mean(0)) ** 2, 1e-12).sum(0)
    print(f'\n{"mod":>4s} {"Re":>7s} {"udzial w filmie":>16s} {"rola":>12s}')
    role = []
    for k in range(len(wolne)):
        rola = 'tozsamosc' if udzial_w_filmie[k] >= args.prog_w_filmie else 'scena'
        role.append(rola)
        print(f'{k:4d} {lam.real[wolne[k]]:7.3f} {100 * udzial_w_filmie[k]:15.0f}% {rola:>12s}')
    kand = [k for k in range(len(wolne)) if role[k] == 'tozsamosc']
    if len(kand) < args.osie:
        print(f'\nza malo modow tozsamosci ({len(kand)}), biore najbardziej zmienne w filmie', flush=True)
        kand = list(np.argsort(-udzial_w_filmie)[:args.osie])
    osie = kand[:args.osie]
    print(f'\nosie tozsamosci: mody {osie}, Re {np.round(lam.real[wolne[osie]], 3).tolist()}', flush=True)
    c = kod[:, osie]

    # ---- zbior oznaczony
    sel = json.loads(args.wybor.read_text(encoding='utf-8'))
    ids = set(sel.get('selected_frame_ids') or sel['frame_ids'])
    oznaczona = np.array([f in ids for f in fid])
    print(f'klatek oznaczonych w puli z opisem: {int(oznaczona.sum())} z {len(ids)}', flush=True)

    # ---- kratka po kwantylach rozkladu puli
    krawedzie = [np.quantile(c[:, j], np.linspace(0, 1, args.kratka + 1)) for j in range(c.shape[1])]
    idx = np.stack([np.clip(np.searchsorted(krawedzie[j][1:-1], c[:, j]), 0, args.kratka - 1)
                    for j in range(c.shape[1])], 1)
    klucz = np.zeros(len(c), dtype=np.int64)
    for j in range(c.shape[1]):
        klucz = klucz * args.kratka + idx[:, j]

    z_pula = {}
    for k, o in zip(klucz, oznaczona):
        a_, b_ = z_pula.get(k, (0, 0))
        z_pula[k] = (a_ + 1, b_ + int(o))
    zajete = len(z_pula)
    puste = {k: v for k, v in z_pula.items() if v[1] == 0}
    klatek_w_pustych = sum(v[0] for v in puste.values())
    print(f'\nkwadratow w kratce {args.kratka ** c.shape[1]}, zajetych przez pule {zajete}')
    print(f'kwadratow bez ani jednej klatki oznaczonej: {len(puste)}')
    print(f'klatek puli w tych kwadratach: {klatek_w_pustych} '
          f'({100 * klatek_w_pustych / len(c):.1f}% puli)', flush=True)

    w_pustym = np.array([z_pula[k][1] == 0 for k in klucz])

    # ---- czym rozni sie zawartosc pustych kwadratow (etykiety TYLKO do opisu)
    box = np.full((len(fid), 4), np.nan)
    at = {f: i for i, f in enumerate(fid)}
    for line in args.manifest.open(encoding='utf-8'):
        r = json.loads(line)
        j = at.get(r['frame_id'])
        ann = r.get('annotations') or []
        if j is not None and ann:
            box[j] = ann[0]['bbox']
    ma = np.isfinite(box).all(1)
    srodek = pudlo[ok][:, :2] + pudlo[ok][:, 2:] / 2
    predkosc = np.zeros(len(fid))
    for v in np.unique(vid):
        i = np.flatnonzero(vid == v)
        i = i[np.argsort(nr[i])]
        d = np.linalg.norm(np.diff(srodek[i], axis=0), axis=1) / np.maximum(np.diff(nr[i]), 1)
        if len(d):
            predkosc[i[1:]] = d
            predkosc[i[0]] = d[0]
    cechy = {'wydluzenie': np.log(np.maximum(box[:, 2], 1) / np.maximum(box[:, 3], 1)),
             'wielkosc (log px2)': np.log(np.maximum(box[:, 2] * box[:, 3], 1)),
             'predkosc (px/klatke)': predkosc,
             'pewnosc detekcji': pula['score'][ok].astype(float)}
    print(f'\n{"cecha":22s} {"puste kwadraty":>15s} {"reszta puli":>13s}')
    opis = {}
    for nazwa, v in cechy.items():
        m = ma if 'px2' in nazwa or nazwa == 'wydluzenie' else np.ones(len(fid), bool)
        a1, a2 = v[w_pustym & m], v[(~w_pustym) & m]
        opis[nazwa] = dict(puste=float(np.median(a1)), reszta=float(np.median(a2)))
        print(f'{nazwa:22s} {np.median(a1):15.3f} {np.median(a2):13.3f}')

    ile_filmow = len(set(vid[w_pustym].tolist()))
    print(f'\nfilmow dotknietych: {ile_filmow} z {len(set(vid.tolist()))}')
    naj = sorted(puste.items(), key=lambda kv: -kv[1][0])[:8]
    print(f'\nnajwieksze puste kwadraty (klatek puli):')
    for k, (n, _) in naj:
        w = np.flatnonzero(klucz == k)
        print(f'   kwadrat {k:6d}  klatek {n:4d}  filmow {len(set(vid[w].tolist())):3d}  '
              f'przyklad {fid[w[0]]}')

    # ---- rysunek
    out_png = args.out.with_suffix('.png')
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    pary = [(0, 1), (0, 2), (1, 2)]
    for a_, (i, j) in zip(ax, pary):
        a_.scatter(c[~w_pustym, i], c[~w_pustym, j], s=3, c='0.8', label='pula, kwadrat obsluzony')
        a_.scatter(c[w_pustym, i], c[w_pustym, j], s=5, c='#d62728', label='pula, kwadrat PUSTY')
        a_.scatter(c[oznaczona, i], c[oznaczona, j], s=6, c='#1f77b4', label='klatki oznaczone')
        for e in krawedzie[i]:
            a_.axvline(e, color='0.6', lw=.5)
        for e in krawedzie[j]:
            a_.axhline(e, color='0.6', lw=.5)
        a_.set_xlabel(f'tozsamosc {i + 1}'); a_.set_ylabel(f'tozsamosc {j + 1}')
    ax[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'Zajetosc przestrzeni tozsamosci; {len(puste)} kwadratow bez ani jednej klatki oznaczonej, '
                 f'{klatek_w_pustych} klatek puli', y=1.0)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches='tight'); plt.close(fig)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        klatek=len(fid), filmow=len(set(vid.tolist())), modow_wolnych=int(len(wolne)),
        osie=[int(v) for v in osie], Re_osi=[float(lam.real[wolne[k]]) for k in osie],
        udzial_w_filmie=[float(udzial_w_filmie[k]) for k in osie],
        kratka=args.kratka, kwadratow_zajetych=int(zajete), kwadratow_pustych=int(len(puste)),
        klatek_w_pustych=int(klatek_w_pustych), filmow_dotknietych=int(ile_filmow),
        opis_pustych=opis, rysunek=str(out_png)), indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, 'oraz', out_png, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
