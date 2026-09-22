#!/usr/bin/env python3
"""Czy film walidacyjny gorzej pokryty przez wybrane klatki uczace ma wiecej pudel (2026-09-22).

Mamy skonczone treningi czterech sposobow wyboru klatek (B1 losowo, B2 rowno w czasie,
B3 pokrycie, K4 pokrycie w modach Koopmana), kazdy na trzech ziarnach, ten sam budzet 1500.
Wybrane klatki sa dla danego sposobu IDENTYCZNE we wszystkich ziarnach, wiec ziarna sluza tu
tylko do oszacowania szumu.

Dla kazdej pary (sposob, film walidacyjny) liczymy:
  pokrycie      mediana odleglosci klatki tego filmu do NAJBLIZSZEJ wybranej klatki uczacej
  pudla         udzial klatek tego filmu, na ktorych najlepszy checkpoint nie trafil narzedzia

Pokrycie liczymy w kilku przestrzeniach o tej samej liczbie liczb, zeby zobaczyc, czy
przestrzen Koopmana mowi wiecej niz zwykle kierunki najwiekszej zmiennosci:
  wolne mody       wspolrzedne w najtrwalszych modach operatora czasu
  skladowe glowne  kierunki najwiekszej zmiennosci (kontrola)
  kierunki losowe  podloga
  pelny opis       wszystkie wspolrzedne wycinka (sufit)

Obie nagrania czastek pochodza z TEGO SAMEGO checkpointu, ale maja wlasne bazy, wiec opis
walidacyjny przepisujemy do bazy puli (sprawdzamy, ile przy tym ginie).

Zaden krok nie uzywa etykiet: wycinek bierzemy z ramki PRZEWIDZIANEJ przez detektor.

    py -3.11 experiments/krystalizacja_latentu/pokrycie_a_bledy.py \
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \
        --walidacja results/koopman_langevin_epochs/czastki/CZASTKI_b0500 \
        --wybory curated/selection_matrix_stage1 \
        --pudla docs/Analiza_ponownej_publikacji/evidence_selection_stage1_2026-09-21/MISSES_PER_VIDEO_best_val.json \
        --out experiments/krystalizacja_latentu/wyniki/REPORT_pokrycie_a_bledy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from krystalizacja import edmd, pairs_within


def crop_descriptor(prof: np.ndarray, box: np.ndarray, grid: tuple[int, int], cells: int) -> np.ndarray:
    h, w = grid
    ys = (np.arange(h) + .5) / h * 480
    xs = (np.arange(w) + .5) / w * 720
    gy, gx = np.meshgrid(ys, xs, indexing='ij')
    gy, gx = gy.ravel(), gx.ravel()
    k = prof.shape[-1]
    out = np.full((len(prof), cells * cells * k), np.nan, dtype=np.float32)
    for t in range(len(prof)):
        x0, y0, bw, bh = box[t]
        if not np.isfinite(box[t]).all() or bw < 8 or bh < 8:
            continue
        u, v = (gx - x0) / bw, (gy - y0) / bh
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        if inside.sum() < 4:
            continue
        cell = (np.clip((v[inside] * cells).astype(int), 0, cells - 1) * cells
                + np.clip((u[inside] * cells).astype(int), 0, cells - 1))
        acc = np.zeros((cells * cells, k))
        cnt = np.zeros(cells * cells)
        np.add.at(acc, cell, prof[t][inside])
        np.add.at(cnt, cell, 1)
        empty = cnt == 0
        acc[~empty] /= cnt[~empty, None]
        if empty.any():
            acc[empty] = acc[~empty].mean(0)
        out[t] = acc.ravel()
    return out


def wczytaj(katalog: Path) -> dict:
    with np.load(katalog / 'czesc000.npz', allow_pickle=False) as d:
        rec = {k: d[k] for k in ('profil_red', 'pred_box', 'score', 'maska_pred', 'frame_ids')}
    with np.load(katalog / 'baza.npz', allow_pickle=False) as d:
        rec['baza'] = {k: d[k] for k in d.files}
    rec['meta'] = json.loads((katalog / 'meta.json').read_text(encoding='utf-8'))
    rec['frame_ids'] = [str(f) for f in rec['frame_ids'].tolist()]
    return rec


def do_bazy(rec: dict, cel: dict) -> np.ndarray:
    """Opis nagrania `rec` przepisany do bazy nagrania `cel` (ta sama siec, inne klatki)."""
    m = rec['baza']['kier'].T @ cel['baza']['kier']
    przesun = (rec['baza']['srodek'] - cel['baza']['srodek']) @ cel['baza']['kier']
    return rec['profil_red'].astype(np.float64) @ m + przesun


def najblizsza(a: np.ndarray, b: np.ndarray, krok: int = 2048) -> np.ndarray:
    """Odleglosc kazdego wiersza a do najblizszego wiersza b."""
    out = np.empty(len(a))
    bb = (b ** 2).sum(1)
    for i in range(0, len(a), krok):
        blok = a[i:i + krok]
        d = (blok ** 2).sum(1)[:, None] + bb[None, :] - 2 * blok @ b.T
        out[i:i + krok] = np.sqrt(np.clip(d.min(1), 0, None))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--walidacja', type=Path, required=True)
    ap.add_argument('--wybory', type=Path, required=True)
    ap.add_argument('--pudla', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cells', type=int, default=3)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--modes', type=int, default=8)
    ap.add_argument('--max-gap', type=int, default=6)
    args = ap.parse_args(argv)

    pula, wal = wczytaj(args.pula), wczytaj(args.walidacja)
    grid = tuple(pula['meta']['grid'])
    s = np.linalg.svd(pula['baza']['kier'].T @ wal['baza']['kier'], compute_uv=False)
    print(f'zgodnosc baz obu nagran: srednia {s.mean():.3f}, najgorszy kierunek {s.min():.3f}', flush=True)

    prof_p = pula['profil_red'].astype(np.float64)
    prof_w = do_bazy(wal, pula)
    x_p = crop_descriptor(prof_p, pula['pred_box'].astype(np.float64), grid, args.cells)
    x_w = crop_descriptor(prof_w, wal['pred_box'].astype(np.float64), grid, args.cells)
    ok_p = np.flatnonzero(np.isfinite(x_p).all(1) & (pula['score'] > .5))
    ok_w = np.flatnonzero(np.isfinite(x_w).all(1) & (wal['score'] > .5))
    print(f'klatek puli z opisem {len(ok_p)} z {len(x_p)}, walidacyjnych {len(ok_w)} z {len(x_w)}', flush=True)

    sr = x_p[ok_p].mean(0)
    a = x_p[ok_p] - sr
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    baza = vt[:args.dim].T / np.maximum(sv[:args.dim] / np.sqrt(len(a)), 1e-9)
    z_p = a @ baza
    z_w = (x_w[ok_w] - sr) @ baza

    fid_p = [pula['frame_ids'][i] for i in ok_p]
    fid_w = [wal['frame_ids'][i] for i in ok_w]
    vid_w = np.array([f.split('/')[-2] for f in fid_w])
    v_all = np.array([f.split('/')[-2] for f in fid_p])
    f_all = np.array([int(f.split('/')[-1]) for f in fid_p], float)
    par = pairs_within(v_all, f_all, args.max_gap)
    lam, u, _ = edmd(z_p[par[:, 0]], z_p[par[:, 1]])
    b_wolne = u[:, :args.modes].real
    b_wolne = b_wolne / np.maximum(np.linalg.norm(b_wolne, axis=0, keepdims=True), 1e-9)
    rng = np.random.default_rng(0)
    b_los = rng.normal(size=(args.dim, args.modes))
    b_los /= np.linalg.norm(b_los, axis=0, keepdims=True)
    b_pca = np.eye(args.dim)[:, :args.modes]
    print(f'par w puli {len(par)}, Re najwolniejszych modow '
          f'{np.round(lam.real[:args.modes], 3).tolist()}', flush=True)

    przestrzenie = {}
    for nazwa, b in (('wolne mody', b_wolne), ('skladowe glowne', b_pca), ('kierunki losowe', b_los)):
        kp, kw = z_p @ b, z_w @ b
        sd = np.maximum(kp.std(0), 1e-9)
        przestrzenie[nazwa] = (kp / sd, kw / sd)
    przestrzenie['pelny opis'] = (z_p / np.maximum(z_p.std(0), 1e-9), z_w / np.maximum(z_p.std(0), 1e-9))

    pudla = json.loads(args.pudla.read_text(encoding='utf-8'))['runs']
    ramiona: dict[str, list[str]] = {}
    for run in pudla:
        ramiona.setdefault(run.split('_b1500')[0].replace('matrix_', ''), []).append(run)
    print(f'ramion {len(ramiona)}: {", ".join(sorted(ramiona))}', flush=True)

    pos = {f: i for i, f in enumerate(fid_p)}
    wybor = {}
    for ramie in ramiona:
        plik = next(args.wybory.glob(f'matrix_{ramie}_b1500_*.json'))
        sel = json.loads(plik.read_text(encoding='utf-8'))
        ids = sel.get('selected_frame_ids') or sel['frame_ids']
        idx = np.array([pos[f] for f in ids if f in pos])
        wybor[ramie] = idx
        print(f'   {ramie:22s} wybranych {len(ids)}, z opisem {len(idx)}', flush=True)

    filmy = sorted(set(vid_w.tolist()) & set(list(pudla.values())[0]['per_video']))
    print(f'filmow walidacyjnych wspolnych {len(filmy)}\n', flush=True)

    raport = {'zgodnosc_baz': float(s.mean()), 'filmy': filmy, 'ramiona': sorted(ramiona),
              'Re_wolnych': [float(v) for v in lam.real[:args.modes]], 'tabela': {}, 'korelacje': {}}
    pudlo = {}
    for ramie, runy in ramiona.items():
        for v in filmy:
            w = [pudla[r]['per_video'][v] for r in runy]
            pudlo[(ramie, v)] = float(np.mean([q['misses'] / max(q['frames'], 1) for q in w]))

    for nazwa, (kp, kw) in przestrzenie.items():
        tab, xs, ys, vv, aa = {}, [], [], [], []
        for ramie, idx in wybor.items():
            d = najblizsza(kw, kp[idx])
            for v in filmy:
                m = vid_w == v
                cov = float(np.median(d[m]))
                tab[f'{ramie}|{v}'] = dict(pokrycie=cov, pudla=pudlo[(ramie, v)])
                xs.append(cov); ys.append(pudlo[(ramie, v)]); vv.append(v); aa.append(ramie)
        xs, ys, vv, aa = np.array(xs), np.array(ys), np.array(vv), np.array(aa)
        surowa = float(np.corrcoef(xs, ys)[0, 1])
        rx, ry = xs.copy(), ys.copy()
        for grupa in (vv, aa):                       # zdjecie sredniej filmu i sredniej ramienia
            for g in np.unique(grupa):
                m = grupa == g
                rx[m] -= rx[m].mean(); ry[m] -= ry[m].mean()
        czysta = float(np.corrcoef(rx, ry)[0, 1])
        raport['tabela'][nazwa] = tab
        raport['korelacje'][nazwa] = dict(surowa=surowa, po_zdjeciu_filmu_i_ramienia=czysta)
        print(f'{nazwa:18s} korelacja pokrycie-pudla: surowa {surowa:+.2f}, '
              f'po zdjeciu filmu i ramienia {czysta:+.2f}', flush=True)

    print('\npudla ramie po ramieniu (srednia po filmach i ziarnach):')
    for ramie in sorted(ramiona):
        print(f'   {ramie:22s} {100 * np.mean([pudlo[(ramie, v)] for v in filmy]):.2f}%')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
