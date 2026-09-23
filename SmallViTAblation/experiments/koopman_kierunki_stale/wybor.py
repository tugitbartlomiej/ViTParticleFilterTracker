#!/usr/bin/env python3
"""Klatki, w ktorych zwykle stale kierunki Koopmana mocno sie zmieniaja (pomysl uzytkownika, 2026-09-23).

Stan klatki: latent detektora ut_b0500 (tokeny backbone, profil zredukowany do 32 liczb) w miejscu, gdzie
WIEMY z adnotacji, ze jest cystotom: opis 3 x 3 kawalkow wewnatrz ramki etykiety (crop_descriptor), wybielony
do 48 wymiarow. Pula: 10 637 klatek uczacych tracking_static36 (te same obrazy sa na Edenie).

1. Koopman (EDMD) na parach sasiednich klatek tego samego filmu (odstep <= --max-gap).
   Kierunki stale: mody o |lambda| >= --prog (zwykle wracaja prawie takie same w nastepnej klatce).
2. Czy tozsamosc narzedzia siedzi w kierunkach stalych: R^2 przewidywania kontrastu wektora tozsamosci
   (RDZEN_bez_etykiet_32) ze wspolrzednych stalych wobec tylu samo najszybszych i losowych kierunkow.
3. Wynik klatki: blad jednokrokowego przewidywania Koopmana ograniczony do kierunkow stalych,
   kazdy kierunek w jednostkach swojego typowego szumu (poprawka Langevina), wieksza z wartosci
   wzgledem poprzedniej i nastepnej klatki.
4. Sprawdzenie offline bez uczenia: czy wynik jest wiekszy na klatkach puli, na ktorych ut_b0500 chybia
   (pewnosc < 0,5 albo IoU < 0,5), z pominieciem jego wlasnych 500 klatek uczacych.
5. Wybor --budzet klatek: od najwiekszego wyniku, z wykluczeniem +-(--odstep) klatek w tym samym filmie,
   plus losowa kontrola z tej samej puli (to samo ziarno selekcji).

    py -3.11 experiments/koopman_kierunki_stale/wybor.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KORZEN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KORZEN / 'experiments/krystalizacja_latentu'))
from krystalizacja import edmd, pairs_within          # noqa: E402
from pokrycie_a_bledy import crop_descriptor, wczytaj  # noqa: E402

DANE = KORZEN / 'feature_cache/eden_static36/data/tracking_static36'


def auc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    x = np.r_[p, n]; r = np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort') + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def iou(a, b):
    x1 = np.maximum(a[:, 0], b[:, 0]); y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]); y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return i / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - i, 1e-9)


def r2(X, y):
    X = np.c_[X, np.ones(len(X))]
    h = len(X) // 2
    b = np.linalg.lstsq(X[:h], y[:h], rcond=None)[0]
    e = y[h:] - X[h:] @ b
    return float(1 - (e ** 2).sum() / ((y[h:] - y[h:].mean()) ** 2).sum())


def wybierz(wynik, vid, nr, budzet, odstep):
    wybrane, zajete = [], {}
    for i in np.argsort(-np.nan_to_num(wynik, nan=-np.inf)):
        if not np.isfinite(wynik[i]):
            break
        z = zajete.setdefault(vid[i], [])
        if any(abs(nr[i] - n) <= odstep for n in z):
            continue
        wybrane.append(i); z.append(nr[i])
        if len(wybrane) == budzet:
            break
    return np.array(wybrane)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, default=KORZEN / 'results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500')
    ap.add_argument('--wektor', type=Path, default=KORZEN / 'experiments/krystalizacja_latentu/wyniki/dowody/RDZEN_bez_etykiet_32.npy')
    ap.add_argument('--uczace-detektora', type=Path, default=DANE / 'variants/tracking_uniform_time_b500_s42.json',
                    help='klatki, na ktorych uczono ut_b0500 (wylaczone z oceny offline)')
    ap.add_argument('--budzet', type=int, default=1500)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--max-gap', type=int, default=6)
    ap.add_argument('--prog', type=float, default=0.85)
    ap.add_argument('--odstep', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'wyniki')
    args = ap.parse_args(argv)

    pula = wczytaj(args.pula)
    fid = pula['frame_ids']
    man = {}
    for line in (DANE / 'pool_manifest.jsonl').open(encoding='utf-8'):
        r = json.loads(line); man[r['frame_id']] = r
    gt = np.array([man[f]['annotations'][0]['bbox'] if man[f]['annotations'] else [np.nan] * 4 for f in fid], float)
    ma = np.isfinite(gt).all(1) & (gt[:, 2] >= 8) & (gt[:, 3] >= 8)
    vid = np.array([f.split('/')[-2] for f in fid]); nr = np.array([int(f.split('/')[-1]) for f in fid])
    P = pula['profil_red'].astype(np.float64)
    x = crop_descriptor(P, gt, tuple(pula['meta']['grid']), 3)
    ok = ma & np.isfinite(x).all(1)
    print(f'klatek {len(fid)}, z cystotomem i opisem {ok.sum()}', flush=True)

    # kontrast wektora tozsamosci pod ramka etykiety
    w = np.load(args.wektor); a = P @ w                                  # klatki x 576
    H = W = 24
    gy, gx = np.meshgrid((np.arange(H) + .5) / H * 480, (np.arange(W) + .5) / W * 720, indexing='ij')
    gy, gx = gy.ravel(), gx.ravel()
    kontrast = np.full(len(fid), np.nan)
    for t in np.flatnonzero(ok):
        x0, y0, bw, bh = gt[t]
        wew = (gx >= x0) & (gx <= x0 + bw) & (gy >= y0) & (gy <= y0 + bh)
        dal = (gx < x0 - .5 * bw) | (gx > x0 + 1.5 * bw) | (gy < y0 - .5 * bh) | (gy > y0 + 1.5 * bh)
        if wew.sum() and dal.sum() > 20:
            kontrast[t] = (a[t, wew].mean() - np.median(a[t, dal])) / (a[t, dal].std() + 1e-9)

    idx = np.flatnonzero(ok)
    xa = x[idx] - x[idx].mean(0)
    _, sv, vt = np.linalg.svd(xa, full_matrices=False)
    z = xa @ (vt[:args.dim].T / np.maximum(sv[:args.dim] / np.sqrt(len(xa)), 1e-9))
    par = pairs_within(vid[idx], nr[idx], args.max_gap)
    lam, u, mu = edmd(z[par[:, 0]], z[par[:, 1]])
    stale = np.flatnonzero(np.abs(lam) >= args.prog)
    szybkie = np.argsort(np.abs(lam))[:len(stale)]
    phi = (z - mu) @ u                                                   # wspolrzedne Koopmana
    print(f'par {len(par)}, kierunkow stalych (|lambda| >= {args.prog}) {len(stale)} z {len(lam)}; '
          f'|lambda| stalych {np.round(np.abs(lam[stale]), 3).tolist()}', flush=True)

    # 2. czy tozsamosc siedzi w kierunkach stalych
    k_ = kontrast[idx]; g = np.isfinite(k_)
    perm = np.random.default_rng(0).permutation(g.sum())
    cz = lambda M: np.c_[M.real, M.imag][g][perm]
    rng = np.random.default_rng(1)
    losowe = [r2(cz(phi[:, rng.choice(len(lam), len(stale), replace=False)]), k_[g][perm]) for _ in range(20)]
    tozsamosc = dict(r2_kontrastu_z_kierunkow_stalych=r2(cz(phi[:, stale]), k_[g][perm]),
                     r2_z_kierunkow_szybkich=r2(cz(phi[:, szybkie]), k_[g][perm]),
                     r2_z_losowych_mediana=float(np.median(losowe)),
                     r2_ze_wszystkich=r2(cz(phi), k_[g][perm]))
    print('tozsamosc w kierunkach:', json.dumps(tozsamosc, indent=1), flush=True)

    # 3. wynik: blad przewidywania w kierunkach stalych, w jednostkach szumu kazdego kierunku
    res = phi[par[:, 1]][:, stale] - phi[par[:, 0]][:, stale] * lam[stale]
    skala = np.median(np.abs(res), 0) + 1e-9
    e = (np.abs(res) / skala) ** 2
    e_pre = e.sum(1)
    res_s = phi[par[:, 1]][:, szybkie] - phi[par[:, 0]][:, szybkie] * lam[szybkie]
    e_sz = ((np.abs(res_s) / (np.median(np.abs(res_s), 0) + 1e-9)) ** 2).sum(1)
    wynik_l = np.full(len(idx), np.nan); wynik_sz = np.full(len(idx), np.nan); zmiana = np.full(len(idx), np.nan)
    dz = np.linalg.norm(z[par[:, 1]] - z[par[:, 0]], axis=1)
    for (i0, i1), v, vs, d in zip(par, e_pre, e_sz, dz):
        for j in (i0, i1):
            wynik_l[j] = np.fmax(wynik_l[j], v); wynik_sz[j] = np.fmax(wynik_sz[j], vs); zmiana[j] = np.fmax(zmiana[j], d)
    wynik = np.full(len(fid), np.nan); wynik[idx] = wynik_l
    wsz = np.full(len(fid), np.nan); wsz[idx] = wynik_sz
    zm = np.full(len(fid), np.nan); zm[idx] = zmiana

    # 4. sprawdzenie offline na chybieniach ut_b0500
    uczace = set(json.loads(args.uczace_detektora.read_text(encoding='utf-8'))['frame_ids'])
    obce = np.array([f not in uczace for f in fid])
    trafia = (pula['score'] >= .5) & (iou(pula['pred_box'].astype(float), gt) >= .5)
    ocena = ok & obce & np.isfinite(wynik)
    ch, tr = ocena & ~trafia, ocena & trafia
    offline = dict(klatek_ocenianych=int(ocena.sum()), chybien=int(ch.sum()),
                   auc_zmiana_kierunkow_stalych=auc(wynik[ch], wynik[tr]),
                   auc_zmiana_kierunkow_szybkich=auc(wsz[ch], wsz[tr]),
                   auc_zwykla_zmiana_latentu=auc(zm[ch], zm[tr]),
                   auc_slaby_wektor_tozsamosci=auc(-kontrast[ch], -kontrast[tr]),
                   auc_niska_pewnosc_detektora=auc(-pula['score'][ch], -pula['score'][tr]))
    print('offline:', json.dumps(offline, indent=1), flush=True)

    # 5. wybor
    sel = wybierz(wynik, vid, nr, args.budzet, args.odstep)
    rng = np.random.default_rng(args.seed)
    kontr = rng.choice(np.flatnonzero(ok), args.budzet, replace=False)
    losowe_b1500 = set(json.loads((DANE / 'variants/tracking_random_b1500_s42.json').read_text(encoding='utf-8'))['frame_ids'])
    opis = {}
    for nazwa, s in (('koopman_stale', sel), ('losowe_z_puli', kontr)):
        s_obce = s[obce[s]]
        opis[nazwa] = dict(klatek=int(len(s)), filmow=int(len(set(vid[s].tolist()))),
                           chybien_ut_b0500_wsrod_obcych=float((~trafia[s_obce]).mean()),
                           mediana_kontrastu_wektora=float(np.nanmedian(kontrast[s])),
                           wspolnych_z_random_b1500_s42=int(len(set(fid[i] for i in s) & losowe_b1500)))
    print('zestawy:', json.dumps(opis, indent=1), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for nazwa, s in (('koopman_stale', sel), ('losowe_z_puli', kontr)):
        rid = f'tracking_{nazwa}_b{args.budzet}_sel{args.seed}'
        ids = [fid[i] for i in s]
        (args.out / f'{rid}.json').write_text(json.dumps({
            'schema_version': 1, 'variant_id': rid, 'run_id': rid, 'strategy': nazwa, 'seed': args.seed,
            'budget': args.budzet, 'frame_ids': ids, 'selected_frame_ids': ids,
            'labels_used_for_ranking': nazwa == 'koopman_stale'}, indent=2), encoding='utf-8')
    raport = dict(parametry={k: str(v) for k, v in vars(args).items()}, kierunkow_stalych=int(len(stale)),
                  lambda_stalych=[float(v) for v in np.abs(lam[stale])], tozsamosc=tozsamosc, offline=offline, zestawy=opis)
    (args.out / 'REPORT_wybor.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(args.out / 'WYNIK_klatek.npz', frame_ids=np.array(fid), wynik=wynik, wynik_szybkie=wsz,
                        zmiana=zm, kontrast=kontrast, trafia=trafia, obce=obce)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
