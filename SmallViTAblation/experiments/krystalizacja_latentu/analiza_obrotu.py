#!/usr/bin/env python3
"""Analiza testu obrotu, wydluzenia i tla (dane z pipeline/koopman_langevin/tozsamosc_obrot.py).

Pytanie autora (2026-09-23): gdzie siec zapisuje w latencie, ze to jest TO narzedzie, a gdzie
jego wariacje (obrot, wydluzenie, skrocenie)? Wedlug Lenca i Vedaldiego: kierunki niezmiennicze
(nie ruszaja sie przy przeksztalceniu) to tozsamosc, kierunki, ktore przeksztalcenie przesuwa
w przewidywalny sposob, to poza.

Dla kazdego detektora i kazdego latentu:
  1. czy detektor dalej widzi narzedzie (IoU z ramka po przeksztalceniu, pewnosc)
  2. zmiana wzgledna  R_g = srednia |L(gx) - L(x)|^2 / srednia |L(x) - srednia|^2
     (0: latent sie nie rusza, 1: rusza sie tyle, ile rozni sie od siebie rozne klatki)
  3. kierunki niezmiennicze: uogolnione zagadnienie wlasne  S_zmian w = lambda S_klatek w,
     liczone na polowie filmow, sprawdzane na drugiej polowie; lambda to zmiana wzgledna
     danego kierunku; licza sie kierunki z lambda < 0,1 na filmach sprawdzajacych
  4. rownowaznosc obrotu: mapa liniowa L(x) -> L(rot x) na polowie filmow, R2 na drugiej,
     wobec mapy tozsamosciowej; katy wartosci wlasnych mapy wobec kata obrotu
  5. tylko ut_b0500: wspolrzedne tozsamosci i sceny z Koopmana (jak w pobudzenie_tozsamosci.py)

    py -3.11 experiments/krystalizacja_latentu/analiza_obrotu.py \\
        --dane experiments/krystalizacja_latentu/dane/obrot \\
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \\
        --out experiments/krystalizacja_latentu/wyniki/REPORT_obrot.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from krystalizacja import pairs_within
from pokrycie_a_bledy import crop_descriptor, wczytaj

GRUPY = {'obrot': ['rot_-45', 'rot_-30', 'rot_-15', 'rot_+15', 'rot_+30', 'rot_+45'],
         'wydluzenie': ['dl_0.7', 'dl_1.4'], 'tlo': ['tlo_a', 'tlo_b']}


def iou(a, b):
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2]); y2 = np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum(a[..., 2] * a[..., 3] + b[..., 2] * b[..., 3] - inter, 1e-9)


def zmiana(L, war, grupa):
    o = L[:, war.index('orig')]
    spread = ((o - o.mean(0)) ** 2).sum(1).mean()
    return float(np.mean([((L[:, war.index(g)] - o) ** 2).sum(1).mean() for g in grupa]) / spread)


def pca(L0, k):
    m = L0.mean(0)
    _, s, vt = np.linalg.svd(L0 - m, full_matrices=False)
    return m, vt[:k].T


def niezmiennicze(L, war, film, k=32, prog=0.1, seed=0):
    """Kierunki, ktore przeksztalcenia ruszaja malo w porownaniu z roznica miedzy klatkami."""
    rng = np.random.default_rng(seed)
    filmy = np.unique(film)
    rng.shuffle(filmy)
    a = np.isin(film, filmy[:len(filmy) // 2]); b = ~a
    o = L[:, war.index('orig')]
    m, P = pca(o[a], k)
    def macierze(maska):
        oo = (o[maska] - m) @ P
        sk = np.cov(oo.T)
        d = np.concatenate([(L[maska, war.index(g)] - L[maska, war.index('orig')]) @ P
                            for grupa in GRUPY.values() for g in grupa])
        sz = d.T @ d / len(d)
        return sk, sz
    sk_a, sz_a = macierze(a)
    sk_b, sz_b = macierze(b)
    wl, V = np.linalg.eigh(sk_a + 1e-6 * np.trace(sk_a) / k * np.eye(k))
    W = V @ np.diag(wl ** -.5)
    lam, U = np.linalg.eigh(W.T @ sz_a @ W)
    kier = W @ U                                               # kolumny: kierunki od najbardziej niezmienniczych
    lam_b = np.array([(kier[:, j] @ sz_b @ kier[:, j]) / (kier[:, j] @ sk_b @ kier[:, j]) for j in range(k)])
    per_grupa = {}
    for gn, grupa in GRUPY.items():
        d = np.concatenate([(L[b, war.index(g)] - L[b, war.index('orig')]) @ P for g in grupa])
        szg = d.T @ d / len(d)
        per_grupa[gn] = [float((kier[:, j] @ szg @ kier[:, j]) / (kier[:, j] @ sk_b @ kier[:, j])) for j in range(5)]
    return dict(lambda_uczace=lam[:10].tolist(), lambda_sprawdzajace=lam_b[:10].tolist(),
                ile_ponizej_progu=int((lam_b < prog).sum()), prog=prog, pierwsze5_wg_grupy=per_grupa)


def rownowaznosc(L, war, film, k=32, alfa=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    filmy = np.unique(film)
    rng.shuffle(filmy)
    a = np.isin(film, filmy[:len(filmy) // 2]); b = ~a
    o = L[:, war.index('orig')]
    m, P = pca(o[a], k)
    wynik = {}
    for g in GRUPY['obrot'] + GRUPY['wydluzenie']:
        X = (o - m) @ P; Y = (L[:, war.index(g)] - m) @ P
        M = np.linalg.solve(X[a].T @ X[a] + alfa * len(X[a]) * np.eye(k), X[a].T @ Y[a]).T
        res = Y[b] - X[b] @ M.T
        r2 = 1 - (res ** 2).sum() / ((Y[b] - Y[b].mean(0)) ** 2).sum()
        r2_id = 1 - ((Y[b] - X[b]) ** 2).sum() / ((Y[b] - Y[b].mean(0)) ** 2).sum()
        ev = np.linalg.eigvals(M)
        ev = ev[np.argsort(-np.abs(ev))][:6]
        wynik[g] = dict(r2_mapa=float(r2), r2_bez_zmiany=float(r2_id),
                        kat_wartosci_wlasnych_stopnie=np.round(np.degrees(np.angle(ev)), 1).tolist(),
                        modul=np.round(np.abs(ev), 3).tolist())
    return wynik


def baza_koopmana(pula_dir: Path, dim=48, prog_re=0.85, prog_w=0.35):
    p = wczytaj(pula_dir)
    x = crop_descriptor(p['profil_red'].astype(np.float64), p['pred_box'].astype(np.float64), tuple(p['meta']['grid']), 3)
    ok = np.flatnonzero(np.isfinite(x).all(1) & (p['score'] > .5))
    fid = [p['frame_ids'][i] for i in ok]
    vid = np.array([f.split('/')[-2] for f in fid]); nr = np.array([int(f.split('/')[-1]) for f in fid])
    xm = x[ok].mean(0)
    _, sv, vt = np.linalg.svd(x[ok] - xm, full_matrices=False)
    W = vt[:dim].T / np.maximum(sv[:dim] / np.sqrt(len(ok)), 1e-9)
    z = (x[ok] - xm) @ W
    par = pairs_within(vid, nr, 6)
    mu = z[par[:, 0]].mean(0)
    z0, z1 = z[par[:, 0]] - mu, z[par[:, 1]] - mu
    k = np.linalg.solve(z0.T @ z0 / len(z0) + 1e-3 * np.eye(dim), z0.T @ z1 / len(z0))
    lam, u = np.linalg.eig(k)
    o = np.argsort(-lam.real); lam, u = lam[o], u[:, o]
    wolne = np.flatnonzero(lam.real > prog_re)
    psi = (z - mu) @ u[:, wolne]
    mid = {f: psi[vid == f].mean(0) for f in np.unique(vid)}
    dyn = psi - np.array([mid[f] for f in vid])
    w_f = (np.abs(dyn) ** 2).sum(0) / (np.abs(psi - psi.mean(0)) ** 2).sum(0)
    def baza(mody):
        kol, byly = [], []
        for m_ in mody:
            if any(abs(lam[m_] - np.conj(lam[j])) < 1e-9 and abs(lam[m_].imag) > 1e-12 for j in byly):
                continue
            byly.append(m_); kol.append(u[:, m_].real)
            if abs(lam[m_].imag) > 1e-12:
                kol.append(u[:, m_].imag)
        return np.linalg.qr(np.stack(kol, 1))[0]
    toz = [int(wolne[j]) for j in range(len(wolne)) if w_f[j] >= prog_w]
    sce = [int(wolne[j]) for j in range(len(wolne)) if w_f[j] < prog_w]
    return dict(xm=xm, W=W, mu=mu, toz=baza(toz), scena=baza(sce), dim=dim)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dane', type=Path, required=True)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    meta = json.loads((args.dane / 'META.json').read_text(encoding='utf-8'))
    war = meta['warianty']
    film = np.array([f.split('/')[-2] for f in meta['frame_ids']])
    raport = {'klatek': len(film), 'filmow': int(len(np.unique(film))), 'detektory': {}}
    bk = baza_koopmana(args.pula)
    for nazwa in meta['detektory']:
        d = dict(np.load(args.dane / f'OBROT_{nazwa}.npz'))
        r = {'iou': {w: float(np.nanmedian(iou(d['ramka'][:, i], d['ramka_gt'][:, i]))) for i, w in enumerate(war)},
             'pewnosc': {w: float(np.median(d['pewnosc'][:, i])) for i, w in enumerate(war)}, 'latenty': {}}
        latenty = {'zapytanie': d['zapytanie'].astype(np.float64), 'tokeny_ramki': d['tokeny_ramki'].astype(np.float64)}
        if 'opis_koopmana' in d:
            x = d['opis_koopmana'].astype(np.float64)
            ok = np.isfinite(x).all((1, 2))
            z = (x - bk['xm']) @ bk['W']
            z[~ok] = np.nan
            latenty['koopman_opis48'] = z
            latenty['koopman_tozsamosc'] = (z - bk['mu']) @ bk['toz']
            latenty['koopman_scena'] = (z - bk['mu']) @ bk['scena']
            rng = np.random.default_rng(0)
            losowe = np.linalg.qr(rng.normal(size=(bk['dim'], bk['toz'].shape[1])))[0]
            latenty['koopman_losowe5'] = (z - bk['mu']) @ losowe
        for ln, L in latenty.items():
            m = np.isfinite(L).all((1, 2))
            Lm, fm = L[m], film[m]
            wyn = {'klatek': int(m.sum()), 'wymiar': int(L.shape[2]),
                   'zmiana_wzgledna': {gn: zmiana(Lm, war, g) for gn, g in GRUPY.items()},
                   'zmiana_wzgledna_obrot_wg_kata': {g: zmiana(Lm, war, [g]) for g in GRUPY['obrot']}}
            if L.shape[2] >= 32:
                wyn['niezmiennicze'] = niezmiennicze(Lm, war, fm)
                wyn['rownowaznosc'] = rownowaznosc(Lm, war, fm)
            r['latenty'][ln] = wyn
        raport['detektory'][nazwa] = r
        print(f'\n=== {nazwa}: IoU orig {r["iou"]["orig"]:.2f}, rot_+45 {r["iou"]["rot_+45"]:.2f}, '
              f'dl_1.4 {r["iou"]["dl_1.4"]:.2f}, tlo_a {r["iou"]["tlo_a"]:.2f}; pewnosc orig {r["pewnosc"]["orig"]:.2f} '
              f'rot_+45 {r["pewnosc"]["rot_+45"]:.2f} tlo_a {r["pewnosc"]["tlo_a"]:.2f}')
        for ln, w in r['latenty'].items():
            zw = w['zmiana_wzgledna']
            linia = f'   {ln:20s} zmiana: obrot {zw["obrot"]:.2f}  wydluzenie {zw["wydluzenie"]:.2f}  tlo {zw["tlo"]:.2f}'
            if 'niezmiennicze' in w:
                n_ = w['niezmiennicze']
                linia += (f' | kierunkow niezmienniczych (<{n_["prog"]}) {n_["ile_ponizej_progu"]}/32, '
                          f'najlepszy {n_["lambda_sprawdzajace"][0]:.3f}')
                rw = w['rownowaznosc']['rot_+30']
                linia += f' | mapa obrotu 30: R2 {rw["r2_mapa"]:.2f} vs bez zmiany {rw["r2_bez_zmiany"]:.2f}'
            print(linia)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
