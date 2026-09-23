#!/usr/bin/env python3
"""Jak w trakcie uczenia zmienia sie zapis tozsamosci narzedzia w latencie (dane z sonda_treningu.py).

Po kazdym kroku uczenia siec widziala te same klatki kontrolne w wersjach: orig, obrot o -30 i +30
stopni, wydluzenie 1,4, narzedzie na tle innej operacji, ta sama scena 5 klatek pozniej. Dla kazdego
kroku liczymy:

  zmiana wzgledna      R_g = srednia |L(gx) - L(x)|^2 / srednia |L(x) - srednia|^2
                       (0: latent nie reaguje na przeksztalcenie, 1: reaguje tak, jak na inna klatke)
  najlepszy kierunek   najmniejsza zmiana wzgledna pojedynczego kierunku (uogolnione zagadnienie
                       wlasne na wspolnej bazie PCA), liczona na polowie klatek i sprawdzana na drugiej
  pewnosc, IoU         czy detektor widzi narzedzie w kazdej wersji

oraz przypisanie klatkom uczacym: zmiana miary w kroku regresowana na obecnosc klatki w paczce
(grzbietowo), co daje, ktore klatki popychaja latent ku niezmienniczosci, a ktore od niej.

    py -3.11 experiments/krystalizacja_latentu/analiza_sondy.py --sonda <katalog *_sonda> \\
        [--telemetria <TELEMETRIA.json>] [--coco-train <plik COCO zbioru uczacego>] --out <REPORT.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def zmiana(L, war, g):
    o = L[:, war.index('orig')]
    spread = ((o - o.mean(0)) ** 2).sum(1).mean()
    return float(((L[:, war.index(g)] - o) ** 2).sum(1).mean() / max(spread, 1e-12))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sonda', type=Path, required=True)
    ap.add_argument('--telemetria', type=Path)
    ap.add_argument('--coco-train', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--k', type=int, default=24)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = dict(np.load(args.sonda / 'SONDA.npz'))
    meta = json.loads((args.sonda / 'SONDA_META.json').read_text(encoding='utf-8'))
    war = meta['warianty']
    kroki, epoki = d['kroki'], d['epoki']
    S = len(kroki)
    n = len(meta['frame_ids'])
    V = len(war)
    iou_gt = d['ramka_gt'].reshape(n, V, 4)
    wynik = {'pomiarow': int(S), 'klatek_sondy': n, 'warianty': war, 'latenty': {}}
    krzywe = {}
    for ln in ('zapytanie', 'tokeny_ramki'):
        L = d[ln].astype(np.float64).reshape(S, n, V, -1)
        # wspolna baza PCA z oryginalow wszystkich krokow, zeby kierunki byly porownywalne w czasie
        o_all = L[:, :, war.index('orig')].reshape(S * n, -1)
        m = o_all.mean(0)
        _, _, vt = np.linalg.svd(o_all[::max(1, S * n // 20000)] - m, full_matrices=False)
        P = vt[:args.k].T
        rng = np.random.default_rng(0)
        pol = rng.permutation(n)
        a, b = pol[:n // 2], pol[n // 2:]
        R = {g: np.zeros(S) for g in war if g != 'orig'}
        najlepszy = np.zeros(S)
        for s in range(S):
            Ls = L[s]
            for g in R:
                R[g][s] = zmiana(Ls, war, g)
            Z = (Ls - m) @ P
            def mac(idx):
                oo = Z[idx, war.index('orig')]
                sk = np.cov(oo.T) + 1e-6 * np.eye(args.k)
                dd = np.concatenate([Z[idx, war.index(g)] - Z[idx, war.index('orig')]
                                     for g in war if g not in ('orig', 'nast')])
                return sk, dd.T @ dd / len(dd)
            ska, sza = mac(a); skb, szb = mac(b)
            wl, U = np.linalg.eigh(ska)
            W = U @ np.diag(wl ** -.5)
            lam, E = np.linalg.eigh(W.T @ sza @ W)
            kier = W @ E[:, :3]
            najlepszy[s] = float(min((kier[:, j] @ szb @ kier[:, j]) / (kier[:, j] @ skb @ kier[:, j]) for j in range(3)))
        krzywe[ln] = dict(R=R, najlepszy=najlepszy)
        wynik['latenty'][ln] = {
            'poczatek': {g: float(R[g][0]) for g in R}, 'koniec': {g: float(R[g][-1]) for g in R},
            'minimum': {g: float(R[g].min()) for g in R}, 'krok_minimum': {g: int(kroki[R[g].argmin()]) for g in R},
            'najlepszy_kierunek_poczatek': float(najlepszy[0]), 'najlepszy_kierunek_koniec': float(najlepszy[-1]),
            'najlepszy_kierunek_minimum': float(najlepszy.min())}

    pew = d['pewnosc'].astype(np.float64).reshape(S, n, V)
    rp = d['ramka'].reshape(S, n, V, 4)
    def iou(p, q):
        x1 = np.maximum(p[..., 0], q[..., 0]); y1 = np.maximum(p[..., 1], q[..., 1])
        x2 = np.minimum(p[..., 0] + p[..., 2], q[..., 0] + q[..., 2]); y2 = np.minimum(p[..., 1] + p[..., 3], q[..., 1] + q[..., 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        return inter / np.maximum(p[..., 2] * p[..., 3] + q[..., 2] * q[..., 3] - inter, 1e-9)
    IoU = iou(rp, iou_gt[None])
    wynik['pewnosc_koniec'] = {w: float(np.median(pew[-1, :, i])) for i, w in enumerate(war)}
    wynik['iou_koniec'] = {w: float(np.median(IoU[-1, :, i])) for i, w in enumerate(war)}

    # przypisanie klatkom uczacym: zmiana niezmienniczosci w kroku ~ obecnosc klatki w paczce
    ids_b = meta['batch_image_ids']
    wszystkie = sorted({i for b_ in ids_b for i in b_})
    if wszystkie and S > 20:
        idx = {i: j for j, i in enumerate(wszystkie)}
        X = np.zeros((S - 1, len(wszystkie)))
        for s in range(1, S):
            for i in ids_b[s]:
                X[s - 1, idx[i]] = 1
        y = np.diff(np.mean([krzywe['zapytanie']['R'][g] for g in ('rot_-30', 'rot_+30', 'dl_1.4', 'tlo_a')], 0))
        Xc = X - X.mean(0); yc = y - y.mean()
        beta = np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(X.shape[1]), Xc.T @ yc)
        nazwy = {}
        if args.coco_train and args.coco_train.exists():
            cc = json.loads(args.coco_train.read_text(encoding='utf-8'))
            nazwy = {im['id']: im.get('frame_id') or im['file_name'] for im in cc['images']}
        o = np.argsort(beta)
        wynik['klatki_najbardziej_ku_niezmienniczosci'] = [[nazwy.get(wszystkie[j], wszystkie[j]), float(beta[j])] for j in o[:15]]
        wynik['klatki_najbardziej_od_niezmienniczosci'] = [[nazwy.get(wszystkie[j], wszystkie[j]), float(beta[j])] for j in o[::-1][:15]]
        wynik['klatek_uczacych_w_paczkach'] = len(wszystkie)
        rng = np.random.default_rng(0)
        def dopasuj(yy):
            return np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(X.shape[1]), Xc.T @ (yy - yy.mean()))
        zero = [np.abs(dopasuj(rng.permutation(y))).max() for _ in range(200)]
        wynik['przypisanie_max_beta'] = float(np.abs(beta).max())
        wynik['przypisanie_podloga_95'] = float(np.quantile(zero, .95))
        wynik['przypisanie_ponad_szum'] = bool(np.abs(beta).max() > np.quantile(zero, .95))

    fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for ln, st in (('zapytanie', '-'), ('tokeny_ramki', '--')):
        for g, c in zip(('rot_-30', 'rot_+30', 'dl_1.4', 'tlo_a', 'nast'), ('C0', 'C1', 'C2', 'C3', 'C4')):
            ax[0].plot(kroki, krzywe[ln]['R'][g], st, color=c, lw=1, label=f'{ln}: {g}')
        ax[1].plot(kroki, krzywe[ln]['najlepszy'], st, color='k', lw=1, label=ln)
    for i, w in enumerate(war):
        ax[2].plot(kroki, np.median(pew[:, :, i], 1), lw=1, label=w)
    ax[0].set_ylabel('zmiana wzgledna latentu'); ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_ylabel('najbardziej niezmienniczy kierunek'); ax[1].legend(fontsize=7)
    ax[2].set_ylabel('pewnosc detekcji (mediana)'); ax[2].set_xlabel('krok uczenia'); ax[2].legend(fontsize=7, ncol=3)
    granice = kroki[np.flatnonzero(np.diff(epoki)) + 1]
    for a_ in ax:
        for x in granice:
            a_.axvline(x, color='0.85', lw=.5)
    if args.telemetria and args.telemetria.exists():
        t = json.loads(args.telemetria.read_text(encoding='utf-8'))
        mp = t.get('val/mAP_50_95', [])
        if mp:
            wynik['mAP_walidacja'] = mp
            bx = ax[2].twinx()
            kr = np.array([e for e, _ in mp]); vv = np.array([v for _, v in mp])   # krok TensorBoard = krok uczenia
            bx.plot(kr, vv, 'ko-', ms=3, label='mAP walidacja'); bx.set_ylabel('mAP walidacja (czarne)')
            wynik['krok_najlepszego_mAP'] = int(kr[vv.argmax()]); wynik['najlepsze_mAP'] = float(vv.max())
    fig.suptitle('Tozsamosc narzedzia w latencie, krok po kroku uczenia (sonda: 32 klatki walidacyjne)')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=130); plt.close(fig)
    args.out.write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in wynik.items() if k in ('pomiarow', 'latenty', 'pewnosc_koniec', 'iou_koniec')},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
