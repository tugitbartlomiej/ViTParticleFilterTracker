#!/usr/bin/env python3
"""Detektor klatek, ktore mocno pobudzaja kierunki tozsamosci wyznaczone przez Koopmana (2026-09-23).

Dla JEDNEJ klatki, bez etykiet, z wstepnego detektora (B2, 1500 klatek):

  opis z      tokeny z wnetrza przewidzianej ramki, siatka 3 x 3, wybielenie do 48 liczb
  Q           baza ortonormalna przestrzeni tozsamosci: mody Koopmana, ktore trwaja (Re lambda > 0,85)
              i zmieniaja sie w filmie (udzial > 0,35); z pary sprzezonej bierzemy Re i Im

Trzy oceny:
  amplituda   A(x) = |Q^T (z - mu)|            jak daleko klatka lezy wzdluz kierunkow tozsamosci
  udzial      S(x) = |Q^T (z - mu)|^2 / |z - mu|^2
              jaka czesc odchylenia klatki idzie w tozsamosc (dla przypadkowego kierunku 5/48)
  wplyw       I(t) = sum_k Re[ (v_k^H G^{-1} z_t)(psi_k(z_{t+1}) - lambda_k psi_k(z_t)) / (v_k^H u_k) ]
              pierwszy rzad zmiany wartosci wlasnej modu tozsamosci po dolozeniu pary (t, t+1);
              dodatni: para robi tozsamosc TRWALSZA (lambda blizej 1), ujemny: rozmywa ja

Etykiety wchodza tylko na koncu, do opisu, czym roznia sie klatki mocno i slabo pobudzajace.

    py -3.11 experiments/krystalizacja_latentu/pobudzenie_tozsamosci.py \\
        --pula results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500 \\
        --manifest feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl \\
        --dataset ../Dataset/Cataract-LMM/3_Object_Tracking/annotations \\
        --out experiments/krystalizacja_latentu/wyniki/pobudzenie_tozsamosci
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from krystalizacja import pairs_within
from pokrycie_a_bledy import crop_descriptor, wczytaj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pula', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dim', type=int, default=48)
    ap.add_argument('--prog-re', type=float, default=0.85)
    ap.add_argument('--prog-w-filmie', type=float, default=0.35)
    ap.add_argument('--ile', type=int, default=60)
    args = ap.parse_args(argv)

    p = wczytaj(args.pula)
    x = crop_descriptor(p['profil_red'].astype(np.float64), p['pred_box'].astype(np.float64),
                        tuple(p['meta']['grid']), 3)
    ok = np.flatnonzero(np.isfinite(x).all(1) & (p['score'] > .5))
    fid = [p['frame_ids'][i] for i in ok]
    vid = np.array([f.split('/')[-2] for f in fid])
    nr = np.array([int(f.split('/')[-1]) for f in fid])
    a = x[ok] - x[ok].mean(0)
    _, sv, vt = np.linalg.svd(a, full_matrices=False)
    z = a @ (vt[:args.dim].T / np.maximum(sv[:args.dim] / np.sqrt(len(a)), 1e-9))

    par = pairs_within(vid, nr, 6)
    mu = z[par[:, 0]].mean(0)
    z0, z1 = z[par[:, 0]] - mu, z[par[:, 1]] - mu
    g = z0.T @ z0 / len(z0) + 1e-3 * np.eye(args.dim)
    k = np.linalg.solve(g, z0.T @ z1 / len(z0))            # Koopman na wspolczynnikach: psi(z') ~ lambda psi(z)
    lam, u = np.linalg.eig(k)
    lamL, v = np.linalg.eig(k.T.conj())                    # lewe wektory wlasne
    o = np.argsort(-lam.real)
    lam, u = lam[o], u[:, o]
    v = v[:, [int(np.argmin(np.abs(lamL.conj() - l_))) for l_ in lam]]

    wolne = np.flatnonzero(lam.real > args.prog_re)
    psi = (z - mu) @ u[:, wolne]
    mid = {f: psi[vid == f].mean(0) for f in np.unique(vid)}
    dyn = psi - np.array([mid[f] for f in vid])
    w_f = (np.abs(dyn) ** 2).sum(0) / np.maximum((np.abs(psi - psi.mean(0)) ** 2).sum(0), 1e-12)
    toz = [int(wolne[j]) for j in range(len(wolne)) if w_f[j] >= args.prog_w_filmie]
    jeden = []                                             # z pary sprzezonej zostaje jeden mod
    for m in toz:
        if not any(abs(lam[m] - np.conj(lam[j])) < 1e-9 and abs(lam[m].imag) > 1e-12 for j in jeden):
            jeden.append(m)
    kol = []
    for m in jeden:
        kol.append(u[:, m].real)
        if abs(lam[m].imag) > 1e-12:
            kol.append(u[:, m].imag)
    # u to wspolczynniki funkcji psi = z^T u; kierunek w przestrzeni opisu to G u (psi = <z, G^{-1}...>),
    # przy wybielonym opisie G ~ I, wiec baze bierzemy wprost z u i ortonormalizujemy
    q, _ = np.linalg.qr(np.stack(kol, 1))
    d = z - mu
    y = d @ q
    amp = np.linalg.norm(y, axis=1)
    udzial = (y ** 2).sum(1) / np.maximum((d ** 2).sum(1), 1e-12)

    # wplyw pary na wartosci wlasne modow tozsamosci (pierwszy rzad, patrz docstring)
    gi_z = np.linalg.solve(g, z0.T).T
    wplyw_par = np.zeros(len(par))
    for m in jeden:
        a_m = gi_z @ v[:, m].conj() / (v[:, m].conj() @ u[:, m])
        res = z1 @ u[:, m] - lam[m] * (z0 @ u[:, m])
        wplyw_par += (a_m * res).real
    wplyw = np.full(len(z), np.nan)
    wplyw[par[:, 0]] = wplyw_par

    # etykiety tylko do opisu
    ann = {}
    for line in args.manifest.open(encoding='utf-8'):
        r = json.loads(line)
        ann[r['frame_id']] = (r.get('annotations') or [{}])[0].get('bbox')
    box = np.array([ann.get(f) or [np.nan] * 4 for f in fid], float)
    pred = p['pred_box'][ok].astype(float)
    def iou(b1, b2):
        x1 = np.maximum(b1[:, 0], b2[:, 0]); y1 = np.maximum(b1[:, 1], b2[:, 1])
        x2 = np.minimum(b1[:, 0] + b1[:, 2], b2[:, 0] + b2[:, 2]); y2 = np.minimum(b1[:, 1] + b1[:, 3], b2[:, 1] + b2[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        return inter / (b1[:, 2] * b1[:, 3] + b2[:, 2] * b2[:, 3] - inter)
    cechy = {'wydluzenie ramki (log dl/szer)': np.log(np.maximum(box[:, 2], 1) / np.maximum(box[:, 3], 1)),
             'wielkosc ramki (log px2)': np.log(np.maximum(box[:, 2] * box[:, 3], 1)),
             'IoU detektora wstepnego': iou(box, pred),
             'pewnosc detekcji': p['score'][ok].astype(float)}

    def spear(a_, b_):
        m_ = np.isfinite(a_) & np.isfinite(b_)
        ra, rb = np.argsort(np.argsort(a_[m_])), np.argsort(np.argsort(b_[m_]))
        return float(np.corrcoef(ra, rb)[0, 1])

    oceny = {'amplituda': amp, 'udzial': udzial, 'wplyw': wplyw}
    raport = {'klatek': len(z), 'mody_tozsamosci': jeden, 'lambda': [str(np.round(lam[m], 4)) for m in jeden],
              'wymiar': int(q.shape[1]), 'udzial_przypadkowy': q.shape[1] / args.dim,
              'udzial_mediana': float(np.median(udzial)), 'korelacje_rangowe': {}, 'gorne_vs_dolne': {}}
    print(f'klatek {len(z)}, mody tozsamosci {jeden}, wymiar {q.shape[1]}; udzial mediana '
          f'{np.median(udzial):.3f} (przypadkowy kierunek {q.shape[1] / args.dim:.3f})')
    print(f'korelacje rangowe ocen: amp-udzial {spear(amp, udzial):.2f}, amp-wplyw {spear(amp, wplyw):.2f}, '
          f'udzial-wplyw {spear(udzial, wplyw):.2f}')
    for on, ov in oceny.items():
        raport['korelacje_rangowe'][on] = {cn: spear(ov, cv) for cn, cv in cechy.items()}
        m_ = np.isfinite(ov)
        gora = m_ & (ov >= np.nanquantile(ov, .9)); dol = m_ & (ov <= np.nanquantile(ov, .1))
        raport['gorne_vs_dolne'][on] = {cn: [float(np.nanmedian(cv[gora])), float(np.nanmedian(cv[dol]))]
                                        for cn, cv in cechy.items()}
        print(f'\n{on}: korelacja rangowa z cechami / mediana w gornych 10% vs dolnych 10%')
        for cn, cv in cechy.items():
            gg, dd = raport['gorne_vs_dolne'][on][cn]
            print(f'   {cn:32s} {raport["korelacje_rangowe"][on][cn]:+.2f}   {gg:8.3f} vs {dd:8.3f}')

    # mozaiki: najmocniej i najslabiej pobudzajace wg udzialu i wplywu
    args.out.mkdir(parents=True, exist_ok=True)
    zipy: dict[str, zipfile.ZipFile] = {}
    def mozaika(idx, plik):
        w, h, kolumn = 192, 128, 10
        img = Image.new('RGB', (kolumn * w, ((len(idx) + kolumn - 1) // kolumn) * h), 'white')
        for n_, i in enumerate(idx):
            film, nrk = vid[i], nr[i]
            zf = zipy.setdefault(film, zipfile.ZipFile(args.dataset / f'{film}.zip'))
            im = Image.open(io.BytesIO(zf.read(f'{film}/{film}_{nrk:06d}.jpg'))).convert('RGB')
            dr = ImageDraw.Draw(im)
            bx, by, bw, bh = pred[i]
            dr.rectangle([bx, by, bx + bw, by + bh], outline=(255, 220, 0), width=4)
            img.paste(im.resize((w, h)), ((n_ % kolumn) * w, (n_ // kolumn) * h))
        img.save(args.out / plik, quality=88)
    for on in ('udzial', 'wplyw'):
        ov = np.where(np.isfinite(oceny[on]), oceny[on], -np.inf)
        mozaika(np.argsort(-ov)[:args.ile], f'MOZAIKA_{on}_najmocniej.jpg')
        ov2 = np.where(np.isfinite(oceny[on]), oceny[on], np.inf)
        mozaika(np.argsort(ov2)[:args.ile], f'MOZAIKA_{on}_najslabiej.jpg')
    np.savez_compressed(args.out / 'OCENY.npz', frame_ids=np.array(fid), amplituda=amp, udzial=udzial,
                        wplyw=wplyw, wspolrzedne=y)
    (args.out / 'REPORT.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nzapisane w', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
