#!/usr/bin/env python3
"""Czy zaskoczenie Koopmana przewiduje chybienia detektora lepiej niz sam wektor narzedzia? (2026-09-23)

Bez uczenia sieci. Detektor ut_b0500 przeszedl juz przez klatki testowe (ocena_test.py, co 10. klatka
25 operacji testowych) i wiadomo, gdzie trafil, a gdzie chybil. Tu:

1. ekstrakcja (--ekstrakcja, venv rfdetr, GPU): dla kazdej klatki testowej stan latentu y_t z tokenow
   backbone, 64 liczby: srednia profilu zredukowanego (32) po calym kadrze i srednia wazona aktywacja
   wektora narzedzia (32), czyli "jak wyglada miejsce, ktore siec uznaje za narzedzie". Bez etykiet.
2. Koopman: y_{t+1} = K y_t + c, dopasowane najmniejszymi kwadratami (z malym grzbietem) na filmach
   z jednej polowy, liczone na drugiej (i odwrotnie). Zaskoczenie klatki t = blad przewidywania y_t z y_{t-1}.
3. Langevin: blad mierzony w jednostkach szumu. Kowariancja reszt Sigma z polowy dopasowania daje
   odleglosc Mahalanobisa e^T Sigma^-1 e (duza tylko wtedy, gdy blad wychodzi poza zwykly szum ruchu),
   a wersja "wzgledem filmu" dzieli ja jeszcze przez mediane danego filmu.

Porownanie (AUC: czy miara jest wieksza na klatkach chybionych niz trafionych, tylko klatki z narzedziem):
  wektor slaby          -kontrast wektora pod obrysem narzedzia (z etykiet, jak w ocena_test)
  obecnosc z mapy       -(najsilniejszy kawalek wobec reszty kadru), bez etykiet
  zmiana klatki         ||y_t - y_{t-1}||, kontrola: czy Koopman daje cos ponad zwykla roznice
  Koopman, Koopman+Langevin, Koopman+Langevin wzgledem filmu
  pewnosc detektora     1 - pewnosc (czesciowo kolowe, bo chybienie obejmuje niska pewnosc)
oraz osobno na "pewnych chybieniach" (pewnosc >= 0,5, IoU < 0,5: zwykle inne narzedzie), gdzie
pewnosc detektora z definicji nic nie mowi.

    <venv rfdetr>/python experiments/krystalizacja_latentu/koopman_zaskoczenie.py --ekstrakcja ...
    py -3.11 experiments/krystalizacja_latentu/koopman_zaskoczenie.py --dane experiments/krystalizacja_latentu/wyniki/test
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np


def auc(p, n):
    p, n = np.asarray(p, float), np.asarray(n, float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    x = np.r_[p, n]; r = np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort') + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def ekstrakcja(args):
    import rfdetr
    import torch
    from PIL import Image
    from test_ksztaltu import wczytaj_film

    split = json.loads(args.split.read_text(encoding='utf-8'))['groups']
    filmy = sorted(f for f, s in split.items() if s == 'test')
    ref = np.load(args.dane / 'TEST_ut_b0500.npz')
    klatki, zipy = [], {}
    for f in filmy:
        z, _, wiersze = wczytaj_film(args.zipy / f'{f}.zip')
        jest = set(z.namelist()); zipy[f] = z
        wiersze = sorted([r for r in wiersze if f'{f}/{r["filename"]}' in jest], key=lambda r: int(r['frame_number']))
        klatki += [(f, int(r['frame_number']), r['filename']) for r in wiersze[::args.krok]]
    assert [k[0] for k in klatki] == list(ref['film']) and [k[1] for k in klatki] == list(ref['nr']), 'inna lista klatek niz w ocena_test'
    baza = dict(np.load(args.baza)); wek = np.load(args.wektor)
    det = rfdetr.RFDETRNano(resolution=384, pretrain_weights=args.wagi)
    net = det.model.model.eval()
    bb = next(m for _, m in net.named_modules() if type(m).__name__ == 'WindowedDinov2WithRegistersBackbone')
    gamma = bb.layernorm.weight.detach().float().cpu().numpy(); beta = bb.layernorm.bias.detach().float().cpu().numpy()
    cap = {}
    bb.register_forward_hook(lambda mod, i, o: cap.__setitem__('fm', (o.feature_maps if hasattr(o, 'feature_maps') else o[0])[-1]))
    Y = np.zeros((len(klatki), 64), np.float32)
    for s in range(0, len(klatki), 32):
        part = [Image.open(io.BytesIO(zipy[f].read(f'{f}/{p}'))).convert('RGB') for f, _, p in klatki[s:s + 32]]
        with torch.inference_mode():
            det.predict(part, threshold=0.0)
        fm = cap['fm'].float(); b, k, h, w = fm.shape
        z = fm.permute(0, 2, 3, 1).reshape(b, h * w, k).cpu().numpy()
        red = ((z - beta) / np.abs(gamma) - baza['srodek']) @ baza['kier']          # b x 576 x 32
        a = red @ wek
        wag = np.exp((a - a.max(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9))
        wag /= wag.sum(1, keepdims=True)
        Y[s:s + b, :32] = red.mean(1); Y[s:s + b, 32:] = (wag[..., None] * red).sum(1)
        if s % 3200 == 0:
            print(f'[ekstrakcja] {s}/{len(klatki)}', flush=True)
    np.savez_compressed(args.dane / 'LATENT_test_ut_b0500.npz', Y=Y, film=ref['film'], nr=ref['nr'])


def analiza(args):
    L = np.load(args.dane / 'TEST_ut_b0500.npz'); Z = np.load(args.dane / 'LATENT_test_ut_b0500.npz')
    assert (L['nr'] == Z['nr']).all()
    Y = Z['Y'].astype(float); film = L['film']; ma = L['narzedzie']; gt = L['ramka_gt']
    rap = json.loads((args.dane / 'REPORT_test.json').read_text(encoding='utf-8'))
    prog = rap['prog']
    from ocena_test import iou_xywh
    filmy = np.unique(film)
    # pary (t-1, t) w obrebie filmu
    poprz = np.r_[-1, np.arange(len(film) - 1)]
    ok = np.r_[False, film[1:] == film[:-1]]
    rng = np.random.default_rng(0); kol = filmy.copy(); rng.shuffle(kol)
    polowy = [np.isin(film, kol[:len(kol) // 2]), np.isin(film, kol[len(kol) // 2:])]
    mu, sd = Y.mean(0), Y.std(0) + 1e-9
    Ys = (Y - mu) / sd
    koop = np.full(len(film), np.nan); mah = np.full(len(film), np.nan)
    for a_, b_ in ((0, 1), (1, 0)):
        fit = ok & polowy[a_]; ev = ok & polowy[b_]
        X0 = np.c_[Ys[poprz[fit]], np.ones(fit.sum())]; X1 = Ys[fit]
        K = np.linalg.solve(X0.T @ X0 + 1e-2 * np.eye(X0.shape[1]), X0.T @ X1)
        E_fit = X1 - X0 @ K
        S = np.cov(E_fit.T) + 1e-3 * np.eye(Y.shape[1]); Si = np.linalg.inv(S)
        E = Ys[ev] - np.c_[Ys[poprz[ev]], np.ones(ev.sum())] @ K
        koop[ev] = np.linalg.norm(E, axis=1); mah[ev] = np.einsum('ij,jk,ik->i', E, Si, E)
        r2 = 1 - (E ** 2).sum() / ((Ys[ev] - Ys[ev].mean(0)) ** 2).sum()
        print(f'Koopman dopasowany na polowie {a_}, R^2 przewidywania na drugiej: {r2:.3f}')
    mah_f = mah.copy()
    for f in filmy:
        m = (film == f) & np.isfinite(mah)
        mah_f[m] = mah[m] / np.median(mah[m])
    zmiana = np.where(ok, np.linalg.norm(Ys - Ys[poprz], axis=1), np.nan)

    def chybienia(nazwa):
        D = np.load(args.dane / f'TEST_{nazwa}.npz')
        p = D['pewnosc'][:, 0]; iou = iou_xywh(D['ramki'][:, 0], gt)
        traf = ma & (p >= prog) & (iou >= .5)
        return p, traf, ma & ~traf, ma & (p >= prog) & (iou < .5)

    miary = {'wektor_slaby': -L['wektor'][:, 0], 'obecnosc_z_mapy_slaba': -L['wektor'][:, 2],
             'zmiana_klatki': zmiana, 'koopman': koop, 'koopman_langevin': mah, 'koopman_langevin_wzgl_filmu': mah_f}
    raport = {}
    for det in ('ut_b0500', 'b1500_best'):
        p, traf, chyb, pewne_chyb = chybienia(det)
        m = dict(miary, pewnosc_detektora_niska=1 - p)
        wsp = ok & ma
        raport[det] = {'klatek_z_narzedziem': int(wsp.sum()), 'chybionych': int((chyb & wsp).sum()),
                       'pewnych_chybien': int((pewne_chyb & wsp).sum()),
                       'auc_chybienie': {k: auc(v[chyb & wsp], v[traf & wsp]) for k, v in m.items()},
                       'auc_pewne_chybienie': {k: auc(v[pewne_chyb & wsp], v[traf & wsp]) for k, v in m.items()}}
        # polaczenie: srednia rang wektora i Koopmana+Langevina
        rk = lambda v: np.argsort(np.argsort(np.nan_to_num(v, nan=np.nanmedian(v)))) / len(v)
        laczna = rk(miary['wektor_slaby']) + rk(miary['koopman_langevin_wzgl_filmu'])
        raport[det]['auc_chybienie']['wektor+koopman_langevin'] = auc(laczna[chyb & wsp], laczna[traf & wsp])
        raport[det]['auc_pewne_chybienie']['wektor+koopman_langevin'] = auc(laczna[pewne_chyb & wsp], laczna[traf & wsp])
        # czy zaskoczenie niesie cos ponad wektor: korelacja rang
        v1, v2 = miary['wektor_slaby'][wsp], miary['koopman_langevin'][wsp]
        g = np.isfinite(v1) & np.isfinite(v2)
        raport[det]['korelacja_rang_wektor_koopman'] = float(np.corrcoef(np.argsort(np.argsort(v1[g])), np.argsort(np.argsort(v2[g])))[0, 1])
    print(json.dumps(raport, indent=1, ensure_ascii=False))
    (args.dane / 'REPORT_koopman_zaskoczenie.json').write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding='utf-8')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dane', type=Path, required=True)
    ap.add_argument('--ekstrakcja', action='store_true')
    ap.add_argument('--zipy', type=Path)
    ap.add_argument('--split', type=Path)
    ap.add_argument('--wagi', default='results/koopman_langevin_epochs/checkpoints/ut_b0500_best.pth')
    ap.add_argument('--baza', type=Path, default=Path('results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500/baza.npz'))
    ap.add_argument('--wektor', type=Path, default=Path('experiments/krystalizacja_latentu/wyniki/dowody/RDZEN_bez_etykiet_32.npy'))
    ap.add_argument('--krok', type=int, default=10)
    args = ap.parse_args(argv)
    if args.ekstrakcja:
        ekstrakcja(args)
    else:
        analiza(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
