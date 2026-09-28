#!/usr/bin/env python3
"""Pętla: mały ViT od zera, latent tożsamości → Koopman/Langevin → FastGAN → prawdziwe klatki → dalsze uczenie.

Metoda użytkownika (2026-09-28): uczymy od zera na 1500 klatkach; po 3 obrotach (checkpointach) znajdujemy w latencie
neurony tożsamości narzędzia (rdzeń i pełny latent, tozsamosc.py); z historii przewidujemy, jak urosną lub zmaleją
w kolejnych obrotach (prognoza.py); FastGAN tworzy obrazy narzędzia pasujące do przewidzianego latentu (generuj.py);
do nich szukamy najbliższych prawdziwych klatek w puli i dodajemy je do zbioru uczącego (wybor.py); uczymy dalej.
Punkty odniesienia z tego samego stanu po rozgrzewce: `losowe` (tyle samo losowych klatek, ta sama reguła
różnorodności) i `bez` (bez dodawania). Równa liczba kroków w każdym obrocie.

    <venv rfdetr>/python -u petla.py --ziarno 42   ->  wyniki/petla_s42/...
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dane as D
import generuj as GN
import prognoza as PR
import tozsamosc as TZ
import trening as T
import wybor as WB

HERE = Path(__file__).resolve().parent
ROZGRZEWKA, OBROTY, KROKI_OBR = 3, 6, 500
NOWYCH, PRZOD, KOTWIC, CO_OCENA = 134, 3, 6, 100
NR_RAMIENIA = dict(koopman=1, losowe=2, bez=3)


def zj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float), encoding='utf-8')


def kotwice_256(opis, tn):
    man = D.jz(D.PROBE / 'manifest.json')
    out = {}
    for i, o in enumerate(opis):
        if o['wariant'] == 'original':
            m = man[i]
            img = torch.from_numpy(np.asarray(Image.open(D.PROBE / m['image']).convert('RGB'))).permute(2, 0, 1)[None].float() / 255
            msk = torch.from_numpy(np.asarray(Image.open(D.PROBE / m['mask']).convert('L')))[None, None].float() / 255
            out[i] = dict(obraz256=img, maska256=msk, tn=tn[i])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ziarno', type=int, required=True)
    ap.add_argument('--ramiona', default='koopman,losowe,bez')
    ap.add_argument('--proba', action='store_true', help='próba kodu: krótkie obroty, wyniki/proba_s*')
    a = ap.parse_args(argv)
    s = a.ziarno
    global ROZGRZEWKA, OBROTY, KROKI_OBR, NOWYCH, KOTWIC, CO_OCENA
    if a.proba:
        ROZGRZEWKA, OBROTY, KROKI_OBR, NOWYCH, KOTWIC, CO_OCENA = 3, 2, 60, 10, 2, 30
    out = HERE / 'wyniki' / (f'proba_s{s}' if a.proba else f'petla_s{s}'); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    baza = D.baza(); wal_w, wal_X = D.walidacja(); sondy = D.sondy()
    X_s, tn, tt, ramki_s, obecn, opis = sondy
    wykl_baz = D.wykluczone_bazowe(baza.wiersze)
    pula_w = D.pula(wykl_baz); pula_id = {r['frame_id']: r for r in pula_w}
    kot256 = kotwice_256(opis, tn)
    poz_obecn = np.where(obecn)[0]
    print(f's{s}: dane gotowe ({time.time() - t0:.0f} s): baza {len(baza)}, val {len(wal_w)}, pula {len(pula_w)}', flush=True)

    def nowa_ocena(lista, ramie, obrot_ref):
        def ocena(u):
            ap_ = T.ap50(u.wykryj(wal_X), [r['ramki'] for r in wal_w])
            q = u.jakosc_sond(sondy)
            lista.append(dict(ramie=ramie, obrot=obrot_ref[0], krok=u.krok, mAP50=ap_, q_sond=float(np.nanmean(q[obecn])),
                              q_puste=float(np.nanmean(q[~obecn]))))
        return ocena

    # ---------- rozgrzewka (wspólna dla ramion)
    wsp = out / 'wspolny'
    if not (wsp / f'k{ROZGRZEWKA:02d}.pt').exists():
        u = T.Uczen(s); rng = np.random.default_rng(s)
        oceny, obr = [], [0]
        ocena = nowa_ocena(oceny, 'wspolny', obr)
        ocena(u)
        lat = [(0, u.latent_sond(sondy))]
        for k in range(1, ROZGRZEWKA + 1):
            obr[0] = k; tk = time.time()
            u.kroki(baza, KROKI_OBR, rng, sondy=sondy, latent_out=lat, co_ocena=CO_OCENA, ocena=ocena)
            (wsp / f'k{k:02d}').mkdir(parents=True, exist_ok=True)
            np.savez_compressed(wsp / f'k{k:02d}' / 'LATENT.npz', kroki=np.array([x for x, _ in lat]),
                                latent=np.stack([l for _, l in lat]))
            st = TZ.statystyki(np.stack([l for _, l in lat[-5:]]).astype(np.float32).mean(0), obecn, opis)
            pod = TZ.podsumowanie(st)
            zj(wsp / f'k{k:02d}' / 'OBROT.json', dict(obrot=k, krok=u.krok, ocena=oceny[-1], tozsamosc=pod, sekundy=time.time() - tk))
            torch.save(u.stan(), wsp / f'k{k:02d}.pt')
            print(f's{s} wspólny k{k}: mAP50 {oceny[-1]["mAP50"]:.4f}  rdzeń {pod["rdzen"]}  pełny {pod["pelny"]}  '
                  f'domknięcie {pod["domkniecie_srednie"]:.2f} (najsłabszy {pod["najslabszy_wariant"]})  ({time.time() - tk:.0f} s)', flush=True)
            lat = []
        zj(wsp / 'OCENY.json', oceny); zj(wsp / 'LOG.json', u.log)
    ocen_wsp = D.jz(wsp / 'OCENY.json')
    lat_wsp = [np.load(wsp / f'k{k:02d}' / 'LATENT.npz')['latent'] for k in range(1, ROZGRZEWKA + 1)]

    # ---------- ramiona
    G = None
    for ramie in a.ramiona.split(','):
        kat = out / ramie
        if (kat / 'KONIEC.json').exists():
            continue
        u = T.Uczen(s); u.wczytaj(torch.load(wsp / f'k{ROZGRZEWKA:02d}.pt', weights_only=False))
        rng = np.random.default_rng(1000 * s + NR_RAMIENIA[ramie])
        zb = D.Zbior(list(baza.wiersze), baza.X.copy())
        zajete = WB.Zajete(sorted(wykl_baz))
        dodane, oceny, obr = [], list(ocen_wsp), [ROZGRZEWKA]
        ocena = nowa_ocena(oceny, ramie, obr)
        lat_hist = list(lat_wsp)
        if ramie == 'koopman' and G is None:
            G = GN.laduj_generator()
        for k in range(ROZGRZEWKA + 1, ROZGRZEWKA + OBROTY + 1):
            tk = time.time(); obr[0] = k
            kd = kat / f'k{k:02d}'; kd.mkdir(parents=True, exist_ok=True)
            L_all = np.concatenate(lat_hist)
            st = TZ.statystyki(L_all[-5:].astype(np.float32).mean(0), obecn, opis)
            info = dict(obrot=k, tozsamosc_przed=TZ.podsumowanie(st))
            nowe = []
            if ramie == 'koopman':
                neur = TZ.sledzone(st)
                Z = TZ.stan(L_all, st, neur, obecn)
                H = PRZOD * KROKI_OBR // T.ZAPIS_CO // PR.B_BLOK
                fut, teraz, droga, diag = PR.prognozuj(Z, H)
                orig = [j for j, i in enumerate(poz_obecn) if opis[i]['wariant'] == 'original']
                kot = sorted(orig, key=lambda j: -droga[j])[:KOTWIC]
                wykl = wykl_baz | {d['frame_id'] for d in dodane}
                listy, fg = [], {}
                for j in kot:
                    i = int(poz_obecn[j])
                    kk = kd / f'kotwica_{i:03d}'
                    fg[str(i)] = GN.generuj(u, G, kot256[i], fut[j], teraz[j], st['mu'], st['sd'], neur, kk, seed=100 * s + k)
                    listy.append(WB.zapytaj(kk / 'target.png', wykl, kk / 'zapytanie_target.json'))
                nowe = WB.koopman(listy, pula_id, zajete, NOWYCH)
                info.update(neuronow_sledzonych=int(len(neur)), prognoza=diag, kotwice=[int(poz_obecn[j]) for j in kot],
                            droga_kotwic=[float(droga[j]) for j in kot], fastgan=fg)
            elif ramie == 'losowe':
                nowe = WB.losowe([r for r in pula_w if r['frame_id'] not in {d['frame_id'] for d in dodane}], zajete, NOWYCH, rng)
            if nowe:
                zb.dolacz(D.zbior_z_puli([pula_id[d['frame_id']] for d in nowe]))
                for d in nowe:
                    d['obrot'] = k
                dodane += nowe
            lat = []
            u.kroki(zb, KROKI_OBR, rng, sondy=sondy, latent_out=lat, co_ocena=CO_OCENA, ocena=ocena)
            arr = np.stack([l for _, l in lat])
            np.savez_compressed(kd / 'LATENT.npz', kroki=np.array([x for x, _ in lat]), latent=arr)
            lat_hist.append(arr)
            st2 = TZ.statystyki(arr[-5:].astype(np.float32).mean(0), obecn, opis)
            info.update(dodane=len(nowe), zbior=len(zb), krok=u.krok, ocena=oceny[-1], tozsamosc_po=TZ.podsumowanie(st2),
                        sekundy=time.time() - tk, wybrane=nowe)
            zj(kd / 'OBROT.json', info)
            zj(kat / 'OCENY.json', oceny)
            pr_ = info.get('prognoza', {})
            print(f's{s} {ramie} k{k}: mAP50 {oceny[-1]["mAP50"]:.4f}  dodane {len(nowe)} (zbiór {len(zb)})  rdzeń '
                  f'{info["tozsamosc_po"]["rdzen"]}  pełny {info["tozsamosc_po"]["pelny"]}  domknięcie '
                  f'{info["tozsamosc_po"]["domkniecie_srednie"]:.2f}'
                  + (f'  prognoza/brak zmiany: koopman {pr_.get("koopman", {}).get("wzgledem_braku_zmiany", float("nan")):.2f}, '
                     f'langevin {pr_.get("langevin", {}).get("wzgledem_braku_zmiany", float("nan")):.2f}' if pr_ else '')
                  + f'  ({time.time() - tk:.0f} s)', flush=True)
        torch.save(u.net.state_dict(), kat / 'wagi_koncowe.pth')
        zj(kat / 'LOG.json', u.log)
        zj(kat / 'KONIEC.json', dict(ramie=ramie, ziarno=s, dodane=dodane, oceny=oceny, sekundy=time.time() - t0))
    print(f's{s}: gotowe ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
