#!/usr/bin/env python3
"""Pętla, wersja 3 (maly_vit_petla_sila_tozsamosci/PLAN.md): Koopman/Langevin przewidują tylko siłę tożsamości
narzędzia, FastGAN steruje obrazem wzdłuż wektora tożsamości o przewidzianą zmianę, podobne prawdziwe klatki trafiają
do uczenia.

Różnice względem maly_vit_wektor_tozsamosci/kod/petla.py (wersja 2):
  - wektor tożsamości: LDA narzędzie/tło na sondach z końca rozgrzewki (krok 1000 = 21 epok), stały do końca biegu;
    stan prognozy = siła tożsamości 180 sond (rzut na wektor), nie 768 neuronów;
  - model prognozy wybierany w każdym obrocie testem wstecznym na historii siły (Koopman albo Langevin);
  - kotwice: 6 z 12 oryginalnych wycinków o największej przewidzianej |zmianie siły|;
  - FastGAN: inwersja kotwicy, sterowanie wejściem z do siły startu + przewidziana zmiana (`sila.generuj`);
  - ramiona: `tozsamosc` (metoda), `current` (to samo, obraz startowy bez zmiany siły), `losowe`, `kolejne`
    (kolejne klatki tych samych filmów), `bez`;
  - rozgrzewka wzięta z wersji 2 (te same ziarna i kod; kopia robiona przez petla_eden.slurm), a gdy jej brak —
    liczona tak samo (2 × 500 kroków).
Iteracja 2 (PLAN.md): --obroty 14 --top-k 180, ziarna 47–66.
Pomiar w każdym obrocie i ramieniu: prognoza siły z poprzedniego obrotu kontra to, co się stało.

    <venv rfdetr>/python -u petla.py --ziarno 47   ->  $MALY_VIT_WYNIKI/petla_s47/...
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
import sila as SI
import tozsamosc as TZ
import trening as T
import wybor as WB

HERE = Path(__file__).resolve().parent
ROZGRZEWKA, KROKI_ROZG, OBROTY, KROKI_OBR = 2, 500, 7, 500
NOWYCH, KOTWIC, CO_OCENA = 134, 6, 100
NR_RAMIENIA = dict(tozsamosc=1, losowe=2, bez=3, kolejne=4, current=5)   # losowe i bez: te same rng co w wersji 2


def zj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float), encoding='utf-8')


def kotwice_256(opis):
    man = D.jz(D.PROBE / 'manifest.json')
    out = {}
    for i, o in enumerate(opis):
        if o['wariant'] == 'original':
            m = man[i]
            img = torch.from_numpy(np.asarray(Image.open(D.PROBE / m['image']).convert('RGB'))).permute(2, 0, 1)[None].float() / 255
            msk = torch.from_numpy(np.asarray(Image.open(D.PROBE / m['mask']).convert('L')))[None, None].float() / 255
            out[i] = dict(obraz256=img, maska256=msk)
    return out


def main(argv=None) -> int:
    global OBROTY, KROKI_OBR, NOWYCH, KOTWIC, CO_OCENA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ziarno', type=int, required=True)
    ap.add_argument('--ramiona', default='tozsamosc,current,losowe,kolejne,bez')
    ap.add_argument('--proba', action='store_true', help='próba kodu: krótkie obroty w ramionach, wyniki/proba_s*')
    ap.add_argument('--obroty', type=int, default=OBROTY, help='obroty w ramionach (iteracja 2: 14)')
    ap.add_argument('--top-k', type=int, default=WB.TOP_K, help='długość listy podobnych klatek na kotwicę (iteracja 2: 180)')
    a = ap.parse_args(argv)
    s = a.ziarno
    OBROTY, WB.TOP_K = a.obroty, a.top_k
    if a.proba:
        OBROTY, KROKI_OBR, NOWYCH, KOTWIC, CO_OCENA = 2, 50, 10, 2, 25
    out = Path(os.environ['MALY_VIT_WYNIKI']) / (f'proba_s{s}' if a.proba else f'petla_s{s}'); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    baza = D.baza(); wal_w, wal_X = D.walidacja(); sondy = D.sondy()
    X_s, tn, tt, ramki_s, obecn, opis = sondy
    wykl_baz = D.wykluczone_bazowe(baza.wiersze)
    pula_w = D.pula(wykl_baz); pula_id = {r['frame_id']: r for r in pula_w}
    kot256 = kotwice_256(opis)
    poz_obecn = np.where(obecn)[0]
    orig = [j for j, i in enumerate(poz_obecn) if opis[i]['wariant'] == 'original']   # 12 kotwic wśród 180 sond
    print(f's{s}: dane gotowe ({time.time() - t0:.0f} s): baza {len(baza)}, val {len(wal_w)}, pula {len(pula_w)}', flush=True)

    def nowa_ocena(lista, ramie, obrot_ref):
        def ocena(u):
            ap_ = T.ap50(u.wykryj(wal_X), [r['ramki'] for r in wal_w])
            q = u.jakosc_sond(sondy)
            lista.append(dict(ramie=ramie, obrot=obrot_ref[0], krok=u.krok, mAP50=ap_, q_sond=float(np.nanmean(q[obecn])),
                              q_puste=float(np.nanmean(q[~obecn]))))
        return ocena

    # ---------- rozgrzewka (wspólna dla ramion; zwykle skopiowana z wersji 2)
    wsp = out / 'wspolny'
    if not (wsp / f'k{ROZGRZEWKA:02d}.pt').exists():
        u = T.Uczen(s); rng = np.random.default_rng(s)
        oceny, obr = [], [0]
        ocena = nowa_ocena(oceny, 'wspolny', obr)
        ocena(u)
        lat = [(0, u.latent_sond(sondy))]
        for k in range(1, ROZGRZEWKA + 1):
            obr[0] = k
            u.kroki(baza, KROKI_ROZG, rng, sondy=sondy, latent_out=lat, co_ocena=100, ocena=ocena)
            (wsp / f'k{k:02d}').mkdir(parents=True, exist_ok=True)
            np.savez_compressed(wsp / f'k{k:02d}' / 'LATENT.npz', kroki=np.array([x for x, _ in lat]),
                                latent=np.stack([l for _, l in lat]))
            torch.save(u.stan(), wsp / f'k{k:02d}.pt')
            print(f's{s} wspólny k{k}: mAP50 {oceny[-1]["mAP50"]:.4f}', flush=True)
            lat = []
        zj(wsp / 'OCENY.json', oceny); zj(wsp / 'LOG.json', u.log)
    ocen_wsp = D.jz(wsp / 'OCENY.json')
    lat_wsp = [np.load(wsp / f'k{k:02d}' / 'LATENT.npz')['latent'] for k in range(1, ROZGRZEWKA + 1)]

    # ---------- wektor tożsamości z końca rozgrzewki (21 epok), stały do końca
    wek = SI.wektor(lat_wsp[-1][-5:].astype(np.float32).mean(0), obecn)
    np.savez(out / 'WEKTOR.npz', mu=wek['mu'], sd=wek['sd'], w=wek['w'])
    zj(out / 'WEKTOR.json', {k: v for k, v in wek.items() if not isinstance(v, np.ndarray)})
    s_wsp = [SI.sila(l[:, obecn, 0], wek) for l in lat_wsp]                          # (T, 180) na obrót
    print(f's{s}: wektor tożsamości — siła narzędzia {wek["sila_narzedzia"]:.2f}, negatywów {wek["sila_negatywow"]:.2f} '
          f'(sd {wek["sd_negatywow"]:.2f})', flush=True)
    H = KROKI_OBR // T.ZAPIS_CO // PR.B_BLOK                                          # horyzont: jeden obrót

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
        s_hist = list(s_wsp)
        prog_poprz = None
        if ramie in ('tozsamosc', 'current') and G is None:
            G = GN.laduj_generator()
        for k in range(ROZGRZEWKA + 1, ROZGRZEWKA + OBROTY + 1):
            tk = time.time(); obr[0] = k
            kd = kat / f'k{k:02d}'; kd.mkdir(parents=True, exist_ok=True)
            Sb = PR.bloki(np.concatenate(s_hist)[:, :, None])
            info = dict(obrot=k, krok=u.krok)
            if prog_poprz is not None:
                info['sprawdzenie_poprz_obrotu'] = SI.sprawdz(prog_poprz, Sb[-1, :, 0], orig)
            pp = SI.prognozy(Sb, H)
            prog_poprz = pp
            ds_all = pp['prognozy'][pp['model']] - pp['teraz']
            info['prognoza'] = dict(model=pp['model'], test_wsteczny=pp['test_wsteczny'], punktow=pp['punktow'],
                                    zmiana_kotwic={m: (pp['prognozy'][m] - pp['teraz'])[orig].tolist() for m in SI.MODELE},
                                    sila_kotwic=pp['teraz'][orig].tolist(), zmiana_srednia=float(ds_all.mean()))
            nowe = []
            if ramie in ('tozsamosc', 'current'):
                kot = sorted(orig, key=lambda j: -abs(ds_all[j]))[:KOTWIC]
                wykl = wykl_baz | {d['frame_id'] for d in dodane}
                rap, sciezki = SI.generuj(u, G, [kot256[int(poz_obecn[j])] for j in kot], [ds_all[j] for j in kot], wek,
                                          kd, seed=100 * s + k, sterowac=(ramie == 'tozsamosc'))
                listy = []
                for b, (tgt, start) in enumerate(sciezki):
                    lt = WB.zapytaj(tgt, wykl, tgt.parent / 'zapytanie_target.json')
                    listy.append(lt)
                    if ramie == 'tozsamosc':                                  # czy prognoza zmienia wyszukane klatki
                        ls_ = WB.zapytaj(start, wykl, start.parent / 'zapytanie_start.json')
                        a_, b_ = [c['frame_id'] for c in lt], [c['frame_id'] for c in ls_]
                        rap[b]['wspolne_top90_ze_startem'] = len(set(a_[:90]) & set(b_[:90]))
                        rap[b]['wspolne_cala_lista_ze_startem'] = len(set(a_) & set(b_))
                        rap[b]['wspolne_top22_ze_startem'] = len(set(a_[:22]) & set(b_[:22]))
                    rap[b]['sonda'] = int(poz_obecn[kot[b]])
                nowe = WB.koopman(listy, pula_id, zajete, NOWYCH)
                info['fastgan'] = rap
            elif ramie == 'losowe':
                nowe = WB.losowe([r for r in pula_w if r['frame_id'] not in {d['frame_id'] for d in dodane}], zajete, NOWYCH, rng)
            elif ramie == 'kolejne':
                juz = {d['frame_id'] for d in dodane}
                nowe = WB.kolejne([r for r in pula_w if r['frame_id'] not in juz],
                                  [r['frame_id'] for r in baza.wiersze] + sorted(juz), NOWYCH, rng)
            if nowe:
                zb.dolacz(D.zbior_z_puli([pula_id[d['frame_id']] for d in nowe]))
                for d in nowe:
                    d['obrot'] = k
                dodane += nowe
            lat = []
            u.kroki(zb, KROKI_OBR, rng, sondy=sondy, latent_out=lat, co_ocena=CO_OCENA, ocena=ocena)
            arr = np.stack([l for _, l in lat])
            np.savez_compressed(kd / 'LATENT.npz', kroki=np.array([x for x, _ in lat]), latent=arr)
            s_hist.append(SI.sila(arr[:, obecn, 0], wek))
            info.update(dodane=len(nowe), zbior=len(zb), krok_po=u.krok, ocena=oceny[-1], sekundy=time.time() - tk, wybrane=nowe)
            zj(kd / 'OBROT.json', info)
            zj(kat / 'OCENY.json', oceny)
            sp = info.get('sprawdzenie_poprz_obrotu', {})
            fg = info.get('fastgan', [])
            txt_fg = ''
            if fg and ramie == 'tozsamosc':
                txt_fg = (f'  FastGAN część zmiany {np.median([r.get("czesc_zmiany", np.nan) for r in fg]):.2f} '
                          f'(odporna {np.median([r.get("czesc_zmiany_odporna", np.nan) for r in fg]):.2f}), '
                          f'wspólne top-90 ze startem {np.mean([r["wspolne_top90_ze_startem"] for r in fg]):.0f}')
            print(f's{s} {ramie} k{k}: mAP50 {oceny[-1]["mAP50"]:.4f}  dodane {len(nowe)} (zbiór {len(zb)})  model {pp["model"]}  '
                  f'poprz. prognoza/brak zmiany {sp.get("wybrany", {}).get("sondy", float("nan")):.2f} '
                  f'(K {sp.get("koopman", {}).get("sondy", float("nan")):.2f}, L {sp.get("langevin", {}).get("sondy", float("nan")):.2f})'
                  f'{txt_fg}  ({time.time() - tk:.0f} s)', flush=True)
        Sb = PR.bloki(np.concatenate(s_hist)[:, :, None])
        spr_k = SI.sprawdz(prog_poprz, Sb[-1, :, 0], orig)
        np.save(kat / 'SILA.npy', np.concatenate(s_hist).astype(np.float32))
        torch.save(u.net.state_dict(), kat / 'wagi_koncowe.pth')
        zj(kat / 'LOG.json', u.log)
        zj(kat / 'KONIEC.json', dict(ramie=ramie, ziarno=s, dodane=dodane, oceny=oceny, sprawdzenie_koncowe=spr_k,
                                     sekundy=time.time() - t0))
    print(f's{s}: gotowe ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
