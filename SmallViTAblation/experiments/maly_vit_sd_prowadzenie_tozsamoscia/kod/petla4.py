#!/usr/bin/env python3
"""Pętla, wersja 4.2 (maly_vit_petla_sd_rownanie/PLAN.md): metoda użytkownika z prognozą latentu tożsamości na JEDEN
krok naprzód (Δz(t+1) = S_pęd(t) + S_paczka(t), `prognoza_krok.py`) i Stable Diffusion z maskami zamiast FastGAN-a.

Obrót (7 × 500 kroków po rozgrzewce 1000 kroków, 134 klatki na obrót, jak w wersji 3), ramię `sd_prognoza`:
  1. prognoza na krok t+1 bez nowych klatek (pęd + średnia paczka obecnego zbioru): 6 kotwic, w których siła
     tożsamości narzędzia będzie po kroku najsłabsza;
  2. dla każdej kotwicy 8 wersji z SD (4 × ten sam układ, 4 × narzędzie przestawione); wycinek każdej wersji →
     DINOv2 → lista podobnych prawdziwych klatek → grupa jej pierwszych 22 klatek, które wybór i tak by wziął;
  3. prognoza na krok t+1 z paczką z tej grupy: gdzie będzie siła tożsamości sond kotwicy; wybór wersji, której
     klatki najmocniej podniosą tożsamość (pęd wspólny, decyduje S_paczka);
  4. 134 klatki na przemian z 6 wybranych list (jak wersja 3), uczenie 500 kroków.
We wszystkich ramionach: sprawdzian równania na prawdziwych krokach uczenia (`prognoza_krok.Sprawdzian`) oraz
przewidziane pchnięcie tożsamości przez dodany zestaw klatek (to samo równanie, przed uczeniem).

Ramiona: `sd_prognoza` (metoda), kontrole bez prognozy `losowe` i `kolejne` (jak w wersji 3), `bez`, pomocnicze
`sd_bez_prognozy` (kotwice według obecnej siły, wersja SD losowa: jedyna różnica wobec metody to prognoza).
Dodatkowe ramiona wyboru przez spadek straty kontrolnej (docs/25, nie metoda użytkownika, tylko przez --ramiona):
`sd_strata`, `sd_strata_losowo`, `kotwice_strata`.

    <venv>/python -u petla4.py --ziarno 47 [--proba] [--ramiona ...]   ->  $MALY_VIT_WYNIKI/petla_krok_s47/...
"""
from __future__ import annotations

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import dane as D
import prognoza_krok as PK
import rownanie as RW
import sila as SI
import trening as T
import uklady as UK
import wybor as WB
import wybor4 as W4

ROZGRZEWKA, KROKI_ROZG, OBROTY, KROKI_OBR = 2, 500, 7, 500
NOWYCH, KOTWIC, CO_OCENA = 134, 6, 100
WERSJI, N_PACZEK_F = 8, 8
N_SD, MAX_SYNT, N_KONTROLA, N_FILMOW_KONTROLI, N_LOSOWYCH_GRUP, N_BEZPOSR = 48, 40, 512, 10, 6, 3
NR_RAMIENIA = dict(losowe=2, bez=3, kolejne=4, sd_prognoza=31, sd_bez_prognozy=32, sd_strata=11, sd_strata_losowo=12,
                   kotwice_strata=13)
RAMIONA = 'sd_prognoza,losowe,kolejne,bez,sd_bez_prognozy'
SPR_NA_POCZATKU, SPR_CO = 10, 50          # sprawdzian równania: pierwsze 10 kroków obrotu i co 50-ty


def zj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float), encoding='utf-8')


def kotwice_png(opis, out):
    man = D.jz(D.PROBE / 'manifest.json'); out.mkdir(parents=True, exist_ok=True)
    sciezki = []
    for i, o in enumerate(opis):
        if o['wariant'] == 'original':
            p = out / f'kotwica_{i:03d}.png'
            if not p.exists():
                Image.open(D.PROBE / man[i]['image']).convert('RGB').save(p)
            sciezki.append(p)
    return sciezki


def manifest_pelny():
    out = {}
    for l in open(D.DATA_ROOT / 'pool_manifest.jsonl', encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            out[r['frame_id']] = dict(frame_id=r['frame_id'], path=r['path'], native_width=int(r.get('native_width', 720)),
                                      native_height=int(r.get('native_height', 480)))
    return out


def obraz_576(sciezka):
    im = cv2.cvtColor(cv2.imread(str(sciezka), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (UK.SD_SZER, UK.SD_WYS), interpolation=cv2.INTER_AREA)


def bank_sd(G, pula_w, pelny, s, k, n, out):
    """Klatki SD obrotu k dla ramion `sd_strata*` (wspólne: te same układy i ziarna)."""
    f = out / 'sd_bank' / f'k{k:02d}.npz'
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z['X_sd'], list(z['uklady'])
    rng = np.random.default_rng(100 * s + k)
    uk = []
    for j in rng.permutation(len(pula_w)):
        r = pelny.get(pula_w[int(j)]['frame_id'])
        u = UK.uklad(r) if r else None
        if u is not None:
            uk.append(u)
        if len(uk) == n:
            break
    pol = len(uk) // 2
    uk = [dict(u, przepis='R2') for u in uk[:pol]] + [dict(UK.przeksztalc(u, rng)[0], przepis='R1') for u in uk[pol:]]
    X_sd = G.rysuj(np.stack([UK.maski(u) for u in uk]), ziarno=100 * s + k)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, X_sd=X_sd, uklady=np.array(uk, dtype=object))
    return X_sd, uk


def zbior_sd(X_sd, uk, s, k):
    X = np.stack([cv2.resize(x, (D.W_, D.H_), interpolation=cv2.INTER_AREA) for x in X_sd])
    w = [dict(frame_id=f'sd_s{s}_k{k:02d}/{i:06d}', ramki=UK.ramka_vit(u), przepis=u['przepis'], zrodlo_ukladu=u['frame_id'])
         for i, u in enumerate(uk)]
    return D.Zbior(w, X)


def grupa_z_listy(lista, pula_id, zajete, n):
    """Pierwsze n klatek listy wyszukiwania, które wybór i tak by wziął: z puli, nie bliżej niż ±30 klatek filmu od
    zajętych ani od siebie nawzajem."""
    gr = []
    for c in lista:
        f = c['frame_id']
        if f not in pula_id or zajete.blisko(f):
            continue
        v, kf = WB._vf(f)
        if any(WB._vf(g)[0] == v and abs(WB._vf(g)[1] - kf) <= WB.OKNO for g in gr):
            continue
        gr.append(f)
        if len(gr) == n:
            break
    return gr


def pchniecie_zestawu(u, fids, pula_id, sondy, obecn, wek, z0, paczka=32):
    """Przewidziane S_paczka zestawu klatek (średnia z paczek po ~32 klatki, obie wersje odbicia): o ile jeden krok
    na tych klatkach podniesie siłę tożsamości 180 sond. Liczone przed uczeniem, przy obecnych wagach."""
    if not fids:
        return None
    zb_d = D.zbior_z_puli([pula_id[f] for f in fids])
    czesci = np.array_split(np.arange(len(zb_d)), int(np.ceil(len(zb_d) / paczka)))
    return np.mean([PK.pchniecie_paczki(u, zb_d, c, sondy, obecn, wek, z0)[0] for c in czesci], 0)


def wersje_krok(G, u, kotw, s, k, rng, kd, prog, sondy, obecn, pula_id, zajete, wykl, wolne, z_prognoza):
    """Wersje SD kotwic, wyszukanie, grupy klatek i prognoza jednego kroku dla każdej grupy.
    kotw: lista dict(uklad, obraz576, sonda, moje) - moje = indeksy sond tej kotwicy wśród 180 sond z narzędziem.
    prog: wynik PK.prognoza_bez_nowych (z0, s0, ped, ...). z_prognoza: True = wybór wersji o największym przewidzianym
    wzroście siły tożsamości sond kotwicy; False = wersja losowa (ramię `sd_bez_prognozy`).
    Zwraca (listy wyszukiwania wybranych wersji, raport, grupy wybranych wersji)."""
    wek = wek_glob[0]; z0 = prog['z0']
    grupa = max(1, NOWYCH // max(1, len(kotw)))
    uk_all, zrodlo = [], []
    for b, kk in enumerate(kotw):
        for w in range(WERSJI):
            uk_all.append(dict(kk['uklad'], przepis='R2') if w < WERSJI // 2 else
                          dict(UK.przeksztalc(kk['uklad'], rng)[0], przepis='R1'))
            zrodlo.append(b)
    X_sd = G.rysuj(np.stack([UK.maski(x) for x in uk_all]), ziarno=100 * s + k)
    zrodlo = np.array(zrodlo)
    listy = []
    for i, (x, uu) in enumerate(zip(X_sd, uk_all)):
        p = kd / 'zapytania' / f'kotwica{zrodlo[i]}_wersja{i % WERSJI}.png'
        Image.fromarray(UK.wycinek_zapytania(x, uu)).save(p)
        listy.append(WB.zapytaj(p, wykl, p.with_name(p.stem + '_zapytanie.json')))
    grupy = [grupa_z_listy(l, pula_id, zajete, grupa) for l in listy]
    los = W4.grupy_losowe(wolne, zajete, set(), N_LOSOWYCH_GRUP, rng, rozmiar=grupa)
    wszystkie = sorted({f for g in grupy for f in g} | {f for g in los for f in g['fids']})
    zb_k = D.zbior_z_puli([pula_id[f] for f in wszystkie]); poz = {f: i for i, f in enumerate(wszystkie)}
    P, N = [], []
    for g in grupy:
        if len(g) >= max(1, grupa // 2):
            m, sz = PK.pchniecie_paczki(u, zb_k, [poz[f] for f in g], sondy, obecn, wek, z0)
        else:
            m, sz = None, None
        P.append(m); N.append(sz)
    P_los = [PK.pchniecie_paczki(u, zb_k, [poz[f] for f in g['fids']], sondy, obecn, wek, z0)[0] for g in los]
    wyb_listy, wyb_grupy, rap, kafle = [], [], [], []
    ref = [UK.wycinek_176(kk['obraz576'], kk['uklad'])[0] for kk in kotw]
    for b, kk in enumerate(kotw):
        ii = [int(i) for i in np.where(zrodlo == b)[0]]
        mo = kk['moje']
        wynik = [float(P[i][mo].mean()) if P[i] is not None else -np.inf for i in ii]
        wazne = [n for n, i in enumerate(ii) if P[i] is not None]
        if not wazne:
            continue
        n_w = wazne[int(np.argmax([wynik[n] for n in wazne]))] if z_prognoza else wazne[int(rng.integers(len(wazne)))]
        j = ii[n_w]
        wyb_listy.append(listy[j]); wyb_grupy.append(grupy[j])
        s0_k, ped_k = float(prog['s0'][mo].mean()), float(prog['ped'][mo].mean())
        los_k = [float(p[mo].mean()) for p in P_los]
        rap.append(dict(sonda=int(kk['sonda']), frame_id=kk['uklad']['frame_id'], sila_teraz=s0_k,
                        sila_za_krok_bez_nowych=float(prog['s1'][mo].mean()), ped=ped_k,
                        paczka_wersji=[w if np.isfinite(w) else None for w in wynik],
                        paczka_wersji_wszystkie_sondy=[float(P[i].mean()) if P[i] is not None else None for i in ii],
                        szum_wersji=[float(N[i][mo].mean()) if N[i] is not None else None for i in ii],
                        przepisy=[uk_all[i]['przepis'] for i in ii], klatek_w_grupie=[len(grupy[i]) for i in ii],
                        wybrana=n_w, paczka_wybranej=wynik[n_w], sila_za_krok_z_wybrana=s0_k + ped_k + wynik[n_w],
                        paczka_grup_losowych=float(np.mean(los_k)) if los_k else None,
                        paczka_grup_losowych_wszystkie_sondy=float(np.mean([p.mean() for p in P_los])) if P_los else None))
        rzad = [ref[b]] + [UK.wycinek_176(X_sd[i], uk_all[i])[0] for i in ii]
        rzad = [cv2.copyMakeBorder(np.ascontiguousarray(x), 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                   value=(40, 220, 40) if n - 1 == n_w else (0, 0, 0)) for n, x in enumerate(rzad)]
        kafle.append(np.concatenate(rzad, 1))
    if kafle:
        cv2.imwrite(str(kd / 'WERSJE_KOTWIC.jpg'), cv2.cvtColor(np.concatenate(kafle, 0), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    return wyb_listy, rap, wyb_grupy


wek_glob = [None]


def main(argv=None) -> int:
    global OBROTY, KROKI_OBR, NOWYCH, KOTWIC, CO_OCENA, N_SD, MAX_SYNT, N_KONTROLA, KROKI_ROZG, WERSJI, N_PACZEK_F
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ziarno', type=int, required=True)
    ap.add_argument('--ramiona', default=RAMIONA)
    ap.add_argument('--proba', action='store_true', help='próba kodu: krótka rozgrzewka i obroty, wyniki/proba_s*')
    ap.add_argument('--obroty', type=int, default=OBROTY)
    a = ap.parse_args(argv)
    s = a.ziarno
    OBROTY = a.obroty
    if a.proba:
        OBROTY, KROKI_OBR, NOWYCH, KOTWIC, CO_OCENA, N_SD, MAX_SYNT, N_KONTROLA, KROKI_ROZG = 2, 40, 24, 2, 20, 16, 8, 128, 60
        WERSJI, N_PACZEK_F = 4, 2
    out = Path(os.environ['MALY_VIT_WYNIKI']) / (f'proba_krok_s{s}' if a.proba else f'petla_krok_s{s}'); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    baza = D.baza(); wal_w, wal_X = D.walidacja(); sondy = D.sondy()
    _, _, _, _, obecn, opis = sondy
    wykl_baz = D.wykluczone_bazowe(baza.wiersze)
    pula_w = D.pula(wykl_baz)
    pelny = manifest_pelny()
    # zbiór kontrolny (tylko dla ramion *_strata): 10 filmów puli spoza filmów bazy, wyjęte z puli wszystkich ramion
    filmy_bazy = {UK.film_i_klatka(r['frame_id'])[0] for r in baza.wiersze}
    filmy = sorted({r['video'] for r in pula_w})
    kand = [f for f in filmy if f not in filmy_bazy] or filmy
    filmy_k = kand[::max(1, len(kand) // N_FILMOW_KONTROLI)][:N_FILMOW_KONTROLI]
    kontr_w = [r for r in pula_w if r['video'] in filmy_k]
    sel = sorted(np.random.default_rng(0).choice(len(kontr_w), min(N_KONTROLA, len(kontr_w)), replace=False).tolist())
    kontrola = RW.Kontrola(D.zbior_z_puli([kontr_w[i] for i in sel]))
    pula_w = [r for r in pula_w if r['video'] not in filmy_k]
    pula_id = {r['frame_id']: r for r in pula_w}
    kot_png = kotwice_png(opis, out / 'kotwice')
    poz_obecn = np.where(obecn)[0]
    orig = [j for j, i in enumerate(poz_obecn) if opis[i]['wariant'] == 'original']      # 12 kotwic wśród 180 sond
    zr_obecn = [opis[i]['zrodlo'] for i in poz_obecn]
    kotwice = {}
    for j in orig:
        i = int(poz_obecn[j]); r = pelny[opis[i]['frame_id']]
        kotwice[j] = dict(uklad=UK.uklad(r), obraz576=obraz_576(D.DATA_ROOT / r['path']), sonda=i,
                          moje=np.array([jj for jj, z in enumerate(zr_obecn) if z == zr_obecn[j]]))
    zj(out / 'KONTROLA.json', dict(filmy=filmy_k, klatek=len(sel), filmy_bazy_wykluczone=len(set(filmy_k) & filmy_bazy) == 0,
                                   kotwice_bez_ukladu=[j for j, kk in kotwice.items() if kk['uklad'] is None]))
    print(f's{s}: dane gotowe ({time.time() - t0:.0f} s): baza {len(baza)}, val {len(wal_w)}, pula {len(pula_w)}, '
          f'kontrola {len(sel)} z {len(filmy_k)} filmów, kotwic z układem {sum(kk["uklad"] is not None for kk in kotwice.values())}',
          flush=True)

    def nowa_ocena(lista, ramie, obrot_ref):
        def ocena(u):
            ap_ = T.ap50(u.wykryj(wal_X), [r['ramki'] for r in wal_w])
            lista.append(dict(ramie=ramie, obrot=obrot_ref[0], krok=u.krok, mAP50=ap_))
        return ocena

    # ---------- rozgrzewka (wspólna dla ramion; na Edenie kopia z wersji 2/3 o tym samym ziarnie)
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
    wek = SI.wektor(lat_wsp[-1][-5:].astype(np.float32).mean(0), obecn)      # wektor tożsamości jak w wersji 3
    wek_glob[0] = wek
    s_wsp = [SI.sila(l[:, obecn, 0], wek) for l in lat_wsp]
    zj(out / 'KOTWICE.json', {int(j): dict(sonda=kk['sonda'], frame_id=opis[kk['sonda']]['frame_id'], sond_kotwicy=len(kk['moje']),
                                           z_ukladem=kk['uklad'] is not None) for j, kk in kotwice.items()})

    # ---------- ramiona
    G = None
    for ramie in a.ramiona.split(','):
        kat = out / ramie
        if (kat / 'KONIEC.json').exists():
            continue
        u = T.Uczen(s); u.wczytaj(torch.load(wsp / f'k{ROZGRZEWKA:02d}.pt', weights_only=False))
        rng = np.random.default_rng(1000 * s + NR_RAMIENIA[ramie])
        rng_f = np.random.default_rng(5000 * s + NR_RAMIENIA[ramie])          # paczki do siły: osobny strumień
        zb = D.Zbior(list(baza.wiersze), baza.X.copy())
        zajete = WB.Zajete(sorted(wykl_baz))
        dodane, oceny, obr = [], list(ocen_wsp), [ROZGRZEWKA]
        ocena = nowa_ocena(oceny, ramie, obr)
        s_hist = list(s_wsp)
        spr = PK.Sprawdzian(sondy, obecn, wek, SPR_NA_POCZATKU, SPR_CO)
        if ramie.startswith('sd_') and G is None:
            import generator_sd as GS
            G = GS.GeneratorSD()
            zj(out / f'SD_{ramie}.json', G.zrodlo); print(f's{s}: SD z {G.zrodlo["unet"]}', flush=True)
        for k in range(ROZGRZEWKA + 1, ROZGRZEWKA + OBROTY + 1):
            tk = time.time(); obr[0] = k
            kd = kat / f'k{k:02d}'; (kd / 'zapytania').mkdir(parents=True, exist_ok=True)
            info = dict(obrot=k, krok=u.krok)
            juz = {d['frame_id'] for d in dodane}
            wolne = [r for r in pula_w if r['frame_id'] not in juz]
            wykl = wykl_baz | juz
            # --- prognoza latentu tożsamości na krok t+1 bez nowych klatek (wszystkie ramiona)
            prog = PK.prognoza_bez_nowych(u, zb, sondy, obecn, wek, rng_f, N_PACZEK_F)
            info['prognoza_krok'] = dict(sila_teraz=float(prog['s0'].mean()), ped=float(prog['ped'].mean()),
                                         paczka_srednia=float(prog['paczka'].mean()), sila_za_krok=float(prog['s1'].mean()),
                                         paczek_w_sredniej=N_PACZEK_F)
            nowe, synt_zb = [], None
            # --- wybór klatek
            if ramie in ('sd_prognoza', 'sd_bez_prognozy'):
                z_prog = ramie == 'sd_prognoza'
                ocena_k = {j: float(prog['s1' if z_prog else 's0'][kotwice[j]['moje']].mean()) for j in orig
                           if kotwice[j]['uklad'] is not None}
                kot = sorted(ocena_k, key=lambda j: ocena_k[j])[:KOTWIC]            # najsłabsza tożsamość
                listy, rap, gr_wyb = wersje_krok(G, u, [kotwice[j] for j in kot], s, k, rng, kd, prog, sondy, obecn,
                                                 pula_id, zajete, wykl, wolne, z_prog)
                nowe = [dict(d, zrodlo='pula') for d in WB.koopman(listy, pula_id, zajete, NOWYCH)]
                w_grupach = {f for g in gr_wyb for f in g}
                info['wersje_sd'] = rap
                info['kotwice_ocena'] = {int(j): v for j, v in ocena_k.items()}
                info['czesc_zestawu_z_ocenionych_grup'] = float(np.mean([d['frame_id'] in w_grupach for d in nowe])) if nowe else None
            elif ramie in ('sd_strata', 'sd_strata_losowo', 'kotwice_strata'):
                rownanie = ramie != 'sd_strata_losowo'
                L0, gk = (kontrola.gradient(u) if rownanie else (kontrola.strata(u), None))
                info['strata_kontrolna_start'] = L0
                uzyte: set = set()
                if ramie.startswith('sd_'):
                    X_sd, uk = bank_sd(G, pula_w, pelny, s, k, N_SD, out)
                    synt_zb = zbior_sd(X_sd, uk, s, k)
                    listy = []
                    for i, (x, uu) in enumerate(zip(X_sd, uk)):
                        p = kd / 'zapytania' / f'sd_{i:03d}.png'
                        Image.fromarray(UK.wycinek_zapytania(x, uu)).save(p)
                        listy.append(WB.zapytaj(p, wykl, kd / 'zapytania' / f'zapytanie_sd_{i:03d}.json'))
                    grupy = W4.grupy_z_list(listy, pula_id, zajete, uzyte, na_liste=1)
                    fs = [w['frame_id'] for w in synt_zb.wiersze]
                    grupy += [dict(fids=fs[i:i + W4.ROZMIAR], lista=-2, syntetyczne=True) for i in range(0, len(fs), W4.ROZMIAR)]
                else:
                    listy = [WB.zapytaj(p, wykl, kd / 'zapytania' / f'zapytanie_{p.stem}.json') for p in kot_png]
                    grupy = W4.grupy_z_list(listy, pula_id, zajete, uzyte, na_liste=2)
                for i, g in enumerate(grupy):
                    g['id'] = i
                los_gr = W4.grupy_losowe(wolne, zajete, uzyte, N_LOSOWYCH_GRUP, rng) if rownanie else []
                prog_b = -np.inf
                if rownanie:
                    real = sorted({f for g in grupy + los_gr if not g['syntetyczne'] for f in g['fids']})
                    zb_r = D.zbior_z_puli([pula_id[f] for f in real]); poz_r = {f: i for i, f in enumerate(real)}
                    poz_s = {w['frame_id']: i for i, w in enumerate(synt_zb.wiersze)} if synt_zb is not None else {}

                    def idx_grupy(g):
                        return (synt_zb, [poz_s[f] for f in g['fids']]) if g['syntetyczne'] else (zb_r, [poz_r[f] for f in g['fids']])
                    for g in grupy + los_gr:
                        z_, i_ = idx_grupy(g)
                        g.update(RW.ocen_grupe(u, z_, np.array(i_), gk))
                    prog_b = W4.prog_bezpiecznika(los_gr)
                    z0 = RW.latent_narzedzia_sond(u, sondy)
                    kal = []
                    for g in sorted(grupy, key=lambda g: -g['zysk'])[:N_BEZPOSR] + los_gr[:N_BEZPOSR]:
                        z_, i_ = idx_grupy(g)
                        kal.append(dict(id=g.get('id', -1), losowa=bool(g.get('losowa')), syntetyczne=g['syntetyczne'],
                                        pierwszy_rzad=g['zysk'], bezposrednio=RW.zysk_bezposredni(u, kontrola, z_, np.array(i_), L0),
                                        sila_tozsamosci=RW.sila_tozsamosci_grupy(u, z_, np.array(i_), sondy, obecn, wek, z0)))
                    info['kalibracja'] = kal
                    info['dryf_pedu'] = RW.dryf_pedu(u, kontrola, gk, L0)
                    info['grupy_losowe'] = [dict(zysk=g['zysk'], szum=g['szum']) for g in los_gr]
                    for g in los_gr:
                        for f in g['fids']:
                            uzyte.discard(f)
                nowe, rap = W4.wybierz(grupy, zajete, NOWYCH, MAX_SYNT, rng, wolne, prog_b,
                                       kolejnosc='zysk' if rownanie else 'losowa')
                info['wybor'] = rap
                info['grupy'] = [dict(id=g['id'], syntetyczne=g['syntetyczne'], lista=g['lista'], n=len(g['fids']),
                                      zysk=g.get('zysk'), szum=g.get('szum')) for g in grupy]
            elif ramie == 'losowe':
                nowe = [dict(d, zrodlo='pula') for d in WB.losowe(wolne, zajete, NOWYCH, rng)]
            elif ramie == 'kolejne':
                nowe = [dict(d, zrodlo='pula') for d in WB.kolejne(wolne, [r['frame_id'] for r in baza.wiersze] + sorted(juz),
                                                                   NOWYCH, rng)]
            if nowe:
                real_n = [d for d in nowe if d['zrodlo'] != 'sd']
                if real_n:
                    zb.dolacz(D.zbior_z_puli([pula_id[d['frame_id']] for d in real_n]))
                sy = [d for d in nowe if d['zrodlo'] == 'sd']
                if sy:
                    poz = {w['frame_id']: i for i, w in enumerate(synt_zb.wiersze)}
                    ii = [poz[d['frame_id']] for d in sy]
                    zb.dolacz(D.Zbior([synt_zb.wiersze[i] for i in ii], synt_zb.X[ii]))
                for d in nowe:
                    d['obrot'] = k
                dodane += nowe
            # --- przewidziane pchnięcie tożsamości przez dodane klatki (to samo równanie, przed uczeniem, każde ramię)
            real_d = [d['frame_id'] for d in nowe if d['zrodlo'] != 'sd']
            pz = pchniecie_zestawu(u, real_d, pula_id, sondy, obecn, wek, prog['z0'])
            if pz is not None:
                info['pchniecie_dodanych'] = dict(wszystkie_sondy=float(pz.mean()),
                                                  sondy_kotwic={int(j): float(pz[kotwice[j]['moje']].mean()) for j in orig})
            lat = []
            u.kroki(zb, KROKI_OBR, rng, sondy=sondy, latent_out=lat, co_ocena=CO_OCENA, ocena=ocena, obserwator=spr)
            arr = np.stack([l for _, l in lat])
            s_rot = SI.sila(arr[:, obecn, 0], wek)
            s_hist.append(s_rot)
            info['sprawdzian_rownania'] = spr.wynik()
            if 'strata_kontrolna_start' in info:
                info['strata_kontrolna_koniec'] = kontrola.strata(u)
            info.update(dodane=len(nowe), syntetycznych=sum(d['zrodlo'] == 'sd' for d in nowe), zbior=len(zb), krok_po=u.krok,
                        ocena=oceny[-1], sila_tozsamosci_srednia=float(s_rot[-5:].mean()), sekundy=time.time() - tk,
                        wybrane=nowe)
            zj(kd / 'OBROT.json', info)
            zj(kat / 'OCENY.json', oceny)
            sp = info['sprawdzian_rownania'] or {}
            txt = (f'  równanie 1 krok: tożsamość {sp.get("tozsamosc", {}).get("rownanie", float("nan")):.1f} % '
                   f'(sam pęd {sp.get("tozsamosc", {}).get("sam_ped", float("nan")):.1f} %), cały latent '
                   f'{sp.get("caly_latent", {}).get("rownanie", float("nan")):.1f} %')
            if 'pchniecie_dodanych' in info:
                txt += f'  pchnięcie dodanych {info["pchniecie_dodanych"]["wszystkie_sondy"]:+.4f}'
            if info.get('wersje_sd'):
                r_ = info['wersje_sd']
                txt += (f'  SD: paczka wybranej {np.mean([r["paczka_wybranej"] for r in r_]):+.4f} vs losowych grup '
                        f'{np.mean([r["paczka_grup_losowych"] for r in r_ if r["paczka_grup_losowych"] is not None]):+.4f}')
            print(f's{s} {ramie} k{k}: mAP50 {oceny[-1]["mAP50"]:.4f}  dodane {len(nowe)} (zbiór {len(zb)}){txt}  '
                  f'({time.time() - tk:.0f} s)', flush=True)
        np.save(kat / 'SILA.npy', np.concatenate(s_hist).astype(np.float32))
        torch.save(u.net.state_dict(), kat / 'wagi_koncowe.pth')
        zj(kat / 'LOG.json', u.log)
        zj(kat / 'KONIEC.json', dict(ramie=ramie, ziarno=s, dodane=dodane, oceny=oceny, sekundy=time.time() - t0))
    print(f's{s}: gotowe ({time.time() - t0:.0f} s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
