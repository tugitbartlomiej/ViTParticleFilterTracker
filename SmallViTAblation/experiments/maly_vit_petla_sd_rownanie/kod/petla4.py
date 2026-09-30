#!/usr/bin/env python3
"""Pętla, wersja 4.1 (maly_vit_petla_sd_rownanie/PLAN.md): metoda użytkownika z nowymi modułami. Równanie ruchu
latentu przewiduje, jak rozwinie się latent tożsamości narzędzia w następnym obrocie; Stable Diffusion z maskami rysuje
wersje kotwic zamiast FastGAN-a; wersja najbliższa przewidzianej sile tożsamości jest zapytaniem do wyszukania podobnych
prawdziwych klatek, które trafiają do uczenia.

Obrót (7 × 500 kroków po rozgrzewce 1000 kroków, 134 klatki na obrót, jak w wersji 3), we wszystkich ramionach:
  1. prognoza siły tożsamości 180 sond na 100 i 500 kroków równaniem (`prognoza4.py`) oraz, dla porównania,
     Koopmanem albo Langevinem z wersji 3; po obrocie ich trafność wobec zmierzonej siły i „braku zmiany”;
  2. ramię `sd_prognoza`: 6 kotwic o największej przewidzianej zmianie; dla każdej 8 wersji z SD (4 × ten sam układ,
     4 × narzędzie przestawione); mały ViT mierzy siłę tożsamości każdej wersji; wybór wersji najbliższej sile
     kotwicy + przewidzianej zmianie; wyszukanie DINOv2 i klatki na przemian z 6 list (jak wersja 3);
  3. uczenie 500 kroków.

Ramiona: `sd_prognoza` (metoda), kontrole bez prognozy `losowe` i `kolejne` (jak w wersji 3), `bez`, oraz pomocnicze
`sd_teraz` (to samo co metoda, ale wersja najbliższa OBECNEJ sile: jedyna różnica to prognoza). Dodatkowe ramiona
wyboru przez spadek straty kontrolnej (propozycja z docs/25, nie metoda użytkownika): `sd_strata`, `sd_strata_losowo`,
`kotwice_strata`.

    <venv>/python -u petla4.py --ziarno 47 [--proba] [--ramiona ...]   ->  $MALY_VIT_WYNIKI/petla_s47/...
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
import prognoza as PR
import prognoza4 as PG
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
NR_RAMIENIA = dict(losowe=2, bez=3, kolejne=4, sd_prognoza=21, sd_teraz=22, sd_strata=11, sd_strata_losowo=12,
                   kotwice_strata=13)
RAMIONA = 'sd_prognoza,losowe,kolejne,bez,sd_teraz'


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


def wersje_kotwic(G, u, kotw, zmiana_przew, s, k, rng, kd, cel_teraz):
    """Wersje SD kotwic i wybór wersji najbliższej sile docelowej (siła kotwicy + przewidziana zmiana, albo sama siła
    kotwicy gdy cel_teraz). kotw: lista dict(uklad, obraz576, sonda). Zwraca (ścieżki zapytań, raport)."""
    uk_all, zrodlo = [], []
    for b, kk in enumerate(kotw):
        for w in range(WERSJI):
            uk_all.append(dict(kk['uklad'], przepis='R2') if w < WERSJI // 2 else
                          dict(UK.przeksztalc(kk['uklad'], rng)[0], przepis='R1'))
            zrodlo.append(b)
    X_sd = G.rysuj(np.stack([UK.maski(x) for x in uk_all]), ziarno=100 * s + k)
    wyc = [UK.wycinek_176(x, uu) for x, uu in zip(X_sd, uk_all)]
    ref = [UK.wycinek_176(kk['obraz576'], kk['uklad']) for kk in kotw]
    s_w = PG.sila_wycinkow(u, np.stack([a for a, _ in wyc]), np.stack([m for _, m in wyc]), wek_glob[0])
    s_r = PG.sila_wycinkow(u, np.stack([a for a, _ in ref]), np.stack([m for _, m in ref]), wek_glob[0])
    zrodlo = np.array(zrodlo)
    sciezki, rap, kafle = [], [], []
    for b, kk in enumerate(kotw):
        ii = np.where(zrodlo == b)[0]
        cel = float(s_r[b] + (0.0 if cel_teraz else zmiana_przew[b]))
        j = int(ii[np.argmin(np.abs(s_w[ii] - cel))])
        p = kd / 'zapytania' / f'wersja_kotwicy_{b}.png'
        Image.fromarray(UK.wycinek_zapytania(X_sd[j], uk_all[j])).save(p)
        sciezki.append(p)
        dz = float(zmiana_przew[b])
        rap.append(dict(sonda=int(kk['sonda']), frame_id=kk['uklad']['frame_id'], sila_kotwicy=float(s_r[b]),
                        zmiana_przewidziana=dz, cel=cel, sily_wersji=s_w[ii].tolist(), przepisy=[uk_all[i]['przepis'] for i in ii],
                        wybrana=int(j - ii[0]), sila_wybranej=float(s_w[j]),
                        czesc_zmiany=float((s_w[j] - s_r[b]) / dz) if abs(dz) > 1e-9 else None,
                        rozrzut_wersji=float(s_w[ii].std())))
        rzad = [ref[b][0]] + [wyc[i][0] for i in ii]
        rzad = [cv2.copyMakeBorder(np.ascontiguousarray(x), 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                   value=(40, 220, 40) if (n > 0 and ii[n - 1] == j) else (0, 0, 0)) for n, x in enumerate(rzad)]
        kafle.append(np.concatenate(rzad, 1))
    cv2.imwrite(str(kd / 'WERSJE_KOTWIC.jpg'), cv2.cvtColor(np.concatenate(kafle, 0), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    return sciezki, rap


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
    out = Path(os.environ['MALY_VIT_WYNIKI']) / (f'proba_s{s}' if a.proba else f'petla_s{s}'); out.mkdir(parents=True, exist_ok=True)
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
    kotwice = {}
    for j in orig:
        i = int(poz_obecn[j]); r = pelny[opis[i]['frame_id']]
        kotwice[j] = dict(uklad=UK.uklad(r), obraz576=obraz_576(D.DATA_ROOT / r['path']), sonda=i)
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
    H_BLOK = max(1, KROKI_OBR // T.ZAPIS_CO // PR.B_BLOK)
    h_krotki = min(100, KROKI_OBR)

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
            # --- prognoza latentu tożsamości (wszystkie ramiona): równanie ruchu i, dla porównania, Koopman/Langevin
            S_all = np.concatenate(s_hist)
            F, n_paczek_f = PG.sila_zbioru(u, zb, sondy, obecn, wek)
            v0 = PG.predkosc(S_all)
            d_dl, d_kr = PG.zmiana(v0, F, KROKI_OBR), PG.zmiana(v0, F, h_krotki)
            b_dl, b_kr = PG.zmiana(v0, 0 * F, KROKI_OBR), PG.zmiana(v0, 0 * F, h_krotki)   # sama bezwładność
            pp = SI.prognozy(PR.bloki(S_all[:, :, None]), H_BLOK)
            teraz_blok = S_all[-PR.B_BLOK:].mean(0)
            info['prognoza'] = dict(sila_paczki_srednia=float(F.mean()), paczek_w_sredniej=n_paczek_f, predkosc_srednia=float(v0.mean()),
                                    zmiana_500_srednia=float(d_dl.mean()), zmiana_100_srednia=float(d_kr.mean()),
                                    model_kl=pp['model'], zmiana_kl_srednia=float((pp['prognozy'][pp['model']] - pp['teraz']).mean()))
            nowe, synt_zb = [], None
            # --- wybór klatek
            if ramie in ('sd_prognoza', 'sd_teraz'):
                kot = sorted([j for j in orig if kotwice[j]['uklad'] is not None], key=lambda j: -abs(d_dl[j]))[:KOTWIC]
                sciezki, rap = wersje_kotwic(G, u, [kotwice[j] for j in kot], [d_dl[j] for j in kot], s, k, rng, kd,
                                             cel_teraz=(ramie == 'sd_teraz'))
                listy = [WB.zapytaj(p, wykl, p.parent / f'{p.stem}_zapytanie.json') for p in sciezki]
                nowe = [dict(d, zrodlo='pula') for d in WB.koopman(listy, pula_id, zajete, NOWYCH)]
                info['wersje_sd'] = rap
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
                prog = -np.inf
                if rownanie:
                    real = sorted({f for g in grupy + los_gr if not g['syntetyczne'] for f in g['fids']})
                    zb_r = D.zbior_z_puli([pula_id[f] for f in real]); poz_r = {f: i for i, f in enumerate(real)}
                    poz_s = {w['frame_id']: i for i, w in enumerate(synt_zb.wiersze)} if synt_zb is not None else {}

                    def idx_grupy(g):
                        return (synt_zb, [poz_s[f] for f in g['fids']]) if g['syntetyczne'] else (zb_r, [poz_r[f] for f in g['fids']])
                    for g in grupy + los_gr:
                        z_, i_ = idx_grupy(g)
                        g.update(RW.ocen_grupe(u, z_, np.array(i_), gk))
                    prog = W4.prog_bezpiecznika(los_gr)
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
                nowe, rap = W4.wybierz(grupy, zajete, NOWYCH, MAX_SYNT, rng, wolne, prog,
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
            lat = []
            u.kroki(zb, KROKI_OBR, rng, sondy=sondy, latent_out=lat, co_ocena=CO_OCENA, ocena=ocena)
            arr = np.stack([l for _, l in lat])
            s_rot = SI.sila(arr[:, obecn, 0], wek)
            s_hist.append(s_rot)
            # --- trafność prognozy latentu tożsamości w tym obrocie
            i_kr = h_krotki // T.ZAPIS_CO - 1
            info['trafnosc'] = dict(
                h100=PG.ocena({'rownanie': S_all[-1] + d_kr, 'bezwladnosc': S_all[-1] + b_kr}, S_all[-1], s_rot[i_kr]),
                h500=PG.ocena({'rownanie': teraz_blok + d_dl, 'bezwladnosc': teraz_blok + b_dl,
                               'koopman_langevin': pp['prognozy'][pp['model']]},
                              teraz_blok, s_rot[-PR.B_BLOK:].mean(0)))
            if 'strata_kontrolna_start' in info:
                info['strata_kontrolna_koniec'] = kontrola.strata(u)
            info.update(dodane=len(nowe), syntetycznych=sum(d['zrodlo'] == 'sd' for d in nowe), zbior=len(zb), krok_po=u.krok,
                        ocena=oceny[-1], sila_tozsamosci_srednia=float(s_rot[-5:].mean()), sekundy=time.time() - tk,
                        wybrane=nowe)
            zj(kd / 'OBROT.json', info)
            zj(kat / 'OCENY.json', oceny)
            tr = info['trafnosc']
            txt = (f'  prognoza/brak zmiany: 100 kroków {tr["h100"]["rownanie"]["wzgl_braku_zmiany"]:.2f}, 500 kroków '
                   f'{tr["h500"]["rownanie"]["wzgl_braku_zmiany"]:.2f} (bezwładność {tr["h500"]["bezwladnosc"]["wzgl_braku_zmiany"]:.2f}, '
                   f'K/L {tr["h500"]["koopman_langevin"]["wzgl_braku_zmiany"]:.2f}), znak {tr["h500"]["rownanie"]["znak"]:.2f}')
            if info.get('wersje_sd'):
                cz = [r['czesc_zmiany'] for r in info['wersje_sd'] if r['czesc_zmiany'] is not None]
                txt += f'  SD: część zmiany {np.median(cz) if cz else float("nan"):.2f}'
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
