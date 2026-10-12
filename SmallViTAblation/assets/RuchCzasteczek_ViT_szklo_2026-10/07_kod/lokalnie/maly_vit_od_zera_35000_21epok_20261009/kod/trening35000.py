"""Mały ViT od zera na 35 000 klatkach (wejscia/MANIFEST_35000.jsonl), 21 epok, wagi po każdej epoce i gęste kroki.

Przepis uczenia bez zmian względem biegu K1000 z gęstą historią (trening.Uczen(47), numpy default_rng(47), paczka 32,
AdamW 5e-4, rozgrzewka lr 300 kroków, potem stały lr, bf16, odbicie i jasność/kontrast). Zmienia się tylko zbiór:
35 000 klatek TRAIN Cataract-LMM Object Tracking zamiast bazy 1500. Klatki czytane prosto z archiwów zip na Edenie,
SHA-256 każdej klatki sprawdzane z manifestem. Każda klatka rozciągana do 480 × 320 (INTER_AREA), ramki skalowane
osobno w X i Y, tak jak w MANIFEST_1500 pomiaru (720×480, 720×576, 640×432, 540×540). Trzymane w RAM (uint8).

Epoka = ceil(35000 / 32) = 1094 kroki (35 008 próbek; Uczen.kroki losuje kolejne permutacje zbioru, więc paczka na
granicy epoki bierze resztę jednej permutacji i początek następnej). Koniec epoki e to krok 1094 · e, a pomiar naturalny
aplikacji z krokiem 1094 wybiera dokładnie końce epok.

Zapis (format pomiaru naturalnego aplikacji: k<krok>.pt = {'net', 'krok'}):
  wagi/k00000.pt ...   wagi sieci: krok 0, co 10 kroków do 1000, co 100 kroków dalej, koniec każdej epoki
  pelny/stan_e<EE>.pt  wagi + AdamW + stan losowania po każdej epoce (zostają 2 ostatnie, do wznowienia)
  dane/OCENY.json      mAP@0,5 na walidacji: krok 0, co 1000 kroków i koniec każdej epoki
  dane/LOG.json        strata co 50 kroków;  dane/metadata.json  opis biegu, sumy wejść, liczby
Pilnowanie kwoty: przed każdym zapisem sprawdzana jest wolna kwota projektu (lfs quota -p 1299). Poniżej
MIN_GESTE_GB gęste kroki są pomijane (zostają epoki), poniżej MIN_STOP_GB bieg zapisuje stan i kończy się.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

import dane as D
import trening as T

ZIARNO = 47
EPOKI = 21
MIN_GESTE_GB = 12.0
MIN_STOP_GB = 4.0


def zj(p, x):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, p)


def sha_pliku(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def wolne_gb(sciezka):
    """Wolna kwota projektu 1299 na /mnt/evafs (GB); gdy lfs niedostępne: wolne miejsce systemu plików."""
    try:
        w = subprocess.run(['lfs', 'quota', '-p', '1299', '/mnt/evafs'], capture_output=True, text=True, timeout=30).stdout
        for linia in w.splitlines():
            pola = linia.split()
            if pola and pola[0] == '/mnt/evafs' and len(pola) >= 3:
                uzyte, kwota = int(pola[1].rstrip('*')), int(pola[2])
                return (kwota - uzyte) / 1024 ** 2
    except Exception:
        pass
    return shutil.disk_usage(sciezka).free / 1e9


def wiersze_35000(manifest):
    out = []
    for linia in open(manifest, encoding='utf-8'):
        r = json.loads(linia)
        assert r['split'] == 'TRAIN' and r['gt'], r['id']
        sx, sy = D.W_ / r['width'], D.H_ / r['height']          # jak MANIFEST_1500: rozciągnięcie do 480 × 320
        b = np.array([[g['bbox_xywh'][0] * sx, g['bbox_xywh'][1] * sy, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * sx,
                       (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * sy] for g in r['gt']], np.float32)
        out.append(dict(frame_id=f'cataract_lmm/tracking/{r["film"]}/{int(r["frame"]):08d}', film=r['film'],
                        archiwum=Path(r['archive'].replace(chr(92), '/')).name, member=r['member'],
                        sha=r['image_sha256'], ramki=b.reshape(-1, 4), wh=(r['width'], r['height'])))
    return out


def wczytaj_z_zipow(wiersze, katalog_zip, watki=16):
    """Klatki z archiwów (każde archiwum w jednym wątku), SHA-256 bajtów zgodne z manifestem, 480 × 320 uint8."""
    X = np.empty((len(wiersze), D.H_, D.W_, 3), np.uint8)
    grupy = {}
    for i, r in enumerate(wiersze):
        grupy.setdefault(r['archiwum'], []).append(i)

    def archiwum(nazwa):
        with zipfile.ZipFile(Path(katalog_zip) / nazwa) as z:
            for i in grupy[nazwa]:
                raw = z.read(wiersze[i]['member'])
                if hashlib.sha256(raw).hexdigest() != wiersze[i]['sha']:
                    raise ValueError(f'SHA-256 klatki niezgodne: {wiersze[i]["frame_id"]}')
                im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if im is None or im.shape[1::-1] != tuple(wiersze[i]['wh']):
                    raise ValueError(f'zła klatka {wiersze[i]["frame_id"]}: {None if im is None else im.shape}')
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                X[i] = cv2.resize(im, (D.W_, D.H_), interpolation=cv2.INTER_AREA)
        return len(grupy[nazwa])

    t = time.time()
    with ThreadPoolExecutor(watki) as ex:
        n = sum(ex.map(archiwum, sorted(grupy)))
    print(f'klatki: {n} z {len(grupy)} archiwów w {time.time() - t:.0f} s', flush=True)
    return X


def plan_zapisu(kroki_lacznie, konce_epok, proba):
    gesty = set(range(0, min(1000, kroki_lacznie) + 1, 10)) | set(range(0, kroki_lacznie + 1, 100))
    if proba:
        gesty = set(range(0, kroki_lacznie + 1, 5))
    return gesty | set(konce_epok) | {0, kroki_lacznie}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--wyniki', required=True)
    a.add_argument('--zipy', required=True, help='katalog z archiwami TR_*.zip (3_Object_Tracking/annotations)')
    a.add_argument('--manifest', required=True)
    a.add_argument('--epoki', type=int, default=EPOKI)
    a.add_argument('--proba', action='store_true', help='512 klatek, 2 epoki, walidacja 200: sprawdzenie kodu')
    a.add_argument('--wznow', action='store_true', help='wznowienie od ostatniego pelny/stan_e*.pt')
    a.add_argument('--obserwuj', action='store_true', help='gęsty zapis ruchu sond i sił co krok (obserwator_gesty.py)')
    a.add_argument('--sondy', default='', help='JSONL z polem pool_row: które wiersze manifestu są sondami (100)')
    a.add_argument('--pelne', type=int, default=8, help='ile pierwszych sond ma pełny zapis 768D')
    a = a.parse_args()
    out = Path(a.wyniki)
    for d in ('wagi', 'pelny', 'dane'):
        (out / d).mkdir(parents=True, exist_ok=True)
    wolne = wolne_gb(out)
    print(f'wolna kwota: {wolne:.1f} GB', flush=True)
    if wolne < MIN_GESTE_GB + 6:
        raise SystemExit(f'za mało miejsca na start: {wolne:.1f} GB (potrzeba ≥ {MIN_GESTE_GB + 6:.0f} GB)')

    wiersze = wiersze_35000(a.manifest)
    if a.proba:
        wiersze = [wiersze[i] for i in np.random.default_rng(0).choice(len(wiersze), 512, replace=False)]
    wal_w, wal_X = D.walidacja()
    ids_tr = {r['frame_id'] for r in wiersze}
    wspolne = [i for i, r in enumerate(wal_w) if r['frame_id'] in ids_tr]
    if wspolne:                                      # walidacja tylko na klatkach spoza treningu
        zostaw = np.setdiff1d(np.arange(len(wal_w)), wspolne)
        wal_w, wal_X = [wal_w[i] for i in zostaw], wal_X[zostaw]
    if a.proba:
        wal_w, wal_X = wal_w[:200], wal_X[:200]
    print(f'walidacja: {len(wal_w)} klatek (usunięto {len(wspolne)} wspólnych z treningiem)', flush=True)
    X = wczytaj_z_zipow(wiersze, a.zipy)
    zb = D.Zbior(wiersze, X)

    epoki = 2 if a.proba else a.epoki
    na_epoke = math.ceil(len(zb) / T.PACZKA)          # 1094 kroki = 35 008 próbek: końce epok w równym odstępie
    konce = [e * na_epoke for e in range(1, epoki + 1)]
    lacznie = konce[-1]
    zapis = plan_zapisu(lacznie, konce, a.proba)
    co_ocena = 20 if a.proba else 1000
    print(f'{len(zb)} klatek, {epoki} epok, {lacznie} kroków; końce epok: {konce}; zapisów wag: {len(zapis)} '
          f'(~{len(zapis) * 11.7 / 1000:.1f} GB)', flush=True)

    u = T.Uczen(ZIARNO); rng = np.random.default_rng(ZIARNO)
    oceny, pominiete = [], []
    meta = dict(opis='Mały ViT od zera, 35 000 klatek TRAIN Cataract-LMM Object Tracking, 21 epok, zapis co epokę i '
                     'gęste kroki (README eksperymentu).',
                przepis='trening.Uczen(47) + numpy default_rng(47), paczka 32, AdamW 5e-4 wd 0.05, rozgrzewka 300, '
                        'stały lr, bf16, odbicie 0.5, jasność/kontrast ±20 %',
                klatki=len(zb), epoki=epoki, krokow=lacznie, konce_epok=konce, kroki_zapisu=sorted(zapis),
                walidacja=len(wal_w), walidacja_usuniete_wspolne=len(wspolne), proba=a.proba,
                manifest=str(a.manifest), manifest_sha256=sha_pliku(a.manifest),
                kod_sha256={p.name: sha_pliku(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__, start=time.strftime('%Y-%m-%d %H:%M:%S'))
    if a.wznow:
        stany = sorted((out / 'pelny').glob('stan_e[0-9][0-9].pt'))
        if not stany:
            raise SystemExit('brak stanu do wznowienia')
        st = torch.load(stany[-1], map_location='cuda', weights_only=False)
        stary_meta = out / 'dane' / 'metadata.json'
        if stary_meta.exists():                      # opis poprzedniego odcinka zostaje obok
            shutil.copy2(stary_meta, out / 'dane' / f'metadata_do_kroku_{st["krok"]:05d}.json')
        meta['wznowiono_od'] = dict(plik=stany[-1].name, krok=st['krok'])
        u.wczytaj(st); rng.bit_generator.state = st['rng']; torch.set_rng_state(st['torch_rng'].cpu())
        oceny = st.get('oceny', []); u.log = st.get('log', [])
        print(f'wznowienie od kroku {u.krok} ({stany[-1].name}); kolejka permutacji zaczyna się od nowa', flush=True)

    def ocen(uu, epoka=None):
        ap = T.ap50(uu.wykryj(wal_X), [r['ramki'] for r in wal_w])
        oceny.append(dict(krok=uu.krok, epoka=epoka, mAP50=float(ap)))
        zj(out / 'dane' / 'OCENY.json', oceny)
        print(f'krok {uu.krok}' + (f' (koniec epoki {epoka})' if epoka else '') + f': mAP@0,5 {ap:.4f}', flush=True)

    def zapisz_wagi(uu):
        p = out / 'wagi' / f'k{uu.krok:05d}.pt'
        if not p.exists():
            torch.save(dict(net={k: v.detach().cpu().clone() for k, v in uu.net.state_dict().items()}, krok=uu.krok,
                            epoka=uu.krok * T.PACZKA / len(zb)), p)

    stop = {'teraz': False}

    def po_kroku(uu):
        t = uu.krok
        epoka = konce.index(t) + 1 if t in konce else None
        if t in zapis:
            wolne_ = wolne_gb(out) if (t % 100 == 0 or epoka) else MIN_GESTE_GB + 1
            if wolne_ < MIN_STOP_GB:
                print(f'KWOTA: {wolne_:.1f} GB < {MIN_STOP_GB} GB - zapisuję stan i kończę', flush=True)
                stop['teraz'] = True
            elif epoka or wolne_ >= MIN_GESTE_GB:
                zapisz_wagi(uu)
            else:
                pominiete.append(t)
        if epoka:
            ocen(uu, epoka)
            st = dict(uu.stan(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), oceny=oceny, log=uu.log)
            torch.save(st, out / 'pelny' / f'stan_e{epoka:02d}.pt')
            for stary in sorted((out / 'pelny').glob('stan_e*.pt'))[:-2]:
                stary.unlink()
            zj(out / 'dane' / 'LOG.json', uu.log)
            print(f'epoka {epoka}/{epoki}: krok {t}, wolna kwota {wolne_gb(out):.1f} GB, '
                  f'{time.strftime("%H:%M:%S")}', flush=True)
        elif t % co_ocena == 0:
            ocen(uu)
        if stop['teraz']:
            raise KeyboardInterrupt

    obs = None
    if a.obserwuj:
        import obserwator_gesty as OG
        if a.wznow:
            raise SystemExit('obserwator gęsty wymaga biegu od kroku 0 (baza PCA32 z kroku 0)')
        if a.proba:
            sondy = list(range(16)); pelne = sondy[:4]
            obs = OG.Obserwator(out / 'gesty', zb, sondy, pelne, gesto_do=20, co_pozniej=2, pelne_do=10, pelne_co=4,
                                porcja=8, porcja_pelne=4, wolne_gb=lambda: wolne_gb(out))
        else:
            wybrane = [json.loads(l)['pool_row'] for l in open(a.sondy, encoding='utf-8') if l.strip()]
            pelne = wybrane[:a.pelne]
            obs = OG.Obserwator(out / 'gesty', zb, wybrane, pelne, wolne_gb=lambda: wolne_gb(out))
        obs.start(u)
        meta['obserwator'] = dict(sondy=len(obs.sondy), pelne=len(obs.pelne), gesto_do=obs.gesto_do, co_pozniej=obs.co_pozniej,
                                  pelne_do=obs.pelne_do, pelne_co=obs.pelne_co, sondy_plik=a.sondy)
    if u.krok == 0:
        zapisz_wagi(u); ocen(u)
    zj(out / 'dane' / 'metadata.json', meta)
    t0 = time.time()
    try:
        u.kroki(zb, lacznie - u.krok, rng, co_ocena=1, ocena=po_kroku, obserwator=obs)
    except KeyboardInterrupt:
        st = dict(u.stan(), rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), oceny=oceny, log=u.log)
        torch.save(st, out / 'pelny' / f'stan_e99_przerwane_k{u.krok}.pt')
    if obs is not None:
        meta['obserwator'].update(obs.zakoncz())
    meta.update(koniec=time.strftime('%Y-%m-%d %H:%M:%S'), czas_s=round(time.time() - t0), ostatni_krok=u.krok,
                pominiete_geste_kroki=pominiete, plikow_wag=len(list((out / 'wagi').glob('k*.pt'))),
                wolna_kwota_gb=round(wolne_gb(out), 1))
    zj(out / 'dane' / 'metadata.json', meta); zj(out / 'dane' / 'LOG.json', u.log)
    print(f'KONIEC: krok {u.krok}, {meta["plikow_wag"]} plików wag, {meta["czas_s"]} s', flush=True)


if __name__ == '__main__':
    main()
