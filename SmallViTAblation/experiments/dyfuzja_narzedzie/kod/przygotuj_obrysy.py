#!/usr/bin/env python3
"""Obrysy narzędzia do klatek, które już są na Edenie (prośba użytkownika 2026-09-29: wycinać narzędzie w trakcie
uczenia prosto z klatek filmów, bez lokalnego wycinania).

Na Edenie: `~/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci/context_train.tar` — 57 073 klatki (co piąta klatka
118 filmów treningowych, 720×480), bez adnotacji. Te same pliki JPEG (bajt w bajt, sprawdzone SHA256) są w lokalnych
archiwach Cataract-LMM razem z obrysem cystotomu każdej klatki (`cystotome_seg`). Ten skrypt zbiera obrysy do klatek
z listy Edenu (`dane/eden_context/context_pool_manifest.jsonl`) w jeden mały plik; wycinanie robi `trenuj.py`.

Wyjście `obrysy.npz`: sciezka (N,) ścieżka klatki w archiwum, video, klatka, ramka (N,4) prostokąt opisany na obrysie
[x0,y0,x1,y1] w pikselach natywnych, pkt (P,2) int16 wierzchołki wszystkich wielokątów, poly_od (Q+1,) początki
wielokątów w pkt, klatka_od (N+1,) początki wielokątów klatki. Tylko klatki z obrysem.
Opcja --obrazy KAT --limit-filmow K: wypakowuje też te klatki z lokalnych archiwów (do próby lokalnej).

    python przygotuj_obrysy.py [--out ../dane/obrysy] [--limit-filmow K --obrazy KAT]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIT = HERE.parents[3]
ARCHIWA = VIT / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
LISTA = HERE.parent / 'dane/eden_context/context_pool_manifest.jsonl'
TAR_EDEN = '/home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci/context_train.tar'
TAR_SHA = '8c838217cf3e40611f9400874d938d01798c535ceed908a45af826a69d00149a'   # sha256sum na Edenie 2026-09-29


def sha_pliku(p, blok=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(blok):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(HERE.parent / 'dane/obrysy'))
    ap.add_argument('--limit-filmow', type=int, default=0)
    ap.add_argument('--obrazy', default=None, help='wypakuj klatki do KAT/images (próba lokalna)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wiersze = [json.loads(l) for l in LISTA.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert all(r['split'] == 'train' for r in wiersze)
    wg_filmu = defaultdict(list)
    for r in wiersze:
        wg_filmu[r['video_id']].append(r)
    filmy = sorted(wg_filmu)[: a.limit_filmow or None]
    sc, vid, kl, ramki, pkt, poly_od, klatka_od = [], [], [], [], [], [0], [0]
    bez, sha_adn, brak_w_adn = 0, {}, 0
    npkt = 0
    for v in filmy:
        with zipfile.ZipFile(ARCHIWA / f'{v}.zip') as z:
            raw = z.read(f'{v}/annotation.csv')
            sha_adn[v] = hashlib.sha256(raw).hexdigest()
            adn = {int(r['frame_number']): r for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))}
            for r in sorted(wg_filmu[v], key=lambda r: r['frame_index']):
                n = int(r['frame_index'])
                x = adn.get(n)
                if x is None:
                    brak_w_adn += 1; continue
                if x['cystotome_seg'] in ('', '[]'):
                    bez += 1; continue
                polys = [np.rint(np.asarray(p, np.float64).reshape(-1, 2)).astype(np.int16) for p in json.loads(x['cystotome_seg'])]
                polys = [p for p in polys if len(p) >= 3]
                if not polys:
                    bez += 1; continue
                wsz = np.concatenate(polys)
                for p in polys:
                    pkt.append(p); npkt += len(p); poly_od.append(npkt)
                klatka_od.append(len(poly_od) - 1)
                ramki.append([wsz[:, 0].min(), wsz[:, 1].min(), wsz[:, 0].max() + 1, wsz[:, 1].max() + 1])
                sc.append(r['path']); vid.append(v); kl.append(n)
                if a.obrazy:
                    cel = Path(a.obrazy) / r['path']; cel.parent.mkdir(parents=True, exist_ok=True)
                    cel.write_bytes(z.read(f'{v}/{x["filename"]}'))
        print(f'  {v}: razem {len(sc)} klatek z obrysem', flush=True)
    np.savez_compressed(out / 'obrysy.npz', sciezka=np.array(sc), video=np.array(vid), klatka=np.array(kl, np.int32),
                        ramka=np.array(ramki, np.float32), pkt=np.concatenate(pkt).astype(np.int16),
                        poly_od=np.array(poly_od, np.int64), klatka_od=np.array(klatka_od, np.int64))
    man = dict(klatek_z_obrysem=len(sc), klatek_bez_obrysu=bez, klatek_bez_adnotacji=brak_w_adn, filmow=len(filmy),
               wielokatow=len(poly_od) - 1, wierzcholkow=npkt, lista_klatek=str(LISTA), lista_sha256=sha_pliku(LISTA),
               archiwum_klatek_eden=TAR_EDEN, archiwum_klatek_sha256=TAR_SHA,
               zgodnosc_bajtow='3 klatki sprawdzone: SHA256 pliku JPEG na Edenie = w lokalnym archiwum',
               adnotacje_sha256=sha_adn, obrysy_sha256=sha_pliku(out / 'obrysy.npz'))
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding='utf-8')
    (out / 'SHA256SUMS').write_text(f'{man["obrysy_sha256"]}  obrysy.npz\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: man[k] for k in ('klatek_z_obrysem', 'klatek_bez_obrysu', 'klatek_bez_adnotacji', 'filmow')}))
    print(f'obrysy.npz: {(out / "obrysy.npz").stat().st_size / 2 ** 20:.1f} MB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
