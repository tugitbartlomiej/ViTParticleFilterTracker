"""Wejścia biegu na Edenie (PLAN.md): 150 nowych klatek, 150 usuwanych pozycji bazy, 64 stałe sondy z bazy, zdjęcie celu.

Nowe: PODOBNE_150_0088_1661 (DINO top-150 do klatki 0088/1661) bez klatek obecnych w bazie 1500 (po SHA256 obrazu
i po filmie/klatce); brakujące uzupełnione kolejnymi z tego samego rankingu DINO (zapytanie j = 17943, bez bazy, bez
zapytania i bez celu j = 10389). Ramki z manifestu vit_dino_faiss_35000 (cystotom), skalowane osobno w x i y do 480 × 320.
Usuwane: default_rng(150).choice(1500, 150) z bazy, posortowane; nowa klatka k trafia na pozycję usuniete[k].
Sondy: default_rng(64).choice z pozostałych 1350 pozycji, 64 sztuki.
Zapis: outputs/vit_k1000_podmiana150_sd/wejscia/ i archiwum WEJSCIA_PODMIANA.tar (klatki JPEG bez ponownej kompresji).
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/vit_k1000_podmiana150_sd/kod/przygotuj.py
"""
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs' / 'vit_k1000_podmiana150_sd' / 'wejscia'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
PODOBNE = ROOT / 'outputs' / 'sd_img2img_sterowanie_latentem' / 'optymalizacja_regresja' / 'PODOBNE_150_0088_1661' / 'PODOBNE.json'
BAZA = ROOT / 'outputs' / 'vit_k1000_1500_pelne_latenty_90_20261008' / 'provenance' / 'MANIFEST_1500.jsonl'
ASSET = ROOT / 'assets' / 'MalyViT_K1000_SD_conditioning' / 'MODEL_VIT_K1000.pt'
J_ZAP, CEL_J, N = 17943, 10389, 150


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    B = [json.loads(l) for l in BAZA.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(B) == 1500 and [b['index'] for b in B] == list(range(1500))
    b_sha = {b['image_sha256'] for b in B}
    b_id = {(b['frame_id'].split('/')[2], int(b['frame_id'].split('/')[3])) for b in B}
    w_bazie = lambda j: MAN[j]['image_sha256'] in b_sha or (MAN[j]['film'], int(MAN[j]['frame'])) in b_id
    P = json.loads(PODOBNE.read_text(encoding='utf-8'))['wyniki']
    pominiete = [dict(poz=p['poz'], j=p['j'], film=p['film'], klatka=p['klatka']) for p in P if w_bazie(p['j'])]
    nowe_j = [p['j'] for p in P if not w_bazie(p['j'])]
    Fp = np.array(np.load(DINO / 'features.npy', mmap_mode='r'), np.float32); Fp /= np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12
    s = Fp @ Fp[J_ZAP]; s[J_ZAP] = -9; s[CEL_J] = -9
    ranking = [int(j) for j in np.argsort(-s)]
    assert ranking[:len(P)] == [p['j'] for p in P], 'ranking DINO niezgodny z PODOBNE.json'
    uzupelnione = []
    for j in ranking[len(P):]:
        if len(nowe_j) >= N:
            break
        if not w_bazie(j) and j not in nowe_j:
            nowe_j.append(j); uzupelnione.append(dict(j=j, film=MAN[j]['film'], klatka=int(MAN[j]['frame']), cos_dino=float(s[j]),
                                                     poz_w_rankingu=ranking.index(j) + 1))
    assert len(nowe_j) == N and len(set(nowe_j)) == N

    usuniete = sorted(np.random.default_rng(150).choice(1500, N, replace=False).tolist())
    reszta = sorted(set(range(1500)) - set(usuniete))
    sondy = sorted(np.random.default_rng(64).choice(reszta, 64, replace=False).tolist())

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'nowe').mkdir(parents=True); (OUT / 'cel').mkdir()

    def wiersz(j, kat):
        m = MAN[j]
        with zipfile.ZipFile(m['archive']) as z:
            dane = z.read(m['member'])
        assert sha(dane) == m['image_sha256'], f'SHA obrazu j={j}'
        plik = f'{m["film"]}_{int(m["frame"]):08d}.jpg'
        (OUT / kat / plik).write_bytes(dane)
        w, h = m['width'], m['height']
        r = [[g['bbox_xywh'][0] * 480 / w, g['bbox_xywh'][1] * 320 / h, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * 480 / w,
              (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * 320 / h] for g in m['gt'] if g['category'] == 'cystotome']
        return dict(j=j, frame_id=f'cataract_lmm/tracking/{m["film"]}/{int(m["frame"]):08d}', film=m['film'], klatka=int(m['frame']),
                    plik=f'{kat}/{plik}', ramki_480=r, zrodlo_wh=[w, h], image_sha256=m['image_sha256'], cos_dino=float(s[j]))

    nowe = [dict(wiersz(j, 'nowe'), pozycja_w_bazie=usuniete[k], zastepuje=B[usuniete[k]]['frame_id']) for k, j in enumerate(nowe_j)]
    cel = wiersz(CEL_J, 'cel')
    opis = dict(zapytanie_dino=dict(j=J_ZAP, film=MAN[J_ZAP]['film'], klatka=int(MAN[J_ZAP]['frame'])),
                pominiete_bo_w_bazie=pominiete, uzupelnione_z_rankingu=uzupelnione,
                usuniete=[dict(pozycja=i, frame_id=B[i]['frame_id'], image_sha256=B[i]['image_sha256']) for i in usuniete],
                sondy=[dict(pozycja=i, frame_id=B[i]['frame_id']) for i in sondy],
                ziarna=dict(usuniete=150, sondy=64), baza_manifest=str(BAZA), baza_manifest_sha256=sha(BAZA.read_bytes()))
    (OUT / 'NOWE.json').write_text(json.dumps(nowe, indent=1, ensure_ascii=False), encoding='utf-8')
    (OUT / 'CEL.json').write_text(json.dumps(cel, indent=1, ensure_ascii=False), encoding='utf-8')
    (OUT / 'PODMIANA.json').write_text(json.dumps(opis, indent=1, ensure_ascii=False), encoding='utf-8')
    shutil.copy2(ASSET, OUT / 'MODEL_VIT_K1000.pt')
    filmy = {}
    for n in nowe:
        filmy[n['film']] = filmy.get(n['film'], 0) + 1
    with tarfile.open(OUT.parent / 'WEJSCIA_PODMIANA.tar', 'w') as t:
        t.add(OUT, arcname='wejscia_podmiana')
    print('nowe', len(nowe), 'pominięte', pominiete, 'uzupełnione', uzupelnione, 'filmy', filmy)
    print('usunięte z filmów', sorted({B[i]['frame_id'].split('/')[2] for i in usuniete}).__len__(), 'sondy', len(sondy))
    print('archiwum', OUT.parent / 'WEJSCIA_PODMIANA.tar', (OUT.parent / 'WEJSCIA_PODMIANA.tar').stat().st_size,
          sha((OUT.parent / 'WEJSCIA_PODMIANA.tar').read_bytes()))


if __name__ == '__main__':
    main()
