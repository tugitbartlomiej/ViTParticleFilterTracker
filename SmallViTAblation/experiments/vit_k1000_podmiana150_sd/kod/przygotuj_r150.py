"""Wejścia ramienia kontrolnego R150 (PLAN.md, rozszerzenie 2026-10-08): 150 klatek wylosowanych jednostajnie z puli TRAIN
vit_dino_faiss_35000 (wszystkie 35 000 mają jedną ramkę cystotomu, tak jak klatki A150), z wykluczeniem bazy 1500
(po SHA256 obrazu i po filmie/klatce), 150 klatek A150, klatki zapytania DINO 0088/1661 (j = 17943) i zdjęcia celu
0055/4019 (j = 10389). Ziarno losowania 1500150. Nowa klatka k trafia na tę samą pozycję co w A150 (PODMIANA.json,
usuniete[k]), więc R150 różni się od A150 wyłącznie tym, jakie 150 obrazów stoi na podmienionych pozycjach.
Ramki skalowane osobno w x i y do 480 × 320, jak w przygotuj.py.
Zapis: outputs/vit_k1000_podmiana150_sd/wejscia_losowe/ (LOSOWE.json, losowe/*.jpg bez ponownej kompresji)
i archiwum WEJSCIA_LOSOWE.tar. Wejść A150 (wejscia/) nie zmienia.
Uruchomienie z SmallViTAblation: F:/Python/3.13/python.exe experiments/vit_k1000_podmiana150_sd/kod/przygotuj_r150.py
"""
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
W = ROOT / 'outputs' / 'vit_k1000_podmiana150_sd'
OUT = W / 'wejscia_losowe'
DINO = ROOT / 'outputs' / 'vit_dino_faiss_35000'
BAZA = ROOT / 'outputs' / 'vit_k1000_1500_pelne_latenty_90_20261008' / 'provenance' / 'MANIFEST_1500.jsonl'
J_ZAP, CEL_J, N, ZIARNO = 17943, 10389, 150, 1500150


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    MAN = [json.loads(l) for l in (DINO / 'manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    B = [json.loads(l) for l in BAZA.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(B) == 1500
    nowe_a = json.loads((W / 'wejscia' / 'NOWE.json').read_text(encoding='utf-8'))
    pod = json.loads((W / 'wejscia' / 'PODMIANA.json').read_text(encoding='utf-8'))
    usuniete = [x['pozycja'] for x in pod['usuniete']]
    assert len(usuniete) == N and [n['pozycja_w_bazie'] for n in nowe_a] == usuniete
    b_sha = {b['image_sha256'] for b in B}
    b_id = {(b['frame_id'].split('/')[2], int(b['frame_id'].split('/')[3])) for b in B}
    wykl = {n['j'] for n in nowe_a} | {J_ZAP, CEL_J}
    a_sha = {n['image_sha256'] for n in nowe_a}
    dozw = [j for j, m in enumerate(MAN)
            if j not in wykl and m['image_sha256'] not in b_sha and m['image_sha256'] not in a_sha
            and (m['film'], int(m['frame'])) not in b_id and m['split'] == 'TRAIN'
            and sum(g['category'] == 'cystotome' for g in m['gt']) == 1]
    losowe_j = sorted(np.random.default_rng(ZIARNO).choice(dozw, N, replace=False).tolist())

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'losowe').mkdir(parents=True)
    wiersze = []
    for k, j in enumerate(losowe_j):
        m = MAN[j]
        with zipfile.ZipFile(m['archive']) as z:
            dane = z.read(m['member'])
        assert sha(dane) == m['image_sha256'], f'SHA obrazu j={j}'
        plik = f'{m["film"]}_{int(m["frame"]):08d}.jpg'
        (OUT / 'losowe' / plik).write_bytes(dane)
        w, h = m['width'], m['height']
        r = [[g['bbox_xywh'][0] * 480 / w, g['bbox_xywh'][1] * 320 / h, (g['bbox_xywh'][0] + g['bbox_xywh'][2]) * 480 / w,
              (g['bbox_xywh'][1] + g['bbox_xywh'][3]) * 320 / h] for g in m['gt'] if g['category'] == 'cystotome']
        wiersze.append(dict(j=j, frame_id=f'cataract_lmm/tracking/{m["film"]}/{int(m["frame"]):08d}', film=m['film'],
                            klatka=int(m['frame']), plik=f'losowe/{plik}', ramki_480=r, zrodlo_wh=[w, h], image_sha256=m['image_sha256'],
                            pozycja_w_bazie=usuniete[k], zastepuje=B[usuniete[k]]['frame_id']))
    (OUT / 'LOSOWE.json').write_text(json.dumps(wiersze, indent=1, ensure_ascii=False), encoding='utf-8')
    (OUT / 'LOSOWANIE.json').write_text(json.dumps(dict(
        ziarno=ZIARNO, pula=len(MAN), dozwolonych=len(dozw), wykluczone='baza 1500 (SHA i film/klatka), 150 klatek A150, j 17943, j 10389',
        baza_manifest_sha256=sha(BAZA.read_bytes()), nowe_a150_sha256=sha((W / 'wejscia' / 'NOWE.json').read_bytes()),
        podmiana_sha256=sha((W / 'wejscia' / 'PODMIANA.json').read_bytes())), indent=1, ensure_ascii=False), encoding='utf-8')
    filmy = {}
    for x in wiersze:
        filmy[x['film']] = filmy.get(x['film'], 0) + 1
    with tarfile.open(W / 'WEJSCIA_LOSOWE.tar', 'w') as t:
        t.add(OUT, arcname='wejscia_losowe')
    print('dozwolonych', len(dozw), 'wylosowane', len(wiersze), 'filmów', len(filmy), 'najwięcej z jednego', max(filmy.values()))
    print('archiwum', W / 'WEJSCIA_LOSOWE.tar', (W / 'WEJSCIA_LOSOWE.tar').stat().st_size, sha((W / 'WEJSCIA_LOSOWE.tar').read_bytes()))


if __name__ == '__main__':
    main()
