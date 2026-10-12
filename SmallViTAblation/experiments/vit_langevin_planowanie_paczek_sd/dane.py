"""Wczytanie dokładnych czterech paczek z prawdziwymi GT i audytem pochodzenia.
Błędny hash, kolejność, split lub etykieta zatrzymuje planowanie przed uczeniem.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/vit_langevin_planowanie_paczek_sd'
PREV=ROOT/'outputs/vit_langevin_cztery_paczki_sd'
RETR=ROOT/'outputs/vit_sd_cztery_obrazy_paczki32'
SOURCE=ROOT/'outputs/vit_latent128_jedno_zdjecie_90'
SEEDS=[2026100511,2026100523,2026100537,2026100549]

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare():
    hashes=json.loads((PREV/'HASHES.json').read_text(encoding='utf-8'))
    cache=json.loads((PREV/'PREPARATION_CACHE.json').read_text(encoding='utf-8'))['hashes']
    images={};boxes={};audit={};seen=set()
    for seed in SEEDS:
        arm=str(seed);manifest=json.loads((RETR/arm/'TOP32.json').read_text(encoding='utf-8'))
        prior_manifest=cache[str((RETR/arm/'TOP32.json').resolve())]
        assert sha(RETR/arm/'TOP32.json')==prior_manifest
        for name in (f'IMAGES_{arm}.npy',f'GROUNDTRUTH_{arm}.json'):
            assert sha(PREV/name)==hashes[name]
            shutil.copy2(PREV/name,OUT/name)
        images[arm]=np.load(OUT/f'IMAGES_{arm}.npy')
        gt=json.loads((OUT/f'GROUNDTRUTH_{arm}.json').read_text(encoding='utf-8'))
        assert len(images[arm])==len(gt)==len(manifest['selected'])==32
        for row,original in zip(gt,manifest['selected']):
            assert row['split']=='train'
            assert (row['archive'],row['member'])==(original['archive'],original['member'])
            assert row['pixel_verified_sha256']==original['pixel_sha256']
            assert sha(Path(original['original_photo']))==original['original_sha256']
            assert row['annotation'] and 'filename' in row['annotation']
            assert original['pixel_sha256'] not in seen;seen.add(original['pixel_sha256'])
        boxes[arm]=[r['boxes_xyxy_480x320'] for r in gt]
        audit[arm]=dict(manifest=str(RETR/arm/'TOP32.json'),manifest_sha256=prior_manifest,
            groundtruth_source=str(PREV/f'GROUNDTRUTH_{arm}.json'),gt_sha256=sha(OUT/f'GROUNDTRUTH_{arm}.json'),
            images_sha256=sha(OUT/f'IMAGES_{arm}.npy'),positive=sum(bool(b) for b in boxes[arm]))
    assert len(seen)==128
    shutil.copy2(SOURCE/'ZDJECIE_ZRODLOWE.jpg',OUT/'SONDA_ORIGINAL.jpg')
    selection=json.loads((SOURCE/'SELECTION.json').read_text(encoding='utf-8'))
    assert selection['source_index']==29 and sha(OUT/'SONDA_ORIGINAL.jpg')==selection['record']['sha256']
    (OUT/'INPUT_PROVENANCE.json').write_text(json.dumps(dict(arms=audit,unique_images=128,
        retrieval_partial_count=88469,GT='real tool annotation.csv; exact prior audited labels; negatives retained',
        probe=selection,target_sha256=sha(SOURCE/'LATENT_PELNY.npz')),ensure_ascii=False,indent=2),encoding='utf-8')
    return images,boxes,selection
