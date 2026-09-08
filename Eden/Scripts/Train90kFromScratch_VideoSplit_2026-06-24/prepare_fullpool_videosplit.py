#!/usr/bin/env python3
"""
Buduje anotacje train/val/test dla PELNEGO poolu 91k (UNCURATED) pod ISTNIEJACYM
video-level splitem (video_split.json, seed 42, 35/8/7 wideo).

Roznica vs AdvancedDatasetSelection/ablation/detr/generate_ablation_datasets.py:
  - train = WSZYSTKIE klatki z wideo treningowych  (BEZ kuracji do 20k!)
  - val/test = WSZYSTKIE klatki z wideo val/test    (identycznie jak ablacje V1-V7)

To jest leakage-free odpowiednik UNCURATED baseline z artykulu (pelny pool 91k),
ale z poprawnym podzialem PER-WIDEO zamiast losowego frame-level. JEDEN plik splitu
(te 3 JSON-y) zasila ZAROWNO DETR (COCO natywnie) jak i YOLO (konwersja w yolo slurm).

Nie wymaga GPU ani obrazow — czysta manipulacja JSON. Uruchamiac na login node Edena.

Wejscie (Eden):
  --annotations  ~/datasets_20250606_test/datasets/DETR_augmented_dataset_20250218/augmented_coco_20250417_030014.json
  --split        ~/DETR/ablation/video_split.json   (mapa stem -> train/val/test)
Wyjscie:
  <output>/annotations_train.json   (~71k klatek, pelny pool z 35 wideo treningowych)
  <output>/annotations_val.json     (~10.78k, 8 wideo)
  <output>/annotations_test.json    (~9.24k, 7 wideo)
  <output>/metadata.json
  <output>/video_split.json         (kopia referencyjna)

Uzycie:
  python3 prepare_fullpool_videosplit.py \
    --annotations ~/datasets_20250606_test/datasets/DETR_augmented_dataset_20250218/augmented_coco_20250417_030014.json \
    --split ~/DETR/ablation/video_split.json \
    --output ~/DETR/fullpool_videosplit
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def parse_video_stem(filename: str) -> str:
    """test01_frame_000123.jpg -> test01 (kopie augmentowane dziela ten sam prefiks)."""
    if '_frame_' in filename:
        return filename.split('_frame_')[0]
    return filename.rsplit('.', 1)[0]


def build_subset(coco: dict, image_ids) -> dict:
    image_ids = set(image_ids)
    return {
        'info': coco.get('info', {}),
        'licenses': coco.get('licenses', []),
        'categories': coco.get('categories', []),
        'images': [im for im in coco['images'] if im['id'] in image_ids],
        'annotations': [an for an in coco['annotations'] if an['image_id'] in image_ids],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-pool 91k annotations under video-level split")
    ap.add_argument('--annotations', required=True, help='Pelny COCO (91k frames)')
    ap.add_argument('--split', required=True, help='video_split.json (klucz video_split: stem->split)')
    ap.add_argument('--output', required=True, help='Katalog wyjsciowy')
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO:  {args.annotations}")
    with open(args.annotations, encoding='utf-8') as f:
        coco = json.load(f)
    print(f"Loading split: {args.split}")
    with open(args.split, encoding='utf-8') as f:
        split = json.load(f)

    if 'video_split' not in split:
        raise SystemExit("ERROR: brak klucza 'video_split' (mapa stem->split) w pliku splitu")
    video_map = split['video_split']  # stem -> 'train'/'val'/'test'

    buckets = {'train': [], 'val': [], 'test': []}
    unknown = Counter()
    for im in coco['images']:
        stem = parse_video_stem(im['file_name'])
        sp = video_map.get(stem)
        if sp in buckets:
            buckets[sp].append(im['id'])
        else:
            unknown[stem] += 1

    print(f"\nFull COCO: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Split map: {len(video_map)} wideo")
    for sp in ('train', 'val', 'test'):
        print(f"  {sp}: {len(buckets[sp])} images")
    if unknown:
        print(f"  WARNING: {sum(unknown.values())} obrazow z {len(unknown)} wideo NIE w split map "
              f"(np. {list(unknown)[:5]}) -> POMINIETE. Sprawdz, czy nazewnictwo sie zgadza.")

    # --- weryfikacja braku przeciekow: zaden stem w dwoch splitach ---
    # (z konstrukcji niemozliwe — kazdy stem ma jedna wartosc w video_map — ale logujemy)
    vids_per_split = {'train': set(), 'val': set(), 'test': set()}
    for stem, sp in video_map.items():
        if sp in vids_per_split:
            vids_per_split[sp].add(stem)
    overlap = (vids_per_split['train'] & vids_per_split['val']) | \
              (vids_per_split['train'] & vids_per_split['test']) | \
              (vids_per_split['val'] & vids_per_split['test'])
    print(f"\nWideo per split: train={len(vids_per_split['train'])} "
          f"val={len(vids_per_split['val'])} test={len(vids_per_split['test'])}")
    print(f"Overlap wideo miedzy splitami: {len(overlap)} (musi byc 0)")
    if overlap:
        raise SystemExit(f"ERROR: PRZECIEK — wideo w >1 splicie: {sorted(overlap)[:10]}")

    for sp in ('train', 'val', 'test'):
        sub = build_subset(coco, buckets[sp])
        p = out / f'annotations_{sp}.json'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(sub, f)
        print(f"  wrote {p.name}: {len(sub['images'])} images, {len(sub['annotations'])} ann")

    meta = {
        'experiment': 'fullpool_91k_videosplit_from_scratch',
        'note': 'UNCURATED full pool; train = ALL train-video frames (NO 20k curation)',
        'seed': split.get('seed', 42),
        'video_counts': split.get('counts', {}).get('videos', {}),
        'image_counts': {sp: len(buckets[sp]) for sp in ('train', 'val', 'test')},
        'source_annotations': str(args.annotations),
        'source_split': str(args.split),
    }
    with open(out / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    try:
        shutil.copy2(args.split, out / 'video_split.json')
    except OSError:
        pass

    print(f"\nDone. Output: {out}")
    print("Oczekiwane (zweryfikowane na Edenie 2026-06-24): train=71316, val=10780, test=9240.")
    print("(val/test identyczne jak ablacja v1; train = pelny uncurated zbior z wideo treningowych.)")
    print("Jesli liczby inne — sprawdz mape video_split i nazewnictwo plikow.")


if __name__ == '__main__':
    main()
