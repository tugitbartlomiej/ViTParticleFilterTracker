#!/usr/bin/env python3
"""
Skrypt do mergowania adnotacji COCO z dwóch źródeł dla datasetu 20k.

Łączy adnotacje z:
1. _annotations.coco.json (Roboflow - 248 obrazów, category_id=1 'tooltip')
2. augmented_coco_20250417_030014.json (Augmented - 91k obrazów, category_id=0 'tool')

Filtruje tylko obrazy faktycznie istniejące w katalogu images/.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_json(filepath):
    """Wczytaj plik JSON."""
    print(f"Wczytywanie: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_filename_to_annotations_map(coco_data, source_name):
    """
    Buduje mapę: filename -> (image_info, [annotations])
    """
    filename_map = {}

    # Mapa image_id -> image_info
    id_to_image = {img['id']: img for img in coco_data.get('images', [])}

    # Mapa image_id -> [annotations]
    id_to_anns = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        id_to_anns[ann['image_id']].append(ann)

    # Buduj mapę po nazwie pliku
    for img_id, img_info in id_to_image.items():
        filename = img_info['file_name']
        annotations = id_to_anns.get(img_id, [])
        filename_map[filename] = {
            'image_info': img_info,
            'annotations': annotations,
            'source': source_name
        }

    return filename_map


def merge_annotations(images_dir, annotation_files, output_path):
    """
    Główna funkcja mergująca adnotacje.

    Args:
        images_dir: Katalog z obrazami (źródło prawdy)
        annotation_files: Lista krotek (filepath, source_name)
        output_path: Ścieżka do wyjściowego pliku JSON
    """
    print("=" * 60)
    print("MERGE ANNOTATIONS FOR 20k DATASET")
    print("=" * 60)

    # 1. Pobierz listę faktycznych plików w katalogu images
    print(f"\n[1/5] Skanowanie katalogu: {images_dir}")
    actual_files = set(os.listdir(images_dir))
    print(f"      Znaleziono {len(actual_files)} plików")

    # 2. Wczytaj i zmapuj adnotacje z wszystkich źródeł
    print(f"\n[2/5] Wczytywanie adnotacji źródłowych...")
    all_filename_maps = {}

    for filepath, source_name in annotation_files:
        coco_data = load_json(filepath)
        filename_map = build_filename_to_annotations_map(coco_data, source_name)
        print(f"      {source_name}: {len(filename_map)} obrazów z adnotacjami")

        # Merge do głównej mapy (późniejsze źródła nadpisują wcześniejsze)
        all_filename_maps.update(filename_map)

    print(f"      Łącznie unikalne nazwy: {len(all_filename_maps)}")

    # 3. Dopasuj adnotacje do faktycznych plików
    print(f"\n[3/5] Dopasowywanie adnotacji do obrazów...")

    matched_images = []
    matched_annotations = []
    missing_annotations = []

    new_image_id = 1
    new_ann_id = 1

    # Zunifikowana kategoria
    unified_category = {
        'id': 1,
        'name': 'tooltip',
        'supercategory': 'surgical-instrument'
    }

    stats = {
        'roboflow': 0,
        'augmented': 0,
        'no_annotation': 0,
        'total_annotations': 0
    }

    for filename in sorted(actual_files):
        if filename in all_filename_maps:
            data = all_filename_maps[filename]
            img_info = data['image_info']
            annotations = data['annotations']
            source = data['source']

            # Nowy wpis obrazu z nowym ID
            new_image = {
                'id': new_image_id,
                'file_name': filename,
                'width': img_info.get('width', 1920),
                'height': img_info.get('height', 1080),
                'license': 1
            }
            matched_images.append(new_image)

            # Nowe adnotacje z przeliczonymi ID
            for ann in annotations:
                new_ann = {
                    'id': new_ann_id,
                    'image_id': new_image_id,
                    'category_id': 1,  # Zunifikowane do tooltip
                    'bbox': ann['bbox'],
                    'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    'iscrowd': ann.get('iscrowd', 0)
                }
                matched_annotations.append(new_ann)
                new_ann_id += 1
                stats['total_annotations'] += 1

            # Statystyki
            if source == 'roboflow':
                stats['roboflow'] += 1
            else:
                stats['augmented'] += 1

            new_image_id += 1
        else:
            # Obraz bez adnotacji - dodaj bez bounding boxów
            missing_annotations.append(filename)
            stats['no_annotation'] += 1

            # Możemy dodać obraz bez adnotacji (opcjonalnie)
            # Na razie pomijamy - DETR potrzebuje adnotacji

    # 4. Raport
    print(f"\n[4/5] RAPORT MERGOWANIA:")
    print(f"      Obrazy z Roboflow: {stats['roboflow']}")
    print(f"      Obrazy z Augmented: {stats['augmented']}")
    print(f"      Obrazy bez adnotacji: {stats['no_annotation']}")
    print(f"      RAZEM obrazów z adnotacjami: {len(matched_images)}")
    print(f"      RAZEM adnotacji: {stats['total_annotations']}")

    if missing_annotations:
        print(f"\n      UWAGA: {len(missing_annotations)} obrazów bez adnotacji!")
        print(f"      Przykłady: {missing_annotations[:5]}")

    # 5. Zapisz wynikowy plik COCO
    print(f"\n[5/5] Zapisywanie do: {output_path}")

    output_coco = {
        'info': {
            'description': 'Merged COCO dataset for 20k selected images',
            'url': '',
            'version': '1.0',
            'year': 2025,
            'contributor': 'AdvancedDatasetSelection',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'licenses': [
            {'id': 1, 'name': 'Research Use', 'url': ''}
        ],
        'categories': [unified_category],
        'images': matched_images,
        'annotations': matched_annotations
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_coco, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SUKCES!")
    print(f"Zapisano {len(matched_images)} obrazów z {len(matched_annotations)} adnotacjami")
    print(f"{'=' * 60}")

    return output_coco, missing_annotations


if __name__ == '__main__':
    # Ścieżki
    base_dir = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset')

    images_dir = base_dir / 'images'

    annotation_files = [
        (base_dir / '_annotations.coco.json', 'roboflow'),
        (base_dir / 'augmented_coco_20250417_030014.json', 'augmented'),
    ]

    output_path = base_dir / 'merged_20k_annotations.json'

    # Uruchom merge
    result, missing = merge_annotations(
        images_dir=str(images_dir),
        annotation_files=[(str(p), n) for p, n in annotation_files],
        output_path=str(output_path)
    )

    # Zapisz listę brakujących adnotacji
    if missing:
        missing_path = base_dir / 'images_without_annotations.txt'
        with open(missing_path, 'w') as f:
            f.write('\n'.join(missing))
        print(f"\nLista obrazów bez adnotacji zapisana do: {missing_path}")
