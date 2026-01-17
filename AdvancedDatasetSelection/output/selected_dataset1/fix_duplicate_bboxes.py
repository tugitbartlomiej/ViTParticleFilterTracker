#!/usr/bin/env python3
"""
Skrypt do naprawy duplikatów bounding boxów w merged_20k_annotations.json

Problem: Niektóre obrazy (szczególnie augmentowane) mają więcej niż 1 bbox,
gdzie drugi bbox to często mały fragment przy krawędzi obrazu powstały
podczas augmentacji geometrycznej.

Rozwiązanie: Dla każdego obrazu zachować tylko bbox z największą powierzchnią (area).
"""

import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime


def fix_duplicate_bboxes(input_json, output_json):
    """
    Napraw duplikaty bbox - zachowaj tylko największy dla każdego obrazu.
    """
    print("=" * 60)
    print("NAPRAWA DUPLIKATÓW BOUNDING BOXÓW")
    print("=" * 60)

    # Wczytaj dane
    print(f"\n[1/4] Wczytywanie: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    print(f"      Obrazów: {len(images)}")
    print(f"      Adnotacji: {len(annotations)}")

    # Grupuj adnotacje po image_id
    print(f"\n[2/4] Analiza duplikatów...")
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)

    # Statystyki przed
    single_bbox = sum(1 for anns in img_to_anns.values() if len(anns) == 1)
    multi_bbox = sum(1 for anns in img_to_anns.values() if len(anns) > 1)
    total_extra = sum(len(anns) - 1 for anns in img_to_anns.values() if len(anns) > 1)

    print(f"      Obrazy z 1 bbox: {single_bbox}")
    print(f"      Obrazy z >1 bbox: {multi_bbox}")
    print(f"      Dodatkowych bbox do usunięcia: {total_extra}")

    # Napraw - zachowaj tylko największy bbox dla każdego obrazu
    print(f"\n[3/4] Naprawianie - zachowuję tylko największy bbox...")

    fixed_annotations = []
    new_ann_id = 1

    removed_examples = []

    for img_id, anns in sorted(img_to_anns.items()):
        if len(anns) == 1:
            # Tylko 1 bbox - zachowaj bez zmian (ale z nowym ID)
            ann = anns[0].copy()
            ann['id'] = new_ann_id
            fixed_annotations.append(ann)
            new_ann_id += 1
        else:
            # Wiele bbox - zachowaj tylko największy
            # Sortuj po area malejąco
            sorted_anns = sorted(anns, key=lambda x: x.get('area', 0), reverse=True)
            largest = sorted_anns[0].copy()
            largest['id'] = new_ann_id
            fixed_annotations.append(largest)
            new_ann_id += 1

            # Zapisz przykład usuniętych
            if len(removed_examples) < 5:
                # Znajdź nazwę pliku
                img_info = next((img for img in images if img['id'] == img_id), None)
                filename = img_info['file_name'] if img_info else f"img_{img_id}"
                removed_examples.append({
                    'filename': filename,
                    'kept': largest['bbox'],
                    'kept_area': largest.get('area', 0),
                    'removed': [a['bbox'] for a in sorted_anns[1:]],
                    'removed_areas': [a.get('area', 0) for a in sorted_anns[1:]]
                })

    # Pokaż przykłady
    print(f"\n      Przykłady naprawionych obrazów:")
    for ex in removed_examples:
        print(f"        {ex['filename']}:")
        print(f"          Zachowano: area={ex['kept_area']:.1f}")
        print(f"          Usunięto: areas={[f'{a:.1f}' for a in ex['removed_areas']]}")

    # Zapisz naprawiony plik
    print(f"\n[4/4] Zapisywanie: {output_json}")

    fixed_data = {
        'info': {
            'description': 'Merged COCO dataset for 20k selected images (fixed duplicates)',
            'url': '',
            'version': '1.1',
            'year': 2025,
            'contributor': 'AdvancedDatasetSelection',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'notes': 'Duplicate bboxes removed - kept only largest by area'
        },
        'licenses': data.get('licenses', [{'id': 1, 'name': 'Research Use', 'url': ''}]),
        'categories': categories,
        'images': images,
        'annotations': fixed_annotations
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2)

    # Raport końcowy
    print(f"\n{'=' * 60}")
    print("RAPORT")
    print("=" * 60)
    print(f"Przed: {len(annotations)} adnotacji")
    print(f"Po:    {len(fixed_annotations)} adnotacji")
    print(f"Usunięto: {len(annotations) - len(fixed_annotations)} duplikatów")
    print(f"\nKażdy obraz ma teraz dokładnie 1 bounding box.")

    return fixed_data


if __name__ == '__main__':
    base_dir = Path(r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset')

    input_json = base_dir / 'merged_20k_annotations.json'
    output_json = base_dir / 'merged_20k_annotations_fixed.json'

    fix_duplicate_bboxes(str(input_json), str(output_json))
