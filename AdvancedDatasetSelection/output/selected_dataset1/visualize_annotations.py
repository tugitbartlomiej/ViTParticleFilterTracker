#!/usr/bin/env python3
"""
Skrypt do wizualizacji bounding boxów z merged_20k_annotations.json
Rysuje bbox na każdym obrazie i zapisuje do katalogu visualizations_bbox/
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import defaultdict
import random


def load_annotations(json_path):
    """Wczytaj adnotacje COCO."""
    print(f"Wczytywanie adnotacji: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def draw_bbox(draw, bbox, color, label=None, width=3):
    """
    Rysuj bounding box w formacie COCO [x, y, width, height].
    """
    x, y, w, h = bbox
    # Prostokąt
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

    # Etykieta
    if label:
        # Tło dla tekstu
        text_bbox = draw.textbbox((x, y - 20), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, y - 20), label, fill='white')


def visualize_dataset(json_path, images_dir, output_dir, max_images=None, sample_random=False):
    """
    Wizualizacja datasetu z bounding boxami.

    Args:
        json_path: Ścieżka do pliku JSON z adnotacjami
        images_dir: Katalog z obrazami
        output_dir: Katalog wyjściowy dla wizualizacji
        max_images: Maksymalna liczba obrazów do przetworzenia (None = wszystkie)
        sample_random: Jeśli True, losowo wybierz obrazy
    """
    # Wczytaj dane
    coco_data = load_annotations(json_path)

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"Kategorie: {categories}")

    images = coco_data['images']
    annotations = coco_data['annotations']

    print(f"Liczba obrazów: {len(images)}")
    print(f"Liczba adnotacji: {len(annotations)}")

    # Mapa image_id -> [annotations]
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)

    # Statystyki
    images_with_anns = sum(1 for img in images if img['id'] in img_to_anns)
    print(f"Obrazy z adnotacjami: {images_with_anns}")

    # Utwórz katalog wyjściowy
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Wybierz obrazy do przetworzenia
    if max_images and max_images < len(images):
        if sample_random:
            selected_images = random.sample(images, max_images)
        else:
            selected_images = images[:max_images]
    else:
        selected_images = images

    print(f"\nPrzetwarzanie {len(selected_images)} obrazów...")

    # Kolory dla bounding boxów
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']

    # Statystyki
    stats = {
        'processed': 0,
        'with_bbox': 0,
        'without_bbox': 0,
        'errors': 0,
        'total_bboxes': 0
    }

    for img_info in tqdm(selected_images, desc="Wizualizacja"):
        image_id = img_info['id']
        filename = img_info['file_name']
        image_path = Path(images_dir) / filename

        if not image_path.exists():
            stats['errors'] += 1
            continue

        try:
            # Wczytaj obraz
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img)

            # Pobierz adnotacje dla tego obrazu
            anns = img_to_anns.get(image_id, [])

            if anns:
                stats['with_bbox'] += 1
                for i, ann in enumerate(anns):
                    bbox = ann['bbox']
                    category_id = ann['category_id']
                    category_name = categories.get(category_id, f'cat_{category_id}')

                    color = colors[i % len(colors)]
                    label = f"{category_name}"

                    draw_bbox(draw, bbox, color, label, width=3)
                    stats['total_bboxes'] += 1
            else:
                stats['without_bbox'] += 1

            # Zapisz obraz
            output_filename = f"viz_{filename}"
            output_file = output_path / output_filename
            img.save(output_file, quality=95)

            stats['processed'] += 1

        except Exception as e:
            print(f"\nBłąd przy przetwarzaniu {filename}: {e}")
            stats['errors'] += 1

    # Raport
    print("\n" + "=" * 50)
    print("RAPORT WIZUALIZACJI")
    print("=" * 50)
    print(f"Przetworzono obrazów: {stats['processed']}")
    print(f"  - z bounding boxami: {stats['with_bbox']}")
    print(f"  - bez bounding boxów: {stats['without_bbox']}")
    print(f"Łączna liczba bbox: {stats['total_bboxes']}")
    print(f"Błędy: {stats['errors']}")
    print(f"\nWizualizacje zapisane w: {output_path}")

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Wizualizacja bounding boxów z adnotacji COCO')
    parser.add_argument('--json', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\merged_20k_annotations.json',
                        help='Ścieżka do pliku JSON z adnotacjami')
    parser.add_argument('--images', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\images',
                        help='Katalog z obrazami')
    parser.add_argument('--output', type=str,
                        default=r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AdvancedDatasetSelection\output\selected_dataset\visualizations_bbox',
                        help='Katalog wyjściowy')
    parser.add_argument('--max', type=int, default=100,
                        help='Maksymalna liczba obrazów (domyślnie 100, 0 = wszystkie)')
    parser.add_argument('--random', action='store_true',
                        help='Losowo wybierz obrazy')

    args = parser.parse_args()

    max_images = args.max if args.max > 0 else None

    visualize_dataset(
        json_path=args.json,
        images_dir=args.images,
        output_dir=args.output,
        max_images=max_images,
        sample_random=args.random
    )
