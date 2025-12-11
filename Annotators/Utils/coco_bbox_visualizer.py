#!/usr/bin/env python3
"""
Renderuje obrazy z oznaczeniami COCO i zapisuje je do katalogu `adnotacje`.

Przykład użycia:

    python Annotators/Utils/coco_bbox_visualizer.py \\
        --dataset /mnt/e/cataract_surgery_Instruments_detection.v1i.coco \\
        --subsets train valid test
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET = Path("/mnt/e/cataract_surgery_Instruments_detection.v1i.coco")
DEFAULT_SUBSETS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rysuje bounding boxy z anotacji COCO i zapisuje podglądy "
            "w katalogu `adnotacje`."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Ścieżka do katalogu z podfolderami train/valid/test (domyślnie %(default)s).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        choices=DEFAULT_SUBSETS,
        help="Które subsety przetworzyć (domyślnie wszystkie).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Ścieżka do katalogu na obrazy z adnotacjami (domyślnie <dataset>/adnotacje).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksymalna liczba obrazów do zapisania na subset (dla szybkich testów).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pomiń renderowanie, jeśli plik już istnieje w katalogu wyjściowym.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=4,
        help="Grubość ramki bounding boxu w pikselach.",
    )
    return parser.parse_args()


def load_coco_annotations(path: Path) -> Tuple[Dict[int, dict], Dict[int, List[dict]], Dict[int, str]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    images = {img["id"]: img for img in data.get("images", [])}
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        grouped[ann["image_id"]].append(ann)
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    return images, grouped, categories


def build_palette(category_ids: Iterable[int]) -> Dict[int, Tuple[int, int, int]]:
    palette = [
        (255, 99, 71),    # tomato
        (30, 144, 255),   # dodger blue
        (60, 179, 113),   # medium sea green
        (255, 215, 0),    # gold
        (147, 112, 219),  # medium purple
        (255, 140, 0),    # dark orange
        (95, 158, 160),   # cadet blue
        (205, 92, 92),    # indian red
        (70, 130, 180),   # steel blue
        (46, 139, 87),    # sea green
    ]
    colors = {}
    for idx, category_id in enumerate(sorted(category_ids)):
        colors[category_id] = palette[idx % len(palette)]
    return colors


def render_image_with_boxes(
    image_path: Path,
    annotations: Sequence[dict],
    categories: Dict[int, str],
    colors: Dict[int, Tuple[int, int, int]],
    line_width: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for ann in annotations:
        bbox = ann["bbox"]
        x1, y1 = bbox[0], bbox[1]
        x2 = x1 + bbox[2]
        y2 = y1 + bbox[3]

        category_id = ann["category_id"]
        label = categories.get(category_id, f"id:{category_id}")
        color = colors.get(category_id, (255, 255, 255))

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        text = label
        text_padding = 2
        text_x = x1
        text_y = max(y1 - font.size - 2 * text_padding, 0)
        text_color = (0, 0, 0)
        background_color = tuple(int(c * 0.75) for c in color)

        bbox_coords = draw.textbbox((text_x, text_y), text, font=font)
        bg_coords = (
            bbox_coords[0] - text_padding,
            bbox_coords[1] - text_padding,
            bbox_coords[2] + text_padding,
            bbox_coords[3] + text_padding,
        )
        draw.rectangle(bg_coords, fill=background_color)
        draw.text((text_x, text_y), text, fill=text_color, font=font)

    return image


def process_subset(
    dataset_dir: Path,
    subset: str,
    output_root: Path,
    limit: int | None,
    skip_existing: bool,
    line_width: int,
) -> int:
    subset_dir = dataset_dir / subset
    annotations_path = subset_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        logging.warning("Brak pliku %s – pomijam subset %s", annotations_path, subset)
        return 0

    images, grouped, categories = load_coco_annotations(annotations_path)
    if not grouped:
        logging.warning("Brak adnotacji w %s", annotations_path)
        return 0

    colors = build_palette(categories.keys())
    font = ImageFont.load_default()

    output_dir = output_root / subset
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for image_id, annotations in grouped.items():
        image_info = images.get(image_id)
        if not image_info:
            logging.debug("Brak wpisu image_id=%s w sekcji images", image_id)
            continue

        image_path = subset_dir / image_info["file_name"]
        if not image_path.exists():
            logging.warning("Brak pliku obrazu: %s", image_path)
            continue

        output_path = output_dir / image_path.name
        if skip_existing and output_path.exists():
            continue

        annotated = render_image_with_boxes(
            image_path,
            annotations,
            categories,
            colors,
            line_width=line_width,
            font=font,
        )
        annotated.save(output_path)
        processed += 1

        if limit and processed >= limit:
            break

    return processed


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_dir = args.dataset.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono katalogu datasetu: {dataset_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else dataset_dir / "adnotacje"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for subset in args.subsets:
        count = process_subset(
            dataset_dir,
            subset,
            output_dir,
            limit=args.limit,
            skip_existing=args.skip_existing,
            line_width=args.line_width,
        )
        logging.info("Subset %s: zapisano %s obrazów do %s", subset, count, output_dir / subset)
        total += count

    logging.info("Łącznie zapisano %s obrazów z bounding boxami.", total)


if __name__ == "__main__":
    main()
