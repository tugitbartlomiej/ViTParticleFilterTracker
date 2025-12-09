"""
COCO format handler for dataset I/O operations.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COCOHandler:
    """Handles COCO format dataset operations."""

    def __init__(self):
        self.annotations = None
        self.images = {}
        self.categories = []

    def load_annotations(self, annotation_path: str) -> Dict:
        """Load COCO annotations from JSON file."""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # Build image lookup
        self.images = {img['id']: img for img in self.annotations.get('images', [])}
        self.categories = self.annotations.get('categories', [])

        logger.info(f"Loaded {len(self.images)} images, {len(self.annotations.get('annotations', []))} annotations")
        return self.annotations

    def get_image_paths(self, images_dir: str) -> List[Tuple[int, str]]:
        """Get list of (image_id, image_path) tuples."""
        paths = []
        for img_id, img_info in self.images.items():
            img_path = os.path.join(images_dir, img_info['file_name'])
            if os.path.exists(img_path):
                paths.append((img_id, img_path))
            else:
                logger.warning(f"Image not found: {img_path}")
        return paths

    def get_annotations_for_image(self, image_id: int) -> List[Dict]:
        """Get all annotations for a specific image."""
        if self.annotations is None:
            return []
        return [
            ann for ann in self.annotations.get('annotations', [])
            if ann['image_id'] == image_id
        ]

    def filter_by_image_ids(self, image_ids: List[int]) -> Dict:
        """Create new COCO dict with only selected images."""
        if self.annotations is None:
            raise ValueError("No annotations loaded")

        selected_images = [
            img for img in self.annotations['images']
            if img['id'] in image_ids
        ]

        selected_annotations = [
            ann for ann in self.annotations.get('annotations', [])
            if ann['image_id'] in image_ids
        ]

        return {
            'info': self.annotations.get('info', self._create_info()),
            'licenses': self.annotations.get('licenses', []),
            'categories': self.categories,
            'images': selected_images,
            'annotations': selected_annotations
        }

    def merge_datasets(self, datasets: List[Dict]) -> Dict:
        """Merge multiple COCO datasets into one."""
        merged = {
            'info': self._create_info(),
            'licenses': [],
            'categories': [],
            'images': [],
            'annotations': []
        }

        # Collect all unique categories
        category_map = {}  # old_id -> new_id
        all_categories = {}

        for dataset in datasets:
            for cat in dataset.get('categories', []):
                cat_name = cat['name']
                if cat_name not in all_categories:
                    new_id = len(all_categories) + 1
                    all_categories[cat_name] = {
                        'id': new_id,
                        'name': cat_name,
                        'supercategory': cat.get('supercategory', '')
                    }

        merged['categories'] = list(all_categories.values())

        # Merge images and annotations with new IDs
        image_id_offset = 0
        ann_id_offset = 0

        for dataset in datasets:
            # Build category mapping for this dataset
            cat_name_to_new_id = {cat['name']: all_categories[cat['name']]['id']
                                  for cat in dataset.get('categories', [])}
            old_cat_to_new = {cat['id']: cat_name_to_new_id[cat['name']]
                              for cat in dataset.get('categories', [])}

            # Process images
            old_img_to_new = {}
            for img in dataset.get('images', []):
                new_img_id = img['id'] + image_id_offset
                old_img_to_new[img['id']] = new_img_id
                new_img = img.copy()
                new_img['id'] = new_img_id
                merged['images'].append(new_img)

            # Process annotations
            for ann in dataset.get('annotations', []):
                new_ann = ann.copy()
                new_ann['id'] = ann['id'] + ann_id_offset
                new_ann['image_id'] = old_img_to_new[ann['image_id']]
                new_ann['category_id'] = old_cat_to_new[ann['category_id']]
                merged['annotations'].append(new_ann)

            # Update offsets
            if dataset.get('images'):
                image_id_offset = max(img['id'] for img in merged['images']) + 1
            if dataset.get('annotations'):
                ann_id_offset = max(ann['id'] for ann in merged['annotations']) + 1

        logger.info(f"Merged {len(datasets)} datasets: {len(merged['images'])} images, "
                   f"{len(merged['annotations'])} annotations")
        return merged

    def save_annotations(self, coco_dict: Dict, output_path: str):
        """Save COCO annotations to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_dict, f, indent=2)
        logger.info(f"Saved annotations to {output_path}")

    def copy_selected_images(self,
                            image_ids: List[int],
                            source_dir: str,
                            output_dir: str,
                            rename: bool = False) -> Dict[int, str]:
        """Copy selected images to output directory."""
        os.makedirs(output_dir, exist_ok=True)

        id_to_new_path = {}
        for idx, img_id in enumerate(image_ids):
            if img_id not in self.images:
                logger.warning(f"Image ID {img_id} not found in annotations")
                continue

            img_info = self.images[img_id]
            src_path = os.path.join(source_dir, img_info['file_name'])

            if not os.path.exists(src_path):
                logger.warning(f"Source image not found: {src_path}")
                continue

            if rename:
                ext = Path(img_info['file_name']).suffix
                new_name = f"img_{idx:05d}{ext}"
            else:
                new_name = img_info['file_name']

            dst_path = os.path.join(output_dir, new_name)
            shutil.copy2(src_path, dst_path)
            id_to_new_path[img_id] = new_name

        logger.info(f"Copied {len(id_to_new_path)} images to {output_dir}")
        return id_to_new_path

    def _create_info(self) -> Dict:
        """Create info section for COCO format."""
        return {
            'description': 'Advanced Dataset Selection - Selected Frames',
            'url': '',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'Advanced Dataset Selection Pipeline',
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.annotations is None:
            return {}

        annotations = self.annotations.get('annotations', [])

        # Count annotations per category
        cat_counts = {}
        for ann in annotations:
            cat_id = ann['category_id']
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1

        # Map to category names
        cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        cat_name_counts = {cat_id_to_name.get(k, f'cat_{k}'): v
                          for k, v in cat_counts.items()}

        # Annotations per image
        img_ann_counts = {}
        for ann in annotations:
            img_id = ann['image_id']
            img_ann_counts[img_id] = img_ann_counts.get(img_id, 0) + 1

        return {
            'num_images': len(self.images),
            'num_annotations': len(annotations),
            'num_categories': len(self.categories),
            'annotations_per_category': cat_name_counts,
            'avg_annotations_per_image': sum(img_ann_counts.values()) / max(len(img_ann_counts), 1),
            'images_without_annotations': len(self.images) - len(img_ann_counts)
        }


if __name__ == "__main__":
    # Test the handler
    handler = COCOHandler()
    print("COCOHandler initialized successfully")
