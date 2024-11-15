import os
import json
import shutil
from typing import Tuple, Dict, List
import random
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np


class COCOtoYOLOConverter:
    def __init__(
            self,
            coco_annotations_path: str,
            images_dir: str,
            output_dir: str,
            splits: Tuple[float, float, float] = (0.7, 0.2, 0.1)
    ):
        """
        Initialize the COCO to YOLO format converter.
        """
        self.coco_annotations_path = Path(coco_annotations_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)

        # Validate split ratios with tolerance
        split_sum = np.sum(splits)
        if not np.isclose(split_sum, 1.0, rtol=1e-5):
            raise ValueError(f"Split ratios must sum to approximately 1.0 (got {split_sum})")

        self.train_ratio, self.val_ratio, self.test_ratio = np.array(splits) / split_sum

        # Create output directory structure
        self.dataset_dirs = {
            'train': self.output_dir / 'train',
            'val': self.output_dir / 'val',
            'test': self.output_dir / 'test'
        }

        # Load and verify data
        self._load_and_verify_data()

    def _load_and_verify_data(self):
        """Load and verify all data sources."""
        print(f"Loading annotations from: {self.coco_annotations_path}")
        print(f"Loading images from: {self.images_dir}")

        # Check if annotations file exists and is not empty
        if not self.coco_annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.coco_annotations_path}")

        if self.coco_annotations_path.stat().st_size == 0:
            raise ValueError(f"Annotations file is empty: {self.coco_annotations_path}")

        # Load COCO annotations with error handling
        try:
            with open(self.coco_annotations_path, 'r', encoding='utf-8') as f:
                self.coco_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            # Opcjonalnie: Wyświetlenie ostatnich kilku linii pliku w celu diagnostyki
            with open(self.coco_annotations_path, 'r', encoding='utf-8') as f:
                data = f.read()
                print("Ostatnie 500 znaków pliku JSON:")
                print(data[-500:])
            raise

        # Get list of actual image files
        self.available_images = set(f.name for f in self.images_dir.glob("*.jpg"))

        # Create image map and verify image existence
        self.image_map = {}
        self.valid_image_info = []

        print("/nVerifying image files...")
        for img in self.coco_data.get('images', []):
            if img['file_name'] in self.available_images:
                self.image_map[img['id']] = img
                self.valid_image_info.append(img)
            else:
                print(f"Warning: Image {img['file_name']} from annotations not found in {self.images_dir}")

        print(f"/nFound {len(self.available_images)} images in directory")
        print(f"Found {len(self.coco_data.get('images', []))} images in annotations")
        print(f"Found {len(self.valid_image_info)} valid images with matching files")

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create the necessary directory structure for YOLO dataset."""
        for split in self.dataset_dirs.values():
            (split / 'images').mkdir(parents=True, exist_ok=True)
            (split / 'labels').mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {split}")

    def _get_image_annotations(self) -> Dict[int, List]:
        """Group annotations by image ID."""
        image_annotations = {}
        for ann in self.coco_data.get('annotations', []):
            if ann['image_id'] in self.image_map:  # Only include annotations for valid images
                if ann['image_id'] not in image_annotations:
                    image_annotations[ann['image_id']] = []
                image_annotations[ann['image_id']].append(ann)
        return image_annotations

    def _convert_bbox_coco_to_yolo(
            self,
            bbox: List[float],
            img_width: float,
            img_height: float
    ) -> List[float]:
        """Convert COCO bbox format to YOLO format."""
        x, y, w, h = bbox

        # Calculate normalized center coordinates, width, and height
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        # Ensure values are within [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))

        return [x_center, y_center, w_norm, h_norm]

    def _process_image(self, image_info: Dict, annotations: List, split_dir: Path) -> bool:
        """Process a single image and its annotations."""
        try:
            image_path = self.images_dir / image_info['file_name']
            if not image_path.exists():
                print(f"Warning: Image file does not exist: {image_path}")
                return False

            # Copy image
            target_image_path = split_dir / 'images' / image_info['file_name']
            shutil.copy2(image_path, target_image_path)

            # Convert and save annotations
            if annotations:
                base_filename = Path(image_info['file_name']).stem
                label_path = split_dir / 'labels' / f"{base_filename}.txt"

                yolo_annotations = []
                img_width = float(image_info['width'])
                img_height = float(image_info['height'])

                for ann in annotations:
                    bbox = self._convert_bbox_coco_to_yolo(
                        ann['bbox'],
                        img_width,
                        img_height
                    )
                    # Zakładając, że klasa jest zawsze 'surgical_tool' z ID 0
                    yolo_ann = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                    yolo_annotations.append(yolo_ann)

                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write("/n".join(yolo_annotations))

            return True

        except Exception as e:
            print(f"Error processing image {image_info['file_name']}: {str(e)}")
            return False

    def convert(self):
        """Convert and split the dataset."""
        print("/nStarting dataset conversion and splitting...")

        # Get valid image IDs and their annotations
        image_annotations = self._get_image_annotations()
        valid_image_ids = list(image_annotations.keys())

        if not valid_image_ids:
            raise ValueError("No valid images with annotations found in the dataset")

        print(f"Processing {len(valid_image_ids)} valid images with annotations")

        # Split the dataset
        random.seed(42)
        random.shuffle(valid_image_ids)

        total_images = len(valid_image_ids)
        train_size = int(self.train_ratio * total_images)
        val_size = int(self.val_ratio * total_images)

        splits_data = {
            'train': valid_image_ids[:train_size],
            'val': valid_image_ids[train_size:train_size + val_size],
            'test': valid_image_ids[train_size + val_size:]
        }

        # Process each split
        successful_conversions = 0
        for split_name, image_ids in splits_data.items():
            print(f"/nProcessing {split_name} split ({len(image_ids)} images)...")
            split_dir = self.dataset_dirs[split_name]

            with tqdm(total=len(image_ids)) as pbar:
                for image_id in image_ids:
                    image_info = self.image_map[image_id]
                    annotations = image_annotations.get(image_id, [])

                    if self._process_image(image_info, annotations, split_dir):
                        successful_conversions += 1
                    pbar.update(1)

        # Create YAML config
        yaml_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {0: 'surgical_tool'},
            'nc': 1
        }

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        # Print summary
        print("/nDataset conversion completed!")
        print(f"Successfully processed {successful_conversions} out of {total_images} images")
        print(f"Train set: {len(splits_data['train'])} images")
        print(f"Validation set: {len(splits_data['val'])} images")
        print(f"Test set: {len(splits_data['test'])} images")
        print(f"/nDataset saved to: {self.output_dir}")
        print("YAML configuration file created: dataset.yaml")


def main():
    # Ustaw ścieżki względem aktualnej struktury folderów
    current_dir = Path(__file__).parent

    # Wyświetl zawartość folderu Raw_Images
    raw_images_dir = current_dir / "output" / "Yolo" / "Raw_Images"
    print("/nContents of Raw_Images directory:")
    for file in raw_images_dir.glob("*"):
        print(f" - {file.name}")

    converter = COCOtoYOLOConverter(
        coco_annotations_path="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/OpencvTrackerAnnotator/output/annotations.json",
        images_dir="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/OpencvTrackerAnnotator/output/Raw_Images",
        output_dir="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/OpencvTrackerAnnotator/output/yolo_dataset"
    )

    converter.convert()


if __name__ == "__main__":
    main()
