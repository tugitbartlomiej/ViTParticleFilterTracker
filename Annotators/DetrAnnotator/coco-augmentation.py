import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm


class COCOAugmenter:
    def __init__(
            self,
            json_path: str,
            images_dir: str,
            output_dir: str,
            augmentations_per_image: int = 3,
            debug: bool = True
    ):
        """
        Initialize the COCO dataset augmenter with more conservative augmentations
        suitable for medical imaging datasets.

        Args:
            json_path: Path to COCO annotations JSON file
            images_dir: Directory containing original images
            output_dir: Directory to save augmented dataset
            augmentations_per_image: Number of augmentations to create per image
            debug: Enable detailed debug messages
        """
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.augmentations_per_image = augmentations_per_image
        self.debug = debug

        # Sprawdzenie, czy pliki wejściowe istnieją
        if not self.json_path.exists():
            raise FileNotFoundError(f"Plik JSON nie istnieje: {self.json_path}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Katalog obrazów nie istnieje: {self.images_dir}")

        # Create output directories
        self.output_images_dir = self.output_dir / "images"
        try:
            self.output_images_dir.mkdir(parents=True, exist_ok=True)
            print(f"Utworzono katalog wyjściowy: {self.output_images_dir}")
        except Exception as e:
            raise RuntimeError(f"Nie można utworzyć katalogu wyjściowego: {str(e)}")

        # Load annotations
        try:
            with open(self.json_path, 'r') as f:
                self.coco_data = json.load(f)
                print(
                    f"Wczytano dane JSON z {len(self.coco_data['images'])} obrazami i {len(self.coco_data['annotations'])} adnotacjami")
        except Exception as e:
            raise RuntimeError(f"Błąd wczytywania pliku JSON: {str(e)}")

        # Initialize ID counters
        if 'images' not in self.coco_data or not self.coco_data['images']:
            raise ValueError("Brak obrazów w pliku JSON")

        if 'annotations' not in self.coco_data or not self.coco_data['annotations']:
            raise ValueError("Brak adnotacji w pliku JSON")

        self.next_image_id = max(img['id'] for img in self.coco_data['images']) + 1
        self.next_ann_id = max(ann['id'] for ann in self.coco_data['annotations']) + 1

        print(f"ID początkowe: obrazy={self.next_image_id}, adnotacje={self.next_ann_id}")

        # Create conservative augmentation pipeline optimized for medical imaging
        self.transform = A.Compose([
            # Mild geometric transformations
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
                A.IAAAffine(
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    shear=None,
                    p=1.0
                ),
            ], p=0.4),

            # Subtle color variations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=10,
                    val_shift_limit=10,
                    p=1.0
                ),
            ], p=0.4),

            # Minimal quality variations
            A.OneOf([
                A.GaussNoise(
                    var_limit=(5.0, 15.0),
                    p=1.0
                ),
                A.ImageCompression(
                    quality_lower=85,
                    quality_upper=100,
                    p=1.0
                ),
            ], p=0.2),

            # Very light blur
            A.GaussianBlur(
                blur_limit=(3, 3),
                p=0.1
            ),

            # Horizontal flips
            A.HorizontalFlip(p=0.3),

        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    def augment_dataset(self) -> None:
        """Augment the entire dataset and create new annotations."""
        print("Starting dataset augmentation with conservative transforms...")

        # Sprawdźmy zawartość katalogu z obrazami
        image_files = list(self.images_dir.glob('*.*'))
        print(f"Znaleziono {len(image_files)} plików w katalogu {self.images_dir}")
        if len(image_files) > 0:
            print(f"Przykładowe pliki: {[f.name for f in image_files[:5]]}")

        # Keep track of original images to copy later
        original_images = []
        processed_images = 0
        augmented_images = 0
        failed_images = 0

        # Process each image in the dataset
        for img_info in tqdm(self.coco_data['images'], desc="Augmenting images"):
            image_id = img_info['id']
            image_filename = img_info['file_name']
            original_images.append(image_filename)

            # Load image
            image_path = self.images_dir / image_filename
            if not image_path.exists():
                print(f"WARNING: Obraz {image_path} nie istnieje, pomijanie...")
                failed_images += 1
                continue

            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"WARNING: Nie można odczytać obrazu {image_path}, pomijanie...")
                    failed_images += 1
                    continue

                if self.debug:
                    print(f"Obraz {image_path} wczytany pomyślnie, rozmiar: {image.shape}")
            except Exception as e:
                print(f"ERROR: Błąd wczytywania obrazu {image_path}: {str(e)}")
                failed_images += 1
                continue

            # Get original annotations for this image
            annotations = [
                ann for ann in self.coco_data['annotations']
                if ann['image_id'] == image_id
            ]

            if not annotations and self.debug:
                print(f"WARNING: Obraz {image_filename} nie ma adnotacji")

            # Create augmentations
            new_images = self._create_augmentations(image, image_filename, annotations)
            augmented_images += new_images
            processed_images += 1

            # Debug co 10 obrazów
            if processed_images % 10 == 0 and self.debug:
                print(f"Przetworzono {processed_images} obrazów, utworzono {augmented_images} augmentacji")

        # Copy original images to output directory
        print("\nKopiowanie oryginalnych obrazów...")
        copied_images = 0
        for filename in tqdm(original_images, desc="Copying originals"):
            src_path = self.images_dir / filename
            dst_path = self.output_images_dir / filename

            if src_path.exists():
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_images += 1
                except Exception as e:
                    print(f"ERROR: Nie można skopiować {src_path}: {str(e)}")

        # Sprawdź, czy jakiekolwiek pliki zostały zapisane
        output_files = list(self.output_images_dir.glob('*.*'))
        print(f"\nLiczba plików w katalogu wyjściowym: {len(output_files)}")

        # Save updated annotations
        output_json = self.output_dir / f"augmented_annotations_{datetime.now():%Y%m%d_%H%M%S}.json"
        try:
            with open(output_json, 'w') as f:
                json.dump(self.coco_data, f, indent=2)
            print(f"Zapisano adnotacje do {output_json}")
        except Exception as e:
            print(f"ERROR: Nie można zapisać adnotacji: {str(e)}")

        print(f"\nPodsumowanie augmentacji:")
        print(f"Oryginalne obrazy: {len(original_images)}")
        print(f"Przetworzono obrazów: {processed_images}")
        print(f"Nie znaleziono/nie wczytano obrazów: {failed_images}")
        print(f"Skopiowano oryginalnych obrazów: {copied_images}")
        print(f"Utworzono nowych augmentowanych obrazów: {augmented_images}")
        print(f"Łączna liczba obrazów w danych: {len(self.coco_data['images'])}")
        print(f"Łączna liczba adnotacji: {len(self.coco_data['annotations'])}")
        print(f"\nKatalog wyjściowy: {self.output_dir}")

    def _create_augmentations(
            self,
            image: np.ndarray,
            image_filename: str,
            annotations: List[Dict]
    ) -> int:
        """
        Create augmented versions of a single image and its annotations.

        Returns:
            int: Number of successful augmentations created
        """
        # Prepare bounding boxes and category ids for transformation
        bboxes = [ann['bbox'] for ann in annotations]
        category_ids = [ann['category_id'] for ann in annotations]

        # Licznik pomyślnych augmentacji
        successful_augmentations = 0

        # Skip images without annotations
        if not bboxes:
            if self.debug:
                print(f"Pomijanie obrazu {image_filename} - brak adnotacji")
            return 0

        # Create multiple augmentations
        for aug_idx in range(self.augmentations_per_image):
            try:
                # Apply transformation
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids
                )

                # Skip if no bounding boxes were preserved after transformation
                if not transformed['bboxes']:
                    if self.debug:
                        print(
                            f"WARNING: Augmentacja {image_filename} (aug_{aug_idx + 1}) nie zachowała żadnych bounding boxów")
                    continue

                # Generate new filename
                base_name = Path(image_filename).stem
                ext = Path(image_filename).suffix
                new_filename = f"{base_name}_aug_{aug_idx + 1}{ext}"

                # Save augmented image
                output_path = self.output_images_dir / new_filename
                try:
                    cv2.imwrite(str(output_path), transformed['image'])
                    if self.debug:
                        print(f"Zapisano augmentowany obraz: {output_path}")
                except Exception as e:
                    print(f"ERROR: Nie można zapisać augmentowanego obrazu {output_path}: {str(e)}")
                    continue

                # Create new image entry
                new_image = {
                    'id': self.next_image_id,
                    'file_name': new_filename,
                    'width': transformed['image'].shape[1],
                    'height': transformed['image'].shape[0],
                    'aug_source': image_filename
                }
                self.coco_data['images'].append(new_image)

                # Create new annotations
                annotations_created = 0
                for bbox, cat_id in zip(transformed['bboxes'], transformed['category_ids']):
                    # Ensure all bbox values are positive
                    bbox = [max(0, val) for val in bbox]

                    # Ensure width and height are positive
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        if self.debug:
                            print(f"WARNING: Pominięto bbox z szerokością/wysokością <= 0: {bbox}")
                        continue

                    new_ann = {
                        'id': self.next_ann_id,
                        'image_id': self.next_image_id,
                        'category_id': cat_id,
                        'bbox': list(map(float, bbox)),
                        'area': float(bbox[2] * bbox[3]),
                        'iscrowd': 0
                    }
                    self.coco_data['annotations'].append(new_ann)
                    self.next_ann_id += 1
                    annotations_created += 1

                if self.debug:
                    print(f"Utworzono {annotations_created} adnotacji dla {new_filename}")

                self.next_image_id += 1
                successful_augmentations += 1

            except Exception as e:
                print(f"ERROR: Nie udało się augmentować {image_filename} (aug_{aug_idx}): {str(e)}")
                continue

        return successful_augmentations


def main():
    # Configuration
    json_path = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/output/coco_annotations_from_yolo_dataset_20250218.json"
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train"
    output_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset"
    augmentations_per_image = 3

    try:
        # Sprawdź, czy ścieżki istnieją
        for path, name in [(json_path, "Plik JSON"), (images_dir, "Katalog obrazów")]:
            if not os.path.exists(path):
                print(f"BŁĄD: {name} nie istnieje: {path}")
                return

        print("=" * 50)
        print(f"Plik JSON: {json_path}")
        print(f"Katalog obrazów: {images_dir}")
        print(f"Katalog wyjściowy: {output_dir}")
        print(f"Liczba augmentacji na obraz: {augmentations_per_image}")
        print("=" * 50)

        # Create and run augmenter
        augmenter = COCOAugmenter(
            json_path=json_path,
            images_dir=images_dir,
            output_dir=output_dir,
            augmentations_per_image=augmentations_per_image,
            debug=True  # Włącz diagnostykę
        )

        augmenter.augment_dataset()

        print("Augmentacja zakończona pomyślnie")

    except Exception as e:
        print(f"Wystąpił krytyczny błąd: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()