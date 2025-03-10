import json
import os
import tarfile
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
            archive_after_n_images: int = 4
    ):
        """
        Inicjalizacja augmentera zbioru COCO.

        Args:
            json_path: Ścieżka do pliku JSON w formacie COCO
            images_dir: Katalog z oryginalnymi obrazami
            output_dir: Katalog, w którym zapiszemy wyniki augmentacji
            augmentations_per_image: Ile augmentacji tworzymy z każdego obrazu
            archive_after_n_images: Po ilu przetworzonych (oryginalnych) obrazach
                                    tworzyć archiwum .tar.gz
        """
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.augmentations_per_image = augmentations_per_image
        self.archive_after_n_images = archive_after_n_images

        # Katalog na Z-AUGMENTOWANE obrazy (powstające w wyniku transformacji)
        self.output_images_dir = self.output_dir / "images"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)

        # Katalog do archiwizacji oryginalnych obrazów, np. "yolo_dateset_2025_test_archivier_archived"
        archived_folder_name = self.images_dir.name + "_archived"
        self.archives_dir = self.images_dir.parent / archived_folder_name
        self.archives_dir.mkdir(parents=True, exist_ok=True)

        # Wczytujemy dane COCO (oryginalne adnotacje)
        with open(self.json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Ustalamy dalsze ID dla nowych obrazów i adnotacji
        self.next_image_id = max(img['id'] for img in self.coco_data['images']) + 1
        self.next_ann_id = max(ann['id'] for ann in self.coco_data['annotations']) + 1

        # Bufor, w którym gromadzimy ścieżki do oryginalnych obrazów –
        # co archive_after_n_images sztuk, będziemy je spakowywać
        self.archive_buffer: List[Path] = []

        # Definicja transformacji Albumentations (przykładowa)
        self.transform = A.Compose([
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
                A.Perspective(scale=(0.05, 0.15), p=1.0),
                A.Affine(
                    scale=1.0,
                    rotate=(-45, 45),
                    shear=(-15, 15),
                    p=1.0
                ),
            ], p=0.7),

            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1.0
                ),
            ], p=0.7),

            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            ], p=0.5),

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.3),

            A.OneOf([
                A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.8, p=1.0),
                A.RandomShadow(
                    num_shadows_lower=1,
                    num_shadows_upper=3,
                    shadow_dimension=5,
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
            ], p=0.3),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    def augment_dataset(self) -> None:
        """
        Wykonuje augmentację całego zbioru COCO, a oryginalne obrazy
        (co 'archive_after_n_images' sztuk) są archiwizowane w .tar.gz
        w folderze obok images_dir (ten sam parent),
        o nazwie images_dir.name + "_archived".
        """
        print("Starting dataset augmentation...")

        # Pętla po wszystkich obrazach z pliku COCO
        for i, img_info in enumerate(tqdm(self.coco_data['images'], desc="Augmenting images")):
            image_id = img_info['id']
            image_filename = img_info['file_name']

            # Ścieżka do oryginalnego obrazu
            image_path = self.images_dir / image_filename
            if not image_path.exists():
                print(f"Warning: Image {image_path} not found, skipping...")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}, skipping...")
                continue

            # Dodajemy ścieżkę do bufora archiwizacji
            self.archive_buffer.append(image_path)
            # Sprawdzamy, czy osiągnęliśmy próg do archiwizacji
            if len(self.archive_buffer) >= self.archive_after_n_images:
                self._archive_original_images()

            # Pobieramy adnotacje (bounding boxy) związane z tym obrazem
            annotations = [
                ann for ann in self.coco_data['annotations']
                if ann['image_id'] == image_id
            ]

            # Tworzymy augmentacje tego obrazu
            self._create_augmentations(image, image_filename, annotations)

        # Jeżeli po zakończeniu pętli coś jeszcze jest w buforze, archiwizujemy
        if self.archive_buffer:
            self._archive_original_images()

        # Na koniec zapis zaktualizowanego pliku JSON
        output_json = self.output_dir / f"augmented_annotations_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(output_json, 'w') as f:
            json.dump(self.coco_data, f, indent=2)

        print("\nAugmentation completed!")
        print(f"Total images in original COCO: {len(self.coco_data['images'])}")
        print(f"New images (augmented): {self.next_image_id - (max(img['id'] for img in self.coco_data['images']) + 1)}")
        print(f"Total images in updated COCO: {len(self.coco_data['images'])}")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Annotations file: {output_json}")

    def _create_augmentations(
            self,
            image: np.ndarray,
            image_filename: str,
            annotations: List[Dict]
    ) -> None:
        """
        Tworzy augmentacje pojedynczego obrazu wraz z jego adnotacjami.
        """
        bboxes = [ann['bbox'] for ann in annotations]
        category_ids = [ann['category_id'] for ann in annotations]

        for aug_idx in range(self.augmentations_per_image):
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids
                )
                # Generujemy nazwę dla zaugmentowanego pliku
                base_name = Path(image_filename).stem
                ext = Path(image_filename).suffix
                new_filename = f"{base_name}_aug_{aug_idx + 1}{ext}"

                # Zapis nowego pliku
                output_path = self.output_images_dir / new_filename
                cv2.imwrite(str(output_path), transformed['image'])

                # Dodanie wpisu o nowym obrazie do COCO
                new_image_info = {
                    'id': self.next_image_id,
                    'file_name': new_filename,
                    'width': transformed['image'].shape[1],
                    'height': transformed['image'].shape[0],
                    'aug_source': image_filename  # opcjonalnie
                }
                self.coco_data['images'].append(new_image_info)

                # Dodanie nowych adnotacji w COCO
                for bbox, cat_id in zip(transformed['bboxes'], transformed['category_ids']):
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

                self.next_image_id += 1

            except Exception as e:
                print(f"Warning: Failed to augment {image_filename} (aug_{aug_idx + 1}): {str(e)}")
                continue

    def _archive_original_images(self) -> None:
        """
        Tworzy archiwum .tar.gz z oryginalnych obrazów (które znajdują się w buforze),
        a następnie USUWA je z dysku i czyści bufor.
        """
        if not self.archive_buffer:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"originals_{timestamp}.tar.gz"
        archive_path = self.archives_dir / archive_name

        try:
            with tarfile.open(archive_path, 'w:gz') as tar:
                for img_path in self.archive_buffer:
                    # arcname – tylko nazwa pliku (bez ścieżek)
                    tar.add(str(img_path), arcname=img_path.name)

            # Teraz usuwamy zarchiwizowane pliki z dysku
            for img_path in self.archive_buffer:
                if img_path.exists():
                    os.remove(img_path)

            print(f"\nCreated archive with {len(self.archive_buffer)} original image(s): {archive_path}")
            print("Archived files removed from disk.")

        except Exception as e:
            print(f"Error while creating archive {archive_path}: {str(e)}")

        # Czyścimy bufor
        self.archive_buffer.clear()


def main():
    # Przykładowe ścieżki
    json_path = "/mnt/evafs/faculty/home/bpiotrowski/DETR/coco_annotations_from_yolo_dataset_20250218.json"
    images_dir = "/mnt/evafs/faculty/home/bpiotrowski/datasets/yolo_dataset_20250218/images/train"
    output_dir = "/mnt/evafs/faculty/home/bpiotrowski/datasets/DETR_augmented_dataset_20250218"

    augmentations_per_image = 3
    archive_after_n_images = 3333

    augmenter = COCOAugmenter(
        json_path=json_path,
        images_dir=images_dir,
        output_dir=output_dir,
        augmentations_per_image=augmentations_per_image,
        archive_after_n_images=archive_after_n_images
    )
    augmenter.augment_dataset()


if __name__ == "__main__":
    main()
