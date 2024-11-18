import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from datetime import datetime
import shutil
from typing import Dict, List, Tuple


class COCOAugmenter:
    def __init__(
            self,
            json_path: str,
            images_dir: str,
            output_dir: str,
            augmentations_per_image: int = 3
    ):
        """
        Initialize the COCO dataset augmenter.

        Args:
            json_path: Path to COCO annotations JSON file
            images_dir: Directory containing original images
            output_dir: Directory to save augmented dataset
            augmentations_per_image: Number of augmentations to create per image
        """
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.augmentations_per_image = augmentations_per_image

        # Create output directories
        self.output_images_dir = self.output_dir / "images"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)

        # Load annotations
        with open(self.json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Initialize ID counters
        self.next_image_id = max(img['id'] for img in self.coco_data['images']) + 1
        self.next_ann_id = max(ann['id'] for ann in self.coco_data['annotations']) + 1

        # Create advanced augmentation pipeline
        self.transform = A.Compose([
            # Group 1: Geometric Transformations
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=45,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
                A.IAAPerspective(scale=(0.05, 0.15), p=1.0),
                A.IAAAffine(
                    scale=1.0,
                    rotate=(-45, 45),
                    shear=(-15, 15),
                    p=1.0
                ),
            ], p=0.7),

            # Group 2: Color Transformations
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

            # Group 3: Noise and Quality
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            ], p=0.5),

            # Group 4: Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ], p=0.3),

            # Group 5: Weather and Environmental
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
        """Augment the entire dataset and create new annotations."""
        print("Starting dataset augmentation...")

        # Keep track of original images to copy later
        original_images = []

        # Process each image in the dataset
        for img_info in tqdm(self.coco_data['images'], desc="Augmenting images"):
            image_id = img_info['id']
            image_filename = img_info['file_name']
            original_images.append(image_filename)

            # Load image
            image_path = self.images_dir / image_filename
            if not image_path.exists():
                print(f"Warning: Image {image_path} not found, skipping...")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}, skipping...")
                continue

            # Get original annotations for this image
            annotations = [
                ann for ann in self.coco_data['annotations']
                if ann['image_id'] == image_id
            ]

            # Create augmentations
            self._create_augmentations(image, image_filename, annotations)

        # Copy original images to output directory
        print("\nCopying original images...")
        for filename in tqdm(original_images, desc="Copying originals"):
            src_path = self.images_dir / filename
            dst_path = self.output_images_dir / filename
            if src_path.exists():
                shutil.copy2(src_path, dst_path)

        # Save updated annotations
        output_json = self.output_dir / f"augmented_annotations_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(output_json, 'w') as f:
            json.dump(self.coco_data, f, indent=2)

        print(f"\nAugmentation completed!")
        print(f"Original images: {len(original_images)}")
        print(f"New images: {(self.next_image_id - len(original_images))}")
        print(f"Total images: {len(self.coco_data['images'])}")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Annotations file: {output_json}")

    def _create_augmentations(
            self,
            image: np.ndarray,
            image_filename: str,
            annotations: List[Dict]
    ) -> None:
        """Create augmented versions of a single image and its annotations."""
        # Prepare bounding boxes and category ids for transformation
        bboxes = [ann['bbox'] for ann in annotations]
        category_ids = [ann['category_id'] for ann in annotations]

        # Create multiple augmentations
        for aug_idx in range(self.augmentations_per_image):
            try:
                # Apply transformation
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids
                )

                # Generate new filename
                base_name = Path(image_filename).stem
                ext = Path(image_filename).suffix
                new_filename = f"{base_name}_aug_{aug_idx + 1}{ext}"

                # Save augmented image
                cv2.imwrite(
                    str(self.output_images_dir / new_filename),
                    transformed['image']
                )

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
                print(f"Warning: Failed to augment {image_filename} (aug_{aug_idx}): {str(e)}")
                continue


def main():
    # Configuration
    json_path = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/output/coco_annotations_76_100.json"
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/OpencvTrackerAnnotator/output/yolo_dataset/train/images"
    output_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset"
    augmentations_per_image = 3

    # Create and run augmenter
    augmenter = COCOAugmenter(
        json_path=json_path,
        images_dir=images_dir,
        output_dir=output_dir,
        augmentations_per_image=augmentations_per_image
    )

    augmenter.augment_dataset()


if __name__ == "__main__":
    main()