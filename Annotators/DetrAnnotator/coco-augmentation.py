import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes in COCO format [x, y, width, height].

    Args:
        box1: First box in COCO format [x, y, width, height]
        box2: Second box in COCO format [x, y, width, height]

    Returns:
        IoU value
    """
    # Convert COCO format [x, y, width, height] to corners [x1, y1, x2, y2]
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]

    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate area of each box
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Check if there is intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    union_area = area1 + area2 - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


class COCOAugmenter:
    def __init__(
            self,
            json_path: str,
            images_dir: str,
            output_dir: str,
            augmentations_per_image: int = 3,
            augmentation_strength: str = "mild",
            preserve_originals: bool = True,
            start_idx: int = None,
            end_idx: int = None,
            debug: bool = True
    ):
        """
        Initialize the COCO dataset augmenter with advanced filtering.

        Args:
            json_path: Path to COCO annotations JSON file
            images_dir: Directory containing original images
            output_dir: Directory to save augmented dataset
            augmentations_per_image: Number of augmentations to create per image
            augmentation_strength: Intensity of augmentations ('mild', 'medium', 'strong')
            preserve_originals: Whether to include original images in output
            start_idx: Optional starting index to process only a range of images
            end_idx: Optional ending index (exclusive)
            debug: Enable detailed debug messages
        """
        self.json_path = Path(json_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.augmentations_per_image = augmentations_per_image
        self.augmentation_strength = augmentation_strength
        self.preserve_originals = preserve_originals
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.debug = debug

        # Create a timestamp for unique output filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Verify input files exist
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file does not exist: {self.json_path}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")

        # Create output directories
        self.output_images_dir = self.output_dir / "images"
        try:
            self.output_images_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Created output directory: {self.output_images_dir}")
        except Exception as e:
            raise RuntimeError(f"Cannot create output directory: {str(e)}")

        # Load annotations
        self._load_annotations()

        # Create augmentation pipeline
        self.transform = self._create_augmentation_pipeline()

        # Statistics
        self.stats = {
            "original_images": 0,
            "processed_images": 0,
            "skipped_images": 0,
            "augmented_images": 0,
            "failed_augmentations": 0,
            "successful_augmentations": 0,
            "annotations_created": 0
        }

    def _load_annotations(self):
        """Load and validate COCO annotations."""
        try:
            with open(self.json_path, 'r') as f:
                self.coco_data = json.load(f)

            if 'images' not in self.coco_data or not self.coco_data['images']:
                raise ValueError("No images found in the JSON file")

            if 'annotations' not in self.coco_data or not self.coco_data['annotations']:
                raise ValueError("No annotations found in the JSON file")

            if 'categories' not in self.coco_data or not self.coco_data['categories']:
                raise ValueError("No categories found in the JSON file")

            self.log(f"Loaded JSON data with {len(self.coco_data['images'])} images, "
                     f"{len(self.coco_data['annotations'])} annotations, and "
                     f"{len(self.coco_data['categories'])} categories")

            # Initialize ID counters for new annotations
            self.next_image_id = max(img['id'] for img in self.coco_data['images']) + 1
            self.next_ann_id = max(ann['id'] for ann in self.coco_data['annotations']) + 1
            self.log(f"Starting IDs: images={self.next_image_id}, annotations={self.next_ann_id}")

        except Exception as e:
            raise RuntimeError(f"Error loading JSON file: {str(e)}")

    def _create_augmentation_pipeline(self):
        """Create augmentation pipeline based on selected strength"""
        # Set parameters based on augmentation strength
        if self.augmentation_strength == 'mild':
            # Very gentle augmentations suitable for medical imaging
            shift_limit = 0.05
            scale_limit = 0.1
            rotate_limit = 15
            brightness_limit = 0.1
            contrast_limit = 0.1
            hue_shift = 5
            sat_shift = 10
            val_shift = 10
            noise_var = (5.0, 15.0)
            quality_range = (85, 100)
            blur_limit = (3, 3)
            flip_prob = 0.3
            geometric_prob = 0.4
            color_prob = 0.4
            noise_prob = 0.2
            blur_prob = 0.1

        elif self.augmentation_strength == 'medium':
            # Moderate augmentations with balanced distortion
            shift_limit = 0.1
            scale_limit = 0.15
            rotate_limit = 30
            brightness_limit = 0.2
            contrast_limit = 0.2
            hue_shift = 10
            sat_shift = 20
            val_shift = 20
            noise_var = (10.0, 30.0)
            quality_range = (75, 95)
            blur_limit = (3, 5)
            flip_prob = 0.4
            geometric_prob = 0.5
            color_prob = 0.5
            noise_prob = 0.3
            blur_prob = 0.2

        else:  # strong
            # More aggressive augmentations for maximum variation
            shift_limit = 0.15
            scale_limit = 0.2
            rotate_limit = 45
            brightness_limit = 0.3
            contrast_limit = 0.3
            hue_shift = 15
            sat_shift = 30
            val_shift = 30
            noise_var = (15.0, 50.0)
            quality_range = (65, 90)
            blur_limit = (3, 7)
            flip_prob = 0.5
            geometric_prob = 0.6
            color_prob = 0.6
            noise_prob = 0.4
            blur_prob = 0.3

        self.log(f"Using {self.augmentation_strength} augmentation settings")

        # Create the augmentation pipeline
        return A.Compose([
            # Geometric transformations
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=shift_limit,
                    scale_limit=scale_limit,
                    rotate_limit=rotate_limit,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=1.0
                ),
                # Using another ShiftScaleRotate instead of IAAAffine which causes errors
                A.ShiftScaleRotate(
                    shift_limit=shift_limit * 1.2,
                    scale_limit=scale_limit * 1.2,
                    rotate_limit=rotate_limit * 1.2,
                    border_mode=cv2.BORDER_REFLECT,  # Different border mode for variety
                    p=1.0
                ),
            ], p=geometric_prob),

            # Color transformations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=hue_shift,
                    sat_shift_limit=sat_shift,
                    val_shift_limit=val_shift,
                    p=1.0
                ),
            ], p=color_prob),

            # Noise and quality variations
            A.OneOf([
                A.GaussNoise(
                    var_limit=noise_var,
                    p=1.0
                ),
                A.ImageCompression(
                    quality_lower=quality_range[0],
                    quality_upper=quality_range[1],
                    p=1.0
                ),
            ], p=noise_prob),

            # Blur
            A.GaussianBlur(
                blur_limit=blur_limit,
                p=blur_prob
            ),

            # Flips
            A.HorizontalFlip(p=flip_prob),

        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    def _get_image_range(self):
        """
        Get the range of images to process based on the specified indices.

        Returns:
            List of image dictionaries to process
        """
        all_images = self.coco_data['images']
        total_images = len(all_images)

        # Sort images by filename for consistent ordering
        sorted_images = sorted(all_images, key=lambda x: x['file_name'])

        # Select range if specified
        if self.start_idx is not None and self.end_idx is not None:
            start = max(0, min(self.start_idx, total_images - 1))
            end = min(self.end_idx, total_images)
            selected_images = sorted_images[start:end]
            self.log(f"Selected {len(selected_images)} images from range {start} to {end}")
        else:
            selected_images = sorted_images
            self.log(f"Processing all {len(selected_images)} images")

        return selected_images

    def _get_annotations_for_image(self, image_id):
        """Get all annotations for a specific image ID."""
        return [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]

    def _filter_boxes(self, bboxes, category_ids, annotations):
        """
        Filter out small and duplicate bounding boxes.

        Args:
            bboxes: List of bounding boxes in COCO format
            category_ids: List of category IDs for each box
            annotations: List of annotation dictionaries

        Returns:
            tuple: (filtered_bboxes, filtered_category_ids, filtered_annotations)
        """
        filtered_bboxes = []
        filtered_category_ids = []
        filtered_annotations = []

        for i, (bbox, cat_id, ann) in enumerate(zip(bboxes, category_ids, annotations)):
            # Skip very small boxes
            if bbox[2] < 10 or bbox[3] < 10:  # Width or height too small
                self.log(f"Skipping small box with dimensions {bbox[2]}x{bbox[3]}", level='warning')
                continue

            # Check if this box overlaps too much with any already filtered box
            is_duplicate = False
            for j, existing_bbox in enumerate(filtered_bboxes):
                iou = calculate_iou(bbox, existing_bbox)
                if iou > 0.8:  # High overlap threshold
                    self.log(f"Found duplicate box with IoU={iou:.2f}", level='warning')
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_bboxes.append(bbox)
                filtered_category_ids.append(cat_id)
                filtered_annotations.append(ann)

        self.log(f"Filtered {len(bboxes) - len(filtered_bboxes)} potentially problematic boxes")
        return filtered_bboxes, filtered_category_ids, filtered_annotations

    def _post_process_transformed_boxes(self, transformed_bboxes, transformed_category_ids):
        """
        Post-process transformed boxes to filter out duplicates and invalid ones.

        Args:
            transformed_bboxes: List of transformed bounding boxes
            transformed_category_ids: List of category IDs for transformed boxes

        Returns:
            tuple: (filtered_bboxes, filtered_category_ids)
        """
        filtered_bboxes = []
        filtered_category_ids = []

        for i, (bbox, cat_id) in enumerate(zip(transformed_bboxes, transformed_category_ids)):
            # Ensure all bbox values are positive
            bbox = [max(0, val) for val in bbox]

            # Ensure width and height are positive and not too small
            if bbox[2] <= 10 or bbox[3] <= 10:
                self.log(f"WARNING: Skipped transformed bbox with width/height too small: {bbox}", level='warning')
                continue

            # Check for duplicates in transformed boxes
            is_duplicate = False
            for existing_bbox in filtered_bboxes:
                iou = calculate_iou(bbox, existing_bbox)
                if iou > 0.7:  # Slightly lower threshold for transformed boxes
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_bboxes.append(bbox)
                filtered_category_ids.append(cat_id)

        return filtered_bboxes, filtered_category_ids

    def augment_dataset(self) -> str:
        """
        Augment the dataset and create new annotations.

        Returns:
            Path to the output JSON file
        """
        self.log(f"Starting dataset augmentation with {self.augmentation_strength} transforms...")

        # Check images directory content
        image_files = list(self.images_dir.glob('*.*'))
        self.log(f"Found {len(image_files)} files in directory {self.images_dir}")
        if len(image_files) > 0:
            self.log(f"Sample files: {[f.name for f in image_files[:5]]}")

        # Get images to process
        selected_images = self._get_image_range()

        # Create new COCO data structure for output
        output_coco_data = {
            "info": {
                "description": f"Augmented COCO dataset with {self.augmentation_strength} augmentations",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "COCOAugmenter",
                "date_created": datetime.now().strftime("%Y-%m-%d")
            },
            "licenses": self.coco_data.get("licenses", [
                {"id": 1, "name": "Unknown", "url": "Unknown"}
            ]),
            "categories": self.coco_data["categories"],
            "images": [],
            "annotations": []
        }

        # Process each image in the dataset
        for img_info in tqdm(selected_images, desc="Augmenting images"):
            image_id = img_info['id']
            image_filename = img_info['file_name']

            # Load image
            image_path = self.images_dir / image_filename
            if not image_path.exists():
                self.log(f"WARNING: Image {image_path} does not exist, skipping...", level='warning')
                self.stats["skipped_images"] += 1
                continue

            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    self.log(f"WARNING: Could not read image {image_path}, skipping...", level='warning')
                    self.stats["skipped_images"] += 1
                    continue
            except Exception as e:
                self.log(f"ERROR: Error loading image {image_path}: {str(e)}", level='error')
                self.stats["skipped_images"] += 1
                continue

            # Get original annotations for this image
            annotations = self._get_annotations_for_image(image_id)

            if not annotations:
                self.log(f"WARNING: Image {image_filename} has no annotations", level='warning')
                if not self.preserve_originals:
                    continue

            # If preserving originals, add original image to output
            if self.preserve_originals:
                try:
                    # Copy original image to output directory
                    dst_path = self.output_images_dir / image_filename
                    shutil.copy2(image_path, dst_path)

                    # Add original image to output data
                    output_coco_data['images'].append(img_info.copy())

                    # Add original annotations to output data
                    for ann in annotations:
                        output_coco_data['annotations'].append(ann.copy())

                    self.stats["original_images"] += 1
                except Exception as e:
                    self.log(f"ERROR: Cannot copy {image_path}: {str(e)}", level='error')

            # Skip augmentation if there are no annotations
            if not annotations:
                continue

            # Prepare bounding boxes and category ids for transformation
            bboxes = [ann['bbox'] for ann in annotations]
            category_ids = [ann['category_id'] for ann in annotations]

            # Filter out small and duplicate boxes
            filtered_bboxes, filtered_category_ids, filtered_annotations = self._filter_boxes(
                bboxes, category_ids, annotations
            )

            # Skip if no boxes left after filtering
            if not filtered_bboxes:
                self.log(f"No valid boxes after filtering for {image_filename}, skipping augmentation")
                continue

            # Create multiple augmentations
            augmented_count = 0
            for aug_idx in range(self.augmentations_per_image):
                try:
                    # Apply transformation
                    transformed = self.transform(
                        image=image,
                        bboxes=filtered_bboxes,
                        category_ids=filtered_category_ids
                    )

                    # Skip if no bounding boxes were preserved after transformation
                    if not transformed['bboxes']:
                        self.log(f"WARNING: Augmentation {image_filename} (aug_{aug_idx + 1}) lost all bounding boxes",
                                level='warning')
                        self.stats["failed_augmentations"] += 1
                        continue

                    # Post-process transformed boxes to filter out duplicates and invalid ones
                    processed_bboxes, processed_category_ids = self._post_process_transformed_boxes(
                        transformed['bboxes'], transformed['category_ids']
                    )

                    # Skip if no valid boxes after post-processing
                    if not processed_bboxes:
                        self.log(f"No valid boxes after post-processing for {image_filename} augmentation {aug_idx + 1}",
                                level='warning')
                        self.stats["failed_augmentations"] += 1
                        continue

                    # Generate new filename
                    base_name = Path(image_filename).stem
                    ext = Path(image_filename).suffix
                    new_filename = f"{base_name}_aug_{aug_idx + 1}{ext}"

                    # Save augmented image
                    output_path = self.output_images_dir / new_filename
                    try:
                        cv2.imwrite(str(output_path), transformed['image'])
                        self.log(f"Saved augmented image: {output_path}")
                    except Exception as e:
                        self.log(f"ERROR: Cannot save augmented image {output_path}: {str(e)}", level='error')
                        self.stats["failed_augmentations"] += 1
                        continue

                    # Create new image entry
                    new_image = {
                        'id': self.next_image_id,
                        'file_name': new_filename,
                        'width': transformed['image'].shape[1],
                        'height': transformed['image'].shape[0],
                        'aug_source': image_filename
                    }
                    output_coco_data['images'].append(new_image)

                    # Create new annotations
                    annotations_created = 0
                    for bbox, cat_id in zip(processed_bboxes, processed_category_ids):
                        new_ann = {
                            'id': self.next_ann_id,
                            'image_id': self.next_image_id,
                            'category_id': cat_id,
                            'bbox': list(map(float, bbox)),
                            'area': float(bbox[2] * bbox[3]),
                            'iscrowd': 0
                        }
                        output_coco_data['annotations'].append(new_ann)
                        self.next_ann_id += 1
                        annotations_created += 1
                        self.stats["annotations_created"] += 1

                    self.log(f"Created {annotations_created} annotations for {new_filename}")
                    self.next_image_id += 1
                    augmented_count += 1
                    self.stats["successful_augmentations"] += 1

                except Exception as e:
                    self.log(f"ERROR: Failed to augment {image_filename} (aug_{aug_idx}): {str(e)}", level='error')
                    self.stats["failed_augmentations"] += 1
                    continue

            # Update augmented images count if any successful augmentations
            if augmented_count > 0:
                self.stats["augmented_images"] += 1

            self.stats["processed_images"] += 1

            # Log progress every 10 images
            if self.stats["processed_images"] % 10 == 0:
                self.log(f"Processed {self.stats['processed_images']} images, "
                        f"created {self.stats['successful_augmentations']} augmentations")

        # Create range identifier for output filename
        range_identifier = ""
        if self.start_idx is not None and self.end_idx is not None:
            range_identifier = f"_{self.start_idx}-{self.end_idx}"

        # Save updated annotations
        output_json = self.output_dir / f"augmented_coco{range_identifier}_{self.timestamp}.json"
        try:
            with open(output_json, 'w') as f:
                json.dump(output_coco_data, f, indent=2)
            self.log(f"Saved annotations to {output_json}")
        except Exception as e:
            self.log(f"ERROR: Cannot save annotations: {str(e)}", level='error')

        # Print summary
        self._print_summary()

        return str(output_json)

    def log(self, message, level='info'):
        """Log a message with a specified level if debug mode is enabled."""
        if not self.debug:
            return

        if level == 'error':
            print(f"ERROR: {message}")
        elif level == 'warning':
            print(f"WARNING: {message}")
        else:
            print(message)

    def _print_summary(self):
        """Print a summary of the augmentation process."""
        self.log("\n" + "=" * 70)
        self.log("AUGMENTATION SUMMARY")
        self.log("=" * 70)
        self.log(f"Images processed: {self.stats['processed_images']}")
        self.log(f"Images skipped: {self.stats['skipped_images']}")
        self.log(f"Original images preserved: {self.stats['original_images']}")
        self.log(f"Images augmented: {self.stats['augmented_images']}")
        self.log(f"Successful augmentations: {self.stats['successful_augmentations']}")
        self.log(f"Failed augmentations: {self.stats['failed_augmentations']}")
        self.log(f"New annotations created: {self.stats['annotations_created']}")
        self.log(f"Output directory: {self.output_dir}")
        self.log("=" * 70)


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='COCO Dataset Augmenter for Medical Images')

    # Required arguments
    parser.add_argument('--json_path', required=False, type=str,
                        default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Detr/coco_annotations_from_yolo_dataset_20250218.json",
                        help='Path to the COCO annotations JSON file')
    parser.add_argument('--images_dir', required=False, type=str,
                        default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train",
                        help='Directory containing the original images')
    parser.add_argument('--output_dir', required=False, type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset',
                        help='Directory to save the augmented dataset')

    # Optional arguments
    parser.add_argument('--augmentations_per_image', type=int, default=3,
                        help='Number of augmentations to create per image (default: 3)')
    parser.add_argument('--augmentation_strength', type=str, default='mild',
                        choices=['mild', 'medium', 'strong'],
                        help='Intensity of augmentations (default: mild)')
    parser.add_argument('--preserve_originals', action='store_true', default=True,
                        help='Include original images in output dataset (default: True)')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Starting index for processing a subset of images (optional)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending index for processing a subset of images (optional)')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Enable detailed debug messages (default: True)')

    # Parse arguments
    args = parser.parse_args()

    try:
        # Check if paths exist
        for path, name in [(args.json_path, "JSON file"), (args.images_dir, "Images directory")]:
            if not os.path.exists(path):
                print(f"ERROR: {name} does not exist: {path}")
                return 1

        print("=" * 70)
        print("COCO DATASET AUGMENTER (IMPROVED VERSION)")
        print("=" * 70)
        print(f"JSON path: {args.json_path}")
        print(f"Images directory: {args.images_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Augmentations per image: {args.augmentations_per_image}")
        print(f"Augmentation strength: {args.augmentation_strength}")
        print(f"Preserve original images: {args.preserve_originals}")

        if args.start_idx is not None and args.end_idx is not None:
            print(f"Processing image range: {args.start_idx} to {args.end_idx}")
        else:
            print("Processing all images")

        print(f"Debug mode: {args.debug}")
        print("=" * 70)

        # Create and run augmenter
        augmenter = COCOAugmenter(
            json_path=args.json_path,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            augmentations_per_image=args.augmentations_per_image,
            augmentation_strength=args.augmentation_strength,
            preserve_originals=args.preserve_originals,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            debug=args.debug
        )

        output_path = augmenter.augment_dataset()

        print("Augmentation completed successfully")
        print(f"Output annotations saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"A critical error occurred: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())