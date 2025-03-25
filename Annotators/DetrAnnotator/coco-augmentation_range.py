import json
import shutil
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm


class COCORangeAugmenter:
    """
    A class for augmenting a specified range of images in a COCO dataset and creating
    a new COCO dataset with the original and augmented images.
    """

    def __init__(self):
        """Initialize the COCO dataset range augmenter with hardcoded parameters."""
        # HARDCODED PARAMETERS - MODIFY THESE AS NEEDED
        # --------------------------------------------------------------------------
        # Input/output paths
        self.json_path = Path(
            "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/output/coco_annotations_from_yolo_dataset_20250218.json")
        self.images_dir = Path(
            "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train")
        self.output_dir = Path(
            "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset")

        # Range selection
        self.start_idx = 200  # Starting index (inclusive)
        self.end_idx = 300  # Ending index (exclusive), None to process until the end

        # Augmentation settings
        self.augmentations_per_image = 10  # Number of augmentations to create per image
        self.augmentation_strength = 'strong'  # Options: 'mild', 'medium', 'strong'
        self.preserve_original_images = True  # Whether to include original images in output

        # Debug/logging
        self.verbose = True  # Enable detailed logging
        # --------------------------------------------------------------------------

        # Create a timestamp for unique output filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directories
        self.output_images_dir = self.output_dir / "images"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.log(f"Created output directory: {self.output_images_dir}")

        # Load and validate COCO annotations
        self._load_annotations()

        # Initialize ID counters for new annotations
        self.next_image_id = max(img['id'] for img in self.coco_data['images']) + 1
        self.next_ann_id = max(ann['id'] for ann in self.coco_data['annotations']) + 1
        self.log(f"Starting IDs: images={self.next_image_id}, annotations={self.next_ann_id}")

        # Create new COCO data structure for the output dataset
        self.new_coco_data = self._initialize_coco_structure()

        # Set up the augmentation pipeline based on the specified strength
        self.transform = self._create_augmentation_pipeline()

        # Performance tracking
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

        except Exception as e:
            raise RuntimeError(f"Error loading JSON file: {str(e)}")

    def _initialize_coco_structure(self):
        """Initialize a new COCO data structure for the output dataset."""
        return {
            "info": {
                "description": f"Augmented COCO dataset (range: {self.start_idx}-{self.end_idx or 'end'})",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "COCORangeAugmenter",
                "date_created": datetime.now().strftime("%Y-%m-%d")
            },
            "licenses": self.coco_data.get("licenses", [
                {"id": 1, "name": "Unknown", "url": ""}
            ]),
            "categories": self.coco_data["categories"],
            "images": [],
            "annotations": []
        }

    def _create_augmentation_pipeline(self):
        """Create an augmentation pipeline based on the specified strength."""
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
                A.IAAAffine(
                    scale=(1.0 - scale_limit, 1.0 + scale_limit),
                    translate_percent=(-shift_limit, shift_limit),
                    rotate=(-rotate_limit, rotate_limit),
                    shear=None,
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

        # Sort images by ID or filename to ensure consistent ordering
        sorted_images = sorted(all_images, key=lambda x: x['file_name'])

        # Determine actual indices to use
        actual_start = min(self.start_idx, total_images)
        actual_end = min(self.end_idx or total_images, total_images)

        # Get the selected range of images
        selected_images = sorted_images[actual_start:actual_end]

        self.log(f"Selected {len(selected_images)} images from range {actual_start} to {actual_end}")
        return selected_images, actual_start, actual_end

    def _get_annotations_for_image(self, image_id):
        """Get all annotations for a specific image ID."""
        return [ann for ann in self.coco_data['annotations'] if ann['image_id'] == image_id]

    def _process_original_image(self, img_info):
        """
        Process an original image and add it to the new dataset.
        Keeps only ONE annotation per image.

        Args:
            img_info: Image information dictionary from COCO data

        Returns:
            bool: Whether the image was successfully processed
        """
        image_id = img_info['id']
        image_filename = img_info['file_name']

        # Check if the image file exists
        image_path = self.images_dir / image_filename
        if not image_path.exists():
            self.log(f"WARNING: Image {image_path} does not exist, skipping...", level='warning')
            return False

        # Get annotations for this image
        annotations = self._get_annotations_for_image(image_id)

        # *** MODIFY HERE ***
        # Keep only one annotation if there are multiple
        if len(annotations) > 1:
            annotations = [annotations[0]]
        # *** END MODIFICATION ***

        if not annotations:
            self.log(f"WARNING: Image {image_filename} has no annotations", level='warning')
            if not self.preserve_original_images:
                return False

        # If we're preserving original images, copy it to the output directory
        if self.preserve_original_images:
            try:
                dst_path = self.output_images_dir / image_filename
                shutil.copy2(image_path, dst_path)

                # Add the original image to the new COCO data
                self.new_coco_data['images'].append(img_info.copy())

                # Add the original annotations to the new COCO data
                for ann in annotations:
                    self.new_coco_data['annotations'].append(ann.copy())

                self.stats["original_images"] += 1
                return True

            except Exception as e:
                self.log(f"ERROR: Could not copy {image_path}: {str(e)}", level='error')
                return False

        return True  # Successfully processed even if we didn't copy the image

    def _create_augmentations(self, image, img_info, annotations):
        """
        Create augmented versions of a single image and its annotations.
        Keeps only ONE annotation per image.

        Args:
            image: Original image as numpy array
            img_info: Original image information dictionary
            annotations: List of annotation dictionaries for the original image

        Returns:
            int: Number of successful augmentations created
        """
        image_filename = img_info['file_name']

        # *** MODIFY HERE ***
        # Select only one annotation (the first one, or you could implement logic to choose the best)
        if len(annotations) > 0:
            # Use only the first annotation
            selected_annotation = annotations[0]
            bboxes = [selected_annotation['bbox']]
            category_ids = [selected_annotation['category_id']]
        else:
            bboxes = []
            category_ids = []
        # *** END MODIFICATION ***

        # Skip augmentation if there are no annotations
        if not bboxes:
            self.log(f"Skipping augmentation for {image_filename} - no annotations")
            return 0

        # Counter for successful augmentations
        successful_augmentations = 0

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
                    self.log(f"WARNING: Augmentation {image_filename} (aug_{aug_idx + 1}) lost all bounding boxes",
                             level='warning')
                    self.stats["failed_augmentations"] += 1
                    continue

                # Generate new filename for augmented image
                base_name = Path(image_filename).stem
                ext = Path(image_filename).suffix
                new_filename = f"{base_name}_aug_{aug_idx + 1}{ext}"

                # Save augmented image
                output_path = self.output_images_dir / new_filename
                try:
                    cv2.imwrite(str(output_path), transformed['image'])
                    self.log(f"Saved augmented image: {output_path}")
                except Exception as e:
                    self.log(f"ERROR: Could not save augmented image {output_path}: {str(e)}", level='error')
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
                self.new_coco_data['images'].append(new_image)

                # Create new annotations for the augmented image
                # *** MODIFY HERE ***
                # Now we should only have one bbox and one category_id
                if transformed['bboxes'] and len(transformed['bboxes']) > 0:
                    bbox = transformed['bboxes'][0]
                    cat_id = transformed['category_ids'][0]

                    # Ensure all bbox values are positive
                    bbox = [max(0, val) for val in bbox]

                    # Ensure width and height are positive
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        self.log(f"WARNING: Skipped bbox with width/height <= 0: {bbox}", level='warning')
                        continue

                    # Create new annotation
                    new_ann = {
                        'id': self.next_ann_id,
                        'image_id': self.next_image_id,
                        'category_id': cat_id,
                        'bbox': list(map(float, bbox)),
                        'area': float(bbox[2] * bbox[3]),
                        'iscrowd': 0
                    }

                    # Add segmentation if present in original annotations
                    if annotations and 'segmentation' in annotations[0]:
                        new_ann['segmentation'] = []  # Empty segmentation for now

                    self.new_coco_data['annotations'].append(new_ann)
                    self.next_ann_id += 1
                    self.stats["annotations_created"] += 1
                # *** END MODIFICATION ***

                # Update counters for next image
                self.next_image_id += 1
                successful_augmentations += 1
                self.stats["successful_augmentations"] += 1

            except Exception as e:
                self.log(f"ERROR: Failed to augment {image_filename} (aug_{aug_idx}): {str(e)}", level='error')
                self.stats["failed_augmentations"] += 1
                continue

        return successful_augmentations

    def log(self, message, level='info'):
        """Log a message with a specified level if verbose mode is enabled."""
        if not self.verbose:
            return

        if level == 'error':
            print(f"ERROR: {message}")
        elif level == 'warning':
            print(f"WARNING: {message}")
        else:
            print(message)

    def augment_dataset(self):
        """
        Augment the dataset and create a new COCO dataset.

        Returns:
            str: Path to the output JSON file
        """
        self.log("=" * 70)
        self.log(f"Starting dataset augmentation ({self.augmentation_strength} strength)")
        self.log(f"Processing images from index {self.start_idx} to {self.end_idx or 'end'}")
        self.log("=" * 70)

        # Get range of images to process
        selected_images, actual_start, actual_end = self._get_image_range()

        # Process each image in the selected range
        for img_info in tqdm(selected_images, desc="Processing images"):
            image_id = img_info['id']
            image_filename = img_info['file_name']

            # Process original image
            if not self._process_original_image(img_info):
                self.stats["skipped_images"] += 1
                continue

            # Get annotations for this image
            annotations = self._get_annotations_for_image(image_id)

            # Skip augmentation if there are no annotations
            if not annotations:
                continue

            # Load image for augmentation
            image_path = self.images_dir / image_filename
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    self.log(f"WARNING: Cannot read image {image_path}, skipping augmentation...", level='warning')
                    self.stats["skipped_images"] += 1
                    continue
            except Exception as e:
                self.log(f"ERROR: Error loading image {image_path}: {str(e)}", level='error')
                self.stats["skipped_images"] += 1
                continue

            # Create augmentations for this image
            augmented_count = self._create_augmentations(image, img_info, annotations)
            self.stats["augmented_images"] += 1 if augmented_count > 0 else 0
            self.stats["processed_images"] += 1

            # Log progress every 10 images
            if self.stats["processed_images"] % 10 == 0:
                self.log(f"Processed {self.stats['processed_images']} images, "
                         f"created {self.stats['successful_augmentations']} augmentations")

        # Save the new COCO dataset
        output_json_path = self.output_dir / f"augmented_coco_{actual_start}-{actual_end}_{self.timestamp}.json"
        try:
            with open(output_json_path, 'w') as f:
                json.dump(self.new_coco_data, f, indent=2)
            self.log(f"Saved annotations to {output_json_path}")
        except Exception as e:
            self.log(f"ERROR: Could not save annotations: {str(e)}", level='error')
            raise

        # Print summary statistics
        self._print_summary()

        return str(output_json_path)

    def _print_summary(self):
        """Print a summary of the augmentation process."""
        self.log("\n" + "=" * 70)
        self.log("AUGMENTATION SUMMARY")
        self.log("=" * 70)
        self.log(f"Images processed: {self.stats['processed_images']}")
        self.log(f"Images skipped: {self.stats['skipped_images']}")
        self.log(f"Original images included: {self.stats['original_images']}")
        self.log(f"Images augmented: {self.stats['augmented_images']}")
        self.log(f"Successful augmentations: {self.stats['successful_augmentations']}")
        self.log(f"Failed augmentations: {self.stats['failed_augmentations']}")
        self.log(f"New annotations created: {self.stats['annotations_created']}")
        self.log(f"Total images in output dataset: {len(self.new_coco_data['images'])}")
        self.log(f"Total annotations in output dataset: {len(self.new_coco_data['annotations'])}")
        self.log(f"Output directory: {self.output_dir}")


def main():
    """Main function to run the augmentation process."""
    try:
        print("COCO Range Augmenter - Hardcoded Parameters Version")
        print("=" * 70)

        # Create and run the augmenter
        augmenter = COCORangeAugmenter()

        # Display parameter values
        print(f"JSON file: {augmenter.json_path}")
        print(f"Images directory: {augmenter.images_dir}")
        print(f"Output directory: {augmenter.output_dir}")
        print(f"Processing image range: {augmenter.start_idx} to {augmenter.end_idx or 'end'}")
        print(f"Augmentations per image: {augmenter.augmentations_per_image}")
        print(f"Augmentation strength: {augmenter.augmentation_strength}")
        print(f"Include original images: {augmenter.preserve_original_images}")
        print("=" * 70)

        # Execute the augmentation
        output_json = augmenter.augment_dataset()

        print(f"\nAugmentation completed successfully!")
        print(f"Output JSON: {output_json}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())