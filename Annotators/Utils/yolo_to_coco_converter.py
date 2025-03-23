import json
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm


class YOLOtoCOCOConverter:
    def __init__(
            self,
            images_dir: str,
            labels_dir: str,
            output_file: str
    ):
        """
        Initialize the YOLOv8 to COCO format converter.

        Args:
            images_dir: Directory containing the images
            labels_dir: Directory containing YOLO format labels
            output_file: Path to save the COCO format JSON file
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_file = Path(output_file)

        # Initialize counters for generating sequential IDs
        self.image_id_counter = 0
        self.annotation_id_counter = 0

        # Dictionary to store mapping between filename and image_id
        self.filename_to_image_id = {}

        # Dictionary to store class mappings
        self.class_map = {}

        # Set to track processed annotations to avoid duplicates
        self.processed_annotations = {}

        # Initialize COCO format structure
        self.coco_format = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "Converted from YOLOv8 format",
                "contributor": "YOLOtoCOCOConverter",
                "date_created": datetime.now().strftime("%Y-%m-%d")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": "Unknown"
                }
            ],
            "categories": [
                {
                    "id": 0,
                    "name": "tool",
                    "supercategory": "none"
                }
            ],
            "images": [],
            "annotations": []
        }

    def convert_bbox_yolo_to_coco(
            self,
            yolo_bbox: list,
            img_width: int,
            img_height: int
    ) -> list:
        """
        Convert YOLO bbox format (x_center, y_center, width, height) to COCO format (x, y, width, height).
        All values in YOLO format are normalized between 0 and 1.
        """
        x_center, y_center, width, height = yolo_bbox

        # Convert normalized values to absolute pixel values
        x = (x_center - width / 2) * img_width
        y = (y_center - height / 2) * img_height
        w = width * img_width
        h = height * img_height

        # Round to 2 decimal places
        return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]

    def process_image(self, image_file: Path) -> None:
        """Process a single image and its corresponding label file."""
        try:
            # Generate a sequential image ID instead of extracting from filename
            self.image_id_counter += 1
            image_id = self.image_id_counter

            # Store mapping between filename and image_id
            self.filename_to_image_id[image_file.name] = image_id

            # Read image dimensions
            with Image.open(image_file) as img:
                img_width, img_height = img.size

            # Add image info to COCO format
            self.coco_format["images"].append({
                "id": image_id,
                "file_name": image_file.name,
                "width": img_width,
                "height": img_height,
                "license": 1
            })

            # Initialize set to track annotations for this image
            if image_id not in self.processed_annotations:
                self.processed_annotations[image_id] = set()

            # Process corresponding label file
            label_file = self.labels_dir / (image_file.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Warning: Invalid format in {label_file}, line: {line.strip()}")
                            continue

                        class_id, x_center, y_center, width, height = map(float, parts)

                        # Create a unique key for this annotation to detect duplicates
                        annotation_key = f"{class_id}_{x_center:.5f}_{y_center:.5f}_{width:.5f}_{height:.5f}"

                        # Skip if this annotation has already been processed for this image
                        if annotation_key in self.processed_annotations[image_id]:
                            print(f"Warning: Duplicate annotation found in {label_file}: {annotation_key}")
                            continue

                        # Add to processed set
                        self.processed_annotations[image_id].add(annotation_key)

                        # Map class ID
                        class_id_int = int(class_id)
                        if class_id_int not in self.class_map:
                            # Add new category if not seen before
                            new_cat_id = len(self.class_map)
                            self.class_map[class_id_int] = new_cat_id

                            # Only add new category if it doesn't already exist
                            if not any(cat["id"] == new_cat_id for cat in self.coco_format["categories"]):
                                self.coco_format["categories"].append({
                                    "id": new_cat_id,
                                    "name": f"tool_{class_id_int}" if class_id_int != 0 else "tool",
                                    "supercategory": "none"
                                })

                        # Get mapped category ID
                        category_id = self.class_map[class_id_int]

                        # Convert YOLO bbox to COCO format
                        bbox = self.convert_bbox_yolo_to_coco(
                            [float(x_center), float(y_center), float(width), float(height)],
                            img_width,
                            img_height
                        )

                        # Validate bbox values
                        if bbox[2] <= 0 or bbox[3] <= 0:
                            print(f"Warning: Invalid bbox dimensions in {label_file}: {bbox}")
                            continue

                        # Calculate area
                        area = bbox[2] * bbox[3]

                        # Add annotation to COCO format
                        self.annotation_id_counter += 1
                        self.coco_format["annotations"].append({
                            "id": self.annotation_id_counter,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0
                        })

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()

    def validate_and_clean(self):
        """Validate the COCO format data and clean up any issues."""
        print("Validating and cleaning COCO data...")

        # Remove categories with no annotations
        used_category_ids = set(ann["category_id"] for ann in self.coco_format["annotations"])
        self.coco_format["categories"] = [
            cat for cat in self.coco_format["categories"]
            if cat["id"] in used_category_ids
        ]

        # Ensure all annotations reference valid images
        valid_image_ids = set(img["id"] for img in self.coco_format["images"])
        self.coco_format["annotations"] = [
            ann for ann in self.coco_format["annotations"]
            if ann["image_id"] in valid_image_ids
        ]

        # Check for images with no annotations
        images_with_annotations = set(ann["image_id"] for ann in self.coco_format["annotations"])
        images_without_annotations = [
            img["file_name"] for img in self.coco_format["images"]
            if img["id"] not in images_with_annotations
        ]

        if images_without_annotations:
            print(f"Warning: {len(images_without_annotations)} images have no annotations")
            print(f"First 5 examples: {images_without_annotations[:5]}")

    def convert(self) -> None:
        """Convert the entire dataset from YOLO to COCO format."""
        print("\nStarting conversion from YOLO to COCO format...")

        # Validate input directories
        if not self.images_dir.exists():
            raise ValueError(f"Images directory does not exist: {self.images_dir}")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory does not exist: {self.labels_dir}")

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(f"*{ext}")))

        # Sort image files alphabetically
        image_files = sorted(image_files)

        if not image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"Found {len(image_files)} images to process")

        # Process each image
        for image_file in tqdm(image_files, desc="Converting", unit="image"):
            self.process_image(image_file)

        # Validate and clean up the data
        self.validate_and_clean()

        # Save COCO format JSON
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.coco_format, f, indent=2)

        print(f"\nConversion completed. Results:")
        print(f"Total images processed: {len(self.coco_format['images'])}")
        print(f"Total annotations: {len(self.coco_format['annotations'])}")
        print(f"Total categories: {len(self.coco_format['categories'])}")
        for cat in self.coco_format['categories']:
            cat_anns = [a for a in self.coco_format['annotations'] if a['category_id'] == cat['id']]
            print(f"  - {cat['name']} (id: {cat['id']}): {len(cat_anns)} annotations")
        print(f"COCO format annotations saved to: {self.output_file}")


def main():
    # Paths to directories
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train"
    labels_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/labels/train"
    output_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Detr/coco_annotations_from_yolo_dataset_20250218.json"

    # Verify paths
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Output file: {output_file}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Initialize and run converter
    converter = YOLOtoCOCOConverter(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_file=output_file
    )

    try:
        converter.convert()
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()