import os
import json
from PIL import Image
from datetime import datetime
from pathlib import Path
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
                    "id": 1,
                    "name": "tool",
                    "supercategory": "none"
                }
            ],
            "images": [],
            "annotations": []
        }

        self.annotation_id = 0

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
            # Extract frame number from filename
            # Assuming filenames are like 'frame_123.jpg'
            frame_number = int(image_file.stem.split('_')[1])

            # Read image dimensions
            with Image.open(image_file) as img:
                img_width, img_height = img.size

            # Add image info to COCO format
            self.coco_format["images"].append({
                "id": frame_number,  # Use frame number as id
                "file_name": image_file.name,
                "width": img_width,
                "height": img_height,
                "license": 1
            })

            # Process corresponding label file
            label_file = self.labels_dir / (image_file.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())

                        # Convert YOLO bbox to COCO format
                        bbox = self.convert_bbox_yolo_to_coco(
                            [x_center, y_center, width, height],
                            img_width,
                            img_height
                        )

                        # Calculate area
                        area = bbox[2] * bbox[3]

                        # Add annotation to COCO format
                        self.coco_format["annotations"].append({
                            "id": self.annotation_id,
                            "image_id": frame_number,  # Use frame number as image_id
                            "category_id": 1,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0
                        })

                        self.annotation_id += 1

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

    def convert(self) -> None:
        """Convert the entire dataset from YOLO to COCO format."""
        print("\nStarting conversion from YOLO to COCO format...")

        # Get all image files
        image_files = sorted(
            [f for f in self.images_dir.glob("*.jpg")],
            key=lambda x: int(x.stem.split('_')[1])
        )

        if not image_files:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"Found {len(image_files)} images to process")

        # Process each image
        for image_file in tqdm(image_files, desc="Converting", unit="image"):
            self.process_image(image_file)

        # Save COCO format JSON
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.coco_format, f, indent=2)

        print(f"\nConversion completed. Results:")
        print(f"Total images processed: {len(self.coco_format['images'])}")
        print(f"Total annotations: {len(self.coco_format['annotations'])}")
        print(f"COCO format annotations saved to: {self.output_file}")


def main():
    # Paths to directories
    images_dir = "output_frames/raw_images/76_100"
    labels_dir = "output_frames/annotations/76_100"
    output_file = "output/coco_annotations_76_100.json"

    converter = YOLOtoCOCOConverter(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_file=output_file
    )

    converter.convert()


if __name__ == "__main__":
    main()
