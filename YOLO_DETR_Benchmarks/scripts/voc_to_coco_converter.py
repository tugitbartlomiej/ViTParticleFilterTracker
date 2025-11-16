"""
VOC to COCO Converter for CaDTD Dataset
Converts Pascal VOC XML annotations to COCO JSON format
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


class VOCToCOCOConverter:
    def __init__(self, voc_labels_dir, images_dir, output_json_path, merge_classes=True):
        """
        Args:
            voc_labels_dir: Path to VOC XML labels
            images_dir: Path to PNG frame images
            output_json_path: Output COCO JSON path
            merge_classes: If True, merge all tool classes into single "surgical_tool" class
        """
        self.voc_labels_dir = Path(voc_labels_dir)
        self.images_dir = Path(images_dir)
        self.output_json_path = Path(output_json_path)
        self.merge_classes = merge_classes

        # COCO structure
        self.coco = {
            "info": {
                "description": "CaDTD Dataset - Setup 2 (Tool Heads)",
                "url": "https://github.com/surgical-vision/CaDTD",
                "version": "1.0",
                "year": 2021,
                "contributor": "Jiang et al., MICCAI 2021",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.image_filename_to_id = {}

    def create_categories(self):
        """Create COCO categories"""
        if self.merge_classes:
            # Single category: all tools
            self.coco["categories"] = [
                {
                    "id": 1,
                    "name": "surgical_tool",
                    "supercategory": "tool"
                }
            ]
        else:
            # Multiple categories (CaDIS tool classes)
            # Based on CaDIS annotations
            tool_names = {
                1: "Bipolar_forceps",
                4: "Irrigation",
                5: "Needle_holder",
                6: "Forceps",
                7: "Scissors",
                8: "Suction",
                9: "Hydrodissection_cannula",
                11: "Phacoemulsifier_handpiece",
                12: "Capsulorhexis_forceps"
            }

            for class_id, class_name in tool_names.items():
                self.coco["categories"].append({
                    "id": class_id,
                    "name": class_name,
                    "supercategory": "surgical_tool"
                })

    def parse_voc_xml(self, xml_path):
        """Parse single VOC XML file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Image info
        filename = root.find('filename').text if root.find('filename') is not None else xml_path.stem + '.png'
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Objects (bounding boxes)
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')

            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # VOC uses 1-indexed, COCO uses 0-indexed
            # But we'll keep absolute coordinates

            objects.append({
                'class_name': name,
                'bbox': [xmin, ymin, xmax, ymax]  # [x1, y1, x2, y2]
            })

        return {
            'filename': filename,
            'width': width,
            'height': height,
            'objects': objects
        }

    def convert(self):
        """Convert all VOC XML to COCO JSON"""
        print(f"\n{'='*60}")
        print("VOC to COCO Conversion")
        print(f"{'='*60}")

        # Create categories
        self.create_categories()
        print(f"\nCategories: {len(self.coco['categories'])}")
        for cat in self.coco['categories']:
            print(f"  - {cat['id']}: {cat['name']}")

        # Get all XML files
        xml_files = sorted(list(self.voc_labels_dir.glob("*.xml")))
        print(f"\nFound {len(xml_files)} XML annotation files")

        # Process each XML
        total_annotations = 0

        for xml_path in tqdm(xml_files, desc="Converting VOC → COCO"):
            # Parse XML
            parsed = self.parse_voc_xml(xml_path)

            # Add image
            # Match PNG filename from Labels directory
            # XML: Video1_frame000090.xml → PNG: Video1_frame000090.png
            png_filename = xml_path.stem + '.png'
            png_path = self.images_dir / png_filename

            if not png_path.exists():
                print(f"⚠️  Warning: Image not found: {png_path}")
                continue

            image_id = self.image_id_counter
            self.image_filename_to_id[png_filename] = image_id

            self.coco["images"].append({
                "id": image_id,
                "file_name": png_filename,
                "width": parsed['width'],
                "height": parsed['height'],
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })

            # Add annotations
            for obj in parsed['objects']:
                x1, y1, x2, y2 = obj['bbox']

                # COCO bbox format: [x, y, width, height]
                bbox_coco = [x1, y1, x2 - x1, y2 - y1]
                area = (x2 - x1) * (y2 - y1)

                # Category ID
                if self.merge_classes:
                    category_id = 1  # All tools → surgical_tool
                else:
                    category_id = int(obj['class_name'])

                self.coco["annotations"].append({
                    "id": self.annotation_id_counter,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox_coco,
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0
                })

                self.annotation_id_counter += 1
                total_annotations += 1

            self.image_id_counter += 1

        # Save COCO JSON
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_json_path, 'w') as f:
            json.dump(self.coco, f, indent=2)

        print(f"\nConversion Complete!")
        print(f"   Images: {len(self.coco['images'])}")
        print(f"   Annotations: {total_annotations}")
        print(f"   Categories: {len(self.coco['categories'])}")
        print(f"   Output: {self.output_json_path}")

        return self.output_json_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert VOC XML to COCO JSON')
    parser.add_argument('--voc-labels', required=True, help='Path to VOC Labels directory')
    parser.add_argument('--images', required=True, help='Path to Images directory')
    parser.add_argument('--output', required=True, help='Output COCO JSON path')
    parser.add_argument('--keep-classes', action='store_true',
                       help='Keep original tool classes instead of merging to single class')

    args = parser.parse_args()

    converter = VOCToCOCOConverter(
        voc_labels_dir=args.voc_labels,
        images_dir=args.images,
        output_json_path=args.output,
        merge_classes=not args.keep_classes
    )

    converter.convert()


if __name__ == "__main__":
    main()
