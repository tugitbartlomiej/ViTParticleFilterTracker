"""
Review and Correct Annotations (OpenCV)
========================================
Loads COCO JSON with DETR detections, allows manual review and correction.
Saves corrected annotations back to COCO JSON.

Usage:
    py -3.11 review_annotations.py
    py -3.11 review_annotations.py --input detr_detections_coco.json --output corrected_coco.json

Controls:
    Left mouse + drag : Draw/correct bounding box
    Right mouse       : Delete annotation for current image
    Enter / Y         : Accept current annotation, go next
    N                 : Next image (keeps annotation as-is)
    S                 : Skip - mark as "no detection" and go next
    A                 : Previous image
    D                 : Delete annotation and go next
    Space             : Toggle show/hide detection info
    Q                 : Quit and save

Author: TestDatasetGenerator
Date: 2025-12-13
"""

import os
import sys
import json
import argparse
from pathlib import Path
from glob import glob
from datetime import datetime

import cv2

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent

DEFAULT_CONFIG = {
    'input_json': SCRIPT_DIR / "output" / "detr_detections_coco.json",
    'output_json': SCRIPT_DIR / "output" / "annotations_reviewed_coco.json",
    'images_dir': SCRIPT_DIR / "output" / "test_frames",
}

# =============================================================================
# COCO DATA HANDLER
# =============================================================================

class COCOAnnotationReviewer:
    def __init__(self, coco_json_path, images_dir, output_path):
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir)
        self.output_path = Path(output_path)

        # Load COCO data
        print(f"Loading annotations from: {coco_json_path}")
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Build indexes
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        # Sort images by filename
        self.sorted_images = sorted(self.coco_data['images'], key=lambda x: x['file_name'])

        # Track modifications
        self.modified = set()  # Set of modified image_ids
        self.deleted = set()   # Set of image_ids with deleted annotations
        self.next_annotation_id = max([ann['id'] for ann in self.coco_data['annotations']], default=0) + 1

        print(f"Loaded {len(self.sorted_images)} images")
        print(f"Loaded {len(self.coco_data['annotations'])} annotations")

    def get_image_path(self, image_info):
        """Get full path to image."""
        return self.images_dir / image_info['file_name']

    def get_annotation(self, image_id):
        """Get annotation for image (first one if multiple)."""
        anns = self.image_id_to_annotations.get(image_id, [])
        return anns[0] if anns else None

    def set_annotation(self, image_id, bbox):
        """Set/update annotation for image."""
        existing = self.get_annotation(image_id)

        if existing:
            # Update existing
            existing['bbox'] = list(bbox)
            existing['area'] = bbox[2] * bbox[3]
            existing.pop('score', None)  # Remove auto-score, now manually verified
            existing['manually_reviewed'] = True
        else:
            # Create new
            new_ann = {
                'id': self.next_annotation_id,
                'image_id': image_id,
                'category_id': 0,
                'bbox': list(bbox),
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
                'manually_reviewed': True
            }
            self.coco_data['annotations'].append(new_ann)
            self.image_id_to_annotations[image_id] = [new_ann]
            self.next_annotation_id += 1

        self.modified.add(image_id)
        if image_id in self.deleted:
            self.deleted.remove(image_id)

    def delete_annotation(self, image_id):
        """Delete annotation for image."""
        self.coco_data['annotations'] = [
            ann for ann in self.coco_data['annotations']
            if ann['image_id'] != image_id
        ]
        self.image_id_to_annotations[image_id] = []
        self.deleted.add(image_id)
        if image_id in self.modified:
            self.modified.remove(image_id)

    def save(self):
        """Save modified COCO data."""
        # Update info
        self.coco_data['info']['last_reviewed'] = datetime.now().isoformat()
        self.coco_data['info']['review_stats'] = {
            'modified_count': len(self.modified),
            'deleted_count': len(self.deleted)
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)

        print(f"Saved to: {self.output_path}")

    def get_stats(self):
        """Get current statistics."""
        total = len(self.sorted_images)
        with_annotations = sum(1 for img in self.sorted_images
                              if self.image_id_to_annotations.get(img['id']))
        return {
            'total_images': total,
            'with_annotations': with_annotations,
            'modified': len(self.modified),
            'deleted': len(self.deleted)
        }


# =============================================================================
# OPENCV REVIEW UI
# =============================================================================

class AnnotationReviewerUI:
    def __init__(self, reviewer: COCOAnnotationReviewer, start_index: int = 0):
        self.reviewer = reviewer
        self.current_index = start_index
        self.total_images = len(reviewer.sorted_images)

        # Drawing state
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.rectangle = None
        self.current_bbox = None

        # Display state
        self.show_info = True
        self.frame = None
        self.frame_display = None

        # Window
        self.window_name = "Annotation Review"

    def load_current_image(self):
        """Load current image and annotation."""
        image_info = self.reviewer.sorted_images[self.current_index]
        image_path = self.reviewer.get_image_path(image_info)

        self.frame = cv2.imread(str(image_path))
        if self.frame is None:
            print(f"ERROR: Cannot load {image_path}")
            return False

        self.current_image_info = image_info
        self.current_image_id = image_info['id']

        # Get existing annotation
        ann = self.reviewer.get_annotation(self.current_image_id)
        self.current_bbox = tuple(ann['bbox']) if ann else None
        self.current_score = ann.get('score') if ann else None
        self.is_reviewed = ann.get('manually_reviewed', False) if ann else False

        return True

    def draw_display(self):
        """Draw current frame with overlays."""
        self.frame_display = self.frame.copy()

        # Draw bbox if exists
        if self.current_bbox:
            x, y, w, h = [int(v) for v in self.current_bbox]
            color = (0, 255, 0) if self.is_reviewed else (0, 255, 255)  # Green if reviewed, Yellow if auto
            cv2.rectangle(self.frame_display, (x, y), (x + w, y + h), color, 2)

            # Label
            label = "Reviewed" if self.is_reviewed else f"DETR {self.current_score*100:.0f}%" if self.current_score else "Auto"
            cv2.putText(self.frame_display, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw rectangle being drawn
        if self.drawing and self.rectangle:
            x0, y0, x1, y1 = self.rectangle
            cv2.rectangle(self.frame_display, (x0, y0), (x1, y1), (0, 0, 255), 2)

        # Info overlay
        if self.show_info:
            self._draw_info_overlay()

    def _draw_info_overlay(self):
        """Draw information overlay."""
        img_info = self.current_image_info
        stats = self.reviewer.get_stats()

        # Background for text
        cv2.rectangle(self.frame_display, (5, 5), (400, 130), (0, 0, 0), -1)
        cv2.rectangle(self.frame_display, (5, 5), (400, 130), (100, 100, 100), 1)

        y = 25
        cv2.putText(self.frame_display, f"[{self.current_index + 1}/{self.total_images}] {img_info['file_name']}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        y += 20
        status = "ANNOTATED" if self.current_bbox else "NO ANNOTATION"
        status_color = (0, 255, 0) if self.current_bbox else (0, 0, 255)
        cv2.putText(self.frame_display, f"Status: {status}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        y += 20
        if self.current_bbox:
            bbox_str = f"BBox: [{self.current_bbox[0]:.0f}, {self.current_bbox[1]:.0f}, {self.current_bbox[2]:.0f}, {self.current_bbox[3]:.0f}]"
            cv2.putText(self.frame_display, bbox_str,
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        y += 20
        cv2.putText(self.frame_display, f"With annotations: {stats['with_annotations']}/{stats['total_images']}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        y += 20
        cv2.putText(self.frame_display, f"Modified: {stats['modified']} | Deleted: {stats['deleted']}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Instructions at bottom
        h = self.frame_display.shape[0]
        cv2.rectangle(self.frame_display, (5, h - 35), (650, h - 5), (0, 0, 0), -1)
        cv2.putText(self.frame_display, "Enter=Accept | N=Next | S=Skip(delete) | A=Prev | D=Delete | Q=Quit",
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.rectangle = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.rectangle = (self.ix, self.iy, x, y)
                self.draw_display()
                cv2.imshow(self.window_name, self.frame_display)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.rectangle:
                x0, y0 = min(self.ix, x), min(self.iy, y)
                x1, y1 = max(self.ix, x), max(self.iy, y)
                w, h = x1 - x0, y1 - y0

                if w > 5 and h > 5:  # Minimum size
                    self.current_bbox = (x0, y0, w, h)
                    self.reviewer.set_annotation(self.current_image_id, self.current_bbox)
                    self.is_reviewed = True
                    print(f"  Annotation set: {self.current_bbox}")

            self.rectangle = None
            self.draw_display()
            cv2.imshow(self.window_name, self.frame_display)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete annotation
            self.reviewer.delete_annotation(self.current_image_id)
            self.current_bbox = None
            self.is_reviewed = False
            print(f"  Annotation deleted")
            self.draw_display()
            cv2.imshow(self.window_name, self.frame_display)

    def run(self):
        """Main review loop."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            # Load current image
            if not self.load_current_image():
                self.current_index += 1
                if self.current_index >= self.total_images:
                    break
                continue

            # Draw display
            self.draw_display()
            cv2.imshow(self.window_name, self.frame_display)

            # Handle keys
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q') or key == 27:  # Q or Escape
                break

            elif key == 13 or key == ord('y'):  # Enter or Y - Accept and next
                if self.current_bbox:
                    self.reviewer.set_annotation(self.current_image_id, self.current_bbox)
                self.current_index += 1
                if self.current_index >= self.total_images:
                    print("Reached end of images.")
                    break

            elif key == ord('n'):  # Next (keep as-is)
                self.current_index += 1
                if self.current_index >= self.total_images:
                    print("Reached end of images.")
                    break

            elif key == ord('s'):  # Skip - delete and next
                self.reviewer.delete_annotation(self.current_image_id)
                self.current_index += 1
                if self.current_index >= self.total_images:
                    print("Reached end of images.")
                    break

            elif key == ord('a'):  # Previous
                if self.current_index > 0:
                    self.current_index -= 1
                else:
                    print("Already at first image.")

            elif key == ord('d'):  # Delete annotation
                self.reviewer.delete_annotation(self.current_image_id)
                self.current_bbox = None
                self.is_reviewed = False
                self.draw_display()
                cv2.imshow(self.window_name, self.frame_display)

            elif key == ord(' '):  # Toggle info
                self.show_info = not self.show_info
                self.draw_display()
                cv2.imshow(self.window_name, self.frame_display)

        cv2.destroyAllWindows()

        # Save on exit
        self.reviewer.save()

        # Print final stats
        stats = self.reviewer.get_stats()
        print("\n" + "=" * 60)
        print("REVIEW COMPLETE")
        print("=" * 60)
        print(f"Total images:      {stats['total_images']}")
        print(f"With annotations:  {stats['with_annotations']}")
        print(f"Modified:          {stats['modified']}")
        print(f"Deleted:           {stats['deleted']}")
        print(f"Saved to:          {self.reviewer.output_path}")
        print("=" * 60)
        print("\nNext step: Run coco_to_yolo.py to convert to YOLO format")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Review and correct annotations")
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input COCO JSON with detections')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output COCO JSON (reviewed)')
    parser.add_argument('--images', type=str, default=None,
                       help='Images directory')
    parser.add_argument('--start', '-s', type=int, default=1,
                       help='Start from image number (1-based, default: 1)')
    args = parser.parse_args()

    # Setup paths
    input_json = Path(args.input) if args.input else DEFAULT_CONFIG['input_json']
    output_json = Path(args.output) if args.output else DEFAULT_CONFIG['output_json']
    images_dir = Path(args.images) if args.images else DEFAULT_CONFIG['images_dir']

    print("=" * 60)
    print("ANNOTATION REVIEW TOOL")
    print("=" * 60)
    print(f"Input:  {input_json}")
    print(f"Output: {output_json}")
    print(f"Images: {images_dir}")
    print("=" * 60)

    # Check paths
    if not input_json.exists():
        print(f"ERROR: Input JSON not found: {input_json}")
        print("Run detr_auto_annotate.py first!")
        sys.exit(1)

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    # Create reviewer
    reviewer = COCOAnnotationReviewer(input_json, images_dir, output_json)

    # Calculate start index (convert from 1-based to 0-based)
    start_index = max(0, args.start - 1)
    if start_index >= len(reviewer.sorted_images):
        print(f"ERROR: Start index {args.start} is beyond total images ({len(reviewer.sorted_images)})")
        sys.exit(1)

    if start_index > 0:
        print(f"Starting from image {args.start} (skipping first {start_index})")

    # Run UI
    ui = AnnotationReviewerUI(reviewer, start_index=start_index)
    ui.run()


if __name__ == "__main__":
    main()
