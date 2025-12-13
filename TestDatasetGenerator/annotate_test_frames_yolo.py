"""
Manual Annotation Tool for Test Dataset - YOLO Format
======================================================
Based on opencv_annotation_tracker_glob_images_yolo.py
Saves annotations in YOLOv8 format (.txt files with normalized coordinates).

Controls:
- Left mouse button + drag: Draw bounding box
- Right mouse button: Clear annotation for current frame
- N: Next image (with tracking)
- S: Skip to next image (without tracking)
- A: Previous image
- T: Toggle continuous mode
- Space: Pause/Resume
- Q: Quit

Output format (YOLOv8):
  class_id x_center y_center width height
  (all values normalized 0-1)

Author: Adapted for TestDatasetGenerator
Date: 2025-12-13
"""

import cv2
import os
import re
import json
from glob import glob

# =============================================================================
# CONFIGURATION - PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: test frames extracted from videos
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "output", "test_frames")

# Existing COCO annotations (will be loaded if exists)
COCO_ANNOTATIONS_FILE = os.path.join(SCRIPT_DIR, "output", "test_annotations_coco.json")

# Output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "yolo_annotations")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "labels")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_sort_key(filename):
    """
    Extract sort key from filename.
    Handles formats like:
    - train05_frame_0001234.jpg
    - test01_frame_0000746.jpg
    """
    basename = os.path.basename(filename)
    match = re.match(r'(.+?)_frame_(\d+)\.jpg', basename, re.IGNORECASE)
    if match:
        video_name = match.group(1)
        frame_num = int(match.group(2))
        return (video_name, frame_num)
    return (basename, 0)


def get_label_filename(image_filename):
    """Get corresponding label filename for an image."""
    basename = os.path.basename(image_filename)
    name_without_ext = os.path.splitext(basename)[0]
    return f"{name_without_ext}.txt"


def load_coco_annotations(coco_file):
    """
    Load existing COCO annotations and convert to filename -> bbox mapping.
    Returns dict: {filename: (x, y, width, height)}
    """
    if not os.path.exists(coco_file):
        return {}

    try:
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        # Build image_id -> filename mapping
        id_to_filename = {}
        for img in coco_data.get("images", []):
            id_to_filename[img["id"]] = img["file_name"]

        # Build filename -> bbox mapping
        filename_to_bbox = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in id_to_filename:
                filename = id_to_filename[img_id]
                bbox = ann["bbox"]  # [x, y, width, height] in COCO format
                filename_to_bbox[filename] = tuple(bbox)

        print(f"Loaded {len(filename_to_bbox)} annotations from COCO file")
        return filename_to_bbox

    except Exception as e:
        print(f"Error loading COCO annotations: {e}")
        return {}


# Global: preloaded COCO annotations
COCO_ANNOTATIONS = {}

# =============================================================================
# LOAD IMAGES
# =============================================================================

print(f"Looking for images in: {IMAGE_FOLDER}")

image_files = sorted(
    glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob(os.path.join(IMAGE_FOLDER, "*.png")),
    key=extract_sort_key
)

if not image_files:
    print(f"ERROR: No images found in {IMAGE_FOLDER}")
    print("Please run extract_random_frames.py first!")
    exit(1)

print(f"Found {len(image_files)} images to annotate")

# Create output directories
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load existing COCO annotations if available
COCO_ANNOTATIONS = load_coco_annotations(COCO_ANNOTATIONS_FILE)
if COCO_ANNOTATIONS:
    print(f"Will use existing COCO annotations as starting point")

# =============================================================================
# STATE VARIABLES
# =============================================================================

tracking = False
continuous_mode = False
paused = False
bbox = None
tracker = None

drawing = False
ix, iy = -1, -1
rectangle = None

current_image_index = 0
total_images = len(image_files)

frame = None
frame_display = None
current_filename = None

# =============================================================================
# INSTRUCTIONS
# =============================================================================

print("\n" + "=" * 60)
print("YOLO ANNOTATION TOOL")
print("=" * 60)
print("Controls:")
print("  Left mouse + drag : Draw bounding box")
print("  Right mouse       : Clear annotation")
print("  N                 : Next image (with tracking)")
print("  S                 : Skip to next (no tracking)")
print("  A                 : Previous image")
print("  T                 : Toggle continuous mode")
print("  Space             : Pause/Resume")
print("  Q                 : Quit")
print("")
print("Output format: YOLOv8 (.txt files)")
print("=" * 60 + "\n")

# =============================================================================
# ANNOTATION FUNCTIONS
# =============================================================================

def save_raw_image(frame, filename):
    """Save the raw image to the images directory."""
    basename = os.path.basename(filename)
    output_path = os.path.join(IMAGES_DIR, basename)
    cv2.imwrite(output_path, frame)
    print(f"Image saved: {output_path}")


def save_yolo_annotation(filename, bbox, image_shape):
    """Save the bounding box annotation in YOLOv8 format."""
    # YOLOv8 expects normalized coordinates: class_id x_center y_center width height
    image_height, image_width = image_shape[:2]

    x_center = (bbox[0] + bbox[2] / 2) / image_width
    y_center = (bbox[1] + bbox[3] / 2) / image_height
    width = bbox[2] / image_width
    height = bbox[3] / image_height

    # Ensure values are between 0 and 1
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    class_id = 0  # Single class: tool

    annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    label_filename = get_label_filename(filename)
    annotation_path = os.path.join(ANNOTATIONS_DIR, label_filename)

    with open(annotation_path, 'w') as f:
        f.write(annotation_line)

    print(f"Annotation saved: {annotation_path}")


def load_yolo_annotation(filename, image_shape):
    """Load existing YOLO annotation if exists, fallback to COCO annotations."""
    label_filename = get_label_filename(filename)
    annotation_path = os.path.join(ANNOTATIONS_DIR, label_filename)
    basename = os.path.basename(filename)

    # First try YOLO format
    if os.path.exists(annotation_path):
        try:
            with open(annotation_path, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    image_height, image_width = image_shape[:2]

                    bbox_width = width * image_width
                    bbox_height = height * image_height
                    bbox_x = (x_center * image_width) - (bbox_width / 2)
                    bbox_y = (y_center * image_height) - (bbox_height / 2)

                    return (bbox_x, bbox_y, bbox_width, bbox_height)
        except Exception as e:
            print(f"Error loading YOLO annotation: {e}")

    # Fallback to COCO annotations
    if basename in COCO_ANNOTATIONS:
        bbox = COCO_ANNOTATIONS[basename]
        return bbox

    return None


def remove_yolo_annotation(filename):
    """Remove the YOLOv8 annotation file and copied image."""
    label_filename = get_label_filename(filename)
    annotation_path = os.path.join(ANNOTATIONS_DIR, label_filename)

    if os.path.exists(annotation_path):
        os.remove(annotation_path)
        print(f"Annotation removed: {annotation_path}")
    else:
        print(f"No annotation found for {filename}")

    # Remove copied image
    basename = os.path.basename(filename)
    image_path = os.path.join(IMAGES_DIR, basename)
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Image removed: {image_path}")


def count_annotations():
    """Count total annotations."""
    return len(glob(os.path.join(ANNOTATIONS_DIR, "*.txt")))


# =============================================================================
# MOUSE CALLBACK
# =============================================================================

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle, bbox, tracking, tracker
    global frame_display, frame, current_filename

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rectangle = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rectangle = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangle = (ix, iy, x, y)
        x0, y0 = min(ix, x), min(iy, y)
        x1, y1 = max(ix, x), max(iy, y)
        bbox = (x0, y0, x1 - x0, y1 - y0)

        # Initialize tracker
        tracker = cv2.TrackerMIL_create()
        tracker.init(frame, bbox)
        tracking = True

        # Save annotation and image
        save_yolo_annotation(current_filename, bbox, frame.shape)
        save_raw_image(frame, current_filename)

        # Update display
        frame_display = frame.copy()
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2)
        add_info_overlay(frame_display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove annotation
        remove_yolo_annotation(current_filename)

        # Clear display
        frame_display = frame.copy()
        add_info_overlay(frame_display)
        cv2.imshow("YOLO Annotation Tool", frame_display)

        # Clear state
        bbox = None
        tracking = False
        rectangle = None


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def add_info_overlay(display_frame):
    """Add information overlay to frame."""
    basename = os.path.basename(current_filename)

    # Frame counter
    text = f"[{current_image_index + 1}/{total_images}] {basename}"
    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Annotation status
    label_filename = get_label_filename(current_filename)
    annotation_path = os.path.join(ANNOTATIONS_DIR, label_filename)

    if os.path.exists(annotation_path):
        cv2.putText(display_frame, "ANNOTATED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "NOT ANNOTATED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Total count
    total_annotated = count_annotations()
    cv2.putText(display_frame, f"Total: {total_annotated}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# =============================================================================
# MAIN LOOP
# =============================================================================

cv2.namedWindow("YOLO Annotation Tool")
cv2.setMouseCallback("YOLO Annotation Tool", draw_rectangle)

while True:
    # Bounds check
    if current_image_index < 0:
        current_image_index = 0
    if current_image_index >= total_images:
        print("End of image sequence reached.")
        break

    # Load current image
    img_path = image_files[current_image_index]
    current_filename = img_path
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"ERROR: Cannot load {img_path}")
        current_image_index += 1
        continue

    frame_display = frame.copy()

    # Check for existing annotation
    existing_bbox = load_yolo_annotation(current_filename, frame.shape)
    if existing_bbox is not None:
        bbox = existing_bbox
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2)
    else:
        bbox = None

    # Handle tracking
    if tracking and not paused:
        success, new_bbox = tracker.update(frame)
        if success:
            bbox = new_bbox
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

            frame_display = frame.copy()
            cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2)

            # Save annotation and image
            save_yolo_annotation(current_filename, bbox, frame.shape)
            save_raw_image(frame, current_filename)
        else:
            cv2.putText(frame_display, "TRACKING FAILED", (100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            tracking = False
            bbox = None

    # Draw rectangle while drawing
    if drawing and rectangle is not None:
        x0, y0, x1, y1 = rectangle
        frame_display = frame.copy()
        cv2.rectangle(frame_display, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Add overlay
    add_info_overlay(frame_display)

    # Display
    cv2.imshow("YOLO Annotation Tool", frame_display)

    # Handle input
    if continuous_mode and not paused:
        key = cv2.waitKey(30) & 0xFF
        current_image_index += 1
        if current_image_index >= total_images:
            print("End of image sequence reached.")
            break
    else:
        key = cv2.waitKey(1) & 0xFF

    # Key handling
    if key == ord('q') or key == 27:  # Q or Escape
        break
    elif key == ord('n'):  # Next with tracking
        if bbox is not None and not tracking:
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame, bbox)
            tracking = True
        current_image_index += 1
    elif key == ord('s'):  # Skip
        tracking = False
        current_image_index += 1
    elif key == ord('a'):  # Previous
        if current_image_index > 0:
            tracking = False
            current_image_index -= 1
        else:
            print("Already at first image.")
    elif key == ord(' '):  # Pause/Resume
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
    elif key == ord('t'):  # Toggle continuous
        continuous_mode = not continuous_mode
        if continuous_mode:
            if bbox is not None and not tracking:
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, bbox)
            tracking = True
            paused = False
            print("Continuous mode ON")
        else:
            tracking = False
            print("Continuous mode OFF")

cv2.destroyAllWindows()

# =============================================================================
# SUMMARY
# =============================================================================

total_annotated = count_annotations()
print("\n" + "=" * 60)
print("ANNOTATION SESSION COMPLETE")
print("=" * 60)
print(f"Total annotations: {total_annotated}")
print(f"Labels saved to: {ANNOTATIONS_DIR}")
print(f"Images saved to: {IMAGES_DIR}")
print("=" * 60)

# Create dataset.yaml for YOLOv8 training
dataset_yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(dataset_yaml_path, 'w') as f:
    f.write(f"""# YOLOv8 Dataset Configuration
# Generated by annotate_test_frames_yolo.py

path: {OUTPUT_DIR}
train: images
val: images

names:
  0: tool

nc: 1
""")
print(f"Dataset config: {dataset_yaml_path}")

# =============================================================================
# GENERATE COCO FORMAT OUTPUT
# =============================================================================

def generate_coco_output():
    """Generate COCO format annotations from YOLO labels."""
    from datetime import datetime

    coco_output = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Test Dataset - Surgical Tool Annotations (YOLO converted)",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "tool", "supercategory": "none"}
        ]
    }

    label_files = glob(os.path.join(ANNOTATIONS_DIR, "*.txt"))
    annotation_id = 1

    for label_path in sorted(label_files):
        label_basename = os.path.basename(label_path)
        image_basename = os.path.splitext(label_basename)[0] + ".jpg"
        image_path = os.path.join(IMAGES_DIR, image_basename)

        if not os.path.exists(image_path):
            # Try original test_frames folder
            image_path = os.path.join(IMAGE_FOLDER, image_basename)

        if not os.path.exists(image_path):
            continue

        # Get image dimensions
        img = cv2.imread(image_path)
        if img is None:
            continue

        img_height, img_width = img.shape[:2]
        image_id = len(coco_output["images"]) + 1

        # Add image entry
        coco_output["images"].append({
            "id": image_id,
            "file_name": image_basename,
            "width": img_width,
            "height": img_height
        })

        # Read YOLO annotation
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)

                        # Convert YOLO to COCO format
                        bbox_width = width * img_width
                        bbox_height = height * img_height
                        bbox_x = (x_center * img_width) - (bbox_width / 2)
                        bbox_y = (y_center * img_height) - (bbox_height / 2)

                        coco_output["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id),
                            "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0
                        })
                        annotation_id += 1
        except Exception as e:
            print(f"Error reading {label_path}: {e}")

    return coco_output

# Generate and save COCO format
coco_output = generate_coco_output()
coco_output_path = os.path.join(OUTPUT_DIR, "annotations_coco.json")
with open(coco_output_path, 'w') as f:
    json.dump(coco_output, f, indent=2)

print(f"COCO annotations: {coco_output_path}")
print(f"  - Images: {len(coco_output['images'])}")
print(f"  - Annotations: {len(coco_output['annotations'])}")
