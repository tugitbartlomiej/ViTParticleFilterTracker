"""
Manual Annotation Tool for Test Dataset
=======================================
Based on opencv_annotation_tracker_glob_images.py
Adapted for TestDatasetGenerator test frames.

Controls:
- Left mouse button + drag: Draw bounding box
- Right mouse button: Clear annotation for current frame
- N: Next image (with tracking)
- S: Skip to next image (without tracking)
- A: Previous image
- T: Toggle continuous mode
- Space: Pause/Resume
- Q: Quit and save

Author: Adapted for TestDatasetGenerator
Date: 2025-12-13
"""

import cv2
import json
import os
import re
from glob import glob
from datetime import datetime

# =============================================================================
# CONFIGURATION - PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: test frames extracted from videos
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "output", "test_frames")

# Output: annotations file
ANNOTATIONS_FILE = os.path.join(SCRIPT_DIR, "output", "test_annotations_coco.json")

# Output directories for annotated/raw copies (optional)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "annotation_output")
ANNOTATED_IMAGES_DIR = os.path.join(OUTPUT_DIR, "Annotated_Images")
RAW_IMAGES_DIR = os.path.join(OUTPUT_DIR, "Raw_Images")

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
    # Try to extract video name and frame number
    match = re.match(r'(.+?)_frame_(\d+)\.jpg', basename, re.IGNORECASE)
    if match:
        video_name = match.group(1)
        frame_num = int(match.group(2))
        return (video_name, frame_num)
    # Fallback: just use filename
    return (basename, 0)


def get_frame_id_from_filename(filename):
    """Generate unique frame ID from filename."""
    basename = os.path.basename(filename)
    # Use hash of filename for unique ID
    return abs(hash(basename)) % (10**9)


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

# =============================================================================
# LOAD OR INITIALIZE ANNOTATIONS
# =============================================================================

if os.path.exists(ANNOTATIONS_FILE):
    try:
        with open(ANNOTATIONS_FILE, 'r') as f:
            coco_data = json.load(f)
            print(f"Loaded existing annotations from {ANNOTATIONS_FILE}")
            print(f"  - {len(coco_data.get('images', []))} images")
            print(f"  - {len(coco_data.get('annotations', []))} annotations")
    except json.JSONDecodeError:
        print("Warning: Invalid JSON in annotations file. Starting fresh.")
        coco_data = None
else:
    coco_data = None

if coco_data is None:
    coco_data = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Test Dataset - Surgical Tool Annotations",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "tool", "supercategory": "none"}
        ]
    }

# Create mapping from filename to existing data
filename_to_image_id = {}
filename_to_annotation = {}

for img in coco_data.get("images", []):
    filename_to_image_id[img["file_name"]] = img["id"]

for ann in coco_data.get("annotations", []):
    img_id = ann["image_id"]
    # Find corresponding filename
    for img in coco_data["images"]:
        if img["id"] == img_id:
            filename_to_annotation[img["file_name"]] = ann
            break

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

current_frame_index = 0
total_frames = len(image_files)
current_frame_image = None
frame_display = None
current_filename = None

# Create output directories
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)

# =============================================================================
# INSTRUCTIONS
# =============================================================================

print("\n" + "=" * 60)
print("MANUAL ANNOTATION TOOL")
print("=" * 60)
print("Controls:")
print("  Left mouse + drag : Draw bounding box")
print("  Right mouse       : Clear annotation")
print("  N                 : Next image (with tracking)")
print("  S                 : Skip to next (no tracking)")
print("  A                 : Previous image")
print("  T                 : Toggle continuous mode")
print("  Space             : Pause/Resume")
print("  Q                 : Quit and save")
print("=" * 60 + "\n")

# =============================================================================
# ANNOTATION FUNCTIONS
# =============================================================================

def get_or_create_image_entry(filename, frame):
    """Get or create image entry in COCO data."""
    basename = os.path.basename(filename)

    if basename in filename_to_image_id:
        return filename_to_image_id[basename]

    # Create new image entry
    existing_ids = [img['id'] for img in coco_data["images"]]
    new_id = max(existing_ids) + 1 if existing_ids else 1

    coco_data["images"].append({
        "id": new_id,
        "file_name": basename,
        "height": frame.shape[0],
        "width": frame.shape[1]
    })

    filename_to_image_id[basename] = new_id
    return new_id


def update_annotation(image_id, bbox, filename):
    """Update or add annotation for image."""
    basename = os.path.basename(filename)

    # Check for existing annotation
    existing = None
    for ann in coco_data["annotations"]:
        if ann['image_id'] == image_id:
            existing = ann
            break

    if existing:
        existing['bbox'] = list(bbox)
        existing['area'] = bbox[2] * bbox[3]
    else:
        existing_ids = [ann['id'] for ann in coco_data["annotations"]]
        new_id = max(existing_ids) + 1 if existing_ids else 1

        new_ann = {
            "id": new_id,
            "image_id": image_id,
            "category_id": 0,  # tool
            "bbox": list(bbox),
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        }
        coco_data["annotations"].append(new_ann)
        filename_to_annotation[basename] = new_ann

    # Save after each annotation
    save_annotations()


def remove_annotation(image_id, filename):
    """Remove annotation for image."""
    basename = os.path.basename(filename)

    coco_data["annotations"] = [
        ann for ann in coco_data["annotations"]
        if ann['image_id'] != image_id
    ]

    if basename in filename_to_annotation:
        del filename_to_annotation[basename]

    save_annotations()
    print(f"Removed annotation for {basename}")


def save_annotations():
    """Save annotations to file."""
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(coco_data, f, indent=2)


def save_frames(frame, frame_display, filename):
    """Save raw and annotated images."""
    basename = os.path.basename(filename)

    raw_path = os.path.join(RAW_IMAGES_DIR, basename)
    cv2.imwrite(raw_path, frame)

    annotated_path = os.path.join(ANNOTATED_IMAGES_DIR, basename)
    cv2.imwrite(annotated_path, frame_display)


# =============================================================================
# MOUSE CALLBACK
# =============================================================================

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle, bbox, tracking, tracker
    global frame_display, current_frame_image, current_filename

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
        tracker.init(current_frame_image, bbox)
        tracking = True

        # Update annotation
        image_id = get_or_create_image_entry(current_filename, current_frame_image)
        update_annotation(image_id, bbox, current_filename)

        # Update display
        frame_display = current_frame_image.copy()
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2)
        add_info_overlay(frame_display)

        print(f"Annotation saved: bbox={bbox}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove annotation
        basename = os.path.basename(current_filename)
        if basename in filename_to_image_id:
            image_id = filename_to_image_id[basename]
            remove_annotation(image_id, current_filename)

        bbox = None
        tracking = False
        rectangle = None

        frame_display = current_frame_image.copy()
        add_info_overlay(frame_display)
        cv2.imshow("Annotation Tool", frame_display)


# =============================================================================
# FRAME LOADING
# =============================================================================

def add_info_overlay(frame):
    """Add information overlay to frame."""
    basename = os.path.basename(current_filename)

    # Frame info
    text = f"[{current_frame_index + 1}/{total_frames}] {basename}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Annotation status
    if basename in filename_to_annotation:
        cv2.putText(frame, "ANNOTATED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NOT ANNOTATED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def load_frame(index):
    """Load and prepare frame at index."""
    global current_frame_image, frame_display, current_filename, bbox, tracking, tracker, rectangle

    if index < 0 or index >= total_frames:
        return False

    img_path = image_files[index]
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"ERROR: Cannot load image: {img_path}")
        return False

    current_filename = img_path
    current_frame_image = frame.copy()
    frame_display = frame.copy()

    # Check for existing annotation
    basename = os.path.basename(img_path)
    if basename in filename_to_annotation:
        ann = filename_to_annotation[basename]
        bbox = tuple(ann['bbox'])
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2)
    else:
        bbox = None

    add_info_overlay(frame_display)
    cv2.imshow("Annotation Tool", frame_display)

    # Reset tracker
    tracking = False
    tracker = None
    rectangle = None

    return True


# =============================================================================
# MAIN LOOP
# =============================================================================

# Load first frame
if not load_frame(current_frame_index):
    print("Failed to load initial frame.")
    exit(1)

# Create window and set mouse callback
cv2.namedWindow("Annotation Tool")
cv2.setMouseCallback("Annotation Tool", draw_rectangle)

while True:
    # Handle tracking in continuous mode
    if tracking and not paused and continuous_mode:
        success, new_bbox = tracker.update(current_frame_image)
        if success:
            bbox = new_bbox
            frame_display = current_frame_image.copy()
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2)
            add_info_overlay(frame_display)

            # Update annotation
            image_id = get_or_create_image_entry(current_filename, current_frame_image)
            update_annotation(image_id, bbox, current_filename)

            save_frames(current_frame_image, frame_display, current_filename)
        else:
            cv2.putText(frame_display, "TRACKING FAILED", (100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            tracking = False
            bbox = None

    # Draw rectangle while drawing
    if drawing and rectangle is not None:
        frame_display = current_frame_image.copy()
        x0, y0, x1, y1 = rectangle
        cv2.rectangle(frame_display, (x0, y0), (x1, y1), (0, 255, 0), 2)
        add_info_overlay(frame_display)

    cv2.imshow("Annotation Tool", frame_display)

    # Handle continuous mode
    if continuous_mode and not paused:
        key = cv2.waitKey(30) & 0xFF
        current_frame_index += 1
        if current_frame_index >= total_frames:
            print("Reached end of images.")
            break
        load_frame(current_frame_index)
    else:
        key = cv2.waitKey(1) & 0xFF

    # Key handling
    if key == ord('q'):
        break
    elif key == ord('n'):  # Next with tracking
        if bbox is not None and not tracking:
            tracker = cv2.TrackerMIL_create()
            tracker.init(current_frame_image, bbox)
            tracking = True
        current_frame_index += 1
        if current_frame_index >= total_frames:
            print("Reached end of images.")
            break
        load_frame(current_frame_index)
    elif key == ord('s'):  # Skip (no tracking)
        tracking = False
        current_frame_index += 1
        if current_frame_index >= total_frames:
            print("Reached end of images.")
            break
        load_frame(current_frame_index)
    elif key == ord('a'):  # Previous
        if current_frame_index > 0:
            current_frame_index -= 1
            load_frame(current_frame_index)
        else:
            print("Already at first image.")
    elif key == ord(' '):  # Pause/Resume
        paused = not paused
        print("PAUSED" if paused else "RESUMED")
    elif key == ord('t'):  # Toggle continuous mode
        continuous_mode = not continuous_mode
        if continuous_mode:
            if bbox is not None and not tracking:
                tracker = cv2.TrackerMIL_create()
                tracker.init(current_frame_image, bbox)
            tracking = True
            paused = False
            print("Continuous mode ON")
        else:
            tracking = False
            print("Continuous mode OFF")

# Cleanup
cv2.destroyAllWindows()
save_annotations()

print("\n" + "=" * 60)
print("ANNOTATION SESSION COMPLETE")
print("=" * 60)
print(f"Annotations saved to: {ANNOTATIONS_FILE}")
print(f"Total images: {len(coco_data['images'])}")
print(f"Total annotations: {len(coco_data['annotations'])}")
print("=" * 60)
