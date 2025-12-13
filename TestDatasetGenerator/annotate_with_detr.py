"""
Manual Annotation Tool with DETR Pre-Detection
===============================================
Uses DETR model to automatically detect bounding boxes,
then allows manual correction via OpenCV interface.

Controls:
- Left mouse button + drag: Draw/correct bounding box
- Right mouse button: Clear annotation for current frame
- Enter/Y: Accept DETR detection and go next
- N: Next image (with tracking)
- S: Skip to next image (without tracking, no save)
- A: Previous image
- R: Re-run DETR detection
- T: Toggle continuous mode
- Space: Pause/Resume
- Q: Quit

Output: YOLO format (.txt) + COCO JSON

Author: Adapted for TestDatasetGenerator with DETR
Date: 2025-12-13
"""

import cv2
import os
import re
import json
import warnings
from glob import glob
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
warnings.filterwarnings("ignore", message=".*pass `assign=True`.*")

import torch
import numpy as np
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: test frames
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "output", "test_frames")

# Existing COCO annotations (optional, will be loaded if exists)
COCO_ANNOTATIONS_FILE = os.path.join(SCRIPT_DIR, "output", "test_annotations_coco.json")

# Output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "yolo_annotations")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "labels")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

# DETR Model Configuration
DETR_CHECKPOINT = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Checkpoints/DETR/DETR_Checkpoints/checkpoint_epoch_170.pth"
DETR_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for auto-detection
DETR_QUERY_ID = 81  # Best query for tooltip detection

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DETR MODEL LOADING
# =============================================================================

print("=" * 60)
print("DETR-ASSISTED ANNOTATION TOOL")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Loading DETR model from: {os.path.basename(DETR_CHECKPOINT)}")

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection

    # Load processor
    DETR_PROCESSOR = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load model
    DETR_MODEL = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    # Load checkpoint
    checkpoint = torch.load(DETR_CHECKPOINT, map_location=DEVICE, weights_only=False)
    DETR_MODEL.load_state_dict(checkpoint['model_state_dict'], strict=False)
    DETR_MODEL.to(DEVICE)
    DETR_MODEL.eval()

    print("DETR model loaded successfully!")
    DETR_AVAILABLE = True

except Exception as e:
    print(f"WARNING: Could not load DETR model: {e}")
    print("Continuing without DETR pre-detection...")
    DETR_AVAILABLE = False
    DETR_MODEL = None
    DETR_PROCESSOR = None

print("=" * 60)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_sort_key(filename):
    """Extract sort key from filename."""
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
    """Load existing COCO annotations."""
    if not os.path.exists(coco_file):
        return {}
    try:
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        id_to_filename = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}
        filename_to_bbox = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id in id_to_filename:
                filename = id_to_filename[img_id]
                filename_to_bbox[filename] = tuple(ann["bbox"])
        print(f"Loaded {len(filename_to_bbox)} existing COCO annotations")
        return filename_to_bbox
    except Exception as e:
        print(f"Error loading COCO: {e}")
        return {}


# =============================================================================
# DETR INFERENCE
# =============================================================================

def run_detr_detection(image_path):
    """
    Run DETR inference on an image.
    Returns: (bbox, confidence) or (None, 0) if no detection
    bbox format: (x, y, width, height)
    """
    if not DETR_AVAILABLE:
        return None, 0

    try:
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        inputs = DETR_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = DETR_MODEL(**inputs)

        # Get outputs
        logits = outputs.logits[0]  # [100, num_classes+1]
        boxes = outputs.pred_boxes[0]  # [100, 4] normalized cxcywh

        # Method 1: Use specific query (Q81)
        query_logits = logits[DETR_QUERY_ID]
        probs = torch.softmax(query_logits, dim=-1)
        class_prob = probs[0].item()  # Probability for class 0 (tool)

        if class_prob >= DETR_CONFIDENCE_THRESHOLD:
            # Convert box from cxcywh normalized to xywh pixels
            cx, cy, w, h = boxes[DETR_QUERY_ID].tolist()
            x = (cx - w/2) * img_width
            y = (cy - h/2) * img_height
            w = w * img_width
            h = h * img_height

            # Clamp to image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = min(w, img_width - x)
            h = min(h, img_height - y)

            return (x, y, w, h), class_prob

        # Method 2: Fallback - find best detection above threshold
        probs_all = torch.softmax(logits, dim=-1)
        class_probs = probs_all[:, 0]  # All queries, class 0
        best_idx = class_probs.argmax().item()
        best_prob = class_probs[best_idx].item()

        if best_prob >= DETR_CONFIDENCE_THRESHOLD:
            cx, cy, w, h = boxes[best_idx].tolist()
            x = (cx - w/2) * img_width
            y = (cy - h/2) * img_height
            w = w * img_width
            h = h * img_height

            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            w = min(w, img_width - x)
            h = min(h, img_height - y)

            return (x, y, w, h), best_prob

        return None, 0

    except Exception as e:
        print(f"DETR inference error: {e}")
        return None, 0


# =============================================================================
# LOAD IMAGES
# =============================================================================

print(f"\nLooking for images in: {IMAGE_FOLDER}")

image_files = sorted(
    glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + glob(os.path.join(IMAGE_FOLDER, "*.png")),
    key=extract_sort_key
)

if not image_files:
    print(f"ERROR: No images found in {IMAGE_FOLDER}")
    exit(1)

print(f"Found {len(image_files)} images to annotate")

# Create output directories
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load existing annotations
COCO_ANNOTATIONS = load_coco_annotations(COCO_ANNOTATIONS_FILE)

# =============================================================================
# STATE VARIABLES
# =============================================================================

tracking = False
continuous_mode = False
paused = False
bbox = None
detr_bbox = None
detr_confidence = 0
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
print("CONTROLS")
print("=" * 60)
print("  Left mouse + drag : Draw/correct bounding box")
print("  Right mouse       : Clear annotation")
print("  Enter/Y           : Accept DETR detection, save & next")
print("  N                 : Next image (with tracking)")
print("  S                 : Skip to next (no save)")
print("  A                 : Previous image")
print("  R                 : Re-run DETR detection")
print("  T                 : Toggle continuous mode")
print("  Space             : Pause/Resume")
print("  Q                 : Quit")
print("")
print("  GREEN box  = DETR detection (auto)")
print("  BLUE box   = Current/saved annotation")
print("  RED box    = Drawing in progress")
print("=" * 60 + "\n")

# =============================================================================
# ANNOTATION FUNCTIONS
# =============================================================================

def save_raw_image(frame, filename):
    """Save the raw image to the images directory."""
    basename = os.path.basename(filename)
    output_path = os.path.join(IMAGES_DIR, basename)
    cv2.imwrite(output_path, frame)


def save_yolo_annotation(filename, bbox, image_shape):
    """Save annotation in YOLOv8 format."""
    image_height, image_width = image_shape[:2]

    x_center = (bbox[0] + bbox[2] / 2) / image_width
    y_center = (bbox[1] + bbox[3] / 2) / image_height
    width = bbox[2] / image_width
    height = bbox[3] / image_height

    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    class_id = 0
    annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    label_filename = get_label_filename(filename)
    annotation_path = os.path.join(ANNOTATIONS_DIR, label_filename)

    with open(annotation_path, 'w') as f:
        f.write(annotation_line)

    print(f"Saved: {label_filename}")


def load_yolo_annotation(filename, image_shape):
    """Load existing YOLO annotation."""
    label_filename = get_label_filename(filename)
    annotation_path = os.path.join(ANNOTATIONS_DIR, label_filename)
    basename = os.path.basename(filename)

    # Try YOLO format first
    if os.path.exists(annotation_path):
        try:
            with open(annotation_path, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) == 5:
                    _, x_center, y_center, width, height = map(float, parts)
                    image_height, image_width = image_shape[:2]

                    bbox_width = width * image_width
                    bbox_height = height * image_height
                    bbox_x = (x_center * image_width) - (bbox_width / 2)
                    bbox_y = (y_center * image_height) - (bbox_height / 2)

                    return (bbox_x, bbox_y, bbox_width, bbox_height)
        except:
            pass

    # Fallback to COCO
    if basename in COCO_ANNOTATIONS:
        return COCO_ANNOTATIONS[basename]

    return None


def remove_yolo_annotation(filename):
    """Remove annotation file."""
    label_filename = get_label_filename(filename)
    annotation_path = os.path.join(ANNOTATIONS_DIR, label_filename)

    if os.path.exists(annotation_path):
        os.remove(annotation_path)
        print(f"Removed: {label_filename}")

    basename = os.path.basename(filename)
    image_path = os.path.join(IMAGES_DIR, basename)
    if os.path.exists(image_path):
        os.remove(image_path)


def count_annotations():
    """Count total annotations."""
    return len(glob(os.path.join(ANNOTATIONS_DIR, "*.txt")))


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def add_info_overlay(display_frame):
    """Add information overlay to frame."""
    basename = os.path.basename(current_filename)

    # Frame counter
    text = f"[{current_image_index + 1}/{total_images}] {basename}"
    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # DETR status
    if detr_bbox is not None:
        detr_text = f"DETR: {detr_confidence*100:.1f}% confidence"
        cv2.putText(display_frame, detr_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "DETR: No detection", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Annotation status
    label_filename = get_label_filename(current_filename)
    annotation_path = os.path.join(ANNOTATIONS_DIR, label_filename)

    if os.path.exists(annotation_path):
        cv2.putText(display_frame, "SAVED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "NOT SAVED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Total count
    total = count_annotations()
    cv2.putText(display_frame, f"Total saved: {total}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Instructions
    cv2.putText(display_frame, "Enter=Accept | N=Next | S=Skip | A=Prev | R=Re-detect | Q=Quit",
                (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def draw_boxes(display_frame):
    """Draw all relevant boxes on the display frame."""
    # Draw DETR detection (GREEN, dashed effect)
    if detr_bbox is not None:
        x, y, w, h = [int(v) for v in detr_bbox]
        # Draw thicker green box for DETR suggestion
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, f"DETR {detr_confidence*100:.0f}%", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw current/saved annotation (BLUE)
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)


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

        # Save immediately
        save_yolo_annotation(current_filename, bbox, frame.shape)
        save_raw_image(frame, current_filename)

        # Update display
        frame_display = frame.copy()
        draw_boxes(frame_display)
        add_info_overlay(frame_display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        remove_yolo_annotation(current_filename)
        bbox = None
        tracking = False
        rectangle = None

        frame_display = frame.copy()
        draw_boxes(frame_display)
        add_info_overlay(frame_display)
        cv2.imshow("DETR Annotation Tool", frame_display)


# =============================================================================
# MAIN LOOP
# =============================================================================

cv2.namedWindow("DETR Annotation Tool")
cv2.setMouseCallback("DETR Annotation Tool", draw_rectangle)

while True:
    # Bounds check
    if current_image_index < 0:
        current_image_index = 0
    if current_image_index >= total_images:
        print("End of images.")
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

    # Run DETR detection
    detr_bbox, detr_confidence = run_detr_detection(current_filename)

    # Check for existing annotation
    existing_bbox = load_yolo_annotation(current_filename, frame.shape)
    if existing_bbox is not None:
        bbox = existing_bbox
    else:
        bbox = None

    # Draw boxes and overlay
    draw_boxes(frame_display)
    add_info_overlay(frame_display)

    # Handle tracking
    if tracking and not paused:
        success, new_bbox = tracker.update(frame)
        if success:
            bbox = new_bbox
            save_yolo_annotation(current_filename, bbox, frame.shape)
            save_raw_image(frame, current_filename)

            frame_display = frame.copy()
            draw_boxes(frame_display)
            add_info_overlay(frame_display)
        else:
            tracking = False
            bbox = None

    # Draw rectangle while drawing
    if drawing and rectangle is not None:
        x0, y0, x1, y1 = rectangle
        frame_display = frame.copy()
        draw_boxes(frame_display)
        cv2.rectangle(frame_display, (x0, y0), (x1, y1), (0, 0, 255), 2)  # RED for drawing
        add_info_overlay(frame_display)

    cv2.imshow("DETR Annotation Tool", frame_display)

    # Handle input
    if continuous_mode and not paused:
        key = cv2.waitKey(30) & 0xFF
        if detr_bbox is not None:
            bbox = detr_bbox
            save_yolo_annotation(current_filename, bbox, frame.shape)
            save_raw_image(frame, current_filename)
        current_image_index += 1
    else:
        key = cv2.waitKey(1) & 0xFF

    # Key handling
    if key == ord('q') or key == 27:  # Q or Escape
        break

    elif key == 13 or key == ord('y'):  # Enter or Y - Accept DETR and go next
        if detr_bbox is not None:
            bbox = detr_bbox
            save_yolo_annotation(current_filename, bbox, frame.shape)
            save_raw_image(frame, current_filename)
            print(f"Accepted DETR detection: {detr_confidence*100:.1f}%")
        elif bbox is not None:
            # Save current bbox if no DETR but manual annotation exists
            save_yolo_annotation(current_filename, bbox, frame.shape)
            save_raw_image(frame, current_filename)
        current_image_index += 1
        tracking = False

    elif key == ord('n'):  # Next with tracking
        if bbox is not None and not tracking:
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame, bbox)
            tracking = True
        current_image_index += 1

    elif key == ord('s'):  # Skip (no save)
        tracking = False
        current_image_index += 1

    elif key == ord('a'):  # Previous
        if current_image_index > 0:
            tracking = False
            current_image_index -= 1
        else:
            print("Already at first image.")

    elif key == ord('r'):  # Re-run DETR
        print("Re-running DETR detection...")
        detr_bbox, detr_confidence = run_detr_detection(current_filename)
        frame_display = frame.copy()
        draw_boxes(frame_display)
        add_info_overlay(frame_display)
        if detr_bbox:
            print(f"DETR: {detr_confidence*100:.1f}% confidence")
        else:
            print("DETR: No detection")

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
            print("Continuous mode ON (auto-accepting DETR)")
        else:
            tracking = False
            print("Continuous mode OFF")

cv2.destroyAllWindows()

# =============================================================================
# GENERATE OUTPUTS
# =============================================================================

total_annotated = count_annotations()
print("\n" + "=" * 60)
print("ANNOTATION SESSION COMPLETE")
print("=" * 60)
print(f"Total annotations: {total_annotated}")
print(f"Labels: {ANNOTATIONS_DIR}")
print(f"Images: {IMAGES_DIR}")

# Create dataset.yaml
dataset_yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
with open(dataset_yaml_path, 'w') as f:
    f.write(f"""# YOLOv8 Dataset Configuration
path: {OUTPUT_DIR}
train: images
val: images

names:
  0: tool

nc: 1
""")
print(f"Dataset config: {dataset_yaml_path}")

# Generate COCO format
def generate_coco_output():
    from datetime import datetime

    coco_output = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Test Dataset - DETR-assisted annotations",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "tool", "supercategory": "none"}]
    }

    label_files = glob(os.path.join(ANNOTATIONS_DIR, "*.txt"))
    annotation_id = 1

    for label_path in sorted(label_files):
        label_basename = os.path.basename(label_path)
        image_basename = os.path.splitext(label_basename)[0] + ".jpg"
        image_path = os.path.join(IMAGES_DIR, image_basename)

        if not os.path.exists(image_path):
            image_path = os.path.join(IMAGE_FOLDER, image_basename)

        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        if img is None:
            continue

        img_height, img_width = img.shape[:2]
        image_id = len(coco_output["images"]) + 1

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_basename,
            "width": img_width,
            "height": img_height
        })

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x_center, y_center, width, height = map(float, parts)

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
        except:
            pass

    return coco_output

coco_output = generate_coco_output()
coco_output_path = os.path.join(OUTPUT_DIR, "annotations_coco.json")
with open(coco_output_path, 'w') as f:
    json.dump(coco_output, f, indent=2)

print(f"COCO annotations: {coco_output_path}")
print(f"  - Images: {len(coco_output['images'])}")
print(f"  - Annotations: {len(coco_output['annotations'])}")
print("=" * 60)
