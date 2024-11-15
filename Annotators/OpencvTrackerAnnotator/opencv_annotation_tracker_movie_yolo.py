import cv2
import json
import os
from ultralytics import YOLO  # Import YOLO model
import warnings
import numpy as np

# Temporarily suppress FutureWarning (if needed)
warnings.filterwarnings("ignore", category=FutureWarning)

# Path to the video file
video_path = 'E:/Cataract/videos/micro/train01.mp4'  # Provide your video file path

# Path to the YOLO model
yolo_model_path = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/surgical_tool_detection/exp11/weights/best.pt'  # Provide your YOLO model path

frame_id = None

# Initialize the YOLO model
try:
    model = YOLO(yolo_model_path)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Initialize video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video file.")
    exit()

# Path to the annotations file
annotations_file = os.path.join("output/Yolo", "annotations.json")

# Path to the YOLO annotations directory
yolo_annotations_dir = os.path.join("output/Yolo", "Yolo_Annotations")
os.makedirs(yolo_annotations_dir, exist_ok=True)

# Load existing annotations or initialize new in COCO format
if os.path.exists(annotations_file):
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
            print("Annotations file successfully loaded.")
    except json.JSONDecodeError:
        print("Error: Annotations file is empty or contains invalid JSON. Initializing new annotation structure.")
        coco_data = {
            "info": {
                "year": 2024,
                "version": "1.0",
                "description": "Tool Tracking Data",
                "date_created": "2024-09-25"
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "tool", "supercategory": "none"}  # Changed category_id to 0 for YOLO compatibility
            ]
        }
else:
    print("Annotations file not found. Initializing new annotation structure.")
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "Tool Tracking Data",
            "date_created": "2024-09-25"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "tool", "supercategory": "none"}  # Changed category_id to 0 for YOLO compatibility
        ]
    }

# Variables for drawing rectangle
drawing = False
ix, iy = -1, -1  # Initial x and y coordinates
rectangle = None

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = -1  # Initialize current frame index to -1

# Folders to save images
output_dir = "output/Yolo"
annotated_images_dir = os.path.join(output_dir, "Annotated_Images")
raw_images_dir = os.path.join(output_dir, "Raw_Images")

os.makedirs(annotated_images_dir, exist_ok=True)
os.makedirs(raw_images_dir, exist_ok=True)

# User instructions
print("Instructions:")
print("- Press 'N' to go to the next frame and perform tool detection using YOLO.")
print("- Use the mouse to manually select an area (left mouse button).")
print("- Right-click to remove existing bounding boxes for the current frame.")
print("- Press 'S' to skip the current frame without detection.")
print("- Press 'A' to go back to the previous frame.")
print("- Press 'K' to save all annotations and raw images.")
print("- Press 'Q' or 'ESC' to quit.")

# Function to save annotations to JSON file
def save_annotations(coco_data, annotations_file):
    try:
        with open(annotations_file, 'w') as f:
            json.dump(coco_data, f, indent=4)
        print(f"Annotations saved to {annotations_file}")
    except TypeError as e:
        print(f"Error saving annotations: {e}")

# Function to save annotations in YOLO format
def save_yolo_annotations(coco_data, yolo_annotations_dir, frame_id):
    """
    Saves the annotations for a specific frame in YOLO format.
    """
    # Get image dimensions
    image_info = next((img for img in coco_data["images"] if img["id"] == frame_id), None)
    if not image_info:
        print(f"No image info found for frame {frame_id}. Skipping YOLO annotation.")
        return

    img_width = image_info["width"]
    img_height = image_info["height"]

    # Get annotations for the frame
    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == frame_id]
    if not annotations:
        # If no annotations, remove existing YOLO annotation file if exists
        yolo_file_path = os.path.join(yolo_annotations_dir, f"frame_{frame_id}.txt")
        if os.path.exists(yolo_file_path):
            os.remove(yolo_file_path)
            print(f"Removed existing YOLO annotation file for frame {frame_id}.")
        return

    yolo_lines = []
    for ann in annotations:
        class_id = ann["category_id"]
        x_min, y_min, width, height = ann["bbox"]

        # Convert to YOLO format
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        # Ensure values are between 0 and 1
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        norm_width = min(max(norm_width, 0), 1)
        norm_height = min(max(norm_height, 0), 1)

        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        yolo_lines.append(yolo_line)

    # Write to YOLO annotation file
    yolo_file_path = os.path.join(yolo_annotations_dir, f"frame_{frame_id}.txt")
    try:
        with open(yolo_file_path, 'w') as f:
            for line in yolo_lines:
                f.write(line + "\n")
        print(f"YOLO annotations saved to {yolo_file_path}")
    except Exception as e:
        print(f"Error saving YOLO annotations for frame {frame_id}: {e}")

# Modified update_annotations function
def update_annotations(coco_data, frame_id, bboxes, frame_shape, source='manual', remove_existing=True):
    """Update annotations for the current frame based on the given bounding boxes."""
    image_height, image_width = frame_shape[:2]

    # Add image info if not exists
    if not any(img['id'] == frame_id for img in coco_data["images"]):
        coco_data["images"].append({
            "id": frame_id,
            "file_name": f"frame_{frame_id}.jpg",
            "height": image_height,
            "width": image_width
        })

    if remove_existing:
        # Remove existing annotations of the same source for this frame
        coco_data["annotations"] = [
            ann for ann in coco_data["annotations"]
            if not (ann['image_id'] == frame_id and ann['source'] == source)
        ]

    existing_ids = [ann['id'] for ann in coco_data["annotations"]]
    ann_id = max(existing_ids) + 1 if existing_ids else 0  # Start from 0 if no annotations

    for bbox in bboxes:
        x_min, y_min, width, height = bbox

        # Convert coordinates to native Python float types
        x_min = float(x_min)
        y_min = float(y_min)
        width = float(width)
        height = float(height)
        area = width * height

        coco_data["annotations"].append({
            "id": ann_id,
            "image_id": frame_id,
            "category_id": 0,  # Assuming one category with ID 0 for YOLO compatibility
            "bbox": [x_min, y_min, width, height],
            "area": area,
            "iscrowd": 0,
            "source": source  # Add source of annotation
        })
        ann_id += 1

    # After updating COCO annotations, also update YOLO annotations
    save_yolo_annotations(coco_data, yolo_annotations_dir, frame_id)

# Function to remove annotations and images
def remove_annotation_and_images(coco_data, frame_id):
    """Remove annotations for a specific frame and associated images from both folders."""
    # Remove all annotations for this frame
    coco_data["annotations"] = [
        ann for ann in coco_data["annotations"] if ann['image_id'] != frame_id
    ]

    # Remove associated images from Raw_Images and Annotated_Images folders
    raw_image_path = os.path.join(raw_images_dir, f"frame_{frame_id}.jpg")
    annotated_image_path = os.path.join(annotated_images_dir, f"frame_{frame_id}.jpg")
    yolo_file_path = os.path.join(yolo_annotations_dir, f"frame_{frame_id}.txt")

    if os.path.exists(raw_image_path):
        os.remove(raw_image_path)
        print(f"Raw image for frame {frame_id} removed from Raw_Images folder.")

    if os.path.exists(annotated_image_path):
        os.remove(annotated_image_path)
        print(f"Annotated image for frame {frame_id} removed from Annotated_Images folder.")

    if os.path.exists(yolo_file_path):
        os.remove(yolo_file_path)
        print(f"YOLO annotation file for frame {frame_id} removed from Yolo_Annotations folder.")

    print(f"Annotations and images for frame {frame_id} removed from all folders.")

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, rectangle, frame_display, coco_data, frame_id, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rectangle = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rectangle = (ix, iy, x, y)
            temp_frame = frame_display.copy()
            x0, y0, x1, y1 = rectangle
            cv2.rectangle(temp_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imshow("Tool Tracker", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangle = (ix, iy, x, y)
        x0, y0 = min(ix, x), min(iy, y)
        x1, y1 = max(ix, x), max(iy, y)
        width = x1 - x0
        height = y1 - y0
        bbox = [x0, y0, width, height]

        # Update annotations as 'manual', removing existing 'manual' annotations
        update_annotations(coco_data, frame_id, [bbox], frame_display.shape, source='manual', remove_existing=True)

        # Refresh frame_display and redraw all annotations
        frame_display = frame.copy()
        existing_annotations = [ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id]
        for ann in existing_annotations:
            bbox_ann = ann['bbox']
            x_min, y_min, width_ann, height_ann = bbox_ann
            x0_ann = int(x_min)          # Corrected assignment
            y0_ann = int(y_min)          # Corrected assignment
            x1_ann = int(x_min + width_ann)
            y1_ann = int(y_min + height_ann)
            if ann.get('source') == 'manual':
                color = (0, 0, 255)  # Manual annotations in red
            else:
                color = (255, 0, 0)  # YOLO annotations in blue
            cv2.rectangle(frame_display, (x0_ann, y0_ann), (x1_ann, y1_ann), color, 2)

        cv2.imshow("Tool Tracker", frame_display)

        # Save images
        raw_image_path = os.path.join(raw_images_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(raw_image_path, frame)

        annotated_image_path = os.path.join(annotated_images_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(annotated_image_path, frame_display)

        # Save annotations after manual update
        save_annotations(coco_data, annotations_file)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove all annotations for the current frame
        remove_annotation_and_images(coco_data, frame_id)

        # Refresh frame display without bounding boxes
        frame_display = frame.copy()
        # Display the frame
        cv2.imshow("Tool Tracker", frame_display)

        # Save annotations after deletion
        save_annotations(coco_data, annotations_file)

# Create window and set mouse callback function
cv2.namedWindow("Tool Tracker")
cv2.setMouseCallback("Tool Tracker", mouse_callback)

# Function to load and display a frame
def load_and_display_frame(cap, current_frame, coco_data):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"Cannot read frame {current_frame}.")
        return None, None, None

    frame_id = current_frame + 1

    frame_display = frame.copy()

    # Add image info to COCO if not already added
    if not any(img['id'] == frame_id for img in coco_data["images"]):
        coco_data["images"].append({
            "id": frame_id,
            "file_name": f"frame_{frame_id}.jpg",
            "height": frame.shape[0],
            "width": frame.shape[1]
        })

    # Check if there are annotations for this frame
    existing_annotations = [ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id]

    # If there are annotations, draw them
    if existing_annotations:
        for ann in existing_annotations:
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            p1 = (int(x_min), int(y_min))
            p2 = (int(x_min + width), int(y_min + height))
            if ann.get('source') == 'manual':
                color = (0, 0, 255)  # Manual annotations in red
            else:
                color = (255, 0, 0)  # YOLO annotations in blue
            cv2.rectangle(frame_display, p1, p2, color, 2)

    # Display frame number in the top-left corner
    cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Tool Tracker", frame_display)

    # After loading, save YOLO annotations
    save_yolo_annotations(coco_data, yolo_annotations_dir, frame_id)

    return frame, frame_display, frame_id

# Function to load the next frame
def load_next_frame():
    global current_frame, frame, frame_display, frame_id
    current_frame += 1
    if current_frame >= total_frames:
        print("End of video.")
        return False

    frame_data = load_and_display_frame(cap, current_frame, coco_data)
    if frame_data[0] is None:
        return False

    frame, frame_display, frame_id = frame_data
    return True

# Initially load the first frame
if not load_next_frame():
    cap.release()
    cv2.destroyAllWindows()
    exit()

while True:
    # Read pressed key (do not block the loop)
    key = cv2.waitKey(1) & 0xFF

    # Key press handling
    if key == ord('q') or key == 27:  # 'q' or 'ESC' to quit
        break
    elif key in [ord('n'), ord('N')]:  # 'N' or 'n' to go to the next frame and YOLO detection
        # Go to the next frame
        if not load_next_frame():
            break

        # Perform detection using YOLO
        results = model(frame)
        detections = results[0].boxes  # Get detections from the first result

        if len(detections) > 0:
            bboxes = []
            for det in detections:
                bbox_xyxy = det.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = bbox_xyxy

                # Convert to native Python float types
                x_min = float(x_min)
                y_min = float(y_min)
                x_max = float(x_max)
                y_max = float(y_max)

                width = x_max - x_min
                height = y_max - y_min
                bbox = [x_min, y_min, width, height]
                bboxes.append(bbox)

            # Update annotations with 'source'='yolo', removing existing 'yolo' annotations
            update_annotations(coco_data, frame_id, bboxes, frame.shape, source='yolo', remove_existing=True)

            # Refresh frame_display and redraw all annotations
            frame_display = frame.copy()
            existing_annotations = [ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id]
            for ann in existing_annotations:
                bbox_ann = ann['bbox']
                x_min, y_min, width_ann, height_ann = bbox_ann
                x0_ann = int(x_min)          # Corrected assignment
                y0_ann = int(y_min)          # Corrected assignment
                x1_ann = int(x_min + width_ann)
                y1_ann = int(y_min + height_ann)
                if ann.get('source') == 'manual':
                    color = (0, 0, 255)  # Manual annotations in red
                else:
                    color = (255, 0, 0)  # YOLO annotations in blue
                cv2.rectangle(frame_display, (x0_ann, y0_ann), (x1_ann, y1_ann), color, 2)

            print(f"Detection completed for frame {frame_id}.")

            # Save annotations after YOLO detection
            save_annotations(coco_data, annotations_file)
        else:
            print("No tools detected in the current frame.")

        # Save images
        raw_image_path = os.path.join(raw_images_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(raw_image_path, frame)

        annotated_image_path = os.path.join(annotated_images_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(annotated_image_path, frame_display)

        # Display the updated frame
        cv2.imshow("Tool Tracker", frame_display)

    elif key in [ord('s'), ord('S')]:  # 'S' or 's' to skip the current frame
        # Go to the next frame without detection
        if not load_next_frame():
            break

    elif key in [ord('a'), ord('A')]:  # 'A' or 'a' to go back to the previous frame
        if current_frame > 0:
            current_frame -= 2  # Go back by 2 because load_next_frame() will increment by 1
            if not load_next_frame():
                break
        else:
            print("You are already at the first frame.")

    elif key in [ord('k'), ord('K')]:  # 'K' or 'k' to save all annotations and images
        # Save annotations
        save_annotations(coco_data, annotations_file)

        # Save all YOLO annotations
        for img in coco_data["images"]:
            frame_num = img['id']
            save_yolo_annotations(coco_data, yolo_annotations_dir, frame_num)

        # Save all raw images
        for img in coco_data["images"]:
            frame_num = img['id']
            raw_image_path = os.path.join(raw_images_dir, f"frame_{frame_num}.jpg")
            annotated_image_path = os.path.join(annotated_images_dir, f"frame_{frame_num}.jpg")

            # Check if raw image exists before saving
            if not os.path.exists(raw_image_path):
                # Reconstruct frame filename based on frame_num
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame_to_save = cap.read()
                if ret:
                    cv2.imwrite(raw_image_path, frame_to_save)
                else:
                    print(f"Failed to read frame {frame_num} for saving raw image.")

            # Check if annotated image exists before saving
            if not os.path.exists(annotated_image_path):
                # Reconstruct frame_display based on annotations
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame_to_save = cap.read()
                if ret:
                    frame_display_to_save = frame_to_save.copy()
                    annotations = [ann for ann in coco_data["annotations"] if ann['image_id'] == frame_num]
                    for ann in annotations:
                        bbox = ann['bbox']
                        x_min, y_min, width, height = bbox
                        x0 = int(x_min)
                        y0 = int(y_min)
                        x1 = int(x_min + width)
                        y1 = int(y_min + height)
                        if ann.get('source') == 'manual':
                            color = (0, 0, 255)  # Manual annotations in red
                        else:
                            color = (255, 0, 0)  # YOLO annotations in blue
                        cv2.rectangle(frame_display_to_save, (x0, y0), (x1, y1), color, 2)
                    cv2.imwrite(annotated_image_path, frame_display_to_save)
                else:
                    print(f"Failed to read frame {frame_num} for saving annotated image.")

        print("All annotations and images have been saved.")

    elif key == ord(' '):  # Space to pause/resume
        # Currently, space is not assigned to any action
        print("Space was pressed. No action is currently assigned.")

# Release the VideoCapture object and close windows
cap.release()
cv2.destroyAllWindows()

# Save COCO data to JSON file upon exit
save_annotations(coco_data, annotations_file)

# Save YOLO annotations for all frames upon exit
for img in coco_data["images"]:
    frame_num = img['id']
    save_yolo_annotations(coco_data, yolo_annotations_dir, frame_num)

print("Final annotations saved. Program terminated successfully.")
