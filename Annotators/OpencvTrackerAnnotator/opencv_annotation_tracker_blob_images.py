import cv2
import json
import os

# Path to the folder containing images
images_dir = "./../DetrAnnotator/output/sorted_frames/0_25/Raw_Frames"

# Path to annotations file
annotations_file = os.path.join("output", "annotations.json")

# Get list of image files in the directory
image_files = [f for f in sorted(os.listdir(images_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Load existing annotations or initialize a new COCO format structure
if os.path.exists(annotations_file):
    try:
        with open(annotations_file, 'r') as f:
            # Attempt to load the JSON file
            coco_data = json.load(f)
            print("Annotations file loaded successfully.")
    except json.JSONDecodeError:
        # Handle the case where the JSON file is empty or invalid
        print("Error: The annotations file is empty or contains invalid JSON. Initializing a new annotations structure.")
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
                {"id": 1, "name": "tool", "supercategory": "none"}
            ]
        }
else:
    # If the file doesn't exist, create a new annotations structure
    print("No annotations file found. Initializing a new annotations structure.")
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
            {"id": 1, "name": "tool", "supercategory": "none"}
        ]
    }

tracking = False
continuous_mode = False
paused = False
bbox = None
tracker = None

# Variables for drawing rectangle
drawing = False
ix, iy = -1, -1  # Initial x and y coordinates
rectangle = None

# Get total number of images
total_images = len(image_files)
current_image_index = 0  # Initialize current image index

# Folders to save images
output_dir = "output"
annotated_images_dir = os.path.join(output_dir, "Annotated_Images")
raw_images_dir = os.path.join(output_dir, "Raw_Images")

if not os.path.exists(annotated_images_dir):
    os.makedirs(annotated_images_dir)
if not os.path.exists(raw_images_dir):
    os.makedirs(raw_images_dir)

# Instructions for the user
print("Use the mouse to draw ROI:")
print("- Left-click and drag to draw the rectangle.")
print("- Release the mouse button to accept the ROI.")
print("Press 'N' to move to the next image with tracking.")
print("Press 'S' to skip to the next image without tracking.")
print("Press 'A' to go back to the previous image.")
print("Press 'T' to toggle continuous mode.")
print("Press 'Space' to pause/resume.")
print("Right-click to clear the ROI.")
print("Press 'Q' to quit.")

# Function to save annotated and raw frames
def save_frames(frame, frame_display, image_filename):
    """Save the raw and annotated frames to their respective directories."""
    # Save raw image
    raw_image_path = os.path.join(raw_images_dir, image_filename)
    cv2.imwrite(raw_image_path, frame)

    # Save annotated image
    annotated_image_path = os.path.join(annotated_images_dir, image_filename)
    cv2.putText(frame_display, image_filename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite(annotated_image_path, frame_display)

# Function to update or add annotations
def update_annotation(coco_data, image_id, bbox):
    """Update or add a new annotation for the current image."""
    # Check if annotation for this image ID already exists
    existing_annotation = next((ann for ann in coco_data["annotations"] if ann['image_id'] == image_id), None)

    if existing_annotation:
        # If exists, update the bounding box and area
        existing_annotation['bbox'] = list(bbox)
        existing_annotation['area'] = bbox[2] * bbox[3]  # width * height
    else:
        # Otherwise, add a new annotation
        # Ensure unique annotation 'id'
        existing_ids = [ann['id'] for ann in coco_data["annotations"]]
        new_id = max(existing_ids) + 1 if existing_ids else 1
        coco_data["annotations"].append({
            "id": new_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": list(bbox),
            "area": bbox[2] * bbox[3],  # width * height
            "iscrowd": 0
        })

def remove_annotation_and_images(coco_data, image_id):
    """Remove the annotation for the specified image_id and associated images from both folders."""
    # Remove annotation
    coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann['image_id'] != image_id]

    # Remove associated images from both Raw_Images and Annotated_Images folders
    raw_image_path = os.path.join(raw_images_dir, f"frame_{image_id:06d}.jpg")
    annotated_image_path = os.path.join(annotated_images_dir, f"frame_{image_id:06d}.jpg")

    if os.path.exists(raw_image_path):
        os.remove(raw_image_path)
        print(f"Raw image for frame {image_id} removed from Raw_Images folder.")

    if os.path.exists(annotated_image_path):
        os.remove(annotated_image_path)
        print(f"Annotated image for frame {image_id} removed from Annotated_Images folder.")

    print(f"Annotation and images for frame {image_id} removed from both folders.")

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle, bbox, tracking, tracker, frame_display, image_id, coco_data

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

        # Initialize tracker with new ROI
        tracker = cv2.TrackerMIL_create()
        tracker.init(frame, bbox)
        tracking = True

        # Update or add bbox to annotations immediately
        update_annotation(coco_data, image_id, bbox)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Check if there's an existing annotation for this image
        existing_annotation = next((ann for ann in coco_data["annotations"] if ann['image_id'] == image_id), None)

        if existing_annotation:
            # Remove the annotation and associated images
            remove_annotation_and_images(coco_data, image_id)

            # Clear the display
            frame_display = frame.copy()
            cv2.putText(frame_display, f"Frame: {image_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Tool Tracker", frame_display)
        else:
            print(f"No annotation found for frame {image_id}.")

        # Clear ROI and stop tracking
        bbox = None
        tracking = False
        rectangle = None

# Create a named window and set the mouse callback
cv2.namedWindow("Tool Tracker")
cv2.setMouseCallback("Tool Tracker", draw_rectangle)

while True:
    # Load the current image
    image_filename = image_files[current_image_index]
    image_path = os.path.join(images_dir, image_filename)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Unable to load image {image_filename}.")
        break

    image_id = current_image_index + 1  # Image IDs start from 1

    # Reset bbox at the beginning of each frame
    bbox = None

    # Copy frames for display and saving
    frame_display = frame.copy()  # For display (with annotations)
    frame_raw = frame.copy()  # For saving raw images

    # Add image information to COCO if not already added
    if not any(img['id'] == image_id for img in coco_data["images"]):
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "height": frame.shape[0],
            "width": frame.shape[1]
        })

    # Check if there is an annotation for this image
    existing_annotation = next((ann for ann in coco_data["annotations"] if ann['image_id'] == image_id), None)

    if tracking and not paused:
        # Update the tracker to get the new bounding box
        success, bbox = tracker.update(frame)
        if success:
            # Draw the bounding box
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

            # Update or add bbox to annotations
            update_annotation(coco_data, image_id, bbox)

            # Save frames with annotations if tracking is enabled and not paused
            save_frames(frame_raw, frame_display, image_filename)
        else:
            cv2.putText(frame_display, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Tracking failed. Stopping tracking.")
            tracking = False
            bbox = None  # Reset bbox if tracking fails

    else:
        # If not tracking, check for existing annotation
        if existing_annotation:
            bbox = existing_annotation['bbox']
        else:
            bbox = None

        if bbox is not None:
            # Draw the bounding box from annotation
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

    # Draw the rectangle while drawing
    if drawing and rectangle is not None:
        x0, y0, x1, y1 = rectangle
        cv2.rectangle(frame_display, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Display the frame number in the top-left corner
    cv2.putText(frame_display, f"Frame: {image_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Tool Tracker", frame_display)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord('n'):  # 'N' key for next image with tracking
        if bbox is not None and not tracking:
            tracker = cv2.TrackerMIL_create()  # Initialize the tracker
            tracker.init(frame, bbox)
            tracking = True

        # After processing the image, remove it from the source folder
        os.remove(image_path)
        print(f"Image {image_filename} removed from source folder.")

        current_image_index += 1
        if current_image_index >= total_images:
            print("End of images reached.")
            break
    elif key == ord('s'):  # 'S' key to skip to next image without tracking
        tracking = False  # Ensure tracking is stopped

        # After skipping the image, remove it from the source folder
        os.remove(image_path)
        print(f"Image {image_filename} removed from source folder.")

        current_image_index += 1
        if current_image_index >= total_images:
            print("Already at the last image.")
            break
    elif key == ord('a'):  # 'A' key to go back to the previous image
        if current_image_index > 0:
            current_image_index -= 1
        else:
            print("Already at the first image.")
    elif key == ord(' '):  # Space to pause/resume
        paused = not paused
        if paused:
            print("Paused.")
        else:
            print("Resumed.")
    elif key == 27:  # Escape key
        break

# Release all resources and close windows
cv2.destroyAllWindows()

# Save the COCO data to a JSON file
with open(annotations_file, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO annotations saved to {annotations_file}")
