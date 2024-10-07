import cv2
import json
import os

# Initialize video capture with your specified path
cap = cv2.VideoCapture('E:/Cataract/videos/micro/train01.mp4')

# Prepare to save bounding box data in COCO format
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

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0  # Initialize current frame index

# Folders to save images
output_dir = "output"
annotated_images_dir = os.path.join(output_dir, "Annotated_Images")
raw_images_dir = os.path.join(output_dir, "Raw_Images")
frames_dir = os.path.join(output_dir, "Frames")  # Folder for saving raw frames

if not os.path.exists(annotated_images_dir):
    os.makedirs(annotated_images_dir)
if not os.path.exists(raw_images_dir):
    os.makedirs(raw_images_dir)
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Instructions for the user
print("Use the mouse to draw ROI:")
print("- Left-click and drag to draw the rectangle.")
print("- Release the mouse button to accept the ROI.")
print("Press 'N' to move to the next frame with tracking.")
print("Press 'S' to skip to the next frame without tracking.")
print("Press 'A' to go back to the previous frame.")
print("Press 'T' to toggle continuous mode.")
print("Press 'Space' to pause/resume.")
print("Right-click to clear the ROI.")
print("Press 'Q' to quit.")

# Function to save raw frame
def save_raw_frame(frame, frame_id):
    """Save the raw frame to the Frames directory."""
    frame_filename = f"frame_{frame_id:06d}.jpg"
    frame_path = os.path.join(frames_dir, frame_filename)
    cv2.imwrite(frame_path, frame)

# Function to save annotated and raw frames
def save_frames(frame, frame_display, frame_id):
    """Save the raw and annotated frames to their respective directories."""
    # Save raw image
    raw_image_filename = f"frame_{frame_id:06d}.jpg"
    raw_image_path = os.path.join(raw_images_dir, raw_image_filename)
    cv2.imwrite(raw_image_path, frame)

    # Save annotated image
    annotated_image_filename = f"frame_{frame_id:06d}.jpg"
    annotated_image_path = os.path.join(annotated_images_dir, annotated_image_filename)
    cv2.imwrite(annotated_image_path, frame_display)

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle, bbox, tracking, tracker, frame_display

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
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Clear ROI
        bbox = None
        tracking = False
        rectangle = None
        print("ROI cleared. Tracking stopped.")

# Create a named window and set the mouse callback
cv2.namedWindow("Tool Tracker")
cv2.setMouseCallback("Tool Tracker", draw_rectangle)

while True:
    # Set the video frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or cannot read the frame.")
        break

    frame_id = current_frame + 1  # Frame IDs start from 1

    # Copy frames for display and saving
    frame_display = frame.copy()  # For display (with annotations)
    frame_raw = frame.copy()      # For saving raw images

    # Add frame information to COCO if not already added
    if not any(img['id'] == frame_id for img in coco_data["images"]):
        coco_data["images"].append({
            "id": frame_id,
            "file_name": f"frame_{frame_id:06d}.jpg",
            "height": frame.shape[0],
            "width": frame.shape[1]
        })

    # Check if there is an annotation for this frame
    existing_annotation = next((ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id), None)

    if tracking and not paused:
        # Update the tracker to get the new bounding box
        success, bbox = tracker.update(frame)
        if success:
            # Draw the bounding box
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

            # Update or add bbox to annotations
            if existing_annotation:
                existing_annotation['bbox'] = list(bbox)
                existing_annotation['area'] = bbox[2] * bbox[3]
            else:
                coco_data["annotations"].append({
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": frame_id,
                    "category_id": 1,
                    "bbox": list(bbox),
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
        else:
            cv2.putText(frame_display, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Tracking failed. Stopping tracking.")
            tracking = False

    else:
        # If not tracking but annotation exists, draw it
        if existing_annotation and bbox is not None:
            bbox = existing_annotation['bbox']
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

    # Draw the rectangle while drawing
    if drawing and rectangle is not None:
        x0, y0, x1, y1 = rectangle
        cv2.rectangle(frame_display, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Display the frame number in the top-left corner
    cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Tool Tracker", frame_display)

    # Save every frame (raw frame without annotations) in the "Frames" folder
    save_raw_frame(frame_raw, frame_id)

    # Save frames with annotations if tracking is enabled and not paused
    if tracking and not paused:
        save_frames(frame_raw, frame_display, frame_id)

    if continuous_mode and not paused:
        key = cv2.waitKey(30) & 0xFF  # Automatically refresh during continuous mode
        current_frame += 1
        if current_frame >= total_frames:
            print("End of video reached.")
            break
    else:
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press or mouse event

    # Handle key presses
    if key == ord('q'):
        break
    elif key == ord('n'):  # 'N' key for next frame with tracking
        if bbox is not None and not tracking:
            tracker = cv2.TrackerMIL_create()  # Initialize the tracker
            tracker.init(frame, bbox)
            tracking = True
        current_frame += 1
        if current_frame >= total_frames:
            print("End of video reached.")
            break
    elif key == ord('s'):  # 'S' key to skip to next frame without tracking
        current_frame += 1
        if current_frame >= total_frames:
            print("Already at the last frame.")
            break
    elif key == ord('a'):  # 'A' key to go back to the previous frame
        if current_frame > 0:
            current_frame -= 1
        else:
            print("Already at the first frame.")
    elif key == ord(' '):  # Space to pause/resume
        paused = not paused
        if paused:
            print("Paused.")
        else:
            print("Resumed.")
    elif key == ord('t'):
        # Toggle continuous mode
        continuous_mode = not continuous_mode
        if continuous_mode:
            if bbox is not None and not tracking:
                tracker = cv2.TrackerMIL_create()  # Initialize the tracker for continuous mode
                tracker.init(frame, bbox)
            tracking = True
            paused = False
            print("Continuous mode started.")
        else:
            tracking = False
            print("Continuous mode stopped.")

    elif key == 27:  # Escape key
        break

    # Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

# Save the COCO data to a JSON file
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
coco_output_file = os.path.join(output_dir, "annotations.json")
with open(coco_output_file, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO annotations saved to {coco_output_file}")
