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
bbox = None
tracker = None

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0  # Initialize current frame index

# Folder do zapisywania zannotowanych obrazów
output_dir = "output"
annotated_images_dir = os.path.join(output_dir, "Annotated_Images")
if not os.path.exists(annotated_images_dir):
    os.makedirs(annotated_images_dir)

# Instructions for the user
print("Press 'S' or Right Arrow to move forward one frame.")
print("Press 'A' or Left Arrow to move backward one frame.")
print("Press Space to select ROI and adjust annotation.")
print("Press 'T' to start/stop tracking.")
print("Press 'Q' to quit.")


# Function to save annotated frames
def save_annotated_frame(frame, frame_id):
    """Save the annotated frame to the Annotated Images directory."""
    image_filename = f"frame_{frame_id:06d}.jpg"
    image_path = os.path.join(annotated_images_dir, image_filename)
    cv2.imwrite(image_path, frame)


while True:
    # Set the video frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print("End of video reached or cannot read the frame.")
        break

    frame_id = current_frame + 1  # Frame IDs start from 1

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
    if existing_annotation:
        bbox = existing_annotation['bbox']
    else:
        bbox = None

    if tracking:
        # Update the tracker to get the new bounding box
        success, bbox = tracker.update(frame)

        # If tracking is successful, draw the bounding box and save to COCO
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            # Save the annotated frame
            save_annotated_frame(frame, frame_id)

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
                    "area": bbox[2] * bbox[3],  # width * height
                    "iscrowd": 0
                })
        else:
            # If tracking fails, display failure message and stop tracking
            cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Tracking failed. Stopping tracking.")
            tracking = False

    else:
        # If not tracking but annotation exists, draw it
        if bbox is not None:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # Display the frame
    cv2.imshow("Tool Tracker", frame)

    if tracking:
        # Automatically proceed to next frame after a short delay
        key = cv2.waitKey(30) & 0xFF  # Adjust delay as needed
        current_frame += 1
        if current_frame >= total_frames:
            print("End of video reached during tracking.")
            break
    else:
        # Wait indefinitely for a key press when not tracking
        key = cv2.waitKey(0) & 0xFF

    # Handle key presses
    if key == ord('q'):
        break
    elif key == ord('s') or key == 83:  # 'S' or Right Arrow key
        # Move forward one frame
        if current_frame < total_frames - 1:
            current_frame += 1
        else:
            print("Already at the last frame.")
    elif key == ord('a') or key == 81:  # 'A' or Left Arrow key
        # Move backward one frame
        if current_frame > 0:
            current_frame -= 1
        else:
            print("Already at the first frame.")
    elif key == ord(' '):
        # Select ROI and adjust annotation
        print("Select the region of interest (ROI)...")
        bbox = cv2.selectROI("Tool Tracker", frame, False)
        if bbox != (0, 0, 0, 0):
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
                    "area": bbox[2] * bbox[3],  # width * height
                    "iscrowd": 0
                })
            # Update tracker with new ROI if tracking is active
            if tracking:
                tracker = cv2.TrackerMIL_create()  # Poprawna inicjalizacja trackera
                tracker.init(frame, bbox)
        else:
            print("ROI selection canceled.")
    elif key == ord('t'):
        # Toggle tracking
        tracking = not tracking
        if tracking:
            if bbox is not None:
                tracker = cv2.TrackerMIL_create()  # Poprawna inicjalizacja trackera
                tracker.init(frame, bbox)
                print("Tracking started.")
            else:
                print("No ROI selected. Please select ROI first.")
                tracking = False
        else:
            print("Tracking stopped.")
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
