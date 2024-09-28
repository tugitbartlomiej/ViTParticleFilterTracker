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

# Function to add annotation to COCO
def add_bbox_to_coco(bbox, frame_id):
    coco_data["annotations"].append({
        "id": frame_id,
        "image_id": frame_id,
        "category_id": 1,
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "area": bbox[2] * bbox[3],  # width * height
        "iscrowd": 0
    })

# Instructions for the user
print("Press 'S' to skip to the next frame and pause.")
print("Press 'F' to move forward one frame.")
print("Press 'D' to move backward one frame.")
print("Press Space to select ROI and start/stop tracking or adjust ROI during tracking.")
print("Press 'Q' to quit.")

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

    if tracking:
        # Update the tracker to get the new bounding box
        success, bbox = tracker.update(frame)

        # If tracking is successful, draw the bounding box and save to COCO
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            add_bbox_to_coco(bbox, frame_id)
        else:
            # If tracking fails, display failure message and stop tracking
            cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Tracking failed. Returning to skip mode.")
            tracking = False

    # Display the frame
    cv2.imshow("Tool Tracker", frame)

    # Wait for key press
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # Skip to the next frame and pause
        current_frame += 1
    elif key == ord('f'):
        # Move forward one frame
        if current_frame < total_frames - 1:
            current_frame += 1
        else:
            print("Already at the last frame.")
    elif key == ord('d'):
        # Move backward one frame
        if current_frame > 0:
            current_frame -= 1
            # Remove last annotations if any
            coco_data["images"] = [img for img in coco_data["images"] if img['id'] != frame_id]
            coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann['image_id'] != frame_id]
        else:
            print("Already at the first frame.")
    elif key == ord(' '):
        if tracking:
            # Pause tracking and allow ROI adjustment
            print("Adjust the region of interest (ROI)...")
            bbox = cv2.selectROI("Tool Tracker", frame, False)
            if bbox != (0, 0, 0, 0):
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, bbox)
                add_bbox_to_coco(bbox, frame_id)
            else:
                print("ROI selection canceled. Continuing tracking.")
        else:
            # Select ROI and start tracking
            print("Select the region of interest (ROI)...")
            bbox = cv2.selectROI("Tool Tracker", frame, False)
            if bbox != (0, 0, 0, 0):
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame, bbox)
                tracking = True
                add_bbox_to_coco(bbox, frame_id)
            else:
                print("ROI selection canceled.")
    else:
        # Continue without changing frame
        continue

    # If not tracking, update current frame
    if not tracking:
        current_frame += 1

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

# Save the COCO data to a JSON file
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
coco_output_file = os.path.join(output_dir, "annotations.json")
with open(coco_output_file, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO annotations saved to {coco_output_file}")
