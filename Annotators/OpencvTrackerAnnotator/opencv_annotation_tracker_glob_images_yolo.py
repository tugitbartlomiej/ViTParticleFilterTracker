import cv2
import os
from glob import glob

# Path to the folder containing images
image_folder = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/output_frames/raw_images/0_25'
image_files = sorted(
    glob(os.path.join(image_folder, "*.jpg")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
)

# Paths to output directories
output_dir = "output"
annotations_dir = os.path.join(output_dir, "Yolo/YOLOv8_Annotations")
raw_images_dir = os.path.join(output_dir, "Yolo/Raw_Images")

if not os.path.exists(annotations_dir):
    os.makedirs(annotations_dir)
if not os.path.exists(raw_images_dir):
    os.makedirs(raw_images_dir)

tracking = False
continuous_mode = False
paused = False
bbox = None
tracker = None

# Variables for drawing rectangle
drawing = False
ix, iy = -1, -1  # Initial x and y coordinates
rectangle = None

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

# Initialize current image index
current_image_index = 0
total_images = len(image_files)

# Initialize frame, frame_display, frame_id
frame = None
frame_display = None
frame_id = None

# Function to save raw images
def save_raw_image(frame, frame_id):
    """Save the raw image to the Raw_Images directory."""
    raw_image_filename = f"frame_{frame_id:06d}.jpg"
    raw_image_path = os.path.join(raw_images_dir, raw_image_filename)
    cv2.imwrite(raw_image_path, frame)
    print(f"Raw image for frame {frame_id} saved to {raw_image_path}")

# Function to save annotations in YOLOv8 format
def save_yolo_annotation(frame_id, bbox, image_shape):
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

    class_id = 0  # Assuming single class; adjust if necessary

    annotation_line = f"{class_id} {x_center} {y_center} {width} {height}\n"

    annotation_filename = f"frame_{frame_id:06d}.txt"
    annotation_path = os.path.join(annotations_dir, annotation_filename)

    with open(annotation_path, 'w') as f:
        f.write(annotation_line)

    print(f"Annotation for frame {frame_id} saved to {annotation_path}")

# Function to remove annotations and raw images
def remove_yolo_annotation(frame_id):
    """Remove the YOLOv8 annotation file and raw image for the specified frame_id."""
    annotation_filename = f"frame_{frame_id:06d}.txt"
    annotation_path = os.path.join(annotations_dir, annotation_filename)

    if os.path.exists(annotation_path):
        os.remove(annotation_path)
        print(f"Annotation file for frame {frame_id} removed.")
    else:
        print(f"No annotation file found for frame {frame_id}.")

    # Remove raw image
    raw_image_filename = f"frame_{frame_id:06d}.jpg"
    raw_image_path = os.path.join(raw_images_dir, raw_image_filename)

    if os.path.exists(raw_image_path):
        os.remove(raw_image_path)
        print(f"Raw image for frame {frame_id} removed.")
    else:
        print(f"No raw image found for frame {frame_id}.")

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle, bbox, tracking, tracker, frame_display, frame_id, frame

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
        tracker.init(frame_display, bbox)
        tracking = True

        # Save the annotation and raw image immediately
        save_yolo_annotation(frame_id, bbox, frame_display.shape)
        save_raw_image(frame, frame_id)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Remove the annotation and raw image
        remove_yolo_annotation(frame_id)

        # Clear the display
        frame_display = frame.copy()
        cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Tool Tracker", frame_display)

        # Clear ROI and stop tracking
        bbox = None
        tracking = False
        rectangle = None

# Create a named window and set the mouse callback
cv2.namedWindow("Tool Tracker")
cv2.setMouseCallback("Tool Tracker", draw_rectangle)

while True:
    # Check if current_image_index is within bounds
    if current_image_index < 0:
        current_image_index = 0
    if current_image_index >= total_images:
        print("End of image sequence reached.")
        break

    # Load the current image
    img_path = image_files[current_image_index]
    frame = cv2.imread(img_path)
    frame_id = int(os.path.splitext(os.path.basename(img_path))[0].split('_')[1])
    frame_display = frame.copy()  # For display (with annotations)

    # Check if there is an annotation for this frame
    annotation_filename = f"frame_{frame_id:06d}.txt"
    annotation_path = os.path.join(annotations_dir, annotation_filename)
    if os.path.exists(annotation_path):
        # Read existing annotation
        with open(annotation_path, 'r') as f:
            annotation_line = f.readline().strip()
            parts = annotation_line.split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                image_height, image_width = frame.shape[:2]
                bbox_width = width * image_width
                bbox_height = height * image_height
                bbox_x = (x_center * image_width) - (bbox_width / 2)
                bbox_y = (y_center * image_height) - (bbox_height / 2)
                bbox = (bbox_x, bbox_y, bbox_width, bbox_height)

                # Draw the bounding box
                p1 = (int(bbox_x), int(bbox_y))
                p2 = (int(bbox_x + bbox_width), int(bbox_y + bbox_height))
                cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)
    else:
        bbox = None

    if tracking and not paused:
        # Update the tracker to get the new bounding box
        success, bbox = tracker.update(frame)
        if success:
            # Draw the bounding box
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

            # Save the updated annotation and raw image
            save_yolo_annotation(frame_id, bbox, frame.shape)
            save_raw_image(frame, frame_id)
        else:
            cv2.putText(frame_display, "Tracking failure", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Tracking failed. Stopping tracking.")
            tracking = False
            bbox = None  # Reset bbox if tracking fails

    # Draw the rectangle while drawing
    if drawing and rectangle is not None:
        x0, y0, x1, y1 = rectangle
        cv2.rectangle(frame_display, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Display the frame number in the top-left corner
    cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("Tool Tracker", frame_display)

    if continuous_mode and not paused:
        key = cv2.waitKey(30) & 0xFF  # Automatically refresh during continuous mode
        current_image_index += 1
        if current_image_index >= total_images:
            print("End of image sequence reached.")
            break
    else:
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press or mouse event

    # Handle key presses
    if key == ord('q'):
        break
    elif key == ord('n'):  # 'N' key for next image with tracking
        if bbox is not None and not tracking:
            tracker = cv2.TrackerMIL_create()  # Initialize the tracker
            tracker.init(frame_display, bbox)
            tracking = True
        current_image_index += 1
        if current_image_index >= total_images:
            print("End of image sequence reached.")
            break
    elif key == ord('s'):  # 'S' key to skip to next image without tracking
        tracking = False  # Ensure tracking is stopped
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
    elif key == ord('t'):
        # Toggle continuous mode
        continuous_mode = not continuous_mode
        if continuous_mode:
            if bbox is not None and not tracking:
                tracker = cv2.TrackerMIL_create()  # Initialize the tracker for continuous mode
                tracker.init(frame_display, bbox)
            tracking = True
            paused = False
            print("Continuous mode started.")
        else:
            tracking = False
            print("Continuous mode stopped.")

    elif key == 27:  # Escape key
        break

cv2.destroyAllWindows()

print("Annotation process completed.")
