import os

import cv2
from ultralytics import YOLO

video_path = "E:/Cataract/videos/micro/train04.mp4"
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_dir = os.path.join("output_frames", video_name)
os.makedirs(output_dir, exist_ok=True)
model = YOLO("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/surgical_tool_detection/exp14/weights/best.pt")
output_dir_raw = os.path.join(output_dir, "raw_images")
output_dir_annotations = os.path.join(output_dir, "annotations")
confidence_folders = {
    "0_25": {
        "annotations": os.path.join(output_dir_annotations, "0_25"),
        "raw_images": os.path.join(output_dir_raw, "0_25")
    },
    "26_50": {
        "annotations": os.path.join(output_dir_annotations, "26_50"),
        "raw_images": os.path.join(output_dir_raw, "26_50")
    },
    "51_75": {
        "annotations": os.path.join(output_dir_annotations, "51_75"),
        "raw_images": os.path.join(output_dir_raw, "51_75")
    },
    "76_100": {
        "annotations": os.path.join(output_dir_annotations, "76_100"),
        "raw_images": os.path.join(output_dir_raw, "76_100")
    },
}
for paths in confidence_folders.values():
    os.makedirs(paths["annotations"], exist_ok=True)
    os.makedirs(paths["raw_images"], exist_ok=True)
cap = cv2.VideoCapture(video_path)
frame_count = 0
classification_log = os.path.join(output_dir, "image_classification.txt")
with open(classification_log, "w") as log_file:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame)
        highest_confidence = 0
        frame_with_boxes = frame.copy()
        annotated_image_path = None
        for result in results:
            boxes = result.boxes.xyxy
            classes = result.boxes.cls
            confidences = result.boxes.conf
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(cls)
                confidence = conf.item()
                if confidence > highest_confidence:
                    highest_confidence = confidence
                if confidence <= 0.25:
                    annotation_subdir = confidence_folders["0_25"]["annotations"]
                elif confidence <= 0.50:
                    annotation_subdir = confidence_folders["26_50"]["annotations"]
                elif confidence <= 0.75:
                    annotation_subdir = confidence_folders["51_75"]["annotations"]
                else:
                    annotation_subdir = confidence_folders["76_100"]["annotations"]
                annotated_image_path = os.path.join(annotation_subdir, f"frame_{frame_count}.jpg")
                label_path = os.path.join(annotation_subdir, f"frame_{frame_count}.txt")
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"Tooltip {class_id} ({confidence * 100:.2f}%)"
                cv2.putText(frame_with_boxes, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                x_center = (x1 + x2) / 2 / frame.shape[1]
                y_center = (y1 + y2) / 2 / frame.shape[0]
                width = (x2 - x1) / frame.shape[1]
                height = (y2 - y1) / frame.shape[0]
                with open(label_path, "a") as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            if annotated_image_path:
                cv2.imwrite(annotated_image_path, frame_with_boxes)
        if highest_confidence <= 0.25:
            raw_image_subdir = confidence_folders["0_25"]["raw_images"]
        elif highest_confidence <= 0.50:
            raw_image_subdir = confidence_folders["26_50"]["raw_images"]
        elif highest_confidence <= 0.75:
            raw_image_subdir = confidence_folders["51_75"]["raw_images"]
        else:
            raw_image_subdir = confidence_folders["76_100"]["raw_images"]
        raw_image_path = os.path.join(raw_image_subdir, f"frame_{frame_count}.jpg")
        cv2.imwrite(raw_image_path, frame)
        log_file.write(f"frame_{frame_count}.jpg -> {raw_image_subdir}\n")
        print(f"Processed frame {frame_count}")
cap.release()
print("Video processing completed.")
