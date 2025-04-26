import os

import cv2
from ultralytics import YOLO

skip_frames = 5
videos_dir = r"E:/Cataract/videos/micro"
output_dir = "output_frames_80_100"
annotated_dir = os.path.join(output_dir, "annotated")
raw_dir = os.path.join(output_dir, "raw")
os.makedirs(annotated_dir, exist_ok=True)
os.makedirs(raw_dir, exist_ok=True)
model = YOLO("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/surgical_tool_detection/Eden/exp/weights/best.pt")

for video_file in os.listdir(videos_dir):
    if not video_file.lower().endswith(".mp4"):
        continue
    video_path = os.path.join(videos_dir, video_file)
    video_name = os.path.splitext(video_file)[0]
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        results = model(frame)
        if len(results) == 0:
            continue
        boxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        classes = results[0].boxes.cls
        valid_idx = [i for i, conf in enumerate(confidences) if conf >= 0.80]
        if not valid_idx:
            continue
        annotated = frame.copy()
        label_data = []
        for i in valid_idx:
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            c = confidences[i].item()
            cls_id = int(classes[i])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Tooltip {cls_id} ({c*100:.2f}%)"
            cv2.putText(annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            x_center = ((x1 + x2) / 2) / frame.shape[1]
            y_center = ((y1 + y2) / 2) / frame.shape[0]
            w = (x2 - x1) / frame.shape[1]
            h = (y2 - y1) / frame.shape[0]
            label_data.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        frame_prefix = f"{video_name}_frame_{frame_count:07d}"
        cv2.imwrite(os.path.join(raw_dir, f"{frame_prefix}.jpg"), frame)
        cv2.imwrite(os.path.join(annotated_dir, f"{frame_prefix}.jpg"), annotated)
        with open(os.path.join(annotated_dir, f"{frame_prefix}.txt"), "w") as f:
            for line in label_data:
                f.write(line + "/n")
    cap.release()
