import os
import random

import cv2
import numpy as np
from ultralytics import YOLO

from Annotators.DeepSortYolo.sort.sort import Sort

# Parametr definiujący częstotliwość przetwarzania klatek
SKIP_FRAMES = 1
# Próg do zapisywania do pliku TXT – wykrywamy, gdy confidence >= 0.8
SAVE_CONF_THRESHOLD = 0.8

class ObjectTracker:
    def __init__(
        self,
        yolo_weights_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        sort_max_age: int = 100,
        sort_min_hits: int = 3,
        sort_iou_threshold: float = 0.1,
        scale_factor: float = 0.7
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.scale_factor = scale_factor

        # Initialize YOLO model
        self.model = YOLO(yolo_weights_path)
        # Initialize SORT tracker
        self.tracker = Sort(
            max_age=sort_max_age,
            min_hits=sort_min_hits,
            iou_threshold=sort_iou_threshold
        )
        # Dictionary for consistent track coloring
        self.color_map = {}

    def generate_random_color(self) -> tuple:
        return tuple(random.randint(0, 255) for _ in range(3))

    def get_track_color(self, track_id: int) -> tuple:
        if track_id not in self.color_map:
            self.color_map[track_id] = self.generate_random_color()
        return self.color_map[track_id]

    def validate_bbox_size(self, bbox: list, frame_shape: tuple, max_ratio: float = 0.5) -> bool:
        frame_height, frame_width = frame_shape[:2]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        return (bbox_width / frame_width < max_ratio) and (bbox_height / frame_height < max_ratio)

    def scale_bbox(self, bbox: list) -> list:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        new_width = width * self.scale_factor
        new_height = height * self.scale_factor

        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        return [new_x1, new_y1, new_x2, new_y2]

    def draw_bbox(self, frame, bbox: list, color: tuple = (0, 255, 0), label: str = "") -> None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def convert_yolo_to_sort_format(self, yolo_box: list, confidence: float) -> list:
        # Format expected by SORT: [x1, y1, x2, y2, confidence]
        return [yolo_box[0], yolo_box[1], yolo_box[2], yolo_box[3], confidence]

    def process_video_file(
        self,
        video_path: str,
        video_name: str,
        raw_folder: str,
        annotated_folder: str,
        sort_change_folder: str,
        skip_frames: int = SKIP_FRAMES
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_count = 0
        prev_track_ids = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Kopie klatek: raw, do rysowania wykryć YOLO oraz SORT
            raw_frame = frame.copy()
            yolo_frame = frame.copy()
            sort_frame = frame.copy()

            label_data = []  # detekcje z confidence >= SAVE_CONF_THRESHOLD, do zapisu TXT
            detections = []  # wszystkie wykrycia dla aktualizacji SORT

            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for box, conf, cls in zip(boxes, confs, classes):
                    # Rysuj wszystkie wykrycia na YOLO frame (zielony)
                    self.draw_bbox(yolo_frame, box, color=(0, 255, 0), label="YOLO")
                    # Dodaj wykrycie do SORT (po skalowaniu) - niezależnie od pewności
                    scaled_box = self.scale_bbox(box)
                    if self.validate_bbox_size(scaled_box, frame.shape):
                        detections.append(self.convert_yolo_to_sort_format(scaled_box, conf))
                    # Jeśli confidence >= SAVE_CONF_THRESHOLD, zbieramy dane do zapisu
                    if conf >= SAVE_CONF_THRESHOLD:
                        x1, y1, x2, y2 = map(int, box)
                        x_center = ((x1 + x2) / 2) / frame.shape[1]
                        y_center = ((y1 + y2) / 2) / frame.shape[0]
                        w_norm = (x2 - x1) / frame.shape[1]
                        h_norm = (y2 - y1) / frame.shape[0]
                        label_data.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

            # Zawsze aktualizujemy tracker SORT
            if detections:
                dets = np.array(detections)
            else:
                dets = np.empty((0, 5))
            tracks = self.tracker.update(dets)
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                color = self.get_track_color(int(track_id))
                label = f"ID:{int(track_id)}"
                self.draw_bbox(sort_frame, [x1, y1, x2, y2], color=color, label=label)

            # Nakładka tekstowa z nazwą wideo i numerem klatki
            overlay_text = f"{video_name} Frame: {frame_count}"
            cv2.putText(yolo_frame, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(sort_frame, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Sprawdzenie zmiany zbioru track ID
            current_track_ids = {int(track[-1]) for track in tracks}
            id_changed = False
            if prev_track_ids is not None and current_track_ids != prev_track_ids:
                id_changed = True
            prev_track_ids = current_track_ids

            # Zapisujemy pliki tylko jeśli mamy przynajmniej jedną detekcję powyżej 80%
            if label_data:
                frame_prefix = f"{video_name}_frame_{frame_count:07d}"
                cv2.imwrite(os.path.join(raw_folder, f"{frame_prefix}.jpg"), raw_frame)
                cv2.imwrite(os.path.join(annotated_folder, f"{frame_prefix}.jpg"), yolo_frame)
                with open(os.path.join(annotated_folder, f"{frame_prefix}.txt"), "w") as f:
                    for line in label_data:
                        f.write(line + "\n")
                # Zapis SORT (jeśli nastąpiła zmiana track ID)
                if id_changed:
                    cv2.imwrite(os.path.join(sort_change_folder, f"{frame_prefix}.jpg"), sort_frame)

            # Opcjonalne wyświetlanie
            cv2.imshow("YOLO", yolo_frame)
            cv2.imshow("SORT", sort_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Processed video: {video_name}")

def process_videos_in_folder(
    input_folder: str,
    output_folder: str,
    yolo_weights_path: str,
    skip_frames: int = SKIP_FRAMES
) -> None:
    raw_folder = os.path.join(output_folder, "RAW")
    annotated_folder = os.path.join(output_folder, "annotated_yolo")
    sort_change_folder = os.path.join(output_folder, "sort_id_changes")
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(annotated_folder, exist_ok=True)
    os.makedirs(sort_change_folder, exist_ok=True)

    tracker = ObjectTracker(
        yolo_weights_path=yolo_weights_path,
        conf_threshold=0.5,
        iou_threshold=0.25,
        sort_max_age=100,
        sort_min_hits=3,
        sort_iou_threshold=0.1,
        scale_factor=0.7
    )

    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]
    if not video_files:
        print("No .mp4 files found in the input folder.")
        return

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_name, _ = os.path.splitext(video_file)
        print(f"[INFO] Processing video: {video_name}")
        tracker.process_video_file(
            video_path=video_path,
            video_name=video_name,
            raw_folder=raw_folder,
            annotated_folder=annotated_folder,
            sort_change_folder=sort_change_folder,
            skip_frames=skip_frames
        )

def main() -> None:
    yolo_weights = r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/surgical_tool_detection/exp14/weights/best.pt"
    input_videos_folder = r"E:/Cataract/videos/micro"
    output_folder = r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos"

    process_videos_in_folder(
        input_folder=input_videos_folder,
        output_folder=output_folder,
        yolo_weights_path=yolo_weights,
        skip_frames=SKIP_FRAMES
    )

if __name__ == "__main__":
    main()
