import os
import random

import cv2
import numpy as np
from ultralytics import YOLO

from Annotators.DeepSortYolo.sort.sort import Sort  # Upewnij się, że moduł SORT jest zainstalowany


class ObjectTracker:
    def __init__(self, yolo_weights_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Inicjalizacja trackera obiektów z użyciem YOLOv8 do detekcji oraz SORT do śledzenia.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Inicjalizacja modelu YOLOv8
        self.model = YOLO(yolo_weights_path)

        # Inicjalizacja algorytmu SORT
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

        # Słownik do przypisywania kolorów dla poszczególnych ID
        self.color_map = {}

    def generate_random_color(self):
        """Generates a random BGR color."""
        return tuple(random.randint(0, 255) for _ in range(3))

    def get_track_color(self, track_id: int):
        """Returns a consistent color for a given track ID."""
        if track_id not in self.color_map:
            self.color_map[track_id] = self.generate_random_color()
        return self.color_map[track_id]

    def validate_bbox_size(self, bbox, frame_shape, max_ratio=0.5):
        """
        Sprawdza, czy rozmiar bounding boxa nie przekracza ustalonego stosunku względem rozmiarów klatki.
        """
        frame_height, frame_width = frame_shape[:2]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        width_ratio = bbox_width / frame_width
        height_ratio = bbox_height / frame_height
        return width_ratio < max_ratio and height_ratio < max_ratio

    def scale_bbox(self, bbox, scale_factor=0.6):
        """
        Skaluje bounding box względem jego środka.
        Dzięki temu, podobnie jak w poprzednich rozwiązaniach, okno wychwytuje końcówkę narzędzia.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        new_width = width * scale_factor
        new_height = height * scale_factor

        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        return [new_x1, new_y1, new_x2, new_y2]

    def draw_debug_info(self, frame, bbox, color=(0, 255, 0), label=""):
        """
        Rysuje bounding box oraz etykietę na klatce.
        """
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def convert_yolo_to_sort_format(self, yolo_box, confidence):
        """
        Konwertuje bounding box z YOLO ([x1, y1, x2, y2]) do formatu SORT ([x1, y1, x2, y2, confidence]).
        """
        x1, y1, x2, y2 = yolo_box
        return [x1, y1, x2, y2, confidence]

    def process_video(self, input_video_path: str, output_video_path: str = None, debug_mode: bool = False):
        """
        Przetwarza wideo: wykrywanie obiektów przy użyciu YOLOv8 oraz śledzenie przy użyciu SORT.
        """
        if output_video_path is None:
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            output_video_path = f"{base_name}_tracked.mp4"

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Nie można otworzyć wideo: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                print(f"\rPrzetwarzanie klatki: {frame_count}/{total_frames}", end="")

                # Detekcja przy użyciu YOLOv8
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
                detections = []

                # Przetwarzanie wyników detekcji
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        # Skalowanie boksu, aby wychwycić końcówkę narzędzia
                        scaled_box = self.scale_bbox(box, scale_factor=0.6)
                        if self.validate_bbox_size(scaled_box, frame.shape):
                            detection = self.convert_yolo_to_sort_format(scaled_box, conf)
                            detections.append(detection)
                            if debug_mode:
                                self.draw_debug_info(frame, box, (0, 255, 0), "YOLO")
                                # self.draw_debug_info(frame, scaled_box, (255, 0, 0), "Scaled")

                # SORT oczekuje numpy array o kształcie (N, 5)
                if len(detections) > 0:
                    dets = np.array(detections)
                else:
                    dets = np.empty((0, 5))

                # Aktualizacja trackera SORT
                tracks = self.tracker.update(dets)
                # Każdy rekord w tracks ma postać [x1, y1, x2, y2, track_id]
                for track in tracks:
                    x1, y1, x2, y2, track_id = track
                    color = self.get_track_color(int(track_id))
                    if debug_mode:
                        self.draw_debug_info(frame, [x1, y1, x2, y2], color, f"ID:{int(track_id)}")
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"ID:{int(track_id)}", (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                writer.write(frame)
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC do przerwania
                    break

        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            print(f"\n[INFO] Wynik zapisano do: {output_video_path}")

def main():
    # Używamy tych samych ścieżek co poprzednio
    YOLO_WEIGHTS = r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/surgical_tool_detection/exp14/weights/best.pt"
    VIDEO_PATH = r"E:/Cataract/videos/micro/one_video/test01.mp4"

    tracker = ObjectTracker(yolo_weights_path=YOLO_WEIGHTS, conf_threshold=0.5, iou_threshold=0.45)
    tracker.process_video(VIDEO_PATH, debug_mode=True)

if __name__ == "__main__":
    main()
