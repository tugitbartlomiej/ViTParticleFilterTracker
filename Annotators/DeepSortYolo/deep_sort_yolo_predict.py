import os
import random

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


class ObjectTracker:
    def __init__(self, yolo_weights_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Inicjalizacja trackera obiektów.

        Args:
            yolo_weights_path: Ścieżka do wag modelu YOLO
            conf_threshold: Próg pewności dla detekcji YOLO
            iou_threshold: Próg IoU dla non-maximum suppression
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Inicjalizacja modelu YOLO
        self.model = YOLO(yolo_weights_path)

        # Inicjalizacja DeepSort z dostosowanymi parametrami
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.3,
            nn_budget=100,
            min_detection_height=20,
        )

        # Słownik do przechowywania kolorów dla poszczególnych ID
        self.color_map = {}

        # Słownik do przechowywania historii trajektorii
        self.track_history = {}

    def generate_random_color(self):
        """Generuje losowy kolor w formacie BGR."""
        return tuple(random.randint(0, 255) for _ in range(3))

    def get_track_color(self, track_id: int):
        """Zwraca kolor dla danego ID, generując nowy jeśli nie istnieje."""
        if track_id not in self.color_map:
            self.color_map[track_id] = self.generate_random_color()
        return self.color_map[track_id]

    def validate_bbox_size(self, bbox, frame_shape, max_ratio=0.5):
        """
        Sprawdza czy bounding box nie jest zbyt duży względem klatki.

        Args:
            bbox: Bounding box w formacie [x1, y1, x2, y2]
            frame_shape: Kształt klatki (height, width, channels)
            max_ratio: Maksymalny dozwolony stosunek wymiarów boksu do wymiarów klatki
        """
        frame_height, frame_width = frame_shape[:2]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        width_ratio = bbox_width / frame_width
        height_ratio = bbox_height / frame_height

        return width_ratio < max_ratio and height_ratio < max_ratio

    def scale_bbox(self, bbox, scale_factor=0.8):
        """
        Skaluje bounding box względem jego środka.

        Args:
            bbox: Bounding box w formacie [x1, y1, x2, y2]
            scale_factor: Współczynnik skalowania (< 1 zmniejsza, > 1 zwiększa)
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        new_width = width * scale_factor
        new_height = height * scale_factor

        x1 = center_x - new_width / 2
        y1 = center_y - new_height / 2
        x2 = center_x + new_width / 2
        y2 = center_y + new_height / 2

        return [x1, y1, x2, y2]

    def draw_debug_info(self, frame, bbox, color=(0, 255, 0), label=""):
        """
        Rysuje bounding box z dodatkowymi informacjami debugowania.
        """
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        debug_text = f"{label} {width}x{height}"
        cv2.putText(frame, debug_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def process_video(self, input_video_path: str, output_video_path: str = None, debug_mode: bool = False):
        """
        Przetwarza wideo, wykonując detekcję i śledzenie obiektów.

        Args:
            input_video_path: Ścieżka do pliku wejściowego
            output_video_path: Ścieżka do pliku wyjściowego (opcjonalna)
            debug_mode: Czy wyświetlać dodatkowe informacje debugowania
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

                # Detekcja YOLO
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
                detections = []

                # Przetwarzanie detekcji
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy()

                    for box, conf, cls_ in zip(boxes, confs, clss):
                        # Skalowanie i walidacja rozmiaru boksu
                        scaled_box = self.scale_bbox(box, scale_factor=0.8)

                        if self.validate_bbox_size(scaled_box, frame.shape):
                            detections.append((scaled_box, conf, int(cls_)))

                            if debug_mode:
                                # Pokaż oryginalne detekcje YOLO
                                self.draw_debug_info(frame, box, (0, 255, 0), "YOLO")
                                # Pokaż przeskalowane boxy
                                self.draw_debug_info(frame, scaled_box, (255, 0, 0), "Scaled")

                # Aktualizacja trackera
                tracks = self.tracker.update_tracks(detections, frame=frame)

                # Rysowanie śledzonych obiektów
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    bbox = track.to_ltrb()
                    track_id = track.track_id
                    color = self.get_track_color(track_id)

                    if debug_mode:
                        self.draw_debug_info(frame, bbox, color, f"ID:{track_id}")
                    else:
                        # Standardowe rysowanie bez informacji debugowania
                        cv2.rectangle(frame,
                                      (int(bbox[0]), int(bbox[1])),
                                      (int(bbox[2]), int(bbox[3])),
                                      color, 2)
                        cv2.putText(frame, f"ID:{track_id}",
                                    (int(bbox[0]), int(bbox[1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                writer.write(frame)

                # Wyświetl podgląd
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            print(f"\n[INFO] Zapisano wynik do: {output_video_path}")


def main():
    # Konfiguracja ścieżek
    YOLO_WEIGHTS = r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/surgical_tool_detection/exp14/weights/best.pt"
    VIDEO_PATH = r"E:/Cataract/videos/micro/one_video/test01.mp4"

    # Inicjalizacja trackera
    tracker = ObjectTracker(
        yolo_weights_path=YOLO_WEIGHTS,
        conf_threshold=0.5,
        iou_threshold=0.45
    )

    # Przetwarzanie wideo z włączonym trybem debugowania
    tracker.process_video(VIDEO_PATH, debug_mode=True)


if __name__ == "__main__":
    main()