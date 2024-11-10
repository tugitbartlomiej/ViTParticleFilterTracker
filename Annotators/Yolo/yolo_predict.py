import cv2
import torch
from ultralytics import YOLO
import os

# Inicjalizacja modelu z pliku best.pt
model = YOLO("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/surgical_tool_detection/exp10/weights/best.pt")

# Ścieżka do wideo
video_path = "E:/Cataract/videos/micro/train01.mp4"

# Ścieżka do głównego folderu na surowe obrazy
output_dir_raw = "output_frames/raw_images/"

# Ścieżka do głównego folderu z adnotacjami (obrazy z bounding boxami i pliki tekstowe)
output_dir_annotations = "output_frames/annotations/"

# Tworzenie głównych folderów wyjściowych dla surowych obrazów i adnotacji
confidence_folders = {
    "0_25": {"annotations": os.path.join(output_dir_annotations, "0_25"), "raw_images": os.path.join(output_dir_raw, "0_25")},
    "26_50": {"annotations": os.path.join(output_dir_annotations, "26_50"), "raw_images": os.path.join(output_dir_raw, "26_50")},
    "51_75": {"annotations": os.path.join(output_dir_annotations, "51_75"), "raw_images": os.path.join(output_dir_raw, "51_75")},
    "76_100": {"annotations": os.path.join(output_dir_annotations, "76_100"), "raw_images": os.path.join(output_dir_raw, "76_100")},
}

# Tworzenie podfolderów dla przedziałów pewności
for paths in confidence_folders.values():
    os.makedirs(paths["annotations"], exist_ok=True)
    os.makedirs(paths["raw_images"], exist_ok=True)

# Inicjalizacja odczytu wideo
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Plik do zapisu listy zdjęć i ich przypisanych folderów
classification_log = "image_classification.txt"
with open(classification_log, "w") as log_file:

    # Przetwarzanie wideo klatka po klatce
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Zwiększ licznik klatek
        frame_count += 1

        # Wykonanie predykcji na danej klatce
        results = model(frame)

        # Flaga do określenia, czy wykryto obiekt o wystarczającej pewności
        highest_confidence = 0  # Śledzenie najwyższego poziomu pewności dla danej klatki
        frame_with_boxes = frame.copy()  # Kopia klatki do rysowania bounding boxów

        # Przetwarzanie wyników i dodawanie bounding boxów
        object_detected = False
        annotated_image_path = None  # Inicjalizacja, aby uniknąć błędów, gdy nie ma detekcji
        for result in results:
            boxes = result.boxes.xyxy  # współrzędne w formacie (x1, y1, x2, y2)
            classes = result.boxes.cls  # klasy obiektów
            confidences = result.boxes.conf  # poziom pewności detekcji

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box.tolist())  # konwersja na int
                class_id = int(cls)
                confidence = conf.item()  # zamiana na float

                # Aktualizacja najwyższej pewności dla danej klatki
                highest_confidence = max(highest_confidence, confidence)
                object_detected = True  # Flaga ustawiona na True, gdy występuje detekcja

                # Wybór odpowiedniego podfolderu na podstawie poziomu pewności detekcji
                if confidence <= 0.25:
                    annotation_subdir = confidence_folders["0_25"]["annotations"]
                elif confidence <= 0.50:
                    annotation_subdir = confidence_folders["26_50"]["annotations"]
                elif confidence <= 0.75:
                    annotation_subdir = confidence_folders["51_75"]["annotations"]
                else:
                    annotation_subdir = confidence_folders["76_100"]["annotations"]

                # Ścieżki do zapisu obrazu z bounding boxem i pliku tekstowego
                annotated_image_path = os.path.join(annotation_subdir, f"frame_{frame_count}.jpg")
                label_path = os.path.join(annotation_subdir, f"frame_{frame_count}.txt")

                # Rysowanie bounding boxa na kopii klatki (frame_with_boxes)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"Tooltip {class_id} ({confidence * 100:.2f}%)"  # Tekst z procentem pewności
                cv2.putText(frame_with_boxes, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Zapis współrzędnych w formacie YOLO: class_id, x_center, y_center, width, height
                x_center = (x1 + x2) / 2 / frame.shape[1]
                y_center = (y1 + y2) / 2 / frame.shape[0]
                width = (x2 - x1) / frame.shape[1]
                height = (y2 - y1) / frame.shape[0]

                # Zapis pliku tekstowego z adnotacją
                with open(label_path, "a") as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Zapis obrazu z narysowanym bounding boxem, jeśli wykryto obiekt
            if annotated_image_path:
                cv2.imwrite(annotated_image_path, frame_with_boxes)

        # Wybór odpowiedniego folderu dla surowego obrazu na podstawie najwyższego poziomu pewności
        if highest_confidence <= 0.25:
            raw_image_subdir = confidence_folders["0_25"]["raw_images"]
        elif highest_confidence <= 0.50:
            raw_image_subdir = confidence_folders["26_50"]["raw_images"]
        elif highest_confidence <= 0.75:
            raw_image_subdir = confidence_folders["51_75"]["raw_images"]
        else:
            raw_image_subdir = confidence_folders["76_100"]["raw_images"]

        # Ścieżka do zapisu surowego obrazu klatki
        raw_image_path = os.path.join(raw_image_subdir, f"frame_{frame_count}.jpg")

        # Zapis surowego obrazu klatki bez bounding boxów
        cv2.imwrite(raw_image_path, frame)

        # Zapis informacji o przypisaniu zdjęcia do folderu w pliku log_file
        log_file.write(f"frame_{frame_count}.jpg -> {raw_image_subdir}\n")

        print(f"Processed frame {frame_count}")

# Zakończenie odczytu wideo
cap.release()
print("Video processing completed.")
