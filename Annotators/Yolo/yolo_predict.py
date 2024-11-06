import cv2
import torch
from ultralytics import YOLO
import os

# Inicjalizacja modelu z pliku best.pt
model = YOLO("best.pt")

# Ścieżka do wideo
video_path = "path/to/your/video.mp4"

# Ścieżka do folderu, w którym będą zapisywane wyniki
output_dir = "output_frames/"
os.makedirs(output_dir, exist_ok=True)

# Inicjalizacja odczytu wideo
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Przetwarzanie wideo klatka po klatce
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Zwiększ licznik klatek
    frame_count += 1

    # Wykonanie predykcji na danej klatce
    results = model(frame)

    # Ścieżki do zapisu wyników
    image_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
    label_path = os.path.join(output_dir, f"frame_{frame_count}.txt")

    # Zapis obrazu klatki
    cv2.imwrite(image_path, frame)

    # Przetwarzanie wyników
    with open(label_path, "w") as f:
        for result in results:
            # Pobranie współrzędnych bounding box i klasy
            boxes = result.boxes.xywhn  # współrzędne normalizowane (YOLO format)
            classes = result.boxes.cls  # klasy obiektów

            for box, cls in zip(boxes, classes):
                x_center, y_center, width, height = box.tolist()
                class_id = int(cls)

                # Zapis w formacie YOLO: class_id, x_center, y_center, width, height
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Processed frame {frame_count}")

# Zakończenie odczytu wideo
cap.release()
print("Video processing completed.")
