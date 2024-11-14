import cv2
import json
import os
from ultralytics import YOLO  # Import modelu YOLO
import warnings

# Tymczasowe wyciszenie FutureWarning (jeśli potrzebne)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ścieżka do pliku wideo
video_path = 'E:/Cataract/videos/micro/train01.mp4'  # Podaj ścieżkę do swojego pliku wideo

# Ścieżka do modelu YOLO
yolo_model_path = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/surgical_tool_detection/exp11/weights/best.pt'  # Podaj ścieżkę do wytrenowanego modelu YOLO

# Inicjalizacja modelu YOLO
try:
    model = YOLO(yolo_model_path)
except Exception as e:
    print(f"Błąd podczas ładowania modelu YOLO: {e}")
    exit()

# Inicjalizacja wideo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Nie można otworzyć pliku wideo.")
    exit()

# Ścieżka do pliku z adnotacjami
annotations_file = os.path.join("output", "annotations.json")

# Wczytaj istniejące adnotacje lub zainicjalizuj nowe w formacie COCO
if os.path.exists(annotations_file):
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
            print("Plik z adnotacjami został pomyślnie wczytany.")
    except json.JSONDecodeError:
        print("Błąd: Plik z adnotacjami jest pusty lub zawiera nieprawidłowy JSON. Inicjalizacja nowej struktury adnotacji.")
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
else:
    print("Nie znaleziono pliku z adnotacjami. Inicjalizacja nowej struktury adnotacji.")
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

# Zmienne do rysowania prostokąta
drawing = False
ix, iy = -1, -1  # Początkowe współrzędne x i y
rectangle = None

# Pobierz całkowitą liczbę klatek
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0  # Inicjalizacja bieżącego indeksu klatki

# Foldery do zapisywania obrazów
output_dir = "output"
annotated_images_dir = os.path.join(output_dir, "Annotated_Images")
raw_images_dir = os.path.join(output_dir, "Raw_Images")

os.makedirs(annotated_images_dir, exist_ok=True)
os.makedirs(raw_images_dir, exist_ok=True)

# Instrukcje dla użytkownika
print("Instrukcje:")
print("- Naciśnij 'N', aby wykonać detekcję narzędzi za pomocą YOLO w bieżącej klatce.")
print("- Użyj myszy, aby ręcznie zaznaczyć obszar (lewy przycisk myszy).")
print("- Kliknij prawym przyciskiem myszy, aby usunąć istniejący bounding box dla bieżącej klatki.")
print("- Naciśnij 'S', aby pominąć bieżącą klatkę.")
print("- Naciśnij 'A', aby cofnąć się do poprzedniej klatki.")
print("- Naciśnij 'Q' lub 'ESC', aby zakończyć.")

# Funkcja do zapisywania adnotacji
def update_annotations(coco_data, frame_id, bbox, frame_shape, source='manual'):
    """Aktualizuj adnotacje dla bieżącej klatki na podstawie podanego bounding boxa."""
    image_height, image_width = frame_shape[:2]

    # Dodaj informacje o obrazie, jeśli nie istnieją
    if not any(img['id'] == frame_id for img in coco_data["images"]):
        coco_data["images"].append({
            "id": frame_id,
            "file_name": f"frame_{frame_id:06d}.jpg",
            "height": image_height,
            "width": image_width
        })

    # Usuń istniejące adnotacje dla tej klatki i tego źródła
    coco_data["annotations"] = [ann for ann in coco_data["annotations"] if not (ann['image_id'] == frame_id and ann.get('source') == source)]

    x_min, y_min, width, height = bbox
    area = width * height

    # Zapewnij unikalne ID adnotacji
    existing_ids = [ann['id'] for ann in coco_data["annotations"]]
    ann_id = max(existing_ids) + 1 if existing_ids else 1

    coco_data["annotations"].append({
        "id": ann_id,
        "image_id": frame_id,
        "category_id": 1,  # Zakładamy jedną kategorię
        "bbox": [x_min, y_min, width, height],
        "area": area,
        "iscrowd": 0,
        "source": source  # Dodaj źródło adnotacji
    })

# Funkcja do usuwania adnotacji i obrazów
def remove_annotation_and_images(coco_data, frame_id):
    """Usuń adnotacje pochodzące z YOLO dla określonej klatki i powiązane obrazy z obu folderów."""
    # Usuń adnotacje pochodzące z YOLO
    coco_data["annotations"] = [ann for ann in coco_data["annotations"] if not (ann['image_id'] == frame_id and ann.get('source') == 'yolo')]

    # Usuń powiązane obrazy z folderów Raw_Images i Annotated_Images
    raw_image_path = os.path.join(raw_images_dir, f"frame_{frame_id:06d}.jpg")
    annotated_image_path = os.path.join(annotated_images_dir, f"frame_{frame_id:06d}.jpg")

    if os.path.exists(raw_image_path):
        os.remove(raw_image_path)
        print(f"Surowy obraz dla klatki {frame_id} usunięty z folderu Raw_Images.")

    if os.path.exists(annotated_image_path):
        os.remove(annotated_image_path)
        print(f"Adnotowany obraz dla klatki {frame_id} usunięty z folderu Annotated_Images.")

    print(f"Adnotacja i obrazy dla klatki {frame_id} usunięte z obu folderów.")

# Funkcja zwrotna myszy
def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, rectangle, frame_display, coco_data, frame_id, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rectangle = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rectangle = (ix, iy, x, y)
            temp_frame = frame_display.copy()
            x0, y0, x1, y1 = rectangle
            cv2.rectangle(temp_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.imshow("Tool Tracker", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangle = (ix, iy, x, y)
        x0, y0 = min(ix, x), min(iy, y)
        x1, y1 = max(ix, x), max(iy, y)
        width = x1 - x0
        height = y1 - y0
        bbox = [x0, y0, width, height]

        # Aktualizuj adnotacje jako ręczne
        update_annotations(coco_data, frame_id, bbox, frame_display.shape, source='manual')

        # Narysuj prostokąt na stałe (czerwony dla ręcznych)
        cv2.rectangle(frame_display, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.imshow("Tool Tracker", frame_display)

        # Zapisz obrazy
        raw_image_path = os.path.join(raw_images_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(raw_image_path, frame)

        annotated_image_path = os.path.join(annotated_images_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(annotated_image_path, frame_display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Usuń adnotacje pochodzące z YOLO dla bieżącej klatki
        remove_annotation_and_images(coco_data, frame_id)

        # Odśwież wyświetlanie klatki bez YOLO bounding boxów
        frame_display = frame.copy()
        # Dodaj ręczne adnotacje, jeśli istnieją
        manual_annotations = [ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id and ann.get('source') == 'manual']
        for ann in manual_annotations:
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            p1 = (int(x_min), int(y_min))
            p2 = (int(x_min + width), int(y_min + height))
            cv2.rectangle(frame_display, p1, p2, (0, 0, 255), 2)  # Ręczne adnotacje na czerwono

        # Dodaj informacje o obrazie do COCO, jeśli nie zostały jeszcze dodane
        if not any(img['id'] == frame_id for img in coco_data["images"]):
            coco_data["images"].append({
                "id": frame_id,
                "file_name": f"frame_{frame_id:06d}.jpg",
                "height": frame.shape[0],
                "width": frame.shape[1]
            })

        # Wyświetl numer klatki w lewym górnym rogu
        cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Wyświetl klatkę
        cv2.imshow("Tool Tracker", frame_display)

# Utwórz okno i ustaw funkcję zwrotną myszy
cv2.namedWindow("Tool Tracker")
cv2.setMouseCallback("Tool Tracker", mouse_callback)

# Funkcja do ładowania i wyświetlania klatki
def load_and_display_frame(cap, current_frame, coco_data):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"Nie można odczytać klatki {current_frame}.")
        return None, None, None

    frame_id = current_frame + 1

    frame_display = frame.copy()
    frame_raw = frame.copy()

    # Dodaj informacje o obrazie do COCO, jeśli nie zostały jeszcze dodane
    if not any(img['id'] == frame_id for img in coco_data["images"]):
        coco_data["images"].append({
            "id": frame_id,
            "file_name": f"frame_{frame_id:06d}.jpg",
            "height": frame.shape[0],
            "width": frame.shape[1]
        })

    # Sprawdź, czy istnieją adnotacje dla tej klatki
    existing_annotations = [ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id]

    # Jeśli są adnotacje, narysuj je
    if existing_annotations:
        for ann in existing_annotations:
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            p1 = (int(x_min), int(y_min))
            p2 = (int(x_min + width), int(y_min + height))
            if ann.get('source') == 'manual':
                color = (0, 0, 255)  # Ręczne adnotacje na czerwono
            else:
                color = (255, 0, 0)  # YOLO adnotacje na niebiesko
            cv2.rectangle(frame_display, p1, p2, color, 2)

    # Wyświetl numer klatki w lewym górnym rogu
    cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Wyświetl klatkę
    cv2.imshow("Tool Tracker", frame_display)

    return frame, frame_display, frame_id

# Ładowanie i wyświetlanie pierwszej klatki
frame, frame_display, frame_id = load_and_display_frame(cap, current_frame, coco_data)
if frame is None:
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Dodaj flagę oznaczającą, czy aktualna klatka została już oznaczona przez YOLO
annotated = False

while True:
    # Odczytaj naciśnięty klawisz (nie blokuj pętli)
    key = cv2.waitKey(1) & 0xFF

    # Obsługa naciśnięć klawiszy
    if key == ord('q') or key == 27:  # 'q' lub 'ESC' aby zakończyć
        break
    elif key in [ord('n'), ord('N')]:  # Klawisz 'N' lub 'n' do wykonania detekcji YOLO i przejścia do następnej klatki
        if not annotated:
            # Sprawdź, czy manualne adnotacje istnieją dla tej klatki
            manual_annotations = [ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id and ann.get('source') == 'manual']
            if manual_annotations:
                print(f"Manualne adnotacje istnieją dla klatki {frame_id}. YOLO nie zostanie uruchomione.")
            else:
                # Wykonaj detekcję za pomocą YOLO
                results = model(frame)
                detections = results[0].boxes  # Pobierz detekcje z pierwszego wyniku

                print(f"Number of YOLO detections: {len(detections)}")

                if len(detections) > 0:
                    for det in detections:
                        bbox_xyxy = det.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                        x_min, y_min, x_max, y_max = bbox_xyxy
                        width = x_max - x_min
                        height = y_max - y_min
                        bbox = [x_min, y_min, width, height]

                        # Aktualizuj adnotacje z 'source'='yolo'
                        update_annotations(coco_data, frame_id, bbox, frame.shape, source='yolo')

                        # Narysuj prostokąt (niebieski dla YOLO)
                        p1 = (int(x_min), int(y_min))
                        p2 = (int(x_max), int(y_max))
                        cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2)

                        print(f"YOLO detected bbox: {bbox}")

                    print(f"Detekcja zakończona dla klatki {frame_id}.")
                else:
                    print("Nie wykryto narzędzi w bieżącej klatce.")

                # Zapisz obrazy
                raw_image_path = os.path.join(raw_images_dir, f"frame_{frame_id:06d}.jpg")
                cv2.imwrite(raw_image_path, frame)
                print(f"Raw image saved to {raw_image_path}")

                annotated_image_path = os.path.join(annotated_images_dir, f"frame_{frame_id:06d}.jpg")
                cv2.imwrite(annotated_image_path, frame_display)
                print(f"Annotated image saved to {annotated_image_path}")

                # Wyświetl zaktualizowaną klatkę
                cv2.imshow("Tool Tracker", frame_display)

                # Ustaw flagę, że aktualna klatka została oznaczona przez YOLO
                annotated = True
        else:
            # Przejdź do następnej klatki
            current_frame += 1
            if current_frame >= total_frames:
                print("Koniec wideo.")
                break

            frame, frame_display, frame_id = load_and_display_frame(cap, current_frame, coco_data)
            if frame is None:
                break

            # Reset flagi
            annotated = False

    elif key in [ord('s'), ord('S')]:  # 'S' lub 's' aby pominąć bieżącą klatkę
        current_frame += 1
        if current_frame >= total_frames:
            print("Koniec wideo.")
            break

        frame, frame_display, frame_id = load_and_display_frame(cap, current_frame, coco_data)
        if frame is None:
            break

        # Reset flagi
        annotated = False

    elif key in [ord('a'), ord('A')]:  # 'A' lub 'a' aby cofnąć się do poprzedniej klatki
        if current_frame > 0:
            current_frame -= 1
            frame, frame_display, frame_id = load_and_display_frame(cap, current_frame, coco_data)
            if frame is None:
                current_frame += 1  # Cofnij, jeśli nie można odczytać
            # Reset flagi
            annotated = False
        else:
            print("Już jesteś na pierwszej klatce.")

    elif key == ord(' '):  # Spacja aby wstrzymać/wznowić
        # Możesz zaimplementować tutaj logikę wstrzymywania/wznawiania wideo
        # Obecnie, przyciski kontrolują wszystko, więc spacja nie jest przypisana do żadnej akcji
        print("Spacja została naciśnięta. Aktualnie nie jest przypisana żadna akcja.")

# Zwolnij obiekt VideoCapture i zamknij okna
cap.release()
cv2.destroyAllWindows()

# Zapisz dane COCO do pliku JSON
with open(annotations_file, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"Adnotacje COCO zapisane do {annotations_file}")
