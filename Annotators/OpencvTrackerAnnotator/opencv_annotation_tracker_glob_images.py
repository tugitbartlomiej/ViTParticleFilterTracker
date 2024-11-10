import cv2
import json
import os
from glob import glob

# Ścieżka do folderu z obrazami
image_folder = 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Yolo/output_frames/raw_images/0_25'
image_files = sorted(
    glob(os.path.join(image_folder, "*.jpg")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
)

# Ścieżka do pliku z adnotacjami
annotations_file = os.path.join("output", "annotations.json")

# Wczytaj istniejące adnotacje lub zainicjalizuj nową strukturę w formacie COCO
if os.path.exists(annotations_file):
    try:
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
            print("Plik z adnotacjami załadowany pomyślnie.")
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
    print("Brak pliku z adnotacjami. Inicjalizacja nowej struktury adnotacji.")
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

# Inicjalizacja zmiennych
tracking = False
continuous_mode = False
paused = False
bbox = None
tracker = None

# Zmienne do rysowania prostokąta
drawing = False
ix, iy = -1, -1  # Początkowe współrzędne
rectangle = None

# Foldery do zapisywania obrazów
output_dir = "output"
annotated_images_dir = os.path.join(output_dir, "Annotated_Images")
raw_images_dir = os.path.join(output_dir, "Raw_Images")

if not os.path.exists(annotated_images_dir):
    os.makedirs(annotated_images_dir)
if not os.path.exists(raw_images_dir):
    os.makedirs(raw_images_dir)

# Instrukcje dla użytkownika
print("Użyj myszy do rysowania ROI:")
print("- Lewy przycisk i przeciągnij, aby narysować prostokąt.")
print("- Zwolnij przycisk myszy, aby zaakceptować ROI.")
print("Naciśnij 'N', aby przejść do następnego obrazu z śledzeniem.")
print("Naciśnij 'S', aby pominąć następny obraz bez śledzenia.")
print("Naciśnij 'A', aby wrócić do poprzedniego obrazu.")
print("Naciśnij 'T', aby przełączyć tryb ciągły.")
print("Naciśnij 'Space', aby pauzować/wznawiać.")
print("Prawy przycisk myszy, aby wyczyścić ROI.")
print("Naciśnij 'Q', aby wyjść.")

# Funkcja do zapisywania klatek
def save_frames(frame, frame_display, frame_id):
    """Zapisz surowe i adnotowane obrazy do odpowiednich folderów."""
    raw_image_filename = f"frame_{frame_id:06d}.jpg"
    raw_image_path = os.path.join(raw_images_dir, raw_image_filename)
    cv2.imwrite(raw_image_path, frame)

    annotated_image_filename = f"frame_{frame_id:06d}.jpg"
    annotated_image_path = os.path.join(annotated_images_dir, annotated_image_filename)
    cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imwrite(annotated_image_path, frame_display)

# Funkcja do aktualizacji adnotacji
def update_annotation(coco_data, frame_id, bbox):
    """Zaktualizuj lub dodaj nową adnotację dla aktualnego obrazu."""
    existing_annotation = next((ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id), None)

    if existing_annotation:
        existing_annotation['bbox'] = list(bbox)
        existing_annotation['area'] = bbox[2] * bbox[3]
    else:
        existing_ids = [ann['id'] for ann in coco_data["annotations"]]
        new_id = max(existing_ids) + 1 if existing_ids else 1
        coco_data["annotations"].append({
            "id": new_id,
            "image_id": frame_id,
            "category_id": 1,
            "bbox": list(bbox),
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        })

def remove_annotation_and_images(coco_data, frame_id):
    """Usuń adnotację dla określonego frame_id i powiązane obrazy z obu folderów."""
    # Usuń adnotację
    coco_data["annotations"] = [ann for ann in coco_data["annotations"] if ann['image_id'] != frame_id]

    # Usuń powiązane obrazy z obu folderów
    raw_image_path = os.path.join(raw_images_dir, f"frame_{frame_id:06d}.jpg")
    annotated_image_path = os.path.join(annotated_images_dir, f"frame_{frame_id:06d}.jpg")

    if os.path.exists(raw_image_path):
        os.remove(raw_image_path)
        print(f"Surowy obraz dla frame {frame_id} usunięty z folderu Raw_Images.")

    if os.path.exists(annotated_image_path):
        os.remove(annotated_image_path)
        print(f"Anotowany obraz dla frame {frame_id} usunięty z folderu Annotated_Images.")

    print(f"Adnotacja i obrazy dla frame {frame_id} usunięte z obu folderów.")

# Funkcja callback myszy
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rectangle, bbox, tracking, tracker, frame_display, frame_id, coco_data

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

        # Inicjalizacja trackera z nowym ROI
        tracker = cv2.TrackerMIL_create()
        tracker.init(current_frame_image, bbox)
        tracking = True

        # Aktualizacja lub dodanie bbox do adnotacji
        update_annotation(coco_data, frame_id, bbox)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Sprawdź, czy istnieje adnotacja dla tego obrazu
        existing_annotation = next((ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id), None)

        if existing_annotation:
            # Usuń adnotację i powiązane obrazy
            remove_annotation_and_images(coco_data, frame_id)

            # Wyświetl czysty obraz
            frame_display = current_frame_image.copy()
            cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Tool Tracker", frame_display)
        else:
            print(f"Brak adnotacji dla frame {frame_id}.")

        # Wyczyść ROI i zatrzymaj śledzenie
        bbox = None
        tracking = False
        rectangle = None

# Utwórz okno i ustaw callback myszy
cv2.namedWindow("Tool Tracker")
cv2.setMouseCallback("Tool Tracker", draw_rectangle)

# Inicjalizacja zmiennych do nawigacji
current_frame_index = 0
total_frames = len(image_files)
current_frame_image = None

# Funkcja do ładowania i przygotowania obrazu
def load_frame(index):
    global current_frame_image, frame_display, frame_id, coco_data
    img_path = image_files[index]
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Nie można wczytać obrazu: {img_path}")
        return False
    current_frame_image = frame.copy()
    frame_id = int(os.path.splitext(os.path.basename(img_path))[0].split('_')[1])
    frame_display = frame.copy()

    # Dodanie informacji o obrazie do struktury COCO, jeśli jeszcze nie dodano
    if not any(img['id'] == frame_id for img in coco_data["images"]):
        coco_data["images"].append({
            "id": frame_id,
            "file_name": f"frame_{frame_id:06d}.jpg",
            "height": frame.shape[0],
            "width": frame.shape[1]
        })

    # Sprawdź, czy istnieje adnotacja dla tego obrazu
    existing_annotation = next((ann for ann in coco_data["annotations"] if ann['image_id'] == frame_id), None)

    if existing_annotation:
        bbox = existing_annotation['bbox']
    else:
        bbox = None

    if bbox is not None:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

    cv2.putText(frame_display, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Tool Tracker", frame_display)
    return True

# Główna pętla
while True:
    if current_frame_image is None:
        if not load_frame(current_frame_index):
            break

    if tracking and not paused:
        success, bbox = tracker.update(current_frame_image)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame_display, p1, p2, (255, 0, 0), 2, 1)

            # Aktualizacja adnotacji
            update_annotation(coco_data, frame_id, bbox)

            # Zapisz klatki z adnotacjami
            save_frames(current_frame_image, frame_display, frame_id)
        else:
            cv2.putText(frame_display, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Śledzenie nie powiodło się. Zatrzymywanie śledzenia.")
            tracking = False
            bbox = None

    # Rysuj aktualny prostokąt podczas rysowania
    if drawing and rectangle is not None:
        x0, y0, x1, y1 = rectangle
        cv2.rectangle(frame_display, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv2.imshow("Tool Tracker", frame_display)

    if continuous_mode and not paused:
        key = cv2.waitKey(30) & 0xFF  # Automatyczne odświeżanie w trybie ciągłym
        current_frame_index += 1
        if current_frame_index >= total_frames:
            print("Osiągnięto koniec folderu z obrazami.")
            break
    else:
        key = cv2.waitKey(1) & 0xFF  # Oczekiwanie na naciśnięcie klawisza lub zdarzenie myszy

    # Obsługa klawiszy
    if key == ord('q'):
        break
    elif key == ord('n'):  # Klawisz 'N' dla następnego obrazu ze śledzeniem
        if bbox is not None and not tracking:
            tracker = cv2.TrackerMIL_create()  # Inicjalizacja trackera
            tracker.init(current_frame_image, bbox)
            tracking = True
        current_frame_index += 1
        if current_frame_index >= total_frames:
            print("Osiągnięto koniec folderu z obrazami.")
            break
        else:
            load_frame(current_frame_index)
    elif key == ord('s'):  # Klawisz 'S' do pominięcia następnego obrazu bez śledzenia
        tracking = False  # Zatrzymanie śledzenia
        current_frame_index += 1
        if current_frame_index >= total_frames:
            print("Osiągnięto koniec folderu z obrazami.")
            break
        else:
            load_frame(current_frame_index)
    elif key == ord('a'):  # Klawisz 'A' do powrotu do poprzedniego obrazu
        if current_frame_index > 0:
            current_frame_index -= 1
            load_frame(current_frame_index)
        else:
            print("Już na pierwszym obrazie.")
    elif key == ord(' '):  # Spacja do pauzowania/wznawiania
        paused = not paused
        if paused:
            print("Pauzowane.")
        else:
            print("Wznawianie.")
    elif key == ord('t'):
        # Przełącz tryb ciągły
        continuous_mode = not continuous_mode
        if continuous_mode:
            if bbox is not None and not tracking:
                tracker = cv2.TrackerMIL_create()  # Inicjalizacja trackera dla trybu ciągłego
                tracker.init(current_frame_image, bbox)
            tracking = True
            paused = False
            print("Tryb ciągły włączony.")
        else:
            tracking = False
            print("Tryb ciągły wyłączony.")

# Zakończenie
cv2.destroyAllWindows()

# Zapisz adnotacje do pliku JSON
with open(annotations_file, 'w') as f:
    json.dump(coco_data, f, indent=4)
print(f"Adnotacje zapisano do {annotations_file}")
