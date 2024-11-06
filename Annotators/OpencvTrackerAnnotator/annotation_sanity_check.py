import cv2
import os
import json
import random

# Ścieżka do folderu z obrazami i pliku z anotacjami
images_dir = "output/Raw_Images"
annotations_file = "output/annotations.json"
checked_dir = "output/CheckedAnnotations"  # Folder do zapisywania obrazów z anotacjami

# Upewnij się, że folder CheckedAnnotations istnieje
if not os.path.exists(checked_dir):
    os.makedirs(checked_dir)


# Funkcja ładująca anotacje z pliku JSON
def load_annotations(annotations_file):
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data


# Funkcja rysująca bounding box na obrazie
def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


# Funkcja losowo sprawdzająca klatki i rysująca bounding boxy
def check_random_frames(images_dir, annotations_file, step=20):
    # Załaduj anotacje
    coco_data = load_annotations(annotations_file)

    # Pobierz listę wszystkich obrazów z anotacji
    images = coco_data['images']
    annotations = coco_data['annotations']

    # Przechowuj mapowanie id obrazu na anotacje
    image_to_annotations = {ann['image_id']: ann for ann in annotations}

    # Wybierz co 20. obraz
    random_indices = list(range(0, len(images), step))
    random.shuffle(random_indices)

    for idx in random_indices:
        image_info = images[idx]
        image_id = image_info['id']
        image_path = os.path.join(images_dir, image_info['file_name'])

        # Załaduj obraz
        image = cv2.imread(image_path)
        if image is None:
            print(f"Nie można załadować obrazu {image_info['file_name']}.")
            continue

        # Znajdź powiązaną anotację
        if image_id in image_to_annotations:
            bbox = image_to_annotations[image_id]['bbox']
            draw_bounding_box(image, bbox)

            # Zapisz obraz z anotacją w folderze CheckedAnnotations
            output_image_path = os.path.join(checked_dir, image_info['file_name'])
            cv2.imwrite(output_image_path, image)
            print(f"Zapisano obraz {image_info['file_name']} z nałożonym bounding boxem do {output_image_path}.")
        else:
            print(f"Brak anotacji dla obrazu {image_info['file_name']}")

    print(f"Zakończono zapis obrazów z bounding boxami do folderu {checked_dir}")


# Sprawdź losowo wybrane klatki co 20. obraz i zapisz wyniki do folderu CheckedAnnotations
check_random_frames(images_dir, annotations_file)
