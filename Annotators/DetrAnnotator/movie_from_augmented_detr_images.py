import json
import os
import random
from pathlib import Path

import cv2


def load_annotations(annotation_file):
    """
    Wczytuje plik z annotacjami w formacie COCO.

    Args:
        annotation_file: Ścieżka do pliku JSON z annotacjami

    Returns:
        Słownik z annotacjami, gdzie kluczami są nazwy plików obrazów
    """
    print(f"Wczytywanie annotacji z pliku: {annotation_file}")

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Tworzymy słownik mapujący ID obrazu na jego nazwę pliku
    image_id_to_name = {}
    for image in data['images']:
        image_id_to_name[image['id']] = image['file_name']

    # Tworzymy słownik kategorii ID -> nazwa
    category_id_to_name = {}
    if 'categories' in data:
        for category in data['categories']:
            category_id_to_name[category['id']] = category['name']

    # Organizujemy annotacje według nazw plików
    annotations_by_image = {}

    for annotation in data['annotations']:
        image_id = annotation['image_id']

        if image_id in image_id_to_name:
            image_name = image_id_to_name[image_id]

            if image_name not in annotations_by_image:
                annotations_by_image[image_name] = []

            # Dodajemy nazwę kategorii, jeśli dostępna
            if 'category_id' in annotation and annotation['category_id'] in category_id_to_name:
                annotation['category_name'] = category_id_to_name[annotation['category_id']]
            else:
                annotation['category_name'] = f"ID: {annotation.get('category_id', 'unknown')}"

            annotations_by_image[image_name].append(annotation)

    print(f"Wczytano annotacje dla {len(annotations_by_image)} obrazów")
    return annotations_by_image


def draw_bounding_boxes(image, annotations):
    """
    Rysuje bounding boxy na obrazie na podstawie annotacji.

    Args:
        image: Obraz numpy array (odczytany przez cv2)
        annotations: Lista annotacji dla danego obrazu

    Returns:
        Obraz z narysowanymi bounding boxami
    """
    img = image.copy()

    # Generowanie różnych kolorów dla różnych kategorii
    colors = {}

    for annotation in annotations:
        # Sprawdzamy, czy annotacja zawiera bbox
        if 'bbox' not in annotation:
            continue

        # W formacie COCO, bbox to [x, y, width, height]
        x, y, w, h = map(int, annotation['bbox'])

        # Pobieramy kategorię lub ID
        category = annotation.get('category_name', str(annotation.get('category_id', 'unknown')))

        # Przypisujemy stały kolor dla danej kategorii
        if category not in colors:
            colors[category] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

        color = colors[category]

        # Rysujemy prostokąt
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Dodajemy etykietę
        cv2.putText(
            img,
            category,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return img


def create_video_from_images(image_dir, annotations_by_image, output_file, fps=10, width=None, height=None):
    """
    Tworzy film z obrazów z naniesionymi bounding boxami.

    Args:
        image_dir: Katalog zawierający obrazy
        annotations_by_image: Słownik z annotacjami dla każdego obrazu
        output_file: Ścieżka do wyjściowego pliku wideo
        fps: Liczba klatek na sekundę w filmie
        width: Opcjonalna szerokość klatek (jeśli None, użyta będzie oryginalna)
        height: Opcjonalna wysokość klatek (jeśli None, użyta będzie oryginalna)
    """
    print(f"Tworzenie filmu z obrazów w katalogu: {image_dir}")

    # Zbieramy wszystkie pliki obrazów
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in valid_extensions:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(image_dir).glob(f'*{ext.upper()}')))

    # Sortujemy pliki
    image_files = sorted(image_files)

    if not image_files:
        print(f"Nie znaleziono obrazów w katalogu: {image_dir}")
        return

    print(f"Znaleziono {len(image_files)} obrazów")

    # Odczytujemy pierwszy obraz, aby określić wymiary wideo
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Nie można odczytać obrazu: {image_files[0]}")
        return

    frame_height, frame_width = first_image.shape[:2]

    # Jeśli podano konkretne wymiary, zmieniamy rozmiar
    if width is not None and height is not None:
        frame_width = width
        frame_height = height

    # Inicjalizujemy obiekt VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Kodek MP4
    video_writer = cv2.VideoWriter(
        output_file,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    # Iterujemy przez wszystkie obrazy
    for i, image_path in enumerate(image_files):
        # Statusy postępu
        if i % 100 == 0:
            print(f"Przetwarzanie obrazu {i + 1}/{len(image_files)}")

        # Odczytujemy obraz
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Nie można odczytać obrazu: {image_path}")
            continue

        # Zmiana rozmiaru, jeśli potrzebna
        if width is not None and height is not None:
            image = cv2.resize(image, (width, height))

        # Pobieramy nazwę pliku
        image_name = os.path.basename(image_path)

        # Sprawdzamy czy mamy annotacje dla tego obrazu
        if image_name in annotations_by_image:
            # Rysujemy bounding boxy
            image = draw_bounding_boxes(image, annotations_by_image[image_name])
        else:
            print(f"Brak annotacji dla obrazu: {image_name}")

        # Dodajemy klatkę do filmu
        video_writer.write(image)

    # Finalizujemy film
    video_writer.release()
    print(f"Film został zapisany do: {output_file}")


def main():
    # Ścieżki podane przez użytkownika
    image_dir = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\DetrAnnotator\augmented_dataset\images"
    annotation_file = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\DetrAnnotator\augmented_dataset\augmented_annotations_20241115_175519.json"

    # Ścieżka do wyjściowego pliku wideo
    output_dir = os.path.dirname(annotation_file)
    output_file = os.path.join(output_dir, "augmented_dataset_video.mp4")

    # Wczytujemy annotacje
    annotations_by_image = load_annotations(annotation_file)

    # Tworzymy film
    create_video_from_images(
        image_dir,
        annotations_by_image,
        output_file,
        fps=15  # Możesz dostosować FPS
    )


if __name__ == "__main__":
    main()