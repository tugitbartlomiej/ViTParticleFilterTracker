import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import yaml
from tqdm import tqdm


class COCOToYOLOConverter:
    """
    Konwertuje dataset w formacie COCO na format YOLO.
    """

    def __init__(
            self,
            coco_json_path: str,
            coco_images_dir: str,
            yolo_output_dir: str,
            train_val_split: float = 0.9,
            random_seed: int = 42
    ):
        """
        Inicjalizacja konwertera.

        Args:
            coco_json_path: Ścieżka do pliku JSON COCO
            coco_images_dir: Katalog z obrazami
            yolo_output_dir: Katalog wyjściowy dla datasetu YOLO
            train_val_split: Podział na zbiór treningowy/walidacyjny (0.9 = 90% train, 10% val)
            random_seed: Ziarno dla generatora liczb losowych
        """
        self.coco_json_path = Path(coco_json_path)
        self.coco_images_dir = Path(coco_images_dir)
        self.yolo_output_dir = Path(yolo_output_dir)
        self.train_val_split = train_val_split
        self.random_seed = random_seed

        # Ustaw ziarno dla powtarzalności
        random.seed(self.random_seed)

        # Wczytaj dane COCO
        print(f"[COCO->YOLO] Wczytywanie pliku COCO: {self.coco_json_path}")
        with open(self.coco_json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Utwórz mapowania
        self._create_mappings()

    def _create_mappings(self):
        """Tworzy mapowania ID -> nazwa dla kategorii i obrazów."""
        # Mapowanie kategorii
        self.category_mapping = {}
        self.category_names = []

        for idx, category in enumerate(self.coco_data['categories']):
            # YOLO używa indeksów zaczynających się od 0
            self.category_mapping[category['id']] = idx
            self.category_names.append(category['name'])

        print(f"[COCO->YOLO] Znaleziono {len(self.category_names)} kategorii: {self.category_names}")

        # Mapowanie obrazów
        self.image_id_to_filename = {}
        for image in self.coco_data['images']:
            self.image_id_to_filename[image['id']] = image['file_name']

        # Grupuj adnotacje według obrazu
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)

    def _convert_bbox_to_yolo(self, coco_bbox: List[float], img_width: int, img_height: int) -> Tuple[
        float, float, float, float]:
        """
        Konwertuje bounding box z formatu COCO na format YOLO.

        COCO: [x_top_left, y_top_left, width, height]
        YOLO: [x_center, y_center, width, height] (znormalizowane do 0-1)

        Args:
            coco_bbox: Bounding box w formacie COCO
            img_width: Szerokość obrazu
            img_height: Wysokość obrazu

        Returns:
            Tuple z wartościami YOLO (x_center, y_center, width, height)
        """
        x, y, w, h = coco_bbox

        # Oblicz środek
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height

        # Znormalizuj wymiary
        width = w / img_width
        height = h / img_height

        # Upewnij się, że wartości są w zakresie [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return x_center, y_center, width, height

    def convert(self):
        """Wykonuje konwersję z formatu COCO na format YOLO."""
        print("\n" + "=" * 80)
        print("[COCO->YOLO] ROZPOCZYNANIE KONWERSJI")
        print("=" * 80)

        # Utwórz strukturę katalogów YOLO
        self._create_yolo_structure()

        # Podziel obrazy na train/val
        all_image_ids = list(self.annotations_by_image.keys())
        train_size = int(len(all_image_ids) * self.train_val_split)

        # Losowe mieszanie
        random.shuffle(all_image_ids)

        train_ids = all_image_ids[:train_size]
        val_ids = all_image_ids[train_size:]

        print(f"[COCO->YOLO] Podział datasetu: {len(train_ids)} train, {len(val_ids)} val")

        # Przetwórz obrazy treningowe
        print("\n[COCO->YOLO] Przetwarzanie zbioru treningowego...")
        self._process_images(train_ids, 'train')

        # Przetwórz obrazy walidacyjne
        print("\n[COCO->YOLO] Przetwarzanie zbioru walidacyjnego...")
        self._process_images(val_ids, 'val')

        # Utwórz plik konfiguracyjny YOLO
        self._create_yaml_config()

        print(f"\n[COCO->YOLO] ✅ Konwersja zakończona! Dataset YOLO zapisany w: {self.yolo_output_dir}")

    def _create_yolo_structure(self):
        """Tworzy strukturę katalogów dla datasetu YOLO."""
        # Katalogi dla obrazów i etykiet
        for split in ['train', 'val']:
            (self.yolo_output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def _process_images(self, image_ids: List[int], split: str):
        """
        Przetwarza obrazy i tworzy pliki etykiet YOLO.

        Args:
            image_ids: Lista ID obrazów do przetworzenia
            split: 'train' lub 'val'
        """
        images_dir = self.yolo_output_dir / 'images' / split
        labels_dir = self.yolo_output_dir / 'labels' / split

        skipped_images = 0
        processed_images = 0
        total_annotations = 0

        for img_id in tqdm(image_ids, desc=f"Przetwarzanie {split}"):
            filename = self.image_id_to_filename.get(img_id)
            if not filename:
                print(f"[WARNING] Brak nazwy pliku dla image_id {img_id}")
                skipped_images += 1
                continue

            # Ścieżka do oryginalnego obrazu
            src_image_path = self.coco_images_dir / filename
            if not src_image_path.exists():
                print(f"[WARNING] Obraz nie istnieje: {src_image_path}")
                skipped_images += 1
                continue

            # Kopiuj obraz do katalogu YOLO
            dst_image_path = images_dir / filename
            try:
                shutil.copy2(src_image_path, dst_image_path)
            except Exception as e:
                print(f"[ERROR] Błąd kopiowania {src_image_path}: {e}")
                skipped_images += 1
                continue

            # Pobierz informacje o obrazie
            image_info = next((img for img in self.coco_data['images'] if img['id'] == img_id), None)
            if not image_info:
                print(f"[WARNING] Brak informacji o obrazie dla ID {img_id}")
                skipped_images += 1
                continue

            img_width = image_info['width']
            img_height = image_info['height']

            # Utwórz plik etykiet YOLO
            label_filename = Path(filename).stem + '.txt'
            label_path = labels_dir / label_filename

            annotations = self.annotations_by_image.get(img_id, [])

            with open(label_path, 'w') as f:
                for ann in annotations:
                    # Konwertuj kategorie ID na indeks YOLO
                    coco_cat_id = ann['category_id']
                    yolo_class_id = self.category_mapping.get(coco_cat_id, 0)

                    # Konwertuj bounding box
                    x_center, y_center, width, height = self._convert_bbox_to_yolo(
                        ann['bbox'], img_width, img_height
                    )

                    # Zapisz w formacie YOLO
                    f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    total_annotations += 1

            processed_images += 1

        print(f"[COCO->YOLO] {split.upper()} - Przetworzono: {processed_images} obrazów, "
              f"Pominięto: {skipped_images}, Adnotacji: {total_annotations}")

    def _create_yaml_config(self):
        """Tworzy plik konfiguracyjny YAML dla YOLO."""
        yaml_path = self.yolo_output_dir / 'dataset.yaml'

        config = {
            'path': str(self.yolo_output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': None,  # Możesz dodać zbiór testowy później

            # Liczba klas
            'nc': len(self.category_names),

            # Nazwy klas
            'names': self.category_names
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n[COCO->YOLO] Plik konfiguracyjny YAML zapisany: {yaml_path}")

        # Wyświetl zawartość
        print("\n[COCO->YOLO] Zawartość dataset.yaml:")
        print("-" * 50)
        with open(yaml_path, 'r') as f:
            print(f.read())
        print("-" * 50)


def main():
    """Główna funkcja uruchamiająca konwersję."""
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")

    # --- Ścieżki z domyślnymi wartościami ---
    parser.add_argument("--coco_json_path", type=str,
                        default=r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/augmented_coco_450-14660_20250417_133558.json",
                        help="Path to COCO annotations JSON file.")

    parser.add_argument("--coco_images_dir", type=str,
                        default=r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/images",
                        help="Directory containing COCO dataset images.")

    parser.add_argument("--yolo_output_dir", type=str,
                        default=r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Datasets/Yolo",
                        help="Output directory for YOLO dataset.")

    # --- Parametry konwersji ---
    parser.add_argument("--train_val_split", type=float, default=0.9,
                        help="Train/validation split ratio (0.9 = 90%% train, 10%% val).")

    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # --- Parametry treningowe YOLO (do wyświetlenia sugerowanej komendy) ---
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt",
                        help="YOLOv8 model to use for training (for display purposes).")

    parser.add_argument("--yolo_epochs", type=int, default=100,
                        help="Number of epochs for YOLO training (for display purposes).")

    parser.add_argument("--yolo_imgsz", type=int, default=640,
                        help="Image size for YOLO training (for display purposes).")

    parser.add_argument("--yolo_batch", type=int, default=16,
                        help="Batch size for YOLO training (for display purposes).")

    args = parser.parse_args()

    # --- Sprawdzenie ścieżek ---
    if not Path(args.coco_json_path).is_file():
        print(f"[ERROR] Plik JSON COCO nie istnieje: {args.coco_json_path}")
        exit(1)

    if not Path(args.coco_images_dir).is_dir():
        print(f"[ERROR] Katalog z obrazami nie istnieje: {args.coco_images_dir}")
        exit(1)

    # --- Wyświetl konfigurację ---
    print("=" * 80)
    print("COCO TO YOLO CONVERTER")
    print("=" * 80)
    print(f"[CONFIG] COCO JSON path: {args.coco_json_path}")
    print(f"[CONFIG] COCO images directory: {args.coco_images_dir}")
    print(f"[CONFIG] YOLO output directory: {args.yolo_output_dir}")
    print(f"[CONFIG] Train/val split: {args.train_val_split}")
    print(f"[CONFIG] Random seed: {args.random_seed}")
    print("=" * 80)

    # Utwórz katalog wyjściowy
    Path(args.yolo_output_dir).mkdir(parents=True, exist_ok=True)

    # Utwórz konwerter
    converter = COCOToYOLOConverter(
        coco_json_path=args.coco_json_path,
        coco_images_dir=args.coco_images_dir,
        yolo_output_dir=args.yolo_output_dir,
        train_val_split=args.train_val_split,
        random_seed=args.random_seed
    )

    # Wykonaj konwersję
    converter.convert()

    # Wyświetl sugerowaną komendę treningową
    print("\n" + "=" * 80)
    print("SUGEROWANA KOMENDA TRENINGU YOLO")
    print("=" * 80)
    print(f"yolo train model={args.yolo_model} data={args.yolo_output_dir}/dataset.yaml "
          f"epochs={args.yolo_epochs} imgsz={args.yolo_imgsz} batch={args.yolo_batch}")
    print("=" * 80)


if __name__ == "__main__":
    main()