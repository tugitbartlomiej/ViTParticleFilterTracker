import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm


class COCOToYOLOLabelsConverter:
    """
    Konwertuje adnotacje z formatu COCO do formatu YOLO z podziałem na train/val/test.
    """

    def __init__(
        self,
        coco_json_path: str,
        images_dir: str,
        output_dir: str,
        class_mapping: Dict[int, int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        verbose: bool = True
    ):
        """
        Inicjalizacja konwertera.

        Args:
            coco_json_path: Ścieżka do pliku COCO JSON
            images_dir: Katalog z obrazami
            output_dir: Katalog wyjściowy główny
            class_mapping: Mapowanie kategorii COCO na YOLO
            train_ratio: Proporcja danych treningowych
            val_ratio: Proporcja danych walidacyjnych
            test_ratio: Proporcja danych testowych
            random_seed: Seed dla losowego podziału
            verbose: Czy wyświetlać szczegółowe informacje
        """
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping or {}
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.verbose = verbose

        # Sprawdź proporcje
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Suma proporcji train/val/test musi wynosić 1.0"

        # Sprawdź czy pliki/katalogi istnieją
        if not self.coco_json_path.exists():
            raise FileNotFoundError(f"Plik COCO JSON nie istnieje: {self.coco_json_path}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Katalog z obrazami nie istnieje: {self.images_dir}")

        # Utwórz strukturę katalogów
        self.setup_directory_structure()

        if self.verbose:
            print(f"COCO JSON: {self.coco_json_path}")
            print(f"Katalog obrazów: {self.images_dir}")
            print(f"Katalog wyjściowy: {self.output_dir}")
            print(f"Podział: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")

    def setup_directory_structure(self):
        """Utwórz strukturę katalogów dla YOLO."""
        # Katalogi dla etykiet
        self.train_labels_dir = self.output_dir / "labels" / "train"
        self.val_labels_dir = self.output_dir / "labels" / "val"
        self.test_labels_dir = self.output_dir / "labels" / "test"

        # Utwórz wszystkie katalogi
        for dir_path in [self.train_labels_dir, self.val_labels_dir, self.test_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_coco_data(self) -> Dict:
        """Wczytaj dane COCO z pliku JSON."""
        if self.verbose:
            print("\nWczytywanie danych COCO...")

        with open(self.coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        if self.verbose:
            print(f"Wczytano {len(coco_data.get('images', []))} obrazów")
            print(f"Wczytano {len(coco_data.get('annotations', []))} adnotacji")
            print(f"Wczytano {len(coco_data.get('categories', []))} kategorii")

        return coco_data

    def coco_to_yolo_bbox(
        self,
        coco_bbox: List[float],
        img_width: int,
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """
        Konwertuj bbox z formatu COCO na format YOLO.
        """
        x, y, w, h = coco_bbox

        # Oblicz center point
        center_x = x + w / 2
        center_y = y + h / 2

        # Znormalizuj do 0-1
        center_x_norm = center_x / img_width
        center_y_norm = center_y / img_height
        width_norm = w / img_width
        height_norm = h / img_height

        return center_x_norm, center_y_norm, width_norm, height_norm

    def get_image_dimensions(self, image_filename: str) -> Tuple[int, int]:
        """Pobierz wymiary obrazu."""
        image_path = self.images_dir / image_filename

        if not image_path.exists():
            raise FileNotFoundError(f"Obraz nie istnieje: {image_path}")

        try:
            with Image.open(image_path) as img:
                return img.size  # (width, height)
        except Exception as e:
            raise RuntimeError(f"Nie można odczytać wymiarów obrazu {image_path}: {e}")

    def split_dataset(self, image_ids: List[int]) -> Dict[str, List[int]]:
        """
        Podziel dataset na train/val/test.
        
        Args:
            image_ids: Lista ID obrazów
            
        Returns:
            Słownik z kluczami 'train', 'val', 'test' i listami ID
        """
        # Ustaw seed dla powtarzalności
        random.seed(self.random_seed)
        
        # Przetasuj listę
        shuffled_ids = image_ids.copy()
        random.shuffle(shuffled_ids)
        
        # Oblicz indeksy podziału
        total = len(shuffled_ids)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        # Podziel
        splits = {
            'train': shuffled_ids[:train_end],
            'val': shuffled_ids[train_end:val_end],
            'test': shuffled_ids[val_end:]
        }
        
        if self.verbose:
            print(f"\nPodział datasetu:")
            print(f"  Train: {len(splits['train'])} obrazów")
            print(f"  Val: {len(splits['val'])} obrazów")
            print(f"  Test: {len(splits['test'])} obrazów")
        
        return splits

    def convert_annotations(self) -> None:
        """Główna funkcja konwertująca adnotacje z COCO na YOLO."""
        # Wczytaj dane COCO
        coco_data = self.load_coco_data()

        # Utwórz mapowania
        images_by_id = {img['id']: img for img in coco_data['images']}
        annotations_by_image = {}

        # Pogrupuj adnotacje według image_id
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)

        # Podziel dataset
        valid_image_ids = [img_id for img_id in images_by_id.keys() 
                          if (self.images_dir / images_by_id[img_id]['file_name']).exists()]
        
        splits = self.split_dataset(valid_image_ids)
        
        # Statystyki
        stats = {
            'train': {'images': 0, 'annotations': 0, 'skipped': 0},
            'val': {'images': 0, 'annotations': 0, 'skipped': 0},
            'test': {'images': 0, 'annotations': 0, 'skipped': 0}
        }

        # Konwertuj dla każdego podziału
        for split_name, image_ids in splits.items():
            if self.verbose:
                print(f"\nKonwertowanie zestawu {split_name}...")
            
            # Wybierz odpowiedni katalog etykiet
            if split_name == 'train':
                labels_dir = self.train_labels_dir
            elif split_name == 'val':
                labels_dir = self.val_labels_dir
            else:
                labels_dir = self.test_labels_dir
            
            # Konwertuj obrazy w tym podziale
            for image_id in tqdm(image_ids, desc=f"Konwertowanie {split_name}"):
                image_info = images_by_id[image_id]
                image_filename = image_info['file_name']
                image_annotations = annotations_by_image.get(image_id, [])

                # Pobierz wymiary obrazu
                try:
                    img_width, img_height = self.get_image_dimensions(image_filename)
                except Exception as e:
                    if self.verbose:
                        print(f"BŁĄD: Nie można odczytać wymiarów obrazu {image_filename}: {e}")
                    stats[split_name]['skipped'] += 1
                    continue

                # Stwórz plik etykiet YOLO
                label_filename = Path(image_filename).stem + ".txt"
                label_path = labels_dir / label_filename

                yolo_lines = []
                valid_annotations = 0

                for annotation in image_annotations:
                    # Mapowanie kategorii
                    category_id = annotation['category_id']
                    yolo_class_id = self.class_mapping.get(category_id, 0)  # Domyślnie 0

                    # Konwertuj bbox
                    coco_bbox = annotation['bbox']
                    try:
                        center_x, center_y, width, height = self.coco_to_yolo_bbox(
                            coco_bbox, img_width, img_height
                        )

                        # Sprawdź czy bbox jest poprawny
                        if all(0 <= val <= 1 for val in [center_x, center_y, width, height]):
                            yolo_line = f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                            yolo_lines.append(yolo_line)
                            valid_annotations += 1

                    except Exception as e:
                        if self.verbose:
                            print(f"BŁĄD: Konwersja bbox dla {image_filename}: {e}")
                        continue

                # Zapisz plik etykiet
                try:
                    with open(label_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(yolo_lines))

                    stats[split_name]['images'] += 1
                    stats[split_name]['annotations'] += valid_annotations

                except Exception as e:
                    if self.verbose:
                        print(f"BŁĄD: Nie można zapisać etykiet dla {image_filename}: {e}")
                    stats[split_name]['skipped'] += 1

        # Wyświetl podsumowanie
        print("\n" + "=" * 70)
        print("PODSUMOWANIE KONWERSJI")
        print("=" * 70)
        total_images = sum(s['images'] for s in stats.values())
        total_annotations = sum(s['annotations'] for s in stats.values())
        total_skipped = sum(s['skipped'] for s in stats.values())
        
        print(f"{'Zestaw':<10} {'Obrazy':<10} {'Adnotacje':<12} {'Śr. adnotacji':<15} {'Pominięte':<10}")
        print("-" * 70)
        
        for split_name, split_stats in stats.items():
            avg_annotations = split_stats['annotations'] / split_stats['images'] if split_stats['images'] > 0 else 0
            print(f"{split_name:<10} {split_stats['images']:<10} {split_stats['annotations']:<12} "
                  f"{avg_annotations:<15.2f} {split_stats['skipped']:<10}")
        
        print("-" * 70)
        avg_total = total_annotations / total_images if total_images > 0 else 0
        print(f"{'RAZEM':<10} {total_images:<10} {total_annotations:<12} "
              f"{avg_total:<15.2f} {total_skipped:<10}")
        print("=" * 70)
        
        # Stwórz pliki list
        self.create_split_files(splits, images_by_id)

    def create_split_files(self, splits: Dict[str, List[int]], images_by_id: Dict[int, Dict]):
        """
        Stwórz pliki train.txt, val.txt, test.txt z listami obrazów.
        """
        for split_name, image_ids in splits.items():
            file_path = self.output_dir / f"{split_name}.txt"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for image_id in image_ids:
                    image_info = images_by_id[image_id]
                    # Zapisz pełną ścieżkę do obrazu
                    image_path = self.images_dir / image_info['file_name']
                    f.write(f"{image_path}\n")
            
            if self.verbose:
                print(f"Stworzono plik listy: {file_path}")

    def create_dataset_yaml(self, dataset_name: str = "surgical_tools") -> None:
        """
        Stwórz plik dataset.yaml dla YOLO.
        """
        yaml_file = self.output_dir / "dataset.yaml"

        # Przeczytaj klasy
        coco_data = self.load_coco_data()
        categories = coco_data.get('categories', [])
        
        if not categories:
            categories = [{'id': 0, 'name': 'surgical_tool'}]

        # Przygotuj mapowanie
        categories.sort(key=lambda x: x['id'])
        class_names = []
        for category in categories:
            category_id = category['id']
            yolo_class_id = self.class_mapping.get(category_id, 0)
            
            while len(class_names) <= yolo_class_id:
                class_names.append("unknown")
            
            class_names[yolo_class_id] = category['name']

        # Stwórz zawartość YAML z absolutnymi ścieżkami
        yaml_content = f"""# {dataset_name} Dataset Configuration
# Generated by COCOToYOLOLabelsConverter

# Dataset paths (absolute paths)
path: {self.output_dir}  # Dataset root directory

# Image paths (absolute paths to the actual image directory)
train: {self.images_dir}  # Train images (actual location)
val: {self.images_dir}    # Validation images (actual location)
test: {self.images_dir}   # Test images (actual location)

# Alternative: use image lists
train_list: {self.output_dir}/train.txt
val_list: {self.output_dir}/val.txt
test_list: {self.output_dir}/test.txt

# Classes
nc: {len(class_names)}  # Number of classes
names: {class_names}  # Class names

# Optional: class indices
"""
        
        # Dodaj indeksy klas
        for i, name in enumerate(class_names):
            yaml_content += f"# {i}: {name}\n"

        try:
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            if self.verbose:
                print(f"\nStworzono plik dataset.yaml: {yaml_file}")

        except Exception as e:
            print(f"BŁĄD: Nie można stworzyć pliku dataset.yaml: {e}")

    def create_data_yaml(self, dataset_name: str = "surgical_tools") -> None:
        """
        Stwórz alternatywny plik data.yaml z prostszą konfiguracją.
        """
        yaml_file = self.output_dir / "data.yaml"

        # Przeczytaj klasy
        coco_data = self.load_coco_data()
        categories = coco_data.get('categories', [])
        
        if not categories:
            categories = [{'id': 0, 'name': 'surgical_tool'}]

        # Przygotuj mapowanie
        categories.sort(key=lambda x: x['id'])
        class_names = []
        for category in categories:
            category_id = category['id']
            yolo_class_id = self.class_mapping.get(category_id, 0)
            
            while len(class_names) <= yolo_class_id:
                class_names.append("unknown")
            
            class_names[yolo_class_id] = category['name']

        # Prostsza konfiguracja
        yaml_content = f"""# YOLOv8 Dataset Configuration
train: {self.output_dir}/train.txt
val: {self.output_dir}/val.txt
test: {self.output_dir}/test.txt

nc: {len(class_names)}
names: {class_names}
"""

        try:
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            if self.verbose:
                print(f"Stworzono plik data.yaml: {yaml_file}")

        except Exception as e:
            print(f"BŁĄD: Nie można stworzyć pliku data.yaml: {e}")


def parse_class_mapping(mapping_str: str) -> Dict[int, int]:
    """Parsuj mapowanie klas z stringa."""
    if not mapping_str:
        return {}
    
    mapping = {}
    for pair in mapping_str.split(','):
        if ':' in pair:
            coco_id, yolo_id = pair.split(':')
            mapping[int(coco_id.strip())] = int(yolo_id.strip())
    
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description='Konwertuj adnotacje COCO na YOLO z podziałem na train/val/test'
    )
    
    # Wymagane argumenty
    parser.add_argument(
        '--coco_json',
        type=str,
        required=True,
        help='Ścieżka do pliku COCO JSON z adnotacjami'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Katalog z obrazami'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Katalog wyjściowy główny'
    )
    
    # Opcjonalne argumenty
    parser.add_argument(
        '--class_mapping',
        type=str,
        default="",
        help='Mapowanie kategorii COCO na YOLO'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Proporcja danych treningowych (domyślnie: 0.7)'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Proporcja danych walidacyjnych (domyślnie: 0.2)'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.1,
        help='Proporcja danych testowych (domyślnie: 0.1)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Seed dla losowego podziału (domyślnie: 42)'
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="surgical_tools",
        help='Nazwa datasetu'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Wyświetlaj szczegółowe informacje'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Tryb cichy'
    )

    args = parser.parse_args()

    # Sprawdź tryb verbose
    verbose = args.verbose and not args.quiet

    try:
        print("=" * 70)
        print("KONWERTER COCO DO YOLO Z PODZIAŁEM NA ZESTAWY")
        print("=" * 70)

        # Parsuj mapowanie klas
        class_mapping = parse_class_mapping(args.class_mapping)

        # Stwórz konwerter
        converter = COCOToYOLOLabelsConverter(
            coco_json_path=args.coco_json,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            class_mapping=class_mapping,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            verbose=verbose
        )

        # Konwertuj adnotacje
        converter.convert_annotations()

        # Stwórz pliki konfiguracyjne
        converter.create_dataset_yaml(args.dataset_name)
        converter.create_data_yaml(args.dataset_name)

        print("\nKonwersja zakończona pomyślnie!")
        print(f"Pliki wyjściowe znajdują się w: {args.output_dir}")

    except Exception as e:
        print(f"BŁĄD: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())