#!/usr/bin/env python3
"""
Skrypt do tworzenia plików train.txt, val.txt, test.txt na podstawie strukturi etykiet YOLO.
Dla każdego pliku etykiety szuka odpowiadającego obrazu i dodaje go do odpowiedniej listy.
"""

import argparse
from pathlib import Path


def find_image_for_label(label_file, images_dir):
    """
    Znajdź obraz odpowiadający plikowi etykiety.
    
    Args:
        label_file: ścieżka do pliku etykiety (np. image001.txt)
        images_dir: katalog z obrazami
    
    Returns:
        ścieżka do obrazu lub None jeśli nie znaleziono
    """
    # Pobierz nazwę podstawową bez rozszerzenia
    base_name = Path(label_file).stem
    
    # Sprawdź różne rozszerzenia obrazów
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    for ext in image_extensions:
        image_path = images_dir / f"{base_name}{ext}"
        if image_path.exists():
            return image_path
    
    return None


def create_image_lists(yolo_dataset_dir, images_dir, verbose=True):
    """
    Stwórz pliki train.txt, val.txt, test.txt na podstawie struktury etykiet.
    
    Args:
        yolo_dataset_dir: katalog główny datasetu YOLO (zawiera labels/)
        images_dir: katalog z obrazami
        verbose: czy wypisywać informacje diagnostyczne
    """
    yolo_dataset_dir = Path(yolo_dataset_dir)
    images_dir = Path(images_dir)
    labels_dir = yolo_dataset_dir / "labels"
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Katalog labels nie istnieje: {labels_dir}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Katalog images nie istnieje: {images_dir}")
    
    print(f"🔍 Tworzę listy obrazów...")
    print(f"  Labels dir: {labels_dir}")
    print(f"  Images dir: {images_dir}")
    
    for split in ['train', 'val', 'test']:
        split_labels_dir = labels_dir / split
        output_list_file = yolo_dataset_dir / f"{split}.txt"
        
        if not split_labels_dir.exists():
            if verbose:
                print(f"  ⚠️ {split}: katalog {split_labels_dir} nie istnieje - pomijam")
            continue
        
        # Znajdź wszystkie pliki etykiet
        label_files = list(split_labels_dir.glob("*.txt"))
        
        if not label_files:
            if verbose:
                print(f"  ⚠️ {split}: brak plików etykiet w {split_labels_dir}")
            continue
        
        # Znajdź odpowiadające obrazy
        image_paths = []
        missing_images = []
        
        for label_file in label_files:
            image_path = find_image_for_label(label_file, images_dir)
            if image_path:
                image_paths.append(str(image_path))
            else:
                missing_images.append(label_file.name)
        
        # Zapisz listę obrazów
        with open(output_list_file, 'w', encoding='utf-8') as f:
            for img_path in sorted(image_paths):
                f.write(f"{img_path}\n")
        
        print(f"  ✓ {split}.txt: {len(image_paths)} obrazów")
        
        if missing_images and verbose:
            print(f"    ⚠️ Nie znaleziono obrazów dla {len(missing_images)} etykiet:")
            for missing in missing_images[:5]:  # Pokaż pierwszych 5
                print(f"      - {missing}")
            if len(missing_images) > 5:
                print(f"      ... i {len(missing_images) - 5} więcej")


def verify_lists(yolo_dataset_dir, verbose=True):
    """Zweryfikuj utworzone listy obrazów."""
    yolo_dataset_dir = Path(yolo_dataset_dir)
    
    print(f"\n🔍 Weryfikuję utworzone listy...")
    
    for split in ['train', 'val', 'test']:
        list_file = yolo_dataset_dir / f"{split}.txt"
        
        if not list_file.exists():
            print(f"  ⚠️ {split}.txt: nie istnieje")
            continue
        
        # Policz linie
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"  ✓ {split}.txt: {len(lines)} obrazów")
        
        # Sprawdź kilka pierwszych obrazów
        if verbose and lines:
            print(f"    Przykładowe ścieżki:")
            for img_path in lines[:3]:
                if Path(img_path).exists():
                    print(f"      ✓ {Path(img_path).name}")
                else:
                    print(f"      ✗ {Path(img_path).name} - nie istnieje!")


def main():
    parser = argparse.ArgumentParser(
        description='Stwórz pliki train.txt, val.txt, test.txt na podstawie struktury etykiet YOLO'
    )
    
    parser.add_argument(
        '--yolo_dataset_dir',
        type=str,
        required=True,
        help='Katalog główny datasetu YOLO (zawierający labels/)'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help='Katalog z obrazami'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='Zweryfikuj utworzone listy'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Wypisuj szczegółowe informacje'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("TWORZENIE LIST OBRAZÓW YOLO")
        print("=" * 70)
        
        # Stwórz listy obrazów
        create_image_lists(
            args.yolo_dataset_dir,
            args.images_dir,
            verbose=args.verbose
        )
        
        # Weryfikacja
        if args.verify:
            verify_lists(args.yolo_dataset_dir, verbose=args.verbose)
        
        print(f"\n✅ Listy obrazów zostały pomyślnie utworzone!")
        
    except Exception as e:
        print(f"❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
