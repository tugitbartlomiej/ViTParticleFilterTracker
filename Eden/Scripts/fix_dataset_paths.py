#!/usr/bin/env python3
"""
Skrypt do automatycznej poprawy ścieżek w plikach konfiguracyjnych YOLO po rozpakowaniu datasetu.
Używany na klastrze Eden po rozpakowaniu archiwum do TMPDIR.
"""

import argparse
import re
import shutil
from pathlib import Path


def find_directories(tmpdir):
    """Znajdź główne katalogi w rozpakownym archiwum."""
    tmpdir = Path(tmpdir)
    
    # Znajdź katalog yolo_dataset
    yolo_dirs = list(tmpdir.glob("**/yolo_dataset"))
    if not yolo_dirs:
        raise FileNotFoundError("Nie znaleziono katalogu yolo_dataset w TMPDIR")
    yolo_dataset_dir = yolo_dirs[0]
    
    # Znajdź katalog images  
    images_dirs = list(tmpdir.glob("**/images"))
    if not images_dirs:
        raise FileNotFoundError("Nie znaleziono katalogu images w TMPDIR")
    images_dir = images_dirs[0]
    
    # Znajdź plik dataset.yaml
    dataset_yamls = list(tmpdir.glob("**/dataset.yaml"))
    if not dataset_yamls:
        raise FileNotFoundError("Nie znaleziono pliku dataset.yaml w TMPDIR")
    dataset_yaml = dataset_yamls[0]
    
    return yolo_dataset_dir, images_dir, dataset_yaml


def backup_file(file_path):
    """Stwórz kopię zapasową pliku."""
    backup_path = str(file_path) + ".backup"
    shutil.copy2(file_path, backup_path)
    print(f"📄 Backup: {backup_path}")
    return backup_path


def update_dataset_yaml(dataset_yaml, yolo_dataset_dir):
    """Aktualizuj ścieżki w pliku dataset.yaml."""
    print(f"🔧 Aktualizuję {dataset_yaml}...")
    
    # Backup
    backup_file(dataset_yaml)
    
    # Czytaj zawartość
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Aktualizuj ścieżki
    old_content = content
    
    # Aktualizuj path
    content = re.sub(r'^path:\s*.*$', f'path: {yolo_dataset_dir}', content, flags=re.MULTILINE)
    
    # Aktualizuj ścieżki do plików list
    content = re.sub(r'^train:\s*.*\.txt\s*$', f'train: {yolo_dataset_dir}/train.txt', content, flags=re.MULTILINE)
    content = re.sub(r'^val:\s*.*\.txt\s*$', f'val: {yolo_dataset_dir}/val.txt', content, flags=re.MULTILINE)
    content = re.sub(r'^test:\s*.*\.txt\s*$', f'test: {yolo_dataset_dir}/test.txt', content, flags=re.MULTILINE)
    
    # Zapisz
    with open(dataset_yaml, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if content != old_content:
        print("  ✓ Zaktualizowano ścieżki w dataset.yaml")
    else:
        print("  ℹ️ Brak zmian w dataset.yaml")


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


def create_or_update_image_lists(yolo_dataset_dir, images_dir):
    """Stwórz lub aktualizuj pliki list obrazów na podstawie etykiet."""
    print(f"🔧 Tworzę/aktualizuję listy obrazów w {yolo_dataset_dir}...")
    
    labels_dir = yolo_dataset_dir / "labels"
    if not labels_dir.exists():
        print(f"  ⚠️ Katalog labels nie istnieje: {labels_dir}")
        return
    
    for split in ['train', 'val', 'test']:
        split_labels_dir = labels_dir / split
        list_path = yolo_dataset_dir / f"{split}.txt"
        
        if not split_labels_dir.exists():
            print(f"  ⚠️ {split}: katalog {split_labels_dir} nie istnieje - pomijam")
            continue
        
        # Znajdź wszystkie pliki etykiet
        label_files = list(split_labels_dir.glob("*.txt"))
        
        if not label_files:
            print(f"  ⚠️ {split}: brak plików etykiet w {split_labels_dir}")
            continue
        
        print(f"  → Tworzę {split}.txt z {len(label_files)} etykiet")
        
        # Backup istniejącego pliku
        if list_path.exists():
            backup_file(list_path)
        
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
        with open(list_path, 'w', encoding='utf-8') as f:
            for img_path in sorted(image_paths):
                f.write(f"{img_path}\n")
        
        print(f"    ✓ Utworzono {split}.txt z {len(image_paths)} obrazami")
        
        if missing_images:
            print(f"    ⚠️ Nie znaleziono obrazów dla {len(missing_images)} etykiet")


def update_image_lists(yolo_dataset_dir, images_dir):
    """Aktualizuj ścieżki w plikach list obrazów (stara funkcja dla kompatybilności)."""
    # Używaj nowej funkcji, która tworzy listy na podstawie etykiet
    create_or_update_image_lists(yolo_dataset_dir, images_dir)


def verify_structure(yolo_dataset_dir, images_dir, dataset_yaml):
    """Weryfikuj czy struktura datasetu jest poprawna."""
    print("\n🔍 Weryfikuję strukturę datasetu...")
    
    # Sprawdź katalogi
    print(f"📁 YOLO dataset: {yolo_dataset_dir}")
    print(f"🖼️ Images: {images_dir}")
    
    # Sprawdź katalogi labels
    labels_dir = yolo_dataset_dir / "labels"
    if labels_dir.exists():
        for split in ['train', 'val', 'test']:
            split_dir = labels_dir / split
            if split_dir.exists():
                label_count = len(list(split_dir.glob("*.txt")))
                print(f"  ✓ labels/{split}/: {label_count} plików etykiet")
            else:
                print(f"  ⚠️ labels/{split}/: nie istnieje")
    else:
        print(f"  ✗ {labels_dir}: nie istnieje")
      # Sprawdź pliki list i przykładowe obrazy
    print("\n📋 Weryfikuję listy obrazów:")
    for list_file in ['train.txt', 'val.txt', 'test.txt']:
        list_path = yolo_dataset_dir / list_file
        
        if list_path.exists():
            with open(list_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            print(f"  ✓ {list_file}: {len(lines)} obrazów")
            
            # Sprawdź kilka pierwszych obrazów
            if lines:
                print(f"    Pierwsze 3 obrazy:")
                for img_path in lines[:3]:
                    if Path(img_path).exists():
                        print(f"      ✓ {Path(img_path).name}")
                    else:
                        print(f"      ✗ {Path(img_path).name} - nie istnieje!")
        else:
            print(f"  ✗ {list_file}: nie istnieje")
    
    # Pokaż zaktualizowaną konfigurację
    print(f"\n📄 Zaktualizowana konfiguracja dataset.yaml:")
    with open(dataset_yaml, 'r') as f:
        lines = f.readlines()[:15]
        for i, line in enumerate(lines, 1):
            print(f"  {i:2d}: {line.rstrip()}")


def main():
    parser = argparse.ArgumentParser(
        description='Popraw ścieżki w plikach konfiguracyjnych YOLO po rozpakowaniu archiwum'
    )
    
    parser.add_argument(
        '--tmpdir',
        type=str,
        required=True,
        help='Katalog tymczasowy gdzie został rozpakowany dataset'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='Weryfikuj strukturę po aktualizacji'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("POPRAWA ŚCIEŻEK DATASETU YOLO")
        print("=" * 70)
        
        # Znajdź katalogi
        yolo_dataset_dir, images_dir, dataset_yaml = find_directories(args.tmpdir)
        
        print(f"🔍 Znalezione lokalizacje:")
        print(f"  YOLO dataset: {yolo_dataset_dir}")
        print(f"  Images: {images_dir}")
        print(f"  Dataset YAML: {dataset_yaml}")
          # Aktualizuj pliki
        update_dataset_yaml(dataset_yaml, yolo_dataset_dir)
        create_or_update_image_lists(yolo_dataset_dir, images_dir)
        
        print("\n✅ Ścieżki zostały pomyślnie zaktualizowane!")
        
        # Weryfikacja
        if args.verify:
            verify_structure(yolo_dataset_dir, images_dir, dataset_yaml)
        
        print(f"\n📝 Zaktualizowany plik konfiguracyjny: {dataset_yaml}")
        
    except Exception as e:
        print(f"❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
