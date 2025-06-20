#!/usr/bin/env python3
"""
Skrypt do kopiowania etykiet YOLO obok obrazów, żeby YOLO mogło je znaleźć.
YOLO szuka etykiet w katalogu images/../labels/, więc musimy skopiować etykiety.
"""

import argparse
import shutil
from pathlib import Path


def copy_labels_to_images_structure(yolo_dataset_dir, images_dir, verbose=True):
    """
    Skopiuj etykiety obok obrazów w strukturze wymaganej przez YOLO.
    
    Args:
        yolo_dataset_dir: katalog główny datasetu YOLO (zawiera labels/)
        images_dir: katalog z obrazami
        verbose: czy wypisywać informacje diagnostyczne
    """
    yolo_dataset_dir = Path(yolo_dataset_dir)
    images_dir = Path(images_dir)
    labels_source_dir = yolo_dataset_dir / "labels"
    
    # Katalog docelowy dla etykiet - obok obrazów
    labels_target_dir = images_dir.parent / "labels"
    
    if not labels_source_dir.exists():
        raise FileNotFoundError(f"Katalog source labels nie istnieje: {labels_source_dir}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Katalog images nie istnieje: {images_dir}")
    
    print(f"🔧 Kopiuję etykiety obok obrazów...")
    print(f"  Source: {labels_source_dir}")
    print(f"  Target: {labels_target_dir}")
    
    # Stwórz katalog docelowy
    labels_target_dir.mkdir(exist_ok=True)
    
    # Skopiuj każdy split
    total_copied = 0
    for split in ['train', 'val', 'test']:
        source_split_dir = labels_source_dir / split
        target_split_dir = labels_target_dir / split
        
        if not source_split_dir.exists():
            if verbose:
                print(f"  ⚠️ {split}: source katalog {source_split_dir} nie istnieje - pomijam")
            continue
        
        # Znajdź pliki etykiet
        label_files = list(source_split_dir.glob("*.txt"))
        
        if not label_files:
            if verbose:
                print(f"  ⚠️ {split}: brak plików etykiet w {source_split_dir}")
            continue
        
        # Stwórz katalog docelowy dla split
        target_split_dir.mkdir(exist_ok=True)
        
        # Skopiuj pliki
        copied_count = 0
        for label_file in label_files:
            target_file = target_split_dir / label_file.name
            try:
                shutil.copy2(label_file, target_file)
                copied_count += 1
            except Exception as e:
                if verbose:
                    print(f"    ⚠️ Błąd kopiowania {label_file.name}: {e}")
        
        print(f"  ✓ {split}: skopiowano {copied_count} plików etykiet")
        total_copied += copied_count
    
    print(f"✅ Łącznie skopiowano {total_copied} plików etykiet")
    print(f"📁 Etykiety dostępne w: {labels_target_dir}")
    
    return labels_target_dir


def verify_labels_structure(images_dir, verbose=True):
    """Zweryfikuj czy etykiety są w odpowiednim miejscu."""
    images_dir = Path(images_dir)
    labels_dir = images_dir.parent / "labels"
    
    print(f"\n🔍 Weryfikuję strukturę etykiet...")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    
    if not labels_dir.exists():
        print(f"  ✗ Katalog labels nie istnieje: {labels_dir}")
        return False
    
    # Sprawdź każdy split
    for split in ['train', 'val', 'test']:
        split_dir = labels_dir / split
        if split_dir.exists():
            label_count = len(list(split_dir.glob("*.txt")))
            print(f"  ✓ labels/{split}/: {label_count} plików etykiet")
        else:
            print(f"  ⚠️ labels/{split}/: nie istnieje")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Skopiuj etykiety YOLO obok obrazów w strukturze wymaganej przez YOLO'
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
        help='Zweryfikuj strukturę po kopiowaniu'
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
        print("KOPIOWANIE ETYKIET YOLO")
        print("=" * 70)
        
        # Skopiuj etykiety
        labels_target_dir = copy_labels_to_images_structure(
            args.yolo_dataset_dir,
            args.images_dir,
            verbose=args.verbose
        )
        
        # Weryfikacja
        if args.verify:
            verify_labels_structure(args.images_dir, verbose=args.verbose)
        
        print(f"\n✅ Etykiety zostały pomyślnie skopiowane!")
        print(f"💡 Teraz YOLO powinno znajdować etykiety w: {labels_target_dir}")
        
    except Exception as e:
        print(f"❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
