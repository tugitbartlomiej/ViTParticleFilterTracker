#!/usr/bin/env python3
"""
Skrypt do kopiowania struktury folderów z archiwum Dataset 2025.06.06
bez kopiowania obrazów - tylko struktura katalogów i pliki YAML.
"""

import argparse
import os
import shutil
from pathlib import Path


def copy_directory_structure(source_dir, target_dir, include_yaml=True, exclude_images=True):
    """
    Kopiuje strukturę katalogów z source_dir do target_dir.
    
    Args:
        source_dir: Katalog źródłowy
        target_dir: Katalog docelowy
        include_yaml: Czy kopiować pliki YAML
        exclude_images: Czy wykluczyć pliki obrazów
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Katalog źródłowy nie istnieje: {source_path}")
      # Rozszerzenia plików obrazów do wykluczenia
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    # Rozszerzenia plików do wykluczenia (etykiety, listy obrazów)
    text_extensions = {'.txt'}
    
    # Rozszerzenia plików do skopiowania
    yaml_extensions = {'.yaml', '.yml'}    json_extensions = {'.json'}
    config_extensions = {'.py', '.slurm', '.sh', '.bat'}
    
    copied_files = []
    created_dirs = []
    skipped_files = []
    
    print(f"📂 Kopiuję strukturę z: {source_path}")
    print(f"📁 Do katalogu: {target_path}")
    print(f"📄 Kopiowanie YAML: {'✓' if include_yaml else '✗'}")
    print(f"🖼️ Wykluczanie obrazów: {'✓' if exclude_images else '✗'}")
    print(f"📝 Wykluczanie plików TXT: ✓")
    print()
    
    # Przejdź przez wszystkie pliki i katalogi
    for root, dirs, files in os.walk(source_path):
        root_path = Path(root)
        
        # Oblicz względną ścieżkę
        rel_path = root_path.relative_to(source_path)
        target_root = target_path / rel_path
        
        # Stwórz katalog w docelowej lokalizacji
        target_root.mkdir(parents=True, exist_ok=True)
        if not target_root in created_dirs:
            created_dirs.append(target_root)
            print(f"📁 Utworzono katalog: {rel_path}")
        
        # Przetwórz pliki w tym katalogu
        for file in files:
            source_file = root_path / file
            target_file = target_root / file
            file_ext = source_file.suffix.lower()
            
            should_copy = False
            reason = ""
              # Sprawdź czy to plik obrazu
            if exclude_images and file_ext in image_extensions:
                should_copy = False
                reason = "obraz (wykluczony)"
                skipped_files.append((source_file, reason))
            
            # Sprawdź czy to plik TXT (etykiety, listy)
            elif file_ext in text_extensions:
                should_copy = False
                reason = "TXT (wykluczony)"
                skipped_files.append((source_file, reason))
            
            # Sprawdź czy to plik YAML
            elif file_ext in yaml_extensions:
                should_copy = include_yaml
                reason = "YAML" if should_copy else "YAML (wykluczony)"
            
            # Sprawdź czy to plik JSON
            elif file_ext in json_extensions:
                should_copy = True
                reason = "JSON"
            
            # Sprawdź czy to plik konfiguracyjny
            elif file_ext in config_extensions:
                should_copy = True
                reason = "konfiguracja"
            
            # Inne pliki
            else:
                should_copy = True
                reason = "inny"
            
            if should_copy:
                try:
                    shutil.copy2(source_file, target_file)
                    copied_files.append((source_file, target_file, reason))
                    print(f"  📄 Skopiowano: {rel_path / file} ({reason})")
                except Exception as e:
                    print(f"  ❌ Błąd kopiowania {rel_path / file}: {e}")
            else:
                if len(skipped_files) <= 10:  # Pokaż tylko pierwsze 10 pominiętych plików
                    print(f"  ⏭️ Pominięto: {rel_path / file} ({reason})")
    
    # Podsumowanie
    print(f"\n{'='*60}")
    print("PODSUMOWANIE KOPIOWANIA")
    print(f"{'='*60}")
    print(f"📁 Utworzono katalogów: {len(created_dirs)}")
    print(f"📄 Skopiowano plików: {len(copied_files)}")
    print(f"⏭️ Pominięto plików: {len(skipped_files)}")
    
    # Szczegóły skopiowanych plików
    if copied_files:
        print(f"\n📋 Skopiowane pliki wg typu:")
        file_types = {}
        for source_file, target_file, reason in copied_files:
            file_types[reason] = file_types.get(reason, 0) + 1
        
        for file_type, count in file_types.items():
            print(f"  {file_type}: {count} plików")
    
    if len(skipped_files) > 10:
        print(f"\n⏭️ Pominięto {len(skipped_files)} plików (głównie obrazy)")
    
    return target_path


def main():
    parser = argparse.ArgumentParser(
        description='Kopiuj strukturę folderów Dataset 2025.06.06 bez obrazów'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default='datasets_20250606.tar/datasets_20250606',
        help='Katalog źródłowy (domyślnie: datasets_20250606.tar/datasets_20250606)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='dataset_structure_copy',
        help='Katalog docelowy (domyślnie: dataset_structure_copy)'
    )
    
    parser.add_argument(
        '--include-yaml',
        action='store_true',
        default=True,
        help='Kopiuj pliki YAML (domyślnie: True)'
    )
    
    parser.add_argument(
        '--include-images',
        action='store_true',
        help='Kopiuj także obrazy (domyślnie: False)'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("KOPIOWANIE STRUKTURY DATASET 2025.06.06")
        print("=" * 60)
        
        # Sprawdź czy katalog źródłowy istnieje
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"❌ Katalog źródłowy nie istnieje: {source_path}")
            print("Sprawdź czy archiwum zostało rozpakowane.")
            return 1
        
        # Kopiuj strukturę
        target_dir = copy_directory_structure(
            source_dir=args.source,
            target_dir=args.target,
            include_yaml=args.include_yaml,
            exclude_images=not args.include_images
        )
        
        print(f"\n✅ Struktura została skopiowana do: {target_dir}")
        print(f"📁 Możesz teraz sprawdzić zawartość katalogu: {target_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
