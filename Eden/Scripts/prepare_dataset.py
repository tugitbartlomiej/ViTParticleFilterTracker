#!/usr/bin/env python3
"""
Dataset Preparation Script for YOLO Training
Przygotowuje dataset z automatycznym podziałem train/val/test (80/15/5%)
"""

import argparse
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm


def find_files_by_extension(directory, extensions):
    """Znajdź wszystkie pliki z podanymi rozszerzeniami"""
    files = []
    for ext in extensions:
        files.extend(Path(directory).rglob(f"*.{ext}"))
    return [str(f) for f in files]

def create_image_label_mapping(images_dir, labels_dir):
    """Stwórz mapowanie między obrazami a etykietami"""
    print("🔍 Szukam obrazów i etykiet...")
    
    # Znajdź wszystkie obrazy
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
    images = find_files_by_extension(images_dir, image_extensions)
    print(f"📸 Znaleziono {len(images)} obrazów")
    
    # Znajdź wszystkie etykiety
    labels = find_files_by_extension(labels_dir, ['txt'])
    print(f"🏷️ Znaleziono {len(labels)} etykiet")
    
    # Stwórz mapowanie: nazwa_pliku_bez_ext -> ścieżka_etykiety
    label_map = {}
    for label_path in labels:
        label_name = Path(label_path).stem
        label_map[label_name] = label_path
    
    # Sprawdź które obrazy mają etykiety
    matched_pairs = []
    missing_labels = []
    
    for image_path in images:
        image_name = Path(image_path).stem
        if image_name in label_map:
            matched_pairs.append((image_path, label_map[image_name]))
        else:
            missing_labels.append(image_name)
    
    print(f"✅ Sparowane: {len(matched_pairs)} par obraz-etykieta")
    print(f"⚠️ Brak etykiet dla: {len(missing_labels)} obrazów")
    
    return matched_pairs, missing_labels

def split_dataset(pairs, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    """Podziel dataset na train/val/test"""
    
    # Sprawdź czy proporcje się zgadzają
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Proporcje muszą sumować się do 1.0"
    
    # Wymieszaj pary
    pairs_shuffled = pairs.copy()
    random.shuffle(pairs_shuffled)
    
    total_pairs = len(pairs_shuffled)
    train_count = int(total_pairs * train_ratio)
    val_count = int(total_pairs * val_ratio)
    test_count = total_pairs - train_count - val_count
    
    train_pairs = pairs_shuffled[:train_count]
    val_pairs = pairs_shuffled[train_count:train_count + val_count]
    test_pairs = pairs_shuffled[train_count + val_count:]
    
    print(f"📈 Podział datasetu:")
    print(f"  Train: {len(train_pairs)} par ({len(train_pairs)/total_pairs*100:.1f}%)")
    print(f"  Val: {len(val_pairs)} par ({len(val_pairs)/total_pairs*100:.1f}%)")
    print(f"  Test: {len(test_pairs)} par ({len(test_pairs)/total_pairs*100:.1f}%)")
    
    return {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }

def copy_files_to_unified_structure(splits, unified_dir):
    """Kopiuj pliki do unified struktury"""
    
    print(f"📁 Tworzę unified strukturę w: {unified_dir}")
    
    # Stwórz katalogi
    for split_name in ['train', 'val', 'test']:
        Path(unified_dir, 'images', split_name).mkdir(parents=True, exist_ok=True)
        Path(unified_dir, 'labels', split_name).mkdir(parents=True, exist_ok=True)
    
    # Kopiuj pliki dla każdego split'a
    copy_stats = defaultdict(lambda: {'images': 0, 'labels': 0, 'errors': 0})
    
    for split_name, pairs in splits.items():
        print(f"\n📋 Kopiuję pliki dla {split_name}...")
        
        # Użyj tqdm do pokazywania postępu
        for image_path, label_path in tqdm(pairs, desc=f"Kopiowanie {split_name}", unit="plik"):
            try:
                # Kopiuj obraz
                image_dest = Path(unified_dir, 'images', split_name, Path(image_path).name)
                shutil.copy2(image_path, image_dest)
                copy_stats[split_name]['images'] += 1
                
                # Kopiuj etykietę
                label_dest = Path(unified_dir, 'labels', split_name, Path(label_path).name)
                shutil.copy2(label_path, label_dest)
                copy_stats[split_name]['labels'] += 1
                
            except Exception as e:
                print(f"❌ Błąd kopiowania {image_path}: {e}")
                copy_stats[split_name]['errors'] += 1
    
    return copy_stats

def verify_unified_structure(unified_dir):
    """Zweryfikuj czy unified struktura jest poprawna"""
    
    print("\n🔍 Weryfikuję unified strukturę...")
    
    verification_results = {}
    
    for split_name in ['train', 'val', 'test']:
        images_dir = Path(unified_dir, 'images', split_name)
        labels_dir = Path(unified_dir, 'labels', split_name)
        
        if images_dir.exists() and labels_dir.exists():
            images_count = len(list(images_dir.glob('*')))
            labels_count = len(list(labels_dir.glob('*.txt')))
            
            verification_results[split_name] = {
                'images': images_count,
                'labels': labels_count,
                'balanced': images_count == labels_count
            }
            
            status = "✅" if images_count == labels_count else "⚠️"
            print(f"  {status} {split_name}: {images_count} obrazów, {labels_count} etykiet")
        else:
            verification_results[split_name] = {
                'images': 0,
                'labels': 0,
                'balanced': False
            }
            print(f"  ❌ {split_name}: brak katalogów")
    
    return verification_results

def create_dataset_yaml(unified_dir, nc=1, names=['tooltip']):
    """Stwórz dataset.yaml dla YOLO"""
    
    dataset_config = {
        'path': str(unified_dir),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': nc,
        'names': names
    }
    
    yaml_path = Path(unified_dir, 'dataset.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Utworzono dataset.yaml: {yaml_path}")
    
    return str(yaml_path)

def main():
    parser = argparse.ArgumentParser(description='Przygotuj dataset dla YOLO')
    parser.add_argument('--tmpdir', required=True, help='Katalog tymczasowy z rozpakowanym archiwum')
    parser.add_argument('--unified_dir', required=True, help='Katalog docelowy dla unified struktury')
    parser.add_argument('--seed', type=int, default=42, help='Seed dla randomizacji')
    
    args = parser.parse_args()
    
    # Ustaw seed dla powtarzalności
    random.seed(args.seed)
    
    print("=" * 60)
    print("🚀 PYTHON DATASET PREPARATION SCRIPT")
    print("=" * 60)
    
    start_time = time.time()
    
    # Znajdź katalogi w rozpakowanym archiwum
    print("🔍 Szukam katalogów...")
    
    # Znajdź katalog images
    images_dirs = list(Path(args.tmpdir).rglob('images'))
    if not images_dirs:
        print("❌ Nie znaleziono katalogu images!")
        sys.exit(1)
    images_dir = images_dirs[0]
    print(f"📁 Images dir: {images_dir}")
    
    # Znajdź katalog yolo_dataset
    yolo_dirs = list(Path(args.tmpdir).rglob('yolo_dataset'))
    if not yolo_dirs:
        print("❌ Nie znaleziono katalogu yolo_dataset!")
        sys.exit(1)
    labels_dir = yolo_dirs[0]
    print(f"📁 Labels dir: {labels_dir}")
    
    # Stwórz mapowanie obraz-etykieta
    pairs, missing_labels = create_image_label_mapping(images_dir, labels_dir)
    
    if not pairs:
        print("❌ Nie znaleziono żadnych par obraz-etykieta!")
        sys.exit(1)
    
    # Podziel dataset
    splits = split_dataset(pairs)
    
    # Kopiuj pliki do unified struktury
    copy_stats = copy_files_to_unified_structure(splits, args.unified_dir)
    
    # Zweryfikuj strukturę
    verification = verify_unified_structure(args.unified_dir)
    
    # Stwórz dataset.yaml
    yaml_path = create_dataset_yaml(args.unified_dir)
    
    # Podsumowanie
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ DATASET PREPARATION COMPLETED!")
    print("=" * 60)
    print(f"⏱️ Czas wykonania: {elapsed_time:.1f} sekund")
    print(f"📄 Dataset YAML: {yaml_path}")
    
    # Sprawdź czy wszystko się udało
    all_balanced = all(v['balanced'] for v in verification.values())
    if all_balanced:
        print("🎉 Wszystkie split'y są zrównoważone!")
        sys.exit(0)
    else:
        print("⚠️ Niektóre split'y mają niezrównoważone liczby plików")
        sys.exit(1)

if __name__ == "__main__":
    main()
