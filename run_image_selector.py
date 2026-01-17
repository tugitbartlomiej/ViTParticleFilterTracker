#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_image_selector.py

Skrypt do uruchomienia selekcji znaczących obrazów z poprawionymi modułami.
"""

import os
import sys
import argparse
import shutil

# Ustal ścieżkę do katalogu skryptów
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Kopiuj poprawione pliki do użycia
original_files = {
    'select_significant_images.py': os.path.join(script_dir, 'Annotators', 'Utils', 'SignificantImageSelector', 'select_significant_images.py'),
    'image_clustering.py': os.path.join(script_dir, 'Annotators', 'Utils', 'SignificantImageSelector', 'image_clustering.py')
}

fixed_files = {
    'select_significant_images_fixed.py': os.path.join(script_dir, 'Annotators', 'Utils', 'SignificantImageSelector', 'select_significant_images_fixed.py'),
    'image_clustering_fixed.py': os.path.join(script_dir, 'Annotators', 'Utils', 'SignificantImageSelector', 'image_clustering_fixed.py')
}

backup_files = {
    'select_significant_images.py.bak': os.path.join(script_dir, 'Annotators', 'Utils', 'SignificantImageSelector', 'select_significant_images.py.bak'),
    'image_clustering.py.bak': os.path.join(script_dir, 'Annotators', 'Utils', 'SignificantImageSelector', 'image_clustering.py.bak')
}


def backup_and_replace_files():
    """Zrób backup oryginalnych plików i zastąp je poprawionymi wersjami."""
    # Zrób backup oryginalnych plików
    for orig_name, orig_path in original_files.items():
        backup_path = f"{orig_path}.bak"
        if os.path.exists(orig_path):
            shutil.copy2(orig_path, backup_path)
            print(f"Utworzono kopię zapasową: {backup_path}")
    
    # Zastąp oryginalne pliki poprawionymi wersjami
    shutil.copy2(
        fixed_files['select_significant_images_fixed.py'],
        original_files['select_significant_images.py']
    )
    shutil.copy2(
        fixed_files['image_clustering_fixed.py'],
        original_files['image_clustering.py']
    )
    print("Zastąpiono pliki poprawionymi wersjami")


def run_selector():
    """Uruchom selektor z parametrami."""
    parser = argparse.ArgumentParser(
        description='Wybór znaczących obrazów do treningu modelu DETR na podstawie adnotacji YOLO'
    )
    
    # Argumenty główne
    parser.add_argument('--images_dir', default='Annotators/Datasets/Yolo/yolo_dataset_20250218_test/images/train', help='Katalog z obrazami')
    parser.add_argument('--labels_dir', default='Annotators/Datasets/Yolo/yolo_dataset_20250218_test/labels/train', help='Katalog z adnotacjami YOLO')
    parser.add_argument('--output_dir', default=os.path.join('Annotators', 'Datasets', 'Yolo', 'significant_images_results_test'), help='Folder docelowy dla wyników')
    
    # Ustawienia clusteringu
    parser.add_argument('--num_clusters', type=int, default=20, help='Liczba klastrów (domyślnie: 20)')
    parser.add_argument('--images_per_cluster', type=int, default=5, help='Liczba obrazów z każdego klastra (domyślnie: 5)')
    parser.add_argument('--annotation_weight', type=float, default=0.3, help='Waga cech adnotacji (0-1)')
    parser.add_argument('--clustering_method', choices=['kmeans', 'dbscan'], default='kmeans', help='Metoda clusteringu')
    
    # Ustawienia ekstrakcji cech
    parser.add_argument('--feature_method', choices=['histogram', 'hog'], default='histogram',
                      help='Metoda ekstrakcji cech wizualnych')
    
    # Opcje wizualizacji i eksportu
    parser.add_argument('--skip_visualization', action='store_true', help='Pomiń tworzenie wizualizacji')
    parser.add_argument('--skip_export', action='store_true', help='Pomiń eksport obrazów')
    parser.add_argument('--copy_annotations', action='store_true', help='Kopiuj pliki adnotacji')
    
    # Opcje debugowania
    parser.add_argument('--keep_temp_files', action='store_true', help='Zachowaj tymczasowe pliki z cechami')
    parser.add_argument('--debug', action='store_true', help='Tryb debugowania (więcej komunikatów)')
    
    args = parser.parse_args()
    
    # Zrób backup i zastąp pliki
    backup_and_replace_files()
    
    # Przygotuj polecenie
    cmd = [sys.executable, original_files['select_significant_images.py']]
    
    # Dodaj wszystkie argumenty
    for arg_name, arg_value in vars(args).items():
        if isinstance(arg_value, bool):
            if arg_value:
                cmd.append(f'--{arg_name}')
        else:
            cmd.append(f'--{arg_name}')
            cmd.append(str(arg_value))
    
    # Uruchom skrypt
    import subprocess
    try:
        subprocess.run(cmd, check=True)
        print("\nSkrypt zakończony pomyślnie!")
    except subprocess.CalledProcessError as e:
        print(f"\nBłąd podczas wykonywania skryptu: {e}")
    finally:
        # Przywróć oryginalne pliki
        for bak_name, bak_path in backup_files.items():
            orig_path = original_files[bak_name.replace('.bak', '')]
            if os.path.exists(bak_path):
                shutil.copy2(bak_path, orig_path)
                os.remove(bak_path)
        print("Przywrócono oryginalne pliki")


if __name__ == '__main__':
    run_selector()
