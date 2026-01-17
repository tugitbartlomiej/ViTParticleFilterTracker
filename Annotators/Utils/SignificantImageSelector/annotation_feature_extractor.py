#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
annotation_feature_extractor.py

Skrypt do ekstrakcji cech z adnotacji YOLO, zakładając jedną klasę 'ToolTip'
(końcówka narzędzia chirurgicznego).
"""

import os
import numpy as np
import glob
import json
from pathlib import Path


class YOLOAnnotationFeatureExtractor:
    """Klasa do ekstrakcji cech z adnotacji YOLO."""
    
    def __init__(self, labels_dir):
        """
        Inicjalizacja ekstraktora cech adnotacji.
        
        Args:
            labels_dir (str): Ścieżka do katalogu z plikami adnotacji YOLO
        """
        self.labels_dir = labels_dir
    
    def parse_annotation_file(self, annotation_path):
        """
        Parsowanie pliku adnotacji YOLO i zwrócenie listy obiektów.
        
        Args:
            annotation_path (str): Ścieżka do pliku adnotacji
            
        Returns:
            list: Lista słowników opisujących wykryte obiekty
        """
        objects = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id, x_center, y_center, width, height
                        obj = {
                            'class_id': int(parts[0]),  # W tym przypadku zawsze 0 (ToolTip)
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        }
                        objects.append(obj)
        return objects
    
    def extract_features(self, annotation_path):
        """
        Ekstrakcja cech z pliku adnotacji.
        
        Args:
            annotation_path (str): Ścieżka do pliku adnotacji
            
        Returns:
            dict: Słownik cech adnotacji
        """
        objects = self.parse_annotation_file(annotation_path)
        
        # Jeśli brak obiektów, zwróć domyślne wartości
        if not objects:
            return {
                'num_tooltips': 0,
                'avg_size': 0.0,
                'size_var': 0.0,
                'x_position': 0.5,  # Środek obrazu
                'y_position': 0.5,  # Środek obrazu
                'width': 0.0,
                'height': 0.0,
                'aspect_ratio': 1.0
            }
        
        # Statystyki dla wykrytych końcówek narzędzi
        widths = [obj['width'] for obj in objects]
        heights = [obj['height'] for obj in objects]
        x_positions = [obj['x_center'] for obj in objects]
        y_positions = [obj['y_center'] for obj in objects]
        sizes = [w * h for w, h in zip(widths, heights)]
        aspect_ratios = [w / h if h > 0 else 1.0 for w, h in zip(widths, heights)]
        
        features = {
            'num_tooltips': len(objects),
            'avg_size': np.mean(sizes),
            'size_var': np.var(sizes) if len(objects) > 1 else 0.0,
            'x_position': np.mean(x_positions),
            'y_position': np.mean(y_positions),
            'width': np.mean(widths),
            'height': np.mean(heights),
            'aspect_ratio': np.mean(aspect_ratios)
        }
        
        return features
    
    def extract_features_for_image(self, image_path):
        """
        Ekstrakcja cech adnotacji dla konkretnego obrazu.
        
        Args:
            image_path (str): Ścieżka do pliku obrazu
            
        Returns:
            dict: Słownik cech adnotacji lub None, jeśli brak adnotacji
        """
        # Wyznaczenie ścieżki do pliku adnotacji
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(self.labels_dir, f"{image_basename}.txt")
        
        # Ekstrakcja cech, jeśli istnieje plik adnotacji
        if os.path.exists(annotation_path):
            return self.extract_features(annotation_path)
        else:
            print(f"Brak pliku adnotacji dla: {image_path}")
            return None
    
    def extract_features_for_directory(self, images_dir, output_file=None):
        """
        Ekstrakcja cech dla wszystkich obrazów w katalogu.
        
        Args:
            images_dir (str): Ścieżka do katalogu z obrazami
            output_file (str, optional): Ścieżka do pliku wyjściowego (JSON)
            
        Returns:
            dict: Słownik z cechami adnotacji dla każdego obrazu
        """
        # Znalezienie wszystkich obrazów
        image_extensions = ['jpg', 'jpeg', 'png']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(images_dir, f'*.{ext}')))
        
        print(f"Znaleziono {len(image_paths)} obrazów w {images_dir}")
        
        # Ekstrakcja cech dla każdego obrazu
        results = {}
        for i, image_path in enumerate(image_paths):
            if i % 100 == 0:
                print(f"Przetwarzanie obrazu {i}/{len(image_paths)}")
            
            features = self.extract_features_for_image(image_path)
            if features:
                results[image_path] = features
        
        # Zapisanie wyników do pliku, jeśli podano ścieżkę
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Zapisano cechy adnotacji do: {output_file}")
        
        return results


def extract_features_cmd():
    """Funkcja uruchamiana z linii poleceń."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ekstrakcja cech z adnotacji YOLO')
    parser.add_argument('--images_dir', required=True, help='Katalog z obrazami')
    parser.add_argument('--labels_dir', help='Katalog z adnotacjami YOLO (opcjonalnie)')
    parser.add_argument('--output', help='Plik wyjściowy do zapisania cech (JSON)')
    args = parser.parse_args()
    
    # Jeśli nie podano katalogu z adnotacjami, użyj domyślnie tego samego co obrazy
    labels_dir = args.labels_dir if args.labels_dir else args.images_dir
    
    # Ekstrakcja cech
    extractor = YOLOAnnotationFeatureExtractor(labels_dir)
    extractor.extract_features_for_directory(args.images_dir, args.output)


if __name__ == '__main__':
    extract_features_cmd()
