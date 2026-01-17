#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
image_feature_extractor.py

Skrypt do ekstrakcji cech wizualnych z obrazów.
Pozwala na ekstrakcję różnych typów deskryptorów:
- histogramy kolorów
- deskryptory HOG
"""

import os
import glob
import json
import numpy as np
import cv2
from pathlib import Path


class ImageFeatureExtractor:
    """Klasa do ekstrakcji cech wizualnych z obrazów."""
    
    def __init__(self, method='histogram'):
        """
        Inicjalizacja ekstraktora cech obrazów.
        
        Args:
            method (str): Metoda ekstrakcji cech ('histogram' lub 'hog')
        """
        self.method = method
        self.supported_methods = ['histogram', 'hog']
        
        if self.method not in self.supported_methods:
            raise ValueError(f"Nieznana metoda ekstrakcji cech: {method}. "
                            f"Dostępne metody: {self.supported_methods}")
    
    def extract_features(self, image_path):
        """
        Ekstrakcja cech z obrazu.
        
        Args:
            image_path (str): Ścieżka do pliku obrazu
            
        Returns:
            dict: Słownik z cechami wizualnymi
        """
        try:
            if self.method == 'histogram':
                return self._extract_histogram_features(image_path)
            elif self.method == 'hog':
                return self._extract_hog_features(image_path)
        except Exception as e:
            print(f"Błąd podczas ekstrakcji cech z {image_path}: {e}")
            # Zwróć puste cechy
            if self.method == 'histogram':
                return {'histogram': np.zeros(32*3).tolist()}
            elif self.method == 'hog':
                return {'hog': np.zeros(36*4*4).tolist()}
    
    def _extract_histogram_features(self, image_path):
        """
        Ekstrakcja cech na podstawie histogramów kolorów.
        
        Args:
            image_path (str): Ścieżka do pliku obrazu
            
        Returns:
            dict: Słownik z cechami histogramu
        """
        # Wczytanie obrazu
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Nie można odczytać obrazu: {image_path}")
        
        # Konwersja do HSV dla lepszej reprezentacji kolorów
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Obliczenie histogramów dla każdego kanału
        hist_h = cv2.calcHist([image], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([image], [2], None, [32], [0, 256])
        
        # Normalizacja
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
        
        # Połączone cechy
        histogram = np.concatenate([
            hist_h.flatten(), 
            hist_s.flatten(), 
            hist_v.flatten()
        ])
        
        # Dodatkowo, oblicz średnie wartości kanałów
        mean_h = np.mean(image[:, :, 0])
        mean_s = np.mean(image[:, :, 1])
        mean_v = np.mean(image[:, :, 2])
        
        # Oblicz odchylenia standardowe kanałów
        std_h = np.std(image[:, :, 0])
        std_s = np.std(image[:, :, 1])
        std_v = np.std(image[:, :, 2])
        
        return {
            'histogram': histogram.tolist(),
            'mean_h': float(mean_h),
            'mean_s': float(mean_s),
            'mean_v': float(mean_v),
            'std_h': float(std_h),
            'std_s': float(std_s),
            'std_v': float(std_v)
        }
    
    def _extract_hog_features(self, image_path):
        """
        Ekstrakcja cech na podstawie deskryptorów HOG.
        
        Args:
            image_path (str): Ścieżka do pliku obrazu
            
        Returns:
            dict: Słownik z cechami HOG
        """
        # Wczytanie obrazu w skali szarości
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Nie można odczytać obrazu: {image_path}")
        
        # Zmiana rozmiaru obrazu dla spójności
        image = cv2.resize(image, (128, 128))
        
        # Konfiguracja deskryptora HOG
        win_size = (128, 128)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        
        # Ekstrakcja cech HOG
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(image)
        
        # Obliczenie prostych statystyk obrazu
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Obliczenie cech krawędzi
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_intensity = np.sqrt(sobelx**2 + sobely**2)
        mean_edge = np.mean(edge_intensity)
        std_edge = np.std(edge_intensity)
        
        return {
            'hog': hog_features.flatten().tolist(),
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'mean_edge': float(mean_edge),
            'std_edge': float(std_edge)
        }
    
    def extract_features_for_directory(self, images_dir, output_file=None):
        """
        Ekstrakcja cech dla wszystkich obrazów w katalogu.
        
        Args:
            images_dir (str): Ścieżka do katalogu z obrazami
            output_file (str, optional): Ścieżka do pliku wyjściowego (JSON)
            
        Returns:
            dict: Słownik z cechami wizualnymi dla każdego obrazu
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
            
            features = self.extract_features(image_path)
            if features:
                results[image_path] = features
        
        # Zapisanie wyników do pliku, jeśli podano ścieżkę
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Zapisano cechy wizualne do: {output_file}")
        
        return results


def extract_features_cmd():
    """Funkcja uruchamiana z linii poleceń."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ekstrakcja cech wizualnych z obrazów')
    parser.add_argument('--images_dir', required=True, help='Katalog z obrazami')
    parser.add_argument('--method', choices=['histogram', 'hog'], default='histogram',
                        help='Metoda ekstrakcji cech (histogram lub hog)')
    parser.add_argument('--output', help='Plik wyjściowy do zapisania cech (JSON)')
    args = parser.parse_args()
    
    # Ekstrakcja cech
    extractor = ImageFeatureExtractor(method=args.method)
    extractor.extract_features_for_directory(args.images_dir, args.output)


if __name__ == '__main__':
    extract_features_cmd()
