#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
image_clustering.py

Skrypt do łączenia cech wizualnych i cech adnotacji,
a następnie clusteringu obrazów w celu wyboru reprezentatywnych przykładów.
"""

import os
import json
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class ImageClusterer:
    """Klasa do clusteringu obrazów na podstawie cech."""
    
    def __init__(self):
        """Inicjalizacja clusterer'a."""
        self.scaler = StandardScaler()
        self.pca = None
    
    def load_features(self, annotation_features_file, image_features_file):
        """
        Wczytanie cech z plików JSON.
        
        Args:
            annotation_features_file (str): Plik z cechami adnotacji
            image_features_file (str): Plik z cechami wizualnymi
            
        Returns:
            tuple: (image_paths, combined_features)
        """
        # Wczytanie cech adnotacji
        with open(annotation_features_file, 'r', encoding='utf-8') as f:
            annotation_features_dict = json.load(f)
        
        # Wczytanie cech wizualnych
        with open(image_features_file, 'r', encoding='utf-8') as f:
            image_features_dict = json.load(f)
        
        # Znajdź wspólne obrazy w obu zbiorach cech
        common_images = set(annotation_features_dict.keys()) & set(image_features_dict.keys())
        print(f"Znaleziono {len(common_images)} obrazów z obiema typami cech")
        
        if not common_images:
            raise ValueError("Brak wspólnych obrazów w obu zbiorach cech!")
        
        # Przygotuj listy obrazów i cech
        image_paths = []
        annotation_features_list = []
        image_features_list = []
        
        for image_path in common_images:
            image_paths.append(image_path)
            
            # Przygotuj wektor cech adnotacji
            annot_features = annotation_features_dict[image_path]
            annot_vector = np.array([
                annot_features['num_tooltips'],
                annot_features['avg_size'],
                annot_features['size_var'],
                annot_features['x_position'],
                annot_features['y_position'],
                annot_features['width'],
                annot_features['height'],
                annot_features['aspect_ratio']
            ])
            annotation_features_list.append(annot_vector)
            
            # Przygotuj wektor cech wizualnych
            vis_features = image_features_dict[image_path]
            if 'histogram' in vis_features:
                vis_vector = np.array(vis_features['histogram'])
                # Dodaj dodatkowe statystyki, jeśli dostępne
                if 'mean_h' in vis_features:
                    extras = np.array([
                        vis_features['mean_h'], 
                        vis_features['mean_s'], 
                        vis_features['mean_v'],
                        vis_features['std_h'], 
                        vis_features['std_s'], 
                        vis_features['std_v']
                    ])
                    vis_vector = np.concatenate([vis_vector, extras])
            elif 'hog' in vis_features:
                vis_vector = np.array(vis_features['hog'])
                # Dodaj dodatkowe statystyki, jeśli dostępne
                if 'mean_intensity' in vis_features:
                    extras = np.array([
                        vis_features['mean_intensity'],
                        vis_features['std_intensity'],
                        vis_features['mean_edge'],
                        vis_features['std_edge']
                    ])
                    vis_vector = np.concatenate([vis_vector, extras])
            
            image_features_list.append(vis_vector)
        
        # Konwersja na tablice NumPy
        annotation_features_array = np.array(annotation_features_list)
        image_features_array = np.array(image_features_list)
        
        return image_paths, annotation_features_array, image_features_array
    
    def combine_features(self, annotation_features, image_features, annotation_weight=0.5):
        """
        Łączenie cech adnotacji i cech wizualnych z odpowiednimi wagami.
        
        Args:
            annotation_features (ndarray): Tablica cech adnotacji
            image_features (ndarray): Tablica cech wizualnych
            annotation_weight (float): Waga cech adnotacji (0-1)
            
        Returns:
            ndarray: Połączone i znormalizowane cechy
        """
        # Normalizacja cech
        annotation_features_scaled = self.scaler.fit_transform(annotation_features)
        image_features_scaled = self.scaler.fit_transform(image_features)
        
        # Redukcja wymiarowości dla cech wizualnych, jeśli są bardzo wysokowymiarowe
        if image_features.shape[1] > 50:
            self.pca = PCA(n_components=min(50, image_features.shape[0]))
            image_features_scaled = self.pca.fit_transform(image_features_scaled)
            print(f"Zredukowano wymiarowość cech wizualnych do {image_features_scaled.shape[1]} wymiarów")
        
        # Zastosowanie wag
        combined_features = np.hstack([
            annotation_features_scaled * annotation_weight,
            image_features_scaled * (1 - annotation_weight)
        ])
        
        return combined_features
    
    def perform_clustering(self, features, n_clusters=20, method='kmeans'):
        """
        Przeprowadzenie clusteringu na podstawie cech.
        
        Args:
            features (ndarray): Tablica cech
            n_clusters (int): Liczba klastrów (dla K-means)
            method (str): Metoda clusteringu ('kmeans' lub 'dbscan')
            
        Returns:
            ndarray: Etykiety klastrów
        """
        if method == 'kmeans':
            # Zabezpieczenie przed zbyt dużą liczbą klastrów
            n_clusters = min(n_clusters, features.shape[0] - 1)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Zapisz centra klastrów
            self.cluster_centers = kmeans.cluster_centers_
            
        elif method == 'dbscan':
            # DBSCAN clustering (automatyczne określanie liczby klastrów)
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(features)
            
            # Obsługa outlierów (etykieta -1)
            n_outliers = np.sum(cluster_labels == -1)
            print(f"DBSCAN wykrył {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)} klastrów")
            print(f"Liczba outlierów: {n_outliers}")
            
            # Oblicz centra klastrów (średnie punktów w każdym klastrze)
            self.cluster_centers = []
            for label in np.unique(cluster_labels):
                if label != -1:  # Pomiń outliery
                    center = np.mean(features[cluster_labels == label], axis=0)
                    self.cluster_centers.append(center)
            self.cluster_centers = np.array(self.cluster_centers)
            
        else:
            raise ValueError(f"Nieznana metoda clusteringu: {method}")
        
        return cluster_labels
    
    def select_representative_images(self, image_paths, features, cluster_labels, images_per_cluster=5):
        """
        Wybór reprezentatywnych obrazów z każdego klastra.
        
        Args:
            image_paths (list): Lista ścieżek do obrazów
            features (ndarray): Tablica cech
            cluster_labels (ndarray): Etykiety klastrów
            images_per_cluster (int): Liczba obrazów do wyboru z każdego klastra
            
        Returns:
            list: Lista ścieżek do reprezentatywnych obrazów
        """
        selected_images = []
        
        # Dla każdego klastra
        for label in np.unique(cluster_labels):
            if label == -1:  # Pomiń outliery dla DBSCAN
                continue
                
            # Indeksy obrazów w tym klastrze
            cluster_indices = np.where(cluster_labels == label)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Znajdź centrum klastra
            try:
                center = self.cluster_centers[label] if hasattr(self, 'cluster_centers') else np.mean(features[cluster_indices], axis=0)
            except IndexError:
                # Dla DBSCAN, znalezienie centrum klastra może być problematyczne
                center = np.mean(features[cluster_indices], axis=0)
            
            # Oblicz odległości do centrum
            distances = []
            for idx in cluster_indices:
                dist = np.linalg.norm(features[idx] - center)
                distances.append((idx, dist))
            
            # Posortuj według odległości i wybierz najlepsze
            distances.sort(key=lambda x: x[1])
            to_select = min(images_per_cluster, len(distances))
            
            # Wybierz obrazy najbliższe centrum klastra
            for i in range(to_select):
                selected_images.append(image_paths[distances[i][0]])
        
        print(f"Wybrano {len(selected_images)} reprezentatywnych obrazów")
        return selected_images
    
    def save_results(self, image_paths, cluster_labels, selected_images, output_file):
        """
        Zapisanie wyników clusteringu do pliku JSON.
        
        Args:
            image_paths (list): Lista ścieżek do obrazów
            cluster_labels (ndarray): Etykiety klastrów
            selected_images (list): Lista wybranych ścieżek
            output_file (str): Ścieżka do pliku wyjściowego
        """
        results = {
            'clusters': {},
            'selected_images': selected_images
        }
        
        # Przypisz obrazy do klastrów
        for i, (path, label) in enumerate(zip(image_paths, cluster_labels)):
            if label not in results['clusters']:
                results['clusters'][int(label)] = []
            results['clusters'][int(label)].append(path)
        
        # Zapisz do pliku
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Zapisano wyniki clusteringu do: {output_file}")


def cluster_images_cmd():
    """Funkcja uruchamiana z linii poleceń."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clustering obrazów na podstawie cech')
    parser.add_argument('--annotation_features', required=True, help='Plik z cechami adnotacji (JSON)')
    parser.add_argument('--image_features', required=True, help='Plik z cechami wizualnymi (JSON)')
    parser.add_argument('--num_clusters', type=int, default=20, help='Liczba klastrów')
    parser.add_argument('--images_per_cluster', type=int, default=5, help='Liczba obrazów z każdego klastra')
    parser.add_argument('--annotation_weight', type=float, default=0.5, help='Waga cech adnotacji (0-1)')
    parser.add_argument('--clustering_method', choices=['kmeans', 'dbscan'], default='kmeans', help='Metoda clusteringu')
    parser.add_argument('--output', required=True, help='Plik wyjściowy z wynikami (JSON)')
    args = parser.parse_args()
    
    # Utworzenie clusterer'a
    clusterer = ImageClusterer()
    
    # Wczytanie cech
    image_paths, annotation_features, image_features = clusterer.load_features(
        args.annotation_features, args.image_features
    )
    
    # Połączenie cech
    combined_features = clusterer.combine_features(
        annotation_features, image_features, args.annotation_weight
    )
    
    # Clustering
    cluster_labels = clusterer.perform_clustering(
        combined_features, args.num_clusters, args.clustering_method
    )
    
    # Wybór reprezentatywnych obrazów
    selected_images = clusterer.select_representative_images(
        image_paths, combined_features, cluster_labels, args.images_per_cluster
    )
    
    # Zapisanie wyników
    clusterer.save_results(image_paths, cluster_labels, selected_images, args.output)


if __name__ == '__main__':
    cluster_images_cmd()
