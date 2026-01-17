#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
select_significant_images.py

Główny skrypt do wyboru najbardziej znaczących obrazów do treningu modelu DETR.
Integruje cały proces od ekstrakcji cech po wybór obrazów.
"""

import os
import argparse
import json
import tempfile
from pathlib import Path

# Importuj poszczególne moduły
from annotation_feature_extractor import YOLOAnnotationFeatureExtractor
from image_feature_extractor import ImageFeatureExtractor
from image_clustering import ImageClusterer
from results_visualizer import ResultsVisualizer


def main():
    """Główna funkcja skryptu."""
    parser = argparse.ArgumentParser(
        description='Wybór znaczących obrazów do treningu modelu DETR na podstawie adnotacji YOLO'
    )
    
    # Argumenty główne
    parser.add_argument('--images_dir', default='Annotators/Datasets/Yolo/yolo_dataset_20250218/images/train', help='Katalog z obrazami')
    parser.add_argument('--labels_dir', default='Annotators/Datasets/Yolo/yolo_dataset_20250218/labels/train', help='Katalog z adnotacjami YOLO (jeśli różny od images_dir)')
    parser.add_argument('--output_dir', default=os.path.join('Annotators', 'Datasets', 'Yolo', 'significant_images_results'), help='Folder docelowy dla wyników')
    
    # Ustawienia clusteringu
    parser.add_argument('--num_clusters', type=int, default=20, help='Liczba klastrów (domyślnie: 20)')
    parser.add_argument('--images_per_cluster', type=int, default=5, help='Liczba obrazów z każdego klastra (domyślnie: 5)')
    parser.add_argument('--annotation_weight', type=float, default=0.5, help='Waga cech adnotacji (0-1)')
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
    
    # Jeśli nie podano katalogu z adnotacjami, użyj domyślnie tego samego co obrazy
    labels_dir = args.labels_dir if args.labels_dir else args.images_dir

    # Utwórz folder docelowy
    os.makedirs(args.output_dir, exist_ok=True)
    # Tymczasowe pliki dla cech - zawsze używamy folderu w projekcie
    temp_dir = os.path.join(args.output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    annotation_features_file = os.path.join(temp_dir, 'annotation_features.json')
    image_features_file = os.path.join(temp_dir, 'image_features.json')
    clustering_results_file = os.path.join(temp_dir, 'clustering_results.json')
    combined_features_file = os.path.join(temp_dir, 'combined_features.json')

    # 1. Ekstrakcja cech z adnotacji YOLO
    print("\n[1/4] Ekstrakcja cech z adnotacji YOLO...")
    annotation_extractor = YOLOAnnotationFeatureExtractor(labels_dir)
    annotation_features = annotation_extractor.extract_features_for_directory(
        args.images_dir, annotation_features_file
    )

    # 2. Ekstrakcja cech wizualnych z obrazów
    print("\n[2/4] Ekstrakcja cech wizualnych z obrazów...")
    image_extractor = ImageFeatureExtractor(method=args.feature_method)
    image_features = image_extractor.extract_features_for_directory(
        args.images_dir, image_features_file
    )

    # 3. Łączenie cech i clustering
    print("\n[3/4] Łączenie cech i clustering obrazów...")
    clusterer = ImageClusterer()

    # Wczytanie cech
    image_paths, annotation_feats_array, image_feats_array = clusterer.load_features(
        annotation_features_file, image_features_file
    )

    # Połączenie cech
    combined_features = clusterer.combine_features(
        annotation_feats_array, image_feats_array, args.annotation_weight
    )

    # Zapisz połączone cechy (do wizualizacji)
    with open(combined_features_file, 'w', encoding='utf-8') as f:
        json.dump({
            'image_paths': image_paths,
            'combined_features': combined_features.tolist()
        }, f)

    # Clustering
    cluster_labels = clusterer.perform_clustering(
        combined_features, args.num_clusters, args.clustering_method
    )

    # Wybór reprezentatywnych obrazów
    selected_images = clusterer.select_representative_images(
        image_paths, combined_features, cluster_labels, args.images_per_cluster
    )

    # Zapisanie wyników clusteringu
    clusterer.save_results(image_paths, cluster_labels, selected_images, clustering_results_file)

    # 4. Wizualizacja i eksport wyników
    print("\n[4/4] Wizualizacja i eksport wyników...")
    visualizer = ResultsVisualizer(clustering_results_file)

    # Wizualizacje
    if not args.skip_visualization:
        dist_viz_path = os.path.join(args.output_dir, 'cluster_distribution.png')
        visualizer.visualize_cluster_distribution(dist_viz_path)

        pca_viz_path = os.path.join(args.output_dir, 'cluster_visualization_pca.png')
        visualizer.create_clusters_visualization(combined_features_file, pca_viz_path, 'pca')

        montage_path = os.path.join(args.output_dir, 'selected_images_montage.png')
        visualizer.create_montage(montage_path)

        report_path = os.path.join(args.output_dir, 'report.html')
        visualizer.create_report(report_path, combined_features_file)

    # Eksport obrazów
    if not args.skip_export:
        images_dir = os.path.join(args.output_dir, 'selected_images')
        visualizer.export_selected_images(images_dir, args.copy_annotations)
    print("\nGotowe! Wybrano najważniejsze obrazy do treningu DETR.")
    print(f"Wyniki zostały zapisane w katalogu: {args.output_dir}")


if __name__ == '__main__':
    main()
