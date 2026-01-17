#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
results_visualizer.py

Skrypt do wizualizacji wyników clusteringu i eksportu wybranych zdjęć.
Umożliwia generowanie różnych typów wizualizacji klastrów oraz
kopiowanie wybranych zdjęć do folderu docelowego.
"""

import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import cv2


class ResultsVisualizer:
    """Klasa do wizualizacji i eksportu wyników clusteringu."""
    
    def __init__(self, clustering_results_file):
        """
        Inicjalizacja wizualizatora wyników.
        
        Args:
            clustering_results_file (str): Ścieżka do pliku z wynikami clusteringu
        """
        # Wczytaj wyniki clusteringu
        with open(clustering_results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # Ekstrakcja informacji
        self.clusters = self.results['clusters']
        self.selected_images = self.results['selected_images']
        
        print(f"Wczytano dane o {len(self.clusters)} klastrach")
        print(f"Liczba wybranych obrazów: {len(self.selected_images)}")
    
    def visualize_cluster_distribution(self, output_file):
        """
        Wizualizacja rozkładu obrazów w klastrach.
        
        Args:
            output_file (str): Ścieżka do zapisania pliku
        """
        # Liczba obrazów w każdym klastrze
        cluster_sizes = [len(images) for _, images in self.clusters.items()]
        cluster_ids = list(self.clusters.keys())
        
        plt.figure(figsize=(12, 6))
        plt.bar(cluster_ids, cluster_sizes)
        plt.title('Rozkład obrazów w klastrach')
        plt.xlabel('ID klastra')
        plt.ylabel('Liczba obrazów')
        plt.grid(True, alpha=0.3)
        
        # Zapisz wykres
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Zapisano wizualizację rozkładu klastrów do: {output_file}")
    
    def create_clusters_visualization(self, features_file, output_file, method='pca'):
        """
        Wizualizacja klastrów w przestrzeni 2D.
        
        Args:
            features_file (str): Plik z połączonymi cechami
            output_file (str): Ścieżka do zapisania pliku
            method (str): Metoda redukcji wymiarowości ('pca' lub 'tsne')
        """
        # Wczytaj cechy
        try:
            with open(features_file, 'r', encoding='utf-8') as f:
                features_data = json.load(f)
            
            features = np.array(features_data['combined_features'])
            image_paths = features_data['image_paths']
            
            # Przygotuj mapowanie obrazów do etykiet klastrów
            image_to_cluster = {}
            for cluster_id, cluster_images in self.clusters.items():
                for img_path in cluster_images:
                    image_to_cluster[img_path] = int(cluster_id)
            
            # Przygotuj etykiety klastrów dla wszystkich obrazów
            labels = np.array([image_to_cluster.get(path, -1) for path in image_paths])
            
            # Redukcja wymiarowości
            if method == 'pca':
                reducer = PCA(n_components=2)
                reduced_features = reducer.fit_transform(features)
                title = 'Wizualizacja klastrów (PCA)'
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42)
                reduced_features = reducer.fit_transform(features)
                title = 'Wizualizacja klastrów (t-SNE)'
            else:
                raise ValueError(f"Nieznana metoda redukcji wymiarowości: {method}")
            
            # Wizualizacja
            plt.figure(figsize=(12, 10))
            
            # Rysuj punkty dla każdego klastra
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == -1:  # Pomiń outliery
                    continue
                    
                mask = labels == label
                plt.scatter(
                    reduced_features[mask, 0],
                    reduced_features[mask, 1],
                    label=f'Cluster {label}',
                    alpha=0.7,
                    s=20
                )
            
            # Oznacz wybrane obrazy
            selected_indices = [i for i, path in enumerate(image_paths) if path in self.selected_images]
            plt.scatter(
                reduced_features[selected_indices, 0],
                reduced_features[selected_indices, 1],
                c='red',
                marker='x',
                s=100,
                label='Wybrane obrazy'
            )
            
            plt.title(title)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Zapisz wykres
            plt.savefig(output_file)
            plt.close()
            
            print(f"Zapisano wizualizację klastrów do: {output_file}")
            
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Błąd podczas tworzenia wizualizacji klastrów: {e}")
            print("Pomijam wizualizację klastrów.")
    
    def create_montage(self, output_file, max_images=20, rows=4, cols=5):
        """
        Stworzenie montażu wybranych obrazów.
        
        Args:
            output_file (str): Ścieżka do zapisania pliku
            max_images (int): Maksymalna liczba obrazów w montażu
            rows (int): Liczba wierszy
            cols (int): Liczba kolumn
        """
        # Wybierz podzbiór obrazów do montażu
        images_to_show = self.selected_images[:max_images]
        num_images = len(images_to_show)
        
        if num_images == 0:
            print("Brak obrazów do utworzenia montażu")
            return
        
        # Dostosuj liczbę wierszy i kolumn
        if num_images < rows * cols:
            rows = int(np.ceil(np.sqrt(num_images)))
            cols = int(np.ceil(num_images / rows))
        
        # Ustaw rozmiar obrazów w montażu
        target_size = (224, 224)
        
        # Utwórz pustą planszę
        montage = np.zeros((rows * target_size[0], cols * target_size[1], 3), dtype=np.uint8)
        
        for i, img_path in enumerate(images_to_show):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            
            try:
                # Wczytaj i zmień rozmiar obrazu
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, target_size)
                
                # Umieść na montażu
                montage[
                    row * target_size[0]:(row + 1) * target_size[0],
                    col * target_size[1]:(col + 1) * target_size[1]
                ] = img
                
                # Dodaj etykietę (nazwę pliku)
                filename = os.path.basename(img_path)
                cv2.putText(
                    montage,
                    filename[:10] + "...",
                    (col * target_size[1] + 5, row * target_size[0] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
                
            except Exception as e:
                print(f"Błąd podczas dodawania obrazu {img_path} do montażu: {e}")
        
        # Zapisz montaż
        cv2.imwrite(output_file, montage)
        print(f"Zapisano montaż wybranych obrazów do: {output_file}")
        
        # Konwertuj także na wersję PNG dla lepszej jakości
        png_output = os.path.splitext(output_file)[0] + ".png"
        cv2.imwrite(png_output, montage)
    
    def export_selected_images(self, output_dir, copy_annotations=True):
        """
        Kopiowanie wybranych obrazów i ich adnotacji do folderu docelowego.
        
        Args:
            output_dir (str): Ścieżka do folderu docelowego
            copy_annotations (bool): Czy kopiować pliki adnotacji
        """
        # Utwórz folder docelowy
        os.makedirs(output_dir, exist_ok=True)
        
        # Kopiuj wybrane obrazy
        for image_path in self.selected_images:
            # Sprawdź, czy plik istnieje
            if not os.path.exists(image_path):
                print(f"Ostrzeżenie: Nie znaleziono pliku {image_path}")
                continue
                
            # Kopiuj obraz
            filename = os.path.basename(image_path)
            dest_path = os.path.join(output_dir, filename)
            shutil.copy2(image_path, dest_path)
            
            # Kopiuj adnotację, jeśli wymagane
            if copy_annotations:
                # Wyznacz ścieżkę do pliku adnotacji
                annotation_path = os.path.splitext(image_path)[0] + '.txt'
                if os.path.exists(annotation_path):
                    annotation_filename = os.path.basename(annotation_path)
                    dest_annotation = os.path.join(output_dir, annotation_filename)
                    shutil.copy2(annotation_path, dest_annotation)
        
        print(f"Skopiowano {len(self.selected_images)} obrazów do {output_dir}")
        
        # Utwórz plik z listą obrazów
        list_file = os.path.join(output_dir, 'selected_images.txt')
        with open(list_file, 'w', encoding='utf-8') as f:
            for image_path in self.selected_images:
                f.write(f"{image_path}\n")
        
        print(f"Zapisano listę obrazów do {list_file}")
    
    def create_report(self, output_file, features_file=None):
        """
        Utworzenie raportu HTML z wynikami.
        
        Args:
            output_file (str): Ścieżka do pliku HTML
            features_file (str): Ścieżka do pliku z cechami (opcjonalnie)
        """
        # Statystyki klastrów
        cluster_stats = []
        for cluster_id, images in self.clusters.items():
            selected_from_cluster = [img for img in self.selected_images if img in images]
            cluster_stats.append({
                'cluster_id': cluster_id,
                'total_images': len(images),
                'selected_images': len(selected_from_cluster),
                'sample_images': images[:3]  # Przykładowe obrazy
            })
        
        # Wygeneruj HTML
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Raport wyboru znaczących obrazów</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .stats {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .thumbnail {{ max-width: 100px; max-height: 100px; margin: 5px; }}
                .selected-images {{ display: flex; flex-wrap: wrap; }}
                .image-container {{ margin: 10px; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Raport wyboru znaczących obrazów</h1>
            
            <div class="stats">
                <h2>Statystyki</h2>
                <p>Liczba klastrów: {num_clusters}</p>
                <p>Całkowita liczba obrazów: {total_images}</p>
                <p>Liczba wybranych obrazów: {num_selected}</p>
            </div>
            
            <div class="clusters">
                <h2>Statystyki klastrów</h2>
                <table>
                    <tr>
                        <th>ID klastra</th>
                        <th>Liczba obrazów</th>
                        <th>Wybrane obrazy</th>
                    </tr>
        """.format(
            num_clusters=len(self.clusters),
            total_images=sum(len(images) for images in self.clusters.values()),
            num_selected=len(self.selected_images)
        )
        
        # Tabela klastrów
        for stat in cluster_stats:
            html_content += """
                    <tr>
                        <td>{cluster_id}</td>
                        <td>{total_images}</td>
                        <td>{selected_images}</td>
                    </tr>
            """.format(**stat)
        
        html_content += """
                </table>
            </div>
            
            <div class="selected-images">
                <h2>Wybrane obrazy</h2>
                <div style="display: flex; flex-wrap: wrap;">
        """
        
        # Lista wybranych obrazów (maksymalnie 100)
        for img_path in self.selected_images[:100]:
            filename = os.path.basename(img_path)
            html_content += """
                    <div class="image-container">
                        <img src="{img_path}" alt="{filename}" class="thumbnail">
                        <div>{filename}</div>
                    </div>
            """.format(img_path=img_path.replace("\\", "/"), filename=filename)
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # Zapisz HTML
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Zapisano raport HTML do: {output_file}")


def visualize_results_cmd():
    """Funkcja uruchamiana z linii poleceń."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wizualizacja i eksport wyników clusteringu')
    parser.add_argument('--results_file', required=True, help='Plik z wynikami clusteringu (JSON)')
    parser.add_argument('--output_dir', required=True, help='Folder docelowy dla wyników')
    parser.add_argument('--features_file', help='Plik z cechami (do wizualizacji klastrów)')
    parser.add_argument('--create_montage', action='store_true', help='Twórz montaż wybranych obrazów')
    parser.add_argument('--create_viz', action='store_true', help='Twórz wizualizacje klastrów')
    parser.add_argument('--create_report', action='store_true', help='Twórz raport HTML')
    parser.add_argument('--export_images', action='store_true', help='Kopiuj wybrane obrazy')
    parser.add_argument('--copy_annotations', action='store_true', help='Kopiuj pliki adnotacji')
    args = parser.parse_args()
    
    # Utwórz folder docelowy
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Utwórz wizualizator
    visualizer = ResultsVisualizer(args.results_file)
    
    # Generuj wizualizację rozkładu klastrów
    if args.create_viz:
        dist_viz_path = os.path.join(args.output_dir, 'cluster_distribution.png')
        visualizer.visualize_cluster_distribution(dist_viz_path)
        
        # Wizualizacja klastrów, jeśli podano plik cech
        if args.features_file:
            pca_viz_path = os.path.join(args.output_dir, 'cluster_visualization_pca.png')
            visualizer.create_clusters_visualization(args.features_file, pca_viz_path, 'pca')
            
            tsne_viz_path = os.path.join(args.output_dir, 'cluster_visualization_tsne.png')
            visualizer.create_clusters_visualization(args.features_file, tsne_viz_path, 'tsne')
    
    # Generuj montaż
    if args.create_montage:
        montage_path = os.path.join(args.output_dir, 'selected_images_montage.jpg')
        visualizer.create_montage(montage_path)
    
    # Generuj raport
    if args.create_report:
        report_path = os.path.join(args.output_dir, 'report.html')
        visualizer.create_report(report_path, args.features_file)
    
    # Eksportuj wybrane obrazy
    if args.export_images:
        images_dir = os.path.join(args.output_dir, 'selected_images')
        visualizer.export_selected_images(images_dir, args.copy_annotations)


if __name__ == '__main__':
    visualize_results_cmd()
