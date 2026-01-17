# Selektor Znaczących Obrazów do Treningu DETR

Zestaw skryptów do wyboru najbardziej znaczących obrazów z adnotacjami YOLO do treningu modelu DETR. Skrypty analizują zarówno cechy wizualne obrazów, jak i informacje z adnotacji YOLO, przeprowadzają clustering i wybierają reprezentatywne przykłady z każdego klastra.

## Struktura projektu

```
SignificantImageSelector/
├── annotation_feature_extractor.py  # Ekstrakcja cech z adnotacji YOLO
├── image_feature_extractor.py       # Ekstrakcja cech wizualnych z obrazów
├── image_clustering.py              # Łączenie cech i clustering
├── results_visualizer.py            # Wizualizacja i eksport wyników
└── select_significant_images.py     # Główny skrypt integrujący cały proces
```

## Wymagania

- Python 3.6+
- NumPy
- OpenCV
- scikit-learn
- Matplotlib

Instalacja wymaganych pakietów:
```bash
pip install numpy opencv-python scikit-learn matplotlib
```

## Jak używać

### Podstawowe użycie

Najłatwiejszym sposobem jest użycie głównego skryptu `select_significant_images.py`, który wykonuje cały proces:

```bash
python select_significant_images.py --images_dir "ścieżka/do/obrazów" --output_dir "ścieżka/do/wyników" --num_clusters 30 --images_per_cluster 3
```

### Szczegółowe opcje

```
--images_dir        Katalog z obrazami
--labels_dir        Katalog z adnotacjami YOLO (opcjonalnie, domyślnie ten sam co obrazy)
--output_dir        Folder docelowy dla wyników
--num_clusters      Liczba klastrów (domyślnie: 20)
--images_per_cluster Liczba obrazów z każdego klastra (domyślnie: 5)
--annotation_weight Waga cech adnotacji vs. cechy wizualne (0-1, domyślnie: 0.5)
--clustering_method Metoda clusteringu ('kmeans' lub 'dbscan', domyślnie: 'kmeans')
--feature_method    Metoda ekstrakcji cech wizualnych ('histogram' lub 'hog', domyślnie: 'histogram')
--skip_visualization Pomiń tworzenie wizualizacji
--skip_export       Pomiń eksport obrazów
--copy_annotations  Kopiuj pliki adnotacji
--keep_temp_files   Zachowaj tymczasowe pliki z cechami
--debug             Tryb debugowania (więcej komunikatów)
```

### Przykłady użycia

1. Wybór 50 najbardziej znaczących obrazów (po 5 z każdego z 10 klastrów):
```bash
python select_significant_images.py --images_dir "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\Datasets\Yolo\yolo_dataset_20250218\images\train" --output_dir "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\Datasets\DETR\selected_images" --num_clusters 10 --images_per_cluster 5
```

2. Użycie deskryptorów HOG zamiast histogramów kolorów:
```bash
python select_significant_images.py --images_dir "ścieżka/do/obrazów" --output_dir "ścieżka/do/wyników" --feature_method hog
```

3. Nadanie większej wagi adnotacjom niż cechom wizualnym:
```bash
python select_significant_images.py --images_dir "ścieżka/do/obrazów" --output_dir "ścieżka/do/wyników" --annotation_weight 0.7
```

## Jak działa?

Proces selekcji składa się z czterech głównych kroków:

1. **Ekstrakcja cech z adnotacji YOLO**
   - Liczba wykrytych ToolTip (końcówek narzędzi)
   - Rozmiar i położenie adnotacji
   - Aspekt ratio i inne cechy geometryczne

2. **Ekstrakcja cech wizualnych z obrazów**
   - Histogramy kolorów w przestrzeni HSV lub
   - Deskryptory HOG (Histogram of Oriented Gradients)

3. **Łączenie cech i clustering**
   - Łączenie cech z adnotacji i cech wizualnych
   - Grupowanie podobnych obrazów za pomocą K-means lub DBSCAN

4. **Wizualizacja i eksport wyników**
   - Wizualizacja klastrów za pomocą PCA lub t-SNE
   - Eksport wybranych obrazów i ich adnotacji
   - Generowanie raportu HTML z wynikami

## Wyniki

Skrypt generuje następujące wyniki w folderze docelowym:

- `selected_images/` - folder z wybranymi obrazami i ich adnotacjami
- `cluster_distribution.png` - wykres rozkładu obrazów w klastrach
- `cluster_visualization_pca.png` - wizualizacja klastrów za pomocą PCA
- `selected_images_montage.png` - montaż wybranych obrazów
- `report.html` - raport HTML z wynikami
- `temp/` - folder z tymczasowymi plikami (jeśli użyto opcji `--keep_temp_files`)

## Narzędzia indywidualne

Możliwe jest również użycie poszczególnych skryptów oddzielnie:

1. Ekstrakcja cech z adnotacji:
```bash
python annotation_feature_extractor.py --images_dir "ścieżka/do/obrazów" --output "cechy_adnotacji.json"
```

2. Ekstrakcja cech wizualnych:
```bash
python image_feature_extractor.py --images_dir "ścieżka/do/obrazów" --method histogram --output "cechy_wizualne.json"
```

3. Clustering:
```bash
python image_clustering.py --annotation_features "cechy_adnotacji.json" --image_features "cechy_wizualne.json" --num_clusters 20 --output "wyniki_clusteringu.json"
```

4. Wizualizacja i eksport:
```bash
python results_visualizer.py --results_file "wyniki_clusteringu.json" --output_dir "wyniki" --create_viz --create_montage --create_report --export_images
```
