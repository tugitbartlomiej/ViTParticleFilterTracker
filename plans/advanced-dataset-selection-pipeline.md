# Plan: Advanced Dataset Selection Pipeline dla DETR Fine-tuning

## Cel

Stworzenie zaawansowanego pipeline'u do selekcji najlepszych zdjęć treningowych dla DETR używając:
- DINO - cechy semantyczne i klasteryzacja
- SAM v3 - precyzyjne maski segmentacji
- Fourier - analiza zmienności w domenie częstotliwości
- EL2N + k-Center - metody selekcji próbek z literatury

## Źródła danych

- Istniejący dataset: ~100k zdjęć z Eden/Datasets/datasets_20250606.tar
- Nowy dataset do dodania: E:\cataract_surgery_Instruments_detection.v1i.coco\train

## Output destination

- Wybrane zdjęcia: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Datasets\SelectedFramesDataset

---

## Struktura projektu

```
F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\
├── External\
│   └── Models\
│       ├── sam_vit_h.pth           # SAM v3 ViT-H checkpoint
│       └── dino_vitb16.pth         # DINO ViT-B/16 (opcjonalnie, można z torch hub)
│
└── AdvancedDatasetSelection\       # NOWY FOLDER
    ├── __init__.py
    ├── config.yaml                 # Konfiguracja pipeline'u
    ├── main_selection_pipeline.py  # Główny orchestrator
    │
    ├── feature_extractors\
    │   ├── __init__.py
    │   ├── dino_extractor.py       # DINO feature extraction
    │   ├── sam_extractor.py        # SAM v3 segmentation masks
    │   └── fourier_analyzer.py     # Fourier frequency analysis
    │
    ├── selection_methods\
    │   ├── __init__.py
    │   ├── el2n_scorer.py          # EL2N difficulty scores
    │   ├── k_center_greedy.py      # k-Center diversity selection
    │   └── combined_selector.py    # EL2N + k-Center combined
    │
    ├── utils\
    │   ├── __init__.py
    │   ├── coco_handler.py         # COCO format I/O
    │   └── visualization.py        # Wizualizacja wyników
    │
    └── output\                     # Wyniki selekcji
        └── selected_dataset\
            ├── images\
            └── annotations.json
```

---

## Etapy implementacji

### ETAP 1: Setup i pobieranie modeli

**Pliki:** config.yaml, setup scripts

1. Stworzenie struktury folderów
2. Pobranie SAM v3 checkpoint (ViT-H, ~2.5GB)
   - URL: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
3. Konfiguracja DINO (torch hub: facebookresearch/dino)
4. Konfiguracja YAML z parametrami

```yaml
# config.yaml
models:
  sam:
    checkpoint: "External/Models/sam_vit_h.pth"
    model_type: "vit_h"
  dino:
    model_name: "dino_vitb16"

datasets:
  existing: "Eden/Datasets/datasets_20250606.tar/datasets_20250606"
  new_source: "E:/cataract_surgery_Instruments_detection.v1i.coco/train"

selection:
  el2n:
    proxy_epochs: 20
    batch_size: 16
  k_center:
    target_size: 5000

fourier:
  similarity_threshold: 0.85
  frequency_bands: ["low", "mid", "high"]
```

---

### ETAP 2: Fourier Analyzer - Analiza zmienności

**Plik:** feature_extractors/fourier_analyzer.py

**Cel:** Eliminacja redundantnych klatek o podobnych rozkładach częstotliwości

```python
class FourierAnalyzer:
    def __init__(self, frequency_bands=["low", "mid", "high"]):
        pass

    def compute_frequency_features(self, image) -> np.ndarray:
        """
        1. Konwersja do grayscale
        2. FFT 2D
        3. Magnitude spectrum
        4. Podział na pasma: low (0-10%), mid (10-50%), high (50-100%)
        5. Obliczenie statystyk per-band: mean, std, energy
        """

    def compute_similarity_matrix(self, features_list) -> np.ndarray:
        """Cosine similarity między wektorami Fouriera"""

    def filter_redundant(self, images, threshold=0.85) -> List[int]:
        """
        Zachłanne usuwanie podobnych obrazów:
        1. Oblicz macierz podobieństwa
        2. Dla każdego klastra podobnych (sim > threshold) zachowaj jeden
        """
```

**Metryki Fouriera:**
- `low_band_energy`: Energia w niskich częstotliwościach (gładkie obszary)
- `high_band_energy`: Energia w wysokich częstotliwościach (krawędzie, szczegóły)
- `spectral_entropy`: Entropia rozkładu częstotliwości
- `frequency_centroid`: Środek ciężkości widma

---

### ETAP 3: SAM v3 Extractor - Maski segmentacji

**Plik:** feature_extractors/sam_extractor.py

**Cel:** Ekstrakcja masek segmentacji i metryk złożoności sceny

```python
class SAMExtractor:
    def __init__(self, checkpoint_path, model_type="vit_h", device="cuda"):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def extract_masks(self, image) -> List[Dict]:
        """Automatic mask generation - zwraca listę masek z confidence"""

    def compute_scene_complexity(self, masks) -> Dict:
        """
        Metryki złożoności:
        - num_segments: liczba wykrytych segmentów
        - avg_segment_size: średni rozmiar segmentu
        - segment_diversity: różnorodność rozmiarów (std)
        - coverage_ratio: % obrazu pokrytego maskami
        - edge_density: gęstość krawędzi między segmentami
        """

    def extract_tool_region_features(self, image, masks) -> np.ndarray:
        """
        Dla największych masek (potencjalne narzędzia):
        - Centroid position
        - Bounding box aspect ratio
        - Mask compactness (4*pi*area/perimeter^2)
        """
```

**SAM Metryki:**
- `scene_complexity_score`: 0-1, jak skomplikowana jest scena
- `tool_presence_confidence`: Czy SAM wykrył obiekty przypominające narzędzia
- `segmentation_quality`: Jakość masek (stability score z SAM)

---

### ETAP 4: DINO Extractor - Rozszerzenie istniejącego

**Plik:** feature_extractors/dino_extractor.py

**Bazuje na:** DINO_Frame_Selection/scripts/dino_feature_extractor.py

```python
class DINOExtractor:
    def __init__(self, model_name="dino_vitb16"):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name)

    def extract_cls_features(self, image) -> np.ndarray:
        """768-dim CLS token features"""

    def extract_attention_features(self, image) -> Dict:
        """
        - attention_entropy: Entropia map atencji
        - attention_diversity: Różnorodność między głowicami
        - spatial_coverage: Jak dużo obrazu jest "obserwowane"
        """

    def compute_semantic_similarity(self, features1, features2) -> float:
        """Cosine similarity w przestrzeni DINO"""
```

---

### ETAP 5: EL2N Scorer - Difficulty scores

**Plik:** selection_methods/el2n_scorer.py

**Implementacja wg wytycznych (compass_artifact):**

```python
class EL2NScorer:
    def __init__(self, proxy_model_path=None, num_epochs=20):
        """
        Proxy model: Mniejszy DETR lub ResNet-18 dla szybkości
        """

    def compute_el2n_scores(self, dataset) -> Dict[str, float]:
        """
        EL2N Score = ||softmax(logits) - one_hot(label)||_2

        Proces:
        1. Trenuj proxy model przez 20 epok
        2. Dla każdego obrazu oblicz EL2N score
        3. Wysoki EL2N = trudny przykład

        Dla object detection (DETR):
        - Agregacja per-image: max(EL2N_per_object) lub mean
        - Uwzględnienie classification + localization uncertainty
        """

    def rank_by_difficulty(self, scores, keep_hard=True) -> List[str]:
        """
        keep_hard=True: Zachowaj trudne przykłady (high EL2N)
        keep_hard=False: Zachowaj łatwe przykłady (low EL2N)

        Wg wytycznych: Przy dużych datasetach lepsze są trudne przykłady
        """
```

**EL2N dla DETR:**

```python
def compute_detr_el2n(model, image, targets):
    outputs = model(image)

    # Classification EL2N
    pred_logits = outputs.pred_logits  # (num_queries, num_classes)
    pred_probs = F.softmax(pred_logits, dim=-1)
    # Dla matched queries (z Hungarian matching)
    cls_el2n = torch.norm(pred_probs - one_hot_targets, p=2, dim=-1)

    # Localization uncertainty (opcjonalnie)
    pred_boxes = outputs.pred_boxes
    box_error = torch.norm(pred_boxes - target_boxes, p=2, dim=-1)

    # Combined score
    image_el2n = cls_el2n.mean() + 0.5 * box_error.mean()
    return image_el2n
```

---

### ETAP 6: k-Center Greedy Selection

**Plik:** selection_methods/k_center_greedy.py

**Implementacja wg wytycznych:**

```python
class KCenterGreedy:
    def __init__(self, feature_dim=768):
        pass

    def select(self, features: np.ndarray, k: int) -> List[int]:
        """
        k-Center Greedy (ICLR 2018):

        1. Start: Wybierz losowy punkt jako pierwszy centroid
        2. Repeat k-1 razy:
           a. Dla każdego niewybranego punktu oblicz min distance do wybranych
           b. Wybierz punkt z MAKSYMALNĄ min distance
        3. Zwróć indeksy k wybranych punktów

        Złożoność: O(k * n) dla n punktów
        """

    def select_with_diversity_guarantee(self, features, k,
                                         min_distance=0.1) -> List[int]:
        """
        Wariant z gwarancją minimalnej odległości między wybranymi
        """
```

---

### ETAP 7: Combined Selector - EL2N + k-Center

**Plik:** selection_methods/combined_selector.py

**Strategia wg wytycznych (compass_artifact):**

```python
class CombinedSelector:
    def __init__(self, el2n_scorer, k_center, dino_extractor,
                 sam_extractor, fourier_analyzer):
        pass

    def select_optimal_subset(self, dataset, target_size: int) -> List[str]:
        """
        Hybrydowa selekcja w 4 krokach:

        KROK 1: Fourier Pre-filtering (usunięcie redundancji)
        - Oblicz Fourier features dla wszystkich obrazów
        - Usuń obrazy z similarity > 0.85 do już wybranych
        - Redukcja: ~20-30% datasetu

        KROK 2: Feature Extraction
        - DINO: 768-dim semantic features
        - SAM: scene complexity metrics
        - Połączenie: concat lub weighted sum

        KROK 3: k-Center Greedy (diversity)
        - Wybierz 2x target_size obrazów dla pokrycia przestrzeni

        KROK 4: EL2N Ranking (difficulty)
        - Z wybranych w kroku 3, zachowaj target_size najtrudniejszych
        - Wg wytycznych: trudne przykłady lepsze przy dużych datasetach
        """

    def compute_combined_score(self, image_path) -> Dict:
        """
        combined_score = {
            'fourier_uniqueness': float,  # Jak unikalny w freq domain
            'dino_features': np.ndarray,  # 768-dim
            'sam_complexity': float,      # 0-1 scene complexity
            'el2n_difficulty': float,     # Trudność dla modelu
            'final_score': float          # Ważona kombinacja
        }
        """
```

**Wagi kombinacji:**

```yaml
weights:
  fourier_uniqueness: 0.15
  dino_diversity: 0.35
  sam_complexity: 0.20
  el2n_difficulty: 0.30
```

---

### ETAP 8: Main Pipeline Orchestrator

**Plik:** main_selection_pipeline.py

```python
class AdvancedDatasetSelectionPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.dino = DINOExtractor(...)
        self.sam = SAMExtractor(...)
        self.fourier = FourierAnalyzer(...)
        self.el2n = EL2NScorer(...)
        self.k_center = KCenterGreedy(...)
        self.selector = CombinedSelector(...)

    def run(self,
            existing_dataset_path: str,
            new_dataset_path: str,
            target_size: int) -> Dict:
        """
        STAGE 1: Load datasets
        - Parse COCO annotations z obu źródeł
        - Merge image lists

        STAGE 2: Feature extraction (parallel processing)
        - DINO features dla każdego obrazu
        - SAM masks + complexity scores
        - Fourier frequency features

        STAGE 3: Fourier pre-filtering
        - Usuń redundantne obrazy (similarity > threshold)

        STAGE 4: Combined selection
        - k-Center na merged features
        - EL2N ranking na wybranych

        STAGE 5: Dataset creation
        - Kopiuj wybrane obrazy do output folder
        - Generuj COCO annotations
        - Generuj raport selekcji
        """

    def generate_selection_report(self) -> Dict:
        """
        Raport zawiera:
        - Statystyki przed/po selekcji
        - Rozkład difficulty scores
        - Pokrycie przestrzeni cech (PCA visualization)
        - Fourier diversity metrics
        - SAM complexity distribution
        """
```

---

## Parametry i thresholdy

| Parametr                     | Wartość    | Opis                                              |
|------------------------------|------------|---------------------------------------------------|
| fourier_similarity_threshold | 0.85       | Próg podobieństwa Fouriera                        |
| k_center_oversampling        | 2.0x       | Ile razy więcej niż target_size wybrać w k-Center |
| el2n_proxy_epochs            | 20         | Epoki treningu proxy modelu                       |
| dino_model                   | ViT-B/16   | Wariant DINO                                      |
| sam_model                    | ViT-H      | Wariant SAM (największy)                          |
| final_target_size            | 5000-10000 | Docelowy rozmiar datasetu                         |

---

## Kolejność implementacji

1. **[Dzień 1]** Setup struktury, config.yaml, pobranie modeli SAM
2. **[Dzień 1-2]** fourier_analyzer.py - kompletna implementacja
3. **[Dzień 2]** sam_extractor.py - integracja SAM v3
4. **[Dzień 2-3]** dino_extractor.py - rozszerzenie istniejącego
5. **[Dzień 3]** el2n_scorer.py - implementacja EL2N dla DETR
6. **[Dzień 3-4]** k_center_greedy.py - algorytm selekcji
7. **[Dzień 4]** combined_selector.py - połączenie wszystkich metod
8. **[Dzień 4-5]** main_selection_pipeline.py - orchestrator
9. **[Dzień 5]** Testy, wizualizacje, raport

---

## Wymagania

```
torch>=2.0
torchvision
transformers
segment-anything  # pip install git+https://github.com/facebookresearch/segment-anything.git
numpy
scipy
scikit-learn
opencv-python
matplotlib
tqdm
pyyaml
```

---

## Output

Po uruchomieniu pipeline'u wybrane zdjęcia trafiają do:

```
F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Datasets\SelectedFramesDataset\
├── images/
│   ├── img_00001.jpg
│   ├── img_00002.jpg
│   └── ...
├── annotations.json          # COCO format
├── selection_report.json     # Szczegółowy raport
├── feature_cache/            # Cache dla powtórzeń
│   ├── dino_features.pkl
│   ├── sam_features.pkl
│   └── fourier_features.pkl
└── visualizations/
    ├── pca_coverage.png
    ├── difficulty_distribution.png
    └── fourier_diversity.png
```
