# Pipeline Inteligentnej Selekcji Datasetu - Dokumentacja Techniczna

**Data:** 2026-01-12
**Lokalizacja kodu:** `AdvancedDatasetSelection/`
**Config:** `AdvancedDatasetSelection/config.yaml`

---

## 1. Przegląd Architektury

W projekcie istnieją **DWA RÓŻNE ALGORYTMY** selekcji:

| Metoda | Klasa | Config | Status |
|--------|-------|--------|--------|
| **CLUSTER** | `ClusterBasedSelector` | `method: cluster` | **DOMYŚLNA (produkcyjna)** |
| KCENTER | `CombinedSelector` | `method: kcenter` | Legacy (opisana w artykule IEEE) |

---

## 2. Metoda CLUSTER (Produkcyjna)

### 2.1 Schemat Pipeline'u

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ETAP 1: EKSTRAKCJA CECH                          │
│  Sekwencyjne ładowanie modeli (oszczędność GPU RAM):                │
│                                                                     │
│  [1/4] Fourier (CPU)     → 9-dim wektor częstotliwościowy           │
│  [2/4] DINO (GPU→unload) → 1024-dim wektor semantyczny              │
│  [3/4] SAM (GPU→unload)  → 1-dim score złożoności                   │
│  [4/4] DETR (GPU→unload) → 1-dim EL2N score trudności               │
├─────────────────────────────────────────────────────────────────────┤
│                    ETAP 2: KOMBINACJA CECH                          │
│                                                                     │
│  Każdy obraz → wektor 1035-dim:                                     │
│  [ DINO (1024) | Fourier (9) | SAM (1) | EL2N (1) ]                 │
│                                                                     │
│  + Z-score normalizacja (StandardScaler)                            │
├─────────────────────────────────────────────────────────────────────┤
│                    ETAP 3: K-MEANS CLUSTERING                       │
│                                                                     │
│  n_clusters = target_size (np. 20,000)                              │
│  Metody: faiss-gpu (najszybsza) / minibatch / kmeans                │
│                                                                     │
│  Każdy klaster = grupa podobnych obrazów w 1035-dim przestrzeni     │
├─────────────────────────────────────────────────────────────────────┤
│                    ETAP 4: WYBÓR REPREZENTANTÓW                     │
│                                                                     │
│  Z każdego klastra wybierany jest 1 obraz (strategie):              │
│  • centroid  - najbliższy centrum klastra (domyślna)                │
│  • max_el2n  - najtrudniejszy w klastrze                            │
│  • medoid    - minimalizuje odległość do wszystkich członków        │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Szczegóły Cech

#### Fourier (9-dim) - `fourier_analyzer.py`
```
[0] low_band_energy      - energia niskich częstotliwości (0-10%)
[1] mid_band_energy      - energia średnich częstotliwości (10-50%)
[2] high_band_energy     - energia wysokich częstotliwości (50-100%)
[3] spectral_entropy     - entropia spektralna (znormalizowana)
[4] frequency_centroid   - środek ciężkości częstotliwości
[5] horizontal_energy    - energia kierunkowa (pozioma)
[6] vertical_energy      - energia kierunkowa (pionowa)
[7] diagonal1_energy     - energia kierunkowa (ukośna 1)
[8] diagonal2_energy     - energia kierunkowa (ukośna 2)
```

**Uwaga:** Fourier NIE filtruje jakości (R < 4.0 nie istnieje w kodzie!).
Służy do: (1) wykrywania duplikatów (similarity threshold 0.7), (2) cechy w klasteryzacji.

#### DINO (1024-dim) - `dino_extractor.py`
- Model: **DINOv3 ViT-L/16** (NIE ViT-B/16 jak w artykule!)
- Wymiar: **1024** (NIE 768 jak w artykule!)
- CLS token embedding - semantyczna reprezentacja obrazu
- Grupuje obrazy o podobnej zawartości wizualnej

#### SAM Complexity (1-dim) - `fastsam_extractor.py` / `sam_extractor.py`
- Preferowany: **FastSAM** (~0.1s/obraz)
- Fallback: proxy method (edge detection + color analysis)
- Mierzy złożoność wizualną sceny

#### EL2N (1-dim) - `detr_el2n_scorer.py`
```python
EL2N(x) = ||softmax(f(x)) - y||₂
```
- Używa **prawdziwego DETR checkpoint** (epoch 170), NIE proxy!
- Mierzy "trudność" obrazu dla modelu
- Wysokie EL2N = model ma problem z detekcją

### 2.3 Kluczowe Parametry (config.yaml)

```yaml
models:
  dino:
    model_name: dinov3_vitl16        # DINOv3 ViT-L/16 (1024-dim)
  detr:
    checkpoint: checkpoint_epoch_170.pth  # Prawdziwy model, nie proxy

fourier:
  similarity_threshold: 0.7         # Do usuwania duplikatów

selection:
  method: cluster                   # DOMYŚLNA METODA
  strategy: centroid                # Wybór reprezentanta

weights:                            # Wagi (dla normalizacji)
  fourier_uniqueness: 0.15
  dino_diversity: 0.35
  sam_complexity: 0.20
  el2n_difficulty: 0.30

output:
  target_size: 20000                # Docelowa liczba obrazów
```

---

## 3. Metoda KCENTER (Legacy - opisana w artykule)

### 3.1 Schemat (4-etapowy pipeline z artykułu)

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: Fourier Pre-filtering (similarity-based)          │
│  → Usuwa duplikaty na podstawie podobieństwa Fourier        │
├──────────────────────────────────────────────────────────────┤
│  Stage 2: Combine DINO + SAM                                │
│  → [DINO | SAM] - BEZ EL2N!                                 │
├──────────────────────────────────────────────────────────────┤
│  Stage 3: k-Center Greedy                                   │
│  → Wybiera 2k próbek (oversampling 2x) dla różnorodności    │
├──────────────────────────────────────────────────────────────┤
│  Stage 4: EL2N Ranking                                      │
│  → Sortuje 2k próbek, wybiera top k najtrudniejszych        │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Kluczowa Różnica od CLUSTER

| Aspekt | CLUSTER | KCENTER |
|--------|---------|---------|
| Rola EL2N | **Cecha w klasteryzacji** | Ranking na końcu |
| Metoda grupowania | K-Means | k-Center Greedy |
| Wymiar przestrzeni | 1035-dim (wszystkie cechy) | ~1025-dim (DINO+SAM) |
| Oversampling | Nie | Tak (2x) |

---

## 4. Rozbieżności Artykuł vs Kod

| Element | Artykuł IEEE | Kod Produkcyjny |
|---------|--------------|-----------------|
| Metoda | k-Center Greedy | **K-Means Clustering** |
| DINO model | ViT-B/16, 768-dim | **ViT-L/16, 1024-dim** |
| Stage 1 Fourier | "Quality filtering" | **Similarity filtering** |
| EL2N | Proxy (10ep, bs=4) | **Real model (170ep)** |
| Rola EL2N | Ranking na końcu | **Cecha w klasteryzacji** |

---

## 5. Pliki Źródłowe

```
AdvancedDatasetSelection/
├── main_selection_pipeline.py      # Główny orchestrator
├── config.yaml                     # Konfiguracja
├── feature_extractors/
│   ├── fourier_analyzer.py         # Fourier 9-dim
│   ├── dino_extractor.py           # DINO 1024-dim
│   ├── fastsam_extractor.py        # SAM complexity (szybki)
│   └── sam_extractor.py            # SAM complexity (fallback)
├── selection_methods/
│   ├── cluster_selector.py         # CLUSTER (produkcyjna)
│   ├── combined_selector.py        # KCENTER (legacy/artykuł)
│   ├── detr_el2n_scorer.py         # EL2N z prawdziwym DETR
│   └── k_center_greedy.py          # k-Center algorithm
└── output/
    └── feature_cache/              # Cache cech (fourier, dino, sam, el2n)
```

---

## 6. Uruchomienie

```bash
# Domyślna metoda (cluster)
python main_selection_pipeline.py --config config.yaml --target-size 20000

# Legacy metoda (kcenter - jak w artykule)
python main_selection_pipeline.py --config config.yaml --method kcenter --target-size 20000

# Strategie wyboru reprezentantów (tylko dla cluster)
python main_selection_pipeline.py --strategy max_el2n  # Najtrudniejsze
python main_selection_pipeline.py --strategy medoid    # Medoidy klastrów
```

---

## 7. Podsumowanie

**Produkcyjny pipeline (CLUSTER):**
1. Ekstrahuje 4 typy cech: DINO (semantyka) + Fourier (częstotliwość) + SAM (złożoność) + EL2N (trudność)
2. Łączy w 1035-wymiarowy wektor per obraz
3. K-Means clustering tworzy k klastrów w tej przestrzeni
4. Z każdego klastra wybierany jest 1 reprezentant

**EL2N jest CECHĄ używaną w klasteryzacji, nie tylko rankingiem na końcu!**

---

*Wygenerowano: 2026-01-12*
