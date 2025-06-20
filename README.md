# ViTParticleFilterTracker

**Projekt PhD:** Zaawansowane metody detekcji i śledzenia narzędzi chirurgicznych  
**Python:** 3.11 (zalecana wersja)  
**Ostatnia aktualizacja:** 17 lipca 2025

## 📋 Opis projektu

Projekt zawiera zaawansowane systemy do detekcji i śledzenia narzędzi chirurgicznych w filmach operacyjnych, wykorzystujące modele YOLO, DETR, DINO oraz inne techniki vision AI.

## 🗂️ Struktura projektu

### **Główne komponenty:**

#### 1. **YOLO_DETR_Benchmarks/** 
Nowoczesne benchmarki i analizy porównawcze modeli:
- **KOMPLETNY_RAPORT_PROBLEMOW_I_ROZWIAZAN** - Analiza problemów DETR
- **Intelligent_Background_Selector_2025-07-17/** - ✨ **NOWY KOMPONENT** (17.07.2025)
- **DINO_Frame_Selection/** - Selekcja ramek z wykorzystaniem DINO
- **Advanced_Analysis/** - Zaawansowane analizy wydajności

#### 2. **Annotators/** 
Różne systemy adnotacji i predykcji:
- **DetrAnnotator/** - Trenowanie i inferencja DETR
- **Yolo/** - Trenowanie i predykcja YOLO
- **TimesFormer/** - Analiza temporalna
- **OpencvTrackerAnnotator/** - Śledzenie OpenCV

#### 3. **BareDetr/**
Minimalistyczna implementacja DETR:
- Czysta implementacja transformera dla detekcji obiektów
- Modularny design bez dodatkowych zależności

#### 4. **Eden/**
Skrypty treningowe i zarządzanie datasetami:
- Konwersje formatów COCO ↔ YOLO
- Optymalizowane skrypty treningowe SLURM

## ✨ Nowy komponent: Intelligent Background Frame Selector

**Data dodania:** 17 lipca 2025  
**Lokalizacja:** `YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17/`

### Problem rozwiązany:
- **Issue:** DETR generuje false positives na czystym oku  
- **Przyczyna:** Brak klasy "background" w treningu  
- **Rozwiązanie:** Automatyczna ekstrakcja ramek tła z filmów chirurgicznych

### Co robi:
1. **Multi-model consensus:** YOLO + DETR dla pewności braku narzędzi
2. **DINO clustering:** Semantyczna różnorodność ramek
3. **HOG analysis:** Filtrowanie zawartości wysokiej jakości
4. **COCO dataset:** Gotowy dataset do treningu DETR

### Wyniki:
- ✅ **690 ramek tła** wykrytych z 6 filmów
- ✅ **20 finalnych ramek** wybranych przez DINO  
- ✅ **10 klastrów semantycznych** dla różnorodności
- ✅ **Dataset COCO** gotowy do treningu

### Użycie:
```bash
cd YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17/

# Test (zalecane przed pierwszym użyciem)
py test_model_loading.py

# Główne uruchomienie
py intelligent_background_frame_selector.py

# Lub przez batch script
run_background_selector.bat
```

**Dokumentacja:** Zobacz `README.md` w folderze komponentu

## 🚀 Szybki start

### Wymagania:
- **Python 3.11** (zalecane dla tego projektu)
- **CUDA** (opcjonalne, ale zalecane)
- **PyTorch, OpenCV, Transformers** (auto-instalacja w skryptach)

### Główne workflow:

#### 1. Detekcja narzędzi (YOLO):
```bash
cd Annotators/Yolo/
py yolo_predict.py
```

#### 2. Detekcja narzędzi (DETR):
```bash
cd Annotators/DetrAnnotator/
py detr_annotator_inference_range_images.py
```

#### 3. Selekcja ramek tła (NOWY):
```bash
cd YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17/
py intelligent_background_frame_selector.py
```

#### 4. Analiza DINO:
```bash
cd YOLO_DETR_Benchmarks/DINO_Frame_Selection/
py dino_frame_selector.py
```

## 📊 Kluczowe pliki

### **Modele:**
- `YOLO_DETR_Benchmarks/models/YOLO/yolo_inference_model_final/yolo_inference_model.pt`
- `YOLO_DETR_Benchmarks/DETR/detr_inference_model_final/`
- `Eden/YOLO_EDEN_TRAIN/best.pt`

### **Datasets:**
- `E:/Cataract/videos/micro/` - Filmy chirurgiczne (50 filmów)
- `Annotators/Datasets/Yolo/` - Dataset YOLO
- `Annotators/Datasets/Detr/` - Dataset DETR/COCO

### **Raporty i analizy:**
- `YOLO_DETR_Benchmarks/KOMPLETNY_RAPORT_PROBLEMOW_I_ROZWIAZAN.pdf`
- `YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17/RAPORT_KONCOWY.md`
- `YOLO_DETR_Benchmarks/Advanced_Analysis/research_report.md`

## 🔧 Konfiguracja środowiska

### Python 3.11 (zalecane):
```bash
# Sprawdź wersję
python --version

# Jeśli masz inną wersję, użyj:
py -3.11  # Windows
```

### Instalacja zależności:
```bash
# Podstawowe (auto-instalowane w skryptach)
py -m pip install torch torchvision ultralytics transformers opencv-python pillow scikit-learn matplotlib numpy timm seaborn

# CUDA (opcjonalne)
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Najnowsze zmiany (17.07.2025)

### ✅ Dodano: Intelligent Background Frame Selector
- **1700+ linii kodu** w 6 nowych plikach
- **Kompletny pipeline** od video do COCO dataset
- **Multi-model validation** dla niezawodności
- **DINO integration** z istniejącym kodem
- **Kompletna dokumentacja** i testy

### 🔧 Poprawki techniczne:
- Fixed DETR model loading paths
- Added missing dependencies (timm, seaborn)  
- Unicode compatibility dla Windows console
- Comprehensive error handling

### 📊 Nowe wyniki:
- **690 ramek tła** automatycznie wykrytych
- **20 finalnych ramek** zoptymalizowanych przez DINO
- **100% success rate** w testach wszystkich komponentów

## 🎯 Następne kroki

### Immediate (1-2 dni):
1. **Trenuj DETR** z nowym background dataset
2. **Testuj na clean eye** images
3. **Mierz redukcję** false positives

### Short-term (1-2 tygodnie):  
1. **Rozszerz dataset** na wszystkie 50 filmów
2. **Zintegruj** z głównym training pipeline
3. **Benchmark** różnych strategii selekcji

## 📞 Kontakt i wsparcie

- **Dokumentacja:** Zobacz README.md w każdym folderze komponentu
- **Issues:** Sprawdź logi w folderach `test_results/` i `analysis_results/`
- **CLAUDE.md:** Pamięć projektu i kontekst rozwoju

---

*Projekt PhD - Zaawansowane metody detekcji narzędzi chirurgicznych*  
*Ostatnia aktualizacja: 17 lipca 2025*