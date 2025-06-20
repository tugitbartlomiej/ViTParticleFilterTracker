# Raport Badawczy: Framework do Benchmarkingu YOLOv8 vs. DETR w Detekcji Narzędzi Chirurgicznych

## 1. Wprowadzenie i Hipoteza Badawcza

W detekcji obiektów w czasie rzeczywistym, szczególnie w dynamicznych i złożonych scenach, takich jak operacje chirurgiczne, wybór odpowiedniej architektury modelu ma kluczowe znaczenie. Modele takie jak YOLOv8, oparte na konwolucyjnych sieciach neuronowych (CNN), przetwarzają obraz w sposób lokalny. W przeciwieństwie do nich, modele oparte na Transformerach, takie jak DETR, wykorzystują mechanizmy atencji do analizy globalnego kontekstu obrazu.

**Hipoteza badawcza:** Globalny kontekst dostarczany przez mechanizm atencji w modelu DETR skutkuje wyższą stabilnością temporalną (mniejsze "drżenie" ramek i "mruganie" detekcji) oraz potencjalnie wyższą dokładnością w sytuacjach częściowej okluzji lub szybkiego ruchu narzędzia, w porównaniu do modelu YOLOv8.

## 2. Proponowana Metodologia Porównawcza

Aby kompleksowo zweryfikować powyższą hipotezę, proponuje się ocenę modeli w trzech kluczowych obszarach:

### 2.1. Metryki Ilościowe (Dokładność Detekcji)
Obliczane na statycznym zbiorze testowym w celu oceny ogólnej zdolności do detekcji.

*   **mAP (mean Average Precision):** Złoty standard w detekcji obiektów.
    *   `mAP@.50`: Ocena przy progu IoU 50%. Wskaźnik ogólnej poprawności detekcji.
    *   `mAP@.50-.95`: Średnia z mAP dla progów IoU od 50% do 95%. Nagradza modele za precyzyjną lokalizację.
*   **Precision & Recall:** Analiza balansu między fałszywymi alarmami (Precision) a pominiętymi detekcjami (Recall).

### 2.2. Metryki Jakościowe i Stabilności Czasowej (Kluczowe dla Wideo)
Te metryki są kluczowe dla weryfikacji hipotezy i oceny modelu w kontekście sekwencji wideo.

*   **Temporal IoU (Jitter/Stability):**
    *   **Opis:** Mierzy średnie IoU (Intersection over Union) bounding boxa dla tego samego obiektu w dwóch kolejnych klatkach. 
    *   **Cel:** Wysoka wartość wskazuje na stabilne, nie "drżące" predykcje, co jest pożądane w śledzeniu. Oczekuje się, że DETR osiągnie tu lepszy wynik.

*   **Detection Flicker/Blink Rate:**
    *   **Opis:** Zlicza, ile razy obiekt, który był widoczny w klatkach `t-1` i `t+1`, został "zgubiony" przez model w klatce `t`.
    *   **Cel:** Niższa wartość oznacza bardziej niezawodną i ciągłą detekcję. Ma to bezpośrednio pokazać przewagę modelu rozumiejącego kontekst sekwencji.

### 2.3. Metryki Wydajności Obliczeniowej
Ocena praktycznej użyteczności modeli.

*   **FPS (Frames Per Second):** Szybkość przetwarzania, kluczowa dla zastosowań w czasie rzeczywistym.
*   **VRAM Usage:** Zapotrzebowanie na pamięć karty graficznej podczas inferencji.
*   **Model Size:** Rozmiar modelu na dysku.

## 3. Implementacja i Wizualizacja

Powyższa metodologia została zaimplementowana w dostarczonych skryptach (`run_inference.py`, `evaluate_metrics.py`). W celu ułatwienia analizy i prezentacji wyników, skrypt `visualize_results.py` generuje:

*   **Filmy porównawcze "side-by-side"**: Bezpośrednia wizualna konfrontacja obu modeli na tych samych danych.
*   **Wykresy słupkowe**: Porównanie kluczowych metryk (mAP, Temporal IoU, FPS).

## 4. Oczekiwany Wniosek
Oczekuje się, że wyniki potwierdzą, iż model DETR, mimo potencjalnie niższej wydajności (niższy FPS), znacząco przewyższa YOLOv8 w metrykach stabilności czasowej, co czyni go bardziej obiecującym kandydatem do zastosowań w precyzyjnej nawigacji chirurgicznej.
