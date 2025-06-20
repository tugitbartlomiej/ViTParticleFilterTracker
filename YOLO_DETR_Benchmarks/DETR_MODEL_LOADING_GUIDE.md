# Przewodnik po ładowaniu modelu DETR w tym projekcie

Ten dokument wyjaśnia, jak poprawnie załadować wytrenowany model DETR do inferencji i dlaczego inne, pozornie oczywiste metody, prowadziły do błędów.

## Problem: Konflikt implementacji i formatu zapisu

Głównym problemem, który napotkaliśmy, był konflikt między sposobem, w jaki model został **wytrenowany i zapisany**, a sposobem, w jaki próbowaliśmy go **załadować**.

Model w tym projekcie został wytrenowany przy użyciu biblioteki `transformers` od Hugging Face. Nasze początkowe błędy wynikały z próby załadowania go przy użyciu innych narzędzi lub w niekompatybilny sposób.

### Błędne podejścia i dlaczego zawiodły

1.  **Próba użycia `torchvision`:**
    *   **Błąd:** `ImportError` lub `RuntimeError` z powodu niedopasowania kluczy (`Missing key(s) in state_dict`).
    *   **Przyczyna:** Biblioteki `torchvision` i `transformers` mają **różne implementacje** architektury DETR. Oznacza to, że nazwy warstw i ogólna struktura modelu różnią się między nimi. Plik z wagami jest jak klucz dopasowany do konkretnego zamka – wagi z modelu `transformers` nie pasują do architektury `torchvision`.

2.  **Próba bezpośredniego załadowania checkpointa `.pth` do `transformers`:**
    *   **Błąd:** `RuntimeError` z powodu niedopasowania kluczy (brak prefiksu `model.` i obecność dodatkowych kluczy jak `optimizer_state_dict`).
    *   **Przyczyna:** Plik `.pth` z treningu to **punkt kontrolny (checkpoint)**, a nie finalny model. Zawiera on dodatkowe informacje (stan optymalizatora, epokę, funkcję straty), które nie są częścią samej architektury modelu. Ponadto, klucze wag w checkpoincie nie zawsze idealnie pasują do struktury, której oczekuje metoda `.from_pretrained()`.

## Rozwiązanie: Proces konwersji i ładowania

Aby poprawnie załadować model, konieczny jest dwuetapowy proces.

### Krok 1: Konwersja checkpointa do formatu inferencyjnego

Należy **zawsze** najpierw uruchomić skrypt `models/convert_detr_checkpoint.py`.

```bash
py -3.11 models/convert_detr_checkpoint.py
```

**Co ten skrypt robi?**
*   Wczytuje "brudny" checkpoint z treningu (np. `checkpoint_epoch_100.pth`).
*   Inicjalizuje "czystą" architekturę modelu DETR z Hugging Face (`facebook/detr-resnet-50`).
*   Wyodrębnia same wagi modelu z checkpointa i wczytuje je do nowej, czystej architektury.
*   Zapisuje finalny, gotowy do inferencji model w **dedykowanym folderze** (`./DETR/detr_inference_model_final`), który zawiera pliki `pytorch_model.bin` i `config.json`.

### Krok 2: Poprawne ładowanie modelu do inferencji

W skrypcie, który ma używać modelu (np. `simple_test/test_single_frame.py`), należy go załadować **bezpośrednio z folderu** stworzonego w Kroku 1.

Poniższy kod jest poprawnym sposobem na załadowanie modelu i procesora:

```python
from transformers import DetrForObjectDetection, DetrImageProcessor

# Ścieżka musi wskazywać na FOLDER, a nie na plik .pth
MODEL_PATH = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/DETR/detr_inference_model_final"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ładowanie modelu i procesora z folderu
model = DetrForObjectDetection.from_pretrained(MODEL_PATH).to(DEVICE)
processor = DetrImageProcessor.from_pretrained(MODEL_PATH)

model.eval()
```

To podejście gwarantuje, że zarówno architektura, jak i wagi są w 100% zgodne, co eliminuje błędy i pozwala na poprawne działanie modelu.
