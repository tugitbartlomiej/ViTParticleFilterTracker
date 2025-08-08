# DETR Local Training - Instrukcja

## 📋 Wymagania

### Hardware
- **GPU:** NVIDIA RTX 3060/4060+ (minimum 8GB VRAM) - dla GPU training
- **RAM:** 16GB+ 
- **Miejsce:** ~5GB wolnego miejsca

### Software
- **Python:** 3.8-3.11
- **CUDA:** 11.8 lub 12.x (jeśli masz GPU NVIDIA)

## 🛠️ Instalacja

### 1. Przygotuj środowisko Python

```bash
# Opcja A: Conda (zalecane)
conda create -n detr_local python=3.11
conda activate detr_local

# Opcja B: venv
python -m venv detr_local
detr_local\Scripts\activate  # Windows
```

### 2. Zainstaluj PyTorch

**Z GPU (CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Tylko CPU (wolne):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Zainstaluj pozostałe biblioteki

```bash
pip install transformers datasets pillow tqdm tensorboard
pip install accelerate  # dla lepszej wydajności
```

### 4. Sprawdź instalację

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
```

## 🚀 Uruchomienie

### Metoda 1: Użyj skryptu BAT (najłatwiejsze)

```cmd
cd F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\scripts\DETR_Background_Training\local
run_local_training.bat
```

### Metoda 2: Ręczne uruchomienie

```cmd
cd F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\scripts\DETR_Background_Training\local

python detr_local_train.py \
    --background_images_dir "..\..\..\..\Datasets\Detr\Background\train" \
    --background_annotations_path "..\..\..\..\Datasets\Detr\Background\annotations\train_annotations.json" \
    --epochs 5 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --augment \
    --use_amp
```

## ⚙️ Parametry treningu

### Podstawowe (dla słabszego komputera)
```bash
--epochs 5                        # Krótki trening testowy
--batch_size 1                    # Mały batch (8GB VRAM)
--gradient_accumulation_steps 8   # Symuluje batch_size=8
--num_workers 0                   # Bez wielowątkowości (Windows)
```

### Zaawansowane (dla mocniejszego komputera)
```bash
--epochs 20                       # Dłuższy trening
--batch_size 4                    # Większy batch (16GB+ VRAM)
--gradient_accumulation_steps 2   # Mniejsza akumulacja
--num_workers 2                   # Więcej workerów
--compile_model                   # Kompilacja (PyTorch 2.0+)
```

## 📊 Monitoring

### TensorBoard
```bash
tensorboard --logdir output_local/logs
# Otwórz: http://localhost:6006
```

### Pliki wyjściowe
- `checkpoints_local/` - checkpointy modelu
- `output_local/logs/` - logi TensorBoard
- `output_local/` - końcowe wyniki

## 🐛 Rozwiązywanie problemów

### "CUDA out of memory"
```bash
# Zmniejsz batch size
--batch_size 1
--gradient_accumulation_steps 16

# Lub użyj CPU
--use_amp false  # wyłącz mixed precision
```

### "FileNotFoundError"
- Sprawdź ścieżki do obrazów i annotations
- Upewnij się że pliki istnieją w folderze `Background/`

### Wolny trening
- Trening na CPU: ~6-12h na 5 epok
- Trening na GPU: ~30-60min na 5 epok

### Windows - problemy z workers
```bash
--num_workers 0  # Zawsze ustaw na 0 w Windows
```

## 📈 Oczekiwane wyniki

**Dataset:** ~55 obrazów background
**Czas treningu (GPU):** 
- 5 epok: ~30-45 min
- 20 epok: ~2-3h

**Metryki:**
- Loss powinien spadać z ~2.0 do ~0.5
- Walidacja powinna być stabilna po 3-5 epokach

## 🔧 Dostosowywanie

### Własny checkpoint
```bash
--checkpoint_path "path/to/your/checkpoint.pth"
```

### Inne zdjęcia
```bash
--background_images_dir "path/to/your/images"
--background_annotations_path "path/to/your/annotations.json"
```

### Resume treningu
```bash
--resume_training
```

## ❓ FAQ

**Q: Czy mogę trenować bez GPU?**
A: Tak, ale będzie bardzo wolno (6-12h). Usuń `--use_amp` i zwiększ `--gradient_accumulation_steps`.

**Q: Mam mało VRAM, co robić?**
A: Ustaw `--batch_size 1` i zwiększ `--gradient_accumulation_steps 16`.

**Q: Jak sprawdzić postęp?**
A: Uruchom TensorBoard lub sprawdzaj logi w konsoli.

**Q: Model się nie uczcy?**
A: Sprawdź czy learning rate nie jest za wysoki/niski. Spróbuj `--lr 1e-5`.