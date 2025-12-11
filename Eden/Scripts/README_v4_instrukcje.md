# Instrukcja uruchomienia YOLO treningu - wersja finalna

## Problem
Trening YOLO na klastrze Eden kończył się błędami związanymi z:
1. Niewłaściwą strukturą datasetu (YOLO nie mogło znaleźć etykiet)
2. Błędami CUDA/NVML na poziomie sterowników
3. Ostrzeżeniami o liczbie workers (16 vs zalecane 1)

## Rozwiązanie

### 1. Skrypt `yolo-train-fixed-v4.slurm`
**Główne zmiany:**
- Tworzy unified strukturę datasetu gdzie obrazy i etykiety są w odpowiednich miejscach
- Kopiuje obrazy i etykiety do standardowej struktury YOLO: `images/train/`, `labels/train/` etc.
- Używa stabilnego skryptu treningu `yolo-train-stable.py`
- Dodaje zmienne środowiskowe dla stabilności CUDA
- Redukuje liczbę workers do 1 (zgodnie z zaleceniami YOLO)

### 2. Skrypt `yolo-train-stable.py`
**Główne funkcje:**
- Walidacja dataset.yaml przed treningiem
- Ustawienia środowiska CUDA dla stabilności
- Wyłączenie problematycznych funkcji (AMP, DDP) na DGX-1
- Lepsze obsługiwanie błędów
- Automatyczne kopiowanie najlepszych modeli

## Uruchomienie

```bash
# 1. Skopiuj nowe pliki na klaster Eden
scp yolo-train-fixed-v4.slurm eden:/mnt/evafs/faculty/home/bpiotrowski/Yolo/
scp yolo-train-stable.py eden:/mnt/evafs/faculty/home/bpiotrowski/Yolo/

# 2. Zaloguj się na klaster
ssh eden

# 3. Przejdź do katalogu
cd /mnt/evafs/faculty/home/bpiotrowski/Yolo/

# 4. Upewnij się że katalog out/ istnieje
mkdir -p out

# 5. Uruchom zadanie
sbatch yolo-train-fixed-v4.slurm
```

## Monitorowanie

```bash
# Sprawdź status zadania
squeue -u $USER

# Monitoruj logi w czasie rzeczywistym
tail -f out/yolo_8gpu_*.out

# Sprawdź błędy
tail -f out/yolo_8gpu_*.err
```

## Struktura wyników

Po udanym treningu:
```
./train_out_8gpu_v4/train/
├── weights/
│   ├── best.pt          # Najlepszy model
│   └── last.pt          # Ostatni checkpoint
├── results.png          # Wykresy treningu
├── confusion_matrix.png # Macierz konfuzji
└── ...

./best_8gpu_v4/
└── best.pt             # Kopia najlepszego modelu

./ckpt_8gpu_v4/
└── last.pt             # Kopia ostatniego checkpointu
```

## Kluczowe poprawki w v4

1. **Unified struktura datasetu:**
   - Kopiowanie obrazów do `$UNIFIED_DATASET_DIR/images/{train,val,test}/`
   - Kopiowanie etykiet do `$UNIFIED_DATASET_DIR/labels/{train,val,test}/`
   - Dataset.yaml wskazuje na tę unified strukturę

2. **Stabilne ustawienia treningu:**
   - `workers=1` (zamiast 16/24)
   - `amp=False` (wyłączone AMP)
   - `cache=False` (oszczędność pamięci)
   - `rect=False` (stabilność)
   - Zmienne CUDA dla stabilności

3. **Lepsze debugowanie:**
   - Sprawdzanie czy pliki rzeczywiście istnieją
   - Walidacja dataset.yaml
   - Szczegółowe logi struktury

## Oczekiwane rezultaty

Trening powinien się rozpocząć bez błędów związanych z:
- ✅ Brakiem etykiet (unified struktura rozwiązuje problem)
- ✅ Błędami CUDA/NVML (stabilne ustawienia środowiska)
- ✅ Ostrzeżeniami o workers (zmniejszone do 1)

Dataset zawiera:
- ~63,884 obrazów treningowych
- ~11,983 obrazów walidacyjnych
- 1 klasa: 'tooltip'
