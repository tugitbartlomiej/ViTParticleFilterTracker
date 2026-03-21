# Ablation Study Plan — Dataset Curation Pipeline (v16)

**Zaktualizowano**: 2026-03-20 23:38:00 — pełny plan wykonawczy dla wszystkich wariantów

## Kluczowe fakty

1. **Nazwy plików kodują przynależność do wideo**: `test01_frame_0000746.jpg` → wideo `test01`
2. **Selekcja z pełnej puli ~91k** (oryginalne + augmentowane) — feature cache policzone na tym zbiorze (90,141 wpisów)
3. 50 unikalnych wideo: `test01`–`test25` + `train01`–`train25`
4. Główna ewaluacja to **cross-dataset** (zewnętrzny Roboflow) — niezależna od splitu treningowego
5. V1 (full pipeline) ma istniejące checkpointy — nie wymaga ponownego treningu

## Stan zasobów na Eden (po czyszczeniu 2026-03-20 20:00:00)

| Zasób | Ścieżka | Rozmiar |
|-------|---------|---------|
| Pula 91k (tar.gz) | `~/datasets_20250606.tar.gz` | 39 GB |
| Pula 91k JSON | `~/datasets_20250606_test/.../augmented_coco_20250417_030014.json` | 37 MB |
| 20k wybrane (tar) | `~/20kSelectedImages.tar` | 8.5 GB |
| 20k JSON | `~/merged_20k_annotations_fixed.json` | ~5 MB |
| Checkpoint ep.170 | `~/DETR/Checkpoints/checkpoint_epoch_170.pth` | ~475 MB |
| 20k best checkpoints | `~/DETR/Checkpoints/20k_finetune_v2_fixed/` | ep 170,210,265,435 + final |
| Feature cache | **LOKALNIE** `F:\...\output\feature_cache2_weighted\` | 369 MB |
| **Wolne miejsce** | `/mnt/evafs` | **87 GB** |
| Wolne GPU | hopper-2: 3×H200, dgx-4: 5×A100, pascal: 4×P100 | — |

## Warianty ablacji (7 wariantów)

| ID | Nazwa | Opis | Cel |
|----|-------|------|-----|
| v1 | FULL | Fourier+DINO+SAM+EL2N (0.15/0.35/0.20/0.30) | Baseline — **istniejące checkpointy** |
| v2 | NO_FOURIER | DINO+SAM+EL2N (0.00/0.45/0.25/0.30) | Czy Fourier coś wnosi? |
| v3 | NO_EL2N | Fourier+DINO+SAM (0.20/0.45/0.35/0.00) | Czy EL2N pomaga czy biasuje? (problem #4) |
| v4 | QUALITY_ONLY | Fourier filtruje + random 20k | Czy sama jakość wystarczy? |
| v5 | RANDOM | Losowe 20k z oryginalnych klatek (seed 42) | Dolna granica — zero selekcji |
| v6 | DIVERSITY_ONLY | DINO+K-Center, bez Fourier i EL2N | Izoluje wpływ samej różnorodności |
| v7 | EL2N_ONLY | Random + EL2N reranking | Czy EL2N sam wystarczy? |

**V1 nie wymaga treningu.** Trenujemy V2–V7 (6 wariantów).

### Selekcja operuje na oryginalnych klatkach

Selekcja operuje na **pełnej puli 91,336 klatek** (oryginalne + augmentowane `_aug_1/2/3`). Feature cache ma 90,141 wpisów z tego zbioru. Każdy wariant wybiera 20k z ~90k = 22% puli — wystarczająco selektywne żeby pokazać różnice między wariantami. Online augmentacje (flip, jitter) stosowane dodatkowo podczas treningu.

## Split treningowy — DECYZJA: random_split(seed=42) dla WSZYSTKICH wariantów

### Uzasadnienie

**Główna metryka to cross-dataset evaluation** (zewnętrzny Roboflow dataset). Cross-dataset jest NIEZALEŻNY od podziału treningowego — model jest testowany na danych z innego szpitala, których nigdy nie widział. Dlatego sposób podziału train/val NIE WPŁYWA na główne wyniki.

Same-distribution validation służy jako **monitoring treningu** (early stopping, wybór checkpointu) i **sanity check** (czy model się nie przeuczył).

### Dlaczego random_split a nie video-level?

1. **Spójność z V1** — oryginalny model (V1 FULL) trenowany był z `random_split(seed=42)`. Użycie tego samego splitu dla V2-V7 gwarantuje **identyczne warunki** dla wszystkich wariantów. Nie trzeba przetrenowywać V1.

2. **Fair comparison** — wszystkie warianty mają ten sam podział train 90% / val 10% z tego samego seeda. Różnice w wynikach wynikają WYŁĄCZNIE z różnej selekcji 20k próbek, nie z różnego podziału.

3. **Cross-dataset rozstrzygna** — nawet jeśli same-dist mAP jest lekko zawyżony przez augmentowane kopie w obu splitach, jest zawyżony **jednakowo** dla wszystkich wariantów. Porównanie V1 vs V2 na same-dist jest fair. A cross-dataset jest niezależny.

### Dla recenzenta (do wpisania w artykule)

> *"All ablation variants use identical train/validation splits (random split with seed 42, 90/10 ratio) to ensure fair comparison. The primary evaluation metric is cross-dataset mAP@0.5 on an external benchmark (Section 5.3), which is independent of the training split. Same-distribution metrics are reported for completeness as a secondary reference."*

### Implementacja

```python
# W detr_train_ablation.py (identyczne jak w detr_train_optimized.py):
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))
```

Seed 42 gwarantuje że ten sam podział jest użyty przy każdym uruchomieniu i dla każdego wariantu.

---

## Plan wykonawczy krok po kroku

### KROK 1: Przesłanie feature cache na Eden (~1 min)

```bash
scp -r /mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/AdvancedDatasetSelection/output/feature_cache2_weighted/ \
    eden-cluster:~/DETR/ablation/feature_cache/
```

### KROK 2: Przesłanie skryptów (~10 sek)

```bash
cd /mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Scripts/AblationStudy/
scp generate_ablation_on_eden.py detr_train_ablation.py evaluate_ablation.py eden-cluster:~/DETR/
scp run_ablation_variant.slurm run_ablation_eval.slurm run_ablation_all.sh eden-cluster:~/DETR/
```

### KROK 3: Generowanie wariantów na Eden (~2-5 min, CPU)

```bash
ssh eden-cluster
cd ~/DETR
conda activate yolo_py310

python generate_ablation_on_eden.py \
    --annotations ~/datasets_20250606_test/datasets/DETR_augmented_dataset_20250218/augmented_coco_20250417_030014.json \
    --cache ~/DETR/ablation/feature_cache/ \
    --output ~/DETR/ablation \
    --seed 42 \
    # selekcja z pełnej puli ~91k (domyślnie, bez filtrowania)
```

Wynik:
```
~/DETR/ablation/
├── video_split.json              # 50 wideo → 35 train / 8 val / 7 test
├── v2/annotations_train.json     # 20k NO_FOURIER
├── v2/annotations_val.json
├── v2/annotations_test.json
├── v3/annotations_train.json     # 20k NO_EL2N
├── v3/annotations_val.json
├── v3/annotations_test.json
├── v4/annotations_train.json     # 20k QUALITY_ONLY
├── v4/annotations_val.json
├── v4/annotations_test.json
├── v5/annotations_train.json     # 20k RANDOM
├── v5/annotations_val.json
├── v5/annotations_test.json
├── v6/annotations_train.json     # 20k DIVERSITY_ONLY
├── v6/annotations_val.json
├── v6/annotations_test.json
├── v7/annotations_train.json     # 20k EL2N_ONLY
├── v7/annotations_val.json
└── v7/annotations_test.json
```

### KROK 3.5: Preflight — weryfikacja stanu Eden TUŻ PRZED uruchomieniem

**OBOWIĄZKOWE** — nie uruchamiaj niczego dopóki WSZYSTKIE checki nie przejdą!

#### A) Jednorazowe (przed pierwszym uruchomieniem)

```bash
ssh eden-cluster

# 1. sklearn — BLOKER dla kroku 3 (generowanie wariantów)
conda activate yolo_py310
python3 -c "import sklearn; print('sklearn OK')" || pip install scikit-learn

# 2. pycocotools — BLOKER dla kroku 7 (ewaluacja COCO mAP)
python3 -c "from pycocotools.coco import COCO; print('pycocotools OK')"

# 3. external_benchmark — BLOKER dla kroku 7b (cross-dataset eval)
# Dataset: Roboflow "cataract_surgery_Instruments_detection.v1i.coco" (2333 obrazów, 175 MB)
# Lokalna ścieżka: /mnt/e/cataract_surgery_Instruments_detection.v1i.coco/
# Format: 3 splity (train/valid/test), każdy z _annotations.coco.json
ls ~/DETR/external_benchmark/valid/_annotations.coco.json && echo "external OK" || echo "STOP: brak external benchmark! Uruchom: scp -r /mnt/e/cataract_surgery_Instruments_detection.v1i.coco/ eden-cluster:~/DETR/external_benchmark/"

# 4. Katalog ablation
mkdir -p ~/DETR/ablation ~/DETR/ablation_checkpoints ~/DETR/ablation_best ~/DETR/ablation_output
```

#### B) Przed KAŻDYM `sbatch`

```bash
ssh eden-cluster
sinfo -p debug,experimental --format="%P %a %D %T %G %N"  # wolne GPU
squeue -u bpiotrowski                                       # moje joby
df -h ~                                                     # wolne miejsce
```

Zaktualizuj SLURM script jeśli:
- Node z planu jest zajęty → zmień `--nodelist` lub usuń (scheduler wybierze)
- Partycja niedostępna → zmień `--partition`
- Quota się zmieniła → dostosuj `--save_interval`

### KROK 4: Canary test — WSZYSTKIE warianty, 2 epoki (~3h)

Test poprawności setupu dla KAŻDEGO wariantu. 2 epoki sprawdzają:
- Ekstrakcja tar.gz do TMPDIR
- Załadowanie checkpoint ep170
- Train loss obliczony
- Val loss obliczony
- Checkpoint zapisany

```bash
ssh eden-cluster
cd ~/DETR

# Canary V2: NO_FOURIER
VARIANT=v2 EPOCHS=2 sbatch --time=1:00:00 --gres=gpu:3 --job-name=canary_v2 run_ablation_variant.slurm

# Canary V3: NO_EL2N
VARIANT=v3 EPOCHS=2 sbatch --time=1:00:00 --gres=gpu:3 --job-name=canary_v3 run_ablation_variant.slurm

# Canary V4: QUALITY_ONLY
VARIANT=v4 EPOCHS=2 sbatch --time=1:00:00 --gres=gpu:3 --job-name=canary_v4 run_ablation_variant.slurm

# Canary V5: RANDOM
VARIANT=v5 EPOCHS=2 sbatch --time=1:00:00 --gres=gpu:3 --job-name=canary_v5 run_ablation_variant.slurm

# Canary V6: DIVERSITY_ONLY
VARIANT=v6 EPOCHS=2 sbatch --time=1:00:00 --gres=gpu:3 --job-name=canary_v6 run_ablation_variant.slurm

# Canary V7: EL2N_ONLY
VARIANT=v7 EPOCHS=2 sbatch --time=1:00:00 --gres=gpu:3 --job-name=canary_v7 run_ablation_variant.slurm
```

Monitorowanie:
```bash
squeue -u bpiotrowski
# Po zakończeniu sprawdź logi:
for v in v2 v3 v4 v5 v6 v7; do
    echo "=== $v ==="
    tail -5 logs/ablation_${v}_*.log 2>/dev/null
done
```

**Kryteria sukcesu canary:**
- [ ] Wszystkie 6 jobów zakończyły się z kodem 0
- [ ] Każdy ma checkpoint (co najmniej 1 plik .pth)
- [ ] Train loss < 1.0 po 2 epokach
- [ ] Val loss obliczony (nie NaN/Inf)
- [ ] Brak OOM (Out of Memory) w logach

**Jeśli któryś canary failuje** → napraw ZANIM przejdziesz do pełnego treningu.

Po canary wyczyść checkpointy:
```bash
for v in v2 v3 v4 v5 v6 v7; do
    rm -rf ~/DETR/ablation_checkpoints/$v/
done
```

### KROK 5: Pełny trening — 6 wariantów × 130 epok

#### Opcja A: Sekwencyjnie (bezpieczna, ~10 dni)

```bash
# V5: RANDOM (najszybszy, baseline dolny)
VARIANT=v5 sbatch --wait --job-name=ablation_v5 run_ablation_variant.slurm
cp ~/DETR/ablation_checkpoints/v5/best_model.pth ~/DETR/ablation_best/v5/best_model.pth
rm ~/DETR/ablation_checkpoints/v5/checkpoint_epoch_*.pth
echo "V5 DONE: $(date)" >> ~/DETR/ablation/progress.log

# V7: EL2N_ONLY
VARIANT=v7 sbatch --wait --job-name=ablation_v7 run_ablation_variant.slurm
cp ~/DETR/ablation_checkpoints/v7/best_model.pth ~/DETR/ablation_best/v7/best_model.pth
rm ~/DETR/ablation_checkpoints/v7/checkpoint_epoch_*.pth
echo "V7 DONE: $(date)" >> ~/DETR/ablation/progress.log

# V4: QUALITY_ONLY
VARIANT=v4 sbatch --wait --job-name=ablation_v4 run_ablation_variant.slurm
cp ~/DETR/ablation_checkpoints/v4/best_model.pth ~/DETR/ablation_best/v4/best_model.pth
rm ~/DETR/ablation_checkpoints/v4/checkpoint_epoch_*.pth
echo "V4 DONE: $(date)" >> ~/DETR/ablation/progress.log

# V6: DIVERSITY_ONLY
VARIANT=v6 sbatch --wait --job-name=ablation_v6 run_ablation_variant.slurm
cp ~/DETR/ablation_checkpoints/v6/best_model.pth ~/DETR/ablation_best/v6/best_model.pth
rm ~/DETR/ablation_checkpoints/v6/checkpoint_epoch_*.pth
echo "V6 DONE: $(date)" >> ~/DETR/ablation/progress.log

# V3: NO_EL2N (kluczowy — rozstrzygnie problem #4 EL2N bias)
VARIANT=v3 sbatch --wait --job-name=ablation_v3 run_ablation_variant.slurm
cp ~/DETR/ablation_checkpoints/v3/best_model.pth ~/DETR/ablation_best/v3/best_model.pth
rm ~/DETR/ablation_checkpoints/v3/checkpoint_epoch_*.pth
echo "V3 DONE: $(date)" >> ~/DETR/ablation/progress.log

# V2: NO_FOURIER (kluczowy — rozstrzygnie wartość Fouriera)
VARIANT=v2 sbatch --wait --job-name=ablation_v2 run_ablation_variant.slurm
cp ~/DETR/ablation_checkpoints/v2/best_model.pth ~/DETR/ablation_best/v2/best_model.pth
rm ~/DETR/ablation_checkpoints/v2/checkpoint_epoch_*.pth
echo "V2 DONE: $(date)" >> ~/DETR/ablation/progress.log
```

#### Opcja B: Po 2 naraz (~5 dni)

```bash
# Runda 1: V5 (random) + V7 (EL2N only)
VARIANT=v5 sbatch --job-name=ablation_v5 run_ablation_variant.slurm
VARIANT=v7 sbatch --job-name=ablation_v7 run_ablation_variant.slurm
# Poczekaj na zakończenie obu, cleanup:
for v in v5 v7; do
    cp ~/DETR/ablation_checkpoints/$v/best_model.pth ~/DETR/ablation_best/$v/best_model.pth
    rm ~/DETR/ablation_checkpoints/$v/checkpoint_epoch_*.pth
done
echo "RUNDA 1 DONE (V5+V7): $(date)" >> ~/DETR/ablation/progress.log

# Runda 2: V4 (quality) + V6 (diversity)
VARIANT=v4 sbatch --job-name=ablation_v4 run_ablation_variant.slurm
VARIANT=v6 sbatch --job-name=ablation_v6 run_ablation_variant.slurm
# Poczekaj, cleanup:
for v in v4 v6; do
    cp ~/DETR/ablation_checkpoints/$v/best_model.pth ~/DETR/ablation_best/$v/best_model.pth
    rm ~/DETR/ablation_checkpoints/$v/checkpoint_epoch_*.pth
done
echo "RUNDA 2 DONE (V4+V6): $(date)" >> ~/DETR/ablation/progress.log

# Runda 3: V2 (no fourier) + V3 (no EL2N)
VARIANT=v2 sbatch --job-name=ablation_v2 run_ablation_variant.slurm
VARIANT=v3 sbatch --job-name=ablation_v3 run_ablation_variant.slurm
# Poczekaj, cleanup:
for v in v2 v3; do
    cp ~/DETR/ablation_checkpoints/$v/best_model.pth ~/DETR/ablation_best/$v/best_model.pth
    rm ~/DETR/ablation_checkpoints/$v/checkpoint_epoch_*.pth
done
echo "RUNDA 3 DONE (V2+V3): $(date)" >> ~/DETR/ablation/progress.log
```

### KROK 6: Monitorowanie postępu

```bash
# Status wszystkich jobów:
squeue -u bpiotrowski

# Postęp per wariant:
cat ~/DETR/ablation/progress.log

# Aktualny epoch per wariant:
for v in v2 v3 v4 v5 v6 v7; do
    latest=$(ls -t ~/DETR/ablation_checkpoints/$v/checkpoint_epoch_*.pth 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        epoch=$(basename $latest | sed "s/checkpoint_epoch_//;s/.pth//")
        echo "$v: epoch $epoch"
    else
        echo "$v: not started / completed"
    fi
done

# TensorBoard (opcjonalnie):
ssh -L 6006:localhost:6006 eden-cluster \
    'tensorboard --logdir ~/DETR/ablation_output/ --port 6006'
```

### KROK 7: Ewaluacja — WSZYSTKIE warianty

```bash
# Osobny job SLURM — rozpakuje 91k na nowo do TMPDIR
sbatch run_ablation_eval.slurm
```

Ewaluacja obejmuje:

#### 7a. Same-distribution (val/test z video-level split)

```bash
python evaluate_ablation.py \
    --variants v1,v2,v3,v4,v5,v6,v7 \
    --variants-dir ~/DETR/ablation_best \
    --v1-checkpoint ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth \
    --test-annotations ~/DETR/ablation/v2/annotations_test.json \
    --images-dir $TMPDIR/images \
    --output ~/DETR/ablation/results_same_dist.json \
    --coco-eval
```

#### 7b. Cross-dataset (zewnętrzny Roboflow benchmark)

```bash
python evaluate_ablation.py \
    --variants v1,v2,v3,v4,v5,v6,v7 \
    --variants-dir ~/DETR/ablation_best \
    --v1-checkpoint ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth \
    --cross-annotations ~/DETR/external_benchmark/valid/_annotations.coco.json \
    --cross-images-dir ~/DETR/external_benchmark/valid/ \
    --test-annotations ~/DETR/ablation/v2/annotations_test.json \
    --images-dir $TMPDIR/images \
    --output ~/DETR/ablation/results_cross_dataset.json \
    --coco-eval
```

#### 7c. Pobranie wyników lokalnie

```bash
scp eden-cluster:~/DETR/ablation/results_same_dist.json \
    /mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Scripts/AblationStudy/results/
scp eden-cluster:~/DETR/ablation/results_cross_dataset.json \
    /mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Scripts/AblationStudy/results/
scp eden-cluster:~/DETR/ablation/progress.log \
    /mnt/f/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Eden/Scripts/AblationStudy/results/
```

---

## Parametry treningu (identyczne dla V2–V7)

| Parametr | Wartość |
|----------|---------|
| Start checkpoint | `checkpoint_epoch_170.pth` |
| Epoki | 170 → 300 (130 epok) |
| LR | 5e-5 |
| Backbone LR | 5e-6 |
| Scheduler | cosine (min 1e-7) |
| Batch/GPU | 4 |
| Patience | 30 |
| Save interval | 10 (co 10 epok) |
| Augment | TAK (online: flip, jitter — jak oryginał) |
| Split | video-level (z nazw plików, seed 42) |

## Szacunki czasowe

| Operacja | Czas |
|----------|------|
| Transfer cache + skrypty | ~2 min |
| Generowanie wariantów (CPU) | ~5 min |
| Canary 6 wariantów × 2 epoki | ~3h (równolegle) lub ~6h (sekwencyjnie) |
| Ekstrakcja 91k per job (tar.gz → TMPDIR) | ~10 min |
| 1 epoka (3×H200, batch 12) | ~18 min |
| 130 epok (3×H200) | ~39h |
| **6 wariantów sekwencyjnie** | **~10 dni** |
| **6 wariantów po 2 naraz** | **~5 dni** |
| Ewaluacja (7 wariantów, same-dist + cross) | ~3h |
| **TOTAL (Opcja B + canary + eval)** | **~6 dni** |

## Zużycie dysku

| Element | Rozmiar |
|---------|---------|
| Feature cache | 369 MB |
| Annotation JSONs (6 wariantów × 3 pliki) | ~60 MB |
| Aktywny trening (1-2 warianty, keep last 3 ckpt) | ~5-10 GB |
| Best models (7 × ~475 MB) | ~3.3 GB |
| Progress log + TensorBoard | ~100 MB |
| **Total na /evafs** | **~14 GB max** |
| **Wolne po ablacji** | **~73 GB** |

87 GB wolne → 14 GB potrzebne → **brak problemu z quota**.

## Oczekiwana tabela wynikowa

| Wariant | Same-dist mAP@0.5 | Cross-dataset mAP@0.5 | Cross-dataset F1 | Δ vs V1 |
|---------|-------------------|----------------------|-------------------|---------|
| V1 FULL | 80.8% (istniejący) | **81.5%** (istniejący) | 50.6% | baseline |
| V2 -FOURIER | ? | ? | ? | ? |
| V3 -EL2N | ? | ? | ? | ? |
| V4 QUALITY | ? | ? | ? | ? |
| V5 RANDOM | ? | ? | ? | ? |
| V6 DIVERSITY | ? | ? | ? | ? |
| V7 EL2N_ONLY | ? | ? | ? | ? |

## Co rozstrzygnie ablacja

| Wynik | Interpretacja | Implikacja dla artykułu |
|-------|--------------|------------------------|
| V2 ≈ V1 | Fourier nic nie wnosi | Usunąć Fourier z pipeline lub złagodzić claims |
| V2 << V1 | Fourier jest kluczowy | Wzmocnić argumentację Fouriera |
| V3 ≈ V1 | EL2N bias pomijalny | Rozwiązuje problem #4, dodać jako limitation |
| V3 << V1 | EL2N kluczowy | Trzeba zbadać model-agnostic alternative |
| V5 << V1 | Pipeline działa (curation > random) | Potwierdza główną tezę artykułu |
| V4 ≈ V5 | Quality filtering samo nie wystarczy | Diversity/difficulty są kluczowe |
| V6 ≈ V1 | Diversity jest wystarczająca | DINO+K-Center = rdzeń pipeline |
| V7 >> V5, V7 << V1 | EL2N pomaga ale nie wystarczy sam | EL2N + diversity = komplementarne |

### Prezentacja wyników w artykule (sugestia Gemini)

W tabeli ablacyjnej w artykule: kolumna **Cross-dataset mAP** pogrubiona jako główna metryka. Same-distribution obok z adnotacją w stopce:

> *"V1 same-distribution results reported for reference; comparison strictly valid on cross-dataset evaluation due to split distribution differences."*

Metryki do raportowania per wariant:
- **mAP@0.5** (główna metryka porównawcza)
- **mAP@0.5:0.95** (precyzja lokalizacji)
- **F1-score** (balans Precision-Recall przy progu 0.3)

## KROK 0: Aktualizacja skryptów PRZED uruchomieniem

*Źródło: weryfikacja Claude_verify (blokery B1-B6) + Gemini (G4-G5), 2026-03-20 20:50*

**PLAN jest poprawny, ale skrypty nie zostały zsynchronizowane z planem v5. Poniższe zmiany MUSZĄ być wprowadzone przed krokiem 1.**

### ~~B1: NIEAKTUALNE~~ — selekcja z pełnej puli 91k

Decyzja: selekcja z **pełnej puli ~91k** (z augmentacjami). Feature cache ma 90,141 wpisów z tego zbioru. Flaga `--skip-augmented` **NIE jest potrzebna**. Skrypt domyślnie operuje na całym cache — to jest poprawne zachowanie. **Nie wymaga zmian.**

### B2: `evaluate_ablation.py` — dodać COCO mAP (pycocotools)

Liczy tylko TP/FP/F1. Artykuł raportuje mAP@0.5 i mAP@0.5:0.95.

Dodać:
```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco_map(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {'mAP@0.5': coco_eval.stats[1], 'mAP@0.5:0.95': coco_eval.stats[0]}
```
Dodać flagę `--coco-eval`.

### B3: `generate_ablation_on_eden.py` — dodać V6 i V7

Skrypt generuje tylko v1-v5.

Dodać:
```python
# V6: DIVERSITY_ONLY — tylko DINO + K-Center Greedy
def generate_v6(features, target_size=20000):
    dino_features = features['dino']
    selected = k_center_greedy(dino_features, target_size)
    return selected

# V7: EL2N_ONLY — random 2x pool, top-k najtrudniejszych wg EL2N
def generate_v7(features, target_size=20000):
    el2n_scores = features['el2n']
    pool = random.sample(range(len(el2n_scores)), min(target_size * 2, len(el2n_scores)))
    sorted_pool = sorted(pool, key=lambda i: el2n_scores[i], reverse=True)
    return sorted_pool[:target_size]
```

### B4: `run_ablation_variant.slurm` — rozszerzyć tablicę VARIANTS

Zmienić (~linia 28-29):
```bash
# Z:
VARIANTS=(v1 v2 v3 v4 v5)
# Na:
VARIANTS=(v1 v2 v3 v4 v5 v6 v7)
```

### B5: `evaluate_ablation.py` — dodać V6/V7 do słownika

Zmienić:
```python
VARIANTS = {
    "v1": "FULL", "v2": "NO_FOURIER", "v3": "NO_EL2N",
    "v4": "QUALITY_ONLY", "v5": "RANDOM",
    "v6": "DIVERSITY_ONLY", "v7": "EL2N_ONLY",  # dodane
}
```

### B6: `run_ablation_variant.slurm` — parametryzować EPOCHS

Zmienić (~linia 208):
```bash
# Z:
    --epochs 300 \
# Na:
    --epochs ${EPOCHS:-300} \
```

### L5: Mapowanie klas (category_id) — Roboflow vs DETR

Model DETR trenowany na naszych danych ma swoje category_id (np. 0="tooltip"). Roboflow dataset może mieć inne (np. 1="Instruments", 2="Pupil"). Jeśli evaluate_ablation.py porównuje predykcje z ground truth bez mapowania → żadna predykcja się nie matchuje → mAP = 0%.

**Jak naprawić:**

1. Sprawdzić klasy w Roboflow:
```python
import json
for split in ['train','valid','test']:
    with open(f'/mnt/e/cataract_surgery_Instruments_detection.v1i.coco/{split}/_annotations.coco.json') as f:
        d = json.load(f)
    print(f'{split}: {d["categories"]}')
```

2. Sprawdzić klasy w modelu DETR:
```python
# Z training annotations:
with open('merged_20k_annotations_fixed.json') as f:
    d = json.load(f)
print(d['categories'])
```

3. Jeśli ID się różnią — dodać mapowanie w `evaluate_ablation.py`:
```python
# Przed ewaluacją cross-dataset: przelicz category_id predykcji
CATEGORY_MAP = {0: 1}  # DETR tooltip(0) → Roboflow Instruments(1)
for pred in predictions:
    pred['category_id'] = CATEGORY_MAP.get(pred['category_id'], pred['category_id'])
```

4. Alternatywnie: w preflight (krok 3.5) dodać automatyczny check:
```bash
# Preflight: sprawdź czy category_id się zgadzają
python3 -c "
import json
with open('merged_20k_annotations_fixed.json') as f: train_cats = {c['id']:c['name'] for c in json.load(f)['categories']}
with open('external_benchmark/valid/_annotations.coco.json') as f: ext_cats = {c['id']:c['name'] for c in json.load(f)['categories']}
print('Train:', train_cats)
print('External:', ext_cats)
if train_cats != ext_cats: print('WARNING: category_id mismatch! Dodaj mapowanie!')
"
```

### G4: Sprawdzić ColorJitter w `detr_train_ablation.py`

Porównać augmentacje online z `detr_train_optimized.py` (działający). Jeśli oryginał ma ColorJitter — dodać do nowego trainera. Wszystkie warianty muszą mieć identyczne augmentacje online.

### G5: Stworzyć `run_ablation_eval.slurm`

Nowy SLURM job do ewaluacji (krok 7). Musi:
1. Rozpakować `datasets_20250606.tar.gz` do TMPDIR (bo treningowy TMPDIR już wyczyszczony)
2. Uruchomić `evaluate_ablation.py` z `--coco-eval` na same-dist + cross-dataset
3. Zapisać wyniki do `~/DETR/ablation/results_*.json`

```bash
#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=ablation_eval
#SBATCH --chdir=/mnt/evafs/faculty/home/bpiotrowski/DETR

source /mnt/evafs/software/anaconda/v.4.0/etc/profile.d/conda.sh
conda activate yolo_py310

TMPDIR=${SLURM_TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID}}
tar -xzf ~/datasets_20250606.tar.gz -C $TMPDIR/

python evaluate_ablation.py \
    --variants v1,v2,v3,v4,v5,v6,v7 \
    --best-models-dir ~/DETR/ablation_best \
    --v1-checkpoint ~/DETR/Checkpoints/20k_finetune_v2_fixed/checkpoint_epoch_210.pth \
    --test-annotations ~/DETR/ablation/v2/annotations_test.json \
    --images-dir $TMPDIR/images \
    --external-dataset ~/DETR/external_benchmark \
    --output ~/DETR/ablation/results.json \
    --coco-eval
```

---

## Status problemów po weryfikacji finalnej (2026-03-20 21:29)

*Źródło: finalna weryfikacja Claude_verify + Codex + Gemini*

### Naprawione (kod zsynchronizowany z planem)

| # | Problem | Status |
|---|---------|--------|
| B1 | ~~`--skip-augmented`~~ | ✅ NIEAKTUALNE — selekcja z pełnej puli 91k |
| B2 | COCO mAP w `evaluate_ablation.py` | ✅ NAPRAWIONE (Claude_worker) |
| B3 | V6/V7 w `generate_ablation_on_eden.py` | ✅ NAPRAWIONE (Claude_worker) |
| B4 | V6/V7 w `run_ablation_variant.slurm` | ✅ NAPRAWIONE (Claude_worker) |
| B5 | V6/V7 w `evaluate_ablation.py` | ✅ NAPRAWIONE (Claude_worker) |
| B6 | EPOCHS parametryzowane w SLURM | ✅ NAPRAWIONE (Claude_worker) |
| G4 | ColorJitter — identyczny jak oryginał (zakomentowany) | ✅ ZWERYFIKOWANE |
| G5 | `run_ablation_eval.slurm` stworzony | ✅ NAPRAWIONE (Claude_worker) |

### Blokery na Eden — wymagają akcji PRZED uruchomieniem

| # | Problem | Jak naprawić | Kto |
|---|---------|-------------|-----|
| **E1** | **`sklearn` brak w `yolo_py310` na Eden** | `ssh eden-cluster 'conda activate yolo_py310 && pip install scikit-learn'` | Claude_worker |
| **E2** | **`~/DETR/external_benchmark/` nie istnieje na Eden** | Przesłać dataset Roboflow: `scp -r /mnt/f/.../external_benchmark eden-cluster:~/DETR/external_benchmark/`. Potrzebne: `images/` + `annotations.json` w formacie COCO. **Blokuje tylko krok 7 (eval), nie trening.** | Claude_worker |
| **E3** | **`hopper` (8×H100) zajęty** | Usunąć `--nodelist=hopper` z SLURM lub zmienić na dostępny node. Alternatywne nody: hopper-2 (H200), dgx-2..4 (A100), pascal (P100). **Krok 3.5 (weryfikacja TUŻ PRZED) rozwiązuje to dynamicznie.** | automatyczny |

### Ostrzeżenia (nie blokują, ale warto naprawić)

| # | Problem | Status |
|---|---------|--------|
| W1 | `run_ablation_all.sh` nieaktualny (5 wariantów, afterany) | DO NAPRAWY — zlecone Claude_worker |
| W2 | Cache mismatch: 90,141 wpisów w cache vs 91,336 w COCO JSON — 1,195 obrazów (w tym 285 oryginalnych) nie ma w cache | Generator powinien logować brakujące, nie failować. Sprawdzić czy braki są rozproszone po wideo, czy to całe wideo. |
| W3 | `detr_train_ablation.py` uproszczony vs oryginał — brak `cosine_warmup`, `compile_model`, rozbudowanego logowania (Codex) | Nie bloker — te features nie wpływają na wyniki, ale trening może być minimalnie inny niż oryginał |
| W4 | Stare komentarze "5 variants" w kilku plikach (Codex) | Kosmetyczne — ryzyko pomyłki operatorskiej |

### Luki z finalnej weryfikacji v8 (2026-03-20 21:35)

*Źródło: Claude_verify + Codex + Gemini — odpowiedzi na plan v8*

| # | Luka | Źródło | Priorytet | Jak naprawić |
|---|------|--------|-----------|-------------|
| **L1** | **Best model format: `.pth` zamiast folder HuggingFace** — Benchmarki w artykule ładowały `checkpoint_epoch_*.pth` (torch.load), nie foldery HF. Trainer zapisuje best jako folder HF (save_pretrained) ale to nigdy nie było używane w benchmarkach. | Claude_verify, Codex | 🔴 DO NAPRAWY | **Decyzja: `.pth` wszędzie.** Zmienić w `detr_train_ablation.py` (linie 379-380): `save_pretrained()` → `torch.save(state_dict, best_model.pth)`. Zmienić w `evaluate_ablation.py`: ładowanie V2-V7 z `.pth` (torch.load) zamiast folderu HF. Spójne z benchmarkami artykułu. |
| **L4** | **`evaluate_ablation.py` niespójny format ładowania** — V2-V7 ładuje z folderu HF (from_pretrained), V1 z `.pth` (torch.load). Po naprawie L1 wszystko ma być `.pth`. | Codex | ✅ NAPRAWIONE | Claude_worker zmienił na torch.load() dla wszystkich wariantów. |
| **L5** | **Mapowanie klas (category_id) Roboflow vs DETR** — model DETR może mieć inne category_id (np. 0="tooltip") niż Roboflow dataset (np. 1="Instruments"). Bez mapowania: mAP=0%. | Gemini | 🔴 DO NAPRAWY | Opis poniżej. |
| **L2** | ~~**V1 same-dist eval unfair**~~ | Claude_verify, Gemini | ✅ ROZWIĄZANE | **Decyzja: random_split(seed=42) dla WSZYSTKICH wariantów (V1-V7).** Identyczny split = identyczne warunki = fair comparison. V1 nie wymaga retreningu. Cross-dataset jako główna metryka (niezależna od splitu). Opis dla recenzenta w sekcji "Split treningowy". |
| **L3** | ~~`--best-models-dir` vs `--variants-dir`~~ | Codex | ✅ NAPRAWIONE | Plan krok 7a/7b + `run_ablation_eval.slurm` zsynchronizowane z flagami evaluate_ablation.py: `--variants-dir`, `--cross-annotations`, `--cross-images-dir`. |

### Decyzje podjęte

| Decyzja | Odpowiedź | Uzasadnienie |
|---------|-----------|-------------|
| Selekcja z 22k orig vs 91k z aug? | **91k z aug** | Feature cache policzone na 91k. 20k z 22k = 91% puli (za mało selektywne). 20k z 91k = 22% (sensowne). |
| 3 seedy? | **1 run najpierw, potem decyzja** | Warianty losowe (V4, V5, V7) są wrażliwe na seed — inne losowanie → inny zbiór 20k → inny mAP. Warianty deterministyczne (V1, V2, V3, V6) używają K-Center/klasteryzacji na cache → ten sam cache daje zawsze ten sam wynik. Strategia: 1 seed (42) dla wszystkich. Jeśli różnice między wariantami < 2pp → wynik niepewny → powtórzyć V4, V5, V7 z 3 seedami (42, 123, 456) i raportować mean ± std. Jeśli różnice > 5pp → 1 seed wystarczy, wynik jednoznaczny. |
| V1 przetrenować? | **NIE — random_split(seed=42) dla wszystkich V1-V7** | Identyczny split = fair comparison. Cross-dataset niezależny. V1 checkpointy reużywane bez retreningu. |

### Kolejność uruchamiania

```
TERAZ (przed treningiem):
  [ ] E1: pip install scikit-learn na Eden
  [ ] E2: przesłać external_benchmark na Eden (lub odłożyć do eval)
  [ ] W1: zaktualizować run_ablation_all.sh (5→7 wariantów)

KROK 1-3: Transfer + generowanie wariantów
KROK 3.5: Weryfikacja stanu Eden (GPU, dysk, node)
KROK 4: Canary — WSZYSTKIE 6 wariantów × 2 epoki
KROK 5: Pełny trening — 6 wariantów × 130 epok
KROK 6: Monitorowanie
KROK 7: Ewaluacja (wymaga E2)
```

---

*Plan v16 — 2026-03-20 23:35:00. L3 naprawione: CLI flagi w planie i run_ablation_eval.slurm zsynchronizowane z evaluate_ablation.py. Ścieżki external benchmark poprawione na valid/_annotations.coco.json.*
