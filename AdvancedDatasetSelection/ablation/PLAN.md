# Ablation Study: Hierarchical Dataset Distillation Pipeline

## Cel

Wykazać, że **każdy etap pipeline'u selekcji danych wnosi mierzalną wartość** do jakości modelu DETR.
Artykuł twierdzi 4-etapowy pipeline (Fourier + DINO + SAM/K-Means + EL2N), ale brak dowodów,
że usunięcie dowolnego etapu pogarsza wyniki. Ablacja to naprawia.

## Hipoteza

H0: Losowa selekcja 20k próbek daje porównywalny mAP do pipeline'u.
H1: Pipeline systematycznie poprawia mAP; usunięcie etapu powoduje mierzalny spadek.

## 5 wariantów (po 20,000 próbek każdy)

| ID | Nazwa | Opis | Wagi (D/F/S/E) |
|----|-------|------|-----------------|
| V1 | FULL_PIPELINE | Kompletny pipeline (baseline) | 0.35/0.15/0.20/0.30 |
| V2 | NO_FOURIER | Bez Fourier features | 0.45/0.00/0.25/0.30 |
| V3 | NO_EL2N | Bez difficulty scoring | 0.45/0.20/0.35/0.00 |
| V4 | NO_CLUSTERING | Fourier quality filter + random sample | N/A (random) |
| V5 | RANDOM | Losowe 20k z pełnej puli 90k | N/A (random) |

## Kluczowe zasady

### Video-level split (NAPRAWIA data leakage!)
- Wszystkie warianty używają **tego samego** video-level splitu
- `generate_video_split.py` tworzy `video_split.json` raz
- 50 wideo → 35 train / 8 val / 7 test (70/15/15)
- Split jest deterministyczny (seed=42)
- Augmentowane kopie (`_aug_1/2/3`) podążają za oryginałem

### Identyczny trening
- Checkpoint startowy: epoch 170 (pretrained na 91k)
- LR: 5e-5, backbone: 5e-6, cosine scheduler
- 300 epok, batch 4/GPU, 6× H100
- Augmentacje w DataLoader: włączone (`--augment`)

### Ewaluacja
- Same-distribution: val split (te same wideo we wszystkich wariantach)
- Cross-dataset: Roboflow Cataract Surgery Instruments (external)
  - Z wykluczeniem 48 leaked images (leaked_valid_images.json)

## Kolejność uruchamiania

```
1. [LOKALNIE] generate_video_split.py    → video_split.json
2. [LOKALNIE] generate_ablation_datasets.py → 5× (annotations + images.tar)
3. [EDEN] scp archives + annotations + scripts to Eden
4. [EDEN] sbatch run_ablation_v1.slurm   (lub pomiń — V1 = istniejący model)
5. [EDEN] sbatch run_ablation_v2.slurm
6. [EDEN] sbatch run_ablation_v3.slurm
7. [EDEN] sbatch run_ablation_v4.slurm
8. [EDEN] sbatch run_ablation_v5.slurm
9. [EDEN/LOKAL] evaluate_ablation.py      → tabela porównawcza
```

## Szacowany czas

| Faza | Czas | Uwagi |
|------|------|-------|
| Generowanie splitów + datasetów | ~2h | Lokalnie, feature cache dostępny |
| Transfer na Eden (5× ~4GB) | ~30min | SCP |
| Trening 1 wariantu (300 ep) | ~18h | 6×H100, batch 4, ~20k images |
| Trening 5 wariantów | ~90h | Sekwencyjnie na 1 nodzie |
| Ewaluacja | ~2h | 5 modeli × 2 datasety |
| **TOTAL** | **~4-5 dni** | Z kolejkowaniem SLURM |

## Interpretacja wyników

### Tabela docelowa (do artykułu)

| Variant | mAP@0.5 (val) | mAP@0.5 (cross) | F1 (val) | F1 (cross) | Δ vs Full |
|---------|---------------|------------------|----------|------------|-----------|
| V1: Full Pipeline | X.XX% | X.XX% | X.XX% | X.XX% | baseline |
| V2: -Fourier | ... | ... | ... | ... | -Y.Ypp |
| V3: -EL2N | ... | ... | ... | ... | -Y.Ypp |
| V4: -Clustering | ... | ... | ... | ... | -Y.Ypp |
| V5: Random | ... | ... | ... | ... | -Y.Ypp |

### Oczekiwane wyniki
- V5 (Random) < V4 (QualityFilter) < V3/V2 < V1 (Full)
- Jeśli V5 ≈ V1 → pipeline nie pomaga (H0 potwierdzone)
- Jeśli V1 >> V5 → pipeline kluczowy (H1 potwierdzone)
- Cross-dataset gap powinien być większy niż same-distribution

## Pliki

```
AdvancedDatasetSelection/ablation/
├── PLAN.md                          ← ten plik
├── generate_video_split.py          ← tworzy video_split.json
├── generate_ablation_datasets.py    ← generuje 5 wariantów
└── evaluate_ablation.py             ← zbiera wyniki

Eden/Scripts/AblationStudy/
├── run_ablation_v1.slurm
├── run_ablation_v2.slurm
├── run_ablation_v3.slurm
├── run_ablation_v4.slurm
├── run_ablation_v5.slurm
└── detr_train_ablation.py           ← zmodyfikowany trainer z video-level split
```
