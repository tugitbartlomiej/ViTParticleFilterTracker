# Ablation Study — Dataset Curation Pipeline (2026-03-20)

## Status: Plan v16 gotowy, canary zlecony

## 7 wariantow ablacji
- V1 FULL: Fourier+DINO+SAM+EL2N (0.15/0.35/0.20/0.30) — istniejace checkpointy
- V2 NO_FOURIER: DINO+SAM+EL2N (0.00/0.45/0.25/0.30)
- V3 NO_EL2N: Fourier+DINO+SAM (0.20/0.45/0.35/0.00) — rozstrzygnie EL2N bias
- V4 QUALITY_ONLY: Fourier filtruje + random 20k
- V5 RANDOM: losowe 20k (seed 42) — dolna granica
- V6 DIVERSITY_ONLY: DINO+K-Center, bez Fourier/EL2N
- V7 EL2N_ONLY: random + EL2N reranking

## Kluczowe decyzje
- Selekcja z pelnej puli ~91k (z augmentacjami)
- random_split(seed=42) dla WSZYSTKICH wariantow (fair comparison, spojne z V1)
- Cross-dataset (Roboflow) jako glowna metryka — niezalezna od splitu
- 1 seed najpierw, 3 seedy jesli roznice < 2pp
- V1 nie wymaga retreningu — reuzywamy istniejace checkpointy

## Pliki
- Plan: Eden/Scripts/AblationStudy/PLAN.md (v16, 701 linii)
- Trainer: Eden/Scripts/AblationStudy/detr_train_ablation.py
- Generator: Eden/Scripts/AblationStudy/generate_ablation_on_eden.py
- Ewaluator: Eden/Scripts/AblationStudy/evaluate_ablation.py (COCO mAP)
- SLURM: Eden/Scripts/AblationStudy/run_ablation_variant.slurm

## Rozwiazane blokery
- E1: sklearn zainstalowany (1.7.2) w yolo_py310 na Eden
- L1/L4: format .pth (torch.save) zamiast HuggingFace folder
- L5: category_id zgodne (id=1, name=tooltip)

## Czyszczenie Eden (2026-03-20)
- Usunieto UCO3D: 513 GB z group storage + ~/uco3d: 31 MB
- Dysk: 100% -> 11% (459 GB odzyskane)

## Data leakage audit
- 3 warstwy: transductive (pipeline na 90k), augmentation (random_split), EL2N circular
- Benchmarki cross-dataset czyste (48 Roboflow leaked images wykluczone)
- Cross-dataset eval niezalezny od splitu treningowego

## Nastepne kroki
1. Przeslac external_benchmark (Roboflow) na Eden
2. Canary 6 wariantow x 2 epoki
3. Pelny trening 6 wariantow x 130 epok
4. Ewaluacja i tabela wynikowa
