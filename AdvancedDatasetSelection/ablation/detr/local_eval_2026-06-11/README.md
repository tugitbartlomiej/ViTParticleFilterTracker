# Lokalna ewaluacja ablacji DETR V1-V7 (2026-06-11)

Ewaluacja 7 wytrenowanych wariantow ablacji pipeline'u selekcji danych
(Multi-Feature Cluster-Based Dataset Distillation) **w calosci na lokalnym
komputerze** (RTX 3070 8 GB) - zadnych operacji na klastrze Eden.

Wyniki zasila tabela "Ablation Study" w artykule IEEE Access
(`F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\Uwagi\Merge\DETR_IEEE_article_merged_v1.tex`).
Plan calosci: `DETR_IEEE\.todo\Todo_2026-06-11_180155_Local_ablation_eval_V1-V7\TODO.md`.

## Status

- [x] Skrypty zaimplementowane (2026-06-11)
- [ ] Smoke testy uruchomione (NAJPIERW v1!)
- [ ] Pelne runy external
- [ ] Pelne runy same-dist
- [ ] Agregacja + sanity-check vs 85.74%

## Struktura

```
local_eval_2026-06-11\
├── README.md                      <- ten plik
├── run_all_smoke_tests.py         <- ORKIESTRATOR smoke: sekwencyjnie v1->v7 w jednym
│                                     procesie + zbiorcze podsumowanie OK/BLAD
├── run_all_full_evals.py          <- ORKIESTRATOR pelnych ewaluacji: faza 1 external
│                                     v1->v7, faza 2 same-dist v1->v7; blad wariantu
│                                     nie przerywa sekwencji; na koncu sanity-check V1
│                                     vs 85.74% + automatyczna tabela SUMMARY.md
├── common\
│   ├── paths_config.py            <- WSZYSTKIE sciezki + etykiety wariantow
│   ├── ablation_eval_lib.py       <- cala logika (ladowanie, inferencja, COCO eval)
│   └── aggregate_results.py       <- sklada wyniki V1-V7 w jedna tabele (SUMMARY.md)
├── v1\ ... v7\                    <- folder per wariant, w kazdym 3 skrypty:
│   ├── run_smoke_test.py          <- 50 obrazow external + 50 same-dist (test poprawnosci)
│   ├── run_eval_external.py       <- pelny cross-dataset (2333 obrazy; FULL+CLEAN+per-split)
│   └── run_eval_samedist.py       <- pelny same-distribution (splity val+test)
└── results\<vid>\{smoke,external,samedist}\   <- REPORT.md, metrics.json, predykcje
```

Skrypty per wariant sa identyczne - wariant rozpoznaja z nazwy folderu,
w ktorym leza. Checkpointy i dane konfiguruje wylacznie `common\paths_config.py`.

## Warianty i checkpointy

| ID | Wariant | Checkpoint |
|----|---------|-----------|
| V1 | FULL pipeline (baseline) | `...\2025-12-13_20kDataset_small_LR\ckpt_20k_finetune\checkpoint_epoch_270.pth` (headline 85.74%) |
| V2 | NO_FOURIER | `...\Ablations\ablation_best\v2\best_model.pth` (jedyna kopia) |
| V3 | NO_EL2N | `...\Ablations\ablation_best\v3\best_model.pth` (jedyna kopia) |
| V4 | QUALITY_ONLY * | `...\Ablations\ablation_best\v4\best_model.pth` (jedyna kopia) |
| V5 | RANDOM | `...\Ablations\ablation_best\v5\best_model.pth` (jedyna kopia) |
| V6 | DIVERSITY_ONLY (DINO+K-Center) | `...\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v6\best_model.pth` |
| V7 | EL2N_ONLY (random+EL2N rerank) | `...\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v7\best_model.pth` |

\* Etykieta V4 do potwierdzenia z logow treningu (PLAN.md z marca mowi
NO_CLUSTERING, sesja 2026-05-24 mowi QUALITY_ONLY). NIE wpisywac do artykulu
przed weryfikacja. UWAGA: dla V6/V7 NIE uzywac starych kopii z
`ablation_best\v6,v7` - sa sprzed udanych rerunow.

## Datasety

1. **External benchmark (cross-dataset):** `E:\cataract_surgery_Instruments_detection.v1i.coco`
   - foldery train(2083)/valid(200)/test(50) scalane do JEDNEGO zbioru (FULL=2333)
     - zgodnie z narracja artykulu "ONE undivided test set",
   - CLEAN = FULL minus 48 leaked images
     (`YOLO_DETR_Benchmarks\scripts\leaked_valid_images.json`),
   - dodatkowo metryki per-split (porownania z historycznymi benchmarkami).
2. **Same-distribution:** anotacje per wariant
   `...\Ablations\ablation\<vid>\annotations_{val,test}.json` (video-level split,
   seed=42, identyczny dla wszystkich wariantow); obrazy:
   `...\Eden\Datasets\datasets_20250606.tar\datasets_20250606\datasets\DETR_augmented_dataset_20250218\images\`
   (90,141 jpg).

## Kolejnosc uruchamiania (gdy zapadnie decyzja o starcie)

```powershell
# KROK 1: smoke testy WSZYSTKICH wariantow po kolei (v1->v7, ~15-30 min lacznie)
py -3.11 run_all_smoke_tests.py
#   warianty wybiorczo:           --variants v1 v5 v6
#   przerwij po pierwszym bledzie: --stop-on-error
# (alternatywnie pojedynczo: py -3.11 v1\run_smoke_test.py)

# KROK 2: pelne ewaluacje WSZYSTKICH wariantow - jeden orkiestrator
#   faza 1: external v1->v7 (~2-4 h), faza 2: same-dist v1->v7 (~10-18 h)
#   blad jednego wariantu NIE przerywa sekwencji; na koncu sanity-check V1
#   vs 85.74% i automatyczna tabela results\SUMMARY.md
py -3.11 run_all_full_evals.py
#   tylko external (najwazniejsze liczby): --mode external
#   tylko same-dist (np. na noc):          --mode samedist
#   wybrane warianty:                      --variants v1 v5 v6

# (alternatywnie pojedynczo:)
py -3.11 v1\run_eval_external.py    # sanity: FULL powinno dac ~85.74%
py -3.11 v1\run_eval_samedist.py    # mozna ograniczyc: --splits test

# W KAZDEJ CHWILI: tabela zbiorcza z tego, co juz policzone
py -3.11 common\aggregate_results.py
```

## Protokol i parametry

- mAP: pycocotools COCOeval (bbox); raportujemy mAP@0.5, mAP@[0.5:0.95], mAP@0.75, AR@100
- TP/FP/FN: conf=0.3, IoU=0.5 (spojnie z wczesniejszymi benchmarkami projektu)
- batch=4 (8 GB VRAM), obrazy przetwarzane przez DetrImageProcessor (DETR-R50)
- predykcje kazdego runu zapisywane do results\ (pelna reprodukowalnosc metryk
  bez ponownej inferencji)
- num_queries / num_labels / typ backbone inferowane z wag checkpointa
  (zabezpieczenie przed niezgodnoscia architektury)

## Znane ryzyka (z TODO)

1. Format best_model.pth (surowy state_dict vs dict) - loader obsluguje oba.
2. transformers 5.1.0 vs API z marca - zweryfikuje smoke test.
3. Sanity: V1 external FULL ~85.74%; rozbieznosc = inny protokol niz LEGACY
   benchmark (sprawdzic `BENCHMARK_ALL_EPOCHS_20260112_194440\BENCHMARK_REPORT.md`).
4. Etykieta V4 (patrz wyzej).
