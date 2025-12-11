# Session Summary: sesja_2025-10-31_16-30-00_training

## Metadata
- **Type:** training
- **Date:** 2025-10-31
- **Time:** 16:30:00 UTC
- **Duration:** n/a
- **Status:** Planned (ready to run)

## Objective
Zbudować lepszy, rzetelny dataset z filmów zaćmy (Cataract), a następnie wytrenować nowy jedno‑klasowy DETR (klasa: `tooltip` – końcówka narzędzia), z uczciwą walidacją (COCO mAP na val i test), bez przecieku między splitami.

## Context
Poprzedni zbiór „DETR_augmented_dataset” zawierał nadmiar bliskich duplikatów, brak negatywów i brak jednoznacznych splitów po nagraniach – co obniżało recall DETR i zawyżało walidację. Nowy pipeline:
- wybór reprezentatywnych klatek (odfiltrowanie rozmyć i duplikatów),
- opcjonalne pseudo‑etykiety YOLO (uściślenie pozytywów),
- podział COCO po video_stem (brak przecieku),
- trening DETR dla jednej klasy z COCO mAP na walidacji.

## Actions Taken
1. Dodano builder datasetu `Cataract_DETR_DatasetBuilder/*` (extract, pseudo‑labels, split, train, eval).
2. Przygotowano config buildera z 1 klasą `tooltip` i negatywami do 30%.
3. Napisano nowy skrypt treningowy DETR „single‑class” z ewaluacją COCO per epoka.
4. Dodano evaluator offline COCO na test.

## Detailed Training Plan (Step‑by‑Step)

### A) Budowa datasetu z filmów
- Wejście: `E:/Cataract/videos/micro` (WSL: `/mnt/e/Cataract/videos/micro`)
- Wybór klatek: `target_fps=2`, `blur_threshold=80` (Variance of Laplacian), `phash_threshold=10` (Hamming) → redukcja rozmyć i near‑dupli.
- Negatywy: włączone, `max_ratio=0.3` (do 30% obrazów bez narzędzia), poprawia kalibrację no‑object.
- Split: po `video_stem` (np. `<nazwa>_frame_00001234.jpg` → grupa `<nazwa>`), proporcje: train 70%, val 15%, test 15%.
- Pseudo‑etykiety YOLO (opcjonalnie): Twoje wagi `.pt`, próg `conf=0.25`.

Komenda (end‑to‑end):
```
python Cataract_DETR_DatasetBuilder/build_dataset.py \
  --config Cataract_DETR_DatasetBuilder/builder_config.yaml
```
Wynik:
- `output_cataract/images/*.jpg`
- `output_cataract/annotations_{train,val,test}.json`

### B) Trening DETR – jedna klasa `tooltip`
- Model: `facebook/detr-resnet-50` (num_labels=1, id2label: `{0: 'tooltip'}`)
- Hiperparametry (startowe):
  - `epochs=50`, `batch_size=8`, `lr=1e-4`, `lr_backbone=1e-5`, `weight_decay=1e-4`, `num_queries=100`
  - AMP: `--use_amp` (jeżeli GPU wspiera)
- Dane:
  - `--images_dir <output_cataract>/images`
  - `--train_json <output_cataract>/annotations_train.json`
  - `--val_json <output_cataract>/annotations_val.json`
  - Bez losowych splitów w skrypcie – używa gotowych COCO splitów.
- Walidacja podczas treningu:
  - `--coco_eval_every 1` → po każdej epoce liczy mAP@.5, mAP@[.5:.95], AR@100.

Komenda:
```
python Cataract_DETR_DatasetBuilder/detr_single_class_train.py \
  --images_dir /mnt/f/.../output_cataract/images \
  --train_json /mnt/f/.../output_cataract/annotations_train.json \
  --val_json /mnt/f/.../output_cataract/annotations_val.json \
  --output_dir /mnt/f/.../output_cataract/detr_tooltip_train \
  --model_checkpoint facebook/detr-resnet-50 \
  --epochs 50 --batch_size 8 --use_amp --coco_eval_every 1
```
Artefakty:
- `best_model_state.pth`, `last_model_state.pth`, `label_mapping.json`

### C) Końcowa ewaluacja na teście (offline)
```
python Cataract_DETR_DatasetBuilder/eval_coco_offline.py \
  --images_dir /mnt/f/.../output_cataract/images \
  --test_json /mnt/f/.../output_cataract/annotations_test.json \
  --model_checkpoint facebook/detr-resnet-50 \
  --weights_path /mnt/f/.../output_cataract/detr_tooltip_train/best_model_state.pth \
  --batch_size 8 --conf_threshold 0.0
```
Zwracane metryki: mAP@0.5:0.95, mAP@0.5, mAP@0.75, AR@100.

## Model Configuration
- **Model:** DETR (ResNet‑50 backbone)
- **Pretrained:** COCO → fine‑tune 1 klasa (tooltip)
- **Classes:** 1 (`tooltip`)
- **num_queries:** 100
- **Optimizer:** AdamW (backbone LR 1e‑5, inne warstwy 1e‑4)
- **Scheduler:** (opcjonalnie – na start brak)

## Dataset
- **Źródło:** filmy `E:/Cataract/videos/micro`
- **Selekcja:** target_fps=2, blur≥80, pHash≥10
- **Negatywy:** do 30% per split
- **Splity:** 70/15/15, grupowanie po video_stem
- **COCO:** `category_id=int` i spójny (1) → w trainerze mapowane do indeksu 0

## Hardware Configuration (zalecenia)
- **GPU:** dowolny z ≥8GB VRAM (AMP zalecany)
- **Batch:** 8 (zwiększ jeśli VRAM pozwala)
- **Precision:** AMP (`--use_amp`)

## Results (oczekiwane)
- Lepsza kalibracja i recall vs poprzedni DETR dzięki negatywom i zmniejszeniu redundancji.
- Uczciwe mAP dzięki splitom bez przecieku.

## Issues Encountered
- Brak (sesja planistyczna/implementacyjna). Ewentualne problemy: brak pycocotools na Windows → użyj `pycocotools-windows`.

## Conclusions
Nowy pipeline budowy datasetu i treningu DETR adresuje wcześniejsze wąskie gardła (redundancja, brak negatywów, przeciek splitów). Oczekiwane: stabilniejszy training loss/val loss, poprawa mAP@[.5:.95] i AR.

## Next Steps
- [ ] Uruchomić budowę datasetu na wszystkich filmach
- [ ] Wytrenować DETR (50 epok), monitorować mAP na val
- [ ] Przeprowadzić finalną ewaluację na teście
- [ ] Porównać z YOLO na tym samym teście

## Files Generated
- `configs/train_detr_tooltip.yaml` – snapshot parametrów treningu
- `configs/builder_config_snapshot.yaml` – snapshot buildera
- `configs/offline_eval.yaml` – snapshot ewaluacji testu

## Commands Used
```bash
# Build dataset
python Cataract_DETR_DatasetBuilder/build_dataset.py --config Cataract_DETR_DatasetBuilder/builder_config.yaml

# Train DETR (single class)
python Cataract_DETR_DatasetBuilder/detr_single_class_train.py \
  --images_dir /mnt/f/.../output_cataract/images \
  --train_json /mnt/f/.../output_cataract/annotations_train.json \
  --val_json /mnt/f/.../output_cataract/annotations_val.json \
  --output_dir /mnt/f/.../output_cataract/detr_tooltip_train \
  --model_checkpoint facebook/detr-resnet-50 \
  --epochs 50 --batch_size 8 --use_amp --coco_eval_every 1

# Offline COCO eval (test)
python Cataract_DETR_DatasetBuilder/eval_coco_offline.py \
  --images_dir /mnt/f/.../output_cataract/images \
  --test_json /mnt/f/.../output_cataract/annotations_test.json \
  --model_checkpoint facebook/detr-resnet-50 \
  --weights_path /mnt/f/.../output_cataract/detr_tooltip_train/best_model_state.pth \
  --batch_size 8 --conf_threshold 0.0
```

## Related Work
- Benchmark: `.sessions/benchmark/sesja_2025-10-29_14-29-09_benchmark`
- Builder i trening: `Cataract_DETR_DatasetBuilder/*`

---

**Session Created:** 2025-10-31 16:30:00 UTC
**Last Updated:** 2025-10-31 16:30:00 UTC
