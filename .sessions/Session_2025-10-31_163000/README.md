# Session: sesja_2025-10-31_16-30-00_training

**Session Type:** training
**Created:** 2025-10-31 16:30:00 UTC
**Description:** Przygotowanie nowego datasetu z filmów Cataract i trening jedno‑klasowego DETR do detekcji końcówki narzędzia (tooltip) z rzetelną walidacją COCO.

## Quick Reference
- **Model:** DETR (facebook/detr-resnet-50) fine-tune, 1 klasa = `tooltip`
- **Skrypty:** `Cataract_DETR_DatasetBuilder/*` (extract, pseudo‑labels, split, train, eval)
- **Dataset (output):** `<output_cataract>/images`, `annotations_{train,val,test}.json`
- **Trening:** `detr_single_class_train.py` (bez losowych splitów, COCO eval co epokę)
- **Ewaluacja test:** `eval_coco_offline.py` (mAP/AR na annotations_test.json)

## Files in This Session
- SESSION_SUMMARY.md – pełny opis, kroki i komendy
- configs/train_detr_tooltip.yaml – snapshot parametrów treningu
- configs/builder_config_snapshot.yaml – snapshot użytego configu buildera
- configs/offline_eval.yaml – snapshot parametrów ewaluacji testu

## Commands (WSL przykłady)
- Budowa datasetu (end‑to‑end):
```
python Cataract_DETR_DatasetBuilder/build_dataset.py \
  --config Cataract_DETR_DatasetBuilder/builder_config.yaml
```
- Trening (1 klasa, COCO eval co epokę):
```
python Cataract_DETR_DatasetBuilder/detr_single_class_train.py \
  --images_dir /mnt/f/.../output_cataract/images \
  --train_json /mnt/f/.../output_cataract/annotations_train.json \
  --val_json /mnt/f/.../output_cataract/annotations_val.json \
  --output_dir /mnt/f/.../output_cataract/detr_tooltip_train \
  --model_checkpoint facebook/detr-resnet-50 \
  --epochs 50 --batch_size 8 --use_amp --coco_eval_every 1
```
- Finalna ewaluacja (test):
```
python Cataract_DETR_DatasetBuilder/eval_coco_offline.py \
  --images_dir /mnt/f/.../output_cataract/images \
  --test_json /mnt/f/.../output_cataract/annotations_test.json \
  --model_checkpoint facebook/detr-resnet-50 \
  --weights_path /mnt/f/.../output_cataract/detr_tooltip_train/best_model_state.pth \
  --batch_size 8 --conf_threshold 0.0
```

## How to Review
1. Przejrzyj SESSION_SUMMARY.md (cele, założenia, pełna instrukcja treningu)
2. Zweryfikuj snapshoty w `configs/` z realnymi ścieżkami
3. Uruchom w/w komendy na swoim środowisku

## Related Sessions
- Benchmark: `.sessions/benchmark/sesja_2025-10-29_14-29-09_benchmark`
