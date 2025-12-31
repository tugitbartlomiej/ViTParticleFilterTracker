# Session: Session_2025-12-31_042306

**Created:** 2025-12-31 04:23:06
**Type:** Benchmark
**Status:** Completed

## Quick Summary
Przeprowadzono kompleksowy benchmark DETR 20k finetune (epochs 170-200) vs YOLO na zewnętrznym datasecie Roboflow. Odkryto problem data leakage w oryginalnym TestDatasetGenerator - adnotacje były generowane przez testowany model DETR ep170. Po przejściu na zewnętrzny dataset Roboflow, wyniki pokazały dramatyczną poprawę generalizacji: DETR 20k ep180 osiąga 77% mAP vs 2% dla oryginalnego modelu.

## Key Results
- **DETR_20k ep180**: Najlepszy model - 74.57% mAP (valid), 50.6% F1 (aggregated)
- **DETR_20k ep195**: Najwyższy mAP 76.83% (valid), ale gorszy F1 38.9%
- **Oryginalny DETR ep170**: Tylko 2% mAP na zewnętrznym datasecie
- **YOLO**: 0% mAP - kompletny brak generalizacji
- **Kluczowe odkrycie**: Finetuning na 20k obrazach drastycznie poprawia cross-dataset generalization

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
