# YOLO Ablation Progress

## Status: not started

## Variants planned
- YE-A: YOLO-EL2N -> 20k -> YOLOv8 fine-tune
- YE-B: YOLO-EL2N -> 20k -> DETR fine-tune (cross-arch comparison)
- YE-C: DETR-EL2N -> 20k -> YOLOv8 fine-tune (cross-arch comparison)

## Checklist
- [ ] generate_yolo_el2n_scores.py — extract per-frame loss z YOLOv8 ep170 (91,336 frames)
- [ ] generate_ablation_datasets.py — top-K po YOLO-EL2N -> 20k subsety dla YE-A, YE-B, YE-C
- [ ] yolo_train_ablation.py — fine-tune YOLOv8 (ye_a, ye_c)
- [ ] detr_train_ablation.py reuse — fine-tune DETR (ye_b)
- [ ] evaluate_ablation.py — same-dist + external benchmark eval
- [ ] Wyniki -> Tab 7 / Discussion 7.3

## Logs
(pusty placeholder — wpisuj kolejne wpisy z datami i checkpointami w trakcie pracy)
