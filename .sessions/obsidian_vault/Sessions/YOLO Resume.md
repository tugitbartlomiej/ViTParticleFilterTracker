---
date: 2026-01-19
time: 04:20
type: Training
topics: [GPU, SSH Eden, Training, YOLO]
aliases: ["YOLO Resume"]
---

# YOLO Resume

> [!info] Session Info
> **Date:** 2026-01-19 04:20
> **Type:** Training
> **ID:** `Session_2026-01-19_042000_YOLO_LR_Fix_Finetune`

## Objective

Naprawić problem z Learning Rate przy fine-tuningu YOLO i uruchomić trening od epoch 170 do 500.

## Topics

[[GPU]] [[SSH Eden]] [[Training]] [[YOLO]]

## Actions Taken

1. Analiza problemu LR - Sprawdzono logi poprzednich sesji i checkpointów - Odkryto że epoch 70 miał LR=0.006535, epoch 100 miał LR=0.00505 - Końcowy LR na epoch 200 ≈ 0.0001

## Key Findings

- `optimizer=auto` ignoruje `lr0` i `momentum` - MUSI być explicit `optimizer='SGD'`
- Oryginalny trening: 0.01→0.0001, Fine-tune: 0.001→0.0001
- epoch170.pt znajduje się w `~/Yolo/20k_finetune/` (root, nie w weights/)
- `sfree` pokazuje lepsze info o dostępnych zasobach niż `sinfo`

## Files Modified

- `Eden/Scripts/Train20kDataset_YOLO_31.12.25/yolo_train_20k_finetune_v2.py`
- `SESSION_SUMMARY.md`
- `README.md`
- `SESSION_SUMMARY.md`

## Next Steps

- [ ] Monitorować trening (loss, mAP)
- [ ] Sprawdzić checkpoint po epoch 10
- [ ] Po zakończeniu: walidacja na test set
- [ ] Porównanie z modelem z epoch 200

## Related Sessions

- [[Session]] (2025-10-20)
- [[Session]] (2025-10-20)
- [[Session]] (2025-10-21)
- [[Session]] (2025-10-22)
- [[Session]] (2025-10-31)
- [[YOLO Fix]] (Unknown)
- [[DETR EL2N]] (2025-12-11)
- [[Dataset Selection]] (2025-12-12)
- [[Session]] (2025-12-12)
- [[Session]] (2025-12-12)
- [[Dataset Selection]] (2025-12-12)
- [[Visualization]] (2025-12-12)
- [[Session]] (2025-12-12)
- [[Query 81]] (2025-12-12)
- [[Visualization]] (2025-12-13)
- [[Visualization]] (2025-12-13)
- [[Query 81]] (2025-12-13)
- [[Session]] (2025-12-13)
- [[Dataset Selection]] (2025-12-31)
- [[Session]] (2026-01-11)
- [[YOLO Fix]] (2026-01-11)
- [[IEEE Article]] (2026-01-12)
- [[IEEE Article]] (2026-01-12)
- [[YOLO Resume]] (2026-01-12)
- [[YOLO Resume]] (2026-01-12)
- [[Dataset Selection]] (2026-01-16)
- [[Dataset Selection]] (2026-01-18)
- [[YOLO Resume]] (2026-01-18)
- [[Session]] (2026-01-20)
- [[Visualization]] (2026-01-26)
- [[Visualization]] (2026-01-26)

---

> [!tip] Navigation
> - [[Sessions Index|Back to Index]]
> - [[Training|All Training Sessions]]
