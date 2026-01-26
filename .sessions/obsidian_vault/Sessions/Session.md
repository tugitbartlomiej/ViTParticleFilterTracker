---
date: 2026-01-20
time: 14:05
type: Training
topics: [GPU, SSH Eden, Training, YOLO]
aliases: ["Session"]
---

# Session

> [!info] Session Info
> **Date:** 2026-01-20 14:05
> **Type:** Training
> **ID:** `Session_2026-01-20_140500`

## Objective

Ponowne uruchomienie fine-tuningu YOLO od epoch 170 z poprawionym Learning Rate po nieudanym poprzednim treningu.

## Topics

[[GPU]] [[SSH Eden]] [[Training]] [[YOLO]]

## Actions Taken

1. Wczytano poprzednia sesje SSH (`sesja_2026-01-19_yolo_finetune_ep170_500`)
2. Przeanalizowano logi - zidentyfikowano problem z LR
3. Utworzono nowy skrypt SLURM z poprawkami: - lr0: 0.001 -> **0.0001** (10x mniejszy) - patience: 50 -> **100** (wiecej cierpliwosci) - GPU: 4 -> **7** (pelne wykorzystanie hopper) - batch: 32 -> **35**
4. Wgrano skrypt na Eden przez SCP
5. Uruchomiono job przez sbatch
6. Naprawiono blad batch size (32 nie podzielne przez 7 GPU)
7. Potwierdzono ze trening dziala - loss spada prawidlowo

## Key Findings

- lr0=0.0001 dziala - model sie uczy, loss spada
- DDP na 7x H100 dziala poprawnie
- Batch 35 (5 per GPU) jest optymalny dla 7 GPU

## Files Modified

- `SESSION_SUMMARY.md`

## Next Steps

- [ ] Monitorowac trening - czy metryki sie poprawiaja przez wiele epok
- [ ] Sprawdzic mAP50-95 po kilku epokach
- [ ] Porownac z baseline (poprzedni best @ epoch 1: mAP50-95=0.961)
- [ ] Zapisac sesje SSH po zakonczeniu treningu
- [ ] Ewentualnie - jesli lr0=0.0001 za maly, sprobowac lr0=0.0005

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
- [[YOLO Resume]] (2026-01-19)
- [[Visualization]] (2026-01-26)
- [[Visualization]] (2026-01-26)

---

> [!tip] Navigation
> - [[Sessions Index|Back to Index]]
> - [[Training|All Training Sessions]]
