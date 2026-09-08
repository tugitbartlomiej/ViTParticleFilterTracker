# Smoke test v1 (po 50 obrazow z obu datasetow)

- Wariant: **v1** (FULL pipeline (baseline, ep270))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR\ckpt_20k_finetune\checkpoint_epoch_270.pth` (epoka 270)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| external_test_smoke | 50 | 66.93% | 21.22% | 52.7% | 39.8% | 78.0% | 39 | 59 | 11 |
| samedist_test_smoke | 50 | 100.00% | 99.77% | 87.0% | 76.9% | 100.0% | 50 | 15 | 0 |
