# Same-distribution v1 (video-level split)

- Wariant: **v1** (FULL pipeline (baseline, ep270))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR\ckpt_20k_finetune\checkpoint_epoch_270.pth` (epoka 270)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | 10652 | 89.11% | 88.04% | 94.7% | 100.0% | 89.9% | 9701 | 0 | 1085 |
| test | 9140 | 89.08% | 87.85% | 94.7% | 99.9% | 90.0% | 8304 | 8 | 927 |
