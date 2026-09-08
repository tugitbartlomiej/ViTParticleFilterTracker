# Smoke test v7 (po 50 obrazow z obu datasetow)

- Wariant: **v7** (EL2N_ONLY (random+EL2N rerank))
- Checkpoint: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\Ablations\Fresh_from_Eden_2026-06-11_v6_v7_completed_runs\v7\best_model.pth` (epoka ?)
- num_queries=100, num_labels=1, timm=True
- Progi: conf=0.3, IoU=0.5

| Zbior | N img | mAP@0.5 | mAP@[0.5:0.95] | F1 | Prec. | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| external_test_smoke | 50 | 1.54% | 0.28% | 3.9% | 2.9% | 6.0% | 3 | 102 | 47 |
| samedist_test_smoke | 50 | 91.41% | 89.67% | 66.7% | 50.0% | 100.0% | 50 | 50 | 0 |
