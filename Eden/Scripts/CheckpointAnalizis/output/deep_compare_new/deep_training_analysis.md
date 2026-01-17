# Deep DETR vs YOLO training analysis (two-phase)

## Phase protocol (critical for interpretation)
- Phase 1: epoch <= 170, dataset ~91,260 images (augmented pool)
- Phase 2: epoch > 170, dataset 20,000 images (fine-tune subset)
- Focus: epoch <= 200 for direct DETR vs YOLO comparability (YOLO ended at 200).

## Plots
- YOLO metrics (mAP/PR + losses): `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\deep_compare_new\deep_yolo_metrics.png`
- DETR fine-tune (loss/grad/time): `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\deep_compare_new\deep_detr_finetune_log.png`
- Weight drift (dW and distance to epoch170): `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\deep_compare_new\deep_weight_drift.png`
- Group drift (where weights change): `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Scripts\CheckpointAnalizis\output\deep_compare_new\deep_group_drift.png`

## Plot reading guide
- `deep_yolo_metrics.png`: YOLO mAP/precision/recall + train/val losses; watch the phase boundary at epoch 170.
- `deep_detr_finetune_log.png`: DETR fine-tune train/val loss + grad norm + epoch time (parsed from the training log).
- `deep_weight_drift.png`: weight-space dynamics; left=relative checkpoint update size, right=drift from epoch170 baseline.
- `deep_group_drift.png`: where the weights change most during fine-tune (170->200); DETR groups vs YOLO layer indices.

## Weight-space drift summary (epoch170 -> end)
| Model | End epoch | ||w_end - w170|| / ||w170|| | cos(w_end, w170) |
|---|---:|---:|---:|
| DETR | 200 | 0.0337 | 0.9994 |
| YOLO | 200 | 0.0567 | 0.9987 |

Interpretation:
- `||w_end - w170|| / ||w170||` measures how much parameters moved during fine-tune (phase2).
- `cos(w_end, w170)` close to 1.0 means the final weights are directionally very similar to epoch170.

## Checkpoint-to-checkpoint update size (median dW/||w||)
| Model | Phase 1 (<=170) | Phase 2 (>170) |
|---|---:|---:|
| DETR | 0.0704 | 0.0140 |
| YOLO | 0.0861 | 0.0178 |

## Key takeaways (training dynamics)
- Phase switch shrinks median dW/||w|| to ~0.1994x of phase1 (~5.0156x smaller) for DETR, and to ~0.2064x of phase1 (~4.8440x smaller) for YOLO.
- Fine-tune drift (170->200): DETR `0.0337` vs YOLO `0.0567` in ||w_end-w170||/||w170|| (both cosine ~1.0).
- DETR fine-tune concentrates updates in: `cls_head` (top drift `0.0973`).
- YOLO fine-tune shows strongest drift in: `layer_1` (top drift `0.1026`).

## Where the model changes during fine-tune (170 -> 200)

**DETR top groups (relative drift):**
- cls_head: 0.0973
- decoder: 0.0383
- encoder: 0.0373
- backbone: 0.0231
- other: 0.0215
- bbox_head: 0.0178

**YOLO top layers (relative drift):**
- layer_1: 0.1026
- layer_9: 0.0769
- layer_2: 0.0599
- layer_4: 0.0454
- layer_6: 0.0422
- layer_3: 0.0320
- layer_5: 0.0297
- layer_22: 0.0293
- layer_15: 0.0267
- layer_12: 0.0195
- layer_8: 0.0183
- layer_7: 0.0182
- layer_16: 0.0084
- layer_0: 0.0053
- layer_18: 0.0023

## DETR fine-tune run details (from log)
- Image count: `20000`
- GPUs: `4x P100`
- Batch per GPU: `4` (effective `16`)
- Resume epoch detected: `172`
- LR reset: old `0.0001` -> new `5e-05` (backbone `5e-06`)
- Cosine scheduler re-init: `128` remaining epochs

## DETR fine-tune loss dynamics (from log)
- Log coverage: epoch `173`..`185` (n=13)
- Start: epoch `173` train `0.3431` / val `0.3410`
- Best val loss: `0.1614` @ epoch `185` (train `0.1787`)
- Last parsed: epoch `185` train `0.1787` / val `0.1614`
- Grad norm: median `52.7339`, min `42.6379`, max `112.5453`
- Epoch time: mean `56.4569` minutes (n=13)

## YOLO phase1 headline (epoch <= 170)
- Best mAP50-95: `0.9408` @ epoch `170`

## YOLO fine-tune headline (from results.csv)
- Epoch range: `172`..`200`
- Peak mAP50-95: `0.9633` @ epoch `189` (gain vs ep170: `0.0225`)
- Peak mAP50: `0.9950` @ epoch `193` (gain vs ep170: `0.0000`)
- Peak precision: `0.9983`
- Peak recall: `0.9975`
