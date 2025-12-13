# DETR training session analysis

- Session dir: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR`
- Job ID: `1454332`
- Start time: `Fri Dec 12 11:55:11 PM CET 2025`
- Node: `pascal`
- Image count: `20000`
- GPUs: `4x P100`
- Effective batch: `16`
- Mixed precision: `DISABLED`
- Best validation loss: `0.161400` at epoch `185`
- Training ended mid-epoch: `E186` at `993/1125` (~88%)

## Session contents

- Log files: `1`
- Checkpoint dirs: `1`
- Checkpoints: `5`
- TensorBoard dirs: `1`
- TensorBoard event files: `13`

## Key training events (from log)

- Resume: found `final_checkpoint.pth`, loaded `ckpt_20k_finetune/final_checkpoint.pth`, epoch `172`
- LR reset: old `0.0001`, new `5e-05` (backbone `5e-06`), cosine remaining epochs `128`

## Configuration (from Namespace)

- `model_checkpoint`: `facebook/detr-resnet-50`
- `num_queries`: `100`
- `train_val_split`: `0.9`
- `augment`: `True`
- `batch_size`: `4`
- `epochs`: `300`
- `lr`: `5e-05`
- `lr_backbone`: `5e-06`
- `weight_decay`: `0.0001`
- `max_grad_norm`: `0.1`
- `use_amp`: `False`
- `lr_scheduler`: `cosine`
- `warmup_epochs`: `5`
- `lr_min`: `1e-07`
- `save_interval`: `5`
- `patience`: `30`
- `resume_training`: `True`
- `checkpoint_dir`: `./ckpt_20k_finetune`
- `best_model_dir`: `./best_20k_finetune`
- `output_dir`: `./train_out_20k_finetune`

## Findings (derived)

- Full epochs in log: `13` (E173..E185)
- Avg epoch time (log): `3387.4s` (~56:27)
- Worst validation loss (log): `0.341000`
- Approx throughput: `5.31` train images/s (all GPUs)
- Suggested resume checkpoint: `checkpoint_epoch_185.pth` (epoch `185`)

## TensorBoard scalar tags found

- `GPU/memory_allocated_GB` (16 points)
- `GPU/memory_reserved_GB` (16 points)
- `Gradients/norm_avg` (13 points)
- `LR/backbone` (13 points)
- `LR/main` (13 points)
- `Loss/train_batch` (465 points)
- `Loss/train_epoch` (13 points)
- `Loss/validation_epoch` (13 points)
- `Loss_Components/bbox_batch` (364 points)
- `Loss_Components/bbox_epoch` (13 points)
- `Loss_Components/ce_batch` (364 points)
- `Loss_Components/ce_epoch` (13 points)
- `Loss_Components/giou_batch` (364 points)
- `Loss_Components/giou_epoch` (13 points)
- `Time/epoch_seconds` (13 points)

## Plots

![](plots/loss_curves.png)

![](plots/lr_schedule.png)

![](plots/grad_norm.png)

![](plots/loss_components.png)

![](plots/gpu_memory.png)

![](plots/epoch_time.png)

![](plots/checkpoint_weight_norm.png)

## Epoch summary table

| Epoch | Train loss (log) | Val loss (log) | Train loss (TB) | Val loss (TB) | Grad norm (log) | LR main (TB) | LR bb (TB) | Epoch time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 171 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 172 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 173 | 0.3431 | 0.3410 | 0.3431 | 0.3410 | 112.55 | 5.00e-05 | 5.00e-06 | 57:00 |
| 174 | 0.3160 | 0.2654 | 0.3160 | 0.2654 | 59.51 | 5.00e-05 | 5.00e-06 | 56:26 |
| 175 | 0.2655 | 0.2498 | 0.2655 | 0.2498 | 48.57 | 4.99e-05 | 4.99e-06 | 56:23 |
| 176 | 0.2518 | 0.2229 | 0.2518 | 0.2229 | 65.93 | 4.99e-05 | 4.99e-06 | 56:22 |
| 177 | 0.2335 | 0.2343 | 0.2335 | 0.2343 | 58.78 | 4.98e-05 | 4.98e-06 | 56:22 |
| 178 | 0.2237 | 0.2149 | 0.2237 | 0.2149 | 56.38 | 4.97e-05 | 4.97e-06 | 56:24 |
| 179 | 0.2030 | 0.1908 | 0.2030 | 0.1908 | 56.29 | 4.96e-05 | 4.96e-06 | 56:24 |
| 180 | 0.2060 | 0.1898 | 0.2060 | 0.1898 | 51.55 | 4.95e-05 | 4.95e-06 | 56:25 |
| 181 | 0.1987 | 0.1941 | 0.1987 | 0.1941 | 48.51 | 4.94e-05 | 4.94e-06 | 56:23 |
| 182 | 0.1926 | 0.1891 | 0.1926 | 0.1891 | 46.17 | 4.93e-05 | 4.93e-06 | 56:22 |
| 183 | 0.1856 | 0.2320 | 0.1856 | 0.2320 | 52.73 | 4.91e-05 | 4.91e-06 | 56:26 |
| 184 | 0.1756 | 0.1718 | 0.1756 | 0.1718 | 42.64 | 4.89e-05 | 4.89e-06 | 56:23 |
| 185 | 0.1787 | 0.1614 | 0.1787 | 0.1614 | 45.67 | 4.87e-05 | 4.88e-06 | 56:37 |
| 186 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Checkpoint analysis

| File | Epoch | Loss | LR main | LR bb | Scheduler last_epoch | Mean weight norm | NaN/Inf tensors |
|---|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint_epoch_170.pth` | 170 | 0.162833 | 1.00e-04 | 1.00e-05 | N/A | 8.7076 | 0/0 |
| `final_checkpoint.pth` | 172 | N/A | 1.00e-04 | 1.00e-05 | 0 | 8.7204 | 0/0 |
| `checkpoint_epoch_175.pth` | 175 | 0.249823 | 4.99e-05 | 4.99e-06 | 3 | 8.7223 | 0/0 |
| `checkpoint_epoch_180.pth` | 180 | 0.189761 | 4.95e-05 | 4.95e-06 | 8 | 8.7251 | 0/0 |
| `checkpoint_epoch_185.pth` | 185 | 0.161407 | 4.87e-05 | 4.88e-06 | 13 | 8.7274 | 0/0 |

## Final model artifacts (HuggingFace)

- Final model dir: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\Checkpoints\DETR\DETR_Training_Sessions\2025-12-13_20kDataset_small_LR\train_out_20k_finetune\final_model`
- `transformers_version`: `4.40.2`
- `num_queries`: `100`
- `id2label`: `{'1': 'tooltip'}`
- Preprocessor `size`: `{'longest_edge': 1333, 'shortest_edge': 800}`
