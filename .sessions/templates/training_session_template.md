# Session Summary: {SESSION_NAME}

## Metadata
- **Type:** training
- **Date:** {DATE}
- **Time:** {TIME} UTC
- **Duration:** {DURATION}
- **Status:** {STATUS}

## Objective
{OBJECTIVE}

## Context
{CONTEXT}

## Model Configuration

### Architecture
- **Model:** {MODEL_NAME}
- **Variant:** {MODEL_VARIANT}
- **Backbone:** {BACKBONE}
- **Pretrained:** {PRETRAINED_STATUS}
- **Classes:** {NUM_CLASSES}

### Training Setup
- **Epochs:** {EPOCHS}
- **Batch Size:** {BATCH_SIZE}
- **Learning Rate:** {LEARNING_RATE}
- **Optimizer:** {OPTIMIZER}
- **Scheduler:** {SCHEDULER}
- **Weight Decay:** {WEIGHT_DECAY}

### Augmentation
- **Augmentation Strategy:** {AUGMENTATION_STRATEGY}
- **Transforms:** {TRANSFORMS_LIST}

## Dataset

### Training Set
- **Name:** {TRAIN_DATASET_NAME}
- **Images:** {TRAIN_NUM_IMAGES}
- **Annotations:** {TRAIN_ANNOTATIONS_PATH}
- **Categories:** {CATEGORIES_LIST}

### Validation Set
- **Name:** {VAL_DATASET_NAME}
- **Images:** {VAL_NUM_IMAGES}
- **Split Ratio:** {SPLIT_RATIO}

### Data Statistics
| Metric | Value |
|--------|-------|
| **Total Images** | {TOTAL_IMAGES} |
| **Total Annotations** | {TOTAL_ANNOTATIONS} |
| **Avg Objects per Image** | {AVG_OBJECTS} |
| **Min Objects** | {MIN_OBJECTS} |
| **Max Objects** | {MAX_OBJECTS} |

## Hardware Configuration

### Compute
- **Device:** {DEVICE}
- **GPUs:** {NUM_GPUS} x {GPU_MODEL}
- **VRAM:** {VRAM_SIZE} MB per GPU
- **CPU:** {CPU_MODEL}
- **RAM:** {RAM_SIZE} GB

### Storage
- **Checkpoint Dir:** {CHECKPOINT_DIR}
- **Log Dir:** {LOG_DIR}
- **Dataset Location:** {DATASET_LOCATION}

## Training Progress

### Epoch-by-Epoch Summary

| Epoch | Train Loss | Val Loss | Val mAP@0.5 | Val mAP@0.5:0.95 | Time |
|-------|------------|----------|-------------|------------------|------|
| {EPOCH_1} | {TL_1} | {VL_1} | {MAP05_1}% | {MAP_1}% | {TIME_1} |
| {EPOCH_2} | {TL_2} | {VL_2} | {MAP05_2}% | {MAP_2}% | {TIME_2} |
| {EPOCH_3} | {TL_3} | {VL_3} | {MAP05_3}% | {MAP_3}% | {TIME_3} |
| ... | ... | ... | ... | ... | ... |
| {EPOCH_FINAL} | {TL_F} | {VL_F} | {MAP05_F}% | {MAP_F}% | {TIME_F} |

### Best Checkpoint
- **Epoch:** {BEST_EPOCH}
- **Val mAP@0.5:** {BEST_MAP05}%
- **Val mAP@0.5:0.95:** {BEST_MAP}%
- **File:** {BEST_CHECKPOINT_PATH}

### Training Curves
- **Loss Plot:** `visualizations/loss_curve.png`
- **mAP Plot:** `visualizations/map_curve.png`
- **Learning Rate Plot:** `visualizations/lr_schedule.png`

## Training Metrics

### Loss Progression
- **Initial Train Loss:** {INITIAL_TRAIN_LOSS}
- **Final Train Loss:** {FINAL_TRAIN_LOSS}
- **Improvement:** {TRAIN_LOSS_IMPROVEMENT}
- **Initial Val Loss:** {INITIAL_VAL_LOSS}
- **Final Val Loss:** {FINAL_VAL_LOSS}
- **Improvement:** {VAL_LOSS_IMPROVEMENT}

### Validation Performance
| Metric | Initial | Final | Improvement |
|--------|---------|-------|-------------|
| **mAP@0.5** | {INITIAL_MAP05}% | {FINAL_MAP05}% | {IMPROVE_MAP05}% |
| **mAP@0.5:0.95** | {INITIAL_MAP}% | {FINAL_MAP}% | {IMPROVE_MAP}% |
| **mAP@0.75** | {INITIAL_MAP75}% | {FINAL_MAP75}% | {IMPROVE_MAP75}% |
| **AR@100** | {INITIAL_AR}% | {FINAL_AR}% | {IMPROVE_AR}% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | AP@0.5 |
|-------|-----------|--------|----------|--------|
| {CLASS_1} | {P_1}% | {R_1}% | {F1_1}% | {AP_1}% |
| {CLASS_2} | {P_2}% | {R_2}% | {F1_2}% | {AP_2}% |

## Training Details

### Training Time
- **Total Time:** {TOTAL_TRAINING_TIME}
- **Time per Epoch:** {TIME_PER_EPOCH}
- **Time per Batch:** {TIME_PER_BATCH}
- **Throughput:** {THROUGHPUT} images/sec

### Resource Usage
- **Peak GPU Memory:** {PEAK_GPU_MEMORY} MB
- **Avg GPU Usage:** {AVG_GPU_USAGE}%
- **Peak CPU Memory:** {PEAK_CPU_MEMORY} GB
- **Avg CPU Usage:** {AVG_CPU_USAGE}%

### Checkpoints Saved
1. `{CHECKPOINT_1}` - Epoch {EPOCH_1}
2. `{CHECKPOINT_2}` - Epoch {EPOCH_2}
3. `{CHECKPOINT_BEST}` - Epoch {BEST_EPOCH} (Best)
4. `{CHECKPOINT_FINAL}` - Epoch {FINAL_EPOCH} (Final)

## Issues Encountered

### Issue 1: {ISSUE_TITLE}
**Epoch:** {ISSUE_EPOCH}
**Description:** {ISSUE_DESCRIPTION}
**Resolution:** {ISSUE_RESOLUTION}
**Impact:** {ISSUE_IMPACT}

## Hyperparameter Tuning

### Experiments Conducted
1. **Learning Rate:** Tested [{LR_VALUES}], Best: {BEST_LR}
2. **Batch Size:** Tested [{BS_VALUES}], Best: {BEST_BS}
3. **Weight Decay:** Tested [{WD_VALUES}], Best: {BEST_WD}

### Ablation Study
| Configuration | mAP@0.5 | mAP@0.5:0.95 | Notes |
|---------------|---------|--------------|-------|
| Baseline | {ABL_BASE_MAP05}% | {ABL_BASE_MAP}% | {ABL_BASE_NOTES} |
| + Augmentation | {ABL_AUG_MAP05}% | {ABL_AUG_MAP}% | {ABL_AUG_NOTES} |
| + Scheduler | {ABL_SCH_MAP05}% | {ABL_SCH_MAP}% | {ABL_SCH_NOTES} |
| Final Config | {ABL_FINAL_MAP05}% | {ABL_FINAL_MAP}% | {ABL_FINAL_NOTES} |

## Analysis

### Overfitting Assessment
**Train-Val Gap:** {TRAIN_VAL_GAP}
**Conclusion:** {OVERFITTING_CONCLUSION}
**Mitigation:** {OVERFITTING_MITIGATION}

### Convergence Status
**Converged:** {CONVERGED_STATUS}
**Evidence:** {CONVERGENCE_EVIDENCE}
**Early Stopping:** {EARLY_STOPPING_STATUS}

### Model Behavior
- {BEHAVIOR_OBSERVATION_1}
- {BEHAVIOR_OBSERVATION_2}
- {BEHAVIOR_OBSERVATION_3}

## Conclusions

### Training Success
{TRAINING_SUCCESS_SUMMARY}

### Model Performance
{MODEL_PERFORMANCE_SUMMARY}

### Comparison to Baseline
- **Baseline mAP@0.5:** {BASELINE_MAP05}%
- **Achieved mAP@0.5:** {ACHIEVED_MAP05}%
- **Improvement:** {IMPROVEMENT_VS_BASELINE}%

## Recommendations

### For Deployment
- {DEPLOYMENT_REC_1}
- {DEPLOYMENT_REC_2}

### For Further Training
- {FURTHER_TRAINING_REC_1}
- {FURTHER_TRAINING_REC_2}

### For Next Experiments
- {NEXT_EXP_REC_1}
- {NEXT_EXP_REC_2}

## Next Steps
- [ ] {NEXT_STEP_1}
- [ ] {NEXT_STEP_2}
- [ ] {NEXT_STEP_3}

## Files Generated
- `checkpoints/checkpoint_epoch_{BEST_EPOCH}.pth` - Best model
- `checkpoints/checkpoint_final.pth` - Final model
- `logs/training.log` - Complete training log
- `configs/training_config.yaml` - Configuration used
- `results/validation_metrics.json` - Val metrics per epoch
- `visualizations/loss_curve.png` - Training curves

## Code & Configuration

### Training Command
```bash
{TRAINING_COMMAND}
```

### Configuration File
```yaml
{TRAINING_CONFIG_YAML}
```

### Key Training Loop
```python
{TRAINING_LOOP_EXCERPT}
```

## Related Work
- **Previous training:** {PREVIOUS_TRAINING}
- **Baseline model:** {BASELINE_MODEL}
- **Dataset preparation:** {DATASET_PREP_SESSION}
- **Benchmark results:** {BENCHMARK_SESSION}

## References
1. {REFERENCE_1}
2. {REFERENCE_2}
3. {REFERENCE_3}

---

**Session Created:** {CREATED_TIMESTAMP}
**Last Updated:** {UPDATED_TIMESTAMP}
