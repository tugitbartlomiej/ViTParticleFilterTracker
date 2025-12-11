# Session Summary: {SESSION_NAME}

## Metadata
- **Type:** benchmark
- **Date:** {DATE}
- **Time:** {TIME} UTC
- **Duration:** {DURATION}
- **Status:** {STATUS}

## Objective
{OBJECTIVE}

## Context
{CONTEXT}

## Models Tested

### Model 1: {MODEL_1_NAME}
- **Architecture:** {MODEL_1_ARCH}
- **Checkpoint:** {MODEL_1_CHECKPOINT}
- **Epoch:** {MODEL_1_EPOCH}
- **Size:** {MODEL_1_SIZE}

### Model 2: {MODEL_2_NAME}
- **Architecture:** {MODEL_2_ARCH}
- **Checkpoint:** {MODEL_2_CHECKPOINT}
- **Epoch:** {MODEL_2_EPOCH}
- **Size:** {MODEL_2_SIZE}

## Test Configuration

### Dataset
- **Name:** {DATASET_NAME}
- **Images:** {NUM_IMAGES}
- **Annotations:** {ANNOTATIONS_PATH}
- **Categories:** {CATEGORIES}

### Inference Parameters
- **Confidence Threshold:** {CONF_THRESHOLD}
- **IoU Threshold:** {IOU_THRESHOLD}
- **Device:** {DEVICE}
- **Batch Size:** {BATCH_SIZE}

### Hardware
- **GPU:** {GPU_MODEL}
- **VRAM:** {VRAM_SIZE}
- **CPU:** {CPU_MODEL}
- **RAM:** {RAM_SIZE}

## Benchmark Results

### Accuracy Metrics

| Metric | {MODEL_1_NAME} | {MODEL_2_NAME} | Delta | Winner |
|--------|----------------|----------------|-------|--------|
| **mAP@0.5** | {M1_MAP_05}% | {M2_MAP_05}% | {DELTA_MAP_05}% | {WINNER_MAP_05} |
| **mAP@0.5:0.95** | {M1_MAP_FULL}% | {M2_MAP_FULL}% | {DELTA_MAP_FULL}% | {WINNER_MAP_FULL} |
| **mAP@0.75** | {M1_MAP_75}% | {M2_MAP_75}% | {DELTA_MAP_75}% | {WINNER_MAP_75} |
| **AR@100** | {M1_AR}% | {M2_AR}% | {DELTA_AR}% | {WINNER_AR} |

### Performance Metrics

| Metric | {MODEL_1_NAME} | {MODEL_2_NAME} | Ratio | Winner |
|--------|----------------|----------------|-------|--------|
| **FPS** | {M1_FPS} | {M2_FPS} | {FPS_RATIO}x | {WINNER_FPS} |
| **VRAM Usage** | {M1_VRAM} MB | {M2_VRAM} MB | {VRAM_RATIO}x | {WINNER_VRAM} |
| **Inference Time** | {M1_TIME} ms | {M2_TIME} ms | {TIME_RATIO}x | {WINNER_TIME} |

### Size-Based Performance

| Object Size | {MODEL_1_NAME} | {MODEL_2_NAME} | Delta | Winner |
|-------------|----------------|----------------|-------|--------|
| **Small** | {M1_SMALL}% | {M2_SMALL}% | {DELTA_SMALL}% | {WINNER_SMALL} |
| **Medium** | {M1_MEDIUM}% | {M2_MEDIUM}% | {DELTA_MEDIUM}% | {WINNER_MEDIUM} |
| **Large** | {M1_LARGE}% | {M2_LARGE}% | {DELTA_LARGE}% | {WINNER_LARGE} |

## Optimization Experiments

### Configuration 1: {CONFIG_1_NAME}
- **Description:** {CONFIG_1_DESC}
- **Parameters:** {CONFIG_1_PARAMS}
- **Result:** mAP@0.5 = {CONFIG_1_RESULT}%
- **Improvement:** {CONFIG_1_IMPROVEMENT}%

### Configuration 2: {CONFIG_2_NAME}
- **Description:** {CONFIG_2_DESC}
- **Parameters:** {CONFIG_2_PARAMS}
- **Result:** mAP@0.5 = {CONFIG_2_RESULT}%
- **Improvement:** {CONFIG_2_IMPROVEMENT}%

## Key Findings

### Finding 1: {FINDING_1_TITLE}
{FINDING_1_DESCRIPTION}

### Finding 2: {FINDING_2_TITLE}
{FINDING_2_DESCRIPTION}

### Finding 3: {FINDING_3_TITLE}
{FINDING_3_DESCRIPTION}

## Issues Encountered

### Issue 1: {ISSUE_TITLE}
**Description:** {ISSUE_DESCRIPTION}
**Resolution:** {ISSUE_RESOLUTION}
**Impact:** {ISSUE_IMPACT}

## Conclusions

### Overall Winner: {WINNER_MODEL}

**Rationale:**
{WINNER_RATIONALE}

### Strengths of {MODEL_1_NAME}
- {M1_STRENGTH_1}
- {M1_STRENGTH_2}
- {M1_STRENGTH_3}

### Weaknesses of {MODEL_1_NAME}
- {M1_WEAKNESS_1}
- {M1_WEAKNESS_2}

### Strengths of {MODEL_2_NAME}
- {M2_STRENGTH_1}
- {M2_STRENGTH_2}
- {M2_STRENGTH_3}

### Weaknesses of {MODEL_2_NAME}
- {M2_WEAKNESS_1}
- {M2_WEAKNESS_2}

## Recommendations

### For Production Deployment
{PRODUCTION_RECOMMENDATION}

### For Research Exploration
{RESEARCH_RECOMMENDATION}

### For Optimization
- {OPTIMIZATION_REC_1}
- {OPTIMIZATION_REC_2}
- {OPTIMIZATION_REC_3}

## Next Steps
- [ ] {NEXT_STEP_1}
- [ ] {NEXT_STEP_2}
- [ ] {NEXT_STEP_3}

## Files Generated
- `results/benchmark_summary.json` - Complete metrics
- `results/model1_predictions.json` - COCO format predictions
- `results/model2_predictions.json` - COCO format predictions
- `configs/benchmark_config.yaml` - Test configuration
- `reports/comparison_report.md` - Detailed analysis

## Commands Used

### Run benchmark
```bash
{BENCHMARK_COMMAND}
```

### Evaluate metrics
```bash
{EVALUATION_COMMAND}
```

### Generate report
```bash
{REPORT_COMMAND}
```

## Related Work
- **Previous benchmark:** {PREVIOUS_BENCHMARK}
- **Related analysis:** {RELATED_ANALYSIS}
- **Model checkpoints:** {CHECKPOINT_LOCATION}
- **Dataset:** {DATASET_LOCATION}

## References
- {REFERENCE_1}
- {REFERENCE_2}
- {REFERENCE_3}

---

**Session Created:** {CREATED_TIMESTAMP}
**Last Updated:** {UPDATED_TIMESTAMP}
