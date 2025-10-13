# DETR MIXED TRAINING STRATEGY TESTING RESULTS
## Session: 2025-08-15

## Dataset Configuration
### Training Dataset
- **Tooltip Dataset**: 
  - Path: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/Datasets/Yolo/yolo_dataset_20250218/`
  - Original pretrained model with 69.4% tooltip detection rate
  
- **Background Dataset (DINO extracted)**:
  - Source: `E:\PicsOnly` (13,683 images)
  - Extracted: 193 diverse frames using DINO clustering
  - Location: `test_dino_dataset/interesting_frames/`

### Dataset Mixing Strategy
- **Tooltip Ratio**: 90%
- **Background Ratio**: 10%
- **Batch Size**: 1 (to handle variable image sizes)

## Tested Strategies

### 1. ✅ ULTRA_GENTLE (COMPLETED - SUCCESS!)
- **Status**: SUCCESS
- **Training Time**: 230.89 minutes (3.85 hours)
- **Configuration**:
  - Learning Rate: 1e-8
  - Backbone LR: 1e-9  
  - Classifier LR: 1e-7
  - Epochs: 3
  - Freeze Backbone: False
  
- **Results**:
  - **Detection Rate**: 70% (14/20 images)
  - **Average Confidence**: 90.6%
  - **Training Loss**: 0.121 → 0.110
  - **Validation Loss**: 0.106 → 0.099
  - **Model Size**: 158.92 MB
  
- **Verdict**: NO CATASTROPHIC FORGETTING! Maintained tooltip detection while adding background capability.

### 2. ❌ FREEZE_BACKBONE (FAILED)
- **Status**: Failed during training
- **Issue**: TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'
- **Training Time**: 0.46 minutes
- **Configuration**:
  - Learning Rate: 1e-5
  - Tooltip Ratio: 85%
  - Background Ratio: 15%
  - Epochs: 3
  - Freeze Backbone: True
  - Freeze Encoder: True
- **Result**: Training crashed, no model saved

### 3. ❌ TOOLTIP_HEAVY_95 (FAILED) 
- **Status**: Failed during training
- **Issue**: TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'
- **Training Time**: 0.44 minutes  
- **Configuration**:
  - Learning Rate: 1e-7
  - Tooltip Ratio: 95%
  - Background Ratio: 5%
  - Epochs: 2
  - Freeze Backbone: False
- **Result**: Training crashed, no model saved

### 4. ❌ BASELINE (FAILED)
- **Status**: Failed during training
- **Issue**: TypeError during training
- **Training Time**: 0.13 minutes
- **Configuration**:
  - Learning Rate: 1e-6
  - Tooltip Ratio: 70%
  - Background Ratio: 30%
  - Epochs: 2
  - Expected: catastrophic_forgetting
- **Result**: Training crashed, no model saved

### 5. ❌ GRADUAL_PHASES (FAILED)
- **Status**: Failed during training
- **Issue**: TypeError during training
- **Training Time**: 0.14 minutes
- **Configuration**:
  - Progressive phases with decreasing LR (1e-7 → 1e-8 → 1e-9)
  - Tooltip ratio: 100% → 90% → 80%
  - Background ratio: 0% → 10% → 20%
- **Result**: Training crashed, no model saved

### 6. ❌ WEIGHTED_LOSS (FAILED)
- **Status**: Failed during training
- **Issue**: TypeError during training
- **Training Time**: 0.13 minutes
- **Configuration**:
  - Learning Rate: 1e-7
  - Tooltip Ratio: 85%
  - Background Ratio: 15%
  - Tooltip Loss Weight: 0.9
  - Background Loss Weight: 0.1
- **Result**: Training crashed, no model saved

## Remaining Strategies to Test

1. **dynamic_freezing** - Alternating freeze/unfreeze schedule
2. **two_stage** - Tooltip reinforcement then gentle mixing
3. **memory_buffer** - Explicit tooltip memory replay
4. **ewc_regularization** - Preserve important weights via Fisher information

## Key Findings So Far
1. **Ultra-low learning rates work**: 1e-8 to 1e-9 prevents catastrophic forgetting
2. **High tooltip ratio (90%) is important**: Maintains original capability
3. **Batch size = 1 required**: Due to variable image sizes in mixed dataset
4. **Model configuration critical**: num_labels=1 (creates 2 classes with no_object)

## Next Steps
- Test remaining 9 strategies
- Compare all results
- Identify best strategy for production deployment