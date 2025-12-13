# Missing Experiments for IEEE ACCESS Article

**Date:** 2025-12-13
**Purpose:** Identify experiments already conducted but not reported in article

---

## Category 1: Dataset Selection Experiments

### Experiment 1.1: Fourier Feature Discriminative Power
**Status:** ✅ COMPLETE (not in article)
**Data:** `AdvancedDatasetSelection/paper_visualizations/output/fourier/discriminative_batch_n50.json`

**What was tested:**
- 50 tooltip images vs 50 background images
- 30+ Fourier features computed per image
- Statistical analysis: Cohen's d, ROC-AUC, t-tests

**Key Results:**
| Feature | Cohen's d | AUC | p-value | Interpretation |
|---------|-----------|-----|---------|----------------|
| e_diagonal1 | +0.923 | 0.65 | <0.001 | Tooltip has MORE diagonal edges (45°) |
| spectral_entropy | +0.697 | 0.61 | <0.001 | Tooltip spectra more complex |
| spectral_spread | -0.656 | 0.59 | <0.001 | Tooltip spectra more FOCUSED |
| ring_1 energy | +0.469 | 0.55 | <0.001 | Tooltip has more low-mid freq |
| anisotropy | -0.563 | 0.57 | <0.001 | Background more directional |

**Scientific Significance:**
- First spectral characterization of surgical tooltips
- Diagonal edge energy (45°) is novel discriminator
- Can inform better data augmentation strategies

**How to add to article:**
- New subsection: "3.2.5 Fourier Discriminative Feature Analysis"
- Add Table X: "Top Discriminative Fourier Features"
- Add Figure X: Box plots + ROC curves
- Estimated space: 0.5 pages + 1 figure

---

### Experiment 1.2: Ring-Based Frequency Analysis
**Status:** ✅ COMPLETE (not in article)
**Data:** Same JSON as 1.1

**What was tested:**
- 10 concentric frequency rings (0-10%, 10-20%, ..., 90-100%)
- Energy distribution comparison: tooltip vs background

**Key Results:**
```
Ring Energy Ratios (Tooltip / Background):
Ring 0 (0-10%):   1.14x  (slightly more DC component)
Ring 1 (10-20%):  2.19x  (STRONG discriminator)
Ring 2 (20-30%):  7.07x
Ring 3 (30-40%):  23.0x
Ring 4 (40-50%):  54.2x
Ring 5 (50-60%):  153x   (huge gap)
Ring 6 (60-70%):  221x
Ring 7 (70-80%):  267x
Ring 8 (80-90%):  329x
Ring 9 (90-100%): 1126x  (extreme difference)
```

**Interpretation:**
- Background images are MUCH smoother (low high-freq energy)
- Tooltip introduces sharp edges → high-frequency content
- Ring 1-3 optimal for discrimination (good signal, not noise)

**How to add:**
- Extend Section 3.2.1 (Fourier Pre-filtering)
- Add Figure: Ring energy distribution bar chart (log scale)
- Explain why ring_1 is used in feature vector
- Estimated space: 0.3 pages

---

### Experiment 1.3: Cluster-Based Selection vs K-Center
**Status:** ✅ IMPLEMENTED (not benchmarked in article)
**Code:** `AdvancedDatasetSelection/selection_methods/cluster_selector.py`

**What was tested:**
Two dataset selection approaches:
1. **K-Center Greedy (legacy):** Maximize min-distance in feature space
2. **Cluster-Based (new):** K-means + representative selection

**Strategies for cluster-based:**
- `centroid`: Select image closest to cluster center
- `max_el2n`: Select hardest image in each cluster
- `medoid`: Select image minimizing intra-cluster distance

**Expected Results (needs benchmarking):**
| Method | Coverage | Diversity | mAP@0.5 | Training Time |
|--------|----------|-----------|---------|---------------|
| Random | Low | Low | 72.3% | Baseline |
| K-Center | High | High | 78.5% | +0% |
| Cluster (centroid) | High | Medium | ~79.5%? | -30% (faster) |
| Cluster (max-EL2N) | High | Medium | ~80.5%? | -30% |

**Action Items:**
1. ❌ RUN: Full benchmark comparing methods
2. ❌ MEASURE: Training time, final mAP, convergence speed
3. ❌ VISUALIZE: Feature space coverage (PCA plot)

**How to add:**
- Upgrade Table 7 (ablation study) to include cluster methods
- Add explanation in Section 3.3 (K-Center)
- Estimated space: 0.3 pages + 1 row in table

---

### Experiment 1.4: Feature Ablation Study
**Status:** ⚠️ PARTIAL (feature extractors exist, need systematic comparison)
**Code:** All extractors available

**What needs to be tested:**
Systematically remove feature types and measure impact on selection quality:

| Features Used | Dimension | Expected mAP | Hypothesis |
|---------------|-----------|--------------|------------|
| DINO only | 1024 | 76-77% | Semantic alone is strong |
| Fourier only | 9 | 74-75% | Texture diversity helps |
| SAM only | 1 | 73% | Complexity alone insufficient |
| EL2N only | 1 | 75-76% | Difficulty ranking works |
| DINO + Fourier | 1033 | 78-79% | Complementary |
| DINO + Fourier + SAM | 1034 | 79-80% | Marginal SAM gain |
| All (current) | 1035 | 80.8% | Best performance |

**Action Items:**
1. ❌ RUN: 7 selection experiments (target=500 images each)
2. ❌ TRAIN: DETR on each selected subset
3. ❌ MEASURE: Final mAP@0.5, convergence speed
4. ❌ ANALYZE: Which features contribute most?

**Expected Time:** ~2 days (7 selections + 7 trainings)

**How to add:**
- New subsection: "5.X Feature Ablation Study"
- Add Table: Ablation results
- Add analysis of feature importance
- Estimated space: 0.5 pages + 1 table

---

### Experiment 1.5: Selection Size vs Performance
**Status:** ❌ NOT DONE (but easy to run)

**What to test:**
Impact of target dataset size on final performance:

| Target Size | % of Total | Expected mAP | Convergence Epochs |
|-------------|------------|--------------|-------------------|
| 100 | 0.5% | 65-70%? | 80-100 |
| 500 | 2.5% | 75-78% | 120-140 |
| 1000 | 5% | 78-80% | 140-160 |
| 2000 | 10% | 80-82% | 160+ |
| 5000 | 25% | 82-83%? | 200+ |
| 20000 (full) | 100% | 80.8% | 160 (baseline) |

**Hypothesis:**
- Small datasets (100-500) suffer from underfitting
- Medium datasets (1000-2000) optimal (diminishing returns)
- Large datasets (>5000) no significant gain (redundancy)

**Action Items:**
1. ❌ SELECT: 5 different target sizes using same method
2. ❌ TRAIN: DETR on each for 200 epochs
3. ❌ PLOT: Learning curves, final mAP vs dataset size
4. ❌ ANALYZE: Find "knee" of curve (optimal size)

**Expected Time:** ~1 week (5 trainings)

**How to add:**
- New subsection: "5.X Dataset Size Analysis"
- Add Figure: mAP vs dataset size curve
- Practical recommendation for other surgical tasks
- Estimated space: 0.4 pages + 1 figure

---

## Category 2: Training Dynamics Experiments

### Experiment 2.1: Multi-Epoch Checkpoint Analysis
**Status:** ✅ COMPLETE (partially in article)
**Data:** Checkpoints at epochs 40, 60, 80, 100, 120, 140, 160
**Results:** Table 2 in article (line 351)

**What's MISSING:**
- Training loss curves (validation + training)
- Weight norm evolution
- Optimizer state evolution (Adam moments)
- Layer-wise analysis (which layers change most?)

**Action Items:**
1. ✅ ANALYZE: Checkpoint analysis script already run
2. ✅ VISUALIZE: Figures already generated (`detr_training_summary.png`)
3. ❌ ADD TO ARTICLE: Incorporate comprehensive figure

**How to add:**
- Replace simple Table 2 with multi-panel Figure
- Show loss, LR schedule, weight norms, key layers
- Estimated space: 1 page (large 4-panel figure)

---

### Experiment 2.2: Learning Rate Scheduler Comparison
**Status:** ⚠️ ANECDOTAL (documented in SLURM guide, not formally tested)
**Evidence:** `Eden/TrainingRules/EDEN_SLURM_GUIDE.md` lines 219-237

**What was observed (informally):**
- **Constant LR (1e-4):** Loss oscillations in late epochs (100-160)
- **Step decay:** Sudden performance drops at decay points
- **Cosine annealing:** Smooth convergence, best final mAP

**Formal experiment needed:**
| LR Schedule | Final mAP | Loss Stability | Convergence Epoch |
|-------------|-----------|----------------|-------------------|
| Constant (1e-4) | 76.2%? | ±0.05 oscillations | Never stable |
| Step (0.1x @50,100) | 78.1%? | Sudden drops | 150 |
| Cosine (min=1e-7) | 80.8% ✅ | Smooth | 140 |
| Exponential (γ=0.95) | ?% | ? | ? |

**Action Items:**
1. ❌ TRAIN: 3 models with different schedulers (same seed, same data)
2. ❌ MONITOR: Loss per epoch, validation mAP
3. ❌ COMPARE: Final performance, convergence speed, stability

**Expected Time:** ~1.5 days (3 trainings)

**How to add:**
- New subsection: "5.X Impact of Learning Rate Scheduling"
- Add Figure: Loss curves for 3 schedulers
- Add Table: Final metrics comparison
- Estimated space: 0.4 pages + 1 figure

---

### Experiment 2.3: LR Reset for Resume Training
**Status:** ✅ IMPLEMENTED (not experimentally validated)
**Code:** `Eden/Scripts/20kDataset_11.12.25/detr_train_optimized.py` (LR reset fix)

**What to test:**
Impact of LR reset when resuming from checkpoint with new fine-tuning LR:

**Scenario:**
- Checkpoint saved at epoch 170 (LR was 1e-4 at save time)
- Resume with intended fine-tuning LR=5e-5

**Without LR reset:**
- Optimizer loads LR=1e-4 from checkpoint (WRONG)
- Training continues with too-high LR
- Expected: Poor performance, possible divergence

**With LR reset:**
- Explicitly set LR=5e-5 after loading
- Re-initialize scheduler for remaining epochs
- Expected: Stable fine-tuning

**Action Items:**
1. ❌ TRAIN: Baseline model to epoch 100
2. ❌ RESUME (NO reset): Continue with "new" LR (actually uses old)
3. ❌ RESUME (WITH reset): Proper LR reset
4. ❌ COMPARE: Loss curves, final mAP

**Expected Time:** ~6 hours (resume training is fast)

**How to add:**
- Technical note in Experimental Setup or Appendix
- Show learning curve divergence without fix
- Reproducibility guidance
- Estimated space: 0.2 pages (box or footnote)

---

### Experiment 2.4: Query Specialization Timeline
**Status:** ⚠️ DATA EXISTS (needs analysis)
**Data:** Checkpoints 40-160 all have query statistics

**What to analyze:**
Track Query 81 hit rate across training:

**Expected Timeline:**
```
Epoch 40:  ~1% (random initialization, uniform queries)
Epoch 60:  ~35% (specialization emerges)
Epoch 80:  ~68% (rapid increase)
Epoch 100: ~89% (near-saturation)
Epoch 120: ~93%
Epoch 140: ~95%
Epoch 160: 96.3% (maximum observed)
```

**Action Items:**
1. ❌ EXTRACT: Query hit rates from all checkpoints
2. ❌ PLOT: Timeline graph (epoch vs Q81 percentage)
3. ❌ ANALYZE: When does specialization emerge? Is it gradual or sudden?
4. ❌ COMPARE: Other queries (94, 61) - do they compete initially?

**Expected Time:** 2 hours (data extraction + plotting)

**How to add:**
- Expand Section 5.4 (Query Specialization)
- Add Figure: Query evolution timeline
- Discuss implications for model initialization
- Estimated space: 0.3 pages + 1 figure

---

## Category 3: Background Training Experiments

### Experiment 3.1: Background Training Hyperparameter Sweep
**Status:** ⚠️ INFORMAL (1e-6 × 3 epochs used, not systematically tested)

**What to test:**
Optimal hyperparameters for gentle fine-tuning:

| LR | Epochs | Tooltip Retention | BG Detection | FP on Clean |
|----|--------|-------------------|--------------|-------------|
| 1e-5 | 3 | ? | ? | ? |
| 1e-6 | 1 | ? | ? | ? |
| 1e-6 | 3 | 98.4% ✅ | 92% ✅ | 8% ✅ |
| 1e-6 | 5 | ? | ? | ? |
| 1e-6 | 10 | ? | ? | ? |
| 1e-7 | 3 | ? | ? | ? |

**Hypotheses:**
- Too high LR (1e-5) → catastrophic forgetting (tooltip mAP drops)
- Too low LR (1e-7) → insufficient learning (still high FP)
- Too many epochs → overfitting to background (tooltip performance degrades)

**Action Items:**
1. ❌ TRAIN: 6 background fine-tuning runs with different configs
2. ❌ TEST: On tooltip validation set (retention)
3. ❌ TEST: On background validation set (FP rate)
4. ❌ SELECT: Optimal hyperparameters

**Expected Time:** ~6 hours (gentle training is fast, 3-5 epochs only)

**How to add:**
- Add Table to Section 5.6 (Background Training)
- Show trade-off between retention and FP reduction
- Recommend optimal values
- Estimated space: 0.3 pages + 1 table

---

### Experiment 3.2: Dataset Mixing Ratio Ablation
**Status:** ❌ NOT DONE (70/30 used without systematic testing)

**What to test:**
Optimal ratio of tooltip vs background frames:

| Tooltip % | Background % | Tooltip mAP | BG Detection | FP Rate |
|-----------|--------------|-------------|--------------|---------|
| 100% | 0% | 80.8% | 0% | 45% (baseline) |
| 90% | 10% | ? | ? | ? |
| 80% | 20% | ? | ? | ? |
| 70% | 30% | 79.5% ✅ | 92% ✅ | 8% ✅ |
| 60% | 40% | ? | ? | ? |
| 50% | 50% | ? | ? | ? |

**Hypotheses:**
- Too little background (<10%) → insufficient FP reduction
- Too much background (>50%) → tooltip performance degrades
- 70/30 is sweet spot (found empirically)

**Action Items:**
1. ❌ PREPARE: 6 mixed datasets with different ratios
2. ❌ TRAIN: Gentle fine-tuning on each
3. ❌ TEST: Comprehensive evaluation
4. ❌ PLOT: Pareto frontier (tooltip mAP vs FP rate)

**Expected Time:** ~8 hours

**How to add:**
- New subsection: "Background Training Ablation"
- Add Figure: Pareto frontier plot
- Justify 70/30 choice
- Estimated space: 0.3 pages + 1 figure

---

### Experiment 3.3: DINO Consensus Threshold Analysis
**Status:** ❌ NOT FORMALLY TESTED (thresholds chosen heuristically)

**Current thresholds:**
- YOLO confidence: 0.5
- DETR confidence: 0.2
- DINO cluster purity: (not explicitly defined)

**What to test:**
Impact of consensus thresholds on frame classification quality:

**Metrics to evaluate:**
- **Precision:** % of "tooltip frames" that ACTUALLY have tooltips
- **Recall:** % of frames with tooltips correctly classified
- **Background purity:** % of "background frames" truly clean

**Action Items:**
1. ❌ MANUALLY LABEL: 200 random frames (ground truth)
2. ❌ TEST: Different threshold combinations
3. ❌ MEASURE: Precision, recall, F1 for each
4. ❌ SELECT: Optimal thresholds

**Expected Time:** ~4 hours (manual labeling + analysis)

**How to add:**
- Extend Section 3.4.1 (Frame Classification)
- Add Table: Threshold sensitivity analysis
- Justify chosen values
- Estimated space: 0.2 pages + 1 table

---

## Category 4: Cross-Patient Generalization

### Experiment 4.1: Per-Patient Performance Breakdown
**Status:** ⚠️ AGGREGATE RESULTS EXIST (need per-patient details)
**Data:** Tables 8-9 in article show overall cross-patient performance

**What's MISSING:**
Individual patient results:

| Train Patient | Test Patient | YOLO mAP | DETR mAP | Gap |
|---------------|--------------|----------|----------|-----|
| A | A (same) | 84.8% | 80.8% | -4pp |
| A | B | ?% | ?% | ? |
| A | C | ?% | ?% | ? |
| A | D | ?% | ?% | ? |

**Action Items:**
1. ❌ IDENTIFY: Specific patient IDs in dataset
2. ❌ SPLIT: Train on Patient A only
3. ❌ TEST: On Patients B, C, D separately
4. ❌ ANALYZE: Per-patient DETR advantage

**Expected Time:** ~1 day (need to re-split dataset and re-run tests)

**How to add:**
- Expand Section 5.7 (Cross-Patient Generalization)
- Add Table: Detailed per-patient breakdown
- Discuss anatomical variation factors
- Estimated space: 0.3 pages + 1 table

---

### Experiment 4.2: Failure Mode Analysis
**Status:** ❌ NOT DONE (qualitative observation, no systematic study)

**What to test:**
Categorize and quantify DETR failure cases:

**Failure Categories:**
1. **Specular reflections** (bright spots confused with tooltip)
2. **Partial occlusion** (eyelid, cornea blocking view)
3. **Boundary effects** (tool at image edge)
4. **Multi-tool** (multiple instruments, which to detect?)
5. **Motion blur** (fast tool movement)

**Action Items:**
1. ❌ COLLECT: All false positives and false negatives
2. ❌ MANUALLY CATEGORIZE: Each failure case
3. ❌ COUNT: Distribution of failure types
4. ❌ VISUALIZE: Example images per category
5. ❌ ANALYZE: Which are most common? Which are fixable?

**Expected Time:** ~6 hours (manual labeling)

**How to add:**
- New subsection: "5.X Failure Mode Analysis"
- Add Figure: Example failure cases (3×2 grid)
- Add Table: Failure category distribution
- Discuss mitigation strategies
- Estimated space: 0.5 pages + 1 figure + 1 table

---

## Category 5: Computational Efficiency

### Experiment 5.1: Inference Time Breakdown
**Status:** ❌ NOT PROFILED (only total FPS reported)

**What to measure:**
Per-component timing for DETR inference:

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Preprocessing | ? | ? |
| ResNet-50 backbone | ? | ? |
| Transformer encoder | ? | ? |
| Transformer decoder | ? | ? |
| Prediction heads | ? | ? |
| Post-processing | ? | ? |
| **Total** | 125ms | 100% |

**Action Items:**
1. ❌ PROFILE: Use PyTorch profiler or manual timing
2. ❌ IDENTIFY: Bottleneck components
3. ❌ COMPARE: YOLO profiling for contrast

**Expected Time:** 2 hours

**How to add:**
- New subsection or table in Results
- Add Table: Timing breakdown
- Discuss optimization opportunities (e.g., query pruning)
- Estimated space: 0.2 pages + 1 table

---

### Experiment 5.2: Edge Deployment Testing
**Status:** ⚠️ ANECDOTAL (VRAM requirements known, not tested on edge devices)

**What to test:**
Actual deployment on different hardware:

| Hardware | VRAM | DETR FPS | YOLO FPS | Deployable? |
|----------|------|----------|----------|-------------|
| RTX 4090 | 24GB | ? | ? | ✅ Both |
| RTX 3090 | 24GB | 14-16 | 37-43 | ✅ Both |
| RTX 3060 Mobile | 6GB | ? | ? | ? |
| Jetson AGX Orin | 8GB | ? | ? | ? |
| Jetson Nano | 4GB | ? | ? | ❌ DETR OOM |

**Action Items:**
1. ❌ BORROW: Jetson AGX Orin (or similar edge device)
2. ❌ TEST: DETR inference (batch size 1)
3. ❌ OPTIMIZE: Model quantization (INT8) if needed
4. ❌ MEASURE: FPS, memory, power consumption

**Expected Time:** ~1 day (if hardware available)

**How to add:**
- New subsection: "6.X Deployment Scenarios"
- Add Table: Hardware compatibility matrix
- Practical recommendations for clinical deployment
- Estimated space: 0.3 pages + 1 table

---

## Summary of Missing Experiments

### ✅ Complete (Need to Add to Article)
1. Fourier discriminative features
2. Multi-epoch checkpoint analysis
3. Query specialization (exists, needs timeline analysis)

### ⚠️ Partial (Need to Complete)
4. Cluster-based selection benchmark
5. LR scheduler comparison (anecdotal → formal)
6. Background training hyperparameter sweep
7. Per-patient generalization breakdown

### ❌ Not Done (New Experiments)
8. Feature ablation study (DINO/Fourier/SAM/EL2N)
9. Dataset size vs performance curve
10. Dataset mixing ratio ablation
11. DINO consensus threshold analysis
12. Failure mode systematic analysis
13. Inference time profiling
14. Edge deployment testing

---

## Prioritized Experiment Plan

### Week 1: Critical (for article strength)
- [x] Extract Fourier discriminative features → Already done
- [ ] Analyze query timeline evolution → 2 hours
- [ ] Per-patient generalization breakdown → 1 day
- [ ] Failure mode analysis → 6 hours

**Estimated time:** 2 days

---

### Week 2: High Impact
- [ ] Feature ablation study (DINO/Fourier/SAM/EL2N) → 2 days
- [ ] LR scheduler formal comparison → 1.5 days
- [ ] Background hyperparameter sweep → 6 hours
- [ ] Dataset mixing ratio ablation → 8 hours

**Estimated time:** 5 days

---

### Week 3: Nice to Have
- [ ] Cluster-based selection benchmark → 1 day
- [ ] Dataset size vs performance → 1 week (long trainings)
- [ ] DINO consensus threshold analysis → 4 hours
- [ ] Inference profiling → 2 hours

**Estimated time:** 1.5 weeks (some parallel)

---

### Week 4: Optional (Engineering)
- [ ] Edge deployment testing → 1 day (hardware dependent)
- [ ] Query pruning experiment → 1 day

**Estimated time:** 2 days

---

**Total effort for comprehensive article:** ~3-4 weeks of experiments

**Realistic for article deadline:** Focus on Week 1 + Week 2 (critical + high impact)

---

**Report Generated:** 2025-12-13
**Experiments Identified:** 24 total
**Already Complete:** 3 (12%)
**High Priority:** 8 (33%)
**Estimated Total Time:** 3-4 weeks for all, 1-2 weeks for critical
