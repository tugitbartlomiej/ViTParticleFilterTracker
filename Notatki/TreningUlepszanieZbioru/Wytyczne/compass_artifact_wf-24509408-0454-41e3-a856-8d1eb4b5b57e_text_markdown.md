# Selecting Informative Training Images for DETR Object Detection

**For eye surgery applications with labeled datasets, a combination of gradient-based scoring (EL2N), forgetting events, and diversity-aware coreset selection offers the most practical path to efficient training.** Research shows that 50% of typical training data can often be pruned without performance loss—and in some cases, careful subset selection actually *improves* model performance by removing redundant or mislabeled samples. The key insight from recent work is that no single method dominates; the optimal strategy depends on your target pruning rate, with score-based methods excelling at moderate pruning (30-50%) and coverage-based methods becoming essential at aggressive pruning (70%+).

This report covers proven methods from 2020-2025, with specific attention to object detection architectures like DETR and the unique challenges of surgical imaging.

---

## Coreset selection forms the methodological backbone

Coreset selection aims to identify a weighted subset of training data that approximates the full dataset's utility. Three families of algorithms have emerged as most effective:

**Gradient-matching methods** select samples whose aggregated gradients match those of the full dataset. **CRAIG** (ICML 2020) formulates this as submodular function maximization, achieving 3-6× speedup while maintaining accuracy. **GradMatch** (ICML 2021) improves this through orthogonal matching pursuit and dynamic subset updates during training—particularly valuable since the most informative samples change as the model learns. Both methods are implemented in the **CORDS library** (github.com/decile-team/cords), which provides adaptive PyTorch data loaders.

**Score-based methods** compute per-sample importance metrics early in training. The **EL2N score** (Error L2-Norm) measures the magnitude of prediction error: samples with high EL2N are "hard" examples the model struggles with. Remarkably, EL2N computed at epoch 20 transfers across architectures—scores from a ResNet-18 can guide pruning for ResNet-50 training. The landmark NeurIPS 2021 paper "Deep Learning on a Data Diet" demonstrated that **50% of CIFAR-10 can be pruned without accuracy loss** using EL2N, while **forgetting scores** (counting how often a sample transitions from correct to incorrect during training) proved even more stable across experimental conditions.

**Geometry-based methods** ensure coverage of the feature space. **k-Center Greedy** (ICLR 2018) iteratively selects points minimizing the maximum distance to any unselected point—critical for detection where spatial diversity matters. A 2023 ICLR paper on **coverage-centric coreset selection** found that at **90% pruning rates**, coverage-based methods achieve 19.56% higher accuracy than importance-based methods because pure difficulty scoring fails to maintain distributional coverage.

| Method | Best Pruning Range | Computational Cost | Key Advantage |
|--------|-------------------|-------------------|---------------|
| EL2N Scores | 30-50% | Low (epoch 20 only) | Fast, cross-architecture transfer |
| Forgetting Events | 10-30% | Low (computed during training) | Most stable scores |
| GradMatch | Any | Low | Dynamic adaptation |
| k-Center Greedy | 1-10% | Medium-high | Geometric coverage |
| Facility Location | 50-90% | Medium | Submodular guarantees |

The **DeepCore library** (github.com/PatrickZH/DeepCore) implements all major methods with standardized benchmarks across CIFAR, TinyImageNet, and ImageNet.

---

## Sample importance metrics enable fine-grained ranking

Beyond coreset selection, several metrics specifically quantify how much individual samples contribute to model learning:

**Data Shapley values** offer theoretically principled attribution based on cooperative game theory, distributing "credit" fairly among training samples. However, exact computation is exponential, making approximations essential. **KNN-Shapley** achieves O(N log N) complexity by computing exact Shapley values for k-nearest neighbor classifiers using last-layer embeddings as a proxy for deep model importance. **Data-OOB** (ICML 2023) provides an even faster alternative for bagging models, scaling to millions of samples in hours on a single CPU while outperforming KNN-Shapley on mislabel detection tasks.

**Influence functions** measure how upweighting a training sample affects model predictions, enabling identification of samples that most influenced specific predictions. However, Basu et al. (2020) demonstrated that influence functions are "fragile in deep learning"—estimates become imprecise for deep architectures like ResNet-50, limiting their utility for modern detectors. The **pyDVL library** (pydvl.org) provides comprehensive implementations of Shapley, influence functions, and KNN-Shapley methods.

For practical deployment, **EL2N and forgetting scores** offer the best trade-off between computational cost and effectiveness. A key finding from the NeurIPS 2022 Outstanding Paper "Beyond Neural Scaling Laws" is that the interpretation of scores depends on your data regime:

- **Data abundant**: Keep high-scoring (hard) examples
- **Data scarce**: Keep low-scoring (easy) examples

This reversal occurs because hard examples provide more gradient signal when you have enough data to learn from them, but easy examples establish foundational patterns when data is limited.

---

## Object detection requires specialized aggregation strategies

Standard sample importance methods assume single outputs per sample, but object detection produces multiple predictions per image. Several recent methods address this:

**MI-AOD** (CVPR 2021) treats images as instance bags and uses Multiple Instance Learning to re-weight object-level uncertainties before aggregation to image-level scores. The method achieves **93.5% of full-data performance with only 20% labeled data** on PASCAL VOC. Implementation is available in MMDetection format (github.com/yuantn/MI-AOD).

**Hierarchical Uncertainty Aggregation** (ICLR 2023) uses Evidential Deep Learning to compute epistemic uncertainty per bounding box, then aggregates bottom-up based on box attributes (nearest object, box size). This significantly outperforms naive entropy averaging.

**PPAL** (CVPR 2024) introduces **Difficulty Calibrated Uncertainty Sampling**, computing category-wise difficulty coefficients that consider both classification and localization uncertainty. Critically, it favors minority object categories—essential for surgical imaging where certain instruments appear rarely.

For DETR specifically, no dedicated sample selection papers exist (representing a research gap), but several DETR properties inform strategy:

- **Cross-attention maps** indicate which image regions each object query attends to—images where queries produce diffuse or conflicting attention patterns may be more informative
- **Bipartite matching cost** from Hungarian algorithm assignment could serve as an image-level difficulty metric
- **Decoder auxiliary losses** (from intermediate layers) identify images where intermediate representations struggle

**CSOD** (CVPR 2024 Workshop) presents the first coreset selection method specifically designed for object detection, using "imagewise-classwise feature vectors" that average features per class within each image. This achieved **+6.4% AP50** over random selection when selecting 200 images from PASCAL VOC.

---

## Curriculum and distillation offer complementary approaches

**Curriculum learning** orders training samples from easy to hard rather than selecting subsets. **MentorNet** (ICML 2018) learns a curriculum dynamically using an auxiliary network that supervises the main training, achieving state-of-the-art results on WebVision's 2.2 million noisy labels. Self-paced learning variants compute sample difficulty via training loss—low loss indicates easy samples that should appear first in training. The key insight from "When Curricula Work" (ICLR 2021) is that even standard SGD exhibits an implicit curriculum, and explicit curriculum design primarily helps with noisy labels or extreme class imbalance.

**Dataset distillation** synthesizes smaller datasets that preserve training utility. **Trajectory Matching (MTT)** optimizes synthetic images to induce similar training dynamics to real data, achieving 71.5% on CIFAR-10 with only 50 images per class. **TESLA** (ICML 2023) scales this to ImageNet with 6× memory reduction. However, a 2024 analysis found that current distillation methods face "fundamental limitations: 20-50% performance gaps, computational intensity often exceeding savings, and limited cross-architecture generalization."

For object detection, **DCOD** (NeurIPS 2024) presents the first dataset condensation framework for detection, using a "Fetch and Forge" approach that stores localization and classification information in model parameters then reconstructs synthetic images via inversion. This is early-stage research not yet validated at scale.

**Practical recommendation**: Curriculum learning offers low overhead and can be combined with subset selection. Apply self-paced ordering after selecting your informative subset, progressing from low-loss to high-loss samples during training.

---

## Medical imaging demands domain-specific considerations

Eye surgery imaging introduces unique challenges that generic methods don't address:

**Class imbalance** is severe in surgical data. The **CaDIS dataset** for cataract surgery explicitly addresses this through hierarchical task design: Task I merges all instruments into a single class, Task II groups instruments by appearance similarity, and Task III uses all 36 classes. For DETR training, consider weighting queries or loss terms to favor rare instruments. **SMOTE** (Synthetic Minority Over-sampling) has been validated for surgical monitoring systems to generate synthetic samples of minority instrument classes.

**Temporal structure** matters for surgical video. **TRandAugment** (IJCARS 2023) treats surgical videos as temporal segments with consistent transformations, achieving 1-6% improvement over random augmentation on the CATARACTS dataset. When selecting frames, ensure coverage across surgical phases—instrument visibility varies dramatically between phases.

**Inter-patient variability** causes domain shift. The 2024 **Cataract-1K dataset** demonstrated that cross-domain performance drops significantly (77% → 67% Dice) between datasets from different surgical centers. Your selection strategy should ensure representation across patient demographics and surgeon experience levels.

**Annotation quality** varies substantially. Studies report 84-92% inter-annotator agreement on frame-level surgical annotations when annotators receive standardized training. Consider weighting samples by annotation confidence if you have multiple annotations per image.

Key benchmark datasets for validation:

| Dataset | Images/Videos | Classes | Focus |
|---------|--------------|---------|-------|
| CATARACTS | 50 videos | 21 instruments | Cataract tool detection |
| CaDIS | 4,670 images | 36 classes | Semantic segmentation |
| Cataract-1K | 1000+ videos | Phase + segmentation | Largest cataract dataset (2024) |
| CholecT50 | 50 videos | Triplets | Action recognition standard |

Validated active learning in medical imaging achieves compelling efficiency: **O-MedAL reached baseline accuracy using only 25% of labeled data**, and multitask active learning for coronary calcium scoring achieved optimal performance with **only 12% of training data**.

---

## Practical implementation roadmap

For a large labeled eye surgery dataset targeting DETR training, the following workflow synthesizes the research findings:

**Phase 1: Compute sample scores (days 1-2)**
1. Train a proxy model (smaller backbone, e.g., ResNet-18) for 20 epochs
2. Compute EL2N scores at epoch 20: `score = ||softmax(logits) - one_hot(label)||_2`
3. Track forgetting events throughout training: count correct→incorrect transitions
4. Extract penultimate layer features for all images

**Phase 2: Diversity-aware selection (day 3)**
1. Apply k-Center Greedy on extracted features to ensure coverage
2. Within each k-center cluster, rank by EL2N scores
3. Select the top-k samples from each cluster (k based on target dataset size)
4. Verify class balance—oversample underrepresented instruments if needed

**Phase 3: Detection-specific refinement**
1. Weight selection toward images with moderate object counts (avoid empty images and extreme density)
2. Prioritize images containing rare instrument classes (using class frequency from annotations)
3. Ensure surgical phase coverage if phase labels are available

**Phase 4: Training with curriculum**
1. Order selected samples by increasing difficulty (EL2N score)
2. Use warmup epochs on easier 50% before introducing harder samples
3. Monitor per-class AP during training to detect instrument-specific failures

**Tools and libraries**:
- **DeepCore**: Comprehensive coreset methods (github.com/PatrickZH/DeepCore)
- **CORDS**: Adaptive data loaders with GradMatch, CRAIG (github.com/decile-team/cords)
- **pyDVL**: Data valuation including KNN-Shapley (pydvl.org)
- **MI-AOD**: Object detection active learning (github.com/yuantn/MI-AOD)
- **PPAL**: Detection-specific calibrated sampling (github.com/ChenhongyiYang/PPAL)

---

## Conclusion: Combine scoring with coverage for robust selection

The most effective approach for selecting DETR training images from labeled surgical data combines multiple complementary signals. **Score-based methods** (EL2N, forgetting) identify individually informative samples but fail at aggressive pruning rates. **Coverage-based methods** (k-center, facility location) ensure distributional representation but may include easy, uninformative samples. **Detection-specific aggregation** (MI-AOD's MIL reweighting, PPAL's class-calibrated scoring) handles the multi-object nature of detection.

Start conservatively with 50% data retention using EL2N selection, validate that DETR performance remains within 2-3% of full-data training, then iteratively prune further if computational constraints demand it. The 2022 finding that data pruning can "beat power law scaling" suggests that careful curation doesn't just save computation—it may genuinely improve model quality by removing redundant and potentially mislabeled samples.

For eye surgery specifically, prioritize rare instrument coverage and surgical phase diversity over pure difficulty-based selection. The field's most successful deployments combine automated scoring with domain expertise to validate that selected samples capture the full spectrum of surgical complexity.