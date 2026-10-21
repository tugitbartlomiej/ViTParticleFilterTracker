# Occlusion of surgical instruments by tissue, and what a "latent microscope" can reveal about it

Scope: detection and tracking of partly hidden instruments (tip under iris, lens fragments, tissue, blood, specular glare, other tools) in cataract-surgery video. The setting is a small ViT with a dense CenterNet/FCOS-style head on a 20x30 token grid plus a particle filter (PF). The question is how to read the latent microscope: tokens as particles, 768 last-block MLP neurons as "lasers", and local force-field traps.

Provenance markers used below:
- **[full text]**: the paper's PDF was read through Undermind `read_pdfs`, and the numbers come from its tables or text.
- **[abstract only]**: only the abstract or metadata was available. Treat these claims as weaker.
- **[graph-read]**: a value read approximately from a plot by the PDF reader. It is not a tabulated number.

Deep searches run for this note (Undermind workspace 40294fa6-...):
- [Occlusion in surgical instrument detection and tracking](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Occlusion%20in%20surgical%20instrument%20detection%20and%20tracking): 156 relevant papers.
- [Internal representation signatures of occlusion and uncertainty](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Internal%20representation%20signatures%20of%20occlusion%20and%20uncertainty%20for%20detection%20and%20tracking): 222 relevant papers.

User's own measurements, used as context (one model, seed 3047, not literature):
- Tool tokens separate from the background cloud. From epoch 40 on, 96–98% of them lie outside the background's 95% radius.
- The last block pushes background toward the tool cluster (the "brake", −11–13% separation).
- IoU plateaus at about 0.76 from epoch 40, while confidence keeps rising, also on frames without an instrument.
- There are 15–60 tool-selective neurons, depending on the epoch.
- On the plateau, epochs with more silent neurons and fewer tool-selective neurons have lower IoU (preliminary).

---

## Q1. Occlusion in surgical tool detection and tracking: datasets, performance drop, and methods that help

### Takeaway
No public **cataract** dataset found here provides explicit per-instrument occlusion or visibility labels. Cataract-1K, CaDIS/CATARACTS-seg, CATARACTS and Cataract-LMM all name occlusion as a difficulty, but none of them labels it. Explicit visibility, occlusion and amodal labels exist only in laparoscopic and robotic data: CholecTrack20, AIS amodal masks, SegSTRONG-C paired corruptions, and ClipAssistNet tip visibility. The most reliable measured gains under occlusion come from **training with synthetic occlusion** (blood ellipses, patch or grid occluders, checkerboards): often +15 to +22 points on the corrupted or occluded test. Temporal recurrence and visibility-aware tracking add smaller gains on top.

### Cited Findings

#### Dataset table (occlusion/visibility labels, size, access)

| Dataset | Domain | Explicit occlusion / visibility label? | Size and annotation | Access | Evidence |
|---|---|---|---|---|---|
| **Cataract-LMM**, subset `3_Object_Tracking` | Cataract (phaco) | **No explicit flag.** Table 2 lists masks, boxes, persistent IDs, tip keypoints and trajectories, with no visibility or occlusion attribute. During annotation, a tracklet buffer and Kalman filtering were used "to bridge occlusion gaps", and Fig. 3 shows "Instrument tip occlusion by tissue" and hand occlusion as challenges. | 170 capsulorhexis clips, 469,118 frames. COCO masks, `[x,y,w,h]` boxes, persistent IDs, `[x,y]` instrument-tip keypoints, trajectories. Whole corpus: 3,000 procedures (1,134.2 h) from 2 centers (Farabi 2,930, Noor 70). | Hugging Face `mjahmadi/Cataract-LMM`, CC-BY 4.0 | [Cataract-LMM, Sci Data](https://doi.org/10.1038/s41597-026-07464-0) [full text] |
| **Cataract-1K** | Cataract | **No** occlusion, visibility or truncation labels. Fig. 3 only illustrates instruments hidden by the corrupted natural lens or by other instruments, reflections, and partly visible tools. | Segmentation subset: 2,256 frames from 30 videos at 1024x768. 3 anatomy classes and 9 instrument classes. Frames with harsh blur or the iris out of scene were **excluded**. | Public (Sci Data) | [Cataract-1K, Sci Data 2024](https://doi.org/10.1038/s41597-024-03193-4) [full text] |
| **CaDIS / 2020 CATARACTS Segmentation Challenge** | Cataract | **No** occlusion or visibility labels are described in the challenge paper. The CaDIS journal paper was available as an abstract only, so its absence of such labels **cannot be fully confirmed**. | Challenge: 4,670 train and 531 hidden test images, 36 classes (29 instrument). Task III separates instrument **tips and handles**. | cataracts-semantic-segmentation2020.grand-challenge.org | [Luengo et al. 2021](https://arxiv.org/abs/2110.10965) [full text]; [CaDIS, MedIA 2021](https://doi.org/10.1016/j.media.2021.102053) [abstract only] |
| **CATARACTS (tool presence)** | Cataract | **No.** A tool counts as "in use" only when it touches the eyeball. Labels are 0, 1, and 0.5 when the two experts disagree (0.5 frames are excluded from AUC). | 50 videos (25/25), 21 tools, about 30 fps, 1920x1080, more than 9 h | cataracts.grand-challenge.org (current availability not established) | [Al Hajj et al. 2019](https://doi.org/10.1016/j.media.2018.11.008) [full text] |
| **CholecTrack20** | Laparoscopic | **Yes.** Eight visual-challenge labels: blurring, bleeding, lens fouling, crowded scene, **occlusion**, smoke, **specular reflection**, trocar view. There are also three trajectory perspectives (visibility, intracorporeal, intraoperative) and out-of-camera-view / out-of-body flags. | 20 videos, 35k frames at 1 fps, about 65k boxes, 2,624 trajectories | GitHub camma-public/cholectrack20, CC-BY-NC-SA 4.0 | [Nwoye et al., CVPR 2025](https://doi.org/10.1109/CVPR52734.2025.00836) [full text] |
| **AIS (amodal instrument segmentation)** | Robotic (EndoVis 2017 re-annotated) | **Yes, amodal masks.** The full instrument shape is drawn, including the parts hidden by tissue or other tools. | 3,000 frames. 7,084 instances, of which 1,455 are occluded. Mean occlusion rate 20.54%. | Not stated in the extracted text | [Shi et al. 2024](https://doi.org/10.1007/s11220-025-00609-2) [full text] |
| **SegSTRONG-C** | Robotic (ex vivo replay) | **Paired clean and corrupted** recordings of the same robot trajectory: bleeding (fake blood), smoke (fog machine), low brightness, background change | 17 sequences, 10 fps, 300 frames per camera | segstrongc.cs.jhu.edu | [Ding et al. 2024](https://doi.org/10.48550/arXiv.2407.11906) [full text] |
| **ROBUST-MIS 2019** | Laparoscopic | **No** per-image artifact or occlusion labels. Robustness is measured by the 5th percentile of per-case scores and by a Stage-3 domain shift. | 10,040 frames from 30 procedures, instance masks | Synapse, non-commercial | [Ross et al. arXiv](https://arxiv.org/abs/2003.10299) [full text]; [MedIA](https://doi.org/10.1016/j.media.2020.101920) |
| **ClipAssistNet (clipper-tip visibility)** | Laparoscopic | **Yes, at frame level.** A frame is "visible" when both jaw tips are in view and **not occluded by another instrument or tissue**. It is "invisible" otherwise, or when visibility cannot be judged. | 300 videos at 5 fps. 111k train frames, 11.3k test frames with 3 annotators. Annotators **disagreed on 18%** of test frames. | Not established | [Aspart et al. 2021](https://doi.org/10.1007/s11548-021-02441-x) [full text] |
| ROBUST-MIPS | Laparoscopic | Unknown (skeletal pose plus instance segmentation derived from ROBUST-MIS) | Not checked | Sci Data | [Han et al. 2025](https://doi.org/10.1038/s41597-026-06938-5) [abstract only] |
| VL-SurgPT | Laparoscopic point tracking | Textual "point status" descriptions under smoke occlusion and reflections | 908 clips, 154 of them instrument-tracking clips | Not checked | [Zhou et al. 2025](https://doi.org/10.48550/arXiv.2511.12026) [abstract only] |

#### How much performance drops under occlusion and corruption
- **SegSTRONG-C.** The paired-recording design makes this the cleanest measurement of the drop. Mean Dice of four supervised baselines (UNet, DeepLabv3+, SETR, Segformer):
  - bleeding: 0.894 → 0.645 (−0.25)
  - smoke: 0.894 → 0.668 (−0.23)
  - low brightness: 0.894 → 0.439 (−0.46)
  - NSD drops are −0.28 to −0.46.

  Source: [Ding et al. 2024](https://doi.org/10.48550/arXiv.2407.11906) [full text].
- **CholecTrack20.** Detection AP on occlusion-labelled frames varies widely by detector: FCOS 40.8, CenterNet 51.7, Cascade R-CNN 46.4, Deformable DETR 57.4, Swin-T 61.1, YOLOv7 82.1. The paper gives **no normal-frame baseline**, so the drop itself cannot be computed. Its text names lens fouling, smoke and occlusion as significant tracking challenges. Source: [Nwoye et al. 2025](https://doi.org/10.1109/CVPR52734.2025.00836) [full text].
- **ROBUST-MIS.** The winner's multi-instance Dice falls 0.82 → 0.71 → 0.62 → 0.45 for 1, 2, 3 and more than 3 instruments. Small, transparent, partly hidden, crossing and border instruments are the named failure modes. Reflections, blood and illumination "generally did not" cause major problems. No occlusion-stratified numbers are given. Source: [Ross et al.](https://arxiv.org/abs/2003.10299) [full text].
- **Cataract-1K.** Instrument IoU tops out at 63.91% (DeepPyramid VGG16), against much higher anatomy scores. The authors attribute the gap to blur, reflections, occlusion and partial visibility. Source: [Ghamsarian et al. 2024](https://doi.org/10.1038/s41597-024-03193-4) [full text].
- **CATARACTS segmentation challenge.** In Task I, failures come "mostly from missed instrument parts". In Task III, the best IoU for the Rycroft cannula **handle** is only 46.12%. Source: [Luengo et al.](https://arxiv.org/abs/2110.10965) [full text].
- **Cataract-LMM.** YOLOv11-L mask mAP is 70.45 within the same center, but falls to 61.19 when trained on Farabi and tested on Noor, and to 47.12 the other way round. Domain shift is therefore a confound at least as large as occlusion. Source: [Ahmadi et al.](https://doi.org/10.1038/s41597-026-07464-0) [full text].
- **ClipAssistNet.** A single-frame ResNet-50 ensemble predicts tip visibility with AUROC 0.9107. It struggles with slightly occluded tips. Source: [Aspart et al. 2021](https://doi.org/10.1007/s11548-021-02441-x) [full text].

#### Methods table (gain under occlusion)

| Method | Mechanism | Gain under occlusion or corruption (exact) | Source |
|---|---|---|---|
| Blood-ellipse occlusion augmentation | Semi-transparent red ellipses (alpha 0.2–0.6) centred on **tool regions** | U-Net on SegSTRONG-C, corrupted-test Dice **0.512 → 0.735 (+22.3 pp)**. Clean Dice falls slightly, 0.945 → 0.929. | [Shao 2025](https://doi.org/10.54254/2755-2721/2025.ld30738) [full text] |
| Corruption-specific augmentation (SegSTRONG-C entrants) | Simulated blood, occlusion augmentation, CycleGAN smoke, brightness reduction | Bleeding-specific augmentation **+0.073 Dice / +0.114 NSD**. Smoke-specific **+0.084 / +0.176**. The winners averaged 0.9394 Dice, **+0.147** over the strongest baseline. The authors caution that architecture also mattered. | [Ding et al. 2024](https://doi.org/10.48550/arXiv.2407.11906) [full text] |
| Grid-patch occlusion augmentation (pose) | Grid cells inside the tool box are replaced by noise or by background patches. The **labels are masked too.** | Occluded test set (not used in training): about 60–67% vs about 50% ADD accuracy at 4 mm, and about 84% vs 65% at 10 mm, i.e. **≈ +15–20 pp** [graph-read] | [Xu et al. 2024](https://doi.org/10.1049/htl2.12100) [full text] |
| Occlusion-trained, then recurrent (6DoF tip) | Checkerboard occlusion on 50% of frames, then a recurrent spatio-temporal extractor | Single-view screwdriver tip error: **73.48 mm → 29.46 mm** from occlusion training alone (non-recurrent), then **→ 25.86 mm** with recurrence. Recurrence brings no benefit when nothing is occluded. | [Agethen et al. 2025](https://doi.org/10.1007/s11548-025-03436-8) [full text] |
| PermaTrack (visibility head and object permanence) | ConvGRU memory, a **visibility head**, amodal supervision with constant-velocity pseudo-labels for hidden objects | KITTI test HOTA **+5.0** (car) and **+8.2** (person) over CenterTrack (whole benchmark, not occlusion-only). Filtering visibility labels alone gave +3.7 Track mAP on PD. | [Tokmakov et al. 2021](https://doi.org/10.1109/ICCV48922.2021.01068) [full text] |
| Uncertainty-aware Kalman filter (UncertaintyTrack) | Predicted box covariance used as KF measurement noise, plus ellipse filtering and entropy matching | ID switches **−19% (BDD100K) to −55% (MOT17)**. HOTA +0.5 to +1.7. Not stratified by occlusion. | [Lee et al. ICRA 2024](https://doi.org/10.1109/ICRA57147.2024.10610458) [full text] |
| Predictive variance as KF noise | Calibrated detector variance used as measurement variance | Variance "can be linked to object occlusion". Tracking position error −1.3% on nuScenes. | [Ulreich et al. 2024](https://doi.org/10.1109/SDF63218.2024.10773809) [abstract only] |
| Context-aware CompositionalNets (detection) | Per-position competition between object and **occluder** likelihoods, with down-weighted context | Heaviest occlusion AP **27.0 → 68.2 (+41.2)**. OccludedCOCO L4 **24.6 → 59.6 (+35.0)**. Unoccluded vehicles are **worse**: 98.0 → 92.6. | [Wang et al. CVPR 2020](https://doi.org/10.1109/cvpr42600.2020.01266) [full text] |
| Parts-biased transformer attention | Attention biased toward discriminative parts | CityPersons miss rate 32.6 → 30.75 (heavy occlusion is the target) | [Shastry et al. 2023](https://doi.org/10.1109/ICIP49359.2023.10222651) [abstract only] |
| Multi-frame tool-tip keypoints (MFCNet) | 3 frames plus optical flow plus depth | Small or mixed: JIGSAWS detection accuracy +1.7 to +2.1 pp, RMSE not better. No occlusion-specific evaluation. | [Ghanekar et al. ISBI 2025](https://doi.org/10.1109/ISBI60581.2025.10980959) [full text] |
| ToolH-YOLO | Multi-branch features and attention fusion | +1.1 pp mAP@0.5 over YOLOv11 on a custom occlusion-heavy set | [Guo et al. 2025](https://doi.org/10.1109/JSEN.2025.3608705) [abstract only] |
| Amodal segmentation baselines on AIS | SAM, AISFormer, C2F-Seg, PLUG with box prompts | Whole-amodal-mask mean IoU 85.94 / 86.65 / 88.17 / **89.25** (PLUG). **No visible vs hidden-part split** is reported. | [Shi et al. 2024](https://doi.org/10.1007/s11220-025-00609-2) [full text] |
| Background compositing (one background) | Tools pasted onto an augmented tissue background | Not occlusion-specific. Synthetic-only training reaches 73.5 vs 82.9 Dice on real data. | [Wang et al. MICCAI 2022](https://doi.org/10.48550/arXiv.2206.11804) [full text] |
| Bayesian unified detection and tracking (retina) | Active testing combined with Bayesian filtering. Handles tools entering and leaving the view. | No occlusion number was available | [Sznitman et al. TPAMI 2013](https://doi.org/10.1109/TPAMI.2012.209) [abstract only] |
| Articulated tip tracking (retina) | Random forests, multi-template tracker, tip-pose estimation | 18 in-vivo sequences. No occlusion-specific number was available. | [Rieke et al. MedIA 2016](https://doi.org/10.1016/j.media.2016.05.003) [abstract only] |

- In CholecTrack20, re-identification after occlusion or re-insertion is the main tracking difficulty. SurgiTrack addresses it by modelling the tool's originating direction as a proxy for the operator. Source: [Nwoye & Padoy 2024](https://doi.org/10.1016/j.media.2024.103438) [abstract only].
- PhacoTrainer (cataract) derives phaco-tip and second-instrument tip positions from YOLACT masks. Mean errors are 9.1 px and 17.1 px. Source: [Yeh et al. 2023](https://doi.org/10.1167/tvst.12.3.23) [abstract only].

### Inferences
- For a cataract PhD project, occlusion labels will have to be **made**. One option is to synthesize them by copy-pasting tissue, iris or lens texture over the tip, with graded occlusion fractions. The other is to annotate a small set of real frames with tip visibility, for example from Cataract-LMM `3_Object_Tracking`, whose tip keypoints and Kalman-bridged trajectories make "tip hidden" frames findable. ClipAssistNet's 18% annotator disagreement suggests a three-level label (visible / partial / hidden) plus an "uncertain" class, rather than a binary one.
- The literature consistently shows that **training-time occlusion augmentation** is the first-order fix (+15 to +22 pp). Temporal memory and visibility states are second-order (a few points). A microscope-guided method should therefore be judged against "plain occlusion augmentation" as the baseline, not against "no augmentation".
- The CenterNet and FCOS heads scored among the lowest occluded-AP values in CholecTrack20 (51.7 and 40.8). Backbones and training differ, so this is not causal. It is still a warning that center-heatmap heads may be sensitive when the visible part of the instrument is shifted away from the box center, as happens when the tip is hidden.

### Gaps
- No cataract dataset with explicit per-instrument occlusion or visibility labels was found. The CaDIS journal paper was not read in full.
- No paper reported **occlusion-stratified** cataract performance, or a separate visible-part vs hidden-part IoU for amodal instrument masks (AIS reports whole-mask IoU only).
- Particle-filter instrument trackers with explicit occlusion handling were found only as older or abstract-level entries: Rilk et al. 2006, a German BVM paper with no abstract available ([DOI](https://doi.org/10.1007/3-540-32137-3_84)), and Sznitman 2013. Their occlusion-specific numbers were not available.
- CholecTrack20's tracking HOTA by visual challenge is shown only as a figure, with no tabulated values and no normal-frame baseline.

---

## Q2. Amodal perception and occlusion robustness in ViTs: known internal signatures of occlusion

### Takeaway
ViTs keep their final representation much closer to the clean-image representation under random patch drop than CNNs do. Their later layers attend to the remaining visible regions. On **real** occlusion, however, the advantage is modest: drops of about 13–14 pp at ≤50% occlusion and 29–32 pp at >50%, similar to CNNs. The best-supported internal signatures are:
1. evidence diverted to background or context, causing false negatives, or false positives when thresholds are lowered;
2. per-position competition between "object" and "occluder or background" explanations;
3. under recurrence, occluded representations drifting back toward the unoccluded target.

"Lower feature norm" is **not** a validated, general occlusion indicator.

### Cited Findings

#### Internal-signatures table

| Signature | Evidence (exact where available) | Setting | Strength | Possible microscope analogue (inference) |
|---|---|---|---|---|
| Final representation stays close to the clean one | Correlation of the final CLS token between the original and the occluded image at 25/50/75% random patch drop: **DeiT-B 0.90/0.77/0.56**, ViT-L 0.92/0.81/0.50, against **ResNet50 0.32/0.13/0.07** | ImageNet classification, random PatchDrop | full text, but classification and synthetic only | Per-token drift between a clean frame and its occluded twin (paired residual distance) |
| Attention shifts to visible regions in later layers | Early layers attend broadly. Later layers focus on non-occluded regions (qualitative). | DeiT-B | qualitative | Do later blocks move "attention mass" from covered tip tokens to the visible shaft tokens? |
| Raw robustness numbers | At 50% random drop: ResNet50 0.1% vs DeiT-S 70% top-1. At 90%: DeiT-B still 37%. | ImageNet | full text | — |
| Real occlusion: the ViT advantage is small | Accuracy drops from level 0 to level 1 (≤50%) and level 2 (>50%): ViT-B −14.1/−32.3 pp, DeiT-B −13.3/−28.8, Swin-B −14.0/−29.7, ResNeXt-50 −14.6/−31.5. Swin-B+Mixup beats the best CNN by about 10 pp in absolute terms. | IRUO (from OVIS), real and synthetic | full text | Expect real degradation even for a ViT, and test it on real occlusion |
| Sub-patch occluders contaminate patch computations | **Hypothesis**: diffuse occluders smaller than the patch size blur attention to the target | IRUO-Diffuse, ViT-B | hypothesis, qualitative | A thin tip or a thin occluder inside one 16-px patch becomes a mixed token, part tool and part tissue |
| Transformer detection under heavy occlusion attends to background | Deformable keypoints "attend to non-discriminative regions or background", which produces false negatives. Biasing attention to parts helps. | Pedestrian DETR-like | abstract only | Occluded tool tokens drifting into the background cloud |
| Context dominates when the object is weak | When the object is strongly occluded and thresholds are lowered, **context features drive false positives** ("airplane in sky without an airplane"). Down-weighting context (ω=0.2) helps. | CompositionalNets, detection | full text | Matches the observed **rising confidence on empty frames**, and the "brake" pushing background toward the tool cluster |
| Object vs occluder likelihood competition | Each feature position is explained either by a class-specific mixture of von Mises–Fisher kernels or by a generic occluder model learned from object-free images. The log-likelihood ratio gives an **occlusion map**. Occlusion is **not** simply "far from all centers". | CompNets | full text | Per-token log-likelihood ratio between a tool-cluster model and a background-cloud model |
| Channel correlations only on visible parts | Channel features of different pedestrian proposals correlate highly only at visible parts. Occluded features can be "completed" adversarially. | FeatComp++ | abstract only | Correlation of a token with the tool-cluster mean as a visibility map |
| Recurrence discounts occluders | Relative distance d_rel = 2·d_target/(d_target + d_occluder) shifts below 1, toward the target, over recurrent time steps. Recurrent minus feedforward accuracy is +6.4 pp (mono) and +11.1 pp (stereo) at 4 occluders. | Small recurrent CNNs, synthetic digits | full text; per-step values not tabulated | The same d_rel computed **across blocks** (depth) and **across frames** for occluded tool tokens |
| Aleatoric localization variance rises with occlusion | Pearson r between occlusion level and predicted width variance up to **0.499**, height variance up to **0.445** (within a prior). **No** epistemic correlation is reported. | YOLOv3, EuroCity Persons | full text | Use a variance-type output or spread, not only the peak confidence, as the PF noise input |
| Attention spread tracks localization quality | AS = det of the covariance of the top-K cross-attention locations. Median AS falls as IoU rises. **Not** tested against occlusion, and **not** fed to a tracker. | DETR-like 3D, nuScenes | full text | Spatial spread or stiffness of the token's trap |
| Feature norm is a confounder, not a signal | Norms vary within and across classes and correlate with the Mahalanobis score. Natural OOD often has **larger** norms, noise-like OOD smaller. L2 normalization before Mahalanobis improves average FPR 42.5% → 34.9% over 44 models. Little benefit for the well-behaved ViT-augreg. | ImageNet OOD | full text | Normalize residuals before any distance measure. Do not interpret "small norm" as "occluded". |

Sources by row:
1. [Naseer et al. 2021](https://arxiv.org/abs/2105.10497) [full text]
2. [Naseer et al. 2021](https://arxiv.org/abs/2105.10497)
3. [Naseer et al. 2021](https://arxiv.org/abs/2105.10497)
4. [Kassaw et al. 2024](https://doi.org/10.48550/arXiv.2409.10775) [full text]
5. [Kassaw et al. 2024](https://doi.org/10.48550/arXiv.2409.10775)
6. [Shastry et al. 2023](https://doi.org/10.1109/ICIP49359.2023.10222651)
7. [Wang et al. CVPR 2020](https://doi.org/10.1109/cvpr42600.2020.01266)
8. [Kortylewski et al. IJCV 2020](https://doi.org/10.1007/s11263-020-01401-3)
9. [Zhang et al. 2024](https://doi.org/10.48550/arXiv.2405.01311)
10. [Ernst et al. 2019](https://doi.org/10.1007/978-3-030-30508-6_24)
11. [Kraus & Dietmayer 2019](https://doi.org/10.1109/ITSC.2019.8917494)
12. [Ruppel et al. 2022](https://doi.org/10.48550/arXiv.2210.14391)
13. [Mueller & Hein 2025](https://doi.org/10.48550/arXiv.2505.18032)

Further findings:
- CompositionalNets classify artificially occluded PASCAL3D+ vehicles better than standard backbones at 60–80% occlusion (level L3): **+14.3 to +26.1 pp**. On real-occluded MS-COCO vehicles at L3 the gain is **+12.4 to +17.8 pp**. Source: [Kortylewski et al. 2020](https://doi.org/10.1007/s11263-020-01401-3) [full text].
- Semantic-part detection can be made robust by voting over "visual concepts", obtained by clustering internal network states, with log-likelihood-ratio tests. Source: [Wang et al. BMVC 2017](https://doi.org/10.5244/C.31.73) [abstract only].
- Naseer et al.'s feature-drop lesion study zeroes a fraction of a block's input features. For ViT-L with 25% dropped, accuracy stays between 73.32% and 76.28% at every block. This points to **redundancy** across features. Source: [Naseer et al. 2021](https://arxiv.org/abs/2105.10497) [full text].

### Inferences
- The user's "brake" (the last block pulling background toward the tool cluster), combined with confidence rising on empty frames, resembles the **context-dominance failure** documented for CompositionalNets. When the head learns to fire on "eye-center context", background tokens near the usual tool location gain tool-like evidence. This is a hypothesis to test (Q4), not an established fact for this model.
- For a hidden tip, the expected signature is a set of **tool-region tokens migrating from the tool cluster toward the background cloud**, with mixed tokens at the occluder boundary (the sub-patch contamination hypothesis). Whether the last block "repairs" these tokens by moving them back toward the tool cluster is the transformer analogue of Ernst et al.'s recurrent discounting, here run over depth instead of time.
- Distance-based signals must be computed on **normalized** residuals, with per-class tied covariance and shrinkage, because norms confound the Mahalanobis score.

### Gaps
- No paper found here measured token-level ViT signatures under **real** occlusion in **dense detection**. The quantitative ViT evidence is for classification with random patch drop.
- No study links occlusion to force-field or Jacobian quantities (trap stiffness, rotational or "vortex" components). These microscope quantities are novel and have no literature anchor.
- The claim that occluded objects have lower feature norms is **not supported** in this evidence set. Mueller & Hein explicitly reject a general "OOD has lower norm" rule.

---

## Q3. Using internal representations to improve detection and tracking: uncertainty for filters, prototypes, neuron and concept interventions

### Takeaway
Feeding detector-predicted uncertainty into a Kalman filter as measurement noise is established and gives moderate gains: ID switches −19% to −55%, HOTA +0.5 to +1.7, position error −1.3%. Feature-space distance (Mahalanobis) beats softmax confidence for flagging out-of-distribution inputs. Prototype or compositional models give large robustness gains under heavy occlusion, at some cost on clean data. Neuron- or SAE-level steering of ViTs is real but sparse: about 10–15% of units are steerable. **No paper found here shows neuron steering improving detection of occluded objects**, in surgical video or elsewhere.

### Cited Findings
- **Uncertainty into the Kalman filter.** UncertaintyTrack converts Prob-YOLOX box covariance into KF measurement noise, rejects detections with oversized confidence ellipses, relaxes boxes for association, and matches greedily by entropy. Results:

  | Dataset / tracker | HOTA | ID switches |
  |---|---|---|
  | BDD100K / ByteTrack | 38.4 → 40.1 | 46,732 → 37,682 |
  | MOT17 / OC-SORT | 67.0 → 67.5 | 2,946 → 1,322 |

  Failure cases involve occluded or small objects with large ellipses, but there is no occlusion-stratified result. Source: [Lee et al. 2024](https://doi.org/10.1109/ICRA57147.2024.10610458) [full text].
- Calibrated predictive variance used as KF measurement variance cuts tracking position error by about 1.3% on nuScenes. The predictive uncertainty "can be linked to object occlusion". Source: [Ulreich et al. 2024](https://doi.org/10.1109/SDF63218.2024.10773809) [abstract only].
- Aleatoric box variance from a single forward pass correlates with pedestrian occlusion (r ≤ 0.50). Monte Carlo dropout needs T=50 passes. Source: [Kraus & Dietmayer 2019](https://doi.org/10.1109/ITSC.2019.8917494) [full text].
- **Feature-space distance vs softmax.** Class-conditional Gaussians with tied covariance, scored by Mahalanobis distance, reach AUROC 99.1% vs 89.9% for maximum softmax (ResNet, CIFAR-10 vs SVHN). The raw score without preprocessing or the layer ensemble gives 93.92%. Source: [Lee et al. 2018](https://arxiv.org/abs/1807.03888) [full text].
- L2-normalizing features before Mahalanobis ("Mahalanobis++") wins in 41 of 44 ImageNet models. Average FPR falls 42.5% → 34.9%, AUROC rises 89.1% → 91.2%. Source: [Mueller & Hein 2025](https://doi.org/10.48550/arXiv.2505.18032) [full text].
- **Prototype or compositional detection.** Context-aware CompositionalNets: +41.2 AP at the heaviest vehicle occlusion, but −5.4 AP on unoccluded vehicles. Source: [Wang et al. 2020](https://doi.org/10.1109/cvpr42600.2020.01266) [full text].
- **Visibility as an explicit state.** PermaTrack's visibility head lets the tracker keep a hidden object's identity and location. Visibility thresholds (T_vis=0.05, T_occl=0.15) decide which labels are negative, ignored, or visible. Fine-tuning on real data alone made the model **forget** occlusion handling. Source: [Tokmakov et al. 2021](https://doi.org/10.1109/ICCV48922.2021.01068) [full text].
- **Neuron and SAE steering in ViTs.** About 10–15% of CLIP-ViT neurons and SAE features are "steerable". SAEs provide thousands more steerable features than the base model. Suppressing SAE features improved Waterbirds, CelebA and typographic-attack defense, best in middle layers. Source: [Joseph et al. 2025](https://doi.org/10.48550/arXiv.2504.08729) [abstract only; PDF read failed].
- SAE patch-level edits on a frozen ViT (layer 11), e.g. suppressing "sand", change segmentation outputs locally. The evidence is **qualitative only**, with no success rate, and the authors acknowledge manual feature selection and possible cherry-picking. Source: [Stevens et al. 2025](https://arxiv.org/abs/2502.06755) [full text].
- PatchSAE shows that most adaptation gains come from concepts already present in the non-adapted CLIP ViT. Source: [Lim et al. 2024](https://doi.org/10.48550/arXiv.2412.05276) [abstract only].
- **Surgical-specific.** No surgical paper was found that uses internal-representation uncertainty (Mahalanobis, prototypes, neuron activity) as a tracker input. The deep-search summary also flags this as the main gap: a "causal, calibrated link from ViT token/feature changes under real occlusion to detection and tracker updates" is missing. Source: [Undermind deep search 2](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Internal%20representation%20signatures%20of%20occlusion%20and%20uncertainty%20for%20detection%20and%20tracking).

### Inferences
- A PF can consume representation-level uncertainty in three ways that mirror the KF literature:
  1. **inflate the measurement noise** R_t, or temper the likelihood (w ∝ p(z|x)^β_t), from a per-detection reliability score;
  2. **gate** detections, analogous to UncertaintyTrack's ellipse filtering;
  3. add a **visibility state** to each particle (visible / occluded), analogous to PermaTrack, so that occluded hypotheses propagate by the motion model and are not killed by missing measurements.

  The KF literature suggests moderate gains, so effect sizes should be expected and reported as moderate.
- The microscope's novelty is to compute reliability from **where the token sits**: its distance to the tool cluster or background cloud, and the trap shape. The alternative is the head's confidence, which the user has already observed rising on empty frames. That is exactly the case where Mahalanobis-type scores beat softmax.
- Steering the 15–60 tool-selective neurons may help recall on occluded tools. The CompNet context result and the user's empty-frame confidence both predict an **FP cost**, and Joseph et al. imply only a fraction of the neurons will be causally effective.

### Gaps
- No direct evidence was found that neuron amplification or SAE steering improves occluded-object **detection**.
- No surgical-video study was found on prototype- or Mahalanobis-based uncertainty for instrument trackers.
- Joseph et al. could not be read in full, so the layer-wise and token-type details are not verified.

---

## Q4. What to measure in the microscope: candidate quantities for (a) an occlusion detector, (b) a PF uncertainty input, (c) a training signal; experiment plan; reviewer risks

### Takeaway
Use **paired clean/occluded frames** (synthetic tissue, iris, lens, blood or glare pasted over the tip) together with a small set of real tip-visibility labels. Then test whether five measures predict tip occlusion better than head confidence:
1. normalized-residual Mahalanobis log-likelihood ratio, tool cluster vs background cloud;
2. drift toward the background cloud across blocks;
3. activity of the tool-selective neurons;
4. the "brake" projection;
5. trap stiffness and rotation.

Then test whether these measures reduce PF error through occlusion episodes, and whether occlusion augmentation reshapes them. All of these quantities are hypotheses grounded in the analogies above. None of them has been validated in the literature.

### Cited Findings (anchors for the plan)
- Paired clean/corrupted recording isolates the effect of corruption. Synthetic occluders rank models similarly to real occlusion, though not identically. Sources: [Ding et al. 2024](https://doi.org/10.48550/arXiv.2407.11906); [Kassaw et al. 2024](https://doi.org/10.48550/arXiv.2409.10775) [full text].
- Visibility labels are ambiguous: annotators disagreed on 18% of test frames, especially at low contrast and with tips close to tissue. Source: [Aspart et al. 2021](https://doi.org/10.1007/s11548-021-02441-x) [full text].
- The relative-distance metric d_rel for "moving toward the target vs the occluder" comes from [Ernst et al. 2019](https://doi.org/10.1007/978-3-030-30508-6_24) [full text]. The per-position object/occluder log-likelihood ratio comes from [Kortylewski et al. 2020](https://doi.org/10.1007/s11263-020-01401-3) [full text]. Feature normalization before Mahalanobis comes from [Mueller & Hein 2025](https://doi.org/10.48550/arXiv.2505.18032) [full text].
- Baselines to beat:
  - softmax or heatmap confidence: [Lee et al. 2018](https://arxiv.org/abs/1807.03888)
  - aleatoric variance: [Kraus & Dietmayer 2019](https://doi.org/10.1109/ITSC.2019.8917494)
  - attention spread: [Ruppel et al. 2022](https://doi.org/10.48550/arXiv.2210.14391)
  - uncertainty-as-noise trackers: [Lee et al. 2024](https://doi.org/10.1109/ICRA57147.2024.10610458)
  - occlusion augmentation: [Shao 2025](https://doi.org/10.54254/2755-2721/2025.ld30738); [Agethen et al. 2025](https://doi.org/10.1007/s11548-025-03436-8)

### Inferences

#### Mapping microscope quantities to the three uses (hypotheses)

| Microscope quantity | Operational definition (proposal) | (a) Occlusion detector | (b) PF uncertainty input | (c) Training signal / augmentation target | Literature analogue |
|---|---|---|---|---|---|
| Token position relative to the tool cluster | LLR_i = log N(r̂_i; μ_tool, Σ) − log N(r̂_i; μ_bg, Σ), with r̂ the L2-normalized residual and Σ tied with shrinkage. Fit on a held-out train split. | Covered tip tokens should show LLR falling toward the background side, while visible shaft tokens stay positive. Use the per-box fraction of tokens with LLR below 0 as an occlusion score. | Aggregate the LLR over a detection to set the likelihood temperature β or the noise R | Consistency loss pulling occluded-twin tokens toward their clean-twin positions (feature completion) | CompNets LLR, FeatComp++, Mahalanobis(++) |
| Distance to the background 95% radius | Normalized Mahalanobis radius of each token in the background cloud | Tokens of the tool box re-entering the radius signal occlusion. On empty frames, background tokens leaving the radius signal a false positive. | Gating: down-weight measurements whose supporting tokens sit inside the background radius | Penalize background tokens that leave the radius on empty frames | Mahalanobis OOD |
| Drift across blocks (d_rel over depth) | d_rel(l) = 2 d_tool(l) / (d_tool(l) + d_bg(l)) for each tool-mask token, per block l | Occluded tokens start near background (d_rel > 1). A "repairing" network lowers d_rel in later blocks. | The final-block d_rel as reliability | Reward a decrease in d_rel across blocks on occluded twins | Ernst et al. recurrent discounting |
| "Brake" component | Projection of the last-block MLP update Δr_i onto the unit vector (μ_tool − μ_bg) | Compare tool tokens (occluded vs visible) with background tokens on empty frames | A large brake on background at the predicted location suggests a context-driven false positive, so inflate R | Regularize the brake on empty frames; keep it on occluded tool tokens if it helps them | Context dominance (CA-CompNets) |
| Tool-selective neuron activity | Sum of the activations of the k tool-selective neurons (fixed set, defined on held-out data) per token | Dose-response curve against occlusion fraction. Check whether activity drops before the heatmap peak does (early warning). | A low activity at the detection used as an uncertainty feature | Amplify or ablate the neurons (causal test). Add a loss keeping their activity on occluded twins. | Steerable neurons and SAE features |
| Trap stiffness and vortex character | Local Jacobian J = ∂F/∂r of the last-block update around the token, in a PCA subspace. Stiffness = −mean Re(λ). Vortex index = ‖antisymmetric part‖ / ‖symmetric part‖. Saddle = mixed-sign Re(λ). | Hypothesis: occluded tool tokens sit in shallower or saddle-like traps | Stiffness as an inverse variance, i.e. a stiffer trap gives a smaller R | Compare trap shape before and after occlusion augmentation | Attention spread vs IoU (Ruppel), but there is **no direct analogue** |
| Head outputs (baseline) | Heatmap peak, box IoU, ECE | Baseline detector | Baseline R(conf) | — | Softmax baseline |

#### Proposed experiment plan (small, feasible)

**Step 0. Data**
- **Synthetic paired set.** Take about 300–500 frames with tool masks or boxes. Candidate sources: the user's own data, Cataract-1K masks, or Cataract-LMM tip keypoints, which are available under CC-BY 4.0.
- For each frame, build occluded twins:
  - occluder textures: tissue, iris or lens patches cut from the **same frame** away from the tool; semi-transparent red ellipses for blood (alpha 0.2–0.6); white specular blobs;
  - placement: at the **tip** or on the **shaft**;
  - occluded fraction of the tool mask: 0, 10–30, 30–50, 50–70%;
  - size: at least one level **smaller than a patch**, to test the sub-patch contamination hypothesis.
- **Real set.** Label about 200 frames for tip visibility (visible / partial / hidden / uncertain), with 2 annotators, and report Cohen's κ.
- **Empty-frame set.** Use frames with no instrument.

**Step 1. Descriptive microscope run on the existing model.** No retraining. Use epochs 40, 100, 200, 300, 400 and at least 3 seeds, not only 3047. Record M1–M5 (the five measures in the mapping table) for:
- tool tokens: clean vs occluded twin, split into covered tokens, boundary tokens and visible-shaft tokens;
- background tokens: on tool frames vs on empty frames.

**Step 2. Occlusion detector test.**
- Compute AUROC and AUPRC of each microscope scalar for "tip hidden or partial vs visible", first on the synthetic labels and then on the real labels.
- Baselines:
  - heatmap peak confidence;
  - MC-dropout or a 3-seed ensemble variance;
  - attention spread (top-K attention covariance of the tool tokens).
- Success criterion (suggested): a microscope score beats confidence by ≥0.05 AUROC on the **real** set, with a bootstrap confidence interval that excludes 0.

**Step 3. PF integration.** Replay sequences that contain occlusion episodes, using Cataract-LMM tracking clips or the user's own videos, and compare:
1. fixed R;
2. R from confidence;
3. R from the microscope score;
4. a visibility-state PF, where particles carry v ∈ {visible, occluded} and the visibility likelihood comes from the microscope score;
5. an oracle visibility.

Metrics:
- tip RMSE during and after occlusion;
- track-loss rate;
- re-acquisition latency;
- ID switches;
- false track births on empty frames.

Expectation: a gain of a few percent to tens of percent in ID switches or losses, consistent with KF literature effect sizes, and largest in occlusion episodes.

**Step 4. Causal interventions on the frozen model.**
- On occluded twins, amplify the tool-selective neurons (×1.5, ×2) or ablate them.
- Controls: count-matched random neurons, and count-matched "silent" neurons.
- Remove the brake by projecting out the Δr component along μ_tool − μ_bg on background tokens.
- Measure IoU and recall on occluded twins, and false positives and confidence on empty frames.
- Expected trade-off: amplification raises occluded recall but also empty-frame false positives. Removing the brake lowers empty-frame confidence. If it also lowers occluded-tool recall, the brake doubles as an amodal "completion" mechanism.

**Step 5. Training-signal test.** Retrain with a short schedule, on the same seeds, for these arms:
1. baseline;
2. tissue/blood copy-paste occlusion augmentation;
3. arm 2 plus a twin-consistency loss on token residuals (occluded token toward its clean twin);
4. arm 2 plus a per-token visibility head;
5. optionally, an empty-frame penalty on brake magnitude.

Evaluate IoU on clean and occluded data, false-positive rate on empty frames, calibration (ECE), and PF metrics. Re-run the microscope. Prediction: augmentation lowers d_rel for occluded tokens, widens the margin from the background cloud, and increases the number of tool-selective neurons.

**Controls (apply throughout)**
- C1: an occluder of the same size pasted on **background away from the tool** (cf. Naseer's non-salient PatchDrop).
- C2: a neutral gray or noise box of the same size, to separate information removal from tissue-like distractor evidence.
- C3: a global corruption of matched strength (blur, brightness), to separate occlusion from generic shift.
- C4: random-neuron and silent-neuron controls for every neuron-level claim.
- C5: at least 3 seeds and fixed test sets. Use paired statistics (Wilcoxon) across frames, and mixed models across seeds and epochs.
- C6: neuron selectivity and cluster statistics defined on a split **disjoint** from evaluation.
- C7: the empty-frame set, to separate "context firing" from occlusion.

**Expected outcomes and how to read them**
- If covered tip tokens fall into the background cloud (LLR < 0) and a token-fraction score predicts real tip visibility better than confidence, the microscope yields a usable **occlusion detector**.
- If the same score improves PF tracking through occlusion more than confidence-based R does, it is a **better uncertainty input**.
- If occlusion augmentation improves occluded IoU while restoring trap margin and the activity of selective neurons, the microscope quantities can serve as **training diagnostics or targets**.
- If the brake is mainly active on empty frames and drives false positives, regularizing it targets the observed confidence drift.

#### Reviewer risks
1. **Single seed and epoch confounds.** The silent-neuron vs IoU relation on the plateau is one model, and epochs are not independent samples. Use at least 3 seeds and report confidence intervals.
2. **Synthetic vs real occlusion.** Rankings agree only approximately (Kassaw et al.). A real-label validation set is required.
3. **Label ambiguity.** About 18% annotator disagreement is plausible. Report κ and use an "uncertain" class.
4. **Circularity.** Tool-selective neurons, the clusters and the 95% radius are defined from GT masks. They must be estimated on disjoint data.
5. **Metric confounds.** Mahalanobis on unnormalized 768-dimensional features is distorted by feature norms (Mueller & Hein), and covariance estimation needs shrinkage or PCA because tool tokens are few.
6. **Metaphor risk.** "Lasers", "traps" and "vortices" need operational definitions (Jacobian eigen-structure). They must also show added predictive value **beyond** simple distances and confidence, or a reviewer will call them decorative.
7. **Off-manifold interventions.** Amplifying neurons may push tokens off-distribution. Report effects on clean and empty frames as well.
8. **Calibration vs occlusion.** Confidence rising on empty frames may be generic overconfidence, not occlusion-related. Report ECE and empty-frame false positives separately.
9. **Domain shift.** Cross-center drops in Cataract-LMM (70.45 → 47.12 mAP) can exceed occlusion effects. Keep the center or video split fixed.
10. **Expected effect sizes.** Uncertainty-aware filtering yields modest gains in the literature (−1.3% position error; HOTA +0.5 to +1.7). Over-claiming will be caught.
11. **Patch-size limits.** On a 20x30 grid, a thin tip may occupy a single mixed token. Report token-level results stratified by occluder size relative to the patch.

### Gaps
- None of the microscope quantities (trap stiffness, vortex index, brake, tool-selective neuron activity) has a literature precedent as an occlusion or uncertainty signal. All mappings above are proposals.
- No cataract-specific benchmark exists for tracking through occlusion. The plan relies on synthetic twins and a small self-labelled set.
- Joseph et al. (neuron/SAE steerability) and several abstract-only sources (Ulreich, Shastry, ToolH-YOLO, FeatComp++, VL-SurgPT, ROBUST-MIPS, CaDIS) were not verified in full text.
