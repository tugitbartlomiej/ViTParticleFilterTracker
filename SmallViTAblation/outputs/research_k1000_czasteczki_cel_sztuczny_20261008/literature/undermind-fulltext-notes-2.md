[Gra25c]:
## What the paper establishes

**“Divergence” means that an intervention produces an activation unlike naturally occurring activations—not necessarily that it changes the model’s output.** In Section 3, the authors show theoretically that coordinate patching can leave a class manifold, except for axis-aligned hyperrectangles (Section 3.1; Appendix A.2). Empirically, they compare natural and intervened activations for mean-difference patching, SAE reconstruction, and DAS (Section 3.2). They visualize the activations with PCA and quantify distributional differences using Earth Mover’s Distance (EMD), compared against a natural-to-natural baseline. Appendix A.1 also describes paired distances and local-manifold/density measures. PCA is illustrative; the paper’s point is that the divergence is present in the full-dimensional distributions too.

**“Harmless” is relative to a particular computation or mechanistic claim.** The paper defines the behavioral null-space of a function \(\psi\) over inputs \(X\) as
\[
N(\psi,X)=\{v:\forall x\in X,\ \psi(x+v)=\psi(x)\}
\]
(Section 4.1, Eq. 4). A change in this null-space does not affect that function’s output on \(X\). But it may still affect internal subcomputations, so it can be harmless to a claim about the whole computation and harmful to a finer-grained claim. The behaviorally binary example (Eqs. 5–7) illustrates another benign case: an activation can be atypical yet preserve the relevant sign and therefore the output.

**“Pernicious” divergence can make an intervention’s apparent causal evidence misleading.** In Section 4.2, the authors describe off-manifold activations that recruit hidden pathways—computations inactive on natural inputs—or prime dormant behavioral changes that appear only in other contexts. Their mean-difference-patching example (Section 4.2.1) changes a ReLU activation pattern and flips the output through a unit that is silent for natural class-A inputs. Section 4.2.2 illustrates a change that appears benign in one context but changes behavior in another. The paper emphasizes that a behavioral success on the tested examples alone does not rule these cases out.

**The Counterfactual Latent (CL) loss encourages intervened states to resemble natural states with the desired causal-variable values.** The original loss combines an L2 term and a cosine term between the intervened activation and a CL target constructed from natural examples (Section 5.1, Eq. 8); it is added to the behavioral DAS loss with weight \(\epsilon\). In the Boundless DAS experiment, increasing the CL weight reduced divergence while maintaining similar interchange-intervention accuracy (IIA). The modified loss applies this idea to causal subspaces (Section 5.2, Eqs. 9–10); in the authors’ synthetic tasks it reduced causal-feature EMD and improved OOD intervention performance. This is evidence in the paper’s DAS/LLM and synthetic settings—not a validated recipe for arbitrary models. The authors explicitly note that their modified loss is limited to simple settings and is not specific to pernicious divergence (Section 6).

**For your frozen vision model, these are concepts and proposed tests, not direct evidence.** The paper does not study a frozen vision model with a 768-to-192 final projection and a tokenwise head, nor does it establish that CL will preserve semantic steering in that architecture. If the head receives only \(Ph\), where \(P:\mathbb R^{768}\to\mathbb R^{192}\), then a perturbation \(\delta\in\ker(P)\) leaves the projected representation unchanged: \(P(h+\delta)=Ph\). That makes the kernel a null-space for the projection output, but not automatically for every claim about upstream computations. If there are bypasses or other inputs to the head, test the full downstream function rather than assuming \(P\) is the whole computation.

## Minimal controls that could falsify a semantic-steering claim

These are suggested tests for your setup, not controls specifically evaluated in the paper.

1. **Projection/kernel control.** Hook the intervention at the claimed layer and measure its effect after the 768-to-192 projection and at the head’s output. Compare the proposed steering direction with a matched perturbation in \(\ker(P)\). If a purported projection-mediated effect survives a true kernel-only perturbation, the stated mechanism is wrong or the intervention is reaching another pathway. Conversely, if the claimed steering direction has no effect at the projected representation or head output, it does not steer through that path.

2. **Matched sham-direction control.** Compare the proposed semantic direction with random or unrelated directions matched for norm (and, where possible, layer and token position). Evaluate the intended semantic change on held-out images and with a semantic measure independent of the same head used to construct or score the intervention. If matched sham directions produce the same effect, or the effect fails on held-out examples or independent semantic evaluation, that undermines the claim of *semantic-specific* steering.

3. **Natural-manifold and pathway control.** Compare intervened activations with natural activations for the intended target semantics, using a natural-to-natural baseline; inspect distances in the relevant activation space and, if feasible, local-manifold or density measures. Test whether the behavioral result survives projection toward the target-semantic natural manifold, and check whether the intervention creates new activation patterns or ReLU gates. If the effect disappears after manifold projection, or depends on newly activated pathways absent from natural examples, the result does not support the claim that it reveals a natural semantic mechanism. It may still show that an artificial perturbation can change the head’s output.

4. **Token/context control.** Because the head is tokenwise, evaluate the effect across token positions and relevant contexts—not just the selected token and tested examples. A result that reverses or produces unintended changes at other positions or contexts is evidence against a broad semantic-steering claim. The paper cautions that exhaustive context testing is infeasible, so passing a finite test cannot prove the absence of dormant behavior.

In short, a **head-score change alone** is insufficient evidence for semantic steering. The stronger claim is weakened or falsified if the effect is not direction-specific, does not generalize to independent semantic evaluation, fails at the stated projection pathway, or depends on clearly off-manifold activations or hidden pathways.

The paper’s visuals may help orient the controls: Figure 1 contrasts harmless and pernicious divergence; Figure 2 shows divergence across several intervention methods; Figure 3 shows the CL-loss results and synthetic OOD comparison.

Figure 1 is a conceptual diagram of natural versus intervention-induced pathways and of harmless versus pernicious regions in a two-dimensional representation. It frames why behavior can look confirmatory even when an intervention recruits a hidden pathway.  
[REGION: page=2, bbox=83,156,531,845]

Figure 2 shows PCA views of natural and intervened activations for coordinate patching, mean-difference patching, SAE reconstruction, and Boundless DAS, plus EMD comparisons. It supports the claim that divergence is observed across methods, while the plots themselves are not a test of whether the divergence is harmful.  
[REGION: page=5, bbox=81,155,410,846]

Figure 3 shows the CL-loss results: reduced divergence in the Boundless DAS setting and, in synthetic tasks, tighter intervened representations and better OOD performance. It is relevant as evidence for the proposed mitigation, but not as evidence that it transfers directly to a frozen vision model.  
[REGION: page=10, bbox=82,155,444,847]

---

[Gei23c]:
**What the paper shows.** In Section 3, the authors test whether feature visualizations for Inception-V1’s last-layer class units activate similar layer representations to natural images of the corresponding class. Their sanity check compares three activation-similarity baselines across the network: natural images from the same class, natural images from different classes, and natural images versus same-class feature visualizations. Similarity is measured primarily with Spearman correlation and normalized so the same-class natural-image comparison is 1 and the different-class comparison is 0. Through roughly the first two-thirds of the network, natural images and their class’s visualizations are often about as similar as natural images and unrelated-class visualizations; similarity rises in the final third. The authors therefore conclude that these optimized visualizations are processed differently from natural images through most of the network. This is an activation-similarity proxy for different processing, not a direct causal trace of every pathway. (Section 3; Appendix C.3.)

Figure 5 plots normalized activation similarity across Inception-V1 layers: the natural-versus-visualization curve stays near the different-class baseline for much of the network and rises toward the same-class baseline only in later layers. It is the paper’s central empirical evidence for the claim.  
[REGION: page=6, bbox=65,69,392,906]

The authors also test whether ordinary distribution shift alone explains the result. Using ImageNet-V2 and NINCO as comparison datasets, they report that same-class natural-image similarity remains substantially higher than natural-versus-feature-visualization similarity through most layers. They conclude that these simpler distribution shifts do not account for the observed difference. (Appendix G.)  
Figure 15 (ImageNet-V2) and Figure 17 (NINCO) show normalized Spearman similarity across layers; in both, the natural-versus-visualization curve remains low through much of the network relative to the same-class natural-image baseline.  
[REGION: page=34, bbox=68,74,316,900]  
[REGION: page=34, bbox=673,73,922,900]

**Natural examples and the sanity check.** Natural images from the same class provide the paper’s upper-bound comparison: they should share class-relevant features and, empirically, have more similar activations than images from different classes. The paper also notes that *highly activating natural samples* would pass this particular sanity check by definition, because they are natural inputs that activate the unit. Appendix A.3 reports that prior human studies found highly activating natural images more interpretable than feature visualizations; Appendix B.7 discusses how those studies relate to the paper’s analysis. But this is not a general reliability guarantee: the authors say natural samples can themselves be manipulated, and their sanity check is **necessary, not sufficient**. For example, a visualization that shows cows on grass could pass while concealing that a unit responds to grass rather than cows. Their practical recommendation is to use visualizations exploratorily, not confirmatorily, and combine them with other evidence, including dataset samples (Section 5; Appendix J).

The Figure 5 comparison identifies the same-class natural-image and different-class natural-image baselines, and places feature visualizations against them. This makes clear both what the check can flag and why passing it is not proof of trustworthiness.  
[REGION: page=6, bbox=65,69,392,906]

**Does this extend to Stable Diffusion inversion of arbitrary frozen-detector hidden targets? Not as an empirical claim.** The paper studies activation-maximization feature visualizations, chiefly Lucent visualizations of last-layer units in Inception-V1; its silent-unit manipulation also demonstrates fooling on ResNet-50. It does **not** study Stable Diffusion, diffusion-based inversion, arbitrary hidden-layer targets, or frozen detectors in general.

There is a **conditional theoretical connection**: if an inversion method is used to find an input that maximizes a scalar hidden-unit activation, the paper’s central identifiability concern is relevant—the maximizing input (even together with the minimum and their values in the paper’s formalization) does not, in general, determine how that unit behaves on other inputs. Table 1 reports no general reliability guarantee for black-box functions or neural-network units without stronger assumptions. But that theorem is about information supplied by extrema, not a result about diffusion priors or inversion algorithms. An inversion objective that matches a feature vector or reconstructs an image may not even be the same formal setup. The paper therefore supports caution and a proposed test, not the conclusion that SD inversions necessarily follow different pathways.

A reasonable **adaptation of the sanity check**, rather than a reported result, would compare inversion outputs’ activations across detector layers with those of relevant natural examples and unrelated controls. For arbitrary hidden targets, however, the paper’s class-based baselines may not be available; choosing a meaningful natural-example comparison is itself a limitation. Passing such a check would still not establish reliability. **Theory pointer:** Table 1 summarizes which function classes and predictions the authors can or cannot guarantee from a maximally activating visualization.  
[REGION: page=8, bbox=64,69,393,907]

---

[Hao20]:
## What LabelEnc does

LabelEnc is **not an oracle that edits a detector’s final MLP or its weights**. It learns a mapping from ground-truth labels to a feature-space target, then uses that target as an auxiliary training signal for the detector backbone.

1. **Represent each image’s labels as a tensor.** For COCO, the paper uses an 80-channel, image-sized map: each object fills its class channel over its box, with values decreasing from 1 at the center to 0.5 at the box boundary. The label-encoding network \(h(y;\psi)\) maps that map into a latent representation (Sec. 3.4).

2. **Learn the label-to-feature mapping in a joint first stage.** The authors motivate \(h\) as an approximation to the inverse of a detection head: ideally, a feature that the head maps back to the ground-truth label (Eqs. 3–6). Since that inverse and the optimal head are not directly available, they train a label-space AutoEncoder using a detection head, while also training an auxiliary image backbone. The joint objective in Eq. 9 has:
   - \(L1\): label reconstruction through the detection head;
   - \(L2\): ordinary detection loss on image features;
   - \(L3\): distance between image-backbone features and \(h(y;\psi)\).

   The detection head is shared by \(L1\) and \(L2\); importantly, they block \(L3\)’s gradient into \(h\). The paper reports a 30k-iteration warmup without \(L3\), because including it from the start led to a degenerate solution (Secs. 3.2, 4.1).

3. **Use the learned representation to update the actual detector.** In stage two, the authors train the detector with its ordinary detection loss plus a feature-distance loss between the image backbone and the fixed label-derived representation \(h(y;\psi^*)\) (Eq. 7, Fig. 2). The auxiliary loss updates the detector backbone; \(h\) is fixed. They reinitialize the backbone, but initialize the detection head from the head learned in stage one. After training, they remove the auxiliary structures, so inference has no added cost (Sec. 3.3). Their distance term includes a feature-adaptation network and layer normalization to handle feature-map sizes and domain differences (Eq. 10, Sec. 3.4).

**The important distinction:** the labels do not directly prescribe edited detector weights. They produce a target representation; a feature loss then supplies gradients to the detector during training.

## Evidence and limits

The main COCO results show gains across one-stage and two-stage detectors. For example, Table 1 reports RetinaNet/ResNet-50 at 36.4 to 38.6 mmAP, FCOS/ResNet-50 at 38.7 to 41.1, and FPN/ResNet-50 at 36.8 to 38.6. The gains remain against the reported 2×-schedule baselines, though the LabelEnc pipeline itself uses two training stages. The authors say the stage-one label encoder can be reused across backbones if the detection head is the same (Table 3, Sec. 4.2–4.3).

Ablations support the role of the method’s components: using only label reconstruction in stage one falls short of the full method (ResNet-50: 37.6 vs. 38.6; Table 2), and in stage two both intermediate supervision and head initialization contribute (Table 5). The paper also reports gains on Mask R-CNN mask prediction (Table 7).

These are empirical results on the paper’s stated COCO setup; they do **not** establish that LabelEnc is a true or unique inverse of the detector head, nor do they test an independently supplied “oracle” target or direct editing of a frozen final MLP. The paper’s inverse-head argument is a motivation for the learned encoding, not a demonstrated recovery of a mathematically defined inverse. The authors also report that stage one alone helps but does not match the two-stage method (Table 4).

The stage-one diagram shows the label encoder and image backbone feeding a shared detection head, with reconstruction, detection, and feature-distance losses. It is useful for seeing how the label-derived representation is learned rather than supplied by an oracle.  
[REGION: page=7, bbox=125,199,480,810]

The stage-two diagram shows the fixed label encoder supplying the auxiliary target while the image backbone and detection head are trained; the label encoder is removed afterward.  
[REGION: page=8, bbox=187,199,540,809]

The main-results table compares LabelEnc with 1× and 2× baselines across RetinaNet, FCOS, and FPN backbones; its reported COCO mmAP values are the central evidence for the performance claim.  
[REGION: page=11, bbox=125,249,359,757]

The ablation table compares the full stage-one training against using only reconstruction loss. It shows that the simpler AutoEncoder-only variant underperforms the full procedure.  
[REGION: page=12, bbox=124,246,337,762]

## Comparison with “independent oracle editing” and whether LabelEnc is minimal

That comparison is **not addressed in the paper**, and “oracle editing a frozen final MLP” could mean more than one thing. In the paper’s method, the label encoder is *learned* using the joint stage-one objective, and the actual detector is then trained with a feature loss and a stage-one-initialized head. It is therefore different from an oracle that directly edits a frozen head or supplies its own desired feature representation.

If by “frozen final MLP” you mean frozen **weights** but gradients still allowed through the MLP to its input, a loss on its output could still update upstream features. But that is not the specific LabelEnc mechanism or experiment: LabelEnc compares backbone features with a label-derived target, and trains the detector head in stage two. If you mean an oracle directly changes the MLP’s weights, that is a different intervention again.

So LabelEnc can be viewed as a **training-only alternative to needing a large external teacher**, but not as a validated minimal substitute for oracle editing. A stripped-down version—fixed label-to-feature mapping plus an auxiliary feature loss—would be conceptually simpler, but the paper does not show that this simplification retains the reported results. Its ablations instead indicate that the joint stage-one training and the stage-two choices matter. The paper also does not measure the cost of stage one separately; it notes that the full pipeline doubles training iterations, with potential amortization when reusing the encoder.

---

[Gen24]:
## What the paper actually evaluates

This is a **classification task-adaptation study**, not an object-detection study. The authors fine-tune a LAION-2B-pretrained **CLIP ViT-B/16** image encoder, using cross-entropy classification, then evaluate zero-shot (ZS) and linear-probe (LP) accuracy. The five benchmarks are ImageNet-1K, DTD, FGVC-Aircraft, StanfordCars, and Flowers102 (Section 4).

For the synthetic arm, they use **Stable Diffusion 1.5**, trained on LAION-2B, to generate class-targeted images from Llama-2 7B–written prompts; they retain the top 30% per class using CLIP similarity filtering (Sections 3.1, 3.3, Appendix C.1–C.2). For the real-data arm, they retrieve targeted images directly from **LAION-2B**, using either caption substring matching or CLIP semantic k-NN, then apply the same top-30% CLIP filtering and remove test-set duplicates (Sections 3.2–3.3). The retrieval method is selected per benchmark using validation accuracy.

## Image and compute budgets

The experiments sweep adaptation-set sizes over roughly an order of magnitude. Figure 2’s size axes range from **16K to 4,096K images**, depending on benchmark; the synthetic sweeps sometimes extend beyond the finite pool of targeted retrieved images. The authors give one especially clear matched comparison for FGVC-Aircraft: the full retrieved set contains **139K post-filter images**. At 139K images, synthetic-data fine-tuning raises LP accuracy from 64.9% to 68.7% (+3.8 points), while retrieved-data fine-tuning raises it to 82.7% (+17.8 points). Scaling synthetic data to 250K raises its LP accuracy to 70.7%; the paper also reports that about **500K synthetic images** are needed to match the LP accuracy of just **15K retrieved aircraft images** (Section 4.1). For ImageNet, the main text notes that retrieval can improve ZS accuracy in a setting where synthetic training can hurt it; Figure 2 shows the broader benchmark-by-benchmark scaling comparison.

These are **image-count comparisons**, not a reported equal-compute comparison. The paper estimates that generating **1M synthetic images before filtering takes about 12 hours on 64 NVIDIA A100 GPUs** (Appendix C.1). Retrieval uses precomputed LAION-2B image embeddings; the authors report computing LAION caption embeddings and their search index as separate infrastructure costs (Appendix D.1), but do not provide an end-to-end, matched retrieval-versus-generation compute budget.

The retrieval pool is finite and task-dependent. For retrieval, they use k=500 for ImageNet and k=2,000 for the other benchmarks, with a per-class cap of 10,000 images; the 139K aircraft count is the explicitly stated post-filter total (Appendix D.1; Section 4.1).

The main scaling figure is useful for seeing the comparison across all five classification benchmarks: synthetic and retrieved fine-tuning are plotted against adaptation-set size for both ZS and LP accuracy. **The central result is that synthetic data is matched or outperformed by retrieved data, including where synthetic performance falls below the off-the-shelf CLIP baseline.**  
[REGION: page=5, bbox=61,155,483,847]

## Where synthetic data did help

The paper does report useful synthetic-data gains over the **off-the-shelf** model in some settings:

- On FGVC-Aircraft, 139K synthetic images improved LP accuracy by 3.8 points (64.9% → 68.7%).
- On DTD, 110K synthetic images improved ZS accuracy by 3.3 points (56.3% → 59.6%).
- The abstract reports a best-case synthetic gain of up to 7.1 percentage points; its point is that these gains are still exceeded by the targeted-real-data baseline.
- On Aircraft, increasing synthetic data to 500K continued to improve LP performance, but with much less data efficiency than retrieval. At roughly similar LP accuracy (~72%), the paper contrasts 500K synthetic images with 15K retrieved images; their ZS accuracies differ substantially, 28.9% versus 39.5% (Sections 4.1 and the discussion preceding Section 5).

Thus, “synthetic helps” in this paper means **it can improve a pretrained classifier relative to using no targeted adaptation data**. It does not mean synthetic images beat the targeted retrieval baseline.

Figure 3 gives examples of the authors’ proposed reasons for the gap: synthetic images can show generator artifacts and can get class-relevant details wrong—such as aircraft engine configurations or the appearance of a flute.  
[REGION: page=7, bbox=485,155,936,849]

## Necessary retrieval-only baseline

The paper’s key methodological recommendation is to compare targeted synthetic training data against **targeted real images retrieved directly from the generator’s upstream training data**—not against a general, untargeted real dataset. In this study that means retrieving relevant images from LAION-2B, which is the upstream dataset used by Stable Diffusion, and evaluating under the same downstream task and training protocol. This is the **retrieval-only baseline** needed to test whether synthesis adds value beyond selecting relevant examples from the generator’s source data (Introduction; Sections 3–4; Discussion).

For a surgical object-detection study, a corresponding baseline would need to retrieve task-relevant real images from the **actual generator’s upstream data**, and compare against synthetic data under comparable labeling, filtering, data-size, and evaluation conditions. Detection also requires localization annotations; the paper’s retrieval method assigns image-level class labels and does not evaluate box annotation or detection metrics. If the upstream training set is unavailable, the paper recognizes that this retrieval baseline may be infeasible (Discussion).

## Scope and limitations—especially for surgical detection or DINO

The authors’ evidence is limited to **image classification**, CLIP, full fine-tuning, and Stable Diffusion 1.x models for which upstream data was available (Section 6, “Limitations”). They explicitly say they did not empirically test other pretrained backbones or adaptation methods; they also leave robustness and fairness comparisons to future work. The paper does **not** study surgical imagery, object detection, or DINO.

Accordingly, its finding does **not** establish that retrieved images will outperform generated images for surgical object detection, nor does it establish anything directly about DINO. Those settings differ in task, domain, labels, and model; the paper’s authors’ conjecture that results may generalize beyond CLIP is not an empirical result for those cases. The appropriate conclusion is narrower: **for their tested CLIP classification setup, targeted retrieval from LAION-2B consistently matched or outperformed targeted Stable Diffusion images.**

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
