[Pru20]:
## Title: “tracing,” not “tracking”

The supplied PDF’s title is **“Estimating Training Data Influence by Tracing Gradient Descent.”** Its first page identifies the paper as arXiv:2002.08484v3 and as a NeurIPS 2020 paper; the displayed title uses **Tracing**. “TracIn” is the method’s name, not “TrackIn.” (PDF p. 1.)

## What the gradient dot product measures

For plain SGD on a single example \(z_t\), the update is \(w_{t+1}-w_t=-\eta_t\nabla\ell(w_t,z_t)\). A first-order expansion of the test example’s loss then gives

\[
\ell(w_t,z')-\ell(w_{t+1},z')
\approx
\eta_t\,\nabla\ell(w_t,z')\cdot\nabla\ell(w_t,z_t).
\]

Thus, when the training example’s gradient points in a direction that also reduces the test example’s loss, the dot product is positive. TracIn sums these contributions over the iterations when that training example was used. With minibatches, the paper assigns each batch member \(1/b\) of the batch’s first-order contribution. (Section 3.2, PDF p. 3.)

In the paper’s terminology, positive influence makes an example a **proponent**—it reduces the test loss—and negative influence makes it an **opponent**. The practical checkpoint approximation, TracInCP, sums corresponding gradient dot products at saved checkpoints (Eq. 1, Section 3.3, PDF p. 4).

## The Adam caveat

The gradient-dot-product formula above follows directly for **SGD**. More generally, the first-order loss change is determined by the test gradient dotted with the **actual parameter update**. For Adam, that update depends on adaptive scaling and optimizer state, so the plain dot product between the test and training loss gradients is not, by itself, the exact first-order contribution of Adam’s update.

The authors explicitly say that the parameter-change formula should be modified for optimizers such as Adam, while the first-order approximation remains valid for small steps (Section 3.2, PDF p. 3). They also say they expect practical TracInCP to remain similar across training variations (Remark 3.4, PDF p. 4). That remark is a heuristic expectation; it does not derive the SGD gradient-dot-product expression as an exact Adam attribution.

## Retrospective attribution versus prospective utility

**Retrospective attribution** is TracIn’s primary interpretation: given a training run, it attributes changes in a specified test example’s loss to training examples along that run’s realized trajectory. In the idealized definition, the contribution is summed over the iterations in which an example was actually used; Lemma 3.1 says the influences over all training examples sum to the total test-loss reduction from the start to the end of training (Section 3.1, PDF p. 2).

**Prospective utility** would instead ask how useful a candidate would be if selected or added for future training—potentially accounting for how including it would alter subsequent training and the resulting model. TracInCP’s counterfactual interpretation can score an example not in the training set as though it had been visited at the saved checkpoints, but the checkpoints remain those produced by the existing run (Remark 3.5, PDF p. 4). So this is a fixed-trajectory proxy, not a guarantee of the candidate’s benefit after retraining or changing the future trajectory.

## Limits for full token targets

The paper does **not** evaluate token-level or full-sequence generation targets. Its text application is DBPedia **text classification**: it predicts one of 14 ontology classes, and computes gradients with respect to the last fully connected layer (Section 5.2, PDF pp. 8–9).

The framework is defined around the loss for a specified test example. In principle, a token or sequence loss could serve as that target, but the attribution would depend on how the target is defined: a sequence-level loss would give an aggregate attribution, while token-specific explanations would require separate token-position targets. The paper does not establish how to interpret or scale such full-token attributions, nor does it report results for them. It also notes that practical use requires choosing suitable checkpoints, layers, and loss heads, and interpreting the results with human judgment (Section 6, PDF p. 9).

---

[Car21b]:
## What the original paper establishes

The 2021 paper studies **DINO**, not DINOv2. For its ImageNet k-NN evaluation, it uses a **global image representation**: Appendix F.1 specifies the ViT’s `[CLS]` output as the representation, and classification is by weighted voting among the nearest stored training-image features. Table 2 reports ImageNet top-1 results, including 78.3% for ViT-S/8. This shows that those global features work well for **image-level class prediction**; it does not establish that nearest embeddings belong to the same object instance or correspond to the same ground-truth bounding box.

Table 2 compares ImageNet linear and k-NN classification across methods and architectures; DINO’s strong k-NN result is an image-classification result, not a box-matching or detection result.  
[REGION: page=5, bbox=71,477,639,914]

The paper also presents **object-related spatial cues**, but those are different evidence. Section 4.2.2 probes last-layer self-attention and thresholds attention maps into masks; Figure 4 reports mask overlap with PASCAL VOC ground-truth segmentations. Section 4.2.1 evaluates patch tokens on DAVIS video segmentation using nearest neighbors between consecutive frames (Table 5). These results concern **segmentation masks or frame-to-frame feature matching**, not whether nearest patch/image embeddings predict the same GT box.

Figure 4 shows attention-thresholded masks for supervised and DINO ViTs and reports their Jaccard similarity to VOC segmentation masks; it supports a claim about segmentation cues, not bounding-box correspondence.  
[REGION: page=8, bbox=72,60,435,492]

Table 5 reports DAVIS video-segmentation scores for frozen features, including DINO variants; the described procedure matches features between consecutive frames, rather than evaluating GT-box identity.  
[REGION: page=7, bbox=69,52,381,498]

## What it does **not** establish

- **Same-GT-box prediction from nearest embeddings:** The paper does not test whether nearest embeddings—global or patch-level—map to the same annotated box, object instance, or box identity. Its k-NN result votes for image classes; its dense results use attention-derived masks or video-frame correspondences.
- **Utility of a proposed gradient update:** DINO training does use gradients to train the student from a teacher’s outputs (Section 3.1, Eq. 2–3). But the paper does not test using nearest-embedding matches to choose, guide, or validate a downstream gradient update, nor show that such an update improves detection.
- **Surgical detection:** The experiments are not on surgical imagery and do not evaluate surgical object detection, bounding-box metrics, or a surgical workflow. The paper therefore provides no direct evidence for those claims.

## DINO original versus DINOv2

The findings above belong to **the original DINO method and experiments in Caron et al. (2021)**. They should not be reported as experiments on **DINOv2**, which is a distinct, later model. If a local pipeline uses DINOv2, this paper may motivate hypotheses about global features or attention cues, but it does not validate that local model, its patch embeddings, box matches, gradient-update strategy, or surgical-detection performance.

---

[Kil21]:
**No—not as a drop-in replacement with the paper’s guarantees.** GradMatch is a *data-subset selection method*, not a replacement optimizer: it selects a weighted subset, and training still proceeds on that subset (Algorithm 1, Section 3).

At the current parameters \(\theta_t\), it chooses subset \(X_t\) and weights \(w_t\) to make the weighted subset gradient,
\[
\sum_{i\in X_t} w_i \nabla_\theta L_T^i(\theta_t),
\]
close to a target gradient \(\nabla_\theta L(\theta_t)\). The target \(L\) can be the full training loss \(L_T\), or the validation loss \(L_V\); the paper uses validation gradients for its class-imbalance setting and generally training gradients otherwise (Sections 2–3; Eq. (1)–(3)). GradMatch-PB applies the matching to mini-batch gradients instead.

The guarantees are conditional. Theorem 1 (Section 2) gives convergence bounds for full gradient descent and the paper extends them to SGD in Appendix B.2; the bounds include an additive term for gradient-matching error. They assume conditions such as bounded parameters and Lipschitz or smooth losses, with convexity used in the stated results. The paper’s experiments use SGD, not AdamW (Section 5; Appendix C.2). Thus those results **do not establish an AdamW convergence guarantee**, nor a guarantee for SD-DINO.

Raw-gradient matching also does not ensure matching AdamW’s parameter updates: those depend on optimizer state (moment estimates), adaptive preconditioning, the learning-rate schedule, and decoupled weight decay. To use GradMatch with an AdamW-based SD-DINO loop, one could use it as a subset selector while retaining AdamW, but that is an adaptation—not a directly validated replacement. A guarantee about the resulting updates would require matching the relevant optimizer-transformed updates and analyzing that setup; the paper does not do this. The paper also does not evaluate SD-DINO.

The reported efficiency results support GradMatch as a promising selector in the paper’s tested settings, not as evidence for SD-DINO: for example, GradMatch-PB-WARM reports about \(7\times\), \(4.2\times\), and \(3\times\) speedups on CIFAR-10 at 10%, 20%, and 30% subsets, respectively, with reported accuracy drops of 2.8%, 1.5%, and 0.9% (Section 5; Figure 1).

Figure 3 compares speedup or energy ratio against relative test error across the tested datasets and methods; it summarizes the empirical efficiency–accuracy trade-offs relevant here, but does not test AdamW or SD-DINO.  
[REGION: page=7, bbox=67,69,621,909]

---

[Bal23b]:
## What the negative result establishes

The paper’s result is specifically about **Selective Backprop**: after a forward pass over a minibatch, choose a smaller subset for backpropagation. Its gradient-matching method selects a weighted subset intended to match the minibatch’s **mean last-layer gradient**, using orthogonal matching pursuit (Section 4, Eq. 1). It is not a general evaluation of dataset curation or all forms of data selection.

In the tested settings, gradient matching generally matched or beat loss-based selection, but **neither consistently beat selecting examples uniformly at random** at the same subsampling fraction (Section 5.1). Importantly, gradient matching *did* produce lower gradient-estimation error than random subsampling in the measured comparison, while loss-based sampling could increase that error (Section 5.3). Thus, a better proxy-gradient estimate did not translate into a consistent test-accuracy advantage.

The comparisons included five model–dataset combinations, with results averaged over three seeds. The main results use a fixed forward-pass batch size: methods take the same number of update steps but have smaller effective backpropagation batches as the subsampling fraction decreases. The appendix also reports a scaled-batch-size setting, where effective batch size is held constant but there are fewer steps; the authors report no substantial change to the comparison (Section 5; Appendix C.2). They additionally tune learning rates and report similar findings (Appendix C.4).

**Compute matters to the claim.** Selective Backprop makes an initial selection forward pass and then a separate forward–backward pass on selected examples; the paper estimates cost proportional to \(M/3+m\), versus \(M\) for full-batch backpropagation, so savings require \(m<2M/3\) (Section 2). The gradient-matching selector adds a small overhead relative to loss-based Selective Backprop—reported as under 5% in the tested implementation—but this does not establish that it is better than random selection at equal end-to-end compute (Appendix B.3).

The practical constraint on claims is therefore: **show that a selection rule beats an appropriately compute-matched random baseline on the outcome that matters, not just that it selects “important” examples or improves a gradient proxy.** The paper does not show that selective backprop is never useful; its evidence is bounded to these tasks, models, methods, budgets, and reported test-accuracy/gradient-error measures.

Figure 1 is the main test-accuracy comparison across subsampling fractions: it shows gradient matching and loss-based selection failing to consistently surpass random selection.  
[REGION: page=5, bbox=72,155,484,849]

Figure 3 compares squared gradient-estimation errors: gradient matching has lower errors than random sampling in the displayed experiments, despite the lack of a consistent test-accuracy win.  
[REGION: page=6, bbox=310,156,535,844]

Figure 2 shows the 10%-label-noise results; the paper reports that loss-based selection suffers substantially in that setting.  
[REGION: page=6, bbox=72,155,323,845]

## What this says—and does not say—about AdamW utility

The paper **does include one AdamW experiment**: fine-tuning pretrained BERT on IMDB, using AdamW for three epochs, a constant learning rate of \(2\times10^{-5}\), and base batch size 32 (Appendix C.1). That is evidence for this particular selective-backprop comparison under one AdamW fine-tuning setup—not a general validation of AdamW-based data utility. The paper does not isolate whether the result depends on AdamW, nor test a range of AdamW tasks, architectures, or utility definitions.

If “AdamW utility” means that a data-utility score is intended to predict which examples actually help **AdamW training**, distinct tests remain necessary. In particular:

- Test the score’s predictions against **observed training outcomes** under AdamW, rather than treating gradient matching or loss as utility by itself.
- Compare selected data with random and relevant selection baselines at **matched compute and training budgets**, including optimizer and schedule controls.
- Evaluate whether the score transfers across AdamW tasks and settings, rather than relying on the single BERT–IMDB experiment.
- If the utility claim concerns a dataset or training run rather than a minibatch, evaluate that separately: this paper’s method selects within minibatches and does not establish dataset-level data-utility rankings.

Those are gaps implied by the paper’s scope, not AdamW utility tests reported by the authors. The AdamW IMDB results, including the learning-rate sensitivity comparison, appear in Figures 1 and 5.  
[REGION: page=5, bbox=72,155,484,849]  
[REGION: page=12, bbox=399,155,635,847]

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
