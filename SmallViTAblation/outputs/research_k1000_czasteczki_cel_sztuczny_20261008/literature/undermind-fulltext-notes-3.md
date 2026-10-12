[Cao21b]:
## What the paper does

Cao and Johnson invert a **trained detector**: given a target layout—object categories and boxes, optionally instance masks—they optimize an image \(I\) so the detector returns that layout. The detector’s weights are held fixed; the optimization changes the image, not the model. The target layout supplies the labels and locations, so this is label-informed inversion, not unconditional image generation.

For a differentiable detector, the paper first gives the direct objective
\[
I^*=\arg\min_I \ell(F(I),\hat L)+R(I)
\tag{1}
\]
(Section 3.1). But detector post-processing, including confidence filtering and NMS, is non-differentiable. For single-stage detectors, the authors introduce pseudo-targets \(Z\) for the raw network outputs \(\Phi(I)\), then alternate between updating the image and updating \(Z\) (Eqs. 2–8, Section 3.2). The image update makes \(\Phi(I)\) match \(Z\); the pseudo-target update minimally adjusts raw outputs so post-processing yields the target layout. The differentiable matching loss combines classification cross-entropy and box-regression GIoU loss (Eq. 9). Two-stage detectors use analogous auxiliary variables for proposal and second-stage outputs (Section 3.3).

## Classification confidence versus box regression

The paper’s comparison is that the two heads rely on qualitatively different cues:

- **Classification** emphasizes category-distinctive, often local cues—such as animal heads or hair/texture.
- **Regression** appears to emphasize object structure, boundaries, and corners, with a more global sense of shape; it is less dependent on texture.

This interpretation is supported by loss-disentangled inversions and attribution visualizations in Section 4.3. In the individual-object experiment, the authors maximize one anchor’s class probability **without constraining its regressed box or other anchors**; this produces the category-specific images and accompanying contextual detections in Figures 8–9. The paper suggests the difference between classification and regression cues may matter for detector-head design, but does not establish a particular design change as beneficial.

The loss-disentanglement figure compares inversions using classification, regression, and mask outputs separately, illustrating that the heads can produce different visual evidence for the same target object.  
[REGION: page=6, bbox=72,61,299,909]

The attribution figures show the image regions identified for classification versus box regression in Mask R-CNN’s second stage; they are the paper’s direct visual evidence for the different cues described above.  
[REGION: page=6, bbox=268,60,608,914]

## Naturalness priors and non-uniqueness

The objective includes an image regularizer \(R(I)\), described as encouraging natural-looking images. The implementation uses periodically applied blurring, TV-norm, and \(p\)-norm regularization (Section 4, Optimization Details). These are visualization regularizers, not a learned natural-image generator or a guarantee that an inversion is a plausible photograph. The paper’s displayed inversions are often highly stylized.

The inverse problem is also underdetermined in the practical sense that the requested output is a detector-recognized layout, not a unique source image. **The paper does not formally analyze uniqueness or measure how many distinct images satisfy the same layout.** It does generate visualizations with different random seeds for its context-frequency analysis (Section 4.4.1), but that is not a systematic study of the solution set for a fixed full layout.

Figure 4 shows examples of synthetic inversions for several detector architectures. The key point for this question is that the images are optimized to reproduce the specified detections, not to recover a unique original scene.  
[REGION: page=5, bbox=70,60,459,917]

## Does detector recognition show the images are useful training data?

**No—not by itself, and this paper does not demonstrate downstream training utility.** Its strongest relevant evidence is *inversion transfer*: the authors generate images using one detector, run them through other detectors, and report COCO-style AP (Section 4.2, Table 1). For example, images inverted with Faster R-CNN obtain 94.8 AP when evaluated by Faster R-CNN, and 74.6 AP with RetinaNet. This supports the claim that the images encode cues recognized by the generating detector—and, to a degree, by other detectors. It does not show that adding these images to a training set improves a detector, generalizes to real data, or provides useful augmentation. The paper reports no such retraining or downstream-utility experiment.

Table 1 gives the inversion-transfer AP values across RetinaNet, Faster R-CNN, Mask R-CNN variants, and FCOS; it is evidence of detector recognition/transfer, rather than a test of training-data value.  
[REGION: page=5, bbox=485,61,722,491]

**Closest evidence overall:** this is label/layout-informed optimization through a fixed trained detector, with regularization and measurable detector recognition. It is not evidence that the resulting synthetic images are useful training examples.

**Source:** Ang Cao and Justin Johnson, “Inverting and Understanding Object Detectors,” arXiv:2106.13933v1, 26 June 2021.

---

[Mak23]:
## The illusion is not “patching a nullspace direction works”

For post-GELU activations \(h\) in an MLP with output projection \(W_{\text{out}}\), an **exact** perturbation \(\delta h\in\ker W_{\text{out}}\) has no effect on that MLP’s output:
\[
W_{\text{out}}(h+\delta h)=W_{\text{out}}h.
\]
So patching *only* along a direction entirely in the fc2/down-projection kernel cannot change downstream computation through that MLP. The paper says this explicitly in §3.2 and defines such a subspace as causally disconnected in §3.5.

The illusion instead arises when the **patched direction is a mixture** of a disconnected direction and a dormant—but causally effective—one. For orthogonal unit vectors \(v_{\text{disconnected}}\) and \(v_{\text{dormant}}\), the paper considers
\[
v_{\text{illusory}}=\frac{v_{\text{disconnected}}+v_{\text{dormant}}}{\sqrt 2}.
\]
The examples’ activations can differ along \(v_{\text{disconnected}}\), while being nearly constant along \(v_{\text{dormant}}\). Patching along the *mixture* uses the difference in the disconnected projection to set the patch’s size, but the update also moves activation along \(v_{\text{dormant}}\). After multiplication by \(W_{\text{out}}\), the disconnected part vanishes, but the dormant part does not: Equation (3) gives an output change proportional to
\[
\big(v_{\text{disconnected}}^\top(h_A-h_B)\big)\,W_{\text{out}}v_{\text{dormant}}.
\]
Thus the observed effect is **not** a nonzero downstream effect of an exact kernel perturbation. It is a nonzero effect of a *mixed* patch whose causal component is dormant on ordinary data and whose patch magnitude is supplied by the disconnected component.

The toy network in §3.3 makes this concrete. Its hidden activation is \(h=(x,0,x)\), and its output is \(y=x\). The \(h_1\) direction is disconnected; \(h_2\) is dormant (normally zero, but it can affect the output); and \(h_3\) faithfully carries the input to the output. Patching the direction \((1,1,0)/\sqrt2\), which mixes \(h_1\) and \(h_2\), changes the output to the source example’s \(x'\)—just as patching \(h_3\) would—even though that mixed direction need not be the model’s ordinary route for computing the output. See Equations (1)–(3) and the toy example in §3.3.

**A related “parallel path” interpretation:** the paper says the intervention can activate a dormant pathway that ordinarily contributes little or nothing on the data distribution. In its IOI example, the established route carries position information from S-Inhibition heads through the residual stream to Name Mover heads; patching the layer-8 MLP subspace can add an alternate, normally dormant route to the same behavior. This is not evidence by itself that the MLP normally mediates that feature.

## What success does—and does not—guarantee

A successful end-to-end patch shows that the intervention can alter behavior in the desired direction. **It does not, by itself, guarantee** that the patched subspace is the model’s faithful representation of the feature, that the model normally uses that subspace, or that the feature has been localized to that component. The authors explicitly question the hoped-for “if and only if” relationship between patch success and faithful representation (§3.2). The mechanism can depend on taking activations off-distribution and recruiting a dormant pathway.

Their IOI results illustrate the distinction. For the DAS-found MLP8 direction, patching gives **46.7% FLDD**, but patching only its causally relevant rowspace component gives **13.5%**, and patching its exact nullspace component gives **0%**; patching the whole MLP gives **−8%**. The nullspace component is therefore not, by itself, producing a downstream effect. Rather, the original mixed direction’s result relies heavily on the disconnected component, while its small causal component can act through a dormant route (Table 1; §4.4). In factual recall, the authors report the analogous pattern: DAS patches can change the recalled fact, while the rowspace-only patch and full-MLP patch largely lose that effect (§6.1).

## The paper’s success control

The counterexample is the IOI **residual-stream** direction \(v_{\text{resid}}\), alongside the attention-gradient direction \(v_{\text{grad}}\) (§5). These patches remain strong when the authors remove their proxy causally disconnected component: \(v_{\text{resid}}\) gives 140.7% FLDD versus 127.5% for its rowspace component; \(v_{\text{grad}}\) gives 111.5% versus 106.47%. Both directions distinguish ABB from BAB examples (Figure 6), and \(v_{\text{resid}}\) is substantially aligned with \(v_{\text{grad}}\) (cosine similarity about 0.78; §5.1). The authors also report that S-Inhibition heads, rather than MLP8, write the relevant residual-stream direction (Appendix C, Figure 20). They take this combination of causal, representational, and circuit-level checks as evidence of a faithful direction—not as a guarantee from patch success alone. Their broader recommendation is to favor activation bottlenecks such as the residual stream and to validate a proposed feature’s role beyond end-to-end patching (§8).

The figures below are useful visual companions: Figure 1 depicts the mixed disconnected/dormant-direction geometry; Figure 3 maps the IOI circuit and marks the MLP and residual-stream interventions; Table 1 gives the patching controls and their metrics; Figure 5 compares how strongly the nullspace and rowspace components discriminate the IOI patterns.

[REGION: page=3, bbox=163,96,836,905]

[REGION: page=12, bbox=78,479,714,911]

[REGION: page=14, bbox=142,450,670,899]

[REGION: page=16, bbox=444,95,818,908]

---

[Bae22b]:
## What the paper distinguishes

- **Leave-one-out (LOO) retraining** asks what model results from training without a data point. The paper’s *cold-start* version restarts from the original initialization; its *warm-start* version starts from the trained parameters. In neural networks these can reach different solutions, so neither is automatically what an influence estimate predicts (Section 4.1).
- **Influence functions (IFs)** are a first-order approximation to a local response to changing a point’s weight (Eq. 4). With damping and estimates taken at possibly non-converged parameters, they need not approximate cold-start LOO retraining.
- **The proximal Bregman response function (PBRF)** instead asks how to change the model in response to a point while keeping predictions on the existing training data close to those of the current model. It uses a Bregman-divergence anchoring term plus a parameter-proximity term (Eq. 8, Section 4.3). The paper shows that the damped, Gauss–Newton IF is the first-order approximation to a *linearized* PBRF (Eq. 49, Appendix B.4).
- **Approximation failure has distinct sources:** warm-start, proximity, and non-convergence *gaps* reflect differences in the questions being asked; linearization and solver errors are approximation errors. In their experiments, the first three generally dominate (Section 4; Figure 4).

## Specifying an online candidate update

The paper studies *removing/downweighting* a training example, not adding a new online candidate. Thus, it does not prescribe or empirically validate an online-addition rule. A defensible extension is to state clearly whether the candidate is being **added** or **removed**, and whether the desired target is retraining or a PBRF-style response.

For **adding** a candidate \(z_*=(x_*,t_*)\) to a current model \(\theta^s\), one possible PBRF-style objective is

\[
\min_\theta\;
\frac{1}{N}\sum_{i=1}^{N}
D_{L^{(i)}}\!\left(f(\theta,x^{(i)}),f(\theta^s,x^{(i)})\right)
+\alpha L(f(\theta,x_*),t_*)
+\frac{\lambda}{2}\|\theta-\theta^s\|^2 .
\]

Here \(\alpha\) must be specified according to the intended weighting—for example, \(\alpha=1/(N+1)\) if using the new point’s weight under a dataset average, with the existing-data term weighted consistently. Linearizing this objective at \(\theta^s\) gives the proposed one-step update

\[
\Delta\theta
= -\alpha (G_s+\lambda I)^{-1}
\nabla_\theta L(f(\theta^s,x_*),t_*),
\qquad
\theta_{\text{new}}=\theta^s+\Delta\theta,
\]

where \(G_s\) is the corresponding Gauss–Newton curvature of the anchoring term. This is a **PBRF-style extension**, not a rule proposed or tested in the paper. It is also not the same as a one-step SGD update: it is a curvature-preconditioned, linearized response. For **removing** an existing point, use the paper’s downweighting convention and sign: Eq. 10 gives a positive \(\epsilon\)-scaled inverse-curvature times the point’s loss gradient.

## Empirical verification

Do not verify only that the update is fast or that it changes predictions plausibly. For the same held-out set of candidate points, compare:

1. the proposed one-step update;
2. optimization of the corresponding *nonlinear* PBRF-style objective;
3. actual retraining on the modified dataset, using clearly identified cold- and/or warm-start protocols.

Report per-candidate prediction differences on a fixed evaluation set, as well as changes in held-out loss; assess both magnitude error and candidate ranking (the paper uses output \(L_2\) distance and Pearson/Spearman correlations). If the claim is that the update approximates retraining, retraining is the relevant comparator. If the claim is that it approximates the local influence-style response, compare it to the nonlinear PBRF-style objective too. Sequentially comparing nonlinear PBRF, linearized PBRF, and the computed IF-style update helps separate linearization error from solver error, as in the paper’s decomposition.

The paper’s findings also suggest useful stress tests: training duration (non-convergence), damping (solver/proximity trade-offs), and the size or fraction of the candidate change (greater nonlinearity can increase linearization error). The paper reports that increasing width reduced linearization error, while greater training time reduced the non-convergence gap (Section 6.2).

**Relevant visual evidence:** Figure 1 contrasts IF with LOO for logistic regression and an MLP, and IF with PBRF for the MLP. It illustrates why the target of the comparison matters: IF aligns poorly with MLP LOO but well with MLP PBRF.  
[REGION: page=2, bbox=96,87,397,885]

Figure 4 decomposes output discrepancies across architectures and datasets; the warm-start, proximity, and non-convergence components are generally larger than linearization and solver components.  
[REGION: page=10, bbox=100,88,351,885]

Figure 5 summarizes ablations over width, depth, training time, regularization, damping, and fraction removed—useful factors for stress-testing the proposed online update.  
[REGION: page=11, bbox=94,88,417,886]

**Bottom line:** specify the online update as an explicitly weighted, signed objective perturbation and identify whether its target is retraining or a PBRF-style local response. The paper supports the latter interpretation for its deletion setting; it does not establish that an addition update follows the same empirical alignment.

---

[Chu22c]:
**No.** DPS does not guarantee that a target has a preimage under the forward operator—whether that operator contains post-GELU activations or is another nonlinear map. The paper does not specifically analyze GELU. It says the method can be applied to nonlinear operators when their gradients can be obtained, and demonstrates that with Fourier phase retrieval and non-uniform deblurring; differentiability enables guidance, but does not imply that the operator is surjective or that every target is attainable.

**What DPS does:** The exact time-dependent likelihood \(p(y\mid x_t)\) is intractable (Section 2.2). DPS approximates it by replacing the expectation over clean images with the likelihood evaluated at the Tweedie estimate \(\hat{x}_0=\mathbb E[x_0\mid x_t]\) (Eqs. 7–11). It then guides reverse diffusion using the gradient of a measurement loss through \(A(\hat{x}_0)\): a squared residual for Gaussian noise, or a weighted residual for the paper’s Poisson approximation (Eqs. 16, 20–22; Algorithms 1–2). For nonlinear \(A\), that gradient can be obtained by backpropagation.

This is **soft guidance, not a feasibility or convergence guarantee**. DPS does not project each iterate onto the set \(A(x)=y\). Theorem 1 (Eq. 14) bounds the likelihood-approximation error under its stated Gaussian-noise assumptions; it is not a guarantee that the approximation is exact, that an inverse exists, or that sampling finds one. The paper also reports phase-retrieval failures associated with non-uniqueness and stochastic samples (Section C.6).

**Feasibility and prior are separate:** feasibility asks whether any \(x\) satisfies \(A(x)=y\); the generative prior favors \(x\)’s that resemble data represented by the diffusion model. A target can be feasible but have no plausible solution under that prior, or be infeasible while still allowing a noisy-data fit. With a noisy likelihood, DPS can seek a compromise between fitting the measurement and following the learned prior; it does not establish an exact preimage. For an arbitrary post-GELU target, therefore, the answer depends on whether that target lies in the activation map’s range—and the paper supplies no general result guaranteeing that it does.

Figure 3 contrasts projection-based guidance with DPS: the authors’ geometry shows DPS avoiding a hard measurement-subspace projection, which they argue helps with noisy measurements. It illustrates the method’s guidance strategy, not a guarantee of target feasibility.  
[REGION: page=6, bbox=87,156,341,846]

Figure 6 shows example DPS reconstructions for two nonlinear problems—phase retrieval and non-uniform deblurring. These examples demonstrate applications, not a guarantee for arbitrary nonlinear operators or targets.  
[REGION: page=9, bbox=87,190,307,814]

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
