[Pro14]:
## Model and identification

The paper’s model is a **discrete-time, input–state relation**:
\[
x_{k+1}=Ax_k+Bu_k. \tag{14, p. 3}
\]
Here \(A\) describes the state’s unforced evolution and \(B\) maps the current input \(u_k\) to the next state. In the snapshot notation, the corresponding equation is
\[
X_0=AX+B\Upsilon, \tag{16, p. 3}
\]
where \(X\) contains current-state snapshots, \(X_0\) their one-step-shifted snapshots, and \(\Upsilon\) the inputs applied at those steps. Thus, the paper’s \(X_0\) (also written \(X'\) in the figure) is a shifted **data matrix**; Eq. 14, not a continuous-time derivative equation, is the state update.

There are two identification cases:

- **Known \(B\):** subtract the measured control contribution, \(X_0-B\Upsilon=AX\), then estimate \(A\) from the remaining state data (Eqs. 17–24, pp. 3–4).
- **Unknown \(B\):** fit \(A\) and \(B\) together by forming \(\Omega=[X;\Upsilon]\) and writing \(X_0=[A\ B]\Omega\) (Eq. 25, p. 4). The paper takes an SVD of \(\Omega\), partitions its left singular vectors into state and input blocks, and estimates reduced-order operators using a second SVD of \(X_0\) (Eqs. 26–34, p. 4; algorithm, pp. 4–5).

The algorithm therefore requires **recorded state and control snapshots**. For the unknown-\(B\) case, the paper says the truncation rank \(p\) of the augmented input matrix should be larger than the state-output truncation rank \(r\) (Sec. III-C, p. 4). More generally, separating \(A\) from \(B\) requires the observed state and input data to contain enough independent information: if the inputs are redundant with the state data or the relevant directions are missing, the joint fit cannot uniquely resolve their effects. The paper does **not** state a formal persistent-excitation condition or a precise minimum number/design of inputs. It does note that DMDc can use arbitrary input histories, in contrast with ERA’s impulse-response focus (Sec. V, pp. 7–9).

**Figure 1** summarizes the collection of state and control snapshots, their matrix construction, and the DMD-versus-DMDc reduction procedures. It is useful for seeing that DMDc explicitly includes the measured control sequence.  
[REGION: page=5, bbox=62,59,958,943]

## Known controls, feedback, and unobserved forcing

“Known \(B\)” in Sec. III-B means the input-to-state map \(B\) is already known or estimated; the control snapshots are still used to remove \(B\Upsilon\) before estimating \(A\) (Eqs. 17–24). In Sec. III-C, both maps are estimated from the state and input snapshots.

The paper also gives a **feedback example**: a proportional controller sets \(u_k=K[x_1]_k\), with \(K=-1\), to stabilize a system with an unstable open-loop eigenvalue (Eq. 37, p. 6). In that example, DMDc uses the control data and **known \(B\)** to recover the underlying \(A\) (Sec. IV-A, pp. 6). This illustrates that feedback-generated inputs can be included when they are measured; it does not mean that the input is independent of the state. As a mathematical consequence, if feedback makes the measured input completely determined by the state and there is no other variation in the data, estimating unknown \(A\) and \(B\) separately can be underdetermined. The paper does not give a specific feedback-identifiability condition.

DMDc’s separation also depends on observing the relevant controls. The paper says ordinary DMD can return incorrect dynamics when external control has been applied (Sec. III-B, p. 3), and presents DMDc as using both state and actuation snapshots. It does not provide a separate method for recovering unobserved forcing. Consequently, an unrecorded disturbance is not a measured input that DMDc can explicitly subtract or attribute to \(B\); the paper does not establish that the fitted \(A,B\) would separate it from the state dynamics.

## Why observed future inputs are not an autonomous forecast

Equation 14 predicts the next state **conditional on both** \(x_k\) and \(u_k\). To forecast beyond the last observed snapshot, one must also supply or predict the future input sequence—or specify a policy that generates it. Knowing historical inputs alone does not specify future optimizer choices or future feedback actions. If the input is set by a known feedback rule, that rule can be applied to make a closed-loop forecast; the DMDc equation by itself does not supply the rule.

Likewise, if an optimizer has its own evolving activation, memory, delays, or other internal dynamics, that is a **different or augmented system** from Eq. 14’s plant model. Eq. 14 treats \(u_k\) as the current input; it does not model how an optimizer generates that input or how an activation variable evolves. To represent such dynamics, they would need to be modeled separately or included among the state variables. The paper does not discuss optimizer-induced activation dynamics specifically.

**Figure 4** contrasts application regimes for modal-decomposition and system-identification methods by underlying rank and number of measurements. It provides rank-context, but is not a formal excitation criterion for identifying \(A\) and \(B\).  
[REGION: page=8, bbox=472,58,826,516]

---

[Los17]:
❌ Error: Could not get PDF for [Los17]. The PDF may not be available for download.

---

[Kil20]:
### Validation-aware subset selection

GLISTER’s exact target is to choose a training subset whose fitted model minimizes loss on a held-out validation set:
\[
\min_{S\subseteq U,\ |S|\le k} L_V\!\left(\arg\min_\theta L_T(\theta,S),V\right).
\]
The paper gives the corresponding log-likelihood maximization form as well. See **Problem Formulation, Eqs. (1)–(2), PDF p. 2**. This is a bilevel objective: the inner problem trains on the selected subset; the outer problem evaluates the resulting model on validation data (**Eq. (3), PDF p. 3**).

Because solving that objective exactly is generally impractical, GLISTER-ONLINE approximates the inner training step with **one gradient-ascent step**:
\[
\theta'=\theta+\eta\nabla_\theta LL_T(\theta,S),
\qquad
S^{t+1}=\arg\max_{S\subseteq U,\ |S|\le k}
LL_V\!\left(\theta+\eta\nabla_\theta LL_T(\theta,S),V\right).
\]
This is **Eq. (4), PDF p. 3**. In loss-minimization terms, the analogous step asks which subset’s training gradient is predicted to reduce validation loss most.

The Taylor approximation used to make greedy selection cheaper scores the effect of adding candidate \(e\) to the current subset \(S\):
\[
\widehat G_\theta(S\cup\{e\})
=LL_V(\theta^S,V)
+\eta\,\nabla_\theta LL_T(\theta,e)^\top
\nabla_\theta LL_V(\theta^S,V),
\]
where \(\theta^S=\theta+\eta\sum_{j\in S}\nabla_\theta LL_T(\theta,j)\). See **“Approximations,” PDF pp. 4–5**. Thus, locally, a candidate scores well when its training gradient aligns with the validation-set log-likelihood gradient. This is a **validation-utility criterion**, not simply a measure of representativeness or similarity. Also, the score is generally conditional on the current parameters and selected set; GLISTER’s greedy selection is not necessarily a single, fixed independent ranking of examples.

### Empirical limitations and qualifications

- **It depends on validation data.** The objective is explicitly tied to the held-out set. The authors note that a small validation set can invite overfitting, and propose facility-location or random-function regularization to mitigate that risk (**Eq. (5) and “Regularization with Other Functions,” PDF p. 5**). If validation data are unrepresentative of the deployment target, the selected subset is optimized for the wrong target.
- **Selection itself costs compute.** The paper says subset selection can exceed full-training cost without approximations, especially for large datasets; it introduces last-layer, Taylor, and \(r\)-greedy approximations to reduce that overhead (**“Approximations,” PDF pp. 4–5; Appendix Table 2, PDF p. 24**). Its timing claims therefore depend on the model, subset budget, and approximation settings.
- **The approximations trade off time and behavior.** The authors use \(L=20\) epochs between selection rounds and typically \(r\approx0.03k\) for smaller models. Their ablation reports that smaller \(r\) can be unstable, while larger \(r\) costs more computation; changing \(L\) likewise trades selection frequency against runtime and convergence (**PDF pp. 6 and 28**).
- **The convergence claims are conditional.** For example, the validation-loss reduction result assumes smoothness and that the selected-subset training gradient has nonnegative inner product with the validation gradient (**Theorem 2, PDF p. 5**). It is not an unconditional guarantee that any selected subset improves validation performance.
- **The reported evidence is empirical, not a DINO comparison.** Experiments include shallow networks, LeNet on MNIST, ResNet-18 on CIFAR-10, and several UCI/LIBSVM datasets. The paper reports speed/accuracy trade-offs and robustness experiments, but does not evaluate DINO-based selection (**experimental setup and results, PDF pp. 7–8**).

Figure 2 summarizes accuracy-versus-budget and timing results, including the noisy-label and class-imbalance settings; it provides the main empirical context for the claimed efficiency/accuracy trade-off.  
[REGION: page=4, bbox=51,66,361,934]

Figure 4 compares the effects of \(L\) and \(r\) on the DNA experiment, illustrating the runtime–accuracy trade-off in the selection approximations.  
[REGION: page=6, bbox=209,66,412,500]

### Relation to DINO similarity or oracle latent closeness

GLISTER ranks examples by their **estimated effect on a labeled held-out validation objective**, using the current model’s gradients. DINO similarity or an oracle notion of latent-space closeness instead ranks by representation-space proximity or similarity. Such a ranking could select representative or nearby points, but the paper does not show that it identifies examples whose training updates improve validation performance, nor does it compare GLISTER with DINO or an oracle latent-closeness selector.

The paper’s synthetic illustrations report that GLISTER tends to select points near decision boundaries, whereas CRAIG, KNN-submodular selection, and random selection tend to choose more representative points (**discussion of Figs. 11–14, PDF pp. 29–30**). That supports the distinction between validation-driven utility and representativeness in those experiments; it does not establish that GLISTER is universally better than latent-space similarity. The paper also illustrates a shifted-validation setting in which its validation-driven selection lowers validation and test loss relative to the shown baselines (**Figs. 15–16, PDF p. 30**).

**Paper:** Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, “GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning,” arXiv:2012.10630, version 4, June 11, 2021.

---

[Yin19]:
## DeepInversion objectives and statistics

The paper’s central point is that optimizing an image to make a teacher predict a chosen class is not enough: the image also needs to produce intermediate features with statistics like those of the teacher’s training data.

- **Class-conditioned inversion and image priors (Sec. 3.1, PDF p. 3, Eqs. 2–3).** Starting from random input \(\hat{x}\), the method minimizes a target-class loss plus an image regularizer:
  \[
  \min_{\hat{x}} L(\hat{x},y)+R(\hat{x}).
  \]
  The DeepDream-style prior is \(R_{\text{prior}}=\alpha_{\text{tv}}R_{\text{TV}}+\alpha_{\ell_2}R_{\ell_2}\), penalizing total variation and the image’s \(\ell_2\) norm. The authors say these priors help stabilize optimization, but alone still produce images far from the natural/training-image distribution.

- **Feature-distribution regularization (Sec. 3.2, PDF p. 3, Eq. 4).** DeepInversion adds penalties for the differences between synthesized-image and training-data feature-map means and variances, summed over convolutional layers:
  \[
  R_{\text{feature}}(\hat{x}) =
  \sum_l \|\mu_l(\hat{x})-\mathbb{E}[\mu_l(x)\mid X]\|_2
  +\sum_l \|\sigma_l^2(\hat{x})-\mathbb{E}[\sigma_l^2(x)\mid X]\|_2 .
  \]
  These are batch-wise feature statistics; the paper assumes feature statistics are adequately described by means and variances.

- **Where the statistics come from (Sec. 3.2, PDF p. 4, Eqs. 5–6).** Rather than needing training images to estimate those expectations, DeepInversion uses each BatchNorm layer’s stored running mean and running variance as estimates:
  \[
  \mathbb{E}[\mu_l(x)\mid X]\approx \mathrm{BN}_l(\text{running mean}),\qquad
  \mathbb{E}[\sigma_l^2(x)\mid X]\approx \mathrm{BN}_l(\text{running variance}).
  \]
  The resulting regularizer is \(R_{\text{DI}}=R_{\text{prior}}+\alpha_fR_{\text{feature}}\) (Eq. 7, PDF p. 4).

- **Adaptive DeepInversion (Sec. 3.3, PDF p. 4, Eqs. 8–9).** To increase diversity during student training, ADI adds \(R_{\text{compete}}=1-\mathrm{JS}(p_T(\hat{x}),p_S(\hat{x}))\) to the inversion regularizer. Minimizing this term encourages teacher–student disagreement—equivalently, larger Jensen–Shannon divergence—while retaining the teacher’s class-conditioned guidance. The total is \(R_{\text{ADI}}=R_{\text{DI}}+\alpha_cR_{\text{compete}}\).

## Evidence from knowledge transfer

The paper evaluates transfer by training a student against the teacher’s soft output distributions on the synthesized images, using a KL-divergence distillation loss (Sec. 3.1, PDF p. 3, Eq. 1). Thus, the images are not merely judged by whether the teacher assigns them a desired class: they are tested by whether they support learning a student.

On CIFAR-10, Table 1 reports student accuracies of **84.16%, 83.82%, and 91.43%** with DeepInversion across its three teacher–student settings; ADI raises these to **90.78%, 90.36%, and 93.26%**. The corresponding teachers reach 92.34%, 92.34%, and 95.42%. By comparison, optimized noise reaches about 13.5%, and DeepDream-style priors reach 29.98–39.67%. The paper also reports that ADI brings the students within 2 percentage points of the teacher in these experiments (Sec. 4.1, PDF p. 5).

On ImageNet, a student ResNet50v1.5 trained from scratch using **140,000 DeepInversion images** reaches **73.8% top-1 accuracy** after 250 epochs with mixup, compared with the teacher’s 77.26% (Table 6; Sec. 4.4, PDF p. 7). In the 90-epoch comparison, DeepInversion reaches 68.0%, versus 64.0% with BigGAN-generated images and 72.3% with MS COCO images. These results support the usefulness of the synthesized data for transfer, while also showing that it does not fully match real-data performance in every setting.

Table 1 compares student accuracy under noise, DeepDream, DeepInversion, and ADI; its results show the large transfer gain from feature-statistic regularization and the further gain from teacher–student competition.  
[REGION: page=5, bbox=70,60,282,490]

The CIFAR-10 image examples contrast noise and DeepDream outputs with DeepInversion and ADI outputs. They illustrate the paper’s qualitative claim that feature regularization produces more recognizable, class-related images, while ADI introduces different samples.  
[REGION: page=5, bbox=254,60,518,490]

Table 6 compares transfer from real ImageNet and proxy/generated sources with transfer from DeepInversion images. It is relevant because it reports student accuracy and training conditions, including the 73.8% result with 140k synthetic images and longer training plus mixup.  
[REGION: page=7, bbox=70,482,234,914]

## What the evidence says about confidence-only or arbitrary activation targets

The paper’s evidence favors **model-guided synthetic data constrained by feature-distribution statistics**, not class-confidence maximization alone. In the authors’ account, DeepDream optimizes a chosen class response while leaving intermediate representations unconstrained; its natural-image priors help, but do not make the resulting images distribution-matched. The CIFAR-10 comparisons support that distinction: DeepDream’s images yield substantially weaker student performance than DeepInversion’s (Table 1). On ImageNet, DeepDream-generated images are also much less transferable to other classifiers: for example, ResNet-18 classifies ResNet-50-generated images at 94.4% with DeepInversion versus 28.0% with DeepDream (Table 2, Sec. 4.2.1, PDF p. 6). The paper further describes DAFL’s generated examples as “unrecognizable,” and reports lower transfer than its method in the CIFAR-10 comparison.

The qualification is important: these experiments **do not establish that every confidence-only method or every hidden-activation-target objective fails**. The paper discusses DeepDream and DAFL as particular baselines; it does not report a systematic comparison against arbitrary choices of hidden-layer activation targets. DeepInversion itself uses class-conditioned teacher guidance, but supplements it with **aggregate, per-layer mean/variance constraints from BatchNorm**, rather than matching arbitrary exact activations. So the supported conclusion is narrower: in the tested settings, adding BN-derived feature-statistic constraints—and, for ADI, student–teacher competition—produced better images and better transfer than the paper’s tested alternatives.

Nor do the results show exact recovery of the original training examples. The method synthesizes class-conditional images that are useful for downstream learning. The authors note limitations including color/background similarity, possible unimodal behavior from Gaussian initialization, sensitivity of image style to optimization hyperparameters, and substantial synthesis time (Sec. 5, PDF pp. 9–10).

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
