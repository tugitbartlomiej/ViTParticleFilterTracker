# Opposing MLP-neuron pairs, neuron copies, condensation and weight decay: literature check for claims C1–C4

Date: 2026-10-21. Builds on (does not repeat) `SmallViTAblation/docs/2026-10-20_literatura_przeciwstawne_neurony/` (README.md, PRACE_PELNE_TEKSTY.md) and `SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-20_literatura_przeciwstawne_neurony_w_uczeniu.md`.

Sources: Undermind workspace 40294fa6-c9a3-4469-9a61-d3eddeaee893 (five existing deep searches on opposing neurons and per-token cancellation were inspected, plus two new deep searches launched today), and Scite (metadata and citation statements).
- New deep search 1: [Hidden neurons merging into near duplicate copies during training… (122 papers)](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Hidden%20neurons%20merging%20into%20near%20duplicate%20copies%20during%20training%20condensation%20and%20symmetry%20induced%20neuron%20splitting)
- New deep search 2: [Weight decay implicit bias on how a function is split across hidden neurons… (121 papers)](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Weight%20decay%20implicit%20bias%20on%20how%20a%20function%20is%20split%20across%20hidden%20neurons%20same%20sign%20copies%20versus%20opposite%20sign%20pairs)

Reading-status tags used below:
- **[FT]**: full text read in this session (Undermind read_pdfs).
- **[FT-prev]**: full text read in the 2026-10-20 session; facts taken from the local notes listed above.
- **[ABS]**: abstract only.
- **[META]**: metadata or title only, with no abstract available.

Our setting is a ViT with a conv stem, 6 pre-LN blocks, d=192, MLP 768 with exact GELU, AdamW (lr 5e-4, wd 0.05) and seed 3047, tracked for 200 epochs. A "beam" is u = GELU(z)·W2[:,c].

---

## Q1. Per-claim verdicts: does the literature support, contradict, or not cover C1–C4?

### Takeaway

| Claim | Verdict | Closest published evidence | What the literature does NOT have (candidate novelty) |
|---|---|---|---|
| **C1**: same-token opposed pairs (≥150°) are absent at init, rise to a peak of 30 at epochs 80–110, decline to 15 at epoch 200, reach ≥170° only once and late, and turn over (5/24 survive from epoch 60 to 200) | **Not covered at the neuron level (likely novel)**. Partial analogues exist at the level of whole layers and in SNNs. | Pat25d: adjacent-layer opposition is absent early and emerges mid-pretraining in OLMo [FT-prev]. Zhu26l: push–pull inhibition grows over SNN training [FT-prev]. Che23an: geometry transitions in a toy model of superposition [FT-prev]. Vel26b and Xu25ad: neuron groupings reorganize over training [FT]. | A rise-then-decline time course of neuron-pair opposition, measured per token in the residual stream, with identity turnover, in any transformer or ViT. |
| **C2**: no anti-parallel W2 (max 142°); opposition comes from parallel W2 plus opposite GELU signs | **Not covered. It contradicts the channel/finite-difference mechanism as an explanation for opposition in this setting.** | Mar25l: channels need a_i·a_j<0 [FT]. Mah21e: the finite-difference construction uses output weights ±n [FT]. Kon25e: the negative GELU/SiLU region is functionally used in LLMs [ABS]. | That opposition between MLP neurons arises mainly from GELU sign mismatch on parallel output weights. |
| **C3**: opposed pairs are not Mar25l channels; late near-duplicate copies appear; hypothesis that weight decay favours same-sign splitting | **Partially supported.** (a) Absence of diverging channels is expected under weight decay. (b) Late near-duplicates are well supported. (c) "WD favours two same-sign halves" is not established: L2 penalizes cancellation, and WD favours identical over distinct neurons, but for ReLU a same-sign split is cost-neutral, and AdamW ≠ L2. | (a) Mar25l discussion [FT]; coercivity (Lia19d) [FT]. (b) Che23z, Ziy23c, Xu26ah, Doi21, Vel26b [FT]; Roy18b, Zha21ar [ABS]. (c) Sav19, Ziy23c, Jac22c [FT]; Xie24h [ABS]; Cas19e, a weak contrast [FT]. | The first direct test of channel signatures in a transformer. Mar25l call such large-scale quantification "ongoing work" [FT]. |
| **C4**: block-6 pair 370/391 merges between epochs 100 and 150 (input cos 0.31→0.96, output angle 72°→7°, norms grow, biases more negative, 91% of tokens in GELU's negative regime, activation correlation 0.03→0.80, adds on 94% and cancels on 6% of tokens, ablation ΔIoU ~1e-5) | **The qualitative pattern is supported; the specifics are not covered.** Norm growth during the merge partly contrasts with Xu26ah's compression-then-condensation ordering. | Che23z: late collapse onto the identical-neuron invariant set, in GELU VGG/ResNet with WD [FT]. Ziy23c: under WD, the midpoint of two neurons beats distinct neurons [FT]. Xu26ah: late angular condensation after fitting [FT]. Bre19: "permutation points" where the in/out weights of two neurons collide [ABS]. Cas19e, Doi21: redundant units/clones are removable [FT]. | A checkpoint-resolved merger of two named neurons in a ViT, with a GELU negative-tail regime and per-token add/cancel accounting. |

### Cited Findings
- **C1.** At layer level, the per-token cosine between the contributions of adjacent layers is ≈ −0.2 (down to −0.4) in 5 of 7 LLM families. In OLMo-7B there is no pattern at ~5% of training (step 33,000); it appears at ~21% (step ~135,500) and then grows [FT-prev] — [Patrawala et al., LLM Layers Immediately Correct Each Other, NeurIPS 38](https://doi.org/10.52202/085713-4643).
- **C1.** In trained spiking networks, the across/within inhibition ratio between oppositely modulated units grows from 1.06±0.15 to 2.0±0.7 over training. The motif disappears without Dale's law [FT-prev] — [Zhu et al., Biol. Cybern. 2026](https://doi.org/10.1007/s00422-025-01030-4).
- **C1.** In a toy model of superposition, SGD passes through k-gon configurations and loss plateaus end in sudden drops (400 trajectories). Antipodal pairs are not counted separately [FT-prev] — [Chen et al. 2023, arXiv 2310.06301](https://arxiv.org/abs/2310.06301).
- **C1/C4.** Over training of an MNIST MLP, the number of distinct "fibers" (groups of synchronized neurons) decreases. Covers can merge but cannot split under GD (Cover Coarse-Graining Theorem) [FT] — [Velarde, Parra, Hashemi, Makse 2026, arXiv 2609.01768](https://arxiv.org/abs/2609.01768).
- **C2.** A channel requires output weights of opposite sign (a_i·a_j<0) and converging input weights [FT] — [Martinelli, van Meegen, Şimşek, Gerstner, Brea 2025, arXiv 2506.14951](https://arxiv.org/abs/2506.14951).
- **C2.** The derivative-approximation construction is h_n(x) = n·ρ(x+1/n) − n·ρ(x): identical input weights, biases differing by 1/n, output weights +n and −n [FT] — [Mahan, King, Cloninger, Neural Networks 2021](https://doi.org/10.1016/j.neunet.2021.01.007).
- **C2.** Zeroing only the negative pre-activations of a small set of "Wasserstein neurons" in GELU/SiLU LLMs sharply degrades grammatical performance. This sign-specific ablation becomes more harmful over training as these neurons emerge [ABS] — [Kong, Ning, Adler, Shavit 2025, arXiv 2509.24198](https://arxiv.org/abs/2509.24198).
- **C3/C4.** Details and numbers for each paper are given in Q2–Q4 and Q7.

### Inferences
- C1 and C2 together describe the mechanism of opposition and its time course. Neither is anticipated at neuron granularity in transformers. C3 and C4 connect to an established theoretical literature (symmetry-induced invariant sets, collapse to identical neurons, weight-decay condensation) that has not been tested on ViT MLP neurons across checkpoints.
- C2 makes C1 partly a consequence of C3/C4. Near-copies with parallel W2 produce opposed beams only on tokens where their pre-activations straddle zero. So part of the C1 decline could be "pairs becoming copies" (opposition shrinks to a thin band near threshold), rather than opposition disappearing. This is our inference and needs a direct count of copy-like pairs among the C1 pairs.

### Gaps
- Kwon & Kim 2005, "Anti-pairing in learning of a neural network with redundant hidden units" (J. Phys. A 38:5627–5646): neither Undermind nor Scite has an abstract. The open-access PDF is at IOP and was not read. It could anticipate opposite-sign pairing in soft-committee machines and must be read before submission — [DOI](https://doi.org/10.1088/0305-4470/38/25/003) [META].

---

## Q2. Martinelli et al. 2025 (flat channels / GLU-like pairs): what exactly is predicted, under which conditions, and does weight decay or finite training remove channels? Follow-ups?

### Takeaway
Mar25l predict pairs of hidden units whose input weights converge while their **scalar** output weights diverge with opposite signs, so that the pair tends to a gated linear unit. This is shown only for small MLPs with scalar outputs, smooth activations (softplus, erf; ReLU in one appendix example), MSE loss, gradient flow and **no weight decay**. They state that regularization, like a finite step size, makes the dynamics "get stuck inside the channel". They do not test GELU, vector outputs, transformers or a weight-decay sweep. No citing paper tests channels in transformers or with weight decay.

### Cited Findings
- **Setup** [FT] — [Mar25l](https://arxiv.org/abs/2506.14951):
  - The main model is a single-hidden-layer, **scalar-output** network trained with MSE by full-batch **gradient flow**, integrated with an ODE solver (KenCarp58; max-norm threshold 1e3; at most 1 h of simulation).
  - Frequency experiments use softplus (Appendix B.3 also reports erf), hidden widths 2–16 and input dimensions 2–16, plus 2–3-layer networks with 4 neurons per layer: 4,420 networks with 50 initializations per configuration.
  - There is no GELU experiment. A ReLU channel example is in Appendix B.4 (Fig. B13).
- **Prediction** [FT] — [Mar25l](https://arxiv.org/abs/2506.14951):
  - The channel criteria are "diverging output weights" with a_i·a_j<0 and input-weight distance d(w_i,w_j)→0. The empirical detector requires parameter norm >1e3, cosine distance of input weights <1e-3, and >90% of updates in the channel subspace.
  - The pair sum tends to c·σ(w·x) + a(Δ·x)·σ'(w·x), a "central finite difference approximation". For softplus the gate is a sigmoid.
  - With a = ε(a_i − a_j) held fixed, the difference of the output weights grows like 1/ε. Loss and approximation error scale as ε² and sharpness as 1/ε² (Fig. 5); the slope along the channel scales as ~ε³.
  - The paper never says verbatim "the two contributions are large and nearly cancel". Cancellation follows from its parameterization.
- **Frequencies** [FT]: about 24% (modified Rosenbrock) and 20/30/61/64% (GP targets with s = 0.1/0.5/2/10); rougher targets give more channels — [Mar25l](https://arxiv.org/abs/2506.14951).
- **Finite step size** [FT]: SGD (lr 0.1, batch 16) and Adam (lr 1e-3) "easily" find the channel entrance in the toy example (2 erf neurons, about 50,000 epochs) but "quickly get stuck there", at the edge of stability where λ_max = 2/η — [Mar25l](https://arxiv.org/abs/2506.14951).
- **Regularization** [FT] — [Mar25l](https://arxiv.org/abs/2506.14951):
  - Section 4 says "a small weight regularization would prevent finding the optimal solution due to the flatness of the channel".
  - The Discussion says "Adding regularization has a similar effect as a finite step size: from a certain point the regularization outweighs the decrease in loss such that the dynamics get stuck inside the channel."
  - There is no L2/weight-decay ablation. The only penalty used is a cubic max-norm control term that activates only above the norm threshold (Appendix A.2).
- **Vector outputs** [FT]: not derived or tested. The paper does not claim that output-weight vectors would become anti-parallel — [Mar25l](https://arxiv.org/abs/2506.14951).
- **Relation to duplication** [FT]: duplicating a neuron with output weights γa and (1−γ)a (the Fukumizu–Amari construction) gives a symmetry-induced saddle line. Channels become **asymptotically parallel** to such saddle lines (Eq. 5) but are distinct objects — [Mar25l](https://arxiv.org/abs/2506.14951).
- **Scale** [FT]: only toy and moderate MLPs were tested. For transformers and CNNs, "a quantification of the occurrence probability … in these large scale settings is ongoing work" — [Mar25l](https://arxiv.org/abs/2506.14951).
- **Venue**: a citing paper's reference list gives NeurIPS 2025; Undermind and Scite list the arXiv DOI 10.48550/arXiv.2506.14951 [FT, via San26c].
- **Follow-ups / citation statements:**
  - Scite shows one citation statement, "mentioning", from Marchetti et al. 2026: intermediate-capacity regimes with unstable dynamics are "potentially related to recent results of Martinelli et al (2025), who show how pairs of neurons can collaborate to approximate gated linear units at the 'edge of stability'" — [Sequential Group Composition, arXiv 2602.03655](https://doi.org/10.48550/arXiv.2602.03655).
  - Undermind lists further citing works: San26c, Mar26p, Bei25, Hua25aa, Sch26n. San26c (Sandbrink … Brea 2026, "Teacher Geometry Shapes Learnability in Teacher-Student Networks") cites Mar25l only for closed-form ReLU Gaussian expectations, and makes no statement about channels, weight decay or GLUs [FT] — [Undermind workspace](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893).
  - No supporting or contrasting Scite statements exist.
- **With L2 on every parameter, the objective is coercive.** Liang, Sun & Srikant prove coercivity ("no decreasing paths to infinity") and zero training error at local minima, but only for **tailored** cubic per-neuron penalties with generic small coefficients (ReQU networks; a deep 1-D CNN with a ReQU last layer). This is not standard L2 [FT] — [SIAM J. Optim., DOI 10.1137/19m1299074](https://doi.org/10.1137/19m1299074).
- Finite-difference pairs with diverging opposite output weights can approximate ρ'. Trained networks (ELU, sigmoid, softsign, ISRLU; Adam; Sobolev loss; no regularization) show norms rising as error falls, but do not follow the explicit construction. The authors note that "regularization techniques may prevent a network from approximating a non-network target function" [FT] — [Mah21e](https://doi.org/10.1016/j.neunet.2021.01.007).

### Inferences
- C3's result (opposed pairs have input cos ≈ 0.07, no cancellation κ≈1, norms ≈3 that do not diverge) is **what Mar25l's own discussion would predict under weight decay** for the diverging part. With wd>0 on all weights, norms cannot run to infinity (coercivity).
- Our data go further. Even the channel **entrance** signature is absent: converging inputs combined with opposite outputs. Mar25l report that finite-step SGD/Adam reach the entrance easily, so the absence is informative. It suggests the channel mechanism is not what generates opposition in this ViT, not merely that weight decay truncates channels.
- To apply Mar25l to a 192-D output, a_i·a_j<0 has to be generalized to anti-parallel W2 columns. This generalization is ours; Mar25l never state it.

### Gaps
- No paper tests channels in transformers or with AdamW/weight decay, or with GELU.
- Whether regularized training stalls at a finite-ε "partial channel" has not been quantified anywhere.

---

## Q3. Condensation, symmetry-induced saddles and neuron splitting/duplication: do neurons merge into duplicates late in training, and with which activations?

### Takeaway
Late merging into near-identical neurons **is reported**. The strongest evidence is Che23z: VGG-16/ResNet-18 **with GELU** and wd 5e-4, where near-identical neurons appear after 1e5–1e6 SGD steps and more so at larger learning rates. Xu26ah shows weight-decay-driven angular condensation that happens late, after fitting (two-layer ReLU, plus an AdamW transformer). Ziy23c, Doi21 and Vel26b add further support.

Classic **condensation** (Xu, Luo, Zhang and colleagues) is an *initial-training*, small-initialization phenomenon, characterized for tanh/sigmoid/xtanh/ReLU. GELU is named but not tested. The symmetry-induced saddle papers (Fukumizu–Amari, Şimşek 2021, Brea 2019, embedding principle) describe *where* duplicates lie in the landscape, not *when* training gets there. None reports a merger of two named neurons in a transformer.

### Cited Findings
- **Condensation at initial training** [FT] — [Zhou et al., NeurIPS 35, DOI 10.52202/068431-0159](https://doi.org/10.52202/068431-0159):
  - An activation's multiplicity p is the order of its first non-zero derivative at 0. Under small initialization there are at most 2p condensed oriented directions, i.e. p lines, each containing a direction w and its opposite −w.
  - tanh, sigmoid and **GELU** are cited as p=1 (one line). GELU is named only in the Introduction and **not tested**; the tested activations are tanh, sigmoid, softplus, xtanh, x²tanh and ReLU.
  - "Initial stage" means before the loss falls to 70% of its initial value. Output-weight signs are not analysed and weight decay is not used.
- **Overview of condensation** [FT] — [Xu, Zhang, Zhou 2025, arXiv 2504.09484](https://arxiv.org/abs/2504.09484):
  - Definition: "neurons in the same layer tend to condense into groups with similar outputs". It is promoted by small initialization and by dropout.
  - In the training process described, the number of clusters "typically increases as training progresses".
  - A single-layer CNN on MNIST shows block structure in weight and output similarities after training (97.62% test). In an ImageNet ResNet-18, condensation is stronger in the last layer than in the first.
  - Transformer evidence is qualitative only (synthetic compositional tasks, a fixed weight decay of 0.01).
  - **Embedding principle:** splitting one neuron into two with the same input weights, and dividing its output weight between them, preserves a critical point. The wider network "can be regarded as being in a condensed state".
  - GELU is not discussed, the effect of weight decay is not established, and the signs of condensed neurons' output weights are not reported.
- **Weight decay and late condensation** [FT] — [Xu … Z.-Q. J. Xu 2026, arXiv 2610.04533](https://arxiv.org/abs/2610.04533):
  - Setting: 1-D input two-layer ReLU network, L2 on all parameters including biases, large initialization (γ<1/2).
  - Training proceeds as rapid fitting → amplitude compression → angular condensation.
  - Main run: m=200, λ=1e-3, full-batch SGD, lr 0.1, 80,000 updates. Minimum data loss is reached at epoch 447, the amplitude plateau around epoch 16,000, and both neuron groups are within 1° at epoch 54,000.
  - **Same-sign** groups condense: positive-output neurons are attracted to negative minima of the residual-correlation field and negative-output neurons to positive maxima.
  - The imbalance a_k² − r_k² decays as e^{−2λt}.
  - A two-layer decoder-only transformer trained with **AdamW** (wd 0.8 vs 0, 400 epochs) shows Q/K norm contraction, row-direction alignment and better held-out accuracy.
  - The paper does **not** prove that weight decay removes opposite-sign cancelling pairs.
- **Stochastic collapse** [FT] — [Chen … Ganguli, J. Stat. Mech. 2024, DOI 10.1088/1742-5468/ad6429](https://doi.org/10.1088/1742-5468/ad6429):
  - Invariant sets: "vanishing neurons" and "identical neurons", the latter meaning equal incoming weights, bias **and outgoing weights**.
  - SGD noise can make these sets attractive, with rate α = L''(0) + ½D''(0) > 0. Diffusion scales as η/β.
  - Experiments: VGG-16 on CIFAR-10 and ResNet-18 on CIFAR-100, SGD with momentum 0.9 and **weight decay 5e-4**. **ReLU was replaced by GELU** to avoid ReLU's extra symmetry.
  - Neurons within 10% normalized distance (incoming and outgoing parameters concatenated) count as identical.
  - Clusters of near-identical neurons appear after 1e5 (VGG) and 1e6 (ResNet) steps; larger learning rates typically give more collapse (Fig. 4). A longer high-LR phase (1e4, 2e4, 4e4 steps before dropping lr from 0.1 to 0.01) gives fewer independent neurons and better test accuracy (Fig. 6).
  - Weight decay is not ablated; the mechanism attributed is gradient noise. Conjecture I.1 equates the noise effect in the linear model with L2-regularized gradient flow with λ = ηζ²/2.
- **Symmetry-induced critical subspaces** [FT] — [Şimşek, Ged, Jacot, Spadaro, Hongler, Gerstner, Brea, ICML 2021 (PMLR 139:9722–9732), arXiv 2105.12221](https://arxiv.org/abs/2105.12221):
  - Copies have output weights β_ti·a_t with Σβ_ti = 1 and no sign restriction, so opposite-sign copies are allowed mathematically but are not analysed. Vector outputs are allowed.
  - The number of such subspaces is G(r,m) = Σ_i (−1)^{r−i} C(r,i) i^m. Expansions of strict saddles stay strict saddles.
  - The setting excludes ReLU, and the all-zero-loss-points theorem needs σ(0)≠0, which **GELU violates**.
  - Lemma 2.1: a gradient flow that starts off a symmetry subspace cannot reach it in finite time.
  - There is no theorem that training approaches the duplicated configurations, and no analysis of regularization.
- **Permutation points** [ABS] — [Brea, Şimşek, Illing, Gerstner 2019, arXiv 1907.02911](https://arxiv.org/abs/1907.02911): points where "the input and output weight vectors of two neurons in the same hidden layer collide and interchange" are critical points with at least n_{k+1} vanishing Hessian eigenvalues.
- **Embedding principle** [ABS]: "Empirically, we find that a wide DNN is often attracted by highly-degenerate critical points that are embedded from narrow DNNs" — [Zhang, Zhang, Luo, Xu, NeurIPS 2021, arXiv 2105.14573](https://arxiv.org/abs/2105.14573).
- **Fukumizu & Amari 2000**, "Local minima and plateaus in hierarchical structures of multilayer perceptrons" (Neural Networks 13(3):317–327) [META]: no abstract was available. Its duplication construction (γa, (1−γ)a) is described in Mar25l [FT]. In Scite it has 147 citation statements: 2 supporting, 0 contrasting — [DOI](https://doi.org/10.1016/S0893-6080(00)00009-5).
- **Duplicate filters** [ABS]: "We observe the emergence of duplicate filters over training iterations"; duplication increases with the number of filters (CNN and FC networks on CIFAR-10) — [RoyChowdhury, Sharma, Learned-Miller 2018](https://www.semanticscholar.org/paper/05769ee8d0731819a6f4213e3987899a06204cb9).
- **Clones** [FT] — [Doimo et al., J. Stat. Mech., DOI 10.1088/1742-5468/aceb4f](https://doi.org/10.1088/1742-5468/aceb4f):
  - "Clones appear only in regularized networks" (§3.6), and "clones appear only if the training is continued until the training error reaches zero" (§4.2).
  - In a Wide-ResNet28-8 on CIFAR-10, residual correlations fall until about epoch 160 (zero training error) and clone formation finishes around epoch 180.
  - Clones are defined over random chunks of neurons, not neuron pairs; the signs of output weights are not analysed.
- **Repeated units** [FT] — [Casper et al., AAAI 2021, DOI 10.1609/aaai.v35i8.16853](https://doi.org/10.1609/aaai.v35i8.16853):
  - Redundancy is measured by PCA rank of activations; models use ReLU.
  - There is no emergence time course. The trends are "invariant to the amount of training time after convergence".
  - "data augmentation, dropout, weight decay, and all three together have only modest effects on the trends in frivolous units".
- **Fibrations** [FT] — [Vel26b](https://arxiv.org/abs/2609.01768):
  - Fibers are defined by equal *summed* incoming weights from equivalent source classes, which is weaker than literal copies, and are detected from weights by clustering.
  - The number of fibers falls over training (MNIST MLP, Adam). In an LSTM the trend is non-monotonic.
  - No weight decay is reported. A transformer (seq2seq, Multi30k, Adam) compresses to 70% of its size without loss increase, and to 18% within a loss tolerance. The abstract says 17%, the full text 18%.
  - Same-sign output weights are not analysed.
- **Early alignment in ReLU networks** is an *early*, small-initialization phenomenon [ABS] — [Boursier & Flammarion 2024, arXiv 2401.10791](https://arxiv.org/abs/2401.10791); [Min, Vidal, Mallada 2023, arXiv 2307.12851](https://arxiv.org/abs/2307.12851).

### Inferences
- C4's merger (input cos 0.31→0.96 and output angle 72°→7° between epochs 100 and 150, under a constant lr of 5e-4 and wd 0.05) best matches the **late, identical-neuron collapse** of Che23z (GELU, WD present, long training, noise-driven) and the **late weight-decay-driven condensation** of Xu26ah. It does not match small-initialization initial condensation (Zho21h).
- Near-copies (cos 0.96) rather than exact copies are expected: under gradient flow the symmetry subspace cannot be reached in finite time (Sim21d, Lemma 2.1).
- Our norms *grow* during the merge (0.6/0.9 → 2.1/1.8). This contrasts with Xu26ah's ordering, in which amplitudes compress and then plateau before angular condensation. It suggests a loss-driven strengthening of a shared feature, not a merge driven purely by shrinkage.

### Gaps
- No paper measures merging of neuron pairs (incoming plus outgoing alignment) checkpoint by checkpoint in a transformer or ViT MLP.
- No paper studies GELU copies whose biases sit in the negative tail.
- Not read: Casper et al. 2019 "Removable and/or Repeated Units Emerge…" (no PDF), RoyChowdhury 2018 (abstract only), Fukumizu & Amari 2000 (metadata only), and the Wei/Amari/Cousseau singularity-dynamics papers (no PDFs).

---

## Q4. Weight decay / L2 implicit bias: is it established that L2 prefers splitting a contribution evenly among same-sign copies rather than large opposite-sign pairs?

### Takeaway
**Half of the hypothesis is established.**
- L2 disfavours opposite-sign cancellation (for ReLU, via the ℓ1/path-norm form).
- Under a permutation symmetry, strong enough weight decay prefers **identical** neurons (the midpoint) over distinct ones (Ziyin, Theorem 4).

**The other half is not established.** For positively homogeneous activations, after rebalancing, one unit and any same-sign split of it have the **same** minimal L2 cost. So L2 does not prefer "two halves". No result exists for non-homogeneous GELU with biases, and **AdamW is not L2-regularized Adam**: its implicit bias is an ℓ∞ constraint.

### Cited Findings
- **ReLU, L2 = ℓ1 on output weights after rebalancing** [FT] — [Savarese, …, Srebro, COLT 2019, arXiv 1902.05040](https://arxiv.org/abs/1902.05040):
  - "minimizing C(θ) is equivalent to constraining the norm of the weights of each hidden unit and minimizing the ℓ1 norm of the top layer" (Section 2, Eq. 5). The minimal cost is Σ|a_j|·‖w_j‖.
  - The proof relies on ReLU's positive homogeneity. Hidden biases are unregularized; footnote 2 says "removing them or regularizing them would substantially change the definitions or results".
  - Consequences, derived from Eq. 5 but **not stated in the paper**: same-sign identical copies cost the same as one unit, and opposite-sign copies that cancel add cost.
- **Representation-level attraction/repulsion** [FT] — [Jacot, …, Gabriel 2022, arXiv 2205.15809](https://arxiv.org/abs/2205.15809):
  - These are forces on whole layer representations, not between neuron pairs. For homogeneous activations, "a group of aligned neurons can be replaced by a single neuron without changing the resulting function".
  - Width bound: N(N+1).
  - There is no merging result for non-homogeneous activations and no analysis of cancelling pairs.
  - Literally copying a neuron's input weights and halving its output weights changes the L2 cost. A cost-preserving split exists only via a 1/√2 rescaling (derived by the reader of the paper, not a theorem in it).
- **Permutation symmetry plus weight decay → identical neurons** [FT] — [Ziyin, "Symmetry Induces Structure and Constraint of Learning", ICML pp. 62847–62866, arXiv 2309.16932](https://arxiv.org/abs/2309.16932):
  - Theorem 4: if θ_a ≠ θ_b, there exists γ0 such that for γ>γ0, ℓ_γ((θ_a+θ_b)/2, (θ_a+θ_b)/2) < ℓ_γ(θ_a, θ_b). The θ's include the input **and** output weights. The paper calls this "homogeneous ensembling" or "neuron collapse".
  - The threshold depends on the loss change and the norm reduction; no numerical value is given.
  - ResNet-18 on CIFAR-10 (wd 0 vs 1e-3, SGD lr 0.1, 100 epochs) shows homogeneous correlation blocks in the penultimate layer only with weight decay.
  - Transformer experiments (Adam, **not AdamW**; wd 0–0.1) measure K/Q rank, not neuron identity.
  - Rescaling symmetry leads to sparsity and rotation symmetry to low rank.
  - The sign-flip example (w,u)→(−w,−u) relies on tanh being odd. GELU is not odd.
- **Weight decay balances in/out norms** [FT]: |a_k|² − ‖(w_k,b_k)‖² decays as e^{−2λt} under gradient flow with L2 (Proposition 1) — [Xu26ah](https://arxiv.org/abs/2610.04533). For gradient flow without weight decay, layers are automatically balanced in homogeneous models [META, title only] — [Du, Hu, Lee 2018, arXiv 1806.00900](https://arxiv.org/abs/1806.00900).
- **AdamW ≠ L2** [ABS]: in the full-batch setting, if AdamW converges, it converges "to a KKT point of the original loss under the constraint that the ℓ∞ norm of the parameter is bounded by the inverse of the weight decay factor" — [Xie & Li, ICML 2024, arXiv 2404.04454](https://arxiv.org/abs/2404.04454). Decoupled weight decay was introduced as distinct from L2 in Adam [META] — [Loshchilov & Hutter, arXiv 1711.05101](https://arxiv.org/abs/1711.05101).
- **Weight decay induces low rank in multiplicative parameterizations** [ABS] — [Akram, Kobayashi, von Oswald 2024, arXiv 2410.23819](https://arxiv.org/abs/2410.23819):
  - Any local minimum of L(ABᵀ) + λ(‖A‖² + ‖B‖²) coincides with a minimum of the nuclear-norm-regularized loss, and the two losses "become identical exponentially quickly during training".
  - Weight decay reduces the rank of W_KᵀW_Q and PW_V in attention layers.
  - This concerns matrix rank, not individual neurons.
- **Weak contrast** [FT]: weight decay has "only modest effects on the trends in frivolous units" — [Cas19e](https://doi.org/10.1609/aaai.v35i8.16853).
- **Support** [FT]: "Clones appear only in regularized networks" — [Doi21](https://doi.org/10.1088/1742-5468/aceb4f).
- **Paths to infinity and cancelling pairs:**
  - Tailored regularizers make the loss coercive, which rules out bounded-loss sequences with diverging parameters [FT] — [Lia19d](https://doi.org/10.1137/19m1299074).
  - Diverging opposite-output finite-difference pairs appear without regularization, and the authors suggest regularization may prevent them [FT] — [Mah21e](https://doi.org/10.1016/j.neunet.2021.01.007).

### Inferences
- The defensible statement of the C3 hypothesis: *"With weight decay on all weights, configurations in which two units carry large opposite contributions that cancel are norm-costly (for ReLU the minimal L2 cost equals the ℓ1 norm of output weights). Moreover, sufficiently strong weight decay makes merging two distinct units into identical ones favourable under permutation symmetry [Ziy23c]."*
- The statement "WD favours splitting into two same-sign halves rather than one unit" is **not supported** for ReLU (cost-neutral) and is **unproven** for GELU.
- Our own inference, unverified: for AdamW, Xie & Li's box constraint ‖θ‖∞ ≤ 1/λ would be 1/0.05 = 20 if their factor is the same λ, which is far above our weight magnitudes. In our regime weight decay therefore probably acts as per-step shrinkage interacting with gradient noise (cf. Che23z, Conjecture I.1), not as a binding norm constraint.
- In a pre-LN ViT with GELU and biases, the rescaling symmetry behind the Savarese/Jacot results is broken (GELU is not homogeneous, and LN gain interacts with W1). Ziyin's permutation theorem still applies, because swapping two hidden units is an exact symmetry for any activation.

### Gaps
- No theorem or measurement compares the L2 cost of "one GELU unit" with "two same-sign GELU copies" with biases, or the cost of copies with opposite-sign GELU outputs.
- No experiment compares AdamW wd>0 with wd=0 for neuron-pair merging in a transformer MLP. Xu26ah's transformer measures Q/K alignment only.
- Pilanci & Ergen's convex reformulation and Yang et al.'s PathProx appear in the deep-search summary as giving an additive group-norm cost under which "same-sign copies can tie while opposite-sign cancellation wastes norm". These were **not read** (summary based on abstracts).

---

## Q5. Opposite-sign / push–pull unit pairs: which papers actually measured opposite contributions over training in transformers or ViTs?

### Takeaway
None at neuron level. The only transformer time course of opposition is **layer-level** (Pat25d, OLMo checkpoints). Neuron-level opposition or counterweight measurements in transformers (Ahm26c, Gon26h) have no training dynamics. Training-time opposition between units is documented only in SNNs (Zhu26l), toy superposition models (Che23an) and CNN channels (Saf26, replaceability rather than opposition).

### Cited Findings
- **Layer level**, over OLMo checkpoints: per-token opposition between adjacent layers (cos ≈ −0.2, down to −0.4) emerges at ~21% of training [FT-prev] — [Pat25d](https://doi.org/10.52202/085713-4643).
- **Counterweights** [FT-prev] — [Ahmad, Seth, Sankarapu 2026, arXiv 2610.02173](https://arxiv.org/abs/2610.02173):
  - 52 counterweights and 16 relays among 81 directions; 7/10 GPT-2 IOI heads are counterweights (e.g. L10.h7, γ = −0.886).
  - Only base vs instruct models are compared; there are no training dynamics.
- **Conditional co-ablation** [FT-prev]: interaction I_uv = δz_uv − δz_u − δz_v plus a search for backup units; no checkpoints — [Gong et al. 2026, arXiv 2607.01940](https://arxiv.org/abs/2607.01940).
- **Push–pull inhibition** in SNNs grows over training (ratio 1.06 → 2.0) [FT-prev] — [Zhu26l](https://doi.org/10.1007/s00422-025-01030-4).
- **Peer replacement** in CNN channels (CIFAR-100) [FT-prev] — [Safaai et al. 2026, arXiv 2605.07086](https://arxiv.org/abs/2605.07086):
  - The correlation between input capture and task relevance is 0.71/0.72 at the start and 0.39/0.05 after ~10 epochs.
  - Replacement by peers helps in 84.3% of lesions and recovers a median 68.7% of the damage.
- **Read/write alignment within single neurons**: the cosine between a neuron's input and output weights is studied in gated LLMs ("enrichment" vs "depletion" by layer). This concerns single neurons, not pairs, and is not tracked over training [ABS, via deep-search summary] — [Gerstner & Schütze 2025, arXiv 2505.17936](https://arxiv.org/abs/2505.17936).
- **Opposite neurons** arise in a firing-rate model of multisensory integration through Hebbian/anti-Hebbian rules. This is biological modelling, not a deep network [ABS] — [Ho, Zhang, Lee 2019, bioRxiv](https://doi.org/10.1101/814483).
- **Opposite singularities** in MLPs, "Stability analysis of opposite singularity in multilayer perceptrons" (Neurocomputing 2017): paywalled, not read [META] — [Guo et al.](https://doi.org/10.1016/j.neucom.2017.12.018).
- **Anti-pairing**, Kwon & Kim 2005 (soft-committee / statistical mechanics of learning): not read [META] — [DOI](https://doi.org/10.1088/0305-4470/38/25/003).

### Inferences
- Our C1 is the first neuron-level per-token analogue of Pat25d's layer-level correction, and it is in a ViT. Pat25d is the right baseline to cite for "opposition emerging during training", and its timing (absent early, emerging mid-training) is compatible with C1's onset.
- The classic opposite-sign mechanisms in the literature (finite differences, channels, anti-pairing) all involve **opposite output weights**. C2's mechanism (parallel outputs, opposite activation signs) is structurally different, and only smooth activations with a negative tail allow it.

### Gaps
- Kwon & Kim 2005 and Guo et al. 2017 remain unread and are the main risk to the claim that C2 is distinct from earlier "anti-pairing".

---

## Q6. Has anyone tracked same-token opposite contributions of MLP neurons in the residual stream across a whole ViT/transformer training run? (novelty)

### Takeaway
**No such paper was found.** Searches covered seven Undermind deep searches (five earlier, two today, together several hundred ranked papers) plus targeted semantic searches today, including one restricted to 2023 and later. The closest works are a layer-level time course (Pat25d), sign-specific ablation dynamics of the negative GELU region in LLMs (Kon25e, abstract only), and activation-sparsity time courses (Wil24b, Li22ap). This supports an explicit novelty statement, with the caveat that the deep-search screening is based on abstracts.

### Cited Findings
- The 2026-10-14 deep search on per-token add/cancel of MLP neuron contributions concluded: "the retrieved literature does not directly quantify how neuron output vectors sum or cancel for a single token; that vector-cooperation measurement—especially in ViTs—appears to be a clear gap" — [Undermind deep search](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/How%20individual%20MLP%20neuron%20contributions%20add%20up%20and%20cancel%20on%20a%20single%20token%20in%20transformers%20and%20how%20this%20evolves%20in%20training).
- The 2026-10-20 prior-art search containing our measured numbers rated "opposition emerges after the main accuracy gain" as "likely novel" and anti-parallel pairing as only "partially anticipated" (Mar25l, Kwo05b) — [Undermind deep search](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Prior%20art%20check%20for%20specific%20measured%20findings%20on%20opposing%20MLP%20neuron%20pairs%20in%20a%20ViT%20detector%20across%20training).
- Today's duplicate-neuron deep search concludes: "Transformer work establishes neuron-level redundancy and ViT MLP compressibility, not this trajectory of within-layer neuron cloning". It notes that "the late timing, AdamW, GELU negative-tail behavior, and joint incoming/outgoing alignment remain the key unaddressed combination" — [Undermind deep search](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Hidden%20neurons%20merging%20into%20near%20duplicate%20copies%20during%20training%20condensation%20and%20symmetry%20induced%20neuron%20splitting).
- A semantic search today for "per-token contributions of individual MLP neurons across ViT checkpoints, opposing pairs and turnover" (2023+) returned only ViT pruning, compression and interpretability papers, with no match.
- Activation sparsity and its evolution over training in ReLU transformers and ViTs is measured, but vector cancellation is not [ABS, via deep-search summary] — [Wild & Anderson 2024, arXiv 2407.07848](https://arxiv.org/abs/2407.07848); [Li et al., ICLR 2023, arXiv 2210.06313](https://arxiv.org/abs/2210.06313).

### Inferences
- A novelty claim is reasonable if worded narrowly: "To our knowledge, no prior work tracks, across the full training of a transformer, the per-token directional opposition between the residual-stream contributions of individual MLP neurons, nor decomposes it into output-weight geometry versus activation sign."

### Gaps
- The screening is abstract-based. Mechanistic-interpretability blog posts and workshop papers (e.g. LessWrong / Alignment Forum) are not indexed and were not searched.

---

## Q7. Per-paper notes: what each paper measured, and its relevance to C1–C4

### Takeaway
The papers that matter most are Mar25l (the hypothesis tested and rejected in C3), Che23z and Ziy23c (mechanisms for late copies, C3/C4), Xu26ah (weight decay and late condensation), Sav19 (what L2 does and does not prefer), Pat25d (time-course baseline for C1) and Kon25e (GELU negative tail, C2). Each is listed below with its measured quantities. Full details for Mar25l are in Q2.

### Cited Findings

**Core theory and mechanisms**

- **Mar25l** [FT] — [arXiv 2506.14951](https://arxiv.org/abs/2506.14951). Martinelli, van Meegen, Şimşek, Gerstner, Brea (2025; NeurIPS 2025 per a citing reference list).
  - Measured: channel frequency in 4,420 small MLPs (20–64%), ε² loss scaling, 1/ε² sharpness, edge-of-stability stalling (SGD lr 0.1, Adam lr 1e-3).
  - Relevance: C3 (the hypothesis tested), C2 (predicts opposite output signs). See Q2.
- **Che23z** [FT] — [DOI 10.1088/1742-5468/ad6429](https://doi.org/10.1088/1742-5468/ad6429). F. Chen … S. Ganguli, "Stochastic collapse: how gradient noise attracts SGD dynamics towards simpler subnetworks", J. Stat. Mech. 2024.
  - Measured: fraction of independent neurons per layer (VGG-16/CIFAR-10, ResNet-18/CIFAR-100, GELU, wd 5e-4, SGD momentum 0.9) at lr 0.01–0.10 after 1e5 steps; "identical" means within 10% normalized distance of concatenated in/out weights. Pairwise distance distributions develop a peak near 0 in collapsing layers (Fig. 8). In a two-neuron toy network, weights become more similar over training and collapse faster with larger lr or noise (Fig. 7).
  - Relevance: the strongest empirical precedent for C3/C4 late copies with GELU. Weight decay is present but not ablated.
- **Ziy23c** [FT] — [arXiv 2309.16932](https://arxiv.org/abs/2309.16932). L. Ziyin, ICML pp. 62847–62866.
  - Measured or proven: Theorem 1 (mirror symmetry plus L2: a threshold γ0 above which projection onto the symmetric subspace lowers the loss); Theorem 4 (permutation: the midpoint neuron beats a distinct pair above γ0); ResNet-18 correlation blocks with wd 1e-3 vs 0; transformer K/Q rank decreasing with wd (Adam).
  - Relevance: theoretical support for "weight decay favours identical copies" (C3/C4). It does not cover opposite-sign pairs specifically.
- **Xu26ah** [FT] — [arXiv 2610.04533](https://arxiv.org/abs/2610.04533). C.-C. Xu … Z.-Q. J. Xu (2026).
  - Measured: stage markers (data-loss minimum at epoch 447, amplitude plateau around 16,000, groups within 1° by 54,000); a transformer trained with AdamW at wd 0.8 shows Q/K row alignment.
  - Relevance: late, weight-decay-driven, same-sign condensation (C3/C4). Its ordering (compress, then condense) contrasts with C4's norm growth.
- **Sim21d** [FT] — [arXiv 2105.12221](https://arxiv.org/abs/2105.12221). Şimşek et al., ICML 2021.
  - Measured: counts of critical subspaces G(r,m) and T(r,m); for widths 10–1000 on MNIST, gradient-norm "zigzags" interpreted as saddle visits (not identified as duplicate configurations).
  - Relevance: C4 framing (an approach towards a symmetry subspace). It does not cover GELU (σ(0)=0).
- **Sav19** [FT] — [arXiv 1902.05040](https://arxiv.org/abs/1902.05040). Savarese, …, Srebro, COLT 2019.
  - Proven: L2 on ReLU weights is equivalent to the ℓ1 norm of output weights at unit-norm input weights; biases are unregularized.
  - Relevance: C3. Cancellation is penalized; same-sign splitting is neutral.
- **Jac22c** [FT] — [arXiv 2205.15809](https://arxiv.org/abs/2205.15809). Jacot, …, Gabriel (2022).
  - Measured: a depth-3 ReLU network on 42 MNIST samples (20,000 steps), with neurons aligned along 3 rays; loss plateaus beyond width ≈10 (N=1000).
  - Relevance: C3. Aligned homogeneous neurons are replaceable by one neuron; the paper does not cover GELU.
- **Zho21h** [FT] — [DOI 10.52202/068431-0159](https://doi.org/10.52202/068431-0159). Zhou … Xu, NeurIPS 35.
  - Measured: input-weight cosine matrices under small initialization for tanh, sigmoid, softplus, xtanh, x²tanh and ReLU (Adam).
  - Relevance: C3 terminology. Our late copies under standard initialization are **not** initial condensation. GELU (p=1) would predict condensation onto ±w lines early, which matches neither our opposed pairs (cos ≈ 0.07) nor our copies (late).
- **Xu25ad** [FT] — [arXiv 2504.09484](https://arxiv.org/abs/2504.09484). Overview of condensation.
  - Measured: CNN similarity blocks after training (MNIST 97.62%); stronger condensation in the last than in the first layer of ResNet-18; qualitative transformer evidence.
  - Relevance: background and terminology for C3/C4.
- **Mah21e** [FT] — [DOI 10.1016/j.neunet.2021.01.007](https://doi.org/10.1016/j.neunet.2021.01.007). Mahan, King, Cloninger 2021.
  - Measured: network norm rises as error falls during training (ELU, sigmoid, softsign, ISRLU; Adam).
  - Relevance: C2. The finite-difference mechanism needs anti-parallel outputs, which we do not observe.
- **Lia19d** [FT] — [DOI 10.1137/19m1299074](https://doi.org/10.1137/19m1299074). Liang, Sun, Srikant.
  - Proven: tailored regularizers make the loss coercive with no bad local minima.
  - Relevance: C3. Diverging channels are impossible under coercive regularization; the result is not specific to standard L2.
- **Xie24h** [ABS] — [arXiv 2404.04454](https://arxiv.org/abs/2404.04454). Xie & Li, ICML 2024.
  - Proven: AdamW converges to KKT points under an ℓ∞ constraint.
  - Relevance: C3 reviewer risk, since AdamW ≠ L2.
- **Akr24** [ABS] — [arXiv 2410.23819](https://arxiv.org/abs/2410.23819).
  - Proven and measured: L2 on factor matrices corresponds to a nuclear-norm penalty on their product; weight decay reduces attention-product rank.
  - Relevance: weight-decay effects in transformers at matrix level, contextual only.

**Redundancy, copies, clones**

- **Doi21** [FT] — [DOI 10.1088/1742-5468/aceb4f](https://doi.org/10.1088/1742-5468/aceb4f).
  - Measured: clone chunk sizes w_c* of about 50 (pMNIST), 100 (CIFAR-10) and 200 (CIFAR-100); clones form by about epoch 180 in a WRN28-8; no clones without weight decay; none in an ImageNet ResNet-50 with 8% training error.
  - Relevance: weight decay plus long training gives redundancy (C3), though at chunk level, not pair level.
- **Cas19e** [FT] — [DOI 10.1609/aaai.v35i8.16853](https://doi.org/10.1609/aaai.v35i8.16853).
  - Measured: redundancy gain per width doubling g ≈ 1.79–2.45 across ImageNet, CIFAR and synthetic settings; weight decay has a "modest" effect.
  - Relevance: C4 (removable redundant units give small ablation effects); weakly contrasts C3's weight-decay claim.
- **Vel26b** [FT] — [arXiv 2609.01768](https://arxiv.org/abs/2609.01768).
  - Measured: fiber counts falling over epochs; compression to 70% (no loss increase) or 18% (with a tolerance).
  - Relevance: merging over training (C1/C4 context). No weight decay is reported.
- **Roy18b** [ABS] — [link](https://www.semanticscholar.org/paper/05769ee8d0731819a6f4213e3987899a06204cb9). Duplicate filters emerge over training on CIFAR-10. Relevance: C4 precedent in CNNs.
- **Bre19** [ABS] — [arXiv 1907.02911](https://arxiv.org/abs/1907.02911). Permutation points where the in/out weights of two neurons collide. Relevance: C4 geometric language.
- **Zha21ar** [ABS] — [arXiv 2105.14573](https://arxiv.org/abs/2105.14573). Wide networks are "often attracted" to critical points embedded from narrower networks. Relevance: C3/C4.

**Opposition and time-course baselines**

- **Pat25d**, **Zhu26l**, **Che23an**, **Ahm26c**, **Gon26h**, **Saf26** [FT-prev]: see Q5. Relevance: C1 baselines and methods (Gon26h's I_uv corresponds to our interaction ratio).
- **Kon25e** [ABS] — [arXiv 2509.24198](https://arxiv.org/abs/2509.24198). Negative pre-activations of a sparse set of GELU/SiLU neurons are functional, and ablating them becomes more harmful over training. Relevance: C2. A reviewer could object that the GELU tail is negligible; this paper counters that.

### Inferences
- Taken together, the literature gives C3/C4 a principled explanation: permutation-symmetric invariant sets are attractive under weight decay (Ziy23c) or gradient noise (Che23z). It gives C1/C2 no ready-made explanation, which is why they are the most novel parts.

### Gaps
- Fukumizu & Amari 2000, Kwon & Kim 2005, Guo et al. 2017, Casper et al. 2019 (arXiv version) and RoyChowdhury 2018 were not read in full.

---

## Q8. How to position C1–C4 in the article (suggested framing), and reviewer risks

### Takeaway
Present C1 and C2 as **new measurements**: neuron-level, per-token opposition dynamics in a ViT, and its decomposition into output geometry versus activation sign. Use Pat25d as the layer-level precedent.

Present C3 as a **negative test of a specific published hypothesis** (Mar25l) in a regime that the hypothesis' authors flagged as untested (transformers, regularization). Present the copies as **consistent with** symmetry-induced collapse under weight decay or noise (Ziy23c, Che23z, Xu26ah), without claiming that weight decay is the cause.

Present C4 as a **case study** of one merger event. Its functional effect is negligible, which matches redundancy literature (Cas19e, Doi21).

### Cited Findings
- **Mar25l does not cover our regime.** It flags transformer-scale quantification as "ongoing work" and expects regularization to stall channels — [Mar25l](https://arxiv.org/abs/2506.14951) [FT].
- **Identical-neuron collapse is documented** with GELU and weight decay in deep CNNs — [Che23z](https://doi.org/10.1088/1742-5468/ad6429) [FT] — and is favoured by weight decay under permutation symmetry — [Ziy23c](https://arxiv.org/abs/2309.16932) [FT].
- **For ReLU, L2 does not prefer same-sign splitting over one unit**; cancellation costs norm — [Sav19](https://arxiv.org/abs/1902.05040) [FT].
- **AdamW's implicit bias differs from L2** — [Xie24h](https://arxiv.org/abs/2404.04454) [ABS].
- **Weight decay's role in redundancy is contested**: "modest" in Cas19e [FT] — [DOI](https://doi.org/10.1609/aaai.v35i8.16853); necessary for clones in Doi21 [FT] — [DOI](https://doi.org/10.1088/1742-5468/aceb4f).
- **The gradient-noise and learning-rate dependence of collapse** (Che23z) means a constant lr of 5e-4 is itself a candidate cause — [Che23z](https://doi.org/10.1088/1742-5468/ad6429) [FT].

### Inferences

**Suggested article framing (sentence-level):**
- **C1**: "Pairs of MLP neurons whose same-token contributions are opposed by ≥150° are absent at initialization and emerge during training. At the level of whole layers, a related correction pattern has been reported to emerge mid-pretraining in LLMs [Pat25d], and push–pull inhibition emerges with task learning in spiking networks [Zhu26l]. To our knowledge, the neuron-level, per-token time course (rise, peak, decline, identity turnover) has not been reported for any transformer." Avoid "first ever"; use "to our knowledge".
- **C2**: "Unlike the finite-difference or channel mechanisms, in which opposition comes from output weights of opposite sign on converging inputs [Mah21e, Mar25l], opposition here comes from nearly parallel output weights combined with opposite GELU signs. This relies on GELU's non-zero negative tail, which has independently been shown to be functionally used in smooth-activation LLMs [Kon25e]."
- **C3**: "We tested the channel hypothesis [Mar25l] directly. Opposed pairs show neither converging input weights nor cancellation or diverging norms. This is consistent with the authors' expectation that regularization stalls channel traversal, and with the coercivity of objectives regularized on all parameters [Lia19d]. Instead, some neurons become near-duplicates late in training, consistent with attraction to permutation-symmetric invariant sets under weight decay [Ziy23c] or gradient noise [Che23z], and with late weight-decay-driven condensation [Xu26ah]."
  - Phrase the weight-decay hypothesis as: "weight decay penalizes norm-costly cancellation and, under permutation symmetry, favours identical over distinct units [Ziy23c, Sav19]".
  - Do **not** phrase it as "weight decay favours splitting into halves", which is unsupported (cost-neutral for ReLU).
- **C4**: "A single merger event illustrates the path towards a permutation-symmetric configuration [Bre19, Sim21d]: input and output weights align while both units move into GELU's negative regime. As expected for redundant units [Cas19e, Doi21], removing the pair has a negligible effect on IoU." Note that exact coincidence is unreachable in finite time under gradient flow [Sim21d].

**Reviewer risks:**
1. **Single seed (3047).** C1–C4 are from one run, and collapse depends on lr and noise (Che23z) and on initialization (Cas19e). Replicate C1 counts and copy formation on several seeds; the 24-seed campaign exists for other measures.
2. **Is "opposition" just copies crossing zero?** C2 plus C4 imply that near-copies with parallel W2 are anti-parallel exactly on tokens where one pre-activation crosses zero. A reviewer may argue that C1 is partly a by-product of C3/C4. Report how many C1 pairs are copy-like (input cos >0.9) at each epoch, and how often each pair's opposition happens in the near-zero band.
3. **Null model for the 150° threshold in 192-D.** Provide shuffled-partner or random-rotation baselines for the count of ≥150° beam pairs. The earlier 3-D projection inflated opposition (prior notes).
4. **AdamW ≠ L2.** L2-based theory (Sav19, Jac22c, Ziy23c) does not transfer directly (Xie24h). Without a wd=0 control run, and a different-lr control (Che23z's noise mechanism), the weight-decay hypothesis stays speculative. Also report whether weight decay was applied to biases and LayerNorm parameters: Sav19's footnote says regularizing biases "would substantially change" the results.
5. **Fair test of Mar25l.** Mar25l's channels are defined for scalar outputs, no weight decay, and very long gradient flow, with a norm >1e3 criterion. Absence under AdamW with wd 0.05 is partly expected. State that we test whether channels *explain opposition* in a practical regularized transformer, not that Mar25l is wrong. Extending a_i·a_j<0 to anti-parallel W2 columns is our assumption.
6. **Pre-LN and GELU break the rescaling symmetry.** Input-weight cosines should be computed on effective weights (W1 with the LN gain folded in, and the bias including LN β). Report both versions.
7. **Post-hoc selection of the "most opposed pair".** C4 is an extreme pair chosen after the fact. Say so, or define selection rules in advance and report the distribution across all copy-like pairs.
8. **Unread potential pre-emptions.** Kwon & Kim 2005 "anti-pairing", Guo et al. 2017 "opposite singularity" and Fukumizu & Amari 2000 must be read before submission.
9. **Terminology.** "Condensation" in the Xu/Luo/Zhang literature means early, small-initialization clustering of input directions, possibly including anti-parallel ±w. Use "late near-duplicate formation" or "collapse to identical neurons" [Che23z], or cite Xu26ah for weight-decay condensation.
10. **Tiny functional effects.** ΔIoU ~1e-5 with a CI including 0 invites a "so what?". Frame C1–C4 as training-dynamics geometry, and keep functional claims to the interaction-ratio result, using the I_uv framing of Gon26h.

### Gaps
- No direct literature evidence decides whether weight decay or SGD/Adam noise causes the late copies in a ViT. This needs our own wd=0 and lr-varied runs.
- No precedent exists for the rise-then-decline shape of C1, so it cannot be benchmarked against published curves.
