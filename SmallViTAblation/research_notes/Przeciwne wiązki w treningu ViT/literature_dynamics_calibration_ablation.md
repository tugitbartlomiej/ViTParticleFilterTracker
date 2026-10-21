# Literature check: training past the performance plateau, confidence drift, and ablation methodology for "opposite beams" in a small ViT detector

Scope note. Sources were found through Undermind (2 new deep searches plus 3 existing workspace searches; full-text PDF reads by sub-agents) and Scite (metadata, full-text excerpts, Smart Citation tallies). A paper is marked **[full text]** when its PDF was read for these notes and **[abstract only]** when only the abstract or deep-search summary was available. Numbers marked "approx./visual" were read off figures by the PDF reader; the papers do not tabulate them. Paraphrases are used instead of verbatim quotes. Our own setup, used in the inferences: AdamW, lr 5e-4 constant after warmup, wd 0.05, bf16, one seed, 200 epochs, 35k training images, VAL/TEST of 500 frames each.

Deep searches used (Undermind workspace 40294fa6-c9a3-4469-9a61-d3eddeaee893):
- New: [Late phase internal change after test performance saturates...](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Late%20phase%20internal%20change%20after%20test%20performance%20saturates%20under%20constant%20learning%20rate%20and%20weight%20decay) and [Confidence and calibration drift with longer training...](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Confidence%20and%20calibration%20drift%20with%20longer%20training%20in%20classifiers%20and%20object%20detectors%20including%20false%20positive%20confidence)
- Existing ones I inspected: "Measuring synergy redundancy and compensation between neurons through training with interventions", "Hidden neurons merging into near duplicate copies...", "Weight decay implicit bias on how a function is split across hidden neurons...", "Prior art check for specific measured findings on opposing MLP neuron pairs..."

---

## Q0. Verdict for each claim (D1, D2, D3, A1)

### Takeaway
None of the four claims is contradicted by published work. D2 and D3 match well-documented mechanisms: confidence keeps growing after accuracy saturates, and weights keep moving and reorganizing after a plateau under AdamW with weight decay. What is new is the measurements themselves: false-positive confidence on frames with no instrument tracked epoch by epoch in a detector, and the life cycle of opposite and duplicate neuron pairs. A1 is methodologically sound in its controls. Reviewers will still ask for mean or mode ablation, direct-effect versus total-effect measurement, a continuous outcome metric and a standard interaction index. Calling additive effects "redundant" clashes with standard usage.

### Cited Findings

**Verdict table**

| Claim | Closest published evidence | Verdict | Key quantities from the literature |
|---|---|---|---|
| **D1** IoU saturates early (VAL 0.721 at ep 9; 0.748–0.775 from ep 40 to 200; TEST 0.745→0.756; CIs overlap) | Guo 2017 Fig. 3; Oksuz 2023 App. E.3; Andriushchenko 2023; Chen-Kunin-Yamamura-Ganguli 2023 (stochastic collapse) | **Consistent / typical; not novel as such.** One caveat: at a large constant LR, a plateau can hide "progress" that only shows after the LR is decayed. | ResNet-110 SD, CIFAR-100: test error 29%→27% while test NLL rises ([Guo 2017](https://arxiv.org/abs/1706.04599)). Faster R-CNN 12→36 epochs: AP 24.8→25.5 ([Oksuz 2023](https://doi.org/10.1109/CVPR52729.2023.00894)). Running the large-step phase longer gives better test error once the step is decayed, even though training loss had stabilized ([Andriushchenko 2023](https://doi.org/10.48550/arXiv.2210.05337); [Chen 2023](https://doi.org/10.1088/1742-5468/ad6429)) |
| **D2** Confidence keeps rising while IoU is flat (positives 0.863→0.899 VAL), including on frames without an instrument (0.629→0.675 VAL n=34; 0.541→0.579 TEST n=28) | Guo 2017; Mukhoti 2020; Wei 2022; Prieto 2025; Minderer 2021; Xie 2026; Oksuz 2023 App. E.3; Park 2024; Huseljic 2024 | **Supported in mechanism and in classifiers. One detector precedent for longer schedule → worse calibration. False-positive confidence on empty frames tracked across epochs: not found in the literature.** | CIFAR-10 ResNet-50: test-set entropy falls throughout training for correctly and incorrectly classified samples, and ECE rises after epoch 150 ([Mukhoti 2020](https://arxiv.org/abs/2002.09437)). Logit norm rises over 200 epochs, also on OOD (SVHN) inputs, approx. 6→10–11 ([Wei 2022](https://doi.org/10.48550/arXiv.2205.09310)). Faster R-CNN 12→36 epochs: mECE 16.4→18.7, attributed to overconfidence ([Oksuz 2023](https://doi.org/10.1109/CVPR52729.2023.00894)). ECE minimum at ep 60 (1.65%) rising to 3.34% at ep 200 ([Xie 2026](https://arxiv.org/abs/2610.12048)) |
| **D3** Internal reorganization after the plateau (opposite pairs peak at ep 80–110 then decline; pair identities turn over; late merging into near-duplicates at ep 100–150; output norms of the merged neurons grow 3×) | Kunin 2021/23; Zhang 2021; Kosson 2024; Hoogland 2024/25; Nanda 2023; Sharon & Dar 2024; Ratzon 2023; Andriushchenko 2023; Chen 2023 stochastic collapse; Martinelli 2025 | **Supported in kind: continued motion and reorganization at a flat metric are predicted and observed. The specific pair life cycle is novel. Late duplicate formation at a constant, relatively large LR is partially anticipated by "stochastic collapse".** | Squared parameter displacement after convergence grows as k^1.075 (ResNet-18, ImageNet) ([Kunin](https://doi.org/10.1162/neco_a_01626)). Under AdamW at equilibrium the angular update per step is sqrt(2ηλ(1−β1)/(1+β1)), and rotation continues at a steady rate regardless of loss convergence ([Kosson 2024](https://arxiv.org/abs/2305.17212)). Near-identical neurons (same incoming and outgoing weights) form clusters, and a longer high-LR phase means fewer independent neurons even after loss and accuracy plateau. Those experiments used GELU ([Chen 2023](https://doi.org/10.1088/1742-5468/ad6429)) |
| **A1** Zero-ablation of the most opposed pair: ΔIoU ≈ 1e-5 (CI includes 0), inside the distribution of 300 control pairs; 768 single ablations; interaction ratio 1.66 means additive, not cancelling; zero ≈ typical value (GELU minimum) | Morcos 2018; Zhou 2018; Meyes 2019; Heimersheim & Nanda 2024; Li & Janson 2024; McGrath 2023; Rushing & Nanda 2024; Pochinkov 2024; Vaidyanathan 2026; Gong 2026; Ahmad 2026; Hütten 2025 | **Consistent with the literature: small single-unit effects on aggregate metrics are common. The methodology partly meets current standards.** The controls are strong. Missing: mean, mode or optimal ablation; direct-effect measurement or LN-frozen ablation; inclusion–exclusion interaction with a CI; an equivalence bound. "Redundant" is non-standard for additive effects. | Single-unit ablation: overall accuracy drops <1% but up to 50% for single classes ([Zhou 2018](https://arxiv.org/abs/1806.02891)). Mean ablation hurts *more* than zero ablation in ReLU CNNs ([Morcos 2018](https://arxiv.org/abs/1803.06959)). Zero, mean and peak ablation do not differ significantly in ViT ([Pochinkov 2024](https://doi.org/10.48550/arXiv.2408.17322)). Downstream compensation has slope 0.69 (r² 0.92) ([McGrath 2023](https://doi.org/10.48550/arXiv.2307.15771)). For the median IOI component, the optimal-ablation gap is 11.1% of the zero-ablation gap and 33.0% of the mean-ablation gap ([Li & Janson 2024](https://doi.org/10.48550/arXiv.2409.09951)) |

**Scite Smart Citation tallies for the anchor papers** (supporting / contrasting / mentioning statements; citing publications in the Scite index):
- Guo 2017: 8 / 0 / 577; 2193 citing publications — [Scite record](https://arxiv.org/abs/1706.04599)
- Minderer 2021: 1 / 2 / 26; 147 citing publications. One contrasting statement comes from a camera-trap calibration paper (10.1002/rse2.412), but its snippet was not returned. The supporting statement concerns ViT and MLP-Mixer being among the best calibrated in- and out-of-distribution — [Scite record](https://arxiv.org/abs/2106.07998)
- Morcos 2018: 2 / 0 / 55; 203 citing publications — [Scite record](https://arxiv.org/abs/1803.06959)
- McGrath 2023: 1 / 0 / 47; 34 citing publications — [Scite record](https://doi.org/10.48550/arXiv.2307.15771)

### Inferences
- D1 is a framing claim, not a discovery. Its value is as the baseline against which D2 and D3 are shown to keep moving. Phrase it as a "plateau of the task metric under a constant LR", not "convergence". The literature shows that a plateau is not a stationary point.
- D2's strongest support for detectors is Oksuz 2023, Appendix E.3. It is small: one detector and two schedule lengths. Our epoch-resolved trajectory, including empty frames, is therefore a real contribution, provided its statistics hold up (n=34 and n=28 empty frames is small).
- D3 needs to be positioned against two near-precedents so that reviewers do not call it rediscovery: Kosson 2024 (continued rotation is expected at constant ηλ) and Chen 2023 stochastic collapse (duplicate neurons form under prolonged large-LR SGD noise, with GELU).
- A1 is the claim most exposed to methodological criticism (see Q3 and Q5). The literature generally accepts the null result itself. What is contested is how to interpret it.

### Gaps
- I found no published study that measures confidence on object-free images epoch by epoch in a detector.
- I found no published study tracking the rise and decline of opposite-sign neuron pairs across checkpoints in a ViT. The workspace's own "prior art check" search reached the same conclusion: closest is Martinelli 2025 on two-layer MLPs; Kwon & Kim 2005 "anti-pairing" has no abstract available.

---

## Q1. Training past the plateau: which theories predict continued internal change at constant LR with weight decay? (D1, D3)

### Takeaway
Several independent lines of work predict that parameters and internal structure keep changing after a task metric plateaus, especially at constant learning rate and with weight decay:
- SGD/AdamW limiting dynamics: anomalous diffusion, no stationary point.
- Weight-decay equilibrium: steady angular updates under AdamW.
- Grokking-style circuit formation and cleanup driven by weight decay.
- Developmental stages that the local learning coefficient (LLC) detects at near-flat loss.
- Noise-driven drift toward sparse or simpler solutions, including duplicate neurons.

Most of this evidence comes from classifiers, toy tasks or language models, not ViT detectors.

### Cited Findings

**(a) Parameters keep moving after performance converges**
- Kunin et al. (Neural Computation; arXiv 2107.09133) **[full text]**: after performance has converged, networks keep moving through parameter space by anomalous diffusion. For ImageNet-pretrained ResNet-18 with resumed training, squared displacement grows as k^c with c = 1.075, while squared local (per-step) displacement stays roughly constant at about 1.2e-6. The fitted c rises with trajectory length. Theory: an underdamped Langevin model in which weight decay λ enters the modified loss through H^{-1}(H+λI). Expected squared step size is η²σ²tr(H)/(S(1−β²)): larger lr means more motion, larger batch means less — [Kunin et al.](https://doi.org/10.1162/neco_a_01626)
- Zhang, Li, Sra & Jadbabaie (ICML 2022) **[abstract only]**: in ImageNet+ResNet-101 and WT103+TransformerXL training, weights do not converge to stationary points even though training loss stabilizes. They model the weight distribution as converging to an approximate invariant measure — [Zhang et al.](https://arxiv.org/abs/2110.06256)

**(b) AdamW + weight decay: equilibrium norms and continued rotation**
- Kosson, Messmer & Jaggi, "Rotational Equilibrium" (ICML 2024, PMLR 235) **[full text]**:
  - For a scale-invariant weight vector of dimension C under AdamW, the equilibrium norm is about sqrt(ηC/(2λ)) when ηλ ≪ 2.
  - The expected angular update per step at equilibrium is sqrt(2ηλ(1−β1)/(1+β1)), independent of C.
  - Squared-norm deviations decay by a factor (1−ηλ)² per step, so the timescale is of order 1/(ηλ) steps.
  - The authors say this equilibrium comes from the geometry of stochastic updates, not from convergence of the loss. Measured rotation in ResNet-50 and GPT-2 settles near the predicted value at constant LR.
  - For weights that are **not** scale-invariant, an outward radial gradient component acts as negative extra decay: λ_e = λ + λ_u with λ_u = −E[u∥]/(η‖ω‖). The norm can then exceed the scale-invariant equilibrium. No closed form exists for AdamW in this case; only the SGDM derivation is in Appendix H.

  — [Kosson et al. 2024](https://arxiv.org/abs/2305.17212)
- Wan, Zhu, Zhang & Sun, "Spherical Motion Dynamics" (NeurIPS 2021) **[abstract only; PDF unavailable]**: for normalized networks trained with SGD(+momentum) and weight decay, they prove an equilibrium is reached at a linear rate under stated assumptions, and propose the "angular update" in place of "effective LR". Their theoretical equilibrium values were verified on ImageNet and MSCOCO — [Wan et al. 2021](https://www.semanticscholar.org/paper/5e94cb6d4d6d0c61439da0658929de5bd4a8d60a)
- Liu, Michaud & Tegmark, "Omnigrok" (ICLR 2023) **[full text]**: when training loss is negligible, weight decay γ shrinks the weight norm roughly as exp(−γt). The time to reach the generalizing "Goldilocks" norm is about ln(w0/wc)/γ. The decoder version gives t ∝ 1/(η_D·γ). The 1/γ scaling held over about two orders of magnitude for modular addition, and approximately for MNIST — [Liu et al.](https://doi.org/10.48550/arXiv.2210.01117)

**(c) Grokking: circuit formation and cleanup while the loss looks flat**
- Nanda, Chan, Lieberum, Smith & Steinhardt (ICLR 2023) **[full text]**: modular addition, full-batch AdamW, lr 1e-3, weight decay λ = 1, 40k epochs, 5 seeds. Three phases:
  - memorization (0–1.4k epochs);
  - circuit formation (1.4k–9.4k), with train and test loss roughly flat while restricted/excluded losses change and the sum of squared weights falls;
  - cleanup (9.4k–14k), in which test loss drops, the weight norm falls more sharply and the Fourier sparsity (Gini) rises.

  After about 14k epochs the weight norm is essentially flat. Grokking emerges from gradual amplification of structured mechanisms followed by removal of memorizing components — [Nanda et al. 2023](https://doi.org/10.48550/arXiv.2301.05217)
- Varma et al., "Circuit efficiency" (2023) **[full text]**: weight decay favours the more parameter-efficient (generalizing) circuit. Between epochs 200 and 1500, the logits of both circuits rise while the parameter norm falls (efficiency improves). After epoch 4000 the generalizing-circuit logit *falls* while test loss keeps improving. So late-phase logit magnitudes can rise and then decline without a change in performance — [Varma et al.](https://doi.org/10.48550/arXiv.2309.02390)
- Prieto, Barsbey, Mediano & Birdal, "Grokking at the edge of numerical stability" (2025) **[full text]**:
  - After 100% training accuracy, cross-entropy gradients align with a "naïve loss minimization" (NLM) direction that scales logits without changing predictions. The cosine between updates and this direction reaches about 0.9 for output layers.
  - Without weight decay, weight norms keep growing. The authors argue weight decay counteracts NLM, which is why it reliably induces grokking.
  - With limited floating-point precision, "Softmax Collapse" (the loss rounding to zero) halts learning, and it sets in earlier at lower precision.

  — [Prieto et al. 2025](https://doi.org/10.48550/arXiv.2501.04697)

**(d) Developmental stages and the local learning coefficient**
- Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet, "Loss Landscape Degeneracy and Stagewise Development in Transformers" (TMLR 07/2025; arXiv 2402.02364, earlier title "The developmental landscape of in-context learning") **[full text]**:
  - Language model: 2-layer attention-only, AdamW, lr 1e-3, **weight decay 0.05, no LR scheduler** (close to our optimizer setup). Five LLC stages; the last (LM5, steps 17k–50k) showed no significant change on the authors' metrics.
  - In-context linear regression model (Adam, OneCycle, wd 0): in stages LR3 (40k–126k) and LR4 (126k–320k), test loss changes only −0.07 and −0.05 while the LLC drops by 12.3 and 44.1. In those stages layer-norm and unembedding weights collapse, and out-of-distribution robustness deteriorates.
  - Similar stage boundaries across 5 seeds.

  — [Hoogland et al.](https://arxiv.org/abs/2402.02364)
- Chen, Lau, Mendel, Wei & Murfet, "Dynamical versus Bayesian phase transitions in a toy model of superposition" (2023) **[abstract only]**: k-gon critical points determine both Bayesian and SGD transitions. Training travels from regions of high loss and low complexity to low loss and high complexity. This concerns transitions while the loss is still changing, not post-plateau change — [Chen et al. 2023](https://doi.org/10.48550/arXiv.2310.06301)

**(e) Representational drift and hidden progress at fixed performance**
- Sharon & Dar (2024) **[full text]**: training runs to 4000 epochs. A "Phase III" of representation evolution continues after the training data are perfectly fit. In ViT-B/16 trained with Adam, CKA similarity blocks line up with phase transitions in encoder blocks 4–12, and the layers evolve in sync. Transitions are more abrupt for ViT than ResNet and for SGD than Adam. Evidence is qualitative heatmaps, with no scalar summary — [Sharon & Dar](https://arxiv.org/abs/2405.17377)
- Ratzon, Derdikman & Barak (eLife 2023/24) **[full text]**: after the loss converges, noisy learning keeps changing representations (predictive-coding model). The fraction of active units falls from about 0.7 to about 0.33 over 1e6 steps (approx./visual), and spatial information rises. The same happens with Adam/RMSprop even without added noise, but not with noiseless GD. The authors interpret this as drift along the zero-loss manifold toward flatter regions. Weight decay was not tested — [Ratzon et al.](https://doi.org/10.7554/eLife.90069)
- Andriushchenko, Varre, Pillaud-Vivien & Flammarion (ICML 2023) **[full text]**: with large step sizes, training loss stabilizes but there is hidden progress toward sparse features. In DenseNet-100, block-4 feature sparsity is 50–60% at small vs 10–20% at large steps. A longer large-step phase gives better test error once the LR is decayed. Their Section 3 experiments use no weight decay — [Andriushchenko et al.](https://doi.org/10.48550/arXiv.2210.05337)

**(f) Late duplicate and opposite pairs**
- Chen, Kunin, Yamamura & Ganguli, "Stochastic collapse" (NeurIPS 2023; JSTAT 2024) **[full text]**: SGD noise pulls the dynamics toward invariant sets of "identical neurons" (equal incoming weights, bias and outgoing weights) and "vanishing neurons", even when such sets are saddles of the loss.
  - Trained VGG-16 (CIFAR-10) and ResNet-18 (CIFAR-100), **with GELU activations**, show large clusters of neurons with near-identical incoming and outgoing parameters.
  - Larger LR gives fewer independent neurons.
  - A longer LR-0.1 phase (drop after 10k/20k/40k steps) leaves fewer independent neurons and better accuracy after the drop, *even though training loss and test accuracy had already plateaued*. Averaged over 8 runs.
  - Weight decay (5e-4) was present but not isolated.

  — [Chen et al. 2023](https://doi.org/10.1088/1742-5468/ad6429)
- Martinelli et al., "Flat channels to infinity" (NeurIPS 2025) **[full text]**: pairs of neurons whose input weights converge while their output weights diverge with opposite signs. In the limit they implement a gated linear unit. The output-weight difference grows as 1/ε, where ε is half the input-weight distance. The authors found these channels in roughly 20–64% of fits on Gaussian-process targets, and 24% on modified Rosenbrock. They appear with softplus, erf and sigmoid (GELU/tanh not tested), in deeper MLPs but not tested in transformers. Regularization stops the dynamics *inside* the channel, leaving an approximate gated linear unit — [Martinelli et al.](https://doi.org/10.48550/arXiv.2506.14951)

**(g) Loss of plasticity and weight growth with long training**
- Dohare et al. (Nature 2024) **[full text]**:
  - Permuted MNIST, 800 tasks: up to 25% of units die, and mean weight magnitude rises from about 0.03 to ≥0.10 (approx./visual).
  - Stationary ant locomotion, 50M steps, PPO: dormant units about 50%, stable rank about 90→25, mean |w| about 0.05→0.115 (approx./visual).
  - L2 stops the continual weight growth but does not fully prevent loss of plasticity.

  — [Dohare et al. 2024](https://doi.org/10.1038/s41586-024-07711-7)
- Papyan, Han & Donoho (PNAS 2020) **[abstract only, via deep-search summary]**: in classifiers trained beyond zero training error, features keep moving toward neural-collapse geometry, a specific trajectory in the terminal phase of training — [Papyan et al.](https://doi.org/10.1073/pnas.2015509117)

### Inferences
- **Which theories predict continued internal change at constant LR with WD?** Kosson (steady rotation at equilibrium), Kunin (anomalous diffusion), Ratzon and Andriushchenko (noise-driven drift toward sparser solutions), Chen et al. (noise-driven attraction to duplicate-neuron sets), and Nanda, Varma and Prieto (weight decay keeps reorganizing circuits after the fit and counteracts logit scaling). Hoogland shows LLC-detectable reorganization at nearly flat loss. Constant LR matters here: Kunin, Kosson and Andriushchenko all tie the size of the ongoing motion to η (or ηλ). An annealed schedule would damp much of what D3 reports.
- **Order of magnitude for our run (Kosson formulas; these assume scale-invariant weights, which MLP weights in pre-LN blocks are only partly):**
  - ηλ = 5e-4 × 0.05 = 2.5e-5 per step. PyTorch AdamW applies decay as θ ← θ − η·λ·θ, so ηλ is this product.
  - Equilibrium timescale is about 1/(ηλ) = 40,000 steps; the squared-norm e-folding time is about 20,000 steps.
  - With 35k images per epoch that is about 35000/B steps per epoch: about 73 epochs at B=64 and about 146 epochs at B=128.
  - So equilibration of norms can plausibly extend into the window where we see the peak at epochs 80–110 and the merging at 100–150. Check this against the real batch size before using it.
  - Predicted angular update per step for β1=0.9: sqrt(2·2.5e-5·0.1/1.9) ≈ 1.6e-3 rad.
  - The 3× growth of output norms for the merged neurons is consistent with Kosson's scale-sensitive case: a persistent outward radial gradient lowers the effective decay. It is also consistent with Prieto's logit-scaling (NLM) direction if those neurons feed the confidence logit. Both readings are hypotheses to test by logging radial gradient components and norms against the equilibrium prediction.
- **The plateau-at-constant-LR caveat for D1.** Andriushchenko and Chen both show that test performance at a large constant LR can understate what the network has gained. Reviewers may ask whether IoU "saturation" is a noise floor of the LR. The cheap test is a short LR-decay probe (e.g., 5–10 epochs of cosine to zero) from checkpoints at epochs 40, 100 and 200. If IoU after the probe rises with checkpoint epoch, the plateau hides progress and the late reorganization (D3) may be functional.
- **Peak then decline of opposite pairs (D3)** has the same shape as "formation then cleanup" (Nanda) and "logits rise then fall" (Varma), but here without any change in performance. Martinelli's channels predict opposite-sign output-weight divergence along converging input weights, and say weight decay halts it inside the channel. That makes a "peak then halt or decline" plausible under wd 0.05. This is speculative.
- **Merging into near-duplicates (D3)** is the closest match to Chen et al.'s stochastic collapse: GELU, late, constant high LR, incoming and outgoing similarity together. Cite it as anticipating the phenomenon. Our addition is the timing (epochs 100–150, well after the plateau) and that the merged neurons sit in GELU's negative tail.
- **Identity turnover (D3)** is consistent with continued rotation (Kosson) and drift (Ratzon, Kunin). Also note that the user's own 24-seed shorter runs, described in the earlier workspace prior-art search, reportedly show partner turnover (about 299 pairs entering and 340 leaving opposition between steps 55k and 65.6k). That is internal evidence of replication across seeds and should be cited internally if it applies to the same model.

### Gaps
- No study found that measures neuron-level turnover or pair life cycles after a validation plateau in a ViT or detector trained with AdamW at constant LR.
- Wan 2021 (SMD) full text was not available; only the abstract is used.
- Kosson gives no closed-form AdamW equilibrium for non-scale-invariant weights, so the 3× norm growth cannot be predicted quantitatively from the literature.
- Hoogland's run with weight decay (the LM) found no significant change in its last stage (17k–50k). The clear near-flat-loss reorganization (LR3/LR4) is in the run without weight decay. Neither is a perfect analogue of our setup.

---

## Q2. Confidence versus accuracy over training; detection-specific calibration (D2)

### Takeaway
That confidence keeps increasing after accuracy saturates is well established for classifiers trained with cross-entropy. The mechanisms: NLL overfitting, logit-norm growth, the logit-scaling direction after fitting, and longer training making models more accurate but worse calibrated, including ViTs. Detection calibration work defines the right metrics (D-ECE, LaECE, LaACE₀) but almost never tracks calibration over training. The one direct detector precedent is Oksuz 2023, Appendix E.3 (longer schedule → higher calibration error). No paper was found that reports *rising false-positive confidence on object-free images over training*. Huseljic 2024 reports overconfidence of a different kind on OOD images: overconfident *background*.

### Cited Findings

**Classifier evidence: confidence drifts after accuracy saturates**
- Guo, Pleiss, Sun & Weinberger (ICML 2017) **[full text]**:
  - Fig. 3 (110-layer ResNet with stochastic depth, CIFAR-100, LR drops at epochs 250 and 375): after the epoch-250 drop, test NLL rises for the rest of training while test error keeps improving (29%→27%). Networks can overfit NLL without overfitting the 0/1 loss.
  - Once nearly all training samples are classified correctly, NLL can still be reduced by raising confidence.
  - Less weight decay worsens calibration: ECE approx. 0.24 at wd 1e-5 falling to about 0.02 before rising to about 0.08 at 1e-2 (visual). More depth and width, and BatchNorm, worsen ECE.
  - ECE uses M=15 equal-width bins. Plain ResNet-110 on CIFAR-100: ECE 16.53% uncalibrated, 1.26% after temperature scaling. Supplementary Fig. S1: the optimal temperature rises over training.

  — [Guo et al. 2017](https://arxiv.org/abs/1706.04599)
- Mukhoti et al. (NeurIPS 2020) **[full text]**:
  - Setup: ResNet-50, CIFAR-10, 350 epochs, LR 0.1 → 0.01 at 150 → 0.001 at 250.
  - After epoch 150, average test NLL rises, driven only by misclassified samples ("curse of misclassified samples"), and test ECE rises.
  - Softmax entropy falls throughout training for both correct and incorrect test samples, so the network grows more confident in its wrong predictions.
  - Proposed mechanism: cross-entropy pushes ‖W‖ upward ("weight magnification").
  - CE baseline ECE: 4.35% (CIFAR-10, T=2.5 → 1.35%) and 17.52% (CIFAR-100, T=2.1 → 3.42%).

  — [Mukhoti et al. 2020](https://arxiv.org/abs/2002.09437)
- Wei et al., LogitNorm (ICML 2022) **[full text]**:
  - Setup: WRN-40-2 on CIFAR-10, 200 epochs.
  - Mean logit norm rises over training: ID train approx. 7–8 → peak about 14 at epoch 100 → about 13 at epoch 200; ID test → about 11–12; **SVHN OOD inputs approx. 6 → 10–11** (visual).
  - Scaling logits leaves the predicted class unchanged and raises softmax confidence (Props. 3.1–3.2).
  - Under CE, final ID and OOD confidence distributions both concentrate at high values. OOD *confidence* is not plotted against epoch, only logit norm.

  — [Wei et al. 2022](https://doi.org/10.48550/arXiv.2205.09310)
- Prieto et al. 2025 **[full text]**: after the training data are fit, updates align with logit scaling (cosine about 0.9 in output layers). Confidence can therefore rise with no change in accuracy, and weight decay counteracts this. The paper has no separate calibration experiment; the link to confidence follows from their stated mechanism — [Prieto et al.](https://doi.org/10.48550/arXiv.2501.04697)
- Minderer et al. (NeurIPS 2021) **[full text]**:
  - ViT and MLP-Mixer are among the best-calibrated families. Their calibration degrades more slowly under ImageNet-C shift.
  - Pretraining **duration** moves a model along its accuracy/calibration Pareto set: longer-trained models are more accurate and *worse* calibrated (BiT, 112k–1M steps; Fig. 10).
  - Pretraining dataset size has little consistent effect.
  - ECE uses equal-mass binning with 100 bins. ViT ECE values are not tabulated.

  — [Minderer et al. 2021](https://arxiv.org/abs/2106.07998)
- Pinto, Torr & Dokania (ECCV 2022) **[abstract only]**: modern CNNs (ConvNeXt) can be as reliable as transformers, or more so, and there is "no clear winner". So "ViTs are well calibrated" should not be assumed for our model — [Pinto et al.](https://doi.org/10.48550/arXiv.2207.11347)
- Xie et al., "Do not train away uncertainty" (arXiv 2610.12048, Oct 2026 preprint) **[full text]**:
  - Validation ECE reaches a minimum early in training and then rises while accuracy improves only marginally. Shown for ResNet-50/CIFAR-10, **ViT-B/32/CIFAR-100** (graphically) and Llama3-8B.
  - Exact ResNet-50 values: ep 40 acc 80.44% / ECE 6.04%; ep 60 acc 92.12% / ECE 1.65%; ep 200 acc 94.74% / ECE 3.34%.
  - Their ViT-B/32 anchor checkpoint is epoch 5.

  — [Xie et al. 2026](https://arxiv.org/abs/2610.12048)
- Tao et al., "Calibrating a deep neural network with its predecessors" (arXiv 2302.06245) **[full text]**:
  - Network blocks overfit for calibration at different stages.
  - ResNet-50/CIFAR-10: early stopping on loss gives error 6.58 / ECE 2.32; early stopping on error gives error 4.89 / ECE 4.03.
  - With the other blocks fixed, the final block's calibration keeps degrading while error stays level.

  — [Tao et al.](https://doi.org/10.48550/arXiv.2302.06245)

**Detection calibration: metrics and findings**
- Küppers et al. (CVPRW 2020) **[full text]**:
  - D-ECE measures confidence against *precision*: a detection counts as correct if its class is right and its IoU is at or above a threshold. Evaluated at IoU 0.6 and 0.75.
  - Bins jointly over confidence and box variables (cx, cy, w, h): 20 bins for confidence-only ECE, 8 per dimension for 3-D D-ECE, 5 per dimension for 5-D.
  - Miscalibration depends on box position, rising toward image borders.
  - COCO baselines, D-ECE at IoU 0.6, confidence only: Faster R-CNN 19.2%, R-FCN 11.7%, SSD 6.6%. Calibration lowers these to about 0.9–1.3%.
  - No analysis of images without objects or of training length.

  — [Küppers et al. 2020](https://doi.org/10.1109/CVPRW50498.2020.00171)
- Oksuz, Joy & Dokania (CVPR 2023) **[full text]**:
  - LaECE target: a true positive (TP) should have confidence equal to its IoU with the matched ground truth; a false positive (FP) should have confidence 0.
  - Faster R-CNN is overconfident and ATSS underconfident. Uncalibrated LaECE on SAOD-Gen: F-RCNN 43.3, RS-RCNN 32.0, ATSS 15.7, D-DETR 15.9.
  - Images without in-distribution objects should be rejected at image level; pseudo-OOD images are made by zeroing pixels inside the ground-truth boxes.
  - **Appendix E.3, Table A.15: F-RCNN trained 12→36 epochs (no multiscale) raises AP 24.8→25.5 but raises mECE 16.4→18.7. The authors attribute this to longer training making the model overconfident.**

  — [Oksuz et al. 2023](https://doi.org/10.1109/CVPR52729.2023.00894)
- Kuzucu, Oksuz et al. (ECCV 2024) **[full text]**:
  - Pitfalls of current evaluation:
    - D-ECE ignores localization and is minimized by confidence 1 for every TP and 0 for every FP.
    - A fixed threshold (usually 0.30) changes detector rankings; AP is maximized at threshold 0, so it cannot pick an operating point.
    - Calibrators are often fitted on shifted data.
  - Proposed metrics: LaECE₀ / LaACE₀ with τ=0, i.e., target = IoU, and FP target = 0. LaECE₀ uses 25 bins; LaACE₀ is the per-detection mean absolute confidence–IoU error.
  - Recommended operating thresholds: chosen by LRP.
  - Recommended post-hoc baselines: Platt scaling and isotonic regression, per class. On COCO, IR gives LaECE₀ 7.7 vs Cal-DETR 11.6.
  - No epoch-wise calibration analysis. Training Cityscapes D-DETR for 200 instead of 50 epochs (together with a larger image scale) gave D-ECE 2.9, but the confound prevents isolating epochs.

  — [Kuzucu et al. 2024](https://doi.org/10.48550/arXiv.2405.20459)
- Munir et al. (NeurIPS 2022) **[full text]**: train-time calibration loss TCD. Baseline FCOS D-ECE: 16.1 on Sim10K, 15.1 on Cityscapes, 18.4 on Cityscapes→Foggy. No analysis over training epochs. Their qualitative claim is that TCD lowers confidence of wrong detections — [Munir et al. 2022](https://doi.org/10.48550/arXiv.2209.07601)
- Munir et al. (CVPR 2023, BPC loss) **[full text]**: groups detections into accurate/inaccurate × confident/not-confident. Baseline D-DETR D-ECE is 12.8 on COCO and 10.8 on CorCOCO; FCOS COCO is 22.0. The numeric IoU threshold is not reported. No D-ECE-vs-epoch analysis — [Munir et al. 2023](https://doi.org/10.1109/CVPR52729.2023.01104)
- Park, Sobolewski & Azizan (IEEE TPAMI; arXiv 2024) **[full text]**: in DETR, Hungarian matching gives one calibrated prediction per object and trains the unmatched queries toward zero foreground confidence (background weight w∅ = 0.1). Unmatched queries can still localize well. Mean negative-query confidence correlates negatively with per-image AP (PCC −0.47 to −0.69). No analysis over training epochs and no metric for empty images — [Park et al.](https://doi.org/10.1109/TPAMI.2026.3686715)
- Huseljic et al. (IJCV 2024/2025) **[full text]**: post-processing (thresholding, NMS, top-k) changes calibration a lot. For post-processed Faster R-CNN on the animals subset, object classes are overconfident and background underconfident; raw outputs show the reverse. On OOD images (Open Images, no COCO classes), pretrained detectors predict *background* with high, overconfident probability. Raising DETR's background-class weight in fine-tuning (best MCE at 8.6 vs default 0.1) improved calibration. No analysis over training epochs — [Huseljic et al.](https://doi.org/10.1007/s11263-024-02219-z)
- Liu et al., COCO-FP (2024) **[abstract only]**: background false positives from non-target clutter are a persistent problem; YOLOv9-E AP50 drops 72.8→65.7 from COCO to COCO-FP — [Liu et al.](https://doi.org/10.48550/arXiv.2409.07907)

### Inferences
- **D2 has firm support at the mechanism level.** The classification part of our loss (heatmap/focal-type, sigmoid) keeps rewarding larger logit margins after localization stops improving. This matches Guo, Mukhoti, Wei and Prieto. Weight decay 0.05 would slow the growth (Guo: more weight decay gives better ECE; Prieto: weight decay counteracts logit scaling) but not stop it.
- **Empty frames.** Our empty-frame "confidence" is the sigmoid of the *maximum* logit over all tokens, an extreme-value statistic. A uniform increase in logit scale or variance raises it even if per-token calibration were constant. This explains why false-positive confidence rises *in parallel* with true-positive confidence: VAL TP +0.036, empty +0.046. The TP–empty gap therefore stays about the same (0.234 → 0.224 on VAL). Our arithmetic suggests that this is mostly a drift in scale, not a loss of discrimination. That must be shown with AUROC/AUPRC for frame-level presence across epochs, plus a temperature-scaled analysis (Minderer uses temperature scaling to remove over- or underconfidence as a confounder).
- **Conventions matter.** CenterNet/FCOS-style targets push the peak score toward 1 whatever the IoU, so "confidence > IoU" on positive frames is partly by design. Under LaECE/LaACE₀ (target = IoU, FP target = 0), our epoch-200 numbers (mean conf 0.899 vs mean IoU 0.765; empty-frame conf 0.675 vs target 0) would read as overconfidence. Under D-ECE (precision at an IoU threshold) the picture can differ. Report both, and state the convention.
- **Defensible novelty:** an epoch-resolved trajectory of false-positive confidence on object-free surgical frames under a fixed IoU plateau. The nearest prior is a two-point comparison (12 vs 36 epochs) in Oksuz 2023.
- **bf16 caveat.** Prieto shows that numerical precision changes late-phase logit dynamics (Softmax Collapse occurs earlier at lower precision). That concerns softmax cross-entropy, not sigmoid/focal losses. A reviewer could still ask whether bf16 affects late logit growth; a short fp32 continuation from epoch 100 would answer it.

### Gaps
- No detector paper found with calibration (D-ECE/LaECE) plotted against training epoch, and none with false-positive confidence on object-free images over training.
- Wei 2022 plots logit norm on OOD data over epochs, not OOD confidence. Xie 2026 gives exact values only for ResNet-50, not ViT-B/32.
- Whether ViT detectors are better or worse calibrated than CNN detectors is not settled. Minderer (classification) and Pinto (no clear winner) point different ways, and no detection-specific ViT-vs-CNN calibration study was retrieved.

---

## Q3. Ablation methodology: zero vs mean vs resample, self-repair, redundancy, interaction (A1)

### Takeaway
A single zero-ablation null on an aggregate metric is common and accepted as an observation. The literature warns against reading it as "the units do not matter", for three reasons:
1. Ablations measure total effects, which downstream compensation (Hydra effect or self-repair, LayerNorm rescaling, standing counterweights) can mask.
2. The replacement value is an uncontrolled "dose".
3. Aggregate metrics hide subset-specific effects.

Current best practice asks for several ablation types (zero, mean, mode/optimal, resample), continuous logit-linear metrics, joint ablations with an inclusion–exclusion interaction term and matched-random controls, and direct-effect checks.

### Cited Findings

**Ablation baselines and replacement values**
- Heimersheim & Nanda (2024) **[full text]**: mean ablation is somewhat more principled than zero ablation because zero has no special meaning for activations. Noising and denoising (resample-style patching) keep the model less out-of-distribution than ablation. Backup components (the Hydra effect) can make important components look unimportant, and the compensation is "lossy". AND/OR toy circuits show that a single-component intervention can miss redundancy depending on its direction. Use several metrics, trusting logit difference most and avoiding probability alone. No numerical sample-size recommendation — [Heimersheim & Nanda](https://doi.org/10.48550/arXiv.2404.15255)
- Zhang & Nanda (arXiv 2309.16042; usually cited as ICLR 2024, a venue the PDF itself does not confirm) **[full text]**:
  - Gaussian noising vs symmetric token replacement changes which components get localized: the GN peak is 2–5× higher in GPT-2 XL factual recall. Recommend STR.
  - Recommend logit difference and avoid probability, which misses negative components. KL is acceptable.
  - Sliding-window (joint) patching gives 1.40×, 1.75× and 1.59× the summed single-layer effects for windows of 3, 5 and 10. Joint effects therefore exceed sums, and joint results should be read as window effects.
  - No systematic zero-vs-mean comparison.

  — [Zhang & Nanda](https://doi.org/10.48550/arXiv.2309.16042)
- Li & Janson, "Optimal ablation" (arXiv 2409.09951; NeurIPS 2024 per common citation, not confirmed by the PDF) **[full text]**:
  - Optimal ablation (OA) replaces a component with the constant that minimizes expected loss against the original model's outputs.
  - On GPT-2-small IOI, the median component's OA gap is 11.1% of its zero-ablation gap, 33.0% of its mean-ablation gap and 17.7% of its resample gap. Mean Δ: zero 0.0584, mean 0.0405, OA 0.0035.
  - Zero and mean ablation can "spoof" downstream computation: the replacement value itself misleads the rest of the network, beyond the information removed.

  — [Li & Janson](https://doi.org/10.48550/arXiv.2409.09951)
- Morcos, Barrett, Rabinowitz & Botvinick (ICLR 2018) **[full text]**: units were ablated by clamping to zero. Clamping to the empirical mean was **more** damaging than zero in CIFAR-10 ConvNets (Appendix A.1). Networks that generalize better are less reliant on single directions. Class selectivity predicts ablation impact poorly: Spearman 0.095 for MNIST, −0.428 for CIFAR-10, r=−0.012 for ImageNet. Reliance on single directions grows as overfitting begins: AUC versus test loss Spearman −0.728. Ablation order was randomized (10 orderings) — [Morcos et al. 2018](https://arxiv.org/abs/1803.06959)
- Pochinkov, Pasero & Shibayama (arXiv 2408.17322) **[full text]**: "peak" ablation sets a neuron to its most frequent (modal) activation. For OPT it is the best of peak, mean and zero: 50% pruning acc 44.54 vs 42.39 vs 41.77. **For ViT, peak, mean and zero do not differ significantly**; the numerical best was zero (50% pruning acc 48.65 ± 2.54, CE 2.46 ± 0.17). Resampling was worst in both. Most neurons are roughly zero-centred and symmetric, and peak ablation targets the minority that are not — [Pochinkov et al.](https://doi.org/10.48550/arXiv.2408.17322)
- Ahmad, Seth & Sankarapu, "Every ablation is a dose" (arXiv 2610.02173, Oct 2026 preprint) **[full text]**: every clamp value v corresponds to a pair-specific dose λ(v) = 2(m−v)/Δ, where m is the clean/counterfactual midpoint and Δ the class contrast. Zero and mean ablation are therefore uncontrolled doses that vary by example. Downstream "counterweights" release a coupling that already existed, which looks like self-repair. 52 of 68 certified couplings across four 7–9B LLMs were counterweights, and 7/7 certified heads in GPT-2 IOI. A small net ablation effect does not show a component is unimportant or that a pair is redundant. Recommendation: dose–response curves and joint dosing — [Ahmad et al.](https://arxiv.org/abs/2610.02173)

**Self-repair and the Hydra effect**
- McGrath, Rahtz, Kramár, Mikulik & Legg (2023) **[full text]**:
  - Model: Chinchilla 7B, trained without dropout. Data: Counterfact (1,209 contexts).
  - Ablating attention layer 23 of 32 triggers downstream compensation with slope 0.69 (r² = 0.92) against the direct effect.
  - At middle layers, compensation plus reduced late-MLP "erasure" restores about 70% of the logit reduction.
  - Resample ablation (15 alternative prompts) is used; zero ablation is out of distribution for models trained without dropout.
  - Total effects (ablation) and direct effects (unembedding) correlate poorly. Measuring only total effects is insufficient.

  — [McGrath et al. 2023](https://doi.org/10.48550/arXiv.2307.15771)
- Rushing & Nanda (arXiv 2402.15390) **[full text]**: self-repair is defined as Δlogit − ΔDE_head (DE = direct effect) and studied with resample ablations of attention heads in Pythia and GPT-2, over up to 1M tokens. It is imperfect and noisy and can over-correct. LayerNorm rescaling accounts for "around 30%" of a head's direct effect (a directional estimate). The clean/ablated LayerNorm-scale ratio exceeds 1 in 90.66% of resample, 98.75% of mean and 100% of zero ablations. A sparse set of neurons does "anti-erasure". Recommendations: avoid zero ablation where possible, freeze LayerNorm when ablating, or use path patching — [Rushing & Nanda](https://doi.org/10.48550/arXiv.2402.15390)

**Redundancy, interactions and co-ablation**
- Meyes, Lu, de Puiseau & Meisen (2019) **[full text]**: ablation by zeroing incoming weights.
  - MLP on MNIST: single units range from −44.5 percentage points (unit 12) to −1.4 points (unit 6).
  - Pairwise ablations reveal redundancy: units 12+19 misclassify more than 50% of class-6 digits that either single ablation left correct. Pair 4+16 is super-additive.
  - VGG-19: recovery after ablating 25% of filters needs about 1 epoch; most performance recovers even after about 80% of filters are ablated with retraining.
  - Report class-specific effects as well as overall ones.

  — [Meyes et al. 2019](https://arxiv.org/abs/1901.08644)
- Zhou, Sun, Bau & Torralba (2018) **[full text]**: ablation by zeroing weights and bias. AlexNet-Places conv5: overall accuracy falls 50.6% → 50.4%, while single classes fall up to 50% (waterfall). More than half of conv5 units cause a drop above 10% in at least one class. Class selectivity correlates weakly with the overall drop (r=0.139) and more strongly with the maximum class drop (r=−0.221) — [Zhou et al. 2018](https://arxiv.org/abs/1806.02891)
- Vaidyanathan et al., "Curse of multiple mediators" (arXiv 2606.27510, 2026) **[full text]**:
  - For components A and B: xINT({A,B}) = E(A,B) − E(A) − E(B).
  - Positive xINT is super-additive and negative is sub-additive, but the sign alone does not label the mechanism (OR vs AND).
  - In IOI under pABC mean ablation, summed single path-specific effects (PIEs) are +5.775 while summed cross-interactions are −4.045. All eight measured NNMH–S-inhibition pairs have negative xINT (−0.086 to −0.606 logit difference).
  - Recommendation: measure interactions as diagnostics and run joint patches for candidate sets.

  — [Vaidyanathan et al.](https://doi.org/10.48550/arXiv.2606.27510)
- Gong et al., "Conditional co-ablation" (COAX; arXiv 2607.01940, 2026) **[full text]**: measures how much a unit's ablation effect grows after a primary set has been ablated.
  - GPT-2 IOI backup heads: ROC-AUC 0.33 (single-ablation saliency) → 0.91.
  - Primary-only ablation drops logit difference by 0.22; adding the COAX backups gives 1.76, vs 1.0 ± 0.7 for a matched-random top-up.
  - Freezing the backups removes about 55% of the self-repair.
  - Joint ablation (1.15) is 1.9× the sum of singles (0.60).
  - Pairwise synergy is defined as I_uv = δz_uv − δz_u − δz_v.
  - Recommendation: matched-random and same-size controls.

  — [Gong et al.](https://doi.org/10.48550/arXiv.2607.01940)
- Hütten, Hölken, Tercan & Meisen, "Detection transformers under the knife" (arXiv 2507.21723, 2025) **[full text]**: post-training ablations of DETR, DDETR and DINO on COCO val by randomly zeroing weights, with 100 random configurations per level (5/15/30/50%).
  - Localization (mgIoU) and classification (F1) are reported *separately*, and they can move in opposite directions.
  - DETR: 50% encoder MHSA ablation costs 39 points of mgIoU and 29 points of F1.
  - DINO: 58% of query-embedding values are near zero. Removing the content queries changes mgIoU by +0.03 points, interpreted as distributed, redundant representations.
  - MLP neurons were not ablated.

  — [Hütten et al.](https://doi.org/10.48550/arXiv.2507.21723)

### Inferences
- **Zero vs typical value.** Both units sit near GELU's minimum (about −0.17 at x ≈ −0.75) on most tokens, so zeroing moves each by about +0.17 per token. In the residual stream that shift is about 0.17·‖w_out‖ per unit, and it grew with the 3× norm growth. Zero is therefore near the *mode* but not equal to it. The *mean* is pulled upward by rare large positive activations. In Ahmad et al.'s terms, zero-ablation is a small but nonzero dose. Report zero, mean and mode ablation; optimal ablation is a cheap scalar fit per neuron. Pochinkov found no significant difference among them for ViT, so we can expect agreement, but it must be shown.
- **The null on IoU is a total effect.** Following McGrath and Rushing, measure (i) the pair's *direct* contribution to the head output (box parameters and the winner logit) with downstream layers frozen, and (ii) the ablation with the final pre-head LayerNorm scale frozen. A large direct effect with a small total effect would mean compensation, not irrelevance.
- **IoU is a coarse outcome.** It is bounded, and the argmax-token selection makes it piecewise in the parameters. Heimersheim & Nanda and Zhang & Nanda recommend continuous logit-linear metrics. Add box-edge L1 deltas for the winner token, the winner logit, the winner-minus-runner-up logit margin, and per-frame distributions. Hütten's precedent is to report localization and confidence separately.
- **The interaction ratio (pair effect / larger single effect = 1.66) is non-standard.** Under exact additivity with singles a ≥ b the ratio is 1 + b/a, so 1.66 is additive only if b ≈ 0.66a. The ratio alone cannot tell additive from mildly sub- or super-additive. Replace or supplement it with the standard inclusion–exclusion term Δ_ij = δ_pair − δ_i − δ_j (Vaidyanathan's xINT; Gong's I_uv), normalized by |δ_i| + |δ_j|, with a film-level bootstrap CI. Compare it with the same quantity for the 300 control pairs.
- **Terminology.** In ablation work, "redundant" usually means OR-like backups: each single ablation is small and the joint ablation is large, i.e., super-additive damage (Meyes's pair 12+19; Gong's backups). Additive effects mean *independent, parallel contributions*. That is what near-duplicate neurons writing the same direction under a near-linear readout should produce. Use "additive (non-interacting) contributions of near-duplicate units", or "parallel copies".
- **Standing offset of an anti-parallel pair in the GELU tail (our own arithmetic; to verify).** If two units have near-anti-parallel output weights (w_j ≈ −w_i) and both sit at about −0.17 on most tokens, their typical residual contributions nearly cancel: −0.17·w_i − 0.17·w_j ≈ 0. Zero-ablating *both* then removes almost nothing on those tokens, while each single ablation removes a standing offset of about 0.17·‖w‖. If the measured logit effects are nevertheless additive (ratio 1.66), they probably come from the minority of tokens where the units are active (positive). Split each ablation effect into tail-token and active-token parts before interpreting it.
- **Opposite (cancelling) pairs.** The expected signature is sub-additive damage: a joint effect smaller than either single effect. The prior-art search summarizes the user's 24-seed data as showing that kind of edge-change cancellation. For the A1 pair, report Δ_ij for box edges and for the logit separately, since the two can differ (Hütten; the user's Finding 4 in the earlier search).

### Gaps
- No ablation-methodology paper specific to dense CenterNet/FCOS-style ViT detector heads was found; Hütten covers DETR attention and queries, not MLP neurons.
- None of the retrieved papers gives a numeric minimum for frames or prompts. Sample size is reported (Rushing up to 1M tokens; McGrath 15 resample patches; Hütten 100 configurations) but not prescribed.
- Ahmad 2026, Vaidyanathan 2026 and Gong 2026 are recent preprints and not yet peer reviewed.

---

## Q4. Pruning and redundant-neuron literature linking duplicate neurons to weight decay or long training

### Takeaway
Duplicate or near-duplicate units are well known and can be merged without loss by summing their outgoing weights. Their *emergence late in training* is best explained by noise-driven attraction to "identical-neuron" invariant sets at large constant LR (stochastic collapse). A specific causal role for weight decay is suggested by theory but not isolated experimentally.

### Cited Findings
- Srinivas & Babu (BMVC 2015) **[full text]**: data-free pruning merges a neuron into a near-duplicate partner with "surgery" a_i ← a_i + a_j, choosing pairs by saliency s_ij = ⟨a_j²⟩·‖W_i − W_j‖². On MNIST (500-neuron FC layer), pruning 440 neurons gives 97.99% vs 99.06%. On CaffeNet, removing 21.27M weights (34.89%) gives 55.6% vs 57.84%. The authors motivate redundancy but do not explain how training produces duplicates — [Srinivas & Babu 2015](https://doi.org/10.5244/C.29.31)
- Casper et al., "Frivolous units" (AAAI 2021) **[full text]**: "prunable" units can be removed with little effect; "redundant" units are linear combinations of others (by PCA at ≤ 0.05 unexplained variance). Both grow faster than width: gain factors of 3.15–3.86 for prunability and 1.79–2.20 for redundancy on ImageNet. Weight decay, dropout and augmentation have only modest effects. In ResNet-18 the counts are stable across post-convergence epochs, so the paper does not show emergence during training — [Casper et al.](https://doi.org/10.1609/aaai.v35i8.16853)
- Chen, Kunin, Yamamura & Ganguli (NeurIPS 2023 / JSTAT 2024) **[full text]**: see Q1(f). Identical-neuron clusters form under SGD noise and are stronger at larger LR and after a longer high-LR phase. The experiments use GELU. Weight decay was present but not tested as a cause — [Chen et al. 2023](https://doi.org/10.1088/1742-5468/ad6429)
- Weight-decay theory for two-layer networks **[abstract-level, via the deep search "Weight decay implicit bias on how a function is split across hidden neurons..."]**. For ReLU two-layer networks, L2 on the parameters becomes a group/variation-norm cost once each neuron's scale is optimized. Same-sign copies then tie with a single neuron, while opposite-sign cancellation wastes norm. These equivalences depend on homogeneity and do not carry over directly to GELU. A positive L2 penalty makes the regularized objective coercive, which rules out minimizing sequences that diverge in norm, such as flat channels to infinity — [deep-search summary](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Weight%20decay%20implicit%20bias%20on%20how%20a%20function%20is%20split%20across%20hidden%20neurons%20same%20sign%20copies%20versus%20opposite%20sign%20pairs); Martinelli et al. also say regularization stops the dynamics inside the channel ([Martinelli et al.](https://doi.org/10.48550/arXiv.2506.14951))
- Dalvi et al. (EMNLP 2020) **[not read; title/abstract-level via the deep search]**: neuron-level redundancy in pretrained transformers (BERT, XLNet) — [Dalvi et al. 2020](https://doi.org/10.18653/v1/2020.emnlp-main.398)
- Dohare et al. 2024 **[full text]**: long stationary training produces dormant units and weight growth, and L2 stops the weight growth (see Q1(g)) — [Dohare et al.](https://doi.org/10.1038/s41586-024-07711-7)

### Inferences
- For our *merged* pair (D3), the strongest citations are Chen et al. 2023 (mechanism and timing at constant LR) and Srinivas & Babu 2015 (merging duplicates by summing output weights preserves function). Simple arithmetic also predicts that zero-ablating one of two near-duplicates writing the same direction removes about half of their joint contribution, and ablating both removes all of it: the additive pattern. A1's pair is described as the *most opposed* pair, so this reasoning carries over to A1 only if that pair functionally behaves like same-direction copies. The text should state explicitly whether the A1 pair and the D3 merged pair are the same neurons.
- Weight decay alone (two-layer ReLU theory) does not strongly favour splitting over a single unit: same-sign copies tie. So weight decay is unlikely to be the driver of the merging; noise at constant LR (Chen) is the better-supported mechanism. In our GELU setting, the negative-tail regime breaks homogeneity, so the ReLU results are suggestive only.

### Gaps
- No study found that isolates weight decay as a cause of duplicate-neuron formation in deep networks or transformers.
- No measurement of duplicate formation over training in a ViT was found.

---

## Q5. Methodological standards reviewers will expect, how to phrase the claims, and reviewer risks

### Takeaway
The biggest risks are a single seed for the trajectory claims (D1–D3), small n for empty-frame confidence (D2), and how the ablation null is interpreted (A1). Most can be addressed cheaply: perturbed continuations from the epoch-60 state, detection calibration metrics with temperature-scaled and AUROC controls, mean/mode/optimal ablations with direct-effect checks, an inclusion–exclusion interaction term, and equivalence bounds.

### Cited Findings

**Standards that published work demonstrates**
- **Seeds and replicates:** Nanda et al. show curves over 5 seeds ([Nanda et al.](https://doi.org/10.48550/arXiv.2301.05217)). Hoogland et al. check stage boundaries across 5 seeds ([Hoogland et al.](https://arxiv.org/abs/2402.02364)). Morcos et al. use 200 networks and 10 random ablation orders ([Morcos et al.](https://arxiv.org/abs/1803.06959)). Chen et al. average 8 runs ([Chen et al.](https://doi.org/10.1088/1742-5468/ad6429)). Hütten et al. use 100 random configurations per ablation level ([Hütten et al.](https://doi.org/10.48550/arXiv.2507.21723)).
- **Calibration metrics:** ECE with 15 equal-width bins and reliability diagrams ([Guo et al.](https://arxiv.org/abs/1706.04599)); equal-mass ECE with 100 bins, plus temperature scaling as a control for confidence scale ([Minderer et al.](https://arxiv.org/abs/2106.07998)); D-ECE at IoU 0.6 and 0.75 with box-variable binning ([Küppers et al.](https://doi.org/10.1109/CVPRW50498.2020.00171)); LaECE/LaECE₀/LaACE₀ with target = IoU for TPs and 0 for FPs, 25 bins, an LRP-chosen operating threshold, and Platt/isotonic baselines fitted on in-distribution data ([Oksuz et al.](https://doi.org/10.1109/CVPR52729.2023.00894); [Kuzucu et al.](https://doi.org/10.48550/arXiv.2405.20459)); NLL, Brier and MCE over all class probabilities including background, raw versus post-processed ([Huseljic et al.](https://doi.org/10.1007/s11263-024-02219-z)).
- **Ablation:**
  - zero vs mean vs resample vs optimal ablation ([Li & Janson](https://doi.org/10.48550/arXiv.2409.09951); [Heimersheim & Nanda](https://doi.org/10.48550/arXiv.2404.15255));
  - mode/peak ablation ([Pochinkov et al.](https://doi.org/10.48550/arXiv.2408.17322));
  - dose–response ([Ahmad et al.](https://arxiv.org/abs/2610.02173));
  - direct vs total effect, frozen LayerNorm ([McGrath et al.](https://doi.org/10.48550/arXiv.2307.15771); [Rushing & Nanda](https://doi.org/10.48550/arXiv.2402.15390));
  - inclusion–exclusion interactions and joint patching ([Vaidyanathan et al.](https://doi.org/10.48550/arXiv.2606.27510)); conditional co-ablation with matched-random controls ([Gong et al.](https://doi.org/10.48550/arXiv.2607.01940));
  - continuous logit-difference metrics ([Zhang & Nanda](https://doi.org/10.48550/arXiv.2309.16042));
  - class- or subset-specific effects ([Zhou et al.](https://arxiv.org/abs/1806.02891); [Meyes et al.](https://arxiv.org/abs/1901.08644));
  - localization and classification reported separately ([Hütten et al.](https://doi.org/10.48550/arXiv.2507.21723)).
- **Dynamics diagnostics:**
  - weight norms and angular updates against the equilibrium prediction ([Kosson et al.](https://arxiv.org/abs/2305.17212));
  - parameter displacement versus steps ([Kunin et al.](https://doi.org/10.1162/neco_a_01626));
  - CKA of each layer against its own earlier checkpoints ([Sharon & Dar](https://arxiv.org/abs/2405.17377));
  - LR-decay probes to expose hidden progress ([Andriushchenko et al.](https://doi.org/10.48550/arXiv.2210.05337); [Chen et al.](https://doi.org/10.1088/1742-5468/ad6429));
  - LLC estimates for stage detection ([Hoogland et al.](https://arxiv.org/abs/2402.02364)).

### Inferences

**Concrete checklist for our paper (priority order)**
1. **Seeds for D3 and D2 (high priority).** Run at least 2–3 more seeds. If compute is short, branch "perturbed continuations" from the exactly restored epoch-60 state with different sampler/augmentation seeds. This tests whether the epoch-80–110 peak, the identity turnover and the epoch-100–150 merging are deterministic given the epoch-60 weights or depend on noise. Chen's and Ratzon's noise-driven accounts predict that the *statistics* repeat while the *identities* differ. The user's 24-seed shorter runs, if from the same model family, can be cited for turnover.
2. **D2 calibration package:**
   - Per-epoch LaACE₀ (|conf − IoU|, with IoU = 0 on empty frames) and D-ECE at IoU 0.5/0.75.
   - Reliability diagrams at epochs 9, 40, 100 and 200.
   - Frame-level presence AUROC (positive vs empty) per epoch, to separate "scale drift" from "loss of discrimination".
   - A temperature-scaled re-analysis: fit T at each checkpoint on VAL, report on TEST.
   - A bootstrap CI on the *paired* change of empty-frame confidence between epochs (the same 34/28 frames), plus a trend test over all checkpoints (Spearman with epoch, film-level block bootstrap).
   - Report the false-positive rate at a fixed operating threshold chosen once on VAL at the plateau epoch.
3. **D1:**
   - Phrase it as a plateau, not convergence.
   - Add an LR-decay probe from epochs 40, 100 and 200.
   - Optionally give equivalence bounds, e.g., "IoU change between epochs 50 and 200 within ±0.01 (TOST)".
4. **A1 ablation package:**
   - Zero, mean, mode (and optionally optimal) ablation.
   - Effects on continuous outputs: box-edge deltas, winner logit, winner-minus-runner-up margin.
   - Direct effect with downstream frozen, and an LN-frozen variant.
   - Inclusion–exclusion Δ_ij with CI, placed in the distribution over the 300 control pairs.
   - A smallest detectable effect: the half-width of the bootstrap CI.
   - Per-film or per-instrument breakdown (Zhou-style subset effects).
   - Multiplicity control (e.g., FDR) across the 768 single ablations if any are called significant.
   - Optionally COAX-style conditional ablation: ablate the pair, then rescan single-neuron effects to look for backups.
5. **Dynamics logging:** per-neuron ‖w_in‖ and ‖w_out‖, angular update per epoch, and the radial gradient component, compared with Kosson's ηλ predictions computed from the real batch size.

**How to phrase it in the article (suggested wording)**
- D1: "Under a constant learning rate, validation IoU reached a plateau by epoch ~40 (0.748–0.775 through epoch 200; film-level bootstrap CIs overlap). A plateau of the task metric does not imply stationary weights [Kunin; Zhang 2021; Kosson 2024]."
- D2: "Winner-token confidence kept rising after the IoU plateau (0.863→0.899 on frames with an instrument), and so did confidence on instrument-free frames (0.629→0.675, n=34). This matches the continued confidence growth and NLL overfitting reported for classifiers after accuracy saturates [Guo 2017; Mukhoti 2020; Wei 2022], and the higher calibration error with longer detector schedules reported by Oksuz et al. (12 vs 36 epochs). To our knowledge, epoch-resolved false-positive confidence on object-free frames has not been reported." Add the AUROC/temperature-scaling result to say whether it is a drift in scale or in discrimination.
- D3: "Internal structure kept reorganizing after the plateau. This is expected under AdamW with weight decay at constant LR, where weights reach a rotational equilibrium with a steady angular update [Kosson 2024] and noise drives the network toward simpler configurations, including near-duplicate neurons [Chen et al. 2023 'stochastic collapse']. What we add is the life cycle of opposite pairs (peak at epochs 80–110, then decline), the turnover of pair identities, and late merging (epochs 100–150) of two GELU-negative-tail neurons into near-duplicates."
- A1: "Zero-ablating the most opposed pair changed IoU by ≈1e-5 (CI includes 0), inside the distribution of 300 control pairs. Because both units sit near the GELU minimum on most tokens, zero-ablation here is close to mode or mean ablation [Pochinkov 2024; Morcos 2018]. Ablation measures total effects, which downstream compensation can mask [McGrath 2023; Rushing & Nanda 2024]; we therefore also report direct effects and LN-frozen ablations. The pair's contributions add (Δ_ij ≈ 0) rather than cancel." Avoid "redundant" unless backup-style super-additivity is shown.
- Terminology: define "opposite beams" once in standard terms ("MLP neurons with anti-parallel output-weight vectors in the residual stream") and cite Martinelli 2025 for the related two-layer phenomenon of opposite-sign output-weight divergence.

**Reviewer risks (expected objections and mitigations)**
1. *"One seed; the pair dynamics may be idiosyncratic."* Mitigate with perturbed continuations and extra seeds; report the statistics of pairs, not identities.
2. *"The plateau is an artifact of constant LR."* Mitigate with an LR-decay probe; cite Andriushchenko and Chen.
3. *"Confidence drift is just logit-scale growth / temperature."* This is probably true. Embrace it: show temperature-scaled curves and AUROC, and tie it to Prieto/Wei.
4. *"n=34/28 empty frames is too small."* Use a paired bootstrap at film level, a trend test across all checkpoints, and pool VAL+TEST as a robustness check.
5. *"The confidence semantics are not calibrated probabilities."* State the target convention (heatmap peak → 1) and report both D-ECE and LaACE₀.
6. *"A zero-ablation null proves nothing (self-repair, spoofing, wrong metric)."* Mitigate with mean/mode/optimal ablation, direct effects, frozen LN, continuous metrics and equivalence bounds.
7. *"The interaction ratio is ad hoc; 'redundant' is misused."* Report inclusion–exclusion Δ_ij with CI and rename.
8. *"Multiplicity across 768 single ablations and 300 controls."* Report the distributions and the percentile of the target pair; apply FDR if testing.
9. *"Novelty: duplicates and continued motion are known."* Cite stochastic collapse, rotational equilibrium and flat channels proactively, and claim only the measured life cycle and the detector-specific confidence trajectory.
10. *"bf16 effects on late logit growth."* Optionally run an fp32 continuation check (cf. Prieto on precision and late logit dynamics).

### Gaps
- No source gives a field-wide numeric minimum for frames, seeds or control pairs in ablation studies. The precedents above are descriptive.
- Venues for several arXiv PDFs (Zhang & Nanda; Li & Janson; Rushing & Nanda; Kunin) were not confirmed from the PDFs read. Cite with arXiv IDs or confirm venues separately.
- The 2026 preprints (Xie, Ahmad, Vaidyanathan, Gong) are recent and not peer reviewed; use them as supporting rather than anchor citations.
