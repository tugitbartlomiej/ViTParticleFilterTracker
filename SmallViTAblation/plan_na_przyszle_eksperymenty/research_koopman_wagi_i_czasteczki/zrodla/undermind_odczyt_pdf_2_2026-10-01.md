[Yeh23b]:
### Node and Edge Representations and Observables
In Section 3.2 (pages 6–7), each individual neural network parameter (each weight and bias) is treated as a graph node $u$ (Section 5, page 16). The observable encoded at time step $t$ is the scalar value of the parameter itself, denoted $v_{u,t}$. 

The node feature vector $x_{u,t}$ is formed by concatenating a learnable node-identifier lookup embedding $\sigma_N(\text{id}_u)$ with an MLP-encoded scalar node value $w(v_{u,t})$:
$$x_{u,t} = \sigma_N(\text{id}_u) \mathbin{\Vert} w(v_{u,t})$$
Edge features $e_{u,v}$ consist solely of learnable lookup table embeddings of edge identifiers $\sigma_E(\text{id}_{u\to v})$.

The predicted quantities decoded by the graph decoder $\phi^{-1}$ and per-node MLPs are the **actual weights and biases** at future time steps $k+t$, **not** weight updates, gradients, or layer activations (Sections 3.1, 3.2, 4.1.1, and 4.3; pages 5–10).

---

### Experimental Setup, Datasets, Architectures, and Splits

* **Datasets & Tasks (Section 4.1.1, Appendix A, pages 8, 20–21):**
  * *Linear regression:* Synthetic data (20 features, 2,000 samples).
  * *Classification:* UCI Wine dataset (13 features, 3 classes, 178 samples).
  * *Image classification:* UCI Digits dataset (8×8 bitmaps, restricted to 2 classes to limit size).
  * *Differential Equation (DE) solver:* Solving $f'(x) - f(x) = y$ with $f(1)=0$.
* **Architectures (Table 3, page 10; Table 5, page 15; Appendix A, page 20):**
  * Networks range between 50 and 300 parameters: 1-fc; 2-fc (106 parameters); 3-fc; 1-conv-2-fc (194 parameters); 2-conv-1-fc.
* **Optimizers (Section 4.1.1, Appendix B.2 & B.3, pages 8, 21–23):**
  * Primary: SGD without momentum. 
  * Robustness tests: Adagrad, Adadelta, Adam, and shuffled mini-batches.
* **Trajectory Splits, History Window, and Horizons (Sections 4.1.1, 4.1.2, Appendix A, pages 8, 20–21):**
  * 10,000 generated trajectories split into 8,000 train, 1,000 validation, and 1,000 test.
  * Trajectories span 50 to 500 epochs, downsampled every 10 epochs.
  * Single-snapshot input ($t=0$ or time step $k$), requiring no history window or warm start.
  * No future gradients are used; predictions advance forward autonomously via $K^t$.

---

### Linearization, Optimizer States, BatchNorm, and Noncausal Information

* **Nature of Linearization:** Empirical, not exact. The model learns an approximate finite-dimensional Koopman matrix $K$ by minimizing multi-task autoencoder loss functions (reconstruction, latent linearity, and dynamics prediction; Section 3.1, page 6).
* **Moments and BatchNorm:** Neither optimizer moments (such as momentum buffers or Adam second moments) nor Batch Normalization parameters/running statistics are incorporated into the graph state.
* **Trajectory Error vs. Task/Loss Performance:** 
  * Table 3 (page 10) measures trajectory mean squared error (MSE) between actual and predicted weights/biases. 
  * Table 4 (page 13) and Table 5 (page 15) measure optimization loss ratio $r = (l_0 - l_{\text{pred}})/(l_0 - l_{\text{true}})$. 
  * As noted in Section 4.3.1 (page 13), the method often deviates from the exact path in weight space while still discovering an alternative smooth trajectory that reaches equivalent final training loss. Task classification accuracies are not reported.
* **Noncausal Information:** None is used during inference; prediction maps initial weights directly to future weights via $x_{k+t} = \phi^{-1}(K^t \phi(x_k))$.

Below is Table 3, which details the trajectory prediction errors on weights and biases across baseline algorithms and network architectures:

Table 3 compares the trajectory prediction mean squared errors of DMD, Extended DMD, Graph DMD, the Lusch autoencoder, and the proposed KMPNN across diverse architectures.
[REGION: page=10, bbox=170,215,395,782]

---

[Liu26p]:
### Authors
Kang Liu and Suyan Li (arXiv:2608.19762v1, 2026).

---

### Key Theoretical Formulations and References

1. **Augmented Joint State ($\theta, m, v$):**  
   Defined on **Page 3, Section 3.2** as $x_s := [\theta_s, m_{s-1}, v_{s-1}]^\top \in \mathbb{R}^{3d}$, incorporating parameters and both AdamW moment buffers.

2. **Perturbation Input Operator ($B_t$):**  
   Derived on **Page 4, Equation (1)** (detailed in Appendix A.4, Page 12) as:
   $$B_t = \begin{bmatrix} \Gamma_t \\ (1-\beta_1)I \\ 2(1-\beta_2)\mathrm{Diag}(g_t^c) \end{bmatrix}$$
   mapping gradient perturbation $\xi_t$ into initial joint-state deviation $\delta x_{t+1}$.

3. **LTV Tangent Transition ($A_s$), Propagator ($\Phi$), and Response Operator:**  
   Given on **Page 4, Equation (2)** (Appendix B.2, Page 14, Eq. 25). The block Jacobian captures parameter retention and coupling feedback ($g \to m \to \theta$ and $g^{\odot 2} \to v \to \theta$). The multi-step propagator is $\Phi_{t+h, t+1} = A_{t+h-1}\cdots A_{t+1}$ (**Page 4, Section 3.4**). Theorem 3.1 (**Pages 4–5**) defines the horizon-$h$ input–output operator:
   $$\mathcal{G}_{t,h} := c_{t+h}^\top \Phi_{t+h, t+1} B_t$$

4. **Multi-step Error Decomposition:**  
   Theorem 3.4 (**Page 6, Equation 8**; proof in Appendix D.2, Page 18, Eq. 33) isolates the exact state error into initial write-in nonlinearity, accumulated smooth Taylor remainders, and activation-switching defects:
   $$e_{t+h} = \Phi_{t+h, t+1} r_{B,t}(\xi_t) + \sum_{j=t+1}^{t+h-1} \Phi_{t+h, j+1} (r_j^{\mathrm{sm}} + r_j^{\mathrm{sw}})$$
   Output probe error bound is given in **Equation (10)** (**Page 6**).

---

### Readouts Tested
- **Fixed test probes:** Fixed quadratic objectives (standard, anisotropic via diagonal scaling, rotating-readout centered between trajectory states; Pages 22–23), test set loss on CIFAR-10 batches (Page 24), and held-out token sequences evaluated by cross-entropy differential on language models (Pages 25–26).
- **Time-varying realized future loss:** $F_s(\theta) = \ell(\theta; B_s)$ (Pages 3, 21).
- **Summary metrics:** Horizon maximum absolute response ($M_{t,H}$), timing ($h^\star$), sign ($s^\star$), accumulated response ($ARE$), and positive excursion ($P^+$) (Pages 5, 15, 21).

---

### Forecasting Under Unseen Future Data vs. Common Future
The full ISO operator is strictly **pathwise**—it conditions on an observed/shared future realization $\omega$ and evaluates intervening trajectory gradients/Hessians. 

To determine whether influence under *unseen* resampled futures can be predicted purely from present-time quantities without oracle future access, the authors test:
- **Prospective variance ratio ($\Pi_H$):** Unseen futures retain strong candidate-specific ranking structure ($\Pi_{32} = 0.730$ in matched quadratics; $\Pi_{12} = 0.921$ for MLP-GELU, $0.700$ for CNN-ReLU; Table 4, Page 9).
- **Predictability without future data:** 
  - Standard gradient/write norms fail to rank the conditional mean under exact one-step matching (correlations $\le 0.05$ in quadratics; Table 17, Page 30).
  - Present-time surrogate estimators (evaluating only current reference batches or a single frozen Jacobian $A_t$) succeed well in neural networks (Spearman $\rho = 0.755\text{--}0.941$ to conditional mean $\mu_i^M$; Table 4, Page 9), but collapse in the controlled quadratic setting ($\rho = -0.066$; Table 4). Thus, predicting unseen futures without future trajectories is highly regime-dependent and lacks general zero-future guarantees.

---

### Treatment of Batch Normalization and Limitations
- **Batch Normalization:** Not used; the architectures evaluated rely on fixed channel normalization (CIFAR-10), standard convolutions/MLPs, or Transformer architectures with LayerNorm (Pythia), keeping activation derivatives piecewise-smooth without batch-dependent coupling.
- **Limitations:** 
  - Pathwise formulation requires observing or assuming the entire future trajectory.
  - Tangent accuracy degrades under activation switching (e.g., CNN local error exponent drops from 2.017 to 1.120 by horizon 12; Page 7).
  - Fails when future dynamical drift or large parameter displacements violate local Lipschitz smoothness.

---

[Wan26m]:
### Diagnostic vs. Prospective Tool
[Wan26m] frames predictability primarily as a **retrospective, parameter-resolved diagnostic** of temporal redundancy in stored checkpoint trajectories (Section 1, p. 1–2; Section 5, p. 7). While Appendix H (pp. 41–42; Table 12, p. 42; Fig. 23, p. 42) demonstrates a prospective online check—showing that restricting Predictive Differential Training (PDT) write-back strictly to auxiliary parameters improves accuracy gains (+0.87–0.94 pp vs. +0.62–0.73 pp full-scope on SGD CIFAR-10) with lower variance—the authors explicitly note this is an initial proof-of-concept rather than a wall-clock acceleration algorithm (App. H.3, p. 42).

---

### Auxiliary (Normalization/Bias) vs. Bulk Matrices
* **Systematic Separation:** Vector-like parameters (normalization scales/shifts, biases) exhibit significantly simpler and higher short-horizon predictability than matrix-like feature transforms (bulk parameters) across both multi-pass vision (CIFAR) and one-pass language pretraining (Pythia-14M to 1.4B) (Sec. 4.2, p. 4; App. C.1–C.3, pp. 20–22; App. F.2, p. 34).
* **Quantitative Evidence:** Auxiliary exceeds bulk predictability in 29/32 trajectory-level comparisons under Element-wise Sign Agreement (ESA) and 32/32 under Dynamic Mode Decomposition (DMD) direction cosine (Sec. 4.2, p. 4; Fig. 3, p. 6; Fig. 17, p. 37). In Pythia, this ordering holds at all 246 evaluated checkpoints (p. 4).

---

### Time-Varying Regimes and Bulk "Predictable Pockets"
* **Anti-persistence:** Bulk parameters rapidly drop below the 0.5 chance sign-agreement floor early in training, indicating oscillatory, diffusion-like steps (Sec. 4.3, pp. 5–6; Fig. 5(a), p. 7).
* **Predictable Pockets:** Bulk tensors are largely unpredictable in aggregate but harbor localized, transient "predictable pockets" (~6% of pooled group–anchor observations) where future residual ratio (FRR) is elevated (Sec. 4.2, p. 5; Fig. 4, p. 6; App. D.1, p. 23).
* **Scale and Depth Relay:** In Pythia-70M $qkv$, head-level predictability fades by ~20k steps, shifting to row-level pockets in deeper layers (Layers 4/5) (Sec. 4.3, p. 6; App. D.2, p. 25; Fig. 9, p. 26).

---

### Probes, Rank, and Koopman/DMD
* **Probe Hierarchy:** Evaluated across three families (Sec. 3, p. 3):
  1. *Displacement-direction:* ESA and Vector Cosine Similarity (VCS) (Eq. 1).
  2. *Subspace-residual:* FRR (Eq. 2) and Low-Residual Group Fraction (LRGF; Eq. 3).
  3. *Predictor-based:* Inertia, linear trend, and Koopman/DMD (Eqs. 1–3, App. B.2, p. 16).
* **Rank & Geometric Ceilings:** History matrices ($n_{\text{past}} = 5$) use Gavish–Donoho truncated SVD rank ($r \le 4$; App. B.3, p. 17). DMD predictions lie strictly in the recent-history subspace, establishing the geometric headroom bound $\cos_{\text{DMD}} \le \sqrt{1 - \text{FRR}^2}$ (Sec. 4.1, p. 4; Fig. 2(b), p. 5).

---

### Baselines, Null Models, Horizons, and Optimizers
* **Causality & Horizons:** Probes use strictly causal pre-anchor history ($n_{\text{past}} = 5$ epochs/checkpoints) to forecast future displacement $\Delta w_a^{(\tau)}$ over horizons $\tau \in \{1, 2, 3, 5, 10, 14\}$ (Sec. 3, p. 3; App. A.4, p. 12).
* **Null Calibration:** Group metrics are calibrated against size-matched coordinate-permutation nulls ($\text{LRGF}_{\text{cal}}$, $\alpha=0.05$; App. B.4, p. 17).
* **Optimization Effects:** In AdamW and SGD, weight decay introduces a slow drift ($-\lambda w$) that increases ESA and lowers FRR, whereas higher momentum depresses directional persistence (Sec. G.3, pp. 40–41; Fig. 22, p. 41). Predictability negatively correlates with final optimizer accuracy (median $\rho_S = -0.71$), proving predictability is not a proxy for optimization quality (App. G.4, p. 41).

---

[Kny24]:
- **Classification:** NiNo is **not** a Koopman operator-based method; it is a **learned graph neural network (GNN) weight nowcaster** that models parameter evolution over neural graphs (Sec. 1, 4, pp. 1–6).
- **Input States & Optimizer Moments:** NiNo takes **only** past parameter weight checkpoints $\Theta_{\tau:\tau_c} \in \mathbb{R}^{n \times c}$ (default context length $c=5$, sampled every 200 steps) mapped onto edge features of a neural graph, plus Laplacian Positional Encodings (LPE) for node features (Sec. 4.2, p. 5). It uses **no gradients and no optimizer moments** (Fig. 4 caption, p. 6; App. A.9.1, line 10, p. 20).
- **Horizons & Alternating Correction:** NiNo uses direct multi-step forecasting to predict future parameter deltas for horizons $k \in [1, K]$ with $K=40$ (Sec. 4.3, Eq. 9, p. 6). Optimization alternates between running a base optimizer (Adam/AdamW) for 1,000 steps and applying a single NiNo parameter jump (Sec. 3.1, p. 3; Sec. 5, p. 6). The future prediction horizon $k$ decays over training via $k \approx K((T - t)/T)^p$ with $p=2$ (Sec. 4.3, p. 6; Fig. 12, p. 18).
- **Transformer Architecture Sensitivity:** Transformers break naive neuron permutation symmetries across attention heads. NiNo resolves this sensitivity by adding dedicated per-head nodes (restricting permutations across heads) and treating $W^q, W^k, W^v$ as separate directional 1D edges (Sec. 4.1, Fig. 2, pp. 4–5). This architectural modification provides a 48.9% average speedup versus 37.1% for naive graphs (Table 2, p. 8) and enables zero-shot generalization to 111M LLaMA3-style Transformers (Sec. 5.2, Table 4, p. 9).
- **Fair Speedup Accounting:** Speedup is measured by the relative reduction in steps required to match the target validation performance of tuned baseline Adam/AdamW (Sec. 5.1, Table 1, p. 7; Table 2, p. 8). Because NiNo executes only once every 1,000 steps, wall-clock time overhead is under 0.5% for a 7.4M parameter model and ~1.47% for a 111.5M parameter model (Sec. 5.3, Table 7, p. 19).
- **Token-Vector Prediction on Images:** **No.** NiNo does not predict full hidden token vectors or activations; it predicts solely parameter/weight deltas $\Delta\theta$ (Sec. 3, 4.3, pp. 3–6; App. A.8.2, p. 18).

Region format: page is 1-indexed; bbox=top,left,bottom,right, normalized 0-1000 from top-left.
